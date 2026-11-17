// tts-outetts.cpp -- `llama-tts-outetts`: text -> speech via the fork's own
// OuteTTS + WavTokenizer pipeline.
//
// GATE-CLAIM-OK: a #ifdef GGML_USE_CUDA / #else banner below prints which vocoder backend THIS
// BUILD was compiled with (a compile-time capability line, ported verbatim from
// examples/tts/tts.cpp:164-168) -- not a runtime claim that any GPU or CPU resource is busy or free.
//
// Pipeline (OuteTTS):
//   text --[backbone LLM, libllama]--> acoustic codes
//        --[WavTokenizer decoder, standalone ggml]--> spectrogram
//        --[ISTFT + overlap-add, audio-common]--> waveform -> WAV
//
// ENGINE-MERGE PORT (row S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01):
// SOURCE-READ: examples/tts/tts.cpp (whole file, 545 lines). That file compiles only into the
// static `wavtokenizer` library's example harness, which tools/tts/CMakeLists.txt never builds
// as an executable (examples/ is not add_subdirectory()'d at all) -- see
// temp/research/findings/S69_STRANDED_PORTS_APPLIED_2026-09-08.md "Port 1" for why the
// upstream mtmd-based tools/tts/tts.cpp was deliberately left unmodified rather than merged
// with this one. This file is a NEW entry point (target `llama-tts-outetts` below), a
// common_params-shaped rewrite of examples/tts/tts.cpp's OuteTTS pipeline, keeping its logic
// (text front-end, prompt shape, code-base detection, 3-stage decode) but updated for the
// current API generation:
//   gpt_params            -> common_params
//   gpt_params_parse()    -> common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, ...)
//   llama_init_from_gpt_params()/.model/.context -> common_init_from_params()->model()/->context()
//   common_sampler_init(model, params.sparams) -> llama_init->sampler(0) (built internally from
//                                                  params.sampling; SOURCE-READ: tools/tts/tts.cpp:79-82)
//   llama_token_is_eog(model, id) -> llama_vocab_is_eog(vocab, id) (SOURCE-READ:
//                                     tools/completion/completion.cpp:156,932)
//   manual llama_free()/llama_free_model() -> RAII via common_init_result_ptr
// common_tokenize()/common_batch_add()/common_batch_clear()/common_sampler_sample()/
// common_sampler_accept() kept the same signatures this fork already had (verified against
// common/common.h:1532-1569 and tools/tts/tts.cpp:151-160) -- no change needed there.
//
// Text front-end and the OuteTTS prompt format follow llama.cpp b4404
// examples/tts/tts.cpp (MIT). The CLI shape (task selectors, --metrics,
// per-request options) follows audio.cpp (Apache-2.0). audio-common.{h,cpp}
// (WAV I/O + ISTFT) is reused directly from examples/audio/ (self-contained,
// no llama/ggml dependency -- SOURCE-READ: examples/audio/audio-common.h:1-19) rather than
// duplicated; see this target's CMakeLists.txt entry.

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"

#include "wavtokenizer.h"
#include "audio-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

//
// ------------------------------------------------------------ CLI options ---
//

struct tts_options {
    std::string vocoder_path;
    int         sample_rate = 24000;
    int         n_fft       = 1280;
    int         n_hop       = 320;
    int         code_base   = -1;    // first acoustic-code token id (-1 = auto)
    int         code_count  = -1;    // codebook size (-1 = from the vocoder)
    int         voc_threads = 0;     // 0 = follow -t
    bool        voc_gpu     = false;
    bool        metrics     = false;
    bool        list_only   = false;
    // SOURCE-READ: examples/tts/tts.cpp:70-74 -- b4404 unconditionally blanked the first 0.25s
    // for a speaker-preamble artifact that no longer exists in this prompt; kept opt-in (0 here).
    float       lead_silence_s = 0.0f;
    // raw OuteTTS + WavTokenizer output measured at -46 dBFS (near-inaudible); normalisation
    // is opt-in so the default keeps reference numerics (SOURCE-READ: examples/tts/tts.cpp:76-79).
    bool        normalize   = false;
    float       peak        = 0.95f;
};

static void tts_print_usage(int /* argc */, char ** argv) {
    const char * argv0 = argv[0];
    LOG_INF("\n");
    LOG_INF("OuteTTS + WavTokenizer text-to-speech (fork-native pipeline)\n\n");
    LOG_INF("usage: %s -m <backbone.gguf> --vocoder-model <wavtokenizer.gguf> -p \"text\" -o out.wav\n\n", argv0);
    LOG_INF("audio options (in addition to the standard llama options):\n");
    LOG_INF("  -vm,  --vocoder-model PATH   WavTokenizer decoder GGUF (required)\n");
    LOG_INF("  -o,   --output PATH          output WAV path (default: output.wav; common_params::out_file)\n");
    LOG_INF("        --out-file PATH        alias of --output (the examples/tts spelling)\n");
    LOG_INF("        --tts-sample-rate N    output sample rate (default: 24000)\n");
    LOG_INF("        --tts-n-fft N          ISTFT window (default: 1280)\n");
    LOG_INF("        --tts-n-hop N          ISTFT hop (default: 320)\n");
    LOG_INF("        --tts-code-base N      first acoustic-code token id (default: auto-detect)\n");
    LOG_INF("        --tts-code-count N     codebook size (default: taken from the vocoder)\n");
    LOG_INF("        --tts-lead-silence S   seconds of silence to blank at the start (default: 0)\n");
    LOG_INF("        --vocoder-threads N    vocoder CPU threads (default: -t)\n");
    LOG_INF("        --vocoder-gpu          run the vocoder on the GPU backend if available\n");
    LOG_INF("        --tts-normalize        peak-normalise the output (raw OuteTTS output is quiet)\n");
    LOG_INF("        --tts-peak X           normalise to this peak, implies --tts-normalize (default 0.95)\n");
    LOG_INF("        --metrics              print wall time / audio length / RTF\n");
    LOG_INF("        --list-audio-caps      print the audio capabilities of this build and exit\n");
    LOG_INF("\n");
}

// Pulls the audio-specific flags out of argv so the remainder can go to
// common_params_parse() untouched (unknown flags are rejected there).
// SOURCE-READ: examples/tts/tts.cpp:107-158 (tts_extract_options), same design.
static std::vector<char *> tts_extract_options(int argc, char ** argv, tts_options & o) {
    std::vector<char *> rest;
    rest.push_back(argv[0]);

    auto next = [&](int & i) -> const char * {
        if (i + 1 >= argc) {
            LOG_ERR("%s: missing value for %s\n", __func__, argv[i]);
            exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-vm" || a == "--vocoder-model") {
            o.vocoder_path = next(i);
        } else if (a == "--tts-sample-rate") {
            o.sample_rate = std::atoi(next(i));
        } else if (a == "--tts-n-fft") {
            o.n_fft = std::atoi(next(i));
        } else if (a == "--tts-n-hop") {
            o.n_hop = std::atoi(next(i));
        } else if (a == "--tts-code-base") {
            o.code_base = std::atoi(next(i));
        } else if (a == "--tts-code-count") {
            o.code_count = std::atoi(next(i));
        } else if (a == "--tts-lead-silence") {
            o.lead_silence_s = (float) std::atof(next(i));
        } else if (a == "--vocoder-threads") {
            o.voc_threads = std::atoi(next(i));
        } else if (a == "--vocoder-gpu") {
            o.voc_gpu = true;
        } else if (a == "--tts-normalize") {
            o.normalize = true;
        } else if (a == "--tts-peak") {
            o.peak = (float) std::atof(next(i));
            o.normalize = true;
        } else if (a == "--metrics") {
            o.metrics = true;
        } else if (a == "--list-audio-caps") {
            o.list_only = true;
        } else if (a == "--out-file") {
            // examples/tts/tts.cpp:123 accepted -o / --out-file / --output; common/arg.cpp names the option
            // -o / --output / --output-file, so --out-file fell through to common_params_parse and was refused.
            static char output_flag[] = "--output";
            rest.push_back(output_flag);
        } else {
            rest.push_back(argv[i]);
        }
    }
    return rest;
}

static void tts_print_caps() {
    LOG_INF("ik_llama.cpp OuteTTS pipeline capabilities\n");
    LOG_INF("  speak (tts)      : OuteTTS backbone (llama arch) + WavTokenizer decoder [built in]\n");
    LOG_INF("  vocoder backend  : ggml, standalone graph (no libllama architecture required)\n");
#ifdef GGML_USE_CUDA
    LOG_INF("  vocoder gpu      : available (--vocoder-gpu)\n");
#else
    LOG_INF("  vocoder gpu      : not built for this backend (--vocoder-gpu unavailable)\n");
#endif
    LOG_INF("  istft            : hann window, cached-twiddle irfft, overlap-add\n");
    LOG_INF("  wav i/o          : read PCM u8/s16/s24/s32 + IEEE float32/64, write PCM s16\n");
    LOG_INF("  resampler        : windowed sinc (Blackman)\n");
}

//
// ------------------------------------------------- OuteTTS text front-end ---
// SOURCE-READ: examples/tts/tts.cpp:180-303, ported unchanged (no API dependency).
//

static const std::map<int, std::string> k_ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> k_tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

static std::string convert_less_than_thousand(int num) {
    std::string result;
    if (num >= 100) {
        result += k_ones.at(num / 100) + " hundred ";
        num %= 100;
    }
    if (num >= 20) {
        result += k_tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + k_ones.at(num % 10);
        }
    } else if (num > 0) {
        result += k_ones.at(num);
    }
    return result;
}

static std::string number_to_words(const std::string & number_str) {
    try {
        const size_t decimal_pos = number_str.find('.');
        const std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                result += convert_less_than_thousand(int_number / 1000000000) + " billion ";
                int_number %= 1000000000;
            }
            if (int_number >= 1000000) {
                result += convert_less_than_thousand(int_number / 1000000) + " million ";
                int_number %= 1000000;
            }
            if (int_number >= 1000) {
                result += convert_less_than_thousand(int_number / 1000) + " thousand ";
                int_number %= 1000;
            }
            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        if (decimal_pos != std::string::npos) {
            result += " point";
            for (char digit : number_str.substr(decimal_pos + 1)) {
                if (digit >= '0' && digit <= '9') {
                    result += " " + k_ones.at(digit - '0');
                }
            }
        }
        return result;
    } catch (const std::exception &) {
        return " ";
    }
}

static std::string replace_numbers_with_words(const std::string & input_text) {
    const std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;

    auto it  = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch & match = *i;
        result.append(input_text, last_pos, (size_t) match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = (size_t) match.position() + (size_t) match.length();
    }
    result.append(input_text, last_pos);
    return result;
}

// OuteTTS v1 prompt processor: lower-case, strip punctuation, join with the
// text separator token.
static std::string process_text(const std::string & text) {
    std::string t = replace_numbers_with_words(text);

    std::transform(t.begin(), t.end(), t.begin(), ::tolower);

    t = std::regex_replace(t, std::regex(R"([-_/,\.\\])"), " ");
    t = std::regex_replace(t, std::regex(R"([^a-z\s])"), "");
    t = std::regex_replace(t, std::regex(R"(\s+)"), " ");
    t = std::regex_replace(t, std::regex(R"(^\s+|\s+$)"), "");
    t = std::regex_replace(t, std::regex(R"(\s)"), "<|text_sep|>");

    return t;
}

static void prompt_add(std::vector<llama_token> & prompt, const std::vector<llama_token> & tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}

static void prompt_add(std::vector<llama_token> & prompt, llama_context * ctx, const std::string & txt, bool add_special, bool parse_special) {
    prompt_add(prompt, common_tokenize(ctx, txt, add_special, parse_special));
}

// The acoustic codes live in a contiguous block of special tokens named
// "<|0|>" .. "<|N-1|>". Detecting the base id from the tokenizer avoids a
// hard-coded id that only held for one OuteTTS version.
static int detect_code_base(llama_context * ctx) {
    const std::vector<llama_token> t = common_tokenize(ctx, "<|0|>", false, true);
    if (t.size() == 1) {
        return (int) t[0];
    }
    return -1;
}

//
// ------------------------------------------------------------------- main ---
//

int main(int argc, char ** argv) {
    common_params params;
    tts_options   opts;

    common_init();

    params.prompt         = "";
    params.n_predict      = 4096;
    params.n_batch        = 8192;
    params.n_ctx          = 8192;
    params.sampling.top_k = 4;
    params.sampling.temp  = 1.0f;

    std::vector<char *> rest = tts_extract_options(argc, argv, opts);

    if (opts.list_only) {
        tts_print_caps();
        return 0;
    }

    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_TTS, tts_print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: no text provided -- use -p \"text to speak\"\n", __func__);
        tts_print_usage(argc, argv);
        return 1;
    }
    if (opts.vocoder_path.empty()) {
        LOG_ERR("%s: no vocoder provided -- use --vocoder-model <wavtokenizer.gguf>\n", __func__);
        tts_print_usage(argc, argv);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    const int64_t t_start_us = ggml_time_us();

    // ---- backbone -----------------------------------------------------
    auto llama_init = common_init_from_params(params);
    llama_model   * model_ttc = llama_init->model();
    llama_context * ctx_ttc   = llama_init->context();
    common_sampler * smpl     = llama_init->sampler(0);
    if (!model_ttc || !ctx_ttc || !smpl) {
        LOG_ERR("%s: failed to load the backbone model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab_ttc = llama_model_get_vocab(model_ttc);

    // ---- vocoder --------------------------------------------------------
    wavtokenizer_params wparams;
    wparams.use_gpu   = opts.voc_gpu;
    wparams.n_threads = opts.voc_threads > 0 ? opts.voc_threads : std::max(1, params.n_threads);
    wparams.verbose   = true;

    std::string err;
    wavtokenizer * voc = wavtokenizer_init_from_file(opts.vocoder_path, wparams, err);
    if (!voc) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return 1;
    }

    const int code_base  = opts.code_base  >= 0 ? opts.code_base  : detect_code_base(ctx_ttc);
    const int code_count = opts.code_count >  0 ? opts.code_count : wavtokenizer_n_vocab(voc);

    if (code_base < 0) {
        LOG_ERR("%s: could not detect the acoustic code token base; pass --tts-code-base N\n", __func__);
        LOG_ERR("%s: (the backbone tokenizer must know the special token \"<|0|>\")\n", __func__);
        return 1;
    }
    LOG_INF("%s: acoustic codes: token ids [%d, %d)\n", __func__, code_base, code_base + code_count);

    // ---- stage 1: text -> acoustic codes -------------------------------
    // OuteTTS v1 prompt:  <|im_start|> <|text_start|> words <|text_end|> <|audio_start|>
    // SOURCE-READ: examples/tts/tts.cpp:389-457 (identical prompt shape and decode loop; the
    // preamble b4404 originally prefixed here is deliberately omitted -- see the fork's own
    // measured finding at that location: with the preamble, this fork's own ASR transcribed
    // back the preamble text, not the requested text).
    std::vector<llama_token> prompt_inp;
    prompt_add(prompt_inp, ctx_ttc, "<|im_start|>\n", true, true);
    prompt_add(prompt_inp, ctx_ttc, "<|text_start|>", false, true);

    const std::string prompt_clean = process_text(params.prompt);
    LOG_INF("%s: normalised prompt: '%s'\n", __func__, prompt_clean.c_str());
    prompt_add(prompt_inp, ctx_ttc, prompt_clean, false, true);
    prompt_add(prompt_inp, ctx_ttc, "<|text_end|>\n", false, true);
    prompt_add(prompt_inp, ctx_ttc, "<|audio_start|>\n", false, true);

    LOG_INF("%s: prompt size: %d tokens\n", __func__, (int) prompt_inp.size());

    llama_batch batch = llama_batch_init(std::max<size_t>(prompt_inp.size(), 1), 0, 1);

    for (size_t i = 0; i < prompt_inp.size(); ++i) {
        common_batch_add(batch, prompt_inp[i], (llama_pos) i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx_ttc, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed on the prompt\n", __func__);
        return 1;
    }
    llama_synchronize(ctx_ttc);

    const int64_t t_gen_start_us = ggml_time_us();

    std::vector<llama_token> codes;
    {
        int n_past   = batch.n_tokens;
        int n_decode = 0;

        while (n_decode <= params.n_predict) {
            const llama_token id = common_sampler_sample(smpl, ctx_ttc, -1);
            common_sampler_accept(smpl, ctx_ttc, id, true);

            if (llama_vocab_is_eog(vocab_ttc, id)) {
                LOG_INF("%s: generation stopped on EOG after %d tokens\n", __func__, n_decode);
                break;
            }
            codes.push_back(id);

            common_batch_clear(batch);
            common_batch_add(batch, id, n_past, { 0 }, true);

            n_past   += 1;
            n_decode += 1;

            if (llama_decode(ctx_ttc, batch)) {
                LOG_ERR("%s: llama_decode() failed while generating codes\n", __func__);
                return 1;
            }
        }
    }
    llama_batch_free(batch);

    const double t_gen_s = (ggml_time_us() - t_gen_start_us) / 1e6;
    LOG_INF("%s: generated %d raw tokens\n", __func__, (int) codes.size());

    // keep only acoustic codes and rebase them onto the vocoder codebook
    std::vector<int32_t> acoustic;
    acoustic.reserve(codes.size());
    for (llama_token t : codes) {
        if (t >= code_base && t < code_base + code_count) {
            acoustic.push_back((int32_t) (t - code_base));
        }
    }

    LOG_INF("%s: %d of them are acoustic codes\n", __func__, (int) acoustic.size());
    if (acoustic.empty()) {
        LOG_ERR("%s: the backbone produced no acoustic codes -- is '%s' an OuteTTS model?\n",
                __func__, params.model.path.c_str());
        return 1;
    }

    // ---- stage 2: codes -> spectrogram ----------------------------------
    const int64_t t_voc_start_us = ggml_time_us();

    std::vector<float> embd;
    if (!wavtokenizer_decode(voc, acoustic.data(), (int) acoustic.size(), embd, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return 1;
    }

    const int n_embd_out = wavtokenizer_n_embd_out(voc);
    const double t_voc_s = (ggml_time_us() - t_voc_start_us) / 1e6;

    // ---- stage 3: spectrogram -> waveform --------------------------------
    const int64_t t_wav_start_us = ggml_time_us();

    std::vector<float> audio = audio_embd_to_wave(
            embd.data(), (int) acoustic.size(), n_embd_out,
            opts.n_fft, opts.n_hop, std::max(1, params.n_threads));

    if (audio.empty()) {
        LOG_ERR("%s: the vocoder produced no audio\n", __func__);
        return 1;
    }

    const size_t lead = std::min(audio.size(), (size_t) (opts.lead_silence_s * opts.sample_rate));
    for (size_t i = 0; i < lead; ++i) {
        audio[i] = 0.0f;
    }

    const double t_wav_s = (ggml_time_us() - t_wav_start_us) / 1e6;

    const float rms_pre = audio_rms(audio.data(), audio.size());
    LOG_INF("%s: output level %.2f dBFS (rms)\n", __func__, audio_dbfs(rms_pre));

    if (opts.normalize) {
        audio_peak_normalize(audio, opts.peak);
        LOG_INF("%s: normalised to peak %.2f -> %.2f dBFS (rms)\n",
                __func__, opts.peak, audio_dbfs(audio_rms(audio.data(), audio.size())));
    }

    if (!audio_wav_write_s16(params.out_file, audio, opts.sample_rate, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return 1;
    }

    const double audio_s = (double) audio.size() / opts.sample_rate;
    const double total_s = (ggml_time_us() - t_start_us) / 1e6;

    LOG_INF("%s: wrote %s -- %.2f s of audio at %d Hz\n", __func__, params.out_file.c_str(), audio_s, opts.sample_rate);
    if (opts.metrics) {
        LOG_INF("%s: metrics: codes %.2fs + vocoder %.2fs + istft %.2fs, total %.2fs, RTF %.3f (%.2fx real time)\n",
                __func__, t_gen_s, t_voc_s, t_wav_s, total_s,
                audio_s > 0 ? total_s / audio_s : 0.0,
                total_s > 0 ? audio_s / total_s : 0.0);
    }

    wavtokenizer_free(voc);
    llama_backend_free();

    return 0;
}
