// audio.cpp -- `llama-audio`: listen, think and speak from ONE binary.
//
// This is the unified-audio entry point adopted from audio.cpp
// (https://github.com/0xShug0/audio.cpp, Apache-2.0). audio.cpp's central idea
// is that one native runtime should own every audio task behind a single
// `--task` selector and a shared set of audio utilities, instead of a separate
// demo binary (and a separate Python environment) per model family. The task
// vocabulary, the option shape (--load-option/--session-option style knobs,
// --metrics, --inspect, --list-*) and the VAD chunking flags follow that CLI.
//
// What runs here is ik_llama.cpp's own stack, not a vendored engine:
//   listen : mtmd audio projectors already in this fork
//            (ultravox / qwen2a / voxtral / gemma4a) over the whisper-style
//            mel front-end in examples/mtmd/mtmd-audio.cpp
//   think  : libllama
//   speak  : examples/tts/wavtokenizer.cpp + the ISTFT head in audio-common
//
//   --task asr   : audio in            -> transcript out
//   --task tts   : text in             -> WAV out
//   --task chat  : text in             -> reply text (+ WAV with --speak)
//   --task s2s   : audio in -> transcript -> reply -> WAV out   (full loop)

#include "log.h"
#include "common.h"
#include "arg.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"

#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "audio-common.h"
#include "asr-silero.h"
#include "asr-stream-vad.h"
#include "asr-fallback.h"
#include "asr-diarize.h"
#include "asr-vad-post.h"
#include "asr-align.h"
#include "asr-align-capture.h"
#include "asr-tdt.h"
#include "audio-stream.h"
#include "speak-stream.h"
#include "audio-capture.h"
#include "mtmd-audio-encoder.h"
#include "wavtokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <csignal>
#include <cmath>
#include <thread>
#include <fstream>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef LOG_ERR
#define LOG_ERR LOG_TEE
#endif
#ifndef LOG_INF
#define LOG_INF LOG_TEE
#endif
// ======================= end compat ========================================

enum audio_task {
    AUDIO_TASK_ASR,
    AUDIO_TASK_TTS,
    AUDIO_TASK_CHAT,
    AUDIO_TASK_S2S,
    AUDIO_TASK_STREAM,
};

struct audio_options {
    audio_task  task = AUDIO_TASK_ASR;

    std::string audio_in;         // --audio
    std::string out_file = "output.wav";
    std::string vocoder_path;     // --vocoder-model
    // Listening and speaking need DIFFERENT backbones unless the model is a true
    // omni model: an audio-understanding LLM (Voxtral) has no acoustic-code
    // tokens to emit, and a TTS backbone (OuteTTS) has no audio projector. s2s
    // therefore loads a second model in the SAME process.
    std::string tts_model;        // --tts-model
    std::string asr_prompt = "Transcribe this audio verbatim.";

    bool speak      = false;      // synthesize the reply in chat mode
    // --speak-stream: speak each sentence the moment it is complete while the reply is still being generated, into a WAV that
    // is valid after every chunk (SOURCE-READ idea: MNN omni.cpp responseInterleaved, see speak-stream.h). Implies --speak.
    bool   speak_stream     = false;
    size_t speak_min_chars  = 16;    // --speak-stream-min-chars N: shorter sentences are merged into the next one
    int    speak_gap_ms     = 100;   // --speak-stream-gap-ms N: silence between two chunks
    bool metrics    = false;
    std::string metrics_json;   // --metrics-json <path>: emit-ingestable metrics artifact
    std::string batch_text_file;    // --batch-text-file <path>: one line = one utterance
    std::string batch_manifest_out; // --batch-manifest-out <path>: per-item results as JSON
    std::string segments_out;       // --segments-out <path>: timestamped ASR segments as JSON
    bool inspect    = false;
    bool list_tasks = false;
    bool list_caps  = false;
    bool voc_gpu    = false;
    int  voc_threads = 0;

    int  sample_rate = 24000;
    int  n_fft       = 1280;
    int  n_hop       = 320;
    int  code_base   = -1;
    int  code_count  = -1;
    bool normalize   = false;   // raw OuteTTS output measured at -46 dBFS
    float peak       = 0.95f;

    // VAD / long-audio chunking -- audio.cpp's --audio-chunk-mode surface
    std::string chunk_mode = "auto"; // auto | vad | fixed | none
    audio_vad_params vad;
    bool vad_report = false;         // --vad-chunks-out style dump to stdout
    // neural VAD (whisper.cpp / Silero ggml file). Empty = the energy VAD above.
    std::string vad_model;
    asr_vad_post_params silero_post;
    // quality-gated temperature fallback (whisper.cpp / faster-whisper semantics), default OFF
    bool                asr_fallback = false;
    asr_fallback_params fb;
    bool                diarize = false; // --diarize: per-segment speaker from stereo channel energy
    // word-level timestamps (whisper.cpp DTW semantics over the LLM's attention to the audio positions)
    bool        word_ts      = false;   // --word-timestamps
    std::string words_out;              // --words-out PATH (implies --word-timestamps)
    int         dtw_l0       = 0;       // --dtw-layers A:B, inclusive; -1 = last layer
    int         dtw_l1       = -1;
    int         dtw_medfilt  = 7;       // --dtw-medfilt N (odd)
    std::string dtw_dump;               // --dtw-dump PATH: raw attention rows per window (diagnostic, binary)
    bool        dtw_speech_weight = true; // --dtw-no-speech-weight: plain whisper.cpp DTW (A/B measurement)
    // SYSTRAN's hallucination_silence_threshold over the word times above (asr-halluc.h); 0 = off
    // (BENCH-PENDING: n/a -- "faster-whisper" in this file is the project's name, no speed is claimed)
    double      halluc_silence = 0.0;   // --hallucination-silence-threshold S (implies --word-timestamps)
    int         halluc_max_decodes = 16; // --hallucination-max-decodes N, per window
    // --task stream (whisper.cpp examples/stream defaults)
    int         stream_step_ms    = 3000;
    int         stream_length_ms  = 10000;
    int         stream_keep_ms    = 200;
    int         stream_eou_frames  = 10;   // --task stream: a line ends after this many blank encoder frames following a token (0 = off)
    int         stream_endpoint_ms = 800;  // with --vad-model: a line ends after this much silence (0 = fixed step count only)
    std::string stream_backend;          // --stream-capture default|null
    int         stream_device     = -1;  // --stream-device N (index from --list-capture-devices)
    double      stream_duration_s = 0.0; // --stream-duration S, 0 = until the input ends / Ctrl+C
    double      stream_speed      = 0.0; // --stream-speed X for a file source: 0 = lockstep, X = X times real time
    bool        list_capture      = false;
};

static const char * task_name(audio_task t) {
    switch (t) {
        case AUDIO_TASK_ASR:  return "asr";
        case AUDIO_TASK_TTS:  return "tts";
        case AUDIO_TASK_CHAT: return "chat";
        case AUDIO_TASK_S2S:  return "s2s";
        case AUDIO_TASK_STREAM: return "stream";
    }
    return "?";
}

static void audio_print_usage(const char * argv0) {
    LOG_TEE("\n");
    LOG_TEE("unified audio for ik_llama.cpp -- listen, think and speak from one binary\n\n");
    LOG_TEE("usage: %s --task <asr|tts|chat|s2s> [options]\n\n", argv0);
    LOG_TEE("  listen : %s --task asr  -m model.gguf --mmproj mmproj.gguf --audio in.wav\n", argv0);
    LOG_TEE("  speak  : %s --task tts  -m outetts.gguf --vocoder-model wavtok.gguf -p \"hello\" -o out.wav\n", argv0);
    LOG_TEE("  think  : %s --task chat -m model.gguf -p \"hello\"\n", argv0);
    LOG_TEE("  loop   : %s --task s2s  -m model.gguf --mmproj mmproj.gguf --audio in.wav \\\n", argv0);
    LOG_TEE("                          --vocoder-model wavtok.gguf -o reply.wav\n");
    LOG_TEE("\naudio options:\n");
    LOG_TEE("        --task NAME            asr | tts | chat | s2s (default: asr)\n");
    LOG_TEE("        --audio PATH           input audio (wav/mp3/flac via the mtmd loader)\n");
    LOG_TEE("  -o,   --out-file PATH        output WAV path (default: output.wav)\n");
    LOG_TEE("  -vm,  --vocoder-model PATH   WavTokenizer decoder GGUF (tts / s2s / --speak)\n");
    LOG_TEE("        --tts-model PATH       speaking backbone for s2s/--speak. -m is the model that\n");
    LOG_TEE("                               LISTENS; an audio-understanding model has no acoustic-code\n");
    LOG_TEE("                               tokens, so the speaker is loaded separately in-process.\n");
    LOG_TEE("        --asr-prompt TEXT      instruction given to the audio model (default: transcribe)\n");
    LOG_TEE("        --speak                also synthesize the reply in chat/s2s mode\n");
    LOG_TEE("        --speak-stream         speak each sentence as soon as it is written (implies --speak): the WAV grows and is\n");
    LOG_TEE("                               valid after every chunk, so the first audio arrives after ONE sentence, not the whole reply\n");
    LOG_TEE("        --speak-stream-min-chars N   merge sentences shorter than N characters into the next one (default 16)\n");
    LOG_TEE("        --speak-stream-gap-ms N      silence between two spoken chunks (default 100)\n");
    LOG_TEE("        --vocoder-threads N    vocoder CPU threads (default: -t)\n");
    LOG_TEE("        --vocoder-gpu          run the vocoder on the GPU backend if available\n");
    LOG_TEE("        --tts-normalize        peak-normalise the output (raw OuteTTS output is quiet)\n");
    LOG_TEE("        --tts-peak X           normalise to this peak, implies --tts-normalize\n");
    LOG_TEE("        --tts-sample-rate N    output sample rate (default: 24000)\n");
    LOG_TEE("        --tts-n-fft N          ISTFT window (default: 1280)\n");
    LOG_TEE("        --tts-n-hop N          ISTFT hop (default: 320)\n");
    LOG_TEE("        --tts-code-base N      first acoustic-code token id (default: auto-detect)\n");
    LOG_TEE("        --tts-code-count N     codebook size (default: from the vocoder)\n");
    LOG_TEE("\nlong-audio / VAD options:\n");
    LOG_TEE("        --audio-chunk-mode M   how a file LONGER than the model window is split for asr:\n");
    LOG_TEE("                               auto  (fits -> one pass; longer -> vad, fixed if silent)\n");
    LOG_TEE("                               vad   (speech segments)  fixed (contiguous windows)\n");
    LOG_TEE("                               none  (never split)      default: auto\n");
    LOG_TEE("        --vad-threshold-db X   dB above the noise floor counted as speech (default: 9)\n");
    LOG_TEE("        --vad-chunk-max-seconds X       hard split length (default: 30)\n");
    LOG_TEE("        --vad-chunk-merge-gap-seconds X merge closer segments (default: 0.4)\n");
    LOG_TEE("        --vad-chunk-padding-seconds X   context kept around a segment (default: 0.2)\n");
    LOG_TEE("        --vad-chunks-out       print the detected speech windows and exit\n");
    LOG_TEE("        --vad-model PATH       neural Silero VAD (whisper.cpp ggml-silero-*.bin) instead of the\n");
    LOG_TEE("                               energy VAD for window planning and --vad-chunks-out. With --task stream:\n");
    LOG_TEE("                               silent steps are not decoded and a line ends on silence\n");
    LOG_TEE("        --stream-eou-frames N  stream: end a line after N blank encoder frames following a piece, no VAD\n");
    LOG_TEE("                               model needed (default: 10 = 800 ms for an 80 ms frame, 0 = off)\n");
    LOG_TEE("        --stream-endpoint-ms N stream + --vad-model: end a line after N ms of silence (default: 800,\n");
    LOG_TEE("                               0 = fixed step count only)\n");
    LOG_TEE("        --vad-silero-threshold X  speech probability threshold (default: 0.5)\n");
    LOG_TEE("        --vad-silero-min-speech-ms N / --vad-silero-min-silence-ms N / --vad-silero-pad-ms N\n");
    LOG_TEE("                               (defaults 250 / 100 / 30, as whisper.cpp)\n");
    LOG_TEE("        --diarize              stereo-energy speaker per segment (channel-separated recordings only):\n");
    LOG_TEE("                               speaker 0 = left louder by >10%%, 1 = right, ? = neither; needs a 2-channel WAV\n");
    LOG_TEE("\nquality-gated decode (asr / chat generation):\n");
    LOG_TEE("        --asr-fallback         re-decode at the next temperature when an attempt looks degenerate\n");
    LOG_TEE("                               (compression ratio, avg logprob, last-32-token entropy); off by default\n");
    LOG_TEE("        --asr-fallback-temp-inc X      temperature step of the ladder (default: 0.2)\n");
    LOG_TEE("        --asr-fallback-compression X   retry above this zlib ratio (default: 2.4)\n");
    LOG_TEE("        --asr-fallback-logprob X       retry below this mean token logprob (default: -1.0)\n");
    LOG_TEE("        --asr-fallback-entropy X       retry below this tail entropy, nats (default: 2.4)\n");
    LOG_TEE("\nintrospection:\n");
    LOG_TEE("        --list-tasks           print the task list and exit\n");
    LOG_TEE("        --list-audio-caps      print the audio capabilities of this build and exit\n");
    LOG_TEE("        --inspect              print what the loaded projector supports and exit\n");
    LOG_TEE("        --metrics              print wall time / audio length / RTF\n");
    LOG_TEE("        --metrics-json PATH    write those metrics as JSON for benchmark_emit\n");
    LOG_TEE("        --batch-text-file PATH one line = one utterance; model loads ONCE\n");
    LOG_TEE("        --batch-manifest-out P per-item results as JSON\n");
    LOG_TEE("        --segments-out PATH    timestamped asr segments as JSON. One entry per\n");
    LOG_TEE("                               window, so it is only finer than the whole file when\n");
    LOG_TEE("                               --audio-chunk-mode actually split it.\n");
    LOG_TEE("\ntransducer asr (no text model):\n");
    LOG_TEE("        --task asr --mmproj X.gguf (or -m X.gguf) where X is a Parakeet-TDT GGUF from\n");
    LOG_TEE("                               examples/audio/convert-parakeet-tdt-to-gguf.py: the FastConformer encoder\n");
    LOG_TEE("                               runs in mtmd and the Token-and-Duration Transducer decodes greedily here;\n");
    LOG_TEE("                               word times come from the duration head (--word-timestamps / --words-out)\n");
    LOG_TEE("\nlive transcription (--task stream, transducer GGUF via --mmproj):\n");
    LOG_TEE("        --stream-step MS       new audio per step (default: 3000)\n");
    LOG_TEE("        --stream-length MS     window length; a line is committed every length/step - 1 steps (default: 10000)\n");
    LOG_TEE("        --stream-keep MS       audio carried from one line into the next (default: 200)\n");
    LOG_TEE("        --stream-capture B     capture backend: default | null (silence, for tests)\n");
    LOG_TEE("        --stream-device N      capture device index (default: the system default)\n");
    LOG_TEE("        --stream-duration S    stop after S seconds of audio (default: until Ctrl+C / end of input)\n");
    LOG_TEE("        --audio PATH           read a file instead of a device; --stream-speed X plays it at X times\n");
    LOG_TEE("                               real time through the capture ring (default 0: one step at a time)\n");
    LOG_TEE("        --list-capture-devices print the capture devices (of --stream-capture) and exit\n");
    LOG_TEE("\nword timestamps (asr):\n");
    LOG_TEE("        --word-timestamps      per-word start/end/probability: DTW (whisper.cpp semantics) over the\n");
    LOG_TEE("                               attention of each generated token to the audio positions. Turns flash\n");
    LOG_TEE("                               attention off (the weights are read from the softmax).\n");
    LOG_TEE("        --words-out PATH       write the words as JSON (implies --word-timestamps)\n");
    LOG_TEE("        --dtw-layers A:B       layers whose heads are averaged, inclusive (default: all)\n");
    LOG_TEE("        --dtw-medfilt N        median filter width over audio positions, odd (default: 7)\n");
    LOG_TEE("        --dtw-no-speech-weight do not weight the DTW cost by the audio's energy, nor trim words to it\n");
    LOG_TEE("                               (plain whisper.cpp DTW; long silences then go to the wrong word)\n");
    LOG_TEE("        --dtw-dump PATH        append every window's raw attention rows + token pieces to PATH\n");
    LOG_TEE("                               (binary, layout in audio.cpp dtw_dump_window; for offline study)\n");
    LOG_TEE("        --hallucination-silence-threshold S\n");
    LOG_TEE("                               hallucination silence skip, SYSTRAN's hallucination_silence_threshold\n");
    LOG_TEE("                               (implies --word-timestamps): a segment whose words are improbable or\n");
    LOG_TEE("                               implausibly short/long and that sits between >= S seconds of silence is\n");
    LOG_TEE("                               dropped, and decoding resumes past the silence. WAV input only (the\n");
    LOG_TEE("                               windows are cut from the samples). default: off\n");
    LOG_TEE("        --hallucination-max-decodes N  re-decodes allowed per window (default: 16)\n");
    LOG_TEE("\n");
}

static void audio_print_caps() {
    LOG_TEE("ik_llama.cpp audio capabilities\n");
    LOG_TEE("  tasks            : asr, tts, chat, s2s, stream\n");
    LOG_TEE("  live capture     : miniaudio (OS default backend, or null for tests) -> sliding-window transducer ASR\n");
    LOG_TEE("  listen (asr)     : mtmd audio projectors -- ultravox, qwen2a, voxtral, gemma4a\n");
    LOG_TEE("  listen front-end : whisper-style log-mel (examples/mtmd/mtmd-audio.cpp)\n");
    LOG_TEE("  speak (tts)      : OuteTTS backbone (llama arch) + WavTokenizer decoder\n");
    LOG_TEE("  vocoder backend  : ggml, standalone graph (no libllama architecture required)\n");
#ifdef GGML_USE_CUDA
    LOG_TEE("  vocoder gpu      : available (--vocoder-gpu)\n");
#else
    LOG_TEE("  vocoder gpu      : not built (CPU only)\n");
#endif
    LOG_TEE("  istft            : hann window, cached-twiddle irfft, overlap-add\n");
    LOG_TEE("  wav i/o          : read PCM u8/s16/s24/s32 + IEEE float32/64, write PCM s16\n");
    LOG_TEE("  resampler        : windowed sinc (Blackman)\n");
    LOG_TEE("  vad              : energy + zero-crossing, adaptive noise floor, chunk merge/split\n");
    LOG_TEE("  long audio (asr) : --audio-chunk-mode auto|vad|fixed|none, per-window KV reset\n");
}

static void audio_print_tasks() {
    LOG_TEE("asr   speech recognition        audio in  -> transcript out\n");
    LOG_TEE("tts   speech synthesis          text in   -> WAV out\n");
    LOG_TEE("chat  text conversation         text in   -> reply text (+ WAV with --speak)\n");
    LOG_TEE("s2s   speech to speech          audio in  -> transcript -> reply -> WAV out\n");
    LOG_TEE("stream live transcription       microphone (or paced file) -> sliding-window lines\n");
}

static std::vector<char *> audio_extract_options(int argc, char ** argv, audio_options & o, bool & help) {
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
        if (a == "--task") {
            const std::string t = next(i);
            if      (t == "asr")  { o.task = AUDIO_TASK_ASR;  }
            else if (t == "tts")  { o.task = AUDIO_TASK_TTS;  }
            else if (t == "chat") { o.task = AUDIO_TASK_CHAT; }
            else if (t == "s2s")  { o.task = AUDIO_TASK_S2S;  }
            else if (t == "stream") { o.task = AUDIO_TASK_STREAM; }
            else {
                LOG_ERR("%s: unknown task '%s' (want asr|tts|chat|s2s|stream)\n", __func__, t.c_str());
                exit(1);
            }
        } else if (a == "--audio") {
            o.audio_in = next(i);
        } else if (a == "-o" || a == "--out-file" || a == "--output") {
            o.out_file = next(i);
        } else if (a == "-vm" || a == "--vocoder-model") {
            o.vocoder_path = next(i);
        } else if (a == "--tts-model") {
            o.tts_model = next(i);
        } else if (a == "--asr-prompt") {
            o.asr_prompt = next(i);
        } else if (a == "--speak") {
            o.speak = true;
        } else if (a == "--speak-stream") {
            o.speak_stream = true;
            o.speak        = true;
        } else if (a == "--speak-stream-min-chars") {
            o.speak_min_chars = (size_t) std::max(1, std::atoi(next(i)));
        } else if (a == "--speak-stream-gap-ms") {
            o.speak_gap_ms = std::max(0, std::atoi(next(i)));
        } else if (a == "--vocoder-threads") {
            o.voc_threads = std::atoi(next(i));
        } else if (a == "--vocoder-gpu") {
            o.voc_gpu = true;
        } else if (a == "--tts-normalize") {
            o.normalize = true;
        } else if (a == "--tts-peak") {
            o.peak = (float) std::atof(next(i));
            o.normalize = true;
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
        } else if (a == "--audio-chunk-mode") {
            o.chunk_mode = next(i);
        } else if (a == "--vad-threshold-db") {
            o.vad.threshold_db = (float) std::atof(next(i));
        } else if (a == "--vad-chunk-max-seconds") {
            o.vad.max_segment_s = (float) std::atof(next(i));
        } else if (a == "--vad-chunk-merge-gap-seconds") {
            o.vad.merge_gap_s = (float) std::atof(next(i));
        } else if (a == "--vad-chunk-padding-seconds") {
            o.vad.padding_ms = 1000.0f * (float) std::atof(next(i));
        } else if (a == "--vad-chunks-out") {
            o.vad_report = true;
        } else if (a == "--diarize") {
            o.diarize = true;
        } else if (a == "--asr-fallback") {
            o.asr_fallback = true;
        } else if (a == "--asr-fallback-temp-inc") {
            o.fb.temperature_inc = (float) std::atof(next(i));
        } else if (a == "--asr-fallback-compression") {
            o.fb.compression_ratio_thold = std::atof(next(i));
        } else if (a == "--asr-fallback-logprob") {
            o.fb.logprob_thold = std::atof(next(i));
        } else if (a == "--asr-fallback-entropy") {
            o.fb.entropy_thold = std::atof(next(i));
        } else if (a == "--vad-model") {
            o.vad_model = next(i);
        } else if (a == "--vad-silero-threshold") {
            o.silero_post.threshold = (float) std::atof(next(i));
        } else if (a == "--vad-silero-min-speech-ms") {
            o.silero_post.min_speech_ms = std::atoi(next(i));
        } else if (a == "--vad-silero-min-silence-ms") {
            o.silero_post.min_silence_ms = std::atoi(next(i));
        } else if (a == "--vad-silero-pad-ms") {
            o.silero_post.speech_pad_ms = std::atoi(next(i));
        } else if (a == "--metrics") {
            o.metrics = true;
        } else if (a == "--metrics-json") {
            o.metrics_json = next(i);
            o.metrics = true;
        } else if (a == "--batch-text-file") {
            o.batch_text_file = next(i);
        } else if (a == "--batch-manifest-out") {
            o.batch_manifest_out = next(i);
        } else if (a == "--segments-out") {
            o.segments_out = next(i);
        } else if (a == "--word-timestamps") {
            o.word_ts = true;
        } else if (a == "--words-out") {
            o.words_out = next(i);
            o.word_ts = true;
        } else if (a == "--dtw-layers") {
            const std::string v = next(i);
            const size_t c = v.find(':');
            if (c == std::string::npos) {
                LOG_ERR("%s: --dtw-layers wants A:B (inclusive, B=-1 for the last layer), got '%s'\n", __func__, v.c_str());
                exit(1);
            }
            o.dtw_l0 = std::atoi(v.substr(0, c).c_str());
            o.dtw_l1 = std::atoi(v.substr(c + 1).c_str());
        } else if (a == "--hallucination-silence-threshold") {
            o.halluc_silence = std::atof(next(i));
            if (!(o.halluc_silence > 0.0)) {
                LOG_ERR("%s: --hallucination-silence-threshold wants seconds > 0\n", __func__);
                exit(1);
            }
            o.word_ts = true;
        } else if (a == "--hallucination-max-decodes") {
            o.halluc_max_decodes = std::atoi(next(i));
            if (o.halluc_max_decodes < 1) {
                LOG_ERR("%s: --hallucination-max-decodes must be >= 1\n", __func__);
                exit(1);
            }
        } else if (a == "--dtw-no-speech-weight") {
            o.dtw_speech_weight = false;
        } else if (a == "--dtw-dump") {
            o.dtw_dump = next(i);
            o.word_ts = true;
        } else if (a == "--dtw-medfilt") {
            o.dtw_medfilt = std::atoi(next(i));
            if (o.dtw_medfilt < 1 || o.dtw_medfilt % 2 == 0) {
                LOG_ERR("%s: --dtw-medfilt must be odd and >= 1\n", __func__);
                exit(1);
            }
        } else if (a == "--stream-step") {
            o.stream_step_ms = std::atoi(next(i));
        } else if (a == "--stream-length") {
            o.stream_length_ms = std::atoi(next(i));
        } else if (a == "--stream-keep") {
            o.stream_keep_ms = std::atoi(next(i));
        } else if (a == "--stream-eou-frames") {
            o.stream_eou_frames = std::atoi(next(i));
        } else if (a == "--stream-endpoint-ms") {
            o.stream_endpoint_ms = std::atoi(next(i));
        } else if (a == "--stream-capture") {
            o.stream_backend = next(i);
        } else if (a == "--stream-device") {
            o.stream_device = std::atoi(next(i));
        } else if (a == "--stream-duration") {
            o.stream_duration_s = std::atof(next(i));
        } else if (a == "--stream-speed") {
            o.stream_speed = std::atof(next(i));
        } else if (a == "--list-capture-devices") {
            o.list_capture = true;
        } else if (a == "--inspect") {
            o.inspect = true;
        } else if (a == "--list-tasks") {
            o.list_tasks = true;
        } else if (a == "--list-audio-caps") {
            o.list_caps = true;
        } else {
            if (a == "-h" || a == "--help") {
                help = true;
            }
            rest.push_back(argv[i]);
        }
    }
    return rest;
}

//
// ------------------------------------------------------------------ think ---
//

// Greedy-free generation over the sampler configured on the command line.
struct gen_stats {
    double               sum_logprob = 0.0; // sum of log p(sampled token) under the RAW (temperature-free) softmax
    int                  n_tokens    = 0;
    std::vector<int32_t> tokens;
};

//
// ------------------------------------------------------- word timestamps ---
//
// An audio LLM keeps the audio embeddings in the same KV cache as the text, so the attention row of the
// query that produced a text token, restricted to the audio cells, plays the part of whisper's
// cross-attention (whisper.cpp src/whisper.cpp, whisper_exp_compute_token_level_timestamps_dtw: alignment
// heads -> median filter -> DTW). The rows are read from the kq_soft_max tensors through the eval
// callback, so flash attention must be off (main() turns it off with --word-timestamps).
//
// Cells: llama-audio clears the cache before every window and fills it in order, so KV cell i holds the
// i-th token of the window; the audio cells are [kv0, kv1). Every eval overwrites `cur` with the LAST
// query row of the batch per layer; generate_text() snapshots it right after sampling, which is the row
// whose logits produced the sampled token.
struct dtw_capture {
    bool    on      = false;    // --word-timestamps
    // the audio KV cells, the selected layers [l0, l1], n_head, the latest-eval row `cur`, `got` and the first
    // capture problem `err` live in the shared capture (asr-align-capture.h, also used by the server word path)
    asr_attn_capture cap;
    std::vector<float>   rows;  // snapshots [n_rows][n_sel*n_head][kv1-kv0]
    int     n_rows  = 0;
    bool    eog_row = false;
    std::vector<asr_align_token> toks;
    // per window
    double  t_offset = 0.0, t_dur = 0.0;
    double  sec_per_pos = 0.0;
    int     medfilt = 7;
    std::vector<asr_word> words;
    size_t  n_windows = 0;
    std::string dump;           // --dtw-dump PATH
    bool    speech_weight = true;   // --dtw-no-speech-weight turns it off (asr_align_speech_weights)
    std::vector<float> pcm;     // the window's samples (for the speech weight)
    int     pcm_sr = 0;
};
static dtw_capture g_dtw;

// --dtw-dump: one record per window, appended. Little-endian:
//   char[4] "DTW1"; int32 n_rows, n_heads, n_frames (valid audio positions), has_eog, n_tokens;
//   double sec_per_frame, t_offset, t_dur;
//   float32 attn[n_rows][n_heads][n_frames]  (heads = selected layers x heads per layer, layer-major)
//   n_tokens x { int32 id; float64 prob; int32 len; char piece[len] }
static void dtw_dump_window(const asr_align_input & in) {
    FILE * f = fopen(g_dtw.dump.c_str(), "ab");
    if (!f) {
        LOG_ERR("word-timestamps: cannot open --dtw-dump %s\n", g_dtw.dump.c_str());
        return;
    }
    const int32_t hdr[5] = { (int32_t) (in.tokens.size() + (in.has_eog_row ? 1 : 0)), in.n_heads, in.n_frames,
                             in.has_eog_row ? 1 : 0, (int32_t) in.tokens.size() };
    const double  d[3]   = { in.sec_per_frame, in.t_offset, in.t_end - in.t_offset };
    fwrite("DTW1", 1, 4, f);
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(d, sizeof(d), 1, f);
    fwrite(in.attn.data(), sizeof(float), in.attn.size(), f);
    for (const auto & t : in.tokens) {
        const int32_t id = t.id, len = (int32_t) t.piece.size();
        fwrite(&id, sizeof(id), 1, f);
        fwrite(&t.prob, sizeof(t.prob), 1, f);
        fwrite(&len, sizeof(len), 1, f);
        fwrite(t.piece.data(), 1, (size_t) len, f);
    }
    fclose(f);
}

// The eval callback is the shared asr_attn_capture_cb (asr-align-capture.h): user_data = &g_dtw.cap. The audio
// cells arrive through g_dtw.cap.arm(); before every audio cell is in the cache no row is wanted (MEASURED
// 2026-10-19: the first prompt batch on Voxtral is [32 kv, 4 q], an error there lost every word of the run).

// Arms the capture for one window: finds the audio cells among the tokenized chunks (KV starts at n_past0).
static bool dtw_arm(const mtmd_input_chunks * chunks, llama_pos n_past0, double t_offset, double t_dur) {
    int64_t cell = n_past0, a0 = -1, a1 = -1;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        const int64_t n = (int64_t) mtmd_input_chunk_get_n_tokens(c);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            if (a0 < 0) { a0 = cell; } else if (a1 != cell) {
                LOG_ERR("%s: audio chunks are separated by text; word timestamps off for this window\n", __func__);
                g_dtw.cap.armed = false;
                return false;
            }
            a1 = cell + n;
        }
        cell += n;
    }
    if (a0 < 0 || a1 <= a0) {
        g_dtw.cap.armed = false;
        return false;
    }
    g_dtw.t_offset = t_offset;
    g_dtw.t_dur    = t_dur;
    if (!g_dtw.cap.arm(a0, a1)) {   // sizes cur/got for [a0, a1) and clears the per-window error
        return false;
    }
    g_dtw.rows.clear();
    g_dtw.toks.clear();
    g_dtw.n_rows  = 0;
    g_dtw.eog_row = false;
    return true;
}

// Called by generate_text() right after a token was sampled (before it is decoded).
static void dtw_snapshot(llama_context * lctx, llama_model * model, llama_token id) {
    if (!g_dtw.cap.armed || g_dtw.eog_row) {
        return;
    }
    for (uint8_t g : g_dtw.cap.got) {
        if (!g) {
            if (g_dtw.cap.err.empty()) { g_dtw.cap.err = "no attention row was captured for a selected layer (flash attention on, or a fused path?)"; }
            return;
        }
    }
    const bool eog = llama_token_is_eog(model, id);
    g_dtw.rows.insert(g_dtw.rows.end(), g_dtw.cap.cur.begin(), g_dtw.cap.cur.end());
    // every row must come from the pass that produced THIS token: a layer that is not captured again before the
    // next sample would otherwise hand its stale row to the next token
    std::fill(g_dtw.cap.got.begin(), g_dtw.cap.got.end(), 0);
    g_dtw.n_rows++;
    if (eog) {
        g_dtw.eog_row = true;
        return;
    }
    asr_align_token at;
    at.id    = id;
    at.piece = common_token_to_piece(lctx, id);
    const float * lg = llama_get_logits_ith(lctx, -1);
    const int     nv = llama_n_vocab(model);
    if (lg && id >= 0 && id < nv) {
        float mx = lg[0];
        for (int k = 1; k < nv; ++k) { mx = std::max(mx, lg[k]); }
        double z = 0.0;
        for (int k = 0; k < nv; ++k) { z += std::exp((double) (lg[k] - mx)); }
        at.prob = std::exp((double) (lg[id] - mx)) / z;
    }
    g_dtw.toks.push_back(std::move(at));
}

// Turns the window's rows into words (appended to g_dtw.words). Disarms.
static void dtw_finish_window() {
    if (!g_dtw.cap.armed) {
        return;
    }
    g_dtw.cap.armed = false;
    g_dtw.n_windows++;
    if (!g_dtw.cap.err.empty()) {
        LOG_ERR("word-timestamps: %s\n", g_dtw.cap.err.c_str());
        return;
    }
    if (g_dtw.toks.empty()) {
        return;
    }
    const int n_pos   = g_dtw.cap.n_audio();
    const int n_valid = asr_align_valid_frames(n_pos, g_dtw.t_dur, g_dtw.sec_per_pos);
    const int n_hd    = g_dtw.cap.n_sel() * g_dtw.cap.n_head;
    asr_align_input in;
    in.tokens        = g_dtw.toks;
    in.n_heads       = n_hd;
    in.n_frames      = n_valid;
    in.has_eog_row   = g_dtw.eog_row;
    in.sec_per_frame = g_dtw.sec_per_pos;
    in.t_offset      = g_dtw.t_offset;
    in.t_end         = g_dtw.t_offset + g_dtw.t_dur;
    in.medfilt_width = g_dtw.medfilt;
    in.attn.resize((size_t) g_dtw.n_rows * n_hd * n_valid);
    for (int r = 0; r < g_dtw.n_rows; ++r) {
        for (int h = 0; h < n_hd; ++h) {
            const float * src = g_dtw.rows.data() + ((size_t) r * n_hd + h) * (size_t) n_pos;
            std::copy(src, src + n_valid, in.attn.data() + ((size_t) r * n_hd + h) * (size_t) n_valid);
        }
    }
    if (!g_dtw.pcm.empty()) {
        in.speech_w = asr_align_speech_weights(g_dtw.pcm.data(), g_dtw.pcm.size(), g_dtw.pcm_sr, g_dtw.sec_per_pos, n_valid);
    }
    if (!g_dtw.dump.empty()) {
        dtw_dump_window(in);
    }
    asr_align_result res;
    std::string err;
    if (!asr_align_words(in, res, &err)) {
        LOG_ERR("word-timestamps: alignment failed: %s\n", err.c_str());
        return;
    }
    int n_speech = 0;
    for (float x : in.speech_w) { n_speech += x > 0.5f; }
    LOG_INF("word-timestamps: window %.3f-%.3f s, %d token(s)%s, %d of %d audio positions (%.4f s each, %s), %d head(s) -> %d word(s)\n",
            g_dtw.t_offset, g_dtw.t_offset + g_dtw.t_dur, (int) g_dtw.toks.size(), g_dtw.eog_row ? " + eog" : "",
            n_valid, n_pos, g_dtw.sec_per_pos,
            in.speech_w.empty() ? "no speech weight" : (std::to_string(n_speech) + " speech-weighted").c_str(),
            n_hd, (int) res.words.size());
    g_dtw.words.insert(g_dtw.words.end(), res.words.begin(), res.words.end());
}

// Seconds per audio position of this projector, MEASURED from the token counts it gives two probe lengths
// (tokenizing runs the mel front end only; nothing is encoded). An equal count for 10 s and 20 s means
// fixed-length padded chunks (whisper style): the chunk length is then found by bisection.
static int dtw_probe_tokens(mtmd_context * mctx, double seconds) {
    const int sr = mtmd_get_audio_sample_rate(mctx);
    std::vector<float> z((size_t) (seconds * sr), 0.0f);
    mtmd_bitmap * b = mtmd_bitmap_init_from_audio(z.size(), z.data());
    if (!b) { return -1; }
    const std::string marker = mtmd_default_marker();
    mtmd_input_text txt;
    txt.text          = marker.c_str();
    txt.text_len      = marker.size();
    txt.add_special   = false;
    txt.parse_special = true;
    mtmd_input_chunks * ch = mtmd_input_chunks_init();
    const mtmd_bitmap * bm[1] = { b };
    int n = -1;
    if (mtmd_tokenize(mctx, ch, &txt, bm, 1) == 0) {
        n = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(ch); ++i) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(ch, i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_AUDIO) { n += (int) mtmd_input_chunk_get_n_tokens(c); }
        }
    }
    mtmd_input_chunks_free(ch);
    mtmd_bitmap_free(b);
    return n;
}

static double dtw_measure_sec_per_pos(mtmd_context * mctx) {
    const int na = dtw_probe_tokens(mctx, 10.0);
    const int nb = dtw_probe_tokens(mctx, 20.0);
    double chunk_s = 0.0;
    if (na > 0 && na == nb) {
        double lo = 20.0, hi = 120.0;
        if (dtw_probe_tokens(mctx, hi) <= na) { return 0.0; }
        while (hi - lo > 0.01) {
            const double mid = 0.5 * (lo + hi);
            if (dtw_probe_tokens(mctx, mid) > na) { hi = mid; } else { lo = mid; }
        }
        chunk_s = std::round(lo * 10.0) / 10.0; // the last length that still fits one chunk, to 0.1 s
    }
    const double spp = asr_align_seconds_per_position(10.0, na, 20.0, nb, chunk_s);
    LOG_INF("word-timestamps: projector gives %d / %d audio positions for 10 / 20 s%s%.1f s -> %.5f s per position\n",
            na, nb, chunk_s > 0.0 ? " (fixed chunks of " : " (proportional", chunk_s > 0.0 ? chunk_s : 0.0, spp);
    return spp;
}

// --speak-stream: called with the reply so far after every piece (final = false) and once more when the text is complete
// (final = true). Set in main() only when --speak-stream is on. A quality-gated attempt (stats != nullptr, the
// --asr-fallback ladder) is never fed: its text may still be thrown away, and speech cannot be taken back.
static std::function<void(const std::string &, bool)> g_speak_feed;

static std::string generate_text(llama_context * lctx,
                                 llama_model   * model,
                                 common_sampler * smpl,
                                 llama_pos & n_past,
                                 int n_predict,
                                 bool stream_to_stdout,
                                 gen_stats * stats = nullptr) {
    std::string out;
    llama_batch batch = llama_batch_init(1, 0, 1);
    const std::function<void(const std::string &, bool)> * feed = (stats == nullptr && g_speak_feed) ? &g_speak_feed : nullptr;

    for (int i = 0; i < n_predict; ++i) {
        const llama_token id = common_sampler_sample(smpl, lctx, -1);
        if (g_dtw.cap.armed) {
            dtw_snapshot(lctx, model, id);
        }
        if (stats && !llama_token_is_eog(model, id)) {
            const float * lg = llama_get_logits_ith(lctx, -1);
            const int     nv = llama_n_vocab(model);
            if (lg && id >= 0 && id < nv) {
                float mx = lg[0];
                for (int k = 1; k < nv; ++k) { mx = std::max(mx, lg[k]); }
                double z = 0.0;
                for (int k = 0; k < nv; ++k) { z += std::exp((double) (lg[k] - mx)); }
                stats->sum_logprob += (double) (lg[id] - mx) - std::log(z);
                stats->n_tokens++;
                stats->tokens.push_back((int32_t) id);
            }
        }
        common_sampler_accept(smpl, lctx, id, true);

        if (llama_token_is_eog(model, id)) {
            break;
        }

        const std::string piece = common_token_to_piece(lctx, id);
        out += piece;
        if (stream_to_stdout) {
            printf("%s", piece.c_str());
            fflush(stdout);
        }
        if (feed) {
            (*feed)(out, false);   // may run the speaker for a finished sentence before the next token is decoded
        }

        common_batch_clear(batch);
        common_batch_add(batch, id, n_past, { 0 }, true);
        n_past++;

        if (llama_decode(lctx, batch)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            break;
        }
    }

    llama_batch_free(batch);
    if (stream_to_stdout) {
        printf("\n");
    }
    if (feed) {
        (*feed)(out, true);
    }
    return out;
}

// Quality-gated decode (--asr-fallback). `prefill` (re)builds the prompt state from `n_past0` and sets
// n_past, `model` is sampled once per temperature of the ladder with a FRESH sampler, and the verdict
// of asr_fallback_check() (compression ratio, mean logprob, tail entropy) decides about the next try.
// An LLM audio model has no no-speech token, so that gate is off and no_speech_prob stays 0.
// Text is never streamed while attempts are running (it could not be taken back); the chosen one is
// printed at the end. Every attempt logs a line, so a run proves the fallback fired.
static const asr_fallback_params *     g_fb = nullptr;      // set in main() with --asr-fallback
static const common_params_sampling * g_fb_base = nullptr;

static std::string generate_with_fallback(llama_context * lctx,
                                          llama_model   * model,
                                          const common_params_sampling & base,
                                          asr_fallback_params fp,
                                          llama_pos n_past0,
                                          const std::function<bool(llama_pos &)> & prefill,
                                          int n_predict,
                                          bool stream_to_stdout,
                                          llama_pos & n_past,
                                          bool * ok) {
    fp.check_no_speech = false;
    *ok = true;
    std::vector<std::string> texts;
    std::vector<llama_pos>   ends;
    const auto res = asr_run_fallback(fp, [&](float temp) {
        asr_decode_attempt att;
        att.temperature = temp;
        llama_kv_cache_seq_rm(lctx, 0, n_past0, -1);
        llama_pos np = n_past0;
        if (!prefill(np)) { *ok = false; texts.push_back(""); ends.push_back(np); return att; }
        common_params_sampling sp = base;
        sp.temp = temp;
        common_sampler * sm = common_sampler_init(model, sp);
        gen_stats st;
        att.text = generate_text(lctx, model, sm, np, n_predict, /*stream*/ false, &st);
        common_sampler_free(sm);
        att.n_tokens     = st.n_tokens;
        att.avg_logprob  = st.n_tokens > 0 ? st.sum_logprob / st.n_tokens : 0.0;
        att.tail_entropy = asr_tail_entropy(st.tokens);
        texts.push_back(att.text);
        ends.push_back(np);
        return att;
    });
    for (size_t i = 0; i < res.attempts.size(); ++i) {
        const auto & a = res.attempts[i];
        const auto & v = res.verdicts[i];
        LOG_INF("asr-fallback: attempt %d T=%.2f tokens=%d avg_logprob=%.3f ratio=%.3f tail_entropy=%.3f -> %s%s%s%s%s\n",
                (int) i, (double) a.temperature, a.n_tokens, a.avg_logprob, v.compression_ratio, a.tail_entropy,
                v.needs_fallback ? "RETRY" : "accept",
                (v.reasons & ASR_FAIL_COMPRESSION) ? " [compression]" : "",
                (v.reasons & ASR_FAIL_LOGPROB) ? " [logprob]" : "",
                (v.reasons & ASR_FAIL_ENTROPY) ? " [entropy]" : "",
                v.is_silence ? " [silence]" : "");
    }
    LOG_INF("asr-fallback: chose attempt %d of %d%s\n", res.chosen, (int) res.attempts.size(), res.exhausted ? " (ladder exhausted)" : "");
    if (res.chosen < 0 || res.chosen >= (int) texts.size()) { n_past = n_past0; return ""; }
    // The KV cache holds the LAST attempt. When another one was chosen, replay it (fresh sampler, same
    // temperature, same seed => same tokens) so the cache matches the text the caller will continue from.
    if (res.chosen != (int) res.attempts.size() - 1) {
        llama_kv_cache_seq_rm(lctx, 0, n_past0, -1);
        llama_pos np = n_past0;
        if (prefill(np)) {
            common_params_sampling sp = base;
            sp.temp = res.attempts[res.chosen].temperature;
            common_sampler * sm = common_sampler_init(model, sp);
            const std::string again = generate_text(lctx, model, sm, np, n_predict, false, nullptr);
            common_sampler_free(sm);
            if (again != texts[res.chosen]) {
                LOG_WRN("asr-fallback: replay of attempt %d differs from the first run; the KV cache follows the replay\n", res.chosen);
            }
            n_past = np;
        } else {
            *ok = false;
        }
    } else {
        n_past = ends.back();
    }
    const std::string & out = texts[res.chosen];
    if (stream_to_stdout) { printf("%s\n", out.c_str()); fflush(stdout); }
    return out;
}

//
// ----------------------------------------------------------------- listen ---
//

// Feeds ONE already-decoded audio bitmap (plus an instruction) through the mtmd
// audio projector and decodes the model's answer -- i.e. transcription.
//
// The bitmap is a PARAMETER rather than something this function loads, because the
// long-audio path (--audio-chunk-mode) has to run the same body N times over N
// windows of the SAME file. The caller owns `bmp` and frees it; this function never
// does, so a chunk loop can hold one bitmap per window without a double free.
static bool run_listen_bitmap(mtmd_context * mctx,
                       llama_context * lctx,
                       llama_model   * model,
                       common_sampler * smpl,
                       common_chat_templates * tmpls,
                       bool use_jinja,
                       mtmd_bitmap * bmp,
                       const std::string & instruction,
                       int n_batch,
                       int n_predict,
                       llama_pos & n_past,
                       std::string & transcript) {

    // The instruction MUST go through the model's own chat template. Voxtral is
    // Mistral-based and expects [INST] ... [/INST]; fed a bare prompt it encodes
    // the audio correctly and then emits EOG on the first sampled token, giving
    // an EMPTY transcript that is indistinguishable from a projector failure.
    // ROOT-CAUSE measured here 2026-08-31: 187 audio tokens encoded and decoded,
    // zero text generated, because no instruction framing surrounded the media
    // marker. The fix is prompt formatting, not a change to the projector.
    const std::string marker  = mtmd_default_marker();
    const std::string content = marker + "\n" + instruction;

    common_chat_msg msg;
    msg.role    = "user";
    msg.content = content;

    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages              = { msg };
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja             = use_jinja;

    const auto formatted = common_chat_templates_apply(tmpls, tmpl_inputs);
    const std::string formatted_prompt = formatted.prompt.str(); // common_chat_prompt (upstream); text.text points into this string

    mtmd_input_text text;
    text.text          = formatted_prompt.c_str();
    // text_len was ADDED to mtmd_input_text upstream and this member-wise fill left it
    // INDETERMINATE. mtmd's tokenizer does `input_text.assign(text->text, text->text_len)`
    // (mtmd.cpp:1144), so the prompt arrived empty or truncated, the media marker was not found
    // in it, and tokenize failed with "number of media markers in text (0) does not match number
    // of bitmaps (1)" -- an error that reads like a prompt-construction bug several lines above.
    // MEASURED: this is what made every ASR check in test_audio_roundtrip.py fail (llama-audio
    // rc=1, 4 of 4 runs) while llama-tts succeeded, on a tree that built with zero errors.
    text.text_len      = formatted_prompt.size();
    text.add_special   = true;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[1] = { bmp };

    const int32_t rc = mtmd_tokenize(mctx, chunks, &text, bitmaps, 1);
    if (rc != 0) {
        LOG_ERR("%s: mtmd_tokenize() failed with %d\n", __func__, (int) rc);
        mtmd_input_chunks_free(chunks);
        return false;
    }

    // --word-timestamps: the window's duration is the bitmap's own (the caller set g_dtw.t_offset)
    const double win_dur = (double) mtmd_bitmap_get_n_bytes(bmp) / sizeof(float) / (double) std::max(1, mtmd_get_audio_sample_rate(mctx));
    const float * bmp_pcm = (g_dtw.on && g_dtw.speech_weight && mtmd_bitmap_is_audio(bmp))
                          ? (const float *) mtmd_bitmap_get_data(bmp) : nullptr;
    if (bmp_pcm) {
        // an audio bitmap holds float32 PCM at the model's sample rate (mtmd_bitmap_init_from_audio)
        const float * p = bmp_pcm;
        g_dtw.pcm.assign(p, p + mtmd_bitmap_get_n_bytes(bmp) / sizeof(float));
        g_dtw.pcm_sr = mtmd_get_audio_sample_rate(mctx);
    } else {
        g_dtw.pcm.clear();
    }

    if (g_fb) {
        const llama_pos n_past0 = n_past;
        bool fb_ok = true;
        transcript = generate_with_fallback(lctx, model, *g_fb_base, *g_fb, n_past0,
            [&](llama_pos & np) {
                np = n_past0;
                llama_pos nn = np;
                // re-armed on every prefill, so the rows kept are those of the attempt whose KV the cache holds
                // at the end (the last one, or the replay of the chosen one)
                if (g_dtw.on) { dtw_arm(chunks, n_past0, g_dtw.t_offset, win_dur); }
                const int32_t rc2 = mtmd_helper_eval_chunks(mctx, lctx, chunks, np, 0, n_batch, true, &nn);
                if (rc2 != 0) { LOG_ERR("%s: mtmd_helper_eval_chunks() failed with %d\n", __func__, (int) rc2); return false; }
                np = nn;
                return true;
            }, n_predict, /*stream*/ true, n_past, &fb_ok);
        mtmd_input_chunks_free(chunks);
        if (g_dtw.on) { dtw_finish_window(); }
        return fb_ok;
    }

    if (g_dtw.on) {
        dtw_arm(chunks, n_past, g_dtw.t_offset, win_dur);
    }
    llama_pos new_n_past = n_past;
    const int32_t erc = mtmd_helper_eval_chunks(mctx, lctx, chunks, n_past, /*seq_id*/ 0,
                                                n_batch, /*logits_last*/ true, &new_n_past);
    mtmd_input_chunks_free(chunks);

    if (erc != 0) {
        LOG_ERR("%s: mtmd_helper_eval_chunks() failed with %d\n", __func__, (int) erc);
        return false;
    }
    n_past = new_n_past;

    transcript = generate_text(lctx, model, smpl, n_past, n_predict, /*stream*/ true);
    if (g_dtw.on) {
        dtw_finish_window();
    }
    return true;
}

// Single-shot path: decode the whole file through mtmd's own loader (which handles
// wav/mp3/flac), then run the body once. Byte-identical to the pre-chunking behaviour.
static bool run_listen(mtmd_context * mctx,
                       llama_context * lctx,
                       llama_model   * model,
                       common_sampler * smpl,
                       common_chat_templates * tmpls,
                       bool use_jinja,
                       const std::string & audio_path,
                       const std::string & instruction,
                       int n_batch,
                       int n_predict,
                       llama_pos & n_past,
                       std::string & transcript) {

    // signature changed upstream: (ctx, fname) -> (ctx, fname, placeholder, opt), returning
    // mtmd_helper_bitmap_wrapper{bitmap, video_ctx} rather than mtmd_bitmap *.
    // placeholder=false: this path wants the REAL decoded audio, not a metadata stand-in.
    // video_ctx stays null for audio input; only .bitmap is used, and the mtmd_bitmap_free(bmp)
    // below still owns exactly that pointer.
    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mctx, audio_path.c_str(),
                                                          false,
                                                          mtmd_helper_init_opt_default()).bitmap;
    if (!bmp) {
        LOG_ERR("%s: failed to load audio '%s'\n", __func__, audio_path.c_str());
        return false;
    }
    if (!mtmd_bitmap_is_audio(bmp)) {
        LOG_ERR("%s: '%s' did not decode as audio\n", __func__, audio_path.c_str());
        mtmd_bitmap_free(bmp);
        return false;
    }

    const bool ok = run_listen_bitmap(mctx, lctx, model, smpl, tmpls, use_jinja, bmp,
                                      instruction, n_batch, n_predict, n_past, transcript);
    mtmd_bitmap_free(bmp);
    return ok;
}

//
// ------------------------------------------------- listen: long audio -------
//
// audio.cpp exposes `--audio-chunk-mode auto|fixed|vad|none` on essentially every
// ASR family, because an audio LLM has a fixed clip limit and a file longer than it
// is not a smaller problem -- it is a DIFFERENT one. This fork parsed that flag and
// documented it in --help and then never read it, so long input silently went
// through as one over-long window. This implements it.
//
// Windowing policy, stated because "auto" is a decision and not a detail:
//   none  : never split. One window, whatever the length. Escape hatch.
//   fixed : contiguous windows of --vad-chunk-max-seconds, no gaps, no analysis.
//   vad   : speech-energy segments from the VAD we already have (merge/pad/split
//           are the existing audio_vad_params knobs).
//   auto  : single window when the file already fits the limit -- so short input is
//           byte-identical to the previous behaviour -- otherwise vad, and fixed if
//           the VAD returns nothing (silence-gated input must not transcribe to "").
// Stereo (2-channel) WAV -> two 16 kHz channel buffers for --diarize.
static bool audio_load_stereo16k(const std::string & path, std::vector<float> & L, std::vector<float> & R, std::string & err) {
    audio_wav w;
    if (!audio_wav_read(path, w, err)) { return false; }
    if (w.channels != 2) { err = "--diarize needs a 2-channel WAV (got " + std::to_string(w.channels) + " channel(s))"; return false; }
    const size_t n = w.n_frames();
    L.resize(n); R.resize(n);
    for (size_t i = 0; i < n; ++i) { L[i] = w.samples[2 * i]; R[i] = w.samples[2 * i + 1]; }
    if (w.sample_rate != 16000) { L = audio_resample(L, w.sample_rate, 16000); R = audio_resample(R, w.sample_rate, 16000); }
    return true;
}

static std::string audio_segment_speaker(const std::vector<float> & L, const std::vector<float> & R, const audio_segment & s) {
    size_t i0 = (size_t) std::max(0.0, s.t0_s * 16000.0), i1 = (size_t) std::max(0.0, s.t1_s * 16000.0);
    if (i1 == 0 || i1 > L.size()) { i1 = L.size(); }   // the single-pass "whole file" segment may carry no end time
    i0 = std::min(i0, i1);
    return asr_stereo_speaker(L.data(), R.data(), i0, std::min(i1, R.size()));
}

// Speech segments from the neural Silero VAD (--vad-model). The probabilities come from
// asr_silero_probs(), the segmentation from asr_vad_postprocess() (hysteresis, min speech /
// silence, max-speech split at the last >= 98 ms silence, merge, padding); max speech is the
// same --vad-chunk-max-seconds that caps the energy VAD windows. Prints an always-on line so a
// run can prove the neural path fired.
static std::vector<audio_segment> audio_vad_segments_silero(const asr_silero_model & sm,
                                                            const std::vector<float> & mono,
                                                            int32_t sample_rate,
                                                            asr_vad_post_params pp,
                                                            float max_segment_s) {
    std::vector<audio_segment> out;
    if (sample_rate != 16000) {
        LOG_WRN("audio_vad_segments_silero: Silero VAD is 16 kHz only (got %d Hz)\n", (int) sample_rate);
        return out;
    }
    pp.window_samples = sm.n_window;
    pp.sample_rate    = sample_rate;
    pp.max_speech_s   = max_segment_s;
    const std::vector<float> probs = asr_silero_probs(sm, mono.data(), mono.size());
    const auto spans = asr_vad_postprocess(probs.data(), probs.size(), pp);
    LOG_INF("vad: silero v%s fired -- %d probabilities -> %d speech segment(s)\n", sm.version.c_str(),
            (int) probs.size(), (int) spans.size());
    for (const auto & sp : spans) {
        audio_segment s;
        s.i0 = (size_t) sp.start;
        s.i1 = std::min((size_t) sp.end, mono.size());
        s.t0_s = (double) s.i0 / sample_rate;
        s.t1_s = (double) s.i1 / sample_rate;
        if (s.i1 > s.i0) { out.push_back(s); }
    }
    return out;
}

static std::vector<audio_segment> audio_plan_windows(const std::vector<float> & mono,
                                                     int32_t sample_rate,
                                                     const std::string & mode,
                                                     const audio_vad_params & vp,
                                                     const char ** why,
                                                     const asr_silero_model * sil = nullptr,
                                                     const asr_vad_post_params * silp = nullptr) {
    auto audio_vad_segments = [&](const std::vector<float> & m, int32_t sr, const audio_vad_params & p) {
        if (sil) { return audio_vad_segments_silero(*sil, m, sr, *silp, p.max_segment_s); }
        return ::audio_vad_segments(m, sr, p);
    };
    std::vector<audio_segment> out;
    const size_t n = mono.size();
    const double dur = sample_rate > 0 ? (double) n / sample_rate : 0.0;

    auto whole = [&]() {
        audio_segment s;
        s.i0 = 0; s.i1 = n; s.t0_s = 0.0; s.t1_s = dur;
        out.assign(1, s);
    };
    auto fixed = [&]() {
        out.clear();
        const size_t w = (size_t) std::max(1.0f, vp.max_segment_s) * (size_t) std::max(1, sample_rate);
        for (size_t i = 0; i < n; i += w) {
            audio_segment s;
            s.i0 = i;
            s.i1 = std::min(n, i + w);
            s.t0_s = (double) s.i0 / sample_rate;
            s.t1_s = (double) s.i1 / sample_rate;
            out.push_back(s);
        }
    };

    if (mode == "none") { whole(); *why = "none: forced single window"; return out; }
    if (mode == "fixed") { fixed(); *why = "fixed windows"; return out; }
    if (mode == "vad")   {
        out = audio_vad_segments(mono, sample_rate, vp);
        if (out.empty()) { fixed(); *why = "vad found no speech -> fixed windows"; }
        else             { *why = "vad segments"; }
        return out;
    }
    // auto
    if (dur <= (double) vp.max_segment_s) { whole(); *why = "auto: fits the window, single pass"; return out; }
    out = audio_vad_segments(mono, sample_rate, vp);
    if (out.empty()) { fixed(); *why = "auto: long, vad found no speech -> fixed windows"; }
    else             { *why = "auto: long -> vad segments"; }
    return out;
}

//
// ------------------------------------------------------------------ speak ---
//

// Shared with llama-tts: text -> acoustic codes -> waveform. Declared here
// rather than exported from tts.cpp so that the two binaries stay independent.
// The prompt text comes from speak_tts_text (speak-stream.cpp), where test-speak-stream covers it.
static bool run_speak(llama_context * ctx_ttc,
                      llama_model   * model_ttc,
                      common_sampler * smpl,
                      wavtokenizer  * voc,
                      const audio_options & o,
                      const std::string & text,
                      int n_predict,
                      int n_threads,
                      std::vector<float> & audio) {

    const int code_base = o.code_base >= 0
        ? o.code_base
        : ([&]() {
              const llama_tokens t = common_tokenize(ctx_ttc, "<|0|>", false, true);
              return t.size() == 1 ? (int) t[0] : -1;
          })();

    const int code_count = o.code_count > 0 ? o.code_count : wavtokenizer_n_vocab(voc);

    if (code_base < 0) {
        LOG_ERR("%s: could not detect the acoustic code token base; pass --tts-code-base N\n", __func__);
        return false;
    }

    llama_tokens prompt;
    auto add = [&](const std::string & s, bool add_special) {
        const llama_tokens t = common_tokenize(ctx_ttc, s, add_special, true);
        prompt.insert(prompt.end(), t.begin(), t.end());
    };

    add("<|im_start|>\n", true);
    const std::string said = speak_tts_text(text);
    if (said.empty()) {
        // an empty prompt makes the model voice whatever it likes (CJK-only text, "...")
        LOG_ERR("%s: nothing this speaker can pronounce in \"%s\" (it reads a-z, digits and accented Latin letters)\n",
                __func__, text.c_str());
        return false;
    }
    add("<|text_start|>", false);
    add(said, false);
    add("<|text_end|>\n", false);
    // Closes the OuteTTS v1 prompt so the model starts emitting acoustic codes for
    // the text above. Without it the model continues in TEXT space and speaks
    // whatever context precedes it -- see the MEASURED note in examples/tts/tts.cpp.
    add("<|audio_start|>\n", false);

    llama_batch batch = llama_batch_init(std::max<size_t>(prompt.size(), 1), 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx_ttc, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed on the TTS prompt\n", __func__);
        llama_batch_free(batch);
        return false;
    }
    llama_synchronize(ctx_ttc);

    std::vector<int32_t> acoustic;
    const int code_budget = speak_code_budget(said, n_predict);   // proportional to the text, never above n_predict
    bool ended = false;
    {
        int n_past = batch.n_tokens;
        for (int i = 0; i < code_budget; ++i) {
            const llama_token id = common_sampler_sample(smpl, ctx_ttc, -1);
            common_sampler_accept(smpl, ctx_ttc, id, true);
            if (llama_token_is_eog(model_ttc, id)) {
                ended = true;
                break;
            }
            if (id >= code_base && id < code_base + code_count) {
                acoustic.push_back((int32_t) (id - code_base));
            }

            common_batch_clear(batch);
            common_batch_add(batch, id, n_past, { 0 }, true);
            n_past++;

            if (llama_decode(ctx_ttc, batch)) {
                LOG_ERR("%s: llama_decode() failed while generating codes\n", __func__);
                break;
            }
        }
    }
    llama_batch_free(batch);

    if (!ended) {
        LOG_WRN("%s: no end token within the %d-code budget for %zu bytes of text (n_predict %d): the audio may be cut off or contain runaway codes\n",
                __func__, code_budget, said.size(), n_predict);
    }

    if (acoustic.empty()) {
        LOG_ERR("%s: the backbone produced no acoustic codes\n", __func__);
        return false;
    }

    std::string err;
    std::vector<float> embd;
    if (!wavtokenizer_decode(voc, acoustic.data(), (int) acoustic.size(), embd, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return false;
    }

    audio = audio_embd_to_wave(embd.data(), (int) acoustic.size(),
                               wavtokenizer_n_embd_out(voc),
                               o.n_fft, o.n_hop, std::max(1, n_threads));
    return !audio.empty();
}

//
// ------------------------------------------------------------------- main ---
//

// JSON string escaping. NOT optional on Windows: a model path like
// models\\x.gguf emits \\J, which is not a legal JSON escape, and the
// artifact then fails json.load with "Invalid \\escape". MEASURED on the first real run.
static std::string json_escape(const std::string & in) {
    std::string o;
    o.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '\"': o += "\\\""; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char) c;
                }
        }
    }
    return o;
}

// ---- transducer ASR (Parakeet-TDT): the encoder runs in mtmd (encoder-only parakeet), the Token-and-Duration
// Transducer decodes here (asr-tdt.h). No text model is loaded, so this route never generates free text: an
// encoder frame either emits a vocabulary token or a blank that skips ahead by the duration head's choice.
// the transducer's two halves, loaded from one GGUF and checked against each other
struct tdt_runtime {
    asr_tdt_model dec;
    std::unique_ptr<mtmd_audio_encoder, void (*)(mtmd_audio_encoder *)> enc{nullptr, mtmd_audio_encoder_free};
    int sr = 0;
};

static bool tdt_runtime_open(const std::string & path, const common_params & params, tdt_runtime & rt, std::string & err) {
    if (!asr_tdt_load(path, rt.dec, err)) {
        return false;
    }
    mtmd_audio_encoder_params ep;
    ep.n_threads = std::max(1, params.n_threads);
    ep.use_gpu   = params.mmproj_use_gpu;
    rt.enc.reset(mtmd_audio_encoder_init(path.c_str(), ep, &err));
    if (!rt.enc) {
        return false;
    }
    mtmd_audio_encoder * enc = rt.enc.get();
    const asr_tdt_model & dec = rt.dec;
    if (mtmd_audio_encoder_n_embd(enc) != dec.n_enc) {
        err = "encoder width " + std::to_string(mtmd_audio_encoder_n_embd(enc)) + " != decoder's encoder_hidden " + std::to_string(dec.n_enc);
        return false;
    }
    rt.sr = mtmd_audio_encoder_sample_rate(enc);
    if (std::fabs(mtmd_audio_encoder_hop(enc) / (double) rt.sr - dec.frame_s) > 1e-6 || mtmd_audio_encoder_subsampling(enc) != dec.subsampling) {
        char buf[256];
        snprintf(buf, sizeof(buf), "the encoder's frame rate (hop %d at %d Hz, subsampling %d) does not match the decoder's (%.4f s, %d)",
                 mtmd_audio_encoder_hop(enc), rt.sr, mtmd_audio_encoder_subsampling(enc), dec.frame_s, dec.subsampling);
        err = buf;
        return false;
    }
    return true;
}

static int run_asr_tdt(const std::string & path, const common_params & params, const audio_options & o) {
    const int64_t t0 = ggml_time_us();
    std::string err;
    tdt_runtime rt;
    if (!tdt_runtime_open(path, params, rt, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return 1;
    }
    const asr_tdt_model & dec = rt.dec;
    mtmd_audio_encoder * enc = rt.enc.get();
    const int sr = rt.sr;
    std::vector<float> mono;
    if (!audio_load_mono(o.audio_in, sr, mono, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return 1;
    }
    const double audio_s = (double) mono.size() / sr;
    const int64_t t_load = ggml_time_us();
    LOG_INF("%s: transducer ASR: vocab %d + blank, durations %d, %.3f s of audio\n", __func__, dec.n_vocab, (int) dec.durations.size(), audio_s);

    std::vector<float> states;
    int n_frames = 0;
    if (!mono.empty() && !mtmd_audio_encoder_encode(enc, mono.data(), mono.size(), states, n_frames)) {
        LOG_ERR("%s: the audio encoder failed\n", __func__);
        return 1;
    }
    const int64_t t_enc = ggml_time_us();
    std::vector<asr_tdt_token> toks;
    if (!asr_tdt_greedy(dec, states.data(), n_frames, toks)) {
        LOG_ERR("%s: TDT decode failed\n", __func__);
        return 1;
    }
    const int64_t t_dec = ggml_time_us();
    const std::string text = asr_tdt_text(dec, toks);
    std::vector<asr_word> words = asr_tdt_words(dec, toks);
    // the encoder pads the last frame, so a final token's duration can reach past the end of the audio
    for (auto & w : words) {
        w.end   = std::min(w.end, audio_s);
        w.start = std::min(w.start, w.end);
    }

    printf("%s\n", text.c_str());
    fflush(stdout);
    LOG_INF("\n%s: transcript: %s\n", __func__, text.c_str());
    LOG_INF("%s: %d encoder frame(s), %d token(s), %d word(s)\n", __func__, n_frames, (int) toks.size(), (int) words.size());
    if (o.word_ts || !o.words_out.empty()) {
        LOG_INF("%s: word-timestamps (TDT duration head): %d word(s)\n", __func__, (int) words.size());
        for (const auto & w : words) {
            LOG_INF("  [%8.3f -> %8.3f] p=%.3f %s\n", w.start, w.end, w.probability, w.word.c_str());
        }
    }
    if (!o.words_out.empty()) {
        FILE * wf = fopen(o.words_out.c_str(), "wb");
        if (!wf) {
            LOG_ERR("%s: could not open %s for words\n", __func__, o.words_out.c_str());
            return 1;
        }
        fprintf(wf, "{\n  \"audio\": \"%s\",\n", json_escape(o.audio_in).c_str());
        fprintf(wf, "  \"method\": \"tdt-duration\",\n");
        fprintf(wf, "  \"seconds_per_encoder_frame\": %.6f,\n", dec.frame_seconds());
        fprintf(wf, "  \"text\": \"%s\",\n", json_escape(text).c_str());
        fprintf(wf, "  \"words\": [\n");
        for (size_t k = 0; k < words.size(); ++k) {
            const auto & w = words[k];
            fprintf(wf, "    {\"word\": \"%s\", \"start\": %.3f, \"end\": %.3f, \"probability\": %.6f}%s\n",
                    json_escape(w.word).c_str(), w.start, w.end, w.probability, k + 1 < words.size() ? "," : "");
        }
        fprintf(wf, "  ]\n}\n");
        fclose(wf);
        LOG_INF("%s: wrote %d word(s) to %s\n", __func__, (int) words.size(), o.words_out.c_str());
    }
    if (o.metrics) {
        const double wall = (t_dec - t0) / 1e6;
        LOG_INF("%s: metrics: load %.3f s, encode %.3f s, decode %.3f s, wall %.3f s, audio %.3f s, RTF %.4f\n", __func__,
                (t_load - t0) / 1e6, (t_enc - t_load) / 1e6, (t_dec - t_enc) / 1e6, wall, audio_s, audio_s > 0 ? wall / audio_s : 0.0);
    }
    return 0;
}

// ---- live transcription (--task stream): whisper.cpp's examples/stream sliding window (audio-stream.h) over the
// transducer above. The audio comes from a capture device (audio-capture.h, miniaudio) or, for reproducible runs,
// from a file: --stream-speed 0 hands the file over one step at a time (lockstep), X > 0 plays it into the same ring
// the microphone uses at X times real time, so falling behind and dropping audio behave as they would live.
static volatile sig_atomic_t g_stream_stop = 0;
static void stream_on_sigint(int) { g_stream_stop = 1; }

static int run_stream_tdt(const std::string & path, const common_params & params, const audio_options & o) {
    std::string err;
    tdt_runtime rt;
    if (!tdt_runtime_open(path, params, rt, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return 1;
    }
    const int sr = rt.sr;
    audio_stream_params sp;
    sp.step_ms     = o.stream_step_ms;
    sp.length_ms   = o.stream_length_ms;
    sp.keep_ms     = o.stream_keep_ms;
    sp.sample_rate = sr;
    audio_stream_window win(sp);
    const int n_step = win.n_step();

    // sources
    const bool from_file = !o.audio_in.empty();
    std::vector<float> file_pcm;
    size_t file_pos = 0;
    if (from_file && !audio_load_mono(o.audio_in, sr, file_pcm, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        return 1;
    }
    const bool lockstep = from_file && !(o.stream_speed > 0.0);
    audio_ring ring((size_t) std::max(win.n_len(), 2 * n_step) * 2);
    std::atomic<bool> feeder_done{false};
    std::thread feeder;
    audio_capture * cap = nullptr;
    if (from_file && !lockstep) {
        feeder = std::thread([&]() {
            const size_t blk = (size_t) sr / 100;   // 10 ms, like a capture callback
            const auto t_begin = std::chrono::steady_clock::now();
            for (size_t p = 0; p < file_pcm.size() && !g_stream_stop; p += blk) {
                const size_t n = std::min(blk, file_pcm.size() - p);
                const auto due = t_begin + std::chrono::microseconds((int64_t) (1e6 * (double) (p + n) / sr / o.stream_speed));
                std::this_thread::sleep_until(due);
                ring.push(file_pcm.data() + p, n);
            }
            feeder_done = true;
        });
    } else if (!from_file) {
        cap = audio_capture_open(o.stream_backend, o.stream_device, sr, ring, err);
        if (!cap) {
            LOG_ERR("%s: %s\n", __func__, err.c_str());
            return 1;
        }
        LOG_INF("%s: capture: backend %s, device '%s', %d Hz mono\n", __func__, audio_capture_backend_name(cap).c_str(),
                audio_capture_device_name(cap).c_str(), sr);
        if (!audio_capture_start(cap, err)) {
            LOG_ERR("%s: %s\n", __func__, err.c_str());
            audio_capture_close(cap);
            return 1;
        }
    }
    const uint64_t max_samples = o.stream_duration_s > 0.0 ? (uint64_t) (o.stream_duration_s * sr) : 0;
    LOG_INF("%s: step %.1f s / length %.1f s / keep %.1f s, a line every %d step(s); source %s%s\n", __func__,
            n_step / (double) sr, win.n_len() / (double) sr, win.n_keep() / (double) sr, win.n_new_line(),
            from_file ? o.audio_in.c_str() : "capture device",
            lockstep ? " (lockstep)" : (from_file ? " (paced)" : ""));
    std::signal(SIGINT, stream_on_sigint);

    std::vector<std::string> lines;
    std::string partial;
    bool partial_open = false;
    uint64_t taken = 0, dropped = 0;
    int n_steps = 0;
    double t_compute = 0.0;
    std::vector<float> fresh, states;

    // Stream VAD (row S69-ENGREV-WHISPERCPP-STREAMVAD-2026-10-27-01): with --vad-model the Silero state is carried across
    // steps (asr_silero_stream), a step with no speech and no open line is not decoded, and a line ends on silence.
    asr_silero_model stream_sil;
    asr_silero_stream stream_sil_state;
    asr_stream_vad_params vad_p;
    asr_stream_vad_state  vad_s;
    bool stream_vad = false;
    if (!o.vad_model.empty()) {
        std::string serr;
        if (sr != 16000) {
            LOG_WRN("%s: stream VAD needs 16 kHz audio (model rate %d Hz): every step is decoded\n", __func__, sr);
        } else if (!asr_silero_load(o.vad_model, stream_sil, &serr)) {
            LOG_ERR("%s: --vad-model: %s\n", __func__, serr.c_str());
            return 1;
        } else {
            asr_silero_stream_reset(stream_sil_state);
            vad_p.threshold           = o.silero_post.threshold;
            vad_p.endpoint_silence_ms = std::max(0, o.stream_endpoint_ms);
            vad_p.prob_ms             = stream_sil.n_window * 1000 / sr;
            stream_vad = true;
            LOG_INF("%s: stream vad: silero v%s, threshold %.2f, line ends after %d ms of silence\n", __func__,
                    stream_sil.version.c_str(), vad_p.threshold, vad_p.endpoint_silence_ms);
        }
    }
    std::vector<float> step_probs;
    while (!g_stream_stop) {
        if (max_samples && taken >= max_samples) {
            break;
        }
        if (lockstep) {
            if (file_pos >= file_pcm.size()) {
                break;
            }
            const size_t n = std::min((size_t) n_step, file_pcm.size() - file_pos);
            fresh.assign(file_pcm.begin() + file_pos, file_pcm.begin() + file_pos + n);
            file_pos += n;
        } else {
            // wait for one step of new audio (or the end of a file feed)
            while (!g_stream_stop && ring.pending() < (size_t) n_step && !(feeder_done && ring.pending() > 0)) {
                if (feeder_done && ring.pending() == 0) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (g_stream_stop || (feeder_done && ring.pending() == 0)) {
                break;
            }
            const size_t had = ring.take(fresh);
            if (had > 2 * (size_t) n_step) {
                // the decoder is slower than real time: keep the newest step, report what was skipped
                const size_t lost = had - (size_t) n_step;
                dropped += lost;
                fresh.erase(fresh.begin(), fresh.end() - n_step);
                LOG_WRN("%s: cannot keep up, dropped %.2f s of audio\n", __func__, lost / (double) sr);
            }
        }
        taken += fresh.size();
        const std::vector<float> & w = win.next(fresh);
        asr_stream_vad_decision vd;   // without --vad-model: decode every step, never endpoint (as before)
        if (stream_vad) {
            step_probs.clear();
            asr_silero_stream_feed(stream_sil, stream_sil_state, fresh.data(), fresh.size(), step_probs);
            vd = asr_stream_vad_step(vad_s, vad_p, step_probs.data(), step_probs.size());
            if (!vd.decode) {
                // silence with no open line: nothing to transcribe; drop the silent audio from the window
                LOG_INF("%s: [%7.2f -> %7.2f] silent step, not decoded\n", __func__, win.t0(), win.t1());
                win.end_line();
                continue;
            }
        }
        const int64_t tc0 = ggml_time_us();
        std::string text;
        bool tok_eou = false;
        if (w.size() >= (size_t) sr / 10) {   // under 100 ms the encoder has no frame worth decoding
            int n_frames = 0;
            std::vector<asr_tdt_token> toks;
            if (!mtmd_audio_encoder_encode(rt.enc.get(), w.data(), w.size(), states, n_frames) ||
                !asr_tdt_greedy(rt.dec, states.data(), n_frames, toks)) {
                LOG_ERR("%s: transcription of the window [%.2f, %.2f) failed\n", __func__, win.t0(), win.t1());
                break;
            }
            text = asr_tdt_text(rt.dec, toks);
            asr_stream_eou_params eou_p;
            eou_p.eou_blank_frames = std::max(0, o.stream_eou_frames);
            tok_eou = asr_stream_eou(eou_p, toks.data(), toks.size(), n_frames);
        }
        t_compute += (ggml_time_us() - tc0) / 1e6;
        ++n_steps;
        LOG_INF("%s: [%7.2f -> %7.2f] partial: %s\n", __func__, win.t0(), win.t1(), text.c_str());
        partial = text;
        partial_open = true;
        // a line ends on a silence endpoint (stream VAD) or on the fixed step count
        bool line_end = false;
        if (vd.endpoint || tok_eou) {
            vad_s.line_has_speech = false;
            win.end_line();
            line_end = true;
        } else if (win.done_step()) {
            line_end = true;
            vad_s.line_has_speech = false;   // the line is closed: silence after it is not decoded
        }
        if (line_end) {
            printf("[%7.2f --> %7.2f] %s\n", win.t0(), win.t1(), text.c_str());
            fflush(stdout);
            if (!text.empty()) {
                lines.push_back(text);
            }
            partial_open = false;
        }
    }
    if (stream_vad) {
        LOG_INF("%s: stream vad: %lld step(s) decoded, %lld silent step(s) skipped, %lld line(s) ended on silence\n", __func__,
                (long long) vad_s.n_decode, (long long) vad_s.n_skip, (long long) vad_s.n_endpoint);
    }
    if (partial_open) {   // the input ended inside a line: its last hypothesis is the line
        printf("[%7.2f --> %7.2f] %s\n", win.t0(), win.t1(), partial.c_str());
        fflush(stdout);
        if (!partial.empty()) {
            lines.push_back(partial);
        }
    }
    if (cap) {
        audio_capture_stop(cap);
    }
    g_stream_stop = 1;   // ends a paced feeder early when --stream-duration stopped the loop
    if (feeder.joinable()) {
        feeder.join();
    }
    std::string all;
    for (const auto & l : lines) {
        all += (all.empty() ? "" : " ") + l;
    }
    LOG_INF("\n%s: transcript: %s\n", __func__, all.c_str());
    LOG_INF("%s: %d step(s), %d line(s), %.3f s of audio taken, %.3f s dropped, captured %llu sample(s)\n", __func__, n_steps,
            (int) lines.size(), taken / (double) sr, dropped / (double) sr, (unsigned long long) ring.total());
    if (o.metrics) {
        LOG_INF("%s: metrics: compute %.3f s for %.3f s of audio, RTF %.4f\n", __func__, t_compute, taken / (double) sr,
                taken > 0 ? t_compute / (taken / (double) sr) : 0.0);
    }
    if (cap) {
        audio_capture_close(cap);
    }
    return 0;
}

int main(int argc, char ** argv) {
    // common_log is asynchronous: without a flush at exit the last log lines (e.g. the asr-fallback verdicts) are
    // sometimes never printed, which made any check on them flaky.
    std::atexit([] { common_log_flush(common_log_main()); });
    common_params params;
    audio_options o;

    params.n_predict = 512;

    bool want_help = false;
    std::vector<char *> rest = audio_extract_options(argc, argv, o, want_help);

    if (o.list_caps) {
        audio_print_caps();
        return 0;
    }
    if (o.list_tasks) {
        audio_print_tasks();
        return 0;
    }
    if (o.list_capture) {
        std::vector<std::string> names;
        std::string err;
        if (!audio_capture_list(o.stream_backend, names, err)) {
            LOG_ERR("%s: %s\n", __func__, err.c_str());
            return 1;
        }
        printf("capture devices (%s): %d\n", o.stream_backend.empty() ? "default" : o.stream_backend.c_str(), (int) names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            printf("  [%d] %s\n", (int) i, names[i].c_str());
        }
        return 0;
    }
    if (want_help) {
        audio_print_usage(argv[0]);
    }

    // --vad-chunks-out only needs the DSP layer, so serve it before touching
    // any model -- this is also the cheapest way to sanity-check audio input.
    if (o.vad_report) {
        if (o.audio_in.empty()) {
            LOG_ERR("%s: --vad-chunks-out needs --audio <file.wav>\n", __func__);
            return 1;
        }
        std::string err;
        std::vector<float> mono;
        if (!audio_load_mono(o.audio_in, 16000, mono, err)) {
            LOG_ERR("%s: %s\n", __func__, err.c_str());
            return 1;
        }
        asr_silero_model sil;
        std::vector<audio_segment> segs;
        if (!o.vad_model.empty()) {
            std::string serr;
            if (!asr_silero_load(o.vad_model, sil, &serr)) {
                LOG_ERR("%s: --vad-model: %s\n", __func__, serr.c_str());
                return 1;
            }
            segs = audio_vad_segments_silero(sil, mono, 16000, o.silero_post, o.vad.max_segment_s);
        } else {
            segs = audio_vad_segments(mono, 16000, o.vad);
        }
        printf("vad: %s\n", o.vad_model.empty() ? "energy" : "silero");
        printf("audio: %s\n", o.audio_in.c_str());
        printf("duration: %.3f s at 16000 Hz (mono)\n", (double) mono.size() / 16000.0);
        printf("segments: %d\n", (int) segs.size());
        std::vector<float> dl, dr;
        if (o.diarize) {
            std::string derr;
            if (!audio_load_stereo16k(o.audio_in, dl, dr, derr)) {
                LOG_ERR("%s: %s\n", __func__, derr.c_str());
                return 1;
            }
            LOG_INF("diarize: stereo-energy fired (%zu frames per channel)\n", dl.size());
        }
        for (size_t i = 0; i < segs.size(); ++i) {
            printf("  [%02d] %8.3f -> %8.3f  (%.3f s)", (int) i,
                   segs[i].t0_s, segs[i].t1_s, segs[i].t1_s - segs[i].t0_s);
            if (o.diarize) { printf("  speaker %s", audio_segment_speaker(dl, dr, segs[i]).c_str()); }
            printf("\n");
        }
        return 0;
    }

    // A Parakeet-TDT GGUF given as --mmproj is the whole ASR model, so there is no text model for -m; the common
    // parser still demands one (common/arg.cpp "--model is required"), so hand it the same file. Nothing loads it
    // as a text model: run_asr_tdt below returns before any llama model load.
    std::string tdt_model_arg;
    {
        bool has_model = false;
        std::string mmproj;
        for (size_t i = 1; i < rest.size(); ++i) {
            const std::string a = rest[i] ? rest[i] : "";
            if (a == "-m" || a == "--model") { has_model = true; }
            if ((a == "-mm" || a == "--mmproj") && i + 1 < rest.size() && rest[i + 1]) { mmproj = rest[i + 1]; }
        }
        if (!has_model && !mmproj.empty() && (o.task == AUDIO_TASK_ASR || o.task == AUDIO_TASK_STREAM) && asr_tdt_is_tdt_file(mmproj)) {
            tdt_model_arg = mmproj;
        }
    }
    static char arg_m[] = "-m";
    if (!tdt_model_arg.empty()) {
        rest.push_back(arg_m);
        rest.push_back(tdt_model_arg.data());
    }
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_MTMD)) {
        audio_print_usage(argv[0]);
        return 1;
    }

    // --inspect only loads the projector and reports what it supports, so it
    // must NOT be gated behind the per-task input requirements below.
    const bool need_listen  = !o.inspect && (o.task == AUDIO_TASK_ASR || o.task == AUDIO_TASK_S2S);
    const bool need_speak   = !o.inspect && (o.task == AUDIO_TASK_TTS || o.task == AUDIO_TASK_S2S || o.speak);

    if (o.task == AUDIO_TASK_STREAM) {
        for (const std::string & p : { params.mmproj.path, params.model.path }) {
            if (!p.empty() && asr_tdt_is_tdt_file(p)) {
                llama_backend_init();
                const int rc = run_stream_tdt(p, params, o);
                llama_backend_free();
                return rc;
            }
        }
        // ROW S69-WHISPERCPP-STREAM-LLM-ROUTE-2026-10-20-01: the LLM listen path re-prefills the whole window per step
        LOG_ERR("%s: --task stream needs a transducer GGUF (Parakeet-TDT, via --mmproj); the audio-LLM route is not wired for streaming\n", __func__);
        return 1;
    }
    if (need_listen && o.audio_in.empty()) {
        LOG_ERR("%s: task '%s' needs --audio <file>\n", __func__, task_name(o.task));
        return 1;
    }
    // a transducer GGUF (Parakeet-TDT) is a whole ASR model: no text model, no projector into one
    if (o.task == AUDIO_TASK_ASR && !o.inspect) {
        for (const std::string & p : { params.mmproj.path, params.model.path }) {
            if (!p.empty() && asr_tdt_is_tdt_file(p)) {
                if (o.halluc_silence > 0.0 || o.asr_fallback) {
                    LOG_WRN("%s: transducer ASR: --hallucination-silence-threshold / --asr-fallback apply to the LLM route only, ignored\n", __func__);
                }
                llama_backend_init();
                const int rc = run_asr_tdt(p, params, o);
                llama_backend_free();
                return rc;
            }
        }
    }
    if (need_listen && params.mmproj.path.empty()) {
        LOG_ERR("%s: task '%s' needs --mmproj <audio-projector.gguf>\n", __func__, task_name(o.task));
        LOG_ERR("%s: (any mtmd audio projector: ultravox, qwen2a, voxtral, gemma4a)\n", __func__);
        return 1;
    }
    if (need_speak && o.vocoder_path.empty()) {
        LOG_ERR("%s: task '%s' needs --vocoder-model <wavtokenizer.gguf>\n", __func__, task_name(o.task));
        return 1;
    }
    if (o.speak_stream && ((o.task != AUDIO_TASK_CHAT && o.task != AUDIO_TASK_S2S) || !o.batch_text_file.empty())) {
        LOG_ERR("%s: --speak-stream speaks the reply of --task chat or s2s as it is written; it has nothing to stream for task '%s'%s\n",
                __func__, task_name(o.task), o.batch_text_file.empty() ? "" : " with --batch-text-file");
        return 1;
    }
    // -m is the LISTENING backbone for asr/s2s, so speaking needs its own.
    if (need_speak && o.task != AUDIO_TASK_TTS && o.tts_model.empty()) {
        LOG_ERR("%s: task '%s' needs --tts-model <outetts.gguf> as well as -m\n", __func__, task_name(o.task));
        LOG_ERR("%s: -m is the model that LISTENS (it carries the audio projector); an\n", __func__);
        LOG_ERR("%s: audio-understanding model has no acoustic-code tokens to SPEAK with,\n", __func__);
        LOG_ERR("%s: so the speaking backbone is loaded separately in this same process.\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    const int64_t t_start_us = ggml_time_us();

    // Audio durations for the metrics block. Tracked separately because ASR is rated against the
    // INPUT it consumed and TTS against the OUTPUT it produced -- dividing wall time by the wrong
    // one silently reports a plausible number for the wrong quantity.
    double in_audio_s  = 0.0;
    double out_audio_s = 0.0;
    double stream_first_audio_s = -1.0;   // --speak-stream: seconds from start until the first chunk was in the output file

    // --word-timestamps reads the attention weights from the kq_soft_max tensors: flash attention keeps them
    // inside one fused op, and attn_max_batch > 0 may split the plain path into per-step pieces, so both go
    // for this run only (opt-in flag; default runs are untouched). RESEARCH-DONE: whisper.cpp makes the same
    // trade the other way round, src/whisper.cpp:3813-3814 (engines/whisper.cpp @ 6e4ab854) "dtw_token_timestamps
    // is not supported with flash_attn - disabling"; the weights DTW needs exist only on the non-flash path.
    // DISABLE-REVERSIBLE: --word-timestamps (absent = flash attention as configured).
    if (o.word_ts && need_listen) {
        if (params.flash_attn) {
            LOG_INF("%s: --word-timestamps: flash attention off for this run (the attention weights are read from the softmax)\n", __func__);
        }
        params.flash_attn        = false;
        params.attn_max_batch    = 0;
        params.cb_eval           = asr_attn_capture_cb;
        params.cb_eval_user_data = &g_dtw.cap;
        g_dtw.on      = true;
        g_dtw.medfilt = o.dtw_medfilt;
        g_dtw.dump    = o.dtw_dump;
        g_dtw.speech_weight = o.dtw_speech_weight;
    }

    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * lctx  = init->context();
    if (!model || !lctx) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }
    if (g_dtw.on) {
        const int n_layer = llama_model_n_layer(model);
        g_dtw.cap.n_head = llama_model_n_head(model);
        g_dtw.cap.l0 = std::max(0, o.dtw_l0);
        g_dtw.cap.l1 = (o.dtw_l1 < 0 || o.dtw_l1 >= n_layer) ? n_layer - 1 : o.dtw_l1;
        if (g_dtw.cap.l0 > g_dtw.cap.l1 || g_dtw.cap.n_head <= 0) {
            LOG_ERR("%s: --dtw-layers %d:%d does not fit a model with %d layers\n", __func__, o.dtw_l0, o.dtw_l1, n_layer);
            return 1;
        }
        LOG_INF("%s: --word-timestamps on: layers %d..%d of %d, %d heads each, median filter %d\n", __func__,
                g_dtw.cap.l0, g_dtw.cap.l1, n_layer, g_dtw.cap.n_head, g_dtw.medfilt);
    }

    if (o.asr_fallback) {
        g_fb      = &o.fb;
        g_fb_base = &params.sampling;
        LOG_INF("%s: --asr-fallback on: ladder from T=%.2f step %.2f, compression>%.2f, logprob<%.2f, entropy<%.2f\n", __func__,
                (double) params.sampling.temp, (double) o.fb.temperature_inc, o.fb.compression_ratio_thold, o.fb.logprob_thold, o.fb.entropy_thold);
        o.fb.temperature = params.sampling.temp > 0.0f ? params.sampling.temp : 0.0f;
    }

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to init the sampler\n", __func__);
        return 1;
    }

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);

    // ---- projector (listen) ----------------------------------------------
    mtmd_context * mctx = nullptr;
    if (need_listen || o.inspect) {
        if (params.mmproj.path.empty()) {
            LOG_ERR("%s: --inspect needs --mmproj <projector.gguf>\n", __func__);
            return 1;
        }
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu       = params.mmproj_use_gpu;
        mp.print_timings = o.metrics;
        mp.n_threads     = std::max(1, params.n_threads);

        mctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mp);
        if (!mctx) {
            LOG_ERR("%s: failed to load the projector '%s'\n", __func__, params.mmproj.path.c_str());
            return 1;
        }
    }

    if (o.inspect) {
        LOG_TEE("projector: %s\n", params.mmproj.path.c_str());
        LOG_TEE("  vision support : %s\n", mtmd_support_vision(mctx) ? "yes" : "no");
        LOG_TEE("  audio support  : %s\n", mtmd_support_audio(mctx)  ? "yes" : "no");
        if (mtmd_support_audio(mctx)) {
            // mtmd_get_audio_bitrate was renamed mtmd_get_audio_sample_rate upstream (mtmd.h:147).
            // The label is corrected with it: the value is in Hz and always was a SAMPLE RATE,
            // not a bitrate -- the old name printed "bitrate ... Hz", which is a unit mismatch.
            LOG_TEE("  audio sample rate : %d Hz\n", mtmd_get_audio_sample_rate(mctx));
        }
        LOG_TEE("  media marker   : %s\n", mtmd_default_marker());
        mtmd_free(mctx);
        return 0;
    }

    if (mctx && !mtmd_support_audio(mctx)) {
        LOG_ERR("%s: '%s' is not an AUDIO projector -- llama-audio needs one of "
                "ultravox / qwen2a / voxtral / gemma4a\n", __func__, params.mmproj.path.c_str());
        return 1;
    }
    if (g_dtw.on && mctx) {
        g_dtw.sec_per_pos = dtw_measure_sec_per_pos(mctx);
        if (!(g_dtw.sec_per_pos > 0.0)) {
            LOG_ERR("%s: --word-timestamps: could not measure the projector's seconds per audio position\n", __func__);
            return 1;
        }
    }

    // ---- speaking backbone, when it is not the same model as -m -----------
    // Same process, second llama context. This is what makes listen->think->speak
    // a single binary rather than a shell pipeline between two of them.
    llama_model    * model_tts = model;
    llama_context  * ctx_tts   = lctx;
    common_sampler * smpl_tts  = smpl;
    common_init_result_ptr init_tts;

    if (need_speak && !o.tts_model.empty()) {
        common_params tp = params;
        tp.model.path    = o.tts_model;
        tp.mmproj        = {};          // the TTS backbone has no projector
        tp.embedding     = false;
        tp.n_ctx         = 8192;
        tp.n_batch       = 8192;
        tp.sampling.top_k = 4;

        LOG_INF("%s: loading the speaking backbone '%s'\n", __func__, o.tts_model.c_str());
        init_tts  = common_init_from_params(tp);
        model_tts = init_tts->model();
        ctx_tts   = init_tts->context();
        if (!model_tts || !ctx_tts) {
            LOG_ERR("%s: failed to load the speaking backbone\n", __func__);
            return 1;
        }
        smpl_tts = common_sampler_init(model_tts, tp.sampling);
        if (!smpl_tts) {
            LOG_ERR("%s: failed to init the speaking sampler\n", __func__);
            return 1;
        }
    }

    // ---- vocoder (speak) --------------------------------------------------
    wavtokenizer * voc = nullptr;
    if (need_speak) {
        wavtokenizer_params wp;
        wp.use_gpu   = o.voc_gpu;
        wp.n_threads = o.voc_threads > 0 ? o.voc_threads : std::max(1, params.n_threads);
        wp.verbose   = true;

        std::string err;
        voc = wavtokenizer_init_from_file(o.vocoder_path, wp, err);
        if (!voc) {
            LOG_ERR("%s: %s\n", __func__, err.c_str());
            return 1;
        }
    }

    llama_pos n_past = 0;
    std::string transcript;
    std::string reply;

    // ---- --speak-stream: speak every finished sentence while the reply is still being generated ----------------------------
    // The two models live in one process, so the loop is: decode a token, and when a sentence is complete stop decoding, run
    // the speaker for that sentence, append it to the WAV, then go on decoding. That is sequential (no overlap), so total
    // time does not shrink; what shrinks is the time to the FIRST audible sample: one sentence of LLM plus one of TTS
    // instead of all of both. Overlapping the two on separate threads is the next step and competes for the same GPU.
    struct speak_stream_state {
        speak_wav_sink sink;
        std::string    err;
        size_t         spoken   = 0;        // bytes of the current text already given to the speaker
        int            n_chunks = 0;
        int            n_skipped = 0;       // chunks with nothing to say ("...", "***")
        double         audio_s  = 0.0;
        double         first_audio_s = -1.0; // seconds from process start until the first chunk was in the file
        double         tts_s    = 0.0;      // time spent in the speaker
        float          gain     = 1.0f;     // --tts-normalize: fixed by the first chunk so later chunks keep their relative level
        bool           gain_set = false;
        bool           failed   = false;
    } sst;
    std::function<void(const std::string &, bool)> speak_feed_fn;
    // declared OUTSIDE the if: speak_feed_fn (copied into g_speak_feed) captures these by reference and is called from generate_text
    // long after the if block ended; block-local captures dangled there (MEASURED: access violation 0xC0000005 at the first sentence)
    speak_cut_params cp;
    std::function<void(const std::string &)> say_chunk;
    if (o.speak_stream) {
        cp.min_chars = o.speak_min_chars;
        say_chunk = [&](const std::string & chunk) {
            if (sst.failed) { return; }
            if (!speak_has_speech(chunk)) { sst.n_skipped++; return; }
            if (speak_tts_text(chunk).empty()) {
                // letters, but none this speaker reads (a CJK-only sentence): skip it, the rest of the reply still plays
                LOG_WRN("%s: speak-stream: nothing pronounceable in \"%s\", skipped\n", __func__, chunk.c_str());
                sst.n_skipped++;
                return;
            }
            const int64_t t0 = ggml_time_us();
            llama_kv_cache_clear(ctx_tts);       // every chunk is an utterance of its own: positions restart at 0
            common_sampler_reset(smpl_tts);
            std::vector<float> audio;
            if (!run_speak(ctx_tts, model_tts, smpl_tts, voc, o, chunk, params.n_predict, params.n_threads, audio)) {
                LOG_ERR("%s: speak-stream: the speaker failed on \"%s\"\n", __func__, chunk.c_str());
                sst.failed = true;
                return;
            }
            if (o.normalize && !sst.gain_set) {
                float mx = 0.0f;
                for (float v : audio) { mx = std::max(mx, std::fabs(v)); }
                sst.gain     = mx > 1e-6f ? o.peak / mx : 1.0f;
                sst.gain_set = true;
            }
            if (!sst.sink.is_open() && !sst.sink.open(o.out_file, o.sample_rate, sst.err)) {
                LOG_ERR("%s: speak-stream: %s\n", __func__, sst.err.c_str());
                sst.failed = true;
                return;
            }
            if (sst.n_chunks > 0 && o.speak_gap_ms > 0) {
                const std::vector<float> gap((size_t) o.sample_rate * (size_t) o.speak_gap_ms / 1000, 0.0f);
                if (!sst.sink.append(gap, 1.0f, sst.err)) { LOG_ERR("%s: speak-stream: %s\n", __func__, sst.err.c_str()); sst.failed = true; return; }
                sst.audio_s += (double) gap.size() / o.sample_rate;
            }
            if (!sst.sink.append(audio, sst.gain, sst.err)) {
                LOG_ERR("%s: speak-stream: %s\n", __func__, sst.err.c_str());
                sst.failed = true;
                return;
            }
            const double now_s  = (ggml_time_us() - t_start_us) / 1e6;
            const double took_s = (ggml_time_us() - t0) / 1e6;
            const double clip_s = (double) audio.size() / o.sample_rate;
            if (sst.first_audio_s < 0.0) { sst.first_audio_s = now_s; }
            sst.tts_s   += took_s;
            sst.audio_s += clip_s;
            sst.n_chunks++;
            LOG_INF("%s: speak-stream: chunk %d \"%s\" -> %.2f s of audio in %.2f s, in the file at %.3f s\n", __func__,
                    sst.n_chunks, chunk.c_str(), clip_s, took_s, now_s);
        };
        speak_feed_fn = [&](const std::string & text, bool final) {
            for (;;) {
                const size_t e = speak_sentence_end(text, sst.spoken, final, cp);
                if (e == std::string::npos) { break; }
                std::string chunk = text.substr(sst.spoken, e - sst.spoken);
                const size_t a = chunk.find_first_not_of(" \t\r\n");
                const size_t b = chunk.find_last_not_of(" \t\r\n");
                chunk = a == std::string::npos ? std::string() : chunk.substr(a, b - a + 1);
                sst.spoken = e;
                say_chunk(chunk);
            }
            if (final) { sst.spoken = 0; }
        };
        g_speak_feed = speak_feed_fn;
    }

    // ---- listen -----------------------------------------------------------
    // INPUT duration for the metrics block, read from the WAV header rather than inferred.
    // Cheap and only done when metrics were asked for.
    // --segments-out needs the duration too, so a single-pass run can still report the span it
    // covered rather than an open-ended segment.
    if ((o.metrics || !o.segments_out.empty()) && !o.audio_in.empty()) {
        audio_wav w;
        std::string werr;
        if (audio_wav_read(o.audio_in, w, werr) && w.sample_rate > 0 && w.channels > 0) {
            in_audio_s = (double) w.samples.size() / (double) w.channels / (double) w.sample_rate;
        }
    }

    // Timestamped ASR segments. The window boundaries are already known exactly, so this costs
    // nothing beyond remembering them -- and a transcript with times is a different artifact
    // from a wall of text when the source is a recording someone has to cite.
    std::vector<audio_segment> seg_times;
    std::vector<std::string>   seg_texts;

    if (need_listen) {
        LOG_INF("%s: listening to '%s'\n", __func__, o.audio_in.c_str());

        // --audio-chunk-mode. `none` and a file that already fits keep the ORIGINAL
        // single-pass path (mtmd's own loader, so mp3/flac still work); only a file
        // that actually needs splitting takes the windowed path, which has to decode
        // to samples itself in order to cut them.
        bool chunked = false;
        asr_silero_model silero;
        bool silero_loaded = false;
        if (!o.vad_model.empty()) {
            std::string serr;
            if (!asr_silero_load(o.vad_model, silero, &serr)) {
                LOG_ERR("%s: --vad-model: %s\n", __func__, serr.c_str());
                return 1;
            }
            silero_loaded = true;
        }
        // --hallucination-silence-threshold re-decodes parts of a window, so it needs the samples: it takes the windowed
        // path even for --audio-chunk-mode none (one window = the whole file)
        const bool halluc = o.halluc_silence > 0.0;
        if (o.chunk_mode != "none" || halluc) {
            const int32_t sr = mctx ? mtmd_get_audio_sample_rate(mctx) : 16000;
            std::vector<float> mono;
            std::string werr;
            if (!audio_load_mono(o.audio_in, sr, mono, werr)) {
                // Not fatal: mtmd's loader handles formats audio-common's WAV reader does
                // not (mp3/flac). Falling back is correct; failing here would REGRESS
                // every non-WAV input that works today.
                LOG_INF("%s: chunk planning skipped (%s) -- single-pass\n", __func__, werr.c_str());
            } else {
                const char * why = "";
                std::vector<audio_segment> wins;
                if (o.chunk_mode != "none") {
                    wins = audio_plan_windows(mono, sr, o.chunk_mode, o.vad, &why,
                                              silero_loaded ? &silero : nullptr, &o.silero_post);
                }
                if (halluc && wins.size() <= 1 && !mono.empty()) {
                    audio_segment whole;
                    whole.i0 = 0; whole.i1 = mono.size();
                    whole.t0_s = 0.0; whole.t1_s = (double) mono.size() / sr;
                    wins.assign(1, whole);
                    why = "one window (hallucination silence skip)";
                }
                const double content_s = (double) mono.size() / sr;
                double last_speech = 0.0;   // end of the last kept word, carried across windows
                if (wins.size() > 1 || (halluc && !wins.empty())) {
                    chunked = true;
                    LOG_INF("%s: long audio %.2f s -> %d window(s) [%s], mode=%s\n", __func__,
                            (double) mono.size() / sr, (int) wins.size(), why, o.chunk_mode.c_str());

                    for (size_t wi = 0; wi < wins.size(); ++wi) {
                        const audio_segment & s = wins[wi];
                        if (s.i1 <= s.i0 || s.i1 > mono.size()) { continue; }

                        // Each window is an INDEPENDENT request. Without clearing the cache the
                        // second window's prompt lands after the first window's tokens and the
                        // model answers a two-turn conversation instead of transcribing -- which
                        // reads as a hallucinating ASR, not as a state bug.
                        llama_kv_cache_clear(lctx);
                        n_past = 0;

                        mtmd_bitmap * wb = mtmd_bitmap_init_from_audio(s.i1 - s.i0, mono.data() + s.i0);
                        if (!wb) {
                            LOG_ERR("%s: could not build the audio bitmap for window %d\n", __func__, (int) wi);
                            return 1;
                        }
                        LOG_INF("%s: window[%d/%d] %.3f -> %.3f s\n", __func__,
                                (int) wi + 1, (int) wins.size(), s.t0_s, s.t1_s);

                        std::string part;
                        g_dtw.t_offset = s.t0_s;
                        bool ok = true;
                        if (!halluc) {
                            ok = run_listen_bitmap(mctx, lctx, model, smpl, tmpls.get(),
                                                   params.use_jinja, wb, o.asr_prompt,
                                                   params.n_batch, params.n_predict, n_past, part);
                        } else {
                            // faster-whisper's seek loop over this window (asr_halluc_decode_span): every decode is an
                            // independent request over [a, b) of the samples; its words come from the DTW aligner in
                            // FILE seconds. The first decode is exactly the window the plain path would run.
                            std::string first_txt;
                            bool any_words = false;
                            const asr_decode_span_fn decode = [&](double a, double b, std::vector<asr_word> & ws) {
                                size_t i0 = (size_t) std::llround(a * sr), i1 = (size_t) std::llround(b * sr);
                                i0 = std::max(i0, s.i0); i1 = std::min(i1, s.i1);
                                if (i1 <= i0) { return true; }
                                llama_kv_cache_clear(lctx);
                                n_past = 0;
                                mtmd_bitmap * sb = (i0 == s.i0 && i1 == s.i1) ? nullptr
                                                 : mtmd_bitmap_init_from_audio(i1 - i0, mono.data() + i0);
                                g_dtw.t_offset = (double) i0 / sr;
                                const size_t w0 = g_dtw.words.size();
                                std::string txt;
                                const bool r = run_listen_bitmap(mctx, lctx, model, smpl, tmpls.get(), params.use_jinja,
                                                                 sb ? sb : wb, o.asr_prompt, params.n_batch, params.n_predict,
                                                                 n_past, txt);
                                if (sb) { mtmd_bitmap_free(sb); }
                                if (first_txt.empty()) { first_txt = txt; }
                                ws.assign(g_dtw.words.begin() + (std::ptrdiff_t) w0, g_dtw.words.end());
                                g_dtw.words.resize(w0);
                                any_words = any_words || !ws.empty();
                                return r;
                            };
                            asr_halluc_span_result hr;
                            ok = asr_halluc_decode_span(s.t0_s, s.t1_s, o.halluc_silence, content_s, last_speech,
                                                        o.halluc_silence, o.halluc_max_decodes, decode, hr,
                                                        g_dtw.sec_per_pos);   // word times step one audio position
                            for (const auto & t : hr.trace) { LOG_INF("%s: hallucination-skip: %s\n", __func__, t.c_str()); }
                            LOG_INF("%s: hallucination-skip: window %.3f-%.3f: %d decode(s), %d leading skip(s), %d truncation(s), %d segment(s) kept\n",
                                    __func__, s.t0_s, s.t1_s, hr.n_decodes, hr.n_skip_leading, hr.n_truncate, (int) hr.kept.size());
                            if (ok && !any_words) {
                                // no word times at all (the aligner failed or the model wrote nothing): nothing to judge,
                                // the window's own transcript stands
                                part = first_txt;
                            } else if (ok) {
                                last_speech = hr.last_speech;
                                for (const auto & g : hr.kept) {
                                    for (const auto & w : g.words) {
                                        if (!part.empty()) { part += ' '; }
                                        part += w.word;
                                        g_dtw.words.push_back(w);
                                    }
                                }
                            }
                        }
                        mtmd_bitmap_free(wb);
                        if (!ok) { return 1; }

                        while (!part.empty() && (part.back() == '\n' || part.back() == ' ')) { part.pop_back(); }
                        // Recorded even when EMPTY: a window that transcribed to nothing is a
                        // real, reportable fact about the audio, and dropping it would silently
                        // renumber the segments after it.
                        seg_times.push_back(s);
                        seg_texts.push_back(part);
                        if (!part.empty()) {
                            if (!transcript.empty()) { transcript += ' '; }
                            transcript += part;
                        }
                    }
                }
            }
        }

        if (!chunked) { g_dtw.t_offset = 0.0; }
        if (!chunked && !run_listen(mctx, lctx, model, smpl, tmpls.get(), params.use_jinja,
                        o.audio_in, o.asr_prompt,
                        params.n_batch, params.n_predict, n_past, transcript)) {
            return 1;
        }
        // Single-pass runs still get exactly one segment, spanning the whole file. That keeps
        // --segments-out's shape identical whether or not the audio was split, so a consumer
        // never has to special-case the unsplit run.
        if (!chunked) {
            audio_segment whole;
            whole.t0_s = 0.0;
            whole.t1_s = in_audio_s;   // 0.0 when the duration could not be read; reported as-is
            whole.i0 = 0;
            whole.i1 = 0;
            seg_times.push_back(whole);
            seg_texts.push_back(transcript);
        }

        // Leading newline on purpose. The generated text is streamed to stdout as
        // it is sampled while the logs go elsewhere, so without it this line lands
        // mid-line and no line-anchored parser can find it -- which is exactly how
        // examples/audio/tests/test_audio_roundtrip.py first failed.
        LOG_INF("\n%s: transcript: %s\n", __func__, transcript.c_str());

        if (g_dtw.on) {
            LOG_INF("%s: word-timestamps: %d word(s) over %d window(s)\n", __func__, (int) g_dtw.words.size(), (int) g_dtw.n_windows);
            for (const auto & w : g_dtw.words) {
                LOG_INF("  [%8.3f -> %8.3f] p=%.3f %s\n", w.start, w.end, w.probability, w.word.c_str());
            }
            if (!o.words_out.empty()) {
                FILE * wf = fopen(o.words_out.c_str(), "wb");
                if (!wf) {
                    LOG_ERR("%s: could not open %s for words\n", __func__, o.words_out.c_str());
                    return 1;
                }
                fprintf(wf, "{\n  \"audio\": \"%s\",\n", json_escape(o.audio_in).c_str());
                fprintf(wf, "  \"method\": \"dtw-llm-attention\",\n");
                fprintf(wf, "  \"layers\": [%d, %d],\n  \"seconds_per_audio_position\": %.6f,\n", g_dtw.cap.l0, g_dtw.cap.l1, g_dtw.sec_per_pos);
                fprintf(wf, "  \"words\": [\n");
                for (size_t k = 0; k < g_dtw.words.size(); ++k) {
                    const auto & w = g_dtw.words[k];
                    fprintf(wf, "    {\"word\": \"%s\", \"start\": %.3f, \"end\": %.3f, \"probability\": %.6f}%s\n",
                            json_escape(w.word).c_str(), w.start, w.end, w.probability, k + 1 < g_dtw.words.size() ? "," : "");
                }
                fprintf(wf, "  ]\n}\n");
                fclose(wf);
                LOG_INF("%s: wrote %d word(s) to %s\n", __func__, (int) g_dtw.words.size(), o.words_out.c_str());
            }
        }
    }

    // ---- think ------------------------------------------------------------
    if (o.task == AUDIO_TASK_CHAT || o.task == AUDIO_TASK_S2S) {
        if (o.task == AUDIO_TASK_CHAT) {
            if (params.prompt.empty()) {
                LOG_ERR("%s: task 'chat' needs -p \"text\"\n", __func__);
                return 1;
            }
            const llama_tokens t = common_tokenize(lctx, params.prompt, true, true);
            auto chat_prefill = [&](llama_pos & np) {
                llama_batch batch = llama_batch_init(std::max<size_t>(t.size(), 1), 0, 1);
                for (size_t i = 0; i < t.size(); ++i) {
                    common_batch_add(batch, t[i], (llama_pos) i, { 0 }, false);
                }
                batch.logits[batch.n_tokens - 1] = true;
                if (llama_decode(lctx, batch) != 0) {
                    LOG_ERR("%s: llama_decode() failed on the chat prompt\n", __func__);
                    llama_batch_free(batch);
                    return false;
                }
                np = batch.n_tokens;
                llama_batch_free(batch);
                return true;
            };
            if (!chat_prefill(n_past)) {
                return 1;
            }

            if (g_fb) {
                bool fb_ok = true;
                reply = generate_with_fallback(lctx, model, *g_fb_base, *g_fb, 0, chat_prefill, params.n_predict, true, n_past, &fb_ok);
                if (!fb_ok) { return 1; }
            } else {
                reply = generate_text(lctx, model, smpl, n_past, params.n_predict, true);
            }
        } else {
            // s2s: the reply continues straight on from the transcript turn
            reply = transcript;
        }
    }

    // ---- speak ------------------------------------------------------------
    // BATCH: the work list. One entry is the pre-existing single-utterance path, so behaviour with
    // no --batch-text-file is byte-identical to before. The point of batching here is that the
    // model and vocoder are already loaded ABOVE this block, so N utterances cost ONE load --
    // which is the whole reason audio.cpp offers batch modes at all.
    std::vector<std::string> batch_texts;
    std::vector<double>      batch_secs;
    std::vector<double>      batch_wall;
    std::vector<std::string> batch_files;

    if (o.speak_stream) {
        // the speaker already ran inside generate_text(); a text that came through a path that does not stream (the
        // --asr-fallback ladder) is spoken here in one go, so --speak-stream never silently speaks nothing
        g_speak_feed = nullptr;
        const std::string whole = !reply.empty() ? reply : transcript;
        if (sst.n_chunks == 0 && sst.n_skipped == 0 && !sst.failed && !whole.empty()) {
            speak_feed_fn(whole, true);
        }
        sst.sink.close();
        if (sst.failed) { return 1; }
        if (sst.n_chunks == 0) {
            LOG_ERR("%s: --speak-stream: nothing to speak\n", __func__);
            return 1;
        }
        out_audio_s += sst.audio_s;
        LOG_INF("%s: speak-stream: %d chunk(s), %.2f s of audio in %s, first audio in the file %.3f s after start, %.3f s in the speaker, %.3f s total\n",
                __func__, sst.n_chunks, sst.audio_s, o.out_file.c_str(), sst.first_audio_s, sst.tts_s, (ggml_time_us() - t_start_us) / 1e6);
        stream_first_audio_s = sst.first_audio_s;
    }

    if (need_speak && !o.speak_stream) {
        if (!o.batch_text_file.empty()) {
            std::ifstream bf(o.batch_text_file);
            if (!bf) {
                LOG_ERR("%s: cannot open --batch-text-file %s\n", __func__, o.batch_text_file.c_str());
                return 1;
            }
            std::string line;
            while (std::getline(bf, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                // Blank lines and # comments are SKIPPED, not spoken: a stray empty line would
                // otherwise produce a silent wav that looks like a vocoder fault.
                if (line.empty() || line[0] == '#') continue;
                batch_texts.push_back(line);
            }
            if (batch_texts.empty()) {
                LOG_ERR("%s: --batch-text-file %s contained no utterances\n", __func__, o.batch_text_file.c_str());
                return 1;
            }
            LOG_INF("%s: batch: %d utterance(s) from %s\n", __func__,
                    (int) batch_texts.size(), o.batch_text_file.c_str());
        } else {
            const std::string say =
                o.task == AUDIO_TASK_TTS ? params.prompt :
                (!reply.empty() ? reply : transcript);

            if (say.empty()) {
                LOG_ERR("%s: nothing to speak -- provide -p \"text\" or an input audio file\n", __func__);
                return 1;
            }
            batch_texts.push_back(say);
        }
    }

    // STOCHASTIC-PATH WARNING. MEASURED 2026-09-01: this TTS path samples (top_k=4) and the seed
    // was never pinned, so identical text produced 1.20 s / 1.39 s / 1.55 s / 2.00 s across runs and
    // the spoken WORDS varied with it -- seed 1234 read back "The fork can speak." 4/4 exact while
    // seed 99 read back only "The". A recorded "verified" result taken from one unpinned draw is a
    // coin flip, not a verification, and that is exactly how a passing result was recorded once and
    // then failed to reproduce. Greedy (--temp 0) is deterministic AND seed-INDEPENDENT here
    // (byte-identical output under seeds 1234 and 99) and read back 4/4 exact on repeat.
    // This WARNS rather than changing the default: promotion-to-default is the user's call.
    if (need_speak && !batch_texts.empty() && params.sampling.temp > 0.0f) {
        LOG_TEE("%s: WARNING: TTS is SAMPLING (temp=%.2f). Output words vary run to run. "
                "Use --temp 0 for a reproducible round trip.\n", __func__, (double) params.sampling.temp);
    }

    for (size_t bi = 0; bi < batch_texts.size(); ++bi) {
        const std::string & say = batch_texts[bi];
        const bool is_batch = !o.batch_text_file.empty();

        // Per-item output name: out.wav -> out_000.wav, so a batch never overwrites itself.
        std::string item_out = o.out_file;
        if (is_batch) {
            const size_t dot = o.out_file.find_last_of('.');
            char sfx[16];
            snprintf(sfx, sizeof(sfx), "_%03d", (int) bi);
            item_out = (dot == std::string::npos)
                     ? o.out_file + sfx
                     : o.out_file.substr(0, dot) + sfx + o.out_file.substr(dot);
        }

        const int64_t t_item_us = ggml_time_us();
        LOG_INF("%s: speaking[%d/%d]: \"%s\"\n", __func__,
                (int) bi + 1, (int) batch_texts.size(), say.c_str());

        std::vector<float> audio;
        if (!run_speak(ctx_tts, model_tts, smpl_tts, voc, o, say, params.n_predict, params.n_threads, audio)) {
            return 1;
        }

        // Always report the level: a silent-looking WAV is otherwise
        // indistinguishable from a broken vocoder.
        LOG_INF("%s: output level %.2f dBFS (rms)\n", __func__, audio_dbfs(audio_rms(audio.data(), audio.size())));
        if (o.normalize) {
            audio_peak_normalize(audio, o.peak);
            LOG_INF("%s: normalised to peak %.2f\n", __func__, o.peak);
        }

        std::string err;
        if (!audio_wav_write_s16(item_out, audio, o.sample_rate, err)) {
            LOG_ERR("%s: %s\n", __func__, err.c_str());
            return 1;
        }
        const double item_secs = (double) audio.size() / o.sample_rate;
        const double item_wall = (ggml_time_us() - t_item_us) / 1e6;
        // out_audio_s feeds the --metrics block. For a batch it accumulates, so the summary RTF
        // rates the WHOLE run against the WHOLE audio produced rather than against the last item.
        out_audio_s += item_secs;
        batch_secs.push_back(item_secs);
        batch_wall.push_back(item_wall);
        batch_files.push_back(item_out);
        LOG_INF("%s: wrote %s -- %.2f s of audio at %d Hz (%.2fs wall, rtf %.3f)\n", __func__,
                item_out.c_str(), item_secs, o.sample_rate, item_wall,
                item_secs > 0.0 ? item_wall / item_secs : 0.0);
    }

    // ---- batch manifest ---------------------------------------------------
    if (!o.batch_manifest_out.empty() && !batch_files.empty()) {
        FILE * mf = fopen(o.batch_manifest_out.c_str(), "wb");
        if (!mf) {
            LOG_ERR("%s: could not open %s for the batch manifest\n", __func__,
                    o.batch_manifest_out.c_str());
        } else {
            fprintf(mf, "{\n  \"tool\": \"llama_audio\",\n");
            fprintf(mf, "  \"model\": \"%s\",\n", json_escape(params.model.path).c_str());
            fprintf(mf, "  \"n_items\": %d,\n", (int) batch_files.size());
            fprintf(mf, "  \"items\": [\n");
            for (size_t k = 0; k < batch_files.size(); ++k) {
                fprintf(mf, "    {\"index\": %d, \"text\": \"%s\", "
                            "\"out_file\": \"%s\", \"out_audio_seconds\": %.6f, "
                            "\"wall_seconds\": %.6f, \"rtf\": %.6f}%s\n",
                        (int) k, json_escape(batch_texts[k]).c_str(),
                        json_escape(batch_files[k]).c_str(), batch_secs[k], batch_wall[k],
                        batch_secs[k] > 0.0 ? batch_wall[k] / batch_secs[k] : 0.0,
                        k + 1 < batch_files.size() ? "," : "");
            }
            fprintf(mf, "  ]\n}\n");
            fclose(mf);
            LOG_INF("%s: batch manifest -> %s (%d items)\n", __func__,
                    o.batch_manifest_out.c_str(), (int) batch_files.size());
        }
    }

    // ---- timestamped segments ---------------------------------------------
    if (!o.segments_out.empty()) {
        if (seg_times.empty()) {
            LOG_ERR("%s: --segments-out was given but no listening happened -- it applies to "
                    "--task asr / s2s\n", __func__);
        } else {
            FILE * sf = fopen(o.segments_out.c_str(), "wb");
            if (!sf) {
                LOG_ERR("%s: could not open %s for segments\n", __func__, o.segments_out.c_str());
            } else {
                fprintf(sf, "{\n  \"tool\": \"llama_audio\",\n");
                fprintf(sf, "  \"audio\": \"%s\",\n", json_escape(o.audio_in).c_str());
                fprintf(sf, "  \"model\": \"%s\",\n", json_escape(params.model.path).c_str());
                fprintf(sf, "  \"chunk_mode\": \"%s\",\n", json_escape(o.chunk_mode).c_str());
                fprintf(sf, "  \"n_segments\": %d,\n", (int) seg_times.size());
                fprintf(sf, "  \"segments\": [\n");
                std::vector<float> dl, dr;
                bool dia = false;
                if (o.diarize) {
                    std::string derr;
                    dia = audio_load_stereo16k(o.audio_in, dl, dr, derr);
                    if (!dia) { LOG_WRN("%s: %s -- segments are written without a speaker\n", __func__, derr.c_str()); }
                    else      { LOG_INF("diarize: stereo-energy fired (%zu frames per channel)\n", dl.size()); }
                }
                for (size_t k = 0; k < seg_times.size(); ++k) {
                    std::string spk;
                    if (dia) { spk = ", \"speaker\": \"" + audio_segment_speaker(dl, dr, seg_times[k]) + "\""; }
                    fprintf(sf, "    {\"index\": %d, \"start\": %.3f, \"end\": %.3f, \"text\": \"%s\"%s}%s\n",
                            (int) k, seg_times[k].t0_s, seg_times[k].t1_s,
                            json_escape(seg_texts[k]).c_str(), spk.c_str(),
                            k + 1 < seg_times.size() ? "," : "");
                }
                fprintf(sf, "  ]\n}\n");
                fclose(sf);
                LOG_INF("%s: segments -> %s (%d)\n", __func__,
                        o.segments_out.c_str(), (int) seg_times.size());
            }
        }
    }

    if (o.metrics) {
        const double total_s = (ggml_time_us() - t_start_us) / 1e6;
        // WHICH duration rates this run: ASR is rated against the audio it CONSUMED, TTS against
        // the audio it PRODUCED. s2s produces output, so it is rated on that.
        const double audio_s = out_audio_s > 0.0 ? out_audio_s : in_audio_s;
        const double rtf     = audio_s > 0.0 ? total_s / audio_s : 0.0;
        const double xrt     = total_s > 0.0 ? audio_s / total_s : 0.0;
        LOG_INF("%s: metrics: task=%s total=%.3fs in_audio=%.3fs out_audio=%.3fs "
                "rtf=%.4f realtime=%.2fx sample_rate=%d channels=1\n",
                __func__, task_name(o.task), total_s, in_audio_s, out_audio_s,
                rtf, xrt, o.sample_rate);

        if (!o.metrics_json.empty()) {
            // Written in the shape cortex/scripts/benchmark_emit.py ingests, so an audio run
            // becomes a store record with MODEL IDENTITY instead of an unattributable number.
            // The contract itself stays in ONE place (python); this only supplies named values.
            FILE * f = fopen(o.metrics_json.c_str(), "wb");
            if (f) {
                fprintf(f, "{\n");
                fprintf(f, "  \"tool\": \"llama_audio\",\n");
                fprintf(f, "  \"model\": \"%s\",\n", json_escape(params.model.path).c_str());
                fprintf(f, "  \"task\": \"%s\",\n", task_name(o.task));
                fprintf(f, "  \"metrics\": {\n");
                fprintf(f, "    \"wall_seconds\": %.6f,\n", total_s);
                fprintf(f, "    \"in_audio_seconds\": %.6f,\n", in_audio_s);
                fprintf(f, "    \"out_audio_seconds\": %.6f,\n", out_audio_s);
                if (stream_first_audio_s >= 0.0) {
                    fprintf(f, "    \"first_audio_seconds\": %.6f,\n", stream_first_audio_s);
                }
                fprintf(f, "    \"rtf\": %.6f,\n", rtf);
                fprintf(f, "    \"realtime_speed\": %.6f,\n", xrt);
                fprintf(f, "    \"sample_rate\": %d,\n", o.sample_rate);
                fprintf(f, "    \"channels\": 1\n");
                fprintf(f, "  }\n}\n");
                fclose(f);
                LOG_INF("%s: metrics json -> %s\n", __func__, o.metrics_json.c_str());
            } else {
                LOG_ERR("%s: could not open %s for metrics json\n", __func__, o.metrics_json.c_str());
            }
        }
    }

    if (smpl_tts && smpl_tts != smpl) { common_sampler_free(smpl_tts); }
    if (voc)  { wavtokenizer_free(voc); }
    if (mctx) { mtmd_free(mctx); }
    common_sampler_free(smpl);
    // `init` / `init_tts` own the model and context: freeing them here as well (llama_free + llama_free_model)
    // double-freed them when the unique_ptrs went out of scope (glibc "double free or corruption (out)").
    init_tts.reset();
    init.reset();
    llama_backend_free();

    return 0;
}
