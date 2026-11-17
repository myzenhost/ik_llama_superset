// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include "mtmd-video-plan.h"
#include "mtmd-audio-route.h"
#include "llama.h"

#include "hash/hash.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cstdlib>
#include <new>
#include <vector>

//#define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

// in-process still-webp decoder (vendor/simplewebp, BSD-3); the one SIMPLEWEBP_IMPLEMENTATION TU
#define SIMPLEWEBP_IMPLEMENTATION
#include "simplewebp/simplewebp.h"

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

#ifdef MTMD_VIDEO
#include "sheredom/subprocess.h"
#include <thread>
#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <pthread.h>
#endif
#endif

//
// internal logging functions
//

void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data) {
    if (log_callback == nullptr) {
        log_callback = g_logger.default_callback;
    }
    g_logger.log_callback = log_callback;
    g_logger.log_callback_user_data = user_data;
    mtmd_log_set(log_callback, user_data);
}

//
// helper functions
//

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks) {
    llama_pos n_pos = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens * chunks, llama_pos pos_0, mtmd_decoder_pos * out_pos) {
    size_t n_tokens = mtmd_image_tokens_get_n_tokens(chunks);
    for (size_t i = 0; i < n_tokens; i++) {
        out_pos[i] = mtmd_image_tokens_get_decoder_pos(chunks, pos_0, i);
    }
}

// Helper class to set non-causal attention via RAII
class scope_non_causal {
public:
    scope_non_causal(llama_context * context, bool enabled) : context_(context), enabled_(enabled) {
        if (enabled_) {
            // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
            llama_set_causal_attn(context_, false);
        }
    }
    ~scope_non_causal() {
        if (enabled_) {
            llama_set_causal_attn(context_, true);
        }
    }

    scope_non_causal(const scope_non_causal &) = delete;
    scope_non_causal & operator=(const scope_non_causal &) = delete;

private:
    llama_context * context_;
    bool enabled_;
};

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past,
        mtmd_helper_post_decode_callback callback,
        void * user_data) {
    GGML_ASSERT(n_batch > 0);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }

    const llama_model * model = llama_get_model(lctx);
    int n_mmproj_embd = llama_model_n_embd_inp(model);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    int32_t i_batch = 0;
    int32_t n_img_batches = (n_tokens + n_batch - 1) / n_batch;
    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    if (mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image_tokens) {
                LOG_ERR("failed to decode chunk: image tokens are null\n");
                return -1;
            }
            const auto n_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);
            std::vector<mtmd_decoder_pos> rel_pos(n_tokens);
            mtmd_helper_image_get_decoder_pos(image_tokens, n_past, rel_pos.data());
            batch_embd.set_position_mrope_2d(rel_pos, seq_id);
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            GGML_ABORT("invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    const bool use_non_causal = mtmd_decode_use_non_causal(ctx, chunk);
    const scope_non_causal non_causal(lctx, use_non_causal);

    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

        int64_t t1 = ggml_time_ms();
        int32_t ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(lctx, pos_offset, n_tokens_batch));
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            return ret;
        }

        if (callback != nullptr) {
            const mtmd_helper_embd_batch view = batch_embd.get_view(pos_offset, n_tokens_batch);
            ret = callback(&view, user_data);
            if (ret != 0) {
                LOG_ERR("post-decode callback failed\n");
                return ret;
            }
        }

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);

        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

    return 0;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past) {
    GGML_ASSERT(n_batch > 0);
    int32_t ret;
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        llama_batch_ext_ptr text_batch(llama_batch_ext_init(lctx));
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            llama_batch_ext_clear(text_batch.get());
            int32_t n_added = 0;
            int32_t idx     = -1;
            for (; i < n_tokens && n_added < n_batch; i++) {
                idx = llama_batch_ext_add_token(text_batch.get(), seq_id, tokens[i]);
                GGML_ASSERT(idx >= 0);
                llama_pos pos = n_past++;
                llama_batch_ext_set_pos(text_batch.get(), idx, &pos);
                n_added++;
            }
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                llama_batch_ext_set_output_logits(text_batch.get(), idx, true);
            }
            ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, text_batch.get());
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                return ret;
            }
            *new_n_past += n_added;
        }

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            return ret;
        }

        LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

        float * embd = mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past, nullptr, nullptr);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            return ret;
        }
    } else {
        GGML_ABORT("chunk type not supported");
    }

    return 0;
}

int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }

    for (size_t i = 0; i < n_chunks; i++) {
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOG_ERR("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
    }

    return 0;
}

namespace audio_helpers {

static bool is_audio_file(const char * buf, size_t len) {
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
}

// Longest clip, in seconds of audio, that is decoded. The decoder reports its length in frames AT THE TARGET RATE, which is
// the file's declared duration times that rate; the declared rate is the file's own claim (a header can say 1 Hz or 100 Hz
// over a million frames), so a small upload could ask for tens of gigabytes before a single sample is read.
// Default 4 h (a 4 h clip is 14400 s * 16 kHz * 4 B = 0.9 GB of f32); LLAMA_MTMD_AUDIO_MAX_SECONDS overrides.
// DISABLE-REVERSIBLE: LLAMA_MTMD_AUDIO_MAX_SECONDS=0 switches the cap off (env, cap is on by default).
// RESEARCH-DONE: https://github.com/ollama/ollama/blob/main/mlxrunner/model/audio/audio.go (read in full 2026-10-27)
//
// SOURCE-READ: ollama f067033f mlxrunner/model/audio/audio.go:11-17 (maxAudioSeconds = 600, maxSampleRate = 384000),
// :33-38 DecodeLimit, :149-154 (length checked BEFORE the sample buffer is allocated), MIT; re-specified, no code copied.
// SOURCE-NUANCE: ollama caps at 600 s because its audio models take one window; this fork chunks long audio
// (mtmd-audio-chunks), so 600 s would refuse legitimate long-form transcription -- hence 4 h. Ollama checks the cap against the
// file's NATIVE rate before any resampling; miniaudio only reports the length after rate conversion, which is the same duration
// times the target rate, so the check here is frames > seconds * target_rate. Ollama also rejects rates above 384 kHz; that
// bounds its own sinc resampler's kernel, whereas miniaudio's resampler cost is bounded by the output length checked here.
static double max_clip_seconds() {
    double s = 14400.0;
    if (const char * e = std::getenv("LLAMA_MTMD_AUDIO_MAX_SECONDS")) {
        char * end = nullptr;
        const double v = std::strtod(e, &end);
        if (end != e && v >= 0.0) {
            s = v;
        }
    }
    return s;
}

// true when `frame_count` frames at `rate` Hz fit under the cap
static bool clip_within_cap(ma_uint64 frame_count, int rate, const char * who) {
    const double cap = max_clip_seconds();
    if (cap > 0.0 && rate > 0 && (double) frame_count > cap * (double) rate) {
        LOG_ERR("%s: audio of %.0f s (%llu frames at %d Hz) is longer than the %.0f s limit; refused before allocating "
                "(LLAMA_MTMD_AUDIO_MAX_SECONDS raises it)\n",
                who, (double) frame_count / (double) rate, (unsigned long long) frame_count, rate, cap);
        return false;
    }
    return true;
}

// returns true if the buffer is a valid audio file
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }
    if (!clip_within_cap(frame_count, target_sampler_rate, "decode_audio_from_buf")) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    try {
        pcmf32_mono.resize(frame_count);
    } catch (const std::bad_alloc &) {
        LOG_ERR("decode_audio_from_buf: cannot allocate %llu frames\n", (unsigned long long) frame_count);
        ma_decoder_uninit(&decoder);
        return false;
    }
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

#ifdef MTMD_AUDIO_DEBUG
    // save audio to wav file
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
    ma_encoder encoder;
    ma_encoder_init_file("output.wav", &config, &encoder);
    ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
    ma_encoder_uninit(&encoder);
#endif

    ma_decoder_uninit(&decoder);
    return true;
}

} // namespace audio_helpers

int32_t mtmd_helper_audio_decode_stereo(const unsigned char * buf, size_t len, int32_t sample_rate,
                                        float * out_lr, size_t max_frames, size_t * n_frames) {
    if (n_frames) { *n_frames = 0; }
    if (buf == nullptr || !audio_helpers::is_audio_file((const char *) buf, len) || sample_rate <= 0) {
        return 0;
    }
    // channels = 0 asks the decoder for the file's native count; probe it before asking for two
    ma_decoder_config probe_cfg = ma_decoder_config_init(ma_format_f32, 0, (ma_uint32) sample_rate);
    ma_decoder probe;
    if (ma_decoder_init_memory(buf, len, &probe_cfg, &probe) != MA_SUCCESS) {
        return 0;
    }
    const int32_t native = (int32_t) probe.outputChannels;
    ma_decoder_uninit(&probe);
    if (native != 2) {
        return native;
    }
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 2, (ma_uint32) sample_rate);
    ma_decoder dec;
    if (ma_decoder_init_memory(buf, len, &cfg, &dec) != MA_SUCCESS) {
        return 0;
    }
    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&dec, &frame_count) != MA_SUCCESS) {
        ma_decoder_uninit(&dec);
        return 0;
    }
    if (!audio_helpers::clip_within_cap(frame_count, sample_rate, "mtmd_helper_audio_decode_stereo")) {
        ma_decoder_uninit(&dec);
        return 0; // the caller sizes its buffer from *n_frames; refuse before it does
    }
    if (out_lr == nullptr) {
        if (n_frames) { *n_frames = (size_t) frame_count; }
        ma_decoder_uninit(&dec);
        return native;
    }
    ma_uint64 frames_read = 0;
    const ma_uint64 want = std::min<ma_uint64>(frame_count, (ma_uint64) max_frames);
    const ma_result r = ma_decoder_read_pcm_frames(&dec, out_lr, want, &frames_read);
    ma_decoder_uninit(&dec);
    if (r != MA_SUCCESS && r != MA_AT_END) {
        return 0;
    }
    if (n_frames) { *n_frames = (size_t) frames_read; }
    return native;
}

int32_t mtmd_helper_audio_decode_mono(const unsigned char * buf, size_t len, int32_t sample_rate,
                                      float * out, size_t max_frames, size_t * n_frames) {
    if (n_frames) { *n_frames = 0; }
    if (buf == nullptr || !audio_helpers::is_audio_file((const char *) buf, len) || sample_rate <= 0) {
        return 0;
    }
    std::vector<float> pcm;
    if (!audio_helpers::decode_audio_from_buf(buf, len, sample_rate, pcm)) {
        return 0;
    }
    if (n_frames) { *n_frames = pcm.size(); }
    if (out != nullptr) {
        std::copy_n(pcm.data(), std::min(pcm.size(), max_frames), out);
    }
    return 1;
}

static bool is_webp_file(const unsigned char * buf, size_t len) {
    // WEBP ref: https://developers.google.com/speed/webp/docs/riff_container
    return len >= 12 && memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WEBP", 4) == 0;
}

// CAPABILITY-GENUINELY-NA: image-format decoder, not a model/harness capability
// SOURCE-READ: tmp/research/webp_1110/xwebp_decode.go (ollama's golang.org/x/image/webp decode.go, BSD, read in full)
// SOURCE-NUANCE: it rejects VP8X len != 10 (:157), frame dims != VP8X canvas (:86,:142), canvas w*h > MaxInt32 (:175),
//   ALPH without the VP8X alpha flag (:59), and an animated file ends in EOF -> invalid (:50); checks below mirror those.
enum webp_rc {
    WEBP_OK = 0,
    WEBP_INVALID,  // spec violation / corrupt: an error, no fallback (ollama: invalid format)
    WEBP_FALLBACK, // animated or feature simplewebp lacks: ffmpeg may still do it (MTMD_VIDEO)
};

static uint32_t webp_le(const unsigned char * p, int n) {
    uint32_t v = 0;
    for (int i = n - 1; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

// Decode a still webp from memory to packed RGB (alpha dropped, as stb does with 3 channels); rgb null = dims only.
static webp_rc decode_webp_in_process(const unsigned char * buf, size_t len, int & w, int & h,
                                      std::vector<unsigned char> * rgb, std::string & why) {
    bool     seen_vp8x = false;
    uint32_t cw = 0, ch = 0;
    bool     want_alpha = false;
    size_t   pos = 12;
    while (pos + 8 <= len) {
        const unsigned char * id = buf + pos;
        const uint64_t size = webp_le(buf + pos + 4, 4);
        const unsigned char * data = buf + pos + 8;
        if (!memcmp(id, "VP8X", 4)) {
            if (seen_vp8x || size != 10 || pos + 8 + 10 > len) { why = "bad VP8X chunk"; return WEBP_INVALID; }
            seen_vp8x  = true;
            if (data[0] & 0x02) { why = "animated webp"; return WEBP_FALLBACK; }
            want_alpha = (data[0] & 0x10) != 0;
            cw = webp_le(data + 4, 3) + 1;
            ch = webp_le(data + 7, 3) + 1;
            if ((uint64_t) cw * ch > 0x7fffffffull) { why = "canvas too large"; return WEBP_INVALID; }
        } else if (!memcmp(id, "ANIM", 4) || !memcmp(id, "ANMF", 4)) {
            why = "animated webp";
            return WEBP_FALLBACK;
        } else if (!memcmp(id, "ALPH", 4)) {
            if (!want_alpha) { why = "ALPH chunk without VP8X alpha flag"; return WEBP_INVALID; }
        } else if (!memcmp(id, "VP8 ", 4) || !memcmp(id, "VP8L", 4)) {
            break; // first image chunk wins
        }
        const uint64_t adv = 8 + size + (size & 1);
        if (adv > len - pos) break;
        pos += (size_t) adv;
    }

    simplewebp * sw = nullptr;
    simplewebp_error err = simplewebp_load_from_memory(const_cast<unsigned char *>(buf), len, nullptr, &sw);
    if (err != SIMPLEWEBP_NO_ERROR) {
        why = simplewebp_get_error_text(err);
        return err == SIMPLEWEBP_UNSUPPORTED_ERROR ? WEBP_FALLBACK : WEBP_INVALID;
    }
    size_t sw_w = 0, sw_h = 0;
    simplewebp_get_dimensions(sw, &sw_w, &sw_h);
    webp_rc rc = WEBP_OK;
    if (sw_w == 0 || sw_h == 0 || sw_w > 0x7fffffffu || sw_h > 0x7fffffffu || (uint64_t) sw_w * sw_h > 0x7fffffffull) {
        why = "bad image dimensions";
        rc = WEBP_INVALID;
    } else if (seen_vp8x && (sw_w != cw || sw_h != ch)) {
        why = "frame dimensions differ from VP8X canvas";
        rc = WEBP_INVALID;
    } else {
        w = (int) sw_w;
        h = (int) sw_h;
        if (rgb) {
            try {
                const size_t n = sw_w * sw_h;
                rgb->assign(n * 4, 0);
                err = simplewebp_decode(sw, rgb->data(), nullptr);
                if (err != SIMPLEWEBP_NO_ERROR) {
                    why = simplewebp_get_error_text(err);
                    rc  = WEBP_INVALID;
                } else {
                    unsigned char * p = rgb->data();
                    for (size_t i = 0; i < n; i++) { // RGBA -> RGB in place (write index never passes read index)
                        p[i * 3 + 0] = p[i * 4 + 0];
                        p[i * 3 + 1] = p[i * 4 + 1];
                        p[i * 3 + 2] = p[i * 4 + 2];
                    }
                    rgb->resize(n * 3);
                }
            } catch (const std::bad_alloc &) {
                why = "out of memory";
                rc  = WEBP_INVALID;
            }
        }
    }
    simplewebp_unload(sw);
    return rc;
}

// Largest still image, in pixels (width x height), that is decoded. The size in an image header is the file's own claim: a
// PNG of about 100 KB can declare 20000 x 20000, and the decoder then allocates 1.2 GB of RGB8 before the model's own resize
// (clip image_max_pixels) ever runs. stb's only built-in bound is per SIDE (STBI_MAX_DIMENSIONS, 1 << 24).
// Default 178956970 pixels = 512 MiB of RGB8, above a 100-megapixel camera frame (11648 x 8736 = 101.8 MP).
// LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS overrides it.
// DISABLE-REVERSIBLE: LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS=0 switches the cap off (env, cap is on by default).
// Row S69-ENGREV-MLXENGINE-IMAGE-PIXEL-CAP-2026-10-27-01.
//
// SOURCE-READ: lmstudio-ai/mlx-engine 86f20eb4 mlx_engine/server/chat.py:107-110 _validate_image_pixel_count (count above
// PIL's Image.MAX_IMAGE_PIXELS -> "Image dimensions are too large."), :113-118 (width * height read from the opened header
// and checked BEFORE the pixels are materialised), :163-165 (the same cap applied to the running total of all images in one
// request), MIT; re-specified, no code copied.
// SOURCE-NUANCE: mlx-engine takes PIL's default cap, 89478485 pixels (256 MiB of RGB8), which refuses a 100 MP camera
// frame; PIL itself only WARNS there and raises its bomb error at twice that. This cap is that doubled figure, so every
// image PIL would open is still accepted here.
uint64_t mtmd_helper_image_max_decode_pixels(void) {
    uint64_t cap = 178956970ull;
    if (const char * e = std::getenv("LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS")) {
        char * end = nullptr;
        errno = 0;
        const unsigned long long v = std::strtoull(e, &end, 10);
        if (end != e && *end == '\0' && errno == 0 && e[0] != '-') {
            cap = (uint64_t) v;
        }
    }
    return cap;
}

// Declared size of a still image, from the header alone: stb formats, then still webp. 0 = not a recognised still image.
uint64_t mtmd_helper_image_declared_pixels(const unsigned char * buf, size_t len) {
    if (!buf || len == 0 || len > (size_t) INT_MAX) {
        return 0;
    }
    int nx = 0, ny = 0, nc = 0;
    if (stbi_info_from_memory(buf, (int) len, &nx, &ny, &nc) && nx > 0 && ny > 0) {
        return (uint64_t) nx * (uint64_t) ny;
    }
    if (is_webp_file(buf, len)) {
        int w = 0, h = 0;
        std::string why;
        if (decode_webp_in_process(buf, len, w, h, nullptr, why) == WEBP_OK && w > 0 && h > 0) {
            return (uint64_t) w * (uint64_t) h;
        }
    }
    return 0;
}

// true when an image declaring `pixels` may be decoded
static bool image_within_cap(uint64_t pixels, const char * who) {
    const uint64_t cap = mtmd_helper_image_max_decode_pixels();
    if (cap > 0 && pixels > cap) {
        LOG_ERR("%s: image declares %llu pixels, above the %llu pixel decode limit; refused before decoding "
                "(LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS raises it)\n",
                who, (unsigned long long) pixels, (unsigned long long) cap);
        return false;
    }
    return true;
}

#ifdef MTMD_VIDEO
static mtmd_bitmap * decode_webp_with_ffmpeg(const mtmd_context * mctx, const unsigned char * buf, size_t len, bool placeholder,
                                             const mtmd_helper_video_init_params & params);
static void mtmd_helper_video_set_id(mtmd_helper_video * vctx, const std::string & id);
// ffmpeg audio fallback: any container ffmpeg reads (ogg/opus, webm, m4a/mp4, amr, adts aac, or a wav/mp3/flac the native
// decoder rejected) to mono f32 PCM at `rate`. false = ffmpeg missing, failed, or the clip is over the length cap.
static bool transcode_audio_with_ffmpeg(const unsigned char * buf, size_t len, int rate, const char * bin_dir,
                                        std::vector<float> & pcm);
#endif

// Decode an audio buffer: native miniaudio for wav/mp3/flac, ffmpeg for the other containers. Logs why on failure.
static bool decode_audio_any(const unsigned char * buf, size_t len, int rate, const mtmd_helper_init_opt & opt,
                             mtmd_audio_route::kind route, std::vector<float> & pcm) {
    if (route == mtmd_audio_route::kind::NATIVE) {
        if (audio_helpers::decode_audio_from_buf(buf, len, rate, pcm)) {
            return true;
        }
        pcm.clear();
    }
#ifdef MTMD_VIDEO
    if (transcode_audio_with_ffmpeg(buf, len, rate, opt.video_params.ffmpeg_bin_dir, pcm)) {
        return true;
    }
    LOG_ERR("Unable to decode audio from buffer (native decoder and ffmpeg both failed; is ffmpeg installed and on PATH?). %s\n",
            mtmd_audio_route::formats_hint().c_str());
#else
    GGML_UNUSED(opt);
    LOG_ERR("Unable to decode audio from buffer (this container needs ffmpeg and MTMD_VIDEO is OFF). %s\n",
            mtmd_audio_route::formats_hint().c_str());
#endif
    return false;
}

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(const mtmd_context * ctx, const unsigned char * buf, size_t len, bool placeholder,
                                                            mtmd_helper_init_opt opt) {
    // calculate the hash if needed
    std::string id;
    mtmd_bitmap * result = nullptr;

    if (!placeholder) {
        // use sha256 to prevent cache poisoning
        id = hash_sha256_hex(buf, len);
    }

    // container routing: wav/flac/mp3 decode natively; ogg/amr/adts-aac go to ffmpeg; mp4/webm try video first
    const mtmd_audio_route::kind route = mtmd_audio_route::classify(buf, len);
    const bool native_sniff = audio_helpers::is_audio_file((const char *)buf, len);
    const bool audio_route  = route == mtmd_audio_route::kind::TRANSCODE ||
                              (native_sniff && route != mtmd_audio_route::kind::MAYBE_VIDEO);
    if (audio_route) {
        std::vector<float> pcmf32;
        const int sample_rate = mtmd_get_audio_sample_rate(ctx);
        if (sample_rate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return {nullptr, nullptr};
        }
        // a sniffed-native buffer the classifier did not name (UNKNOWN) still gets the native decoder first
        const mtmd_audio_route::kind eff = route == mtmd_audio_route::kind::TRANSCODE ? route : mtmd_audio_route::kind::NATIVE;
        if (!decode_audio_any(buf, len, sample_rate, opt, eff, pcmf32)) {
            return {nullptr, nullptr};
        }
        result = mtmd_bitmap_init_from_audio(pcmf32.size(), placeholder ? nullptr : pcmf32.data());
        mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
        return {result, nullptr};
    }

    // otherwise, we assume it's an image
    if (!result) {
        int nx, ny, nc;
        // the header's claim is checked before stb allocates for it (the decode runs for a placeholder too)
        if (len <= (size_t) INT_MAX && stbi_info_from_memory(buf, (int) len, &nx, &ny, &nc) && nx > 0 && ny > 0
            && !image_within_cap((uint64_t) nx * (uint64_t) ny, __func__)) {
            return {nullptr, nullptr};
        }
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (data) {
            result = mtmd_bitmap_init(nx, ny, placeholder ? nullptr : data);
            mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
            stbi_image_free(data);
            return {result, nullptr};
        }
        // otherwise, fallthrough to video decoding (if supported)
    }

    // stb_image does not support webp: decode in-process first (no ffmpeg needed), ffmpeg only for animated/unsupported
    bool webp_try_ffmpeg = false;
    if (!result && is_webp_file(buf, len)) {
        int w = 0, h = 0;
        std::vector<unsigned char> rgb;
        std::string why;
        webp_rc rc = decode_webp_in_process(buf, len, w, h, nullptr, why); // dimensions only: nothing is allocated for pixels
        if (rc == WEBP_OK && !image_within_cap((uint64_t) w * (uint64_t) h, __func__)) {
            return {nullptr, nullptr};
        }
        if (rc == WEBP_OK && !placeholder) {
            rc = decode_webp_in_process(buf, len, w, h, &rgb, why);
        }
        if (rc == WEBP_OK) {
            result = mtmd_bitmap_init(w, h, placeholder ? nullptr : rgb.data());
            mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
            return {result, nullptr};
        }
        if (rc == WEBP_INVALID) {
            LOG_ERR("%s: invalid webp image: %s\n", __func__, why.c_str());
            return {nullptr, nullptr};
        }
        webp_try_ffmpeg = true;
#ifndef MTMD_VIDEO
        LOG_ERR("%s: webp not decodable in-process (%s) and ffmpeg fallback is not built (MTMD_VIDEO is OFF)\n", __func__, why.c_str());
        return {nullptr, nullptr};
#endif
    }

#ifdef MTMD_VIDEO
    if (!result && webp_try_ffmpeg) {
        result = decode_webp_with_ffmpeg(ctx, buf, len, placeholder, opt.video_params);
        if (!result) {
            LOG_ERR("%s: failed to decode webp buffer\n", __func__);
            return {nullptr, nullptr};
        }
        mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
        return {result, nullptr};
    }
#endif

    // last try: load as video
#ifdef MTMD_VIDEO
    if (!result) {
        auto video_ctx = mtmd_helper_video_init_from_buf(ctx, buf, len, opt.video_params);
        if (!video_ctx && route == mtmd_audio_route::kind::MAYBE_VIDEO) {
            // m4a / audio-only webm: no video stream, so try an audio-only ffmpeg transcode
            const int sample_rate = mtmd_get_audio_sample_rate(ctx);
            std::vector<float> pcmf32;
            if (sample_rate >= 0 && transcode_audio_with_ffmpeg(buf, len, sample_rate, opt.video_params.ffmpeg_bin_dir, pcmf32)) {
                result = mtmd_bitmap_init_from_audio(pcmf32.size(), placeholder ? nullptr : pcmf32.data());
                mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
                return {result, nullptr};
            }
        }
        if (!video_ctx) {
            LOG_ERR("%s: failed to decode buffer as either image/audio/video\n", __func__);
            return {nullptr, nullptr};
        }
        mtmd_helper_video_set_id(video_ctx, id); // propagate the hash to the frames
        result = mtmd_bitmap_init_lazy(ctx,
            id.empty() ? nullptr : id.c_str(),
            video_ctx,
            [](size_t, void * user_data, mtmd_bitmap ** out_bitmap, char ** out_text) -> int {
                auto * vctx = static_cast<mtmd_helper_video *>(user_data);
                char * text = nullptr;
                int ret = mtmd_helper_video_read_next(vctx, out_bitmap, &text);
                *out_text = text; // heap-allocated by read_next; freed automatically by mtmd
                return ret;
            });
         return {result, video_ctx};
    }
#else
    if (!result) {
        LOG_ERR("%s: failed to decode buffer as either image or audio (video support not compiled in)\n", __func__);
        return {nullptr, nullptr};
    }
#endif

    // should not reach here
    return {nullptr, nullptr};
}

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(const mtmd_context * ctx, const char * fname, bool placeholder,
                                                             mtmd_helper_init_opt opt) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, NULL, 0);
    if (!wlen) {
        LOG_ERR("Unable to convert filename to UTF-16: %s\n", fname);
        return {nullptr, nullptr};
    }
    std::vector<wchar_t> wfname(wlen);
    wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, wfname.data(), wlen);
    if (!wlen) {
        LOG_ERR("Unable to convert filename to UTF-16: %s\n", fname);
        return {nullptr, nullptr};
    }
    FILE * f = _wfopen(wfname.data(), L"rb");
#else
    FILE * f = fopen(fname, "rb");
#endif
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return {nullptr, nullptr};
    }

    std::vector<unsigned char> buf;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size < 0) {
        LOG_ERR("Failed to get file size of %s\n", fname);
        fclose(f);
        return {nullptr, nullptr};
    }
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return {nullptr, nullptr};
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size(), placeholder, opt);
}

//
// Video input helpers
//

#ifdef MTMD_VIDEO

// LLM tokens one video item costs at (w, h): tokenize `tg` placeholder frames (no pixel data) the way a real clip would be, so the count comes from
// the model's own preprocessor and not from arithmetic. -1 when the context has no vision or tokenization fails.
static int64_t video_item_tokens(mtmd_context * mctx, int32_t w, int32_t h, int tg) {
    if (!mctx || !mtmd_support_vision(mctx) || w <= 0 || h <= 0) {
        return -1;
    }
    std::vector<mtmd_bitmap *> bitmaps;
    std::string text;
    for (int i = 0; i < tg; ++i) {
        mtmd_bitmap * b = mtmd_bitmap_init((uint32_t) w, (uint32_t) h, nullptr);
        mtmd_bitmap_set_mergeable(b, tg > 1);
        bitmaps.push_back(b);
        text += mtmd_get_marker(mctx);
    }
    std::vector<const mtmd_bitmap *> cb(bitmaps.begin(), bitmaps.end());
    mtmd_input_text it = { text.c_str(), text.size(), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int64_t n = -1;
    if (mtmd_tokenize(mctx, chunks, &it, cb.data(), cb.size()) == 0) {
        n = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            n += (int64_t) mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, i));
        }
    }
    mtmd_input_chunks_free(chunks);
    for (auto * b : bitmaps) {
        mtmd_bitmap_free(b);
    }
    return n;
}

// MSVC deprecates strdup(), use a std::string version directly
static char * mtmd_strdup(const std::string & s) {
    size_t n = s.size() + 1;
    char * p = (char *) malloc(n);
    if (p) {
        memcpy(p, s.c_str(), n);
    }
    return p;
}

struct mtmd_helper_video {
    const mtmd_context * mctx;
    std::string path;
    std::vector<uint8_t> input_buf; // non-empty when initialized from buffer
    std::string ffmpeg_bin;
    std::string ffprobe_bin;
    float fps_target = 0.0f;
    mtmd_helper_video_info info = {};
    std::string id; // hash of the input video

    // sampling limits (row S69-MNN-PICK-VIDEO-FRAME-CAP-AND-VISION-TOKEN-BUDGET-2026-10-05-01); 0 = off
    int32_t max_frames        = 0;
    int64_t max_vision_tokens = 0;
    int32_t frames_budget     = 0;     // stop reading after this many frames (planned count, or max_frames when the duration is unknown)
    bool    scaled            = false; // ffmpeg scales every frame to info.width x info.height

    // RAII wrapper for managing subprocess
    struct subprocess_handle {
        struct subprocess_s proc = {};
        bool created = false; // process exists and must be cleaned up
        bool alive   = false; // process can still give us data
        std::thread feeder;

        subprocess_handle() = default;
        subprocess_handle(const subprocess_handle &) = delete;
        subprocess_handle & operator=(const subprocess_handle &) = delete;
        ~subprocess_handle() { stop(); }

        void stop() {
            // note: alive becomes false on stdout EOF, but the process still needs cleanup
            if (!created) {
                return;
            }
            subprocess_terminate(&proc);
#ifdef _WIN32
            // no SIGPIPE on windows: a blocked feeder only gets a broken pipe once we close our read end of the child stdin
            if (proc.hStdInput) {
                CloseHandle(proc.hStdInput);
                proc.hStdInput = nullptr;
            }
#endif
            // join before destroy: feeder holds a FILE* from subprocess_stdin;
            // subprocess_destroy closes it, so the thread must finish first
            if (feeder.joinable()) {
                feeder.join();
            }
            subprocess_join(&proc, nullptr); // reap the child, or else it stays a zombie
            subprocess_destroy(&proc);
            created = false;
            alive   = false;
        }

        FILE * stdout_pipe() {
            return subprocess_stdout(&proc);
        }

        // buf is tied to lifetime of mtmd_helper_video, so it's guaranteed to outlive the feeder thread
        void start_feeder(const std::vector<uint8_t> & buf) {
            feeder = std::thread([this, &buf]() {
#ifndef _WIN32
                // ffmpeg can exit before it reads all the input, for example when ffprobe already got the metadata.
                // the write below must then fail with EPIPE, instead of killing the process with SIGPIPE
                sigset_t sigpipe_set;
                sigemptyset(&sigpipe_set);
                sigaddset(&sigpipe_set, SIGPIPE);
                pthread_sigmask(SIG_BLOCK, &sigpipe_set, nullptr); // linux sends the signal to the writing thread
#endif
                FILE * f = subprocess_stdin(&proc);
                if (!f) {
                    return;
                }
#ifdef F_SETNOSIGPIPE
                fcntl(fileno(f), F_SETNOSIGPIPE, 1); // macos/bsd send it to the process, so turn it off per fd
#endif
                fwrite(buf.data(), 1, buf.size(), f);
                fclose(f);
                proc.stdin_file = nullptr; // prevent double-close in subprocess_destroy
            });
        }
    };

    subprocess_handle sp;
    int32_t current_frame = 0;

    std::string prompt_start         = "Video:";
    int32_t     timestamp_interval_ms = 5000; // emit a timestamp text every N ms (0 = disabled)
    float       next_timestamp_ms     = 0.0f; // next elapsed-ms threshold at which to emit

    std::vector<uint8_t> frame_buf;
    std::string pending_text; // text queued to be returned before the next frame
    bool        start_emitted = false;

    bool is_buf_input() const {
        return !input_buf.empty();
    }

    bool probe(float fps_target_arg) {
        const char * input_arg = is_buf_input() ? "pipe:0" : path.c_str();
        const char * cmd[] = {
            ffprobe_bin.c_str(),
            "-v", "quiet",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-select_streams", "v:0",
            "-of", "default=noprint_wrappers=1",
            input_arg,
            nullptr,
        };

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) { LOG_DBG(" %s", cmd[i]); }
        LOG_DBG("\n");

        subprocess_handle probe_sp;
        if (subprocess_create(cmd,
                subprocess_option_search_user_path | subprocess_option_inherit_environment,
                &probe_sp.proc) != 0) {
            LOG_ERR("%s: failed to launch ffprobe\n", __func__);
            return false;
        }
        probe_sp.created = true;
        probe_sp.alive   = true;

        if (is_buf_input()) {
            probe_sp.start_feeder(input_buf);
        }

        uint32_t width  = 0;
        uint32_t height = 0;
        float orig_fps = 0.0f;
        float duration = -1.0f;
        int32_t n_frames_orig = -1;
        char line[256];
        FILE * fp = probe_sp.stdout_pipe();

        while (fgets(line, sizeof(line), fp)) {
            char * eq = strchr(line, '=');
            if (!eq) continue;
            *eq = '\0';
            const char * key = line;
            const char * val = eq + 1;
            char * nl = (char *)strchr(val, '\n');
            if (nl) *nl = '\0';

            if (strcmp(key, "width") == 0) {
                width = (uint32_t)atoi(val);
            } else if (strcmp(key, "height") == 0) {
                height = (uint32_t)atoi(val);
            } else if (strcmp(key, "r_frame_rate") == 0) {
                orig_fps = parse_rational(val);
            } else if (strcmp(key, "nb_frames") == 0 && strcmp(val, "N/A") != 0) {
                n_frames_orig = atoi(val);
            } else if (strcmp(key, "duration") == 0 && strcmp(val, "N/A") != 0) {
                duration = (float)atof(val);
            }
        }

        probe_sp.stop();

        if (width == 0 || height == 0 || orig_fps <= 0.0f) {
            return false;
        }

        if (duration < 0.0f && n_frames_orig > 0) {
            duration = (float)n_frames_orig / orig_fps;
        }

        fps_target = fps_target_arg > 0.0f ? fps_target_arg : orig_fps;
        info.width    = width;
        info.height   = height;
        info.fps      = fps_target;
        LOG_DBG("%s: %ux%u fps=%.2f duration=%.2fs n_frames=%d\n",
                __func__, width, height, fps_target, duration, info.n_frames);
        info.n_frames = duration > 0.0f ? (int32_t)(duration * fps_target + 0.5f) : -1;

        if (max_frames > 0 || max_vision_tokens > 0) {
            if (duration > 0.0f) {
                mtmd_video_plan_in pin;
                pin.duration_s     = duration;
                pin.fps_target     = fps_target;
                pin.width          = (int32_t) width;
                pin.height         = (int32_t) height;
                pin.max_frames     = max_frames;
                pin.max_tokens     = max_vision_tokens;
                const int cell     = mctx ? mtmd_get_vision_cell_px(mctx) : 0;
                const int tg       = mctx ? mtmd_get_vision_temporal_group(mctx) : 1;
                pin.cell_px        = cell > 0 ? cell : 28;
                pin.temporal_group = tg;
                // mtmd_tokenize / mtmd_get_marker take a mutable context; video_item_tokens only reads it
                mtmd_context * mc = const_cast<mtmd_context *>(mctx);
                if (max_vision_tokens > 0 && cell > 0 && video_item_tokens(mc, pin.width, pin.height, tg) > 0) {
                    pin.tokens_of = [mc, tg](int32_t w, int32_t h) { return video_item_tokens(mc, w, h, tg); };
                }
                const mtmd_video_plan_out plan = mtmd_video_plan(pin);
                if (!plan.ok) {
                    LOG_ERR("%s: video does not fit the limits: %s\n", __func__, plan.error.c_str());
                    return false;
                }
                if (plan.limited) {
                    LOG_INF("%s: video limited to %d frames (%d items, %.3f fps) at %dx%d, about %lld vision tokens (native %ux%u at %.2f fps, %.1fs)\n",
                            __func__, plan.n_frames, plan.n_items, plan.fps_eff, plan.width_eff, plan.height_eff,
                            (long long) plan.est_tokens, width, height, fps_target, duration);
                }
                fps_target    = (float) plan.fps_eff;
                info.fps      = fps_target;
                info.n_frames = plan.n_frames;
                frames_budget = plan.n_frames;
                if ((uint32_t) plan.width_eff != width || (uint32_t) plan.height_eff != height) {
                    scaled = true;
                    width  = (uint32_t) plan.width_eff;
                    height = (uint32_t) plan.height_eff;
                    info.width  = width;
                    info.height = height;
                }
            } else if (max_frames > 0) {
                // no duration: the sampling cannot be planned, so the cap is enforced while reading
                frames_budget = max_frames;
                LOG_WRN("%s: the video duration is unknown, the frame cap (%d) is applied while reading and no size reduction is possible\n",
                        __func__, max_frames);
            }
        }
        frame_buf.resize((size_t)width * height * 3);
        return true;
    }

    bool start_ffmpeg(float seek_seconds) {
        char seek_buf[64];
        char fps_buf[64];

        std::vector<const char *> cmd;
        cmd.push_back(ffmpeg_bin.c_str());

        if (!is_buf_input() && seek_seconds > 0.0f) {
            // input-side seek: fast, keyframe-accurate; only valid for seekable file inputs
            snprintf(seek_buf, sizeof(seek_buf), "%.6f", seek_seconds);
            cmd.push_back("-ss");
            cmd.push_back(seek_buf);
        }

        cmd.push_back("-nostdin");
        if (is_buf_input()) {
            // remove the 64KB read-ahead limit of cache:, or else ffmpeg cannot reach a moov atom at end of file
            cmd.push_back("-read_ahead_limit");
            cmd.push_back("-1");
        }
        cmd.push_back("-i");
        // cache:pipe:0 wraps stdin with a seekable in-memory cache, letting ffmpeg seek
        // backwards for container headers (e.g. MP4 moov atom at end of file)
        cmd.push_back(is_buf_input() ? "cache:pipe:0" : path.c_str());

        if (seek_seconds > 0.0f && is_buf_input()) {
            // output-side seek: frame-accurate but decodes and discards frames up to seek point
            snprintf(seek_buf, sizeof(seek_buf), "%.6f", seek_seconds);
            cmd.push_back("-ss");
            cmd.push_back(seek_buf);
        }

        char vf_buf[160];
        {
            // fps filter (sampling rate) and, when a vision-token budget shrank the frames, an area-averaging scale to the planned size
            int n_vf = 0;
            vf_buf[0] = '\0';
            if (fps_target > 0.0f) {
                n_vf += snprintf(vf_buf + n_vf, sizeof(vf_buf) - n_vf, "fps=%.6f", fps_target);
            }
            if (scaled) {
                n_vf += snprintf(vf_buf + n_vf, sizeof(vf_buf) - n_vf, "%sscale=%u:%u:flags=area", n_vf > 0 ? "," : "", info.width, info.height);
            }
            if (n_vf > 0) {
                cmd.push_back("-vf");
                cmd.push_back(vf_buf);
            }
        }
        GGML_UNUSED(fps_buf);

        cmd.push_back("-f");
        cmd.push_back("rawvideo");
        cmd.push_back("-pix_fmt");
        cmd.push_back("rgb24");
        cmd.push_back("pipe:1");
        cmd.push_back("-loglevel");
        cmd.push_back("error");
        cmd.push_back(nullptr);

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) {
            LOG_DBG(" %s", cmd[i]);
        }
        LOG_DBG("\n");

        int ret = subprocess_create(
            cmd.data(),
            subprocess_option_search_user_path | subprocess_option_inherit_environment,
            &sp.proc);

        sp.created = (ret == 0);
        sp.alive   = (ret == 0);
        LOG_DBG("%s: subprocess_create ret=%d proc_alive=%d\n", __func__, ret, (int)sp.alive);

        if (sp.alive && is_buf_input()) {
            LOG_DBG("%s: starting feeder thread for %zu-byte buffer\n", __func__, input_buf.size());
            sp.start_feeder(input_buf);
        }

        return sp.alive;
    }

    void stop_ffmpeg() {
        sp.stop();
    }

    mtmd_bitmap * read_next_frame() {
        if (!sp.alive) return nullptr;
        if (frames_budget > 0 && current_frame >= frames_budget) {
            // the planned frame count is reached (ffmpeg's fps filter can emit one more than planned): end the stream here
            LOG_DBG("%s: frame cap %d reached\n", __func__, frames_budget);
            sp.alive = false;
            return nullptr;
        }

        FILE * fp = sp.stdout_pipe();
        const size_t frame_size = (size_t)info.width * info.height * 3;
        LOG_DBG("%s: reading frame %d, expecting %zu bytes (%ux%u)\n",
                __func__, current_frame, frame_size, info.width, info.height);

        size_t total_read = 0;
        while (total_read < frame_size) {
            size_t n = fread(frame_buf.data() + total_read, 1, frame_size - total_read, fp);
            if (n == 0) {
                // clean EOF only if no bytes read yet; partial frame is an error
                LOG_DBG("%s: fread returned 0 after %zu/%zu bytes (ferror=%d)\n",
                        __func__, total_read, frame_size, ferror(fp));
                sp.alive = false;
                return nullptr;
            }
            total_read += n;
        }

        LOG_DBG("%s: frame %d read OK\n", __func__, current_frame);
        mtmd_bitmap * frame = mtmd_bitmap_init(info.width, info.height, frame_buf.data());
        mtmd_bitmap_set_mergeable(frame, true);
        if (!id.empty()) {
            // each frame gets a unique id in the form of {hash}+{frame}, so that it can be identified in cache
            std::string frame_id = id + "+" + std::to_string(current_frame);
            mtmd_bitmap_set_id(frame, frame_id.c_str());
        }
        current_frame++;
        return frame;
    }

    int32_t read_next(mtmd_bitmap ** out_bitmap, char ** out_text) {
        *out_bitmap = nullptr;
        *out_text   = nullptr;

        if (!pending_text.empty()) {
            *out_text = mtmd_strdup(pending_text);
            pending_text.clear();
            return *out_text ? 0 : -2;
        }

        LOG_DBG("%s: proc_alive=%d start_emitted=%d current_frame=%d\n",
                __func__, (int)sp.alive, (int)start_emitted, current_frame);

        if (!sp.alive) {
            return (current_frame == 0) ? -2 : -1;
        }

        if (!start_emitted) {
            start_emitted = true;
            if (!prompt_start.empty()) {
                *out_text = mtmd_strdup(prompt_start);
                return *out_text ? 0 : -2;
            }
        }

        mtmd_bitmap * frame = read_next_frame();
        if (!frame) return -1;
        *out_bitmap = frame;

        if (timestamp_interval_ms > 0) {
            // current_frame was already incremented by read_next_frame(); undo for elapsed calc
            float elapsed_ms = (float)(current_frame - 1) / info.fps * 1000.0f;
            if (elapsed_ms >= next_timestamp_ms) {
                char ts_buf[32];
                float elapsed_s = elapsed_ms / 1000.0f;
                int   minutes   = (int)(elapsed_s / 60);
                float seconds   = elapsed_s - minutes * 60.0f;
                snprintf(ts_buf, sizeof(ts_buf), "[%dm%.2fs]", minutes, seconds);
                pending_text = ts_buf;
                next_timestamp_ms += (float)timestamp_interval_ms;
            }
        }

        return 0;
    }

    static float parse_rational(const char * s) {
        int num = 0, den = 1;
        if (sscanf(s, "%d/%d", &num, &den) == 2 && den > 0) {
            return (float)num / (float)den;
        }
        float val;
        if (sscanf(s, "%f", &val) == 1) {
            return val;
        }
        return 0.0f;
    }
};
#endif

mtmd_helper_video_init_params mtmd_helper_video_init_params_default() {
    return {
        /* fps_target             */ 4.0f,
        /* ffmpeg_bin_dir         */ nullptr,
        /* timestamp_interval_ms  */ 5000,
        /* max_frames             */ 0,
        /* max_vision_tokens      */ 0,
    };
}

mtmd_helper_init_opt mtmd_helper_init_opt_default() {
    return {
        /* video_params */ mtmd_helper_video_init_params_default(),
    };
}


static std::string video_resolve_bin(const char * bin_dir, const char * name) {
    if (!bin_dir || bin_dir[0] == '\0') {
        return name; // rely on PATH
    }
    std::string result = bin_dir;
    char last = result.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        result += '\\';
#else
        result += '/';
#endif
    }
    result += name;
#ifdef _WIN32
    result += ".exe";
#endif
    return result;
}

#ifdef MTMD_VIDEO
static void mtmd_helper_video_set_id(mtmd_helper_video * vctx, const std::string & id) {
    vctx->id = id;
}

static mtmd_bitmap * decode_webp_with_ffmpeg(const mtmd_context * mctx, const unsigned char * buf, size_t len, bool placeholder,
                                             const mtmd_helper_video_init_params & params) {
    mtmd_helper_video vctx;
    vctx.mctx        = mctx;
    vctx.input_buf.assign(buf, buf + len);
    vctx.ffmpeg_bin  = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    vctx.ffprobe_bin = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    if (!vctx.probe(0.0f)) {
        return nullptr;
    }
    if (placeholder) {
        return mtmd_bitmap_init(vctx.info.width, vctx.info.height, nullptr);
    }
    // still image: the fps filter would output no frame, so disable it
    vctx.fps_target = 0.0f;
    if (!vctx.start_ffmpeg(0.0f)) {
        return nullptr;
    }
    mtmd_bitmap * frame = vctx.read_next_frame();
    if (frame) {
        mtmd_bitmap_set_mergeable(frame, false);
    }
    return frame;
}

// SOURCE-READ: the named project's decode_audio (every container through a libav resampler to 16 kHz mono); no code copied.
// MEASURED: none. No ffmpeg binary exists on the build host, so this path is compile-checked and the argv/routing are unit
// tested; the end-to-end transcode is BENCH-PENDING on a host with ffmpeg.
static bool transcode_audio_with_ffmpeg(const unsigned char * buf, size_t len, int rate, const char * bin_dir,
                                        std::vector<float> & pcm) {
    pcm.clear();
    if (rate <= 0 || len == 0) {
        return false;
    }
    const std::vector<std::string> args = mtmd_audio_route::ffmpeg_argv(video_resolve_bin(bin_dir, "ffmpeg"), rate);
    std::vector<const char *> cmd;
    for (const auto & a : args) {
        cmd.push_back(a.c_str());
    }
    cmd.push_back(nullptr);

    mtmd_helper_video::subprocess_handle sp;
    if (subprocess_create(cmd.data(), subprocess_option_search_user_path | subprocess_option_inherit_environment, &sp.proc) != 0) {
        LOG_ERR("%s: failed to launch ffmpeg\n", __func__);
        return false;
    }
    sp.created = true;
    sp.alive   = true;

    const std::vector<uint8_t> input(buf, buf + len); // outlives the feeder thread (joined in sp.stop())
    sp.start_feeder(input);

    const uint64_t cap_samples = mtmd_audio_route::max_samples(audio_helpers::max_clip_seconds(), rate);
    FILE * fp = sp.stdout_pipe();
    if (!fp) {
        return false;
    }
    std::vector<float> chunk(16384);
    bool too_long = false;
    for (;;) {
        const size_t n = fread(chunk.data(), sizeof(float), chunk.size(), fp);
        if (n == 0) {
            break;
        }
        pcm.insert(pcm.end(), chunk.begin(), chunk.begin() + n);
        if (mtmd_audio_route::over_cap((uint64_t) pcm.size(), cap_samples)) {
            too_long = true;
            break;
        }
    }
    if (too_long) {
        LOG_ERR("%s: audio is longer than the %.0f s limit; refused (LLAMA_MTMD_AUDIO_MAX_SECONDS raises it)\n",
                __func__, audio_helpers::max_clip_seconds());
        pcm.clear();
        return false;
    }
    if (pcm.empty()) {
        LOG_ERR("%s: ffmpeg produced no audio samples\n", __func__);
        return false;
    }
    return true;
}
#endif

mtmd_helper_video * mtmd_helper_video_init(
        const mtmd_context * mctx,
        const char * path,
        mtmd_helper_video_init_params params) {
#ifdef MTMD_VIDEO
    auto * ctx = new mtmd_helper_video();

    ctx->mctx                 = mctx;
    ctx->path                 = path;
    ctx->ffmpeg_bin           = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    ctx->ffprobe_bin          = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    ctx->timestamp_interval_ms = params.timestamp_interval_ms;
    ctx->max_frames            = params.max_frames;
    ctx->max_vision_tokens     = params.max_vision_tokens;

    if (!ctx->probe(params.fps_target)) {
        LOG_ERR("%s: ffprobe failed for '%s' (is ffprobe in PATH?)\n", __func__, path);
        delete ctx;
        return nullptr;
    }

    if (!ctx->start_ffmpeg(0.0f)) {
        LOG_ERR("%s: failed to start ffmpeg for '%s' (is ffmpeg in PATH?)\n", __func__, path);
        delete ctx;
        return nullptr;
    }

    return ctx;
#else
    GGML_UNUSED(mctx);
    GGML_UNUSED(path);
    GGML_UNUSED(params);
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
    return nullptr;
#endif
}

mtmd_helper_video * mtmd_helper_video_init_from_buf(
        const mtmd_context * mctx,
        const unsigned char * buf, size_t len,
        mtmd_helper_video_init_params params) {
#ifdef MTMD_VIDEO
    auto * ctx = new mtmd_helper_video();

    ctx->mctx                  = mctx;
    ctx->input_buf.assign(buf, buf + len);
    ctx->ffmpeg_bin            = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    ctx->ffprobe_bin           = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    ctx->timestamp_interval_ms = params.timestamp_interval_ms;
    ctx->max_frames            = params.max_frames;
    ctx->max_vision_tokens     = params.max_vision_tokens;

    if (!ctx->probe(params.fps_target)) {
        LOG_ERR("%s: ffprobe failed on buffer (is ffprobe in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }

    if (!ctx->start_ffmpeg(0.0f)) {
        LOG_ERR("%s: failed to start ffmpeg on buffer (is ffmpeg in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }

    return ctx;
#else
    GGML_UNUSED(mctx);
    GGML_UNUSED(buf);
    GGML_UNUSED(len);
    GGML_UNUSED(params);
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
    return nullptr;
#endif
}

void mtmd_helper_video_free(mtmd_helper_video * ctx) {
#ifdef MTMD_VIDEO
    if (!ctx) return;
    ctx->stop_ffmpeg();
    delete ctx;
#else
    GGML_UNUSED(ctx);
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
#endif
}

mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video * ctx) {
#ifdef MTMD_VIDEO
    return ctx->info;
#else
    GGML_UNUSED(ctx);
    GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

int32_t mtmd_helper_video_read_next(mtmd_helper_video * ctx,
        mtmd_bitmap ** out_bitmap, char ** out_text) {
#ifdef MTMD_VIDEO
    if (!ctx) return -2;
    return ctx->read_next(out_bitmap, out_text);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(out_bitmap);
    GGML_UNUSED(out_text);
    GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

mtmd_helper_cap mtmd_helper_get_capabilities(const llama_model * model, const mtmd_context * mctx) {
    mtmd_helper_cap cap = {};
    cap.chat     = true;
    cap.text_gen = true;
    if (!mctx) {
        return cap;
    }

    cap.vision = mtmd_support_vision(mctx);
    cap.audio  = mtmd_support_audio(mctx);
    cap.audio_stream = mtmd_support_audio_stream(mctx);
#ifdef MTMD_VIDEO
    cap.video  = cap.vision;
#endif
    const auto gen_audio_type = mtmd_gen_audio_get_info(mctx).type;
    cap.audio_gen = gen_audio_type != MTMD_GEN_AUDIO_TYPE_NONE;
    cap.audio_gen_spk_ref = gen_audio_type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS
                         || gen_audio_type == MTMD_GEN_AUDIO_TYPE_POCKETTTS;

    // tts-only model cannot be used for chat (no chat template)
    cap.chat = !cap.audio_gen || llama_model_chat_template(model, nullptr) != nullptr;

    return cap;
}

bool mtmd_helper_model_can_chat(const llama_context * lctx, const mtmd_context * mctx) {
    return mtmd_helper_get_capabilities(llama_get_model(lctx), mctx).chat;
}

bool mtmd_helper_support_video(const mtmd_context * ctx) {
#ifdef MTMD_VIDEO
    return mtmd_support_vision(ctx);
#else
    GGML_UNUSED(ctx);
    return false;
#endif
}
