// Real-audio comparison of an mtmd encoder provider against the ggml graph (the acceptance test of the
// "mtmd encoder offload hook" pick). Not an add_test(): it needs an mmproj GGUF, an exported model and wav files.
//
// usage: test-mtmd-encoder-compare <mmproj.gguf> <provider module> <provider model> <device> <tol> <threads> <wav> [<wav> ...]
//
// For every wav: whisper-style mel (the SAME mtmd_audio_preprocessor_whisper mtmd.cpp uses), the first 30 s chunk,
// encoded once by the ggml graph (provider env cleared) and once through the provider; prints nmse / cosine, then runs
// the negative control: the provider output of clip i against the ggml output of clip j != i must be FAR above tol,
// otherwise the comparison could not have failed.
//
// A clip_ctx reads the MTMD_ENCODER_PROVIDER* environment once in clip_init(), so the two contexts are made with the
// environment cleared / set around the call.
#include "clip.h"
#include "clip-impl.h"
#include "mtmd-audio.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void set_env(const char * k, const char * v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

// RIFF/WAVE PCM16 or float32, any channel count (mixed down), linear resample to 16 kHz
static bool read_wav16k(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 44 || std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(b.data() + 8, "WAVE", 4) != 0) return false;
    auto u16 = [&](size_t o) { return (uint16_t) (b[o] | (b[o + 1] << 8)); };
    auto u32 = [&](size_t o) { return (uint32_t) (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | ((uint32_t) b[o + 3] << 24)); };
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    size_t pos = 12, data_off = 0, data_len = 0;
    while (pos + 8 <= b.size()) {
        const uint32_t sz = u32(pos + 4);
        if (!std::memcmp(b.data() + pos, "fmt ", 4)) {
            fmt = u16(pos + 8); ch = u16(pos + 10); rate = u32(pos + 12); bits = u16(pos + 22);
        } else if (!std::memcmp(b.data() + pos, "data", 4)) {
            data_off = pos + 8;
            data_len = std::min<size_t>(sz, b.size() - data_off);
            break;
        }
        pos += 8 + sz + (sz & 1);
    }
    if (!ch || !rate || !data_len) return false;
    std::vector<float> mono;
    if (fmt == 1 && bits == 16) {
        const size_t n = data_len / (2 * ch);
        mono.resize(n);
        for (size_t i = 0; i < n; i++) {
            float s = 0;
            for (int c = 0; c < ch; c++) s += (int16_t) u16(data_off + (i * ch + c) * 2) / 32768.0f;
            mono[i] = s / ch;
        }
    } else if (fmt == 3 && bits == 32) {
        const size_t n = data_len / (4 * ch);
        mono.resize(n);
        for (size_t i = 0; i < n; i++) {
            float s = 0;
            for (int c = 0; c < ch; c++) { float v; std::memcpy(&v, &b[data_off + (i * ch + c) * 4], 4); s += v; }
            mono[i] = s / ch;
        }
    } else {
        return false;
    }
    if (rate == 16000) { out = std::move(mono); return true; }
    const size_t n_out = (size_t) ((double) mono.size() * 16000.0 / rate);
    out.resize(n_out);
    for (size_t i = 0; i < n_out; i++) {
        const double x = (double) i * rate / 16000.0;
        const size_t i0 = (size_t) x;
        const size_t i1 = std::min(i0 + 1, mono.size() - 1);
        const double fr = x - i0;
        out[i] = (float) (mono[i0] * (1.0 - fr) + mono[i1] * fr);
    }
    return true;
}

static clip_ctx * open_ctx(const char * mmproj) {
    clip_context_params p{};
    p.use_gpu = false;
    p.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    p.image_min_tokens = -1;
    p.image_max_tokens = -1;
    clip_init_result r = clip_init(mmproj, p);
    return r.ctx_a;
}

static bool encode(clip_ctx * ctx, const mtmd_audio_mel & mel, int threads, std::vector<float> & out, double & secs) {
    clip_image_f32 img;
    img.set_size({(int) mel.n_len, (int) mel.n_mel}, false, true);
    img.cpy_buf(mel.data);
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(img);
    const int n_tok = clip_n_output_tokens(ctx, &batch.entries.front());
    out.assign((size_t) n_tok * clip_n_mmproj_embd(ctx), 0.0f);
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = clip_image_batch_encode(ctx, threads, &batch, out);
    secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

static void diff(const std::vector<float> & a, const std::vector<float> & ref, double & nmse, double & cosine) {
    double num = 0, den = 0, dot = 0, na = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double e = (double) a[i] - ref[i];
        num += e * e; den += (double) ref[i] * ref[i]; dot += (double) a[i] * ref[i]; na += (double) a[i] * a[i];
    }
    nmse = den > 0 ? num / den : num;
    cosine = (na > 0 && den > 0) ? dot / std::sqrt(na * den) : 0.0;
}

int main(int argc, char ** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: %s <mmproj.gguf> <provider module> <provider model> <device> <tol> <threads> <wav>...\n", argv[0]);
        return 2;
    }
    const char * mmproj = argv[1];
    const std::string provider = argv[2], pmodel = argv[3], device = argv[4];
    const double tol = std::atof(argv[5]);
    const int threads = std::atoi(argv[6]);

    // reference context: provider env cleared
    set_env("MTMD_ENCODER_PROVIDER", "");
    clip_ctx * ref_ctx = open_ctx(mmproj);
    if (!ref_ctx) { std::fprintf(stderr, "cannot open %s\n", mmproj); return 1; }

    std::vector<std::vector<float>> mels_ref, mels_got;
    std::vector<mtmd_audio_mel> mels;
    std::vector<std::string> names;
    mtmd_audio_preprocessor_whisper pre(ref_ctx);
    pre.initialize();
    for (int i = 7; i < argc; i++) {
        std::vector<float> pcm;
        if (!read_wav16k(argv[i], pcm)) { std::fprintf(stderr, "skip (not a readable wav): %s\n", argv[i]); continue; }
        std::vector<mtmd_audio_mel> chunks;
        if (!pre.preprocess(pcm.data(), pcm.size(), chunks) || chunks.empty()) { std::fprintf(stderr, "preprocess failed: %s\n", argv[i]); continue; }
        mels.push_back(std::move(chunks.front()));
        names.push_back(argv[i]);
        std::printf("clip %zu: %s  %.1f s of audio, mel %lld x %lld\n", names.size() - 1, argv[i], pcm.size() / 16000.0,
                    (long long) mels.back().n_mel, (long long) mels.back().n_len);
    }
    if (mels.size() < 2) { std::fprintf(stderr, "need at least 2 readable clips (the negative control compares different clips)\n"); return 2; }

    double t_ref = 0;
    for (size_t i = 0; i < mels.size(); i++) {
        std::vector<float> o; double s = 0;
        if (!encode(ref_ctx, mels[i], threads, o, s)) { std::fprintf(stderr, "ggml encode failed on clip %zu\n", i); return 1; }
        t_ref += s;
        mels_ref.push_back(std::move(o));
        std::printf("clip %zu: ggml graph %.2f s, %zu floats\n", i, s, mels_ref.back().size());
    }
    clip_free(ref_ctx);

    // provider context
    set_env("MTMD_ENCODER_PROVIDER", provider.c_str());
    set_env("MTMD_ENCODER_PROVIDER_MODEL", pmodel.c_str());
    set_env("MTMD_ENCODER_PROVIDER_DEVICE", device.c_str());
    set_env("MTMD_ENCODER_PROVIDER_VERIFY", "");
    clip_ctx * pctx = open_ctx(mmproj);
    if (!pctx) { std::fprintf(stderr, "cannot open %s with the provider\n", mmproj); return 1; }
    // a provider that fails is DROPPED by clip_encode and the ggml graph answers instead; the two outputs are then
    // identical (nmse 0), which looks like a perfect pass. So the provider must still be attached after every encode.
    if (!clip_audio_provider_active(pctx)) { std::fprintf(stderr, "FAIL: provider did not open (see the clip_init log line above)\n"); return 1; }
    double t_got = 0;
    int n_bad = 0;
    for (size_t i = 0; i < mels.size(); i++) {
        std::vector<float> o; double s = 0;
        if (!encode(pctx, mels[i], threads, o, s)) { std::fprintf(stderr, "provider-path encode failed on clip %zu\n", i); return 1; }
        t_got += s;
        double nmse, cosine;
        diff(o, mels_ref[i], nmse, cosine);
        const bool served = clip_audio_provider_active(pctx);   // false: the provider failed and the ggml graph answered
        std::printf("clip %zu: provider path %.2f s  nmse %.3e  cosine %.8f  %s%s\n", i, s, nmse, cosine,
                    nmse <= tol ? "OK" : "OVER TOL", served ? "" : "  PROVIDER DROPPED (output is the ggml graph's)");
        if (!(nmse <= tol) || !served) n_bad++;
        mels_got.push_back(std::move(o));
    }
    clip_free(pctx);

    // negative control: a different clip's reference must be far away
    double min_cross = 1e300;
    for (size_t i = 0; i < mels.size(); i++) {
        const size_t j = (i + 1) % mels.size();
        double nmse, cosine;
        diff(mels_got[i], mels_ref[j], nmse, cosine);
        if (nmse < min_cross) min_cross = nmse;
        std::printf("control: provider(clip %zu) vs ggml(clip %zu): nmse %.3e cosine %.5f\n", i, j, nmse, cosine);
    }
    const bool control_ok = min_cross > 100.0 * tol;
    std::printf("total ggml %.2f s, provider path %.2f s\n", t_ref, t_got);
    std::printf("RESULT: %s (clips %zu, over tol %d, negative control %s, min cross nmse %.3e, tol %.1e)\n",
                (n_bad == 0 && control_ok) ? "PASS" : "FAIL", mels.size(), n_bad, control_ok ? "ok" : "TOO CLOSE", min_cross, tol);
    return (n_bad == 0 && control_ok) ? 0 : 1;
}
