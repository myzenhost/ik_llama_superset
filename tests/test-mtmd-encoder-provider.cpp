// Unit test of the mtmd audio-encoder provider hook (tools/mtmd/mtmd-encoder-provider.{h,cpp}).
// usage: test-mtmd-encoder-provider <path of the test plugin module>
//
// What it proves, with a negative control for each claim that could pass vacuously:
//   - compare(): identical -> nmse 0; a scaled copy -> known nmse; a NaN -> infinite nmse; all-zero reference is not NaN
//   - open_api(): bad abi / null entry / opt-out init are refused with a reason, a good table opens
//   - encode(): output equals an independent recomputation (nmse 0), a mel shifted by one frame does NOT (nmse above tol:
//     the control that shows the comparison can fail), a failing provider and a NaN-writing provider are refused
//   - open_library(): loads the plugin by path, missing file and missing symbol give a reason, no crash
//   - open_from_env(): feature off with no env, set env reaches the plugin, VERIFY/TOL are parsed
//   - eligibility list matches the projector types that run models/whisper-enc.cpp
#include "mtmd-encoder-provider.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static void set_env(const char * k, const char * v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

// the plugin's function, recomputed here independently
static void reference(const std::vector<float> & mel, std::vector<float> & out) {
    const int64_t n = (int64_t) mel.size();
    for (int64_t i = 0; i < (int64_t) out.size(); i++) {
        out[(size_t) i] = 0.5f * mel[(size_t) (i % n)] + 0.25f * mel[(size_t) ((i * 7 + 3) % n)];
    }
}

static std::vector<float> make_mel(int n_step, int n_mel, int shift) {
    std::vector<float> m((size_t) n_step * n_mel);
    for (int r = 0; r < n_mel; r++) {
        for (int c = 0; c < n_step; c++) {
            const int cc = c + shift;
            m[(size_t) r * n_step + c] = std::sin(0.013f * cc * (r + 1)) + 0.1f * (float) ((cc * 31 + r * 17) % 11) - 0.5f;
        }
    }
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <plugin module>\n", argv[0]);
        return 2;
    }
    const std::string plugin = argv[1];

    // ---- compare()
    {
        std::vector<float> a = { 1, -2, 3, 4 };
        auto d = mtmd_encoder_provider_compare(a.data(), a.data(), a.size());
        CHECK(d.nmse == 0.0 && std::fabs(d.cosine - 1.0) < 1e-12 && d.max_abs == 0.0 && d.n_nonfinite == 0);
        std::vector<float> b = { 2, -4, 6, 8 };            // a == 0.5 * b
        d = mtmd_encoder_provider_compare(a.data(), b.data(), a.size());
        CHECK(std::fabs(d.nmse - 0.25) < 1e-9);            // sum((b/2)^2) / sum(b^2) = 1/4
        CHECK(std::fabs(d.cosine - 1.0) < 1e-9);
        std::vector<float> n = { 1, NAN, 3, 4 };
        d = mtmd_encoder_provider_compare(n.data(), a.data(), a.size());
        CHECK(d.n_nonfinite == 1 && std::isinf(d.nmse));
        std::vector<float> z(4, 0.0f);
        d = mtmd_encoder_provider_compare(a.data(), z.data(), a.size());
        CHECK(std::isfinite(d.nmse) && d.nmse > 0.0);      // zero reference must not turn into NaN
    }

    // ---- open_api() refusals
    {
        std::string err;
        CHECK(!mtmd_encoder_provider_session::open_api(nullptr, "m", "CPU", "", err) && !err.empty());
        mtmd_encoder_provider_api bad = { 99, "bad-abi", nullptr, nullptr, nullptr };
        err.clear();
        CHECK(!mtmd_encoder_provider_session::open_api(&bad, "m", "CPU", "", err) && err.find("abi_version") != std::string::npos);
        bad.abi_version = MTMD_ENCODER_PROVIDER_ABI;
        err.clear();
        CHECK(!mtmd_encoder_provider_session::open_api(&bad, "m", "CPU", "", err) && err.find("null entry") != std::string::npos);
    }

    // ---- through the real shared library
    const int n_step = 40, n_mel = 8, n_out = 96;
    {
        std::string err;
        auto s = mtmd_encoder_provider_session::open_library(plugin, "ok", "CPU", "", err);
        CHECK(s != nullptr);
        if (s) {
            CHECK(std::string(s->name()) == "test-plugin");
            const auto mel = make_mel(n_step, n_mel, 0);
            std::vector<float> got((size_t) n_out), ref((size_t) n_out);
            CHECK(s->encode(mel.data(), n_step, n_mel, got.data(), n_out, err));
            reference(mel, ref);
            auto d = mtmd_encoder_provider_compare(got.data(), ref.data(), (size_t) n_out);
            CHECK(d.nmse == 0.0);                           // exact: same float expression on both sides

            // negative control: the SAME comparison must flag a mel shifted by one frame
            const auto mel2 = make_mel(n_step, n_mel, 1);
            std::vector<float> got2((size_t) n_out);
            CHECK(s->encode(mel2.data(), n_step, n_mel, got2.data(), n_out, err));
            d = mtmd_encoder_provider_compare(got2.data(), ref.data(), (size_t) n_out);
            CHECK(d.nmse > 1e-3);

            // bad arguments never reach the provider
            CHECK(!s->encode(nullptr, n_step, n_mel, got.data(), n_out, err));
            CHECK(!s->encode(mel.data(), 0, n_mel, got.data(), n_out, err));
        }
        // opt-out, failing and NaN-writing providers
        err.clear();
        CHECK(!mtmd_encoder_provider_session::open_library(plugin, "optout", "CPU", "", err) && err.find("opted out") != std::string::npos);
        auto f = mtmd_encoder_provider_session::open_library(plugin, "fail", "CPU", "", err);
        CHECK(f != nullptr);
        if (f) {
            const auto mel = make_mel(n_step, n_mel, 0);
            std::vector<float> out((size_t) n_out);
            err.clear();
            CHECK(!f->encode(mel.data(), n_step, n_mel, out.data(), n_out, err) && err.find("returned 5") != std::string::npos);
            CHECK(f->n_failed == 1);
        }
        auto nn = mtmd_encoder_provider_session::open_library(plugin, "nan", "CPU", "", err);
        CHECK(nn != nullptr);
        if (nn) {
            const auto mel = make_mel(n_step, n_mel, 0);
            std::vector<float> out((size_t) n_out);
            err.clear();
            CHECK(!nn->encode(mel.data(), n_step, n_mel, out.data(), n_out, err) && err.find("non-finite") != std::string::npos);
        }
        // a provider that "succeeds" without writing: the caller's compare in verify mode is what catches it
        auto sh = mtmd_encoder_provider_session::open_library(plugin, "short", "CPU", "", err);
        CHECK(sh != nullptr);
        if (sh) {
            const auto mel = make_mel(n_step, n_mel, 0);
            std::vector<float> out((size_t) n_out, 0.0f), ref((size_t) n_out);
            CHECK(sh->encode(mel.data(), n_step, n_mel, out.data(), n_out, err));
            reference(mel, ref);
            CHECK(mtmd_encoder_provider_compare(out.data(), ref.data(), (size_t) n_out).nmse > 0.5);
        }
        // missing file, and a real DLL that lacks the entry symbol
        err.clear();
        CHECK(!mtmd_encoder_provider_session::open_library(plugin + ".does-not-exist", "ok", "CPU", "", err) && err.find("cannot load") != std::string::npos);
        // a real library that does not export the entry symbol
#if defined(_WIN32)
        const char * no_entry_lib = "kernel32.dll";
#else
        const char * no_entry_lib = "libm.so.6";
#endif
        err.clear();
        CHECK(!mtmd_encoder_provider_session::open_library(no_entry_lib, "ok", "CPU", "", err) && err.find("does not export") != std::string::npos);
    }

    // ---- environment selection
    {
        std::string err;
        set_env("MTMD_ENCODER_PROVIDER", nullptr);
        set_env("MTMD_ENCODER_PROVIDER", "");
        CHECK(mtmd_encoder_provider_session::open_from_env(err) == nullptr && err.empty());   // feature off: silent
        set_env("MTMD_ENCODER_PROVIDER", plugin.c_str());
        set_env("MTMD_ENCODER_PROVIDER_MODEL", "ok");
        set_env("MTMD_ENCODER_PROVIDER_VERIFY", "1");
        set_env("MTMD_ENCODER_PROVIDER_TOL", "2.5e-4");
        auto s = mtmd_encoder_provider_session::open_from_env(err);
        CHECK(s != nullptr);
        if (s) {
            CHECK(s->verify() && std::fabs(s->tol() - 2.5e-4) < 1e-12);
        }
        set_env("MTMD_ENCODER_PROVIDER_VERIFY", "0");
        set_env("MTMD_ENCODER_PROVIDER_TOL", "-1");     // invalid -> default
        s = mtmd_encoder_provider_session::open_from_env(err);
        CHECK(s != nullptr && !s->verify() && std::fabs(s->tol() - 1e-3) < 1e-12);
        set_env("MTMD_ENCODER_PROVIDER_MODEL", "optout");
        err.clear();
        CHECK(mtmd_encoder_provider_session::open_from_env(err) == nullptr && !err.empty());   // requested but refused: reason given
        set_env("MTMD_ENCODER_PROVIDER", "");
    }

    // ---- eligibility
    CHECK(mtmd_encoder_provider_projector_ok("voxtral") && mtmd_encoder_provider_projector_ok("ultravox") &&
          mtmd_encoder_provider_projector_ok("qwen2a") && mtmd_encoder_provider_projector_ok("glma") &&
          mtmd_encoder_provider_projector_ok("meralion") && mtmd_encoder_provider_projector_ok("musicflamingo"));
    CHECK(!mtmd_encoder_provider_projector_ok("mlp") && !mtmd_encoder_provider_projector_ok("qwen3a") && !mtmd_encoder_provider_projector_ok(nullptr));

    if (g_fail) {
        std::fprintf(stderr, "test-mtmd-encoder-provider: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("test-mtmd-encoder-provider: all checks passed\n");
    return 0;
}
