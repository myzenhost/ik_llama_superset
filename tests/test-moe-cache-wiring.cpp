// ik_llama #2644 MoE expert cache: is it REACHABLE in this fork? (row S69-FORK-MOE-CACHE-WIRING-IK-2644-NOT-WIRED-2026-10-27-01)
//
// The cache itself (src/llama-moe-cache.cpp) and its graph hooks merged with ik 894191d6e, but llama_context refused any
// moe_cache_size > 0 and no CLI flag set it. This test checks the three things that make it reachable:
//
//   A. CLI (model-free): --moe-cache-mib N parses to N MiB, reaches llama_context_params, and bad values are refused.
//   B. CPU arm (model, no GPU device): a context asked for a cache goes through cache CONSTRUCTION, which refuses for
//      lack of a GPU backend (upstream design, llama-moe-cache.cpp:188) -- not the old "not wired" throw. A cache-off
//      context on the same model is the control.
//   C. GPU arm (model, a GPU device present): experts kept in host memory (-cmoe), every other layer on the GPU. A
//      cache-on context is created, its logits over a prompt and greedy single-token steps match a cache-off context,
//      and the cache's own teardown statistics show it served lookups (hits + misses > 0, hits > 0).
//
// Usage: test-moe-cache-wiring [moe.gguf]   (no model: part A only, exit 0 when it passes)
// Env:   MOE_CACHE_TEST_MIB (GPU arm budget, default 8 MiB: below stories15M_MOE's ~32 MiB of experts, so it evicts)
//
// GATE-CLAIM-OK: "no GPU device" below is measured from the ggml device registry at run time, not assumed.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) {
        g_failures++;
    }
}

struct log_capture {
    std::string text;
};

static void capture_log(ggml_log_level level, const char * text, void * user_data) {
    auto * cap = static_cast<log_capture *>(user_data);
    cap->text += text;
    if (level >= GGML_LOG_LEVEL_WARN || std::strstr(text, "moe_cache") || std::strstr(text, "MoE cache")) {
        fputs(text, stdout);
    }
}

static bool parse_args(std::vector<std::string> args, common_params & params) {
    // common_params_parse post-checks "--model is required" (common/arg.cpp:1318) and fails the whole parse without one,
    // which would hide the flag's own verdict. -m only stores the path (arg.cpp:4925); nothing is opened here.
    args.insert(args.begin(), { "test-moe-cache-wiring", "-m", "parse-only-not-opened.gguf" });
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    return common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON);
}

static void part_a_cli() {
    printf("A. --moe-cache-mib parsing\n");
    {
        common_params params;
        const bool ok = parse_args({ "--moe-cache-mib", "64" }, params);
        check(ok, "--moe-cache-mib 64 is accepted");
        check(params.moe_cache_size == (size_t) 64 * 1024 * 1024,
              "params.moe_cache_size = " + std::to_string(params.moe_cache_size) + " (want 67108864)");
        const llama_context_params cparams = common_context_params_to_llama(params);
        check(cparams.moe_cache_size == (size_t) 64 * 1024 * 1024,
              "llama_context_params.moe_cache_size = " + std::to_string(cparams.moe_cache_size));
    }
    {
        common_params params;
        check(parse_args({ "--moe-cache-mib", "0" }, params) && params.moe_cache_size == 0, "--moe-cache-mib 0 = no cache");
    }
    {
        // "-1" is the spelling of "auto" since the --fit sizing (common/arg.cpp), not an error
        common_params params;
        check(parse_args({ "--moe-cache-mib", "-1" }, params) && params.moe_cache_auto && params.moe_cache_size == 0, "--moe-cache-mib -1 = auto");
    }
    for (const char * bad : { "12x", "", "1.5", "99999999999999999999" }) {
        common_params params;
        const bool ok = parse_args({ "--moe-cache-mib", bad }, params);
        check(!ok && params.moe_cache_size == 0, std::string("--moe-cache-mib '") + bad + "' is refused");
    }
    // upstream llama.cpp #30966 spelling; 'auto' is this fork's addition
    {
        common_params params;
        check(parse_args({ "--moe-cache-mib", "64" }, params) && params.moe_cache_cpu_share == 0.0f && params.moe_cache_policy == 0,
              "defaults: --moe-cache-cpu-share 0, policy slru");
    }
    {
        common_params params;
        check(parse_args({ "--moe-cache-cpu-share", "0.5", "--moe-cache-policy", "classlru" }, params) && params.moe_cache_cpu_share == 0.5f && params.moe_cache_policy == 1,
              "--moe-cache-cpu-share 0.5 --moe-cache-policy classlru");
        const llama_context_params cparams = common_context_params_to_llama(params);
        check(cparams.moe_cache_cpu_share == 0.5f && cparams.moe_cache_policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU, "both reach llama_context_params");
    }
    {
        common_params params;
        check(parse_args({ "--moe-cache-cpu-share", "auto" }, params) && params.moe_cache_cpu_share < 0.0f, "--moe-cache-cpu-share auto");
    }
    for (const char * bad : { "-0.1", "1.5", "x", "" }) {
        common_params params;
        check(!parse_args({ "--moe-cache-cpu-share", bad }, params), std::string("--moe-cache-cpu-share '") + bad + "' is refused");
    }
    {
        common_params params;
        check(!parse_args({ "--moe-cache-policy", "lru" }, params), "--moe-cache-policy lru is refused");
    }
}

static bool has_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return true;
        }
    }
    return false;
}

static llama_context * make_ctx(llama_model * model, size_t cache_bytes) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = 256;
    cparams.n_batch        = 64;
    cparams.n_ubatch       = 64;
    cparams.n_threads      = 4;
    cparams.n_threads_batch = 4;
    cparams.no_perf        = true;
    cparams.moe_cache_size = cache_bytes;
    return llama_init_from_model(model, cparams);
}

static void part_b_cpu(llama_model * model, log_capture & cap) {
    printf("B. CPU arm: a cache request reaches cache construction\n");
    cap.text.clear();
    llama_context * ctx_off = make_ctx(model, 0);
    check(ctx_off != nullptr, "control: cache-off context is created");
    if (ctx_off) {
        llama_free(ctx_off);
    }

    cap.text.clear();
    llama_context * ctx_on = make_ctx(model, (size_t) 64 * 1024 * 1024);
    const bool not_wired = cap.text.find("not wired") != std::string::npos;
    const bool reached   = cap.text.find("failed to initialize MoE cache: MoE cache requires a GPU backend") != std::string::npos;
    check(!not_wired, "the old 'MoE-cache creation is not wired' refusal is gone");
    check(reached, "construction ran and refused for lack of a GPU backend (llama-moe-cache.cpp:188)");
    check(ctx_on == nullptr, "no context is returned for a cache that cannot exist (upstream: llama_init_from_model returns nullptr)");
    if (ctx_on) {
        llama_free(ctx_on);
    }
}

static bool decode_one(llama_context * ctx, const std::vector<llama_token> & toks, int pos0, std::vector<float> & last) {
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (int) i, { 0 }, i + 1 == toks.size());
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * lg = llama_get_logits_ith(ctx, -1);
    last.assign(lg, lg + n_vocab);
    return true;
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & x) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (double) (x[i] - ref[i]) * (x[i] - ref[i]);
        den += (double) ref[i] * ref[i];
    }
    return den > 0.0 ? num / den : num;
}

static int argmax(const std::vector<float> & v) {
    int best = 0;
    for (int i = 1; i < (int) v.size(); ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

static void part_c_gpu(llama_model * model, log_capture & cap) {
    printf("C. GPU arm: cache on vs off, experts in host memory\n");
    const char * env_mib = getenv("MOE_CACHE_TEST_MIB");
    const size_t mib = env_mib ? (size_t) std::strtoull(env_mib, nullptr, 10) : 8;

    llama_context * ctx_off = make_ctx(model, 0);
    cap.text.clear();
    llama_context * ctx_on = make_ctx(model, mib * 1024 * 1024);
    check(ctx_off != nullptr, "cache-off context is created");
    check(ctx_on != nullptr, "cache-on context is created (" + std::to_string(mib) + " MiB)");
    check(cap.text.find("MoE cache uses") != std::string::npos, "the cache allocated device banks ('MoE cache uses ...')");
    if (!ctx_off || !ctx_on) {
        if (ctx_off) llama_free(ctx_off);
        if (ctx_on) llama_free(ctx_on);
        return;
    }

    // <= 32 tokens: the cache serves ubatches up to max_batch = 32 (llama-moe-cache.cpp:153)
    std::vector<llama_token> prompt = common_tokenize(ctx_off, "Once upon a time, there was a little girl named Lily who loved to", true, false);
    if (prompt.size() > 32) prompt.resize(32);

    std::vector<float> l_off, l_on;
    bool ok = decode_one(ctx_off, prompt, 0, l_off) && decode_one(ctx_on, prompt, 0, l_on);
    check(ok, "prompt of " + std::to_string(prompt.size()) + " tokens decodes on both");
    double worst = ok ? nmse(l_off, l_on) : 1.0;
    int top1_same = ok && argmax(l_off) == argmax(l_on) ? 1 : 0;
    int pos = (int) prompt.size();
    const int n_steps = 24;
    for (int s = 0; ok && s < n_steps; ++s) {
        const llama_token next = argmax(l_off); // both contexts are fed the SAME token
        ok = decode_one(ctx_off, { next }, pos, l_off) && decode_one(ctx_on, { next }, pos, l_on);
        if (ok) {
            worst = std::max(worst, nmse(l_off, l_on));
            top1_same += argmax(l_off) == argmax(l_on);
        }
        ++pos;
    }
    check(ok, std::to_string(n_steps) + " single-token steps decode on both");
    char buf[160];
    snprintf(buf, sizeof(buf), "logits cache-on vs cache-off: worst NMSE %.3e (< 1e-3), top-1 agree %d/%d", worst, top1_same, n_steps + 1);
    check(worst < 1e-3, buf);

    llama_free(ctx_off);
    cap.text.clear();
    llama_free(ctx_on); // the cache logs its statistics in its destructor (llama-moe-cache.cpp:420-436)
    unsigned long long hits = 0, misses = 0;
    const size_t at = cap.text.find("ubatch <= 8 hits=");
    if (at != std::string::npos) {
        sscanf(cap.text.c_str() + at, "ubatch <= 8 hits=%llu misses=%llu", &hits, &misses);
    }
    check(hits + misses > 0, "cache served single-token lookups: hits=" + std::to_string(hits) + " misses=" + std::to_string(misses));
    check(hits > 0, "repeat experts hit the cache (hits > 0)");
}

int main(int argc, char ** argv) {
    part_a_cli();

    if (argc < 2) {
        printf("no model given: parts B/C skipped (run with a MoE gguf, e.g. ggml-org/stories15M_MOE-F16.gguf)\n");
    } else {
        log_capture cap;
        llama_log_set(capture_log, &cap);
        llama_backend_init();

        const bool gpu = has_gpu_device();
        llama_model_params mparams = llama_model_default_params();
        llama_model_tensor_buft_override ovr[2] = { llm_ffn_exps_cpu_override(), { nullptr, nullptr } };
        if (gpu) {
            mparams.n_gpu_layers          = 999;
            mparams.tensor_buft_overrides = ovr; // experts in host memory: what the cache is for (-cmoe)
        } else {
            mparams.n_gpu_layers = 0;
        }
        llama_model * model = llama_model_load_from_file(argv[1], mparams);
        check(model != nullptr, std::string("load ") + argv[1]);
        if (model) {
            if (gpu) {
                part_c_gpu(model, cap);
            } else {
                part_b_cpu(model, cap);
            }
            llama_model_free(model);
        }
        llama_backend_free();
        llama_log_set(nullptr, nullptr);
    }

    printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
