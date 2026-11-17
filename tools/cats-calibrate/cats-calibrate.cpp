// tools/cats-calibrate -- offline calibration for CATS and TEAL activation sparsity.
// S69-CATS-STAGE1-2026-09-28 (CATS mode, default) + S69-TEAL-STAGE1-2026-09-28 (--teal mode).
// References:
//   CATS: Lee et al., "CATS: Contextually-Aware Thresholding for Sparsity in LLMs",
//         arXiv:2404.08763, Sec. 3.1 (mask definition, eq. 4-5) and Sec. 4.1 (calibration).
//   TEAL: Liu et al., "Training-Free Activation Sparsity in LLMs", arXiv:2408.14690,
//         Sec. 4.2 Definition 1 (magnitude threshold tp: the s-quantile of |x|, applied to the
//         INPUT of a linear projection, not an activation's output).
// No code copied from either reference repo -- neither is on disk and neither was cloned (task
// constraint); this is an independent implementation from the papers' equations plus this fork's
// own graph/tensor-naming conventions (see llama-build-context.cpp's cb() calls).
//
// WHAT THIS DOES (CATS mode, default -- unchanged from the original S69-CATS-STAGE1 tool)
//   1. Loads the model with cparams.cats_calibrate=true (src/llama-context.cpp), which forces
//      llm_build_ffn's un-fused silu(gate) path for every gated-SiLU layer WITHOUT masking -- the
//      only way to observe the raw activation, since the fused kernels never materialize it.
//   2. Registers a ggml_backend_sched_eval_callback that intercepts tensors named "ffn_silu-<il>"
//      and accumulates |value| into a per-layer buffer.
//   3. Runs the calibration text through llama_decode in n_batch-sized chunks (single sequence,
//      causal prefill -- no perplexity is computed here, this is a stats pass only).
//   4. For each layer, the threshold for a target sparsity s is the s-quantile of the collected
//      |silu(gate)| values (nth_element on a copy) -- s fraction of activations fall below it.
//   5. Writes {"arch","n_layer","sparsity_target","thresholds":[...]}. Parsed by
//      src/llama-context.cpp's cats_load_thresholds().
//
// WHAT THIS DOES (--teal mode)
//   Same shape, three families instead of one, because TEAL sparsifies three distinct input
//   distributions per layer (see llama-build-context.cpp build_std_attention / llm_build_ffn):
//     attn_in  <- cb name "attn_norm-<il>"       (ALWAYS materialized, no routing needed)
//     ffn_in   <- cb name "teal_ffn_in-<il>"     (needs cparams.teal_calibrate to force the
//     ffn_mid  <- cb name "teal_ffn_mid-<il>"     un-fused FFN path, same reason as CATS above)
//   Verify mode reads the POST-mask names "teal_attn_in_masked-<il>" / "teal_ffn_in_masked-<il>" /
//   "teal_ffn_mid_masked-<il>" and counts exact zeros, same logic as CATS verify.
//   Writes {"arch","n_layer","sparsity_target","thresholds_attn_in":[...],"thresholds_ffn_in":[...],
//   "thresholds_ffn_mid":[...]}. Parsed by src/llama-context.cpp's teal_load_thresholds().
//
// ANTI-LEAKAGE: this tool does not itself enforce it, but per the task's own CHALLENGE requirement
// the calibration corpus (-f) and any later evaluation corpus (llama-perplexity -f) must be
// DIFFERENT files/spans -- see the ledger for which files were actually used.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// One family = one (calibrate-name-prefix, verify-name-prefix, json-key) triple. CATS mode uses
// exactly one; TEAL mode uses three. Generalizes the original single-family CATS collector.
struct calib_family {
    std::string calib_prefix;   // e.g. "ffn_silu-" or "attn_norm-"
    std::string verify_prefix;  // e.g. "ffn_cats_masked-" or "teal_attn_in_masked-"
    std::string json_key;       // e.g. "thresholds" or "thresholds_attn_in"
    std::vector<std::vector<float>> abs_vals;  // [il] -> collected |value| (calibrate mode)
    std::vector<int64_t> zero_count;            // [il] (verify mode)
    std::vector<int64_t> total_count;            // [il] (verify mode)
    calib_family(std::string c, std::string v, std::string j)
        : calib_prefix(std::move(c)), verify_prefix(std::move(v)), json_key(std::move(j)) {}
};

struct calib_collector {
    std::mutex mutex;
    std::vector<calib_family> families;
    int  n_layer = 0;
    bool verify_mode = false;

    void init(int n_layer_, bool verify) {
        n_layer = n_layer_;
        verify_mode = verify;
        for (auto & f : families) {
            f.abs_vals.assign(n_layer, {});
            f.zero_count.assign(n_layer, 0);
            f.total_count.assign(n_layer, 0);
        }
    }

    // returns {family index, layer index}, family index -1 if no family's prefix matches `name`
    std::pair<int,int> parse(const char * name) const {
        for (size_t fi = 0; fi < families.size(); ++fi) {
            const std::string & prefix = verify_mode ? families[fi].verify_prefix : families[fi].calib_prefix;
            const size_t plen = prefix.size();
            if (strncmp(name, prefix.c_str(), plen) != 0) continue;
            const char * p = name + plen;
            if (!*p) continue;
            bool all_digit = true;
            for (const char * q = p; *q; ++q) {
                if (*q < '0' || *q > '9') { all_digit = false; break; }
            }
            if (!all_digit) continue;
            int il = atoi(p);
            if (il < 0 || il >= n_layer) continue;
            return {(int) fi, il};
        }
        return {-1, -1};
    }

    bool eval_cb(struct ggml_tensor * t, bool ask) {
        auto [fi, il] = parse(t->name);
        if (fi < 0) {
            return false;
        }
        if (ask) {
            return true;
        }
        // Defensive checks (S69-CATS-STAGE1 crash postmortem): a non-F32 or not-yet-materialized
        // tensor here would read past the real buffer -- skip rather than trust blindly.
        if (t->type != GGML_TYPE_F32) {
            fprintf(stderr, "[calibrate] WARNING: %s has type %s, not F32 -- skipping this call\n",
                    t->name, ggml_type_name(t->type));
            fflush(stderr);
            return true;
        }
        if (t->buffer == nullptr || t->data == nullptr) {
            fprintf(stderr, "[calibrate] WARNING: %s has null buffer/data -- skipping this call\n", t->name);
            fflush(stderr);
            return true;
        }
        std::lock_guard<std::mutex> lock(mutex);
        const int64_t n = ggml_nelements(t);
        std::vector<float> host(n);
        if (ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(host.data(), t->data, n * sizeof(float));
        } else {
            ggml_backend_tensor_get(t, host.data(), 0, n * sizeof(float));
        }
        auto & fam = families[fi];
        if (verify_mode) {
            fam.total_count[il] += n;
            for (int64_t i = 0; i < n; ++i) {
                if (host[i] == 0.0f) ++fam.zero_count[il];
            }
        } else {
            auto & dst = fam.abs_vals[il];
            dst.reserve(dst.size() + n);
            for (int64_t i = 0; i < n; ++i) {
                dst.push_back(std::fabs(host[i]));
            }
        }
        return true;
    }
};

static calib_collector g_collector;

// ggml_backend_sched_eval_callback is `int (*)(ggml_tensor*, bool, void*)` (ggml-backend.h), not
// bool -- MEASURED here: g++ -fpermissive flags the bool-returning version as an invalid
// function-pointer conversion for `params.cb_eval` (build log, original CATS tool first attempt).
static int calib_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    return g_collector.eval_cb(t, ask) ? 1 : 0;
}

static std::string read_text_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        LOG_ERR("%s: could not open calibration text file '%s'\n", __func__, path.c_str());
        return {};
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx   = 512;
    params.n_batch = 512;

    std::string calib_file;
    std::string out_file;
    std::string verify_thresholds_file; // when set: VERIFY mode instead of calibrate
    float target_sparsity = 0.25f;
    int   max_tokens = 2000;
    bool  teal_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) { LOG_ERR("missing value for %s\n", flag); exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") params.model.path = next(a.c_str());
        else if (a == "-f" || a == "--calib-file") calib_file = next(a.c_str());
        else if (a == "-o" || a == "--output") out_file = next(a.c_str());
        else if (a == "-c" || a == "--ctx-size") params.n_ctx = std::stoi(next(a.c_str()));
        else if (a == "-b" || a == "--batch-size") params.n_batch = std::stoi(next(a.c_str()));
        else if (a == "--target-sparsity") target_sparsity = std::stof(next(a.c_str()));
        else if (a == "--verify-thresholds") verify_thresholds_file = next(a.c_str());
        else if (a == "-n" || a == "--n-tokens") max_tokens = std::stoi(next(a.c_str()));
        else if (a == "-t" || a == "--threads") params.cpuparams.n_threads = std::stoi(next(a.c_str()));
        else if (a == "--teal") teal_mode = true;
        else {
            LOG_ERR("unknown argument: %s\n", a.c_str());
            LOG_ERR("usage (calibrate): cats-calibrate [--teal] -m MODEL -f CALIB_TEXT -o OUT.json --target-sparsity F "
                     "[-c CTX] [-b BATCH] [-n MAX_TOKENS] [-t THREADS]\n");
            LOG_ERR("usage (verify achieved sparsity on an eval file): cats-calibrate [--teal] -m MODEL "
                     "-f EVAL_TEXT --verify-thresholds THRESHOLDS.json [-c CTX] [-b BATCH] [-n MAX_TOKENS] [-t THREADS]\n");
            return 1;
        }
    }

    const bool verify_mode = !verify_thresholds_file.empty();
    if (params.model.path.empty() || calib_file.empty() || (!verify_mode && out_file.empty())) {
        LOG_ERR("usage (calibrate): cats-calibrate [--teal] -m MODEL -f CALIB_TEXT -o OUT.json --target-sparsity F "
                 "[-c CTX] [-b BATCH] [-n MAX_TOKENS] [-t THREADS]\n");
        LOG_ERR("usage (verify achieved sparsity on an eval file): cats-calibrate [--teal] -m MODEL "
                 "-f EVAL_TEXT --verify-thresholds THRESHOLDS.json [-c CTX] [-b BATCH] [-n MAX_TOKENS] [-t THREADS]\n");
        return 1;
    }

    // common_cpu_params::n_threads defaults to -1 ("auto, resolve me") -- normally resolved by
    // common_params_parser_init()'s post-processing, which this tool's hand-rolled arg loop above
    // does not run. MEASURED as the actual cause of the original CATS tool's first-run crash
    // (ACCESS_VIOLATION inside common_threadpools::init(), not the graph code) -- fix: resolve
    // both n_threads and n_threads_batch explicitly.
    if (params.cpuparams.n_threads <= 0) {
        params.cpuparams.n_threads = (int32_t) std::thread::hardware_concurrency();
        if (params.cpuparams.n_threads <= 0) params.cpuparams.n_threads = 4;
    }
    params.cpuparams_batch.n_threads = params.cpuparams.n_threads;
    if (target_sparsity < 0.0f || target_sparsity >= 1.0f) {
        LOG_ERR("--target-sparsity must be in [0, 1)\n");
        return 1;
    }

    // Set up the families for this mode.
    if (teal_mode) {
        g_collector.families = {
            { "attn_norm-",    "teal_attn_in_masked-", "thresholds_attn_in" },
            { "teal_ffn_in-",  "teal_ffn_in_masked-",  "thresholds_ffn_in"  },
            { "teal_ffn_mid-", "teal_ffn_mid_masked-", "thresholds_ffn_mid" },
        };
    } else {
        g_collector.families = {
            { "ffn_silu-", "ffn_cats_masked-", "thresholds" },
        };
    }

    if (verify_mode) {
        if (teal_mode) {
            params.teal_calibrate = false;
            params.teal_sparsity = target_sparsity > 0.0f ? target_sparsity : 0.01f;
            params.teal_thresholds_file = verify_thresholds_file;
        } else {
            // Verify mode routes through the REAL masking path (cats_calibrate=false,
            // cats_sparsity>0, cats_thresholds loaded from file) and reads the POST-mask tensor
            // "ffn_cats_masked-<il>" -- an exact-0.0f value there IS a masked activation (the mask
            // is a hard 0/1 multiply, not a soft shrink), so counting exact zeros over total IS
            // the achieved sparsity, not an approximation of it.
            params.cats_calibrate = false;
            params.cats_sparsity = target_sparsity > 0.0f ? target_sparsity : 0.01f;
            params.cats_thresholds_file = verify_thresholds_file;
        }
    } else {
        if (teal_mode) {
            params.teal_calibrate = true;  // forces the un-fused FFN path (attn_in needs no forcing)
        } else {
            params.cats_calibrate = true;  // forces the un-fused, unmasked silu(gate) path
        }
    }
    params.cb_eval = calib_eval_cb;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    const int n_layer = llama_model_n_layer(model);
    g_collector.init(n_layer, verify_mode);

    std::string arch_name(llama_model_arch_string(model) ? llama_model_arch_string(model) : "unknown");

    std::string text = read_text_file(calib_file);
    // Cap the text BEFORE tokenizing -- ~8 chars/token is a safe (loose) upper bound for English
    // BPE text, so this over-reads rather than under-reads.
    const size_t char_cap = (size_t) max_tokens * 8;
    if (text.size() > char_cap) {
        text.resize(char_cap);
    }
    if (text.empty()) {
        LOG_ERR("%s: calibration text is empty\n", __func__);
        return 1;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
    if ((int) tokens.size() > max_tokens) {
        tokens.resize(max_tokens);
    }
    LOG_INF("%s: %s corpus '%s' -> %zu tokens (capped at %d), n_layer=%d, target_sparsity=%.3f, mode=%s\n",
            __func__, teal_mode ? "TEAL" : "CATS", calib_file.c_str(), tokens.size(), max_tokens, n_layer,
            target_sparsity, verify_mode ? "verify" : "calibrate");

    const int n_batch = params.n_batch;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // Each batch is an INDEPENDENT causal chunk (positions reset to 0, KV cache cleared first) --
    // matching tools/imatrix's per-chunk llama_memory_clear pattern (fixed after review: the first
    // CATS-tool version cleared the cache once and reused positions across chunks, contaminating
    // activations with a stale/new-token mix). Independent chunks are the correct unit for a
    // per-layer quantile, not one continuous sequence.
    for (size_t start = 0; start < tokens.size(); start += n_batch) {
        size_t end = std::min(tokens.size(), start + (size_t) n_batch);
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch_clear(batch);
        for (size_t k = start; k < end; ++k) {
            common_batch_add(batch, tokens[k], (llama_pos)(k - start), { 0 }, true);
        }
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: llama_decode failed at token %zu\n", __func__, start);
            llama_batch_free(batch);
            return 1;
        }
        LOG_INF("%s: processed %zu/%zu tokens\n", __func__, end, tokens.size());
    }
    llama_batch_free(batch);

    if (verify_mode) {
        LOG_INF("%s: ACHIEVED SPARSITY (from '%s', on '%s', %zu tokens):\n",
                __func__, verify_thresholds_file.c_str(), calib_file.c_str(), tokens.size());
        for (auto & fam : g_collector.families) {
            LOG_INF("%s: -- family %s --\n", __func__, fam.json_key.c_str());
            for (int il = 0; il < n_layer; ++il) {
                int64_t zc = fam.zero_count[il];
                int64_t tc = fam.total_count[il];
                if (tc == 0) {
                    LOG_ERR("%s: layer %2d: NO masked activations observed (arch-excluded, or threshold "
                            "file has 0 for this layer) -- cannot report achieved sparsity here\n", __func__, il);
                    continue;
                }
                double achieved = (double) zc / (double) tc;
                LOG_INF("%s: layer %2d: achieved=%.4f  (zeros=%lld / total=%lld)%s\n",
                        __func__, il, achieved, (long long) zc, (long long) tc,
                        achieved <= 0.0 ? "  <-- WARNING: threshold zeroed NOTHING at this layer" : "");
            }
        }
        llama_backend_free();
        return 0;
    }

    // Compute the target-sparsity quantile threshold per layer, per family.
    std::ofstream out(out_file);
    if (!out) {
        LOG_ERR("%s: could not open output file '%s'\n", __func__, out_file.c_str());
        return 1;
    }
    out << "{\n";
    out << "  \"arch\": \"" << arch_name << "\",\n";
    out << "  \"n_layer\": " << n_layer << ",\n";
    out << "  \"sparsity_target\": " << target_sparsity << ",\n";
    out << "  \"calib_file\": \"" << calib_file << "\",\n";
    out << "  \"calib_tokens\": " << tokens.size() << ",\n";
    for (size_t fi = 0; fi < g_collector.families.size(); ++fi) {
        auto & fam = g_collector.families[fi];
        std::vector<float> thresholds(n_layer, 0.0f);
        for (int il = 0; il < n_layer; ++il) {
            auto & v = fam.abs_vals[il];
            if (v.empty()) {
                LOG_ERR("%s: family %s layer %d collected ZERO activations -- this layer never took "
                        "the un-fused path (arch exclusion, or not a gated-SiLU/PAR layer); threshold "
                        "left at 0 (no masking will occur there)\n", __func__, fam.json_key.c_str(), il);
                continue;
            }
            size_t k = (size_t)(target_sparsity * (double) v.size());
            if (k >= v.size()) k = v.size() - 1;
            std::nth_element(v.begin(), v.begin() + k, v.end());
            thresholds[il] = v[k];
            LOG_INF("%s: family %s layer %2d: n=%zu threshold=%.6f (target sparsity=%.3f)\n",
                    __func__, fam.json_key.c_str(), il, v.size(), thresholds[il], target_sparsity);
        }
        out << "  \"" << fam.json_key << "\": [";
        for (int il = 0; il < n_layer; ++il) {
            out << thresholds[il];
            if (il + 1 < n_layer) out << ", ";
        }
        out << "]" << (fi + 1 < g_collector.families.size() ? ",\n" : "\n");
    }
    out << "}\n";
    out.close();

    LOG_INF("%s: wrote %s\n", __func__, out_file.c_str());

    llama_backend_free();
    return 0;
}
