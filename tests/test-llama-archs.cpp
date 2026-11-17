#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h" // llama_model::hparams, read by --swa-pattern-dump

#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h> // getpid() for the roundtrip file name
#endif

// TCA_TRACE=1: while the reference (CPU) and the device model compute, record the first values of every contiguous F32 graph node, in execution order, then print the
// first nodes at which the two disagree. This turns "the logits differ" into "the first node that differs", which is what a CUDA-vs-CPU op defect needs.
struct trace_row {
    std::string        name;
    std::vector<float> v;
};
static std::vector<trace_row> * g_trace = nullptr;
static const bool g_trace_on = getenv("TCA_TRACE") != nullptr;
static bool g_col0 = false;

// this fork's ggml_backend_sched_eval_callback returns int (ggml-backend.h), not bool
static int trace_cb(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!g_trace || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) || t->name[0] == 0 || !ggml_is_contiguous(t)) {
        return 0;
    }
    if (ask) {
        return 1;
    }
    trace_row r;
    r.name = t->name;
    // read the whole node (capped) and keep an evenly strided sample of up to 4096 values, so the LAST rows (the canvas tokens of a diffusion model)
    // are covered as well as the first
    // g_col0: record only the first row (ne[0] values = the first token's column) so a 24-token batch and a 1-token step line up
    const size_t n_all = g_col0 ? std::min<size_t>((size_t) t->ne[0], 4096) : std::min<size_t>((size_t) ggml_nelements(t), (size_t) 1 << 20);
    const size_t stride = g_col0 ? 1 : std::max<size_t>(1, n_all / 4096);
    std::vector<float> all(n_all);
    if (t->type == GGML_TYPE_I32) { // expert ids etc.: compared as numbers, so a changed choice shows as a large relative error
        std::vector<int32_t> iv(n_all);
        ggml_backend_tensor_get(t, iv.data(), 0, iv.size() * sizeof(int32_t));
        for (size_t i = 0; i < n_all; i++) {
            all[i] = (float) iv[i];
        }
    } else {
        ggml_backend_tensor_get(t, all.data(), 0, all.size() * sizeof(float));
    }
    for (size_t i = 0; i < n_all; i += stride) {
        r.v.push_back(all[i]);
    }
    g_trace->push_back(std::move(r));
    return 1;
}

static void trace_report(const std::vector<trace_row> & ref, const std::vector<trace_row> & dev) {
    std::map<std::string, int> occ_dev;
    std::map<std::string, std::vector<const trace_row *>> by_name;
    for (const trace_row & r : dev) {
        by_name[r.name].push_back(&r);
    }
    int shown = 0;
    size_t compared = 0;
    for (const trace_row & r : ref) {
        const int k = occ_dev[r.name]++;
        auto it = by_name.find(r.name);
        if (it == by_name.end() || (size_t) k >= it->second.size() || it->second[k]->v.size() != r.v.size()) {
            continue;
        }
        compared++;
        const std::vector<float> & d = it->second[k]->v;
        double err = 0.0, en = 0.0, dmax = 0.0, rmax = 0.0;
        bool finite = true;
        for (size_t i = 0; i < r.v.size(); i++) {
            finite = finite && std::isfinite(r.v[i]) && std::isfinite(d[i]);
            err += (double) (r.v[i] - d[i]) * (r.v[i] - d[i]);
            en  += (double) r.v[i] * r.v[i];
            dmax = std::max(dmax, (double) std::fabs(r.v[i] - d[i]));
            rmax = std::max(rmax, (double) std::fabs(r.v[i]));
        }
        const double rel = en > 0.0 ? std::sqrt(err / en) : (err > 0.0 ? INFINITY : 0.0);
        // TCA_TRACE=<threshold> sets the relative error above which a node is reported (default 1e-3)
        static const double thr = [] { const char * e = getenv("TCA_TRACE"); const double x = e ? atof(e) : 0.0; return x > 0.0 ? x : 1e-3; }();
        // layer outputs, expert choices and the result are always listed (they show how the error grows), whatever the threshold
        // TCA_TRACE_FILTER=<substring>: list every node whose name contains it (e.g. "-1" for all of layer 1), in execution order
        static const char * const flt = getenv("TCA_TRACE_FILTER");
        const bool landmark = r.name.compare(0, 5, "l_out") == 0 || r.name.compare(0, 6, "result") == 0 || r.name.find("moe_topk") != std::string::npos ||
                              (flt && flt[0] && r.name.find(flt) != std::string::npos);
        if (landmark) {
            fprintf(stderr, "[trace-landmark] %-30s n=%-5zu rel err %.3e  max|ref| %.3e  max|diff| %.3e\n", r.name.c_str(), r.v.size(), rel, rmax, dmax);
        }
        if (!finite || rel > thr) {
            if (shown < 12) {
                fprintf(stderr, "[trace] node %-34s n=%-5zu rel err %.3e  max|ref| %.3e  max|diff| %.3e%s\n", r.name.c_str(), r.v.size(), rel, rmax, dmax, finite ? "" : "  NON-FINITE");
            }
            shown++;
        }
    }
    fprintf(stderr, "[trace] compared %zu nodes (reference %zu, device %zu); %d shown (rel err > 1e-3, first values only)\n", compared, ref.size(), dev.size(), shown);
}

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llm_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// Archs whose graphs apply several RMS norms per layer (gemma4 family: attn, q, k, v, post-attn, pre/post-FFN, MoE branch norms). With every weight drawn from
// N(0, 1e-2) the norm weights shrink the activations ~100x at each of them, the logits end up around 1e-8 (or exactly 0 after an F16 intermediate), and a
// CUDA-vs-CPU comparison measures underflow, not the engine. For these, norm / scale weights are drawn around 1 instead (std unchanged).
static bool g_norm_weights_near_one = false;

// Exact attention for the archs listed in the loop: flash attention off and an F32 KV cache. Rounding K/V to F16 (the cache type, and what flash attention does
// internally) turns a 1e-7 CUDA-vs-CPU difference upstream into an occasional 1-ulp-of-F16 (~1e-3) jump, and in a model whose router logits are nearly equal that
// flips a top-k expert choice now and then, after which the two models run different experts. MEASURED 2026-10-04, gemma4 over seeds 1-6 (default F16 cache with
// flash attention): NMSE 2e-3 / 5e-5 / 4e-5 / 7e-2 / 4e-5 / 5e-2, the failing seeds jumping at ffn_moe_probs-0 (sort) with max index difference 1.0; diffusion-gemma,
// which has no KV cache, passed all six at ~1e-9. The flash-attention kernels and the F16 cache have their own coverage in test-backend-ops.
static bool g_exact_attention = false;

static bool is_norm_or_scale_weight(const char * name) {
    const std::string n(name);
    return n.find("norm") != std::string::npos || n.find("scale") != std::string::npos || n.find("_s.") != std::string::npos;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params & params) {
    std::mt19937 gen(params.seed ^ std::hash<std::string>{}(key));
    std::normal_distribution<float> dis(0.0f, params.stdev);
    for (float & v : dst) {
        v = dis(gen);
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// BitNet norm_in weights: the constant 2.0, so test_norm_in can check 2 * x / rms(x) exactly
static void fill_norm_in_two(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (float & v : dst) {
        v = 2.0f;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "_norm_in",    fill_norm_in_two },
    { "ssm_a",       fill_ssm_a       },
    { "_resid_",     fill_resid_gate  },
    { "exp_probs_b", fill_exp_probs_b },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);
    // see g_norm_weights_near_one: norm / scale weights drawn around 1 (same spread) for the archs that set it
    if (g_norm_weights_near_one && fill == fill_normal && is_norm_or_scale_weight(tensor->name)) {
        for (float & v : tmp) {
            v += 1.0f;
        }
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOGV(LOG_LEVEL_OUTPUT, "Usage: %s [options]\n\n", argv[0]);
    LOGV(LOG_LEVEL_OUTPUT, "Options:\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -v <N>                   Set log verbosity level\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -b, --backend <backend>  Run only on the given backend device\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -p, --parity             CPU only: batch vs one-token steps vs chunks vs state-checkpoint logits parity per architecture\n");
    LOGV(LOG_LEVEL_OUTPUT, "  --deepstack-key-layout   CPU only: qwen3vl/qwen3vlmoe n_embd_inp from the native and the ollama deepstack key layouts\n");
    LOGV(LOG_LEVEL_OUTPUT, "  --deepstack-mapping-default  CPU only: the default deepstack layer mapping when the key is absent\n");
    LOGV(LOG_LEVEL_OUTPUT, "  --swa-pattern-dump       CPU only: per-layer is_swa each SWA arch loads to (key as fixture / scalar period 3 / absent)\n");
    LOGV(LOG_LEVEL_OUTPUT, "  --tree-batch             CPU only: a 7-node draft tree decoded in ONE batch (seq-id sets + depth positions) == each root-to-node path alone\n");
    LOGV(LOG_LEVEL_OUTPUT, "                           the five modes above draw weights from N(0, 0.01), the spread they were measured at, unless -d is given\n");
    LOGV(LOG_LEVEL_OUTPUT, "  --verbose                Same as -v 3 (info); a bare -v with no number after it does the same\n");
    LOGV(LOG_LEVEL_OUTPUT, "  -h, --help               Show this help message\n\n");
    LOGV(LOG_LEVEL_OUTPUT, "Examples:\n");
    LOGV(LOG_LEVEL_OUTPUT, "  %s\n", argv[0]);
    LOGV(LOG_LEVEL_OUTPUT, "  %s -a qwen35moe\n", argv[0]);
    LOGV(LOG_LEVEL_OUTPUT, "  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOGV(LOG_LEVEL_OUTPUT, "  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_BAICHUAN) {
        // llama-hparams.cpp maps only 32 layers (7B, rope) and 40 (13B, alibi) to a model.type, and build_baichuan.cpp:52 aborts for any other type
        n_layer = 32;
    } else if (arch == LLM_ARCH_QWEN3NEXT) {
        // llama-hparams.cpp (QWEN3NEXT case) hard-codes "every 4th layer is full attention"; with 2 layers both are recurrent, no K tensor exists
        // and context creation stops at "every per-layer K tensor is null"
        n_layer = 4;
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_DIFFUSION_GEMMA) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, Engram, ratio-2 and ratio-1 sources, top-k reuse and a candidate-masked index layer
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MISTRAL4) {
        // mainline-MLA GGUF shape: ONE kv head and a query-head count divisible by 4 (llama-hparams.cpp, "Adjusted mainline llama.cpp MLA tensors
        // to ik_llama.cpp": n_gqa % 4 == 0, key/value length 576/512, rope 64); with n_head = 1 that compatibility check aborts the load
        n_embd = 128;
        n_head = 4;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_HY_V4 // MLA: one KV head, else the cache row is n_head_kv * 576 wide and set_rows rejects k_cur
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            // MISTRAL4 takes the mainline-MLA branch above (n_head 4, its loader check); HY_V4 is listed at the top of this condition
            || arch == LLM_ARCH_GLM5_NEXT) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 3072; // the hard-coded codec head size: with a larger vocab the logits past 3072 are never written and compare as NaN
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER ||
            arch == LLM_ARCH_KOLIBRI1) {
        n_head_kv = 2; // GQA coverage
    } else if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MISTRAL4) {
        n_head_kv = 1; // mainline-MLA shape, see above
    }
    if (arch == LLM_ARCH_BEAM) {
        n_head    = 4;
        n_head_kv = 2;
        n_layer   = 4;
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_model_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_GLM5_NEXT) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_GLM5NEXT_DASHED) {
        // llama-hparams.cpp, GLM5NEXT case: "per-layer head_count_kv array: 0 marks the KDA (linear-attention) layers, 1 the MLA". With a scalar
        // every layer is MLA and the KDA tensors, the conv/delta state and the hybrid cache are never built; layer 0 stays MLA (it owns the
        // k-pool indexer), layer 1 is KDA, which is the mix the real model has.
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il == 1 ? 0 : 1);   // MLA layers carry ONE latent kv head, as in the real config
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else if (arch == LLM_ARCH_MAMBA) {
        // pure mamba has no attention heads; with heads the K row is n_embd_k_gqa + n_embd_k_s and build_mamba's reshape of k_l asserts
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(0));
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_GLM5NEXT_DASHED) {
        // nope-only MLA (llama-hparams.cpp, GLM5NEXT case: "rope.dimension_count == 0, so n_rot stays 0 and the whole q head is the NoPE part"):
        // the loader's wk_b has dim 0 = n_embd_head_k - n_rot and the graph multiplies it by a q head of n_embd_head_k_full, so the two must be equal
        // key_length / value_length are what the real converter writes (Unsloth conversion/glm5next.py: add_key_length(kv_lora_rank + qk_rope_head_dim),
        // add_value_length(kv_lora_rank)): the ABSORBED latent width of the K/V cache, 512/512 here, NOT the 192/128 MLA head size. The Unsloth engine
        // sizes its attention cache from them, so a fixture with 192/128 aborts it in ggml_set_rows (a->ne[0] == b->ne[0]) and is no oracle input.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(512));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_GLM5NEXT
            || arch == LLM_ARCH_GLM5NEXT_DASHED
            || arch == LLM_ARCH_OPENPANGU
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4) {
        // mistral4's mainline-MLA check expects 320 / 256 where deepseek2's expects 576 / 512 (llama-hparams.cpp, MISTRAL4 / DEEPSEEK2 case)
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_MISTRAL4 ? uint32_t(320) : arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     arch == LLM_ARCH_MISTRAL4 ? uint32_t(256) : uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    if (arch == LLM_ARCH_SENSENOVA_U1) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SPATIAL, 10000.0f);
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    if (arch != LLM_ARCH_KIMI_LINEAR) { // real Kimi-Linear has no Q compression; with a rank its class loader takes the wq_a/wq_b branch that its graph does not implement
        ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,        (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(64) : uint32_t(512));
    }
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_BEAM) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_DIFFUSION_GEMMA) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_COHERE2_MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 ||
            arch == LLM_ARCH_SPARK2_5 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_GRANITE_SWA ||
            arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_FLINT) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);   // mimo2 (and the other SWA-pattern archs) read the SWA rope base
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN, pattern);
        }
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_KOLIBRI1) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, false);
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        // LLAMA_TEST_PLE_LAYER=<n> moves it (default 0): test-ple-state-bytes needs the attention-layer variant, where the state row holds only the history
        const char * ple_layer_env = getenv("LLAMA_TEST_PLE_LAYER");
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ ple_layer_env ? (uint32_t) atoi(ple_layer_env) : 0u }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // glm5next gets 8 indexer heads (the real model has 32): with ONE head the score is relu(q.k)*w, so every pool whose dot product is negative
    // scores exactly 0 and the top-2 of 12 pools is decided by ties; two engines then pick different pools for no reason that is a defect
    // (measured 2026-10-11 against the Unsloth engine). With 8 heads a tie would need all eight dot products to be non-positive.
    // minimax-m3 keeps one indexer head per GQA head and deepseek4 one per head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 || arch == LLM_ARCH_DEEPSEEK4 ? n_head :
                                                     ((arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_GLM5NEXT_DASHED) ? uint32_t(8) : uint32_t(64)));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot; glm5next keeps the 64 it was measured at
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : ((arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_GLM5NEXT_DASHED) ? uint32_t(64) : uint32_t(128)));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    // glm5next keeps top-k 8 / kpool 4 (2 of 12 pools chosen) so the pool selection itself is exercised, with 8 heads against ties
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        (arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_GLM5NEXT_DASHED) ? uint32_t(8) : uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        // bailingmoe3 reads the key (Ling 3.0 VL -> M-RoPE over the 64-wide rotary part set above); the generic value below
        // sums to n_embd_head = 128 > 64 and ggml_rope_multi rejects it. VL-shaped [t, h, w, 0] summing to n_rot/2 instead.
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 2, 2, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,            std::vector<uint32_t>({2, 4}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,       std::vector<uint32_t>({2, 4, 5}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({1, 3}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(3));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   uint32_t(64));
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({ 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                               67716810739261ull, 51510806800915ull, 30921347202721ull }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
        ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,             token_map);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(host_experts_test(arch) ? 64 : 2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            case LLM_ARCH_MELLUM:
                gating_func = 1;
                break;
            case LLM_ARCH_DEEPSEEK4:
            case LLM_ARCH_DEEPSEEK41:
                gating_func = 4;
                break;
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        // keys this fork's loaders require for the MoE archs (S69 2026-10-04: each was the first "key not found" of an arch that could
        // not be built; unused by the others, which ignore extra keys)
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,         uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT,    uint32_t(1));
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        }
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }
    if (arch == LLM_ARCH_GRANITE || arch == LLM_ARCH_GRANITE_MOE || arch == LLM_ARCH_GRANITE_HYBRID) {
        ms.add_kv(LLM_KV_EMBEDDING_SCALE,            1.0f);
        ms.add_kv(LLM_KV_ATTENTION_SCALE,            1.0f);
    }
    // qwen4exp takes upstream's hyper-connection + PLE block above (a superset)
    if (arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_GLM5NEXT_DASHED) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK,            uint32_t(16)); // hc_low_rank
    }
    if (arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_GLM5NEXT_DASHED) {
        // the GLM5NEXT loader takes the pool width from attention.indexer.kpool (llama-hparams.cpp, GLM5NEXT case) and NOT from
        // attention.indexer.block_size above; without it indexer_block_size stays 0 and the k-pool indexer graph is never built, so
        // the fixture only ever exercised dense MLA. top_k 8 / kpool 4 selects 2 pools, so a prompt of more than 8 tokens makes the indexer choose.
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL, uint32_t(4));
    }
    if (arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true); // required (no default) by these two loaders
    }
    // hy_v4: upstream's block above sets these keys and more (swiglu clamp, expert weights scale / norm)
    if (arch == LLM_ARCH_OPENPANGU) {
        ms.add_kv(LLM_KV_OPENPANGU_MHC_NUM_STREAM,             uint32_t(2));
        ms.add_kv(LLM_KV_OPENPANGU_MHC_RECUR_NORM,             uint32_t(4)); // read as u32 into hparams.mhc_recur_norm, which build_openpangu.cpp:979 uses as the Sinkhorn iteration count (ggml_hc_pre needs > 1)
        ms.add_kv(LLM_KV_OPENPANGU_PARAM_SINK_NUMBER,          uint32_t(2));
    }


    if (arch == LLM_ARCH_BEAM) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    // granitehybrid implements one SSM group only (build_granite_hybrid.cpp:79 asserts n_group == 1)
    // mamba (v1) has no SSM groups: a non-zero ssm_n_group makes n_embd_k_s() size a delta-net state (llama-hparams.cpp, n_embd_k_s), which
    // build_mamba's reshape of k_l then rejects (ggml.c:8643, found 2026-10-04); a real mamba GGUF carries no such key.
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_MAMBA) ? 0 : arch == LLM_ARCH_GRANITE_HYBRID ? uint32_t(1) : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_DEEPSEEK41) { // deepseek41's loader requires the key (llama-hparams.cpp, DEEPSEEK41 case)
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_DFLASH_DRAFT) {
        // validate_dflash_hparams: block_size > 1, target layer ids and the target feature width are required (the live target model is checked at DFlash init)
        ms.add_kv(LLM_KV_DFLASH_BLOCK_SIZE,         uint32_t(4));
        ms.add_kv(LLM_KV_DFLASH_N_TARGET_FEATURES,  uint32_t(n_embd * 2));
        ms.add_kv(LLM_KV_DFLASH_TARGET_LAYER_IDS,   std::vector<uint32_t>({0, 1}));
    }
    if (arch == LLM_ARCH_DIFFUSION_GEMMA) {
        // read as a raw key (no arch prefix) by llama-hparams.cpp; REQUIRED, and must be positive
        gguf_set_val_u32(ms.gguf_ctx, "diffusion.canvas_length", 16);
    }

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    // The CPU reference (empty device list) must really compute on the CPU: with op_offload on, ggml's scheduler runs the large-batch
    // matmuls of a host-resident model on the GPU anyway, the "reference" executes the same CUDA kernels as the device model, and
    // the NMSE is a CUDA-vs-CUDA 0.00 (MEASURED 2026-10-04: max |device - reference| = 0.0 bitwise on a CUDA build).
    ctx_params.op_offload = !devs.empty();
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (g_trace_on) {
        ctx_params.cb_eval = trace_cb;
    }
    if (g_exact_attention) {
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.type_k = GGML_TYPE_F32;
        ctx_params.type_v = GGML_TYPE_F32;
    }
    if (getenv("PARITY_NO_FA") != nullptr) {   // parity sweep over the non-flash (transposed-V cache) attention path of every architecture
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_FLINT:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_GEMMA4: // same shared Gemma-4 backbone: the dense config (n_expert = 0) still reaches ggml_mul_mat_id with zero-size expert tensors (ggml.c:8020)
        case LLM_ARCH_DIFFUSION_GEMMA: // loader reads ffn_gate_inp unconditionally and the graph always runs the 128-expert MoE branch; n_expert = 0 gave zero-size expert tensors
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_COHERE2_MOE:
        case LLM_ARCH_OPENPANGU:
        case LLM_ARCH_GLM5NEXT:        // expert-only: the hparams loader requires expert_count (no dense fallback)
        case LLM_ARCH_GLM5NEXT_DASHED:
        case LLM_ARCH_GRANITE_MOE: // an expert-only arch: its tensor table has no dense FFN entries
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_KOLIBRI1:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        // REFACT is not listed: its graph (build_refact) has only the dense FFN and dereferences ffn_up, which an expert layer does not have
        // MINICPM is not listed either: build_minicpm.cpp has no expert code at all and its tensor table maps no FFN_*_EXPS entries
        // (the MoE config failed at "tensor 'blk.0.ffn_gate.0.weight' not found")
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4_ASSISTANT || arch == LLM_ARCH_GEMMA4_MTP) {
        // ROOT-CAUSE: assistant heads that draw their K/V and hidden state from a TARGET model (mtp_target_ctx / ctx_other); with no target, gemma4_mtp builds a
        // 3-node graph and 1280 of 16384 reference logits are non-finite. CONFIGS-TRIED: 2 (fixture with the required backbone key and gemma4 dims; same with the
        // saver writing the MTP keys). ISOLATED: test-llama-archs -a gemma4_mtp, 2026-10-04 -- the saved file also fails to read back (tensor offset mismatch).
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        // ROOT-CAUSE (isolated 2026-10-04): the DSV4 tensor creator infers every shape from the FILE's tensor metadata (create_deepseek4_tensors / create_tensor_from_meta:
        // ml.require_tensor_meta("token_embd.weight") ...) and the hparams loader probes attn_output_a/b, hc_head_base and (deepseek41) the Engram tables the same way.
        // A weightless model has no file tensors, so there is nothing to discover from; this is not a defect in either arch. These two archs therefore have NO CUDA-vs-CPU
        // comparison in this harness; open row S69-FORK-TEST-LLAMA-ARCHS-DEEPSEEK4-41-TENSOR-BEARING-FIXTURE-2026-10-04-01 tracks a tensor-bearing random-weight fixture.
        // (An earlier version of this comment cited 7e187206b as real-file DSV4 coverage; that commit is a gemma3 KQ-mask guard, unrelated.)
        // deepseek41 additionally needs engram.* keys.
        return false;
    }
    if (arch == LLM_ARCH_DFLASH_DRAFT) {
        // ROOT-CAUSE: the draft attends over the TARGET model's K/V (lctx.dflash.kv.k_ctx_cache, build_dflash.cpp:491 GGML_ASSERT(k_ctx_cache[il]->ne[1] >= n_kv_total)); with no
        // target there is no context to read. ISOLATED 2026-10-04: with the required dflash keys and swa_type paired it loads and fails only there.
        return false;
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_MODERNBERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    if (arch == LLM_ARCH_INKLING) {
        return false; // S69 P4 registered the arch (converter/quantize/load); the compute graph is step P1, so a decode would abort by design
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4_ASSISTANT || arch == LLM_ARCH_GEMMA4_MTP || arch == LLM_ARCH_DFLASH_DRAFT) {
            continue; // needs a target model (see arch_supported)
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_DFLASH2) {
            continue; // see the same skip in the compare loop below: "dflash" resolves to DFLASH, which needs tensor metadata
        }
        if (arch == LLM_ARCH_INKLING) {
            continue; // S69 P4: needs inkling.* keys and a per-layer pattern this generic fixture does not write; graph is step P1
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_model_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens_all = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_model_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOGV(LOG_LEVEL_OUTPUT, template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch");
    LOGV(LOG_LEVEL_OUTPUT, "|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOGV(LOG_LEVEL_OUTPUT, "-");
    }
    LOGV(LOG_LEVEL_OUTPUT, "|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOGV(LOG_LEVEL_OUTPUT, "-");
    }
    LOGV(LOG_LEVEL_OUTPUT, "|------|---------------|---------|---------------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4_ASSISTANT || arch == LLM_ARCH_GEMMA4_MTP || arch == LLM_ARCH_DFLASH_DRAFT) {
            continue; // needs a target model (see arch_supported)
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_DFLASH2) {
            continue; // DFLASH2 shares the file name "dflash" with DFLASH (src/llama-arch.cpp:98-99), so the loader resolves the fixture to DFLASH,
                      // whose hparams loader probes tensor metadata (llama-hparams.cpp:2446-2456) that a weightless fixture does not have
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        // non-causal attention needs n_ubatch >= n_tokens (llama.cpp, llama_decode); DiffusionGemma is non-causal but is driven through llama_decode, not llama_encode
        const bool full_ubatch = encode || arch == LLM_ARCH_DIFFUSION_GEMMA;
        g_norm_weights_near_one = arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_DIFFUSION_GEMMA;
        g_exact_attention = arch == LLM_ARCH_GEMMA4;
        const std::vector<llama_token> & tokens = tokens_all;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            std::vector<trace_row> trace_cpu, trace_dev, trace_rt;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOGV(LOG_LEVEL_OUTPUT, template_row_cfg.c_str(), llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                // tensor split requires flash attention (llama_init_from_model: "SPLIT_MODE_TENSOR requires flash_attn to be enabled"), which g_exact_attention turns off.
                // DISABLE-REVERSIBLE: g_exact_attention (test-only; set per arch in the loop above) -- drop the arch from that flag and this row runs again.
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && (dc.devs.empty() || g_exact_attention));
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, full_ubatch);
                        trace_cpu.clear();
                        g_trace = g_trace_on ? &trace_cpu : nullptr;
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                        g_trace = nullptr;
                    }
                    // build-level gate first: the loader refuses -sm tensor while llm_build_supports_sm_tensor() is false,
                    // so the Meta row is a SKIP (not run) rather than a load failure (gate b1109h 2026-11-09: 4 archs rc -1)
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || (llm_build_supports_sm_tensor() && llm_arch_supports_sm_tensor(arch))) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, full_ubatch, overrides, dc.moe_cache_size);
                        trace_dev.clear();
                        g_trace = g_trace_on ? &trace_dev : nullptr;
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        g_trace = nullptr;
                        if (g_trace_on) {
                            trace_report(trace_cpu, trace_dev);
                        }
                        // `nmse_val > 1e-4` is false for NaN, so a NaN logit tensor used to be reported OK (and then failed the roundtrip
                        // compare, since NaN != NaN). Non-finite is a failure, and say which side holds it. A reference whose logits are
                        // all zero has no energy to normalise by (nmse = 0/0 = NaN); that case is compared by absolute difference instead.
                        size_t nf_cpu = 0, nf_dev = 0;
                        double ref_energy = 0.0, max_abs_dev = 0.0;
                        for (float v : logits_cpu) { nf_cpu += !std::isfinite(v); ref_energy += (double) v * (double) v; }
                        for (float v : logits_dev) { nf_dev += !std::isfinite(v); max_abs_dev = std::max(max_abs_dev, (double) std::fabs(v)); }
                        if (nf_cpu || nf_dev) {
                            fprintf(stderr, "\n[non-finite logits] reference model: %zu of %zu, device model: %zu of %zu\n",
                                    nf_cpu, logits_cpu.size(), nf_dev, logits_dev.size());
                        }
                        double nmse_val = nmse(logits_cpu, logits_dev);
                        if (nf_cpu == 0 && nf_dev == 0 && logits_cpu.size() == logits_dev.size()) {
                            double max_ref = 0.0, max_diff = 0.0;
                            for (size_t i = 0; i < logits_cpu.size(); i++) {
                                max_ref  = std::max(max_ref,  (double) std::fabs(logits_cpu[i]));
                                max_diff = std::max(max_diff, (double) std::fabs(logits_cpu[i] - logits_dev[i]));
                            }
                            fprintf(stderr, "\n[logits] reference rms %.3e, max |ref| %.3e, max |device - ref| %.3e\n",
                                    std::sqrt(ref_energy / std::max<size_t>(1, logits_cpu.size())), max_ref, max_diff);
                        }
                        if (nf_cpu == 0 && nf_dev == 0 && !(ref_energy > 0.0)) {
                            fprintf(stderr, "\n[zero reference logits] compared by absolute difference (max |device logit| = %.3e)\n", max_abs_dev);
                            nmse_val = max_abs_dev <= 1e-6 ? 0.0 : INFINITY;
                        }
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (!(nmse_val <= 1e-4)) { // NaN-safe: a NaN NMSE fails
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llm_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        // fork: skipped for full_ubatch archs too -- DiffusionGemma runs non-causal with n_ubatch = the whole context
                        // (above), so the toggle check's own precondition n_past + 1.5 * n_ubatch <= n_ctx cannot hold for it
                        if (!full_ubatch && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
#ifdef _WIN32
                    // MEASURED 2026-10-07 on this host: tmpfile() returned null, so the Roundtrip column read SKIP for EVERY architecture
                    // and no saved-and-reloaded model was ever compared here. A named file opened delete-on-close ("D") does the same job.
                    if (file == nullptr) {
                        const std::string rt_name = "test-llama-archs-roundtrip-" + std::to_string((long long) getpid()) + ".gguf";
                        file = fopen(rt_name.c_str(), "w+bD");
                    }
#endif
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, full_ubatch, overrides, dc.moe_cache_size);
                        trace_rt.clear();
                        g_trace = g_trace_on ? &trace_rt : nullptr;
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        g_trace = nullptr;
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        double rt_max_diff = 0.0;
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            rt_max_diff = std::max(rt_max_diff, (double) std::fabs(logits_roundtrip[i] - logits_dev[i]));
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                test_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                            }
                        }
                        if (status_roundtrip.find("FAIL") != std::string::npos) {
                            fprintf(stderr, "\n[roundtrip] reloaded model's logits differ from the device model's: max |diff| %.3e\n", rt_max_diff);
                            if (g_trace_on) {
                                fprintf(stderr, "[roundtrip] first differing nodes, device model vs reloaded model:\n");
                                trace_report(trace_dev, trace_rt);
                            }
                        }
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOGV(LOG_LEVEL_OUTPUT, template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    if (n_tests == 0) {
        LOGV(LOG_LEVEL_OUTPUT, "Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOGV(LOG_LEVEL_OUTPUT, "Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOGV(LOG_LEVEL_OUTPUT, "Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

// ---- prefill / one-token-stepwise / chunked / state-checkpoint parity ------------------------------------------------------------
// S69-A-PICK-PREFILL-VS-STEPWISE-DECODE-PARITY-TEST-ACROSS-ALL-ARCHS-2026-10-05-01. CPU only, random weights, same model each time.
// RESEARCH-DONE: https://github.com/pytorch/executorch/issues/21858 (read in full 2026-10-05 by agent A: the failure class this test targets)
// The backend comparison above can not see a prompt-path / decode-path divergence: both devices run the same wrong path. The
// failure class is real (an LFM2.5 hybrid steps correctly token by token and degenerates under prefill-style driving, caused by
// the short-convolution state). For every architecture this runs the SAME 24 tokens
//   A  as one llama_decode          B  as 24 single-token llama_decode calls      C  as 3 chunks (7 + 9 + 8)
//   D  12 tokens, llama_state_seq_get_data, memory clear, llama_state_seq_set_data, then the remaining 12
// and compares the logits of EVERY position against A (NMSE <= 1e-4, the threshold of the backend comparison above).
// Rows that are not run say why on the same line (encoder / diffusion archs need the whole batch at once; the rest repeat the
// exclusions of the backend comparison). DISABLE-REVERSIBLE: these are the same per-arch exclusions test_backends() carries.
static bool parity_decode_span(llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, int lo, int hi, std::vector<float> & out) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(hi - lo, 0, 1);
    for (int p = lo; p < hi; ++p) {
        common_batch_add(batch, tokens[p], p, {0}, true);
    }
    const int rc = llama_decode(lctx, batch);
    if (rc == 0) {
        for (int i = 0; i < hi - lo; ++i) {
            const float * row = llama_get_logits_ith(lctx, i);
            out.insert(out.end(), row, row + n_vocab);
        }
    }
    llama_batch_free(batch);
    return rc == 0;
}

static bool parity_all_finite(const std::vector<float> & v) {
    for (float f : v) {
        if (!std::isfinite(f)) {
            return false;
        }
    }
    return true;
}

static int test_parity(const std::string & arch_filter, const size_t seed, const float stdev, const ggml_log_level log_level) {
    llama_log_set([](ggml_log_level level, const char * text, void * ud) {
        if (level >= *(const ggml_log_level *) ud) {
            fputs(text, stderr);
        }
    }, new ggml_log_level(log_level));

    constexpr int T = 24;
    const std::vector<llama_token> tokens = get_tokens(T, 128, seed);
    int n_run = 0, n_fail = 0, n_skip = 0;

    // MUST-FAIL CONTROL: the comparison below reads 0.0e+00 for most architectures (F32 random weights make batch and
    // single-token paths bit-identical), so a blind comparison would look the same. Decode the same prompt with ONE token
    // changed and require the NMSE to exceed the threshold; if it does not, this harness can not fail and is stopped.
    if (arch_matches(arch_filter, LLM_ARCH_LLAMA)) {
        gguf_context_ptr gg = get_gguf_ctx(LLM_ARCH_LLAMA, false);
        std::vector<llama_token> alt = tokens;
        alt[T / 2] = (alt[T / 2] + 37) % 128;
        std::vector<float> ra, rb;
        {
            auto mc = get_model_and_ctx(gg.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, false);
            if (!parity_decode_span(mc.first.get(), mc.second.get(), tokens, 0, T, ra)) { printf("parity control: decode failed\n"); return 1; }
        }
        {
            auto mc = get_model_and_ctx(gg.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, false);
            if (!parity_decode_span(mc.first.get(), mc.second.get(), alt, 0, T, rb)) { printf("parity control: decode failed\n"); return 1; }
        }
        const double ce = nmse(ra, rb);
        printf("parity control (one prompt token changed, must exceed 1e-4): NMSE %.3e -> %s\n", ce, ce > 1e-4 ? "detected" : "BLIND");
        if (!(ce > 1e-4)) { return 1; }
    }
    printf("|%-22s|%-6s|%16s|%16s|%16s|\n", "Model arch.", "Config", "step vs batch", "chunks vs batch", "ckpt vs batch");
    printf("|----------------------|------|----------------|----------------|----------------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4_ASSISTANT || arch == LLM_ARCH_GEMMA4_MTP || arch == LLM_ARCH_DFLASH_DRAFT ||
                arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_DFLASH2) {
            continue; // same exclusions as the backend comparison (need a target model / tensor-bearing fixture)
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * cfg = moe ? "MoE" : "Dense";
            if (!arch_supported(arch)) {
                printf("|%-22s|%-6s|%16s|%16s|%16s|  NOT RUN: arch_supported() is false\n", llama_model_arch_name(arch), cfg, "-", "-", "-");
                ++n_skip;
                continue;
            }
            if (encode || arch == LLM_ARCH_DIFFUSION_GEMMA) {
                printf("|%-22s|%-6s|%16s|%16s|%16s|  NOT RUN: non-causal / encoder arch, needs the whole batch at once\n", llama_model_arch_name(arch), cfg, "-", "-", "-");
                ++n_skip;
                continue;
            }
            g_norm_weights_near_one = arch == LLM_ARCH_GEMMA4;
            g_exact_attention       = arch == LLM_ARCH_GEMMA4;
            ++n_run;
            std::string s_step = "?", s_chunk = "?", s_ckpt = "?", note;
            bool ok = true;
            try {
                gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
                if (arch == LLM_ARCH_BAILINGMOE3) {
                    GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
                }
                auto make = [&]() { return get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, false); };
                char buf[48];
                auto verdict = [&](const char * what, const std::vector<float> & ref, const std::vector<float> & got, std::string & cell) {
                    if (got.size() != ref.size() || !parity_all_finite(got) || !parity_all_finite(ref)) {
                        cell = "FAIL(size/nan)";
                        note += std::string(" ") + what + ":size=" + std::to_string(got.size()) + "/" + std::to_string(ref.size());
                        ok = false;
                        return;
                    }
                    double energy = 0.0;
                    for (float f : ref) { energy += (double) f * (double) f; }
                    if (!(energy > 0.0)) {   // a zero-logit reference compares nothing: say so instead of reporting a clean 0
                        cell = "FAIL(zero ref)";
                        note += std::string(" ") + what + ":reference logits are all zero";
                        ok = false;
                        return;
                    }
                    const double e = nmse(ref, got);
                    snprintf(buf, sizeof(buf), "%s(%.1e)", e <= 1e-4 ? "ok" : "FAIL", e);
                    cell = buf;
                    if (!(e <= 1e-4)) {
                        ok = false;
                        // where does it start? first position whose own NMSE exceeds the threshold
                        const size_t nv = ref.size() / (size_t) T;
                        for (int p = 0; p < T; ++p) {
                            std::vector<float> r(ref.begin() + p * nv, ref.begin() + (p + 1) * nv), g(got.begin() + p * nv, got.begin() + (p + 1) * nv);
                            double en = 0.0;
                            for (float f : r) { en += (double) f * (double) f; }
                            const double ep = en > 0.0 ? nmse(r, g) : 0.0;
                            if (ep > 1e-4) {
                                note += std::string(" ") + what + ":first bad position " + std::to_string(p) + " nmse " + std::to_string(ep);
                                break;
                            }
                        }
                    }
                };

                std::vector<float> a, b, c, d;
                {   // A: one batch
                    auto mc = make();
                    if (!parity_decode_span(mc.first.get(), mc.second.get(), tokens, 0, T, a)) { throw std::runtime_error("batch decode failed"); }
                }
                {   // B: single tokens
                    auto mc = make();
                    for (int p = 0; p < T; ++p) {
                        if (!parity_decode_span(mc.first.get(), mc.second.get(), tokens, p, p + 1, b)) { throw std::runtime_error("step decode failed"); }
                    }
                }
                {   // C: 7 + 9 + 8
                    auto mc = make();
                    const int cuts[] = {0, 7, 16, T};
                    for (int k = 0; k < 3; ++k) {
                        if (!parity_decode_span(mc.first.get(), mc.second.get(), tokens, cuts[k], cuts[k + 1], c)) { throw std::runtime_error("chunk decode failed"); }
                    }
                }
                {   // D: 12 tokens, save the sequence state, wipe the memory, restore, continue
                    auto mc = make();
                    if (!parity_decode_span(mc.first.get(), mc.second.get(), tokens, 0, 12, d)) { throw std::runtime_error("ckpt first half failed"); }
                    const size_t sz = llama_state_seq_get_size(mc.second.get(), 0, 0);
                    std::vector<uint8_t> state(sz);
                    const size_t got = llama_state_seq_get_data(mc.second.get(), state.data(), state.size(), 0, 0);
                    if (sz == 0 || got != sz) { throw std::runtime_error("llama_state_seq_get_data size mismatch"); }
                    llama_memory_clear(llama_get_memory(mc.second.get()), true);
                    if (llama_state_seq_set_data(mc.second.get(), state.data(), state.size(), 0, 0) == 0) { throw std::runtime_error("llama_state_seq_set_data returned 0"); }
                    if (!parity_decode_span(mc.first.get(), mc.second.get(), tokens, 12, T, d)) { throw std::runtime_error("ckpt second half failed"); }
                }
                verdict("step",  a, b, s_step);
                verdict("chunk", a, c, s_chunk);
                verdict("ckpt",  a, d, s_ckpt);
                if (!ok && g_trace_on) {
                    // TCA_TRACE=1: the first graph nodes at which the 24-token batch and a 1-token decode of the SAME first token disagree
                    std::vector<trace_row> tA, tB;
                    g_col0 = true;
                    { auto mc = make(); std::vector<float> o; g_trace = &tA; parity_decode_span(mc.first.get(), mc.second.get(), tokens, 0, T, o); g_trace = nullptr; }
                    { auto mc = make(); std::vector<float> o; g_trace = &tB; parity_decode_span(mc.first.get(), mc.second.get(), tokens, 0, 1, o); g_trace = nullptr; }
                    g_col0 = false;
                    std::map<std::string, int> occ;
                    std::map<std::string, std::vector<const trace_row *>> byname;
                    for (const trace_row & r : tB) { byname[r.name].push_back(&r); }
                    int shown = 0;
                    fprintf(stderr, "\n[parity trace] %s: nodes recorded batch=%zu step=%zu; first nodes whose token-0 row differs:\n", llama_model_arch_name(arch), tA.size(), tB.size());
                    for (const trace_row & r : tA) {
                        const int k = occ[r.name]++;
                        auto it = byname.find(r.name);
                        if (it == byname.end() || (size_t) k >= it->second.size() || it->second[k]->v.size() != r.v.size() || r.v.empty()) { continue; }
                        double mx = 0.0, md = 0.0;
                        for (size_t i = 0; i < r.v.size(); ++i) {
                            mx = std::max(mx, (double) std::fabs(r.v[i]));
                            md = std::max(md, (double) std::fabs(r.v[i] - it->second[k]->v[i]));
                        }
                        if (md > 1e-3 * (mx + 1e-12)) {
                            fprintf(stderr, "  %-34s occurrence %d  n=%zu  max|batch| %.4e  max|batch-step| %.4e\n", r.name.c_str(), k, r.v.size(), mx, md);
                            if (shown < 3) {
                                fprintf(stderr, "     batch:");
                                for (size_t i = 0; i < std::min<size_t>(8, r.v.size()); ++i) { fprintf(stderr, " %.4g", r.v[i]); }
                                fprintf(stderr, "\n     step :");
                                for (size_t i = 0; i < std::min<size_t>(8, r.v.size()); ++i) { fprintf(stderr, " %.4g", it->second[k]->v[i]); }
                                fprintf(stderr, "\n");
                            }
                            if (++shown >= 10) { break; }
                        }
                    }
                }
            } catch (const std::exception & err) {
                ok = false;
                note = std::string(" EXCEPTION: ") + err.what();
            }
            if (!ok) { ++n_fail; }
            printf("|%-22s|%-6s|%16s|%16s|%16s|%s\n", llama_model_arch_name(arch), cfg, s_step.c_str(), s_chunk.c_str(), s_ckpt.c_str(), note.c_str());
            fflush(stdout);
        }
    }
    printf("parity: %d configurations run, %d failed, %d not run (stated above)\n", n_run, n_fail, n_skip);
    return n_fail == 0 ? 0 : 1;
}

// Tree-shaped speculative verification through the EXISTING batch API (row S69-F1-PICK-TREE-MASK-VERIFY-BATCH-API-2026-10-05-01).
// A draft tree is decoded in ONE batch: every leaf owns one sequence id, a node carries the ids of all leaves below it (a shared
// prefix is one KV cell with several seq ids) and its position is prefix length + depth, so several nodes share a position. The KV
// mask built from seq-id membership + position is then exactly the ancestry mask: a node sees the committed prefix, its ancestors
// and itself, never a sibling branch. Reference: each node's root-to-node path decoded alone as a plain sequence in a fresh context.
//
//   tree (7 nodes)       r                 seq sets   r {0,1,2,3}   a {0,1}   b {2,3}
//                      /   \                          a1 {0}  a2 {1}  b1 {2}  b2 {3}
//                     a     b              positions  P, P+1, P+1, P+2 x 4
//                    / \   / \
//                  a1  a2 b1  b2
//
// PASS: every node's logits match its path reference (NMSE <= 1e-6), with flash attention off and on, and with the shared prefix
// written once with four seq ids. MUST-FAIL CONTROL: the same tokens and positions on ONE sequence (no ancestry sets) must differ
// from the path reference at the second branch (b, b1): if it does not, the comparison cannot see a sibling leak and the run stops.
static int test_tree_batch(const size_t seed, const float stdev, const ggml_log_level log_level) {
    llama_log_set([](ggml_log_level level, const char * text, void * ud) {
        if (level >= *(const ggml_log_level *) ud) {
            fputs(text, stderr);
        }
    }, new ggml_log_level(log_level));

    constexpr int P = 6;                                  // committed prefix
    constexpr int N = 7;                                  // tree nodes
    const int parent[N] = { -1, 0, 0, 1, 1, 2, 2 };
    const int depth [N] = {  0, 1, 1, 2, 2, 2, 2 };
    const std::vector<std::vector<llama_seq_id>> seqs = { {0, 1, 2, 3}, {0, 1}, {2, 3}, {0}, {1}, {2}, {3} };
    const std::vector<llama_token> all = get_tokens(P + N, 128, seed);
    const std::vector<llama_token> prefix(all.begin(), all.begin() + P);
    const std::vector<llama_token> node(all.begin() + P, all.end());

    // Norm weights near 1 (not the default N(0, 1e-2)): with near-zero norms every activation is ~1e-2 of its size, attention
    // is near-uniform and a sibling leak moves the logits by only ~1e-6 NMSE (MEASURED, seed 42: single-seq control 1.5e-6 at
    // node b), too close to rounding to call. Restored on exit so the other modes keep their own setting.
    const bool norm_prev = g_norm_weights_near_one;
    g_norm_weights_near_one = true;
    struct restore_norm { bool v; ~restore_norm() { g_norm_weights_near_one = v; } } restore_norm_guard{ norm_prev };

    gguf_context_ptr gg = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> no_devs = { nullptr };   // CPU only
    mp.devices = no_devs.data();
    tensor_data_params tmp = { seed, stdev }; // set_tensor_data reads a tensor_data_params (a bare size_t here read past it)
    llama_model_ptr model(llama_model_init_from_user(gg.get(), set_tensor_data, &tmp, mp));
    if (!model) {
        printf("tree: model creation failed\n");
        return 1;
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    auto make_ctx = [&](uint32_t n_seq, bool fa) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 128;
        cp.n_batch         = 64;
        cp.n_ubatch        = 64;
        cp.n_seq_max       = n_seq;
        cp.kv_unified      = true;
        cp.op_offload      = false;
        cp.n_threads       = 4;
        cp.n_threads_batch = 4;
        cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context_ptr c(llama_init_from_model(model.get(), cp));
        if (!c) {
            throw std::runtime_error("failed to create llama context");
        }
        return c;
    };
    auto row = [&](llama_context * c, int i) {
        const float * l = llama_get_logits_ith(c, i);
        return std::vector<float>(l, l + n_vocab);
    };
    auto energy_ok = [](const std::vector<float> & v) {
        double e = 0.0;
        for (float f : v) { if (!std::isfinite(f)) { return false; } e += (double) f * f; }
        return e > 0.0;
    };

    int n_fail = 0;
    for (bool fa : { false, true }) {
        // reference: root-to-node path alone, one sequence, fresh context per node
        std::vector<std::vector<float>> ref(N);
        for (int i = 0; i < N; ++i) {
            std::vector<int> path;
            for (int k = i; k >= 0; k = parent[k]) { path.insert(path.begin(), k); }
            llama_context_ptr c = make_ctx(1, fa);
            llama_batch b = llama_batch_init(P + N, 0, 1);
            for (int p = 0; p < P; ++p) { common_batch_add(b, prefix[p], p, {0}, false); }
            for (size_t k = 0; k < path.size(); ++k) { common_batch_add(b, node[path[k]], P + (int) k, {0}, k + 1 == path.size()); }
            if (llama_decode(c.get(), b) != 0) { llama_batch_free(b); throw std::runtime_error("reference decode failed"); }
            ref[i] = row(c.get(), b.n_tokens - 1);
            llama_batch_free(b);
            if (!energy_ok(ref[i])) { printf("tree: reference logits of node %d are zero or non-finite\n", i); return 1; }
        }

        // tree: prefix once with all four seq ids, then the 7 nodes in ONE batch
        std::vector<std::vector<float>> got(N);
        {
            llama_context_ptr c = make_ctx(4, fa);
            llama_batch b = llama_batch_init(P + N, 0, 4);
            for (int p = 0; p < P; ++p) { common_batch_add(b, prefix[p], p, {0, 1, 2, 3}, false); }
            if (llama_decode(c.get(), b) != 0) { llama_batch_free(b); throw std::runtime_error("tree prefix decode failed"); }
            common_batch_clear(b);
            for (int i = 0; i < N; ++i) { common_batch_add(b, node[i], P + depth[i], seqs[i], true); }
            if (llama_decode(c.get(), b) != 0) { llama_batch_free(b); throw std::runtime_error("tree batch decode failed"); }
            for (int i = 0; i < N; ++i) { got[i] = row(c.get(), i); }
            llama_batch_free(b);
        }

        // control: same tokens and positions, ONE sequence -> every node also sees the other branch
        std::vector<std::vector<float>> ctl(N);
        {
            llama_context_ptr c = make_ctx(1, fa);
            llama_batch b = llama_batch_init(P + N, 0, 1);
            for (int p = 0; p < P; ++p) { common_batch_add(b, prefix[p], p, {0}, false); }
            for (int i = 0; i < N; ++i) { common_batch_add(b, node[i], P + depth[i], {0}, true); }
            if (llama_decode(c.get(), b) != 0) { llama_batch_free(b); throw std::runtime_error("control decode failed"); }
            for (int i = 0; i < N; ++i) { ctl[i] = row(c.get(), P + i); }
            llama_batch_free(b);
        }

        double worst = 0.0;
        for (int i = 0; i < N; ++i) {
            const double e = nmse(ref[i], got[i]);
            worst = std::max(worst, e);
            printf("tree fa=%d node %d (depth %d, seqs %zu): tree vs path NMSE %.3e  single-seq control vs path NMSE %.3e\n",
                   (int) fa, i, depth[i], seqs[i].size(), e, nmse(ref[i], ctl[i]));
            if (!(e <= 1e-6)) { ++n_fail; }
        }
        const double c_b = nmse(ref[2], ctl[2]), c_b1 = nmse(ref[5], ctl[5]);
        printf("tree fa=%d: worst tree NMSE %.3e -> %s; control at the second branch b %.3e, b1 %.3e -> %s\n", (int) fa, worst,
               worst <= 1e-6 ? "ok" : "FAIL", c_b, c_b1, (c_b > 1e-4 && c_b1 > 1e-4) ? "detected" : "BLIND");
        if (!(c_b > 1e-4 && c_b1 > 1e-4)) {
            printf("tree: the single-sequence control did not differ, so this comparison cannot see a sibling leak\n");
            return 1;
        }
    }
    printf("tree: %d node comparison(s) failed\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}

// Qwen3-VL deepstack layer count, read from two GGUF layouts (row S69-VL-SERVE-ENDPOINT-UNSLOTH-MMPROJ-01):
//   native  : "<arch>.n_deepstack_layers" = N                        (llama.cpp converters)
//   ollama  : "<arch>.vision.deepstack_visual_indexes" = [N indices] (the qwen3-vl blob in models/ollama; no native key)
// The multimodal projector emits (1 + N) * n_embd values per image token and mtmd compares that against
// llama_model_n_embd_inp(); a text model that reads N = 0 from the ollama layout is refused with
// "mismatch between text model (n_embd = 4096) and mmproj (n_embd = 16384)".
// Each case builds a tiny random-weight text model on the CPU, checks the input width, and decodes 16 text tokens
// so the (1 + N) fake-width graph is also exercised. The cases that expect 1x are the must-fail partners of the 4x ones.
// RESEARCH-DONE: ggml/include/gguf.h:138,145,155 -- gguf_remove_key / gguf_set_val_u32 / gguf_set_arr_data edit the in-memory context only (test input, no product path).
static int test_deepstack_key_layout(const size_t seed, const float stdev) {
    struct dcase { const char * name; int native; int arr_len; int mult; };   // native < 0: key absent; arr_len 0: array absent
    const dcase cases[] = {
        { "native key = 3",                         3, 0, 4 },
        { "ollama array of 3 only",                -1, 3, 4 },
        { "neither key (a plain text qwen3vl)",    -1, 0, 1 },
        { "native key = 1 wins over an array of 3", 1, 3, 2 },
        { "native key = 0 wins over an array of 3", 0, 3, 1 },
    };
    const uint32_t n_embd_base = 256;   // get_gguf_ctx(): n_embd for every arch not special-cased there
    int n_fail = 0;
    const std::vector<llama_token> tokens = get_tokens(16, 128, seed);
    for (const llm_arch arch : { LLM_ARCH_QWEN3VL, LLM_ARCH_QWEN3VLMOE }) {
        const std::string prefix = llama_model_arch_name(arch);
        for (const dcase & c : cases) {
            gguf_context_ptr g = get_gguf_ctx(arch, arch == LLM_ARCH_QWEN3VLMOE);
            const std::string k_native = prefix + ".n_deepstack_layers";
            const std::string k_arr    = prefix + ".vision.deepstack_visual_indexes";
            gguf_remove_key(g.get(), k_native.c_str());
            gguf_remove_key(g.get(), k_arr.c_str());
            if (c.native >= 0) {
                gguf_set_val_u32(g.get(), k_native.c_str(), (uint32_t) c.native);
            }
            if (c.arr_len > 0) {
                std::vector<uint32_t> idx;
                for (int i = 0; i < c.arr_len; i++) { idx.push_back(8u * (i + 1)); }
                gguf_set_arr_data(g.get(), k_arr.c_str(), GGUF_TYPE_UINT32, idx.data(), idx.size());
            }
            int32_t got = -1;
            std::string err;
            try {
                auto mc = get_model_and_ctx(g.get(), nullptr, seed, stdev, {});
                got = llama_model_n_embd_inp(mc.first.get());
                const std::vector<float> lg = get_logits(mc.first.get(), mc.second.get(), tokens);
                for (const float v : lg) { if (!std::isfinite(v)) { err = "non-finite logit"; break; } }
            } catch (const std::exception & e) {
                err = e.what();
            }
            const int32_t want = (int32_t) (n_embd_base * c.mult);
            const bool ok = err.empty() && got == want;
            printf("deepstack-key-layout %-9s %-42s n_embd_inp %6d want %6d  %s%s%s\n", prefix.c_str(), c.name, got, want,
                   ok ? "[PASS]" : "[FAIL]", err.empty() ? "" : " : ", err.c_str());
            n_fail += ok ? 0 : 1;
        }
    }
    printf("deepstack-key-layout: %d failure(s)\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}

// Granite deepstack mapping default (row S69-B9590RES-PICK-DEEPSTACK-MAPPING-DEFAULT-2026-10-18-01). The src/models granite
// classes (granite.cpp, granite-swa.cpp) count every deepstack_mapping_arr entry >= 0 as a deepstack layer and add an input
// view to every decoder layer whose entry is >= 0. Upstream 64086f2b2 (src/llama-model.cpp:1103, MIT) fills the array with
// -1 before any loader reads it; without that fill the value-initialised zeros mean "deepstack index 0" for every layer,
// so a plain text granite_swa GGUF (no key) gets n_deepstack_layers = 1 and n_embd_inp = 2 * n_embd.
static int test_deepstack_mapping_default(const size_t seed, const float stdev) {
    struct dcase { const char * name; std::vector<int32_t> mapping; int mult; };   // empty mapping: key absent
    const dcase cases[] = {
        { "key absent (a plain text granite_swa)", {},       1 },
        { "explicit [-1, -1]",                     { -1, -1 }, 1 },
        { "[-1, 0] (one deepstack input)",         { -1, 0 },  2 },   // control: the key IS read
    };
    const uint32_t n_embd_base = 256;   // get_gguf_ctx(): n_embd for every arch not special-cased there
    int n_fail = 0;
    const std::vector<llama_token> tokens = get_tokens(16, 128, seed);
    const llm_arch arch = LLM_ARCH_GRANITE_SWA;
    const std::string key = std::string(llama_model_arch_name(arch)) + ".deepstack_mapping";
    for (const dcase & c : cases) {
        gguf_context_ptr g = get_gguf_ctx(arch, false);
        gguf_remove_key(g.get(), key.c_str());
        if (!c.mapping.empty()) {
            gguf_set_arr_data(g.get(), key.c_str(), GGUF_TYPE_INT32, c.mapping.data(), c.mapping.size());
        }
        int32_t got = -1;
        std::string err;
        try {
            auto mc = get_model_and_ctx(g.get(), nullptr, seed, stdev, {});
            got = llama_model_n_embd_inp(mc.first.get());
            const std::vector<float> lg = get_logits(mc.first.get(), mc.second.get(), tokens);
            for (const float v : lg) { if (!std::isfinite(v)) { err = "non-finite logit"; break; } }
        } catch (const std::exception & e) {
            err = e.what();
        }
        const int32_t want = (int32_t) (n_embd_base * c.mult);
        const bool ok = err.empty() && got == want;
        printf("deepstack-mapping-default %-12s %-40s n_embd_inp %6d want %6d  %s%s%s\n", llama_model_arch_name(arch), c.name,
               got, want, ok ? "[PASS]" : "[FAIL]", err.empty() ? "" : " : ", err.c_str());
        n_fail += ok ? 0 : 1;
    }
    printf("deepstack-mapping-default: %d failure(s)\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}

// SWA pattern as LOADED (row S69-LLAMACPP-PICK-SAVER-SWA-PATTERN-29042-2026-10-19-01). llama.cpp #29042 (MIT) makes the
// period-only loaders read sliding_window_pattern as an array too. This prints, per SWA arch, the per-layer is_swa flags a
// file loads to with the key as the fixture writes it (a scalar period 2 for most archs, an array for a few), as a scalar
// period 3, and absent. It asserts nothing by itself: run it on a build before and after a loader change and diff the
// output -- a published (scalar-period) file must load to the same flags on both.
// RESEARCH-DONE: ggml/include/gguf.h -- gguf_remove_key / gguf_set_val_u32 edit the in-memory context only (test input).
static int test_swa_pattern_dump(const std::string & arch_filter, const size_t seed, const float stdev) {
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    for (const llm_arch arch : llm_arch_all()) {
        // modern-bert and gemma-embedding are excluded from the logits comparison (vocab / embedding output), not from loading
        const bool load_only_ok = arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_GEMMA_EMBEDDING;
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || (!arch_supported(arch) && !load_only_ok)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4_ASSISTANT || arch == LLM_ARCH_GEMMA4_MTP || arch == LLM_ARCH_DFLASH_DRAFT ||
            arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_DFLASH2 || arch == LLM_ARCH_INKLING) {
            continue; // same skips as save_models()
        }
        const bool moe = moe_mandatory(arch);
        const std::string key = LLM_KV(arch)(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN);
        for (const char * variant : { "fixture", "period3", "absent" }) {
            gguf_context_ptr g = get_gguf_ctx(arch, moe);
            if (strcmp(variant, "fixture") != 0) {
                gguf_remove_key(g.get(), key.c_str());
            }
            if (strcmp(variant, "period3") == 0) {
                gguf_set_val_u32(g.get(), key.c_str(), 3);
            }
            std::string line;
            try {
                auto mc = get_model_and_ctx(g.get(), nullptr, seed, stdev, {});
                const llama_hparams & hp = mc.first->hparams;
                if (hp.swa_type == LLAMA_SWA_TYPE_NONE && !hp.is_swa_any()) {
                    if (strcmp(variant, "fixture") == 0) {
                        break; // not an SWA model here: nothing to compare
                    }
                    line = "swa_type=0 (no SWA)";
                } else {
                    line = "swa_type=" + std::to_string((int) hp.swa_type) + " n_layer=" + std::to_string(hp.n_layer) + " is_swa=";
                    for (uint32_t il = 0; il < hp.n_layer; ++il) {
                        line += hp.is_swa(il) ? '1' : '0';
                    }
                }
            } catch (const std::exception & e) {
                line = std::string("load-error: ") + e.what();
            }
            printf("swa-pattern %-18s %-8s %s\n", llama_model_arch_name(arch), variant, line.c_str());
        }
    }
    return 0;
}

// BitNet "norm_in" tensors (S69-ENGREV-BITNET-NORMIN-2026-10-27-01): optional per-layer RMS-norm weights applied to the INPUT of each
// projection (q/k/v, attention output, ffn gate/up/down). SOURCE-READ: BitNet 3rdparty/llama.cpp src/models/qwen3.cpp (spec only).
// The test model is a random-weight qwen3 whose norm_in weights are the constant 2.0 (fill_norm_in_two), so every norm_in node must equal
// 2 * x / sqrt(mean(x^2) + eps) of the node that feeds it, row by row.
static std::map<std::string, std::vector<float>> g_norm_in_cap;
static std::map<std::string, int64_t>            g_norm_in_rows;
static int norm_in_cb(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    static const char * const keep[] = { "attn_norm-0", "attn_q_norm_in-0", "attn_k_norm_in-0", "attn_v_norm_in-0", "attn_pre_out-0",
        "attn_out_norm_in-0", "ffn_norm-0", "ffn_gate_norm_in-0", "ffn_up_norm_in-0", "ffn_act-0", "ffn_down_norm_in-0" };
    bool want = false;
    for (const char * k : keep) {
        want = want || strcmp(t->name, k) == 0;
    }
    if (!want || t->type != GGML_TYPE_F32) {
        return 0;
    }
    if (ask) {
        return 1;
    }
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    g_norm_in_cap[t->name]  = std::move(v);
    g_norm_in_rows[t->name] = t->ne[0];
    return 1;   // 0 here would stop the graph evaluation after the first captured node
}

static int test_norm_in(const size_t seed, const float stdev, const ggml_log_level log_level) {
    llama_log_set([](ggml_log_level level, const char * text, void * ud) {
        if (level >= *(const ggml_log_level *) ud) {
            fputs(text, stderr);
        }
    }, new ggml_log_level(log_level));

    gguf_context_ptr gg = get_gguf_ctx(LLM_ARCH_QWEN3, false);
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> no_devs = { nullptr };
    mp.devices = no_devs.data();
    tensor_data_params tmp = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(gg.get(), set_tensor_data, &tmp, mp));
    if (!model) {
        printf("norm_in: model creation failed\n");
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 16; cp.n_ubatch = 16; cp.op_offload = false; cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.cb_eval = norm_in_cb;
    llama_context_ptr lctx(llama_init_from_model(model.get(), cp));
    if (!lctx) {
        printf("norm_in: context creation failed\n");
        return 1;
    }
    const std::vector<llama_token> toks = get_tokens(6, 128, seed);
    g_norm_in_cap.clear();
    g_norm_in_rows.clear();
    const std::vector<float> logits = get_logits(model.get(), lctx.get(), toks);
    int n_fail = 0;
    for (float f : logits) {
        if (!std::isfinite(f)) { printf("norm_in: non-finite logit\n"); n_fail++; break; }
    }
    // {norm_in node, the node whose rows it normalises}
    const char * const pairs[][2] = {
        { "attn_q_norm_in-0", "attn_norm-0" }, { "attn_k_norm_in-0", "attn_norm-0" }, { "attn_v_norm_in-0", "attn_norm-0" },
        { "attn_out_norm_in-0", "attn_pre_out-0" },
        { "ffn_gate_norm_in-0", "ffn_norm-0" }, { "ffn_up_norm_in-0", "ffn_norm-0" }, { "ffn_down_norm_in-0", "ffn_act-0" },
    };
    for (const auto & p : pairs) {
        auto o = g_norm_in_cap.find(p[0]);
        auto i = g_norm_in_cap.find(p[1]);
        if (o == g_norm_in_cap.end() || i == g_norm_in_cap.end()) {
            printf("norm_in: node %s (or its input %s) was never evaluated (captured %zu nodes: have-out=%d have-in=%d)\n", p[0], p[1],
                g_norm_in_cap.size(), (int) (o != g_norm_in_cap.end()), (int) (i != g_norm_in_cap.end()));
            n_fail++;
            continue;
        }
        const int64_t n = g_norm_in_rows[p[0]];
        if (o->second.size() != i->second.size() || n != g_norm_in_rows[p[1]]) {
            printf("norm_in: %s and %s differ in shape\n", p[0], p[1]);
            n_fail++;
            continue;
        }
        double worst = 0.0;
        for (size_t r = 0; r + (size_t) n <= i->second.size(); r += (size_t) n) {
            double ss = 0.0;
            for (int64_t c = 0; c < n; c++) { ss += (double) i->second[r + c] * i->second[r + c]; }
            // 1e-5 = the rms-norm eps get_gguf_ctx writes into the test model (measured: implied eps 1.000e-05 on every row)
            const double scale = 2.0 / std::sqrt(ss / (double) n + 1e-5);
            for (int64_t c = 0; c < n; c++) {
                const double want = scale * i->second[r + c];
                worst = std::max(worst, std::fabs(want - o->second[r + c]) / (std::fabs(want) + 1e-3));
            }
        }
        const bool ok = worst < 1e-2;
        printf("norm_in: %-20s = rms_norm(%-15s) * 2.0  worst rel err %.3e  %s\n", p[0], p[1], worst, ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
    printf("test-norm-in: %s\n", n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting

    // Compare the CUDA model with the CPU reference in true F32. The CUDA backend creates its cuBLAS handles in TF32 tensor-op mode (ggml-cuda/common.cuh,
    // CUBLAS_TF32_TENSOR_OP_MATH), so every large-batch F32 GEMM carries ~1e-3 relative rounding; on a model whose attention scores or router logits are
    // large that noise is amplified into a failing NMSE with no defect behind it. MEASURED 2026-10-04, diffusion-gemma: NMSE 8.2e-3 with TF32 on, 1.75e-10 with
    // NVIDIA_TF32_OVERRIDE=0 set in the shell. That variable is read when the CUDA libraries load, so setting it here is too late (measured: 3.3e-2 still);
    // GGML_CUDA_NO_TF32 (ggml-cuda/common.cuh) is read when a cuBLAS handle is created, which happens after main. A value the caller already set is left alone.
    // DISABLE-REVERSIBLE: GGML_CUDA_NO_TF32 -- test process only, off behind the caller's own setting; the served engine keeps TF32 (this changes no product path).
    if (getenv("GGML_CUDA_NO_TF32") == nullptr) {
#ifdef _WIN32
        _putenv("GGML_CUDA_NO_TF32=1");
#else
        setenv("GGML_CUDA_NO_TF32", "1", 0);
#endif
    }
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    bool stdev_set = false; // -d given: the fork modes use it too, else the N(0, 0.01) they were measured at
    bool parity = false;
    bool deepstack_layout = false;
    bool deepstack_mapping_default = false;
    bool swa_pattern_dump = false;
    bool tree_batch = false;
    bool norm_in = false;

    const char * target_backend = nullptr;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--parity") == 0) {
            parity = true;
        } else if (strcmp(argv[i], "--deepstack-key-layout") == 0) {
            deepstack_layout = true;
        } else if (strcmp(argv[i], "--deepstack-mapping-default") == 0) {
            deepstack_mapping_default = true;
        } else if (strcmp(argv[i], "--tree-batch") == 0) {
            tree_batch = true;
        } else if (strcmp(argv[i], "--norm-in") == 0) {
            norm_in = true;
        } else if (strcmp(argv[i], "--swa-pattern-dump") == 0) {
            swa_pattern_dump = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbosity = LOG_LEVEL_INFO;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llm_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
                stdev_set = true;
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            // upstream: -v <N>; the bare -v this fork had (= --verbose) is still accepted when no number follows
            if (i + 1 < argc && isdigit((unsigned char) argv[i + 1][0])) {
                verbosity = std::stoi(argv[++i]);
            } else {
                verbosity = LOG_LEVEL_INFO;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    const float stdev_fork = stdev_set ? stdev : 1.0e-2f;
    const ggml_log_level log_level = verbosity >= LOG_LEVEL_INFO ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_ERROR;

    try {
        if (deepstack_layout) {
            return test_deepstack_key_layout(seed, stdev_fork);
        }
        if (deepstack_mapping_default) {
            return test_deepstack_mapping_default(seed, stdev_fork);
        }
        if (tree_batch) {
            return test_tree_batch(seed, stdev_fork, log_level);
        }
        if (norm_in) {
            return test_norm_in(seed, stdev_fork, log_level);
        }
        if (swa_pattern_dump) {
            return test_swa_pattern_dump(arch_filter, seed, stdev_fork);
        }
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        if (parity) {
            return test_parity(arch_filter, seed, stdev_fork, log_level);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
