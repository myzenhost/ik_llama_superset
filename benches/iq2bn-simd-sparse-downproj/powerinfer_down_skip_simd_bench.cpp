// powerinfer_down_skip_simd_bench: SIMD sparse iq2_bn down-proj micro-bench for
// S69-IQ2BN-SIMD-SPARSE-DOWNPROJ-KERNEL-2026-09-27-01.
// Builds on the design/decode SOURCE-READ already recorded in
// tmp/_agent_ledger_s69-engine-review-powerinfer-sparse-ffn-2026-09-27-01.md and
// tmp/ffn_sparsity_probe/powerinfer_down_skip_bench.cpp (that file's scalar AXPY
// candidate measured 0.327x of dense -- scalar, non-SIMD). This file adds a SIMD-vectorized sparse
// kernel over the SAME lossless transposed-code design (codeT[i][j], per-row scale d[j] factored out,
// re-verified in the prior ledger from ggml/src/iqk/iqk_quantize.cpp:216-241) and measures it against
// the real production dense kernel (mul_mat_iq2_bn_r4_q8_k16, AVX2, via the public ggml graph API --
// same dense-timing code path as the prior bench, unchanged in spirit) on real BitNet-2B4T decode
// activations.
//
// BREAK-EVEN (from the task row): at measured skip s, sparse wins only if its cost per LIVE row is
// <= 1/(1-s) x dense's cost per (all) row. s is measured fresh below, not assumed.
//
// TWO SIMD VARIANTS (anti-bias mandate: >=2 layouts tried, enumerated):
//   VARIANT A "widen-per-step": for each live i, sign_epi8(broadcast(q8[i]), codeT[i][j:j+32]) then
//     immediately widen 8-lane groups to int32 and accumulate. Simple, more instructions/lane per i.
//   VARIANT B "block-accumulate/int16-flush": accumulate signed int16 partial sums over a whole
//     QK=64-wide quant block's worth of live i (int16 accumulator, headroom |125*64|=8000 << 32767),
//     flush (widen+scale-multiply) into a float accumulator once per block. Fewer flushes, cheaper
//     per-i inner step (sign_epi8 + int16 add only).
//   EXCLUDED, with the arithmetic bound that rules each out (not a research pivot, plain range check):
//     (c) int8-only running accumulator, no widen: signed int8 range is [-128,127] (ISO C++
//     <cstdint> INT8_MIN/INT8_MAX); two per-lane terms of magnitude up to 127 already sum to 254,
//     outside that range, so correctness would require a widen after every 1-2 terms -- strictly more
//     widen ops per live-i than variant B's per-64-terms widen, for no accuracy or simplicity gain
//     over A. Excluded on this arithmetic bound alone, not benchmarked.
//     (d) AVX-512 VPCOMPRESSB-based live-list compress: excluded because this host has no AVX-512
//     (memory note project-host-rebar-off-bar1-256mib.md / Zen 2 CPU family notes) -- VPCOMPRESSB is
//     an AVX-512BW instruction, unavailable here; the movemask+tzcnt idiom below is the AVX2-only
//     substitute actually used for both variants' live-list build.
//
// ACTIVATION QUANTIZATION: per-QK(=64)-block int8, scale = max(|cur|) in that block / 127 (mirrors the
// production Q8_K64 activation quantizer's block granularity, QK_IQ1BN=64, so precision is comparable
// to what the dense kernel itself already accepts -- this is not a harsher approximation than
// production takes on the SAME activation vector).
//
// CORRECTNESS, two-tier (float-vs-int8-quant error is an inherent, expected quantization effect, not a
// SIMD bug; isolating it from actual SIMD-arithmetic bugs needs two separate comparisons):
//   TIER 1 (SIMD-arithmetic correctness): quantized-SCALAR reference (plain double-accumulate loop
//     over the same q8 activations, same per-block scale, same codeT) vs each SIMD variant's output.
//     Must be exact (both operate on the same already-quantized inputs; int32/int64/double
//     accumulation of the same integer terms is exact regardless of summation order).
//   TIER 2 (transpose+quantization vs float dense_ref, informational): the ORIGINAL float dense_ref
//     from the earlier bench (kept, per the task's "keep that test") vs the int8-quantized SIMD output
//     -- expected to show the quantization step's inherent error, reported but not gating on the
//     original bench's 1e-4 float-vs-float tolerance (comparing float math to int8-quantized math at
//     1e-4 is comparing two different numerical methods, not validating the SIMD code).
//
// exit: 0 measured, 2 setup/type error, 3 no ffn_relu/ffn_up tensor for the layer.

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct CaptureState {
    int target_il = 0;
    bool armed = false;
    bool got_up = false, got_relu = false;
    std::vector<float> up, relu;
};

static int cb_eval(struct ggml_tensor * t, bool ask, void * user) {
    auto * st = (CaptureState *) user;
    if (!st->armed) return ask ? false : true;
    const char * nm = ggml_get_name(t);
    char want_up[64], want_relu[64];
    std::snprintf(want_up, sizeof(want_up), "ffn_up-%d", st->target_il);
    std::snprintf(want_relu, sizeof(want_relu), "ffn_relu-%d", st->target_il);
    const bool is_up = std::strcmp(nm, want_up) == 0 && t->type == GGML_TYPE_F32;
    const bool is_relu = std::strcmp(nm, want_relu) == 0 && t->type == GGML_TYPE_F32;
    if (ask) return is_up || is_relu;
    if (!is_up && !is_relu) return true;
    if (!ggml_is_contiguous(t)) return true;
    const int64_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    if (is_up)   { st->up = std::move(buf);   st->got_up = true; }
    if (is_relu) { st->relu = std::move(buf); st->got_relu = true; }
    return true;
}

static inline int8_t iq2bn_code_to_val(int code) { return (int8_t)(code - 1); }

static float iq2bn_decode_row(const uint8_t * row_bytes, int n_ff, int8_t * out_codes) {
    float d;
    std::memcpy(&d, row_bytes, sizeof(float));
    const uint8_t * qs = row_bytes + sizeof(float);
    const int nblock = n_ff / 64;
    for (int blk = 0; blk < nblock; ++blk) {
        const uint8_t * bqs = qs + blk * 16;
        int base = blk * 64;
        for (int b = 0; b < 16; ++b) {
            uint8_t byte = bqs[b];
            out_codes[base + b]      = iq2bn_code_to_val((byte >> 0) & 3);
            out_codes[base + b + 16] = iq2bn_code_to_val((byte >> 2) & 3);
            out_codes[base + b + 32] = iq2bn_code_to_val((byte >> 4) & 3);
            out_codes[base + b + 48] = iq2bn_code_to_val((byte >> 6) & 3);
        }
    }
    return d;
}

struct Stat { double mean, stdev; };
static Stat stat_of(const std::vector<double> & v) {
    double m = 0; for (double x : v) m += x; m /= v.size();
    double s = 0; for (double x : v) s += (x - m) * (x - m); s = v.size() > 1 ? std::sqrt(s / (v.size() - 1)) : 0.0;
    return {m, s};
}

// ---- activation quantization: per-QK(=64)-block int8 ----
static constexpr int QK = 64;
static void quantize_q8_blocks(const std::vector<float> & cur, std::vector<int8_t> & q, std::vector<float> & blk_scale) {
    const int64_t n = (int64_t) cur.size();
    const int nb = (int) (n / QK);
    q.resize(n);
    blk_scale.resize(nb);
    for (int b = 0; b < nb; ++b) {
        float amax = 0.f;
        for (int k = 0; k < QK; ++k) amax = std::max(amax, std::fabs(cur[b*QK+k]));
        float scale = amax > 0.f ? amax / 127.f : 1.f;
        blk_scale[b] = scale;
        float inv = amax > 0.f ? 127.f / amax : 0.f;
        for (int k = 0; k < QK; ++k) {
            int v = (int) std::lround(cur[b*QK+k] * inv);
            v = std::max(-127, std::min(127, v));
            q[b*QK+k] = (int8_t) v;
        }
    }
}

// ---- live-index list via vectorized compare + movemask + tzcnt extraction (AVX2, no AVX-512) ----
static void build_live_list(const std::vector<int8_t> & q, std::vector<int32_t> & live_idx) {
    const int64_t n = (int64_t) q.size();
    live_idx.clear();
    live_idx.reserve(n);
    __m256i zero = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(q.data() + i));
        __m256i eqz = _mm256_cmpeq_epi8(v, zero);
        uint32_t mask = (uint32_t) ~(uint32_t) _mm256_movemask_epi8(eqz); // 1 bit per NONZERO lane
        while (mask) {
            uint32_t bit = (uint32_t) __builtin_ctz(mask); // GCC builtin, portable vs _mm_tzcnt_u32 (BMI-only name mismatch under MinGW g++)
            live_idx.push_back((int32_t)(i + bit));
            mask &= mask - 1;
        }
    }
    for (; i < n; ++i) if (q[i] != 0) live_idx.push_back((int32_t) i);
}

// ---- VARIANT A: widen-per-step, int32 accumulate, 32 lanes/insn group ----
static void sparse_simd_variant_a(const std::vector<int32_t> & live_idx, const std::vector<int8_t> & q8act,
                                   const std::vector<int8_t> & codeT, int64_t n_embd, std::vector<int64_t> & acc) {
    std::fill(acc.begin(), acc.end(), 0);
    const int64_t j32 = (n_embd / 32) * 32;
    for (int32_t i : live_idx) {
        __m256i ci = _mm256_set1_epi8(q8act[i]);
        const int8_t * row = &codeT[(size_t) i * n_embd];
        int64_t j = 0;
        for (; j < j32; j += 32) {
            __m256i code = _mm256_loadu_si256((const __m256i *)(row + j));
            __m256i signed_val = _mm256_sign_epi8(ci, code); // per-lane in {-q8[i],0,+q8[i]}
            __m128i lo = _mm256_castsi256_si128(signed_val);
            __m128i hi = _mm256_extracti128_si256(signed_val, 1);
            for (int half = 0; half < 2; ++half) {
                __m128i h = half == 0 ? lo : hi;
                for (int quarter = 0; quarter < 2; ++quarter) {
                    __m128i q16 = quarter == 0 ? h : _mm_srli_si128(h, 8);
                    __m256i widened = _mm256_cvtepi8_epi32(q16); // 8 int32 lanes
                    int64_t base = j + half*16 + quarter*8;
                    alignas(32) int32_t lanes[8];
                    _mm256_storeu_si256((__m256i *)lanes, widened); // _mm256_extract_epi32 needs a
                    // compile-time immediate index, not a runtime loop var (MinGW g++ 14.2 error:
                    // "the last argument must be a 1-bit immediate" / "selector must be an integer
                    // constant") -- store-then-loop is the standard portable fix, same total lane
                    // extraction, one store instead of 8 immediate-indexed extracts.
                    for (int lane = 0; lane < 8; ++lane) acc[base + lane] += lanes[lane];
                }
            }
        }
        for (; j < n_embd; ++j) acc[j] += (int64_t) q8act[i] * (int64_t) row[j];
    }
}

// ---- VARIANT B: block-accumulate int16, flush to double accumulator once per QK=64 block ----
static void sparse_simd_variant_b(const std::vector<int32_t> & live_idx, const std::vector<int8_t> & q8act,
                                   const std::vector<float> & blk_scale, const std::vector<int8_t> & codeT,
                                   int64_t n_embd, std::vector<double> & acc_out) {
    acc_out.assign(n_embd, 0.0);
    if (live_idx.empty()) return;
    std::vector<int16_t> acc16(n_embd, 0);
    const int64_t j16 = (n_embd / 16) * 16;
    int cur_block = live_idx[0] / QK;
    size_t k = 0;
    auto flush_block = [&](int blk) {
        float sc = blk_scale[blk];
        for (int64_t j = 0; j < n_embd; ++j) { acc_out[j] += (double) sc * (double) acc16[j]; acc16[j] = 0; }
    };
    for (; k < live_idx.size(); ++k) {
        int32_t i = live_idx[k];
        int blk = i / QK;
        if (blk != cur_block) { flush_block(cur_block); cur_block = blk; }
        __m128i ci16 = _mm_set1_epi8(q8act[i]);
        const int8_t * row = &codeT[(size_t) i * n_embd];
        int64_t j = 0;
        for (; j < j16; j += 16) {
            __m128i code16 = _mm_loadu_si128((const __m128i *)(row + j));
            __m128i signed_val = _mm_sign_epi8(ci16, code16); // 16 lanes, {-q8[i],0,+q8[i]}
            __m256i widened = _mm256_cvtepi8_epi16(signed_val); // 16 int16 lanes
            __m256i acc_cur = _mm256_loadu_si256((const __m256i *)(&acc16[j]));
            __m256i sum = _mm256_add_epi16(acc_cur, widened);
            _mm256_storeu_si256((__m256i *)(&acc16[j]), sum);
        }
        for (; j < n_embd; ++j) acc16[j] = (int16_t)(acc16[j] + row[j]*q8act[i]);
    }
    flush_block(cur_block);
}

int main(int argc, char ** argv) {
    std::string model_path, text_path;
    int n_prompt = 256, n_decode = 64, n_threads = 8, target_il = 0, reps = 3, iters = 400;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m") model_path = argv[++i];
        else if (a == "-f") text_path = argv[++i];
        else if (a == "-p") n_prompt = std::atoi(argv[++i]);
        else if (a == "-n") n_decode = std::atoi(argv[++i]);
        else if (a == "-t") n_threads = std::atoi(argv[++i]);
        else if (a == "-il") target_il = std::atoi(argv[++i]);
        else if (a == "-reps") reps = std::atoi(argv[++i]);
        else if (a == "-iters") iters = std::atoi(argv[++i]);
    }
    if (model_path.empty() || text_path.empty()) {
        std::fprintf(stderr, "usage: %s -m model.gguf -f text.txt [-p 256] [-n 64] [-t 8] [-il 0] [-reps 3] [-iters 400]\n", argv[0]);
        return 2;
    }
    std::ifstream in(text_path, std::ios::binary);
    std::stringstream ss; ss << in.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 2; }

    CaptureState st; st.target_il = target_il;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_prompt + n_decode + 16;
    cp.n_batch = n_prompt;
    cp.n_ubatch = n_prompt;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 2; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < n_prompt) { std::fprintf(stderr, "text too short: %d tokens\n", nt); return 2; }
    toks.resize(n_prompt);

    llama_batch b = llama_batch_get_one(toks.data(), n_prompt);
    if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "prefill decode failed\n"); return 2; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_token tok = 0;
    for (int i = 0; i < n_decode; ++i) {
        st.armed = (i == n_decode - 1);
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        tok = best;
        llama_batch d = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, d) != 0) { std::fprintf(stderr, "decode step %d failed\n", i); return 2; }
    }
    if (!st.got_up || !st.got_relu) {
        std::printf("RESULT: ffn_up-%d / ffn_relu-%d not both captured -- no measurement.\n", target_il, target_il);
        return 3;
    }

    const int64_t n_ff = (int64_t) st.relu.size();
    std::vector<float> cur(n_ff);
    int64_t zeros = 0;
    for (int64_t i = 0; i < n_ff; ++i) {
        float r = st.relu[i];
        cur[i] = st.up[i] * r * r;
        if (cur[i] == 0.0f) zeros++;
    }
    const double zero_frac = (double) zeros / (double) n_ff;
    std::printf("captured cur[n_ff=%lld] real decode-step zero_frac=%.4f\n", (long long) n_ff, zero_frac);

    char down_name[64];
    std::snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down.weight", target_il);
    struct ggml_tensor * down = llama_get_model_tensor(model, down_name);
    if (!down) { std::fprintf(stderr, "tensor %s not found\n", down_name); return 2; }
    if (down->type != GGML_TYPE_IQ2_BN) {
        std::fprintf(stderr, "RESULT: %s is type %d, not GGML_TYPE_IQ2_BN\n", down_name, (int) down->type);
        return 2;
    }
    if (down->ne[0] != n_ff) { std::fprintf(stderr, "down ne[0]=%lld != captured n_ff=%lld\n", (long long) down->ne[0], (long long) n_ff); return 2; }
    const int64_t n_embd = down->ne[1];
    const size_t row_size = sizeof(float) + (n_ff / 64) * sizeof(uint8_t) * 16;
    const size_t nbytes = (size_t) n_embd * row_size;
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(down, raw.data(), 0, nbytes);

    // float dense_ref (TIER 2 comparator, kept from the prior bench, unmodified math)
    std::vector<float> dense_ref(n_embd), d_scale(n_embd);
    std::vector<int8_t> row_codes(n_ff);
    for (int64_t j = 0; j < n_embd; ++j) {
        float d = iq2bn_decode_row(raw.data() + j * row_size, (int) n_ff, row_codes.data());
        d_scale[j] = d;
        double sum = 0;
        for (int64_t i = 0; i < n_ff; ++i) sum += (double) row_codes[i] * (double) cur[i];
        dense_ref[j] = (float) (d * sum);
    }

    // transposed code table (load-time cost, NOT in the timed loop)
    std::vector<int8_t> codeT((size_t) n_ff * n_embd);
    for (int64_t j = 0; j < n_embd; ++j) {
        iq2bn_decode_row(raw.data() + j * row_size, (int) n_ff, row_codes.data());
        for (int64_t i = 0; i < n_ff; ++i) codeT[(size_t) i * n_embd + j] = row_codes[i];
    }

    // activation quantization (per-QK=64 block int8) + live-list build -- COUNTED in the timed cost below
    std::vector<int8_t> q8act; std::vector<float> blk_scale;
    quantize_q8_blocks(cur, q8act, blk_scale);
    std::vector<int32_t> live_idx;
    build_live_list(q8act, live_idx);
    const double live_frac = (double) live_idx.size() / (double) n_ff;
    std::printf("quantized live count=%zu / n_ff=%lld (live_frac=%.4f, skip=%.4f)\n",
                live_idx.size(), (long long) n_ff, live_frac, 1.0 - live_frac);

    // TIER 1: quantized-scalar reference (double accumulate, same q8act/codeT/blk_scale)
    std::vector<double> scalar_q_out(n_embd, 0.0);
    {
        std::vector<double> acc(n_embd, 0.0);
        for (int32_t i : live_idx) {
            int blk = i / QK;
            float sc = blk_scale[blk];
            const int8_t * row = &codeT[(size_t) i * n_embd];
            double v = (double) sc * (double) q8act[i];
            for (int64_t j = 0; j < n_embd; ++j) acc[j] += v * (double) row[j];
        }
        for (int64_t j = 0; j < n_embd; ++j) scalar_q_out[j] = d_scale[j] * acc[j];
    }

    // VARIANT A output: run per-block (raw q8*code sum, no scale applied inline), scale+accumulate after
    std::vector<double> a_out(n_embd, 0.0);
    {
        size_t k = 0;
        std::vector<int64_t> blk_acc(n_embd);
        while (k < live_idx.size()) {
            int blk = live_idx[k] / QK;
            size_t k0 = k;
            while (k < live_idx.size() && live_idx[k] / QK == blk) ++k;
            std::vector<int32_t> blk_live(live_idx.begin() + k0, live_idx.begin() + k);
            sparse_simd_variant_a(blk_live, q8act, codeT, n_embd, blk_acc);
            float sc = blk_scale[blk];
            for (int64_t j = 0; j < n_embd; ++j) a_out[j] += (double) sc * (double) blk_acc[j];
        }
        for (int64_t j = 0; j < n_embd; ++j) a_out[j] = d_scale[j] * a_out[j];
    }

    // VARIANT B output (already block-aware/scaled internally, apply final per-row d)
    std::vector<double> b_raw;
    sparse_simd_variant_b(live_idx, q8act, blk_scale, codeT, n_embd, b_raw);
    std::vector<double> b_out(n_embd);
    for (int64_t j = 0; j < n_embd; ++j) b_out[j] = d_scale[j] * b_raw[j];

    auto check = [&](const char * name, const std::vector<double> & ref, const std::vector<double> & got, double tol) {
        double max_abs = 0, max_rel = 0; int64_t n_bad = 0;
        for (int64_t j = 0; j < n_embd; ++j) {
            double diff = std::fabs(ref[j] - got[j]);
            double rel = diff / std::max(1e-6, std::fabs(ref[j]));
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, rel);
            if (rel > tol && diff > tol) n_bad++;
        }
        std::printf("RESULT correctness[%s]: max_abs_diff=%.3e max_rel_diff=%.3e n_over_tol(%.0e)=%lld  %s\n",
                    name, max_abs, max_rel, tol, (long long) n_bad, n_bad == 0 ? "PASS" : "FAIL");
        return n_bad == 0;
    };
    bool ok_a_vs_scalar = check("A_vs_scalarQ_TIER1_SIMD_arith", scalar_q_out, a_out, 1e-6);
    bool ok_b_vs_scalar = check("B_vs_scalarQ_TIER1_SIMD_arith", scalar_q_out, b_out, 1e-6);
    std::vector<double> dense_ref_d(n_embd);
    for (int64_t j = 0; j < n_embd; ++j) dense_ref_d[j] = (double) dense_ref[j];
    check("scalarQ_vs_floatDense_TIER2_quant_error_informational", dense_ref_d, scalar_q_out, 5e-2);

    if (!ok_a_vs_scalar || !ok_b_vs_scalar) {
        std::printf("RESULT: a SIMD variant FAILED tier-1 (SIMD-vs-scalar-same-inputs) -- refusing to time or recommend landing.\n");
        return 2;
    }

    // ---- SPEED: dense = real production kernel via ggml graph (unchanged from prior bench) ----
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    struct ggml_init_params gip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx0 = ggml_init(gip);
    struct ggml_tensor * act = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_ff);
    struct ggml_tensor * res = ggml_mul_mat(ctx0, down, act);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, res);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend);
    if (!buf) { std::fprintf(stderr, "alloc_ctx_tensors failed\n"); return 2; }
    ggml_backend_tensor_set(act, cur.data(), 0, n_ff * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<double> dense_us;
    for (int r = 0; r < reps; ++r) {
        int64_t t0 = ggml_time_us();
        for (int k = 0; k < iters; ++k) ggml_backend_graph_compute(backend, gf);
        int64_t t1 = ggml_time_us();
        dense_us.push_back((double) (t1 - t0) / iters);
    }

    // SPEED single-thread SIMD sparse (both variants), INCLUDING quantize+live-list build cost per call
    auto time_variant = [&](const char * name, auto fn) {
        std::vector<double> us;
        for (int r = 0; r < reps; ++r) {
            int64_t t0 = ggml_time_us();
            for (int k = 0; k < iters; ++k) {
                std::vector<int8_t> q; std::vector<float> sc;
                quantize_q8_blocks(cur, q, sc);
                std::vector<int32_t> lv;
                build_live_list(q, lv);
                fn(lv, q, sc);
            }
            int64_t t1 = ggml_time_us();
            us.push_back((double) (t1 - t0) / iters);
        }
        Stat s = stat_of(us);
        std::printf("RESULT speed[%s] (us/call, single-thread, n=%d iters x %d reps): mean=%.2f stdev=%.2f\n",
                    name, iters, reps, s.mean, s.stdev);
        return s;
    };

    std::vector<int64_t> tmp_a(n_embd);
    Stat sa = time_variant("variantA_widen_per_step", [&](std::vector<int32_t> & lv, std::vector<int8_t> & q, std::vector<float> &) {
        sparse_simd_variant_a(lv, q, codeT, n_embd, tmp_a);
    });
    std::vector<double> tmp_b(n_embd);
    Stat sb = time_variant("variantB_block_accum_int16", [&](std::vector<int32_t> & lv, std::vector<int8_t> & q, std::vector<float> & sc) {
        sparse_simd_variant_b(lv, q, sc, codeT, n_embd, tmp_b);
    });

    Stat ds = stat_of(dense_us);
    const double n_ff_d = (double) n_ff;
    const double live_d = std::max(1.0, (double) live_idx.size());
    double dense_per_row = ds.mean / n_ff_d;
    double a_per_row = sa.mean / live_d;
    double b_per_row = sb.mean / live_d;
    double breakeven = 1.0 / (1.0 - (1.0 - live_frac));
    std::printf("RESULT SUMMARY: n_ff=%lld n_embd=%lld live=%zu skip=%.4f breakeven_ratio=%.3fx (=1/(1-skip))\n",
                (long long) n_ff, (long long) n_embd, live_idx.size(), 1.0 - live_frac, breakeven);
    std::printf("RESULT SUMMARY: dense_us=%.2f dense_per_row_us=%.5f\n", ds.mean, dense_per_row);
    std::printf("RESULT SUMMARY: variantA_us=%.2f a_per_live_row_us=%.5f a_ratio_to_dense_per_row=%.3fx %s\n",
                sa.mean, a_per_row, a_per_row / dense_per_row, (a_per_row / dense_per_row) <= breakeven ? "UNDER_BREAKEVEN" : "OVER_BREAKEVEN");
    std::printf("RESULT SUMMARY: variantB_us=%.2f b_per_live_row_us=%.5f b_ratio_to_dense_per_row=%.3fx %s\n",
                sb.mean, b_per_row, b_per_row / dense_per_row, (b_per_row / dense_per_row) <= breakeven ? "UNDER_BREAKEVEN" : "OVER_BREAKEVEN");
    std::printf("RESULT SUMMARY: variantA_total_vs_dense_total speedup=%.3fx  variantB_total_vs_dense_total speedup=%.3fx\n",
                ds.mean / sa.mean, ds.mean / sb.mean);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
