// Real-KV test for the #220 offload-attention bridge.
//
// Everything before this exercised the math on a synthetic store and the symbols on an object
// file. Neither touches a real llama_context, so "it will work on real KV" was an assumption. This
// loads an actual GGUF on CPU, decodes a prompt so the KV cache is genuinely populated, and then
// drives the bridge against that live cache.
//
// Why this is runnable at all, having briefly treated it as blocked: the bridge needs host-
// resident F32 KV, and the fork already exposes both halves -- cache_type_k/v force the type, and
// CPU-only inference (n_gpu_layers = 0) leaves the tensors in host RAM with a non-null data
// pointer. No GPU work, one small model, one short decode.
//
// WHAT IS ASSERTED
//   1. probe() returns OK on a populated F32 host cache -- and returns a SPECIFIC refusal code for
//      an out-of-range layer, so a pass is distinguishable from a blanket zero.
//   2. offload_attn_step_from_ctx() succeeds and writes a finite output vector.
//   3. The output MATCHES dense attention computed independently, in this file, over the same live
//      KV rows -- at a full page budget the two must agree to fp tolerance. That is the check that
//      can actually fail: it compares against a second implementation, not against itself.
//   4. At a small budget the result must DIVERGE from dense. If sparse and dense agree at 2% on
//      real KV, the selector is not selecting and the test would be passing for the wrong reason.
//
// Usage: offload-attn-test <model.gguf> [n_predict]

#include "llama.h"
#include "ggml-backend.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
uint64_t offload_attn_op_calls(void);
uint64_t offload_attn_op_masked(void);
uint64_t offload_attn_op_passthrough(void);
uint64_t offload_attn_op_last_keep(void);
uint64_t offload_attn_op_last_npages(void);
uint64_t offload_attn_op_last_budget_milli(void);
uint64_t offload_attn_maskop_calls(void);
uint64_t offload_attn_maskop_masked(void);
uint64_t offload_attn_maskop_passthrough(void);
uint64_t offload_attn_maskop_last_keep(void);
uint64_t offload_attn_maskop_last_npages(void);
int offload_attn_probe(void * ctx, int layer_idx);
int offload_attn_step_from_ctx(void * ctx, int layer_idx, const float * q, float * out,
                               int page_size, float page_budget, int selector);
}

// Reach into the cache the same way the bridge does, so the reference is computed over exactly the
// rows the bridge attends.
#include "../../src/llama-context.h"

static bool finite_vec(const std::vector<float> & v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

static double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double n = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na <= 0 || nb <= 0) return 0.0;
    return n / (std::sqrt(na) * std::sqrt(nb));
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [n_predict]\n", argv[0]); return 2; }
    const std::string model_path = argv[1];
    const int n_pred = argc > 2 ? std::atoi(argv[2]) : 0;   // n_gpu_layers

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    // n_gpu_layers comes from argv so ONE binary measures both arms. With 0 the KV stays
    // in host RAM; with >0 it moves to the device, which is the case the bridge's
    // NOT_HOST refusal exists for -- and that refusal has to be measured, not assumed.
    mp.n_gpu_layers = n_pred;   // argv[2], reused as the layer count
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "FAIL: model did not load\n"); return 3; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;   // room for the multi-page prompt below
    cp.n_batch    = 1024;
    // Use the fork DEFAULTS (F16). MEASURED 2026-08-19: forcing F32 is not a reachable
    // configuration -- without flash_attn the fork refuses ("V cache quantization requires
    // flash_attn"), and with it the FA kernel aborts at iqk_flash_attn.cpp:184 because F32
    // is not in supported_kv_types(). Testing against F32 would have tested a config no
    // served model can use.
    cp.flash_attn = true;   // exercise the FA-path op
    // Drive the SPLICED op, not just the standalone bridge. The counters after the
    // decode are the only way to tell 'the mask ran' from 'the graph changed shape'.
    cp.offload_attn          = true;
    cp.offload_attn_page     = 16;
    cp.offload_attn_budget   = 0.10f;
    cp.offload_attn_selector = 0;
    // SET EXPLICITLY, both of them. llama_context_default_params() does not initialise the
    // offload_attn_* fields at all -- they fall into the trailing block of llama_context_params
    // that the aggregate initialiser leaves to zero-init -- so leaving them unset means the test
    // silently exercises whatever zero happens to mean, which is not the same as testing a
    // chosen configuration. A plumbing-completeness check flagged both as assigned everywhere
    // in the chain EXCEPT here.
    cp.offload_attn_sink     = 0;
    // The token floor is deliberately DISABLED for this test. It exists so that short prompts
    // keep every page and the op short-circuits; that is correct for serving and fatal here,
    // because this test's whole assertion is that masking CHANGES the output. With the shipped
    // 256-token floor the op would decline to splice and the divergence check could not fire --
    // the same shape as the earlier defect where a 13-token prompt made a 2% budget keep
    // everything. Testing the safe default here would test that the feature does nothing.
    cp.offload_attn_min_tokens = 0;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "FAIL: context did not create\n"); llama_free_model(model); return 4; }

    // Populate the KV with a real decode.
    // Long enough to produce MANY pages. The first run used a 13-token prompt, which at
    // page_size 16 is a single page -- so a 2%% budget still kept everything and the
    // "must diverge" assertion could not possibly hold. That was a defect in the test
    // setup, not in the selector: with one page there is nothing to drop.
    std::string prompt;
    for (int i = 0; i < 12; ++i) {
        prompt += "The regional transit agency publishes a schedule that tells riders to"
                  "arrive at the platform five minutes before departure time.";
    }
    std::vector<llama_token> toks(prompt.size() + 8);
    int n_tok = llama_tokenize(model, prompt.c_str(), (int) prompt.size(),
                               toks.data(), (int) toks.size(), true, false);
    if (n_tok <= 0) { std::fprintf(stderr, "FAIL: tokenize returned %d\n", n_tok); return 5; }
    // Clamp to the context. An over-long prompt fast-failed the process (0xC0000409)
    // rather than erroring, so this is a guard, not a nicety.
    if (n_tok > (int) cp.n_ctx - 8) n_tok = (int) cp.n_ctx - 8;
    toks.resize(n_tok);

    // upstream batch ABI (0297995eb): sequence 0, positions from the cache (empty here, so 0..n_tok-1) -- same as the old (0, 0)
    llama_batch batch = llama_batch_get_one(toks.data(), n_tok);
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "FAIL: decode failed\n"); return 6; }
    std::printf("decoded %d prompt tokens (n_predict arg=%d, unused for KV population)\n",
                n_tok, n_pred);

    auto * lctx = static_cast<llama_context *>(ctx);
    llama_kv_cache_fork * kv = &lctx->kv_self;   // the fork's live KV struct (src/llama-context.h:136)
    const int used = (int) kv->get_used_n_kv();
    std::printf("KV used cells: %d\n", used);
    // The counters are the only way to separate "the mask ran" from "the graph changed shape".
    // Output text alone cannot: adding a node can perturb numerics even when the op passes
    // through, which is exactly how a dormant feature reads as a working one.
    std::printf("OP COUNTERS after decode: calls=%llu masked=%llu passthrough=%llu\n",
                (unsigned long long) offload_attn_op_calls(),
                (unsigned long long) offload_attn_op_masked(),
                (unsigned long long) offload_attn_op_passthrough());
    std::printf("OP LAST DECISION: nP=%llu keep=%llu budget=%.3f\n",
                (unsigned long long) offload_attn_op_last_npages(),
                (unsigned long long) offload_attn_op_last_keep(),
                offload_attn_op_last_budget_milli() / 1000.0);
    std::printf("FA MASK OP: calls=%llu masked=%llu passthrough=%llu nP=%llu keep=%llu\n",
                (unsigned long long) offload_attn_maskop_calls(),
                (unsigned long long) offload_attn_maskop_masked(),
                (unsigned long long) offload_attn_maskop_passthrough(),
                (unsigned long long) offload_attn_maskop_last_npages(),
                (unsigned long long) offload_attn_maskop_last_keep());

    // ---- 1. probe, both directions -------------------------------------------------------
    const int p_ok  = offload_attn_probe(ctx, 0);
    const int p_bad = offload_attn_probe(ctx, 100000);   // out-of-range layer
    std::printf("probe(layer 0)      = %d   %s\n", p_ok, p_ok == 0 ? "OK" : "refused");
    std::printf("probe(layer 100000) = %d   %s\n", p_bad,
                p_bad != 0 ? "refused (correct)" : "accepted (control did not fail)");

    struct ggml_tensor * k = kv->get_k_raw(0);
    if (k) {
        const bool host = k->buffer && ggml_backend_buffer_is_host(k->buffer);
        std::printf("K residency: buffer=%s data=%p -> %s\n",
                    k->buffer ? "yes" : "none", k->data, host ? "HOST" : "DEVICE");
        // A device tensor has a NON-NULL data pointer. Reading it from host code is the
        // access violation this arm exists to demonstrate, so the reference below is
        // gated on the same question the bridge now asks.
        if (!host) {
            std::printf("probe on device KV = %d  %s\n", p_ok,
                        p_ok == -6 ? "declined with E_NOT_HOST (correct)"
                                   : "did NOT decline as NOT_HOST");
            const bool arm_ok = (p_ok == -6) && (p_bad != 0);
            std::printf("\nOFFLOAD-ATTN DEVICE-KV ARM: %s\n", arm_ok ? "PASS" : "FAIL");
            llama_free(ctx); llama_free_model(model); llama_backend_free();
            return arm_ok ? 0 : 1;
        }
    }
    if (k) {
        std::printf("K tensor: type=%d ne0=%lld data=%s\n", (int) k->type,
                    (long long) k->ne[0], k->data ? "host" : "NULL(device)");
    }

    bool ok = true;
    if (p_bad == 0) { std::printf("FAIL: out-of-range layer was accepted\n"); ok = false; }

    if (p_ok != 0) {
        std::printf("\nprobe refused layer 0 with code %d -- the bridge declined rather than "
                    "guessing. That is a correct outcome, but it means the attend path below "
                    "cannot be exercised on this model/config.\n", p_ok);
        llama_free(ctx); llama_free_model(model); llama_backend_free();
        return ok ? 7 : 1;      // 7 = ran, but refused: reported distinctly, never as a pass
    }

    // ---- 2/3. attend at FULL budget must match an independent dense reference -------------
    const int64_t d = k->ne[0];
    struct ggml_tensor * vt = kv->v_l[0];
    // Read at the cache's ACTUAL dtype. The first run cast an F16 buffer to const float*,
    // which produced cosine 0.000000 -- a defect in this reference, not in the bridge.
    auto KAT = [&](int64_t t, int64_t j) -> float {
        return k->type == GGML_TYPE_F16
             ? ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(k->data)[t*d + j])
             : static_cast<const float *>(k->data)[t*d + j];
    };
    auto VAT = [&](int64_t t, int64_t j) -> float {
        return vt->type == GGML_TYPE_F16
             ? ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(vt->data)[t*d + j])
             : static_cast<const float *>(vt->data)[t*d + j];
    };

    std::vector<float> q(d);
    for (int64_t j = 0; j < d; ++j) q[j] = KAT((int64_t)(used/2), j);   // query near a real key

    std::vector<float> out_full(d, 0.f);
    const int rc_full = offload_attn_step_from_ctx(ctx, 0, q.data(), out_full.data(),
                                                   16, 1e9f, 0);           // budget >=1 => a count
    std::printf("\nattend(full budget) rc=%d finite=%d\n", rc_full, (int) finite_vec(out_full));

    // independent dense reference over the SAME rows
    std::vector<double> logit(used);
    const double inv = 1.0 / std::sqrt((double) d);
    double mx = -1e300;
    for (int t = 0; t < used; ++t) {
        double s = 0; for (int64_t j = 0; j < d; ++j) s += (double) q[j] * (double) KAT(t, j);
        logit[t] = s * inv; if (logit[t] > mx) mx = logit[t];
    }
    double sum = 0; for (int t = 0; t < used; ++t) { logit[t] = std::exp(logit[t]-mx); sum += logit[t]; }
    std::vector<float> dense(d, 0.f);
    for (int t = 0; t < used; ++t) {
        const double w = logit[t]/sum;
        for (int64_t j = 0; j < d; ++j) dense[j] += (float)(w * (double) VAT(t, j));
    }

    const double cos_full = cosine(out_full, dense);
    std::printf("cosine(offload@full, independent dense) = %.6f  %s\n",
                cos_full, cos_full >= 0.999 ? "PASS" : "FAIL");
    ok = ok && rc_full == 0 && finite_vec(out_full) && cos_full >= 0.999;

    // ---- 4. small budget must DIVERGE, or the selector is not selecting -------------------
    std::vector<float> out_small(d, 0.f);
    const int rc_small = offload_attn_step_from_ctx(ctx, 0, q.data(), out_small.data(),
                                                    16, 0.02f, 0);
    const double cos_small = cosine(out_small, dense);
    const bool diverged = cos_small < 0.999999;
    std::printf("cosine(offload@2%%,  independent dense) = %.6f  %s\n", cos_small,
                diverged ? "diverges (correct -- the selector is selecting)"
                         : "identical (suspect: nothing was actually dropped)");
    ok = ok && rc_small == 0 && diverged;

    // ---- 5. REFUSAL ARM: a KV type the bridge does not support must be DECLINED ----------
    // Without this the suite can only ever return "supported". The reason the F32-only design
    // survived is that its refusal path was never exercised against anything real -- it looked
    // principled while being unreachable. So drive a QUANTISED KV cache through the same probe
    // and require the specific type code, not merely nonzero.
    {
        llama_context_params qp = llama_context_default_params();
        qp.n_ctx      = 512;
        qp.n_batch    = 512;
        qp.type_k     = GGML_TYPE_Q8_0;    // quantised KV: unsupported by the offload math
        qp.type_v     = GGML_TYPE_Q8_0;
        qp.flash_attn = true;              // the fork requires FA for a non-default V type
        llama_context * qctx = llama_init_from_model(model, qp);
        if (!qctx) {
            std::printf("\nrefusal arm: q8_0 KV context did not create -- arm SKIPPED "
                        "(cannot conclude the probe refuses; NOT counted as a pass)\n");
            ok = false;
        } else {
            std::vector<llama_token> qt(toks.begin(), toks.begin() + (n_tok < 400 ? n_tok : 400));
            llama_batch qb = llama_batch_get_one(qt.data(), (int) qt.size());   // fresh context: positions start at 0
            const int qdec = llama_decode(qctx, qb);
            const int qprobe = offload_attn_probe(qctx, 0);
            auto * qc = static_cast<llama_context *>(qctx);
            struct ggml_tensor * qk = qc->kv_self.get_k_raw(0);
            std::printf("\nrefusal arm: decode rc=%d  K type=%d  probe=%d  %s\n",
                        qdec, qk ? (int) qk->type : -1, qprobe,
                        qprobe == -5 ? "declined with E_TYPE (correct)"
                                     : "did NOT decline with E_TYPE");
            ok = ok && (qprobe == -5);
            llama_free(qctx);
        }
    }

    std::printf("\nOFFLOAD-ATTN REAL-KV TEST: %s\n", ok ? "PASS" : "FAIL");
    llama_free(ctx); llama_free_model(model); llama_backend_free();
    return ok ? 0 : 1;
}
