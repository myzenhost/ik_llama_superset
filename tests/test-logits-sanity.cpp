// Is the engine producing usable logits at all, and does its state advance between decodes?
//
// GATE-CLAIM-OK: "CPU" below names a BACKEND SOURCE DIRECTORY / a build configuration, not a
// claim that any device is unavailable. This probe runs whatever backend the binary was built
// with and allocates no accelerator memory of its own.
//
// WHY THIS EXISTS. With the context, KV and decode paths repaired, the engine generates tokens --
// and every one of them is the same token. On two different models (SmolLM2-135M-Instruct-Q8_0
// and the earlier one) `llama-cli -n 12` returns twelve identical `!`. In a BPE/tiktoken-style
// vocabulary `!` is token id 0, and "argmax is always 0" is the signature of logits that carry no
// information -- all equal, all zero, or all NaN -- because argmax of a flat array returns the
// first index. That is a HYPOTHESIS about the cause, and the point of this file is to test it
// rather than act on it: "it always prints !" is a symptom, and at least three different defects
// produce it.
//
// THE TWO QUESTIONS IT SEPARATES, which need different fixes:
//   1. ARE THE LOGITS DEGENERATE?  If the final-token logits are constant, zero or NaN, the fault
//      is in the graph or the output head, and sampling is an innocent bystander.
//   2. DOES THE STATE ADVANCE?     If the logits are perfectly healthy but IDENTICAL at step N and
//      step N+1, the graph is fine and the fault is that the decode is not moving: KV not written,
//      position not advancing, or the same cached graph being replayed. Healthy-but-frozen logits
//      would also produce the same token forever, and would be invisible to question 1 alone.
// A probe that only asked question 1 would report "logits fine" and leave the bug untouched.
//
// CONTROLS, because a statistic that cannot fail proves nothing:
//   * the summary statistics are first run on SYNTHETIC arrays with known answers -- a flat array
//     (which must be reported as degenerate) and a spiked array (which must NOT be). If the
//     degeneracy test cannot tell those two apart, every verdict below is meaningless.
//   * n_vocab is checked to be non-trivial, so a model that failed to load cannot masquerade as a
//     model with boring logits.
//   * the step-N vs step-N+1 comparison is run against ITSELF first (must report identical), so
//     "they differ" cannot come from a comparator that always reports a difference.
//
// This probe DOES NOT sample, so nothing here depends on the sampler chain, the temperature, the
// seed, or any of the sampling work done earlier in this fork.
//
// Usage: test-logits-sanity <model.gguf> [prompt]
// Skips with exit 0 and a printed reason when no model path is given, so it can sit in the test
// suite on machines with no model file.

#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

struct logit_stats {
    int    n        = 0;
    int    n_nan    = 0;
    int    n_inf    = 0;
    float  min      = 0.0f;
    float  max      = 0.0f;
    double mean     = 0.0;
    double stddev   = 0.0;
    int    argmax   = -1;
    size_t n_unique = 0;
};

static logit_stats summarise(const float * v, int n) {
    logit_stats s;
    s.n = n;
    if (n <= 0) {
        return s;
    }
    s.min = s.max = 0.0f;
    bool have_finite = false;
    double sum = 0.0;
    int    n_finite = 0;
    for (int i = 0; i < n; i++) {
        const float x = v[i];
        if (std::isnan(x)) { s.n_nan++; continue; }
        if (std::isinf(x)) { s.n_inf++; continue; }
        if (!have_finite) { s.min = s.max = x; have_finite = true; }
        s.min = std::min(s.min, x);
        s.max = std::max(s.max, x);
        sum += (double) x;
        n_finite++;
    }
    s.mean = n_finite ? sum / n_finite : 0.0;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        const float x = v[i];
        if (std::isnan(x) || std::isinf(x)) continue;
        const double d = (double) x - s.mean;
        var += d*d;
    }
    s.stddev = n_finite ? std::sqrt(var / n_finite) : 0.0;

    // argmax computed EXACTLY as a greedy sampler would: first strict maximum wins. This is what
    // makes "argmax == 0 on a flat array" a prediction rather than a coincidence.
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    s.argmax = best;

    std::vector<float> sorted(v, v + n);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    s.n_unique = sorted.size();

    return s;
}

// The degeneracy call, in one place so the controls below exercise the SAME predicate the verdict
// uses. A separate copy for the controls would test nothing.
static bool is_degenerate(const logit_stats & s) {
    return s.n_nan > 0 || s.n_unique <= 1 || s.stddev == 0.0;
}

static void print_stats(const char * label, const logit_stats & s) {
    printf("  %-22s n=%d  unique=%zu  min=%.6g  max=%.6g  mean=%.6g  sd=%.6g  NaN=%d  Inf=%d  argmax=%d\n",
           label, s.n, s.n_unique, (double) s.min, (double) s.max, s.mean, s.stddev,
           s.n_nan, s.n_inf, s.argmax);
}

static void controls(void) {
    printf("CONTROLS on the statistics themselves (they decide whether any verdict below counts)\n");

    std::vector<float> flat(4096, 0.0f);
    const logit_stats sf = summarise(flat.data(), (int) flat.size());
    const bool ok_flat = is_degenerate(sf) && sf.argmax == 0 && sf.n_unique == 1;
    printf("  %-4s flat array reported degenerate, argmax=%d, unique=%zu\n",
           ok_flat ? "ok" : "FAIL", sf.argmax, sf.n_unique);
    if (!ok_flat) g_failures++;

    std::vector<float> spike(4096, 0.0f);
    for (size_t i = 0; i < spike.size(); i++) {
        spike[i] = 0.001f * (float) (i % 97);      // genuinely varied, not one spike on a flat bed
    }
    spike[1234] = 99.0f;
    const logit_stats ss = summarise(spike.data(), (int) spike.size());
    const bool ok_spike = !is_degenerate(ss) && ss.argmax == 1234;
    printf("  %-4s varied array NOT reported degenerate, argmax=%d (expected 1234)\n",
           ok_spike ? "ok" : "FAIL", ss.argmax);
    if (!ok_spike) g_failures++;

    // The comparator, both directions.
    const bool same_is_same = std::memcmp(flat.data(), flat.data(), flat.size()*sizeof(float)) == 0;
    const bool diff_is_diff = std::memcmp(flat.data(), spike.data(), flat.size()*sizeof(float)) != 0;
    const bool ok_cmp = same_is_same && diff_is_diff;
    printf("  %-4s comparator: identical reads identical AND different reads different\n",
           ok_cmp ? "ok" : "FAIL");
    if (!ok_cmp) g_failures++;

    printf("\n");
}

// ---------------------------------------------------------------------------------------------
// NODE SCAN, run DURING graph execution.
//
// WHY IT IS A CALLBACK AND NOT A LOOP OVER gf->nodes AFTERWARDS. ggml's graph allocator REUSES
// buffers: a node's output memory is handed to a later node once its last consumer has run. A
// post-hoc sweep of gf->nodes therefore reads recycled memory for everything except the surviving
// outputs, and reports it as that node's value. MEASURED: such a sweep reported inp_tokens as
// -2621440 on one run and -524288 on the next, for the same prompt and the same token -- a value
// that changes between identical runs is the scratch buffer, not the input. That reading was on
// its way to being recorded as a root cause. The scheduler's eval callback fires immediately
// after each node computes, before its buffer can be reused, which is the whole reason the hook
// exists.
//
// The callback returns the fork's three-valued code: 1 = give me the data and synchronize first.
struct scan_state {
    int  node_idx  = 0;
    int  first_bad = -1;
    int  n_scanned = 0;
    int  n_skipped = 0;
};

static void describe_src(const ggml_tensor * sr, int s) {
    char vals[256] = {0};
    if (sr->data && sr->buffer && ggml_backend_buffer_is_host(sr->buffer)) {
        const int64_t ns = std::min<int64_t>(8, ggml_nelements(sr));
        int off = 0;
        if (sr->type == GGML_TYPE_I32) {
            const int32_t * p = (const int32_t *) sr->data;
            for (int64_t k = 0; k < ns && off < 200; k++)
                off += snprintf(vals+off, sizeof(vals)-off, " %d", p[k]);
        } else if (sr->type == GGML_TYPE_F32) {
            const float * p = (const float *) sr->data;
            for (int64_t k = 0; k < ns && off < 200; k++)
                off += snprintf(vals+off, sizeof(vals)-off, " %g", (double) p[k]);
        } else {
            const uint8_t * p = (const uint8_t *) sr->data;
            for (int64_t k = 0; k < 8 && off < 200; k++)
                off += snprintf(vals+off, sizeof(vals)-off, " %02x", p[k]);
        }
    } else {
        snprintf(vals, sizeof(vals), " <not host-readable>");
    }
    printf("     src[%d] '%s' op=%s type=%s ne=[%lld,%lld] first8:%s\n",
           s, sr->name, ggml_op_name(sr->op), ggml_type_name(sr->type),
           (long long) sr->ne[0], (long long) sr->ne[1], vals);
}

static int nan_scan_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return 1;   // want every node's data, synchronized before we read it
    }
    scan_state * st = (scan_state *) user_data;
    const int ni = st->node_idx++;

    if (!t || t->type != GGML_TYPE_F32 || !t->buffer ||
        !ggml_backend_buffer_is_host(t->buffer) || !t->data) {
        st->n_skipped++;
        return 1;
    }
    st->n_scanned++;

    const int64_t n = ggml_nelements(t);
    const float * d = (const float *) t->data;
    int64_t n_nan = 0, n_inf = 0;
    float mn = 0.0f, mx = 0.0f;
    bool have = false;
    for (int64_t k = 0; k < n; k++) {
        if (std::isnan(d[k]))      { n_nan++; continue; }
        if (std::isinf(d[k]))      { n_inf++; continue; }
        if (!have) { mn = mx = d[k]; have = true; }
        mn = std::min(mn, d[k]);
        mx = std::max(mx, d[k]);
    }

    if (ni < 8) {
        printf("  node #%-3d %-22s op=%-14s ne=[%lld,%lld] finite-min=%-12.6g finite-max=%-12.6g nan=%lld inf=%lld of %lld\n",
               ni, t->name, ggml_op_name(t->op),
               (long long) t->ne[0], (long long) t->ne[1],
               (double) mn, (double) mx, (long long) n_nan, (long long) n_inf, (long long) n);
    }

    if ((n_nan || n_inf) && st->first_bad < 0) {
        st->first_bad = ni;
        printf("  >>> FIRST NON-FINITE at node #%d '%s' op=%s  nan=%lld inf=%lld of %lld\n",
               ni, t->name, ggml_op_name(t->op),
               (long long) n_nan, (long long) n_inf, (long long) n);
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
            describe_src(t->src[s], s);
        }
    }
    return 1;
}

int main(int argc, char ** argv) {
    controls();

    if (argc < 2) {
        printf("SKIP -- no model path given. Usage: %s <model.gguf> [prompt]\n", argv[0]);
        printf("The controls above ran and are the only thing this exit code covers.\n");
        return g_failures == 0 ? 0 : 1;
    }

    const std::string model_path = argv[1];
    const std::string prompt     = argc >= 3 ? argv[2] : "The capital of France is";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;      // this probe is about the graph, not about placement

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        printf("FAIL -- could not load %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;

    // Opt-in: LLAMA_NAN_SCAN=1 installs the per-node scan above. Off by default so the ordinary
    // test run stays quiet and fast.
    scan_state scan;
    const bool do_scan = getenv("LLAMA_NAN_SCAN") != nullptr;
    if (do_scan) {
        cparams.cb_eval           = nan_scan_cb;
        cparams.cb_eval_user_data = &scan;
        printf("NODE SCAN ENABLED (values read during execution, before buffer reuse)\n");
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("FAIL -- could not create a context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int n_vocab = llama_n_vocab(model);
    printf("model   : %s\n", model_path.c_str());
    printf("prompt  : \"%s\"\n", prompt.c_str());
    printf("n_vocab : %d\n\n", n_vocab);

    if (n_vocab < 1000) {
        printf("FAIL -- n_vocab %d is implausible; the model did not load as expected, so nothing\n"
               "        below would be a statement about the engine.\n", n_vocab);
        g_failures++;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    printf("tokens  : %zu ->", tokens.size());
    for (size_t i = 0; i < tokens.size() && i < 16; i++) {
        printf(" %d", tokens[i]);
    }
    printf("\n\n");

    if (tokens.empty()) {
        printf("FAIL -- the prompt tokenized to nothing.\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ---- STEP 1: the prompt, from scratch -----------------------------------------------------
    // upstream batch ABI (0297995eb): sequence 0, positions start at 0 on the empty cache (was explicit pos 0, seq 0)
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        printf("FAIL -- llama_decode returned non-zero on the prompt.\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (do_scan) {
        printf("  scan coverage step 1: %d node(s) inspected, %d skipped (not f32/host), "
               "first_bad=%d\n\n", scan.n_scanned, scan.n_skipped, scan.first_bad);
        scan = scan_state();
    }

    const float * l1 = llama_get_logits(ctx);
    if (!l1) {
        printf("FAIL -- llama_get_logits returned NULL after a successful decode.\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::vector<float> step1(l1, l1 + n_vocab);
    const logit_stats s1 = summarise(step1.data(), n_vocab);

    printf("QUESTION 1 -- are the logits degenerate?\n");
    print_stats("step 1 (prompt)", s1);

    // ---- STEP 2: feed step 1's own argmax back, as a generator would ---------------------------
    llama_token next = (llama_token) s1.argmax;
    // sequence 0 continues from its cached positions, i.e. tokens.size() after the prompt decode above (was explicit pos, seq 0)
    if (llama_decode(ctx, llama_batch_get_one(&next, 1))) {
        printf("FAIL -- llama_decode returned non-zero on the second step.\n");
        g_failures++;
    }
    const float * l2 = llama_get_logits(ctx);
    std::vector<float> step2(l2 ? l2 : l1, (l2 ? l2 : l1) + n_vocab);
    const logit_stats s2 = summarise(step2.data(), n_vocab);
    print_stats("step 2 (+1 token)", s2);

    const bool deg1 = is_degenerate(s1);
    const bool deg2 = is_degenerate(s2);
    printf("  %-4s step 1 logits carry information (not constant, no NaN)\n", deg1 ? "FAIL" : "ok");
    printf("  %-4s step 2 logits carry information (not constant, no NaN)\n", deg2 ? "FAIL" : "ok");
    if (deg1) g_failures++;
    if (deg2) g_failures++;

    // ---- QUESTION 2: did anything change? -----------------------------------------------------
    printf("\nQUESTION 2 -- does the state advance between decodes?\n");
    const bool identical = std::memcmp(step1.data(), step2.data(), n_vocab*sizeof(float)) == 0;
    double max_abs_delta = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        const double d = std::fabs((double) step1[i] - (double) step2[i]);
        if (d > max_abs_delta) max_abs_delta = d;
    }
    printf("  step1 argmax = %d   step2 argmax = %d   max|delta| = %.6g\n",
           s1.argmax, s2.argmax, max_abs_delta);
    printf("  %-4s the two steps produced DIFFERENT logits\n", identical ? "FAIL" : "ok");
    if (identical) {
        printf("       byte-identical logits after decoding an extra token means the decode did\n");
        printf("       not advance the state. The graph may be perfectly correct; what is frozen\n");
        printf("       is the KV write, the position, or graph reuse. This is a DIFFERENT defect\n");
        printf("       from degenerate logits and is not fixed by touching the output head.\n");
        g_failures++;
    }

    // ---- what a greedy sampler would actually emit --------------------------------------------
    printf("\nTop 5 by logit at step 1 (what greedy sampling would pick, in order):\n");
    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return step1[a] > step1[b]; });
        for (int k = 0; k < 5; k++) {
            const int id = idx[k];
            const char * txt = llama_vocab_get_text(vocab, id);
            printf("   %d. id=%-7d logit=%12.6f  %s\n", k + 1, id, (double) step1[id],
                   txt ? txt : "(null)");
        }
    }

    printf("\n%s -- %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    printf("SCOPE: one model, one prompt, two decode steps, no sampling. A PASS here says the\n"
           "logits are informative and the state moves; it does NOT say the numbers are correct.\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return g_failures == 0 ? 0 : 1;
}
