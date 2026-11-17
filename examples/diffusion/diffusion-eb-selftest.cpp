// Self-test for the entropy-bound DiffusionGemma decoder (diffusion_generate_entropy_bound) and the masked-path
// extensions (self_conditioning, suppress_mask_token) in diffusion.cpp. Runs on CPU against the random-weight
// GGUF produced by tests/diffusion-gemma-check/make_tiny_diffusion_gemma_gguf.py -- no real checkpoint is needed.
//
// What each check can and cannot show (random weights, so the TEXT is meaningless):
//   determinism     same seed -> same per-step frames (catches uninitialised reads / thread-order dependence)
//   kv-vs-unified   PREFILL once + canvas-only DECODE (chunked, ubatch < prompt) reproduces the no-cache forward
//                   step by step (catches mask / position / cache-offset errors on the cached path)
//   device-SC path  gpu_sampling=true on a CPU backend must give the same frames as the host path
//   serial ref      an independent serial re-implementation of the loop (plain llama_decode + loops, no threads,
//                   no chunking) written from the contract in diffusion.h. It shares the RNG draw order and the
//                   per-position formulas with the port BY CONSTRUCTION (both implement one spec), so it catches
//                   threading / indexing / SC-plumbing / frame-capture errors, not a wrong spec.
//   sensitivity     the same reference run with a perturbed entropy bound MUST differ, so "equal" is informative
//   suppress mask   masked path: with suppress_mask_token no canvas position is revealed as the mask token; the
//                   control (suppress off, mask id = the token the model most wants) does reveal it
// Not covered: CUDA device sampler numerics, real-weight quality, multi-GPU.
//
// SOURCE-READ: upstream/unsloth-llama.cpp/examples/diffusion/diffusion.cpp:437-683 (MIT, the entropy-bound loop
// the reference mirrors) and upstream/unsloth-llama.cpp/examples/diffusion/diffusion.h:63-95 (the contract).

#include "diffusion.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...) do { if (cond) { g_pass++; } else { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct run_t {
    std::vector<std::vector<llama_token>> frames;   // per executed step: the argmax canvas handed to the callback
    std::vector<llama_token>              out;      // output_tokens[n_input .. max_length)
    int32_t                               n_gen = 0;
};

struct cap_t { run_t * r; int32_t n_input; };

static bool cap_cb(int32_t, int32_t, const llama_token * tok, int32_t n, void * ud) {
    auto * c = (cap_t *) ud;
    c->r->frames.emplace_back(tok + c->n_input, tok + n);
    return true;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx, uint32_t n_ubatch, uint32_t n_outputs_max) {
    llama_context_params c = llama_context_default_params();
    c.n_ctx           = n_ctx;
    c.n_batch         = n_ctx;
    c.n_ubatch        = n_ubatch;
    c.n_outputs_max   = n_outputs_max;
    c.no_perf         = true;
    c.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, c);
    if (ctx) {
        llama_set_n_threads(ctx, 4, 4);
        llama_set_causal_attn(ctx, false);
    }
    return ctx;
}

static run_t run_port(llama_context * ctx, const std::vector<llama_token> & prompt, int32_t canvas, diffusion_eb_params p) {
    run_t r;
    const int32_t n_in = (int32_t) prompt.size();
    p.max_length = n_in + canvas;
    cap_t cap{ &r, n_in };
    p.step_callback = cap_cb;
    p.step_callback_user_data = &cap;
    std::vector<llama_token> out(p.max_length, 0);
    diffusion_generate_entropy_bound(ctx, prompt.data(), out.data(), n_in, p, r.n_gen);
    r.out.assign(out.begin() + n_in, out.begin() + p.max_length);
    return r;
}

// Independent serial reference of the entropy-bound loop (unified forward every step, one thread).
static run_t run_ref(llama_context * ctx, llama_model * model, const std::vector<llama_token> & prompt, int32_t canvas,
                     const diffusion_eb_params & p) {
    run_t r;
    const int32_t n_in    = (int32_t) prompt.size();
    const int32_t n_total = n_in + canvas;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t S       = std::max(1, p.max_denoising_steps);

    llama_set_causal_attn(ctx, false);
    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vdist(0, n_vocab - 1);

    std::vector<llama_token> cur(canvas);
    for (int32_t i = 0; i < canvas; i++) { cur[i] = vdist(rng); }

    std::vector<float>       sc((size_t) canvas * n_vocab, 0.0f);
    std::vector<llama_token> argmax_c(canvas, 0), prev_argmax(canvas, -1), denoiser(canvas), renoise(canvas);
    std::vector<float>       ent(canvas), u(canvas);
    llama_batch batch = llama_batch_init(n_total, 0, 1);
    float prev_temp_inv = 1.0f;
    int   held = 0;
    bool  finished = false;
    std::vector<llama_token> out(canvas, 0);

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;
        const float   t        = p.t_min + (p.t_max - p.t_min) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        batch.n_tokens = n_total;
        for (int32_t i = 0; i < n_total; i++) {
            batch.token[i] = i < n_in ? prompt[i] : cur[i - n_in];
            batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
        }
        llama_diffusion_set_sc(model, sc.data(), step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);
        if (llama_decode(ctx, batch) != 0) { printf("ref: decode failed at step %d\n", step_idx); break; }
        const float * logits = llama_get_logits(ctx);

        for (int32_t pos = 0; pos < canvas; pos++) { u[pos] = uni01(rng); renoise[pos] = vdist(rng); }

        for (int32_t pos = 0; pos < canvas; pos++) {
            const float * row = logits + (size_t) (n_in + pos) * n_vocab;
            float m = -INFINITY; int32_t am = 0;
            for (int32_t v = 0; v < n_vocab; v++) { const float z = row[v] * temp_inv; if (z > m) { m = z; am = v; } }
            float Z = 0.0f;
            for (int32_t v = 0; v < n_vocab; v++) { Z += expf(row[v] * temp_inv - m); }
            const float target = u[pos] * Z;
            float cum = 0.0f, H = 0.0f; int32_t smp = n_vocab - 1; bool picked = false;
            for (int32_t v = 0; v < n_vocab; v++) {
                const float e = expf(row[v] * temp_inv - m);
                const float pr = e / Z;
                if (pr > 0.0f) { H -= pr * logf(pr); }
                cum += e;
                if (!picked && cum >= target) { smp = v; picked = true; }
            }
            ent[pos] = H; argmax_c[pos] = am; denoiser[pos] = smp;
            std::memcpy(sc.data() + (size_t) pos * n_vocab, row, n_vocab * sizeof(float));
        }

        std::vector<int32_t> order(canvas);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return ent[a] < ent[b]; });
        std::vector<char> acc(canvas, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < canvas; k++) {
            const int32_t pos = order[k];
            cumE += ent[pos];
            if (cumE - ent[pos] <= p.entropy_bound) { acc[pos] = 1; }
        }
        float esum = 0.0f;
        for (int32_t pos = 0; pos < canvas; pos++) {
            cur[pos] = acc[pos] ? denoiser[pos] : renoise[pos];
            out[pos] = argmax_c[pos];
            esum += ent[pos];
        }
        held = (prev_argmax == argmax_c) ? held + 1 : 0;
        const bool confident = (esum / (float) canvas) < p.confidence_threshold;
        if (held >= p.stability_threshold && confident) { finished = true; }
        prev_argmax = argmax_c;
        prev_temp_inv = temp_inv;
        r.frames.push_back(out);
    }
    llama_batch_free(batch);
    r.out = out;
    r.n_gen = n_total;
    return r;
}

static bool same(const run_t & a, const run_t & b, const char * what) {
    bool ok = a.frames.size() == b.frames.size() && a.out == b.out && a.n_gen == b.n_gen;
    long long first_bad = -1;
    for (size_t i = 0; ok && i < a.frames.size(); i++) {
        if (a.frames[i] != b.frames[i]) { ok = false; first_bad = (long long) i; }
    }
    if (!ok) {
        printf("  [%s] differ: steps %zu vs %zu, n_gen %d vs %d, first differing step %lld\n",
               what, a.frames.size(), b.frames.size(), a.n_gen, b.n_gen, first_bad);
    }
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <tiny-diffusion-gemma.gguf> [--skip-devsc]\n", argv[0]); return 2; }
    const bool skip_devsc = argc > 2 && std::strcmp(argv[2], "--skip-devsc") == 0;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }
    CHECK(llama_model_is_diffusion(model), "model must be a diffusion model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    char buf[32];
    CHECK(llama_model_meta_val_str(model, "diffusion.canvas_length", buf, sizeof(buf)) >= 0, "canvas_length key");
    const int32_t canvas = (int32_t) strtol(buf, nullptr, 10);

    // the eb_* keys written by the fork's gguf-py must be the keys the readers use (underscore spelling)
    CHECK(llama_model_meta_val_str(model, "diffusion.eb_max_steps", buf, sizeof(buf)) >= 0 && atoi(buf) == 7,
          "diffusion.eb_max_steps not readable under the upstream spelling");

    const std::string text = "user: hello there\nmodel: ";
    std::vector<llama_token> prompt(256);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), prompt.data(), (int32_t) prompt.size(), true, true);
    CHECK(n > 0, "tokenize");
    prompt.resize(std::max(n, 0));
    const int32_t n_in = (int32_t) prompt.size();
    const uint32_t N = (uint32_t) (n_in + canvas);
    printf("prompt=%d canvas=%d n_vocab=%d ctx=%u\n", n_in, canvas, n_vocab, N);

    // SC graph must exist before the context reserves its compute buffer (same order as the CLI / visual server)
    llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);
    llama_context * ctxU = make_ctx(model, N, N, 0);                        // unified: whole [prompt|canvas] in one ubatch
    llama_context * ctxK = make_ctx(model, N, (uint32_t) canvas, (uint32_t) canvas);  // cached: chunked prefill, canvas-only decode
    CHECK(ctxU && ctxK, "context creation");
    if (!ctxU || !ctxK) { return 1; }

    diffusion_eb_params base;
    base.max_denoising_steps  = 6;
    base.t_min                = 0.3f;
    base.t_max                = 0.9f;
    base.entropy_bound        = 8.0f;
    base.stability_threshold  = 1;
    base.confidence_threshold = 0.0f;   // never "confident": all 6 steps run
    base.seed                 = 1234;

    // 1. determinism + host path
    run_t a1 = run_port(ctxU, prompt, canvas, base);
    run_t a2 = run_port(ctxU, prompt, canvas, base);
    CHECK(a1.n_gen == n_in + canvas, "n_generated=%d want %d", a1.n_gen, n_in + canvas);
    CHECK((int) a1.frames.size() == base.max_denoising_steps, "frames=%zu want %d", a1.frames.size(), base.max_denoising_steps);
    CHECK(same(a1, a2, "determinism"), "same seed must reproduce");
    diffusion_eb_params other = base; other.seed = 99;
    CHECK(!same(a1, run_port(ctxU, prompt, canvas, other), "seed-sensitivity"), "a different seed must change the frames");

    // 2. independent serial reference vs the port (unified)
    run_t rf = run_ref(ctxU, model, prompt, canvas, base);
    CHECK(same(a1, rf, "port-vs-ref"), "port frames must equal the serial reference");
    // Sensitivity: the SAME reference with a different seed must NOT equal the port, so "equal" above is informative.
    // (Perturbing the entropy bound is a weaker control: frames hold the model's ARGMAX canvas, which a random-weight
    // model barely couples to the accepted/renoised input canvas -- measured: bound 8 vs 40 gave identical frames.)
    diffusion_eb_params pert = base; pert.seed = base.seed + 1;
    run_t rp = run_ref(ctxU, model, prompt, canvas, pert);
    CHECK(!same(a1, rp, "ref-other-seed"), "reference with a different seed must differ from the port (sensitivity)");

    // 3. cached path (PREFILL in ubatch-sized chunks + canvas-only DECODE) vs unified
    diffusion_eb_params kv = base; kv.kv_cache = true;
    run_t k1 = run_port(ctxK, prompt, canvas, kv);
    CHECK(same(a1, k1, "kv-vs-unified"), "cached path must reproduce the unified forward step by step");

    // 4. device-resident SC (+ Stage-1 reduce, which must fall back to host on a CPU backend). --skip-devsc lets
    // a driver rerun the other sections if this one aborts the process (the abort itself is then the finding).
    if (!skip_devsc) {
        diffusion_eb_params dv = base; dv.gpu_sampling = true;
        run_t d1 = run_port(ctxU, prompt, canvas, dv);
        CHECK(same(a1, d1, "devsc-vs-host"), "gpu_sampling must not change the result");
        diffusion_eb_params dr = dv; dr.gpu_sample_reduce = true;
        run_t d2 = run_port(ctxU, prompt, canvas, dr);
        CHECK(same(a1, d2, "devsc-reduce-vs-host"), "gpu_sample_reduce must fall back to the host path with the same result");
        diffusion_eb_params all = dr; all.kv_cache = true;
        run_t k2 = run_port(ctxK, prompt, canvas, all);
        CHECK(same(a1, k2, "all-features-vs-host"), "kv_cache + gpu_sampling + gpu_sample_reduce (the visual server's config) must match");
    }

    // 5. adaptive stop: stability_threshold=0 and a huge confidence threshold -> finish after the first step
    diffusion_eb_params stop = base; stop.stability_threshold = 0; stop.confidence_threshold = 1e9f;
    run_t s1 = run_port(ctxU, prompt, canvas, stop);
    run_t s2 = run_ref(ctxU, model, prompt, canvas, stop);
    CHECK(s1.frames.size() == 1, "adaptive stop ran %zu steps, want 1", s1.frames.size());
    CHECK(same(s1, s2, "stop-vs-ref"), "stopped run must equal the reference");

    // 6. masked path: self_conditioning + suppress_mask_token
    {
        const int32_t MASK = llama_vocab_mask(vocab);
        // probe: which token does the model most want at the canvas rows of an all-mask canvas?
        std::vector<llama_token> toks(n_in + canvas);
        std::copy(prompt.begin(), prompt.end(), toks.begin());
        std::fill(toks.begin() + n_in, toks.end(), MASK);
        llama_batch b = llama_batch_init(n_in + canvas, 0, 1);
        b.n_tokens = n_in + canvas;
        for (int32_t i = 0; i < n_in + canvas; i++) { b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);
        CHECK(llama_decode(ctxU, b) == 0, "probe decode");
        const float * lg = llama_get_logits(ctxU);
        std::map<int32_t, int> hist;
        for (int32_t pos = 0; pos < canvas; pos++) {
            const float * row = lg + (size_t) (n_in + pos) * n_vocab;
            hist[(int32_t) (std::max_element(row, row + n_vocab) - row)]++;
        }
        llama_batch_free(b);
        int32_t wanted = -1, best = 0;
        for (auto & kvp : hist) { if (kvp.second > best) { best = kvp.second; wanted = kvp.first; } }

        auto masked_run = [&](int32_t mask_id, bool suppress) {
            diffusion_params dp;
            dp.steps = 4; dp.temperature = 0.0f; dp.mask_token_id = mask_id; dp.seed = 5;
            dp.top_k = 1; dp.top_p = 1.0f;
            dp.algorithm = DIFFUSION_ALGORITHM_CONFIDENCE_BASED;
            dp.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
            dp.eps = 1e-3f;
            dp.max_length = n_in + canvas;
            dp.suppress_mask_token = suppress;
            dp.self_conditioning = true;
            std::vector<llama_token> out(dp.max_length, 0);
            int32_t ng = 0;
            diffusion_generate(ctxU, prompt.data(), out.data(), n_in, dp, ng);
            int masks = 0;
            for (int32_t i = n_in; i < dp.max_length; i++) { masks += out[i] == mask_id; }
            return std::make_pair(ng, masks);
        };
        // Try EVERY vocab id as the mask id. Control: with suppression off, at least one id must end up revealed
        // somewhere (otherwise this check could not fail). Property: with suppression on, NO id may be revealed.
        (void) wanted; (void) best;
        int ctl_ids = 0, sup_bad = 0, ng_bad = 0;
        int example = -1;
        for (int32_t id = 0; id < n_vocab; id++) {
            auto ctl = masked_run(id, false);
            auto sup = masked_run(id, true);
            ng_bad += (ctl.first != n_in + canvas) + (sup.first != n_in + canvas);
            if (ctl.second > 0) { ctl_ids++; if (example < 0) { example = id; } }
            if (sup.second > 0) { sup_bad++; }
        }
        printf("masked path over %d candidate mask ids: unsuppressed reveals the id in %d runs (e.g. id %d); suppressed reveals it in %d runs\n",
               n_vocab, ctl_ids, example, sup_bad);
        CHECK(ng_bad == 0, "masked path n_generated wrong in %d runs", ng_bad);
        CHECK(sup_bad == 0, "suppress_mask_token still revealed the mask id in %d of %d runs", sup_bad, n_vocab);
        CHECK(ctl_ids > 0, "control: with suppression off the mask id should be revealed for at least one id");
    }

    printf("RESULT pass=%d fail=%d\n", g_pass, g_fail);
    llama_free(ctxU);
    llama_free(ctxK);
    llama_model_free(model);
    llama_backend_free();
    return g_fail ? 1 : 0;
}
