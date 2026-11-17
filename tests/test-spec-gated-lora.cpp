// SCOUT-NA: unit test in the fork tree, no new tool.  GATE-CLAIM-OK: unit test, no resource claims.
// Row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01 (self-speculative decoding with a draft_rows gated LoRA).
//
// Semantics: in a self-speculative round the DRAFT pass rows (the drafted tokens, positions >= pos0 + 1) carry a gated adapter and the
// VERIFY pass rows (target decode of [id_last, draft...]) carry gate 0, so the base model judges acceptance and the output distribution
// stays the base model's. The test drives the real pooled draft speculator (common_speculative_init -> draft-model impl) with the draft
// context on the SAME model as the target, a synthetic adapter (gguf_init_empty, attn_q on every layer, GGUF adapter.lora.gate =
// draft_rows) on the draft context.
//
//   A  DRAFT ROWS GATED   the drafted chain with the adapter attached differs from the base chain, its first token (drafted from id_last,
//                         an accepted token at pos0, gate 0) equals the base chain's first token, and the whole chain equals a reference
//                         chain decoded by hand with llama_set_draft_start(pos0 + 1) on a third context.
//   B  VERIFY ROWS BASE   the same adapter attached to the TARGET context (never given a draft_start) yields logits for [id_last, draft...]
//                         equal to a context without the adapter.
//   C  GATE CLEARED       after the draft call the draft context holds no draft start: re-decoding [id_last, draft...] on it equals the base.
//   D  ADAPTER HAS EFFECT the by-hand gated rows differ from the base rows, so A is not satisfied by a zero-effect adapter.
//
// argv[1] = model (any GGUF of a llama-family arch with attn_q). Exit 77 (skip) without it.
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s  ", __FILE__, __LINE__, #c); fprintf(stderr, "" __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static llama_token argmax(const float * l, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) { if (l[i] > l[b]) { b = i; } }
    return b;
}

static llama_context * mk_ctx(llama_model * m) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 128; cp.n_ubatch = 128; cp.n_seq_max = 1; cp.n_threads = 4; cp.n_threads_batch = 4;
    return llama_init_from_model(m, cp);
}

static std::string meta(const llama_model * m, const char * key) {
    char buf[256];
    const int32_t n = llama_model_meta_val_str(m, key, buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf);
}

// deterministic pseudo-random in [-amp, amp]
struct rng {
    uint32_t s = 12345u;
    float next(float amp) {
        s = s * 1664525u + 1013904223u;
        return ((float) (s >> 8) / (float) (1u << 24) * 2.0f - 1.0f) * amp;
    }
};

// a rank-4 attn_q adapter on every layer with the draft_rows gate; alpha 32 / rank 4 = delta x8
static bool write_gated_adapter(const llama_model * model, const std::string & path) {
    const std::string arch  = meta(model, "general.architecture");
    const int n_embd        = llama_model_n_embd(model);
    const int n_layer       = llama_model_n_layer(model);
    const int n_head        = llama_model_n_head(model);
    const std::string kl    = meta(model, (arch + ".attention.key_length").c_str());
    const int64_t q_out     = kl.empty() ? n_embd : (int64_t) n_head * std::stoi(kl);
    const int64_t rank      = 4;

    ggml_init_params ip = { (size_t) (n_layer * (n_embd + q_out) * rank * 4 + 8 * 1024 * 1024), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g   = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", arch.c_str());
    gguf_set_val_str(g, "general.type", "adapter");
    gguf_set_val_str(g, "adapter.type", "lora");
    gguf_set_val_f32(g, "adapter.lora.alpha", 32.0f);
    gguf_set_val_str(g, "adapter.lora.gate", "draft_rows");
    rng r;
    for (int il = 0; il < n_layer; ++il) {
        const std::string base = "blk." + std::to_string(il) + ".attn_q.weight";
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rank);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, q_out);
        ggml_set_name(a, (base + ".lora_a").c_str());
        ggml_set_name(b, (base + ".lora_b").c_str());
        float * pa = (float *) a->data;
        float * pb = (float *) b->data;
        for (int64_t i = 0; i < ggml_nelements(a); ++i) { pa[i] = r.next(0.15f); }
        for (int64_t i = 0; i < ggml_nelements(b); ++i) { pb[i] = r.next(0.15f); }
        gguf_add_tensor(g, a);
        gguf_add_tensor(g, b);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

static float max_abs_diff(const float * a, const float * b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) { m = std::max(m, std::fabs(a[i] - b[i])); }
    return m;
}

// decode [first, rest...] at pos0.. with logits on every row; returns n_vocab * n_rows logits
static std::vector<float> decode_rows(llama_context * c, int nv, llama_token first, const std::vector<llama_token> & rest, llama_pos pos0) {
    llama_batch b = llama_batch_init(128, 0, 1);
    common_batch_add(b, first, pos0, { 0 }, true);
    for (size_t i = 0; i < rest.size(); ++i) { common_batch_add(b, rest[i], pos0 + 1 + (llama_pos) i, { 0 }, true); }
    std::vector<float> out;
    if (llama_decode(c, b) == 0) {
        out.assign(llama_get_logits_ith(c, 0), llama_get_logits_ith(c, 0) + nv);
        for (int i = 1; i < b.n_tokens; ++i) { out.insert(out.end(), llama_get_logits_ith(c, i), llama_get_logits_ith(c, i) + nv); }
    }
    llama_batch_free(b);
    return out;
}

static void prefill(llama_context * c, const std::vector<llama_token> & prompt) {
    llama_batch b = llama_batch_init(128, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) { common_batch_add(b, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size()); }
    CHECK(llama_decode(c, b) == 0);
    llama_batch_free(b);
}

// one draft call of the pooled draft speculator (draft context = c_dft, same model as the target c_tgt)
static llama_tokens draft_with(llama_context * c_tgt, llama_context * c_dft, const std::vector<llama_token> & prompt, llama_token id_last, int n_max) {
    common_params_speculative sp;
    sp.type    = COMMON_SPECULATIVE_TYPE_DRAFT;
    sp.n_max   = n_max;
    sp.n_min   = 0;
    sp.ctx_tgt = c_tgt;
    sp.ctx_dft = c_dft;
    common_speculative * spec = common_speculative_init(sp, 1);
    CHECK(spec != nullptr);
    llama_tokens D;
    if (!spec) { return D; }
    llama_batch b = llama_batch_init(128, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) { common_batch_add(b, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size()); }
    CHECK(common_speculative_process(spec, b));
    llama_batch_free(b);

    std::vector<llama_token> P = prompt;
    auto & dp   = common_speculative_get_draft_params(spec, 0);
    dp.drafting = true;
    dp.n_max    = -1;
    dp.pos0     = (llama_pos) P.size();
    dp.id_last  = id_last;
    dp.prompt   = &P;
    dp.result   = &D;
    common_speculative_draft(spec);
    common_speculative_free(spec);
    return D;
}

int main(int argc, char ** argv) {
    // model: argv[1], else the LLAMA_TEST_MODEL environment variable (lets a bare run, as the pre-run harness does, find a fixture)
    const char * model_path = argc >= 2 ? argv[1] : getenv("LLAMA_TEST_MODEL");
    if (!model_path || !*model_path) {
        fprintf(stderr, "usage: test-spec-gated-lora <model.gguf> | LLAMA_TEST_MODEL=<model.gguf>   (skipping)\n");
        return 77;
    }
    if (FILE * f = fopen(model_path, "rb")) { fclose(f); } else { fprintf(stderr, "model %s not present, skipping\n", model_path); return 77; }
    ggml_time_init();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);

    const std::string apath = (std::filesystem::temp_directory_path() / "test-spec-gated-lora-adapter.gguf").string();
    if (!write_gated_adapter(model, apath)) { fprintf(stderr, "cannot write the adapter\n"); return 1; }
    llama_adapter_lora * ad = llama_adapter_lora_init(model, apath.c_str());
    if (!ad) { fprintf(stderr, "cannot load the synthetic adapter\n"); return 1; }
    CHECK(llama_adapter_lora_is_draft_gated(ad), "the synthetic adapter must load as draft_rows gated");
    float scale = 1.0f;

    std::vector<llama_token> prompt;
    {
        const std::string text = "Once upon a time there was a little girl named Lily. She loved to play in the park with her friends.";
        prompt.resize(text.size() + 8);
        const int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), prompt.data(), (int) prompt.size(), true, false);
        prompt.resize(std::max(n, 0));
    }
    if (prompt.size() < 4) { fprintf(stderr, "tokenization failed\n"); return 1; }
    const llama_pos pos0 = (llama_pos) prompt.size();
    const int N = 6;

    // the id_last: the base model's next token after the prompt
    llama_context * c_base = mk_ctx(model);
    prefill(c_base, prompt);
    const llama_token id_last = argmax(llama_get_logits_ith(c_base, (int) prompt.size() - 1), nv);

    // --- A: draft chains ---------------------------------------------------------------------------------------------------------
    llama_context * t0 = mk_ctx(model);
    llama_context * d0 = mk_ctx(model);
    prefill(t0, prompt);
    const llama_tokens D_base = draft_with(t0, d0, prompt, id_last, N);

    llama_context * t1 = mk_ctx(model);
    llama_context * d1 = mk_ctx(model);
    llama_set_adapters_lora(t1, &ad, 1, &scale);   // verify side: attached, never given a draft_start
    llama_set_adapters_lora(d1, &ad, 1, &scale);   // draft side
    prefill(t1, prompt);
    const llama_tokens D_gated = draft_with(t1, d1, prompt, id_last, N);

    // the reference chain by hand: gated rows start at pos0 + 1
    llama_context * cr = mk_ctx(model);
    llama_set_adapters_lora(cr, &ad, 1, &scale);
    prefill(cr, prompt);
    llama_set_draft_start(cr, 0, pos0 + 1);
    llama_tokens D_ref;
    {
        llama_batch b = llama_batch_init(8, 0, 1);
        llama_token t = id_last;
        for (int i = 0; i < N; ++i) {
            common_batch_clear(b);
            common_batch_add(b, t, pos0 + i, { 0 }, true);
            CHECK(llama_decode(cr, b) == 0);
            t = argmax(llama_get_logits_ith(cr, 0), nv);
            D_ref.push_back(t);
        }
        llama_batch_free(b);
    }
    llama_set_draft_start(cr, 0, -1);

    printf("base draft :");  for (auto t : D_base)  { printf(" %d", t); }  printf("\n");
    printf("gated draft:");  for (auto t : D_gated) { printf(" %d", t); }  printf("\n");
    printf("ref   draft:");  for (auto t : D_ref)   { printf(" %d", t); }  printf("\n");

    CHECK((int) D_base.size() == N && (int) D_gated.size() == N, "draft lengths base %zu gated %zu", D_base.size(), D_gated.size());
    if (!D_base.empty() && !D_gated.empty()) {
        CHECK(D_gated[0] == D_base[0], "the first drafted token comes from id_last (gate 0) and must equal the base's");
    }
    CHECK(D_gated != D_base, "A: the gated adapter on the draft context must change the drafted chain (draft rows are gated)");
    CHECK(D_gated == D_ref, "A: the drafted chain must equal the by-hand chain with draft_start = pos0 + 1");

    // --- D: the adapter has an effect on gated rows -----------------------------------------------------------------------------
    {
        llama_context * cg = mk_ctx(model);
        llama_set_adapters_lora(cg, &ad, 1, &scale);
        prefill(cg, prompt);
        llama_set_draft_start(cg, 0, pos0 + 1);
        const std::vector<float> lg = decode_rows(cg, nv, id_last, D_base, pos0);
        llama_context * cb = mk_ctx(model);
        prefill(cb, prompt);
        const std::vector<float> lb = decode_rows(cb, nv, id_last, D_base, pos0);
        CHECK(!lg.empty() && lg.size() == lb.size());
        if (lg.size() == lb.size() && !lg.empty()) {
            const float d_first = max_abs_diff(lg.data(), lb.data(), nv);
            const float d_rest  = max_abs_diff(lg.data() + nv, lb.data() + nv, (int) (lg.size() - nv));
            printf("D: id_last row diff %.6f, draft rows diff %.6f\n", d_first, d_rest);
            CHECK(d_first < 1e-3f, "D: row of id_last (pos0) must stay base even with draft_start = pos0 + 1 (diff %f)", d_first);
            CHECK(d_rest > 1e-2f, "D: the gated draft rows must differ from base (diff %f): a zero-effect adapter proves nothing", d_rest);
        }
        llama_free(cg);
        llama_free(cb);
    }

    // --- B: verify rows on a context that carries the adapter but never a draft_start ------------------------------------------------
    {
        llama_context * cv = mk_ctx(model);
        llama_set_adapters_lora(cv, &ad, 1, &scale);
        prefill(cv, prompt);
        const std::vector<float> lv = decode_rows(cv, nv, id_last, D_gated, pos0);
        llama_context * cb = mk_ctx(model);
        prefill(cb, prompt);
        const std::vector<float> lb = decode_rows(cb, nv, id_last, D_gated, pos0);
        CHECK(!lv.empty() && lv.size() == lb.size());
        if (lv.size() == lb.size() && !lv.empty()) {
            const float d = max_abs_diff(lv.data(), lb.data(), (int) lv.size());
            printf("B: verify rows (adapter attached, no draft_start) vs base: max diff %.6f\n", d);
            CHECK(d < 1e-3f, "B: verify rows must be gate 0 = base (diff %f)", d);
        }
        llama_free(cv);
        llama_free(cb);
    }

    // --- C: the gate is cleared after the draft call: the draft context (adapter attached) now decodes as base ------------------------
    {
        CHECK(llama_memory_seq_rm(llama_get_memory(d1), 0, pos0, -1));
        const std::vector<float> lc = decode_rows(d1, nv, id_last, D_gated, pos0);
        const std::vector<float> lb = decode_rows(c_base, nv, id_last, D_gated, pos0);
        CHECK(!lc.empty() && lc.size() == lb.size());
        if (lc.size() == lb.size() && !lc.empty()) {
            const float d = max_abs_diff(lc.data(), lb.data(), (int) lc.size());
            printf("C: draft context after the draft call vs base: max diff %.6f\n", d);
            CHECK(d < 1e-3f, "C: a draft_start left set would gate process()/accepted rows (diff %f)", d);
        }
    }

    for (llama_context * c : { c_base, t0, d0, t1, d1, cr }) { llama_free(c); }
    llama_adapter_lora_free(ad);
    llama_model_free(model);
    std::remove(apath.c_str());
    llama_backend_free();
    if (n_fail == 0) { printf("test-spec-gated-lora: OK\n"); }
    return n_fail == 0 ? 0 : 1;
}
