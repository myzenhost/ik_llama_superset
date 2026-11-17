// test-prism-hdmd-fill: the fork has two Prism Hadamard implementations, one per graph path. The legacy builder applies
// model.hadamard_map through ggml_hadamard (the op computes H*x for each block); the class-path graphs (llm_graph_context) apply
// model.hdmd.rot / model.hdmd.inv, upstream's layout (a block x block matrix tensor per weight, matmul'd). Nothing filled hdmd, so a
// Prism GGUF on a class-path arch ran its folded weights unrotated. llama_hdmd_fill makes hdmd from the loader's recorded names, sign
// data and flags. Row S69-ENGREV-UPSTREAM-MERGE-PRISM-HADAMARD-TWO-IMPLEMENTATIONS-2026-11-04-01.
// GATE-CLAIM-OK: documentation comment, no resource is declared unavailable here.
//
// Checks on a synthetic model (host-memory tensors, no GGUF):
//   F1 rot holds every folded weight (keyed by the weight tensor), inv holds the latent token table
//   F2 each rot matrix equals the LEGACY matrix element-wise: the legacy op ggml_hadamard run on an identity (H is symmetric, so
//      output row r = H[r]), tolerance 1e-6
//   F3 rot * inv = identity within 1e-5 (H*H = I)
//   F4 hdmd sign vectors equal the legacy hadamard_map sign tensors element-wise
//   F5 one matrix per block size shared by every weight of that size (no per-weight copies)
//   F6 GDN value-head permutation geometry on .ssm_out. (perm_hd/nk/rep) matches the legacy fill formula
//   F7 inconsistent input throws (token_embd missing for tied_output)
// Usage: test-prism-hdmd-fill            -> fills hdmd with llama_hdmd_fill, expects PASS
//        test-prism-hdmd-fill --no-fill  -> the same checks WITHOUT the fill (the state before this change), expects FAIL
#include "llama-model.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

static std::vector<float> get_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// legacy matrix: ggml_hadamard(identity [n, n], n) computed by the ggml host backend
static std::vector<float> legacy_matrix(int n) {
    ggml_init_params p = { 16u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    float * d = (float *) id->data;
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) d[(size_t) r * n + c] = r == c ? 1.0f : 0.0f;
    ggml_tensor * out = ggml_hadamard(ctx, id, n);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> m((float *) out->data, (float *) out->data + (size_t) n * n);
    ggml_free(ctx);
    return m;
}

static ggml_tensor * new_host_tensor(llama_model & model, int64_t ne0, int64_t ne1, const char * name, const std::vector<float> * data) {
    ggml_init_params p = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (data) ggml_backend_tensor_set(t, data->data(), 0, data->size() * sizeof(float));
    model.ctxs.push_back(ctx);
    model.bufs.push_back(buf);
    return t;
}

int main(int argc, char ** argv) {
    const bool do_fill = !(argc > 1 && strcmp(argv[1], "--no-fill") == 0);
    printf("test-prism-hdmd-fill (%s)\n", do_fill ? "with fill" : "NO fill: pre-change state, expected to FAIL");

    const uint32_t block = 64;
    llama_model model;
    model.hparams.ssm_dt_rank = 8;
    model.hparams.ssm_n_group = 2;

    // folded weights: ne0 = input width. two widths so two sign vectors; ssm_out takes the GDN permutation
    struct W { const char * name; int64_t ne0; };
    const W ws[] = { { "blk.0.attn_q.weight", 128 }, { "blk.0.ffn_down.weight", 256 }, { "blk.0.ssm_out.weight", 256 } };
    std::map<int64_t, std::vector<float>> signs;
    for (const int64_t w : { 128, 256 }) {
        std::vector<float> s(w);
        for (int64_t i = 0; i < w; ++i) s[i] = ((i * 7 + w) % 3 == 0) ? -1.0f : 1.0f;
        signs[w] = s;
    }
    std::map<int64_t, ggml_tensor *> legacy_signs;
    for (const auto & [w, s] : signs) legacy_signs[w] = new_host_tensor(model, w, 0, "legacy.signs", &s);

    std::vector<ggml_tensor *> wt;
    for (const W & w : ws) {
        ggml_tensor * t = new_host_tensor(model, w.ne0, 4, w.name, nullptr);
        model.tensors_by_name.emplace_back(w.name, t);
        wt.push_back(t);
        // legacy fill (what llm_load_hadamard writes)
        llama_hadamard_transform & lt = model.hadamard_map[w.name];
        lt.block_size = block;
        lt.signs = legacy_signs[w.ne0];
    }
    ggml_tensor * emb = new_host_tensor(model, 128, 4, "token_embd.weight", nullptr);
    model.tok_embd = emb;
    model.tensors_by_name.emplace_back("token_embd.weight", emb);
    { llama_hadamard_transform & lt = model.hadamard_map["token_embd.weight"]; lt.block_size = block; lt.signs = legacy_signs[128]; }

    // what the loader records in hdmd from the same metadata
    for (const W & w : ws) model.hdmd.weight_blocks.emplace(w.name, block);
    model.hdmd.inverse_blocks.emplace("token_embd.weight", block);
    for (const auto & [w, s] : signs) {
        auto & v = model.hdmd.sign_data[(uint32_t) w];
        for (float f : s) v.push_back((int32_t) f);
    }
    model.hdmd.gdn_v_grouped = true;

    if (do_fill) {
        const size_t n = llama_hdmd_fill(model);
        printf("  filled %zu transform(s)\n", n);
    }

    // F1
    check(model.hdmd.rot.size() == 3 && model.hdmd.inv.size() == 1, "F1 rot has 3 folded weights, inv has the token table");
    bool keyed = true;
    for (ggml_tensor * t : wt) keyed = keyed && model.hdmd.rot.count(t) == 1;
    check(keyed && model.hdmd.inv.count(emb) == 1, "F1 transforms are keyed by the model's weight tensors");

    // F2 / F3 / F4 / F6 over every transform
    const std::vector<float> H = legacy_matrix((int) block);
    bool f2 = !model.hdmd.rot.empty(), f3 = !model.hdmd.rot.empty(), f4 = !model.hdmd.rot.empty(), f6 = !model.hdmd.rot.empty();
    for (ggml_tensor * t : wt) {
        const auto it = model.hdmd.rot.find(t);
        if (it == model.hdmd.rot.end()) { f2 = f3 = f4 = f6 = false; continue; }
        const llama_hdmd_transform & tr = it->second;
        const std::vector<float> R = get_f32(tr.rot);
        f2 = f2 && tr.rot->ne[0] == (int64_t) block && tr.rot->ne[1] == (int64_t) block && R.size() == H.size();
        for (size_t i = 0; f2 && i < R.size(); ++i) f2 = std::fabs(R[i] - H[i]) <= 1e-6f;
        const auto inv = model.hdmd.inv.find(emb);
        if (inv == model.hdmd.inv.end()) { f3 = false; } else {
            const std::vector<float> I = get_f32(inv->second.rot);
            for (uint32_t r = 0; f3 && r < block; ++r) for (uint32_t c = 0; f3 && c < block; ++c) {
                double s = 0;
                for (uint32_t k = 0; k < block; ++k) s += (double) R[(size_t) r * block + k] * I[(size_t) k * block + c];
                f3 = std::fabs(s - (r == c ? 1.0 : 0.0)) <= 1e-5;
            }
        }
        const ggml_tensor * ls = model.hadamard_map.at(t->name).signs;
        f4 = f4 && tr.signs != nullptr && tr.signs->ne[0] == ls->ne[0];
        if (f4) { const auto a = get_f32(tr.signs), b = get_f32(ls); for (size_t i = 0; i < a.size(); ++i) f4 = f4 && a[i] == b[i]; }
        const bool ssm = strstr(t->name, ".ssm_out.") != nullptr;
        if (ssm) f6 = f6 && tr.perm_hd == t->ne[0] / 8 && tr.perm_nk == 2 && tr.perm_rep == 4;
        else     f6 = f6 && tr.perm_rep == 0;
    }
    check(f2, "F2 rot matrix == legacy ggml_hadamard matrix element-wise (1e-6)");
    check(f3, "F3 rot * inv == identity (1e-5)");
    check(f4, "F4 hdmd signs == legacy hadamard_map signs element-wise");
    bool f5 = !model.hdmd.rot.empty();
    for (ggml_tensor * t : wt) { const auto it = model.hdmd.rot.find(t); f5 = f5 && it != model.hdmd.rot.end() && it->second.rot == model.hdmd.rot.begin()->second.rot; }
    check(f5, "F5 one matrix tensor shared by all weights of the block size");
    check(f6, "F6 GDN value-head permutation geometry on ssm_out only");

    // F7: tied_output without a latent token table in inverse_blocks must throw, not run unrotated
    {
        llama_model m2;
        m2.hdmd.weight_blocks.emplace("blk.0.attn_q.weight", block);
        m2.hdmd.tied_output = true;
        bool threw = false;
        if (do_fill) { try { llama_hdmd_fill(m2); } catch (const std::runtime_error &) { threw = true; } }
        check(threw, "F7 tied_output without a latent token embedding throws");
    }

    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
