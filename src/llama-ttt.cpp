// In-Place TTT graph route and state layout -- see src/llama-ttt.h for the semantics and the official-code references.
// Row S69-FORK-INPLACE-TTT-2026-10-26-01.

#include "llama.h"
#include "llama-ttt.h"
#include "llama-build-context.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cfloat>

llama_ttt_layout llama_ttt_row_layout(const llama_hparams & hparams, uint32_t il) {
    llama_ttt_layout l;
    if (!hparams.is_ttt(il)) {
        return l;
    }
    const int64_t d = hparams.n_embd;
    l.n_s = (int64_t) hparams.n_ff(il) * d;
    l.n_x = d * hparams.ttt_chunk;
    l.n_t = hparams.ttt_target == 1 ? d * hparams.ttt_chunk : 0;
    return l;
}

int32_t llama_ttt_chunk_draft_cap(uint32_t chunk, int32_t pos_last) {
    if (chunk == 0) {
        return INT32_MAX;
    }
    const int32_t c = (int32_t) chunk;
    const int32_t r = ((pos_last % c) + c) % c;
    // first i >= 1 with (r + i) % c == c - 1 is i = c - 1 - r (or c when r == c - 1); the cap is one less
    return r == c - 1 ? c - 1 : c - 2 - r;
}

int32_t llama_model_ttt_chunk(const llama_model * model) {
    return model != nullptr && model->hparams.has_ttt() ? (int32_t) model->hparams.ttt_chunk : 0;
}

int32_t llama_ttt_draft_cap(const llama_model * model, llama_pos pos_last) {
    return llama_ttt_chunk_draft_cap((uint32_t) llama_model_ttt_chunk(model), pos_last);
}

// FFN of a TTT layer, residual included (the same contract as llm_build_ffn(..., add_input = true)).
//   ffn_inp  : the residual stream entering the FFN, [n_embd, n_tokens]
//   inp_embd : the token embeddings of this ubatch (the target when ttt.target = input_embed), may be null otherwise
// Returns ffn_inp + (W_down + S) h for every token of the ubatch; when the ubatch completes the chunk it also writes
// S <- S + dS (apply-then-update: this ubatch's own outputs use the OLD S, modeling_qwen3.py:141-151).
ggml_tensor * llm_build_context::build_ttt_ffn(ggml_cgraph * gf, ggml_tensor * ffn_inp, ggml_tensor * inp_embd, int il) {
    const llama_layer & layer = model.layers[il];
    const int64_t C  = hparams.ttt_chunk;
    const int64_t n  = n_tokens;
    const int64_t d  = n_embd;
    const size_t  f  = sizeof(float);
    const llama_ttt_layout lay = llama_ttt_row_layout(hparams, il);
    const int64_t n_ff_l = lay.n_s / d;

    GGML_ASSERT((size_t) il < kv_self.ttt_l.size() && kv_self.ttt_l[il] != nullptr);
    ggml_tensor * st = kv_self.ttt_l[il];
    GGML_ASSERT(st->type == GGML_TYPE_F32 && st->ne[0] == lay.row());
    GGML_ASSERT(layer.ffn_ttt_conv != nullptr);
    GGML_ASSERT(hparams.ttt_target == 0 || inp_embd != nullptr);

    // Position and sequence of this ubatch. The worst-case reserve graph has neither (llama_batch_get_one), and it
    // must size the LARGEST graph: a completing ubatch past chunk 0 whose tail holds the rest of the chunk.
    const bool reserve = batch.pos == nullptr || batch.seq_id == nullptr;
    llama_pos    p0  = 0;
    llama_seq_id seq = 0;
    int64_t      j0  = 0;
    bool complete    = false;
    bool use_s       = false;
    if (reserve) {
        j0       = n < C ? C - n : 0;
        complete = true;
        use_s    = true;
    } else {
        p0  = batch.pos[0];
        seq = batch.seq_id[0][0];
        GGML_ASSERT(p0 >= 0 && seq >= 0 && seq < st->ne[1]);
        j0       = p0 % C;
        GGML_ASSERT(j0 + n <= C && "TTT ubatch crosses a chunk boundary (llama_ttt_check_ubatch should have refused it)");
        complete = j0 + n == C;
        use_s    = p0 >= C;   // S holds every update of chunks 0 .. p0/C - 1; in chunk 0 the weight is W_down alone
    }
    const size_t row_off = (size_t) seq * st->nb[1];

    // FFN input x = ffn_norm(ffn_inp); the official target_states for hidden_states (modeling_qwen3.py:347-349)
    ggml_tensor * x = llm_build_norm(ctx0, ffn_inp, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(x, "ffn_norm", il);

    auto ffn_h = [&](ggml_tensor * in) {
        ggml_tensor * g = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate, in);
        ggml_tensor * u = llm_build_lora_mm(lctx, ctx0, layer.ffn_up,   in);
        return ggml_mul(ctx0, ggml_silu(ctx0, g), u);   // act_fn(gate_proj(x)) * up_proj(x), modeling_qwen3.py:118
    };
    ggml_tensor * h = ffn_h(x);
    cb(h, "ffn_ttt_h", il);

    ggml_tensor * S = ggml_view_2d(ctx0, st, n_ff_l, d, n_ff_l * f, row_off);
    ggml_tensor * y = llm_build_lora_mm(lctx, ctx0, layer.ffn_down, h);
    if (use_s) {
        y = ggml_add(ctx0, y, ggml_mul_mat(ctx0, S, h));
    }
    cb(y, "ffn_ttt_y", il);
    ggml_tensor * out = ggml_add(ctx0, y, ffn_inp);
    cb(out, "ffn_out", il);
    // Every read of S that produces this ubatch's output is now in the graph; the S write below comes after it.
    ggml_build_forward_expand(gf, out);

    ggml_tensor * t = hparams.ttt_target == 1 ? inp_embd : x;
    auto tail = [&](int64_t base, int64_t col0, int64_t ncol) {
        return ggml_view_2d(ctx0, st, d, ncol, d * f, row_off + (size_t) (base + col0 * d) * f);
    };

    if (!complete) {
        // the chunk is still open: keep this ubatch's inputs (and targets) for the update that completes it
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, x, tail(lay.n_s, j0, n)));
        if (lay.n_t > 0) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, t, tail(lay.n_s + lay.n_x, j0, n)));
        }
        return out;
    }

    // ---- the chunk completes in this ubatch: dS from the whole chunk (tail + this ubatch), modeling_qwen3.py:142-150
    ggml_tensor * Hc = h;
    ggml_tensor * Tc = t;
    if (j0 > 0) {
        ggml_tensor * xt = tail(lay.n_s, 0, j0);
        Hc = ggml_concat(ctx0, ffn_h(xt), h, 1);
        Tc = ggml_concat(ctx0, lay.n_t > 0 ? tail(lay.n_s + lay.n_x, 0, j0) : xt, t, 1);
    }
    const int64_t Cc = j0 + n;   // == C, except in the reserve graph when n > C

    // depthwise conv over the chunk alone, zero outside it:
    //   symmetric (official, Conv1d padding=2): tc[:, j] = sum_k w[:, k] * T[:, j - 2 + k]
    //   causal    (paper text, opt-in):         tc[:, j] = sum_k w[:, k] * T[:, j - 4 + k]
    ggml_tensor * w = layer.ffn_ttt_conv;   // ne {n_embd, 5}, row k = tap k
    auto tap = [&](int k) { return ggml_view_2d(ctx0, w, d, 1, w->nb[1], (size_t) k * w->nb[1]); };
    const int s0 = hparams.ttt_conv_padding == 1 ? -4 : -2;
    const int kc = -s0;                    // the tap with shift 0 covers every column
    ggml_tensor * tc = ggml_mul(ctx0, Tc, tap(kc));
    for (int k = 0; k < 5; ++k) {
        if (k == kc) {
            continue;
        }
        const int64_t s   = s0 + k;        // tc[:, j] += w[:, k] * T[:, j + s] for 0 <= j + s < Cc
        const int64_t len = Cc - (s < 0 ? -s : s);
        if (len <= 0) {
            continue;
        }
        ggml_tensor * src  = ggml_view_2d(ctx0, Tc, d, len, Tc->nb[1], (size_t) std::max<int64_t>(0, s) * Tc->nb[1]);
        ggml_tensor * term = ggml_mul(ctx0, src, tap(k));
        tc = ggml_acc(ctx0, tc, term, tc->nb[1], tc->nb[2], tc->nb[3], (size_t) std::max<int64_t>(0, -s) * tc->nb[1]);
    }
    cb(tc, "ffn_ttt_tconv", il);

    // Z = conv(T) @ P (the contraction "c d, d e"); ffn_ttt_proj holds P^T, so mul_mat gives z[:, c] = Z[c, :]
    ggml_tensor * z = layer.ffn_ttt_proj ? ggml_mul_mat(ctx0, layer.ffn_ttt_proj, tc) : tc;

    // dS[h, e] = lr * sum_c H[h, c] * Z[e, c]  -- ggml layout of dW[e, h] = lr * sum_c Z[c, e] H[c, h]
    ggml_tensor * dS = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, Hc)), ggml_cont(ctx0, ggml_transpose(ctx0, z)));
    dS = ggml_scale(ctx0, dS, hparams.ttt_lr);
    if (hparams.ttt_clip_tau > 0.0f) {
        // paper variant: dW <- tau * dW / ||dW||_F when ||dW||_F > tau (divide by max(1, ||dW||_F / tau))
        ggml_tensor * nrm = ggml_sqrt(ctx0, ggml_sum(ctx0, ggml_sqr(ctx0, dS)));
        dS = ggml_div(ctx0, dS, ggml_clamp(ctx0, ggml_scale(ctx0, nrm, 1.0f / hparams.ttt_clip_tau), 1.0f, FLT_MAX));
    }
    cb(dS, "ffn_ttt_dS", il);

    ggml_tensor * S_new = use_s ? ggml_add(ctx0, S, dS) : dS;
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, S_new, S));
    return out;
}
