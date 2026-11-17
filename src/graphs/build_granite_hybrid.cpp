#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

//
// GraniteHybrid / Granite 4.0-H (IBM): MAMBA2 layers interleaved with conventional attention,
// on top of the Granite scale family.
//
// granite-4.0-h-350m is 32 blocks: 28 mamba2 + 4 attention. head_count_kv is written per layer
// and is 0 on the mamba2 layers, so the array itself says which is which -- the convention
// bailingmoe3, kimi-k3 and lfm2 already use in this fork, here applied to a fourth mechanism.
//
// WHY THIS RUNS ON A MAMBA1 KERNEL, which is the whole reason the port is small.
//
// This fork's ggml_ssm_scan is mamba1-shaped: 8 arguments, no group or ids argument. mamba2
// differs from mamba1 in three ways that matter, and only one of them is a problem:
//
//   1. B and C are shared across a GROUP of channels rather than being per-channel. The file
//      writes ssm.group_count = 1, and with a single group B and C are [d_state, n_tokens]
//      shared across every channel -- which is exactly mamba1's layout. No problem.
//   2. A, D and dt are per-HEAD scalars rather than per-channel vectors. The kernel wants them
//      per-channel. Solved by materialising the broadcast in the graph: each head's scalar is
//      repeated across that head's head_dim channels. Cheap here (A becomes 128x1536 per layer).
//   3. The recurrence itself. It is the same. ggml.c:23818-23832 computes
//          dt_soft_plus = softplus(dt[i1])
//          state        = s0[i]*expf(dt_soft_plus * A[i]) + B[i0]*(x[i1]*dt_soft_plus)
//          y[i1]        = sum_i0(state * C[i0])
//      which IS mamba2's SSD recurrence once A is a broadcast per-head scalar.
//
// So no new kernel. What the kernel does NOT do, and this graph therefore must: add the dt bias
// BEFORE the softplus (the kernel softplusses raw dt), the D skip connection, the z gate, the
// ssm_norm, the conv bias and the SiLU after the convolution.
//
// A IS STORED ALREADY NEGATIVE. Measured from the file, not assumed: blk.0.ssm_a is 48/48
// negative, min -1.9427, max -0.6295, mean -1.0008. There is no A = -exp(A_log) step here. That
// transform appears in some mamba2 conversions and applying it as well would flip the sign of
// the decay -- a graph that runs and diverges rather than one that fails.
//
// STATE LAYOUT. Both states live in kv_self.s_l[il], sized by hparams.n_embd_v_s(), packed as
//     [ conv state: (d_conv-1) * conv_dim ][ ssm state: d_state * d_inner ]
// and indexed by SEQUENCE slot. build_mamba.cpp uses kv_head/n_kv, which are TOKEN-space offsets
// into the attention cache; that is correct there only because every mamba layer is recurrent, so
// the KV cache IS the sequence-slot array. In a hybrid the two are different axes and using the
// token offset runs off the end -- measured on lfm2 as a view-bounds assert at ggml.c:5477. Each
// contiguous same-sequence run of the ubatch uses ITS sequence's row (recurrent_runs), exposed as a
// single-slot view so sq->ne[0] == s->ne[2] == 1 still holds, and starts from zero state at pos 0.
//
// GATE-THEN-NORM, and why that specific order.
// RESEARCH-DONE: the reference implementation is state-spaces/mamba, whose gated norm lives in
// mamba_ssm/ops/triton/layernorm_gated.py. It is not a fixed order -- RMSNormGated carries an
// explicit `norm_before_gate` flag, and Mamba-2's mixer constructs it with norm_before_gate=False,
// i.e. compute x = x * silu(z) FIRST and RMS-normalise the gated result. That is what this file
// does. See also arXiv:2405.21060 (Dao & Gu, "Transformers are SSMs: Generalized Models and
// Efficient Algorithms Through Structured State Space Duality"), which places the gated norm on
// the SSD output before the out-projection.
// Because that flag exists, the other order is a real configuration rather than a mistake, and
// both settings produce a graph that runs. If generation is fluent but the model underperforms
// its published behaviour, swapping those two lines is the cheapest thing to test first.
//

ggml_cgraph * llm_build_context::build_granite_hybrid() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t d_conv     = hparams.ssm_d_conv;         // 4
    const int64_t d_inner    = hparams.ssm_d_inner;        // 1536
    const int64_t d_state    = hparams.ssm_d_state;        // 128
    const int64_t n_group    = hparams.ssm_n_group;        // 1
    const int64_t n_ssm_head = hparams.ssm_dt_rank;        // 48
    const int64_t head_dim   = n_ssm_head > 0 ? d_inner / n_ssm_head : 0;   // 32
    const int64_t conv_dim   = d_inner + 2*n_group*d_state;                 // 1792
    const int64_t n_embd_head = hparams.n_embd_head_k(0);                   // 64

    // Granite writes an explicit attention multiplier (0.015625 = 1/64 = 1/head_dim), which is
    // NOT 1/sqrt(head_dim) = 0.125. Use it when present rather than deriving one.
    const float kq_scale = hparams.f_attention_scale != 0.0f
                         ? hparams.f_attention_scale
                         : 1.0f / sqrtf(float(n_embd_head));

    GGML_ASSERT(n_group == 1 && "granitehybrid: only ssm.group_count == 1 is implemented");
    GGML_ASSERT(head_dim > 0 && d_inner == n_ssm_head * head_dim);
    GGML_ASSERT(d_conv > 1);

    // applies hparams.f_embedding_scale for us (llama-build-context.cpp:898)
    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    // Only build the position input when RoPE consumes it (see the rope_finetuned note in the
    // attention branch). An input tensor no node reads is never allocated, and set_inputs then
    // asserts "tensor buffer not set" (ggml-backend.cpp:344) -- measured 2026-09-25.
    // llama.cpp's set_inputs skips a null lctx.inp_pos.
    ggml_tensor * inp_pos = hparams.rope_finetuned ? build_inp_pos() : nullptr;
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();

    // the per-token sequence-slot input the hybrid recurrent path uses; build_inp_s_mask /
    // build_inp_s_seq are the MAMBA-family token-space inputs and are deliberately not used
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    const int64_t n_state_kv   = 1;
    const int64_t conv_state_n = (d_conv - 1) * conv_dim;   // 5376 floats, then the ssm state

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * prev = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(cur, "operator_norm", il);

        if (hparams.is_recurrent(il)) {
            // ---- mamba2 -------------------------------------------------------------------
            GGML_ASSERT(lctx.inp_s_seq_qnext != nullptr);
            GGML_ASSERT(kv_self.s_l[il] != nullptr);

            const size_t ss        = ggml_element_size(kv_self.s_l[il]);
            const size_t row_bytes = kv_self.s_l[il]->nb[1];

            // fused in-projection: [ z | x B C | dt ] along ne0
            ggml_tensor * zxbcdt = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_in, cur);
            cb(zxbcdt, "ssm_in", il);
            const size_t es = ggml_element_size(zxbcdt);

            // ONE STATE ROW PER SEQUENCE -- see build_lfm2.cpp for the full note. Row r of s_l[il]
            // holds sequence r's [conv | ssm] state; each contiguous same-sequence run of the
            // ubatch reads and writes its own row, and starts from zero state at pos == 0.
            // FIXED 2026-09-26: this block used row 0 for every sequence and never zeroed it, so
            // under llama-server one request's state leaked into the next and -np > 1 shared it.
            ggml_tensor * y_all = nullptr;

            // Fused gate + norm (GGML_OP_GATED_GROUP_RMS_NORM, n_group 1): decided once per layer
            // from a probe node built on this layer's shape; a layer whose device answers
            // supports_op == false keeps the unfused mul/silu/cont + fused_rms_norm chain below.
            // Per-run is exact: both the gate and the RMS are per token.
            bool fuse_gated_norm = cparams.fused_gated_norm && model.layers[il].ssm_norm &&
                                   model.layers[il].ssm_norm->type == GGML_TYPE_F32 &&
                                   ggml_nelements(model.layers[il].ssm_norm) == d_inner &&
                                   zxbcdt->type == GGML_TYPE_F32;
            if (fuse_gated_norm) {
                ggml_backend_dev_t dev = model.dev_layer(il);
                if (dev != nullptr) {
                    ggml_tensor * probe_y = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_inner, 1);
                    ggml_tensor * probe_z = ggml_view_2d(ctx0, zxbcdt, d_inner, 1, zxbcdt->nb[1], 0);
                    ggml_tensor * probe   = ggml_gated_group_rms_norm(ctx0, probe_y, probe_z,
                            model.layers[il].ssm_norm, hparams.f_norm_rms_eps, 1);
                    fuse_gated_norm = ggml_backend_dev_supports_op(dev, probe);
                }
            }

            for (const auto & run : recurrent_runs()) {
                const int64_t n_tok   = run.len;
                const size_t  row_off = (size_t) run.seq * row_bytes;
                const size_t  tok_off = (size_t) run.start * zxbcdt->nb[1];

                ggml_tensor * conv_states = ggml_view_3d(ctx0, kv_self.s_l[il],
                        d_conv - 1, conv_dim, n_state_kv,
                        (d_conv - 1) * ss,
                        (d_conv - 1) * conv_dim * ss,
                        row_off);
                if (run.reset) {
                    conv_states = ggml_scale(ctx0, conv_states, 0.0f);
                }
                // (the ssm state is viewed further down, in the 4-D layout the scan takes)

                ggml_tensor * z   = ggml_view_2d(ctx0, zxbcdt, d_inner,    n_tok, zxbcdt->nb[1], tok_off);
                ggml_tensor * xBC = ggml_view_2d(ctx0, zxbcdt, conv_dim,   n_tok, zxbcdt->nb[1],
                                                 tok_off + es * d_inner);
                ggml_tensor * dt  = ggml_view_2d(ctx0, zxbcdt, n_ssm_head, n_tok, zxbcdt->nb[1],
                                                 tok_off + es * (d_inner + conv_dim));
                ggml_tensor * sq_r = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, n_tok,
                                                  lctx.inp_s_seq_qnext->nb[1],
                                                  (size_t) run.start * lctx.inp_s_seq_qnext->nb[1]);

                // ---- short causal convolution over x, B and C together ----
                // this fork's ggml_ssm_conv takes the carried state SEPARATELY and returns the new
                // state packed into the tail of the result; mainline's 3-arg form pre-concatenates
                ggml_tensor * x_conv = ggml_ssm_conv(ctx0, conv_states, ggml_cont(ctx0, xBC),
                        model.layers[il].ssm_conv1d, sq_r, nullptr);
                const size_t ces = ggml_element_size(x_conv);

                // the result packs the output (conv_dim*n_tok) then the state
                // (d_conv*conv_dim*n_kv); the new carried state is the last d_conv-1 of each
                // d_conv-length group, hence the offset of one and the stride of d_conv
                ggml_build_forward_expand(gf,
                        ggml_cpy(ctx0,
                            ggml_view_2d(ctx0, x_conv, d_conv - 1, conv_dim * n_state_kv,
                                         d_conv * ces, (1 + conv_dim * n_tok) * ces),
                            ggml_view_1d(ctx0, kv_self.s_l[il],
                                         (d_conv - 1) * conv_dim * n_state_kv, row_off)));

                ggml_tensor * xBC_c = ggml_view_2d(ctx0, x_conv, conv_dim, n_tok,
                                                   conv_dim * ces, 0);
                xBC_c = ggml_add(ctx0, xBC_c, model.layers[il].ssm_conv1d_b);
                xBC_c = ggml_silu(ctx0, xBC_c);
                cb(xBC_c, "ssm_conv", il);

                const size_t cs = ggml_element_size(xBC_c);
                ggml_tensor * x = ggml_view_2d(ctx0, xBC_c, d_inner,           n_tok,
                                               xBC_c->nb[1], 0);
                ggml_tensor * B = ggml_view_2d(ctx0, xBC_c, n_group * d_state, n_tok,
                                               xBC_c->nb[1], cs * d_inner);
                ggml_tensor * C = ggml_view_2d(ctx0, xBC_c, n_group * d_state, n_tok,
                                               xBC_c->nb[1], cs * (d_inner + n_group * d_state));
                x = ggml_cont(ctx0, x);

                // ---- the scan, in the layout the kernel implements (2026-09-25) ----
                // ggml_ssm_scan is now the Mamba-2-capable upstream kernel (ggml-cpu/ops.cpp):
                //     s {d_state, head_dim, n_head, n_seqs}   x  {head_dim, n_head, n_tok, n_seqs}
                //     dt {n_head, n_tok, n_seqs}               A  {1, n_head}  (scalar decay per head)
                //     B/C {d_state, n_group, n_tok, n_seqs}    ids {n_seqs}
                // so the per-channel broadcasts of dt and A this graph used to build for the old
                // Mamba-1-shaped kernel are gone: the kernel takes Mamba-2 natively. The memory is the
                // same -- d_inner is head-major (channel = h*head_dim + j), which is exactly the
                // kernel's ii = j + h*head_dim -- so the state view below reinterprets the same bytes
                // the old {d_state, d_inner} view addressed, and the stores after the scan are unchanged.
                // The old call failed the constructor at ggml.c GGML_ASSERT(s->ne[2] == n_head), so
                // this path had never run. One sequence per run: n_seqs = 1, that sequence's state row.
                //
                // dt: the kernel softplusses whatever it is given, so the BIAS is added here first
                ggml_tensor * dt_b = ggml_add(ctx0, ggml_cont(ctx0, dt), model.layers[il].ssm_dt_b);
                ggml_tensor * dt_s = ggml_reshape_3d(ctx0, dt_b, n_ssm_head, n_tok, 1);
                // A: one scalar per head, already negative in the file
                ggml_tensor * A_s  = ggml_reshape_2d(ctx0, model.layers[il].ssm_a, 1, n_ssm_head);
                ggml_tensor * x_s  = ggml_reshape_4d(ctx0, x, head_dim, n_ssm_head, n_tok, 1);
                ggml_tensor * B_s  = ggml_reshape_4d(ctx0, ggml_cont(ctx0, B), d_state, n_group, n_tok, 1);
                ggml_tensor * C_s  = ggml_reshape_4d(ctx0, ggml_cont(ctx0, C), d_state, n_group, n_tok, 1);
                ggml_tensor * s_s  = ggml_view_4d(ctx0, kv_self.s_l[il],
                        d_state, head_dim, n_ssm_head, n_state_kv,
                        d_state * ss, d_state * head_dim * ss, d_state * d_inner * ss,
                        row_off + conv_state_n * ss);
                if (run.reset) {
                    s_s = ggml_scale(ctx0, s_s, 0.0f);
                }
                // ids = {n_seqs} = {1}: one element of inp_s_seq_qnext (all zeros) -- index 0 of the
                // single state row s_s exposes
                ggml_tensor * ids  = ggml_view_1d(ctx0, lctx.inp_s_seq_qnext, 1,
                                                  (size_t) run.start * lctx.inp_s_seq_qnext->nb[1]);

                // D: [1, n_ssm_head] -> [d_inner]
                ggml_tensor d_rep;
                d_rep.ne[0] = head_dim; d_rep.ne[1] = n_ssm_head;
                d_rep.ne[2] = 1; d_rep.ne[3] = 1;
                ggml_tensor * D_e = ggml_repeat(ctx0,
                        ggml_reshape_2d(ctx0, model.layers[il].ssm_d, 1, n_ssm_head), &d_rep);
                D_e = ggml_reshape_1d(ctx0, D_e, d_inner);

                // K = 1: single-chunk scan (the scalar-A path requires it). y occupies the first
                // d_inner*n_tok floats of the result and the new state the next d_state*d_inner,
                // the same packing as before, so the stores below are unchanged.
                ggml_tensor * y_ssm = ggml_ssm_scan(ctx0, s_s, x_s, dt_s, A_s, B_s, C_s, ids, /*K=*/1);
                const size_t ys = ggml_element_size(y_ssm);

                // second region of the result is the new ssm state; it goes after the conv state
                ggml_build_forward_expand(gf,
                        ggml_cpy(ctx0,
                            ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_state_kv,
                                         d_inner * n_tok * ys),
                            ggml_view_1d(ctx0, kv_self.s_l[il], d_state * d_inner * n_state_kv,
                                         row_off + conv_state_n * ss)));

                ggml_tensor * y = ggml_view_2d(ctx0, y_ssm, d_inner, n_tok, d_inner * ys, 0);

                // D skip connection, per head broadcast to that head's channels
                y = ggml_add(ctx0, y, ggml_mul(ctx0, x, D_e));

                if (fuse_gated_norm) {
                    // gate THEN norm in one node; z is read through its strided view (no cont)
                    y = ggml_gated_group_rms_norm(ctx0, y, z, model.layers[il].ssm_norm,
                            hparams.f_norm_rms_eps, 1);
                } else {
                    // gate (per run, it is elementwise) -- the norm follows on the joined runs
                    y = ggml_mul(ctx0, y, ggml_silu(ctx0, ggml_cont(ctx0, z)));
                }
                y_all = y_all ? ggml_concat(ctx0, y_all, y, 1) : y;
            }
            GGML_ASSERT(y_all != nullptr);

            // gate, THEN norm -- norm_before_gate=False, see the note at the top of this file
            ggml_tensor * y = fuse_gated_norm ? y_all
                    : llm_build_norm(ctx0, y_all, hparams, model.layers[il].ssm_norm, NULL,
                                     LLM_NORM_RMS, cb, il);
            cb(y, "ssm_y", il);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_out, y);
            cb(cur, "ssm_out", il);
        } else {
            // ---- conventional attention -----------------------------------------------------
            ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            ggml_tensor * k = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            ggml_tensor * v = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(q, "wq", il);
            cb(k, "wk", il);
            cb(v, "wv", il);

            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
            k = ggml_reshape_3d(ctx0, k, n_embd_head, hparams.n_head_kv(il), n_tokens);

            // RoPE ONLY when the file says so. Granite uses rope.scaling.finetuned as the rope
            // switch (upstream llm_build_granite_hybrid: `use_rope = hparams.rope_finetuned`), and
            // Granite-4.0-H is a NoPE model -- granite-4.0-h-350m-Q8_0.gguf carries
            // granitehybrid.rope.scaling.finetuned = False. This graph roped Q/K unconditionally.
            // MEASURED 2026-09-25, greedy on CPU vs upstream llama.cpp on the same GGUF: upstream
            // counted one..fifteen, this graph stopped after "One".
            if (hparams.rope_finetuned) {
                q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, nullptr,
                    k, v, q, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev = ggml_get_rows(ctx0, prev, inp_out_ids);
        }

        // Granite scales every residual branch before it is added back
        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        cur = ggml_add(ctx0, prev, cur);
        ggml_tensor * ffn_inp = cur;

        // dense FFN on both layer kinds; expert_count is 0 in the -h- files
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false);
        cb(cur, "ffn_out", il);

        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        cur = ggml_add(ctx0, ffn_inp, cur);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);

    // Granite divides the logits
    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }

    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);

    return gf;
}
