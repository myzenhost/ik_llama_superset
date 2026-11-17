// test-ple-state-bytes: the qwen4exp PLE convolution history rides in the legacy kv_self state row; the saved sequence state must carry exactly
// those rows, restore must reproduce the next-token logits, and a seq_cp must continue like a fresh run.
// Row S69-ENGREV-UPSTREAM-MERGE-PLE-STATE-TWO-LAYOUTS-2026-11-04-01.
//
// Layouts at d7e438595: the fork packs the PLE history into the tail of kv_self.s_l[il] (llama.cpp:1805 recurrent layer, :1981 attention layer);
// upstream's llama_memory_recurrent keeps a separate p_l row (llama-memory-recurrent.cpp:112), but qwen4exp has no class dispatch
// (llama-model-create.cpp), so that memory module is metadata-only and the public state calls route to kv_self (llama-context.cpp:6958).
//
// Checks (each prints ok/FAIL):
//   L1 per layer: kv_self.s_l row width == n_embd_v_s + n_embd_ple_conv (recurrent) or n_embd_attn_layer_state (attention PLE layer)
//   L2 hparams.ple_conv_row_width() == n_embd_ple_conv(il) on every PLE layer (the two widths are one number)
//   L3 recurrent_state_layout_for(il): ple_offset is where the history really starts in the row (0 on an attention layer: the row holds only it)
//   L4 the memory module holds no bytes (memory_breakdown() == 0): the second layout is dead storage, not a second copy
//   B1 llama_state_seq_get_size == bytes written, and the stream holds every live s_l row back to back (header triple + payload, adjacent)
//   B2 the PLE tail of the stream payload == the PLE tail of the live tensor, and is not all zero (the history was really captured)
//   R1 save -> restore into a fresh context -> next-token logits equal the original's
//   R2 control: the same restore with the PLE tail zeroed in the stream gives DIFFERENT logits (the bytes matter; the check can fail)
//   C1 seq_cp(0 -> 1) then decode on seq 1 == a fresh run of the same tokens; seq 0 continues unchanged
// Usage: test-ple-state-bytes <qwen4exp.gguf> (exit 77 when no model). A tiny random-weight model proves wiring only.
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_seq_max) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 32; cp.n_ubatch = 8; cp.n_seq_max = n_seq_max; cp.n_threads = 2; cp.n_threads_batch = 2;
    try { return llama_init_from_model(model, cp); } catch (const std::exception & e) { printf("    (context creation: %s)\n", e.what()); return nullptr; }
}

static llama_token tok_at(int i, int n_vocab) { return 3 + (i*37 + 5) % (n_vocab - 3); }

static bool decode(llama_context * ctx, int from, int to, llama_seq_id seq, int n_vocab) {
    llama_batch b = llama_batch_init(to - from, 0, 1);
    b.n_tokens = to - from;
    for (int i = from; i < to; ++i) {
        const int j = i - from;
        b.token[j] = tok_at(i, n_vocab); b.pos[j] = i; b.n_seq_id[j] = 1; b.seq_id[j][0] = seq; b.logits[j] = (i + 1 == to);
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(l, l + n_vocab);
}

static double max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::fmax(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_PLE_MODEL");
    if (!path) { printf("SKIP: no model (arg 1 or LLAMA_TEST_PLE_MODEL)\n"); return 77; }
    if (FILE * f = std::fopen(path, "rb")) { std::fclose(f); } else { printf("SKIP: no model at %s\n", path); return 77; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { printf("FAIL: model load\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const llama_hparams & hp = ((const llama_model *) model)->hparams;
    const int n_layer = (int) hp.n_layer;
    const int P = 12;   // prefix: 8 + 4 -> the PLE tail has to cross a ubatch boundary

    llama_context * A = make_ctx(model, 2);
    if (!A) { printf("FAIL: context\n"); return 1; }
    auto & kv = A->kv_self;

    // ---- layout map -------------------------------------------------------------------------------------------------------------------------
    printf("layout map (%s): n_layer=%d hc=%u n_embd=%u ple_conv_state=%u ple_conv_row_width=%u\n", path, n_layer,
           hp.dsv4_hc_mult, hp.n_embd, hp.ple_conv_state(), hp.ple_conv_row_width());
    int n_ple = 0;
    size_t sum_rows = 0, sum_ple_bytes = 0;
    for (int il = 0; il < n_layer; ++il) {
        const bool ple  = hp.is_ple(il);
        const bool recr = hp.is_recurrent(il);
        const ggml_tensor * s = il < (int) kv.s_l.size() ? kv.s_l[il] : nullptr;
        const uint32_t expect = recr ? hp.n_embd_v_s() + hp.n_embd_ple_conv(il) : hp.n_embd_attn_layer_state(il);
        const auto lay = hp.recurrent_state_layout_for(il);
        printf("  il=%2d ple=%d recurrent=%d s_l.ne0=%lld expect=%u layout.ple_offset=%u layout.row_width=%u\n", il, ple, recr,
               s ? (long long) s->ne[0] : -1LL, expect, lay.ple_offset, lay.row_width);
        if (s) { sum_rows += ggml_row_size(s->type, s->ne[0]); }
        if (recr || ple) {
            check(s != nullptr && (uint32_t) s->ne[0] == expect, "L1 s_l row width == recurrent/attention-state formula");
        }
        if (ple) {
            ++n_ple;
            sum_ple_bytes += (size_t) hp.n_embd_ple_conv(il) * sizeof(float);
            check(hp.ple_conv_row_width() == hp.n_embd_ple_conv(il), "L2 ple_conv_row_width() == n_embd_ple_conv(il)");
            const uint32_t want_off = recr ? hp.n_embd_v_s() : 0;
            check(s != nullptr && lay.ple_offset == want_off && lay.ple_offset + lay.ple_width == (uint32_t) s->ne[0],
                  "L3 layout.ple_offset + ple_width == the live row width (history sits at the row tail)");
        }
    }
    check(n_ple > 0, "model has a PLE layer (else this test proves nothing)");
    {
        size_t mem_bytes = 0;
        for (const auto & kvp : llama_get_memory(A)->memory_breakdown()) { mem_bytes += kvp.second; }
        check(mem_bytes == 0, "L4 memory module holds 0 bytes (metadata-only: the separate p_l row is not a second copy)");
    }

    // ---- prefix on seq 0, save --------------------------------------------------------------------------------------------------------------
    if (!decode(A, 0, P, 0, n_vocab)) { printf("FAIL: prefix decode\n"); return 1; }
    llama_synchronize(A);
    const llama_state_seq_flags flags[2] = { LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, 0 };
    std::vector<uint8_t> state;
    for (int fi = 0; fi < 2; ++fi) {
        state.assign(llama_state_seq_get_size_ext(A, 0, flags[fi]), 0);
        const size_t n = llama_state_seq_get_data_ext(A, state.data(), state.size(), 0, flags[fi]);
        printf("B1 [%s] get_size=%zu get_data=%zu sum_live_rows=%zu (PLE bytes in them: %zu) header+other=%zd\n",
               fi == 0 ? "partial" : "full", state.size(), n, sum_rows, sum_ple_bytes, (ptrdiff_t) state.size() - (ptrdiff_t) sum_rows);
        check(n == state.size() && n > 0, "B1 get_data wrote exactly get_size bytes");
    }
    // the stream: per layer [i32 type][u64 row_bytes][u32 rows] then rows*row_bytes; walk it layer by layer from the first live row
    std::vector<uint8_t> part(llama_state_seq_get_size_ext(A, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    llama_state_seq_get_data_ext(A, part.data(), part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    size_t cursor = 0, prev_end = 0;
    int prev_il = -1;
    bool adj_ok = true, tail_ok = true, tail_nonzero = false;
    size_t ple_tail_len = 0;
    for (int il = 0; il < n_layer; ++il) {
        const ggml_tensor * s = il < (int) kv.s_l.size() ? kv.s_l[il] : nullptr;
        if (!s) continue;
        const uint64_t row = ggml_row_size(s->type, s->ne[0]);
        uint8_t pat[16]; const int32_t ty = (int32_t) s->type; const uint32_t one = 1;
        memcpy(pat, &ty, 4); memcpy(pat + 4, &row, 8); memcpy(pat + 12, &one, 4);
        size_t at = (size_t) -1;
        for (size_t o = cursor; o + 16 + row <= part.size(); ++o) { if (memcmp(&part[o], pat, 16) == 0) { at = o; break; } }
        if (at == (size_t) -1) { adj_ok = false; printf("    layer %d: header triple not found in the stream\n", il); break; }
        if (prev_il >= 0 && at != prev_end + 16u * (size_t) (il - prev_il - 1)) { adj_ok = false; printf("    layer %d: not adjacent to layer %d\n", il, prev_il); }
        // payload == live tensor row of slot 0 (seq 0 maps to cell 0); compare whole row
        std::vector<uint8_t> live(row);
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(s), live.data(), 0, row);
        const uint8_t * pay = &part[at + 16];
        if (memcmp(pay, live.data(), row) != 0) { tail_ok = false; printf("    layer %d: stream payload != live row\n", il); }
        if (hp.is_ple(il)) {
            ple_tail_len = (size_t) hp.n_embd_ple_conv(il) * sizeof(float);
            for (size_t k = 0; k < ple_tail_len; ++k) { if (pay[row - ple_tail_len + k]) { tail_nonzero = true; break; } }
        }
        prev_il = il; prev_end = at + 16 + row; cursor = prev_end;
    }
    check(adj_ok, "B1 stream holds every live s_l row back to back (header triple + payload, adjacent in layer order)");
    check(tail_ok, "B1 each row payload equals the live kv_self row bytes");
    check(tail_nonzero, "B2 the PLE history tail in the stream is not all zero");

    // ---- R1/R2: restore into fresh contexts, compare the next token --------------------------------------------------------------------------
    if (!decode(A, P, P + 1, 0, n_vocab)) { printf("FAIL: ref decode\n"); return 1; }
    const std::vector<float> ref = last_logits(A, n_vocab);
    llama_context * S = make_ctx(model, 2);   // twin at exactly P tokens (A has advanced one token), source of the saved state
    decode(S, 0, P, 0, n_vocab); llama_synchronize(S);
    std::vector<uint8_t> st(llama_state_seq_get_size_ext(S, 0, 0));
    llama_state_seq_get_data_ext(S, st.data(), st.size(), 0, 0);
    {
        llama_context * B = make_ctx(model, 2);
        const size_t n = llama_state_seq_set_data_ext(B, st.data(), st.size(), 0, 0);
        check(n > 0, "R1 restore accepted");
        decode(B, P, P + 1, 0, n_vocab);
        const double d = max_diff(ref, last_logits(B, n_vocab));
        printf("       restore max|dlogit| = %.6g\n", d);
        check(d < 1e-5, "R1 save -> restore -> next-token logits equal the original");
        llama_free(B);
    }
    {
        // control: zero the PLE tail inside the full stream (same locate-by-triple walk on the full stream)
        std::vector<uint8_t> bad = st;
        bool found = false;
        size_t cur = 0;
        for (int il = 0; il < n_layer && !found; ++il) {
            const ggml_tensor * s = il < (int) kv.s_l.size() ? kv.s_l[il] : nullptr;
            if (!s || !hp.is_ple(il)) continue;
            const uint64_t row = ggml_row_size(s->type, s->ne[0]);
            uint8_t pat[16]; const int32_t ty = (int32_t) s->type; const uint32_t one = 1;
            memcpy(pat, &ty, 4); memcpy(pat + 4, &row, 8); memcpy(pat + 12, &one, 4);
            for (size_t o = cur; o + 16 + row <= bad.size(); ++o) {
                if (memcmp(&bad[o], pat, 16) == 0) {
                    const size_t tl = (size_t) hp.n_embd_ple_conv(il) * sizeof(float);
                    memset(&bad[o + 16 + row - tl], 0, tl);
                    found = true; break;
                }
            }
        }
        check(found, "R2 located the PLE tail in the full stream");
        llama_context * Cc = make_ctx(model, 2);
        llama_state_seq_set_data_ext(Cc, bad.data(), bad.size(), 0, 0);
        decode(Cc, P, P + 1, 0, n_vocab);
        const double d = max_diff(ref, last_logits(Cc, n_vocab));
        printf("       zeroed-PLE-tail restore max|dlogit| = %.6g\n", d);
        check(d > 1e-6, "R2 control: zeroing the PLE tail changes the next-token logits (the check can fail)");
        llama_free(Cc);
    }

    // ---- C1: seq_cp 0 -> 1 ---------------------------------------------------------------------------------------------------------------------
    {
        llama_memory_seq_cp(llama_get_memory(S), 0, 1, -1, -1);
        const bool ok1 = decode(S, P, P + 1, 1, n_vocab);
        const std::vector<float> l1 = ok1 ? last_logits(S, n_vocab) : std::vector<float>();
        const bool ok0 = decode(S, P, P + 1, 0, n_vocab);
        const std::vector<float> l0 = ok0 ? last_logits(S, n_vocab) : std::vector<float>();
        const double d1 = ok1 ? max_diff(ref, l1) : 1e9, d0 = ok0 ? max_diff(ref, l0) : 1e9;
        printf("       seq_cp: seq1 max|dlogit| vs fresh = %.6g, seq0 continues = %.6g\n", d1, d0);
        check(ok1 && d1 < 1e-5, "C1 seq_cp(0 -> 1) then decode on seq 1 == a fresh run");
        check(ok0 && d0 < 1e-5, "C1 seq 0 continues unchanged after the copy");
    }

    llama_free(S); llama_free(A);
    llama_model_free(model);
    llama_backend_free();
    printf("RESULT: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
