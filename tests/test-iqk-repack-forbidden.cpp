// Run-time repacking (-rtr, and the iqk_repacked_type() answer llama-quantize reads) must leave the tensors a row-interleaved
// layout cannot serve: token_embd / per_layer_token_embd (exact names) and the DeepSeek-V4 engram tensors
// blk.N.engram_embd.weight / blk.N.engram_k.weight / blk.N.engram_q.weight.
//
// WHY. engram_embd is read with ggml_get_rows (src/graphs/build_deepseek4.cpp:1102) and engram_k / engram_q go through as_f32()
// (:1129-1130); a Q4_0 engram tensor repacked to Q4_0_R8 has no get_rows / dequantize row for the interleaved layout. ik_llama
// fixed the same gap in 11b54e80f "Do not repack engram tensors" (#2612, MIT, ik/main 2026-10-07): is_forbidden_tensor() grew a
// substring list. Row S69-ENGINE-MERGE-THIREUS-IK-LLAMA-2026-09-09-01 (ik commits surfaced by the Thireus delta).
//
// CONTROLS, so a pass means the probe can see: an ordinary blk.0.ffn_up.weight Q4_0 tensor with 8 rows MUST come back as
// Q4_0_R8 (the repack path is live in this build); token_embd.weight must stay Q4_0 (the pre-existing exact-name rule).
// A look-alike name that merely contains "engram" but none of the three suffixes must still be repacked (the match is on the
// suffix, not the word).
//
// Exit 0 all pass, 1 any check failed.
#include "ggml.h"

#include <cstdio>

extern "C" int iqk_repacked_type(const struct ggml_tensor * tensor);   // ggml/src/iqk/iqk_quantize.h:392

static int g_fail = 0;

static void check(const char * what, bool ok) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

int main() {
    ggml_init_params ip = { /*mem_size*/ 1 << 20, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::printf("ggml_init failed\n"); return 1; }

    auto make = [&](const char * name) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 8);  // 8 rows: Q4_0 -> Q4_0_R8 is eligible
        ggml_set_name(t, name);
        return t;
    };
    auto repacked = [&](const char * name) { return (ggml_type) iqk_repacked_type(make(name)); };

    // controls
    check("control: blk.0.ffn_up.weight Q4_0 is repacked to Q4_0_R8", repacked("blk.0.ffn_up.weight") == GGML_TYPE_Q4_0_R8);
    check("control: token_embd.weight stays Q4_0",                    repacked("token_embd.weight") == GGML_TYPE_Q4_0);
    check("control: per_layer_token_embd.weight stays Q4_0",          repacked("per_layer_token_embd.weight") == GGML_TYPE_Q4_0);
    check("control: blk.3.engram_gate.weight (look-alike) IS repacked", repacked("blk.3.engram_gate.weight") == GGML_TYPE_Q4_0_R8);

    // the fix
    check("blk.3.engram_embd.weight stays Q4_0", repacked("blk.3.engram_embd.weight") == GGML_TYPE_Q4_0);
    check("blk.3.engram_k.weight stays Q4_0",    repacked("blk.3.engram_k.weight") == GGML_TYPE_Q4_0);
    check("blk.3.engram_q.weight stays Q4_0",    repacked("blk.3.engram_q.weight") == GGML_TYPE_Q4_0);
    check("blk.27.engram_embd.weight stays Q4_0", repacked("blk.27.engram_embd.weight") == GGML_TYPE_Q4_0);

    ggml_free(ctx);
    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
