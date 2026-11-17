// CPU backend supports_op must not CLAIM an op it has no row to compute.
//
// WHY. ggml-cpu.cpp answered TRUE for MUL_MAT / MUL_MAT_ID / CPY / GET_ROWS on every type, whether or not ggml-cpu.c's
// type_traits_cpu[] had a vec_dot / from_float for it. Executing the claimed op on such a type dies: MUL_MAT segfaults
// (0xC0000005), CPY aborts. MEASURED 2026-09-30, one op per process, on 22 types (TURBO2/3/4_0, Q8_1_X4, Q8_KR8, q8_1, q8_K,
// q8_K16/32/64/128, q8_0_x4, q8_2_x4, i8, i16, i32, i64, f64, q4_0_4x4/4x8/8x8, i2_s); the controls (q4_0, iq4_ks_r16,
// q1_0_g128_r8) ran. A wrong TRUE sends the scheduler to a backend that crashes instead of one that can do the work.
//
// WHAT IT CHECKS. For every type with a name: build MUL_MAT (type weights x F32), MUL_MAT_ID, and CPY (F32 -> type) and ask the CPU
// backend. A claimed MUL_MAT/MUL_MAT_ID needs type_traits_cpu[type].vec_dot; a claimed CPY needs .from_float (F32 -> F32 and the
// copy-to-self cases are the only ones that do not). Exit 0 none violated, 1 a violation, 2 refused (nothing was probed).
//
// CONTROLS, so a pass means the probe can see: (a) the predicate is run on fabricated rows and must flag the bad one and pass the good
// one; (b) Q4_0 must be CLAIMED for all three ops (a backend that refuses everything would pass the main check vacuously);
// (c) Q8_KR8 must be REFUSED (a type this test exists for: an internal layout with no row). TURBO3_0 was in this list until
// 2026-10-07; it now has a traits row (from_float + an f32 vec_dot, the CPU TurboQuant KV path), so the control became the
// opposite and stronger one: it must be CLAIMED for all three ops AND carry both rows. A claim without the rows is the
// violation the main check reports; claiming it with them is the intended state.
//
// LIMIT: it checks the answer against the traits table, not the numerics; a claimed op with a row is not proven correct here.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cstdio>

// a claim is only honest if the row it needs exists
static bool violates(bool claims_matmul, bool claims_matmul_id, bool claims_cpy, bool has_vec_dot, bool has_from_float) {
    return ((claims_matmul || claims_matmul_id) && !has_vec_dot) || (claims_cpy && !has_from_float);
}

int main() {
    printf("EXIT-CODE-SEMANTICS: findings=1\n");

    // control (a): the predicate itself
    if (!violates(true, false, false, false, true) || !violates(false, false, true, true, false) || violates(true, true, true, true, true) ||
        violates(false, false, false, false, false)) {
        printf("CONTROL FAILED: the violation predicate does not separate a bad claim from a good one\n");
        return 2;
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) {
        printf("REFUSED: no CPU backend\n");
        return 2;
    }
    ggml_init_params ip = { 64u * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const int K = 4096;   // divisible by every block size in this fork (1, 16, 32, 64, 128, 256)
    int probed = 0, bad = 0;
    bool q4_0_claimed = false, turbo3_claimed_with_rows = false, q8kr8_refused = false;

    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = (ggml_type) t;
        const auto * tr = ggml_get_type_traits(type);
        if (!tr || !tr->type_name || tr->blck_size <= 0 || K % tr->blck_size != 0) {
            continue;
        }
        const auto * trc = ggml_get_type_traits_cpu(type);
        const bool has_vec_dot    = trc && trc->vec_dot;
        const bool has_from_float = trc && trc->from_float;

        ggml_tensor * w   = ggml_new_tensor_2d(ctx, type, K, 8);
        ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 2);
        ggml_tensor * mm  = ggml_mul_mat(ctx, w, x);
        ggml_tensor * wx  = ggml_new_tensor_3d(ctx, type, K, 8, 4);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
        ggml_tensor * x3  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 2, 2);
        ggml_tensor * mmi = ggml_mul_mat_id(ctx, wx, x3, ids);
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 8);
        ggml_tensor * dst = ggml_new_tensor_2d(ctx, type, K, 8);
        ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

        const bool s_mm = ggml_backend_supports_op(be, mm);
        const bool s_mi = ggml_backend_supports_op(be, mmi);
        // F32 -> F32 is a plain copy and needs no from_float
        const bool s_cp = type == GGML_TYPE_F32 ? false : ggml_backend_supports_op(be, cpy);
        probed++;

        if (violates(s_mm, s_mi, s_cp, has_vec_dot, has_from_float)) {
            printf("  VIOLATION %-12s claims MUL_MAT=%d MUL_MAT_ID=%d CPY(f32->type)=%d but cpu row has vec_dot=%d from_float=%d\n",
                   tr->type_name, s_mm, s_mi, s_cp, has_vec_dot, has_from_float);
            bad++;
        }
        if (type == GGML_TYPE_Q4_0) {
            q4_0_claimed = s_mm && s_mi && s_cp;
        }
        if (type == GGML_TYPE_TURBO3_0) {
            turbo3_claimed_with_rows = s_mm && s_mi && s_cp && has_vec_dot && has_from_float;
        }
        if (type == GGML_TYPE_Q8_KR8) {
            q8kr8_refused = !s_mm && !s_mi && !s_cp;
        }
    }

    printf("SCOPE: %d type(s) probed on the CPU backend (MUL_MAT, MUL_MAT_ID, CPY f32->type); %d violation(s).\n", probed, bad);
    printf("CONTROL Q4_0 is claimed for all three ops:     %s\n", q4_0_claimed ? "yes" : "NO");
    printf("CONTROL TURBO3_0 is claimed for all three ops, with both rows: %s\n", turbo3_claimed_with_rows ? "yes" : "NO");
    printf("CONTROL Q8_KR8 is refused for all three ops:   %s\n", q8kr8_refused ? "yes" : "NO");

    ggml_free(ctx);
    ggml_backend_free(be);

    if (probed < 50 || !q4_0_claimed) {
        printf("REFUSED: the probe did not reach what it should have (probed %d, q4_0 claimed %d) -- a pass here would be vacuous.\n",
               probed, (int) q4_0_claimed);
        return 2;
    }
    if (!turbo3_claimed_with_rows || !q8kr8_refused) {
        printf("FAIL: a control type is not in its expected state (TURBO3_0 claimed with rows, Q8_KR8 refused).\n");
        return 1;
    }
    return bad == 0 ? 0 : 1;
}
