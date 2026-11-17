// MTP drafter embd-input rows (common/spec-mtp-embd.h).  Row S69-ENGREV-SPEC-MTP-MTMD-EMBD-INPUT-2026-11-14-01.
// GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the header did not exist and common_speculative_impl_mtp::process() returned true for every batch with `embd` set, so an image
// chunk left the MTP KV cache without a row for any of its positions. Before the fix, kind_of(false, true) was the "skip" case (NONE).
#include "spec-mtp-embd.h"

#include <cstdio>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    // input kind
    CHECK(common_spec_mtp_input_kind_of(true,  false) == COMMON_SPEC_MTP_INPUT_TOKENS);
    CHECK(common_spec_mtp_input_kind_of(false, true ) == COMMON_SPEC_MTP_INPUT_EMBD);   // was skipped before the fix
    CHECK(common_spec_mtp_input_kind_of(false, false) == COMMON_SPEC_MTP_INPUT_NONE);
    CHECK(common_spec_mtp_input_kind_of(true,  true ) == COMMON_SPEC_MTP_INPUT_EMBD);

    // gather: 5 rows x width 3, two interleaved sequences; seq B owns rows 1 and 3
    const int32_t w = 3;
    float src[5 * 3];
    for (int r = 0; r < 5; ++r) for (int c = 0; c < w; ++c) src[r * w + c] = (float) (10 * r + c);
    {
        std::vector<int32_t> rows = { 1, 3 };
        float dst[2 * 3] = {};
        CHECK(common_spec_mtp_gather_embd_rows(src, w, rows, dst));
        CHECK(dst[0] == 10.0f && dst[1] == 11.0f && dst[2] == 12.0f);
        CHECK(dst[3] == 30.0f && dst[4] == 31.0f && dst[5] == 32.0f);
    }
    {   // all rows in order is an identity copy
        std::vector<int32_t> rows = { 0, 1, 2, 3, 4 };
        float dst[5 * 3] = {};
        CHECK(common_spec_mtp_gather_embd_rows(src, w, rows, dst));
        for (int i = 0; i < 15; ++i) CHECK(dst[i] == src[i]);
    }
    {   // refusal writes nothing
        std::vector<int32_t> rows = { 0 };
        float dst[3] = { -1, -1, -1 };
        CHECK(!common_spec_mtp_gather_embd_rows(nullptr, w, rows, dst));
        CHECK(!common_spec_mtp_gather_embd_rows(src, 0, rows, dst));
        std::vector<int32_t> bad = { -1 };
        CHECK(!common_spec_mtp_gather_embd_rows(src, w, bad, dst));
        CHECK(dst[0] == -1 && dst[1] == -1 && dst[2] == -1);
    }
    if (n_fail) { fprintf(stderr, "%d failure(s)\n", n_fail); return 1; }
    printf("test-spec-mtp-embd: OK\n");
    return 0;
}
