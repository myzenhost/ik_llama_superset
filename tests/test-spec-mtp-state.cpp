// MTP boundary-state blob (common/spec-mtp-state.h).  Row S69-ENGREV-BEELLAMA-MTP-CARRY-STATE-CKPT-CACHE-2026-10-27-01.
// GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the header did not exist; the checkpoint blob of commit b79cd70914 was raw floats with no magic, version or width, so a blob saved
// by a model of another hidden width was accepted only by luck of the size check, and a prompt-cache entry carried no spec bytes at all.
#include "spec-mtp-state.h"

#include <cstdio>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    const uint32_t n = 6;
    const float h[6] = { 0.5f, -1.0f, 2.25f, 0.0f, 3.5f, -0.125f };
    std::vector<uint8_t> blob;
    common_spec_mtp_state_encode(h, n, 41, blob);
    CHECK(blob.size() == common_spec_mtp_state_header_bytes() + n * sizeof(float));

    {   // round trip
        float o[6] = {}; int32_t pos = -9;
        CHECK(common_spec_mtp_state_decode(blob, n, o, pos) == COMMON_SPEC_MTP_STATE_OK);
        CHECK(pos == 41);
        for (int i = 0; i < 6; i++) CHECK(o[i] == h[i]);
    }
    {   // empty blob -> EMPTY, outputs untouched
        float o[6] = { 7, 7, 7, 7, 7, 7 }; int32_t pos = -9;
        CHECK(common_spec_mtp_state_decode(std::vector<uint8_t>{}, n, o, pos) == COMMON_SPEC_MTP_STATE_EMPTY);
        CHECK(pos == -9 && o[0] == 7);
    }
    {   // wrong width (state saved by another model)
        float o[8] = {}; int32_t pos = -9;
        CHECK(common_spec_mtp_state_decode(blob, 8, o, pos) == COMMON_SPEC_MTP_STATE_BAD_WIDTH);
        CHECK(pos == -9);
    }
    {   // wrong magic
        auto b = blob; b[0] ^= 0xff;
        float o[6] = {}; int32_t pos = -9;
        CHECK(common_spec_mtp_state_decode(b, n, o, pos) == COMMON_SPEC_MTP_STATE_BAD_MAGIC);
    }
    {   // wrong version
        auto b = blob; b[4] = 2;
        float o[6] = {}; int32_t pos = -9;
        CHECK(common_spec_mtp_state_decode(b, n, o, pos) == COMMON_SPEC_MTP_STATE_BAD_VERSION);
    }
    {   // truncated body, truncated header, trailing junk
        float o[6] = {}; int32_t pos = -9;
        auto b = blob; b.pop_back();
        CHECK(common_spec_mtp_state_decode(b, n, o, pos) == COMMON_SPEC_MTP_STATE_BAD_SIZE);
        b = blob; b.resize(5);
        CHECK(common_spec_mtp_state_decode(b, n, o, pos) == COMMON_SPEC_MTP_STATE_BAD_SIZE);
        b = blob; b.push_back(0);
        CHECK(common_spec_mtp_state_decode(b, n, o, pos) == COMMON_SPEC_MTP_STATE_BAD_SIZE);
    }
    {   // legacy raw blob (row only, no header) is rejected, not reinterpreted
        std::vector<uint8_t> raw(n * sizeof(float), 0);
        float o[6] = {}; int32_t pos = -9;
        CHECK(common_spec_mtp_state_decode(raw, n, o, pos) != COMMON_SPEC_MTP_STATE_OK);
    }
    if (n_fail) { fprintf(stderr, "%d failure(s)\n", n_fail); return 1; }
    printf("test-spec-mtp-state: OK\n");
    return 0;
}
