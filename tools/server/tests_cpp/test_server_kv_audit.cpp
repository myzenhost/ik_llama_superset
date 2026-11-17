// Standalone test for server-kv-audit.h (row S69-ENGREV-SGLANG-KV-CANARY-2026-10-27-01).
// Build: g++ -std=c++17 -I.. test_server_kv_audit.cpp -o t && ./t     No model.
#include "server-kv-audit.h"

#include <cstdio>
#include <limits>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    std::vector<int32_t> toks;
    for (int i = 0; i < 64; ++i) {
        toks.push_back(100 + i * 3);
    }

    // the chain depends on token AND position AND history
    {
        CHECK(kv_audit_mix3(1, 5, 7) == kv_audit_mix3(1, 5, 7));
        CHECK(kv_audit_mix3(1, 5, 7) != kv_audit_mix3(1, 6, 7));
        CHECK(kv_audit_mix3(1, 5, 7) != kv_audit_mix3(1, 5, 8));
        CHECK(kv_audit_mix3(1, 5, 7) != kv_audit_mix3(2, 5, 7));
    }

    // canary: control (clean) vs injected wrong token -> exactly one violation at that position
    {
        kv_audit_canary c;
        CHECK(c.audit(1, toks) == -1);              // first stamp
        toks.push_back(7);                           // the slot grows
        CHECK(c.audit(1, toks) == -1);              // clean growth: no violation
        auto bad = toks;
        bad[20] ^= 0x55;                             // the injected wrong cell token
        CHECK(c.audit(1, bad) == 20);               // detected at that position
        CHECK(c.audit(1, bad) == -1);               // re-stamped: reported ONCE, not on every audit
    }
    // legal rewrites are not violations: a new owner, a shorter list (rewind / shift / restore)
    {
        kv_audit_canary c;
        c.audit(1, toks);
        auto other = toks;
        other[3] = 9999;
        CHECK(c.audit(2, other) == -1);             // new request on the slot
        std::vector<int32_t> shorter(other.begin(), other.begin() + 10);
        shorter[4] = 1;
        CHECK(c.audit(2, shorter) == -1);           // rewind then different text
        // and the control for that control: same owner, same length, one token changed IS a violation
        CHECK(c.audit(2, shorter) == -1);
        shorter[4] = 2;
        CHECK(c.audit(2, shorter) == 4);
    }
    // 1000 normal audits over a growing list = zero violations
    {
        kv_audit_canary c;
        std::vector<int32_t> t;
        int viol = 0;
        for (int i = 0; i < 1000; ++i) {
            t.push_back(i * 7 % 1000);
            viol += c.audit(1, t) >= 0;
        }
        CHECK(viol == 0);
    }
    // pool leak check: one-sided
    {
        CHECK(!kv_audit_pool_check(100, 100, 0).leak);   // identity
        CHECK(!kv_audit_pool_check(60, 100, 0).leak);    // shared prefix / sliding window: fewer cells than tokens is normal
        CHECK(!kv_audit_pool_check(105, 100, 8).leak);   // in-flight batch slack
        auto r = kv_audit_pool_check(130, 100, 8);
        CHECK(r.leak && r.unowned == 22);                // 130 - (100 + 8)
        CHECK(kv_audit_pool_check(101, 100, 0).leak);    // one leaked cell with no slack
    }
    // non-finite logits: NaN and +inf flagged, -inf (a mask) is not
    {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        float row[6] = { 0.5f, -inf, 2.0f, -3.0f, 1.0f, 0.0f };
        CHECK(kv_audit_first_nonfinite(row, 6) == -1);
        row[4] = nan;
        CHECK(kv_audit_first_nonfinite(row, 6) == 4);
        row[4] = 1.0f;
        row[2] = inf;
        CHECK(kv_audit_first_nonfinite(row, 6) == 2);
        CHECK(kv_audit_first_nonfinite(row, 2) == -1);   // only the prefix asked for is read
    }
    CHECK(kv_audit_describe(3, 20, "x").find("slot 3 position 20") != std::string::npos);

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
