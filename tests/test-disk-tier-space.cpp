// test-disk-tier-space: the free-space guard of common_disk_tier (common/disk-tier.h), row
// S69-MLX-PICK-DISK-TIER-FREE-SPACE-GUARD-2026-10-05-01. The free-space query is injected, so no disk is filled; payloads are
// patterned bytes, no model. Prints "N checks, M failed" and exits non-zero on a failure. Links only disk-tier.cpp.
// Policy from lmstudio-ai/mlx-engine @86f20eb4 (MIT) mlx_engine/model_kit/batched_vision/prompt_cache/disk_budget.py:27-29,82-91
// (under 10 GiB free: no disk records; otherwise at most free/4); rewritten, no code copied.
#include "disk-tier.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_checks = 0;
static int g_fails  = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s (line %d)\n", what, __LINE__); } } while (0)

static std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) { v[i] = (uint8_t) (seed + i * 31 + (i >> 8) * 7); }
    return v;
}

static size_t n_files(const fs::path & dir) {
    size_t n = 0;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(dir, ec)) { (void) e; n++; }
    return n;
}

static constexpr int64_t GiB = 1ll << 30;

int main() {
    const auto root = fs::temp_directory_path() / "test-disk-tier-space";
    fs::remove_all(root);

    // ---- the floor: under 10 GiB free the tier writes nothing, and the refusal is counted, not an error -------------------
    {
        const std::string dir = (root / "floor").string();
        int64_t free_now = 10 * GiB - 1; // one byte under the default floor
        common_disk_tier t(dir, 64u << 20, 0);
        t.set_free_space_fn([&](const std::string & d) { CHECK(d == dir, "the query is asked about the tier's directory"); return free_now; });
        int n_stale = 0; std::string err;
        CHECK(t.init(n_stale, err), "init");

        const auto p = pattern(4096, 1);
        CHECK(!t.space_allows_bytes(t.file_size_for(p.size(), 0)), "space_allows_bytes says no under the floor");
        const uint64_t id = t.put(p, {}, 1);
        CHECK(id == 0, "put is refused when free space is under 10 GiB");
        CHECK(t.stats().n_refused_space == 1, "the refusal is counted in n_refused_space");
        CHECK(t.stats().n_put_failed == 0 && t.stats().n_refused_cap == 0 && t.stats().n_refused_budget == 0,
              "and not as an I/O failure, a cap or a budget refusal");
        CHECK(t.stats().n_put == 0 && t.stats().bytes_written == 0 && t.used_bytes() == 0, "nothing was written");
        CHECK(n_files(dir) == 0, "no file (and no .tmp) left in the directory");

        // the guard is re-checked on every put(): free space coming back re-opens the tier
        free_now = 10 * GiB;
        const uint64_t id2 = t.put(p, {}, 2);
        CHECK(id2 != 0, "put succeeds once 10 GiB are free again");
        CHECK(t.stats().n_refused_space == 1, "no further refusal counted");
        std::vector<uint8_t> ra, rb;
        CHECK(t.get(id2, 2, ra, rb) && ra == p, "the blob reads back intact");

        // and another process filling the disk closes it again, mid-run
        free_now = 2 * GiB;
        CHECK(t.put(pattern(4096, 3), {}, 3) == 0 && t.stats().n_refused_space == 2, "a disk filled by someone else closes the tier again");
        CHECK(t.has(id2) && t.get(id2, 2, ra, rb), "what the tier already holds is kept and still readable");
    }

    // ---- the fraction cap: the tier may hold at most (free + used) / divisor, below the configured cap ---------------------
    // The real floor and divisor (10 GiB, 4) would need GiB-sized blobs to reach the fraction, so the policy is scaled down.
    {
        const std::string dir = (root / "fraction").string();
        int64_t free_now = 400 * 1024; // 400 KiB free -> allowance 100 KiB for an empty tier with divisor 4
        common_disk_tier t(dir, 64u << 20, 0);
        t.set_free_space_fn([&](const std::string &) { return free_now; });
        t.set_free_space_policy(64 * 1024, 4);
        int n_stale = 0; std::string err;
        CHECK(t.init(n_stale, err), "init");

        const uint64_t a = t.put(pattern(60 * 1024, 4), {}, 4);
        CHECK(a != 0, "a 60 KiB blob fits in an allowance of 100 KiB");
        free_now -= (int64_t) t.used_bytes(); // the write itself took that much free space
        const uint64_t b = t.put(pattern(60 * 1024, 5), {}, 5);
        CHECK(b == 0 && t.stats().n_refused_space == 1, "a second one would take the tier past (free + used) / 4 and is refused");
        CHECK(t.stats().n_refused_cap == 0, "the configured cap (64 MiB) was not the reason");
        const uint64_t c = t.put(pattern(30 * 1024, 6), {}, 6);
        CHECK(c != 0, "a smaller one that stays inside the allowance still goes in");

        // the floor of the scaled policy
        free_now = 64 * 1024 - 1;
        CHECK(t.put(pattern(16, 7), {}, 7) == 0 && t.stats().n_refused_space == 2, "under the scaled floor nothing is written");
    }

    // ---- ample space changes nothing: the configured cap and the budget behave exactly as before ---------------------------
    {
        const std::string dir = (root / "ample").string();
        common_disk_tier t(dir, 100 * 1024, 0);
        t.set_free_space_fn([](const std::string &) { return 500 * GiB; });
        int n_stale = 0; std::string err;
        CHECK(t.init(n_stale, err), "init");
        CHECK(t.put(pattern(60 * 1024, 8), {}, 8) != 0, "first blob in");
        CHECK(t.put(pattern(60 * 1024, 9), {}, 9) == 0 && t.stats().n_refused_cap == 1 && t.stats().n_refused_space == 0,
              "with ample free space the configured cap is what refuses, as before");
    }

    // ---- an unknown free space (the query fails) leaves put() as it was ---------------------------------------------------
    {
        const std::string dir = (root / "unknown").string();
        common_disk_tier t(dir, 1 << 20, 0);
        t.set_free_space_fn([](const std::string &) { return (int64_t) -1; });
        int n_stale = 0; std::string err;
        CHECK(t.init(n_stale, err), "init");
        CHECK(t.put(pattern(4096, 10), {}, 10) != 0 && t.stats().n_refused_space == 0, "no information, no refusal");
    }

    // ---- the guard can be turned off, and the default query is the real file system -----------------------------------------
    {
        const std::string dir = (root / "off").string();
        common_disk_tier t(dir, 1 << 20, 0);
        t.set_free_space_fn([](const std::string &) { return (int64_t) 1; }); // 1 byte free
        t.set_free_space_policy(0, 0);
        int n_stale = 0; std::string err;
        CHECK(t.init(n_stale, err), "init");
        CHECK(t.put(pattern(4096, 11), {}, 11) != 0 && t.stats().n_refused_space == 0, "policy (0, 0) disables the guard");

        // back to the default query: whatever this machine's temp volume has, the decision must match the real numbers
        t.set_free_space_fn(nullptr);
        t.set_free_space_policy(common_disk_tier::DEFAULT_MIN_FREE_BYTES, common_disk_tier::DEFAULT_FREE_SPACE_DIVISOR);
        std::error_code ec;
        const auto si = fs::space(dir, ec);
        CHECK(!ec, "std::filesystem::space works on the temp directory");
        const size_t fb = t.file_size_for(4096, 0);
        const bool expect = (int64_t) si.available >= common_disk_tier::DEFAULT_MIN_FREE_BYTES &&
                            t.used_bytes() + fb <= (si.available + t.used_bytes()) / 4;
        CHECK(t.space_allows_bytes(fb) == expect, "the default query agrees with std::filesystem::space");
        std::printf("temp volume: %.2f GiB available, guard %s\n", si.available / (double) GiB, expect ? "open" : "ENGAGED");
    }

    fs::remove_all(root);
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
