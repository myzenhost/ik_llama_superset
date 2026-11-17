// probe-disk-tier: common_disk_tier (common/disk-tier.h), the NVMe tier under the host prompt cache (--cache-disk-dir).
// No model needed: the payloads are patterned bytes. Prints "N checks, M failed" and exits non-zero on a failure.
// Every guarantee has a negative control (a corrupted byte, a short file, a refused write...) so a probe that can only
// say "ok" would fail here. Row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01.
#include "disk-tier.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_checks = 0;
static int g_fails  = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s\n", what); } } while (0)

static std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = (uint8_t) (seed + i * 31 + (i >> 8) * 7);
    }
    return v;
}

static void flip_byte(const std::string & path, std::streamoff at) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(at);
    char c = 0;
    f.read(&c, 1);
    c = (char) (c ^ 0x5a);
    f.seekp(at);
    f.write(&c, 1);
}

static size_t n_files(const fs::path & dir) {
    size_t n = 0;
    for (const auto & e : fs::directory_iterator(dir)) { (void) e; n++; }
    return n;
}

int main() {
    const auto root = fs::temp_directory_path() / "probe-disk-tier";
    fs::remove_all(root);
    const std::string dir = (root / "tier").string();

    // ---- checksum: deterministic, and sensitive to a change anywhere ---------------------------------------------------
    {
        auto v = pattern(1000, 3);
        const uint64_t h = common_disk_tier::checksum(v.data(), v.size(), 7);
        CHECK(h == common_disk_tier::checksum(v.data(), v.size(), 7), "checksum is deterministic");
        CHECK(h != common_disk_tier::checksum(v.data(), v.size(), 8), "the seed changes the checksum");
        for (size_t at : { (size_t) 0, (size_t) 31, (size_t) 32, (size_t) 500, (size_t) 991, (size_t) 999 }) {
            auto w = v;
            w[at] ^= 1;
            CHECK(common_disk_tier::checksum(w.data(), w.size(), 7) != h, "a one-bit change anywhere changes the checksum");
        }
        std::vector<uint8_t> z1(64, 0), z2(65, 0);
        CHECK(common_disk_tier::checksum(z1.data(), z1.size()) != common_disk_tier::checksum(z2.data(), z2.size()),
              "zeros of different length differ");
        CHECK(common_disk_tier::checksum(nullptr, 0) == common_disk_tier::checksum(nullptr, 0), "empty input is fine");
    }

    // ---- init: creates the directory (with parents), removes only our stale files --------------------------------------
    {
        fs::create_directories(dir);
        std::ofstream(dir + "/pcache-9.bin") << "stale";
        std::ofstream(dir + "/pcache-9.tmp") << "stale";
        std::ofstream(dir + "/keep.txt") << "mine";
        common_disk_tier t(dir, 1 << 20, 0);
        int n = -1;
        std::string err;
        CHECK(t.init(n, err), "init succeeds");
        CHECK(n == 2, "init removed the two stale pcache files");
        CHECK(fs::exists(dir + "/keep.txt"), "control: init left an unrelated file alone");
        CHECK(!fs::exists(dir + "/pcache-9.bin") && !fs::exists(dir + "/pcache-9.tmp"), "stale files are gone");
        fs::remove(dir + "/keep.txt");
    }
    {
        common_disk_tier t((root / "a" / "b" / "c").string(), 1 << 20, 0);
        int n = 0;
        std::string err;
        CHECK(t.init(n, err) && fs::is_directory(root / "a" / "b" / "c"), "init creates missing parent directories");
    }

    // ---- round trip ------------------------------------------------------------------------------------------------------
    {
        common_disk_tier t(dir, 64u << 20, 0);
        int n; std::string err;
        t.init(n, err);
        const auto a = pattern(3u << 20, 1), b = pattern(777, 2);
        const uint64_t id = t.put(a, b, 1234);
        CHECK(id != 0, "put succeeds");
        CHECK(t.has(id) && fs::exists(t.path_of(id)), "the blob exists on disk");
        CHECK(fs::file_size(t.path_of(id)) == t.file_size_for(a.size(), b.size()), "file size is header + payloads");
        CHECK(t.used_bytes() == t.file_size_for(a.size(), b.size()), "used_bytes counts the file");
        CHECK(t.stats().bytes_written == t.used_bytes() && t.stats().n_put == 1, "bytes_written is the endurance counter");
        CHECK(!fs::exists(t.path_of(id).substr(0, t.path_of(id).size() - 4) + ".tmp"), "no .tmp file is left behind");

        std::vector<uint8_t> ra, rb;
        CHECK(t.get(id, 1234, ra, rb), "get succeeds");
        CHECK(ra == a && rb == b, "both payloads come back byte-identical (3 MiB + 777 bytes)");
        CHECK(t.has(id), "the blob stays after a read (the cache entry may be hit again)");
        CHECK(t.stats().n_get == 1 && t.stats().bytes_read == t.used_bytes(), "read counters");

        // empty second payload is legal (no draft model)
        const uint64_t id2 = t.put(pattern(100, 5), {}, 5);
        std::vector<uint8_t> xa, xb;
        CHECK(id2 != 0 && t.get(id2, 5, xa, xb) && xa.size() == 100 && xb.empty(), "an empty second payload round-trips");
        CHECK(t.put({}, pattern(10, 1), 1) == 0, "control: an empty first payload is refused");

        t.erase(id);
        CHECK(!t.has(id) && !fs::exists(t.path_of(id)), "erase removes the file");
        CHECK(t.used_bytes() == t.file_size_for(100, 0), "erase gives the bytes back");
        CHECK(!t.get(id, 1234, ra, rb), "control: reading an erased blob is a miss");
    }
    CHECK(n_files(dir) == 0, "the destructor removed every file it wrote");

    // ---- verification: a bad blob is a miss, is deleted, and is counted ------------------------------------------------
    {
        common_disk_tier t(dir, 64u << 20, 0);
        int n; std::string err;
        t.init(n, err);
        std::vector<uint8_t> ra, rb;

        // wrong tag
        uint64_t id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        CHECK(!t.get(id, 43, ra, rb), "control: a different tag is refused");
        CHECK(!t.has(id) && !fs::exists(t.path_of(id)), "control: the refused blob was deleted");
        CHECK(t.used_bytes() == 0, "control: its bytes were given back");

        // flipped payload byte (same length, wrong content)
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        flip_byte(t.path_of(id), 40 + 2500);
        CHECK(!t.get(id, 42, ra, rb), "control: a flipped payload byte is caught");
        CHECK(t.stats().n_checksum_failed == 1, "it is counted as a checksum failure");
        CHECK(!fs::exists(t.path_of(id)) && ra.empty() && rb.empty(), "control: deleted, and no bytes were handed out");

        // flipped byte in the second payload
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        flip_byte(t.path_of(id), 40 + 5000 + 10);
        CHECK(!t.get(id, 42, ra, rb) && t.stats().n_checksum_failed == 2, "control: a flipped byte in the second payload is caught");

        // truncated file
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        fs::resize_file(t.path_of(id), 2000);
        CHECK(!t.get(id, 42, ra, rb), "control: a truncated file is refused");
        CHECK(t.stats().n_get_failed >= 1 && !fs::exists(t.path_of(id)), "control: it is counted and deleted");

        // grown file (trailing junk)
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        { std::ofstream f(t.path_of(id), std::ios::binary | std::ios::app); f << "junk"; }
        CHECK(!t.get(id, 42, ra, rb), "control: a file with trailing bytes is refused");

        // wrong magic
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        flip_byte(t.path_of(id), 0);
        CHECK(!t.get(id, 42, ra, rb), "control: a wrong magic is refused");

        // lengths in the header that do not add up must never become an allocation
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        flip_byte(t.path_of(id), 8 + 8 + 7); // top byte of len_a
        CHECK(!t.get(id, 42, ra, rb), "control: an absurd length in the header is refused, not allocated");

        // missing file
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        fs::remove(t.path_of(id));
        CHECK(!t.get(id, 42, ra, rb), "control: a missing file is a miss, no crash");
        CHECK(!t.has(id) && t.used_bytes() == 0, "control: the bookkeeping is cleaned up");

        // unknown id
        CHECK(!t.get(999999, 1, ra, rb), "control: an unknown id is a miss");

        // and an untouched blob still reads (the controls above did not just make get() always fail)
        id = t.put(pattern(5000, 1), pattern(50, 2), 42);
        CHECK(t.get(id, 42, ra, rb) && ra == pattern(5000, 1) && rb == pattern(50, 2), "control: an untouched blob reads fine");
    }

    // ---- size cap --------------------------------------------------------------------------------------------------------
    {
        common_disk_tier t(dir, 100 * 1024, 0);
        int n; std::string err;
        t.init(n, err);
        const uint64_t i1 = t.put(pattern(40 * 1024, 1), {}, 1);
        const uint64_t i2 = t.put(pattern(40 * 1024, 2), {}, 2);
        CHECK(i1 != 0 && i2 != 0, "two blobs fit under the cap");
        CHECK(t.used_bytes() <= t.limit_bytes(), "used_bytes is within the cap");
        CHECK(!t.fits(40 * 1024, 0), "a third does not fit");
        CHECK(t.put(pattern(40 * 1024, 3), {}, 3) == 0, "put refuses a blob that would exceed the cap");
        CHECK(t.stats().n_refused_cap == 1, "the refusal is counted");
        CHECK(n_files(dir) == 2, "a refused put wrote nothing");
        t.erase(i1);
        CHECK(t.fits(40 * 1024, 0) && t.put(pattern(40 * 1024, 3), {}, 3) != 0, "after an erase the same blob fits");
        common_disk_tier off(dir, 0, 0);
        CHECK(off.put(pattern(10, 1), {}, 1) == 0, "a tier with cap 0 stores nothing");
    }

    // ---- write budget (injected clock, no sleeping) ----------------------------------------------------------------------
    {
        double now = 1000.0;
        // 10 KiB per hour
        common_disk_tier t(dir, 64u << 20, 10 * 1024, [&] { return now; });
        int n; std::string err;
        t.init(n, err);
        const auto p = pattern(6 * 1024 - 40, 1); // file = exactly 6 KiB
        CHECK(t.put(p, {}, 1) != 0, "first write is inside the budget (bucket starts full)");
        CHECK(t.stats().bytes_written == 6 * 1024, "6 KiB written");
        CHECK(!t.budget_allows(p.size(), 0), "the second 6 KiB is over what is left (4 KiB)");
        CHECK(t.put(p, {}, 2) == 0 && t.stats().n_refused_budget == 1, "put refuses it and counts a budget refusal");
        CHECK(t.stats().bytes_written == 6 * 1024, "control: the refused write cost no wear");
        now += 1800.0; // half an hour: +5 KiB
        CHECK(t.budget_allows(p.size(), 0), "after half an hour the bucket has refilled enough");
        CHECK(t.put(p, {}, 2) != 0, "the write is accepted");
        now += 100000.0; // far in the future: the bucket must cap at one hour (10 KiB), not grow without bound
        CHECK(t.put(p, {}, 3) != 0, "a refilled bucket pays for one 6 KiB write");
        CHECK(t.put(p, {}, 4) == 0, "control: but not for two in a row (the bucket holds one hour, not 28 hours)");
        // a blob bigger than the whole hourly budget can never be written
        CHECK(!t.budget_allows(20 * 1024, 0), "a blob larger than the hourly budget is never allowed");

        common_disk_tier u(dir, 64u << 20, 0, [&] { return now; });
        u.init(n, err);
        bool all = true;
        for (int i = 0; i < 20; i++) { all = all && u.put(p, {}, i) != 0; }
        CHECK(all, "control: budget 0 means unlimited");
    }

    // ---- an I/O failure leaves nothing behind -----------------------------------------------------------------------------
    {
        const std::string gone = (root / "vanishing").string();
        common_disk_tier t(gone, 1 << 20, 0);
        int n; std::string err;
        t.init(n, err);
        fs::remove_all(gone); // the directory disappears under a running server
        CHECK(t.put(pattern(1000, 1), {}, 1) == 0, "put fails when the directory is gone");
        CHECK(t.stats().n_put_failed == 1 && t.used_bytes() == 0, "the failure is counted and no bytes are accounted");
        CHECK(!fs::exists(gone), "control: nothing was recreated");
    }

    // ---- checkpoint blobs: same cap, same budget, same checksum; own file name and counters ------------------------------
    // Row S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01.
    {
        double now = 5000.0;
        common_disk_tier t(dir, 1 << 20, 100 * 1024, [&] { return now; });
        int n; std::string err;
        t.init(n, err);
        const uint64_t ic = t.put(pattern(10000, 1), pattern(100, 2), 77, /*is_ckpt=*/true);
        const uint64_t ip = t.put(pattern(10000, 3), pattern(100, 4), 78);
        CHECK(ic != 0 && ip != 0, "a checkpoint blob and a prompt blob both fit");
        CHECK(fs::path(t.path_of(ic)).filename().string().rfind("pckpt-", 0) == 0, "a checkpoint blob is named pckpt-<id>.bin");
        CHECK(fs::path(t.path_of(ip)).filename().string().rfind("pcache-", 0) == 0, "control: a prompt blob keeps the pcache-<id>.bin name");
        CHECK(t.stats().n_put == 2 && t.stats().n_put_ckpt == 1, "n_put counts both, n_put_ckpt only the checkpoint");
        CHECK(t.stats().bytes_written_ckpt == t.file_size_for(10000, 100) && t.stats().bytes_written == 2 * t.file_size_for(10000, 100),
              "the checkpoint bytes are part of the one endurance counter, and also reported on their own");
        // the one write budget pays for both kinds: 100 KiB per hour, 2 x ~10 KiB spent, a 90 KiB blob must be refused
        CHECK(!t.budget_allows_bytes(90 * 1024), "control: the budget is shared, what the two blobs spent is gone");
        CHECK(t.budget_allows_bytes(70 * 1024), "and what is left is still available");
        CHECK(t.put(pattern(90 * 1024, 5), {}, 1, true) == 0 && t.stats().n_refused_budget == 1,
              "a checkpoint blob over the remaining budget is refused and counted like any other");
        // the one cap too
        common_disk_tier small(dir + "-small", 16 * 1024, 0);   // its own directory: init() wipes the stale files of the directory it is given
        small.init(n, err);
        CHECK(small.put(pattern(10000, 1), {}, 1, true) != 0, "a checkpoint blob fits under the cap");
        CHECK(small.put(pattern(10000, 2), {}, 2, true) == 0 && small.stats().n_refused_cap == 1, "the second does not: the cap is shared");

        std::vector<uint8_t> ra, rb;
        CHECK(t.get(ic, 77, ra, rb) && ra == pattern(10000, 1) && rb == pattern(100, 2), "a checkpoint blob round-trips both payloads");
        CHECK(t.stats().n_get_ckpt == 1 && t.stats().bytes_read_ckpt == t.file_size_for(10000, 100), "checkpoint read counters");
        CHECK(t.get(ip, 78, ra, rb) && t.stats().n_get == 2 && t.stats().n_get_ckpt == 1, "control: a prompt read does not count as a checkpoint read");
        // the same corruption controls as the prompt blobs
        flip_byte(t.path_of(ic), 40 + 5000);
        CHECK(!t.get(ic, 77, ra, rb) && t.stats().n_checksum_failed == 1, "a flipped byte in a checkpoint blob is caught by the shared checksum");
        CHECK(!t.get(ic, 77, ra, rb), "and the blob is gone afterwards");
    }
    {
        // stale files of both kinds from a crashed run are removed at init, other files are not
        fs::create_directories(dir);
        std::ofstream(dir + "/pckpt-3.bin") << "stale";
        std::ofstream(dir + "/pckpt-3.tmp") << "stale";
        std::ofstream(dir + "/keep2.txt") << "mine";
        common_disk_tier t(dir, 1 << 20, 0);
        int n = -1; std::string err;
        t.init(n, err);
        CHECK(n == 2 && !fs::exists(dir + "/pckpt-3.bin") && !fs::exists(dir + "/pckpt-3.tmp"), "init also removes stale pckpt-* files");
        CHECK(fs::exists(dir + "/keep2.txt"), "control: an unrelated file is left alone");
        fs::remove(dir + "/keep2.txt");
    }

    // ---- common_disk_blob: a counted reference ---------------------------------------------------------------------------
    {
        auto tier = std::make_shared<common_disk_tier>(dir, 1 << 20, 0);
        int n; std::string err;
        tier->init(n, err);
        const uint64_t id = tier->put(pattern(1000, 1), {}, 1, true);
        const std::string path = tier->path_of(id);
        auto h1 = std::make_shared<common_disk_blob>(tier, id);
        auto h2 = h1;                                      // a second copy of the checkpoint shares the reference
        CHECK(h1->id() == id && h1->tier().get() == tier.get(), "the handle names its blob and its tier");
        h1.reset();
        CHECK(tier->has(id) && fs::exists(path), "control: one of two references released, the blob stays");
        std::weak_ptr<common_disk_tier> weak = tier;
        tier.reset();                                      // the owner lets go of the tier: the handle keeps it alive
        CHECK(!weak.expired() && fs::exists(path), "the handle keeps the tier alive, so a late release is always safe");
        h2.reset();
        CHECK(weak.expired(), "the last release lets the tier go");
        CHECK(!fs::exists(path), "and the last release erased the blob's file");
        CHECK(n_files(dir) == 0, "nothing left in the directory");
    }
    {
        // a blob the tier already dropped (failed its checks) can be released without harm
        auto tier = std::make_shared<common_disk_tier>(dir, 1 << 20, 0);
        int n; std::string err;
        tier->init(n, err);
        const uint64_t id = tier->put(pattern(1000, 1), {}, 1, true);
        auto h = std::make_shared<common_disk_blob>(tier, id);
        std::vector<uint8_t> ra, rb;
        CHECK(!tier->get(id, 2, ra, rb), "wrong tag: the tier drops the blob");
        const auto n_erased = tier->stats().n_erased;
        h.reset();
        CHECK(tier->stats().n_erased == n_erased && tier->used_bytes() == 0, "releasing a handle whose blob is already gone is a no-op, no double count");
    }

    fs::remove_all(root);
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
