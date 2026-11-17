// test-disk-tier-async: asynchronous restore of common_disk_tier blobs (common_disk_loader), row
// S69-ENGREV-RTPLLM-ASYNC-DISK-RESTORE-2026-10-27-01. Design from rtp-llm BlockTreeLoader.cc:277-357 (async load, load-join registry,
// per-task deadline; Apache-2.0), clean-room. Patterned bytes, no model. Links only disk-tier.cpp.
#include "disk-tier.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;

static int g_checks = 0;
static int g_fails  = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s (line %d)\n", what, __LINE__); } } while (0)

static std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) { v[i] = (uint8_t) (seed + i * 31 + (i >> 8) * 7); }
    return v;
}

static common_disk_loader::state wait_done(common_disk_loader & L, uint64_t id) {
    for (int i = 0; i < 5000; i++) {
        const auto s = L.poll(id);
        if (s != common_disk_loader::state::LOADING) { return s; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return common_disk_loader::state::LOADING;
}

int main() {
    const auto root = fs::temp_directory_path() / "test-disk-tier-async";
    fs::remove_all(root);
    auto tier = std::make_shared<common_disk_tier>((root / "t").string(), 256u << 20, 0);
    tier->set_free_space_fn([](const std::string &) { return (int64_t) 1 << 40; });
    int n_stale = 0; std::string err;
    CHECK(tier->init(n_stale, err), "init");

    const auto pa = pattern(300000, 3), pb = pattern(70000, 9);
    const uint64_t id = tier->put(pa, pb, 42);
    CHECK(id != 0, "put");
    const uint64_t id2 = tier->put(pa, {}, 7, true);
    CHECK(id2 != 0, "put ckpt");

    // 1. async read returns the same bytes as the sync get() and books the same stats
    {
        common_disk_loader L(tier);
        CHECK(L.start(id, 42, 30.0), "start");
        CHECK(wait_done(L, id) == common_disk_loader::state::READY, "ready");
        std::vector<uint8_t> a, b;
        CHECK(L.take(id, a, b), "take");
        CHECK(a == pa && b == pb, "async payload equals what was put");
        CHECK(tier->stats().n_get == 1 && tier->stats().bytes_read > 0, "async read is booked like get()");
        CHECK(L.poll(id) == common_disk_loader::state::NONE, "task forgotten after take");
        CHECK(!L.take(id, a, b), "second take refused");
    }
    // 2. join-dedup: three starts for one blob, one read
    {
        common_disk_loader L(tier);
        CHECK(L.start(id2, 7, 30.0) && L.start(id2, 7, 30.0) && L.start(id2, 7, 30.0), "three starts");
        CHECK(L.n_started() == 1 && L.n_joined() == 2, "two joined the read in flight");
        CHECK(wait_done(L, id2) == common_disk_loader::state::READY, "ready 2");
        std::vector<uint8_t> a, b;
        CHECK(L.take(id2, a, b) && a == pa && b.empty(), "payload 2");
        CHECK(tier->stats().n_get_ckpt == 1, "ckpt counter booked once");
    }
    // 3. unknown blob: nothing started
    {
        common_disk_loader L(tier);
        CHECK(!L.start(999999, 1, 1.0) && L.n_started() == 0, "unknown id refused");
    }
    // 4. wrong tag and corruption fail and delete the blob, exactly as get() does (negative controls)
    {
        common_disk_loader L(tier);
        CHECK(L.start(id, 41, 30.0), "start wrong tag");
        CHECK(wait_done(L, id) == common_disk_loader::state::FAILED, "wrong tag fails");
        CHECK(tier->stats().n_get_failed == 1, "failure counted");
        std::vector<uint8_t> a, b;
        CHECK(!tier->get(id, 42, a, b), "the blob was deleted by the failed async read");

        const uint64_t idc = tier->put(pa, pb, 5);
        {
            std::fstream f(tier->path_of(idc), std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(200); char c = 0x55; f.write(&c, 1);
        }
        CHECK(L.start(idc, 5, 30.0), "start corrupt");
        CHECK(wait_done(L, idc) == common_disk_loader::state::FAILED, "corrupt byte fails");
        CHECK(tier->stats().n_checksum_failed == 1, "checksum failure counted");
    }
    // 5. deadline: a fake clock past the deadline reports TIMEOUT; the read is still booked by reap()
    {
        double now = 0.0;
        common_disk_loader L(tier, [&]() { return now; });
        const uint64_t idt = tier->put(pa, pb, 11);
        const auto before = tier->stats().n_get;
        CHECK(L.start(idt, 11, 5.0), "start deadline");
        now = 100.0;
        auto s = L.poll(idt);
        // the read may already be done (tiny file): then it is READY, not TIMEOUT; both are legal, a LOADING past deadline is not
        CHECK(s == common_disk_loader::state::TIMEOUT || s == common_disk_loader::state::READY, "past deadline is never LOADING");
        if (s == common_disk_loader::state::TIMEOUT) {
            CHECK(L.n_timeout() == 1, "timeout counted");
            for (int i = 0; i < 2000 && tier->stats().n_get == before; i++) {
                L.reap();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            CHECK(tier->stats().n_get == before + 1, "abandoned read still booked by reap()");
            CHECK(L.poll(idt) == common_disk_loader::state::NONE, "timed-out task is gone");
        }
    }
    fs::remove_all(root);
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
