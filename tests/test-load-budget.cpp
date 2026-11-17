// test-load-budget: the host-RAM budget that bounds parallel tensor loads in
// llama_model_loader::load_all_data (src/llama-load-budget.h). Model-free, CPU, no CUDA needed.
//
// Covers: /proc/meminfo and cgroup v2 parsing, the leaf-to-root cgroup walk through a fake file
// reader (so the Linux logic runs on every host), the 2/3 budget, admission (cap at ONE in-flight
// tensor when RAM is short, never zero), unlimited when the probe fails, the semaphore under real
// threads with controls in both directions, and the live OS probe on this host.
#include "llama-load-budget.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", what, __FILE__, __LINE__); g_fail = 1; } else { std::printf("ok:   %s\n", what); } } while (0)

namespace lb = llama_load_budget;

static constexpr uint64_t KiB = 1024ull;
static constexpr uint64_t MiB = 1024ull * KiB;
static constexpr uint64_t GiB = 1024ull * MiB;

static void test_meminfo() {
    const std::string mi =
        "MemTotal:       131072000 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:   65536000 kB\n"
        "Buffers:          123 kB\n";
    auto a = lb::meminfo_available(mi);
    CHECK(a.has_value() && *a == 65536000ull * KiB, "meminfo: MemAvailable kB -> bytes");
    CHECK(!lb::meminfo_available("MemTotal: 1 kB\nMemFree: 2 kB\n").has_value(), "meminfo: no MemAvailable key -> nullopt");
    CHECK(!lb::meminfo_available("MemAvailable: lots\n").has_value(), "meminfo: non-numeric value -> nullopt");
    CHECK(!lb::meminfo_available("").has_value(), "meminfo: empty text -> nullopt");
    // a key that merely starts with the same letters must not match
    CHECK(!lb::meminfo_available("MemAvailableX: 5 kB\n").has_value(), "meminfo: prefix-only key does not match");
}

static void test_cgroup_parse() {
    auto p = lb::cgroup_v2_path("12:cpuset:/old\n0::/user.slice/user-1000.slice/session.scope\n");
    CHECK(p.has_value() && *p == "/user.slice/user-1000.slice/session.scope", "cgroup: v2 leaf path from /proc/self/cgroup");
    CHECK(!lb::cgroup_v2_path("12:cpuset:/old\n4:memory:/x\n").has_value(), "cgroup: v1-only host -> no v2 path");

    const std::string stat = "anon 100\nfile 1073741824\nkernel 5\nshmem 0\n";
    auto a = lb::cgroup_available("8589934592\n", "4294967296\n", stat);
    CHECK(a.has_value() && *a == 8 * GiB - (4 * GiB - 1 * GiB), "cgroup: limit - (current - reclaimable file cache)");
    CHECK(!lb::cgroup_available("max\n", "4294967296\n", stat).has_value(), "cgroup: memory.max = max -> no limit at this level");
    auto shm = lb::cgroup_available("8589934592", "4294967296", "file 1073741824\nshmem 536870912\n");
    CHECK(shm.has_value() && *shm == 8 * GiB - (4 * GiB - 512 * MiB), "cgroup: shmem is not reclaimable");
    auto full = lb::cgroup_available("1073741824", "2147483648", "file 0\nshmem 0\n");
    CHECK(full.has_value() && *full == 0, "cgroup: charged above the limit -> 0 available (not unlimited)");
    CHECK(!lb::cgroup_available("8589934592", "", stat).has_value(), "cgroup: unreadable memory.current -> level skipped");
}

// Fake filesystem for the Linux walk.
struct fake_fs {
    std::map<std::string, std::string> files;
    lb::read_fn reader() const {
        return [this](const std::string & path) -> std::optional<std::string> {
            auto it = files.find(path);
            if (it == files.end()) { return std::nullopt; }
            return it->second;
        };
    }
};

static void test_linux_walk() {
    fake_fs fs;
    fs.files["/proc/meminfo"] = "MemAvailable: 16777216 kB\n"; // 16 GiB
    fs.files["/proc/self/cgroup"] = "0::/a/b\n";
    fs.files["/sys/fs/cgroup/a/b/memory.max"]     = "4294967296\n";        // 4 GiB
    fs.files["/sys/fs/cgroup/a/b/memory.current"] = "1073741824\n";        // 1 GiB
    fs.files["/sys/fs/cgroup/a/b/memory.stat"]    = "file 0\nshmem 0\n";
    auto a = lb::linux_allocatable_bytes(fs.reader());
    CHECK(a.has_value() && *a == 3 * GiB, "walk: leaf cgroup limit is tighter than MemAvailable");

    fs.files["/sys/fs/cgroup/a/memory.max"]     = "2147483648\n";          // parent 2 GiB, empty
    fs.files["/sys/fs/cgroup/a/memory.current"] = "0\n";
    fs.files["/sys/fs/cgroup/a/memory.stat"]    = "file 0\nshmem 0\n";
    a = lb::linux_allocatable_bytes(fs.reader());
    CHECK(a.has_value() && *a == 2 * GiB, "walk: a tighter ANCESTOR wins (leaf -> root)");

    fs.files["/sys/fs/cgroup/a/memory.max"] = "max\n";
    fs.files["/sys/fs/cgroup/a/b/memory.max"] = "max\n";
    a = lb::linux_allocatable_bytes(fs.reader());
    CHECK(a.has_value() && *a == 16 * GiB, "walk control: no cgroup limit -> MemAvailable");

    fake_fs none;
    CHECK(!lb::linux_allocatable_bytes(none.reader()).has_value(), "walk: nothing readable -> nullopt (probe failed)");

    fake_fs cg_only;
    cg_only.files["/proc/self/cgroup"] = "0::/c\n";
    cg_only.files["/sys/fs/cgroup/c/memory.max"]     = "1073741824\n";
    cg_only.files["/sys/fs/cgroup/c/memory.current"] = "0\n";
    cg_only.files["/sys/fs/cgroup/c/memory.stat"]    = "file 0\nshmem 0\n";
    a = lb::linux_allocatable_bytes(cg_only.reader());
    CHECK(a.has_value() && *a == 1 * GiB, "walk: cgroup limit alone (no /proc/meminfo) still bounds");
}

// ggml_host_mem::linux_host_memory: what the CPU backend device reports as total/free on Linux
// (ggml_backend_cpu_device_get_memory). Same fake file system.
static void test_cpu_device_memory() {
    namespace hm = ggml_host_mem;
    fake_fs fs;
    fs.files["/proc/meminfo"] = "MemTotal: 33554432 kB\nMemFree: 1048576 kB\nMemAvailable: 16777216 kB\n"; // 32 GiB / 16 GiB
    auto m = hm::linux_host_memory(fs.reader());
    CHECK(m.total && *m.total == 32 * GiB && m.free && *m.free == 16 * GiB, "cpu mem control: no cgroup -> MemTotal / MemAvailable");

    fs.files["/proc/self/cgroup"] = "0::/docker/abc\n";
    fs.files["/sys/fs/cgroup/docker/abc/memory.max"]     = "2147483648\n";   // docker --memory 2g
    fs.files["/sys/fs/cgroup/docker/abc/memory.current"] = "536870912\n";    // 512 MiB charged
    fs.files["/sys/fs/cgroup/docker/abc/memory.stat"]    = "file 268435456\nshmem 0\n"; // 256 MiB of it page cache
    m = hm::linux_host_memory(fs.reader());
    CHECK(m.total && *m.total == 2 * GiB, "cpu mem: container limit caps total (was the whole host)");
    CHECK(m.free && *m.free == 2 * GiB - 256 * MiB, "cpu mem: free = limit - (current - reclaimable cache)");

    fs.files["/sys/fs/cgroup/docker/memory.max"]     = "1073741824\n";       // tighter parent slice, 1 GiB
    fs.files["/sys/fs/cgroup/docker/memory.current"] = "0\n";
    m = hm::linux_host_memory(fs.reader());
    CHECK(m.total && *m.total == 1 * GiB && m.free && *m.free == 1 * GiB, "cpu mem: a tighter ANCESTOR limit wins for total and free");

    fs.files["/sys/fs/cgroup/docker/memory.max"]     = "max\n";
    fs.files["/sys/fs/cgroup/docker/abc/memory.max"] = "max\n";
    m = hm::linux_host_memory(fs.reader());
    CHECK(m.total && *m.total == 32 * GiB && m.free && *m.free == 16 * GiB, "cpu mem: memory.max = max at every level -> host values");

    fake_fs old_kernel;   // Linux < 3.14: no MemAvailable
    old_kernel.files["/proc/meminfo"] = "MemTotal: 8388608 kB\nMemFree: 1048576 kB\nBuffers: 524288 kB\nCached: 2097152 kB\n";
    m = hm::linux_host_memory(old_kernel.reader());
    CHECK(m.free && *m.free == 3584 * MiB, "cpu mem: no MemAvailable -> MemFree + Buffers + Cached");
    CHECK(!lb::meminfo_available(old_kernel.files["/proc/meminfo"]).has_value(), "cpu mem: the loader budget stays strict (no estimate)");

    fake_fs over;          // charged above its limit: 0 free, not a wrap-around
    over.files["/proc/meminfo"] = "MemTotal: 33554432 kB\nMemAvailable: 16777216 kB\n";
    over.files["/proc/self/cgroup"] = "0::/x\n";
    over.files["/sys/fs/cgroup/x/memory.max"]     = "1073741824\n";
    over.files["/sys/fs/cgroup/x/memory.current"] = "2147483648\n";
    m = hm::linux_host_memory(over.reader());
    CHECK(m.free && *m.free == 0 && m.total && *m.total == 1 * GiB, "cpu mem: charged above the limit -> free 0");

    // Docker on cgroup v2 gives the container its own cgroup namespace: /proc/self/cgroup reads "0::/" and the limit is
    // in /sys/fs/cgroup/memory.max itself (measured: docker run --memory 2g debian:bookworm-slim).
    fake_fs ns;
    ns.files["/proc/meminfo"] = "MemTotal: 92160000 kB\nMemAvailable: 90000000 kB\n";
    ns.files["/proc/self/cgroup"] = "0::/\n";
    ns.files["/sys/fs/cgroup/memory.max"]     = "2147483648\n";
    ns.files["/sys/fs/cgroup/memory.current"] = "2183168\n";
    ns.files["/sys/fs/cgroup/memory.stat"]    = "file 0\nshmem 0\n";
    m = hm::linux_host_memory(ns.reader());
    CHECK(m.total && *m.total == 2 * GiB, "cpu mem: namespaced container (0::/) -> root memory.max caps total");
    CHECK(m.free && *m.free == 2 * GiB - 2183168ull, "cpu mem: namespaced container -> free = limit - current");
    auto nsa = lb::linux_allocatable_bytes(ns.reader());
    CHECK(nsa && *nsa == 2 * GiB - 2183168ull, "walk: namespaced container (0::/) bounds the loader budget too");
    ns.files.erase("/sys/fs/cgroup/memory.max");   // a real host root has no memory.max
    m = hm::linux_host_memory(ns.reader());
    CHECK(m.total && *m.total == 92160000ull * KiB, "cpu mem control: 0::/ with no root memory.max -> host MemTotal");

    fake_fs none;
    m = hm::linux_host_memory(none.reader());
    CHECK(!m.total && !m.free, "cpu mem: nothing readable -> both unknown (caller keeps sysconf)");
}

static void test_budget_math() {
    CHECK(lb::budget_bytes(std::nullopt) == lb::UNLIMITED, "budget: probe failed -> UNLIMITED (today's behaviour)");
    CHECK(lb::budget_bytes(3 * GiB) == 2 * GiB, "budget: 2/3 of what the OS can allocate");
    CHECK(lb::budget_bytes(0) == 0, "budget: nothing available -> 0 bytes");
    CHECK(lb::budget_bytes(UINT64_MAX) != lb::UNLIMITED && lb::budget_bytes(UINT64_MAX) > 0, "budget: no overflow at the top of the range");

    CHECK(lb::admit(0, 10 * GiB, 1 * GiB), "admit: an empty budget always admits ONE tensor, even larger than the budget");
    CHECK(lb::admit(0, 10 * GiB, 0), "admit: budget 0 still admits one (cap at 1, never deadlock)");
    CHECK(!lb::admit(512 * MiB, 600 * MiB, 1 * GiB), "admit: second tensor that would exceed the budget waits");
    CHECK(lb::admit(512 * MiB, 512 * MiB, 1 * GiB), "admit control: second tensor that fits is admitted");
    CHECK(lb::admit(UINT64_MAX - 1, UINT64_MAX - 1, lb::UNLIMITED), "admit: UNLIMITED admits everything");
    CHECK(!lb::admit(1, UINT64_MAX, 1 * GiB), "admit: used + n overflow does not wrap to 'fits'");

    CHECK(lb::max_parallel(2 * GiB, 512 * MiB, 8) == 4, "parallel: 2 GiB budget / 512 MiB tensors = 4");
    CHECK(lb::max_parallel(100 * MiB, 512 * MiB, 8) == 1, "parallel: RAM short -> capped at 1");
    CHECK(lb::max_parallel(0, 512 * MiB, 8) == 1, "parallel: zero budget -> 1, not 0");
    CHECK(lb::max_parallel(lb::UNLIMITED, 512 * MiB, 8) == 8, "parallel control: unlimited -> every worker");
    CHECK(lb::max_parallel(64 * GiB, 1 * MiB, 8) == 8, "parallel control: huge budget is still capped at the worker count");
    CHECK(lb::max_parallel(1 * GiB, 0, 8) == 8, "parallel: empty tensors never limit");
}

// 8 workers each hold `n` bytes for a while; returns the peak number of concurrent holders.
static int run_workers(uint64_t budget, uint64_t n, int n_workers) {
    lb::semaphore sem(budget);
    std::atomic<int> live{0};
    std::atomic<int> peak{0};
    std::atomic<int> arrived{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < n_workers; ++t) {
        pool.emplace_back([&] {
            lb::semaphore::guard g(sem, n);
            const int now = ++live;
            int p = peak.load();
            while (now > p && !peak.compare_exchange_weak(p, now)) {}
            ++arrived;
            // hold until everyone that CAN be admitted is in (bounded wait so a budget of 1 cannot hang)
            const auto t0 = std::chrono::steady_clock::now();
            while (arrived.load() < n_workers && std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(150)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            --live;
        });
    }
    for (auto & th : pool) { th.join(); }
    if (sem.used() != 0) { std::printf("FAIL: semaphore leaked %llu bytes\n", (unsigned long long) sem.used()); g_fail = 1; }
    return peak.load();
}

static void test_semaphore_threads() {
    const int unl = run_workers(lb::UNLIMITED, 512 * MiB, 8);
    std::printf("      peak holders unlimited = %d\n", unl);
    CHECK(unl == 8, "threads control: unlimited -> all 8 workers load at once (today's behaviour)");
    const int two = run_workers(1 * GiB, 512 * MiB, 8);
    std::printf("      peak holders 1 GiB / 512 MiB = %d\n", two);
    CHECK(two == 2, "threads: 1 GiB budget, 512 MiB tensors -> at most 2 in flight");
    const int one = run_workers(100 * MiB, 512 * MiB, 8);
    std::printf("      peak holders short RAM = %d\n", one);
    CHECK(one == 1, "threads: RAM short -> exactly 1 in flight, and every worker still finishes");
}

static void test_live_probe() {
    auto a = lb::host_allocatable_bytes();
    std::printf("      live probe: %s %.1f MiB\n", a ? "available" : "FAILED", a ? (double) *a / (double) MiB : 0.0);
#if defined(_WIN32) || defined(__linux__)
    CHECK(a.has_value() && *a > 0, "live probe: this host reports allocatable RAM (GlobalMemoryStatusEx / meminfo+cgroup)");
#else
    CHECK(!a.has_value(), "live probe: other platforms report nothing -> unlimited");
#endif
}

int main() {
    test_meminfo();
    test_cgroup_parse();
    test_linux_walk();
    test_cpu_device_memory();
    test_budget_math();
    test_semaphore_threads();
    test_live_probe();
    std::printf(g_fail ? "test-load-budget: FAILED\n" : "test-load-budget: OK\n");
    return g_fail;
}
