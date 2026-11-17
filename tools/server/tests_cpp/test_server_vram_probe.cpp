// GATE-CLAIM-OK: unit test of the free-GPU-memory probes' parsing and of the dispatch in server-vram.cpp; it reads fixture trees it creates
// under the temp directory and makes no claim about the host's devices.
//
// Row S69-A-PICK-VENDOR-NEUTRAL-VRAM-TELEMETRY-FOR-SERVER-VRAM-AND-WATCHDOGS-2026-10-05-01.
// Usage: test-server-vram-probe            run every check (exit 0 = all passed)
//        test-server-vram-probe --live     also print what the probes read on THIS host, with and without NVML (no assertion on the values)
//
// Fail-before: the integration block (LLAMA_SERVER_VRAM_NO_NVML + LLAMA_SERVER_VRAM_DRM_ROOT) cannot pass on the NVML-only server-vram.cpp:
// it ignores both variables, so it returns the NVIDIA reading (or -1 without NVML) instead of the fixture sum.
#include "server-vram.h"
#include "server-vram-probe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
static int g_fail = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            ++g_fail;                                             \
            std::printf("FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                             \
            std::printf("\n");                                    \
        }                                                         \
    } while (0)

static void put(const fs::path & p, const std::string & s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << s;
}

static void setenv_(const char * k, const char * v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) {
        setenv(k, v, 1);
    } else {
        unsetenv(k);
    }
#endif
}

int main(int argc, char ** argv) {
    const bool live = argc > 1 && std::strcmp(argv[1], "--live") == 0;
    using namespace vram_probe;

    // ---- read_i64 / read_word --------------------------------------------------------------------------------------------------------
    const fs::path tmp = fs::temp_directory_path() / "vram_probe_fixture";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    put(tmp / "n1", "17163091968\n");
    put(tmp / "n2", "12 34\n");
    put(tmp / "n3", "");
    put(tmp / "n4", "-5\n");
    CHECK(read_i64(tmp / "n1") == 17163091968LL, "plain number with newline");
    CHECK(read_i64(tmp / "n2") == -1, "two tokens are not one number");
    CHECK(read_i64(tmp / "n3") == -1, "empty file");
    CHECK(read_i64(tmp / "n4") == -1, "sign is not a sysfs unsigned");
    CHECK(read_i64(tmp / "missing") == -1, "missing file");

    // ---- DRM sysfs fixture -----------------------------------------------------------------------------------------------------------
    const fs::path drm = tmp / "drm";
    auto card = [&](const std::string & name, const std::string & vendor, const std::string & total, const std::string & used,
                    const std::string & gtt_total = "", const std::string & gtt_used = "") {
        put(drm / name / "device" / "vendor", vendor + "\n");
        if (!total.empty()) {
            put(drm / name / "device" / "mem_info_vram_total", total + "\n");
            put(drm / name / "device" / "mem_info_vram_used", used + "\n");
        }
        if (!gtt_total.empty()) {
            put(drm / name / "device" / "mem_info_gtt_total", gtt_total + "\n");
            put(drm / name / "device" / "mem_info_gtt_used", gtt_used + "\n");
        }
    };
    card("card0", "0x1002", "17163091968", "1073741824");                                   // AMD discrete 16 GiB, 1 GiB used
    card("card0-DP-1", "0x1002", "999999999999", "1");                                      // connector node: never a device
    card("card1", "0x10DE", "34190917632", "2000000000");                                   // NVIDIA: NVML's job (upper-case hex in vendor)
    card("card2", "0x1002", "536870912", "100000000", "33554432000", "2000000000");         // AMD APU: 512 MiB carve-out + 31 GiB GTT
    card("card3", "0x8086", "", "");                                                        // driver exposes no vram files: skipped, not zero
    put(drm / "renderD128" / "device" / "vendor", "0x1002\n");
    put(drm / "version", "1\n");
    card("card10", "0x1002", "8589934592", "0");                                            // sorts after card2 only numerically: names sort as text

    const auto devs = scan_drm(drm.string());
    CHECK(devs.size() == 3, "expected card0, card10, card2 (3 devices), got %zu", devs.size());
    int64_t sum_total = 0, sum_free = 0;
    int n_unified = 0;
    for (const auto & d : devs) {
        sum_total += d.total;
        sum_free += d.free_bytes();
        n_unified += d.unified ? 1 : 0;
    }
    const int64_t apu_total = 536870912LL + 33554432000LL;
    CHECK(sum_total == 17163091968LL + 8589934592LL + apu_total, "total sum %lld", (long long) sum_total);
    CHECK(sum_free == (17163091968LL - 1073741824LL) + 8589934592LL + (apu_total - 100000000LL - 2000000000LL), "free sum %lld", (long long) sum_free);
    CHECK(n_unified == 1, "only the APU is unified, got %d", n_unified);
    CHECK(scan_drm((tmp / "nope").string()).empty(), "a missing root yields no devices");
    {
        // the NVIDIA skip is a parameter: with an empty skip set card1 appears (4 devices)
        CHECK(scan_drm(drm.string(), {}).size() == 4, "empty skip set keeps the NVIDIA card");
    }
    // a driver that reports used > total must not produce a negative free
    {
        device d;
        d.total = 10;
        d.used = 20;
        CHECK(d.free_bytes() == 0, "free is clamped at zero");
    }
    // two cardN names for ONE pci device are one device (symlink; skipped where the OS refuses the link)
    {
        const fs::path drm2 = tmp / "drm2";
        put(drm2 / "pci0" / "vendor", "0x1002\n");
        put(drm2 / "pci0" / "mem_info_vram_total", "4294967296\n");
        put(drm2 / "pci0" / "mem_info_vram_used", "0\n");
        std::error_code lk;
        fs::create_directories(drm2 / "card0");
        fs::create_directories(drm2 / "card1");
        fs::create_directory_symlink(drm2 / "pci0", drm2 / "card0" / "device", lk);
        std::error_code lk2;
        fs::create_directory_symlink(drm2 / "pci0", drm2 / "card1" / "device", lk2);
        if (lk || lk2) {
            std::printf("SKIP dedupe-by-symlink: the OS refused a directory symlink (%s)\n", (lk ? lk : lk2).message().c_str());
        } else {
            CHECK(scan_drm(drm2.string()).size() == 1, "two cardN names for one device count once");
        }
    }

    // ---- /proc/meminfo -------------------------------------------------------------------------------------------------------------
    const std::string mi = "MemTotal:       131072000 kB\nMemFree:         1000 kB\nMemAvailable:   65536000 kB\nHugePages_Total:       0\n";
    CHECK(meminfo_bytes(mi, "MemTotal") == 131072000LL * 1024, "MemTotal");
    CHECK(meminfo_bytes(mi, "MemAvailable") == 65536000LL * 1024, "MemAvailable");
    CHECK(meminfo_bytes(mi, "Missing") == -1, "absent key");
    CHECK(meminfo_bytes(mi, "HugePages_Total") == 0, "a count without a unit is taken as is");
    CHECK(nvml_memory_is_unified(3) && !nvml_memory_is_unified(0) && !nvml_memory_is_unified(2), "NVML NOT_SUPPORTED is code 3 only");

    // ---- ioreg / Apple budget ----------------------------------------------------------------------------------------------------------
    const std::string io = "  | |   \"model\" = \"Apple M2 Max\"\n  | |   \"PerformanceStatistics\" = {\"Device Utilization %\"=12,\"In use system memory\"=3221225472,\"Alloc system memory\"=9}\n";
    CHECK(parse_ioreg_in_use(io) == 3221225472LL, "ioreg in-use bytes");
    CHECK(parse_ioreg_in_use("nothing here") == -1, "ioreg without the key");
    CHECK(parse_ioreg_in_use("\"In use system memory\"=") == -1, "key with no digits");
    const int64_t gib = (int64_t) 1 << 30;
    CHECK(apple_gpu_budget(16 * gib, 0) == 16 * gib / 3 * 2, "16 GiB default = 2/3");
    CHECK(apple_gpu_budget(36 * gib, 0) == 36 * gib / 3 * 2, "36 GiB still 2/3");
    CHECK(apple_gpu_budget(64 * gib, 0) == 64 * gib / 4 * 3, "64 GiB default = 3/4");
    CHECK(apple_gpu_budget(64 * gib, 40000) == 40000LL * 1024 * 1024, "explicit wired limit wins");
    CHECK(apple_gpu_budget(16 * gib, 999999) == 16 * gib, "a wired limit above RAM is clamped to RAM");
    CHECK(apple_gpu_budget(0, 0) == -1, "no memsize, no budget");

    // ---- PDH instance names ------------------------------------------------------------------------------------------------------------
    uint64_t luid = 0;
    int phys = -1;
    CHECK(parse_pdh_instance("luid_0x00000000_0x0001A1A1_phys_0", luid, phys) && luid == 0x1A1A1ull && phys == 0, "PDH instance, upper-case hex");
    CHECK(parse_pdh_instance("luid_0x00000001_0x00000002_phys_1", luid, phys) && luid == 0x100000002ull && phys == 1, "high part is kept");
    CHECK(!parse_pdh_instance("pid_1234_luid_0x0_0x1_phys_0_eng_0", luid, phys), "an engine instance is not an adapter instance");
    CHECK(!parse_pdh_instance("luid_0x0_0x1_phys_0_extra", luid, phys), "trailing text is refused");

    // ---- integration: the dispatch in server-vram.cpp ----------------------------------------------------------------------------------
    setenv_("LLAMA_SERVER_VRAM_NO_NVML", "1");
    setenv_("LLAMA_SERVER_VRAM_DRM_ROOT", drm.string().c_str());
    CHECK(server_vram_total_bytes() == sum_total, "server_vram_total_bytes %lld, fixture %lld", (long long) server_vram_total_bytes(), (long long) sum_total);
    CHECK(server_vram_free_bytes() == sum_free, "server_vram_free_bytes %lld, fixture %lld", (long long) server_vram_free_bytes(), (long long) sum_free);
    setenv_("LLAMA_SERVER_VRAM_DRM_ROOT", (tmp / "empty-root").string().c_str());
    fs::create_directories(tmp / "empty-root");
    CHECK(server_vram_free_bytes() == -1, "no NVML and an empty DRM root is 'unknown' (-1), as before");

    if (live) {
        setenv_("LLAMA_SERVER_VRAM_DRM_ROOT", nullptr);
        setenv_("LLAMA_SERVER_VRAM_NO_NVML", nullptr);
        const int64_t t0 = server_vram_total_bytes(), f0 = server_vram_free_bytes();
        setenv_("LLAMA_SERVER_VRAM_NO_NVML", "1");
        const int64_t t1 = server_vram_total_bytes(), f1 = server_vram_free_bytes();
        std::printf("LIVE with NVML    : total %.2f GiB  free %.2f GiB\n", t0 / 1073741824.0, f0 / 1073741824.0);
        std::printf("LIVE without NVML : total %.2f GiB  free %.2f GiB (vendor-neutral probes of this OS)\n", t1 / 1073741824.0, f1 / 1073741824.0);
    }

    fs::remove_all(tmp, ec);
    std::printf(g_fail ? "FAILED: %d checks\n" : "all vram probe checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
