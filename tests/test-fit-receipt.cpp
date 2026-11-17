// common_fit_receipt (common/fit.cpp): the one operator-visible block printed after common_fit_params.
// Row S69-ENGREV-ACCELERATE-FIT-RESULT-RECEIPT-2026-10-27-01. Every fit decision line is LOG_TRC, so a normal start printed nothing
// about what the fit chose; the host-resident bytes were never compared with free RAM at all. The formatter is pure so it runs with no GPU.
#include "fit.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static bool has(const std::string & s, const std::string & what) { return s.find(what) != std::string::npos; }

int main() {
    constexpr int64_t MiB = 1024 * 1024;
    const std::vector<common_fit_receipt_dev> devs = {
        { "CUDA0", 20000 * MiB, 3000 * MiB, 1024 * MiB },
        { "CUDA1", 9000 * MiB,  500 * MiB,  1024 * MiB },
    };

    // margin satisfied: host holds 4000 MiB of 64000 MiB free
    bool warn = true;
    const std::string ok = common_fit_receipt(33, 0, 16384, devs, 4000 * MiB, 64000 * MiB, 1024 * MiB, &warn);
    CHECK(!warn, "no host warning when the host-resident bytes fit in free RAM minus the margin");
    CHECK(has(ok, "fit:"), "every line carries the fit: tag");
    CHECK(has(ok, "n_gpu_layers=33"), "names the chosen n_gpu_layers: %s", ok.c_str());
    CHECK(has(ok, "auto -> 16384"), "context chosen from auto is shown: %s", ok.c_str());
    CHECK(has(ok, "CUDA0") && has(ok, "CUDA1"), "names every device");
    CHECK(has(ok, "20000 MiB") && has(ok, "3000 MiB") && has(ok, "9000 MiB") && has(ok, "500 MiB"), "per-device used and free in MiB: %s", ok.c_str());
    CHECK(has(ok, "4000 MiB") && has(ok, "64000 MiB"), "host resident and host free in MiB: %s", ok.c_str());
    CHECK(!has(ok, "WARN"), "no WARN line when the margin is satisfied: %s", ok.c_str());

    // host-resident above free minus margin: mmap paging thrash is predicted
    warn = false;
    const std::string bad = common_fit_receipt(10, 8192, 4096, devs, 63500 * MiB, 64000 * MiB, 1024 * MiB, &warn);
    CHECK(warn, "host warning flag set");
    CHECK(has(bad, "fit: WARN"), "a line starts with the WARN tag: %s", bad.c_str());
    CHECK(has(bad, "8192 -> 4096"), "context before and after: %s", bad.c_str());

    // boundary: resident == free - margin is NOT over the margin
    warn = true;
    const std::string edge = common_fit_receipt(10, 8192, 8192, devs, 62976 * MiB, 64000 * MiB, 1024 * MiB, &warn);
    CHECK(!warn && !has(edge, "WARN"), "resident exactly at free - margin does not warn: %s", edge.c_str());

    // a null warn pointer and an empty device list (CPU only) must not crash
    const std::string cpu = common_fit_receipt(0, 4096, 4096, {}, 100 * MiB, 200 * MiB, 0, nullptr);
    CHECK(has(cpu, "n_gpu_layers=0"), "cpu-only receipt: %s", cpu.c_str());

    // measured peak after warmup next to the projection (row S69-FIT-PROJECTED-VS-MEASURED-PEAK-NANO-VLLM-PICK-2026-09-27-01)
    {
        bool mw = true;
        const std::string same = common_fit_measured_report({ { "CUDA0", 1000 * MiB, 1000 * MiB }, { "CUDA1", 2000 * MiB, 1500 * MiB } }, &mw);
        CHECK(!mw, "measured at or below the projection does not warn: %s", same.c_str());
        CHECK(has(same, "CUDA0") && has(same, "CUDA1") && has(same, "1000 MiB") && has(same, "1500 MiB"),
              "both devices show projected and measured: %s", same.c_str());
        CHECK(!has(same, "WARN"), "no WARN line: %s", same.c_str());

        mw = true;
        const std::string edge5 = common_fit_measured_report({ { "CUDA0", 2000 * MiB, 2100 * MiB } }, &mw); // exactly +5 percent
        CHECK(!mw && !has(edge5, "WARN"), "exactly 5 percent over does not warn: %s", edge5.c_str());

        mw = false;
        const std::string over = common_fit_measured_report({ { "CUDA0", 2000 * MiB, 2101 * MiB } }, &mw); // just above 5 percent
        CHECK(mw && has(over, "fit: WARN") && has(over, "CUDA0"), "more than 5 percent over warns and names the device: %s", over.c_str());

        mw = false;
        const std::string zero = common_fit_measured_report({ { "CUDA0", 0, 10 * MiB } }, &mw);
        CHECK(mw, "a zero projection with measured use warns (no divide by zero): %s", zero.c_str());

        mw = true;
        const std::string none = common_fit_measured_report({}, &mw);
        CHECK(!mw && none.empty(), "no devices: empty report, no warn");
        CHECK(common_fit_measured_report({ { "CUDA0", 100 * MiB, 100 * MiB } }, nullptr).size() > 0, "null warn pointer is accepted");
    }

    if (g_fails) {
        std::printf("test-fit-receipt: %d failure(s)\n", g_fails);
        return 1;
    }
    std::printf("test-fit-receipt: all pass\n");
    return 0;
}
