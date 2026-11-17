// test-vram-spill: CPU-only checks of the VRAM-spill placement policy (ggml/src/ggml-vram-spill.cpp).
// Standalone: compiles the policy source directly, links no ggml/llama library.
//   - default policy is OFF and every size is placed on the device (bit-identical to a build without the feature)
//   - mode names parse; junk is rejected
//   - threshold boundary is inclusive (size >= threshold), as in the spec in docs/agent-reports/link-vram-spill-1014.md
//   - managed / hostmap send >= threshold to the spill tier regardless of free VRAM
//   - auto spills only when the allocation does not fit in (free - reserve) AND is >= threshold
//   - the free-VRAM reserve cannot underflow
//   - counters record each decision
#include "ggml-vram-spill.h"

#include <cstdio>
#include <cstdint>
#include <string_view>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static const uint64_t MiB = 1024ull * 1024ull;
static const uint64_t GiB = 1024ull * MiB;

int main() {
    // ---- default is OFF and a pure pass-through
    {
        ggml_vram_spill_policy p = ggml_vram_spill_get_policy();
        CHECK(p.mode == GGML_VRAM_SPILL_OFF);
        const uint64_t sizes[] = {0, 1, 255 * MiB, 256 * MiB, 40 * GiB, UINT64_MAX};
        for (uint64_t s : sizes) {
            CHECK(ggml_vram_spill_choose(&p, s, 0) == GGML_VRAM_SPILL_PLACE_DEVICE);
            CHECK(ggml_vram_spill_choose(&p, s, 100 * GiB) == GGML_VRAM_SPILL_PLACE_DEVICE);
        }
    }

    // ---- mode parsing
    {
        ggml_vram_spill_mode m = GGML_VRAM_SPILL_AUTO;
        CHECK(ggml_vram_spill_parse_mode("off", &m) && m == GGML_VRAM_SPILL_OFF);
        CHECK(ggml_vram_spill_parse_mode("managed", &m) && m == GGML_VRAM_SPILL_MANAGED);
        CHECK(ggml_vram_spill_parse_mode("hostmap", &m) && m == GGML_VRAM_SPILL_HOSTMAP);
        CHECK(ggml_vram_spill_parse_mode("auto", &m) && m == GGML_VRAM_SPILL_AUTO);
        m = GGML_VRAM_SPILL_HOSTMAP;
        CHECK(!ggml_vram_spill_parse_mode("bogus", &m) && m == GGML_VRAM_SPILL_HOSTMAP); // untouched on failure
        CHECK(!ggml_vram_spill_parse_mode("", &m));
        CHECK(!ggml_vram_spill_parse_mode(nullptr, &m));
        CHECK(std::string_view(ggml_vram_spill_mode_name(GGML_VRAM_SPILL_AUTO)) == "auto");
    }

    // ---- managed: inclusive threshold, free VRAM irrelevant
    {
        ggml_vram_spill_policy p = {GGML_VRAM_SPILL_MANAGED, 256 * MiB, 0};
        CHECK(ggml_vram_spill_choose(&p, 256 * MiB - 1, 100 * GiB) == GGML_VRAM_SPILL_PLACE_DEVICE);
        CHECK(ggml_vram_spill_choose(&p, 256 * MiB,     100 * GiB) == GGML_VRAM_SPILL_PLACE_MANAGED);
        CHECK(ggml_vram_spill_choose(&p, 8 * GiB,       0)         == GGML_VRAM_SPILL_PLACE_MANAGED);
    }

    // ---- hostmap
    {
        ggml_vram_spill_policy p = {GGML_VRAM_SPILL_HOSTMAP, 64 * MiB, 0};
        CHECK(ggml_vram_spill_choose(&p, 64 * MiB - 1, 0) == GGML_VRAM_SPILL_PLACE_DEVICE);
        CHECK(ggml_vram_spill_choose(&p, 64 * MiB,     0) == GGML_VRAM_SPILL_PLACE_HOSTMAP);
    }

    // ---- auto: fits -> device; does not fit and big enough -> hostmap; small never spills
    {
        ggml_vram_spill_policy p = {GGML_VRAM_SPILL_AUTO, 256 * MiB, 1 * GiB};
        // free 10 GiB, reserve 1 GiB -> 9 GiB usable
        CHECK(ggml_vram_spill_choose(&p, 9 * GiB,     10 * GiB) == GGML_VRAM_SPILL_PLACE_DEVICE);   // exactly fits
        CHECK(ggml_vram_spill_choose(&p, 9 * GiB + 1, 10 * GiB) == GGML_VRAM_SPILL_PLACE_HOSTMAP);  // one byte over
        CHECK(ggml_vram_spill_choose(&p, 100 * MiB,   0)        == GGML_VRAM_SPILL_PLACE_DEVICE);   // below threshold: never spill
        CHECK(ggml_vram_spill_choose(&p, 256 * MiB,   0)        == GGML_VRAM_SPILL_PLACE_HOSTMAP);
        // free < reserve must not underflow into "huge usable"
        CHECK(ggml_vram_spill_choose(&p, 2 * GiB, 512 * MiB) == GGML_VRAM_SPILL_PLACE_HOSTMAP);
        // fits in raw free VRAM but free < reserve: usable is 0, so it must still spill
        CHECK(ggml_vram_spill_choose(&p, 400 * MiB, 512 * MiB) == GGML_VRAM_SPILL_PLACE_HOSTMAP);
    }

    // ---- configure / get round trip and counters
    {
        ggml_vram_spill_policy p = {GGML_VRAM_SPILL_AUTO, 128 * MiB, 2 * GiB};
        ggml_vram_spill_configure(&p);
        ggml_vram_spill_policy q = ggml_vram_spill_get_policy();
        CHECK(q.mode == p.mode && q.threshold_bytes == p.threshold_bytes && q.reserve_bytes == p.reserve_bytes);

        ggml_vram_spill_stats_reset();
        ggml_vram_spill_record(GGML_VRAM_SPILL_PLACE_DEVICE,  10);
        ggml_vram_spill_record(GGML_VRAM_SPILL_PLACE_HOSTMAP, 300 * MiB);
        ggml_vram_spill_record(GGML_VRAM_SPILL_PLACE_HOSTMAP, 100 * MiB);
        ggml_vram_spill_record(GGML_VRAM_SPILL_PLACE_MANAGED, 5);
        ggml_vram_spill_stats s = ggml_vram_spill_get_stats();
        CHECK(s.n_device == 1 && s.bytes_device == 10);
        CHECK(s.n_hostmap == 2 && s.bytes_hostmap == 400 * MiB);
        CHECK(s.n_managed == 1 && s.bytes_managed == 5);

        // configuring OFF restores pass-through
        ggml_vram_spill_policy off = {GGML_VRAM_SPILL_OFF, 0, 0};
        ggml_vram_spill_configure(&off);
        ggml_vram_spill_policy r = ggml_vram_spill_get_policy();
        CHECK(ggml_vram_spill_choose(&r, 40 * GiB, 0) == GGML_VRAM_SPILL_PLACE_DEVICE);
    }

    if (g_fail) { std::printf("test-vram-spill: %d FAILED\n", g_fail); return 1; }
    std::printf("test-vram-spill: all checks passed\n");
    return 0;
}
