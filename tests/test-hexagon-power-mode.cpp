// Hexagon HTP power-mode name mapping (ggml/src/ggml-hexagon/htp-power-mode.h).
//
// The mapping is header-only and SDK-free precisely so it can be checked on any host; the
// DSP-side vote (htp/main.c) and the FastRPC call need a Snapdragon and are DEVICE-PENDING
// (compile-check with the Hexagon SDK; run: GGML_HEXAGON_POWER_MODE=balanced llama-cli ...
// and read the "power-mode N : dcvs-enable 1 ..." line from the DSP log).
//
// Ported behaviour: Qualcomm GenieX llama.cpp patch, BSD-3-Clause (see docs/THIRD_PARTY_NOTICES_E_SERVING.md). No code copied;
// the cases below were written from the behaviour list in the pick row.
#include "../ggml/src/ggml-hexagon/htp-power-mode.h"

#include <cstdio>
#include <cstring>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    // the enum is the wire format of htp_iface.start(power_mode): values are append-only
    CHECK(HTP_POWER_MODE_LOW_POWER_SAVER == 0);
    CHECK(HTP_POWER_MODE_BURST == 7);
    CHECK(HTP_POWER_MODE_COUNT == 8);

    // every canonical name round-trips
    const char * names[] = { "low_power_saver", "power_saver", "high_power_saver", "low_balanced",
                             "balanced", "high_performance", "sustained_high_performance", "burst" };
    for (int i = 0; i < 8; i++) {
        CHECK(htp_power_mode_from_str(names[i]) == i);
        CHECK(strcmp(htp_power_mode_to_str(i), names[i]) == 0);
    }

    // case-insensitive
    CHECK(htp_power_mode_from_str("BURST") == HTP_POWER_MODE_BURST);
    CHECK(htp_power_mode_from_str("Balanced") == HTP_POWER_MODE_BALANCED);
    CHECK(htp_power_mode_from_str("High_Performance") == HTP_POWER_MODE_HIGH_PERFORMANCE);

    // "default" aliases burst and never shadows it in the reverse lookup
    CHECK(htp_power_mode_from_str("default") == HTP_POWER_MODE_BURST);
    CHECK(strcmp(htp_power_mode_to_str(HTP_POWER_MODE_BURST), "burst") == 0);

    // must-fail partners: unknown / prefix / suffix / empty / null are rejected (-1), never "close enough"
    CHECK(htp_power_mode_from_str("turbo") == -1);
    CHECK(htp_power_mode_from_str("burs") == -1);
    CHECK(htp_power_mode_from_str("burst ") == -1);
    CHECK(htp_power_mode_from_str("power") == -1);
    CHECK(htp_power_mode_from_str("") == -1);
    CHECK(htp_power_mode_from_str(nullptr) == -1);

    // out-of-range value has a printable name
    CHECK(strcmp(htp_power_mode_to_str(HTP_POWER_MODE_COUNT), "unknown") == 0);
    CHECK(strcmp(htp_power_mode_to_str(-1), "unknown") == 0);

    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-hexagon-power-mode: all checks passed\n");
    return 0;
}
