// Name <-> value mapping for the Hexagon HTP session power mode (enum htp_power_mode).
//
// Header-only and free of any Hexagon SDK include, so it compiles on every host and can be
// unit-tested without a Snapdragon (tests/test-hexagon-power-mode.cpp).
//
// Behaviour ported from the Qualcomm GenieX llama.cpp patch (BSD-3-Clause, Qualcomm
// Technologies; see docs/THIRD_PARTY_NOTICES_E_SERVING.md): case-insensitive names, "default" is an alias for the mode the
// backend historically used (burst), an unknown name is reported to the caller as -1 so the
// caller can warn and keep its default.
#pragma once

#include <cstdint>
#include "htp/htp-ops.h"

#include <cstring>

struct htp_power_mode_name {
    const char * name;
    int          mode;
};

// "default" aliases burst. Change that one row to change the default mode.
static const htp_power_mode_name htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_LOW_POWER_SAVER            },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER                },
    { "high_power_saver",           HTP_POWER_MODE_HIGH_POWER_SAVER           },
    { "low_balanced",               HTP_POWER_MODE_LOW_BALANCED               },
    { "balanced",                   HTP_POWER_MODE_BALANCED                   },
    { "high_performance",           HTP_POWER_MODE_HIGH_PERFORMANCE           },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_HIGH_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST                      },
    { "default",                    HTP_POWER_MODE_BURST                      },
};

static inline bool htp_power_mode_name_eq(const char * a, const char * b) {
    for (; *a && *b; ++a, ++b) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char) (ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char) (cb - 'A' + 'a');
        if (ca != cb) return false;
    }
    return *a == *b;
}

// Returns the mode value, or -1 for a null / unknown name.
static inline int htp_power_mode_from_str(const char * s) {
    if (!s) {
        return -1;
    }
    for (const auto & e : htp_power_mode_names) {
        if (htp_power_mode_name_eq(s, e.name)) {
            return e.mode;
        }
    }
    return -1;
}

// First name registered for the value ("burst", never the "default" alias), or "unknown".
static inline const char * htp_power_mode_to_str(int mode) {
    for (const auto & e : htp_power_mode_names) {
        if (e.mode == mode) {
            return e.name;
        }
    }
    return "unknown";
}
