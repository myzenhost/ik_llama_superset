#pragma once

// Normalise per-device split weights into cumulative split points in (0, 1].
//
// The default weights are each device's free memory minus a safety margin, clamped to 0 when free is below the
// margin (src/llama.cpp, load_tensors). Under Windows/WDDM a card that is already full commonly reports free == 0,
// so on a multi-GPU run every weight can be 0 -- exactly when a draft model loads second, after the target has
// filled VRAM. The old normalisation then divided 0 by 0: every split point became NaN, std::upper_bound over NaNs
// returned past-the-end, and devices.at() threw ("invalid vector subscript" on MSVC).
// Upstream report and fix: ggml-org/llama.cpp#24795 / #27647 (read 2026-09-29, same arithmetic as ours).
// Our fix: when the weights do not sum to a positive finite value, fall back to an even split and say so.
//
// Returns true when the weights were used as given, false when the even-split fallback was taken.

#include <cmath>
#include <vector>

static inline bool llama_split_points_from_weights(std::vector<float> & splits) {
    const size_t n = splits.size();
    if (n == 0) {
        return true;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += splits[i];
        splits[i] = sum;
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        for (size_t i = 0; i < n; ++i) {
            splits[i] = float(i + 1) / float(n);
        }
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        splits[i] /= sum;
    }
    return true;
}
