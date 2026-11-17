#pragma once

// Repetition / frequency / presence penalty with a linear fade-out past the window.
//
// Row S69-EXLLAMAV3-PICK-PENALTY-FADE-2026-10-18-01. ExLlamaV3 (turboderp-org/exllamav3, MIT,
// sha fa714d0a917297f9647b4353e915113e9358e422) applies its penalties over a "sustain" range at full
// strength and then over a "decay" range whose strength falls linearly to zero, so a token does not
// flip from fully penalised to free the moment it leaves the window.
//
// SOURCE-READ (rule 1.87): exllamav3/exllamav3_ext/generator/rep_pen.cu
//   :47-66   window and factor: only past tokens with dist = past_len - i < sustain + decay count;
//            factor = 1 for dist <= sustain, else clamp(1 - (dist - sustain) / decay, 0, 1); a token
//            seen several times keeps its LARGEST factor (shmemAtomicMaxF).
//   :71-84   repetition: w = v > 0 ? v / rep : v * rep, out = v * (1 - f) + w * f (a linear blend
//            between the unpenalised and the fully penalised logit).
//   :163-199 presence / frequency: frequency subtracts freq * SUM of the factors (each occurrence
//            counts by its own fade), presence subtracts pres * MAX factor.
// TabbyAPI (theroyallab/tabbyAPI, AGPL-3.0, read for the parameter names only, no code copied)
// backends/exllamav3/sampler.py:57-69 maps penalty_range -> sustain and repetition_decay -> decay.
// This is a fresh C++ rewrite of those semantics for the host sampling path; no code copied.
//
// Mapping in this fork: sustain = the existing repeat_last_n window, decay = the new penalty_decay.
// penalty_decay = 0 is the old uniform window, bit for bit (the caller does not enter this path).

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

struct common_penalty_fade_weight {
    float f_max = 0.0f; // largest fade factor among this token's occurrences (repetition, presence)
    float f_sum = 0.0f; // sum of fade factors over its occurrences (frequency)
};

// Fade factor of an occurrence `dist` tokens back (1 = the most recent token).
inline float common_penalty_fade_factor(int64_t dist, int32_t sustain, int32_t decay) {
    if (dist <= sustain) {
        return 1.0f;
    }
    if (decay <= 0) {
        return 0.0f;
    }
    const float f = 1.0f - (float) (dist - sustain) / (float) decay;
    return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
}

// Per-token weights over the newest sustain + decay entries of hist[0..n_hist) (oldest first).
inline std::unordered_map<llama_token, common_penalty_fade_weight> common_penalty_fade_weights(
        const llama_token * hist, size_t n_hist, int32_t sustain, int32_t decay) {
    std::unordered_map<llama_token, common_penalty_fade_weight> w;
    if (hist == nullptr || n_hist == 0 || sustain < 0) {
        return w;
    }
    // Only dist <= sustain + decay can have a non-zero factor (the one at exactly sustain + decay is 0).
    // NOTE rep_pen.cu:50 skips dist >= sustain + decay, which with decay = 0 also drops the token at
    // dist == sustain (a window one short); TabbyAPI works around it (sampler.py:69). Here decay = 0 never
    // reaches this function, and the factor test below is the only gate, so there is no such edge.
    const int64_t reach = (int64_t) sustain + (decay > 0 ? decay : 0);
    const size_t first = (int64_t) n_hist > reach ? n_hist - (size_t) reach : 0;
    for (size_t i = first; i < n_hist; ++i) {
        const int64_t dist = (int64_t) (n_hist - i);
        const float f = common_penalty_fade_factor(dist, sustain, decay);
        if (f <= 0.0f) {
            continue;
        }
        auto & e = w[hist[i]];
        e.f_max = f > e.f_max ? f : e.f_max;
        e.f_sum += f;
    }
    return w;
}

// Applies the faded penalties to every candidate in cur_p. Returns the number of candidates changed.
inline size_t common_penalty_fade_apply(llama_token_data_array * cur_p, const llama_token * hist, size_t n_hist,
        int32_t sustain, int32_t decay, float penalty_repeat, float penalty_freq, float penalty_present) {
    if (cur_p == nullptr) {
        return 0;
    }
    const auto w = common_penalty_fade_weights(hist, n_hist, sustain, decay);
    if (w.empty()) {
        return 0;
    }
    size_t changed = 0;
    for (size_t k = 0; k < cur_p->size; ++k) {
        auto it = w.find(cur_p->data[k].id);
        if (it == w.end()) {
            continue;
        }
        const float f = it->second.f_max;
        float v = cur_p->data[k].logit;
        if (penalty_repeat != 1.0f) {
            const float full = v > 0.0f ? v / penalty_repeat : v * penalty_repeat;
            v = v * (1.0f - f) + full * f;
        }
        v -= penalty_freq * it->second.f_sum + penalty_present * f;
        cur_p->data[k].logit = v;
        ++changed;
    }
    if (changed > 0) {
        cur_p->sorted = false;
    }
    return changed;
}
