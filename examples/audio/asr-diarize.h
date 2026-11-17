// asr-diarize.h -- stereo-energy speaker estimate for channel-separated recordings.
//
// SOURCE-READ (MIT): ggml-org/whisper.cpp examples/cli/cli.cpp:316-351 (estimate_diarization_speaker): sum |x| of
// each channel over the segment; one channel must exceed the other by 10% to win, otherwise "?".
// Meaningful only when the speakers are on SEPARATE channels (phone-call style recordings); on a
// normal stereo mix it reports "?" or an arbitrary channel. Written independently.
#pragma once

#include <cmath>
#include <cstddef>
#include <string>

// "0" (left louder by > 10%), "1" (right louder by > 10%), "?" otherwise (including an empty range).
inline std::string asr_stereo_speaker(const float * left, const float * right, size_t i0, size_t i1) {
    double e0 = 0.0, e1 = 0.0;
    for (size_t j = i0; j < i1; j++) {
        e0 += std::fabs((double) left[j]);
        e1 += std::fabs((double) right[j]);
    }
    if (e0 > 1.1 * e1) { return "0"; }
    if (e1 > 1.1 * e0) { return "1"; }
    return "?";
}
