#pragma once

// Video sampling planner: how many frames and at what resolution a clip may be read so that the vision tokens it produces fit a budget.
// Row S69-MNN-PICK-VIDEO-FRAME-CAP-AND-VISION-TOKEN-BUDGET-2026-10-05-01.
//
// Before this, tools/mtmd/mtmd-helper.cpp sampled every clip at --fps (default 4) at native resolution, so a 10 minute 1080p clip became
// 2400 frames and the only protection was the context-overflow error after all the decoding was done.
//
// SOURCE-READ (idea only; Apache-2.0): alibaba/MNN transformers/llm/engine/src/omni.cpp:97-189, 1349-1440, 1585-1659 -- video_max_frames trims the
// sampled list (aligned DOWN to the temporal patch), video_max_vision_tokens is divided over the temporal groups, converted to a per-frame pixel
// ceiling BEFORE the resize, and a request that still does not fit fails with a message instead of overflowing. This planner is written from
// that description with its own structure and names; no code copied.
//
// NUANCES THE ROW DID NOT SAY (found while writing the tests):
//   1. Their budget counts pre-merge patches; ours counts the tokens the LLM sees (items x tokens per item), because mtmd pairs two mergeable
//      frames into ONE item (tools/mtmd/mtmd.cpp:1115 mtmd_group_mergeable_bitmaps) and the 2x2 spatial merge happens inside the projector.
//   2. A frame cap must be applied in units of the temporal group, otherwise an odd count leaves a lone trailing frame that becomes its own
//      item and costs as much as a pair.
//   3. When the cap leaves fewer frames than one temporal group the clip is read as a single frame (never zero).
//   4. Shrinking is by whole cells (patch x merge pixels): an unaligned size would be rounded UP again by the model preprocessor and break
//      the budget the planner just computed. The caller can pass `tokens_of(w, h)` (a real tokenisation of a placeholder bitmap) so the plan
//      is verified against the actual preprocessor instead of this file's cell arithmetic.
//   5. Both budgets may be 0 (= unlimited, the old behaviour); an unknown duration cannot be planned, so the helper falls back to counting
//      frames while it reads and ends the stream at the cap.
//   6. A budget below min_cells_per_item for even one item is an error that carries the numbers, not a silent 1-token frame.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>

struct mtmd_video_plan_in {
    double  duration_s       = 0.0;   // > 0
    double  fps_target       = 4.0;   // requested sampling rate
    int32_t width            = 0;     // native frame size
    int32_t height           = 0;
    int32_t cell_px          = 28;    // patch size x spatial merge: the size granule the model preprocessor works in
    int32_t temporal_group   = 2;     // frames that become one item (2 for Qwen-VL style merging, 1 otherwise)
    int32_t max_frames       = 0;     // 0 = unlimited
    int64_t max_tokens       = 0;     // 0 = unlimited; counts LLM tokens over the whole clip
    int32_t min_cells_per_item = 4;   // never shrink an item below a 2x2 grid of cells
    // optional exact oracle: LLM tokens of ONE item at (w, h); when set the plan is verified and tightened against it
    std::function<int64_t(int32_t, int32_t)> tokens_of;
};

struct mtmd_video_plan_out {
    bool        ok          = true;
    std::string error;
    double      fps_eff     = 0.0;
    int32_t     width_eff   = 0;
    int32_t     height_eff  = 0;
    int32_t     n_frames    = 0;      // frames that will be read
    int32_t     n_items     = 0;      // items the LLM sees (frames / temporal_group, rounded up)
    int64_t     est_tokens  = 0;      // n_items x tokens per item at the planned size
    bool        limited     = false;  // true when a budget changed the sampling
};

namespace mtmd_video_plan_detail {

inline int32_t cells(int32_t px, int32_t cell) { return std::max<int32_t>(1, (px + cell - 1) / cell); }

inline int64_t item_tokens(const mtmd_video_plan_in & in, int32_t w, int32_t h) {
    if (in.tokens_of) {
        return in.tokens_of(w, h);
    }
    return (int64_t) cells(w, in.cell_px) * cells(h, in.cell_px);
}

} // namespace mtmd_video_plan_detail

inline mtmd_video_plan_out mtmd_video_plan(const mtmd_video_plan_in & in) {
    using namespace mtmd_video_plan_detail;
    mtmd_video_plan_out out;
    auto fail = [&](const char * what, long long a, long long b) {
        char buf[240];
        snprintf(buf, sizeof(buf), "%s (%lld vs %lld)", what, a, b);
        out.ok    = false;
        out.error = buf;
        return out;
    };
    if (in.duration_s <= 0.0 || in.fps_target <= 0.0 || in.width <= 0 || in.height <= 0 || in.cell_px <= 0 || in.temporal_group <= 0) {
        return fail("video plan input is not positive", (long long) in.width, (long long) in.height);
    }
    const int32_t tg = in.temporal_group;

    // 1. frames
    int64_t n = std::max<int64_t>(1, (int64_t) std::llround(in.duration_s * in.fps_target));
    int64_t n_raw = n;
    if (in.max_frames > 0 && n > in.max_frames) {
        n = in.max_frames;
        if (n >= tg) {
            n = (n / tg) * tg;  // whole temporal groups only (nuance 2)
        } else {
            n = 1;              // nuance 3
        }
    }

    // 2. token budget
    int32_t w = in.width;
    int32_t h = in.height;
    const int64_t min_item = std::max<int32_t>(1, in.min_cells_per_item);
    if (in.max_tokens > 0) {
        if (in.max_tokens < min_item) {
            return fail("vision token budget is below the smallest useful item", (long long) in.max_tokens, (long long) min_item);
        }
        int64_t items = (n + tg - 1) / tg;
        if (items * min_item > in.max_tokens) {                     // too many items even at the minimum size: fewer frames
            items = in.max_tokens / min_item;
            n     = std::max<int64_t>(1, std::min<int64_t>(n, items * tg));
            items = (n + tg - 1) / tg;
        }
        const int64_t allowance = in.max_tokens / items;           // tokens one item may cost
        const int32_t cw0 = cells(w, in.cell_px);
        const int32_t ch0 = cells(h, in.cell_px);
        if ((int64_t) cw0 * ch0 > allowance) {
            // shrink by whole cells keeping the aspect ratio; floor first, then spend any leftover on the dimension that is further below its ideal
            const double scale = std::sqrt((double) allowance / ((double) cw0 * ch0));
            int32_t cw = std::max<int32_t>(1, (int32_t) std::floor(cw0 * scale));
            int32_t ch = std::max<int32_t>(1, (int32_t) std::floor(ch0 * scale));
            while ((int64_t) cw * ch > allowance && (cw > 1 || ch > 1)) {
                if (cw >= ch && cw > 1) { cw--; } else if (ch > 1) { ch--; } else { cw--; }
            }
            for (;;) {
                const double ew = cw0 * scale - cw;
                const double eh = ch0 * scale - ch;
                const bool can_w = (int64_t) (cw + 1) * ch <= allowance && cw + 1 <= cw0;
                const bool can_h = (int64_t) cw * (ch + 1) <= allowance && ch + 1 <= ch0;
                if (can_w && (!can_h || ew >= eh)) { cw++; } else if (can_h) { ch++; } else { break; }
            }
            w = cw * in.cell_px;
            h = ch * in.cell_px;
        }
        // verify against the exact oracle when there is one (nuance 4)
        for (int guard = 0; in.tokens_of && guard < 64 && items * item_tokens(in, w, h) > in.max_tokens; ++guard) {
            const int32_t cw = cells(w, in.cell_px);
            const int32_t ch = cells(h, in.cell_px);
            if (cw <= 1 && ch <= 1) {
                break;
            }
            if (cw >= ch && cw > 1) { w = (cw - 1) * in.cell_px; } else { h = (ch - 1) * in.cell_px; }
        }
        if (in.tokens_of && items * item_tokens(in, w, h) > in.max_tokens) {
            return fail("vision tokens still exceed the budget at the smallest size", (long long) (items * item_tokens(in, w, h)), (long long) in.max_tokens);
        }
    }

    out.n_frames   = (int32_t) n;
    out.n_items    = (int32_t) ((n + tg - 1) / tg);
    out.width_eff  = w;
    out.height_eff = h;
    out.fps_eff    = (double) n / in.duration_s;
    out.est_tokens = (int64_t) out.n_items * item_tokens(in, w, h);
    out.limited    = n != n_raw || w != in.width || h != in.height;
    return out;
}
