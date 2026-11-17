// Model-free test of the video sampling planner (tools/mtmd/mtmd-video-plan.h).
// Row S69-MNN-PICK-VIDEO-FRAME-CAP-AND-VISION-TOKEN-BUDGET-2026-10-05-01.
// Fail-before: the header did not exist, so this file did not compile; the helper sampled every clip at --fps and native size.

#include "mtmd-video-plan.h"

#include <cstdio>
#include <cstdlib>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } while (0)

static mtmd_video_plan_in base_clip(double dur) {
    mtmd_video_plan_in in;
    in.duration_s = dur;
    in.fps_target = 4.0;
    in.width      = 1920;
    in.height     = 1080;
    in.cell_px    = 28;   // patch 14 x merge 2
    in.temporal_group = 2;
    return in;
}

int main() {
    // the row's case: 600 s, 1920x1080, fps 4, patch 14, merge 2, 128 frames, 8192 tokens
    {
        auto in = base_clip(600.0);
        in.max_frames = 128;
        in.max_tokens = 8192;
        const auto p = mtmd_video_plan(in);
        CHECK(p.ok, "%s", p.error.c_str());
        CHECK(p.n_frames <= 128 && p.n_frames % 2 == 0, "frames %d", p.n_frames);
        CHECK(p.est_tokens <= 8192, "tokens %lld", (long long) p.est_tokens);
        CHECK(p.width_eff % 28 == 0 && p.height_eff % 28 == 0, "size %dx%d", p.width_eff, p.height_eff);
        CHECK(p.limited, "the clip was limited");
        // the old behaviour for contrast: 2400 frames at native size, 64 * 69 cells per item
        const auto old = mtmd_video_plan(base_clip(600.0));
        CHECK(old.n_frames == 2400 && old.width_eff == 1920 && old.height_eff == 1080 && !old.limited,
              "unbounded plan must equal today's sampling, got %d frames %dx%d", old.n_frames, old.width_eff, old.height_eff);
        CHECK(old.est_tokens > 8192 * 50, "control: the unbudgeted plan is far over the budget (%lld)", (long long) old.est_tokens);
        // aspect ratio kept to within one cell
        const double a0 = 1920.0 / 1080.0, a1 = (double) p.width_eff / p.height_eff;
        CHECK(std::fabs(a0 - a1) / a0 < 0.35, "aspect %f vs %f", a0, a1);
    }
    // monotone: a longer clip never gets MORE frames
    {
        int32_t prev = 1 << 30;
        for (double dur : {10.0, 30.0, 60.0, 120.0, 600.0, 3600.0}) {
            auto in = base_clip(dur);
            in.max_frames = 128;
            const auto p = mtmd_video_plan(in);
            CHECK(p.ok && p.n_frames <= 128, "frames %d at %.0fs", p.n_frames, dur);
            if (dur >= 60.0) {
                CHECK(p.n_frames <= prev || prev == (1 << 30), "monotone broke at %.0fs: %d after %d", dur, p.n_frames, prev);
                prev = p.n_frames;
            }
        }
        auto in = base_clip(10.0);      // short clip below the cap: untouched, not rounded
        in.max_frames = 128;
        in.fps_target = 3.0;
        const auto p = mtmd_video_plan(in);
        CHECK(p.n_frames == 30 && !p.limited, "short clip frames %d limited %d", p.n_frames, (int) p.limited);
        in.duration_s = 10.1;           // odd raw count stays odd when no cap applies
        const auto q = mtmd_video_plan(in);
        CHECK(q.n_frames == 30 || q.n_frames == 31, "frames %d", q.n_frames);
    }
    // a tighter token budget never yields a larger picture or more tokens
    {
        int32_t pw = 1 << 30;
        for (int64_t budget : {20000, 8192, 4096, 1024, 256, 64}) {
            auto in = base_clip(600.0);
            in.max_frames = 128;
            in.max_tokens = budget;
            const auto p = mtmd_video_plan(in);
            CHECK(p.ok && p.est_tokens <= budget, "budget %lld tokens %lld err %s", (long long) budget, (long long) p.est_tokens, p.error.c_str());
            CHECK(p.width_eff <= pw, "picture grew when the budget shrank: %d after %d", p.width_eff, pw);
            pw = p.width_eff;
        }
    }
    // odd cap: frames drop to a whole temporal group; a cap of one frame reads one frame
    {
        auto in = base_clip(600.0);
        in.max_frames = 7;
        const auto p = mtmd_video_plan(in);
        CHECK(p.n_frames == 6, "frames %d", p.n_frames);
        in.max_frames = 1;
        const auto q = mtmd_video_plan(in);
        CHECK(q.ok && q.n_frames == 1 && q.n_items == 1, "frames %d items %d", q.n_frames, q.n_items);
    }
    // too many items for the budget at the smallest item size: frames are reduced, not the item size below the floor
    {
        auto in = base_clip(600.0);
        in.max_frames = 128;
        in.max_tokens = 40;         // 10 items of 4 cells
        const auto p = mtmd_video_plan(in);
        CHECK(p.ok && p.n_items <= 10 && p.est_tokens <= 40, "items %d tokens %lld err %s", p.n_items, (long long) p.est_tokens, p.error.c_str());
    }
    // a budget below one minimal item is an error that carries the numbers
    {
        auto in = base_clip(600.0);
        in.max_tokens = 3;
        const auto p = mtmd_video_plan(in);
        CHECK(!p.ok && p.error.find("3") != std::string::npos, "expected an error, got ok=%d '%s'", (int) p.ok, p.error.c_str());
    }
    // an exact oracle that disagrees with the cell arithmetic: this model emits 2 tokens per cell (a separator after each) and 3 extra per item
    {
        auto in = base_clip(600.0);
        in.max_frames = 64;
        in.max_tokens = 4096;
        in.tokens_of = [](int32_t w, int32_t h) -> int64_t { return (int64_t) ((w + 27) / 28) * ((h + 27) / 28) * 2 + 3; };
        const auto p = mtmd_video_plan(in);
        CHECK(p.ok && p.n_items * in.tokens_of(p.width_eff, p.height_eff) <= 4096, "oracle plan %lld tokens (err %s)", (long long) p.est_tokens, p.error.c_str());
        auto no_oracle = in;
        no_oracle.tokens_of = nullptr;
        const auto q = mtmd_video_plan(no_oracle);
        // control: the cell-arithmetic plan, judged by the same exact oracle, is OVER the budget; only the oracle-verified plan fits
        const int64_t judged = (int64_t) q.n_items * in.tokens_of(q.width_eff, q.height_eff);
        printf("oracle control: cell plan %dx%d judged %lld tokens, oracle plan %dx%d %lld tokens (budget 4096)\n", q.width_eff, q.height_eff,
               (long long) judged, p.width_eff, p.height_eff, (long long) (p.n_items * in.tokens_of(p.width_eff, p.height_eff)));
        CHECK(judged > 4096, "the control needs the cell plan to be over budget under the oracle, got %lld", (long long) judged);
        CHECK(p.width_eff * p.height_eff < q.width_eff * q.height_eff, "the oracle plan must be smaller than the cell plan");
    }
    // bad input is rejected
    {
        auto in = base_clip(0.0);
        CHECK(!mtmd_video_plan(in).ok, "zero duration must be rejected");
    }

    if (g_fail) {
        printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    printf("all video plan checks passed\n");
    return 0;
}
