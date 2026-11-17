#pragma once
// Reference arithmetic for the DeepSeek-V4 / V4.1 vision image-size solver, transcribed VERBATIM (by script) from the
// ik_llama.cpp source so that tools/mtmd's port can be compared against the code it was ported from.
//
//   SOURCE-READ: ik_llama.cpp 32cddbfce examples/mtmd/clip.cpp (MIT)
//     deepseek4v : clip_image_preprocess(), PROJECTOR_TYPE_DEEPSEEK4V case, source lines 5260..5341  (commit 60c74d9c4)
//     deepseek41 : clip_image_preprocess(), PROJECTOR_TYPE_DEEPSEEK41 case, source lines 5361..5426  (commit 248e77729)
//   The statements are copied unchanged. The only edits are (1) wrapping them in a function and replacing the
//   `params` / `original_size` objects by locals of the same name, and (2) in deepseek41 the GGML_ASSERT that aborts when
//   the solver does not fit becomes IK_REF_ASSERT, which records instead of aborting, so a sweep can report it.
//   The image resize / pad / normalize tail (img_tool::resize ...) is NOT included: it needs pixel data.
//   dsv4_get_block_layout() is the one from tools/mtmd/clip-impl.h (identical to ik's clip-impl.h in 60c74d9c4).
//   Regenerate with: pythonw temp/_dsv41_genref.py  (reads a `git show 32cddbfce:examples/mtmd/clip.cpp` dump).

#include "clip-impl.h"

#include <algorithm>
#include <cmath>

struct ik_ref_result {
    int  best_width    = 0;
    int  best_height   = 0;
    bool assert_failed = false; // an ik assertion would have aborted
    bool stretch       = false; // ik takes the "extreme aspect ratio" no-padding branch
};

#define IK_REF_ASSERT(cond) do { if (!(cond)) { ik_ref_assert_failed = true; } } while (0)

static inline ik_ref_result ik_ref_deepseek4v(int orig_w, int orig_h, int patch_size, int n_merge,
                                              int dsv4_max_n_token, int dsv4_max_wh_ratio, int image_min_pixels) {
    struct { int patch_size, n_merge, dsv4_max_n_token, dsv4_max_wh_ratio, image_min_pixels; }
        params = { patch_size, n_merge, dsv4_max_n_token, dsv4_max_wh_ratio, image_min_pixels };
    struct { int width, height; } original_size = { orig_w, orig_h };
    bool ik_ref_assert_failed = false;
    // ---- begin verbatim (ik 60c74d9c4 / 32cddbfce clip.cpp 5260..5341) ----
    const int p           = params.patch_size;
    const int r           = params.n_merge;
    const int max_n_token = params.dsv4_max_n_token;
    const int max_wh      = params.dsv4_max_wh_ratio;

    struct grid_info {
        int n_llm_h;
        int n_llm_w;
        int n_tokens; // token count of the block (incl. newline/pad rows and start/end, excl. lead pads)
    };

    // ref: grid_tokens()
    auto grid_tokens = [](int best_height, int best_width, int patch_size, int r) -> grid_info {
        grid_info g;
        g.n_llm_h = ((best_height / patch_size) + r - 1) / r;
        g.n_llm_w = ((best_width  / patch_size) + r - 1) / r;
        g.n_tokens = dsv4_get_block_layout(g.n_llm_w, g.n_llm_h, 0).n_out;
        return g;
    };

    // ref: solve_resize_ratio()
    auto solve_resize_ratio = [](int height, int width, int p, int r, int max_n_token,
                                 int & best_height, int & best_width) {
        const double ratio   = (double) height / width;
        const double max_w_f = std::sqrt((max_n_token - 2) / ratio + 0.25) - 0.5;
        const double max_h_f = max_w_f * ratio;
        if (max_w_f < 1.0) {
            const int max_w = 1;
            int max_h = (max_n_token - 2) / (max_w + 1);
            if (max_h % 2 == 1) {
                max_h -= 1;
            }
            best_width  = max_w * p * r;
            best_height = max_h * p * r;
        } else if (max_h_f < 2.0) {
            const int max_h = 2;
            // guard tiny budgets; cannot be hit with the current lower bound on max_n_token
            const int max_w = std::max(((max_n_token - 2) / max_h) - 1, 2);
            best_width  = max_w * p * r;
            best_height = max_h * p * r;
        } else {
            const int max_w_i = (int) std::floor(max_w_f);
            int max_h_i = (int) std::floor(max_h_f);
            if (max_h_i % 2 == 1) {
                max_h_i -= 1;
            }
            const double beta = std::min(
                (double) max_w_i * p * r / width,
                (double) max_h_i * p * r / height);
            best_width  = (int) std::floor(width  * beta / p) * p;
            best_height = (int) std::floor(height * beta / p) * p;
        }
    };

    // ref: safe_resize()
    auto safe_resize = [&](int height, int width, int & best_height, int & best_width,
                           int p, int r, int max_n_token) {
        max_n_token -= 4 - 1; // reserve room for the position-dependent lead pads (COMPRESS_PAD_TO - 1)
        grid_info g = grid_tokens(best_height, best_width, p, r);
        int budget = max_n_token;
        while (g.n_tokens > max_n_token) {
            solve_resize_ratio(height, width, p, r, budget, best_height, best_width);
            g = grid_tokens(best_height, best_width, p, r);
            budget -= 1;
        }
    };

    // ref: load_image()
    int width  = original_size.width;
    int height = original_size.height;
    if (max_wh > 0 && width > height * max_wh) {
        width = height * max_wh;
    }
    if (params.image_min_pixels > 0 && width * height > 0
            && width * height < params.image_min_pixels) {
        const double up = std::sqrt((double) params.image_min_pixels / ((double) width * height));
        width  = (int) (width  * up);
        height = (int) (height * up);
    }
    int best_width  = CLIP_ALIGN(width,  p);
    int best_height = CLIP_ALIGN(height, p);
    safe_resize(height, width, best_height, best_width, p, r, max_n_token);
    // ---- end verbatim ----
    ik_ref_result res;
    res.best_width = best_width; res.best_height = best_height; res.assert_failed = ik_ref_assert_failed;
    res.stretch = max_wh > 0 && original_size.width >= max_wh * original_size.height;
    return res;
}

static inline ik_ref_result ik_ref_deepseek41(int orig_w, int orig_h, int patch_size, int n_merge,
                                              int dsv4_max_n_token, int dsv4_max_wh_ratio, int image_min_pixels) {
    struct { int patch_size, n_merge, dsv4_max_n_token, dsv4_max_wh_ratio, image_min_pixels; }
        params = { patch_size, n_merge, dsv4_max_n_token, dsv4_max_wh_ratio, image_min_pixels };
    struct { int width, height; } original_size = { orig_w, orig_h };
    bool ik_ref_assert_failed = false;
    // ---- begin verbatim (ik 248e77729 / 32cddbfce clip.cpp 5361..5426) ----
    const int p           = params.patch_size;
    const int r           = params.n_merge;
    const int max_n_token = params.dsv4_max_n_token;
    const int max_wh      = params.dsv4_max_wh_ratio;

    struct grid_info {
        int n_llm_h;
        int n_llm_w;
        int n_tokens;
    };

    auto grid_tokens = [](int best_height, int best_width, int patch_size, int r) -> grid_info {
        grid_info g;
        g.n_llm_h = (int) std::ceil((double)(best_height / patch_size) / r);
        g.n_llm_w = (int) std::ceil((double)(best_width  / patch_size) / r);
        g.n_tokens = g.n_llm_h * (g.n_llm_w + 1) + 2;
        return g;
    };

    auto solve_resize_ratio = [](int height, int width, int p, int r, int max_n_token,
                                 int & best_height, int & best_width) {
        const double ratio   = (double) height / width;
        const double max_w_f = std::sqrt((double)(max_n_token - 2) / ratio + 0.25) - 0.5;
        const double max_h_f = max_w_f * ratio;
        const int    cell    = p * r;
        if (max_w_f < 1.0) {
            best_height = ((max_n_token - 2) / 2) * cell;
            best_width  = cell;
        } else if (max_h_f < 1.0) {
            best_height = cell;
            best_width  = (max_n_token - 3) * cell;
        } else {
            const int max_w_i = (int) std::floor(max_w_f);
            const int max_h_i = (int) std::floor(max_h_f);
            const double beta = std::min(
                (double) max_w_i * cell / width,
                (double) max_h_i * cell / height);
            best_height = (int) std::floor(height * beta / p) * p;
            best_width  = (int) std::floor(width  * beta / p) * p;
        }
    };

    auto safe_resize = [&](int height, int width, int & best_height, int & best_width,
                           int p, int r, int max_n_token) {
        grid_info g = grid_tokens(best_height, best_width, p, r);
        if (g.n_tokens > max_n_token) {
            solve_resize_ratio(height, width, p, r, max_n_token, best_height, best_width);
            g = grid_tokens(best_height, best_width, p, r);
            IK_REF_ASSERT(g.n_tokens <= max_n_token && "resize solver failed to fit the token budget");
        }
    };

    int width  = original_size.width;
    int height = original_size.height;
    if (max_wh > 0 && width > height * max_wh) {
        width = height * max_wh;
    }
    if (params.image_min_pixels > 0 && width * height > 0
            && width * height < params.image_min_pixels) {
        const double up = std::sqrt((double) params.image_min_pixels / ((double) width * height));
        width  = (int) (width  * up);
        height = (int) (height * up);
    }
    int best_width  = CLIP_ALIGN(width,  p);
    int best_height = CLIP_ALIGN(height, p);
    safe_resize(height, width, best_height, best_width, p, r, max_n_token);
    // ---- end verbatim ----
    ik_ref_result res;
    res.best_width = best_width; res.best_height = best_height; res.assert_failed = ik_ref_assert_failed;
    res.stretch = max_wh > 0 && original_size.width >= max_wh * original_size.height;
    return res;
}

// number of tokens ik's clip_n_output_tokens() returns for an image of the given (already resized) size
static inline int ik_ref_n_tokens_deepseek4v(int nx, int ny, int patch_size, int n_merge, int lead_pad) {
    const int out_patch_size = patch_size * n_merge;           // ik clip.cpp 5671..5677
    const int n_llm_w = CLIP_ALIGN(nx, out_patch_size) / out_patch_size;
    const int n_llm_h = CLIP_ALIGN(ny, out_patch_size) / out_patch_size;
    return dsv4_get_block_layout(n_llm_w, n_llm_h, lead_pad).n_out;
}
static inline int ik_ref_n_tokens_deepseek41(int nx, int ny, int patch_size, int n_merge) {
    const int out_patch_size = patch_size * n_merge;           // ik clip.cpp 5678..5684
    const int n_llm_w = CLIP_ALIGN(nx, out_patch_size) / out_patch_size;
    const int n_llm_h = CLIP_ALIGN(ny, out_patch_size) / out_patch_size;
    return n_llm_h * (n_llm_w + 1) + 2;
}
