#include "testing.h"

#include "clip-impl.h"
#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"
#include "dsv4-ik-reference.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_longest_edge) {
    clip_hparams hparams;
    hparams.image_longest_edge = 1540;
    hparams.patch_size = 14;
    hparams.n_merge = 2;

    const std::vector<std::pair<clip_image_size, clip_image_size>> cases = {
        { {  463,  463 }, {  476,  476 } },
        { {  463,  300 }, {  476,  308 } },
        { {  448,  280 }, {  448,  280 } },
        { { 2000, 1000 }, { 1540,  784 } },
    };

    for (const auto & [input, expected] : cases) {
        const clip_image_size actual = mtmd_image_preprocessor_longest_edge::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_dyn_size) {
    // { patch_size, image size, expected size }
    // a side that is an even number plus .5 merged patches rounds down, like Python round() in smart_resize
    const std::vector<std::tuple<int, clip_image_size, clip_image_size>> cases = {
        { 16, {  768, 400 }, {  768, 384 } },
        { 16, {  800, 336 }, {  800, 320 } },
        { 16, {  400, 400 }, {  384, 384 } },
        { 16, {  640, 240 }, {  640, 256 } },
        { 14, {  966, 560 }, {  952, 560 } },
    };

    for (const auto & [patch_size, input, expected] : cases) {
        clip_hparams hparams;
        hparams.patch_size = patch_size;
        hparams.n_merge = 2;
        hparams.image_round_half_even = true;
        hparams.set_limit_image_tokens(8, 4096);

        const clip_image_size actual = mtmd_image_preprocessor_dyn_size::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

static std::string image_size_to_str(const clip_image_size & size) {
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

static std::string resize_algo_to_str(resize_algo algo) {
    std::string str;
    switch (algo & ~RESIZE_ALGO_FLAG_TORCH) {
        case RESIZE_ALGO_BILINEAR: str = "bilinear"; break;
        case RESIZE_ALGO_BICUBIC:  str = "bicubic";  break;
        case RESIZE_ALGO_LANCZOS:  str = "lanczos";  break;
        default:                   str = "unknown";  break;
    }
    if (algo & RESIZE_ALGO_FLAG_TORCH) {
        str += "_torch";
    }
    return str;
}

// pseudo-random pixels: s = s * 1103515245 + 12345 from s = 1, one byte (s >> 16) & 255 per channel
static clip_image_u8 make_test_image(const clip_image_size & size) {
    clip_image_u8 img;
    img.set_size(size, false);
    uint32_t s = 1;
    auto next = [&s]() {
        s = s * 1103515245u + 12345u;
        return static_cast<uint8_t>(s >> 16);
    };
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            img.set_pixel(x, y, { next(), next(), next() });
        }
    }
    return img;
}

// FNV-1a of the pixels
static uint32_t hash_pixels(const clip_image_u8 & img, uint32_t hash = 2166136261u) {
    for (uint8_t v : img.get_ro_buf()) {
        hash = (hash ^ v) * 16777619u;
    }
    return hash;
}

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }

    // { image size, expected size from HF smart_resize }
    const std::vector<std::pair<clip_image_size, clip_image_size>> resize_cases = {
        { {  400, 600 }, { 384, 608 } }, // 400 / 32 = 12.5 rounds half to even
        { {  480, 336 }, { 480, 320 } },
        { {   64, 169 }, { 160, 416 } }, // 169 * beta / 32 is exactly 13 in double
        { { 1024, 768 }, { 576, 416 } },
    };

    for (const auto & [size, expected] : resize_cases) {
        t.assert_equal("smart_resize for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::smart_resize(hparams, size)));
    }

    // { image size, expected grid from HF }
    const std::vector<std::pair<clip_image_size, clip_image_size>> grid_cases = {
        { {  720, 1600 }, { 2, 4 } }, // 0.45 is an exact tie between 2/4 and 2/5 in double, not in float
        { {  420, 1440 }, { 1, 3 } },
        { {  255, 2160 }, { 1, 8 } },
        { { 1024,  768 }, { 3, 2 } },
    };

    for (const auto & [size, expected] : grid_cases) {
        t.assert_equal("grid for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::get_grid_layout(hparams, size.height, size.width)));
    }

    // do_image_splitting=false
    hparams.preproc_min_tiles = 1;
    hparams.preproc_max_tiles = 1;
    t.assert_equal("tiling for 1024x977 with max_tiles=1", std::string("single"),
            std::string(mtmd_image_preprocessor_lfm2::should_tile(hparams, { 1024, 977 }) ? "tiled" : "single"));
}

MAKE_TEST(test_image_preprocessor_lfm2_torch_resize) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;
    // Lfm2VlImageProcessor stretches the image to the tile grid
    hparams.image_pad_rf = PAD_NONE;

    // { resize algo, image size, expected hash of the tiles and the thumbnail from Lfm2VlImageProcessor (torchvision 0.29, native uint8 resize: arm64, or x86 with AVX2/AVX512) }
    const std::vector<std::tuple<resize_algo, clip_image_size, uint32_t>> cases = {
        { RESIZE_ALGO_BILINEAR, {  620,  410 }, 0xf5e9c941 }, // 608x416
        { RESIZE_ALGO_BILINEAR, { 2000, 1500 }, 0x3331f56c }, // 3x2 tiles, 576x416 thumbnail
        { RESIZE_ALGO_BICUBIC,  {  620,  410 }, 0x2b34eac9 },
        { RESIZE_ALGO_BICUBIC,  { 2000, 1500 }, 0xb035d942 },
    };

    for (const auto & [base_algo, size, expected] : cases) {
        const resize_algo algo = static_cast<resize_algo>(base_algo | RESIZE_ALGO_FLAG_TORCH);
        hparams.image_resize_algo_rf = algo;
        hparams.image_resize_algo_ov = algo;
        const mtmd_image_preprocessor_lfm2 preproc(hparams);
        const auto sliced = preproc.slice_image(make_test_image(size), preproc.get_slice_instructions(size));

        uint32_t hash = 2166136261u;
        for (const auto & slice : sliced.slices) {
            hash = hash_pixels(slice, hash);
        }
        hash = hash_pixels(sliced.overview, hash);
        t.assert_equal("pixels of " + std::to_string(size.width) + "x" + std::to_string(size.height) + " with resize algo " + resize_algo_to_str(algo), expected, hash);
    }
}

//
// DeepSeek-V4 / V4.1 vision: image-size solver vs the ik_llama code it was ported from
//

// Compares tools/mtmd's solvers (deepseek4v::plan_size, deepseek41v::plan_grid -- upstream's preprocessors, the code preprocess()
// runs) with ik's lambdas, copied verbatim into
// dsv4-ik-reference.h, over a sweep of image sizes and solver parameters. Expected to be IDENTICAL except where ik's code
// aborts or emits a 0-sized image, which are counted and reported separately.
static void dsv4_solver_sweep(testing & t, bool is_v41) {
    using P   = mtmd_image_preprocessor_deepseek4v;
    using P41 = mtmd_image_preprocessor_deepseek41v;
    const int patches[]  = { 4, 14, 16 };
    const int merges[]   = { 1, 2, 3, 4 };
    const int caps[]     = { 16, 17, 24, 40, 64, 100, 257, 384, 1024 };
    const int min_pixs[] = { 0, 4096, 200000 };
    const int whs[]      = { 0, 8 };   // deepseek41 forces 0, deepseek4v 8; the other value is swept too so both branches are covered

    std::vector<int> dims = { 1, 2, 3, 5, 7, 8, 13, 16, 17, 31, 33, 50, 64, 97, 100, 128, 255, 256, 257, 333, 480, 512, 640, 719, 1000, 1024, 1333, 2048, 3000, 4000, 8000 };

    long n_cmp = 0, n_equal = 0, n_ik_abort = 0, n_ik_zero = 0, n_other_diff = 0, n_overbudget = 0;
    std::string first_diff;
    for (int patch : patches) for (int merge : merges) for (int cap : caps) for (int minpx : min_pixs) for (int wh : whs) {
        for (int w : dims) for (int h : dims) {
            // V4.1 has no aspect clamp: clip.cpp sets dsv4_max_wh_ratio = 0 for it ("vision_max_wh_ratio = null (converter asserts
            // both)") and upstream's plan_grid, a port of image_processor.py (V4.1), takes no wh at all. The fork's pre-merge solver was
            // one function for V4 and V4.1, so this sweep fed V4.1 wh=8 to cover that shared branch; the merged V4.1 solver has no such
            // branch, so ik's V4.1 code is compared at the wh it runs with. V4 still sweeps both wh values.
            const ik_ref_result ik = is_v41
                ? ik_ref_deepseek41(w, h, patch, merge, cap, 0, minpx)
                : ik_ref_deepseek4v(w, h, patch, merge, cap, wh, minpx);

            bool stretch = false;
            clip_image_size ours;
            int n_tokens = 0;
            if (is_v41) {
                // deepseek41v has no aspect clamp and no stretch path (upstream mtmd-image.cpp: "a single solve, no stretch path")
                int bw = 0, bh = 0;
                P41::plan_grid(w, h, patch, merge, cap, minpx, bw, bh);
                ours     = { bw, bh };
                n_tokens = P41::block_tokens(ours.height, ours.width, patch, merge);
            } else {
                ours     = P::plan_size({ w, h }, patch, merge, cap, wh, minpx, stretch);
                n_tokens = P::block_tokens(ours.height, ours.width, patch, merge);
            }
            n_cmp++;

            // our result must always be a valid size that fits the budget (a 1-patch image is the floor)
            const int budget = is_v41 ? cap : cap - 3; // deepseek4v keeps 3 tokens for the lead pads
            if (n_tokens > budget && ours.width > patch && ours.height > patch) {
                n_overbudget++;
            }

            if (ik.assert_failed) { n_ik_abort++; continue; }
            if (ik.best_width <= 0 || ik.best_height <= 0) { n_ik_zero++; continue; }
            if (ik.best_width == ours.width && ik.best_height == ours.height && ik.stretch == stretch) {
                n_equal++;
            } else {
                n_other_diff++;
                if (first_diff.empty()) {
                    first_diff = "patch=" + std::to_string(patch) + " merge=" + std::to_string(merge) + " cap=" + std::to_string(cap) +
                                 " minpx=" + std::to_string(minpx) + " wh=" + std::to_string(wh) + " img=" + std::to_string(w) + "x" + std::to_string(h) +
                                 " ik=" + std::to_string(ik.best_width) + "x" + std::to_string(ik.best_height) +
                                 " ours=" + std::to_string(ours.width) + "x" + std::to_string(ours.height);
                }
            }
        }
    }
    t.log("compared " + std::to_string(n_cmp) + " cases: identical " + std::to_string(n_equal) +
          ", ik aborted " + std::to_string(n_ik_abort) + ", ik emitted a 0-sized image " + std::to_string(n_ik_zero) +
          ", other differences " + std::to_string(n_other_diff) + ", over budget " + std::to_string(n_overbudget));
    if (!first_diff.empty()) {
        t.log("first difference: " + first_diff);
    }
    t.assert_equal("no difference from the ik solver outside its abort / 0-size cases", 0L, n_other_diff);
    t.assert_equal("our result always fits the token budget", 0L, n_overbudget);
    t.assert_true("the sweep actually compared something", n_equal > 1000);
}

MAKE_TEST(test_dsv4_solver_matches_ik_deepseek4v) { dsv4_solver_sweep(t, false); }
MAKE_TEST(test_dsv4_solver_matches_ik_deepseek41) { dsv4_solver_sweep(t, true); }

// token counts: clip-impl.h's block layout against hand-worked values of the layout ik documents (60c74d9c4):
//   [PAD]*lead [START] <rows padded to an even count, each w+1 wide> [PAD]*tail [END], tail = 2 when (rows/2*(w+1)) is odd
MAKE_TEST(test_dsv4_block_layout) {
    struct tc { int w, h, lead, n_out, tail; };
    const tc cases[] = {
        { 8, 8, 0, 74, 0 },  // 72 grid slots, 36 even
        { 5, 3, 3, 29, 0 },  // 3 rows -> 4, 24 slots, 12 even
        { 4, 2, 1, 15, 2 },  // 2 rows, 10 slots, 5 odd -> tail 2
        { 1, 1, 2,  8, 0 },  // 1 row -> 2, 4 slots, 2 even
        { 2, 1, 0, 10, 2 },  // 1 row -> 2, 6 slots, 3 odd -> tail 2
    };
    for (const auto & c : cases) {
        const auto bl = dsv4_get_block_layout(c.w, c.h, c.lead);
        const std::string name = std::to_string(c.w) + "x" + std::to_string(c.h) + " lead " + std::to_string(c.lead);
        t.assert_equal("n_out for " + name, c.n_out, bl.n_out);
        t.assert_equal("pad_last for " + name, c.tail, bl.pad_last);
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// projector registry
//
// Idea from unslothai/llama.cpp#176 (MIT, same tree family as ours), written for this fork's table, which also carries
// a read-only legacy alias map. A projector whose enum value survives a merge while its PROJECTOR_TYPE_NAMES entry is
// lost still compiles, and every model using it then loads as PROJECTOR_TYPE_UNKNOWN; two projectors sharing a name
// compile too and silently route one model to the other's graph. Neither needs a model file to detect.
//
// Projectors the clip loader assigns to itself and never reads from a GGUF are unnameable by construction. Keep this
// list at exactly those: an entry added to quiet a failure is a projector no model can ever select.
// Returns one line per defect found in `names`; empty means the table is sound. Takes the table as an argument so
// the negative controls below can feed it a deliberately broken copy: an audit that has only ever been shown a
// good table has not been shown to detect anything.
static std::vector<std::string> projector_registry_problems(const std::map<projector_type, std::string> & names,
                                                            const std::set<projector_type> & internal_only) {
    std::vector<std::string> problems;
    std::map<std::string, projector_type> seen;
    for (int i = 0; i < PROJECTOR_TYPE_UNKNOWN; i++) {
        const projector_type ty = static_cast<projector_type>(i);
        const bool internal = internal_only.count(ty) > 0;
        const auto it = names.find(ty);
        if (it == names.end()) {
            if (!internal) {
                problems.push_back("projector #" + std::to_string(i) + " is missing from the name table");
            }
            continue;
        }
        if (internal) {
            problems.push_back("\"" + it->second + "\" is named but listed internal-only");
        }
        const auto dup = seen.find(it->second);
        if (dup != seen.end()) {
            problems.push_back("\"" + it->second + "\" names projectors #" + std::to_string(static_cast<int>(dup->second)) +
                               " and #" + std::to_string(i));
        }
        seen[it->second] = ty;
    }
    for (const auto & kv : names) {
        if (static_cast<int>(kv.first) >= static_cast<int>(PROJECTOR_TYPE_UNKNOWN)) {
            problems.push_back("table entry \"" + kv.second + "\" is not a projector the enum declares");
        }
    }
    return problems;
}

MAKE_TEST(test_projector_registry) {
    const std::set<projector_type> internal_only = {
        PROJECTOR_TYPE_MLP_NORM, // chosen from the tensor shapes for Yi-type llava
    };

    const auto problems = projector_registry_problems(PROJECTOR_TYPE_NAMES, internal_only);
    for (const auto & p : problems) {
        t.assert_equal("registry defect", std::string("none"), p);
    }
    t.assert_equal("the real table has no defect", size_t(0), problems.size());

    // through the lookup every model load uses, not the table read backwards
    for (const auto & kv : PROJECTOR_TYPE_NAMES) {
        t.assert_equal("\"" + kv.second + "\" resolves to its own projector",
                       static_cast<int>(kv.first), static_cast<int>(clip_projector_type_from_string(kv.second)));
    }

    // clip reads the string straight out of the GGUF: a typo must not become a real graph
    t.assert_equal("an unregistered name resolves to UNKNOWN", static_cast<int>(PROJECTOR_TYPE_UNKNOWN),
                   static_cast<int>(clip_projector_type_from_string("no-such-projector")));
    // legacy spellings resolve to a real projector but are not themselves in the table (read-only aliases)
    for (const auto & a : PROJECTOR_TYPE_LEGACY_ALIASES) {
        t.assert_equal("legacy alias \"" + a.first + "\" resolves to a real projector", true, a.second != PROJECTOR_TYPE_UNKNOWN);
        bool canonical = false;
        for (const auto & kv : PROJECTOR_TYPE_NAMES) {
            canonical = canonical || kv.second == a.first;
        }
        t.assert_equal("legacy alias \"" + a.first + "\" is not also a canonical name", false, canonical);
    }
}

// NEGATIVE CONTROLS for the audit above: each broken copy of the real table must be reported, and the unbroken
// copy must not be. Without these the audit passing means only "the audit ran".
MAKE_TEST(test_projector_registry_audit_detects_defects) {
    const std::set<projector_type> internal_only = { PROJECTOR_TYPE_MLP_NORM };

    auto good = PROJECTOR_TYPE_NAMES;
    t.assert_equal("control: an unbroken copy is clean", size_t(0), projector_registry_problems(good, internal_only).size());

    auto lost = PROJECTOR_TYPE_NAMES;                      // an enum value whose name entry was lost in a merge
    lost.erase(PROJECTOR_TYPE_QWEN3VL);
    t.assert_equal("control: a lost name entry is reported", size_t(1), projector_registry_problems(lost, internal_only).size());

    auto dup = PROJECTOR_TYPE_NAMES;                       // two projectors sharing one name
    dup[PROJECTOR_TYPE_GEMMA3NV] = dup[PROJECTOR_TYPE_GEMMA3];
    t.assert_equal("control: a duplicated name is reported", size_t(1), projector_registry_problems(dup, internal_only).size());

    auto quieted = PROJECTOR_TYPE_NAMES;                   // an internal-only projector that was given a selectable name
    quieted[PROJECTOR_TYPE_MLP_NORM] = "mlp_norm";
    t.assert_equal("control: naming an internal-only projector is reported", size_t(1), projector_registry_problems(quieted, internal_only).size());

    auto stale = PROJECTOR_TYPE_NAMES;                     // a row for a value the enum no longer declares
    stale[static_cast<projector_type>(static_cast<int>(PROJECTOR_TYPE_UNKNOWN) + 5)] = "stale-row";
    t.assert_equal("control: a stale table row is reported", size_t(1), projector_registry_problems(stale, internal_only).size());
}

// mtmd_audio
//

MAKE_TEST(test_audio_preprocessor_conformer) {
    clip_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // 0.4 s of tones, 0.2 s of silence, 0.4 s of a quiet tone
    const double pi = 3.14159265358979323846;
    const int sr = hparams.audio_sample_rate;
    std::vector<float> samples(sr, 0.0f);
    for (int i = 0; i < sr; i++) {
        const double ts = (double) i / sr;
        if (i < 0.4 * sr) {
            samples[i] = (float) (0.3 * std::sin(2 * pi * 300 * ts) + 0.2 * std::sin(2 * pi * 1200 * ts) + 0.1 * std::sin(2 * pi * 3500 * ts));
        } else if (i >= 0.6 * sr) {
            samples[i] = (float) (1e-3 * std::sin(2 * pi * 800 * ts));
        }
    }

    mtmd_audio_preprocessor_conformer preproc(hparams);
    preproc.initialize();
    std::vector<mtmd_audio_mel> mels;
    if (!t.assert_true("preprocess", preproc.preprocess(samples.data(), samples.size(), mels))) {
        return;
    }
    const mtmd_audio_mel & mel = mels[0];
    t.assert_equal("n_len", (int64_t) 101, mel.n_len);

    // reference values from NeMo AudioToMelSpectrogramPreprocessor (liquid-audio 1.3.0)
    // { mel bin, frame, value }
    const std::vector<std::tuple<int, int, float>> cases = {
        {   8,  0,  3.6104f }, // tones
        {  24, 50, -0.6371f }, // silence
        {  24, 60, -0.0476f }, // silence -> quiet tone
        { 100, 20, -0.0913f }, // tones, high bin
    };
    for (const auto & [bin, frame, expected] : cases) {
        const float actual = mel.data[bin * mel.n_len + frame];
        t.assert_true("mel[" + std::to_string(bin) + "][" + std::to_string(frame) + "] = " + std::to_string(actual) + ", expected " + std::to_string(expected), std::fabs(actual - expected) < 1e-3f);
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
