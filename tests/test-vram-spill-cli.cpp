// test-vram-spill-cli: drives the REAL path  CLI -> common_params -> common_vram_spill_apply -> ggml policy.
//   flag absent  -> policy stays OFF, apply() returns false (bit-identical default)
//   flag present -> params carry the values, apply() returns true and the ggml policy reads back the same values
//   bad mode     -> parse fails
#include "arg.h"
#include "common.h"
#include "ggml-vram-spill.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static bool parse(common_params & params, std::vector<std::string> args) {
    std::vector<char *> argv;
    args.insert(args.begin(), {"test-vram-spill-cli", "-m", "does-not-need-to-exist.gguf"});
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    return common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON);
}

int main() {
    const ggml_vram_spill_policy off = {GGML_VRAM_SPILL_OFF, 256ull << 20, 1024ull << 20};

    {   // default
        ggml_vram_spill_configure(&off);
        common_params p;
        CHECK(parse(p, {}));
        CHECK(p.vram_spill_mode == "off");
        CHECK(!common_vram_spill_apply(p));
        CHECK(ggml_vram_spill_get_policy().mode == GGML_VRAM_SPILL_OFF);
    }
    {   // flags on
        ggml_vram_spill_configure(&off);
        common_params p;
        CHECK(parse(p, {"--vram-spill", "auto", "--vram-spill-threshold", "512", "--vram-spill-reserve", "2048"}));
        CHECK(p.vram_spill_mode == "auto" && p.vram_spill_threshold_mib == 512 && p.vram_spill_reserve_mib == 2048);
        CHECK(common_vram_spill_apply(p));
        const ggml_vram_spill_policy g = ggml_vram_spill_get_policy();
        CHECK(g.mode == GGML_VRAM_SPILL_AUTO && g.threshold_bytes == (512ull << 20) && g.reserve_bytes == (2048ull << 20));
        // and the policy the parser configured really decides placements
        CHECK(ggml_vram_spill_choose(&g, 600ull << 20, 0) == GGML_VRAM_SPILL_PLACE_HOSTMAP);
        CHECK(ggml_vram_spill_choose(&g, 100ull << 20, 0) == GGML_VRAM_SPILL_PLACE_DEVICE);
    }
    {   // explicit off after a previous on does not leave the policy configured by apply()
        ggml_vram_spill_configure(&off);
        common_params p;
        CHECK(parse(p, {"--vram-spill", "off"}));
        CHECK(!common_vram_spill_apply(p));
        CHECK(ggml_vram_spill_get_policy().mode == GGML_VRAM_SPILL_OFF);
    }
    {   // bad mode / negative numbers rejected
        common_params p;
        CHECK(!parse(p, {"--vram-spill", "bogus"}));
        common_params q;
        CHECK(!parse(q, {"--vram-spill-threshold", "-1"}));
    }

    if (g_fail) { std::printf("test-vram-spill-cli: %d FAILED\n", g_fail); return 1; }
    std::printf("test-vram-spill-cli: all checks passed\n");
    return 0;
}
