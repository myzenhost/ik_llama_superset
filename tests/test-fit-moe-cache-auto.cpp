// --moe-cache-mib auto: the sizing rule (common_fit_moe_cache_pick, common/fit.cpp), unit-tested without a GPU.
// Rule (unslothai/llama.cpp PR 251, adapted: this fork's dry-run excludes the cache, so the size is exact):
//   cache = floor_MiB(target - mem_cpu_moe); none when the dense weights do not fit, when the layer split would keep >= 80% of the
//   expert weights resident, or when the cache is below 2 x llama_model_moe_cache_min_size.
#include "fit.h"

#include <cstdio>
#include <string>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static const int64_t MiB = 1024*1024;
static const int64_t GiB = 1024*MiB;

int main() {
    std::string why;

    // 24 GiB card, 4 GiB of dense weights + context, a 60 GiB expert bank: 20 GiB cache
    CHECK(common_fit_moe_cache_pick(24*GiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 20*GiB, "plain case: %s", why.c_str());

    // MiB rounding: 10 GiB + 700 KiB of room caches 10 GiB
    CHECK(common_fit_moe_cache_pick(14*GiB + 700*1024, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 10*GiB, "floor to MiB: %s", why.c_str());

    // the dense weights do not fit
    CHECK(common_fit_moe_cache_pick(4*GiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 0 && why.find("do not fit") != std::string::npos, "no room: %s", why.c_str());
    CHECK(common_fit_moe_cache_pick(3*GiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 0, "deficit");

    // the layer split would hold >= 80% of the experts on the device: no cache (60 GiB of experts, 50 GiB of room = 83%)
    CHECK(common_fit_moe_cache_pick(54*GiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 0 && why.find("layer split") != std::string::npos, "resident: %s", why.c_str());
    // just under the threshold still caches (47 GiB of 60 = 78%)
    CHECK(common_fit_moe_cache_pick(51*GiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 47*GiB, "78%% resident: %s", why.c_str());

    // below twice the minimum a cache thrashes: 100 MiB of room against a 64 MiB minimum
    CHECK(common_fit_moe_cache_pick(4*GiB + 100*MiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 0 && why.find("minimum") != std::string::npos, "below minimum: %s", why.c_str());
    CHECK(common_fit_moe_cache_pick(4*GiB + 128*MiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, &why) == 128*MiB, "exactly 2x minimum: %s", why.c_str());

    // no layer can be cached (min size 0: not a one-GPU MoE with host-resident experts)
    CHECK(common_fit_moe_cache_pick(24*GiB, 4*GiB, 64*GiB, 4*GiB, 0, &why) == 0 && why.find("no layer") != std::string::npos, "min 0: %s", why.c_str());

    // no expert bytes known: treated as fully resident, no cache
    CHECK(common_fit_moe_cache_pick(24*GiB, 4*GiB, 4*GiB, 4*GiB, 64*MiB, &why) == 0, "no experts");

    // why == nullptr is allowed
    CHECK(common_fit_moe_cache_pick(24*GiB, 4*GiB, 64*GiB, 4*GiB, 64*MiB, nullptr) == 20*GiB, "null why");

    std::printf("%d failed\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
