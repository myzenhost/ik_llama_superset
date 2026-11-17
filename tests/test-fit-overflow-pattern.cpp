// common_fit_overflow_pattern (common/fit.cpp): the tensor-name regex the fit writes into tensor_buft_overrides for a layer.
// Row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01 (D1). It used to keep the patterns in three static
// std::array<std::string, 1000> and throw "at most 1000 model layers are supported": a model with more layers could not be fitted.
// The layer count is a property of the model (<arch>.block_count), so no layer index may be refused here.
#include "fit.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static std::string expected(size_t il, common_layer_fraction_t lf) {
    const std::string blk = "blk\\." + std::to_string(il) + "\\.";
    switch (lf) {
        case LAYER_FRACTION_ATTN: return blk + "ffn_(gate|up|gate_up|down).*";
        case LAYER_FRACTION_UP:   return blk + "ffn_(gate|gate_up|down).*";
        case LAYER_FRACTION_GATE: return blk + "ffn_down.*";
        case LAYER_FRACTION_MOE:  return blk + "ffn_(up|down|gate_up|gate)_(ch|)exps";
        default:                  return "";
    }
}

int main() {
    const common_layer_fraction_t fr[] = { LAYER_FRACTION_ATTN, LAYER_FRACTION_UP, LAYER_FRACTION_GATE, LAYER_FRACTION_MOE };

    // layer indices on both sides of the old 1000 cap, and far past it
    for (const size_t il : { (size_t) 0, (size_t) 1, (size_t) 999, (size_t) 1000, (size_t) 1001, (size_t) 1500, (size_t) 100000 }) {
        for (const auto lf : fr) {
            std::string got;
            bool threw = false;
            try {
                got = common_fit_overflow_pattern(il, lf);
            } catch (const std::exception & e) {
                threw = true;
                std::printf("layer %zu fraction %d threw: %s\n", il, (int) lf, e.what());
            }
            CHECK(!threw, "layer %zu fraction %d must not be refused", il, (int) lf);
            CHECK(!threw && got == expected(il, lf), "layer %zu fraction %d pattern '%s' != '%s'", il, (int) lf, got.c_str(), expected(il, lf).c_str());
        }
    }

    // the fit keeps the pointers in the caller's buffer after it returns: they must stay valid and unchanged while later layers
    // (more than the old cap) are added, and the same layer must give the same pointer
    std::vector<const char *> first;
    for (const auto lf : fr) { first.push_back(common_fit_overflow_pattern(0, lf)); }
    for (size_t il = 1; il < 3000; ++il) {
        for (const auto lf : fr) { (void) common_fit_overflow_pattern(il, lf); }
    }
    for (size_t k = 0; k < 4; ++k) {
        CHECK(common_fit_overflow_pattern(0, fr[k]) == first[k], "pointer of layer 0 fraction %zu moved", k);
        CHECK(std::strcmp(first[k], expected(0, fr[k]).c_str()) == 0, "layer 0 fraction %zu text changed", k);
    }

    if (g_fails) {
        std::printf("test-fit-overflow-pattern: %d FAILED\n", g_fails);
        return 1;
    }
    std::printf("test-fit-overflow-pattern: ok\n");
    return 0;
}
