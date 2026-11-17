// CPU-only test of ggml-cuda/fattn-force.h (GGML_CUDA_FATTN_FORCE parsing and legality).
#include "../ggml/src/ggml-cuda/fattn-force.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)

int main() {
    // parse: default and bad input are NONE (today's behaviour)
    CHECK(fattn_force_parse(nullptr) == FATTN_FORCE_NONE);
    CHECK(fattn_force_parse("")      == FATTN_FORCE_NONE);
    CHECK(fattn_force_parse("VEC")   == FATTN_FORCE_NONE);   // case-sensitive: a typo never forces
    CHECK(fattn_force_parse("wmma")  == FATTN_FORCE_NONE);   // not offered (legality unread)
    CHECK(fattn_force_parse("vec ")  == FATTN_FORCE_NONE);
    CHECK(fattn_force_parse("vec")   == FATTN_FORCE_VEC);
    CHECK(fattn_force_parse("tile")  == FATTN_FORCE_TILE);
    CHECK(fattn_force_parse("mma")   == FATTN_FORCE_MMA);

    // decide: exhaustive over every request and every legality combination.
    const fattn_force_kind wants[] = { FATTN_FORCE_NONE, FATTN_FORCE_VEC, FATTN_FORCE_TILE, FATTN_FORCE_MMA };
    for (fattn_force_kind w : wants) {
        for (int m = 0; m < 8; ++m) {
            const bool v = m & 1, t = m & 2, a = m & 4;
            const fattn_force_kind got = fattn_force_decide(w, v, t, a);
            const bool legal = (w == FATTN_FORCE_VEC && v) || (w == FATTN_FORCE_TILE && t) || (w == FATTN_FORCE_MMA && a);
            // never an illegal family, and exactly the request when legal
            CHECK(got == (legal ? w : FATTN_FORCE_NONE));
        }
    }
    // an illegal request declines even when other families are legal
    CHECK(fattn_force_decide(FATTN_FORCE_TILE, true, false, true) == FATTN_FORCE_NONE);

    std::printf("OK\n");
    return 0;
}
