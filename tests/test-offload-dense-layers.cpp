// Row S69-ENGREV-HIPATTENTION-FLAG-SURFACE-2026-10-27-01: --offload-attn-dense-layers accepts N | -N | an explicit list.
// Model-free (src/offload_attention_dense.h is header-only). Before the row only a bare integer (first N) existed, and a
// list such as "0-2,45-47" went through std::stoi, which reads the leading 0 and ignores the rest. Here the same strings are
// checked against the resolver. Includes must-fail partners (reversed range, out of range, junk).
#include "../src/offload_attention_dense.h"

#include <cstdio>
#include <string>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); g_fail++; } } while (0)

using ik_offload::DenseLayers;
using ik_offload::parse_dense_layers;

static bool parse(const char * s, int n, DenseLayers & d, std::string & err) { err.clear(); return parse_dense_layers(s, n, d, err); }

int main() {
    DenseLayers d; std::string err;

    // first N (original meaning, unchanged)
    CHECK(parse("4", 48, d, err) && !d.listed && d.first_n == 4, "N=4 is first-4, not listed");
    CHECK(d.contains(0) && d.contains(3) && !d.contains(4), "first-4 membership");
    CHECK(parse("0", 48, d, err) && !d.contains(0), "0 = none");
    CHECK(d.describe(48) == "none", "describe none");
    CHECK(parse("48", 48, d, err) && d.contains(47), "first-N equal to n_layer covers everything");

    // last N
    CHECK(parse("-3", 48, d, err) && d.listed, "-3 parses as a set");
    CHECK(!d.contains(44) && d.contains(45) && d.contains(47) && !d.contains(48), "last-3 of 48 = 45..47");
    CHECK(parse("-48", 48, d, err) && d.contains(0) && d.contains(47), "last-all");
    CHECK(!parse("-49", 48, d, err), "last-49 of 48 refused");

    // explicit set: the Qwen3-235B HiP config [0,1,2,47,46,45] over 48 layers
    CHECK(parse("0-2,45-47", 48, d, err) && d.listed, "0-2,45-47 parses");
    for (int il = 0; il < 48; il++) {
        const bool want = il <= 2 || il >= 45;
        CHECK(d.contains(il) == want, "0-2,45-47 membership");
    }
    CHECK(d.describe(48) == "0-2,45-47", "describe round-trips");

    // the QwQ-32B shape over 64 layers: 0-5, every 4th 9..57, 59-63
    CHECK(parse("0-5,9,13,17,21,25,29,33,37,41,45,49,53,57,59-63", 64, d, err), "qwq-like list parses");
    CHECK(d.contains(9) && d.contains(57) && !d.contains(10) && d.contains(59) && d.contains(63) && !d.contains(58), "qwq-like membership");

    // single layer needs a range or a comma; whitespace tolerated; negative index inside a list counts from the end
    CHECK(parse(" 5-5 ", 48, d, err) && d.contains(5) && !d.contains(4) && !d.contains(6), "5-5 = layer 5 only");
    CHECK(parse("1,-1", 48, d, err) && d.contains(1) && d.contains(47) && !d.contains(46), "negative index in a list");

    // an out-of-model layer index is never dense
    CHECK(!d.contains(100) && !d.contains(-1), "out-of-model il is never dense");

    // must-fail partners
    CHECK(!parse("0-48", 48, d, err) && !err.empty(), "range past the last layer refused, with a message");
    CHECK(!parse("5-2", 48, d, err), "reversed range refused");
    CHECK(!parse("a,b", 48, d, err), "junk refused");
    CHECK(!parse("1,,2", 48, d, err), "empty item refused");
    CHECK(!parse("1-", 48, d, err), "open range refused");
    CHECK(!parse("0-2", 0, d, err), "n_layer 0 refused");

    // cparams round trip
    {
        DenseLayers a; parse("0-2,45-47", 48, a, err);
        DenseLayers b = ik_offload::dense_layers_from_cparams(a.first_n, a.listed, a.mask);
        bool same = true;
        for (int il = 0; il < 48; il++) same = same && (a.contains(il) == b.contains(il));
        CHECK(same, "dense_layers_from_cparams round-trips the set");
        DenseLayers f = ik_offload::dense_layers_from_cparams(3, false, a.mask);
        CHECK(f.contains(2) && !f.contains(3), "unlisted cparams = first N");
    }

    if (g_fail) { std::printf("test-offload-dense-layers: %d FAILED\n", g_fail); return 1; }
    std::printf("test-offload-dense-layers: OK\n");
    return 0;
}
