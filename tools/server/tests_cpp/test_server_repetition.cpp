// Standalone test for server-repetition.h. Build: g++ -std=c++17 -I.. test_server_repetition.cpp -o t && ./t
#include "server-repetition.h"
#include <cstdio>
#include <vector>

using server_repetition::noncontiguous_span_repeated;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // a 6-token block, then other tokens, the block again, other tokens, the block a third time
    std::vector<int> block = {11, 12, 13, 14, 15, 16};
    std::vector<int> t;
    auto add = [&](const std::vector<int> & v) { t.insert(t.end(), v.begin(), v.end()); };
    add({1, 2, 3}); add(block); add({7, 8, 9, 10}); add(block); add({21, 22}); add(block);
    CHECK(noncontiguous_span_repeated(t, 6, 3));   // 3 separated copies, ends on the block  -> TRUE
    CHECK(!noncontiguous_span_repeated(t, 6, 4));  // only 3 copies                          -> FALSE (opposite answer)
    CHECK(!noncontiguous_span_repeated(t, 7, 3));  // span longer than the block, tail is not repeated
    // the same tokens but the output does NOT end on the block: tail span differs
    std::vector<int> u = t; u.push_back(99);
    CHECK(!noncontiguous_span_repeated(u, 6, 3));
    // overlapping self-similarity must not count as separate copies: 5 x token 4 and min_len 2 count 3 needs 6 tokens
    std::vector<int> v = {4, 4, 4, 4, 4};
    CHECK(!noncontiguous_span_repeated(v, 2, 3));
    CHECK(noncontiguous_span_repeated(v, 1, 3));   // single tokens: 5 non-overlapping copies of "4"
    // disabled inputs
    CHECK(!noncontiguous_span_repeated(t, 0, 3));
    CHECK(!noncontiguous_span_repeated(t, 6, 1));
    CHECK(!noncontiguous_span_repeated(std::vector<int>{}, 1, 2));
    // a long token run well past the window still works on the recent tail
    std::vector<int> w(10000, 5);
    CHECK(noncontiguous_span_repeated(w, 8, 3));
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
