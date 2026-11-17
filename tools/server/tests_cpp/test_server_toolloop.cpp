// Standalone test for server-toolloop.h. Build: g++ -std=c++17 -I.. test_server_toolloop.cpp -o t && ./t
#include "server-toolloop.h"
#include <cstdio>
#include <vector>

using server_toolloop::toolcall_loop_detected;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    const int B = 100, E = 101;
    std::vector<int> call = {B, 7, 8, 9, E};
    std::vector<int> other = {B, 7, 8, 5, E};
    std::vector<int> all;
    auto add = [&](const std::vector<int> & v) { all.insert(all.end(), v.begin(), v.end()); };

    // two identical calls in history, the third one completes in the output
    add({1, 2}); add(call); add({3}); add(call); add({4});
    const size_t n_hist = all.size();
    add(call);
    CHECK(toolcall_loop_detected(all, n_hist, B, E, 3));    // run of 3, ends in output          -> TRUE
    CHECK(!toolcall_loop_detected(all, n_hist, B, E, 4));   // only 3                            -> FALSE (opposite)

    // same three calls but ALL in history: the output holds no completed span
    std::vector<int> h = all;
    CHECK(!toolcall_loop_detected(h, h.size(), B, E, 3));

    // a different call in the middle breaks the run
    std::vector<int> d;
    d.insert(d.end(), call.begin(), call.end());
    d.insert(d.end(), other.begin(), other.end());
    d.insert(d.end(), call.begin(), call.end());
    CHECK(!toolcall_loop_detected(d, 0, B, E, 3));
    CHECK(!toolcall_loop_detected(d, 0, B, E, 2));          // last two spans (other, call) differ

    // unterminated trailing span does not count; begin restarts an open span
    std::vector<int> u = {B, 1, B, 7, 8, 9, E, B, 7, 8, 9, E, B, 7, 8};
    CHECK(!toolcall_loop_detected(u, 0, B, E, 3));
    CHECK(toolcall_loop_detected(u, 0, B, E, 2));           // two identical completed spans, last one in "output"
    // degenerate inputs
    CHECK(!toolcall_loop_detected(all, n_hist, B, B, 3));
    CHECK(!toolcall_loop_detected(all, n_hist, B, E, 1));
    CHECK(!toolcall_loop_detected(std::vector<int>{}, 0, B, E, 2));
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
