// Standalone test for server-reasoning-count.h. Build: g++ -std=c++17 -I.. test_server_reasoning_count.cpp -o t && ./t
// RESEARCH-DONE: sglang b4584ab schedule_batch.py:2151-2170 read for the counting rule (see the header).
#include "server-reasoning-count.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

using toks = std::vector<int32_t>;

static int32_t run(server_reasoning_counter & rc, const toks & gen) {
    for (const int32_t t : gen) {
        rc.accept(t);
    }
    return rc.n;
}

int main() {
    // ids: 100 "<think>"  101 "</think>"  7/8 "</" "think>" (a two-token closer)  10 "\n"  1..5 ordinary text
    const toks              open_tag  = { 100 };
    const std::vector<toks> close_tag = { { 101 } };

    // 1. generation prompt leaves the block open ("<think>\n"): 3 reasoning tokens + the closer = 4, then 2 answer tokens
    {
        server_reasoning_counter rc;
        rc.init(open_tag, close_tag, { 9, 100, 10 });
        CHECK(rc.open);
        CHECK(run(rc, { 1, 2, 3, 101, 4, 5 }) == 4);
        CHECK(!rc.open);
        CHECK(rc.n_gen == 6);
    }
    // 2. the model writes the opener itself: opener + 2 + closer = 4
    {
        server_reasoning_counter rc;
        rc.init(open_tag, close_tag, { 9 });
        CHECK(!rc.open);
        CHECK(run(rc, { 100, 1, 2, 101, 3 }) == 4);
    }
    // 3. no thinking at all: 0
    {
        server_reasoning_counter rc;
        rc.init(open_tag, close_tag, { 9 });
        CHECK(run(rc, { 1, 2, 3 }) == 0);
    }
    // 4. a template without thinking tags: the counter has no markers and stays 0, even if the ids appear
    {
        server_reasoning_counter rc;
        rc.init({}, {}, { 100 });
        CHECK(!rc.enabled());
        CHECK(run(rc, { 100, 1, 101 }) == 0);
    }
    // 5. a multi-token closer split across two tokens, plus a second closer listed (the tool-call opener a template may
    //    list among its end tags): every token up to and including the full closer counts
    {
        server_reasoning_counter rc;
        rc.init(open_tag, { { 7, 8 }, { 55 } }, { 100 });
        CHECK(run(rc, { 1, 7, 2, 7, 8, 3 }) == 5);   // 1, 7 (lone "</" is text), 2, 7, 8
        CHECK(!rc.open);
    }
    // 6. interleaved thinking: two blocks add up; text between them does not count
    {
        server_reasoning_counter rc;
        rc.init(open_tag, close_tag, { 100 });
        CHECK(run(rc, { 1, 101, 2, 2, 100, 3, 101, 4 }) == 2 + 3);
    }
    // 7. generation stopped inside the block (max_tokens): everything generated is reasoning
    {
        server_reasoning_counter rc;
        rc.init(open_tag, close_tag, { 100, 10 });
        CHECK(run(rc, { 1, 2, 3 }) == 3);
        CHECK(rc.open);
    }
    // 8. prefill that opens and closes the block (a template that pre-closes thinking: "<think>\n\n</think>\n\n"): 0
    {
        server_reasoning_counter rc;
        rc.init(open_tag, close_tag, { 100, 10, 10, 101, 10 });
        CHECK(!rc.open);
        CHECK(run(rc, { 1, 2 }) == 0);
    }
    // 9. a multi-token opener completed by the model after the prompt wrote its first half: only the generated half counts
    {
        server_reasoning_counter rc;
        rc.init({ 40, 41 }, close_tag, { 9, 40 });
        CHECK(!rc.open);
        CHECK(run(rc, { 41, 1, 101 }) == 3);   // 41, 1, 101
    }
    // 10. init() resets a reused counter (slots are reused across tasks)
    {
        server_reasoning_counter rc;
        rc.init(open_tag, close_tag, { 100 });
        run(rc, { 1, 2 });
        rc.init(open_tag, close_tag, { 9 });
        CHECK(rc.n == 0 && rc.n_gen == 0 && !rc.open);
    }

    if (fails == 0) {
        std::printf("all reasoning-count checks passed\n");
    }
    return fails == 0 ? 0 : 1;
}
