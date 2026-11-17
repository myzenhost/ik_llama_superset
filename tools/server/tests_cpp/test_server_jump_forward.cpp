// Standalone test for server-jump-forward.h. Build: g++ -std=c++17 -I.. test_server_jump_forward.cpp -o t && ./t
#include "server-jump-forward.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

using namespace server_jump_forward;

int main() {
    // a small vocabulary: id = index. "" is a control token (not indexed).
    //  0 ""   1 "{"   2 "\""   3 "name"   4 "na"   5 "me"   6 "\":"   7 " "   8 "a"   9 "ab"   10 "abc"   11 "n"   12 "name"(dup of 3)
    vocab_index vi;
    vi.build({ "", "{", "\"", "name", "na", "me", "\":", " ", "a", "ab", "abc", "n", "name" });

    // ---- the index ------------------------------------------------------------------------------------------------------
    CHECK(vi.by_piece.at("name") == 3);          // duplicate text: the lowest id wins
    CHECK(vi.by_piece.count("") == 0);           // control token is not indexed
    CHECK(vi.max_len == 4);
    CHECK(vi.is_prefix_token(8));                // "a"  is a prefix of "ab"
    CHECK(vi.is_prefix_token(9));                // "ab" is a prefix of "abc"
    CHECK(!vi.is_prefix_token(10));              // "abc" is a prefix of nothing
    CHECK(vi.is_prefix_token(4));                // "na" is a prefix of "name"
    CHECK(vi.is_prefix_token(11));               // "n"  is a prefix of "na"
    CHECK(!vi.is_prefix_token(5));               // "me"
    CHECK(!vi.is_prefix_token(3) && !vi.is_prefix_token(12));   // "name" (both ids) is a prefix of nothing
    CHECK(vi.is_prefix_token(2));                // the quote is a prefix of the two-character token quote-colon
    CHECK(!vi.is_prefix_token(0));               // "" never
    CHECK(!vi.is_prefix_token(-1) && !vi.is_prefix_token(99));

    // ---- longest-match tokenisation -------------------------------------------------------------------------------------
    {
        std::vector<int> t;
        CHECK(greedy_tokenize(vi, "{\"name\":", t));
        CHECK(t == std::vector<int>({1, 2, 3, 6}));
        CHECK(join(vi, t) == "{\"name\":");
        CHECK(greedy_tokenize(vi, "abc", t) && t == std::vector<int>({10}));          // longest wins over a + ab
        CHECK(greedy_tokenize(vi, "abab", t) && t == std::vector<int>({9, 9}));
        CHECK(!greedy_tokenize(vi, "nam", t));                                       // na + m: there is no "m" token
        CHECK(!greedy_tokenize(vi, "z", t));                                         // no token for it
        CHECK(!greedy_tokenize(vi, "{z}", t));
        CHECK(greedy_tokenize(vi, "", t) && t.empty());
    }

    // ---- make_plan -------------------------------------------------------------------------------------------------------
    {
        // tokenizer list is used when its text IS the forced text
        plan p = make_plan(vi, "{\"name\":", {1, 2, 3, 6}, 16);
        CHECK(p.tokens == std::vector<int>({1, 2, 3, 6}) && p.text == "{\"name\":");
        // a canonical list with a different text (e.g. a leading space added by the tokenizer) is replaced by the longest match
        p = make_plan(vi, "{\"name\":", {7, 1, 2, 3, 6}, 16);
        CHECK(p.tokens == std::vector<int>({1, 2, 3, 6}) && p.text == "{\"name\":");
        // empty canonical list: longest match
        p = make_plan(vi, "{\"name\":", {}, 16);
        CHECK(p.tokens == std::vector<int>({1, 2, 3, 6}));
        // MLC rule: the LAST token is dropped once when it is a strict prefix of another token
        p = make_plan(vi, "{\"na", {}, 16);                 // longest match: { " na ; "na" is a prefix of "name" -> dropped
        CHECK(p.tokens == std::vector<int>({1, 2}) && p.text == "{\"");
        // ... only the last one, and only once
        p = make_plan(vi, "aab", {}, 16);                   // a ab ; "ab" is a prefix of "abc" -> dropped, "a" stays
        CHECK(p.tokens == std::vector<int>({8}) && p.text == "a");
        // a forced string that is one prefix token and nothing else -> no jump
        p = make_plan(vi, "ab", {}, 16);
        CHECK(p.tokens.empty() && p.text.empty());
        // the cap, then the prefix rule on the new last token
        p = make_plan(vi, "{\"name\":", {}, 3);             // { " name  -> 3 tokens, "name" is not a prefix -> kept
        CHECK(p.tokens == std::vector<int>({1, 2, 3}) && p.text == "{\"name");
        p = make_plan(vi, "{\"abc", {}, 3);                 // { " abc
        CHECK(p.tokens == std::vector<int>({1, 2, 10}) && p.text == "{\"abc");
        p = make_plan(vi, "{\"ab", {}, 3);                  // { " ab -> "ab" is a prefix -> dropped
        CHECK(p.tokens == std::vector<int>({1, 2}));
        // controls
        CHECK(make_plan(vi, "", {}, 16).tokens.empty());
        CHECK(make_plan(vi, "{", {1}, 0).tokens.empty());   // max 0
        CHECK(make_plan(vi, "zzz", {}, 16).tokens.empty()); // untokenisable
        p = make_plan(vi, "{", {1}, 16);
        CHECK(p.tokens == std::vector<int>({1}));
    }

    // ---- UTF-8 text passes through unchanged (pieces are bytes of text) ---------------------------------------------------
    {
        vocab_index u;
        u.build({ "", "\xC3\xA9", "e", "\xE2\x82\xAC", "\xE2\x82" });   // e-acute, e, euro sign, a partial euro (broken UTF-8 piece)
        std::vector<int> t;
        CHECK(greedy_tokenize(u, "\xC3\xA9" "e\xE2\x82\xAC", t) && t == std::vector<int>({1, 2, 3}));
        CHECK(u.is_prefix_token(4));                       // the broken piece is a prefix of the euro sign
    }

    if (fails == 0) {
        std::printf("OK server-jump-forward: 0 fails\n");
        return 0;
    }
    std::printf("%d FAIL(s)\n", fails);
    return 1;
}
