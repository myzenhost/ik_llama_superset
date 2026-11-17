#include "token-ban.h"

#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // classification
    CHECK(token_ban_classify("", 1) == token_ban_kind::DROP);
    CHECK(token_ban_classify("abc", 0) == token_ban_kind::DROP);
    CHECK(token_ban_classify("ABC", 1) == token_ban_kind::TOKEN);
    CHECK(token_ban_classify("elevenchars", 1) == token_ban_kind::TOKEN);   // 11 bytes: still short
    CHECK(token_ban_classify("twelve chars", 1) == token_ban_kind::PHRASE); // 12 bytes: phrase
    CHECK(token_ban_classify("ab", 2) == token_ban_kind::PHRASE);
    CHECK(token_ban_phrase_budget(4) == 7);

    // substring scan
    const std::vector<std::string> vocab = {"hello", "xAbY", "", "AB", "zzz", "ab", "q"};
    auto ids = token_ban_scan(vocab, {"ab"});
    CHECK((ids == std::vector<int>{1, 3, 5}));            // case-insensitive, substring, ascending, once each
    ids = token_ban_scan(vocab, {"q", "AB", "q"});
    CHECK((ids == std::vector<int>{1, 3, 5, 6}));         // duplicates harmless
    CHECK(token_ban_scan(vocab, {}).empty());
    CHECK(token_ban_scan(vocab, {""}).empty());           // empty pattern never bans everything
    CHECK(token_ban_scan({}, {"a"}).empty());
    // non-ASCII bytes compared exactly
    CHECK((token_ban_scan({"\xC3\x89", "\xC3\xA9"}, {"\xC3\xA9"}) == std::vector<int>{1}));

    printf(fails ? "token-ban: %d FAILED\n" : "token-ban: all passed\n", fails);
    return fails ? 1 : 0;
}
