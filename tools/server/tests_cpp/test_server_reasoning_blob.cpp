// Standalone test for server-reasoning-blob.h.
// Build: g++ -std=c++17 -I.. -I../../../common test_server_reasoning_blob.cpp -o t && ./t
#include "server-reasoning-blob.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

namespace rb = server_reasoning_blob;

int main() {
    // round trip, including multi-byte UTF-8, newlines and text that looks like markup
    for (const std::string text : { std::string("I should call the tool."), std::string("line1\nline2\n<think>x</think>"),
                                    std::string("\xC3\xA9t\xC3\xA9 \xE2\x9C\x93"), std::string(1, 'a'), std::string(4097, 'z') }) {
        const std::string blob = rb::encode(text);
        CHECK(blob.compare(0, rb::prefix().size(), rb::prefix()) == 0);
        CHECK(blob.find('\n') == std::string::npos);   // one opaque token, safe in any JSON string
        std::string back;
        CHECK(rb::decode(blob, back));
        CHECK(back == text);
    }
    // a foreign blob (OpenAI's are long base64 with no prefix) is refused, so the caller skips it as before
    {
        std::string out = "untouched";
        CHECK(!rb::decode("gAAAAABoX3k2Zm9yZWlnbiBibG9i", out));
        CHECK(!rb::decode("", out));
        CHECK(!rb::decode(rb::prefix(), out));            // prefix with nothing after it
        CHECK(!rb::decode("ikfork-rs0:aGVsbG8=", out));     // another version of the prefix
    }
    // a damaged blob with our prefix is refused, never thrown out of decode()
    {
        std::string out;
        CHECK(!rb::decode(rb::prefix() + "!!!not base64!!!", out));
    }
    CHECK(std::string(rb::include_key()) == "reasoning.encrypted_content");

    if (fails == 0) {
        std::printf("all reasoning-blob checks passed\n");
    }
    return fails == 0 ? 0 : 1;
}
