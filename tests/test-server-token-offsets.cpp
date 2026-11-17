// /tokenize with_offsets: the code-point spans of tokens over the text they spell.
// Standard library only; the helper is a header so no model or server build is needed.
#include "server-token-offsets.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

// utf-8 code points of s, one string per code point
static std::vector<std::string> codepoints(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const unsigned char c = (unsigned char) s[i];
        if ((c & 0xF8) == 0xF0) n = 4; else if ((c & 0xF0) == 0xE0) n = 3; else if ((c & 0xE0) == 0xC0) n = 2;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

static std::string slice(const std::vector<std::string> & cps, size_t a, size_t b) {
    std::string r;
    for (size_t i = a; i < b; i++) r += cps[i];
    return r;
}

int main() {
    // multibyte text split into whole-character pieces: text[start:stop] == piece, monotone, contiguous
    const std::vector<std::string> pieces = { "This", " is", " a", " test", " \xE5\xAA\xBD", " and", " emoji", " \xF0\x9F\xA4\x97" };
    std::string text;
    for (const auto & p : pieces) text += p;
    const auto cps = codepoints(text);
    const auto spans = token_char_offsets(pieces);
    CHECK(spans.size() == pieces.size());
    size_t prev_stop = 0;
    for (size_t i = 0; i < pieces.size(); i++) {
        CHECK(spans[i].start == prev_stop);
        CHECK(spans[i].stop >= spans[i].start);
        CHECK(slice(cps, spans[i].start, spans[i].stop) == pieces[i]);
        prev_stop = spans[i].stop;
    }
    CHECK(prev_stop == cps.size());
    // the byte count would be wrong for this text: the offsets are code points, not bytes
    CHECK(text.size() > cps.size());

    // a character split across byte-fallback tokens: first token holds the lead bytes, the tail is empty
    const std::vector<std::string> split = { "a", "\xE5\xAA", "\xBD", "b" };
    const auto s2 = token_char_offsets(split);
    CHECK(s2[0].start == 0 && s2[0].stop == 1);
    CHECK(s2[1].start == 1 && s2[1].stop == 2);
    CHECK(s2[2].start == 2 && s2[2].stop == 2);
    CHECK(s2[3].start == 2 && s2[3].stop == 3);

    // empty piece (a special token that renders nothing) and an empty list
    const auto s3 = token_char_offsets({ "x", "", "y" });
    CHECK(s3[1].start == 1 && s3[1].stop == 1 && s3[2].start == 1 && s3[2].stop == 2);
    CHECK(token_char_offsets({}).empty());

    std::printf("test-server-token-offsets: OK\n");
    return 0;
}
