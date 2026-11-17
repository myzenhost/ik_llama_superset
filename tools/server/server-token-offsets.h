#pragma once

// Character offsets of tokens inside the text they spell, for /tokenize with_offsets and
// /chat_tokenize. Standard library only, so a unit test can include it without a model.
//
// The offsets are counted in Unicode code points over the concatenation of the token pieces (the
// detokenized text), which is what a citation highlighter needs when it renders the text it got back.
// A code point is counted when its first byte is reached. A byte-fallback token that holds only the
// tail of a multibyte character therefore gets an empty span [n, n] right after the span that holds
// the character's first bytes; offsets stay monotone and never overlap.
//
// Shape of the idea from text-generation-inference router/src/server.rs (Apache-2.0), where
// /tokenize returns {id, text, start, stop} per token; this is an independent implementation.

#include <cstddef>
#include <string>
#include <vector>

struct token_char_span {
    size_t start;
    size_t stop;
};

inline std::vector<token_char_span> token_char_offsets(const std::vector<std::string> & pieces) {
    std::vector<token_char_span> out;
    out.reserve(pieces.size());
    size_t cps = 0; // code points whose first byte has been seen so far
    for (const std::string & piece : pieces) {
        const size_t start = cps;
        for (unsigned char c : piece) {
            if ((c & 0xC0) != 0x80) { // not a continuation byte
                cps++;
            }
        }
        out.push_back({ start, cps });
    }
    return out;
}
