#include "unicode.h"
#include "unicode-data.h"
#include "unicode-nfkc-data.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        if (result < 0x80) {
            throw std::invalid_argument("invalid character");
        }
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        if (result < 0x800) {
            throw std::invalid_argument("invalid character");
        }
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        if (result < 0x10000 || result > 0x10ffff) {
            throw std::invalid_argument("invalid character");
        }
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cpt) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cpt && */ cpt <= 0xffff) {
//        result.emplace_back(cpt);
//        return result;
//    }
//    if (0x10000 <= cpt && cpt <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cpt - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cpt - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static std::vector<int> unicode_cpt_to_byte_map() {
    std::vector<int> map;
    for (int ch = 0; ch < 256; ++ch) {
        size_t offset = 0;
        const uint32_t cpt = unicode_cpt_from_utf8(unicode_byte_to_utf8(ch), offset);
        if (cpt >= map.size()) {
            map.resize(cpt + 1, -1);
        }
        map[cpt] = ch;
    }
    return map;
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            // regex: <space>?\p{L}+
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom_k2_horizon(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        // K2-Horizon: letter runs are (?:\p{L}|\p{M}|\u200C|\u200D)+
        auto _is_k2_letter = [&] (const size_t pos) -> bool {
            const uint32_t c = _get_cpt(pos);
            if (c == 0x200C || c == 0x200D) {
                return true;
            }
            const auto f = _get_flags(pos);
            return f.is_letter || f.is_accent_mark;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 0x017F) {
                    cpt_next = 's'; // Unicode case-folding of long s
                }
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (_is_k2_letter(pos) || _is_k2_letter(pos+1)) {  // one or more letters/marks/ZWNJ/ZWJ
                    pos++;
                    while (_is_k2_letter(pos)) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// Qwen2 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
static std::vector<size_t> unicode_regex_split_custom_qwen2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}
            if (flags.is_number) {
                pos++;
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// Qwen3.5 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
// Compared to Qwen2, letter-runs also consume Unicode combining marks (\p{M}): [\p{L}\p{M}]+ instead of \p{L}+
static std::vector<size_t> unicode_regex_split_custom_qwen35(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || flags.is_accent_mark || _get_flags(pos + 1).is_accent_mark || _get_flags(pos+1).is_letter) {
                    pos++;
                    while (_get_flags(pos).is_letter || _get_flags(pos).is_accent_mark) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}
            if (flags.is_number) {
                pos++;
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_accent_mark | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_accent_mark | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

template <typename CharT>
static std::vector<size_t> unicode_regex_split_stl(const std::basic_string<CharT> & text, const std::basic_string<CharT> & regex, const std::vector<size_t> & offsets) {
    using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
    // Bypass bug in MSVC: https://github.com/ggml-org/llama.cpp/issues/17830
    constexpr auto regex_flags = std::regex_constants::ECMAScript;
#else
    constexpr auto regex_flags = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
    std::basic_regex<CharT> expr(regex, regex_flags);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::regex_iterator<BidirIt> it(text.begin() + start, text.begin() + start + offset, expr);
        std::regex_iterator<BidirIt> end;

        int64_t start_idx = 0;
        while (it != end) {
            std::match_results<BidirIt> match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// K2 system regex patterns (from tokenization_kimi.py):
// [\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_kimi_k2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // Pattern 1: [\p{Han}]+ (Chinese characters)
            if (unicode_cpt_is_han(cpt)) {
                while (unicode_cpt_is_han(_get_cpt(pos))) {
                    pos++;
                }
                _add_token(pos);
                continue;
            }

            // Pattern 2 & 3: Letter words excluding Han characters with optional contractions
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // Check if current char is a letter OR if current char could be a leading char and next char is a letter
            bool is_letter_pattern = (flags.is_letter && !unicode_cpt_is_han(cpt)) ||
                                     (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number) &&
                                      _get_flags(pos + 1).is_letter && !unicode_cpt_is_han(_get_cpt(pos + 1)));

            if (is_letter_pattern) {
                // Handle optional leading non-letter/non-number character
                bool has_leading_char = false;
                if (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number)) {
                    has_leading_char = true;
                    pos++;
                }

                // Match letter sequence (excluding Han characters)
                bool has_letters = false;
                while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                    has_letters = true;
                    pos++;
                }

                // Only proceed if we found letters (after potentially skipping leading char)
                if (has_letters || (!has_leading_char && _get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos)))) {
                    if (!has_letters) pos++; // consume the first letter if we didn't already

                    // Continue consuming letters
                    while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                        pos++;
                    }

                    // Check for optional contractions (?:'s|'t|'re|'ve|'m|'ll|'d)
                    if (_get_cpt(pos) == '\'' && pos + 1 < offset_end) {
                        uint32_t cpt_next = unicode_tolower(_get_cpt(pos + 1));
                        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                            pos += 2;
                        } else if (pos + 2 < offset_end) {
                            uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
                            if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                                (cpt_next == 'v' && cpt_next_next == 'e') ||
                                (cpt_next == 'l' && cpt_next_next == 'l')) {
                                pos += 3;
                            }
                        }
                    }

                    _add_token(pos);
                    continue;
                } else if (has_leading_char) {
                    // We consumed a leading char but found no letters, backtrack
                    pos--;
                }
            }

            // Pattern 4: \p{N}{1,3} (numbers 1-3 digits)
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]* (optional space + non-word chars + optional newlines)
            auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
            if (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                // Match optional [\r\n]*
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            // Count whitespace characters
            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // Pattern 6: \s*[\r\n]+ (whitespace with newlines)
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // Pattern 7: \s+(?!\S) (trailing whitespace)
            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // Pattern 8: \s+ (general whitespace)
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // No matches - consume single character
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// General_Category Nd (decimal digit), Unicode 16.0.0 UnicodeData.txt (sha256 ff58e582...edb48f). 16.0.0 and not latest because that
// is what HF tokenizers' \p{Nd} matches: measured, it groups the 16.0 Ol Onal digits (U+1E5F1) and not the 17.0 block at U+11DE0.
// The codepoint flags only carry is_number (\p{N}), which also covers Nl and No (U+00BD, U+2167, ...).
static bool unicode_cpt_is_decimal_digit(const uint32_t cpt) {
    static const uint32_t ranges[][2] = {
        {0x00030, 0x00039}, {0x00660, 0x00669}, {0x006F0, 0x006F9}, {0x007C0, 0x007C9},
        {0x00966, 0x0096F}, {0x009E6, 0x009EF}, {0x00A66, 0x00A6F}, {0x00AE6, 0x00AEF},
        {0x00B66, 0x00B6F}, {0x00BE6, 0x00BEF}, {0x00C66, 0x00C6F}, {0x00CE6, 0x00CEF},
        {0x00D66, 0x00D6F}, {0x00DE6, 0x00DEF}, {0x00E50, 0x00E59}, {0x00ED0, 0x00ED9},
        {0x00F20, 0x00F29}, {0x01040, 0x01049}, {0x01090, 0x01099}, {0x017E0, 0x017E9},
        {0x01810, 0x01819}, {0x01946, 0x0194F}, {0x019D0, 0x019D9}, {0x01A80, 0x01A89},
        {0x01A90, 0x01A99}, {0x01B50, 0x01B59}, {0x01BB0, 0x01BB9}, {0x01C40, 0x01C49},
        {0x01C50, 0x01C59}, {0x0A620, 0x0A629}, {0x0A8D0, 0x0A8D9}, {0x0A900, 0x0A909},
        {0x0A9D0, 0x0A9D9}, {0x0A9F0, 0x0A9F9}, {0x0AA50, 0x0AA59}, {0x0ABF0, 0x0ABF9},
        {0x0FF10, 0x0FF19}, {0x104A0, 0x104A9}, {0x10D30, 0x10D39}, {0x10D40, 0x10D49},
        {0x11066, 0x1106F}, {0x110F0, 0x110F9}, {0x11136, 0x1113F}, {0x111D0, 0x111D9},
        {0x112F0, 0x112F9}, {0x11450, 0x11459}, {0x114D0, 0x114D9}, {0x11650, 0x11659},
        {0x116C0, 0x116C9}, {0x116D0, 0x116E3}, {0x11730, 0x11739}, {0x118E0, 0x118E9},
        {0x11950, 0x11959}, {0x11BF0, 0x11BF9}, {0x11C50, 0x11C59}, {0x11D50, 0x11D59},
        {0x11DA0, 0x11DA9}, {0x11F50, 0x11F59}, {0x16130, 0x16139}, {0x16A60, 0x16A69},
        {0x16AC0, 0x16AC9}, {0x16B50, 0x16B59}, {0x16D70, 0x16D79}, {0x1CCF0, 0x1CCF9},
        {0x1D7CE, 0x1D7FF}, {0x1E140, 0x1E149}, {0x1E2F0, 0x1E2F9}, {0x1E4F0, 0x1E4F9},
        {0x1E5F1, 0x1E5FA}, {0x1E950, 0x1E959}, {0x1FBF0, 0x1FBF9},
    };
    size_t lo = 0, hi = sizeof(ranges) / sizeof(ranges[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cpt < ranges[mid][0]) {
            hi = mid;
        } else if (cpt > ranges[mid][1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

// AFMOE digit handling. tokenizer.json of arcee-ai/Trinity: Digits (individual_digits=false, so any \p{N} run is one piece), then three Splits
// that only touch \p{Nd}: each Nd run inside the numeric run is isolated, and an Nd run of length L splits into a leading L%3 (when nonzero)
// and groups of 3. A non-Nd stretch of the numeric run (U+00BD, U+2167, ...) stays one piece. The 510-digit chunking of the first Split
// changes nothing here because 510 is a multiple of 3.
static std::vector<size_t> unicode_regex_split_custom_afmoe(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const auto flags = _get_flags(pos);

            // Handle digit sequences with special splitting logic
            if (flags.is_number) {
                size_t digit_start = pos;
                size_t digit_count = 0;

                // Count consecutive digits
                while (_get_flags(pos).is_number && pos < offset_end) {
                    digit_count++;
                    pos++;
                }

                // The non-digit stretch before the run is its own piece (Split/Isolated). Without this flush the first group below
                // swallowed it ("  1" instead of "  " + "1"), so the main regex then split the spaces one by one.
                _add_token(digit_start);

                // Within the numeric run, each Nd stretch splits on its own length modulo 3; a non-Nd stretch is one piece.
                // Grouping the whole \p{N} run by its length put "1234567" + U+00BD into 12|345|67+U+00BD (HF: 1|234|567|U+00BD).
                const size_t digit_end = digit_start + digit_count;
                for (size_t sub = digit_start; sub < digit_end; ) {
                    const bool nd = unicode_cpt_is_decimal_digit(cpts[sub]);
                    size_t sub_end = sub;
                    while (sub_end < digit_end && unicode_cpt_is_decimal_digit(cpts[sub_end]) == nd) {
                        sub_end++;
                    }
                    if (!nd) {
                        _add_token(sub_end);
                    } else {
                        size_t current = sub;
                        const size_t remainder = (sub_end - sub) % 3;
                        if (remainder > 0) {
                            _add_token(current + remainder);
                            current += remainder;
                        }
                        while (current < sub_end) {
                            _add_token(current + 3);
                            current += 3;
                        }
                    }
                    sub = sub_end;
                }
                continue;
            }

            // For non-digits, just move forward
            pos++;
        }

        // Add any remaining content
        if (_prev_end < offset_end) {
            _add_token(offset_end);
        }
    }

    return bpe_offsets;
}

// AFMOE CJK / Asian-script split. tokenizer.json of arcee-ai/Trinity (pre_tokenizer[4], Split, Isolated):
//   (?:[一-鿿぀-ゟ゠-ヿ㐀-䶿豈-﫿･-ﾟ⼀-⿟]+|[฀-๿]+|[຀-໿]+|[ក-៿]+
//    |[က-႟ꩠ-ꩿꧠ-꧿]+|[가-힯ᄀ-ᇿ]+)
// i.e. six disjoint script groups, each matched as its OWN run (a Han run directly followed by a Thai run is two matches).
// This is custom C++ and not the generic std::wregex stage because that stage cannot express it: (a) the old class literal in llama-vocab.cpp had
// U+F900 NFC-normalized to U+8C48, which widened the Han range to U+8C48..U+FAFF; (b) with a 16-bit wchar_t (MinGW, MSVC) the std::wstring built at
// the wregex stage truncates astral codepoints, so U+1F999 aliased U+F999 and matched the class.
static int unicode_afmoe_cjk_group(const uint32_t cpt) {
    if ((cpt >= 0x4E00 && cpt <= 0x9FFF) || (cpt >= 0x3040 && cpt <= 0x309F) || (cpt >= 0x30A0 && cpt <= 0x30FF) || (cpt >= 0x3400 && cpt <= 0x4DBF) ||
        (cpt >= 0xF900 && cpt <= 0xFAFF) || (cpt >= 0xFF65 && cpt <= 0xFF9F) || (cpt >= 0x2F00 && cpt <= 0x2FDF)) {
        return 1;
    }
    if (cpt >= 0x0E00 && cpt <= 0x0E7F) return 2;  // Thai
    if (cpt >= 0x0E80 && cpt <= 0x0EFF) return 3;  // Lao
    if (cpt >= 0x1780 && cpt <= 0x17FF) return 4;  // Khmer
    if ((cpt >= 0x1000 && cpt <= 0x109F) || (cpt >= 0xAA60 && cpt <= 0xAA7F) || (cpt >= 0xA9E0 && cpt <= 0xA9FF)) return 5;  // Myanmar
    if ((cpt >= 0xAC00 && cpt <= 0xD7AF) || (cpt >= 0x1100 && cpt <= 0x11FF)) return 6;  // Hangul
    return 0;
}

static std::vector<size_t> unicode_regex_split_custom_afmoe_cjk(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t prev_end = offset_ini;  // end of the last emitted piece
        size_t pos = offset_ini;
        while (pos < offset_end) {
            const int group = unicode_afmoe_cjk_group(cpts[pos]);
            if (group == 0) {
                pos++;
                continue;
            }
            size_t run_end = pos + 1;
            while (run_end < offset_end && unicode_afmoe_cjk_group(cpts[run_end]) == group) {
                run_end++;
            }
            if (pos > prev_end) {
                bpe_offsets.push_back(pos - prev_end);  // the unmatched stretch before this run
            }
            bpe_offsets.push_back(run_end - pos);
            prev_end = run_end;
            pos = run_end;
        }
        if (prev_end < offset_end) {
            bpe_offsets.push_back(offset_end - prev_end);
        }
    }

    return bpe_offsets;
}

// Oniguruma's \w as HF tokenizers sees it: Alphabetic | M | Nd | Pc from Unicode 16.0.0 (UnicodeData.txt + DerivedCoreProperties.txt),
// plus U+00B2 U+00B3 U+00B9 U+00BC U+00BD U+00BE, which onig's Latin-1 ctype table marks as word characters. Checked against HF over
// every codepoint (the '1' + c probe of temp/arex1014/_tinyaya_wordset_1015.py): equal, 144671 codepoints.
static bool unicode_cpt_is_onig_word(const uint32_t cpt) {
    static const uint32_t ranges[][2] = {
        {0x00030, 0x00039}, {0x00041, 0x0005A}, {0x0005F, 0x0005F}, {0x00061, 0x0007A},
        {0x000AA, 0x000AA}, {0x000B2, 0x000B3}, {0x000B5, 0x000B5}, {0x000B9, 0x000BA},
        {0x000BC, 0x000BE}, {0x000C0, 0x000D6}, {0x000D8, 0x000F6}, {0x000F8, 0x002C1},
        {0x002C6, 0x002D1}, {0x002E0, 0x002E4}, {0x002EC, 0x002EC}, {0x002EE, 0x002EE},
        {0x00300, 0x00374}, {0x00376, 0x00377}, {0x0037A, 0x0037D}, {0x0037F, 0x0037F},
        {0x00386, 0x00386}, {0x00388, 0x0038A}, {0x0038C, 0x0038C}, {0x0038E, 0x003A1},
        {0x003A3, 0x003F5}, {0x003F7, 0x00481}, {0x00483, 0x0052F}, {0x00531, 0x00556},
        {0x00559, 0x00559}, {0x00560, 0x00588}, {0x00591, 0x005BD}, {0x005BF, 0x005BF},
        {0x005C1, 0x005C2}, {0x005C4, 0x005C5}, {0x005C7, 0x005C7}, {0x005D0, 0x005EA},
        {0x005EF, 0x005F2}, {0x00610, 0x0061A}, {0x00620, 0x00669}, {0x0066E, 0x006D3},
        {0x006D5, 0x006DC}, {0x006DF, 0x006E8}, {0x006EA, 0x006FC}, {0x006FF, 0x006FF},
        {0x00710, 0x0074A}, {0x0074D, 0x007B1}, {0x007C0, 0x007F5}, {0x007FA, 0x007FA},
        {0x007FD, 0x007FD}, {0x00800, 0x0082D}, {0x00840, 0x0085B}, {0x00860, 0x0086A},
        {0x00870, 0x00887}, {0x00889, 0x0088E}, {0x00897, 0x008E1}, {0x008E3, 0x00963},
        {0x00966, 0x0096F}, {0x00971, 0x00983}, {0x00985, 0x0098C}, {0x0098F, 0x00990},
        {0x00993, 0x009A8}, {0x009AA, 0x009B0}, {0x009B2, 0x009B2}, {0x009B6, 0x009B9},
        {0x009BC, 0x009C4}, {0x009C7, 0x009C8}, {0x009CB, 0x009CE}, {0x009D7, 0x009D7},
        {0x009DC, 0x009DD}, {0x009DF, 0x009E3}, {0x009E6, 0x009F1}, {0x009FC, 0x009FC},
        {0x009FE, 0x009FE}, {0x00A01, 0x00A03}, {0x00A05, 0x00A0A}, {0x00A0F, 0x00A10},
        {0x00A13, 0x00A28}, {0x00A2A, 0x00A30}, {0x00A32, 0x00A33}, {0x00A35, 0x00A36},
        {0x00A38, 0x00A39}, {0x00A3C, 0x00A3C}, {0x00A3E, 0x00A42}, {0x00A47, 0x00A48},
        {0x00A4B, 0x00A4D}, {0x00A51, 0x00A51}, {0x00A59, 0x00A5C}, {0x00A5E, 0x00A5E},
        {0x00A66, 0x00A75}, {0x00A81, 0x00A83}, {0x00A85, 0x00A8D}, {0x00A8F, 0x00A91},
        {0x00A93, 0x00AA8}, {0x00AAA, 0x00AB0}, {0x00AB2, 0x00AB3}, {0x00AB5, 0x00AB9},
        {0x00ABC, 0x00AC5}, {0x00AC7, 0x00AC9}, {0x00ACB, 0x00ACD}, {0x00AD0, 0x00AD0},
        {0x00AE0, 0x00AE3}, {0x00AE6, 0x00AEF}, {0x00AF9, 0x00AFF}, {0x00B01, 0x00B03},
        {0x00B05, 0x00B0C}, {0x00B0F, 0x00B10}, {0x00B13, 0x00B28}, {0x00B2A, 0x00B30},
        {0x00B32, 0x00B33}, {0x00B35, 0x00B39}, {0x00B3C, 0x00B44}, {0x00B47, 0x00B48},
        {0x00B4B, 0x00B4D}, {0x00B55, 0x00B57}, {0x00B5C, 0x00B5D}, {0x00B5F, 0x00B63},
        {0x00B66, 0x00B6F}, {0x00B71, 0x00B71}, {0x00B82, 0x00B83}, {0x00B85, 0x00B8A},
        {0x00B8E, 0x00B90}, {0x00B92, 0x00B95}, {0x00B99, 0x00B9A}, {0x00B9C, 0x00B9C},
        {0x00B9E, 0x00B9F}, {0x00BA3, 0x00BA4}, {0x00BA8, 0x00BAA}, {0x00BAE, 0x00BB9},
        {0x00BBE, 0x00BC2}, {0x00BC6, 0x00BC8}, {0x00BCA, 0x00BCD}, {0x00BD0, 0x00BD0},
        {0x00BD7, 0x00BD7}, {0x00BE6, 0x00BEF}, {0x00C00, 0x00C0C}, {0x00C0E, 0x00C10},
        {0x00C12, 0x00C28}, {0x00C2A, 0x00C39}, {0x00C3C, 0x00C44}, {0x00C46, 0x00C48},
        {0x00C4A, 0x00C4D}, {0x00C55, 0x00C56}, {0x00C58, 0x00C5A}, {0x00C5D, 0x00C5D},
        {0x00C60, 0x00C63}, {0x00C66, 0x00C6F}, {0x00C80, 0x00C83}, {0x00C85, 0x00C8C},
        {0x00C8E, 0x00C90}, {0x00C92, 0x00CA8}, {0x00CAA, 0x00CB3}, {0x00CB5, 0x00CB9},
        {0x00CBC, 0x00CC4}, {0x00CC6, 0x00CC8}, {0x00CCA, 0x00CCD}, {0x00CD5, 0x00CD6},
        {0x00CDD, 0x00CDE}, {0x00CE0, 0x00CE3}, {0x00CE6, 0x00CEF}, {0x00CF1, 0x00CF3},
        {0x00D00, 0x00D0C}, {0x00D0E, 0x00D10}, {0x00D12, 0x00D44}, {0x00D46, 0x00D48},
        {0x00D4A, 0x00D4E}, {0x00D54, 0x00D57}, {0x00D5F, 0x00D63}, {0x00D66, 0x00D6F},
        {0x00D7A, 0x00D7F}, {0x00D81, 0x00D83}, {0x00D85, 0x00D96}, {0x00D9A, 0x00DB1},
        {0x00DB3, 0x00DBB}, {0x00DBD, 0x00DBD}, {0x00DC0, 0x00DC6}, {0x00DCA, 0x00DCA},
        {0x00DCF, 0x00DD4}, {0x00DD6, 0x00DD6}, {0x00DD8, 0x00DDF}, {0x00DE6, 0x00DEF},
        {0x00DF2, 0x00DF3}, {0x00E01, 0x00E3A}, {0x00E40, 0x00E4E}, {0x00E50, 0x00E59},
        {0x00E81, 0x00E82}, {0x00E84, 0x00E84}, {0x00E86, 0x00E8A}, {0x00E8C, 0x00EA3},
        {0x00EA5, 0x00EA5}, {0x00EA7, 0x00EBD}, {0x00EC0, 0x00EC4}, {0x00EC6, 0x00EC6},
        {0x00EC8, 0x00ECE}, {0x00ED0, 0x00ED9}, {0x00EDC, 0x00EDF}, {0x00F00, 0x00F00},
        {0x00F18, 0x00F19}, {0x00F20, 0x00F29}, {0x00F35, 0x00F35}, {0x00F37, 0x00F37},
        {0x00F39, 0x00F39}, {0x00F3E, 0x00F47}, {0x00F49, 0x00F6C}, {0x00F71, 0x00F84},
        {0x00F86, 0x00F97}, {0x00F99, 0x00FBC}, {0x00FC6, 0x00FC6}, {0x01000, 0x01049},
        {0x01050, 0x0109D}, {0x010A0, 0x010C5}, {0x010C7, 0x010C7}, {0x010CD, 0x010CD},
        {0x010D0, 0x010FA}, {0x010FC, 0x01248}, {0x0124A, 0x0124D}, {0x01250, 0x01256},
        {0x01258, 0x01258}, {0x0125A, 0x0125D}, {0x01260, 0x01288}, {0x0128A, 0x0128D},
        {0x01290, 0x012B0}, {0x012B2, 0x012B5}, {0x012B8, 0x012BE}, {0x012C0, 0x012C0},
        {0x012C2, 0x012C5}, {0x012C8, 0x012D6}, {0x012D8, 0x01310}, {0x01312, 0x01315},
        {0x01318, 0x0135A}, {0x0135D, 0x0135F}, {0x01380, 0x0138F}, {0x013A0, 0x013F5},
        {0x013F8, 0x013FD}, {0x01401, 0x0166C}, {0x0166F, 0x0167F}, {0x01681, 0x0169A},
        {0x016A0, 0x016EA}, {0x016EE, 0x016F8}, {0x01700, 0x01715}, {0x0171F, 0x01734},
        {0x01740, 0x01753}, {0x01760, 0x0176C}, {0x0176E, 0x01770}, {0x01772, 0x01773},
        {0x01780, 0x017D3}, {0x017D7, 0x017D7}, {0x017DC, 0x017DD}, {0x017E0, 0x017E9},
        {0x0180B, 0x0180D}, {0x0180F, 0x01819}, {0x01820, 0x01878}, {0x01880, 0x018AA},
        {0x018B0, 0x018F5}, {0x01900, 0x0191E}, {0x01920, 0x0192B}, {0x01930, 0x0193B},
        {0x01946, 0x0196D}, {0x01970, 0x01974}, {0x01980, 0x019AB}, {0x019B0, 0x019C9},
        {0x019D0, 0x019D9}, {0x01A00, 0x01A1B}, {0x01A20, 0x01A5E}, {0x01A60, 0x01A7C},
        {0x01A7F, 0x01A89}, {0x01A90, 0x01A99}, {0x01AA7, 0x01AA7}, {0x01AB0, 0x01ACE},
        {0x01B00, 0x01B4C}, {0x01B50, 0x01B59}, {0x01B6B, 0x01B73}, {0x01B80, 0x01BF3},
        {0x01C00, 0x01C37}, {0x01C40, 0x01C49}, {0x01C4D, 0x01C7D}, {0x01C80, 0x01C8A},
        {0x01C90, 0x01CBA}, {0x01CBD, 0x01CBF}, {0x01CD0, 0x01CD2}, {0x01CD4, 0x01CFA},
        {0x01D00, 0x01F15}, {0x01F18, 0x01F1D}, {0x01F20, 0x01F45}, {0x01F48, 0x01F4D},
        {0x01F50, 0x01F57}, {0x01F59, 0x01F59}, {0x01F5B, 0x01F5B}, {0x01F5D, 0x01F5D},
        {0x01F5F, 0x01F7D}, {0x01F80, 0x01FB4}, {0x01FB6, 0x01FBC}, {0x01FBE, 0x01FBE},
        {0x01FC2, 0x01FC4}, {0x01FC6, 0x01FCC}, {0x01FD0, 0x01FD3}, {0x01FD6, 0x01FDB},
        {0x01FE0, 0x01FEC}, {0x01FF2, 0x01FF4}, {0x01FF6, 0x01FFC}, {0x0203F, 0x02040},
        {0x02054, 0x02054}, {0x02071, 0x02071}, {0x0207F, 0x0207F}, {0x02090, 0x0209C},
        {0x020D0, 0x020F0}, {0x02102, 0x02102}, {0x02107, 0x02107}, {0x0210A, 0x02113},
        {0x02115, 0x02115}, {0x02119, 0x0211D}, {0x02124, 0x02124}, {0x02126, 0x02126},
        {0x02128, 0x02128}, {0x0212A, 0x0212D}, {0x0212F, 0x02139}, {0x0213C, 0x0213F},
        {0x02145, 0x02149}, {0x0214E, 0x0214E}, {0x02160, 0x02188}, {0x024B6, 0x024E9},
        {0x02C00, 0x02CE4}, {0x02CEB, 0x02CF3}, {0x02D00, 0x02D25}, {0x02D27, 0x02D27},
        {0x02D2D, 0x02D2D}, {0x02D30, 0x02D67}, {0x02D6F, 0x02D6F}, {0x02D7F, 0x02D96},
        {0x02DA0, 0x02DA6}, {0x02DA8, 0x02DAE}, {0x02DB0, 0x02DB6}, {0x02DB8, 0x02DBE},
        {0x02DC0, 0x02DC6}, {0x02DC8, 0x02DCE}, {0x02DD0, 0x02DD6}, {0x02DD8, 0x02DDE},
        {0x02DE0, 0x02DFF}, {0x02E2F, 0x02E2F}, {0x03005, 0x03007}, {0x03021, 0x0302F},
        {0x03031, 0x03035}, {0x03038, 0x0303C}, {0x03041, 0x03096}, {0x03099, 0x0309A},
        {0x0309D, 0x0309F}, {0x030A1, 0x030FA}, {0x030FC, 0x030FF}, {0x03105, 0x0312F},
        {0x03131, 0x0318E}, {0x031A0, 0x031BF}, {0x031F0, 0x031FF}, {0x03400, 0x04DBF},
        {0x04E00, 0x0A48C}, {0x0A4D0, 0x0A4FD}, {0x0A500, 0x0A60C}, {0x0A610, 0x0A62B},
        {0x0A640, 0x0A672}, {0x0A674, 0x0A67D}, {0x0A67F, 0x0A6F1}, {0x0A717, 0x0A71F},
        {0x0A722, 0x0A788}, {0x0A78B, 0x0A7CD}, {0x0A7D0, 0x0A7D1}, {0x0A7D3, 0x0A7D3},
        {0x0A7D5, 0x0A7DC}, {0x0A7F2, 0x0A827}, {0x0A82C, 0x0A82C}, {0x0A840, 0x0A873},
        {0x0A880, 0x0A8C5}, {0x0A8D0, 0x0A8D9}, {0x0A8E0, 0x0A8F7}, {0x0A8FB, 0x0A8FB},
        {0x0A8FD, 0x0A92D}, {0x0A930, 0x0A953}, {0x0A960, 0x0A97C}, {0x0A980, 0x0A9C0},
        {0x0A9CF, 0x0A9D9}, {0x0A9E0, 0x0A9FE}, {0x0AA00, 0x0AA36}, {0x0AA40, 0x0AA4D},
        {0x0AA50, 0x0AA59}, {0x0AA60, 0x0AA76}, {0x0AA7A, 0x0AAC2}, {0x0AADB, 0x0AADD},
        {0x0AAE0, 0x0AAEF}, {0x0AAF2, 0x0AAF6}, {0x0AB01, 0x0AB06}, {0x0AB09, 0x0AB0E},
        {0x0AB11, 0x0AB16}, {0x0AB20, 0x0AB26}, {0x0AB28, 0x0AB2E}, {0x0AB30, 0x0AB5A},
        {0x0AB5C, 0x0AB69}, {0x0AB70, 0x0ABEA}, {0x0ABEC, 0x0ABED}, {0x0ABF0, 0x0ABF9},
        {0x0AC00, 0x0D7A3}, {0x0D7B0, 0x0D7C6}, {0x0D7CB, 0x0D7FB}, {0x0F900, 0x0FA6D},
        {0x0FA70, 0x0FAD9}, {0x0FB00, 0x0FB06}, {0x0FB13, 0x0FB17}, {0x0FB1D, 0x0FB28},
        {0x0FB2A, 0x0FB36}, {0x0FB38, 0x0FB3C}, {0x0FB3E, 0x0FB3E}, {0x0FB40, 0x0FB41},
        {0x0FB43, 0x0FB44}, {0x0FB46, 0x0FBB1}, {0x0FBD3, 0x0FD3D}, {0x0FD50, 0x0FD8F},
        {0x0FD92, 0x0FDC7}, {0x0FDF0, 0x0FDFB}, {0x0FE00, 0x0FE0F}, {0x0FE20, 0x0FE2F},
        {0x0FE33, 0x0FE34}, {0x0FE4D, 0x0FE4F}, {0x0FE70, 0x0FE74}, {0x0FE76, 0x0FEFC},
        {0x0FF10, 0x0FF19}, {0x0FF21, 0x0FF3A}, {0x0FF3F, 0x0FF3F}, {0x0FF41, 0x0FF5A},
        {0x0FF66, 0x0FFBE}, {0x0FFC2, 0x0FFC7}, {0x0FFCA, 0x0FFCF}, {0x0FFD2, 0x0FFD7},
        {0x0FFDA, 0x0FFDC}, {0x10000, 0x1000B}, {0x1000D, 0x10026}, {0x10028, 0x1003A},
        {0x1003C, 0x1003D}, {0x1003F, 0x1004D}, {0x10050, 0x1005D}, {0x10080, 0x100FA},
        {0x10140, 0x10174}, {0x101FD, 0x101FD}, {0x10280, 0x1029C}, {0x102A0, 0x102D0},
        {0x102E0, 0x102E0}, {0x10300, 0x1031F}, {0x1032D, 0x1034A}, {0x10350, 0x1037A},
        {0x10380, 0x1039D}, {0x103A0, 0x103C3}, {0x103C8, 0x103CF}, {0x103D1, 0x103D5},
        {0x10400, 0x1049D}, {0x104A0, 0x104A9}, {0x104B0, 0x104D3}, {0x104D8, 0x104FB},
        {0x10500, 0x10527}, {0x10530, 0x10563}, {0x10570, 0x1057A}, {0x1057C, 0x1058A},
        {0x1058C, 0x10592}, {0x10594, 0x10595}, {0x10597, 0x105A1}, {0x105A3, 0x105B1},
        {0x105B3, 0x105B9}, {0x105BB, 0x105BC}, {0x105C0, 0x105F3}, {0x10600, 0x10736},
        {0x10740, 0x10755}, {0x10760, 0x10767}, {0x10780, 0x10785}, {0x10787, 0x107B0},
        {0x107B2, 0x107BA}, {0x10800, 0x10805}, {0x10808, 0x10808}, {0x1080A, 0x10835},
        {0x10837, 0x10838}, {0x1083C, 0x1083C}, {0x1083F, 0x10855}, {0x10860, 0x10876},
        {0x10880, 0x1089E}, {0x108E0, 0x108F2}, {0x108F4, 0x108F5}, {0x10900, 0x10915},
        {0x10920, 0x10939}, {0x10980, 0x109B7}, {0x109BE, 0x109BF}, {0x10A00, 0x10A03},
        {0x10A05, 0x10A06}, {0x10A0C, 0x10A13}, {0x10A15, 0x10A17}, {0x10A19, 0x10A35},
        {0x10A38, 0x10A3A}, {0x10A3F, 0x10A3F}, {0x10A60, 0x10A7C}, {0x10A80, 0x10A9C},
        {0x10AC0, 0x10AC7}, {0x10AC9, 0x10AE6}, {0x10B00, 0x10B35}, {0x10B40, 0x10B55},
        {0x10B60, 0x10B72}, {0x10B80, 0x10B91}, {0x10C00, 0x10C48}, {0x10C80, 0x10CB2},
        {0x10CC0, 0x10CF2}, {0x10D00, 0x10D27}, {0x10D30, 0x10D39}, {0x10D40, 0x10D65},
        {0x10D69, 0x10D6D}, {0x10D6F, 0x10D85}, {0x10E80, 0x10EA9}, {0x10EAB, 0x10EAC},
        {0x10EB0, 0x10EB1}, {0x10EC2, 0x10EC4}, {0x10EFC, 0x10F1C}, {0x10F27, 0x10F27},
        {0x10F30, 0x10F50}, {0x10F70, 0x10F85}, {0x10FB0, 0x10FC4}, {0x10FE0, 0x10FF6},
        {0x11000, 0x11046}, {0x11066, 0x11075}, {0x1107F, 0x110BA}, {0x110C2, 0x110C2},
        {0x110D0, 0x110E8}, {0x110F0, 0x110F9}, {0x11100, 0x11134}, {0x11136, 0x1113F},
        {0x11144, 0x11147}, {0x11150, 0x11173}, {0x11176, 0x11176}, {0x11180, 0x111C4},
        {0x111C9, 0x111CC}, {0x111CE, 0x111DA}, {0x111DC, 0x111DC}, {0x11200, 0x11211},
        {0x11213, 0x11237}, {0x1123E, 0x11241}, {0x11280, 0x11286}, {0x11288, 0x11288},
        {0x1128A, 0x1128D}, {0x1128F, 0x1129D}, {0x1129F, 0x112A8}, {0x112B0, 0x112EA},
        {0x112F0, 0x112F9}, {0x11300, 0x11303}, {0x11305, 0x1130C}, {0x1130F, 0x11310},
        {0x11313, 0x11328}, {0x1132A, 0x11330}, {0x11332, 0x11333}, {0x11335, 0x11339},
        {0x1133B, 0x11344}, {0x11347, 0x11348}, {0x1134B, 0x1134D}, {0x11350, 0x11350},
        {0x11357, 0x11357}, {0x1135D, 0x11363}, {0x11366, 0x1136C}, {0x11370, 0x11374},
        {0x11380, 0x11389}, {0x1138B, 0x1138B}, {0x1138E, 0x1138E}, {0x11390, 0x113B5},
        {0x113B7, 0x113C0}, {0x113C2, 0x113C2}, {0x113C5, 0x113C5}, {0x113C7, 0x113CA},
        {0x113CC, 0x113D3}, {0x113E1, 0x113E2}, {0x11400, 0x1144A}, {0x11450, 0x11459},
        {0x1145E, 0x11461}, {0x11480, 0x114C5}, {0x114C7, 0x114C7}, {0x114D0, 0x114D9},
        {0x11580, 0x115B5}, {0x115B8, 0x115C0}, {0x115D8, 0x115DD}, {0x11600, 0x11640},
        {0x11644, 0x11644}, {0x11650, 0x11659}, {0x11680, 0x116B8}, {0x116C0, 0x116C9},
        {0x116D0, 0x116E3}, {0x11700, 0x1171A}, {0x1171D, 0x1172B}, {0x11730, 0x11739},
        {0x11740, 0x11746}, {0x11800, 0x1183A}, {0x118A0, 0x118E9}, {0x118FF, 0x11906},
        {0x11909, 0x11909}, {0x1190C, 0x11913}, {0x11915, 0x11916}, {0x11918, 0x11935},
        {0x11937, 0x11938}, {0x1193B, 0x11943}, {0x11950, 0x11959}, {0x119A0, 0x119A7},
        {0x119AA, 0x119D7}, {0x119DA, 0x119E1}, {0x119E3, 0x119E4}, {0x11A00, 0x11A3E},
        {0x11A47, 0x11A47}, {0x11A50, 0x11A99}, {0x11A9D, 0x11A9D}, {0x11AB0, 0x11AF8},
        {0x11BC0, 0x11BE0}, {0x11BF0, 0x11BF9}, {0x11C00, 0x11C08}, {0x11C0A, 0x11C36},
        {0x11C38, 0x11C40}, {0x11C50, 0x11C59}, {0x11C72, 0x11C8F}, {0x11C92, 0x11CA7},
        {0x11CA9, 0x11CB6}, {0x11D00, 0x11D06}, {0x11D08, 0x11D09}, {0x11D0B, 0x11D36},
        {0x11D3A, 0x11D3A}, {0x11D3C, 0x11D3D}, {0x11D3F, 0x11D47}, {0x11D50, 0x11D59},
        {0x11D60, 0x11D65}, {0x11D67, 0x11D68}, {0x11D6A, 0x11D8E}, {0x11D90, 0x11D91},
        {0x11D93, 0x11D98}, {0x11DA0, 0x11DA9}, {0x11EE0, 0x11EF6}, {0x11F00, 0x11F10},
        {0x11F12, 0x11F3A}, {0x11F3E, 0x11F42}, {0x11F50, 0x11F5A}, {0x11FB0, 0x11FB0},
        {0x12000, 0x12399}, {0x12400, 0x1246E}, {0x12480, 0x12543}, {0x12F90, 0x12FF0},
        {0x13000, 0x1342F}, {0x13440, 0x13455}, {0x13460, 0x143FA}, {0x14400, 0x14646},
        {0x16100, 0x16139}, {0x16800, 0x16A38}, {0x16A40, 0x16A5E}, {0x16A60, 0x16A69},
        {0x16A70, 0x16ABE}, {0x16AC0, 0x16AC9}, {0x16AD0, 0x16AED}, {0x16AF0, 0x16AF4},
        {0x16B00, 0x16B36}, {0x16B40, 0x16B43}, {0x16B50, 0x16B59}, {0x16B63, 0x16B77},
        {0x16B7D, 0x16B8F}, {0x16D40, 0x16D6C}, {0x16D70, 0x16D79}, {0x16E40, 0x16E7F},
        {0x16F00, 0x16F4A}, {0x16F4F, 0x16F87}, {0x16F8F, 0x16F9F}, {0x16FE0, 0x16FE1},
        {0x16FE3, 0x16FE4}, {0x16FF0, 0x16FF1}, {0x17000, 0x187F7}, {0x18800, 0x18CD5},
        {0x18CFF, 0x18D08}, {0x1AFF0, 0x1AFF3}, {0x1AFF5, 0x1AFFB}, {0x1AFFD, 0x1AFFE},
        {0x1B000, 0x1B122}, {0x1B132, 0x1B132}, {0x1B150, 0x1B152}, {0x1B155, 0x1B155},
        {0x1B164, 0x1B167}, {0x1B170, 0x1B2FB}, {0x1BC00, 0x1BC6A}, {0x1BC70, 0x1BC7C},
        {0x1BC80, 0x1BC88}, {0x1BC90, 0x1BC99}, {0x1BC9D, 0x1BC9E}, {0x1CCF0, 0x1CCF9},
        {0x1CF00, 0x1CF2D}, {0x1CF30, 0x1CF46}, {0x1D165, 0x1D169}, {0x1D16D, 0x1D172},
        {0x1D17B, 0x1D182}, {0x1D185, 0x1D18B}, {0x1D1AA, 0x1D1AD}, {0x1D242, 0x1D244},
        {0x1D400, 0x1D454}, {0x1D456, 0x1D49C}, {0x1D49E, 0x1D49F}, {0x1D4A2, 0x1D4A2},
        {0x1D4A5, 0x1D4A6}, {0x1D4A9, 0x1D4AC}, {0x1D4AE, 0x1D4B9}, {0x1D4BB, 0x1D4BB},
        {0x1D4BD, 0x1D4C3}, {0x1D4C5, 0x1D505}, {0x1D507, 0x1D50A}, {0x1D50D, 0x1D514},
        {0x1D516, 0x1D51C}, {0x1D51E, 0x1D539}, {0x1D53B, 0x1D53E}, {0x1D540, 0x1D544},
        {0x1D546, 0x1D546}, {0x1D54A, 0x1D550}, {0x1D552, 0x1D6A5}, {0x1D6A8, 0x1D6C0},
        {0x1D6C2, 0x1D6DA}, {0x1D6DC, 0x1D6FA}, {0x1D6FC, 0x1D714}, {0x1D716, 0x1D734},
        {0x1D736, 0x1D74E}, {0x1D750, 0x1D76E}, {0x1D770, 0x1D788}, {0x1D78A, 0x1D7A8},
        {0x1D7AA, 0x1D7C2}, {0x1D7C4, 0x1D7CB}, {0x1D7CE, 0x1D7FF}, {0x1DA00, 0x1DA36},
        {0x1DA3B, 0x1DA6C}, {0x1DA75, 0x1DA75}, {0x1DA84, 0x1DA84}, {0x1DA9B, 0x1DA9F},
        {0x1DAA1, 0x1DAAF}, {0x1DF00, 0x1DF1E}, {0x1DF25, 0x1DF2A}, {0x1E000, 0x1E006},
        {0x1E008, 0x1E018}, {0x1E01B, 0x1E021}, {0x1E023, 0x1E024}, {0x1E026, 0x1E02A},
        {0x1E030, 0x1E06D}, {0x1E08F, 0x1E08F}, {0x1E100, 0x1E12C}, {0x1E130, 0x1E13D},
        {0x1E140, 0x1E149}, {0x1E14E, 0x1E14E}, {0x1E290, 0x1E2AE}, {0x1E2C0, 0x1E2F9},
        {0x1E4D0, 0x1E4F9}, {0x1E5D0, 0x1E5FA}, {0x1E7E0, 0x1E7E6}, {0x1E7E8, 0x1E7EB},
        {0x1E7ED, 0x1E7EE}, {0x1E7F0, 0x1E7FE}, {0x1E800, 0x1E8C4}, {0x1E8D0, 0x1E8D6},
        {0x1E900, 0x1E94B}, {0x1E950, 0x1E959}, {0x1EE00, 0x1EE03}, {0x1EE05, 0x1EE1F},
        {0x1EE21, 0x1EE22}, {0x1EE24, 0x1EE24}, {0x1EE27, 0x1EE27}, {0x1EE29, 0x1EE32},
        {0x1EE34, 0x1EE37}, {0x1EE39, 0x1EE39}, {0x1EE3B, 0x1EE3B}, {0x1EE42, 0x1EE42},
        {0x1EE47, 0x1EE47}, {0x1EE49, 0x1EE49}, {0x1EE4B, 0x1EE4B}, {0x1EE4D, 0x1EE4F},
        {0x1EE51, 0x1EE52}, {0x1EE54, 0x1EE54}, {0x1EE57, 0x1EE57}, {0x1EE59, 0x1EE59},
        {0x1EE5B, 0x1EE5B}, {0x1EE5D, 0x1EE5D}, {0x1EE5F, 0x1EE5F}, {0x1EE61, 0x1EE62},
        {0x1EE64, 0x1EE64}, {0x1EE67, 0x1EE6A}, {0x1EE6C, 0x1EE72}, {0x1EE74, 0x1EE77},
        {0x1EE79, 0x1EE7C}, {0x1EE7E, 0x1EE7E}, {0x1EE80, 0x1EE89}, {0x1EE8B, 0x1EE9B},
        {0x1EEA1, 0x1EEA3}, {0x1EEA5, 0x1EEA9}, {0x1EEAB, 0x1EEBB}, {0x1F130, 0x1F149},
        {0x1F150, 0x1F169}, {0x1F170, 0x1F189}, {0x1FBF0, 0x1FBF9}, {0x20000, 0x2A6DF},
        {0x2A700, 0x2B739}, {0x2B740, 0x2B81D}, {0x2B820, 0x2CEA1}, {0x2CEB0, 0x2EBE0},
        {0x2EBF0, 0x2EE5D}, {0x2F800, 0x2FA1D}, {0x30000, 0x3134A}, {0x31350, 0x323AF},
        {0xE0100, 0xE01EF},
    };
    size_t lo = 0, hi = sizeof(ranges) / sizeof(ranges[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cpt < ranges[mid][0]) {
            hi = mid;
        } else if (cpt > ranges[mid][1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

// regex (tiny_aya tokenizer.json, Split, Isolated): \d{1,3}(?=(?:\d{3})*\b)
// A \d run splits into a leading length%3 and groups of 3 only when a word boundary follows the run: the last digit is a word
// character, so the next codepoint must be a non-word one, or the piece must end there. Otherwise nothing in the run matches
// and the digits stay in the unmatched stretch ("2024" + U+5E74, "1234_5", "1234567" + U+00BD are not split here).
static std::vector<size_t> unicode_regex_split_custom_tiny_aya_digits(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t prev_end = offset_ini;
        size_t pos = offset_ini;
        while (pos < offset_end) {
            if (!unicode_cpt_is_decimal_digit(cpts[pos])) {
                pos++;
                continue;
            }
            size_t run_end = pos;
            while (run_end < offset_end && unicode_cpt_is_decimal_digit(cpts[run_end])) {
                run_end++;
            }
            if (run_end == offset_end || !unicode_cpt_is_onig_word(cpts[run_end])) {
                if (pos > prev_end) {
                    bpe_offsets.push_back(pos - prev_end);
                }
                const size_t lead = (run_end - pos) % 3;
                if (lead > 0) {
                    bpe_offsets.push_back(lead);
                }
                for (size_t g = pos + lead; g < run_end; g += 3) {
                    bpe_offsets.push_back(3);
                }
                prev_end = run_end;
            }
            pos = run_end;
        }
        if (prev_end < offset_end) {
            bpe_offsets.push_back(offset_end - prev_end);
        }
    }

    return bpe_offsets;
}

// regex: [0-9]
// isolates every ASCII digit as its own piece. Custom (not std::wregex) so that no astral codepoint can alias an ASCII digit through a
// 16-bit wchar_t truncation (U+10030 -> '0').
static std::vector<size_t> unicode_regex_split_custom_ascii_digits(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t prev_end = offset_ini;
        for (size_t pos = offset_ini; pos < offset_end; ++pos) {
            if (cpts[pos] >= '0' && cpts[pos] <= '9') {
                if (pos > prev_end) {
                    bpe_offsets.push_back(pos - prev_end);
                }
                bpe_offsets.push_back(1);
                prev_end = pos + 1;
            }
        }
        if (prev_end < offset_end) {
            bpe_offsets.push_back(offset_end - prev_end);
        }
    }

    return bpe_offsets;
}

// regex: [^\n]+|[\n]+
// splits text into runs of non-newline characters and runs of newline characters
static std::vector<size_t> unicode_regex_split_custom_newlines(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t pos = offset_ini;
        while (pos < offset_end) {
            const bool is_newline = (cpts[pos] == '\n');
            const size_t run_start = pos;
            while (pos < offset_end && (cpts[pos] == '\n') == is_newline) {
                pos++;
            }
            bpe_offsets.push_back(pos - run_start);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
    } else if (
           regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_qwen2(text, offsets);
    } else if (
           regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_qwen35(text, offsets);
    } else if (
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        // K2-Horizon: llama3 splitter with marks + ZWNJ/ZWJ inside letter runs
        // (the generic std::regex fallback cannot parse \p{..} on MSVC)
        bpe_offsets = unicode_regex_split_custom_k2_horizon(text, offsets);
    } else if (regex_expr == "\\p{Han}+") {
        // K2's first pattern - handle all K2 patterns together
        bpe_offsets = unicode_regex_split_custom_kimi_k2(text, offsets);
    } else if (regex_expr == "\\p{AFMoE_digits}") {
        // AFMOE digit pattern - use custom implementation for proper splitting
        bpe_offsets = unicode_regex_split_custom_afmoe(text, offsets);
    } else if (regex_expr == "\\p{AFMoE_cjk}") {
        bpe_offsets = unicode_regex_split_custom_afmoe_cjk(text, offsets);
    } else if (regex_expr == "[0-9]") {
        bpe_offsets = unicode_regex_split_custom_ascii_digits(text, offsets);
    } else if (regex_expr == "[^\\n]+|[\\n]+") {
        bpe_offsets = unicode_regex_split_custom_newlines(text, offsets);
    } else if (regex_expr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
        // tiny_aya digit grouping pattern from tokenizer.json:
        //   {"type": "Split", "pattern": {"Regex": "\\d{1,3}(?=(?:\\d{3})*\\b)"}, "behavior": "Isolated"}
        // Splits a digit run into groups of 3 from the right (1234567 -> 1, 234, 567), but only when \b follows the run.
        // It used to reuse the afmoe splitter, which ignores \b and groups non-\d numbers too.
        bpe_offsets = unicode_regex_split_custom_tiny_aya_digits(text, offsets);
    }

    return bpe_offsets;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;

    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        result.push_back(cpt);
        return result;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

// NFKC (UAX #15): full compatibility decomposition, canonical ordering, canonical composition.
// Tables: src/unicode-nfkc-data.cpp (scripts/gen-unicode-nfkc-data.py); Hangul syllables are algorithmic (UAX #15 / ch. 3.12).
static constexpr uint32_t HANGUL_S_BASE = 0xAC00, HANGUL_L_BASE = 0x1100, HANGUL_V_BASE = 0x1161, HANGUL_T_BASE = 0x11A7;
static constexpr uint32_t HANGUL_L_COUNT = 19, HANGUL_V_COUNT = 21, HANGUL_T_COUNT = 28;
static constexpr uint32_t HANGUL_N_COUNT = HANGUL_V_COUNT * HANGUL_T_COUNT, HANGUL_S_COUNT = HANGUL_L_COUNT * HANGUL_N_COUNT;

static uint8_t unicode_cpt_ccc(uint32_t cpt) {
    const unicode_ccc_range * end = unicode_ccc_ranges + unicode_ccc_ranges_size;
    const unicode_ccc_range * it  = std::upper_bound(unicode_ccc_ranges, end, cpt,
        [] (uint32_t c, const unicode_ccc_range & r) { return c < r.first; });
    if (it == unicode_ccc_ranges) {
        return 0;
    }
    --it;
    return cpt <= it->last ? it->ccc : 0;
}

static void unicode_cpt_append_nfkd(uint32_t cpt, std::vector<uint32_t> & out) {
    if (cpt - HANGUL_S_BASE < HANGUL_S_COUNT) {
        const uint32_t s = cpt - HANGUL_S_BASE;
        out.push_back(HANGUL_L_BASE + s / HANGUL_N_COUNT);
        out.push_back(HANGUL_V_BASE + (s % HANGUL_N_COUNT) / HANGUL_T_COUNT);
        if (s % HANGUL_T_COUNT != 0) {
            out.push_back(HANGUL_T_BASE + s % HANGUL_T_COUNT);
        }
        return;
    }
    const unicode_nfkd_entry * end = unicode_nfkd_map + unicode_nfkd_map_size;
    const unicode_nfkd_entry * it  = std::lower_bound(unicode_nfkd_map, end, cpt,
        [] (const unicode_nfkd_entry & e, uint32_t c) { return e.cpt < c; });
    if (it != end && it->cpt == cpt) {
        out.insert(out.end(), unicode_nfkd_data + it->offset, unicode_nfkd_data + it->offset + it->length);
    } else {
        out.push_back(cpt);
    }
}

// the primary composite of (first, second), or 0 when there is none
static uint32_t unicode_cpt_compose(uint32_t first, uint32_t second) {
    if (first - HANGUL_L_BASE < HANGUL_L_COUNT && second - HANGUL_V_BASE < HANGUL_V_COUNT) {
        return HANGUL_S_BASE + ((first - HANGUL_L_BASE) * HANGUL_V_COUNT + (second - HANGUL_V_BASE)) * HANGUL_T_COUNT;
    }
    if (first - HANGUL_S_BASE < HANGUL_S_COUNT && (first - HANGUL_S_BASE) % HANGUL_T_COUNT == 0 &&
        second - HANGUL_T_BASE - 1 < HANGUL_T_COUNT - 1) {
        return first + (second - HANGUL_T_BASE);
    }
    const unicode_compose_entry * end = unicode_compose_pairs + unicode_compose_pairs_size;
    const unicode_compose_entry * it  = std::lower_bound(unicode_compose_pairs, end, std::make_pair(first, second),
        [] (const unicode_compose_entry & e, const std::pair<uint32_t, uint32_t> & k) {
            return e.first < k.first || (e.first == k.first && e.second < k.second);
        });
    return (it != end && it->first == first && it->second == second) ? it->composite : 0;
}

std::vector<uint32_t> unicode_cpts_normalize_nfkc(const std::vector<uint32_t> & cpts) {
    std::vector<uint32_t> d;
    d.reserve(cpts.size());
    for (const uint32_t cpt : cpts) {
        unicode_cpt_append_nfkd(cpt, d);
    }

    // canonical ordering: a stable sort by ccc inside each run of non-starters
    std::vector<uint8_t> ccc(d.size());
    for (size_t i = 0; i < d.size(); ++i) {
        ccc[i] = unicode_cpt_ccc(d[i]);
    }
    for (size_t i = 1; i < d.size(); ++i) {
        for (size_t j = i; j > 0 && ccc[j] != 0 && ccc[j - 1] > ccc[j]; --j) {
            std::swap(d[j - 1], d[j]);
            std::swap(ccc[j - 1], ccc[j]);
        }
    }

    // canonical composition: C joins the last starter S unless a character between them is blocking (ccc 0, or a ccc
    // >= C's). Every character kept between S and C is a non-starter (a kept starter becomes the new S), so the last
    // one's ccc decides.
    std::vector<uint32_t> out;
    out.reserve(d.size());
    size_t  starter = SIZE_MAX;
    uint8_t last_cc = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        const uint32_t c  = d[i];
        const uint8_t  cc = ccc[i];
        if (starter != SIZE_MAX) {
            const bool adjacent = out.size() - 1 == starter;
            if (adjacent || (last_cc != 0 && last_cc < cc)) {
                const uint32_t comp = unicode_cpt_compose(out[starter], c);
                if (comp != 0) {
                    out[starter] = comp;
                    continue;
                }
            }
        }
        if (cc == 0) {
            starter = out.size();
            last_cc = 0;
        } else {
            last_cc = cc;
        }
        out.push_back(c);
    }
    return out;
}

std::string unicode_normalize_nfkc(const std::string & text) {
    // ASCII is NFKC-invariant: skip the decode for the common case
    if (std::all_of(text.begin(), text.end(), [] (char ch) { return static_cast<uint8_t>(ch) < 0x80; })) {
        return text;
    }
    std::vector<uint32_t> cpts;
    cpts.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        try {
            cpts.push_back(unicode_cpt_from_utf8(text, offset));
        } catch (const std::invalid_argument &) {
            // not valid UTF-8: leave the bytes as they are for the byte-level tokenizer rather than invent U+FFFD
            return text;
        }
    }
    return unicode_cpts_to_utf8(unicode_cpts_normalize_nfkc(cpts));
}

static uint8_t unicode_canonical_class(uint32_t cpt) {
    const auto it = unicode_combining_class.find(cpt);
    return it == unicode_combining_class.end() ? 0 : it->second;
}

static void unicode_canonical_decompose(uint32_t cpt, std::vector<uint32_t> & result, std::vector<uint32_t> & pending) {
    pending.push_back(cpt);
    while (!pending.empty()) {
        cpt = pending.back();
        pending.pop_back();
        // Hangul syllables use algorithmic decomposition (UAX #15).
        if (cpt >= 0xAC00 && cpt < 0xAC00 + 11172) {
            const uint32_t index = cpt - 0xAC00;
            result.push_back(0x1100 + index / 588);
            result.push_back(0x1161 + (index % 588) / 28);
            if (index % 28) {
                result.push_back(0x11A7 + index % 28);
            }
            continue;
        }
        const auto it = unicode_decomposition.find(cpt);
        if (it == unicode_decomposition.end()) {
            result.push_back(cpt);
            continue;
        }
        if (it->second.second) {
            pending.push_back(it->second.second);
        }
        pending.push_back(it->second.first);
    }
}

static uint32_t unicode_canonical_compose(uint32_t first, uint32_t second) {
    if (first >= 0x1100 && first < 0x1100 + 19 && second >= 0x1161 && second < 0x1161 + 21) {
        return 0xAC00 + ((first - 0x1100) * 21 + second - 0x1161) * 28;
    }
    if (first >= 0xAC00 && first < 0xAC00 + 11172 && (first - 0xAC00) % 28 == 0 && second > 0x11A7 && second < 0x11A7 + 28) {
        return first + second - 0x11A7;
    }
    const auto it = unicode_composition.find((uint64_t(first) << 32) | second);
    return it == unicode_composition.end() ? 0 : it->second;
}

std::vector<uint32_t> unicode_cpts_normalize_nfc(const std::vector<uint32_t> & cpts) {
    std::vector<uint32_t> decomposed;
    decomposed.reserve(cpts.size());
    std::vector<uint32_t> pending;
    for (const uint32_t cpt : cpts) {
        unicode_canonical_decompose(cpt, decomposed, pending);
    }

    // Sort each run of combining marks; equal classes retain their order.
    for (size_t i = 0; i < decomposed.size();) {
        if (unicode_canonical_class(decomposed[i]) == 0) {
            ++i;
        }
        const size_t begin = i;
        while (i < decomposed.size() && unicode_canonical_class(decomposed[i]) != 0) {
            ++i;
        }
        std::stable_sort(decomposed.begin() + begin, decomposed.begin() + i, [](uint32_t a, uint32_t b) {
            return unicode_canonical_class(a) < unicode_canonical_class(b);
        });
    }

    std::vector<uint32_t> result;
    result.reserve(decomposed.size());
    size_t starter = 0;
    uint8_t previous_class = 0;
    for (const uint32_t cpt : decomposed) {
        const uint8_t current_class = unicode_canonical_class(cpt);
        const uint32_t composite = !result.empty() && (previous_class == 0 || previous_class < current_class)
            ? unicode_canonical_compose(result[starter], cpt) : 0;
        if (composite) {
            result[starter] = composite;
        } else {
            if (current_class == 0) {
                starter = result.size();
            }
            result.push_back(cpt);
            previous_class = current_class;
        }
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument & /*ex*/) {
            // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond llama_tokenize
            ++offset;
            result.emplace_back(0xFFFD); // replacement character
        }
    }
    return result;
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;  // undefined
    }
    size_t offset = 0;
    return unicode_cpt_flags_from_cpt(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint8_t unicode_cpt_to_byte(uint32_t cpt) {
    static const std::vector<int> map = unicode_cpt_to_byte_map();
    if (cpt >= map.size() || map[cpt] < 0) {
        throw std::out_of_range("unicode_cpt_to_byte: not a byte token code point");
    }
    return map[cpt];
}

uint32_t unicode_tolower(uint32_t cpt) {
    // NOTE (S69, 2026-09-07): the 2026-09-02 fix (67ea3f9e) made this a hash `.find()` because
    // at that time unicode-data.h declared `unicode_map_lowercase` as a std::unordered_map, and
    // running std::lower_bound over an unordered_map's bucket order is what silently mapped 0/26
    // ASCII letters (measured then: all-MiniLM-L6-v2 "The Access to Public Records Act" ->
    // 100,100,...,100). THIS MERGE resolved unicode-data.h/.cpp (a sibling file, not touched here)
    // to upstream's std::initializer_list<std::pair<uint32_t,uint32_t>> form -- confirmed by
    // reading unicode-data.h/.cpp directly, not by memory of the old fix. A hash `.find()` does
    // not compile against an initializer_list (no such member) -- caught by an MSVC /Zs syntax
    // check, not by inspection. Re-verified the FIX'S INTENT against the new container: parsed
    // unicode-data.cpp's unicode_map_lowercase table (1433 entries) and confirmed by script it is
    // strictly ascending by key (0 sortedness violations) and that a lower_bound lookup over it
    // maps all 26/26 of A-Z to the correct lowercase codepoint. Binary search is therefore VALID
    // again for this container (its precondition -- sortedness -- now holds, unlike the
    // unordered_map it used to be), and matches the idiom already used for the sibling tables in
    // this same file (unicode_ranges_nfd via std::upper_bound, unicode_ranges_flags materialized
    // once into a vector). Restored to binary search rather than re-adding a hash map.
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // Return the original code point if no lowercase mapping is found
}

bool unicode_cpt_is_han(uint32_t cpt) {
    // Han character ranges (Chinese/CJK characters)
    // CJK Unified Ideographs (most common)
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;

    // CJK Extension A
    if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;

    // CJK Extension B
    if (cpt >= 0x20000 && cpt <= 0x2A6DF) return true;

    // CJK Extension C
    if (cpt >= 0x2A700 && cpt <= 0x2B73F) return true;

    // CJK Extension D
    if (cpt >= 0x2B740 && cpt <= 0x2B81F) return true;

    // CJK Extension E
    if (cpt >= 0x2B820 && cpt <= 0x2CEAF) return true;

    // CJK Extension F
    if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) return true;

    // CJK Compatibility Ideographs
    if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;

    // CJK Compatibility Ideographs Supplement
    if (cpt >= 0x2F800 && cpt <= 0x2FA1F) return true;

    return false;
}

// General category of a cased or modifier letter (Lu, Ll, Lt, Lm), Unicode 16.0.0 UnicodeData.txt (sha256 ff58e582...edb48f):
// runs of consecutive codepoints with the same category. A letter not listed is Lo. These are exactly the sets HF tokenizers'
// Oniguruma matches for \p{Lu} \p{Ll} \p{Lt} \p{Lm} \p{Lo}: measured over every codepoint (1858 / 2258 / 31 / 404 / 136477),
// no difference in either direction. The codepoint flags cannot give this: is_lowercase / is_uppercase only mark codepoints
// that are the target of a case mapping, so they miss an Ll with no case partner such as U+0138.
enum unicode_letter_case : uint8_t { UCASE_NONE = 0, UCASE_LU, UCASE_LL, UCASE_LT, UCASE_LM };

static unicode_letter_case unicode_cpt_letter_case(const uint32_t cpt) {
    struct case_range { uint32_t first; uint32_t last; unicode_letter_case cat; };
    static const case_range ranges[] = {
    {0x00041,0x0005A,UCASE_LU}, {0x00061,0x0007A,UCASE_LL}, {0x000B5,0x000B5,UCASE_LL}, {0x000C0,0x000D6,UCASE_LU},
    {0x000D8,0x000DE,UCASE_LU}, {0x000DF,0x000F6,UCASE_LL}, {0x000F8,0x000FF,UCASE_LL}, {0x00100,0x00100,UCASE_LU},
    {0x00101,0x00101,UCASE_LL}, {0x00102,0x00102,UCASE_LU}, {0x00103,0x00103,UCASE_LL}, {0x00104,0x00104,UCASE_LU},
    {0x00105,0x00105,UCASE_LL}, {0x00106,0x00106,UCASE_LU}, {0x00107,0x00107,UCASE_LL}, {0x00108,0x00108,UCASE_LU},
    {0x00109,0x00109,UCASE_LL}, {0x0010A,0x0010A,UCASE_LU}, {0x0010B,0x0010B,UCASE_LL}, {0x0010C,0x0010C,UCASE_LU},
    {0x0010D,0x0010D,UCASE_LL}, {0x0010E,0x0010E,UCASE_LU}, {0x0010F,0x0010F,UCASE_LL}, {0x00110,0x00110,UCASE_LU},
    {0x00111,0x00111,UCASE_LL}, {0x00112,0x00112,UCASE_LU}, {0x00113,0x00113,UCASE_LL}, {0x00114,0x00114,UCASE_LU},
    {0x00115,0x00115,UCASE_LL}, {0x00116,0x00116,UCASE_LU}, {0x00117,0x00117,UCASE_LL}, {0x00118,0x00118,UCASE_LU},
    {0x00119,0x00119,UCASE_LL}, {0x0011A,0x0011A,UCASE_LU}, {0x0011B,0x0011B,UCASE_LL}, {0x0011C,0x0011C,UCASE_LU},
    {0x0011D,0x0011D,UCASE_LL}, {0x0011E,0x0011E,UCASE_LU}, {0x0011F,0x0011F,UCASE_LL}, {0x00120,0x00120,UCASE_LU},
    {0x00121,0x00121,UCASE_LL}, {0x00122,0x00122,UCASE_LU}, {0x00123,0x00123,UCASE_LL}, {0x00124,0x00124,UCASE_LU},
    {0x00125,0x00125,UCASE_LL}, {0x00126,0x00126,UCASE_LU}, {0x00127,0x00127,UCASE_LL}, {0x00128,0x00128,UCASE_LU},
    {0x00129,0x00129,UCASE_LL}, {0x0012A,0x0012A,UCASE_LU}, {0x0012B,0x0012B,UCASE_LL}, {0x0012C,0x0012C,UCASE_LU},
    {0x0012D,0x0012D,UCASE_LL}, {0x0012E,0x0012E,UCASE_LU}, {0x0012F,0x0012F,UCASE_LL}, {0x00130,0x00130,UCASE_LU},
    {0x00131,0x00131,UCASE_LL}, {0x00132,0x00132,UCASE_LU}, {0x00133,0x00133,UCASE_LL}, {0x00134,0x00134,UCASE_LU},
    {0x00135,0x00135,UCASE_LL}, {0x00136,0x00136,UCASE_LU}, {0x00137,0x00138,UCASE_LL}, {0x00139,0x00139,UCASE_LU},
    {0x0013A,0x0013A,UCASE_LL}, {0x0013B,0x0013B,UCASE_LU}, {0x0013C,0x0013C,UCASE_LL}, {0x0013D,0x0013D,UCASE_LU},
    {0x0013E,0x0013E,UCASE_LL}, {0x0013F,0x0013F,UCASE_LU}, {0x00140,0x00140,UCASE_LL}, {0x00141,0x00141,UCASE_LU},
    {0x00142,0x00142,UCASE_LL}, {0x00143,0x00143,UCASE_LU}, {0x00144,0x00144,UCASE_LL}, {0x00145,0x00145,UCASE_LU},
    {0x00146,0x00146,UCASE_LL}, {0x00147,0x00147,UCASE_LU}, {0x00148,0x00149,UCASE_LL}, {0x0014A,0x0014A,UCASE_LU},
    {0x0014B,0x0014B,UCASE_LL}, {0x0014C,0x0014C,UCASE_LU}, {0x0014D,0x0014D,UCASE_LL}, {0x0014E,0x0014E,UCASE_LU},
    {0x0014F,0x0014F,UCASE_LL}, {0x00150,0x00150,UCASE_LU}, {0x00151,0x00151,UCASE_LL}, {0x00152,0x00152,UCASE_LU},
    {0x00153,0x00153,UCASE_LL}, {0x00154,0x00154,UCASE_LU}, {0x00155,0x00155,UCASE_LL}, {0x00156,0x00156,UCASE_LU},
    {0x00157,0x00157,UCASE_LL}, {0x00158,0x00158,UCASE_LU}, {0x00159,0x00159,UCASE_LL}, {0x0015A,0x0015A,UCASE_LU},
    {0x0015B,0x0015B,UCASE_LL}, {0x0015C,0x0015C,UCASE_LU}, {0x0015D,0x0015D,UCASE_LL}, {0x0015E,0x0015E,UCASE_LU},
    {0x0015F,0x0015F,UCASE_LL}, {0x00160,0x00160,UCASE_LU}, {0x00161,0x00161,UCASE_LL}, {0x00162,0x00162,UCASE_LU},
    {0x00163,0x00163,UCASE_LL}, {0x00164,0x00164,UCASE_LU}, {0x00165,0x00165,UCASE_LL}, {0x00166,0x00166,UCASE_LU},
    {0x00167,0x00167,UCASE_LL}, {0x00168,0x00168,UCASE_LU}, {0x00169,0x00169,UCASE_LL}, {0x0016A,0x0016A,UCASE_LU},
    {0x0016B,0x0016B,UCASE_LL}, {0x0016C,0x0016C,UCASE_LU}, {0x0016D,0x0016D,UCASE_LL}, {0x0016E,0x0016E,UCASE_LU},
    {0x0016F,0x0016F,UCASE_LL}, {0x00170,0x00170,UCASE_LU}, {0x00171,0x00171,UCASE_LL}, {0x00172,0x00172,UCASE_LU},
    {0x00173,0x00173,UCASE_LL}, {0x00174,0x00174,UCASE_LU}, {0x00175,0x00175,UCASE_LL}, {0x00176,0x00176,UCASE_LU},
    {0x00177,0x00177,UCASE_LL}, {0x00178,0x00179,UCASE_LU}, {0x0017A,0x0017A,UCASE_LL}, {0x0017B,0x0017B,UCASE_LU},
    {0x0017C,0x0017C,UCASE_LL}, {0x0017D,0x0017D,UCASE_LU}, {0x0017E,0x00180,UCASE_LL}, {0x00181,0x00182,UCASE_LU},
    {0x00183,0x00183,UCASE_LL}, {0x00184,0x00184,UCASE_LU}, {0x00185,0x00185,UCASE_LL}, {0x00186,0x00187,UCASE_LU},
    {0x00188,0x00188,UCASE_LL}, {0x00189,0x0018B,UCASE_LU}, {0x0018C,0x0018D,UCASE_LL}, {0x0018E,0x00191,UCASE_LU},
    {0x00192,0x00192,UCASE_LL}, {0x00193,0x00194,UCASE_LU}, {0x00195,0x00195,UCASE_LL}, {0x00196,0x00198,UCASE_LU},
    {0x00199,0x0019B,UCASE_LL}, {0x0019C,0x0019D,UCASE_LU}, {0x0019E,0x0019E,UCASE_LL}, {0x0019F,0x001A0,UCASE_LU},
    {0x001A1,0x001A1,UCASE_LL}, {0x001A2,0x001A2,UCASE_LU}, {0x001A3,0x001A3,UCASE_LL}, {0x001A4,0x001A4,UCASE_LU},
    {0x001A5,0x001A5,UCASE_LL}, {0x001A6,0x001A7,UCASE_LU}, {0x001A8,0x001A8,UCASE_LL}, {0x001A9,0x001A9,UCASE_LU},
    {0x001AA,0x001AB,UCASE_LL}, {0x001AC,0x001AC,UCASE_LU}, {0x001AD,0x001AD,UCASE_LL}, {0x001AE,0x001AF,UCASE_LU},
    {0x001B0,0x001B0,UCASE_LL}, {0x001B1,0x001B3,UCASE_LU}, {0x001B4,0x001B4,UCASE_LL}, {0x001B5,0x001B5,UCASE_LU},
    {0x001B6,0x001B6,UCASE_LL}, {0x001B7,0x001B8,UCASE_LU}, {0x001B9,0x001BA,UCASE_LL}, {0x001BC,0x001BC,UCASE_LU},
    {0x001BD,0x001BF,UCASE_LL}, {0x001C4,0x001C4,UCASE_LU}, {0x001C5,0x001C5,UCASE_LT}, {0x001C6,0x001C6,UCASE_LL},
    {0x001C7,0x001C7,UCASE_LU}, {0x001C8,0x001C8,UCASE_LT}, {0x001C9,0x001C9,UCASE_LL}, {0x001CA,0x001CA,UCASE_LU},
    {0x001CB,0x001CB,UCASE_LT}, {0x001CC,0x001CC,UCASE_LL}, {0x001CD,0x001CD,UCASE_LU}, {0x001CE,0x001CE,UCASE_LL},
    {0x001CF,0x001CF,UCASE_LU}, {0x001D0,0x001D0,UCASE_LL}, {0x001D1,0x001D1,UCASE_LU}, {0x001D2,0x001D2,UCASE_LL},
    {0x001D3,0x001D3,UCASE_LU}, {0x001D4,0x001D4,UCASE_LL}, {0x001D5,0x001D5,UCASE_LU}, {0x001D6,0x001D6,UCASE_LL},
    {0x001D7,0x001D7,UCASE_LU}, {0x001D8,0x001D8,UCASE_LL}, {0x001D9,0x001D9,UCASE_LU}, {0x001DA,0x001DA,UCASE_LL},
    {0x001DB,0x001DB,UCASE_LU}, {0x001DC,0x001DD,UCASE_LL}, {0x001DE,0x001DE,UCASE_LU}, {0x001DF,0x001DF,UCASE_LL},
    {0x001E0,0x001E0,UCASE_LU}, {0x001E1,0x001E1,UCASE_LL}, {0x001E2,0x001E2,UCASE_LU}, {0x001E3,0x001E3,UCASE_LL},
    {0x001E4,0x001E4,UCASE_LU}, {0x001E5,0x001E5,UCASE_LL}, {0x001E6,0x001E6,UCASE_LU}, {0x001E7,0x001E7,UCASE_LL},
    {0x001E8,0x001E8,UCASE_LU}, {0x001E9,0x001E9,UCASE_LL}, {0x001EA,0x001EA,UCASE_LU}, {0x001EB,0x001EB,UCASE_LL},
    {0x001EC,0x001EC,UCASE_LU}, {0x001ED,0x001ED,UCASE_LL}, {0x001EE,0x001EE,UCASE_LU}, {0x001EF,0x001F0,UCASE_LL},
    {0x001F1,0x001F1,UCASE_LU}, {0x001F2,0x001F2,UCASE_LT}, {0x001F3,0x001F3,UCASE_LL}, {0x001F4,0x001F4,UCASE_LU},
    {0x001F5,0x001F5,UCASE_LL}, {0x001F6,0x001F8,UCASE_LU}, {0x001F9,0x001F9,UCASE_LL}, {0x001FA,0x001FA,UCASE_LU},
    {0x001FB,0x001FB,UCASE_LL}, {0x001FC,0x001FC,UCASE_LU}, {0x001FD,0x001FD,UCASE_LL}, {0x001FE,0x001FE,UCASE_LU},
    {0x001FF,0x001FF,UCASE_LL}, {0x00200,0x00200,UCASE_LU}, {0x00201,0x00201,UCASE_LL}, {0x00202,0x00202,UCASE_LU},
    {0x00203,0x00203,UCASE_LL}, {0x00204,0x00204,UCASE_LU}, {0x00205,0x00205,UCASE_LL}, {0x00206,0x00206,UCASE_LU},
    {0x00207,0x00207,UCASE_LL}, {0x00208,0x00208,UCASE_LU}, {0x00209,0x00209,UCASE_LL}, {0x0020A,0x0020A,UCASE_LU},
    {0x0020B,0x0020B,UCASE_LL}, {0x0020C,0x0020C,UCASE_LU}, {0x0020D,0x0020D,UCASE_LL}, {0x0020E,0x0020E,UCASE_LU},
    {0x0020F,0x0020F,UCASE_LL}, {0x00210,0x00210,UCASE_LU}, {0x00211,0x00211,UCASE_LL}, {0x00212,0x00212,UCASE_LU},
    {0x00213,0x00213,UCASE_LL}, {0x00214,0x00214,UCASE_LU}, {0x00215,0x00215,UCASE_LL}, {0x00216,0x00216,UCASE_LU},
    {0x00217,0x00217,UCASE_LL}, {0x00218,0x00218,UCASE_LU}, {0x00219,0x00219,UCASE_LL}, {0x0021A,0x0021A,UCASE_LU},
    {0x0021B,0x0021B,UCASE_LL}, {0x0021C,0x0021C,UCASE_LU}, {0x0021D,0x0021D,UCASE_LL}, {0x0021E,0x0021E,UCASE_LU},
    {0x0021F,0x0021F,UCASE_LL}, {0x00220,0x00220,UCASE_LU}, {0x00221,0x00221,UCASE_LL}, {0x00222,0x00222,UCASE_LU},
    {0x00223,0x00223,UCASE_LL}, {0x00224,0x00224,UCASE_LU}, {0x00225,0x00225,UCASE_LL}, {0x00226,0x00226,UCASE_LU},
    {0x00227,0x00227,UCASE_LL}, {0x00228,0x00228,UCASE_LU}, {0x00229,0x00229,UCASE_LL}, {0x0022A,0x0022A,UCASE_LU},
    {0x0022B,0x0022B,UCASE_LL}, {0x0022C,0x0022C,UCASE_LU}, {0x0022D,0x0022D,UCASE_LL}, {0x0022E,0x0022E,UCASE_LU},
    {0x0022F,0x0022F,UCASE_LL}, {0x00230,0x00230,UCASE_LU}, {0x00231,0x00231,UCASE_LL}, {0x00232,0x00232,UCASE_LU},
    {0x00233,0x00239,UCASE_LL}, {0x0023A,0x0023B,UCASE_LU}, {0x0023C,0x0023C,UCASE_LL}, {0x0023D,0x0023E,UCASE_LU},
    {0x0023F,0x00240,UCASE_LL}, {0x00241,0x00241,UCASE_LU}, {0x00242,0x00242,UCASE_LL}, {0x00243,0x00246,UCASE_LU},
    {0x00247,0x00247,UCASE_LL}, {0x00248,0x00248,UCASE_LU}, {0x00249,0x00249,UCASE_LL}, {0x0024A,0x0024A,UCASE_LU},
    {0x0024B,0x0024B,UCASE_LL}, {0x0024C,0x0024C,UCASE_LU}, {0x0024D,0x0024D,UCASE_LL}, {0x0024E,0x0024E,UCASE_LU},
    {0x0024F,0x00293,UCASE_LL}, {0x00295,0x002AF,UCASE_LL}, {0x002B0,0x002C1,UCASE_LM}, {0x002C6,0x002D1,UCASE_LM},
    {0x002E0,0x002E4,UCASE_LM}, {0x002EC,0x002EC,UCASE_LM}, {0x002EE,0x002EE,UCASE_LM}, {0x00370,0x00370,UCASE_LU},
    {0x00371,0x00371,UCASE_LL}, {0x00372,0x00372,UCASE_LU}, {0x00373,0x00373,UCASE_LL}, {0x00374,0x00374,UCASE_LM},
    {0x00376,0x00376,UCASE_LU}, {0x00377,0x00377,UCASE_LL}, {0x0037A,0x0037A,UCASE_LM}, {0x0037B,0x0037D,UCASE_LL},
    {0x0037F,0x0037F,UCASE_LU}, {0x00386,0x00386,UCASE_LU}, {0x00388,0x0038A,UCASE_LU}, {0x0038C,0x0038C,UCASE_LU},
    {0x0038E,0x0038F,UCASE_LU}, {0x00390,0x00390,UCASE_LL}, {0x00391,0x003A1,UCASE_LU}, {0x003A3,0x003AB,UCASE_LU},
    {0x003AC,0x003CE,UCASE_LL}, {0x003CF,0x003CF,UCASE_LU}, {0x003D0,0x003D1,UCASE_LL}, {0x003D2,0x003D4,UCASE_LU},
    {0x003D5,0x003D7,UCASE_LL}, {0x003D8,0x003D8,UCASE_LU}, {0x003D9,0x003D9,UCASE_LL}, {0x003DA,0x003DA,UCASE_LU},
    {0x003DB,0x003DB,UCASE_LL}, {0x003DC,0x003DC,UCASE_LU}, {0x003DD,0x003DD,UCASE_LL}, {0x003DE,0x003DE,UCASE_LU},
    {0x003DF,0x003DF,UCASE_LL}, {0x003E0,0x003E0,UCASE_LU}, {0x003E1,0x003E1,UCASE_LL}, {0x003E2,0x003E2,UCASE_LU},
    {0x003E3,0x003E3,UCASE_LL}, {0x003E4,0x003E4,UCASE_LU}, {0x003E5,0x003E5,UCASE_LL}, {0x003E6,0x003E6,UCASE_LU},
    {0x003E7,0x003E7,UCASE_LL}, {0x003E8,0x003E8,UCASE_LU}, {0x003E9,0x003E9,UCASE_LL}, {0x003EA,0x003EA,UCASE_LU},
    {0x003EB,0x003EB,UCASE_LL}, {0x003EC,0x003EC,UCASE_LU}, {0x003ED,0x003ED,UCASE_LL}, {0x003EE,0x003EE,UCASE_LU},
    {0x003EF,0x003F3,UCASE_LL}, {0x003F4,0x003F4,UCASE_LU}, {0x003F5,0x003F5,UCASE_LL}, {0x003F7,0x003F7,UCASE_LU},
    {0x003F8,0x003F8,UCASE_LL}, {0x003F9,0x003FA,UCASE_LU}, {0x003FB,0x003FC,UCASE_LL}, {0x003FD,0x0042F,UCASE_LU},
    {0x00430,0x0045F,UCASE_LL}, {0x00460,0x00460,UCASE_LU}, {0x00461,0x00461,UCASE_LL}, {0x00462,0x00462,UCASE_LU},
    {0x00463,0x00463,UCASE_LL}, {0x00464,0x00464,UCASE_LU}, {0x00465,0x00465,UCASE_LL}, {0x00466,0x00466,UCASE_LU},
    {0x00467,0x00467,UCASE_LL}, {0x00468,0x00468,UCASE_LU}, {0x00469,0x00469,UCASE_LL}, {0x0046A,0x0046A,UCASE_LU},
    {0x0046B,0x0046B,UCASE_LL}, {0x0046C,0x0046C,UCASE_LU}, {0x0046D,0x0046D,UCASE_LL}, {0x0046E,0x0046E,UCASE_LU},
    {0x0046F,0x0046F,UCASE_LL}, {0x00470,0x00470,UCASE_LU}, {0x00471,0x00471,UCASE_LL}, {0x00472,0x00472,UCASE_LU},
    {0x00473,0x00473,UCASE_LL}, {0x00474,0x00474,UCASE_LU}, {0x00475,0x00475,UCASE_LL}, {0x00476,0x00476,UCASE_LU},
    {0x00477,0x00477,UCASE_LL}, {0x00478,0x00478,UCASE_LU}, {0x00479,0x00479,UCASE_LL}, {0x0047A,0x0047A,UCASE_LU},
    {0x0047B,0x0047B,UCASE_LL}, {0x0047C,0x0047C,UCASE_LU}, {0x0047D,0x0047D,UCASE_LL}, {0x0047E,0x0047E,UCASE_LU},
    {0x0047F,0x0047F,UCASE_LL}, {0x00480,0x00480,UCASE_LU}, {0x00481,0x00481,UCASE_LL}, {0x0048A,0x0048A,UCASE_LU},
    {0x0048B,0x0048B,UCASE_LL}, {0x0048C,0x0048C,UCASE_LU}, {0x0048D,0x0048D,UCASE_LL}, {0x0048E,0x0048E,UCASE_LU},
    {0x0048F,0x0048F,UCASE_LL}, {0x00490,0x00490,UCASE_LU}, {0x00491,0x00491,UCASE_LL}, {0x00492,0x00492,UCASE_LU},
    {0x00493,0x00493,UCASE_LL}, {0x00494,0x00494,UCASE_LU}, {0x00495,0x00495,UCASE_LL}, {0x00496,0x00496,UCASE_LU},
    {0x00497,0x00497,UCASE_LL}, {0x00498,0x00498,UCASE_LU}, {0x00499,0x00499,UCASE_LL}, {0x0049A,0x0049A,UCASE_LU},
    {0x0049B,0x0049B,UCASE_LL}, {0x0049C,0x0049C,UCASE_LU}, {0x0049D,0x0049D,UCASE_LL}, {0x0049E,0x0049E,UCASE_LU},
    {0x0049F,0x0049F,UCASE_LL}, {0x004A0,0x004A0,UCASE_LU}, {0x004A1,0x004A1,UCASE_LL}, {0x004A2,0x004A2,UCASE_LU},
    {0x004A3,0x004A3,UCASE_LL}, {0x004A4,0x004A4,UCASE_LU}, {0x004A5,0x004A5,UCASE_LL}, {0x004A6,0x004A6,UCASE_LU},
    {0x004A7,0x004A7,UCASE_LL}, {0x004A8,0x004A8,UCASE_LU}, {0x004A9,0x004A9,UCASE_LL}, {0x004AA,0x004AA,UCASE_LU},
    {0x004AB,0x004AB,UCASE_LL}, {0x004AC,0x004AC,UCASE_LU}, {0x004AD,0x004AD,UCASE_LL}, {0x004AE,0x004AE,UCASE_LU},
    {0x004AF,0x004AF,UCASE_LL}, {0x004B0,0x004B0,UCASE_LU}, {0x004B1,0x004B1,UCASE_LL}, {0x004B2,0x004B2,UCASE_LU},
    {0x004B3,0x004B3,UCASE_LL}, {0x004B4,0x004B4,UCASE_LU}, {0x004B5,0x004B5,UCASE_LL}, {0x004B6,0x004B6,UCASE_LU},
    {0x004B7,0x004B7,UCASE_LL}, {0x004B8,0x004B8,UCASE_LU}, {0x004B9,0x004B9,UCASE_LL}, {0x004BA,0x004BA,UCASE_LU},
    {0x004BB,0x004BB,UCASE_LL}, {0x004BC,0x004BC,UCASE_LU}, {0x004BD,0x004BD,UCASE_LL}, {0x004BE,0x004BE,UCASE_LU},
    {0x004BF,0x004BF,UCASE_LL}, {0x004C0,0x004C1,UCASE_LU}, {0x004C2,0x004C2,UCASE_LL}, {0x004C3,0x004C3,UCASE_LU},
    {0x004C4,0x004C4,UCASE_LL}, {0x004C5,0x004C5,UCASE_LU}, {0x004C6,0x004C6,UCASE_LL}, {0x004C7,0x004C7,UCASE_LU},
    {0x004C8,0x004C8,UCASE_LL}, {0x004C9,0x004C9,UCASE_LU}, {0x004CA,0x004CA,UCASE_LL}, {0x004CB,0x004CB,UCASE_LU},
    {0x004CC,0x004CC,UCASE_LL}, {0x004CD,0x004CD,UCASE_LU}, {0x004CE,0x004CF,UCASE_LL}, {0x004D0,0x004D0,UCASE_LU},
    {0x004D1,0x004D1,UCASE_LL}, {0x004D2,0x004D2,UCASE_LU}, {0x004D3,0x004D3,UCASE_LL}, {0x004D4,0x004D4,UCASE_LU},
    {0x004D5,0x004D5,UCASE_LL}, {0x004D6,0x004D6,UCASE_LU}, {0x004D7,0x004D7,UCASE_LL}, {0x004D8,0x004D8,UCASE_LU},
    {0x004D9,0x004D9,UCASE_LL}, {0x004DA,0x004DA,UCASE_LU}, {0x004DB,0x004DB,UCASE_LL}, {0x004DC,0x004DC,UCASE_LU},
    {0x004DD,0x004DD,UCASE_LL}, {0x004DE,0x004DE,UCASE_LU}, {0x004DF,0x004DF,UCASE_LL}, {0x004E0,0x004E0,UCASE_LU},
    {0x004E1,0x004E1,UCASE_LL}, {0x004E2,0x004E2,UCASE_LU}, {0x004E3,0x004E3,UCASE_LL}, {0x004E4,0x004E4,UCASE_LU},
    {0x004E5,0x004E5,UCASE_LL}, {0x004E6,0x004E6,UCASE_LU}, {0x004E7,0x004E7,UCASE_LL}, {0x004E8,0x004E8,UCASE_LU},
    {0x004E9,0x004E9,UCASE_LL}, {0x004EA,0x004EA,UCASE_LU}, {0x004EB,0x004EB,UCASE_LL}, {0x004EC,0x004EC,UCASE_LU},
    {0x004ED,0x004ED,UCASE_LL}, {0x004EE,0x004EE,UCASE_LU}, {0x004EF,0x004EF,UCASE_LL}, {0x004F0,0x004F0,UCASE_LU},
    {0x004F1,0x004F1,UCASE_LL}, {0x004F2,0x004F2,UCASE_LU}, {0x004F3,0x004F3,UCASE_LL}, {0x004F4,0x004F4,UCASE_LU},
    {0x004F5,0x004F5,UCASE_LL}, {0x004F6,0x004F6,UCASE_LU}, {0x004F7,0x004F7,UCASE_LL}, {0x004F8,0x004F8,UCASE_LU},
    {0x004F9,0x004F9,UCASE_LL}, {0x004FA,0x004FA,UCASE_LU}, {0x004FB,0x004FB,UCASE_LL}, {0x004FC,0x004FC,UCASE_LU},
    {0x004FD,0x004FD,UCASE_LL}, {0x004FE,0x004FE,UCASE_LU}, {0x004FF,0x004FF,UCASE_LL}, {0x00500,0x00500,UCASE_LU},
    {0x00501,0x00501,UCASE_LL}, {0x00502,0x00502,UCASE_LU}, {0x00503,0x00503,UCASE_LL}, {0x00504,0x00504,UCASE_LU},
    {0x00505,0x00505,UCASE_LL}, {0x00506,0x00506,UCASE_LU}, {0x00507,0x00507,UCASE_LL}, {0x00508,0x00508,UCASE_LU},
    {0x00509,0x00509,UCASE_LL}, {0x0050A,0x0050A,UCASE_LU}, {0x0050B,0x0050B,UCASE_LL}, {0x0050C,0x0050C,UCASE_LU},
    {0x0050D,0x0050D,UCASE_LL}, {0x0050E,0x0050E,UCASE_LU}, {0x0050F,0x0050F,UCASE_LL}, {0x00510,0x00510,UCASE_LU},
    {0x00511,0x00511,UCASE_LL}, {0x00512,0x00512,UCASE_LU}, {0x00513,0x00513,UCASE_LL}, {0x00514,0x00514,UCASE_LU},
    {0x00515,0x00515,UCASE_LL}, {0x00516,0x00516,UCASE_LU}, {0x00517,0x00517,UCASE_LL}, {0x00518,0x00518,UCASE_LU},
    {0x00519,0x00519,UCASE_LL}, {0x0051A,0x0051A,UCASE_LU}, {0x0051B,0x0051B,UCASE_LL}, {0x0051C,0x0051C,UCASE_LU},
    {0x0051D,0x0051D,UCASE_LL}, {0x0051E,0x0051E,UCASE_LU}, {0x0051F,0x0051F,UCASE_LL}, {0x00520,0x00520,UCASE_LU},
    {0x00521,0x00521,UCASE_LL}, {0x00522,0x00522,UCASE_LU}, {0x00523,0x00523,UCASE_LL}, {0x00524,0x00524,UCASE_LU},
    {0x00525,0x00525,UCASE_LL}, {0x00526,0x00526,UCASE_LU}, {0x00527,0x00527,UCASE_LL}, {0x00528,0x00528,UCASE_LU},
    {0x00529,0x00529,UCASE_LL}, {0x0052A,0x0052A,UCASE_LU}, {0x0052B,0x0052B,UCASE_LL}, {0x0052C,0x0052C,UCASE_LU},
    {0x0052D,0x0052D,UCASE_LL}, {0x0052E,0x0052E,UCASE_LU}, {0x0052F,0x0052F,UCASE_LL}, {0x00531,0x00556,UCASE_LU},
    {0x00559,0x00559,UCASE_LM}, {0x00560,0x00588,UCASE_LL}, {0x00640,0x00640,UCASE_LM}, {0x006E5,0x006E6,UCASE_LM},
    {0x007F4,0x007F5,UCASE_LM}, {0x007FA,0x007FA,UCASE_LM}, {0x0081A,0x0081A,UCASE_LM}, {0x00824,0x00824,UCASE_LM},
    {0x00828,0x00828,UCASE_LM}, {0x008C9,0x008C9,UCASE_LM}, {0x00971,0x00971,UCASE_LM}, {0x00E46,0x00E46,UCASE_LM},
    {0x00EC6,0x00EC6,UCASE_LM}, {0x010A0,0x010C5,UCASE_LU}, {0x010C7,0x010C7,UCASE_LU}, {0x010CD,0x010CD,UCASE_LU},
    {0x010D0,0x010FA,UCASE_LL}, {0x010FC,0x010FC,UCASE_LM}, {0x010FD,0x010FF,UCASE_LL}, {0x013A0,0x013F5,UCASE_LU},
    {0x013F8,0x013FD,UCASE_LL}, {0x017D7,0x017D7,UCASE_LM}, {0x01843,0x01843,UCASE_LM}, {0x01AA7,0x01AA7,UCASE_LM},
    {0x01C78,0x01C7D,UCASE_LM}, {0x01C80,0x01C88,UCASE_LL}, {0x01C89,0x01C89,UCASE_LU}, {0x01C8A,0x01C8A,UCASE_LL},
    {0x01C90,0x01CBA,UCASE_LU}, {0x01CBD,0x01CBF,UCASE_LU}, {0x01D00,0x01D2B,UCASE_LL}, {0x01D2C,0x01D6A,UCASE_LM},
    {0x01D6B,0x01D77,UCASE_LL}, {0x01D78,0x01D78,UCASE_LM}, {0x01D79,0x01D9A,UCASE_LL}, {0x01D9B,0x01DBF,UCASE_LM},
    {0x01E00,0x01E00,UCASE_LU}, {0x01E01,0x01E01,UCASE_LL}, {0x01E02,0x01E02,UCASE_LU}, {0x01E03,0x01E03,UCASE_LL},
    {0x01E04,0x01E04,UCASE_LU}, {0x01E05,0x01E05,UCASE_LL}, {0x01E06,0x01E06,UCASE_LU}, {0x01E07,0x01E07,UCASE_LL},
    {0x01E08,0x01E08,UCASE_LU}, {0x01E09,0x01E09,UCASE_LL}, {0x01E0A,0x01E0A,UCASE_LU}, {0x01E0B,0x01E0B,UCASE_LL},
    {0x01E0C,0x01E0C,UCASE_LU}, {0x01E0D,0x01E0D,UCASE_LL}, {0x01E0E,0x01E0E,UCASE_LU}, {0x01E0F,0x01E0F,UCASE_LL},
    {0x01E10,0x01E10,UCASE_LU}, {0x01E11,0x01E11,UCASE_LL}, {0x01E12,0x01E12,UCASE_LU}, {0x01E13,0x01E13,UCASE_LL},
    {0x01E14,0x01E14,UCASE_LU}, {0x01E15,0x01E15,UCASE_LL}, {0x01E16,0x01E16,UCASE_LU}, {0x01E17,0x01E17,UCASE_LL},
    {0x01E18,0x01E18,UCASE_LU}, {0x01E19,0x01E19,UCASE_LL}, {0x01E1A,0x01E1A,UCASE_LU}, {0x01E1B,0x01E1B,UCASE_LL},
    {0x01E1C,0x01E1C,UCASE_LU}, {0x01E1D,0x01E1D,UCASE_LL}, {0x01E1E,0x01E1E,UCASE_LU}, {0x01E1F,0x01E1F,UCASE_LL},
    {0x01E20,0x01E20,UCASE_LU}, {0x01E21,0x01E21,UCASE_LL}, {0x01E22,0x01E22,UCASE_LU}, {0x01E23,0x01E23,UCASE_LL},
    {0x01E24,0x01E24,UCASE_LU}, {0x01E25,0x01E25,UCASE_LL}, {0x01E26,0x01E26,UCASE_LU}, {0x01E27,0x01E27,UCASE_LL},
    {0x01E28,0x01E28,UCASE_LU}, {0x01E29,0x01E29,UCASE_LL}, {0x01E2A,0x01E2A,UCASE_LU}, {0x01E2B,0x01E2B,UCASE_LL},
    {0x01E2C,0x01E2C,UCASE_LU}, {0x01E2D,0x01E2D,UCASE_LL}, {0x01E2E,0x01E2E,UCASE_LU}, {0x01E2F,0x01E2F,UCASE_LL},
    {0x01E30,0x01E30,UCASE_LU}, {0x01E31,0x01E31,UCASE_LL}, {0x01E32,0x01E32,UCASE_LU}, {0x01E33,0x01E33,UCASE_LL},
    {0x01E34,0x01E34,UCASE_LU}, {0x01E35,0x01E35,UCASE_LL}, {0x01E36,0x01E36,UCASE_LU}, {0x01E37,0x01E37,UCASE_LL},
    {0x01E38,0x01E38,UCASE_LU}, {0x01E39,0x01E39,UCASE_LL}, {0x01E3A,0x01E3A,UCASE_LU}, {0x01E3B,0x01E3B,UCASE_LL},
    {0x01E3C,0x01E3C,UCASE_LU}, {0x01E3D,0x01E3D,UCASE_LL}, {0x01E3E,0x01E3E,UCASE_LU}, {0x01E3F,0x01E3F,UCASE_LL},
    {0x01E40,0x01E40,UCASE_LU}, {0x01E41,0x01E41,UCASE_LL}, {0x01E42,0x01E42,UCASE_LU}, {0x01E43,0x01E43,UCASE_LL},
    {0x01E44,0x01E44,UCASE_LU}, {0x01E45,0x01E45,UCASE_LL}, {0x01E46,0x01E46,UCASE_LU}, {0x01E47,0x01E47,UCASE_LL},
    {0x01E48,0x01E48,UCASE_LU}, {0x01E49,0x01E49,UCASE_LL}, {0x01E4A,0x01E4A,UCASE_LU}, {0x01E4B,0x01E4B,UCASE_LL},
    {0x01E4C,0x01E4C,UCASE_LU}, {0x01E4D,0x01E4D,UCASE_LL}, {0x01E4E,0x01E4E,UCASE_LU}, {0x01E4F,0x01E4F,UCASE_LL},
    {0x01E50,0x01E50,UCASE_LU}, {0x01E51,0x01E51,UCASE_LL}, {0x01E52,0x01E52,UCASE_LU}, {0x01E53,0x01E53,UCASE_LL},
    {0x01E54,0x01E54,UCASE_LU}, {0x01E55,0x01E55,UCASE_LL}, {0x01E56,0x01E56,UCASE_LU}, {0x01E57,0x01E57,UCASE_LL},
    {0x01E58,0x01E58,UCASE_LU}, {0x01E59,0x01E59,UCASE_LL}, {0x01E5A,0x01E5A,UCASE_LU}, {0x01E5B,0x01E5B,UCASE_LL},
    {0x01E5C,0x01E5C,UCASE_LU}, {0x01E5D,0x01E5D,UCASE_LL}, {0x01E5E,0x01E5E,UCASE_LU}, {0x01E5F,0x01E5F,UCASE_LL},
    {0x01E60,0x01E60,UCASE_LU}, {0x01E61,0x01E61,UCASE_LL}, {0x01E62,0x01E62,UCASE_LU}, {0x01E63,0x01E63,UCASE_LL},
    {0x01E64,0x01E64,UCASE_LU}, {0x01E65,0x01E65,UCASE_LL}, {0x01E66,0x01E66,UCASE_LU}, {0x01E67,0x01E67,UCASE_LL},
    {0x01E68,0x01E68,UCASE_LU}, {0x01E69,0x01E69,UCASE_LL}, {0x01E6A,0x01E6A,UCASE_LU}, {0x01E6B,0x01E6B,UCASE_LL},
    {0x01E6C,0x01E6C,UCASE_LU}, {0x01E6D,0x01E6D,UCASE_LL}, {0x01E6E,0x01E6E,UCASE_LU}, {0x01E6F,0x01E6F,UCASE_LL},
    {0x01E70,0x01E70,UCASE_LU}, {0x01E71,0x01E71,UCASE_LL}, {0x01E72,0x01E72,UCASE_LU}, {0x01E73,0x01E73,UCASE_LL},
    {0x01E74,0x01E74,UCASE_LU}, {0x01E75,0x01E75,UCASE_LL}, {0x01E76,0x01E76,UCASE_LU}, {0x01E77,0x01E77,UCASE_LL},
    {0x01E78,0x01E78,UCASE_LU}, {0x01E79,0x01E79,UCASE_LL}, {0x01E7A,0x01E7A,UCASE_LU}, {0x01E7B,0x01E7B,UCASE_LL},
    {0x01E7C,0x01E7C,UCASE_LU}, {0x01E7D,0x01E7D,UCASE_LL}, {0x01E7E,0x01E7E,UCASE_LU}, {0x01E7F,0x01E7F,UCASE_LL},
    {0x01E80,0x01E80,UCASE_LU}, {0x01E81,0x01E81,UCASE_LL}, {0x01E82,0x01E82,UCASE_LU}, {0x01E83,0x01E83,UCASE_LL},
    {0x01E84,0x01E84,UCASE_LU}, {0x01E85,0x01E85,UCASE_LL}, {0x01E86,0x01E86,UCASE_LU}, {0x01E87,0x01E87,UCASE_LL},
    {0x01E88,0x01E88,UCASE_LU}, {0x01E89,0x01E89,UCASE_LL}, {0x01E8A,0x01E8A,UCASE_LU}, {0x01E8B,0x01E8B,UCASE_LL},
    {0x01E8C,0x01E8C,UCASE_LU}, {0x01E8D,0x01E8D,UCASE_LL}, {0x01E8E,0x01E8E,UCASE_LU}, {0x01E8F,0x01E8F,UCASE_LL},
    {0x01E90,0x01E90,UCASE_LU}, {0x01E91,0x01E91,UCASE_LL}, {0x01E92,0x01E92,UCASE_LU}, {0x01E93,0x01E93,UCASE_LL},
    {0x01E94,0x01E94,UCASE_LU}, {0x01E95,0x01E9D,UCASE_LL}, {0x01E9E,0x01E9E,UCASE_LU}, {0x01E9F,0x01E9F,UCASE_LL},
    {0x01EA0,0x01EA0,UCASE_LU}, {0x01EA1,0x01EA1,UCASE_LL}, {0x01EA2,0x01EA2,UCASE_LU}, {0x01EA3,0x01EA3,UCASE_LL},
    {0x01EA4,0x01EA4,UCASE_LU}, {0x01EA5,0x01EA5,UCASE_LL}, {0x01EA6,0x01EA6,UCASE_LU}, {0x01EA7,0x01EA7,UCASE_LL},
    {0x01EA8,0x01EA8,UCASE_LU}, {0x01EA9,0x01EA9,UCASE_LL}, {0x01EAA,0x01EAA,UCASE_LU}, {0x01EAB,0x01EAB,UCASE_LL},
    {0x01EAC,0x01EAC,UCASE_LU}, {0x01EAD,0x01EAD,UCASE_LL}, {0x01EAE,0x01EAE,UCASE_LU}, {0x01EAF,0x01EAF,UCASE_LL},
    {0x01EB0,0x01EB0,UCASE_LU}, {0x01EB1,0x01EB1,UCASE_LL}, {0x01EB2,0x01EB2,UCASE_LU}, {0x01EB3,0x01EB3,UCASE_LL},
    {0x01EB4,0x01EB4,UCASE_LU}, {0x01EB5,0x01EB5,UCASE_LL}, {0x01EB6,0x01EB6,UCASE_LU}, {0x01EB7,0x01EB7,UCASE_LL},
    {0x01EB8,0x01EB8,UCASE_LU}, {0x01EB9,0x01EB9,UCASE_LL}, {0x01EBA,0x01EBA,UCASE_LU}, {0x01EBB,0x01EBB,UCASE_LL},
    {0x01EBC,0x01EBC,UCASE_LU}, {0x01EBD,0x01EBD,UCASE_LL}, {0x01EBE,0x01EBE,UCASE_LU}, {0x01EBF,0x01EBF,UCASE_LL},
    {0x01EC0,0x01EC0,UCASE_LU}, {0x01EC1,0x01EC1,UCASE_LL}, {0x01EC2,0x01EC2,UCASE_LU}, {0x01EC3,0x01EC3,UCASE_LL},
    {0x01EC4,0x01EC4,UCASE_LU}, {0x01EC5,0x01EC5,UCASE_LL}, {0x01EC6,0x01EC6,UCASE_LU}, {0x01EC7,0x01EC7,UCASE_LL},
    {0x01EC8,0x01EC8,UCASE_LU}, {0x01EC9,0x01EC9,UCASE_LL}, {0x01ECA,0x01ECA,UCASE_LU}, {0x01ECB,0x01ECB,UCASE_LL},
    {0x01ECC,0x01ECC,UCASE_LU}, {0x01ECD,0x01ECD,UCASE_LL}, {0x01ECE,0x01ECE,UCASE_LU}, {0x01ECF,0x01ECF,UCASE_LL},
    {0x01ED0,0x01ED0,UCASE_LU}, {0x01ED1,0x01ED1,UCASE_LL}, {0x01ED2,0x01ED2,UCASE_LU}, {0x01ED3,0x01ED3,UCASE_LL},
    {0x01ED4,0x01ED4,UCASE_LU}, {0x01ED5,0x01ED5,UCASE_LL}, {0x01ED6,0x01ED6,UCASE_LU}, {0x01ED7,0x01ED7,UCASE_LL},
    {0x01ED8,0x01ED8,UCASE_LU}, {0x01ED9,0x01ED9,UCASE_LL}, {0x01EDA,0x01EDA,UCASE_LU}, {0x01EDB,0x01EDB,UCASE_LL},
    {0x01EDC,0x01EDC,UCASE_LU}, {0x01EDD,0x01EDD,UCASE_LL}, {0x01EDE,0x01EDE,UCASE_LU}, {0x01EDF,0x01EDF,UCASE_LL},
    {0x01EE0,0x01EE0,UCASE_LU}, {0x01EE1,0x01EE1,UCASE_LL}, {0x01EE2,0x01EE2,UCASE_LU}, {0x01EE3,0x01EE3,UCASE_LL},
    {0x01EE4,0x01EE4,UCASE_LU}, {0x01EE5,0x01EE5,UCASE_LL}, {0x01EE6,0x01EE6,UCASE_LU}, {0x01EE7,0x01EE7,UCASE_LL},
    {0x01EE8,0x01EE8,UCASE_LU}, {0x01EE9,0x01EE9,UCASE_LL}, {0x01EEA,0x01EEA,UCASE_LU}, {0x01EEB,0x01EEB,UCASE_LL},
    {0x01EEC,0x01EEC,UCASE_LU}, {0x01EED,0x01EED,UCASE_LL}, {0x01EEE,0x01EEE,UCASE_LU}, {0x01EEF,0x01EEF,UCASE_LL},
    {0x01EF0,0x01EF0,UCASE_LU}, {0x01EF1,0x01EF1,UCASE_LL}, {0x01EF2,0x01EF2,UCASE_LU}, {0x01EF3,0x01EF3,UCASE_LL},
    {0x01EF4,0x01EF4,UCASE_LU}, {0x01EF5,0x01EF5,UCASE_LL}, {0x01EF6,0x01EF6,UCASE_LU}, {0x01EF7,0x01EF7,UCASE_LL},
    {0x01EF8,0x01EF8,UCASE_LU}, {0x01EF9,0x01EF9,UCASE_LL}, {0x01EFA,0x01EFA,UCASE_LU}, {0x01EFB,0x01EFB,UCASE_LL},
    {0x01EFC,0x01EFC,UCASE_LU}, {0x01EFD,0x01EFD,UCASE_LL}, {0x01EFE,0x01EFE,UCASE_LU}, {0x01EFF,0x01F07,UCASE_LL},
    {0x01F08,0x01F0F,UCASE_LU}, {0x01F10,0x01F15,UCASE_LL}, {0x01F18,0x01F1D,UCASE_LU}, {0x01F20,0x01F27,UCASE_LL},
    {0x01F28,0x01F2F,UCASE_LU}, {0x01F30,0x01F37,UCASE_LL}, {0x01F38,0x01F3F,UCASE_LU}, {0x01F40,0x01F45,UCASE_LL},
    {0x01F48,0x01F4D,UCASE_LU}, {0x01F50,0x01F57,UCASE_LL}, {0x01F59,0x01F59,UCASE_LU}, {0x01F5B,0x01F5B,UCASE_LU},
    {0x01F5D,0x01F5D,UCASE_LU}, {0x01F5F,0x01F5F,UCASE_LU}, {0x01F60,0x01F67,UCASE_LL}, {0x01F68,0x01F6F,UCASE_LU},
    {0x01F70,0x01F7D,UCASE_LL}, {0x01F80,0x01F87,UCASE_LL}, {0x01F88,0x01F8F,UCASE_LT}, {0x01F90,0x01F97,UCASE_LL},
    {0x01F98,0x01F9F,UCASE_LT}, {0x01FA0,0x01FA7,UCASE_LL}, {0x01FA8,0x01FAF,UCASE_LT}, {0x01FB0,0x01FB4,UCASE_LL},
    {0x01FB6,0x01FB7,UCASE_LL}, {0x01FB8,0x01FBB,UCASE_LU}, {0x01FBC,0x01FBC,UCASE_LT}, {0x01FBE,0x01FBE,UCASE_LL},
    {0x01FC2,0x01FC4,UCASE_LL}, {0x01FC6,0x01FC7,UCASE_LL}, {0x01FC8,0x01FCB,UCASE_LU}, {0x01FCC,0x01FCC,UCASE_LT},
    {0x01FD0,0x01FD3,UCASE_LL}, {0x01FD6,0x01FD7,UCASE_LL}, {0x01FD8,0x01FDB,UCASE_LU}, {0x01FE0,0x01FE7,UCASE_LL},
    {0x01FE8,0x01FEC,UCASE_LU}, {0x01FF2,0x01FF4,UCASE_LL}, {0x01FF6,0x01FF7,UCASE_LL}, {0x01FF8,0x01FFB,UCASE_LU},
    {0x01FFC,0x01FFC,UCASE_LT}, {0x02071,0x02071,UCASE_LM}, {0x0207F,0x0207F,UCASE_LM}, {0x02090,0x0209C,UCASE_LM},
    {0x02102,0x02102,UCASE_LU}, {0x02107,0x02107,UCASE_LU}, {0x0210A,0x0210A,UCASE_LL}, {0x0210B,0x0210D,UCASE_LU},
    {0x0210E,0x0210F,UCASE_LL}, {0x02110,0x02112,UCASE_LU}, {0x02113,0x02113,UCASE_LL}, {0x02115,0x02115,UCASE_LU},
    {0x02119,0x0211D,UCASE_LU}, {0x02124,0x02124,UCASE_LU}, {0x02126,0x02126,UCASE_LU}, {0x02128,0x02128,UCASE_LU},
    {0x0212A,0x0212D,UCASE_LU}, {0x0212F,0x0212F,UCASE_LL}, {0x02130,0x02133,UCASE_LU}, {0x02134,0x02134,UCASE_LL},
    {0x02139,0x02139,UCASE_LL}, {0x0213C,0x0213D,UCASE_LL}, {0x0213E,0x0213F,UCASE_LU}, {0x02145,0x02145,UCASE_LU},
    {0x02146,0x02149,UCASE_LL}, {0x0214E,0x0214E,UCASE_LL}, {0x02183,0x02183,UCASE_LU}, {0x02184,0x02184,UCASE_LL},
    {0x02C00,0x02C2F,UCASE_LU}, {0x02C30,0x02C5F,UCASE_LL}, {0x02C60,0x02C60,UCASE_LU}, {0x02C61,0x02C61,UCASE_LL},
    {0x02C62,0x02C64,UCASE_LU}, {0x02C65,0x02C66,UCASE_LL}, {0x02C67,0x02C67,UCASE_LU}, {0x02C68,0x02C68,UCASE_LL},
    {0x02C69,0x02C69,UCASE_LU}, {0x02C6A,0x02C6A,UCASE_LL}, {0x02C6B,0x02C6B,UCASE_LU}, {0x02C6C,0x02C6C,UCASE_LL},
    {0x02C6D,0x02C70,UCASE_LU}, {0x02C71,0x02C71,UCASE_LL}, {0x02C72,0x02C72,UCASE_LU}, {0x02C73,0x02C74,UCASE_LL},
    {0x02C75,0x02C75,UCASE_LU}, {0x02C76,0x02C7B,UCASE_LL}, {0x02C7C,0x02C7D,UCASE_LM}, {0x02C7E,0x02C80,UCASE_LU},
    {0x02C81,0x02C81,UCASE_LL}, {0x02C82,0x02C82,UCASE_LU}, {0x02C83,0x02C83,UCASE_LL}, {0x02C84,0x02C84,UCASE_LU},
    {0x02C85,0x02C85,UCASE_LL}, {0x02C86,0x02C86,UCASE_LU}, {0x02C87,0x02C87,UCASE_LL}, {0x02C88,0x02C88,UCASE_LU},
    {0x02C89,0x02C89,UCASE_LL}, {0x02C8A,0x02C8A,UCASE_LU}, {0x02C8B,0x02C8B,UCASE_LL}, {0x02C8C,0x02C8C,UCASE_LU},
    {0x02C8D,0x02C8D,UCASE_LL}, {0x02C8E,0x02C8E,UCASE_LU}, {0x02C8F,0x02C8F,UCASE_LL}, {0x02C90,0x02C90,UCASE_LU},
    {0x02C91,0x02C91,UCASE_LL}, {0x02C92,0x02C92,UCASE_LU}, {0x02C93,0x02C93,UCASE_LL}, {0x02C94,0x02C94,UCASE_LU},
    {0x02C95,0x02C95,UCASE_LL}, {0x02C96,0x02C96,UCASE_LU}, {0x02C97,0x02C97,UCASE_LL}, {0x02C98,0x02C98,UCASE_LU},
    {0x02C99,0x02C99,UCASE_LL}, {0x02C9A,0x02C9A,UCASE_LU}, {0x02C9B,0x02C9B,UCASE_LL}, {0x02C9C,0x02C9C,UCASE_LU},
    {0x02C9D,0x02C9D,UCASE_LL}, {0x02C9E,0x02C9E,UCASE_LU}, {0x02C9F,0x02C9F,UCASE_LL}, {0x02CA0,0x02CA0,UCASE_LU},
    {0x02CA1,0x02CA1,UCASE_LL}, {0x02CA2,0x02CA2,UCASE_LU}, {0x02CA3,0x02CA3,UCASE_LL}, {0x02CA4,0x02CA4,UCASE_LU},
    {0x02CA5,0x02CA5,UCASE_LL}, {0x02CA6,0x02CA6,UCASE_LU}, {0x02CA7,0x02CA7,UCASE_LL}, {0x02CA8,0x02CA8,UCASE_LU},
    {0x02CA9,0x02CA9,UCASE_LL}, {0x02CAA,0x02CAA,UCASE_LU}, {0x02CAB,0x02CAB,UCASE_LL}, {0x02CAC,0x02CAC,UCASE_LU},
    {0x02CAD,0x02CAD,UCASE_LL}, {0x02CAE,0x02CAE,UCASE_LU}, {0x02CAF,0x02CAF,UCASE_LL}, {0x02CB0,0x02CB0,UCASE_LU},
    {0x02CB1,0x02CB1,UCASE_LL}, {0x02CB2,0x02CB2,UCASE_LU}, {0x02CB3,0x02CB3,UCASE_LL}, {0x02CB4,0x02CB4,UCASE_LU},
    {0x02CB5,0x02CB5,UCASE_LL}, {0x02CB6,0x02CB6,UCASE_LU}, {0x02CB7,0x02CB7,UCASE_LL}, {0x02CB8,0x02CB8,UCASE_LU},
    {0x02CB9,0x02CB9,UCASE_LL}, {0x02CBA,0x02CBA,UCASE_LU}, {0x02CBB,0x02CBB,UCASE_LL}, {0x02CBC,0x02CBC,UCASE_LU},
    {0x02CBD,0x02CBD,UCASE_LL}, {0x02CBE,0x02CBE,UCASE_LU}, {0x02CBF,0x02CBF,UCASE_LL}, {0x02CC0,0x02CC0,UCASE_LU},
    {0x02CC1,0x02CC1,UCASE_LL}, {0x02CC2,0x02CC2,UCASE_LU}, {0x02CC3,0x02CC3,UCASE_LL}, {0x02CC4,0x02CC4,UCASE_LU},
    {0x02CC5,0x02CC5,UCASE_LL}, {0x02CC6,0x02CC6,UCASE_LU}, {0x02CC7,0x02CC7,UCASE_LL}, {0x02CC8,0x02CC8,UCASE_LU},
    {0x02CC9,0x02CC9,UCASE_LL}, {0x02CCA,0x02CCA,UCASE_LU}, {0x02CCB,0x02CCB,UCASE_LL}, {0x02CCC,0x02CCC,UCASE_LU},
    {0x02CCD,0x02CCD,UCASE_LL}, {0x02CCE,0x02CCE,UCASE_LU}, {0x02CCF,0x02CCF,UCASE_LL}, {0x02CD0,0x02CD0,UCASE_LU},
    {0x02CD1,0x02CD1,UCASE_LL}, {0x02CD2,0x02CD2,UCASE_LU}, {0x02CD3,0x02CD3,UCASE_LL}, {0x02CD4,0x02CD4,UCASE_LU},
    {0x02CD5,0x02CD5,UCASE_LL}, {0x02CD6,0x02CD6,UCASE_LU}, {0x02CD7,0x02CD7,UCASE_LL}, {0x02CD8,0x02CD8,UCASE_LU},
    {0x02CD9,0x02CD9,UCASE_LL}, {0x02CDA,0x02CDA,UCASE_LU}, {0x02CDB,0x02CDB,UCASE_LL}, {0x02CDC,0x02CDC,UCASE_LU},
    {0x02CDD,0x02CDD,UCASE_LL}, {0x02CDE,0x02CDE,UCASE_LU}, {0x02CDF,0x02CDF,UCASE_LL}, {0x02CE0,0x02CE0,UCASE_LU},
    {0x02CE1,0x02CE1,UCASE_LL}, {0x02CE2,0x02CE2,UCASE_LU}, {0x02CE3,0x02CE4,UCASE_LL}, {0x02CEB,0x02CEB,UCASE_LU},
    {0x02CEC,0x02CEC,UCASE_LL}, {0x02CED,0x02CED,UCASE_LU}, {0x02CEE,0x02CEE,UCASE_LL}, {0x02CF2,0x02CF2,UCASE_LU},
    {0x02CF3,0x02CF3,UCASE_LL}, {0x02D00,0x02D25,UCASE_LL}, {0x02D27,0x02D27,UCASE_LL}, {0x02D2D,0x02D2D,UCASE_LL},
    {0x02D6F,0x02D6F,UCASE_LM}, {0x02E2F,0x02E2F,UCASE_LM}, {0x03005,0x03005,UCASE_LM}, {0x03031,0x03035,UCASE_LM},
    {0x0303B,0x0303B,UCASE_LM}, {0x0309D,0x0309E,UCASE_LM}, {0x030FC,0x030FE,UCASE_LM}, {0x0A015,0x0A015,UCASE_LM},
    {0x0A4F8,0x0A4FD,UCASE_LM}, {0x0A60C,0x0A60C,UCASE_LM}, {0x0A640,0x0A640,UCASE_LU}, {0x0A641,0x0A641,UCASE_LL},
    {0x0A642,0x0A642,UCASE_LU}, {0x0A643,0x0A643,UCASE_LL}, {0x0A644,0x0A644,UCASE_LU}, {0x0A645,0x0A645,UCASE_LL},
    {0x0A646,0x0A646,UCASE_LU}, {0x0A647,0x0A647,UCASE_LL}, {0x0A648,0x0A648,UCASE_LU}, {0x0A649,0x0A649,UCASE_LL},
    {0x0A64A,0x0A64A,UCASE_LU}, {0x0A64B,0x0A64B,UCASE_LL}, {0x0A64C,0x0A64C,UCASE_LU}, {0x0A64D,0x0A64D,UCASE_LL},
    {0x0A64E,0x0A64E,UCASE_LU}, {0x0A64F,0x0A64F,UCASE_LL}, {0x0A650,0x0A650,UCASE_LU}, {0x0A651,0x0A651,UCASE_LL},
    {0x0A652,0x0A652,UCASE_LU}, {0x0A653,0x0A653,UCASE_LL}, {0x0A654,0x0A654,UCASE_LU}, {0x0A655,0x0A655,UCASE_LL},
    {0x0A656,0x0A656,UCASE_LU}, {0x0A657,0x0A657,UCASE_LL}, {0x0A658,0x0A658,UCASE_LU}, {0x0A659,0x0A659,UCASE_LL},
    {0x0A65A,0x0A65A,UCASE_LU}, {0x0A65B,0x0A65B,UCASE_LL}, {0x0A65C,0x0A65C,UCASE_LU}, {0x0A65D,0x0A65D,UCASE_LL},
    {0x0A65E,0x0A65E,UCASE_LU}, {0x0A65F,0x0A65F,UCASE_LL}, {0x0A660,0x0A660,UCASE_LU}, {0x0A661,0x0A661,UCASE_LL},
    {0x0A662,0x0A662,UCASE_LU}, {0x0A663,0x0A663,UCASE_LL}, {0x0A664,0x0A664,UCASE_LU}, {0x0A665,0x0A665,UCASE_LL},
    {0x0A666,0x0A666,UCASE_LU}, {0x0A667,0x0A667,UCASE_LL}, {0x0A668,0x0A668,UCASE_LU}, {0x0A669,0x0A669,UCASE_LL},
    {0x0A66A,0x0A66A,UCASE_LU}, {0x0A66B,0x0A66B,UCASE_LL}, {0x0A66C,0x0A66C,UCASE_LU}, {0x0A66D,0x0A66D,UCASE_LL},
    {0x0A67F,0x0A67F,UCASE_LM}, {0x0A680,0x0A680,UCASE_LU}, {0x0A681,0x0A681,UCASE_LL}, {0x0A682,0x0A682,UCASE_LU},
    {0x0A683,0x0A683,UCASE_LL}, {0x0A684,0x0A684,UCASE_LU}, {0x0A685,0x0A685,UCASE_LL}, {0x0A686,0x0A686,UCASE_LU},
    {0x0A687,0x0A687,UCASE_LL}, {0x0A688,0x0A688,UCASE_LU}, {0x0A689,0x0A689,UCASE_LL}, {0x0A68A,0x0A68A,UCASE_LU},
    {0x0A68B,0x0A68B,UCASE_LL}, {0x0A68C,0x0A68C,UCASE_LU}, {0x0A68D,0x0A68D,UCASE_LL}, {0x0A68E,0x0A68E,UCASE_LU},
    {0x0A68F,0x0A68F,UCASE_LL}, {0x0A690,0x0A690,UCASE_LU}, {0x0A691,0x0A691,UCASE_LL}, {0x0A692,0x0A692,UCASE_LU},
    {0x0A693,0x0A693,UCASE_LL}, {0x0A694,0x0A694,UCASE_LU}, {0x0A695,0x0A695,UCASE_LL}, {0x0A696,0x0A696,UCASE_LU},
    {0x0A697,0x0A697,UCASE_LL}, {0x0A698,0x0A698,UCASE_LU}, {0x0A699,0x0A699,UCASE_LL}, {0x0A69A,0x0A69A,UCASE_LU},
    {0x0A69B,0x0A69B,UCASE_LL}, {0x0A69C,0x0A69D,UCASE_LM}, {0x0A717,0x0A71F,UCASE_LM}, {0x0A722,0x0A722,UCASE_LU},
    {0x0A723,0x0A723,UCASE_LL}, {0x0A724,0x0A724,UCASE_LU}, {0x0A725,0x0A725,UCASE_LL}, {0x0A726,0x0A726,UCASE_LU},
    {0x0A727,0x0A727,UCASE_LL}, {0x0A728,0x0A728,UCASE_LU}, {0x0A729,0x0A729,UCASE_LL}, {0x0A72A,0x0A72A,UCASE_LU},
    {0x0A72B,0x0A72B,UCASE_LL}, {0x0A72C,0x0A72C,UCASE_LU}, {0x0A72D,0x0A72D,UCASE_LL}, {0x0A72E,0x0A72E,UCASE_LU},
    {0x0A72F,0x0A731,UCASE_LL}, {0x0A732,0x0A732,UCASE_LU}, {0x0A733,0x0A733,UCASE_LL}, {0x0A734,0x0A734,UCASE_LU},
    {0x0A735,0x0A735,UCASE_LL}, {0x0A736,0x0A736,UCASE_LU}, {0x0A737,0x0A737,UCASE_LL}, {0x0A738,0x0A738,UCASE_LU},
    {0x0A739,0x0A739,UCASE_LL}, {0x0A73A,0x0A73A,UCASE_LU}, {0x0A73B,0x0A73B,UCASE_LL}, {0x0A73C,0x0A73C,UCASE_LU},
    {0x0A73D,0x0A73D,UCASE_LL}, {0x0A73E,0x0A73E,UCASE_LU}, {0x0A73F,0x0A73F,UCASE_LL}, {0x0A740,0x0A740,UCASE_LU},
    {0x0A741,0x0A741,UCASE_LL}, {0x0A742,0x0A742,UCASE_LU}, {0x0A743,0x0A743,UCASE_LL}, {0x0A744,0x0A744,UCASE_LU},
    {0x0A745,0x0A745,UCASE_LL}, {0x0A746,0x0A746,UCASE_LU}, {0x0A747,0x0A747,UCASE_LL}, {0x0A748,0x0A748,UCASE_LU},
    {0x0A749,0x0A749,UCASE_LL}, {0x0A74A,0x0A74A,UCASE_LU}, {0x0A74B,0x0A74B,UCASE_LL}, {0x0A74C,0x0A74C,UCASE_LU},
    {0x0A74D,0x0A74D,UCASE_LL}, {0x0A74E,0x0A74E,UCASE_LU}, {0x0A74F,0x0A74F,UCASE_LL}, {0x0A750,0x0A750,UCASE_LU},
    {0x0A751,0x0A751,UCASE_LL}, {0x0A752,0x0A752,UCASE_LU}, {0x0A753,0x0A753,UCASE_LL}, {0x0A754,0x0A754,UCASE_LU},
    {0x0A755,0x0A755,UCASE_LL}, {0x0A756,0x0A756,UCASE_LU}, {0x0A757,0x0A757,UCASE_LL}, {0x0A758,0x0A758,UCASE_LU},
    {0x0A759,0x0A759,UCASE_LL}, {0x0A75A,0x0A75A,UCASE_LU}, {0x0A75B,0x0A75B,UCASE_LL}, {0x0A75C,0x0A75C,UCASE_LU},
    {0x0A75D,0x0A75D,UCASE_LL}, {0x0A75E,0x0A75E,UCASE_LU}, {0x0A75F,0x0A75F,UCASE_LL}, {0x0A760,0x0A760,UCASE_LU},
    {0x0A761,0x0A761,UCASE_LL}, {0x0A762,0x0A762,UCASE_LU}, {0x0A763,0x0A763,UCASE_LL}, {0x0A764,0x0A764,UCASE_LU},
    {0x0A765,0x0A765,UCASE_LL}, {0x0A766,0x0A766,UCASE_LU}, {0x0A767,0x0A767,UCASE_LL}, {0x0A768,0x0A768,UCASE_LU},
    {0x0A769,0x0A769,UCASE_LL}, {0x0A76A,0x0A76A,UCASE_LU}, {0x0A76B,0x0A76B,UCASE_LL}, {0x0A76C,0x0A76C,UCASE_LU},
    {0x0A76D,0x0A76D,UCASE_LL}, {0x0A76E,0x0A76E,UCASE_LU}, {0x0A76F,0x0A76F,UCASE_LL}, {0x0A770,0x0A770,UCASE_LM},
    {0x0A771,0x0A778,UCASE_LL}, {0x0A779,0x0A779,UCASE_LU}, {0x0A77A,0x0A77A,UCASE_LL}, {0x0A77B,0x0A77B,UCASE_LU},
    {0x0A77C,0x0A77C,UCASE_LL}, {0x0A77D,0x0A77E,UCASE_LU}, {0x0A77F,0x0A77F,UCASE_LL}, {0x0A780,0x0A780,UCASE_LU},
    {0x0A781,0x0A781,UCASE_LL}, {0x0A782,0x0A782,UCASE_LU}, {0x0A783,0x0A783,UCASE_LL}, {0x0A784,0x0A784,UCASE_LU},
    {0x0A785,0x0A785,UCASE_LL}, {0x0A786,0x0A786,UCASE_LU}, {0x0A787,0x0A787,UCASE_LL}, {0x0A788,0x0A788,UCASE_LM},
    {0x0A78B,0x0A78B,UCASE_LU}, {0x0A78C,0x0A78C,UCASE_LL}, {0x0A78D,0x0A78D,UCASE_LU}, {0x0A78E,0x0A78E,UCASE_LL},
    {0x0A790,0x0A790,UCASE_LU}, {0x0A791,0x0A791,UCASE_LL}, {0x0A792,0x0A792,UCASE_LU}, {0x0A793,0x0A795,UCASE_LL},
    {0x0A796,0x0A796,UCASE_LU}, {0x0A797,0x0A797,UCASE_LL}, {0x0A798,0x0A798,UCASE_LU}, {0x0A799,0x0A799,UCASE_LL},
    {0x0A79A,0x0A79A,UCASE_LU}, {0x0A79B,0x0A79B,UCASE_LL}, {0x0A79C,0x0A79C,UCASE_LU}, {0x0A79D,0x0A79D,UCASE_LL},
    {0x0A79E,0x0A79E,UCASE_LU}, {0x0A79F,0x0A79F,UCASE_LL}, {0x0A7A0,0x0A7A0,UCASE_LU}, {0x0A7A1,0x0A7A1,UCASE_LL},
    {0x0A7A2,0x0A7A2,UCASE_LU}, {0x0A7A3,0x0A7A3,UCASE_LL}, {0x0A7A4,0x0A7A4,UCASE_LU}, {0x0A7A5,0x0A7A5,UCASE_LL},
    {0x0A7A6,0x0A7A6,UCASE_LU}, {0x0A7A7,0x0A7A7,UCASE_LL}, {0x0A7A8,0x0A7A8,UCASE_LU}, {0x0A7A9,0x0A7A9,UCASE_LL},
    {0x0A7AA,0x0A7AE,UCASE_LU}, {0x0A7AF,0x0A7AF,UCASE_LL}, {0x0A7B0,0x0A7B4,UCASE_LU}, {0x0A7B5,0x0A7B5,UCASE_LL},
    {0x0A7B6,0x0A7B6,UCASE_LU}, {0x0A7B7,0x0A7B7,UCASE_LL}, {0x0A7B8,0x0A7B8,UCASE_LU}, {0x0A7B9,0x0A7B9,UCASE_LL},
    {0x0A7BA,0x0A7BA,UCASE_LU}, {0x0A7BB,0x0A7BB,UCASE_LL}, {0x0A7BC,0x0A7BC,UCASE_LU}, {0x0A7BD,0x0A7BD,UCASE_LL},
    {0x0A7BE,0x0A7BE,UCASE_LU}, {0x0A7BF,0x0A7BF,UCASE_LL}, {0x0A7C0,0x0A7C0,UCASE_LU}, {0x0A7C1,0x0A7C1,UCASE_LL},
    {0x0A7C2,0x0A7C2,UCASE_LU}, {0x0A7C3,0x0A7C3,UCASE_LL}, {0x0A7C4,0x0A7C7,UCASE_LU}, {0x0A7C8,0x0A7C8,UCASE_LL},
    {0x0A7C9,0x0A7C9,UCASE_LU}, {0x0A7CA,0x0A7CA,UCASE_LL}, {0x0A7CB,0x0A7CC,UCASE_LU}, {0x0A7CD,0x0A7CD,UCASE_LL},
    {0x0A7D0,0x0A7D0,UCASE_LU}, {0x0A7D1,0x0A7D1,UCASE_LL}, {0x0A7D3,0x0A7D3,UCASE_LL}, {0x0A7D5,0x0A7D5,UCASE_LL},
    {0x0A7D6,0x0A7D6,UCASE_LU}, {0x0A7D7,0x0A7D7,UCASE_LL}, {0x0A7D8,0x0A7D8,UCASE_LU}, {0x0A7D9,0x0A7D9,UCASE_LL},
    {0x0A7DA,0x0A7DA,UCASE_LU}, {0x0A7DB,0x0A7DB,UCASE_LL}, {0x0A7DC,0x0A7DC,UCASE_LU}, {0x0A7F2,0x0A7F4,UCASE_LM},
    {0x0A7F5,0x0A7F5,UCASE_LU}, {0x0A7F6,0x0A7F6,UCASE_LL}, {0x0A7F8,0x0A7F9,UCASE_LM}, {0x0A7FA,0x0A7FA,UCASE_LL},
    {0x0A9CF,0x0A9CF,UCASE_LM}, {0x0A9E6,0x0A9E6,UCASE_LM}, {0x0AA70,0x0AA70,UCASE_LM}, {0x0AADD,0x0AADD,UCASE_LM},
    {0x0AAF3,0x0AAF4,UCASE_LM}, {0x0AB30,0x0AB5A,UCASE_LL}, {0x0AB5C,0x0AB5F,UCASE_LM}, {0x0AB60,0x0AB68,UCASE_LL},
    {0x0AB69,0x0AB69,UCASE_LM}, {0x0AB70,0x0ABBF,UCASE_LL}, {0x0FB00,0x0FB06,UCASE_LL}, {0x0FB13,0x0FB17,UCASE_LL},
    {0x0FF21,0x0FF3A,UCASE_LU}, {0x0FF41,0x0FF5A,UCASE_LL}, {0x0FF70,0x0FF70,UCASE_LM}, {0x0FF9E,0x0FF9F,UCASE_LM},
    {0x10400,0x10427,UCASE_LU}, {0x10428,0x1044F,UCASE_LL}, {0x104B0,0x104D3,UCASE_LU}, {0x104D8,0x104FB,UCASE_LL},
    {0x10570,0x1057A,UCASE_LU}, {0x1057C,0x1058A,UCASE_LU}, {0x1058C,0x10592,UCASE_LU}, {0x10594,0x10595,UCASE_LU},
    {0x10597,0x105A1,UCASE_LL}, {0x105A3,0x105B1,UCASE_LL}, {0x105B3,0x105B9,UCASE_LL}, {0x105BB,0x105BC,UCASE_LL},
    {0x10780,0x10785,UCASE_LM}, {0x10787,0x107B0,UCASE_LM}, {0x107B2,0x107BA,UCASE_LM}, {0x10C80,0x10CB2,UCASE_LU},
    {0x10CC0,0x10CF2,UCASE_LL}, {0x10D4E,0x10D4E,UCASE_LM}, {0x10D50,0x10D65,UCASE_LU}, {0x10D6F,0x10D6F,UCASE_LM},
    {0x10D70,0x10D85,UCASE_LL}, {0x118A0,0x118BF,UCASE_LU}, {0x118C0,0x118DF,UCASE_LL}, {0x16B40,0x16B43,UCASE_LM},
    {0x16D40,0x16D42,UCASE_LM}, {0x16D6B,0x16D6C,UCASE_LM}, {0x16E40,0x16E5F,UCASE_LU}, {0x16E60,0x16E7F,UCASE_LL},
    {0x16F93,0x16F9F,UCASE_LM}, {0x16FE0,0x16FE1,UCASE_LM}, {0x16FE3,0x16FE3,UCASE_LM}, {0x1AFF0,0x1AFF3,UCASE_LM},
    {0x1AFF5,0x1AFFB,UCASE_LM}, {0x1AFFD,0x1AFFE,UCASE_LM}, {0x1D400,0x1D419,UCASE_LU}, {0x1D41A,0x1D433,UCASE_LL},
    {0x1D434,0x1D44D,UCASE_LU}, {0x1D44E,0x1D454,UCASE_LL}, {0x1D456,0x1D467,UCASE_LL}, {0x1D468,0x1D481,UCASE_LU},
    {0x1D482,0x1D49B,UCASE_LL}, {0x1D49C,0x1D49C,UCASE_LU}, {0x1D49E,0x1D49F,UCASE_LU}, {0x1D4A2,0x1D4A2,UCASE_LU},
    {0x1D4A5,0x1D4A6,UCASE_LU}, {0x1D4A9,0x1D4AC,UCASE_LU}, {0x1D4AE,0x1D4B5,UCASE_LU}, {0x1D4B6,0x1D4B9,UCASE_LL},
    {0x1D4BB,0x1D4BB,UCASE_LL}, {0x1D4BD,0x1D4C3,UCASE_LL}, {0x1D4C5,0x1D4CF,UCASE_LL}, {0x1D4D0,0x1D4E9,UCASE_LU},
    {0x1D4EA,0x1D503,UCASE_LL}, {0x1D504,0x1D505,UCASE_LU}, {0x1D507,0x1D50A,UCASE_LU}, {0x1D50D,0x1D514,UCASE_LU},
    {0x1D516,0x1D51C,UCASE_LU}, {0x1D51E,0x1D537,UCASE_LL}, {0x1D538,0x1D539,UCASE_LU}, {0x1D53B,0x1D53E,UCASE_LU},
    {0x1D540,0x1D544,UCASE_LU}, {0x1D546,0x1D546,UCASE_LU}, {0x1D54A,0x1D550,UCASE_LU}, {0x1D552,0x1D56B,UCASE_LL},
    {0x1D56C,0x1D585,UCASE_LU}, {0x1D586,0x1D59F,UCASE_LL}, {0x1D5A0,0x1D5B9,UCASE_LU}, {0x1D5BA,0x1D5D3,UCASE_LL},
    {0x1D5D4,0x1D5ED,UCASE_LU}, {0x1D5EE,0x1D607,UCASE_LL}, {0x1D608,0x1D621,UCASE_LU}, {0x1D622,0x1D63B,UCASE_LL},
    {0x1D63C,0x1D655,UCASE_LU}, {0x1D656,0x1D66F,UCASE_LL}, {0x1D670,0x1D689,UCASE_LU}, {0x1D68A,0x1D6A5,UCASE_LL},
    {0x1D6A8,0x1D6C0,UCASE_LU}, {0x1D6C2,0x1D6DA,UCASE_LL}, {0x1D6DC,0x1D6E1,UCASE_LL}, {0x1D6E2,0x1D6FA,UCASE_LU},
    {0x1D6FC,0x1D714,UCASE_LL}, {0x1D716,0x1D71B,UCASE_LL}, {0x1D71C,0x1D734,UCASE_LU}, {0x1D736,0x1D74E,UCASE_LL},
    {0x1D750,0x1D755,UCASE_LL}, {0x1D756,0x1D76E,UCASE_LU}, {0x1D770,0x1D788,UCASE_LL}, {0x1D78A,0x1D78F,UCASE_LL},
    {0x1D790,0x1D7A8,UCASE_LU}, {0x1D7AA,0x1D7C2,UCASE_LL}, {0x1D7C4,0x1D7C9,UCASE_LL}, {0x1D7CA,0x1D7CA,UCASE_LU},
    {0x1D7CB,0x1D7CB,UCASE_LL}, {0x1DF00,0x1DF09,UCASE_LL}, {0x1DF0B,0x1DF1E,UCASE_LL}, {0x1DF25,0x1DF2A,UCASE_LL},
    {0x1E030,0x1E06D,UCASE_LM}, {0x1E137,0x1E13D,UCASE_LM}, {0x1E4EB,0x1E4EB,UCASE_LM}, {0x1E900,0x1E921,UCASE_LU},
    {0x1E922,0x1E943,UCASE_LL}, {0x1E94B,0x1E94B,UCASE_LM},
    };
    size_t lo = 0, hi = sizeof(ranges) / sizeof(ranges[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cpt < ranges[mid].first) {
            hi = mid;
        } else if (cpt > ranges[mid].last) {
            lo = mid + 1;
        } else {
            return ranges[mid].cat;
        }
    }
    return UCASE_NONE;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode) {
    if (text.empty()) {
        return {};
    }

    // Unicode categories on the collapsed path: each category becomes the collapse byte(s) of the non-ASCII codepoints in it
    // plus its ASCII members, which the collapsed text keeps as they are. Letters collapse by case (Lu 0xD6, Ll 0xD7, Lt 0xD8,
    // Lm 0xD9, Lo 0xD2), so \p{Lu} and \p{Ll} are not both plain \p{L}: with one letter byte the o200k-style pattern
    // [\p{Lu}...]*[\p{Ll}...]+ kept "aX" and "camelCase" in one piece where HF splits a|X and camel|Case.
    struct ucat_class { const char * collapsed; const char * ascii; };
    static const std::map<std::string, ucat_class> k_ucat_class = {
        { "\\p{N}",  { "\xD1", "\x30-\x39" } }, // 0-9
        { "\\p{L}",  { "\xD2\xD6\xD7\xD8\xD9", "\x41-\x5A\x61-\x7A" } }, // A-Za-z
        { "\\p{P}",  { "\xD3", "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" } }, // !-#%-*,-/:-;?-@\[-\]_\{\}
        { "\\p{M}",  { "\xD4", "" } }, // no sub-128 codepoints
        { "\\p{S}",  { "\xD5", "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C\\\x7E" } }, // $+<=>^`|~
        { "\\p{Lu}", { "\xD6", "\x41-\x5A" } }, // Uppercase letter, A-Z
        { "\\p{Ll}", { "\xD7", "\x61-\x7A" } }, // Lowercase letter, a-z
        { "\\p{Lt}", { "\xD8", "" } },         // Titlecase letter
        { "\\p{Lm}", { "\xD9", "" } },         // Modifier letter
        { "\\p{Lo}", { "\xD2", "" } },         // Other letter
    };

    static const std::map<int, int> k_ucat_cpt = {
        { unicode_cpt_flags::NUMBER,      0xD1 },
        { unicode_cpt_flags::LETTER,      0xD2 },
        { unicode_cpt_flags::PUNCTUATION, 0xD3 },
        { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
        { unicode_cpt_flags::SYMBOL,      0xD5 },
    };

    // compute collapsed codepoints only if needed by at least one regex
    bool need_collapse = false;
    for (const auto & regex_expr : regex_exprs) {
        // search for unicode categories
        for (const auto & ucat : k_ucat_class) {
            if (std::string::npos != regex_expr.find(ucat.first)) {
                need_collapse = true;
                break;
            }
        }
    }

    const auto cpts = unicode_cpts_from_utf8(text);

    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    // ref: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2081479935
    std::string text_collapsed;
    if (need_collapse) {
        // collapse all unicode categories
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            // keep single-byte codepoints as is
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);
            const auto lcase = unicode_cpt_letter_case(cpts[i]);

            if (flags.is_whitespace) {
                //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                text_collapsed[i] = (char) 0x0B;    // <vertical tab> as whitespace fallback
            } else if (lcase != UCASE_NONE) {
                static const char k_case_byte[] = { (char) 0xD2, (char) 0xD6, (char) 0xD7, (char) 0xD8, (char) 0xD9 };
                text_collapsed[i] = k_case_byte[lcase];
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = (char) 0xD0; // fallback
            }
        }
    }

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // fallback to general-purpose std::regex / std::wregex
        try {
            // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
            // with the corresponding collapsed representation
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_class) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }
            const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);

            if (use_collapsed) {
                // sanity-check that the original regex does not contain any non-ASCII characters
                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                // generate a collapsed representation of the regex
                std::string regex_expr_collapsed;

                // track if we are inside [], because nested [] are not allowed
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    // Match \p{...} Unicode properties of varying lengths
                    if (regex_expr[i + 0] == '\\' && i + 3 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{') {
                        // Find the closing brace
                        size_t closing_brace = regex_expr.find('}', i + 3);
                        if (closing_brace != std::string::npos && closing_brace <= i + 10) { // reasonable limit
                            const std::string pat = regex_expr.substr(i, closing_brace - i + 1);
                            if (k_ucat_class.find(pat) != k_ucat_class.end()) {
                                if (!inside) {
                                    regex_expr_collapsed += '[';
                                }
                                regex_expr_collapsed += k_ucat_class.at(pat).collapsed;
                                regex_expr_collapsed += k_ucat_class.at(pat).ascii;
                                if (!inside) {
                                    regex_expr_collapsed += ']';
                                }
                                i = closing_brace;
                                continue;
                            }
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                //printf("text_collapsed: %s\n", text_collapsed.c_str());
                //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // no unicode category used, we can use std::wregex directly
                std::wstring wregex_expr(cpts_regex.begin(), cpts_regex.end());

                // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                //
                // With a 16-bit wchar_t (MinGW, MSVC) the std::wstring below would TRUNCATE every astral codepoint to
                // its low 16 bits, so U+10030 became '0' and matched [0-9]: measured on the FALCON pre-type, whose
                // "[0-9][0-9][0-9]" rule split "a" U+10030 U+10031 U+10032 "b" into three pieces where HF keeps one.
                // Each astral codepoint is replaced, one for one, by the noncharacter U+FFFF: offsets stay aligned with
                // cpts (the pieces are cut from cpts, not wtext), and U+FFFF matches what an unknown non-ASCII
                // codepoint should -- negated classes and '.', never an ASCII class. Classification uses the real
                // codepoint, so the whitespace fallback still sees astral whitespace (there is none today).
                std::wstring wtext(cpts.size(), L'\0');
                for (size_t i = 0; i < cpts.size(); ++i) {
                    const uint32_t c = cpts[i];
                    if (c > 0x7F && unicode_cpt_flags_from_cpt(c).is_whitespace) {
                        wtext[i] = 0x0B;
                    } else if (sizeof(wchar_t) < 4 && c > 0xFFFF) {
                        wtext[i] = static_cast<wchar_t>(0xFFFF);
                    } else {
                        wtext[i] = static_cast<wchar_t>(c);
                    }
                }

                //printf("text: %s\n", text.c_str());
                //printf("regex_expr: %s\n", regex_expr.c_str());
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    if (byte_encode) {
        return unicode_byte_encoding_process(bpe_words);
    }

    return bpe_words;
}
