#include "speak-stream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>

namespace {

bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }

// the UTF-8 sequence at text[i] is one of the CJK full stops (U+3002 ideographic full stop, U+FF01 fullwidth !, U+FF1F fullwidth ?)
size_t cjk_stop_len(const std::string & t, size_t i) {
    if (i + 3 <= t.size()) {
        const unsigned char a = (unsigned char) t[i], b = (unsigned char) t[i + 1], c = (unsigned char) t[i + 2];
        if (a == 0xE3 && b == 0x80 && c == 0x82) { return 3; }   // U+3002
        if (a == 0xEF && b == 0xBC && (c == 0x81 || c == 0x9F)) { return 3; }   // U+FF01, U+FF1F
    }
    return 0;
}

bool is_closer(unsigned char c) { return c == '"' || c == '\'' || c == ')' || c == ']' || c == '}'; }

// the word that ends right before text[dot] (the '.'): an abbreviation or a lone initial does not end a sentence
bool dot_is_abbreviation(const std::string & t, size_t from, size_t dot) {
    size_t b = dot;
    while (b > from && (std::isalpha((unsigned char) t[b - 1]) || t[b - 1] == '.')) { --b; }
    std::string w = t.substr(b, dot - b);
    std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (w.size() == 1 && std::isalpha((unsigned char) w[0])) { return true; }          // J. Smith
    static const char * k_abbr[] = { "mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "no", "inc", "ltd", "jr", "sr", "approx", "fig", nullptr };
    for (int i = 0; k_abbr[i]; ++i) {
        if (w == k_abbr[i]) {
            return true;
        }
    }
    return false;
}

size_t trimmed_len(const std::string & t, size_t a, size_t b) {
    while (a < b && is_space((unsigned char) t[a])) { ++a; }
    while (b > a && is_space((unsigned char) t[b - 1])) { --b; }
    return b - a;
}

}  // namespace

size_t speak_sentence_end(const std::string & text, size_t from, bool final_flush, const speak_cut_params & p) {
    const size_t n = text.size();
    if (from >= n) {
        return std::string::npos;
    }
    size_t i = from;
    while (i < n) {
        const unsigned char c = (unsigned char) text[i];
        size_t end = std::string::npos;   // exclusive end of a sentence ending at this character
        if (c == '\n') {
            end = i + 1;
        } else if (c == '.' || c == '!' || c == '?' || cjk_stop_len(text, i) > 0) {
            const bool cjk = cjk_stop_len(text, i) > 0;
            size_t j = i + (cjk ? 3 : 1);
            // swallow a run of terminators and closing quotes / brackets ("?!", "...", ".\"")
            while (j < n && (text[j] == '.' || text[j] == '!' || text[j] == '?' || is_closer((unsigned char) text[j]) || cjk_stop_len(text, j) > 0)) {
                j += cjk_stop_len(text, j) > 0 ? 3 : 1;
            }
            if (j >= n) {
                // the buffer ends on the terminator: the next token may continue it
                if (final_flush) { end = n; }
                else { return std::string::npos; }
            } else if (cjk || is_space((unsigned char) text[j])) {
                const bool plain_dot = !cjk && c == '.' && j == i + 1;
                if (!(plain_dot && dot_is_abbreviation(text, from, i))) {
                    end = j;
                }
            }
            if (end == std::string::npos) {
                i = j;
                continue;
            }
        }
        if (end != std::string::npos) {
            if (trimmed_len(text, from, end) >= p.min_chars) {
                return end;
            }
            i = end;
            continue;
        }
        ++i;
        // no sentence end for a long stretch: cut at a clause boundary (, ; :) or at a word boundary before max_chars
        if (p.max_chars > 0 && i - from >= p.max_chars && is_space((unsigned char) text[i - 1])) {
            size_t cut = std::string::npos;
            for (size_t k = i; k > from + p.min_chars; --k) {
                const char ch = text[k - 1];
                if ((ch == ',' || ch == ';' || ch == ':') && k < n && is_space((unsigned char) text[k])) { cut = k; break; }
            }
            return cut != std::string::npos ? cut : i;
        }
    }
    if (final_flush) {
        return trimmed_len(text, from, n) > 0 ? n : std::string::npos;
    }
    return std::string::npos;
}

bool speak_has_speech(const std::string & chunk) {
    for (unsigned char c : chunk) {
        if (std::isalnum(c) || c >= 0x80) {
            return true;
        }
    }
    return false;
}

namespace {

const char * const k_ones[] = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
const char * const k_tens[] = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// 1..999
std::string below_thousand(int n) {
    std::string r;
    if (n >= 100) {
        r = std::string(k_ones[n / 100]) + " hundred";
        n %= 100;
        if (n > 0) { r += ' '; }
    }
    if (n >= 20) {
        r += k_tens[n / 10];
        if (n % 10 > 0) { r += std::string("-") + k_ones[n % 10]; }
    } else if (n > 0) {
        r += k_ones[n];
    }
    return r;
}

// 0..999,999,999,999
std::string integer_words(long long n) {
    if (n == 0) { return "zero"; }
    static const struct { long long v; const char * name; } k_scale[] = {
        { 1000000000LL, "billion" }, { 1000000LL, "million" }, { 1000LL, "thousand" },
    };
    std::string r;
    for (const auto & s : k_scale) {
        if (n >= s.v) {
            r += below_thousand((int) (n / s.v)) + " " + s.name;
            n %= s.v;
            if (n > 0) { r += ' '; }
        }
    }
    if (n > 0) { r += below_thousand((int) n); }
    return r;
}

std::string digit_words(const std::string & digits) {
    std::string r;
    for (char c : digits) {
        if (!r.empty()) { r += ' '; }
        r += k_ones[c - '0'];
    }
    return r;
}

// text[j] is ',' followed by exactly three digits (and no fourth)
bool is_thousands_group(const std::string & t, size_t j) {
    return j + 3 < t.size() && t[j] == ',' && is_digit(t[j + 1]) && is_digit(t[j + 2]) && is_digit(t[j + 3])
        && (j + 4 >= t.size() || !is_digit(t[j + 4]));
}

} // namespace

std::string speak_numbers_to_words(const std::string & text) {
    std::string out;
    out.reserve(text.size() * 2);
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        if (!is_digit(text[i])) { out += text[i++]; continue; }

        size_t j = i;
        while (j < n && is_digit(text[j])) { ++j; }
        std::string integer = text.substr(i, j - i);
        // "1,234,567": only a 1-3 digit lead can start a grouped number
        if (integer.size() <= 3) {
            while (is_thousands_group(text, j)) {
                integer += text.substr(j + 1, 3);
                j += 4;
            }
        }
        std::string words = (integer.size() > 1 && integer[0] == '0') || integer.size() > 12
            ? digit_words(integer)
            : integer_words(std::stoll(integer));
        // a decimal part only when a digit follows the '.': "3." at a sentence end keeps its full stop
        if (j + 1 < n && text[j] == '.' && is_digit(text[j + 1])) {
            size_t k = j + 1;
            while (k < n && is_digit(text[k])) { ++k; }
            words += " point " + digit_words(text.substr(j + 1, k - j - 1));
            j = k;
        }

        if (!out.empty() && is_alpha(out.back())) { out += ' '; }
        out += words;
        if (j < n && is_alpha(text[j])) { out += ' '; }
        i = j;
    }
    return out;
}

namespace {

// lower-case ASCII spelling of a Latin-1 Supplement / Latin Extended-A letter (U+00C0..U+017F), nullptr for anything else
const char * latin_fold(uint32_t cp) {
    static const char * const k_c0[64] = {   // U+00C0..U+00FF; x and / (U+00D7, U+00F7) are not letters
        "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i",
        "d", "n", "o", "o", "o", "o", "o", nullptr, "o", "u", "u", "u", "u", "y", "th", "ss",
        "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i",
        "d", "n", "o", "o", "o", "o", "o", nullptr, "o", "u", "u", "u", "u", "y", "th", "y",
    };
    static const struct { uint32_t last; const char * s; } k_ext_a[] = {   // U+0100..U+017F as runs, each up to `last`
        { 0x0105, "a" }, { 0x010D, "c" }, { 0x0111, "d" }, { 0x011B, "e" }, { 0x0123, "g" }, { 0x0127, "h" },
        { 0x0131, "i" }, { 0x0133, "ij" }, { 0x0135, "j" }, { 0x0138, "k" }, { 0x0142, "l" }, { 0x014B, "n" },
        { 0x0151, "o" }, { 0x0153, "oe" }, { 0x0159, "r" }, { 0x0161, "s" }, { 0x0167, "t" }, { 0x0173, "u" },
        { 0x0175, "w" }, { 0x0178, "y" }, { 0x017E, "z" }, { 0x017F, "s" },
    };
    if (cp >= 0xC0 && cp <= 0xFF) { return k_c0[cp - 0xC0]; }
    if (cp >= 0x100 && cp <= 0x17F) {
        for (const auto & r : k_ext_a) {
            if (cp <= r.last) { return r.s; }
        }
    }
    return nullptr;
}

} // namespace

std::string speak_tts_text(const std::string & text) {
    // digits first: only letters survive below, so "42" or a "2024." chunk would otherwise be spoken as nothing
    const std::string t = speak_numbers_to_words(text);
    std::string out;
    out.reserve(t.size() * 2);

    bool pending_sep = false;
    auto put = [&](const char * s) {
        if (pending_sep && !out.empty()) { out += "<|text_sep|>"; }
        pending_sep = false;
        out += s;
    };
    const size_t n = t.size();
    for (size_t i = 0; i < n; ) {
        const unsigned char c = (unsigned char) t[i];
        if (c < 0x80) {
            const char lc = (char) std::tolower(c);
            if (lc >= 'a' && lc <= 'z') {
                const char s[2] = { lc, 0 };
                put(s);
            } else {
                pending_sep = true;
            }
            ++i;
            continue;
        }
        // every letter latin_fold knows is a two-byte sequence; any other byte (CJK, emoji, stray continuation) separates
        const char * folded = nullptr;
        size_t len = 1;
        if ((c & 0xE0) == 0xC0 && i + 1 < n && ((unsigned char) t[i + 1] & 0xC0) == 0x80) {
            folded = latin_fold(((uint32_t) (c & 0x1F) << 6) | ((unsigned char) t[i + 1] & 0x3F));
            len = 2;
        }
        if (folded) { put(folded); } else { pending_sep = true; }
        i += len;
    }
    if (!out.empty()) {
        out += "<|text_sep|>";
    }
    return out;
}

speak_wav_sink::~speak_wav_sink() { close(); }

void speak_wav_sink::close() {
    if (m_f) {
        std::fclose(m_f);
        m_f = nullptr;
    }
}

namespace {
void put_u32(unsigned char * q, uint32_t v) { q[0] = (unsigned char) v; q[1] = (unsigned char) (v >> 8); q[2] = (unsigned char) (v >> 16); q[3] = (unsigned char) (v >> 24); }
void put_u16(unsigned char * q, uint16_t v) { q[0] = (unsigned char) v; q[1] = (unsigned char) (v >> 8); }
}  // namespace

bool speak_wav_sink::open(const std::string & path, int32_t sample_rate, std::string & err) {
    close();
    m_n = 0;
    m_f = std::fopen(path.c_str(), "wb");
    if (!m_f) {
        err = "cannot write '" + path + "'";
        return false;
    }
    unsigned char h[44];
    std::memcpy(h, "RIFF", 4);  put_u32(h + 4, 36);
    std::memcpy(h + 8, "WAVEfmt ", 8);  put_u32(h + 16, 16);
    put_u16(h + 20, 1);  put_u16(h + 22, 1);
    put_u32(h + 24, (uint32_t) sample_rate);  put_u32(h + 28, (uint32_t) sample_rate * 2);
    put_u16(h + 32, 2);  put_u16(h + 34, 16);
    std::memcpy(h + 36, "data", 4);  put_u32(h + 40, 0);
    if (std::fwrite(h, 1, sizeof(h), m_f) != sizeof(h) || std::fflush(m_f) != 0) {
        err = "short write on '" + path + "'";
        close();
        return false;
    }
    return true;
}

bool speak_wav_sink::append(const std::vector<float> & mono, float gain, std::string & err) {
    if (!m_f) {
        err = "speak_wav_sink: not open";
        return false;
    }
    std::vector<unsigned char> pcm(mono.size() * 2);
    for (size_t i = 0; i < mono.size(); ++i) {
        float v = mono[i] * gain * 32767.0f;
        v = std::min(32767.0f, std::max(-32768.0f, v));
        put_u16(&pcm[i * 2], (uint16_t) (int16_t) v);
    }
    if (std::fseek(m_f, 0, SEEK_END) != 0 || (!pcm.empty() && std::fwrite(pcm.data(), 1, pcm.size(), m_f) != pcm.size())) {
        err = "short write while appending to the WAV";
        return false;
    }
    m_n += mono.size();
    const uint32_t data_bytes = (uint32_t) (m_n * 2);
    unsigned char sz[4];
    put_u32(sz, 36 + data_bytes);
    if (std::fseek(m_f, 4, SEEK_SET) != 0 || std::fwrite(sz, 1, 4, m_f) != 4) { err = "cannot patch the RIFF size"; return false; }
    put_u32(sz, data_bytes);
    if (std::fseek(m_f, 40, SEEK_SET) != 0 || std::fwrite(sz, 1, 4, m_f) != 4) { err = "cannot patch the data size"; return false; }
    if (std::fflush(m_f) != 0) { err = "flush failed"; return false; }
    return true;
}

int speak_code_budget(const std::string & said, int n_predict) {
    // count only the spoken letters: every word is followed by the 12-byte separator token
    static const std::string sep = "<|text_sep|>";
    int letters = 0;
    for (size_t i = 0; i < said.size(); ) {
        if (said.compare(i, sep.size(), sep) == 0) { i += sep.size(); continue; }
        ++letters;
        ++i;
    }
    // 12 codes per letter is about twice the slowest natural pace (75 codes/s at ~6 letters/s); +50 covers lead-in/out
    const long long want = std::max<long long>(100, 12LL * letters + 50);
    return (int) std::min<long long>(want, std::max(n_predict, 1));
}
