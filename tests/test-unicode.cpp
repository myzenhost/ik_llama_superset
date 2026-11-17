// NFKC cases follow UAX #15 (https://www.unicode.org/reports/tr15/), Section 1.3 / Table 1: NFKC = compatibility
// decomposition, then canonical composition; its long-s example (1E9B 0323 -> 1E69) is one of the cases below.
#include "../src/unicode.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

static int test_nfkc() {
    // inputs hit each part of UAX #15; expected values from Python unicodedata 14.0.0 NFKC (the table generator's source)
    const std::vector<std::tuple<const char *, std::vector<uint32_t>, std::vector<uint32_t>>> nfkc_cases = {
        { "combining acute composes", {0x0065, 0x0301}, {0x00E9} },
        { "Angstrom sign singleton", {0x212B}, {0x00C5} },
        { "Ohm sign singleton", {0x2126}, {0x03A9} },
        { "fi ligature", {0xFB01}, {0x0066, 0x0069} },
        { "vulgar one half", {0x00BD}, {0x0031, 0x2044, 0x0032} },
        { "fullwidth tilde", {0xFF5E}, {0x007E} },
        { "fullwidth Latin and digits", {0xFF21, 0xFF42, 0xFF43, 0xFF11, 0xFF12, 0xFF13}, {0x0041, 0x0062, 0x0063, 0x0031, 0x0032, 0x0033} },
        { "halfwidth katakana", {0xFF71}, {0x30A2} },
        { "halfwidth katakana + voiced mark", {0xFF76, 0xFF9E}, {0x30AC} },
        { "no-break space", {0x0061, 0x00A0, 0x0062}, {0x0061, 0x0020, 0x0062} },
        { "ideographic space", {0x3000}, {0x0020} },
        { "circled digit", {0x2460}, {0x0031} },
        { "Hangul jamo L V T compose", {0x1100, 0x1161, 0x11A8}, {0xAC01} },
        { "Hangul syllable LV + T compose", {0xAC00, 0x11A8}, {0xAC01} },
        { "Hangul syllables stay", {0xD55C, 0xAE00}, {0xD55C, 0xAE00} },
        { "reorder ogonek before acute", {0x0061, 0x0301, 0x0328}, {0x0105, 0x0301} },
        { "sequential composition A ring acute", {0x0041, 0x030A, 0x0301}, {0x01FA} },
        { "UAX15 long s dot below dot above", {0x1E9B, 0x0323}, {0x1E69} },
        { "blocked: same ccc twice", {0x0061, 0x0301, 0x0301}, {0x00E1, 0x0301} },
        { "composition exclusion stays decomposed", {0x0958}, {0x0915, 0x093C} },
        { "non-starter decomposition", {0x0F73}, {0x0F71, 0x0F72} },
        { "superscript two", {0x0078, 0x00B2}, {0x0078, 0x0032} },
        { "leading combining mark", {0x0301, 0x0065}, {0x0301, 0x0065} },
        { "mixed ASCII and CJK unchanged", {0x0061, 0x0070, 0x0070, 0x006C, 0x0065, 0x5DE5, 0x4F5C}, {0x0061, 0x0070, 0x0070, 0x006C, 0x0065, 0x5DE5, 0x4F5C} },
    };

    int n_fail = 0;
    for (const auto & [name, input, expected] : nfkc_cases) {
        const auto actual = unicode_cpts_normalize_nfkc(input);
        std::string in_utf8;
        for (const uint32_t cpt : input) {
            in_utf8 += unicode_cpt_to_utf8(cpt);
        }
        const auto actual_str = unicode_cpts_from_utf8(unicode_normalize_nfkc(in_utf8));
        if (actual != expected || actual_str != expected) {
            fprintf(stderr, "NFKC %s: got", name);
            for (const uint32_t cpt : actual) {
                fprintf(stderr, " %04X", cpt);
            }
            fprintf(stderr, " (string path:");
            for (const uint32_t cpt : actual_str) {
                fprintf(stderr, " %04X", cpt);
            }
            fprintf(stderr, "), expected");
            for (const uint32_t cpt : expected) {
                fprintf(stderr, " %04X", cpt);
            }
            fprintf(stderr, "\n");
            ++n_fail;
        }
    }

    // invalid UTF-8 is handed back untouched for the byte-level tokenizer
    const std::string invalid = std::string("a\xC2\xBD") + "\xFF";
    if (unicode_normalize_nfkc(invalid) != invalid) {
        fprintf(stderr, "NFKC: invalid UTF-8 was changed\n");
        ++n_fail;
    }
    return n_fail;
}

// The generic std::wregex stage (an expression with no \p category) must not see an astral codepoint as the BMP
// codepoint sharing its low 16 bits. With a 16-bit wchar_t, U+10030 used to become '0'. Reference: tiiuae/falcon-7b
// tokenizer.json, whose Split "[0-9][0-9][0-9]" uses the Rust regex crate, where [0-9] is ASCII only. HF makes no cut at the
// letter/astral boundaries. It does cut byte B2 out of U+10032, because its Digits stage runs after ByteLevel. llama.cpp does not
// model that cut, and it changes no ids (11/11 probes, 2026-10-15).
static int test_wregex_astral() {
    const std::vector<std::string> falcon = {
        "[\\p{P}\\$\\+<=>\\^~\\|`]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "[0-9][0-9][0-9]",
    };
    const std::string a3 = unicode_cpt_to_utf8(0x10030) + unicode_cpt_to_utf8(0x10031) + unicode_cpt_to_utf8(0x10032);
    const std::string b3 = unicode_cpt_to_utf8(0x20030) + unicode_cpt_to_utf8(0x20031) + unicode_cpt_to_utf8(0x20032);
    const std::vector<std::tuple<const char *, std::string, std::vector<std::string>>> cases = {
        // control: the rule itself still isolates real ASCII digit runs
        { "ascii digits split", "a012b", { "a", "012", "b" } },
        { "U+10030..32 is not [0-9]", "a" + a3 + "b", { "a" + a3 + "b" } },
        { "U+20030..32 is not [0-9]", "x" + b3 + "y", { "x" + b3 + "y" } },
    };
    int n_fail = 0;
    for (const auto & [name, text, want] : cases) {
        const auto got = unicode_regex_split(text, falcon, false);
        if (got != want) {
            fprintf(stderr, "wregex astral: %s: got %zu piece(s), want %zu\n", name, got.size(), want.size());
            ++n_fail;
        }
    }
    return n_fail;
}

// afmoe digit stage: tokenizer.json Digits groups any \p{N} run, then its Splits group only \p{Nd} by 3 from the right.
// A non-Nd number (U+00BD vulgar fraction, U+216B roman numeral) inside the run must stay its own piece and must not shift
// the grouping. Expected pieces are HF's (arcee-ai/Trinity tokenizer.json, Digits + the three Nd Splits).
static int test_afmoe_digits_nd() {
    const std::vector<std::string> afmoe = { "\\p{AFMoE_digits}" };
    const std::string half = "\xC2\xBD", three_q = "\xC2\xBE", xii = "\xE2\x85\xAB";
    const std::string ar1 = "\xD9\xA1", ar234 = "\xD9\xA2\xD9\xA3\xD9\xA4";  // U+0661, U+0662..0664 (Nd)
    const std::vector<std::tuple<const char *, std::string, std::vector<std::string>>> cases = {
        // controls: plain ASCII and non-ASCII Nd runs keep the 1-2 + groups-of-3 split
        { "ascii 7 digits", "1234567", { "1", "234", "567" } },
        { "arabic-indic 4 digits", ar1 + ar234, { ar1, ar234 } },
        { "Nd run then fraction", "1234567" + half, { "1", "234", "567", half } },
        { "fraction then Nd run", half + "1234", { half, "1", "234" } },
        { "fraction inside", "12" + half + "345", { "12", half, "345" } },
        { "roman numeral then Nd", xii + "12", { xii, "12" } },
        { "two fractions inside", "7" + half + three_q + "8888", { "7", half + three_q, "8", "888" } },
    };
    int n_fail = 0;
    for (const auto & [name, text, want] : cases) {
        const auto got = unicode_regex_split(text, afmoe, false);
        if (got != want) {
            fprintf(stderr, "afmoe digits: %s: got", name);
            for (const auto & p : got) {
                fprintf(stderr, " [%s]", p.c_str());
            }
            fprintf(stderr, "\n");
            ++n_fail;
        }
    }
    return n_fail;
}

// tiny_aya digit stage, tokenizer.json Split \d{1,3}(?=(?:\d{3})*\b): a \d run is grouped only when a word boundary follows
// it. Onig's word set includes letters, '_', CJK and (Latin-1 quirk) U+00BD and U+00B2, so after those the run is NOT split.
// Expected pieces are HF's (tokenizers.pre_tokenizers.Split with that regex alone).
static int test_tiny_aya_digits() {
    const std::vector<std::string> tiny_aya = { "\\d{1,3}(?=(?:\\d{3})*\\b)" };
    const std::string half = "\xC2\xBD", sup2 = "\xC2\xB2", nian = "\xE5\xB9\xB4";  // U+00BD, U+00B2, U+5E74
    const std::vector<std::tuple<const char *, std::string, std::vector<std::string>>> cases = {
        // controls: a run that ends the text or meets a space/comma is grouped
        { "ascii 7 digits", "1234567", { "1", "234", "567" } },
        { "run then space", "1234 x", { "1", "234", " x" } },
        { "comma groups", "12,345", { "12", ",", "345" } },
        { "fraction before run", half + "1234", { half, "1", "234" } },
        { "letter after run", "1234abc", { "1234abc" } },
        { "underscore after run", "1234_5", { "1234_", "5" } },  // the trailing 5 ends the text, so it is isolated
        { "fraction after run", "1234567" + half, { "1234567" + half } },
        { "superscript after run", "1234" + sup2, { "1234" + sup2 } },
        { "CJK after run", "2024" + nian, { "2024" + nian } },
    };
    int n_fail = 0;
    for (const auto & [name, text, want] : cases) {
        const auto got = unicode_regex_split(text, tiny_aya, false);
        if (got != want) {
            fprintf(stderr, "tiny_aya digits: %s: got", name);
            for (const auto & p : got) {
                fprintf(stderr, " [%s]", p.c_str());
            }
            fprintf(stderr, "\n");
            ++n_fail;
        }
    }
    return n_fail;
}

// o200k pattern with real case classes (GPT4O, MINIMAX_M2, TINY_AYA, INKLING, GRANITE_EMB_MULTI): [\p{Lu}..]*[\p{Ll}..]+ must
// see the case of every letter, ASCII or not. Expected pieces are HF's (Split of the Xenova/gpt-4o tokenizer.json regex alone).
static int test_case_classes() {
    const std::vector<std::string> o200k = {
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    const auto u = [](std::initializer_list<uint32_t> cpts) {
        std::string s;
        for (const uint32_t c : cpts) {
            s += unicode_cpt_to_utf8(c);
        }
        return s;
    };
    const std::vector<std::tuple<const char *, std::string, std::vector<std::string>>> cases = {
        // controls: Devanagari marks stay on their letters, ASCII contractions stay attached
        { "devanagari word", u({0x0915, 0x093F, 0x0924, 0x093E, 0x092C}), { u({0x0915, 0x093F, 0x0924, 0x093E, 0x092C}) } },
        { "contractions", "IT'S it's", { "IT'S", " it's" } },
        { "ascii lower then upper", "aX380", { "a", "X", "380" } },
        { "camel case", "camelCase", { "camel", "Case" } },
        { "upper run then word", "getHTTPResponse", { "get", "HTTPResponse" } },
        { "latin-1 upper", "a" + u({0x00C9}), { "a", u({0x00C9}) } },
        { "latin-1 mixed", u({0x00E9, 0x00C9, 0x0074, 0x00E9}), { u({0x00E9}), u({0x00C9, 0x0074, 0x00E9}) } },
        { "titlecase Lt", "a" + u({0x01C5}), { "a", u({0x01C5}) } },
        { "greek", u({0x03A3, 0x03C3, 0x03A3}), { u({0x03A3, 0x03C3}), u({0x03A3}) } },
        { "cyrillic", u({0x0414, 0x0434, 0x0414}), { u({0x0414, 0x0434}), u({0x0414}) } },
        { "modifier letter Lm", "a" + u({0x02B0}) + "B", { "a" + u({0x02B0}), "B" } },
        { "caseless Ll U+0138", u({0x0138}) + "K", { u({0x0138}), "K" } },
        { "astral upper", "a" + u({0x10400}), { "a", u({0x10400}) } },
    };
    int n_fail = 0;
    for (const auto & [name, text, want] : cases) {
        const auto got = unicode_regex_split(text, o200k, false);
        if (got != want) {
            fprintf(stderr, "case classes: %s: got", name);
            for (const auto & p : got) {
                fprintf(stderr, " [%s]", p.c_str());
            }
            fprintf(stderr, "\n");
            ++n_fail;
        }
    }
    return n_fail;
}

// The flag and case tables follow the Unicode version HF tokenizers use (16.0, scripts/gen-unicode-data.py). Every code point
// below is new in 16.0 (absent from 15.1.0 UnicodeData.txt), so a 15.1 table calls them undefined and no \p{N}/\p{L}/case
// rule matches them. Expected values are the 16.0.0 UnicodeData.txt records.
static int test_ucd16() {
    const auto u = [](std::initializer_list<uint32_t> cpts) {
        std::string s;
        for (const uint32_t c : cpts) {
            s += unicode_cpt_to_utf8(c);
        }
        return s;
    };
    int n_fail = 0;
    // controls first: ASCII is the same in every version
    const std::vector<std::tuple<const char *, uint32_t, bool, bool>> flag_cases = {  // name, cpt, is_number, is_letter
        { "ASCII 0", 0x0030, true, false },
        { "ASCII A", 0x0041, false, true },
        { "Ol Onal digit zero Nd", 0x1E5F1, true, false },
        { "Ol Onal letter o Lo", 0x1E5D0, false, true },
        { "Sunuwar digit zero Nd", 0x11BF0, true, false },
        { "Garay digit zero Nd", 0x10D40, true, false },
        { "Latin capital rams horn Lu", 0xA7CB, false, true },
    };
    for (const auto & [name, cpt, number, letter] : flag_cases) {
        const auto f = unicode_cpt_flags_from_cpt(cpt);
        if (f.is_undefined || f.is_number != number || f.is_letter != letter) {
            fprintf(stderr, "ucd16 flags: %s U+%04X: undefined %d number %d letter %d\n", name, cpt, f.is_undefined, f.is_number, f.is_letter);
            ++n_fail;
        }
    }
    const std::vector<std::tuple<const char *, uint32_t, uint32_t>> lower_cases = {
        { "ASCII A", 0x0041, 0x0061 },
        { "Latin capital rams horn", 0xA7CB, 0x0264 },
        { "Garay capital a", 0x10D50, 0x10D70 },
    };
    for (const auto & [name, cpt, want] : lower_cases) {
        if (unicode_tolower(cpt) != want) {
            fprintf(stderr, "ucd16 tolower: %s U+%04X -> U+%04X, want U+%04X\n", name, cpt, unicode_tolower(cpt), want);
            ++n_fail;
        }
    }
    // the splitters see the new classes: afmoe groups an Nd run 1 + 3 from the right, o200k cuts before an uppercase letter
    const std::string ol = u({0x1E5F2, 0x1E5F3, 0x1E5F4, 0x1E5F1});  // Ol Onal 1 2 3 0
    const std::vector<std::tuple<const char *, std::vector<std::string>, std::string, std::vector<std::string>>> split_cases = {
        { "afmoe Ol Onal digits", { "\\p{AFMoE_digits}" }, ol, { u({0x1E5F2}), u({0x1E5F3, 0x1E5F4, 0x1E5F1}) } },
        { "number run Ol Onal", { "\\p{N}+" }, "a" + ol + "b", { "a", ol, "b" } },
        { "o200k rams horn upper", { "[\\p{Lu}]+[\\p{Ll}]*|[\\p{Ll}]+" }, "a" + u({0xA7CB}), { "a", u({0xA7CB}) } },
    };
    for (const auto & [name, exprs, text, want] : split_cases) {
        const auto got = unicode_regex_split(text, exprs, false);
        if (got != want) {
            fprintf(stderr, "ucd16 split: %s: got", name);
            for (const auto & p : got) {
                fprintf(stderr, " [%s]", p.c_str());
            }
            fprintf(stderr, "\n");
            ++n_fail;
        }
    }
    return n_fail;
}

// S69-FORK-BPE-UPPERCASE-CONTRACTION-REGRESSION-UNMEASURED-01: the llama3/qwen2/qwen35 BPE splitters fold the char after an
// apostrophe with unicode_tolower, so a tolower that silently maps 0/26 ASCII letters (the 2026-09-02 bug) splits 'RE 'S 'T 'D
// 'LL 'VE 'M off as punctuation + word. Pins A-Z -> a-z and the uppercase-contraction split for the llama3 regex.
static int test_uppercase_contractions() {
    int n_fail = 0;
    for (uint32_t c = 'A'; c <= 'Z'; ++c) {
        if (unicode_tolower(c) != c - 'A' + 'a') {
            fprintf(stderr, "tolower: U+%04X -> U+%04X, want U+%04X\n", c, unicode_tolower(c), c - 'A' + 'a');
            ++n_fail;
        }
    }
    const std::vector<std::string> llama3 = {
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
        { "'RE you", { "'RE", " you" } },
        { "'S 'T 'M 'D", { "'S", " '", "T", " '", "M", " '", "D" } },
        { "We'Ve a'lL", { "We", "'Ve", " a", "'lL" } },
        { "it's", { "it", "'s" } },
    };
    for (const auto & [text, want] : cases) {
        const auto got = unicode_regex_split(text, llama3, false);
        if (got != want) {
            fprintf(stderr, "uppercase contraction: [%s] got", text.c_str());
            for (const auto & p : got) {
                fprintf(stderr, " [%s]", p.c_str());
            }
            fprintf(stderr, "\n");
            ++n_fail;
        }
    }
    return n_fail;
}

int main() {
    if (test_uppercase_contractions() != 0) {
        return 1;
    }
    if (test_ucd16() != 0) {
        return 1;
    }
    if (test_case_classes() != 0) {
        return 1;
    }
    if (test_wregex_astral() != 0) {
        return 1;
    }
    if (test_afmoe_digits_nd() != 0) {
        return 1;
    }
    if (test_tiny_aya_digits() != 0) {
        return 1;
    }
    bool success = true;
    const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> normalization = {
        {{}, {}},
        {{0x65, 0x301}, {0xE9}},
        {{0xE9}, {0xE9}},
        {{0x212B}, {0xC5}},
        {{0x61, 0x315, 0x300}, {0xE0, 0x315}},
        {{0x61, 0x305, 0x301}, {0x61, 0x305, 0x301}},
        {{0x301, 0x327}, {0x327, 0x301}},
        {{0x1100, 0x1161, 0x11A8}, {0xAC01}},
        {{0xAC01}, {0xAC01}},
        {{0xFB01, 0xFF21}, {0xFB01, 0xFF21}},
        {{0x958}, {0x915, 0x93C}},
        {{0x1D15E}, {0x1D157, 0x1D165}},
    };
    for (const auto & test : normalization) {
        if (unicode_cpts_normalize_nfc(test.first) != test.second || unicode_cpts_normalize_nfc(test.second) != test.second) {
            fprintf(stderr, "unexpected NFC normalization\n");
            success = false;
            break;
        }
    }

    const struct {
        std::string text;
        std::vector<uint32_t> expected;
    } utf8_tests[] = {
        { "\x7f",                 { 0x7f } },
        { "\xc2\x80",             { 0x80 } },
        { "\xdf\xbf",             { 0x7ff } },
        { "\xe0\xa0\x80",         { 0x800 } },
        { "\xef\xbf\xbf",         { 0xffff } },
        { "\xf0\x90\x80\x80",     { 0x10000 } },
        { "\xf4\x8f\xbf\xbf",     { 0x10ffff } },
        // Invalid-input cases preserve the existing recovery behavior: one U+FFFD per byte.
        { "\xc0\xa0",             { 0xfffd, 0xfffd } },
        { "\xc1\xbf",             { 0xfffd, 0xfffd } },
        { "\xe0\x80\xa0",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xe0\x9f\xbf",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x80\x80\xa0",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x8f\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf4\x90\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf5\x80\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf7\xbf\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xe2\x80",             { 0xfffd, 0xfffd } },
    };

    for (const auto & test : utf8_tests) {
        auto expected = test.expected;
        expected.insert(expected.begin(), 'a');
        expected.push_back('b');
        if (unicode_cpts_from_utf8("a" + test.text + "b") != expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        success = false;
    }

    return (success && test_nfkc() == 0) ? 0 : 1;
}
