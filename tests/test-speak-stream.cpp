// test-speak-stream.cpp -- examples/audio/speak-stream: the sentence cutter and the incremental WAV sink behind
// `llama-audio --speak-stream`. Expected values are worked out by hand from the rules written in speak-stream.h, not
// computed with the code under test; the text is fed the way tokens arrive (a few characters at a time) and the chunks
// must not depend on how it is sliced.
//
// REWRITE-SAFE: the only file written is a fresh "test-speak-stream.tmp.wav" in the working directory, removed at the end.

#include "speak-stream.h"
#include "audio-common.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\n')) { ++a; }
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n')) { --b; }
    return s.substr(a, b - a);
}

// feed `text` `step` bytes at a time, cut after every append exactly like the streamer does
static std::vector<std::string> stream_chunks(const std::string & text, size_t step, const speak_cut_params & p = speak_cut_params()) {
    std::vector<std::string> out;
    std::string buf;
    size_t spoken = 0;
    for (size_t i = 0; i < text.size(); i += step) {
        buf += text.substr(i, step);
        for (;;) {
            const size_t e = speak_sentence_end(buf, spoken, false, p);
            if (e == std::string::npos) { break; }
            out.push_back(trim(buf.substr(spoken, e - spoken)));
            spoken = e;
        }
    }
    const size_t e = speak_sentence_end(buf, spoken, true, p);
    if (e != std::string::npos) { out.push_back(trim(buf.substr(spoken, e - spoken))); }
    return out;
}

static void expect_chunks(const char * name, const std::string & text, const std::vector<std::string> & want, const speak_cut_params & p = speak_cut_params()) {
    for (size_t step : { (size_t) 1, (size_t) 3, (size_t) 7, text.size() }) {
        const std::vector<std::string> got = stream_chunks(text, step, p);
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); ++i) { same = got[i] == want[i]; }
        if (!same) {
            ++g_fail;
            printf("FAIL %s (step %zu): want %zu chunk(s), got %zu\n", name, step, want.size(), got.size());
            for (size_t i = 0; i < got.size(); ++i) { printf("   got[%zu] = [%s]\n", i, got[i].c_str()); }
        }
    }
}

static std::vector<uint8_t> read_all(const char * path) {
    std::vector<uint8_t> v;
    FILE * f = fopen(path, "rb");
    if (!f) { return v; }
    uint8_t b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof(b), f)) > 0) { v.insert(v.end(), b, b + n); }
    fclose(f);
    return v;
}
static uint32_t rd32(const std::vector<uint8_t> & v, size_t o) { return v[o] | (v[o + 1] << 8) | (v[o + 2] << 16) | ((uint32_t) v[o + 3] << 24); }

int main() {
    // 1. plain sentences; "Fine." is short but the text is over, so the final flush speaks it
    expect_chunks("basic", "Hello there, friend. How are you today? Fine.",
                  { "Hello there, friend.", "How are you today?", "Fine." });

    // 2. a decimal point is not a sentence end, and a '.' at the very end of the buffer waits for the next token
    expect_chunks("decimal", "The value is 3.5 meters long. Next.", { "The value is 3.5 meters long.", "Next." });
    {
        // NEGATIVE CONTROL: a naive "cut at the first '.'" would have spoken "The value is 3." -- the case above can fail
        const std::string t = "The value is 3.5 meters long. Next.";
        CHECK(t.substr(0, t.find('.') + 1) == "The value is 3.", "control: the naive cutter splits the decimal");
        CHECK(speak_sentence_end("The value is 3.", 0, false) == std::string::npos, "a buffer that ends on '3.' must wait");
    }

    // 3. abbreviations and lone initials
    expect_chunks("abbrev", "Dr. Smith went to St. Louis today. He left.", { "Dr. Smith went to St. Louis today.", "He left." });
    expect_chunks("initials", "J. R. R. Tolkien wrote many long books. Yes.", { "J. R. R. Tolkien wrote many long books.", "Yes." });

    // 4. a candidate shorter than min_chars is merged into the next sentence
    expect_chunks("min_chars", "Yes. I think that is right. Done.", { "Yes. I think that is right.", "Done." });
    expect_chunks("run of terminators", "Really?! Yes it is. Okay.", { "Really?! Yes it is.", "Okay." });
    {
        speak_cut_params p; p.min_chars = 1;
        expect_chunks("min_chars=1", "Yes. I think that is right.", { "Yes.", "I think that is right." }, p);
    }

    // 5. closing quotes and brackets stay with their sentence
    expect_chunks("quote", "He said \"Stop right there.\" Then he left.", { "He said \"Stop right there.\"", "Then he left." });

    // 6. a newline ends a chunk (list items, paragraphs)
    expect_chunks("newline", "Item one is here\nItem two is also here\n", { "Item one is here", "Item two is also here" });

    // 7. CJK full stop: 8 characters of 3 bytes + the 3-byte stop = 27 bytes
    {
        const std::string t = "\xE4\xBB\x8A\xE5\xA4\xA9\xE7\x9A\x84\xE5\xA4\xA9\xE6\xB0\x94\xE9\x9D\x9E\xE5\xB8\xB8\xE5\xA5\xBD\xE3\x80\x82\xE6\x98\x8E\xE5\xA4\xA9";
        CHECK(speak_sentence_end(t, 0, false) == 27, "CJK stop: got %zu", speak_sentence_end(t, 0, false));
    }

    // 8. nothing to cut yet / nothing left
    CHECK(speak_sentence_end("no terminator here yet", 0, false) == std::string::npos, "no terminator, not final");
    CHECK(speak_sentence_end("no terminator here yet", 0, true) == 22, "final flush returns the remainder");
    CHECK(speak_sentence_end("   \n  ", 0, true) != 0 && speak_sentence_end("   ", 0, true) == std::string::npos, "whitespace-only remainder is not speakable");
    CHECK(speak_sentence_end("", 0, true) == std::string::npos, "empty");

    // 9. no sentence end for max_chars: a word boundary (60 words of "word " -> the space at index 239 -> cut at 240) ...
    {
        std::string t;
        for (int i = 0; i < 60; ++i) { t += "word "; }
        CHECK(speak_sentence_end(t, 0, false) == 240, "max_chars word cut: got %zu", speak_sentence_end(t, 0, false));
        // ... or, when there is a clause break in the first part, at the last one: 100 a, ", ", 100 b, " ", 100 c, " " -> cut after the comma (101)
        const std::string c = std::string(100, 'a') + ", " + std::string(100, 'b') + " " + std::string(100, 'c') + " ";
        CHECK(speak_sentence_end(c, 0, false) == 101, "max_chars clause cut: got %zu", speak_sentence_end(c, 0, false));
    }

    // 10. speak_has_speech
    CHECK(!speak_has_speech("...") && !speak_has_speech(" *** ") && speak_has_speech("a.") && speak_has_speech("7") && speak_has_speech("\xE4\xBB\x8A"), "speak_has_speech");

    // 10b. speak_numbers_to_words: a chunk speak_has_speech accepts for its digits must keep them as words
    {
        struct { const char * in; const char * want; } k_cases[] = {
            { "I owe 42 dollars.",         "I owe forty-two dollars." },
            { "2024.",                     "two thousand twenty-four." },
            { "0",                         "zero" },
            { "7 of 10",                   "seven of ten" },
            { "100",                       "one hundred" },
            { "115",                       "one hundred fifteen" },
            { "1,234,567",                 "one million two hundred thirty-four thousand five hundred sixty-seven" },
            { "3.14",                      "three point one four" },
            { "The value is 3.",           "The value is three." },
            { "007",                       "zero zero seven" },
            { "42nd",                      "forty-two nd" },
            { "x2",                        "x two" },
            { "12,34",                     "twelve,thirty-four" },
            { "1234,567",                  "one thousand two hundred thirty-four,five hundred sixty-seven" },
            { "2000000000",                "two billion" },
            { "9999999999999",             "nine nine nine nine nine nine nine nine nine nine nine nine nine" },
            { "no digits here",            "no digits here" },
            { "",                          "" },
        };
        for (const auto & c : k_cases) {
            const std::string got = speak_numbers_to_words(c.in);
            CHECK(got == c.want, "numbers_to_words(\"%s\"): got \"%s\" want \"%s\"", c.in, got.c_str(), c.want);
        }
    }

    // 10c. speak_tts_text: the prompt the speaker sees; accented Latin folds to ASCII, anything else separates or vanishes
    {
        #define S "<|text_sep|>"
        struct { const char * in; const char * want; } k_cases[] = {
            { "Hi 42",                         "hi" S "forty" S "two" S },
            { "caf\xC3\xA9",                   "cafe" S },                    // café
            { "Stra\xC3\x9F" "e",              "strasse" S },                 // Straße
            { "\xC5\x92uvre",                  "oeuvre" S },                  // Œuvre
            { "\xC5\x81\xC3\xB3" "d\xC5\xBA",  "lodz" S },                    // Łódź
            { "na\xC3\xAFve 2024.",            "naive" S "two" S "thousand" S "twenty" S "four" S },
            { "a\xE2\x80\x94" "b",             "a" S "b" S },                 // em dash separates
            { "3\xC3\x97" "4",                 "three" S "four" S },          // U+00D7 is not a letter
            { "\x80" "a",                      "a" S },                       // stray continuation byte
            { "\xE4\xBB\x8A\xE6\x97\xA5",      "" },                          // CJK only
            { "...",                           "" },
            { "",                              "" },
        };
        #undef S
        for (const auto & c : k_cases) {
            const std::string got = speak_tts_text(c.in);
            CHECK(got == c.want, "speak_tts_text(\"%s\"): got \"%s\" want \"%s\"", c.in, got.c_str(), c.want);
        }
        // the case the streamer must skip: speak_has_speech accepts it, the speaker has nothing to say
        CHECK(speak_has_speech("\xE4\xBB\x8A") && speak_tts_text("\xE4\xBB\x8A").empty(), "CJK chunk: has_speech but empty prompt");
    }

    // 11. the sink: the file is a valid WAV after EVERY append, and the bytes equal audio_wav_encode_s16 of the whole signal
    {
        const char * path = "test-speak-stream.tmp.wav";
        std::vector<float> a(1000), b(777);
        for (size_t i = 0; i < a.size(); ++i) { a[i] = 0.5f * (float) ((int) (i % 40) - 20) / 20.0f; }
        for (size_t i = 0; i < b.size(); ++i) { b[i] = 1.4f * (float) ((int) (i % 30) - 15) / 15.0f; }   // overdriven: must clip, not wrap
        speak_wav_sink s;
        std::string err;
        CHECK(s.open(path, 24000, err), "open: %s", err.c_str());
        CHECK(s.append(a, 1.0f, err), "append a: %s", err.c_str());
        std::vector<uint8_t> f1 = read_all(path);           // read while the sink is still open
        CHECK(f1.size() == 44 + 2000, "size after the first append: %zu", f1.size());
        CHECK(f1.size() >= 44 && rd32(f1, 4) == 36 + 2000 && rd32(f1, 40) == 2000, "header after the first append: riff %u data %u", f1.size() >= 44 ? rd32(f1, 4) : 0, f1.size() >= 44 ? rd32(f1, 40) : 0);
        CHECK(s.append(b, 1.0f, err), "append b: %s", err.c_str());
        s.close();
        std::vector<float> all = a;
        all.insert(all.end(), b.begin(), b.end());
        const std::vector<uint8_t> want = audio_wav_encode_s16(all, 24000);
        const std::vector<uint8_t> got = read_all(path);
        CHECK(got == want, "sink bytes differ from audio_wav_encode_s16 (sizes %zu vs %zu)", got.size(), want.size());
        // gain: half of 0.5 -> 0.25 -> 8191 (truncating, as encode does)
        speak_wav_sink g;
        CHECK(g.open(path, 16000, err) && g.append(std::vector<float>(3, 0.5f), 0.5f, err), "gain open/append");
        g.close();
        const std::vector<uint8_t> gg = read_all(path);
        CHECK(gg.size() == 44 + 6 && (int16_t) (gg[44] | (gg[45] << 8)) == 8191 && rd32(gg, 24) == 16000, "gain: size %zu", gg.size());
        std::remove(path);
    }

    // speak_code_budget: max(100, 12 * letters + 50) capped at n_predict; separators are not letters. Worked by hand.
    {
        CHECK(speak_code_budget("", 512) == 100, "empty -> floor 100, got %d", speak_code_budget("", 512));
        CHECK(speak_code_budget("a<|text_sep|>", 512) == 100, "1 letter -> 62 -> floor 100");
        const std::string hw = speak_tts_text("hello world");   // 10 letters, separators in between
        CHECK(speak_code_budget(hw, 512) == 170, "10 letters -> 170, got %d", speak_code_budget(hw, 512));
        CHECK(speak_code_budget(hw, 120) == 120, "cap by n_predict 120, got %d", speak_code_budget(hw, 120));
        CHECK(speak_code_budget(std::string(30, 'a'), 1000) == 410, "30 letters -> 410");
        CHECK(speak_code_budget(std::string(100, 'a'), 512) == 512, "100 letters -> 1250 capped at 512");
        CHECK(speak_code_budget("abc", 0) == 1, "n_predict 0 -> 1, got %d", speak_code_budget("abc", 0));
        CHECK(speak_code_budget(std::string(5000000, 'a'), 2147483647) == 2147483647 || speak_code_budget(std::string(5000000, 'a'), 2147483647) == 60000050, "no overflow on huge text");
    }

    if (g_fail) { printf("%d FAILURE(S)\n", g_fail); return 1; }
    printf("test-speak-stream: OK\n");
    return 0;
}
