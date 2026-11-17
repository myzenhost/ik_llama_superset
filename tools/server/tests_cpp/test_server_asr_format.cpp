// Standalone test for server-asr-format.h. Build: g++ -std=c++17 -I.. test_server_asr_format.cpp -o t && ./t
// Expected strings are written out by hand from the SRT / WebVTT / OpenAI verbose_json shapes, not produced by the code under test.
#include "server-asr-format.h"
#include "server-asr-words.h"
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
#define EQ(a, b) do { const std::string _a = (a), _b = (b); if (_a != _b) { std::printf("FAIL line %d:\n--- got\n%s\n--- want\n%s\n", __LINE__, _a.c_str(), _b.c_str()); ++fails; } } while (0)

int main() {
    using namespace asrfmt;
    format f;
    CHECK(parse("", f) && f == format::json);
    CHECK(parse("json", f) && f == format::json);
    CHECK(parse("text", f) && f == format::text);
    CHECK(parse("srt", f) && f == format::srt);
    CHECK(parse("vtt", f) && f == format::vtt);
    CHECK(parse("verbose_json", f) && f == format::verbose_json);
    CHECK(!parse("xml", f));          // refused, not silently json
    CHECK(!parse("SRT", f));          // exact (OpenAI is case sensitive)

    // two sentences of 12 and 27 code points over 4 s: the cut falls at 12/39 * 4 = 1.2308 s -> 1.231
    const std::string t = "Hello world.  This is the second one, ok!";
    EQ(render(format::text, t, 4.0), "Hello world.  This is the second one, ok!\n");
    EQ(render(format::srt, t, 4.0),
       "1\n00:00:00,000 --> 00:00:01,231\nHello world.\n\n"
       "2\n00:00:01,231 --> 00:00:04,000\nThis is the second one, ok!\n\n");
    EQ(render(format::vtt, t, 4.0),
       "WEBVTT\n\n"
       "00:00:00.000 --> 00:00:01.231\nHello world.\n\n"
       "00:00:01.231 --> 00:00:04.000\nThis is the second one, ok!\n\n");
    EQ(render(format::verbose_json, t, 4.0, "en"),
       "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":4.000,\"text\":\"Hello world.  This is the second one, ok!\","
       "\"timestamps\":\"interpolated\",\"segments\":["
       "{\"id\":0,\"seek\":0,\"start\":0.000,\"end\":1.231,\"text\":\"Hello world.\",\"tokens\":[],\"temperature\":0.000,\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null},"
       "{\"id\":1,\"seek\":0,\"start\":1.231,\"end\":4.000,\"text\":\"This is the second one, ok!\",\"tokens\":[],\"temperature\":0.000,\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null}]}");

    // hours roll over, and abbreviations / decimals inside a sentence do not split it
    EQ(timestamp(3725.5, ','), "01:02:05,500");
    CHECK(split_sentences("Pi is 3.14 roughly.").size() == 1);   // a decimal point is not a sentence end
    CHECK(split_sentences("See e.g. this.").size() == 2);        // KNOWN LIMIT: an abbreviation followed by a space does split
    CHECK(split_sentences("One. Two? Three!").size() == 3);
    // CJK full stops split without a following space; code points, not bytes, weigh the segments
    CHECK(split_sentences("\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82\xe5\x86\x8d\xe8\xa7\x81\xe3\x80\x82").size() == 2);
    CHECK(n_codepoints("\xe4\xbd\xa0\xe5\xa5\xbd") == 2);

    // unknown duration: no invented times (all cues are zero-length at 0)
    EQ(render(format::srt, "Hi.", 0.0), "1\n00:00:00,000 --> 00:00:00,000\nHi.\n\n");
    // empty transcript: no cues
    EQ(render(format::srt, "", 5.0), "");
    EQ(render(format::vtt, "  ", 5.0), "WEBVTT\n\n");
    // escaping
    EQ(render(format::verbose_json, "say \"hi\"\n", 1.0),
       "{\"task\":\"transcribe\",\"language\":\"unknown\",\"duration\":1.000,\"text\":\"say \\\"hi\\\"\",\"timestamps\":\"interpolated\",\"segments\":[{\"id\":0,\"seek\":0,\"start\":0.000,\"end\":1.000,\"text\":\"say \\\"hi\\\"\",\"tokens\":[],\"temperature\":0.000,\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null}]}");
    CHECK(std::string(content_type(format::vtt)).find("text/vtt") == 0);
    CHECK(std::string(content_type(format::srt)).find("application/x-subrip") == 0);
    CHECK(std::string(content_type(format::text)).find("text/plain") == 0);
    CHECK(std::string(content_type(format::json)).find("application/json") == 0);

    // ---- tier 1: timestamps the model wrote itself (whisper-style markers); expected strings written out by hand ----
    const std::string m = "<|0.00|> Hello there. <|2.50|> Second part. <|5.00|>";
    EQ(render(format::srt, m, 6.0),
       "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n"
       "2\n00:00:02,500 --> 00:00:05,000\nSecond part.\n\n");
    EQ(render(format::text, m, 6.0), "Hello there.  Second part.\n");     // the markers are not part of the transcript
    EQ(render(format::verbose_json, "<|1.00|> Hi.", 3.0, "en"),            // a run with no closing marker ends at the audio duration
       "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":3.000,\"text\":\"Hi.\",\"timestamps\":\"model\",\"segments\":["
       "{\"id\":0,\"seek\":0,\"start\":1.000,\"end\":3.000,\"text\":\"Hi.\",\"tokens\":[],\"temperature\":0.000,\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null}]}");
    {
        std::vector<segment> sg;
        std::string plain;
        CHECK(!parse_markers("no markers here", 2.0, sg, plain) && plain == "no markers here" && sg.empty());
        // a special token such as <|endoftext|> is not a number: it stays text and does not switch the tier
        CHECK(!parse_markers("Hi.<|endoftext|>", 2.0, sg, plain));
        CHECK(parse_markers("<|0.5|>a<|1.5|>", 0.0, sg, plain) && sg.size() == 1 && sg[0].t0 == 0.5 && sg[0].t1 == 1.5);
    }
    // avg_logprob rides on every segment of verbose_json when the caller has it, and is null otherwise
    EQ(render(format::verbose_json, "Hi.", 1.0, "en", true, -0.25),
       "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":1.000,\"text\":\"Hi.\",\"timestamps\":\"interpolated\",\"segments\":["
       "{\"id\":0,\"seek\":0,\"start\":0.000,\"end\":1.000,\"text\":\"Hi.\",\"tokens\":[],\"temperature\":0.000,\"avg_logprob\":-0.250000,\"compression_ratio\":null,\"no_speech_prob\":null}]}");

    // The rest of OpenAI's verbose_json segment (seek, tokens, temperature, compression_ratio, no_speech_prob) is always present so a client
    // that reads those fields never meets a missing key. The request's temperature, a per-segment compression ratio from the injected
    // function (honest: computed from the segment text, not invented) and a raw usage object are added when the caller supplies them.
    {
        verbose_extras vx;
        vx.temperature       = 0.4;
        vx.compression_ratio = [](const std::string & s) { return s.size() > 5 ? 2.5 : 1.0; };
        vx.usage_json        = "{\"prompt_tokens\":7}";
        EQ(render(format::verbose_json, "Hello world. Ok.", 2.0, "en", true, -0.5, &vx),
           "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":2.000,\"text\":\"Hello world. Ok.\",\"timestamps\":\"interpolated\",\"segments\":["
           "{\"id\":0,\"seek\":0,\"start\":0.000,\"end\":1.600,\"text\":\"Hello world.\",\"tokens\":[],\"temperature\":0.400,"
           "\"avg_logprob\":-0.500000,\"compression_ratio\":2.5000,\"no_speech_prob\":null},"
           "{\"id\":1,\"seek\":0,\"start\":1.600,\"end\":2.000,\"text\":\"Ok.\",\"tokens\":[],\"temperature\":0.400,"
           "\"avg_logprob\":-0.500000,\"compression_ratio\":1.0000,\"no_speech_prob\":null}],\"usage\":{\"prompt_tokens\":7}}");
    }

    // no_speech_prob: the caller's first-position end-of-generation probability, written on every segment (one decode covers the audio);
    // 0 is a real value and is written, only a negative (unknown) stays null.
    {
        verbose_extras vx;
        vx.no_speech_prob = 0.125;
        EQ(render(format::verbose_json, "Hello world. Ok.", 2.0, "en", true, -0.5, &vx),
           "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":2.000,\"text\":\"Hello world. Ok.\",\"timestamps\":\"interpolated\",\"segments\":["
           "{\"id\":0,\"seek\":0,\"start\":0.000,\"end\":1.600,\"text\":\"Hello world.\",\"tokens\":[],\"temperature\":0.000,"
           "\"avg_logprob\":-0.500000,\"compression_ratio\":null,\"no_speech_prob\":0.125000},"
           "{\"id\":1,\"seek\":0,\"start\":1.600,\"end\":2.000,\"text\":\"Ok.\",\"tokens\":[],\"temperature\":0.000,"
           "\"avg_logprob\":-0.500000,\"compression_ratio\":null,\"no_speech_prob\":0.125000}]}");
        vx.no_speech_prob = 0.0;
        EQ(render(format::verbose_json, "Hi.", 1.0, "en", true, -0.25, &vx),
           "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":1.000,\"text\":\"Hi.\",\"timestamps\":\"interpolated\",\"segments\":["
           "{\"id\":0,\"seek\":0,\"start\":0.000,\"end\":1.000,\"text\":\"Hi.\",\"tokens\":[],\"temperature\":0.000,\"avg_logprob\":-0.250000,\"compression_ratio\":null,\"no_speech_prob\":0.000000}]}");
    }

    // speaker (diarize): called with each segment's own start/end, written as a string after no_speech_prob; with no callback
    // there is no "speaker" key at all (the blocks above pin that).
    {
        verbose_extras vx;
        std::string seen;
        vx.speaker = [&seen](double t0, double t1) {
            seen += fixed(t0, 3) + "-" + fixed(t1, 3) + ";";
            return t0 < 1.0 ? std::string("0") : std::string("1");
        };
        EQ(render(format::verbose_json, "Hello world. Ok.", 2.0, "en", false, 0.0, &vx),
           "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":2.000,\"text\":\"Hello world. Ok.\",\"timestamps\":\"interpolated\",\"segments\":["
           "{\"id\":0,\"seek\":0,\"start\":0.000,\"end\":1.600,\"text\":\"Hello world.\",\"tokens\":[],\"temperature\":0.000,"
           "\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null,\"speaker\":\"0\"},"
           "{\"id\":1,\"seek\":0,\"start\":1.600,\"end\":2.000,\"text\":\"Ok.\",\"tokens\":[],\"temperature\":0.000,"
           "\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null,\"speaker\":\"1\"}]}");
        EQ(seen, std::string("0.000-1.600;1.600-2.000;"));
    }

    // time_remap (--asr-vad-trim): the model heard 2 s, the upload was 120 s. Segment times are rewritten to the upload's, "duration" is
    // the upload's, the speaker callback sees the rewritten times, and the info object is the top-level "asr_vad". The rewrite here is
    // written by hand: what starts before 1.6 s is 10 s later in the upload, what starts at or after it is at 100 s + (t - 1.6); what
    // ends at or before 1.6 s ends 10 s later (the join belongs to the piece before it when it is an end, to the next when a start).
    {
        time_remap rm;
        rm.segment = [](double & t0, double & t1) {
            t0 = t0 < 1.6 ? t0 + 10.0 : 100.0 + (t0 - 1.6);
            t1 = t1 <= 1.6 ? t1 + 10.0 : 100.0 + (t1 - 1.6);
        };
        rm.duration_s = 120.0;
        rm.info_json  = "{\"trimmed\":true}";
        verbose_extras vx;
        std::string seen;
        vx.speaker = [&seen](double t0, double t1) {
            seen += fixed(t0, 3) + "-" + fixed(t1, 3) + ";";
            return std::string("0");
        };
        EQ(render(format::verbose_json, "Hello world. Ok.", 2.0, "en", false, 0.0, &vx, &rm),
           "{\"task\":\"transcribe\",\"language\":\"en\",\"duration\":120.000,\"text\":\"Hello world. Ok.\",\"timestamps\":\"interpolated\",\"segments\":["
           "{\"id\":0,\"seek\":0,\"start\":10.000,\"end\":11.600,\"text\":\"Hello world.\",\"tokens\":[],\"temperature\":0.000,"
           "\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null,\"speaker\":\"0\"},"
           "{\"id\":1,\"seek\":0,\"start\":100.000,\"end\":100.400,\"text\":\"Ok.\",\"tokens\":[],\"temperature\":0.000,"
           "\"avg_logprob\":null,\"compression_ratio\":null,\"no_speech_prob\":null,\"speaker\":\"0\"}],\"asr_vad\":{\"trimmed\":true}}");
        EQ(seen, std::string("10.000-11.600;100.000-100.400;"));
        EQ(render(format::srt, "Hello world. Ok.", 2.0, "en", false, 0.0, nullptr, &rm),
           "1\n00:00:10,000 --> 00:00:11,600\nHello world.\n\n"
           "2\n00:01:40,000 --> 00:01:40,400\nOk.\n\n");
        EQ(render(format::vtt, "Hello world. Ok.", 2.0, "en", false, 0.0, nullptr, &rm),
           "WEBVTT\n\n00:00:10.000 --> 00:00:11.600\nHello world.\n\n"
           "00:01:40.000 --> 00:01:40.400\nOk.\n\n");
        // the model's own markers are times on what it heard, so they are rewritten too
        EQ(render(format::srt, "<|0.50|> Hi.<|1.00|>", 2.0, "en", false, 0.0, nullptr, &rm),
           "1\n00:00:10,500 --> 00:00:11,000\nHi.\n\n");
        // text carries no time: unchanged
        EQ(render(format::text, "Hello world. Ok.", 2.0, "en", false, 0.0, nullptr, &rm), "Hello world. Ok.\n");
        // an empty time_remap changes nothing
        const time_remap none;
        EQ(render(format::verbose_json, "Hello world. Ok.", 2.0, "en", false, 0.0, nullptr, &none),
           render(format::verbose_json, "Hello world. Ok.", 2.0, "en"));
        EQ(render(format::srt, "Hello world. Ok.", 2.0, "en", false, 0.0, nullptr, &none), render(format::srt, "Hello world. Ok.", 2.0, "en"));
    }

    // wrap_opts (max_len / split_on_word, whisper.cpp whisper_wrap_segment semantics). One sentence of 27 code points over 2.7 s, so
    // one code point = 0.1 s. Expected cues written out by hand:
    //   max_len 10, by word: "one two" (7) | "three" (+" four" would be 11) | "four five" (10, not > 10) | "six"
    //     times at code points 0, 7, 13, 23, 27 -> 0.0, 0.7, 1.3, 2.3, 2.7
    //   max_len 10, by code point: "one two th" | "ree four f" | "ive six" at 0, 10, 20, 27 -> 0.0, 1.0, 2.0, 2.7
    {
        const std::string s = "one two three four five six";
        wrap_opts w;
        w.max_len = 10;
        EQ(render(format::srt, s, 2.7, "en", false, 0.0, nullptr, nullptr, &w),
           "1\n00:00:00,000 --> 00:00:00,700\none two\n\n"
           "2\n00:00:00,700 --> 00:00:01,300\nthree\n\n"
           "3\n00:00:01,300 --> 00:00:02,300\nfour five\n\n"
           "4\n00:00:02,300 --> 00:00:02,700\nsix\n\n");
        w.split_on_word = false;
        EQ(render(format::vtt, s, 2.7, "en", false, 0.0, nullptr, nullptr, &w),
           "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\none two th\n\n"
           "00:00:01.000 --> 00:00:02.000\nree four f\n\n"
           "00:00:02.000 --> 00:00:02.700\nive six\n\n");
        // a word longer than max_len stays whole on its own line (split_on_word never cuts a word)
        wrap_opts w5;
        w5.max_len = 5;
        const auto cues = wrap_segments({{0.0, 2.8, "supercalifragilistic is long"}}, w5);
        CHECK(cues.size() == 3);
        if (cues.size() == 3) {
            EQ(cues[0].text, std::string("supercalifragilistic"));
            EQ(cues[1].text, std::string("is"));
            EQ(cues[2].text, std::string("long"));
            CHECK(cues[0].t0 == 0.0 && cues[2].t1 == 2.8 && cues[0].t1 <= cues[1].t0 && cues[1].t1 <= cues[2].t0);
        }
        // every cue of a by-word wrap is at most max_len code points unless it is one word
        for (const auto & c : wrap_segments({{0.0, 9.0, "the quick brown fox jumps over the lazy dog again and again"}}, w)) {
            CHECK(n_codepoints(c.text) <= 10 || c.text.find(' ') == std::string::npos);
        }
        // UTF-8: code points, not bytes ("éé" is 2 code points, 4 bytes)
        wrap_opts w3;
        w3.max_len = 3;
        w3.split_on_word = false;
        const auto u = wrap_segments({{0.0, 1.0, "\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9"}}, w3);
        CHECK(u.size() == 2 && u[0].text == "\xc3\xa9\xc3\xa9\xc3\xa9" && u[1].text == "\xc3\xa9");
        // max_len 0 / no wrap_opts: unchanged
        const wrap_opts off;
        EQ(render(format::srt, s, 2.7, "en", false, 0.0, nullptr, nullptr, &off), render(format::srt, s, 2.7, "en"));
        // wrap runs BEFORE the remap: a cue boundary is a time on what the model heard, then mapped like any other
        time_remap rm;
        rm.segment = [](double & t0, double & t1) { t0 += 100.0; t1 += 100.0; };
        w.split_on_word = true;
        EQ(render(format::srt, s, 2.7, "en", false, 0.0, nullptr, &rm, &w),
           "1\n00:01:40,000 --> 00:01:40,700\none two\n\n"
           "2\n00:01:40,700 --> 00:01:41,300\nthree\n\n"
           "3\n00:01:41,300 --> 00:01:42,300\nfour five\n\n"
           "4\n00:01:42,300 --> 00:01:42,700\nsix\n\n");
    }

    // strip_wrapping_quotes: one balanced outer pair only (ec6520161 port)
    EQ(strip_wrapping_quotes("\"From the New York Times, July 21st, 1969.\""), std::string("From the New York Times, July 21st, 1969."));
    EQ(strip_wrapping_quotes("  'Hello there.'\n"), std::string("Hello there."));
    EQ(strip_wrapping_quotes("He said \"stop\" twice."), std::string("He said \"stop\" twice."));   // interior quotes: untouched
    EQ(strip_wrapping_quotes("\"He said \"stop\" twice.\""), std::string("\"He said \"stop\" twice.\"")); // quote inside: not a wrapper
    EQ(strip_wrapping_quotes("\"unbalanced"), std::string("\"unbalanced"));
    EQ(strip_wrapping_quotes("\"mixed'"), std::string("\"mixed'"));
    EQ(strip_wrapping_quotes("\""), std::string("\""));
    EQ(strip_wrapping_quotes(""), std::string(""));
    EQ(strip_wrapping_quotes("It's fine, isn't it"), std::string("It's fine, isn't it"));
    EQ(strip_wrapping_quotes("\"\""), std::string(""));

    // strip_language_tag: Qwen3-ASR "language <Name><asr_text>..." (expected values from the transformers model card example
    // and the two measured leaks on the row; written by hand)
    {
        language_tagged r = strip_language_tag("language English<asr_text>The New York Times from July twenty-first.");
        CHECK(r.tagged);
        EQ(r.language, std::string("English"));
        EQ(r.text, std::string("The New York Times from July twenty-first."));
        r = strip_language_tag("language None<asr_text>");                  // no speech
        CHECK(r.tagged);
        EQ(r.language, std::string(""));
        EQ(r.text, std::string(""));
        r = strip_language_tag("LANGUAGE NONE<asr_text> ");                 // case-insensitive, as the reference
        CHECK(r.tagged && r.language.empty() && r.text.empty());
        r = strip_language_tag("<asr_text>bare marker");                    // forced-language prefill: the tag was in the prompt
        CHECK(r.tagged);
        EQ(r.language, std::string(""));
        EQ(r.text, std::string("bare marker"));
        r = strip_language_tag("\n Chinese \n<asr_text> 你好");              // first non-empty line, no "language " word
        EQ(r.language, std::string("Chinese"));
        EQ(r.text, std::string("你好"));
        r = strip_language_tag("language English<asr_text>a <asr_text> b"); // split on the FIRST marker only
        EQ(r.text, std::string("a <asr_text> b"));
        r = strip_language_tag("Just speech, language English, no marker."); // gated on the marker: untouched
        CHECK(!r.tagged);
        EQ(r.text, std::string("Just speech, language English, no marker."));
        CHECK(r.language.empty());
        // composed as the route does: tag first, then the wrapper-quote rule
        EQ(strip_wrapping_quotes(strip_language_tag("language English<asr_text>\"Hello there.\"").text), std::string("Hello there."));
    }

    // language_tag_stream: the streaming filter (stream=true deltas, realtime socket). Pieces are fed as a tokenizer splits
    // them; the joined output must equal the final event's text, and no piece of the tag may go out as a delta.
    {
        auto run = [](const std::vector<std::string> & pieces, std::vector<std::string> * outs = nullptr) {
            language_tag_stream f;
            std::string all;
            for (const auto & p : pieces) {
                const std::string o = f.push(p);
                if (outs && !o.empty()) { outs->push_back(o); }
                all += o;
            }
            const std::string tail = f.flush();
            if (outs && !tail.empty()) { outs->push_back(tail); }
            return all + tail;
        };
        std::vector<std::string> outs;
        EQ(run({"language", " English", "<", "asr", "_text", ">", " The", " New", " York"}, &outs), std::string("The New York"));
        CHECK(outs.size() == 3);                           // nothing before the speech
        EQ(outs[0], std::string("The"));                   // leading space after the marker dropped, as the final event trims
        EQ(run({"language None", "<asr_text>"}), std::string(""));          // no speech: nothing at all
        EQ(run({"<asr", "_text>", "bare"}), std::string("bare"));            // forced-language prefill: bare marker
        EQ(run({"LANGUAGE", " Chinese<asr_text>", "  ", "\xe4\xbd\xa0"}), std::string("\xe4\xbd\xa0")); // case, whitespace-only piece
        outs.clear();
        EQ(run({"Hello", ",", " world"}, &outs), std::string("Hello, world")); // cannot open "language": released at once
        CHECK(outs.size() == 3);
        EQ(outs[0], std::string("Hello"));
        EQ(run({"Lang", "uage", " is", " fun", "."}), std::string("Language is fun."));    // looked like a tag, was speech
        EQ(run({"language English is spoken here"}), std::string("language English is spoken here")); // undecided -> flush
        EQ(run({"language English <b>"}), std::string("language English <b>")); // a '<' that is not the marker releases
        EQ(run({"a", " <asr_text> b"}), std::string("a <asr_text> b"));      // marker after released speech: pass-through
        EQ(run({"a <asr_text> b"}), strip_language_tag("a <asr_text> b").text); // one piece: first marker, as the final event
        // agrees with the final event on every case above that has a marker
        EQ(run({"language English<asr_text>The New York Times."}), strip_language_tag("language English<asr_text>The New York Times.").text);
        // reset() starts a new item (realtime socket: one filter per committed item)
        language_tag_stream f;
        EQ(f.push("language English<asr_text>one"), std::string("one"));
        EQ(f.push(" two"), std::string(" two"));
        f.reset();
        EQ(f.push("language"), std::string(""));
        EQ(f.push(" English<asr_text>three"), std::string("three"));
        // bound: a long letters-only opening is released at 64 bytes, never held for the whole stream
        language_tag_stream g;
        std::string got = g.push("language ");
        for (int i = 0; i < 70 && got.empty(); ++i) { got = g.push("a"); }
        CHECK(!got.empty());
    }

    // timestamp_granularities[]=word: the request values and the "words" array of verbose_json
    {
        using namespace asrwords;
        CHECK(!parse_granularities({}).word && parse_granularities({}).bad.empty());
        auto g = parse_granularities({"word"});
        CHECK(g.word && !g.segment && g.bad.empty());
        g = parse_granularities({"segment", "word"});
        CHECK(g.word && g.segment && g.bad.empty());
        g = parse_granularities({"segment"});
        CHECK(!g.word && g.segment);
        g = parse_granularities({"word", "phoneme", "other"});
        CHECK(g.word && g.bad == "phoneme");   // the first unknown value is reported
        g = parse_granularities({"Word"});
        CHECK(!g.word && g.bad == "Word");     // exact, like response_format

        verbose_extras vx;
        const std::string base = render(format::verbose_json, "Hello world.", 2.0, "en", false, 0.0, &vx);
        CHECK(base.find("\"words\"") == std::string::npos);   // field absent unless asked: the old bytes
        vx.words_on = true;
        vx.words.push_back({"Hello", 0.1, 0.4});
        vx.words.push_back({"world.", 0.45, 1.2345});
        const std::string with = render(format::verbose_json, "Hello world.", 2.0, "en", false, 0.0, &vx);
        CHECK(with.find(",\"words\":[{\"word\":\"Hello\",\"start\":0.100,\"end\":0.400},{\"word\":\"world.\",\"start\":0.450,\"end\":1.234}]") != std::string::npos
              || with.find(",\"words\":[{\"word\":\"Hello\",\"start\":0.100,\"end\":0.400},{\"word\":\"world.\",\"start\":0.450,\"end\":1.235}]") != std::string::npos);
        verbose_extras ve;
        ve.words_on = true;   // asked, nothing aligned (no speech): an empty list, present
        CHECK(render(format::verbose_json, "", 1.0, "", false, 0.0, &ve).find(",\"words\":[]") != std::string::npos);
        vx.words_on = false;
        CHECK(render(format::verbose_json, "Hello world.", 2.0, "en", false, 0.0, &vx) == base);   // words_on off -> identical bytes

        // --asr-vad-trim: word times from the aligner are on the trimmed signal; time_remap::word takes them back to the upload
        vx.words_on = true;
        time_remap rm;
        rm.word = [](double & t0, double & t1) { t0 += 100.0; t1 += 100.0; };
        rm.duration_s = 120.0;
        const std::string mapped = render(format::verbose_json, "Hello world.", 2.0, "en", false, 0.0, &vx, &rm);
        CHECK(mapped.find("{\"word\":\"Hello\",\"start\":100.100,\"end\":100.400}") != std::string::npos);
        CHECK(mapped.find("{\"word\":\"world.\",\"start\":100.450,") != std::string::npos);
        CHECK(vx.words[0].t0 == 0.1);   // the caller's words are not rewritten in place
        time_remap seg_only;            // no word callback: the words keep their times (old behaviour)
        seg_only.segment = [](double & t0, double & t1) { t0 += 100.0; t1 += 100.0; };
        CHECK(render(format::verbose_json, "Hello world.", 2.0, "en", false, 0.0, &vx, &seg_only).find("{\"word\":\"Hello\",\"start\":0.100,") != std::string::npos);
    }

    std::printf(fails ? "FAILED %d\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
