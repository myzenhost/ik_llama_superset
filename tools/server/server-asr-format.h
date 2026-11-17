// server-asr-format.h -- /v1/audio/transcriptions response_format: json | text | srt | vtt | verbose_json.
//
// Header-only, std only (unit-tested without a server or a model: tests_cpp/test_server_asr_format.cpp).
//
// SOURCE-READ (MIT): ggml-org/whisper.cpp examples/server/server.cpp:1010-1173 (response formats; text is the
// bare transcript + newline, srt "N\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n", vtt "WEBVTT\n\n" +
// "HH:MM:SS.mmm --> ..." cues, verbose_json task/language/duration/text/segments). Written independently.
//
// An LLM audio model has no per-token timestamps (whisper.cpp reads them from the decoder). Segment times come, in order:
//   (1) from timestamp markers the model wrote into its own text (whisper-style "<|0.00|> text <|2.50|>"): real, "timestamps":"model";
//   (2) otherwise INTERPOLATED: the transcript is split into sentences and each sentence gets a share of the audio duration proportional
//       to its length in code points. verbose_json says so ("timestamps": "interpolated"); they are subtitle-grade approximations,
//       not alignments. With an unknown duration (0) the subtitle formats refuse to invent times and use a single zero-length cue.
// Tier (1) and the per-segment avg_logprob were carried over from the e-serving agent's separate implementation of the same row
// (S69-A-PICK-TRANSCRIPTIONS-ENDPOINT-TEXT-SRT-VTT-VERBOSE-JSON-SEGMENTS-AND-STEREO-DIARIZE-2026-10-05-01) when it was merged into this one.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace asrfmt {

enum class format { json, text, srt, vtt, verbose_json };

inline bool parse(const std::string & s, format & out) {
    if (s.empty() || s == "json")  { out = format::json;         return true; }
    if (s == "text")               { out = format::text;         return true; }
    if (s == "srt")                { out = format::srt;          return true; }
    if (s == "vtt")                { out = format::vtt;          return true; }
    if (s == "verbose_json")       { out = format::verbose_json; return true; }
    return false;
}

inline const char * content_type(format f) {
    switch (f) {
        case format::json:
        case format::verbose_json: return "application/json; charset=utf-8";
        case format::vtt:          return "text/vtt; charset=utf-8";
        case format::srt:          return "application/x-subrip; charset=utf-8";   // as tools/server/README.md documents
        default:                   return "text/plain; charset=utf-8";   // text
    }
}

struct segment { double t0 = 0, t1 = 0; std::string text; };

inline size_t n_codepoints(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) { if ((c & 0xC0) != 0x80) { n++; } }
    return n;
}

inline std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\t' || s[a] == '\r')) { a++; }
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\n' || s[b-1] == '\t' || s[b-1] == '\r')) { b--; }
    return s.substr(a, b - a);
}

// Sentence ends: '.', '!', '?' followed by whitespace or the end of the text, or the CJK full stops
// (U+3002, U+FF01, U+FF1F) anywhere. "3.5" or "e.g." mid-word do not split.
inline std::vector<std::string> split_sentences(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < text.size(); i++) {
        cur += text[i];
        bool end = false;
        const char c = text[i];
        if ((c == '.' || c == '!' || c == '?') && (i + 1 == text.size() || text[i+1] == ' ' || text[i+1] == '\n')) { end = true; }
        if (i + 2 < text.size() + 0 && (unsigned char) text[i] == 0xE3 && (unsigned char) text[i+1] == 0x80 && (unsigned char) text[i+2] == 0x82) { cur += text[i+1]; cur += text[i+2]; i += 2; end = true; }
        if (i + 2 < text.size() + 0 && (unsigned char) text[i] == 0xEF && (unsigned char) text[i+1] == 0xBC && ((unsigned char) text[i+2] == 0x81 || (unsigned char) text[i+2] == 0x9F)) { cur += text[i+1]; cur += text[i+2]; i += 2; end = true; }
        if (end) {
            const std::string t = trim(cur);
            if (!t.empty()) { out.push_back(t); }
            cur.clear();
        }
    }
    const std::string t = trim(cur);
    if (!t.empty()) { out.push_back(t); }
    return out;
}

// Split the model's text on whisper-style timestamp markers  <|12.34|>text<|15.00|>  (a marker is "<|" number "|>", number >= 0) into timed
// segments. Text before the first marker starts at 0; a run after the last marker with no closing marker ends at the audio duration when it is
// known, else where it starts. Returns false (segs empty, plain = text) when the text carries no marker at all. `plain` is the text minus
// every marker. A "<|...|>" that is not a number (a special token such as <|endoftext|>) is ordinary text and is left alone here.
inline bool parse_markers(const std::string & text, double duration_s, std::vector<segment> & segs, std::string & plain) {
    segs.clear();
    plain.clear();
    struct tok { bool is_time; double t; std::string s; };
    std::vector<tok> toks;
    std::string cur;
    size_t i = 0;
    while (i < text.size()) {
        if (text.compare(i, 2, "<|") == 0) {
            const size_t close = text.find("|>", i + 2);
            if (close != std::string::npos) {
                const std::string inner = text.substr(i + 2, close - i - 2);
                char * e = nullptr;
                const double v = std::strtod(inner.c_str(), &e);
                if (!inner.empty() && e && *e == '\0' && v >= 0.0) {
                    if (!cur.empty()) { toks.push_back({false, 0.0, cur}); cur.clear(); }
                    toks.push_back({true, v, ""});
                    i = close + 2;
                    continue;
                }
            }
        }
        cur.push_back(text[i]);
        i++;
    }
    if (!cur.empty()) { toks.push_back({false, 0.0, cur}); }
    bool any_time = false;
    for (const auto & t : toks) {
        any_time = any_time || t.is_time;
        if (!t.is_time) { plain += t.s; }
    }
    if (!any_time) {
        plain = text;
        return false;
    }
    double last_time = 0.0;
    for (size_t k = 0; k < toks.size(); k++) {
        if (toks[k].is_time) { last_time = toks[k].t; continue; }
        segment g;
        g.text = trim(toks[k].s);
        if (g.text.empty()) { continue; }
        g.t0 = last_time;
        g.t1 = last_time;
        bool closed = false;
        for (size_t j = k + 1; j < toks.size(); j++) {
            if (toks[j].is_time) { g.t1 = toks[j].t; closed = true; break; }
        }
        if (!closed && duration_s > g.t0) { g.t1 = duration_s; }
        segs.push_back(std::move(g));
    }
    return !segs.empty();
}

inline std::vector<segment> make_segments(const std::string & text, double duration_s) {
    std::vector<segment> segs;
    const auto sents = split_sentences(text);
    size_t total = 0;
    for (const auto & s : sents) { total += n_codepoints(s); }
    if (total == 0) { return segs; }
    size_t acc = 0;
    for (const auto & s : sents) {
        segment g;
        g.text = s;
        g.t0 = duration_s > 0 ? duration_s * (double) acc / (double) total : 0.0;
        acc += n_codepoints(s);
        g.t1 = duration_s > 0 ? duration_s * (double) acc / (double) total : 0.0;
        segs.push_back(std::move(g));
    }
    return segs;
}

// Subtitle line wrapping: whisper.cpp's --max-len / --split-on-word (src/whisper.cpp whisper_wrap_segment; MIT, SOURCE-READ, no
// code copied). whisper.cpp walks a segment's tokens and closes the line BEFORE a token when the line would pass max_len UTF-8
// code points (never before the first one); with split_on_word it closes only before a token that starts with a space, so no
// word is cut. A segment here is text with times, so the units are words (each with the whitespace before it) or, with
// split_on_word off, single code points; a cue's time is interpolated by code points inside its segment, the same rule
// make_segments uses. A word longer than max_len stays whole on its own line (split_on_word never cuts a word).
//   max_len 0 = no wrapping (the default, and what every caller that does not ask gets).
struct wrap_opts {
    int  max_len       = 0;
    bool split_on_word = true;
};

// units of `text`: by_word -> runs of [whitespace][non-whitespace]; else one UTF-8 code point each
inline std::vector<std::string> wrap_units(const std::string & text, bool by_word) {
    std::vector<std::string> u;
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    for (size_t i = 0; i < text.size();) {
        if (!by_word) {
            size_t j = i + 1;
            while (j < text.size() && (((unsigned char) text[j]) & 0xC0) == 0x80) { j++; }
            u.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        size_t j = i;
        while (j < text.size() && is_ws((unsigned char) text[j])) { j++; }
        while (j < text.size() && !is_ws((unsigned char) text[j])) { j++; }
        u.push_back(text.substr(i, j - i));
        i = j;
    }
    return u;
}

inline std::vector<segment> wrap_segments(const std::vector<segment> & segs, const wrap_opts & w) {
    if (w.max_len <= 0) { return segs; }
    std::vector<segment> out;
    for (const auto & g : segs) {
        const auto units = wrap_units(g.text, w.split_on_word);
        size_t total = 0;
        for (const auto & x : units) { total += n_codepoints(x); }
        if (total == 0) { out.push_back(g); continue; }
        auto at = [&](size_t cp) { return g.t0 + (g.t1 - g.t0) * (double) cp / (double) total; };
        std::string line;
        size_t acc = 0, done = 0, line_start = 0;
        for (size_t i = 0; i < units.size(); i++) {
            const size_t cur = n_codepoints(units[i]);
            const bool word_start = !units[i].empty() && (units[i][0] == ' ' || units[i][0] == '\t' || units[i][0] == '\n');
            if (acc + cur > (size_t) w.max_len && i > 0 && (!w.split_on_word || word_start)) {
                const std::string t = trim(line);
                if (!t.empty()) { out.push_back({at(line_start), at(done), t}); }
                line.clear();
                acc = 0;
                line_start = done;
            }
            line += units[i];
            acc += cur;
            done += cur;
        }
        const std::string t = trim(line);
        if (!t.empty()) { out.push_back({at(line_start), g.t1, t}); }
    }
    return out;
}

inline std::string timestamp(double t, char ms_sep) {
    if (!(t > 0)) { t = 0; }
    const int64_t ms_total = (int64_t) std::llround(t * 1000.0);
    const int64_t h = ms_total / 3600000, m = (ms_total / 60000) % 60, s = (ms_total / 1000) % 60, ms = ms_total % 1000;
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld%c%03lld", (long long) h, (long long) m, (long long) s, ms_sep, (long long) ms);
    return buf;
}

inline std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; } else { o += (char) c; }
        }
    }
    return o;
}

inline std::string fixed(double v, int prec) { char b[48]; std::snprintf(b, sizeof(b), "%.*f", prec, v); return b; }

// What a caller may add to verbose_json beyond the transcript. Every member is optional.
//   temperature        the sampling temperature of the request (OpenAI writes it on every segment); 0 when the caller does not say
//   compression_ratio  per-segment len(text)/len(zlib(text)); called with the segment text. Left empty, the field is null: no value is invented.
//   usage_json         a raw JSON object, written as the top-level "usage" member (the plain json response carries the same object)
//   no_speech_prob     the end-of-generation probability at the first generated position (an LLM audio decoder has no no-speech head; a
//                      model that would end its reply at once is saying it heard nothing). One decode covers the whole audio, so it is written
//                      on every segment, like avg_logprob. Negative = not known, written as null.
//   speaker            stereo speaker label for a segment, called with its start and end in seconds ("0", "1" or "?", whisper.cpp
//                      server's --diarize field). Left empty, segments carry no "speaker" key at all: diarize is off unless asked.
struct verbose_extras {
    double                                      temperature = 0.0;
    std::function<double(const std::string &)>  compression_ratio;
    std::string                                 usage_json;
    double                                      no_speech_prob = -1.0;
    std::function<std::string(double, double)>  speaker;
    // timestamp_granularities[]=word: OpenAI's top-level "words" [{word,start,end}], written only when `words_on`. The times come from
    // the aligner (server-asr-words.h), never from interpolation; a request whose alignment failed is an error, not an empty list.
    bool                                        words_on = false;
    struct word { std::string text; double t0 = 0.0, t1 = 0.0; };
    std::vector<word>                           words;
};

// When the model was given a SHORTER signal than the client sent (--asr-vad-trim: the silence is cut out before the model runs), every time
// the renderer knows -- the model's own markers and the interpolated ones alike -- is a time on that shorter signal. Every member is optional.
//   segment     called once per segment with its [t0, t1] on the processed signal; rewrites them to times on the upload
//   word        called once per verbose_extras word with its [t0, t1] on the processed signal; rewrites them to times on the upload
//               (a word lies in one piece of the cut, so this is not the segment rule); absent = word times are written as given
//   duration_s  the length of the upload, written as verbose_json "duration" (> 0; else the processed length is kept)
//   info_json   a raw JSON object, written as the top-level "asr_vad" member of verbose_json
// The speaker label (verbose_extras::speaker) is asked for AFTER the rewrite, so it reads the upload's channels at the upload's times.
struct time_remap {
    std::function<void(double &, double &)>  segment;
    std::function<void(double &, double &)>  word;
    double                                   duration_s = 0.0;
    std::string                              info_json;
};

// Body for a non-json format. `text` is the model's transcript, `duration_s` the decoded audio length (0 = unknown).
// `has_lp` / `avg_logprob`: the mean log-probability of the generated tokens, one value for the whole reply (the server asks for token probabilities
// for verbose_json); it is written on every segment as "avg_logprob" and is null when not known.
//
// verbose_json carries OpenAI's full segment field set so a client that reads any of them never meets a missing key (the e-serving implementation of
// this row did the same, S69-A-PICK-TRANSCRIPTIONS-ENDPOINT-...): id, seek (0: this server never seeks within a window), start, end, text,
// tokens ([]: an LLM decoder's token ids cannot be attributed to a segment), temperature, avg_logprob, compression_ratio, no_speech_prob (the
// first-position end-of-generation proxy from verbose_extras; null when the caller has none).
inline std::string render(format f, const std::string & text, double duration_s, const std::string & language = "", bool has_lp = false, double avg_logprob = 0.0,
                          const verbose_extras * vx = nullptr, const time_remap * remap = nullptr, const wrap_opts * wrap = nullptr) {
    std::vector<segment> segs;
    std::string plain;
    const bool from_model = parse_markers(text, duration_s, segs, plain);   // tier 1: times the model wrote itself
    const std::string t = trim(from_model ? plain : text);
    if (f == format::text) { return t + "\n"; }
    if (!from_model) {
        segs = make_segments(t, duration_s);                                   // tier 2: interpolated
        if (segs.empty() && !t.empty()) { segs.push_back({0.0, duration_s, t}); }
    }
    if (wrap) {
        // on the times of the signal the model heard, BEFORE any remap: a cue interpolated inside a segment that spans a
        // removed silence would otherwise be placed inside that silence
        segs = wrap_segments(segs, *wrap);
    }
    if (remap) {
        // the times above are on the signal the model was given; from here on they are times on the upload
        if (remap->segment) {
            for (auto & g : segs) { remap->segment(g.t0, g.t1); }
        }
        if (remap->duration_s > 0.0) { duration_s = remap->duration_s; }
    }
    if (f == format::srt || f == format::vtt) {
        const char sep = f == format::srt ? ',' : '.';
        std::string o = f == format::vtt ? "WEBVTT\n\n" : "";
        for (size_t i = 0; i < segs.size(); i++) {
            if (f == format::srt) { o += std::to_string(i + 1) + "\n"; }
            o += timestamp(segs[i].t0, sep) + " --> " + timestamp(segs[i].t1, sep) + "\n" + segs[i].text + "\n\n";
        }
        return o;
    }
    // verbose_json (OpenAI shape, see above)
    const double temperature = vx ? vx->temperature : 0.0;
    std::string o = "{\"task\":\"transcribe\",\"language\":\"" + json_escape(language.empty() ? "unknown" : language) + "\",\"duration\":" +
                    fixed(duration_s, 3) + ",\"text\":\"" + json_escape(t) + "\",\"timestamps\":\"" + (from_model ? "model" : "interpolated") + "\",\"segments\":[";
    for (size_t i = 0; i < segs.size(); i++) {
        if (i) { o += ","; }
        const std::string ratio = (vx && vx->compression_ratio && !segs[i].text.empty()) ? fixed(vx->compression_ratio(segs[i].text), 4) : std::string("null");
        o += "{\"id\":" + std::to_string(i) + ",\"seek\":0,\"start\":" + fixed(segs[i].t0, 3) + ",\"end\":" + fixed(segs[i].t1, 3) +
             ",\"text\":\"" + json_escape(segs[i].text) + "\",\"tokens\":[],\"temperature\":" + fixed(temperature, 3) +
             ",\"avg_logprob\":" + (has_lp ? fixed(avg_logprob, 6) : std::string("null")) + ",\"compression_ratio\":" + ratio +
             ",\"no_speech_prob\":" + ((vx && vx->no_speech_prob >= 0.0) ? fixed(vx->no_speech_prob, 6) : std::string("null")) +
             ((vx && vx->speaker) ? ",\"speaker\":\"" + json_escape(vx->speaker(segs[i].t0, segs[i].t1)) + "\"" : std::string()) + "}";
    }
    o += "]";
    if (vx && vx->words_on) {
        o += ",\"words\":[";
        for (size_t i = 0; i < vx->words.size(); i++) {
            if (i) { o += ","; }
            double w0 = vx->words[i].t0, w1 = vx->words[i].t1;   // a copy: the caller's list stays on the processed signal
            if (remap && remap->word) { remap->word(w0, w1); }
            o += "{\"word\":\"" + json_escape(vx->words[i].text) + "\",\"start\":" + fixed(w0, 3) + ",\"end\":" + fixed(w1, 3) + "}";
        }
        o += "]";
    }
    if (vx && !vx->usage_json.empty()) { o += ",\"usage\":" + vx->usage_json; }
    if (remap && !remap->info_json.empty()) { o += ",\"asr_vad\":" + remap->info_json; }
    return o + "}";
}

// Remove ONE balanced pair of outer quotes from a transcript, and nothing else.
//
// An OpenAI client treats `text` AS the transcript, so a wrapper quote is ingested as a spoken word. Fork commit
// ec6520161 (2026-09-02) measured that Voxtral-Mini-3B still wraps some transcripts in quotes after the system-prompt and
// temperature fixes (test-2.mp3: "From the New York Times, July 21st, 1969 ..."), and that no prompt wording that kept
// full coverage removed it. That fix lived in examples/server, which the build does not compile; this is its port.
// Deliberately narrow: strips only when the trimmed text opens and closes with the SAME quote character (" or ') and
// that character occurs nowhere in between, so a transcript with real quoted speech is left untouched. No preamble
// stripping: a heuristic cannot tell a preamble from speech that really begins "The audio recording shows...".
inline std::string strip_wrapping_quotes(const std::string & in) {
    const size_t b = in.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return in;
    }
    const size_t e = in.find_last_not_of(" \t\r\n");
    const std::string t = in.substr(b, e - b + 1);
    if (t.size() < 2) {
        return in;
    }
    const char q = t.front();
    if ((q != '"' && q != '\'') || t.back() != q) {
        return in;
    }
    if (t.find(q, 1) != t.size() - 1) {
        return in; // the quote character appears inside: real quoted speech, not a wrapper
    }
    return t.substr(1, t.size() - 2);
}

// Qwen3-ASR's raw output is "language <Name><asr_text><transcript>" ("language None<asr_text>" for no speech). The route
// returned it verbatim, so every json / text / srt / vtt / verbose_json consumer read "language English<asr_text>" as
// spoken words (row S69-ASR-QWEN3ASR-LANGUAGE-TAG-LEAKS-INTO-TRANSCRIPT-2026-10-17-01, measured on Qwen3-ASR-0.6B Q8_0).
// SOURCE-READ (Apache-2.0): huggingface/transformers src/transformers/models/qwen3_asr/processing_qwen3_asr.py:107-144
// _parse_single_output -- split on the FIRST "<asr_text>"; prefix "language none" (any case) = no language; otherwise the
// first non-empty prefix line, minus a leading "language ", is the language. Written independently, no code copied.
// Gated on the MARKER, not on the model arch: text without "<asr_text>" comes back unchanged with tagged == false.
// Deviation from the reference: its repetition fixer and "assistant\n" split are not applied here (the chat parser already
// removed the turn header, and repetition is the --asr-quality retry loop's job).
struct language_tagged {
    bool        tagged   = false; // "<asr_text>" was present and removed
    std::string language;         // the model's language name ("English"), empty for "None" or a bare marker
    std::string text;             // the transcript after the marker (trimmed), or the input unchanged when !tagged
};

inline language_tagged strip_language_tag(const std::string & in) {
    static const std::string marker = "<asr_text>";
    language_tagged r;
    const size_t m = in.find(marker);
    if (m == std::string::npos) {
        r.text = in;
        return r;
    }
    auto trim = [](const std::string & s) {
        const size_t b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) {
            return std::string();
        }
        return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
    };
    auto lower = [](std::string s) {
        for (auto & c : s) { if (c >= 'A' && c <= 'Z') { c = (char) (c - 'A' + 'a'); } }
        return s;
    };
    r.tagged = true;
    r.text   = trim(in.substr(m + marker.size()));
    const std::string prefix = trim(in.substr(0, m));
    if (lower(prefix) == "language none") {
        return r;
    }
    size_t pos = 0;
    while (pos <= prefix.size()) {
        size_t nl = prefix.find('\n', pos);
        if (nl == std::string::npos) { nl = prefix.size(); }
        const std::string line = trim(prefix.substr(pos, nl - pos));
        pos = nl + 1;
        if (line.empty()) {
            continue;
        }
        if (lower(line).rfind("language ", 0) == 0) {
            r.language = trim(line.substr(9));
        } else {
            r.language = line;
        }
        break; // only the first non-empty line, as the reference does
    }
    if (lower(r.language) == "none") {
        r.language.clear();
    }
    return r;
}

// Streaming counterpart of strip_language_tag (row S69-ASR-QWEN3ASR-LANGUAGE-TAG-STREAMING-DELTAS-2026-11-08-01): a
// stream=true transcription and the realtime socket sent the tag as deltas ("language", " English", "<asr_text>") before
// the speech. push() holds pieces while the text so far can still be a tag -- optional whitespace, then a prefix of
// "language <Name>" (letters, spaces, '-'; bounded) or of the bare marker, optionally followed by a prefix of
// "<asr_text>" -- and releases them once the marker arrives (tag dropped, leading whitespace of the speech dropped, as the
// final event trims) or the text stops looking like a tag (held text released unchanged). flush() hands back what is
// still held when the stream ends undecided. The final event is unaffected: it strips the whole text.
// Not covered: the reference's "first non-empty line without the word language" form ("Chinese\n<asr_text>") is released
// as text by push() (the final event still strips it); Qwen3-ASR's measured output is "language <Name><asr_text>".
struct language_tag_stream {
    std::string held;
    bool        done   = false; // decided: pass everything through
    bool        lstrip = false; // marker seen, speech not started: drop leading whitespace

    static bool could_be_tag(const std::string & h) {
        static const std::string marker = "<asr_text>";
        static const std::string kw     = "language";
        const size_t b = h.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) {
            return h.size() <= 16;
        }
        const std::string s = h.substr(b);
        if (s.size() > 64) {
            return false;
        }
        const size_t lt   = s.find('<');
        const std::string head = s.substr(0, lt);
        if (lt != std::string::npos && marker.compare(0, s.size() - lt, s, lt, std::string::npos) != 0) {
            return false; // a '<' that is not the start of the marker
        }
        if (head.empty()) {
            return true; // bare marker prefix
        }
        std::string l = head;
        for (auto & c : l) { if (c >= 'A' && c <= 'Z') { c = (char) (c - 'A' + 'a'); } }
        if (l.size() <= kw.size()) {
            return kw.compare(0, l.size(), l) == 0 && (lt == std::string::npos || l == kw);
        }
        if (l.compare(0, kw.size(), kw) != 0 || (l[kw.size()] != ' ' && l[kw.size()] != '\t')) {
            return false;
        }
        for (size_t i = kw.size(); i < l.size(); ++i) {
            const char c = l[i];
            if (!((c >= 'a' && c <= 'z') || c == ' ' || c == '\t' || c == '-')) {
                return false;
            }
        }
        return true;
    }

    std::string push(const std::string & piece) {
        static const std::string marker = "<asr_text>";
        std::string out;
        if (done) {
            out = piece;
        } else {
            held += piece;
            const size_t m = held.find(marker);
            if (m != std::string::npos) {
                done   = true;
                lstrip = true;
                out    = held.substr(m + marker.size());
                held.clear();
            } else if (!could_be_tag(held)) {
                done = true;
                out.swap(held);
                return out;
            } else {
                return out;
            }
        }
        if (lstrip) {
            const size_t b = out.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) {
                return std::string();
            }
            lstrip = false;
            out    = out.substr(b);
        }
        return out;
    }

    std::string flush() {
        done = true;
        std::string out;
        out.swap(held);
        return out;
    }

    void reset() {
        held.clear();
        done   = false;
        lstrip = false;
    }
};

} // namespace asrfmt
