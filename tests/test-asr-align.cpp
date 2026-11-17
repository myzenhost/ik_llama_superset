// test-asr-align.cpp -- word timestamps from LLM attention over audio positions (examples/audio/asr-align.h).
// Synthetic attention with a known answer; the real-model check is examples/audio/tests/test-asr-words-cli.py.
#include "asr-align.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int main() {
    // ---- grouping ----
    {
        const auto g = asr_align_group_words({"Hello", ",", " wor", "ld", ".", " Bye"});
        CHECK(g.size() == 3);
        CHECK(g[0].first == 0 && g[0].second == 2);   // "Hello,"
        CHECK(g[1].first == 2 && g[1].second == 5);   // " world."
        CHECK(g[2].first == 5 && g[2].second == 6);   // " Bye"
        const auto g2 = asr_align_group_words({"\"", "Hi", " there"});
        CHECK(g2.size() == 2 && g2[0].second == 2);    // leading quote joins the first word
        const auto g3 = asr_align_group_words({" ", " ok"});
        CHECK(g3.size() == 1 && g3[0].first == 0 && g3[0].second == 2);
        CHECK(asr_align_is_punct(" ,."));
        CHECK(!asr_align_is_punct("\xc3\xa9"));        // UTF-8 letter
    }
    // ---- seconds per position ----
    {
        CHECK(near(asr_align_seconds_per_position(10.0, 375, 20.0, 375, 30.0), 0.08, 1e-12)); // whisper/Voxtral
        CHECK(near(asr_align_seconds_per_position(10.0, 130, 20.0, 260, 30.0), 10.0 / 130, 1e-12));
        CHECK(asr_align_seconds_per_position(10.0, 200, 20.0, 100, 30.0) <= 0.0);
        CHECK(asr_align_valid_frames(375, 3.5, 0.08) == 44);
        CHECK(asr_align_valid_frames(375, 60.0, 0.08) == 375);
    }
    // ---- alignment: 4 tokens + EOG row, 40 frames of 0.1 s, offset 5 s ----
    // token k attends to frames [band[k], band[k+1]); word 1 = tokens 0,1; word 2 = token 2 (+ ","), EOG = silence
    {
        const int T = 4, R = T + 1, H = 2, F = 40;
        const int band[R + 1] = {2, 8, 14, 22, 30, 40};
        asr_align_input in;
        in.tokens = {{1, " Hel", 0.9}, {2, "lo", 0.7}, {3, " there", 0.5}, {4, ",", 0.3}};
        in.n_heads = H; in.n_frames = F; in.has_eog_row = true;
        in.sec_per_frame = 0.1; in.t_offset = 5.0; in.t_end = 9.0; in.medfilt_width = 3;
        in.attn.assign((size_t) R * H * F, 0.001f);
        for (int r = 0; r < R; ++r) {
            for (int h = 0; h < H; ++h) {
                for (int f = band[r]; f < band[r + 1]; ++f) { in.attn[((size_t) r * H + h) * F + f] = 0.5f; }
            }
        }
        asr_align_result out;
        std::string err;
        CHECK(asr_align_words(in, out, &err));
        CHECK(out.tok_t0.size() == 4);
        CHECK(out.words.size() == 2);
        if (out.words.size() == 2) {
            std::printf("word0 '%s' %.2f-%.2f p=%.3f\n", out.words[0].word.c_str(), out.words[0].start, out.words[0].end, out.words[0].probability);
            std::printf("word1 '%s' %.2f-%.2f p=%.3f\n", out.words[1].word.c_str(), out.words[1].start, out.words[1].end, out.words[1].probability);
            CHECK(out.words[0].word == "Hello");
            CHECK(out.words[1].word == "there,");
            CHECK(near(out.words[0].start, 5.0, 0.21));   // first band starts at frame 2 -> DTW enters at 0..2
            CHECK(near(out.words[0].end,   6.4, 0.21));   // token 2 starts at frame 14
            CHECK(near(out.words[1].start, 6.4, 0.21));
            CHECK(near(out.words[1].end,   7.2, 0.21));   // "," is punctuation: the word ends with " there" (frame 22)
            CHECK(near(out.words[0].probability, 0.8, 1e-9));
            CHECK(near(out.words[1].probability, 0.4, 1e-9));
            CHECK(out.words[1].end <= 9.0 && out.words[0].start >= 5.0);
        }
        // monotone token starts
        for (size_t i = 1; i < out.tok_t0.size(); ++i) { CHECK(out.tok_t0[i] >= out.tok_t0[i - 1]); }

        // negative control: the SAME tokens with the attention bands moved ({2,4,6,30,34,40}: tokens 2 and 3 start at
        // frames 6 and 30). The times must follow the attention -- the expectation above must now FAIL, and the
        // new bands' answer must hold. (A first version swapped the rows of tokens 0 and 3; monotone DTW still
        // reproduced the same word times there, because rows 1 and 2 fix the boundaries and token 3 is punctuation,
        // so that control could not fail -- measured 2026-10-19, ut1_after.log.)
        asr_align_input moved = in;
        const int band2[R + 1] = {2, 4, 6, 30, 34, 40};
        moved.attn.assign((size_t) R * H * F, 0.001f);
        for (int r = 0; r < R; ++r) {
            for (int h = 0; h < H; ++h) {
                for (int f = band2[r]; f < band2[r + 1]; ++f) { moved.attn[((size_t) r * H + h) * F + f] = 0.5f; }
            }
        }
        asr_align_result ob;
        CHECK(asr_align_words(moved, ob, &err));
        CHECK(ob.words.size() == 2);
        if (ob.words.size() == 2) {
            std::printf("moved word0 '%s' %.2f-%.2f  word1 '%s' %.2f-%.2f\n", ob.words[0].word.c_str(), ob.words[0].start,
                        ob.words[0].end, ob.words[1].word.c_str(), ob.words[1].start, ob.words[1].end);
            CHECK(!(near(ob.words[0].end, 6.4, 0.21) && near(ob.words[1].end, 7.2, 0.21)));  // original answer rejected
            CHECK(near(ob.words[0].end,   5.6, 0.21));   // token 2 now starts at frame 6
            CHECK(near(ob.words[1].start, 5.6, 0.21));
            CHECK(near(ob.words[1].end,   8.0, 0.21));   // token 3 (",") starts at frame 30
        }
    }
    // ---- long word after a sentence end (a pause the DTW path handed to the next word) ----
    {
        auto mk = [](const char * s, double a, double b) { asr_word w; w.word = s; w.start = a; w.end = b; return w; };
        // durations 0.5 0.5 3.5 0.5 -> median 0.5, max_duration 1.0
        std::vector<asr_word> w = { mk("Hello", 0.0, 0.5), mk("world.", 0.5, 1.0), mk("The", 1.0, 4.5), mk("end", 4.5, 5.0) };
        asr_align_truncate_long_words(w);
        std::printf("trunc: '%s' %.2f-%.2f\n", w[2].word.c_str(), w[2].start, w[2].end);
        CHECK(near(w[2].start, 3.5, 1e-9) && near(w[2].end, 4.5, 1e-9));
        CHECK(near(w[1].end, 1.0, 1e-9) && near(w[3].start, 4.5, 1e-9));
        // negative control: the same pause INSIDE a sentence (no mark before it) is left alone, as upstream does
        std::vector<asr_word> v = { mk("Hello", 0.0, 0.5), mk("world", 0.5, 1.0), mk("the", 1.0, 4.5), mk("end", 4.5, 5.0) };
        asr_align_truncate_long_words(v);
        CHECK(near(v[2].start, 1.0, 1e-9));
        // CJK full stop counts as a sentence end; median cap 0.7 (durations 2,2,6,2 -> median 2 -> capped 0.7, max 1.4)
        std::vector<asr_word> c = { mk("a", 0.0, 2.0), mk("b\xE3\x80\x82", 2.0, 4.0), mk("c", 4.0, 10.0), mk("d", 10.0, 12.0) };
        asr_align_truncate_long_words(c);
        CHECK(near(c[2].start, 10.0 - 1.4, 1e-9));
        CHECK(near(c[3].start, 10.0, 1e-9));   // "d" is long too but follows no sentence end
    }
    // ---- speech weights from the samples ----
    {
        const int sr = 1000; const double spf = 0.1;
        std::vector<float> pcm(4 * sr, 0.0f);   // 1 s tone, 2 s digital silence, 1 s tone
        for (int i = 0; i < sr; ++i) { pcm[i] = 0.1f * (float) std::sin(i * 0.7); pcm[3 * sr + i] = 0.1f * (float) std::sin(i * 0.7); }
        const auto w = asr_align_speech_weights(pcm.data(), pcm.size(), sr, spf, 40);
        CHECK(w.size() == 40);
        if (w.size() == 40) {
            CHECK(w[0] == 1.0f && w[9] == 1.0f && w[30] == 1.0f && w[39] == 1.0f);
            CHECK(w[10] == 0.0f && w[25] == 0.0f && w[29] == 0.0f);
        }
        // continuous tone at two levels 12 dB apart: ref - floor < 20 dB, nothing is weighted down
        std::vector<float> c(2 * sr);
        for (int i = 0; i < 2 * sr; ++i) { c[i] = (i < sr ? 0.1f : 0.025f) * (float) std::sin(i * 0.7); }
        const auto wc = asr_align_speech_weights(c.data(), c.size(), sr, spf, 20);
        bool all1 = wc.size() == 20;
        for (float x : wc) { all1 = all1 && x == 1.0f; }
        CHECK(all1);
        CHECK(asr_align_speech_weights(nullptr, 10, sr, spf, 4).empty());
    }
    // ---- a long silence the attention cannot place (the Voxtral failure, 2026-10-19) ----
    // 3 tokens + EOG over 40 frames of 0.1 s: speech in frames 0..14 (token k on [5k, 5k+5)); frames 15..39 are
    // silence where row 1 is a little higher than the others and row 2 peaks at the very end -- a self-attention
    // pattern over silent audio. Plain DTW gives token 2 the end; the speech weight keeps it at frame 10.
    {
        const int T = 3, R = T + 1, H = 1, F = 40;
        asr_align_input in;
        in.tokens = {{1, " to", 0.9}, {2, " its", 0.9}, {3, " place.", 0.9}};
        in.n_heads = H; in.n_frames = F; in.has_eog_row = true;
        in.sec_per_frame = 0.1; in.t_offset = 0.0; in.t_end = 4.0; in.medfilt_width = 1;
        in.attn.assign((size_t) R * H * F, 0.001f);
        auto A = [&](int r, int f) -> float & { return in.attn[(size_t) r * F + f]; };
        for (int k = 0; k < T; ++k) { for (int f = 5 * k; f < 5 * k + 5; ++f) { A(k, f) = 0.5f; } }
        for (int f = 15; f < 40; ++f) { A(1, f) = 0.05f; }
        for (int f = 36; f < 40; ++f) { A(2, f) = 0.6f; }
        asr_align_result plain, weighted;
        std::string err;
        CHECK(asr_align_words(in, plain, &err));
        in.speech_w.assign(F, 0.0f);
        for (int f = 0; f < 15; ++f) { in.speech_w[f] = 1.0f; }
        CHECK(asr_align_words(in, weighted, &err));
        CHECK(plain.words.size() == 3 && weighted.words.size() == 3);
        if (plain.words.size() == 3 && weighted.words.size() == 3) {
            std::printf("silence plain:    its %.2f-%.2f  place. %.2f-%.2f\n", plain.words[1].start, plain.words[1].end,
                        plain.words[2].start, plain.words[2].end);
            std::printf("silence weighted: its %.2f-%.2f  place. %.2f-%.2f\n", weighted.words[1].start, weighted.words[1].end,
                        weighted.words[2].start, weighted.words[2].end);
            CHECK(plain.words[2].start > 3.0);                 // negative control: the failure is reproduced
            CHECK(near(weighted.words[1].start, 0.5, 0.11) && near(weighted.words[1].end, 1.0, 0.11));
            CHECK(near(weighted.words[2].start, 1.0, 0.11) && near(weighted.words[2].end, 1.5, 0.11));  // trimmed to speech
        }
        in.speech_w.assign(5, 1.0f);
        CHECK(!asr_align_words(in, weighted, &err) && err == "speech weight size mismatch");
    }
    // ---- invalid input ----
    {
        asr_align_input in; asr_align_result out; std::string err;
        CHECK(!asr_align_words(in, out, &err));
        in.tokens = {{1, " a", 1.0}}; in.n_heads = 1; in.n_frames = 4; in.sec_per_frame = 0.1;
        in.attn.assign(3, 0.0f);
        CHECK(!asr_align_words(in, out, &err) && err == "attention size mismatch");
    }
    std::printf("%s (%d failure(s))\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
