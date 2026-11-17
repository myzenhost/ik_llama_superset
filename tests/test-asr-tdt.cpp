// test-asr-tdt.cpp -- the Token-and-Duration Transducer decoder (examples/audio/asr-tdt.h) on hand-built models whose
// answer is known: the encoder frames carry a one-hot (token, duration) choice straight into the joint, so every
// step of the greedy loop is predictable. The real-model check is examples/audio/tests/test-asr-tdt-cli.py.
#include "asr-tdt.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static const char * W_ = "\xE2\x96\x81"; // word marker
enum { THE, CAT, S, DOT, SAT, NOSPEECH, NV_TOK };   // token ids; blank = NV_TOK
static const int BLANK = NV_TOK;
static const int ND = 5;                             // durations 0..4
static const int NV = NV_TOK + 1;                    // token logits incl. blank
static const int NENC = NV + ND;                     // encoder frame: one-hot token | one-hot duration
static const int J = NENC + 1;                       // joint hidden: the one-hots + one penalty unit
static const int H = 2;                              // prediction-network width

// feedback = false: the prediction network has no effect on the logits
// feedback = true : right after CAT is consumed, the penalty unit pushes CAT's logit far down
static asr_tdt_model make_model(bool feedback, int max_symbols) {
    asr_tdt_model m;
    m.n_vocab = NV_TOK; m.n_pred = H; m.n_joint = J; m.n_enc = NENC;
    m.max_symbols = max_symbols; m.subsampling = 8; m.frame_s = 0.01;
    m.durations = {0, 1, 2, 3, 4};
    m.vocab = {std::string(W_) + "the", std::string(W_) + "cat", "s", ".", std::string(W_) + "sat", "<|nospeech|>"};
    m.embd.assign((size_t) (m.n_vocab + 1) * H, 0.f);
    m.embd[(size_t) CAT * H + 0] = 3.f;                // only CAT has a non-zero input
    for (int l = 0; l < 2; ++l) {
        asr_tdt_lstm_layer L;
        L.n_in = H;
        L.w_ih.assign((size_t) 4 * H * H, 0.f);
        L.w_hh.assign((size_t) 4 * H * H, 0.f);
        L.b.assign((size_t) 4 * H, 0.f);
        for (int j = 0; j < H; ++j) {
            L.b[0 * H + j] =  20.f;                     // input gate open
            L.b[1 * H + j] = -20.f;                     // forget gate shut: the state is the last input only
            L.b[3 * H + j] =  20.f;                     // output gate open
            L.w_ih[(size_t) (2 * H + j) * H + j] = l == 0 ? 1.f : 3.f; // g = tanh(x)
        }
        m.lstm.push_back(L);
    }
    m.pred_w.assign((size_t) J * H, 0.f);
    m.pred_b.assign(J, 0.f);
    if (feedback) { m.pred_w[(size_t) (J - 1) * H + 0] = 10.f; }
    m.enc_w.assign((size_t) J * NENC, 0.f);
    m.enc_b.assign(J, 0.f);
    for (int i = 0; i < NENC; ++i) { m.enc_w[(size_t) i * NENC + i] = 1.f; }
    m.out_w.assign((size_t) m.n_out() * J, 0.f);
    m.out_b.assign(m.n_out(), 0.f);
    for (int i = 0; i < NENC; ++i) { m.out_w[(size_t) i * J + i] = 1.f; }
    m.out_w[(size_t) CAT * J + (J - 1)] = -10.f;      // only reached when the penalty unit is non-zero
    m.out_b[BLANK] = 0.5f;                             // blank beats a frame whose token one-hot is absent
    return m;
}

static void frame(std::vector<float> & e, int tok, int dur) {
    std::vector<float> f(NENC, 0.f);
    f[tok] = 1.f;
    f[NV + dur] = 1.f;
    e.insert(e.end(), f.begin(), f.end());
}

// f0 blank d0 | f1 the d2 | f2 sat (skipped) | f3 cat d0 | f4 s d1 | f5 . d3 | f6,f7 sat (skipped) | f8 blank d4 |
// f9..f11 sat (skipped) | f12 <|nospeech|> d1  -> 13 frames
static std::vector<float> stimulus() {
    std::vector<float> e;
    frame(e, BLANK, 0); frame(e, THE, 2); frame(e, SAT, 1); frame(e, CAT, 0); frame(e, S, 1); frame(e, DOT, 3);
    frame(e, SAT, 1); frame(e, SAT, 1); frame(e, BLANK, 4); frame(e, SAT, 1); frame(e, SAT, 1); frame(e, SAT, 1);
    frame(e, NOSPEECH, 1);
    return e;
}

static std::string ids(const std::vector<asr_tdt_token> & t) {
    std::string s;
    for (const auto & x : t) { char b[32]; snprintf(b, sizeof(b), "%d@%d/%d ", x.id, x.frame, x.duration); s += b; }
    return s;
}

static bool write_gguf(const asr_tdt_model & m, const char * path) {
    const size_t mem = 64 * ggml_tensor_overhead() + 4 * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "tdt.vocab_size", m.n_vocab);
    gguf_set_val_u32(g, "tdt.blank_id", m.n_vocab);
    gguf_set_val_u32(g, "tdt.pred_hidden", m.n_pred);
    gguf_set_val_u32(g, "tdt.pred_layers", (uint32_t) m.lstm.size());
    gguf_set_val_u32(g, "tdt.joint_hidden", m.n_joint);
    gguf_set_val_u32(g, "tdt.encoder_hidden", m.n_enc);
    gguf_set_val_u32(g, "tdt.max_symbols_per_step", m.max_symbols);
    gguf_set_val_u32(g, "tdt.subsampling_factor", m.subsampling);
    gguf_set_val_f32(g, "tdt.frame_seconds", (float) m.frame_s);
    std::vector<int32_t> d(m.durations.begin(), m.durations.end());
    gguf_set_arr_data(g, "tdt.durations", GGUF_TYPE_INT32, d.data(), d.size());
    std::vector<const char *> v;
    for (const auto & s : m.vocab) { v.push_back(s.c_str()); }
    gguf_set_arr_str(g, "tdt.vocab", v.data(), v.size());
    auto add = [&](const char * name, const std::vector<float> & data, int64_t ne0) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, (int64_t) data.size() / ne0);
        memcpy(t->data, data.data(), data.size() * sizeof(float));
        ggml_set_name(t, name);
        gguf_add_tensor(g, t);
    };
    add("tdt.pred.embd.weight", m.embd, m.n_pred);
    for (size_t l = 0; l < m.lstm.size(); ++l) {
        char a[64], b[64], c[64];
        snprintf(a, sizeof(a), "tdt.pred.lstm.%zu.w_ih", l);
        snprintf(b, sizeof(b), "tdt.pred.lstm.%zu.w_hh", l);
        snprintf(c, sizeof(c), "tdt.pred.lstm.%zu.b", l);
        add(a, m.lstm[l].w_ih, m.lstm[l].n_in);
        add(b, m.lstm[l].w_hh, m.n_pred);
        add(c, m.lstm[l].b, 4 * m.n_pred);
    }
    add("tdt.joint.pred.weight", m.pred_w, m.n_pred); add("tdt.joint.pred.bias", m.pred_b, m.n_joint);
    add("tdt.joint.enc.weight", m.enc_w, m.n_enc);    add("tdt.joint.enc.bias", m.enc_b, m.n_joint);
    add("tdt.joint.out.weight", m.out_w, m.n_joint);  add("tdt.joint.out.bias", m.out_b, m.n_out());
    const bool ok = gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

int main() {
    const std::vector<float> enc = stimulus();
    const int T = (int) enc.size() / NENC;
    std::string err;

    // ---- the loop without predictor feedback: skipping, blank-advance, the per-frame symbol cap ----
    {
        const asr_tdt_model m = make_model(false, 3);
        CHECK(asr_tdt_check(m, err));
        std::vector<asr_tdt_token> t;
        CHECK(asr_tdt_greedy(m, enc.data(), T, t));
        std::printf("no-feedback: %s\n", ids(t).c_str());
        // the@1 d2 (skips f2), cat@3 d0 three times (cap 3, then one frame on), s@4 d1, .@5 d3 (skips f6,f7),
        // blank@8 d4 (skips f9..f11), nospeech@12
        CHECK(ids(t) == "0@1/2 1@3/0 1@3/0 1@3/0 2@4/1 3@5/3 5@12/1 ");
        CHECK(asr_tdt_text(m, t) == "the cat cat cats.");
        // p of the@1: logits the 1, blank 0.5, the other 5 tokens 0
        const double p_the = std::exp(1.0) / (std::exp(1.0) + std::exp(0.5) + 5.0);
        CHECK(!t.empty() && std::fabs(t[0].p - p_the) < 1e-5);

        const auto w = asr_tdt_words(m, t, 10.0);
        CHECK(w.size() == 4);
        if (w.size() == 4) {
            const double fs = 0.08;
            CHECK(w[0].word == "the"   && std::fabs(w[0].start - (10 + 1 * fs)) < 1e-9 && std::fabs(w[0].end - (10 + 3 * fs)) < 1e-9);
            CHECK(w[1].word == "cat"   && std::fabs(w[1].start - (10 + 3 * fs)) < 1e-9 && std::fabs(w[1].end - (10 + 3 * fs)) < 1e-9);
            CHECK(w[3].word == "cats." && std::fabs(w[3].start - (10 + 3 * fs)) < 1e-9);
            // the "." spans f5..f8 by its duration, but punctuation collapses to the end of "s" (f5)
            CHECK(std::fabs(w[3].end - (10 + 5 * fs)) < 1e-9);
        }

        // a cap of 1 symbol per frame: one cat, then the next frame
        const asr_tdt_model m1 = make_model(false, 1);
        std::vector<asr_tdt_token> t1;
        CHECK(asr_tdt_greedy(m1, enc.data(), T, t1));
        CHECK(ids(t1) == "0@1/2 1@3/0 2@4/1 3@5/3 5@12/1 ");

        // empty input and a model that fails its shape check
        std::vector<asr_tdt_token> t0;
        CHECK(asr_tdt_greedy(m, enc.data(), 0, t0) && t0.empty());
        asr_tdt_model broken = m;
        broken.out_b.pop_back();
        CHECK(!asr_tdt_check(broken, err));
    }

    // ---- predictor feedback: after CAT is consumed the joint sees it and stops repeating it ----
    {
        const asr_tdt_model m = make_model(true, 3);
        std::vector<asr_tdt_token> t;
        CHECK(asr_tdt_greedy(m, enc.data(), T, t));
        std::printf("feedback:    %s\n", ids(t).c_str());
        // one cat on f3; the second look at f3 sees blank win (cat pushed down) -> blank d0 -> advance 1
        CHECK(ids(t) == "0@1/2 1@3/0 2@4/1 3@5/3 5@12/1 ");
        CHECK(asr_tdt_text(m, t) == "the cats.");
    }

    // ---- GGUF round trip: the loader reads back exactly the model that was written ----
    {
        const asr_tdt_model m = make_model(true, 3);
        const char * path = "test-asr-tdt-tmp.gguf";
        CHECK(write_gguf(m, path));
        CHECK(asr_tdt_is_tdt_file(path));
        asr_tdt_model r;
        const bool ok = asr_tdt_load(path, r, err);
        if (!ok) { std::printf("load: %s\n", err.c_str()); }
        CHECK(ok);
        CHECK(r.vocab == m.vocab && r.durations == m.durations && r.out_w == m.out_w && r.lstm.size() == 2 &&
              r.lstm[1].w_ih == m.lstm[1].w_ih && r.max_symbols == 3 && std::fabs(r.frame_s - 0.01) < 1e-9);
        std::vector<asr_tdt_token> a, b;
        CHECK(asr_tdt_greedy(m, enc.data(), T, a) && asr_tdt_greedy(r, enc.data(), T, b) && ids(a) == ids(b));
        std::remove(path);
        CHECK(!asr_tdt_is_tdt_file("no-such-file.gguf"));
    }

    // ---- pieces ----
    CHECK(asr_tdt_is_punct(",") && asr_tdt_is_punct(std::string(W_) + "?") && !asr_tdt_is_punct("..") && !asr_tdt_is_punct("a"));
    CHECK(asr_tdt_is_special("<unk>") && asr_tdt_is_special("<|en|>") && !asr_tdt_is_special("<") && !asr_tdt_is_special(std::string(W_) + "the"));

    std::printf("%s (%d failure(s))\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
