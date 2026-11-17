// asr-tdt.cpp -- see asr-tdt.h
#include "asr-tdt.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

// ---------------------------------------------------------------------------------------------------------------
// loading

bool asr_tdt_is_tdt_file(const std::string & path) {
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) { return false; }
    const bool yes = gguf_find_key(g, "tdt.vocab_size") >= 0;
    gguf_free(g);
    return yes;
}

bool asr_tdt_check(const asr_tdt_model & m, std::string & err) {
    auto bad = [&](const char * what) { err = std::string("tdt model: ") + what; return false; };
    if (m.n_vocab <= 0 || m.n_pred <= 0 || m.n_joint <= 0 || m.n_enc <= 0) { return bad("zero size"); }
    if (m.durations.empty())                                   { return bad("no durations"); }
    for (int d : m.durations) { if (d < 0) { return bad("negative duration"); } }
    if (m.max_symbols <= 0 || m.subsampling <= 0 || !(m.frame_s > 0)) { return bad("bad max_symbols / subsampling / frame_seconds"); }
    if ((int) m.vocab.size() != m.n_vocab)                     { return bad("vocab size != n_vocab"); }
    if (m.embd.size() != (size_t) (m.n_vocab + 1) * m.n_pred)  { return bad("embedding shape"); }
    if (m.lstm.empty())                                        { return bad("no LSTM layer"); }
    for (size_t l = 0; l < m.lstm.size(); ++l) {
        const auto & L = m.lstm[l];
        const int n_in = m.n_pred; // layer 0 reads the embedding, which is n_pred wide too
        if (L.n_in != n_in || L.w_ih.size() != (size_t) 4 * m.n_pred * n_in || L.w_hh.size() != (size_t) 4 * m.n_pred * m.n_pred ||
            L.b.size() != (size_t) 4 * m.n_pred) {
            return bad("LSTM layer shape");
        }
    }
    if (m.pred_w.size() != (size_t) m.n_joint * m.n_pred || m.pred_b.size() != (size_t) m.n_joint) { return bad("joint.pred shape"); }
    if (m.enc_w.size()  != (size_t) m.n_joint * m.n_enc  || m.enc_b.size()  != (size_t) m.n_joint) { return bad("joint.enc shape"); }
    if (m.out_w.size()  != (size_t) m.n_out() * m.n_joint || m.out_b.size() != (size_t) m.n_out()) { return bad("joint.out shape"); }
    return true;
}

namespace {

struct gguf_reader {
    gguf_context * g    = nullptr;
    ggml_context * meta = nullptr;
    std::ifstream  f;
    std::string    path;
    std::string    err;

    ~gguf_reader() {
        if (g)    { gguf_free(g); }
        if (meta) { ggml_free(meta); }
    }
    bool open(const std::string & p) {
        path = p;
        gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &meta };
        g = gguf_init_from_file(p.c_str(), ip);
        if (!g) { err = p + ": not a readable GGUF"; return false; }
        f.open(p, std::ios::binary);
        if (!f) { err = p + ": cannot open"; return false; }
        return true;
    }
    bool u32(const char * key, int & v) {
        const int64_t k = gguf_find_key(g, key);
        if (k < 0) { err = std::string("missing key ") + key; return false; }
        if (gguf_get_kv_type(g, k) != GGUF_TYPE_UINT32) { err = std::string("key ") + key + " is not u32"; return false; }
        v = (int) gguf_get_val_u32(g, k);
        return true;
    }
    bool f32(const char * key, double & v) {
        const int64_t k = gguf_find_key(g, key);
        if (k < 0) { err = std::string("missing key ") + key; return false; }
        if (gguf_get_kv_type(g, k) != GGUF_TYPE_FLOAT32) { err = std::string("key ") + key + " is not f32"; return false; }
        v = gguf_get_val_f32(g, k);
        return true;
    }
    bool ints(const char * key, std::vector<int> & v) {
        const int64_t k = gguf_find_key(g, key);
        if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY) { err = std::string("missing array ") + key; return false; }
        const size_t n = gguf_get_arr_n(g, k);
        const void * d = gguf_get_arr_data(g, k);
        v.resize(n);
        switch (gguf_get_arr_type(g, k)) {
            case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; ++i) { v[i] = ((const int32_t  *) d)[i]; } break;
            case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; ++i) { v[i] = (int) ((const uint32_t *) d)[i]; } break;
            case GGUF_TYPE_INT64:  for (size_t i = 0; i < n; ++i) { v[i] = (int) ((const int64_t  *) d)[i]; } break;
            default: err = std::string("array ") + key + " is not integer"; return false;
        }
        return true;
    }
    bool strs(const char * key, std::vector<std::string> & v) {
        const int64_t k = gguf_find_key(g, key);
        if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, k) != GGUF_TYPE_STRING) {
            err = std::string("missing string array ") + key;
            return false;
        }
        const size_t n = gguf_get_arr_n(g, k);
        v.resize(n);
        for (size_t i = 0; i < n; ++i) { v[i] = gguf_get_arr_str(g, k, i); }
        return true;
    }
    // F32 tensor with exactly ne0 x ne1 elements (ggml order: ne0 is the row length)
    bool tensor(const char * name, int64_t ne0, int64_t ne1, std::vector<float> & out) {
        const int64_t id = gguf_find_tensor(g, name);
        if (id < 0) { err = std::string("missing tensor ") + name; return false; }
        if (gguf_get_tensor_type(g, id) != GGML_TYPE_F32) { err = std::string("tensor ") + name + " is not F32"; return false; }
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        if (!t || t->ne[0] != ne0 || ggml_nelements(t) != ne0 * ne1) {
            char buf[256];
            snprintf(buf, sizeof(buf), "tensor %s has shape [%lld, %lld], expected [%lld, %lld]", name,
                     t ? (long long) t->ne[0] : -1LL, t ? (long long) (ggml_nelements(t) / std::max<int64_t>(1, t->ne[0])) : -1LL,
                     (long long) ne0, (long long) ne1);
            err = buf;
            return false;
        }
        out.resize((size_t) (ne0 * ne1));
        const uint64_t off = (uint64_t) gguf_get_data_offset(g) + (uint64_t) gguf_get_tensor_offset(g, id);
        f.seekg((std::streamoff) off, std::ios::beg);
        f.read((char *) out.data(), (std::streamsize) (out.size() * sizeof(float)));
        if (!f) { err = std::string("short read of tensor ") + name; return false; }
        return true;
    }
};

} // namespace

bool asr_tdt_load(const std::string & path, asr_tdt_model & m, std::string & err) {
    m = asr_tdt_model();
    gguf_reader r;
    if (!r.open(path)) { err = r.err; return false; }
    int n_layers = 0;
    bool ok = r.u32("tdt.vocab_size", m.n_vocab) && r.u32("tdt.pred_hidden", m.n_pred) && r.u32("tdt.pred_layers", n_layers) &&
              r.u32("tdt.joint_hidden", m.n_joint) && r.u32("tdt.encoder_hidden", m.n_enc) &&
              r.u32("tdt.max_symbols_per_step", m.max_symbols) && r.u32("tdt.subsampling_factor", m.subsampling) &&
              r.f32("tdt.frame_seconds", m.frame_s) && r.ints("tdt.durations", m.durations) && r.strs("tdt.vocab", m.vocab);
    if (ok) {
        int blank = -1;
        ok = r.u32("tdt.blank_id", blank);
        if (ok && blank != m.n_vocab) { r.err = "tdt.blank_id must equal tdt.vocab_size (blank is the last token)"; ok = false; }
    }
    if (ok) { ok = r.tensor("tdt.pred.embd.weight", m.n_pred, m.n_vocab + 1, m.embd); }
    for (int l = 0; ok && l < n_layers; ++l) {
        asr_tdt_lstm_layer L;
        L.n_in = m.n_pred;
        char n1[64], n2[64], n3[64];
        snprintf(n1, sizeof(n1), "tdt.pred.lstm.%d.w_ih", l);
        snprintf(n2, sizeof(n2), "tdt.pred.lstm.%d.w_hh", l);
        snprintf(n3, sizeof(n3), "tdt.pred.lstm.%d.b", l);
        ok = r.tensor(n1, L.n_in, 4 * m.n_pred, L.w_ih) && r.tensor(n2, m.n_pred, 4 * m.n_pred, L.w_hh) &&
             r.tensor(n3, 4 * m.n_pred, 1, L.b);
        if (ok) { m.lstm.push_back(std::move(L)); }
    }
    ok = ok && r.tensor("tdt.joint.pred.weight", m.n_pred, m.n_joint, m.pred_w) && r.tensor("tdt.joint.pred.bias", m.n_joint, 1, m.pred_b) &&
         r.tensor("tdt.joint.enc.weight", m.n_enc, m.n_joint, m.enc_w) && r.tensor("tdt.joint.enc.bias", m.n_joint, 1, m.enc_b) &&
         r.tensor("tdt.joint.out.weight", m.n_joint, m.n_vocab + 1 + (int) m.durations.size(), m.out_w) &&
         r.tensor("tdt.joint.out.bias", m.n_vocab + 1 + (int) m.durations.size(), 1, m.out_b);
    if (!ok) { err = path + ": " + r.err; return false; }
    return asr_tdt_check(m, err);
}

// ---------------------------------------------------------------------------------------------------------------
// decoding

namespace {

// y = W x + b, W row-major [rows x cols]
void matvec(const float * W, const float * b, const float * x, int rows, int cols, float * y) {
    for (int r = 0; r < rows; ++r) {
        const float * w = W + (size_t) r * cols;
        float s = 0.f;
        for (int c = 0; c < cols; ++c) { s += w[c] * x[c]; }
        y[r] = s + (b ? b[r] : 0.f);
    }
}

float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

struct pred_state {
    std::vector<std::vector<float>> h, c;
};

// one prediction-network step: consume `label`, advance the state, return the top layer's h in `out`
void pred_step(const asr_tdt_model & m, int label, pred_state & st, std::vector<float> & out) {
    const int H = m.n_pred;
    std::vector<float> x(m.embd.begin() + (size_t) label * H, m.embd.begin() + (size_t) (label + 1) * H);
    std::vector<float> gates(4 * (size_t) H), gh(4 * (size_t) H);
    for (size_t l = 0; l < m.lstm.size(); ++l) {
        const auto & L = m.lstm[l];
        matvec(L.w_ih.data(), L.b.data(), x.data(), 4 * H, L.n_in, gates.data());
        matvec(L.w_hh.data(), nullptr, st.h[l].data(), 4 * H, H, gh.data());
        for (int j = 0; j < 4 * H; ++j) { gates[j] += gh[j]; }
        auto & h = st.h[l];
        auto & c = st.c[l];
        for (int j = 0; j < H; ++j) {
            const float i_ = sigm(gates[j]);
            const float f_ = sigm(gates[H + j]);
            const float g_ = std::tanh(gates[2 * H + j]);
            const float o_ = sigm(gates[3 * H + j]);
            c[j] = f_ * c[j] + i_ * g_;
            h[j] = o_ * std::tanh(c[j]);
        }
        x = h;
    }
    out = x;
}

} // namespace

bool asr_tdt_greedy(const asr_tdt_model & m, const float * enc, int n_frames, std::vector<asr_tdt_token> & out) {
    out.clear();
    if (n_frames <= 0) { return true; }
    if (!enc) { return false; }
    const int H = m.n_pred, J = m.n_joint, nV = m.n_vocab + 1, nD = (int) m.durations.size();

    pred_state st;
    st.h.assign(m.lstm.size(), std::vector<float>(H, 0.f));
    st.c.assign(m.lstm.size(), std::vector<float>(H, 0.f));

    // the decode starts from the blank label (a zero input: blank-as-pad) and a zero state
    std::vector<float> g, gp(J), ep(J), hid(J), logits(m.n_out());
    pred_step(m, m.blank(), st, g);
    matvec(m.pred_w.data(), m.pred_b.data(), g.data(), J, H, gp.data());

    int t = 0, ep_t = -1, emitted = 0;
    while (t < n_frames) {
        if (ep_t != t) {
            matvec(m.enc_w.data(), m.enc_b.data(), enc + (size_t) t * m.n_enc, J, m.n_enc, ep.data());
            ep_t = t;
        }
        for (int j = 0; j < J; ++j) { hid[j] = std::max(0.f, ep[j] + gp[j]); }
        matvec(m.out_w.data(), m.out_b.data(), hid.data(), m.n_out(), J, logits.data());

        int best = 0;
        for (int i = 1; i < nV; ++i) { if (logits[i] > logits[best]) { best = i; } }
        int bd = 0;
        for (int i = 1; i < nD; ++i) { if (logits[nV + i] > logits[nV + bd]) { bd = i; } }
        int dur = m.durations[bd];

        if (best == m.blank()) {
            t += std::max(1, dur);
            emitted = 0;
            continue;
        }

        double z = 0.0;
        for (int i = 0; i < nV; ++i) { z += std::exp((double) logits[i] - logits[best]); }
        asr_tdt_token tok;
        tok.id = best;
        tok.frame = t;
        tok.duration = dur;
        tok.p = (float) (1.0 / z);
        out.push_back(tok);

        pred_step(m, best, st, g);
        matvec(m.pred_w.data(), m.pred_b.data(), g.data(), J, H, gp.data());

        if (dur > 0) {
            t += dur;
            emitted = 0;
            continue;
        }
        if (++emitted >= m.max_symbols) {
            t += 1;
            emitted = 0;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// text and words

static const char * k_word_marker = "\xE2\x96\x81"; // U+2581, SentencePiece's word start

static bool starts_word(const std::string & p) { return p.compare(0, 3, k_word_marker) == 0; }

static std::string strip_marker(const std::string & p) {
    std::string s;
    for (size_t i = 0; i < p.size();) {
        if (p.compare(i, 3, k_word_marker) == 0) { s += ' '; i += 3; } else { s += p[i++]; }
    }
    return s;
}

bool asr_tdt_is_punct(const std::string & piece) {
    std::string s = starts_word(piece) ? piece.substr(3) : piece;
    static const std::string punct = ".,!?;:'\"-()[]{}";
    return s.size() == 1 && punct.find(s[0]) != std::string::npos;
}

bool asr_tdt_is_special(const std::string & piece) {
    if (piece == "<unk>" || piece == "<pad>" || piece == "<s>" || piece == "</s>") { return true; }
    return piece.size() >= 4 && piece.compare(0, 2, "<|") == 0 && piece.compare(piece.size() - 2, 2, "|>") == 0;
}

std::string asr_tdt_text(const asr_tdt_model & m, const std::vector<asr_tdt_token> & toks) {
    std::string s;
    for (const auto & t : toks) {
        if (t.id < 0 || t.id >= m.n_vocab) { continue; }
        const std::string & p = m.vocab[t.id];
        if (asr_tdt_is_special(p)) { continue; }
        s += strip_marker(p);
    }
    size_t a = s.find_first_not_of(' ');
    size_t b = s.find_last_not_of(' ');
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

std::vector<asr_word> asr_tdt_words(const asr_tdt_model & m, const std::vector<asr_tdt_token> & toks, double t_offset) {
    std::vector<asr_word> words;
    const double fs = m.frame_seconds();
    double last_end = -1.0; // end of the last non-punctuation token
    double p_sum = 0.0;
    int    p_n   = 0;
    auto close_word = [&]() {
        if (!words.empty() && p_n > 0) { words.back().probability = p_sum / p_n; }
        p_sum = 0.0;
        p_n   = 0;
    };
    for (const auto & t : toks) {
        if (t.id < 0 || t.id >= m.n_vocab) { continue; }
        const std::string & p = m.vocab[t.id];
        if (asr_tdt_is_special(p)) { continue; }
        double t0 = t_offset + t.frame * fs;
        double t1 = t_offset + (t.frame + t.duration) * fs;
        if (asr_tdt_is_punct(p)) {
            if (last_end >= 0.0) { t0 = t1 = last_end; }
        } else {
            last_end = t1;
        }
        std::string text = strip_marker(p);
        const bool new_word = words.empty() || (starts_word(p) && !asr_tdt_is_punct(p));
        if (new_word) {
            close_word();
            size_t a = text.find_first_not_of(' ');
            asr_word w;
            w.word  = a == std::string::npos ? std::string() : text.substr(a);
            w.start = t0;
            w.end   = t1;
            words.push_back(w);
        } else {
            // a continuation piece, or punctuation (attached to the word before it even after a word marker)
            size_t a = text.find_first_not_of(' ');
            if (a != std::string::npos) { words.back().word += text.substr(a); }
            words.back().end = std::max(words.back().end, t1);
        }
        p_sum += t.p;
        ++p_n;
    }
    close_word();
    // a leading piece without a word marker may have produced an empty word
    words.erase(std::remove_if(words.begin(), words.end(), [](const asr_word & w) { return w.word.empty(); }), words.end());
    return words;
}
