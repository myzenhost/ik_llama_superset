#include "asr-silero.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

namespace {

float half_to_float(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    float v;
    if (e == 0)       v = std::ldexp((float) f, -24);
    else if (e == 31) v = f ? NAN : INFINITY;
    else              v = std::ldexp((float) (f | 0x400), (int) e - 25);
    return s ? -v : v;
}

struct reader {
    FILE * f;
    bool ok = true;
    template <typename T> T get() { T v{}; if (std::fread(&v, sizeof(T), 1, f) != 1) ok = false; return v; }
    bool bytes(void * p, size_t n) { if (n && std::fread(p, 1, n, f) != n) ok = false; return ok; }
};

} // namespace

bool asr_silero_load(const std::string & path, asr_silero_model & m, std::string * err) {
    auto fail = [&](const std::string & s) { if (err) *err = s; return false; };
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open " + path);
    struct closer { FILE * f; ~closer() { std::fclose(f); } } c{f};
    reader r{f};

    if (r.get<uint32_t>() != 0x67676d6cu) return fail("bad magic (not a ggml file)");
    const int32_t tlen = r.get<int32_t>();
    if (!r.ok || tlen < 0 || tlen > 64) return fail("bad model type length");
    m.type.assign((size_t) tlen, '\0');
    r.bytes(&m.type[0], (size_t) tlen);
    const int32_t v0 = r.get<int32_t>(), v1 = r.get<int32_t>(), v2 = r.get<int32_t>();
    m.version = std::to_string(v0) + "." + std::to_string(v1) + "." + std::to_string(v2);
    m.n_window  = r.get<int32_t>();
    m.n_context = r.get<int32_t>();
    const int32_t n_enc = r.get<int32_t>();
    if (!r.ok) return fail("truncated header");
    if (n_enc != 4)            return fail("expected 4 encoder layers, got " + std::to_string(n_enc));
    if (m.n_window != 512)     return fail("only 512-sample windows (16 kHz) are supported, got " + std::to_string(m.n_window));
    for (int i = 0; i < 4; i++) {
        m.enc_ic[i] = r.get<int32_t>(); m.enc_oc[i] = r.get<int32_t>(); m.enc_k[i] = r.get<int32_t>();
    }
    const int32_t lstm_in = r.get<int32_t>(), lstm_h = r.get<int32_t>(), fin_in = r.get<int32_t>(), fin_out = r.get<int32_t>();
    if (!r.ok) return fail("truncated header");
    if (lstm_in != 128 || lstm_h != 128 || fin_in != 128 || fin_out != 1) return fail("unexpected LSTM / final conv sizes");
    const int32_t exp_ic[4] = {129, 128, 64, 64}, exp_oc[4] = {128, 64, 64, 128};
    for (int i = 0; i < 4; i++) {
        if (m.enc_ic[i] != exp_ic[i] || m.enc_oc[i] != exp_oc[i] || m.enc_k[i] != 3) return fail("unexpected encoder shape");
    }

    std::map<std::string, std::vector<float>> t;
    while (true) {
        int32_t n_dims;
        if (std::fread(&n_dims, 4, 1, f) != 1) break; // clean EOF between tensors
        const int32_t len = r.get<int32_t>(), ttype = r.get<int32_t>();
        if (!r.ok || n_dims < 0 || n_dims > 4 || len <= 0 || len > 256 || (ttype != 0 && ttype != 1)) return fail("bad tensor header");
        int64_t ne = 1;
        for (int i = 0; i < n_dims; i++) { const int32_t d = r.get<int32_t>(); if (d <= 0 || d > (1 << 20)) return fail("bad tensor dim"); ne *= d; }
        std::string name((size_t) len, '\0');
        r.bytes(&name[0], (size_t) len);
        if (!r.ok) return fail("truncated tensor header");
        std::vector<float> v((size_t) ne);
        if (ttype == 0) {
            r.bytes(v.data(), v.size() * 4);
        } else {
            std::vector<uint16_t> h((size_t) ne);
            r.bytes(h.data(), h.size() * 2);
            for (size_t i = 0; i < h.size(); i++) v[i] = half_to_float(h[i]);
        }
        if (!r.ok) return fail("truncated tensor data for " + name);
        t[name] = std::move(v);
    }

    auto need = [&](const char * name, size_t n, std::vector<float> & dst) {
        auto it = t.find(name);
        if (it == t.end()) return fail(std::string("missing tensor ") + name);
        if (it->second.size() != n) return fail(std::string("wrong size for tensor ") + name);
        dst = std::move(it->second);
        return true;
    };
    if (!need("_model.stft.forward_basis_buffer", 258 * 256, m.stft)) return false;
    for (int i = 0; i < 4; i++) {
        const std::string b = "_model.encoder." + std::to_string(i) + ".reparam_conv.";
        if (!need((b + "weight").c_str(), (size_t) m.enc_oc[i] * m.enc_ic[i] * 3, m.enc_w[i])) return false;
        if (!need((b + "bias").c_str(), (size_t) m.enc_oc[i], m.enc_b[i])) return false;
    }
    std::vector<float> ob;
    if (!need("_model.decoder.rnn.weight_ih", 512 * 128, m.lstm_wih) || !need("_model.decoder.rnn.weight_hh", 512 * 128, m.lstm_whh) ||
        !need("_model.decoder.rnn.bias_ih", 512, m.lstm_bih) || !need("_model.decoder.rnn.bias_hh", 512, m.lstm_bhh) ||
        !need("_model.decoder.decoder.2.weight", 128, m.out_w) || !need("_model.decoder.decoder.2.bias", 1, ob)) return false;
    m.out_b = ob[0];
    return true;
}

namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y[oc][t] = b[oc] + sum_ic sum_k w[oc][ic][k] * x[ic][t*stride + k - pad], zero outside; ReLU.
void conv_relu(const std::vector<float> & w, const std::vector<float> & b, int ic, int oc, int K, int stride, int pad,
               const std::vector<float> & x, int n_in, std::vector<float> & y, int & n_out) {
    n_out = (n_in + 2 * pad - K) / stride + 1;
    y.assign((size_t) oc * n_out, 0.0f);
    for (int o = 0; o < oc; o++) {
        for (int t = 0; t < n_out; t++) {
            double s = b[o];
            for (int i = 0; i < ic; i++) {
                for (int k = 0; k < K; k++) {
                    const int p = t * stride + k - pad;
                    if (p < 0 || p >= n_in) continue;
                    s += (double) w[((size_t) o * ic + i) * K + k] * x[(size_t) i * n_in + p];
                }
            }
            y[(size_t) o * n_out + t] = s > 0.0 ? (float) s : 0.0f;
        }
    }
}

// One 512-sample window -> one probability; h, c are the LSTM state, updated in place. left_ctx: the 64 samples to the
// left of the window (previous window's tail), or nullptr for a reflect pad of the window itself.
float silero_window_prob(const asr_silero_model & m, const float * win, const float * left_ctx,
                         std::vector<float> & h, std::vector<float> & c) {
    const int W = m.n_window, P = 64;
    std::vector<float> pad(W + 2 * P);
    {
        // left: carried context, or reflect pad (edge sample not repeated); right: always reflect pad
        if (left_ctx) std::copy(left_ctx, left_ctx + P, pad.begin());
        else          for (int i = 0; i < P; i++) pad[i] = win[P - i];
        std::copy(win, win + W, pad.begin() + P);
        for (int j = 0; j < P; j++) pad[P + W + j] = win[W - 2 - j];

        // STFT: 258 filters, k=256, stride 128 -> 4 frames; magnitude of (re = first 129, im = last 129)
        const int n_fr = (W + 2 * P - 256) / 128 + 1; // 4
        std::vector<float> mag((size_t) 129 * n_fr);
        for (int fb = 0; fb < 129; fb++) {
            for (int t = 0; t < n_fr; t++) {
                double re = 0, im = 0;
                for (int k = 0; k < 256; k++) {
                    const float x = pad[(size_t) t * 128 + k];
                    re += (double) m.stft[(size_t) fb * 256 + k] * x;
                    im += (double) m.stft[(size_t) (fb + 129) * 256 + k] * x;
                }
                mag[(size_t) fb * n_fr + t] = (float) std::sqrt(re * re + im * im);
            }
        }

        std::vector<float> a = mag, b;
        int n = n_fr;
        const int strides[4] = {1, 2, 2, 1};
        for (int l = 0; l < 4; l++) {
            int n_out;
            conv_relu(m.enc_w[l], m.enc_b[l], m.enc_ic[l], m.enc_oc[l], 3, strides[l], 1, a, n, b, n_out);
            a.swap(b);
            n = n_out;
        }
        // a is [128][n]; take time index 0 (pytorch [:, :, 0])
        std::vector<float> x(128);
        for (int i = 0; i < 128; i++) x[i] = a[(size_t) i * n];

        // one LSTM step, gate order i, f, g, o
        std::vector<float> g(512);
        for (int o = 0; o < 512; o++) {
            double s = (double) m.lstm_bih[o] + m.lstm_bhh[o];
            for (int i = 0; i < 128; i++) s += (double) m.lstm_wih[(size_t) o * 128 + i] * x[i] + (double) m.lstm_whh[(size_t) o * 128 + i] * h[i];
            g[o] = (float) s;
        }
        for (int i = 0; i < 128; i++) {
            const float ig = sigmoidf(g[i]), fg = sigmoidf(g[128 + i]), gg = std::tanh(g[256 + i]), og = sigmoidf(g[384 + i]);
            c[i] = fg * c[i] + ig * gg;
            h[i] = og * std::tanh(c[i]);
        }
        double s = m.out_b;
        for (int i = 0; i < 128; i++) s += (double) m.out_w[i] * (h[i] > 0.0f ? h[i] : 0.0f);
        return sigmoidf((float) s);
    }
}

void stream_window(const asr_silero_model & m, asr_silero_stream & s, const float * win, std::vector<float> & probs) {
    const int W = m.n_window, P = 64;
    if (s.h.size() != 128) asr_silero_stream_reset(s);
    probs.push_back(silero_window_prob(m, win, s.left_context ? s.ctx.data() : nullptr, s.h, s.c));
    std::copy(win + W - P, win + W, s.ctx.begin());
}

} // namespace

void asr_silero_stream_reset(asr_silero_stream & s) {
    s.h.assign(128, 0.0f);
    s.c.assign(128, 0.0f);
    s.ctx.assign(64, 0.0f);
    s.pending.clear();
}

void asr_silero_stream_feed(const asr_silero_model & m, asr_silero_stream & s, const float * samples, size_t n_samples,
                            std::vector<float> & probs) {
    const size_t W = (size_t) m.n_window;
    if (s.h.size() != 128) asr_silero_stream_reset(s);
    size_t pos = 0;
    if (!s.pending.empty()) {                      // finish the window a previous call left open
        const size_t take = std::min(W - s.pending.size(), n_samples);
        s.pending.reserve(W);
        for (size_t i = 0; i < take; i++) s.pending.push_back(samples[i]);
        pos = take;
        if (s.pending.size() < W) return;
        stream_window(m, s, s.pending.data(), probs);
        s.pending.clear();
    }
    for (; pos + W <= n_samples; pos += W) stream_window(m, s, samples + pos, probs);
    s.pending.assign(samples + pos, samples + n_samples);
}

void asr_silero_stream_flush(const asr_silero_model & m, asr_silero_stream & s, std::vector<float> & probs) {
    if (s.pending.empty()) return;
    std::vector<float> win((size_t) m.n_window, 0.0f);
    std::copy(s.pending.begin(), s.pending.end(), win.begin());
    s.pending.clear();
    stream_window(m, s, win.data(), probs);
}

std::vector<float> asr_silero_probs(const asr_silero_model & m, const float * samples, size_t n_samples) {
    asr_silero_stream s;
    asr_silero_stream_reset(s);
    std::vector<float> probs;
    probs.reserve((n_samples + (size_t) m.n_window - 1) / (size_t) m.n_window);
    asr_silero_stream_feed(m, s, samples, n_samples, probs);
    asr_silero_stream_flush(m, s, probs);
    return probs;
}
