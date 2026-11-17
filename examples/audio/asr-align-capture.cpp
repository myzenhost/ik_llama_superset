// asr-align-capture.cpp -- see asr-align-capture.h. Written independently; no code copied.
#include "asr-align-capture.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int asr_attn_layer_of(const char * name) {
    if (!name) {
        return -1;
    }
    const char * dash = strrchr(name, '-');
    if (!dash || dash[1] < '0' || dash[1] > '9') {
        return -1;
    }
    return std::atoi(dash + 1);
}

bool asr_attn_capture::arm(int64_t a0, int64_t a1) {
    armed    = false;
    all_rows = false;
    err.clear();
    if (a1 <= a0 || n_head <= 0 || n_sel() <= 0) {
        return false;
    }
    kv0 = a0;
    kv1 = a1;
    cur.assign((size_t) n_sel() * (size_t) n_head * (size_t) n_audio(), 0.0f);
    got.assign((size_t) n_sel(), 0);
    all.assign((size_t) n_sel(), std::vector<float>());
    armed = true;
    return true;
}

bool asr_attn_capture::take_row0_and_begin_all_rows(std::vector<float> & row0) {
    if (!armed) {
        return false;
    }
    if (err.empty()) {
        for (uint8_t g : got) {
            if (!g) {
                err = "no attention row was captured for a selected layer (flash attention on, or a fused path?)";
                break;
            }
        }
    }
    if (!err.empty()) {
        return false;
    }
    row0 = cur;
    all_rows = true;
    for (auto & v : all) {
        v.clear();
    }
    return true;
}

bool asr_attn_capture::take_all_rows(int n_tok, std::vector<float> & tok_rows) {
    if (!armed || !all_rows || n_tok < 1) {
        return false;
    }
    armed    = false;
    all_rows = false;
    if (!err.empty()) {
        return false;
    }
    const size_t n_a = (size_t) n_audio(), nh = (size_t) n_head, per_row = nh * n_a;
    for (int s = 0; s < n_sel(); ++s) {
        if (all[(size_t) s].size() != (size_t) n_tok * per_row) {
            char b[160];
            snprintf(b, sizeof(b), "layer %d delivered %zu attention rows, expected %d", l0 + s,
                     per_row ? all[(size_t) s].size() / per_row : 0, n_tok);
            err = b;
            return false;
        }
    }
    tok_rows.assign((size_t) n_tok * (size_t) n_sel() * per_row, 0.0f);
    for (int r = 0; r < n_tok; ++r) {
        for (int s = 0; s < n_sel(); ++s) {
            const float * src = all[(size_t) s].data() + (size_t) r * per_row;
            float *       dst = tok_rows.data() + ((size_t) r * (size_t) n_sel() + (size_t) s) * per_row;
            std::copy(src, src + per_row, dst);
        }
    }
    for (auto & v : all) {
        v.clear();
        v.shrink_to_fit();
    }
    return true;
}

bool asr_attn_capture::on_softmax(int il, bool is_f32, const char * type_name, const int64_t ne[4], const size_t nb[4],
                                  const reader_fn & read) {
    if (!armed || il < l0 || il > l1) {
        return true;
    }
    if (!is_f32) {
        if (err.empty()) { err = std::string("kq softmax is not F32 (") + (type_name ? type_name : "?") + ")"; }
        return true;
    }
    const int64_t n_kv = ne[0], n_q = ne[1], n_h = ne[2] * ne[3];
    if (n_kv < kv1) {
        return true;   // a batch evaluated before every audio cell is in the cache: no row of interest yet
    }
    if (n_q < 1 || n_h != n_head || ne[3] != 1) {
        if (err.empty()) {
            char b[160];
            snprintf(b, sizeof(b), "kq softmax shape mismatch [%lld,%lld,%lld,%lld] (audio cells %lld..%lld, n_head %d)",
                     (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], (long long) kv0, (long long) kv1, n_head);
            err = b;
        }
        return true;
    }
    const int     s   = il - l0;
    const size_t  n_a = (size_t) n_audio();
    if (all_rows) {
        std::vector<float> & dst = all[(size_t) s];
        const size_t base = dst.size();
        dst.resize(base + (size_t) n_q * (size_t) n_head * n_a);
        for (int64_t q = 0; q < n_q; ++q) {
            for (int64_t h = 0; h < n_h; ++h) {
                const size_t off = (size_t) h * nb[2] + (size_t) q * nb[1] + (size_t) kv0 * nb[0];
                read(dst.data() + base + ((size_t) q * (size_t) n_head + (size_t) h) * n_a, off, n_a * sizeof(float));
            }
        }
    } else {
        for (int64_t h = 0; h < n_h; ++h) {
            const size_t off = (size_t) h * nb[2] + (size_t) (n_q - 1) * nb[1] + (size_t) kv0 * nb[0];
            read(cur.data() + ((size_t) s * (size_t) n_head + (size_t) h) * n_a, off, n_a * sizeof(float));
        }
        got[(size_t) s] = 1;
    }
    return true;
}

// int, not bool: the fork's eval callback is three-valued (0 = skip, 1 = wanted + synchronize first, 2 = wanted without).
int asr_attn_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (strncmp(t->name, "kq_soft_max", 11) != 0) {
        return ask ? 0 : 1;
    }
    auto * c = (asr_attn_capture *) user_data;
    const int il = asr_attn_layer_of(t->name);
    const bool sel = c && c->armed && il >= c->l0 && il <= c->l1;
    if (ask) {
        return sel ? 1 : 0;
    }
    if (!sel) {
        return 1;
    }
    const int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    const size_t  nb[4] = { t->nb[0], t->nb[1], t->nb[2], t->nb[3] };
    c->on_softmax(il, t->type == GGML_TYPE_F32, ggml_type_name(t->type), ne, nb,
                  [t](void * dst, size_t off, size_t n) { ggml_backend_tensor_get(t, dst, off, n); });
    return 1;
}

std::vector<float> asr_attn_assemble(const std::vector<float> & row0, const std::vector<float> & tok_rows,
                                     int n_tok, bool has_eog_row, int n_hd, int n_pos, int n_frames) {
    std::vector<float> out;
    if (n_tok < 1 || n_hd < 1 || n_frames < 1 || n_frames > n_pos ||
        row0.size() != (size_t) n_hd * (size_t) n_pos || tok_rows.size() != (size_t) n_tok * (size_t) n_hd * (size_t) n_pos) {
        return out;
    }
    const int n_rows = has_eog_row ? n_tok + 1 : n_tok;
    out.resize((size_t) n_rows * (size_t) n_hd * (size_t) n_frames);
    for (int r = 0; r < n_rows; ++r) {
        const float * base = r == 0 ? row0.data() : tok_rows.data() + (size_t) (r - 1) * (size_t) n_hd * (size_t) n_pos;
        for (int h = 0; h < n_hd; ++h) {
            const float * src = base + (size_t) h * (size_t) n_pos;
            std::copy(src, src + n_frames, out.data() + ((size_t) r * (size_t) n_hd + (size_t) h) * (size_t) n_frames);
        }
    }
    return out;
}
