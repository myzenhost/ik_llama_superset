// Weight-level head-to-head of quantisation TYPES on REAL tensors of a GGUF, same imatrix for every type.
//
// WHY. End-to-end KLD/PPL (scripts/iq1-h2h/h2h_run.py) is the verdict, but at 1-1.5 bpw a small model is so damaged
// that the end-to-end numbers say little about WHERE types differ. This asks the narrower question for every type
// at once, cheaply: on the same rows of the same tensors, with the same activation-importance vector, how large
// is the reconstruction error, and how long does the quantiser take?
//
//   nrmse_plain    = sqrt( sum (w-q)^2 / sum w^2 )
//   nrmse_weighted = sqrt( sum_j I_j (w_j-q_j)^2 / sum_j I_j w_j^2 )   I = imatrix column importance (E[x_j^2])
// The weighted one is what the imatrix quantisers minimise and is the better proxy for output error.
//
// SCOPE: block tensors blk.<L>.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down}.weight of the chosen layers,
// an even sample of ROWS rows per tensor, row length a multiple of 256, host CPU, single thread. Row-interleaved
// (_r4/_r8) types need a multi-row layout and are skipped by name. It says nothing about kernel speed, activation
// quantisation, or how the errors compound through 16+ layers -- that is what the KLD run is for.
//
// USAGE: probe-quant-h2h-weights <model.gguf> <imatrix> <type,type,...> [layers=0,7,15] [rows=128]

#include "ggml.h"
#include "gguf.h"
#include "imatrix-loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define strcasecmp _stricmp   // POSIX name; MSVC spells it _stricmp (the probe built under MinGW only)
#endif

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, c)) if (!item.empty()) out.push_back(item);
    return out;
}

static ggml_type find_type(const std::string & name) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && strcasecmp(n, name.c_str()) == 0) return (ggml_type) i;
    }
    return GGML_TYPE_COUNT;
}

struct Acc { double num = 0, den = 0, wnum = 0, wden = 0, secs = 0, nw = 0; };

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf imatrix types [layers] [rows]\n", argv[0]); return 2; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string model = argv[1], imfile = argv[2];
    std::vector<ggml_type> types;
    for (auto & t : split(argv[3], ',')) {
        const ggml_type ty = find_type(t);
        if (ty == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type '%s'\n", t.c_str()); return 2; }
        types.push_back(ty);
    }
    std::vector<int> layers;
    for (auto & l : split(argc > 4 ? argv[4] : "0,7,15", ',')) layers.push_back(atoi(l.c_str()));
    const int ROWS = argc > 5 ? atoi(argv[5]) : 128;

    ggml_context * ctx = nullptr;
    gguf_init_params gp = { false, &ctx };
    gguf_context * g = gguf_init_from_file(model.c_str(), gp);
    if (!g || !ctx) { fprintf(stderr, "cannot read %s\n", model.c_str()); return 2; }

    common_imatrix im;
    if (!common_imatrix_load(imfile, im)) { fprintf(stderr, "cannot read imatrix %s\n", imfile.c_str()); return 2; }
    printf("model %s  imatrix entries=%zu legacy=%d  layers=%zu rows/tensor=%d\n", model.c_str(), im.entries.size(), (int) im.is_legacy, layers.size(), ROWS);

    const char * kinds[] = { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" };
    std::map<std::pair<int, std::string>, Acc> acc;   // (type, kind) and (type, "ALL")
    int n_tensors = 0, n_noim = 0;

    for (int L : layers) for (const char * kind : kinds) {
        char nm[128];
        snprintf(nm, sizeof(nm), "blk.%d.%s.weight", L, kind);
        ggml_tensor * t = ggml_get_tensor(ctx, nm);
        if (!t) { fprintf(stderr, "missing tensor %s\n", nm); continue; }
        const int64_t K = t->ne[0], M = t->ne[1];
        if (K % 256) { fprintf(stderr, "%s row length %lld not a multiple of 256\n", nm, (long long) K); continue; }
        // importance vector, normalised exactly as tools/quantize/quantize.cpp load_imatrix does (expert 0)
        std::vector<float> imp(K, 1.0f);
        auto it = im.entries.find(nm);
        if (it == im.entries.end() && strcmp(kind, "ffn_gate") == 0) {
            // same aliasing as src/llama-quantize.cpp: gate and up read the same activations, and the imatrix records the up name
            char up[128];
            snprintf(up, sizeof(up), "blk.%d.ffn_up.weight", L);
            it = im.entries.find(up);
        }
        if (it == im.entries.end() || (int64_t) it->second.sums.size() < K) {
            ++n_noim;
            printf("note: no usable imatrix entry for %s -> uniform importance for that tensor\n", nm);
        } else {
            const auto & e = it->second;
            const double cnt = e.counts.empty() ? 1.0 : (double) e.counts[0];
            for (int64_t j = 0; j < K; ++j) imp[j] = cnt > 0 ? (float) (e.sums[j] / cnt) : 1.0f;
        }
        // an even sample of rows, dequantised to f32 from the source type
        const int R = (int) std::min<int64_t>(ROWS, M);
        std::vector<float> w((size_t) R*K);
        const ggml_type_traits * st = ggml_get_type_traits(t->type);
        for (int r = 0; r < R; ++r) {
            const int64_t row = (int64_t) r * M / R;
            if (t->type == GGML_TYPE_F32) memcpy(w.data() + (size_t) r*K, (const char *) t->data + row*t->nb[1], K*4);
            else st->to_float((const char *) t->data + row*t->nb[1], w.data() + (size_t) r*K, K);
        }
        ++n_tensors;
        for (ggml_type ty : types) {
            const std::string nmty = ggml_type_name(ty);
            const char * us = strrchr(nmty.c_str(), '_');
            if (us && us[1] == 'r' && us[2] >= '0' && us[2] <= '9') continue;   // interleaved
            ggml_quantize_init(ty);
            const size_t rs = ggml_row_size(ty, K);
            std::vector<uint8_t> q(rs*R);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_quantize_chunk(ty, w.data(), q.data(), 0, R, K, imp.data(), nullptr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::vector<float> dq(K);
            Acc a;
            for (int r = 0; r < R; ++r) {
                ggml_get_type_traits(ty)->to_float(q.data() + r*rs, dq.data(), K);
                const float * wr = w.data() + (size_t) r*K;
                for (int64_t j = 0; j < K; ++j) {
                    const double d = (double) wr[j] - dq[j];
                    a.num += d*d; a.den += (double) wr[j]*wr[j];
                    a.wnum += imp[j]*d*d; a.wden += imp[j]*(double) wr[j]*wr[j];
                }
            }
            a.secs = secs; a.nw = (double) R*K;
            for (const char * key : { kind, "ALL" }) {
                Acc & s = acc[{ (int) ty, key }];
                s.num += a.num; s.den += a.den; s.wnum += a.wnum; s.wden += a.wden; s.secs += a.secs; s.nw += a.nw;
            }
        }
    }
    printf("tensors=%d  without-imatrix-entry=%d\n\n", n_tensors, n_noim);
    printf("%-10s %7s %10s %10s %9s | weighted nrmse by tensor kind\n", "type", "bpw", "nrmse", "nrmse_w", "Mw/s");
    printf("%-10s %7s %10s %10s %9s | ", "", "", "", "", "");
    for (const char * k : kinds) printf("%11s", k);
    printf("\n");
    for (ggml_type ty : types) {
        auto f = acc.find({ (int) ty, "ALL" });
        if (f == acc.end()) continue;
        const Acc & s = f->second;
        const double bpw = 8.0 * (double) ggml_type_size(ty) / (double) ggml_blck_size(ty);
        printf("%-10s %7.4f %10.4f %10.4f %9.2f | ", ggml_type_name(ty), bpw, std::sqrt(s.num/s.den), std::sqrt(s.wnum/s.wden), s.nw/1e6/(s.secs > 0 ? s.secs : 1e-9));
        for (const char * k : kinds) {
            auto fk = acc.find({ (int) ty, k });
            printf("%11.4f", fk == acc.end() ? -1.0 : std::sqrt(fk->second.wnum/fk->second.wden));
        }
        printf("\n");
    }
    gguf_free(g);
    ggml_free(ctx);
    return 0;
}
