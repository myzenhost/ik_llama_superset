// wavtokenizer.cpp -- see wavtokenizer.h for provenance and rationale.

#include "wavtokenizer.h"

#include "ggml.h"
// ADDED 2026-09-13. All 26 of this file's compile errors were one missing declaration each:
//   gguf_init_params / gguf_context / gguf_init_from_file / gguf_free / gguf_get_n_tensors /
//   gguf_get_tensor_name / gguf_get_tensor_offset / gguf_find_key / gguf_get_kv_type /
//   gguf_get_val_f32 / gguf_get_val_u32 / gguf_get_data_offset / GGUF_TYPE_FLOAT32 / GGUF_TYPE_UINT32
// all live in ggml/include/gguf.h (:84-85 for the two init entry points), and
// ggml_backend_cpu_init / ggml_backend_cpu_set_n_threads live in ggml/include/ggml-cpu.h:143.
// Neither header was included here: the gguf uplift moved those symbols OUT of ggml.h, and this
// file still assumed the pre-uplift layout. The tell was gcc's suggestions -- every one of them
// offered a ggml_ name for a gguf_ symbol, which is what a missing header looks like, not a
// renamed API.
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

//
// -------------------------------------------------------------- internals ---
//

namespace {

struct posnet_layer {
    // resnet blocks (posnet layers 0, 1, 3, 4)
    ggml_tensor * norm1   = nullptr;
    ggml_tensor * norm1_b = nullptr;
    ggml_tensor * conv1   = nullptr;
    ggml_tensor * conv1_b = nullptr;
    ggml_tensor * norm2   = nullptr;
    ggml_tensor * norm2_b = nullptr;
    ggml_tensor * conv2   = nullptr;
    ggml_tensor * conv2_b = nullptr;

    // attention block (posnet layer 2) and the trailing norm (posnet layer 5)
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * attn_q      = nullptr;
    ggml_tensor * attn_q_b    = nullptr;
    ggml_tensor * attn_k      = nullptr;
    ggml_tensor * attn_k_b    = nullptr;
    ggml_tensor * attn_v      = nullptr;
    ggml_tensor * attn_v_b    = nullptr;
    ggml_tensor * attn_o      = nullptr;
    ggml_tensor * attn_o_b    = nullptr;
};

struct convnext_layer {
    ggml_tensor * dw     = nullptr;
    ggml_tensor * dw_b   = nullptr;
    ggml_tensor * norm   = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * pw1    = nullptr;
    ggml_tensor * pw1_b  = nullptr;
    ggml_tensor * pw2    = nullptr;
    ggml_tensor * pw2_b  = nullptr;
    ggml_tensor * gamma  = nullptr;
};

} // namespace

struct wavtokenizer {
    wavtokenizer_params params;

    // hparams -- every one of these is derived from tensor SHAPES rather than
    // read from GGUF metadata keys, because the wavtokenizer key names differ
    // between converter versions while the tensor geometry does not.
    int64_t n_embd          = 0; // codebook embedding width
    int64_t n_vocab         = 0;
    int64_t posnet_n_embd   = 0;
    int64_t posnet_n_layer  = 0;
    int64_t convnext_n_embd = 0;
    int64_t convnext_n_layer= 0;
    int64_t n_ff            = 0;
    int64_t n_embd_out      = 0;

    float eps_norm  = 1e-5f;
    float eps_group = 1e-5f;
    int   n_groups  = 32;

    ggml_tensor * tok_embd   = nullptr;
    ggml_tensor * conv1d     = nullptr;
    ggml_tensor * conv1d_b   = nullptr;
    ggml_tensor * tok_norm   = nullptr;
    ggml_tensor * tok_norm_b = nullptr;
    ggml_tensor * out_norm   = nullptr;
    ggml_tensor * out_norm_b = nullptr;
    ggml_tensor * output     = nullptr;
    ggml_tensor * output_b   = nullptr;

    // zero column used to build reflection-free symmetric padding for the
    // depthwise convolution (ggml_pad only pads at the end).
    ggml_tensor * dw_zero_pad = nullptr;

    std::vector<posnet_layer>   posnet;
    std::vector<convnext_layer> convnext;

    ggml_context          * ctx_w   = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf_w   = nullptr;
    ggml_gallocr_t          galloc  = nullptr;

    std::map<std::string, ggml_tensor *> tensors;

    ~wavtokenizer() {
        if (galloc)  { ggml_gallocr_free(galloc); }
        if (buf_w)   { ggml_backend_buffer_free(buf_w); }
        if (ctx_w)   { ggml_free(ctx_w); }
        if (backend) { ggml_backend_free(backend); }
    }
};

namespace {

std::string fmt_name(const char * pattern, int i) {
    char buf[128];
    snprintf(buf, sizeof(buf), pattern, i);
    return std::string(buf);
}

ggml_tensor * get_opt(wavtokenizer & wt, const std::string & name) {
    auto it = wt.tensors.find(name);
    return it == wt.tensors.end() ? nullptr : it->second;
}

ggml_tensor * get_req(wavtokenizer & wt, const std::string & name, std::string & err) {
    ggml_tensor * t = get_opt(wt, name);
    if (!t && err.empty()) {
        err = "wavtokenizer: missing tensor '" + name + "'";
    }
    return t;
}

//
// ----------------------------------------------------------- graph helpers --
//

// LLM_NORM: plain layer norm over ne0, then scale + shift.
ggml_tensor * wt_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b, float eps) {
    cur = ggml_norm(ctx, cur, eps);
    if (w) {
        cur = ggml_mul(ctx, cur, w);
    }
    if (b) {
        cur = ggml_add(ctx, cur, b);
    }
    return cur;
}

// LLM_NORM_GROUP: llama.cpp reshapes [L, C] -> [L, 1, C] so that ggml_group_norm
// (which normalises over ne0*ne1 per group of ne2) sees one channel per slot.
ggml_tensor * wt_norm_group(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b, int n_groups, float eps) {
    cur = ggml_reshape_3d(ctx, cur, cur->ne[0], 1, cur->ne[1]);
    cur = ggml_group_norm(ctx, cur, n_groups, eps);
    cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[2]);

    if (w) {
        cur = ggml_mul(ctx, cur, w);
    }
    if (b) {
        cur = ggml_add(ctx, cur, b);
    }
    return cur;
}

// Depthwise 1-D convolution with "same" padding.
//
// ik_llama.cpp's ggml has no ggml_conv_1d_dw_ph (upstream added it alongside
// the OuteTTS work), and ggml.c is owned by another workstream, so this is
// expressed with primitives that DO exist: symmetric zero padding via
// ggml_concat, then an explicit tap-by-tap accumulation
//
//     out[l][c] = sum_k w[k][c] * padded[l + k][c]
//
// which is exactly a depthwise convolution and avoids depending on im2col's
// depthwise layout conventions.
ggml_tensor * wt_conv_1d_dw_ph(ggml_context * ctx,
                               ggml_tensor  * kernel, // [K, 1, C]
                               ggml_tensor  * cur,    // [L, C]
                               ggml_tensor  * zero_pad /* [K/2, C] */) {
    const int64_t K = kernel->ne[0];
    const int64_t C = cur->ne[1];

    ggml_tensor * padded = ggml_concat(ctx, zero_pad, cur, 0);
    padded = ggml_concat(ctx, padded, zero_pad, 0);
    padded = ggml_cont(ctx, padded);

    ggml_tensor * kern2d = ggml_reshape_2d(ctx, kernel, K, C);

    const int64_t L = cur->ne[0];

    ggml_tensor * acc = nullptr;
    for (int64_t k = 0; k < K; ++k) {
        ggml_tensor * slice = ggml_view_2d(ctx, padded,
                L, C,
                padded->nb[1],
                (size_t) k * padded->nb[0]);

        ggml_tensor * wk = ggml_view_2d(ctx, kern2d,
                1, C,
                kern2d->nb[1],
                (size_t) k * kern2d->nb[0]);

        ggml_tensor * term = ggml_mul(ctx, ggml_cont(ctx, slice), wk);
        acc = acc ? ggml_add(ctx, acc, term) : term;
    }

    return acc;
}

} // namespace

//
// ------------------------------------------------------------------ load ----
//

wavtokenizer * wavtokenizer_init_from_file(const std::string & path,
                                           const wavtokenizer_params & params,
                                           std::string & err) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta };

    gguf_context * gguf = gguf_init_from_file(path.c_str(), gparams);
    if (!gguf) {
        err = "wavtokenizer: failed to open GGUF '" + path + "'";
        return nullptr;
    }

    wavtokenizer * wt = new wavtokenizer();
    wt->params = params;

    auto fail = [&](const std::string & msg) -> wavtokenizer * {
        if (err.empty()) {
            err = msg;
        }
        gguf_free(gguf);
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
        delete wt;
        return nullptr;
    };

    // ---- backend ----------------------------------------------------------
#ifdef GGML_USE_CUDA
    if (params.use_gpu) {
        // ik_llama's signature DIVERGES from upstream's here:
        //     upstream  ggml_backend_cuda_init(int device)
        //     this fork ggml_backend_cuda_init(int device, const void * params, const void * model)
        // The audio.cpp port carried the upstream one-argument call through, and it does not
        // compile (C2660). Both extra arguments are optional context: llama.cpp passes cparams and
        // the model, and the fork's own STANDALONE consumer -- examples/rpc/rpc-server.cpp:266 --
        // passes (device, nullptr, nullptr) for exactly this case. wavtokenizer is standalone too:
        // it loads its own GGUF and has no llama_model or cuda_params to hand over.
        wt->backend = ggml_backend_cuda_init(0, nullptr, nullptr);
        if (!wt->backend && params.verbose) {
            fprintf(stderr, "%s: CUDA backend unavailable, falling back to CPU\n", __func__);
        }
    }
#endif
    if (!wt->backend) {
        wt->backend = ggml_backend_cpu_init();
    }
    if (!wt->backend) {
        return fail("wavtokenizer: no usable ggml backend");
    }
    ggml_backend_cpu_set_n_threads(wt->backend, std::max(1, params.n_threads));

    // ---- weight context ---------------------------------------------------
    const int n_tensors = gguf_get_n_tensors(gguf);
    if (n_tensors <= 0) {
        return fail("wavtokenizer: GGUF has no tensors");
    }

    {
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_tensors + 8),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        wt->ctx_w = ggml_init(ip);
        if (!wt->ctx_w) {
            return fail("wavtokenizer: ggml_init failed");
        }
    }

    // Every tensor is materialised as F32. The decoder is small (tens of
    // millions of parameters), and a single dtype removes an entire class of
    // "elementwise op got an F16 operand" failures from the graph below.
    struct src_tensor {
        std::string name;
        ggml_tensor * meta = nullptr;
        size_t offset = 0;
    };
    std::vector<src_tensor> srcs;
    srcs.reserve((size_t) n_tensors);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * meta = ggml_get_tensor(ctx_meta, name);
        if (!meta) {
            return fail(std::string("wavtokenizer: tensor '") + name + "' has no metadata");
        }
        if (meta->type != GGML_TYPE_F32 && meta->type != GGML_TYPE_F16) {
            return fail(std::string("wavtokenizer: tensor '") + name + "' has unsupported type " +
                        ggml_type_name(meta->type) + " (only F32/F16 decoder weights are supported)");
        }

        ggml_tensor * dst = ggml_new_tensor(wt->ctx_w, GGML_TYPE_F32, GGML_MAX_DIMS, meta->ne);
        ggml_set_name(dst, name);
        wt->tensors[name] = dst;

        srcs.push_back({ name, meta, gguf_get_tensor_offset(gguf, i) });
    }

    // the zero column used by the depthwise convolution -- sized once the
    // convnext width is known, so provision it after the shape scan below
    // (placeholder created here to keep it in the same allocation context)

    // ---- shapes -----------------------------------------------------------
    {
        ggml_tensor * tok = get_req(*wt, "token_embd.weight", err);
        ggml_tensor * c1w = get_req(*wt, "conv1d.weight",     err);
        if (!err.empty()) {
            return fail(err);
        }
        wt->n_embd        = tok->ne[0];
        wt->n_vocab       = tok->ne[1];
        wt->posnet_n_embd = c1w->ne[2];

        while (get_opt(*wt, fmt_name("posnet.%d.attn_norm.weight", (int) wt->posnet_n_layer)) ||
               get_opt(*wt, fmt_name("posnet.%d.norm1.weight",     (int) wt->posnet_n_layer))) {
            wt->posnet_n_layer++;
        }

        while (get_opt(*wt, fmt_name("convnext.%d.dw.weight", (int) wt->convnext_n_layer))) {
            wt->convnext_n_layer++;
        }

        if (wt->posnet_n_layer == 0 || wt->convnext_n_layer == 0) {
            return fail("wavtokenizer: could not find posnet/convnext blocks -- is this a WavTokenizer decoder GGUF?");
        }

        ggml_tensor * pw1 = get_req(*wt, "convnext.0.pw1.weight", err);
        ggml_tensor * dw0 = get_req(*wt, "convnext.0.dw.weight",  err);
        ggml_tensor * out = get_req(*wt, "output.weight",         err);
        if (!err.empty()) {
            return fail(err);
        }
        wt->convnext_n_embd = dw0->ne[2];
        wt->n_ff            = pw1->ne[1];
        wt->n_embd_out      = out->ne[1];

        if (wt->convnext_n_embd != wt->posnet_n_embd) {
            return fail("wavtokenizer: posnet width " + std::to_string(wt->posnet_n_embd) +
                        " != convnext width " + std::to_string(wt->convnext_n_embd));
        }

        // group count must divide the channel count; 32 is the WavTokenizer default
        if (wt->posnet_n_embd % wt->n_groups != 0) {
            wt->n_groups = 1;
            for (int g = 32; g >= 1; --g) {
                if (wt->posnet_n_embd % g == 0) {
                    wt->n_groups = g;
                    break;
                }
            }
        }

        const int64_t dw_k = dw0->ne[0];
        const int64_t pad  = dw_k / 2;
        wt->dw_zero_pad = ggml_new_tensor_2d(wt->ctx_w, GGML_TYPE_F32, pad, wt->convnext_n_embd);
        ggml_set_name(wt->dw_zero_pad, "wt.dw_zero_pad");
    }

    // optional metadata overrides for the epsilons
    {
        const char * eps_keys[] = {
            "wavtokenizer-dec.attention.layer_norm_epsilon",
            "wavtokenizer.attention.layer_norm_epsilon",
        };
        for (const char * k : eps_keys) {
            const int id = gguf_find_key(gguf, k);
            if (id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_FLOAT32) {
                wt->eps_norm = gguf_get_val_f32(gguf, id);
                break;
            }
        }
        const char * gkeys[] = {
            "wavtokenizer-dec.attention.group_norm_epsilon",
            "wavtokenizer.attention.group_norm_epsilon",
        };
        for (const char * k : gkeys) {
            const int id = gguf_find_key(gguf, k);
            if (id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_FLOAT32) {
                wt->eps_group = gguf_get_val_f32(gguf, id);
                break;
            }
        }
        const char * nkeys[] = {
            "wavtokenizer-dec.attention.group_norm_groups",
            "wavtokenizer.attention.group_norm_groups",
        };
        for (const char * k : nkeys) {
            const int id = gguf_find_key(gguf, k);
            if (id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_UINT32) {
                const int g = (int) gguf_get_val_u32(gguf, id);
                if (g > 0 && wt->posnet_n_embd % g == 0) {
                    wt->n_groups = g;
                }
                break;
            }
        }
    }

    // ---- allocate + upload ------------------------------------------------
    wt->buf_w = ggml_backend_alloc_ctx_tensors(wt->ctx_w, wt->backend);
    if (!wt->buf_w) {
        return fail("wavtokenizer: failed to allocate weight buffer");
    }

    {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            return fail("wavtokenizer: cannot reopen '" + path + "' to read tensor data");
        }
        const size_t data_off = gguf_get_data_offset(gguf);

        std::vector<uint8_t> raw;
        std::vector<float>   f32;

        for (const auto & s : srcs) {
            ggml_tensor * dst = wt->tensors[s.name];
            const size_t n_elem = (size_t) ggml_nelements(s.meta);
            const size_t nbytes = ggml_nbytes(s.meta);

            raw.resize(nbytes);
            fin.seekg((std::streamoff) (data_off + s.offset), std::ios::beg);
            if (!fin.read(reinterpret_cast<char *>(raw.data()), (std::streamsize) nbytes)) {
                return fail("wavtokenizer: short read for tensor '" + s.name + "'");
            }

            if (s.meta->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(dst, raw.data(), 0, nbytes);
            } else {
                f32.resize(n_elem);
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(raw.data()), f32.data(), (int64_t) n_elem);
                ggml_backend_tensor_set(dst, f32.data(), 0, n_elem * sizeof(float));
            }
        }

        // zero-fill the depthwise padding column
        std::vector<float> zeros((size_t) ggml_nelements(wt->dw_zero_pad), 0.0f);
        ggml_backend_tensor_set(wt->dw_zero_pad, zeros.data(), 0, zeros.size() * sizeof(float));
    }

    // ---- bind named weights ----------------------------------------------
    wt->tok_embd   = get_req(*wt, "token_embd.weight",       err);
    wt->conv1d     = get_req(*wt, "conv1d.weight",           err);
    wt->conv1d_b   = get_req(*wt, "conv1d.bias",             err);
    wt->tok_norm   = get_req(*wt, "token_embd_norm.weight",  err);
    wt->tok_norm_b = get_req(*wt, "token_embd_norm.bias",    err);
    wt->out_norm   = get_req(*wt, "output_norm.weight",      err);
    wt->out_norm_b = get_req(*wt, "output_norm.bias",        err);
    wt->output     = get_req(*wt, "output.weight",           err);
    wt->output_b   = get_req(*wt, "output.bias",             err);

    wt->posnet.resize((size_t) wt->posnet_n_layer);
    for (int i = 0; i < (int) wt->posnet_n_layer; ++i) {
        auto & l = wt->posnet[(size_t) i];
        l.norm1   = get_opt(*wt, fmt_name("posnet.%d.norm1.weight", i));
        l.norm1_b = get_opt(*wt, fmt_name("posnet.%d.norm1.bias",   i));
        l.conv1   = get_opt(*wt, fmt_name("posnet.%d.conv1.weight", i));
        l.conv1_b = get_opt(*wt, fmt_name("posnet.%d.conv1.bias",   i));
        l.norm2   = get_opt(*wt, fmt_name("posnet.%d.norm2.weight", i));
        l.norm2_b = get_opt(*wt, fmt_name("posnet.%d.norm2.bias",   i));
        l.conv2   = get_opt(*wt, fmt_name("posnet.%d.conv2.weight", i));
        l.conv2_b = get_opt(*wt, fmt_name("posnet.%d.conv2.bias",   i));

        l.attn_norm   = get_opt(*wt, fmt_name("posnet.%d.attn_norm.weight",   i));
        l.attn_norm_b = get_opt(*wt, fmt_name("posnet.%d.attn_norm.bias",     i));
        l.attn_q      = get_opt(*wt, fmt_name("posnet.%d.attn_q.weight",      i));
        l.attn_q_b    = get_opt(*wt, fmt_name("posnet.%d.attn_q.bias",        i));
        l.attn_k      = get_opt(*wt, fmt_name("posnet.%d.attn_k.weight",      i));
        l.attn_k_b    = get_opt(*wt, fmt_name("posnet.%d.attn_k.bias",        i));
        l.attn_v      = get_opt(*wt, fmt_name("posnet.%d.attn_v.weight",      i));
        l.attn_v_b    = get_opt(*wt, fmt_name("posnet.%d.attn_v.bias",        i));
        l.attn_o      = get_opt(*wt, fmt_name("posnet.%d.attn_output.weight", i));
        l.attn_o_b    = get_opt(*wt, fmt_name("posnet.%d.attn_output.bias",   i));
    }

    wt->convnext.resize((size_t) wt->convnext_n_layer);
    for (int i = 0; i < (int) wt->convnext_n_layer; ++i) {
        auto & l = wt->convnext[(size_t) i];
        l.dw     = get_req(*wt, fmt_name("convnext.%d.dw.weight",    i), err);
        l.dw_b   = get_req(*wt, fmt_name("convnext.%d.dw.bias",      i), err);
        l.norm   = get_req(*wt, fmt_name("convnext.%d.norm.weight",  i), err);
        l.norm_b = get_req(*wt, fmt_name("convnext.%d.norm.bias",    i), err);
        l.pw1    = get_req(*wt, fmt_name("convnext.%d.pw1.weight",   i), err);
        l.pw1_b  = get_req(*wt, fmt_name("convnext.%d.pw1.bias",     i), err);
        l.pw2    = get_req(*wt, fmt_name("convnext.%d.pw2.weight",   i), err);
        l.pw2_b  = get_req(*wt, fmt_name("convnext.%d.pw2.bias",     i), err);
        l.gamma  = get_req(*wt, fmt_name("convnext.%d.gamma.weight", i), err);
    }

    if (!err.empty()) {
        return fail(err);
    }

    wt->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(wt->backend));
    if (!wt->galloc) {
        return fail("wavtokenizer: ggml_gallocr_new failed");
    }

    if (params.verbose) {
        fprintf(stderr, "%s: %s\n", __func__, wavtokenizer_describe(wt).c_str());
    }

    gguf_free(gguf);
    ggml_free(ctx_meta);
    return wt;
}

void wavtokenizer_free(wavtokenizer * wt) {
    delete wt;
}

int wavtokenizer_n_embd_out(const wavtokenizer * wt) {
    return wt ? (int) wt->n_embd_out : 0;
}

int wavtokenizer_n_vocab(const wavtokenizer * wt) {
    return wt ? (int) wt->n_vocab : 0;
}

std::string wavtokenizer_describe(const wavtokenizer * wt) {
    if (!wt) {
        return "wavtokenizer: <null>";
    }
    char buf[512];
    snprintf(buf, sizeof(buf),
             "wavtokenizer-dec: n_vocab=%lld n_embd=%lld posnet=%lldx%lld convnext=%lldx%lld n_ff=%lld n_embd_out=%lld groups=%d",
             (long long) wt->n_vocab, (long long) wt->n_embd,
             (long long) wt->posnet_n_layer,   (long long) wt->posnet_n_embd,
             (long long) wt->convnext_n_layer, (long long) wt->convnext_n_embd,
             (long long) wt->n_ff, (long long) wt->n_embd_out, wt->n_groups);
    return std::string(buf);
}

//
// ----------------------------------------------------------------- decode ---
//

bool wavtokenizer_decode(wavtokenizer * wt,
                         const int32_t * codes,
                         int             n_codes,
                         std::vector<float> & out_embd,
                         std::string & err) {
    if (!wt) {
        err = "wavtokenizer: null context";
        return false;
    }
    if (n_codes <= 0) {
        err = "wavtokenizer: no codes to decode";
        return false;
    }

    for (int i = 0; i < n_codes; ++i) {
        if (codes[i] < 0 || codes[i] >= wt->n_vocab) {
            err = "wavtokenizer: code " + std::to_string(codes[i]) + " out of range [0, " +
                  std::to_string(wt->n_vocab) + ")";
            return false;
        }
    }

    // graph size: ~30 nodes per convnext layer (the depthwise expansion is the
    // heavy one) plus ~20 per posnet layer, with generous headroom
    const size_t graph_size = (size_t) (wt->convnext_n_layer * 64 + wt->posnet_n_layer * 32 + 128);

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        err = "wavtokenizer: ggml_init failed while building the decode graph";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_codes);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * cur = ggml_get_rows(ctx, wt->tok_embd, inp_tokens); // [n_embd, n_codes]

    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));                   // [n_codes, n_embd]

    cur = ggml_conv_1d_ph(ctx, wt->conv1d, cur, 1, 1);                // [n_codes, posnet_n_embd]
    cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
    cur = ggml_add(ctx, cur, wt->conv1d_b);

    // ---- posnet -----------------------------------------------------------
    for (int il = 0; il < (int) wt->posnet_n_layer; ++il) {
        const auto & layer = wt->posnet[(size_t) il];
        ggml_tensor * inpL = cur;

        if (layer.conv1 && layer.conv2) {
            // resnet block
            cur = wt_norm_group(ctx, cur, layer.norm1, layer.norm1_b, wt->n_groups, wt->eps_group);
            cur = ggml_mul(ctx, ggml_sigmoid(ctx, cur), cur);

            cur = ggml_conv_1d_ph(ctx, layer.conv1, cur, 1, 1);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
            cur = ggml_add(ctx, cur, layer.conv1_b);

            cur = wt_norm_group(ctx, cur, layer.norm2, layer.norm2_b, wt->n_groups, wt->eps_group);
            cur = ggml_mul(ctx, ggml_sigmoid(ctx, cur), cur);

            cur = ggml_conv_1d_ph(ctx, layer.conv2, cur, 1, 1);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
            cur = ggml_add(ctx, cur, layer.conv2_b);

            cur = ggml_add(ctx, cur, inpL);
        } else if (layer.attn_q) {
            // attention block
            cur = wt_norm_group(ctx, cur, layer.attn_norm, layer.attn_norm_b, wt->n_groups, wt->eps_group);

            ggml_tensor * q = ggml_conv_1d_ph(ctx, layer.attn_q, cur, 1, 1);
            ggml_tensor * k = ggml_conv_1d_ph(ctx, layer.attn_k, cur, 1, 1);
            ggml_tensor * v = ggml_conv_1d_ph(ctx, layer.attn_v, cur, 1, 1);

            q = ggml_add(ctx, ggml_reshape_2d(ctx, q, q->ne[0], q->ne[1]), layer.attn_q_b);
            k = ggml_add(ctx, ggml_reshape_2d(ctx, k, k->ne[0], k->ne[1]), layer.attn_k_b);
            v = ggml_add(ctx, ggml_reshape_2d(ctx, v, v->ne[0], v->ne[1]), layer.attn_v_b);

            q = ggml_cont(ctx, ggml_transpose(ctx, q));
            k = ggml_cont(ctx, ggml_transpose(ctx, k));

            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / sqrtf((float) wt->posnet_n_embd), 0.0f);

            cur = ggml_mul_mat(ctx, kq, v);

            cur = ggml_conv_1d_ph(ctx, layer.attn_o, cur, 1, 1);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
            cur = ggml_add(ctx, cur, layer.attn_o_b);

            cur = ggml_add(ctx, cur, inpL);
        } else {
            // trailing group norm
            cur = wt_norm_group(ctx, cur, layer.attn_norm, layer.attn_norm_b, wt->n_groups, wt->eps_group);
        }
    }

    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));                   // [posnet_n_embd, n_codes]
    cur = wt_norm(ctx, cur, wt->tok_norm, wt->tok_norm_b, wt->eps_norm);
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));                   // [n_codes, n_embd]

    // ---- convnext ---------------------------------------------------------
    ggml_tensor * inpL = cur;
    for (int il = 0; il < (int) wt->convnext_n_layer; ++il) {
        const auto & layer = wt->convnext[(size_t) il];

        cur = inpL;

        cur = wt_conv_1d_dw_ph(ctx, layer.dw, cur, wt->dw_zero_pad);
        cur = ggml_add(ctx, cur, layer.dw_b);

        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));               // [n_embd, n_codes]

        cur = wt_norm(ctx, cur, layer.norm, layer.norm_b, wt->eps_norm);

        // point-wise FFN: pw1 -> GELU -> pw2 (LLM_FFN_GELU / LLM_FFN_SEQ)
        cur = ggml_mul_mat(ctx, layer.pw1, cur);
        cur = ggml_add(ctx, cur, layer.pw1_b);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, layer.pw2, cur);
        cur = ggml_add(ctx, cur, layer.pw2_b);

        cur = ggml_mul(ctx, cur, layer.gamma);

        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));               // [n_codes, n_embd]

        inpL = ggml_add(ctx, cur, inpL);
    }
    cur = inpL;

    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));                   // [n_embd, n_codes]
    cur = wt_norm(ctx, cur, wt->out_norm, wt->out_norm_b, wt->eps_norm);

    cur = ggml_mul_mat(ctx, wt->output, cur);                         // [n_embd_out, n_codes]
    cur = ggml_add(ctx, cur, wt->output_b);

    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_gallocr_alloc_graph(wt->galloc, gf)) {
        ggml_free(ctx);
        err = "wavtokenizer: failed to allocate the decode graph";
        return false;
    }

    ggml_backend_tensor_set(inp_tokens, codes, 0, (size_t) n_codes * sizeof(int32_t));

    const ggml_status st = ggml_backend_graph_compute(wt->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "wavtokenizer: graph compute failed with status " + std::to_string((int) st);
        return false;
    }

    out_embd.resize((size_t) ggml_nelements(cur));
    ggml_backend_tensor_get(cur, out_embd.data(), 0, ggml_nbytes(cur));

    ggml_free(ctx);
    return true;
}
