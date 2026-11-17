#include "llama-kv-kivi-sim.h"

#include "llama-context.h"
#include "llama-kv-kivi.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct llama_kivi_sim {
    uint32_t bits = 2, G = 32, R = 128;
    uint32_t n_layer = 0, n_head_kv = 0, head_dim = 0;
    bool     v_trans = false;
    std::vector<std::unique_ptr<llama_kivi::cache>> caches; // [il * n_head_kv + h]; null for layers without KV
    uint32_t fed       = 0;   // tokens already appended to the containers
    bool     stopped   = false;
    uint64_t k_quant   = 0;   // K tokens written back dequantised (summed over layers and heads)
    uint64_t v_quant   = 0;
    uint64_t n_resets  = 0;
    uint64_t n_ubatch  = 0;
};

uint64_t llama_kivi_sim_k_tokens_quantised(const llama_kivi_sim * s) { return s ? s->k_quant : 0; }
uint64_t llama_kivi_sim_v_tokens_quantised(const llama_kivi_sim * s) { return s ? s->v_quant : 0; }

static void kivi_fail(const std::string & why) {
    throw std::runtime_error("--kivi-bits: " + why);
}

llama_kivi_sim * llama_kivi_sim_init(llama_context & lctx) {
    const auto & cp = lctx.cparams;
    if (cp.kivi_bits == 0) {
        return nullptr;
    }
    const auto & kv = lctx.kv_self;
    const auto & hp = lctx.model.hparams;

    if (cp.kivi_bits != 2 && cp.kivi_bits != 4 && cp.kivi_bits != 8) {
        kivi_fail("bits must be 2, 4 or 8 (got " + std::to_string(cp.kivi_bits) + ")");
    }
    if (cp.kivi_group <= 0 || cp.kivi_residual <= 0 || cp.kivi_residual % cp.kivi_group != 0) {
        kivi_fail("need --kivi-group > 0 and --kivi-residual a positive multiple of it");
    }
    if (kv.type_k != GGML_TYPE_F16 || kv.type_v != GGML_TYPE_F16) {
        kivi_fail(std::string("needs an f16 K and V cache (got ") + ggml_type_name(kv.type_k) + "/" + ggml_type_name(kv.type_v) +
                  "); KIVI replaces cache quantisation, it does not stack on --cache-type-k/-v. The attn_rot_k/--k-cache-hadamard path only applies to quantised K, so it is off.");
    }
    if (kv.recurrent || kv.hybrid || kv.any_compacted() || kv.k_l.empty() || kv.v_l.empty()) {
        kivi_fail("unsupported KV layout (recurrent/hybrid/compacted/MLA-latent cache)");
    }

    auto sim = std::make_unique<llama_kivi_sim>();
    sim->bits = cp.kivi_bits; sim->G = cp.kivi_group; sim->R = cp.kivi_residual;
    sim->n_layer = hp.n_layer;
    sim->v_trans = kv.v_trans;
    for (uint32_t il = 0; il < hp.n_layer; il++) {
        if (il >= kv.k_l.size() || !kv.k_l[il] || il >= kv.v_l.size() || !kv.v_l[il]) {
            continue; // layer without attention KV
        }
        if (kv.k_l[il]->extra || kv.v_l[il]->extra) {
            kivi_fail("split (multi-GPU) KV cache is not supported");
        }
        const uint32_t nh = hp.n_head_kv(il), dk = hp.n_embd_head_k(il), dv = hp.n_embd_head_v(il);
        if (nh == 0 || dk != dv) {
            kivi_fail("needs n_head_kv > 0 and equal K/V head size");
        }
        if (sim->head_dim == 0) { sim->head_dim = dk; sim->n_head_kv = nh; }
        if (dk != sim->head_dim || nh != sim->n_head_kv) {
            kivi_fail("layers with different head counts/sizes are not supported");
        }
        if (dk % sim->G != 0 || sim->G % (8 / sim->bits) != 0) {
            kivi_fail("head_dim must be a multiple of --kivi-group (head_dim=" + std::to_string(dk) + ")");
        }
        if ((int64_t) kv.k_l[il]->ne[1] != (int64_t) nh * kv.size) {
            kivi_fail("K tensor is not [head_dim, n_head_kv*size]");
        }
    }
    if (sim->head_dim == 0) {
        kivi_fail("model has no attention KV layers");
    }
    sim->caches.resize((size_t) sim->n_layer * sim->n_head_kv);
    LLAMA_LOG_INFO("%s: KIVI simulated KV quantisation ON: bits=%u group=%u residual=%u head_dim=%u n_head_kv=%u v_trans=%d "
                   "(K per-channel over G tokens, V per-token over G channels; cache tensors stay f16, only the values are quantised; "
                   "tokens older than the residual window are replaced by their dequantised value after each ubatch)\n",
                   __func__, sim->bits, sim->G, sim->R, sim->head_dim, sim->n_head_kv, (int) sim->v_trans);
    return sim.release();
}

void llama_kivi_sim_free(llama_kivi_sim * sim, bool log_summary) {
    if (!sim) return;
    if (log_summary) {
        LLAMA_LOG_INFO("%s: KIVI summary: ubatches=%llu fed_tokens=%u K tokens quantised (layers x heads x tokens)=%llu "
                       "V tokens quantised=%llu resets=%llu stopped=%d\n", __func__,
                       (unsigned long long) sim->n_ubatch, sim->fed, (unsigned long long) sim->k_quant,
                       (unsigned long long) sim->v_quant, (unsigned long long) sim->n_resets, (int) sim->stopped);
    }
    delete sim;
}

static void kivi_reset(llama_kivi_sim & s) {
    for (auto & c : s.caches) c.reset();
    s.fed = 0;
}

void llama_kivi_sim_after_ubatch(llama_context & lctx, int64_t n_tokens) {
    llama_kivi_sim * sp = lctx.kivi_sim;
    if (!sp || sp->stopped || n_tokens <= 0) return;
    llama_kivi_sim & s = *sp;
    auto & kv = lctx.kv_self;
    s.n_ubatch++;

    // 1. cells -> T tokens, require one sequence with cell index == position
    uint32_t T = 0;
    while (T < kv.size && !kv.cells[T].is_empty()) T++;
    bool ok = true;
    llama_seq_id seq0 = -1;
    for (uint32_t i = 0; i < kv.size && ok; i++) {
        const auto & c = kv.cells[i];
        if (i < T) {
            ok = c.pos == (llama_pos) i && c.seq_id.size() == 1;
            if (ok) { if (seq0 < 0) seq0 = *c.seq_id.begin(); ok = *c.seq_id.begin() == seq0; }
        } else {
            ok = c.is_empty();
        }
    }
    if (!ok) {
        LLAMA_LOG_WARN("%s: KIVI simulation stopped: the KV cache is not one sequence with cell==position (shift/multi-seq/defrag); "
                       "values already quantised stay quantised\n", __func__);
        s.stopped = true;
        return;
    }
    if (T < (uint32_t) n_tokens) return; // nothing sensible to do
    const uint32_t first_pos = T - (uint32_t) n_tokens; // this ubatch wrote [first_pos, T)
    if (first_pos != s.fed) {
        // cache was cleared / rolled back / a different prefix was loaded: rebuild the state from what the tensors hold now
        s.n_resets++;
        kivi_reset(s);
    }

    ggml_backend_sched_synchronize(lctx.sched);

    const uint32_t D = s.head_dim, NH = s.n_head_kv, from = s.fed, nnew = T - from;
    const size_t   esz = sizeof(ggml_fp16_t);
    std::vector<ggml_fp16_t> bufk, bufv;
    std::vector<float> kf(D), vf(D);

    for (uint32_t il = 0; il < s.n_layer; il++) {
        ggml_tensor * K = kv.k_l[il], * V = kv.v_l[il];
        if (!K || !V) continue;
        for (uint32_t h = 0; h < NH; h++) {
            auto & c = s.caches[(size_t) il * NH + h];
            if (!c) c.reset(new llama_kivi::cache(D, s.bits, s.G, s.R));
        }
        // read the new K rows [from, T) : rows (t*NH + h), D halves each
        bufk.resize((size_t) nnew * NH * D);
        ggml_backend_tensor_get(K, bufk.data(), (size_t) from * NH * D * esz, bufk.size() * esz);
        // V
        const uint32_t E = NH * D;
        std::vector<ggml_fp16_t> vall; // trans layout: whole tensor
        if (!s.v_trans) {
            bufv.resize((size_t) nnew * E);
            ggml_backend_tensor_get(V, bufv.data(), (size_t) from * E * esz, bufv.size() * esz);
        } else {
            vall.resize((size_t) kv.size * E);
            ggml_backend_tensor_get(V, vall.data(), 0, vall.size() * esz);
        }
        auto vat = [&](uint32_t t, uint32_t e) -> ggml_fp16_t {
            return s.v_trans ? vall[(size_t) e * kv.size + t] : bufv[(size_t) (t - from) * E + e];
        };

        std::vector<size_t> kq_before(NH), vq_before(NH);
        for (uint32_t h = 0; h < NH; h++) {
            auto & c = *s.caches[(size_t) il * NH + h];
            kq_before[h] = c.quantized_k_tokens();
            vq_before[h] = c.quantized_v_tokens();
            for (uint32_t t = from; t < T; t++) {
                for (uint32_t d = 0; d < D; d++) {
                    kf[d] = ggml_fp16_to_fp32(bufk[((size_t) (t - from) * NH + h) * D + d]);
                    vf[d] = ggml_fp16_to_fp32(vat(t, h * D + d));
                }
                c.append(kf.data(), vf.data());
            }
        }
        // write back the tokens that became quantised in this step
        // K: all heads flush together (same token count), so one contiguous token range
        const size_t kq0 = kq_before[0], kq1 = s.caches[(size_t) il * NH]->quantized_k_tokens();
        if (kq1 > kq0) {
            std::vector<ggml_fp16_t> out((kq1 - kq0) * NH * D);
            for (uint32_t h = 0; h < NH; h++) {
                auto & c = *s.caches[(size_t) il * NH + h];
                for (size_t t = kq0; t < kq1; t++) {
                    c.read_k(t, kf.data());
                    for (uint32_t d = 0; d < D; d++) out[((t - kq0) * NH + h) * D + d] = ggml_fp32_to_fp16(kf[d]);
                }
            }
            ggml_backend_tensor_set(K, out.data(), kq0 * NH * D * esz, out.size() * esz);
            s.k_quant += (uint64_t) (kq1 - kq0) * NH;
        }
        const size_t vq0 = vq_before[0], vq1 = s.caches[(size_t) il * NH]->quantized_v_tokens();
        if (vq1 > vq0) {
            std::vector<ggml_fp16_t> out;
            if (!s.v_trans) out.resize((vq1 - vq0) * E);
            for (uint32_t h = 0; h < NH; h++) {
                auto & c = *s.caches[(size_t) il * NH + h];
                for (size_t t = vq0; t < vq1; t++) {
                    c.read_v(t, vf.data());
                    for (uint32_t d = 0; d < D; d++) {
                        const ggml_fp16_t x = ggml_fp32_to_fp16(vf[d]);
                        if (!s.v_trans) out[(t - vq0) * E + h * D + d] = x;
                        else vall[(size_t) (h * D + d) * kv.size + t] = x;
                    }
                }
            }
            if (!s.v_trans) ggml_backend_tensor_set(V, out.data(), vq0 * E * esz, out.size() * esz);
            else            ggml_backend_tensor_set(V, vall.data(), 0, vall.size() * esz);
            s.v_quant += (uint64_t) (vq1 - vq0) * NH;
        }
    }
    s.fed = T;
}
