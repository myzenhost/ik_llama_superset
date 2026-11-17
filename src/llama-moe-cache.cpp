#include "llama-moe-cache.h"

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache-share.h"
#include "llama-moe-cache-lru.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Merge of two implementations of one design (2026-11-03 upstream merge, both read in full):
//   * fork / ik_llama #2644 base: device mapping through model.default_layer_device, budget split by the fork's
//     device_shares(model.splits), RPC guard, the scheduler's QUERY/EXECUTE copy phases, overflow-checked alloc_size,
//     eviction + skip statistics.
//   * upstream llama.cpp (#30112, #30637; unsloth/master 977c48a25 src/llama-moe-cache.cpp): SLRU policy (probation +
//     protected halves, hits promote) whose victim() never evicts an expert the current request selected, copy_experts
//     for large batches (read-only use of the cache), memory_breakdown + the no_alloc measuring path, n_expert_used_max.

namespace {

// SLRU of the experts of a group of layers; the slot of each expert lives in the slot map of its layer
struct moe_cache_slru {
    struct fill { int32_t expert; int32_t slot; };

    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] slot-map data, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation (0) and protected (1) LRU lists; hits promote a slot to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        head[1] = tail[1] = -1;
        is_protected.assign(n_slots, false);
        n_protected = 0;
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head[from] = next[s];
        if (next[s] >= 0) prev[next[s]] = prev[s]; else tail[from] = prev[s];
        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) next[tail[protect]] = s; else head[protect] = s;
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    // least valuable slot that holds no expert selected by this request (probation first)
    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    // give a slot to each expert selected by ids in layer il, evicting from probation first.
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        fills.clear();
        n_hit = 0;
        n_eviction = 0;
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) return false;
        int32_t * slots = slot_map[il];
        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                ++n_hit;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads merge
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) continue;
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
                ++n_eviction;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

using llama_moe_cache_detail::moe_cache_lru;

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_up_gate_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t) res.push_back(t);
    }
    return res;
}

static bool is_host_weight(const ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static bool same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) return false;
    }
    return true;
}

} // namespace

struct llama_moe_cache::impl {
    struct device {
        ggml_backend * backend = nullptr;
        ggml_backend_buffer_type * buft = nullptr;
        int32_t model_device = -1;
        double split = 0.0;
        size_t host_bytes = 0;
        size_t buffer_size = 0;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
    };
    struct group {
        int32_t device = -1;
        std::vector<ggml_tensor *> ref;
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;
        moe_cache_lru lru; // used instead of slru when classlru
        bool classlru = false;
    };
    struct layer {
        int32_t group = -1;
        ggml_tensor * slot_map = nullptr;
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part; its data is not used
        ggml_tensor * cpu_map = nullptr;      // I32 [1, n_expert] host: e for the experts computed on the CPU, -1 for the others
        std::vector<int32_t> cpu_experts;     // experts set in cpu_map
        uint32_t cpu_acc = 0;                 // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts;
    };
    struct binding {
        int32_t layer;
        int32_t part;
        ggml_tensor * cached;
    };
    struct stats { size_t hits = 0; size_t misses = 0; size_t cpu = 0; size_t evictions = 0; size_t bytes = 0; };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings;
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part
    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;
    std::vector<moe_cache_lru::fill> fills_lru;
    float    cpu_share = 0.0f;   // 0 = off; resolved from -1 (auto) by calibrate_cpu_share()
    uint32_t cpu_share_q = 0;    // cpu_share in 1/65536, integer so that the tuner can reproduce the rounding
    int32_t  n_zero = 0;         // zero slots per bank after the cached ones
    bool     classlru = false;
    FILE *   trace = nullptr;    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for scripts/moe-cache-tune.py
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;
    ggml_context_ptr ctx_views; // views used by copy_experts
    stats stats_small, stats_large, stats_copy;
    size_t requested_size;
    mutable size_t large_batch_skips = 0;
    mutable size_t incompatible_layer_skips = 0;

    impl(const llama_model & model, const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts, size_t size, int policy, float share)
            : n_expert_used((int32_t) model.hparams.n_expert_used_max()), layers(model.layers.size()), requested_size(size),
              cpu_share(share < 0.0f ? -1.0f : std::min(share, 1.0f)) {
        if (model.hparams.n_expert == 0 || n_expert_used <= 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (!model.rpc_servers.empty()) throw std::runtime_error("MoE cache does not support RPC devices");
        if (backends.size() != bufts.size()) throw std::runtime_error("MoE cache backend metadata is inconsistent");

        for (size_t i = 0; i < backends.size(); ++i) {
            if (!ggml_backend_is_cpu(backends[i])) {
                device d;
                d.backend = backends[i];
                d.buft = bufts[i];
                devices.push_back(std::move(d));
            }
        }
        if (devices.empty()) throw std::runtime_error("MoE cache requires a GPU backend");

        // only layers that keep all of their experts in host memory, cached on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), is_host_weight)) continue;
            if (!std::all_of(experts.begin(), experts.end(), [&](const ggml_tensor * t) {
                    return t->ne[2] == experts[0]->ne[2];
                })) continue;
            if (il >= model.default_layer_device.size()) continue;
            const int32_t model_device = model.default_layer_device[il];
            if (model_device < 0 || (size_t) model_device >= model.devices.size()) continue;
            const auto * layer_buft = model.default_buffer_type_offload(model_device);
            auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.buft == layer_buft; });
            if (it_dev == devices.end()) continue;
            const int32_t id = (int32_t) (it_dev - devices.begin());
            it_dev->model_device = model_device;
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.device == id && same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->device = id;
                it->ref = experts;
            }
            it->layers.push_back((int32_t) il);
            for (ggml_tensor * t : experts) {
                it->host_bytes += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no complete host-resident expert layer is eligible. MoE cache is inactive\n", __func__);
            return;
        }

        std::vector<bool> eligible(model.splits.size(), false);
        for (const device & d : devices) {
            if (d.host_bytes && d.model_device >= 0 && (size_t) d.model_device < eligible.size()) {
                eligible[d.model_device] = true;
            }
        }
        const std::vector<double> shares = llama_moe_cache_detail::device_shares(model.splits, eligible);
        for (device & d : devices) {
            d.split = d.model_device >= 0 && (size_t) d.model_device < shares.size() ? shares[d.model_device] : 0.0;
        }

        double split_sum = 0.0;
        for (const device & d : devices) if (d.host_bytes) split_sum += d.split;
        if (split_sum <= 0.0) {
            for (device & d : devices) if (d.host_bytes) d.split = 1.0;
            split_sum = std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes != 0; });
        }
        // upstream e95426f16: with misses on the CPU the n_zero slots after the cached ones stay zero, the device computes these misses as zero.
        // The CPU share needs the CRF the classlru policy keeps, so it selects classlru; slru stays the default and is bit-for-bit as before.
        classlru = policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU;
        if (cpu_share != 0.0f) {
            if (!classlru) LLAMA_LOG_INFO("%s: --moe-cache-cpu-share needs the classlru policy, using it\n", __func__);
            classlru = true;
            n_zero = std::min<int32_t>(n_expert_used, 64);
        }
        // one extra slot at the end: CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.device].buft);
            size_t total = 0;
            for (const ggml_tensor * t : g.ref) {
                const size_t bytes = t->nb[2] * (size_t) (slots + n_zero + 1);
                if (bytes > SIZE_MAX - alignment) return SIZE_MAX;
                const size_t padded = GGML_PAD(bytes, alignment);
                if (padded > SIZE_MAX - total) return SIZE_MAX;
                total += padded;
            }
            return total;
        };

        // within a device the budget is split by expert bytes, so each group caches the same fraction of its experts
        size_t host_layer_count = 0;
        std::vector<size_t> device_tensor_counts(devices.size(), 0);
        for (group & g : groups) {
            const device & d = devices[g.device];
            const size_t budget = (size_t) ((double) size * d.split / split_sum * g.host_bytes / d.host_bytes);
            const int32_t n_expert = (int32_t) g.ref[0]->ne[2];
            const int32_t max_slots = (int32_t) std::min<size_t>(INT32_MAX, g.layers.size() * (size_t) n_expert);
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) ++g.n_slots;
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: cache budget leaves %zu expert layers uncached because they cannot hold one token\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.classlru = classlru;
            if (classlru) g.lru.init((int32_t) layers.size(), n_expert, g.n_slots); else g.slru.init((int32_t) layers.size(), n_expert, g.n_slots);
            device_tensor_counts[g.device] += g.ref.size() * (1 + g.layers.size());
            host_layer_count += g.layers.size() * (n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        if (host_layer_count == 0) throw std::runtime_error("MoE cache budget is too small to hold one token");

        auto make_ctx = [](size_t n_tensors) {
            ggml_init_params params = { n_tensors * ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) throw std::runtime_error("failed to create MoE cache tensor metadata");
            return ctx;
        };
        for (size_t id = 0; id < devices.size(); ++id) if (device_tensor_counts[id]) devices[id].ctx = make_ctx(device_tensor_counts[id]);
        ctx_host  = make_ctx(host_layer_count);
        ctx_views = make_ctx(2);
        const auto buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (group & g : groups) {
            if (g.n_slots == 0) continue;
            auto & d = devices[g.device];
            for (ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(d.ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                if (bank->nb[2] != t->nb[2]) throw std::runtime_error("MoE cache expert layout is not contiguous by expert");
                ggml_format_name(bank, "moe_cache.%d.%s", (int) (&g - groups.data()), t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.group = (int32_t) (&g - groups.data());
                l.experts = layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(d.ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero,
                            bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            d.buffer_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only measuring the memory use (llama_context::memory_breakdown): zero-size dummy buffers, tensors point at them
            for (device & d : devices) {
                if (!d.ctx) continue;
                d.buffer.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buffer.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) continue;
                d.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buffer) throw std::runtime_error("failed to allocate MoE cache buffers");
                ggml_backend_buffer_clear(d.buffer.get(), 0);
                d.buffer_size = ggml_backend_buffer_get_size(d.buffer.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) throw std::runtime_error("failed to allocate MoE cache slot maps");
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // every slot -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            for (group & g : groups) for (int32_t il : g.layers) {
                if (!layers[il].slot_map) continue;
                int32_t * slots = (int32_t *) layers[il].slot_map->data;
                if (g.classlru) {
                    std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty); // n_slots or later = not cached
                    g.lru.slot_map[il] = slots;
                } else {
                    g.slru.slot_map[il] = slots;
                }
            }
        }
        // as weights: the ops reading the banks run on the device and the slot maps go through the copy callback
        for (device & d : devices) if (d.buffer) ggml_backend_buffer_set_usage(d.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (buf_host) ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t i = 0; i < devices.size(); ++i) {
            const device & d = devices[i];
            if (!d.host_bytes || !d.buffer_size) continue;
            LLAMA_LOG_INFO("%s: %s MoE cache uses %.2f MiB for %.2f MiB of host experts\n", __func__,
                    ggml_backend_buft_name(d.buft), d.buffer_size / 1048576.0, d.host_bytes / 1048576.0);
            for (const group & g : groups) if (g.device == (int32_t) i && g.n_slots) {
                LLAMA_LOG_INFO("%s: %zu layer(s), %s: %d expert slots (%.1f%% coverage)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots,
                        100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
            }
        }
        cpu_share_q = (uint32_t) std::lround(std::max(cpu_share, 0.0f)*65536.0f);
        if (model.hparams.no_alloc) {
            return;
        }
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb"); // atomic-ok: debug trace stream written record by record for scripts/moe-cache-tune.py, not a state file
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        if (cpu_share < 0.0f) {
            calibrate_cpu_share();
        }
        if (cpu_share > 0.0f) {
            LLAMA_LOG_INFO("%s: %.1f%% of the misses are computed on the CPU once the cache is full\n", __func__, 100.0*cpu_share);
        }
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    // (upstream e95426f16 format, with this fork's group fields)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32((uint32_t) layers.size());
        u32((uint32_t) n_expert_used);
        u32((uint32_t) groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.device];
            u32((uint32_t) g.device);
            u32((uint32_t) g.ref[0]->ne[2]);
            u32((uint32_t) g.layers.size());
            for (int32_t il : g.layers) u32((uint32_t) il);
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32((uint32_t) g.ref.size());
            for (const ggml_tensor * t : g.ref) u64(t->nb[2]);
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    // --moe-cache-cpu-share auto. Measures, on the host experts as the cache will read them (mmap'd or pinned), the speed at which the device
    // takes an expert upload (U) and the speed at which the CPU threads read experts (C, the rate the CPU matmul is bound by). Per byte of
    // missed experts the device needs (1-F)/U and the CPU F/C, in parallel; this is moe-cache-tune.py's model with its rule, the lowest share
    // within 5% of the fastest. The tuner fits the same two speeds from timed runs; this one measures them directly in a few ms.
    void calibrate_cpu_share() {
        const group * gr = nullptr;
        for (const group & g : groups) if (g.n_slots > 0) { gr = &g; break; }
        if (gr == nullptr) { cpu_share = 0.0f; cpu_share_q = 0; return; }
        const group & g = *gr;
        struct piece { const uint8_t * src; size_t size; ggml_tensor * bank; size_t off; };
        std::vector<piece> pieces;
        const int32_t n_expert = (int32_t) g.ref[0]->ne[2];
        const int32_t n_sample = std::min<int32_t>(16, g.n_slots);
        size_t total = 0;
        for (int32_t k = 0; k < n_sample; ++k) {
            const int32_t il = g.layers[(size_t) k % g.layers.size()];
            const int32_t e  = (int32_t) (((int64_t) k*2654435761LL) % n_expert);
            const layer & l = layers[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                pieces.push_back({ (const uint8_t *) w->data + (size_t) e*w->nb[2], w->nb[2], g.banks[ip], (size_t) k*w->nb[2] });
                total += w->nb[2];
            }
        }
        if (pieces.empty() || total == 0) { cpu_share = 0.0f; cpu_share_q = 0; return; }
        using clk = std::chrono::steady_clock;
        const device & d = devices[g.device];
        const unsigned n_thr = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        auto read_all = [&]() {
            std::vector<std::thread> th;
            std::vector<uint64_t> acc(n_thr, 0);
            for (unsigned t = 0; t < n_thr; ++t) {
                th.emplace_back([&, t]() {
                    uint64_t s = 0;
                    for (size_t i = t; i < pieces.size(); i += n_thr) {
                        const uint64_t * p = (const uint64_t *) pieces[i].src;
                        for (size_t j = 0; j < pieces[i].size/8; ++j) s += p[j];
                    }
                    acc[t] = s;
                });
            }
            for (auto & x : th) x.join();
            uint64_t s = 0;
            for (uint64_t a : acc) s += a;
            return s;
        };
        volatile uint64_t sink = read_all(); // first touch: pages of a mmap'd model come in from disk here, not in the timed passes
        auto upload_all = [&]() {
            for (const piece & p : pieces) ggml_backend_tensor_set_async(d.backend, p.bank, p.src, p.off, p.size);
            ggml_backend_synchronize(d.backend);
        };
        upload_all();
        const int reps = 3;
        auto t0 = clk::now();
        for (int r = 0; r < reps; ++r) upload_all();
        const double t_up = std::chrono::duration<double>(clk::now() - t0).count();
        t0 = clk::now();
        for (int r = 0; r < reps; ++r) sink = sink + read_all();
        const double t_cpu = std::chrono::duration<double>(clk::now() - t0).count();
        const double U = reps*(double) total/std::max(t_up, 1e-9);
        const double C = reps*(double) total/std::max(t_cpu, 1e-9);
        auto cost = [&](double f) { return std::max((1.0 - f)/U, f/C); };
        double best = cost(0.0);
        const double grid[] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 };
        for (double f : grid) best = std::min(best, cost(f));
        double pick = 0.0;
        for (double f : grid) if (cost(f) <= 1.05*best) { pick = f; break; }
        cpu_share   = (float) pick;
        cpu_share_q = (uint32_t) std::lround(pick*65536.0);
        LLAMA_LOG_INFO("llama_moe_cache: cpu share auto: upload %.2f GiB/s, CPU read %.2f GiB/s (%u threads, %.1f MiB sampled) -> share %.2f\n",
                U/1073741824.0, C/1073741824.0, n_thr, total/1048576.0, pick);
        (void) sink;
    }

    ~impl() {
        log_stats();
        if (trace != nullptr) fclose(trace);
    }

    ggml_backend * backend(int32_t il) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return nullptr;
        return devices[groups[layers[il].group].device].backend;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.group];
        // large batches use most experts of a layer: little gain, and they would evict the generation experts
        if (n_tokens <= 0 || n_tokens > max_batch) {
            ++large_batch_skips;
            return nullptr;
        }
        if (std::min<int64_t>(n_tokens*n_expert_used, l.slot_map->ne[1]) > g.n_slots) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    // the experts of layer il computed on the CPU: I32 [1, n_expert] host, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].cpu_map == nullptr || cpu_share_q == 0) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * t) const {
        auto it = bindings.find(t);
        return it == bindings.end() ? nullptr : it->second.cached;
    }

    int64_t copy_experts(ggml_backend * backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) return 0;
        const binding & b = it->second;
        const group & g = groups[layers[b.layer].group];
        if (backend != devices[g.device].backend) return 0;
        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.classlru ? g.lru.slot_map[b.layer] : g.slru.slot_map[b.layer];
        if (slots == nullptr || slots[e] < 0 || slots[e] >= g.n_slots) return 0; // classlru: n_slots or later = not cached
        int64_t n = 1;
        while (e + n <= last && slots[e + n] >= 0 && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) n++;
        ggml_tensor * bank = g.banks[b.part];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);
        return n;
    }

    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase) {
        auto it = layer_of.find(src);
        if (it == layer_of.end()) return false;
        if (phase == GGML_BACKEND_SCHED_COPY_PHASE_QUERY) return true;
        const int32_t il = it->second.first;
        const bool has_cpu = it->second.second;
        layer & l = layers[il];
        group & g = groups[l.group];
        const device & d = devices[g.device];
        GGML_ASSERT(backend == d.backend);
        // the get_rows that looks up the slots of the selected experts
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && !lookup; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) lookup = node;
        }
        GGML_ASSERT(lookup && "MoE cache slot map must feed GET_ROWS");
        // the selected experts must be computed in an earlier split
        const ggml_tensor * selected = lookup->src[1];
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == selected || node == selected->view_src) {
                GGML_ABORT("MoE experts for layer %d are selected in the same scheduler split as the cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(selected));
        ids.resize(ggml_nelements(selected));
        ggml_backend_tensor_get_async(backend, selected, ids.data(), 0, ggml_nbytes(selected));
        ggml_backend_synchronize(backend);
        size_t n_hit = 0;
        size_t n_eviction = 0;
        cpu.clear();
        if (g.classlru) {
            GGML_ASSERT(selected->op == GGML_OP_RESHAPE && selected->src[0] != nullptr); // of the [n_expert_used, n_tokens] ids, see the graph builders
            const int32_t n_used = (int32_t) selected->src[0]->ne[0];
            const int64_t n_tok  = selected->src[0]->ne[1];
            if (trace != nullptr) write_trace(il, n_used);
            // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
            const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_slots ? cpu_share_q : 0;
            auto n_cpu = [&](size_t n_miss) {
                const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
                l.cpu_acc = (uint32_t) (acc & 0xffff);
                return (size_t) (acc >> 16);
            };
            if (!g.lru.plan(il, ids.data(), ids.size(), n_tok, n_used, n_zero, n_cpu, fills_lru, cpu, n_hit)) {
                GGML_ABORT("MoE cache slot count is smaller than the routed expert set");
            }
            n_eviction = g.lru.n_evict;
            fills.clear();
            for (const auto & f : fills_lru) fills.push_back({ f.expert, f.slot });
            if (has_cpu) {
                int32_t * cpu_map = (int32_t *) l.cpu_map->data;
                for (int32_t e : l.cpu_experts) cpu_map[e] = -1;
                l.cpu_experts.clear();
                for (const auto & f : cpu) {
                    cpu_map[f.expert] = f.expert;
                    l.cpu_experts.push_back(f.expert);
                }
            }
        } else if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit, n_eviction)) {
            GGML_ABORT("MoE cache slot count is smaller than the routed expert set");
        }
        // upload the missing experts; consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w = l.experts[ip];
            ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i+n < fills.size() && fills[i+n].expert == fills[i].expert + (int32_t)n &&
                        fills[i+n].slot == fills[i].slot + (int32_t)n) ++n;
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size,
                        fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu += cpu.size();
        st.evictions += n_eviction;
        st.bytes += bytes;
        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map)); // slot_map_cpu carries no data of its own
        return true;
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const {
        std::map<ggml_backend_buffer_type_t, size_t> res;
        for (const device & d : devices) {
            if (d.buffer) res[ggml_backend_buffer_get_type(d.buffer.get())] += d.buffer_size;
        }
        if (buf_host) res[ggml_backend_buffer_get_type(buf_host.get())] += buf_host_size;
        return res;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & s) {
            const size_t n = s.hits + s.misses;
            if (n) LLAMA_LOG_INFO("llama_moe_cache: %s hits=%zu misses=%zu evictions=%zu hit-rate=%.2f%% uploaded=%.2f MiB cpu=%zu\n",
                    name, s.hits, s.misses, s.evictions, 100.0*s.hits/n, s.bytes/1048576.0, s.cpu);
        };
        size_t actual_size = buf_host_size;
        for (const device & d : devices) actual_size += d.buffer_size;
        LLAMA_LOG_INFO("llama_moe_cache: requested %.2f MiB, allocated %.2f MiB\n",
                requested_size / 1048576.0, actual_size / 1048576.0);
        log("ubatch <= 8", stats_small);
        log("ubatch 9-32", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n",
                    stats_copy.hits, stats_copy.bytes/1048576.0);
        }
        if (large_batch_skips || incompatible_layer_skips) {
            LLAMA_LOG_INFO("llama_moe_cache: skipped %zu large-batch and %zu ineligible layer accesses\n",
                    large_batch_skips, incompatible_layer_skips);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model,
        const std::vector<ggml_backend *> & backends,
        const std::vector<ggml_backend_buffer_type *> & bufts, size_t size, int policy, float cpu_share)
        : pimpl(new impl(model, backends, bufts, size, policy, cpu_share)) {}

llama_moe_cache::~llama_moe_cache() = default;

// smallest budget that gives every group of host-resident expert layers the slots for the experts of one token. The cache splits its
// budget between the groups of a device by their share of the host expert bytes, so a group g gets size * g.bytes / device.bytes and
// needs alloc(n_expert_used) of it: size >= alloc * device.bytes / g.bytes for every group. Same grouping as the constructor above,
// for the one-GPU case (the only one --moe-cache-mib auto serves).
size_t llama_model_moe_cache_min_size(const llama_model * model) {
    const llama_hparams & hp = model->hparams;
    const int32_t n_expert_used = (int32_t) hp.n_expert_used_max();
    if (hp.n_expert == 0 || n_expert_used <= 0 || model->devices.size() != 1) {
        return 0;
    }
    ggml_backend_buffer_type_t buft = const_cast<ggml_backend_buffer_type_t>(model->default_buffer_type_offload(0));
    if (!buft || ggml_backend_buft_is_host(buft)) {
        return 0;
    }
    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    std::vector<std::vector<ggml_tensor *>> refs;
    std::vector<size_t> group_bytes;
    size_t host_bytes = 0;
    for (size_t il = 0; il < model->layers.size(); ++il) {
        auto experts = layer_experts(model->layers[il]);
        if (experts.empty() || !std::all_of(experts.begin(), experts.end(), is_host_weight)) continue;
        if (!std::all_of(experts.begin(), experts.end(), [&](const ggml_tensor * t) { return t->ne[2] == experts[0]->ne[2]; })) continue;
        if (il >= model->default_layer_device.size() || model->default_layer_device[il] != 0) continue;
        auto it = std::find_if(refs.begin(), refs.end(), [&](const std::vector<ggml_tensor *> & ref) { return same_layout(ref, experts); });
        if (it == refs.end()) {
            refs.push_back(experts);
            group_bytes.push_back(0);
            it = refs.end() - 1;
        }
        for (const ggml_tensor * t : experts) {
            group_bytes[it - refs.begin()] += ggml_nbytes(t);
            host_bytes                     += ggml_nbytes(t);
        }
    }

    size_t res = 0;
    for (size_t ig = 0; ig < refs.size(); ++ig) {
        size_t need = 0; // n_expert_used slots + the extra guard slot
        for (const ggml_tensor * t : refs[ig]) {
            need += GGML_PAD(t->nb[2]*(size_t) (n_expert_used + 1), alignment);
        }
        res = std::max(res, (size_t) std::ceil((double) need*(double) host_bytes/(double) group_bytes[ig]) + 1);
    }
    return res;
}
ggml_backend * llama_moe_cache::backend(int32_t il) const { return pimpl->backend(il); }
ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const { return pimpl->get_slot_map(il, n_tokens, n_expert_used, cpu_part); }
ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il, int64_t n_tokens) const { return pimpl->get_cpu_map(il, n_tokens); }
ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * t) const { return pimpl->get_experts(t); }
bool llama_moe_cache::copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
        ggml_backend_sched_copy_phase phase) { return pimpl->copy(backend, src, dst, graph, phase); }
int64_t llama_moe_cache::copy_experts(ggml_backend * backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}
std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const { return pimpl->memory_breakdown(); }
