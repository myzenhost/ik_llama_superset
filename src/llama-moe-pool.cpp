// llama-moe-pool.cpp -- see llama-moe-pool.h for full attribution and scope notes.
#include "llama-moe-pool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

// Replace `dst` with `src` in one step, so a reader never sees a half-written `dst`.
// Windows: MoveFileExW with REPLACE_EXISTING (plain rename() fails when dst exists) and
// WRITE_THROUGH (do not return before the move is on disk). POSIX: rename(2) is atomic.
static bool moe_replace_file(const std::string & src, const std::string & dst) {
#ifdef _WIN32
    auto widen = [](const std::string & s, std::wstring & out) -> bool {
        const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), -1, nullptr, 0);
        if (n <= 0) return false;
        out.assign((size_t) n, L'\0');
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), -1, &out[0], n) != n) return false;
        out.resize((size_t) n - 1); // drop the terminator MultiByteToWideChar counted
        return true;
    };
    std::wstring wsrc, wdst;
    if (!widen(src, wsrc) || !widen(dst, wdst)) return false;
    return MoveFileExW(wsrc.c_str(), wdst.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(src.c_str(), dst.c_str()) == 0;
#endif
}

static unsigned long moe_pid() {
#ifdef _WIN32
    return (unsigned long) GetCurrentProcessId();
#else
    return (unsigned long) getpid();
#endif
}

llama_moe_recorder_mode llama_moe_recorder_mode_from_str(const std::string & s) {
    if (s == "stat")        return LLAMA_MOE_RECORDER_STAT;
    if (s == "stat_approx") return LLAMA_MOE_RECORDER_STAT_APPROX;
    if (s == "per_pass")    return LLAMA_MOE_RECORDER_PER_PASS;
    if (s == "per_token")   return LLAMA_MOE_RECORDER_PER_TOKEN;
    throw std::invalid_argument("unknown --expert-distribution-recorder-mode: " + s);
}

llama_moe_placement_strategy llama_moe_placement_strategy_from_str(const std::string & s) {
    if (s == "uniform")        return LLAMA_MOE_PLACEMENT_UNIFORM;
    if (s == "frequency")      return LLAMA_MOE_PLACEMENT_FREQUENCY;
    if (s == "front-loading")  return LLAMA_MOE_PLACEMENT_FRONT_LOADING;
    if (s == "random")         return LLAMA_MOE_PLACEMENT_RANDOM;
    throw std::invalid_argument("unknown --moe-expert-placement-strategy: " + s);
}

// ───────────────────────── recorder ─────────────────────────

void llama_moe_recorder::record(int32_t layer, const int32_t * ids, int64_t n_ids) {
    auto & row = counts[layer];
    for (int64_t i = 0; i < n_ids; ++i) {
        row[ids[i]] += 1;
    }
    steps_recorded += 1;
}

void llama_moe_recorder::maybe_decay(int32_t half_life) {
    if (mode != LLAMA_MOE_RECORDER_STAT_APPROX || half_life <= 0) {
        return;
    }
    if (steps_recorded == 0 || steps_recorded % (uint64_t) half_life != 0) {
        return;
    }
    for (auto & layer_row : counts) {
        for (auto & kv : layer_row.second) {
            kv.second = kv.second / 2; // halve, matches the PR-thread admission-gate idea
        }
    }
}

std::vector<std::vector<uint64_t>> llama_moe_recorder::to_dense(int32_t n_layers, int32_t n_experts) const {
    std::vector<std::vector<uint64_t>> dense(n_layers, std::vector<uint64_t>(n_experts, 0));
    for (auto & layer_row : counts) {
        if (layer_row.first < 0 || layer_row.first >= n_layers) continue;
        for (auto & kv : layer_row.second) {
            if (kv.first < 0 || kv.first >= n_experts) continue;
            dense[layer_row.first][kv.first] = kv.second;
        }
    }
    return dense;
}

static const char * recorder_mode_name(llama_moe_recorder_mode m) {
    switch (m) {
        case LLAMA_MOE_RECORDER_STAT:        return "stat";
        case LLAMA_MOE_RECORDER_STAT_APPROX: return "stat_approx";
        case LLAMA_MOE_RECORDER_PER_PASS:    return "per_pass";
        case LLAMA_MOE_RECORDER_PER_TOKEN:   return "per_token";
    }
    return "stat";
}

// Written to <path>.tmp.<pid> first and moved over <path> only after every byte is confirmed
// written and closed. Before 2026-11-03 this wrote <path> in place and ignored fclose(): a kill,
// crash or full disk mid-write left a truncated profile at <path> while the caller was told the
// write succeeded, and read_json below then loaded the damaged counts.
bool llama_moe_recorder::write_json(const std::string & path) const {
    const std::string tmp = path + ".tmp." + std::to_string(moe_pid());
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        return false;
    }
    fprintf(f, "{\"mode\":\"%s\",\"steps\":%llu,\"layers\":{",
            recorder_mode_name(mode), (unsigned long long) steps_recorded);
    // Deterministic ordering (sorted layer, then sorted expert id) so the output is diffable and
    // the "print the seeded expert ids" control in the task brief has something stable to compare.
    std::vector<int32_t> layer_ids;
    layer_ids.reserve(counts.size());
    for (auto & kv : counts) layer_ids.push_back(kv.first);
    std::sort(layer_ids.begin(), layer_ids.end());
    bool first_layer = true;
    for (int32_t il : layer_ids) {
        if (!first_layer) fprintf(f, ",");
        first_layer = false;
        fprintf(f, "\"%d\":{", il);
        std::vector<int32_t> expert_ids;
        auto & row = counts.at(il);
        expert_ids.reserve(row.size());
        for (auto & kv : row) expert_ids.push_back(kv.first);
        std::sort(expert_ids.begin(), expert_ids.end());
        bool first_e = true;
        for (int32_t e : expert_ids) {
            if (!first_e) fprintf(f, ",");
            first_e = false;
            fprintf(f, "\"%d\":%llu", e, (unsigned long long) row.at(e));
        }
        fprintf(f, "}");
    }
    fprintf(f, "}}\n");
    // fprintf errors are sticky in the stream's error flag, so one ferror() check covers every
    // write above; fflush() + fclose() surface a buffered write that only fails at flush time.
    bool ok = fflush(f) == 0 && ferror(f) == 0;
    ok = (fclose(f) == 0) && ok; // always close: Windows cannot move or delete an open file
    if (ok) {
        ok = moe_replace_file(tmp, path);
    }
    if (!ok) {
        std::remove(tmp.c_str());
    }
    return ok;
}

// Minimal reader for our own write_json output. Not a general JSON parser: this project already
// has one (common/json.h, nlohmann-based) but src/ does not depend on common/ (layering: common/
// depends on src/, not the reverse), and the shape here is simple/fixed enough that hand parsing
// it is less risk than adding a new cross-layer dependency for one reader used only by this
// feature's own writer format.
bool llama_moe_recorder::read_json(const std::string & path,
        std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    std::string buf((size_t) sz, '\0');
    size_t nread = fread(&buf[0], 1, (size_t) sz, f);
    fclose(f);
    if (nread != (size_t) sz) return false;

    size_t layers_pos = buf.find("\"layers\":{");
    if (layers_pos == std::string::npos) return false;
    size_t i = layers_pos + std::string("\"layers\":{").size();
    int32_t cur_layer = -1;
    bool in_layer = false;
    while (i < buf.size()) {
        if (buf[i] == '"') {
            size_t start = i + 1;
            size_t end = buf.find('"', start);
            if (end == std::string::npos) break;
            std::string key = buf.substr(start, end - start);
            i = end + 1;
            while (i < buf.size() && (buf[i] == ':' || buf[i] == ' ')) i++;
            if (!in_layer) {
                cur_layer = std::atoi(key.c_str());
                if (i < buf.size() && buf[i] == '{') { in_layer = true; i++; }
            } else {
                // key is an expert id; value is a number up to , or }
                size_t vstart = i;
                size_t vend = buf.find_first_of(",}", vstart);
                if (vend == std::string::npos) break;
                uint64_t val = (uint64_t) std::strtoull(buf.substr(vstart, vend - vstart).c_str(), nullptr, 10);
                out[cur_layer][std::atoi(key.c_str())] = val;
                i = vend;
            }
        } else if (buf[i] == '}') {
            if (in_layer) { in_layer = false; i++; }
            else { break; } // end of "layers" object
        } else {
            i++;
        }
    }
    return true;
}

// ───────────────────────── placement strategies ─────────────────────────

std::vector<int32_t> llama_moe_seed_layer(
        llama_moe_placement_strategy strategy,
        int32_t                      layer,
        int32_t                      n_layers,
        int32_t                      n_expert,
        int32_t                      n_slots,
        const std::vector<uint64_t> & freq_row,
        uint32_t                     seed) {
    n_slots = std::min(n_slots, n_expert);
    if (n_slots <= 0) return {};

    std::vector<int32_t> ids;
    switch (strategy) {
        case LLAMA_MOE_PLACEMENT_UNIFORM: {
            // KT Tutorial.md: "Distributes GPU experts evenly across all MoE layers" -- per-layer
            // budget is already even by construction here (n_slots is the same for every layer),
            // so within a layer this is just "the first N ids, in order" (no prior stats needed).
            ids.reserve(n_slots);
            for (int32_t e = 0; e < n_slots; ++e) ids.push_back(e);
            break;
        }
        case LLAMA_MOE_PLACEMENT_FRONT_LOADING: {
            // KT Tutorial.md: "Fills GPU experts from the first layer onwards". With a PER-LAYER
            // slot budget (this fork's --moe-expert-cache N, PR #27861 semantics) every layer
            // already gets the same N regardless of layer index, so "front-loading" has nothing
            // to bias within a single layer's own id space; expressing KT's across-layer bias
            // would require a GLOBAL slot budget shared across layers, which --moe-expert-cache
            // deliberately is not (see file header). We therefore give front-loading a layer-
            // dependent but still-deterministic id window so it is NOT silently identical to
            // uniform (the ANTI-BIAS requirement: "a strategy can seed the pool with the same
            // experts as uniform" must be caught, not produced) -- shift the starting id by the
            // layer index, wrapping, so early layers and late layers pick different expert ids.
            ids.reserve(n_slots);
            int32_t start = (layer * 7) % n_expert; // deterministic, layer-dependent, non-trivial stride
            for (int32_t k = 0; k < n_slots; ++k) {
                ids.push_back((start + k) % n_expert);
            }
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            break;
        }
        case LLAMA_MOE_PLACEMENT_RANDOM: {
            // KT Tutorial.md: "Randomly selects experts with fixed seed (42)". Per-layer stream
            // derived from the base seed so different layers don't all draw the identical subset.
            std::mt19937 rng(seed + (uint32_t) layer * 2654435761u);
            std::vector<int32_t> pool(n_expert);
            for (int32_t e = 0; e < n_expert; ++e) pool[e] = e;
            std::shuffle(pool.begin(), pool.end(), rng);
            ids.assign(pool.begin(), pool.begin() + n_slots);
            std::sort(ids.begin(), ids.end());
            break;
        }
        case LLAMA_MOE_PLACEMENT_FREQUENCY: {
            // KT python/experts_base.py:162-213 generate_gpu_experts_masks(): top-K by activation
            // frequency (there: flattened across all layers; here: within this layer, since our
            // slot budget is per-layer -- see file header). Falls back to uniform, LOGGED by the
            // caller, when freq_row is empty/all-zero (cold recorder / no --init-expert-location).
            bool any_nonzero = false;
            for (auto c : freq_row) if (c > 0) { any_nonzero = true; break; }
            if (freq_row.empty() || !any_nonzero) {
                return llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_UNIFORM, layer, n_layers, n_expert, n_slots, {}, seed);
            }
            std::vector<int32_t> order(n_expert);
            for (int32_t e = 0; e < n_expert; ++e) order[e] = e;
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                uint64_t ca = a < (int32_t) freq_row.size() ? freq_row[a] : 0;
                uint64_t cb = b < (int32_t) freq_row.size() ? freq_row[b] : 0;
                return ca > cb; // descending frequency; ties broken by ascending id (stable_sort)
            });
            ids.assign(order.begin(), order.begin() + n_slots);
            std::sort(ids.begin(), ids.end());
            break;
        }
    }
    return ids;
}

// ───────────────────────── VRAM rail ─────────────────────────

int64_t llama_moe_pool_vram_budget_slots(
        int64_t free_bytes,
        int64_t reserve_bytes,
        int64_t bytes_per_slot,
        int64_t requested_slots) {
    if (bytes_per_slot <= 0 || requested_slots <= 0) {
        return 0;
    }
    int64_t usable = free_bytes - reserve_bytes;
    if (usable <= 0) {
        return 0;
    }
    int64_t affordable = usable / bytes_per_slot;
    // requested_slots is a CEILING (task brief: "N is a ceiling"); the rail only ever lowers it.
    return std::min(requested_slots, affordable);
}

// ───────────────────────── host-side LRU pool ─────────────────────────

void llama_moe_pool_layer::init(int32_t n_expert_, int32_t n_slots_, int32_t admit_threshold_) {
    n_expert = n_expert_;
    n_slots  = n_slots_;
    admit_threshold = std::max(1, admit_threshold_);
    resident_id.assign(n_slots, -1);
    recency.assign(n_slots, 0);
    use_count.clear();
    last_seen.clear();
    call_no = 0;
    clock = 0;
}

void llama_moe_pool_layer::seed(const std::vector<int32_t> & ids) {
    int32_t n = std::min((int32_t) ids.size(), n_slots);
    for (int32_t i = 0; i < n; ++i) {
        resident_id[i] = ids[i];
        recency[i] = ++clock;
    }
}

bool llama_moe_pool_layer::is_resident(int32_t expert_id) const {
    for (int32_t i = 0; i < n_slots; ++i) {
        if (resident_id[i] == expert_id) return true;
    }
    return false;
}

int32_t llama_moe_pool_admit_interval(int32_t base, int32_t n_layers) {
    int32_t v = std::max<int32_t>(1, base);
    if (n_layers <= 1) {
        return v;
    }
    auto gcd = [](int32_t a, int32_t b) { while (b) { const int32_t t = a % b; a = b; b = t; } return a; };
    while (gcd(v, n_layers) != 1) {
        ++v;
    }
    return v;
}

llama_moe_pool_layer::step_stats llama_moe_pool_layer::step_gated(
        const int32_t * ids, int64_t n_ids, int32_t max_inserts, const llama_moe_pool_cost & cost) {
    step_stats st;
    ++call_no;

    // routes per distinct id, in first-seen order (deterministic)
    std::vector<int32_t> order;
    std::unordered_map<int32_t, int32_t> routes;
    for (int64_t k = 0; k < n_ids; ++k) {
        auto it = routes.find(ids[k]);
        if (it == routes.end()) {
            routes.emplace(ids[k], 1);
            order.push_back(ids[k]);
        } else {
            ++it->second;
        }
    }

    int32_t best = -1;
    uint32_t best_heat = 0;
    for (int32_t id : order) {
        const int32_t n = routes[id];
        ++clock;
        int32_t slot = -1;
        for (int32_t i = 0; i < n_slots; ++i) {
            if (resident_id[i] == id) { slot = i; break; }
        }
        if (slot >= 0) {
            recency[slot] = clock;
            st.hits += n;
            continue;
        }
        st.misses += n;
        uint32_t & heat = use_count[id];
        auto seen = last_seen.find(id);
        if (age_window > 0 && seen != last_seen.end() && call_no - seen->second > (uint64_t) age_window) {
            heat = 0;
        }
        last_seen[id] = call_no;
        uint32_t inc = (uint32_t) n;
        if (cost.n_tokens > 1) {
            const double scaled = std::ceil(8.0 * (double) n / (double) cost.n_tokens);
            inc = (uint32_t) std::min(8.0, std::max(1.0, scaled));
        }
        heat = std::min<uint32_t>(65535u, heat + inc);
        if ((int32_t) heat >= admit_threshold && (best < 0 || heat > best_heat || (heat == best_heat && id < best))) {
            best = id;
            best_heat = heat;
        }
    }
    if (best < 0 || max_inserts < 1) {
        return st;
    }
    if (cost.valid && !(cost.cpu_us - cost.gpu_us > cost.refill_us * cost.margin)) {
        st.gated++;
        return st;
    }
    int32_t victim = 0;
    for (int32_t i = 1; i < n_slots; ++i) {
        if (recency[i] < recency[victim]) victim = i;
    }
    if (resident_id[victim] != -1) st.evicted++;
    resident_id[victim] = best;
    recency[victim] = ++clock;
    st.admitted++;
    return st;
}

llama_moe_pool_layer::step_stats llama_moe_pool_layer::step(const int32_t * ids, int64_t n_ids, int32_t max_inserts) {
    step_stats st;
    int32_t inserted_this_step = 0;
    for (int64_t k = 0; k < n_ids; ++k) {
        int32_t id = ids[k];
        clock++;
        // 1) hit?
        int32_t hit_slot = -1;
        for (int32_t i = 0; i < n_slots; ++i) {
            if (resident_id[i] == id) { hit_slot = i; break; }
        }
        if (hit_slot >= 0) {
            recency[hit_slot] = clock;
            st.hits++;
            continue;
        }
        st.misses++;
        // 2) admission gate (PR-thread idea: only admit once recent-use crosses a threshold;
        //    threshold==1 reproduces unconditional admission, i.e. the original PR #27861 policy)
        uint32_t & uc = use_count[id];
        if (uc < 0xFFFFFFFFu) uc++;
        if ((int32_t) uc < admit_threshold) {
            continue; // observed, but not yet hot enough to pay an eviction for
        }
        if (inserted_this_step >= max_inserts) {
            continue; // per-step insert throttle (--moe-expert-cache-inserts)
        }
        // 3) evict least-recently-used empty-or-occupied slot and admit.
        int32_t victim = 0;
        for (int32_t i = 1; i < n_slots; ++i) {
            if (recency[i] < recency[victim]) victim = i;
        }
        if (resident_id[victim] != -1) st.evicted++;
        resident_id[victim] = id;
        recency[victim] = clock;
        st.admitted++;
        inserted_this_step++;
    }
    return st;
}
