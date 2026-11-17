// lora-pool.h -- admission / eviction POLICY for an on-demand LoRA adapter pool: bounded memory budget, least-recently-used eviction of IDLE adapters only,
// fail-fast memo for adapters that failed to load, refusal to unload an adapter that has requests in flight.  Header-only, no llama dependency: the
// caller does the actual llama_adapter_lora_init / free and tells the pool what happened (so the policy is testable without a model).
// Row S69-LORAX-PICK-ON-DEMAND-ADAPTER-POOL-2026-09-29-01 (step 1: the pool policy; the server wiring -- load/unload endpoints, per-request lookup by
// name, queueing a request behind a load -- lives in tools/server and is the follow-up of that row).
//
// SPEC READ FROM (behaviour only, own implementation; Apache-2.0, predibase/lorax @ a7394ba3): router/src/loader.rs:1-343 (load command loop and the per-adapter
// error memo, :122-141), router/src/queue.rs:100-430 (AdapterQueuesState: active set bounded by a memory budget, learned per-adapter cost, idle adapters
// offloaded, oldest-active selection so one busy adapter cannot starve the pending ones).
// Not taken: the time-slice cycle (adapter_cycle_time_s) -- this pool evicts on pressure only, because the fork's server has one decode loop and no per-adapter
// queues yet; fairness belongs to the scheduler row.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

class common_lora_pool {
public:
    enum state { STATE_REGISTERED, STATE_READY, STATE_ERRORED };

    enum plan_status {
        PLAN_READY,        // already loaded: acquire it
        PLAN_LOAD,         // load it; `evict` lists the idle adapters to free first (may be empty)
        PLAN_TOO_BIG,      // larger than the whole budget: can never be loaded
        PLAN_BUSY,         // would fit if in-flight adapters finished: wait and retry
        PLAN_ERRORED,      // failed before: fail fast with the remembered message
        PLAN_UNKNOWN,      // not registered
    };

    struct plan {
        plan_status              status = PLAN_UNKNOWN;
        std::vector<std::string> evict;     // ids, in the order they must be freed
        std::string              error;     // PLAN_ERRORED
    };

    struct entry {
        std::string path;
        size_t      bytes     = 0;
        state       st        = STATE_REGISTERED;
        uint64_t    last_used = 0;
        int         in_flight = 0;
        std::string error;
    };

    explicit common_lora_pool(size_t budget_bytes) : budget(budget_bytes) {}

    // metadata only (no weights loaded); re-registering an id with another path or size replaces it unless it is loaded or in flight
    bool add(const std::string & id, const std::string & path, size_t bytes) {
        auto it = entries.find(id);
        if (it != entries.end() && (it->second.st == STATE_READY || it->second.in_flight > 0)) {
            return false;
        }
        entry e;
        e.path  = path;
        e.bytes = bytes;
        entries[id] = e;
        return true;
    }

    plan plan_load(const std::string & id) const {
        plan p;
        auto it = entries.find(id);
        if (it == entries.end()) {
            return p;
        }
        const entry & e = it->second;
        if (e.st == STATE_READY) {
            p.status = PLAN_READY;
            return p;
        }
        if (e.st == STATE_ERRORED) {
            p.status = PLAN_ERRORED;
            p.error  = e.error;
            return p;
        }
        if (e.bytes > budget) {
            p.status = PLAN_TOO_BIG;
            return p;
        }
        size_t need = used() + e.bytes;
        if (need <= budget) {
            p.status = PLAN_LOAD;
            return p;
        }
        // evict idle READY adapters, least recently used first, until it fits
        std::vector<std::pair<uint64_t, std::string>> idle;
        for (const auto & kv : entries) {
            if (kv.second.st == STATE_READY && kv.second.in_flight == 0) {
                idle.emplace_back(kv.second.last_used, kv.first);
            }
        }
        std::sort(idle.begin(), idle.end());
        for (const auto & c : idle) {
            p.evict.push_back(c.second);
            need -= entries.at(c.second).bytes;
            if (need <= budget) {
                p.status = PLAN_LOAD;
                return p;
            }
        }
        p.evict.clear();
        p.status = PLAN_BUSY;
        return p;
    }

    // the caller finished freeing `id` (after a PLAN_LOAD eviction list or an explicit unload)
    void mark_evicted(const std::string & id) {
        auto it = entries.find(id);
        if (it != entries.end() && it->second.st == STATE_READY && it->second.in_flight == 0) {
            it->second.st = STATE_REGISTERED;
        }
    }
    void mark_loaded(const std::string & id) {
        auto it = entries.find(id);
        if (it != entries.end()) {
            it->second.st        = STATE_READY;
            it->second.last_used = ++clock;
        }
    }
    void mark_failed(const std::string & id, const std::string & message) {
        auto it = entries.find(id);
        if (it != entries.end()) {
            it->second.st    = STATE_ERRORED;
            it->second.error = message;
        }
    }
    // forget a remembered failure (the file was replaced): the next plan_load tries again
    void clear_error(const std::string & id) {
        auto it = entries.find(id);
        if (it != entries.end() && it->second.st == STATE_ERRORED) {
            it->second.st = STATE_REGISTERED;
            it->second.error.clear();
        }
    }

    // a request starts / ends using a READY adapter
    bool acquire(const std::string & id) {
        auto it = entries.find(id);
        if (it == entries.end() || it->second.st != STATE_READY) {
            return false;
        }
        it->second.in_flight++;
        it->second.last_used = ++clock;
        return true;
    }
    void release(const std::string & id) {
        auto it = entries.find(id);
        if (it != entries.end() && it->second.in_flight > 0) {
            it->second.in_flight--;
        }
    }

    // explicit unload: refused while requests are in flight; on success the caller frees the weights
    bool can_unload(const std::string & id) const {
        auto it = entries.find(id);
        return it != entries.end() && it->second.st == STATE_READY && it->second.in_flight == 0;
    }

    size_t used() const {
        size_t u = 0;
        for (const auto & kv : entries) {
            if (kv.second.st == STATE_READY) {
                u += kv.second.bytes;
            }
        }
        return u;
    }
    size_t           budget_bytes() const { return budget; }
    const entry *    find(const std::string & id) const {
        auto it = entries.find(id);
        return it == entries.end() ? nullptr : &it->second;
    }
    std::vector<std::string> ready_ids() const {
        std::vector<std::string> r;
        for (const auto & kv : entries) {
            if (kv.second.st == STATE_READY) {
                r.push_back(kv.first);
            }
        }
        return r;
    }

private:
    size_t                     budget;
    uint64_t                   clock = 0;
    std::map<std::string, entry> entries;
};
