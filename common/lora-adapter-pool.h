#pragma once

// On-demand LoRA adapter pool (S69-LORAX-PICK-ON-DEMAND-ADAPTER-POOL-2026-09-29-01).
//
// Design idea (bounded active set, byte budget, round-robin time slices) follows predibase/lorax
// router/src/queue.rs (Apache-2.0); this is an independent implementation, no code copied.
//
// NOT wired into the server: the fork applies LoRA per llama_context, not per sequence (see
// docs/agent-reports/lorax-adapter-pool.md), so the pool has no consumer yet. Header-only, no llama.h
// dependency; the adapter itself is an opaque std::shared_ptr<void> produced by a loader callback.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class lora_adapter_pool {
public:
    struct loaded {
        std::shared_ptr<void> handle;   // opaque adapter object; destroyed when evicted and unreferenced
        size_t                bytes = 0;
    };

    // Fills `out` and returns true, or returns false on failure. Called with the pool lock held, so it
    // must not call back into the pool. Loads are therefore serialised.
    using loader_fn = std::function<bool(const std::string & id, loaded & out)>;
    // Optional: called after an entry is evicted (handle still valid). Same lock caveat.
    using evict_fn  = std::function<void(const std::string & id)>;

    enum class status {
        ok,
        load_failed,   // loader returned false
        over_budget,   // the adapter alone exceeds max_bytes
        no_room,       // fits the budget alone, but the pinned (in-use) adapters leave no room
    };

    struct acquired {
        status                st     = status::load_failed;
        std::shared_ptr<void> handle;
        bool                  loaded_now = false;
    };

    // max_adapters == 0 or max_bytes == 0 means "no limit" on that axis.
    lora_adapter_pool(size_t max_adapters, size_t max_bytes, loader_fn loader, evict_fn on_evict = nullptr)
        : max_adapters_(max_adapters), max_bytes_(max_bytes), loader_(std::move(loader)), on_evict_(std::move(on_evict)) {}

    // Pins the adapter (refcount++) and returns it, loading and evicting LRU unreferenced entries as needed.
    acquired acquire(const std::string & id) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(id);
        if (it != map_.end()) {
            it->second.refs++;
            touch(it->second);
            return { status::ok, it->second.handle, false };
        }
        loaded l;
        if (!loader_ || !loader_(id, l)) {
            return { status::load_failed, nullptr, false };
        }
        if (max_bytes_ && l.bytes > max_bytes_) {
            return { status::over_budget, nullptr, false };
        }
        // evict LRU unreferenced entries until the newcomer fits both axes
        while (!fits(l.bytes)) {
            if (!evict_one_lru()) {
                return { status::no_room, nullptr, false };
            }
        }
        lru_.push_back(id);
        entry e;
        e.handle = l.handle;
        e.bytes  = l.bytes;
        e.refs   = 1;
        e.pos    = std::prev(lru_.end());
        bytes_  += l.bytes;
        map_.emplace(id, std::move(e));
        return { status::ok, l.handle, true };
    }

    // refcount--; returns false if the id is unknown or was not pinned.
    bool release(const std::string & id) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(id);
        if (it == map_.end() || it->second.refs == 0) {
            return false;
        }
        it->second.refs--;
        touch(it->second);
        return true;
    }

    size_t size() const        { std::lock_guard<std::mutex> lk(mu_); return map_.size(); }
    size_t bytes() const       { std::lock_guard<std::mutex> lk(mu_); return bytes_; }
    bool   contains(const std::string & id) const { std::lock_guard<std::mutex> lk(mu_); return map_.count(id) != 0; }
    size_t refs(const std::string & id) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(id);
        return it == map_.end() ? 0 : it->second.refs;
    }
    // resident ids, least recently used first
    std::vector<std::string> lru_order() const {
        std::lock_guard<std::mutex> lk(mu_);
        return std::vector<std::string>(lru_.begin(), lru_.end());
    }

private:
    struct entry {
        std::shared_ptr<void>          handle;
        size_t                         bytes = 0;
        size_t                         refs  = 0;
        std::list<std::string>::iterator pos;
    };

    bool fits(size_t add_bytes) const {
        if (max_adapters_ && map_.size() + 1 > max_adapters_) return false;
        if (max_bytes_    && bytes_ + add_bytes > max_bytes_) return false;
        return true;
    }
    void touch(entry & e) { lru_.splice(lru_.end(), lru_, e.pos); }
    bool evict_one_lru() {
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            auto m = map_.find(*it);
            if (m->second.refs == 0) {
                const std::string id = *it;
                bytes_ -= m->second.bytes;
                if (on_evict_) on_evict_(id);
                map_.erase(m);
                lru_.erase(it);
                return true;
            }
        }
        return false;
    }

    mutable std::mutex                       mu_;
    size_t                                   max_adapters_;
    size_t                                   max_bytes_;
    loader_fn                                loader_;
    evict_fn                                 on_evict_;
    std::unordered_map<std::string, entry>   map_;
    std::list<std::string>                   lru_;     // front = least recently used
    size_t                                   bytes_ = 0;
};

// Round-robin scheduler across adapters with waiting requests. One adapter is "current" for a time slice
// (slice_quanta picks and/or slice_ms of the injected clock); when the slice ends, or the current adapter
// has nothing waiting, the next adapter in arrival ring order with waiting requests becomes current.
// If only the current adapter has work, it keeps going (new slice). So one busy adapter cannot starve others.
class lora_fair_scheduler {
public:
    using clock_fn = std::function<uint64_t()>; // milliseconds

    lora_fair_scheduler(size_t slice_quanta, uint64_t slice_ms = 0, clock_fn clock = nullptr)
        : slice_quanta_(slice_quanta ? slice_quanta : 1), slice_ms_(slice_ms), clock_(std::move(clock)) {}

    void enqueue(const std::string & adapter, uint64_t request) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = queues_.find(adapter);
        if (it == queues_.end()) {
            ring_.push_back(adapter);
            it = queues_.emplace(adapter, std::deque<uint64_t>()).first;
        }
        it->second.push_back(request);
    }

    // Next (adapter, request) to run, FIFO within an adapter; nullopt if nothing waits.
    std::optional<std::pair<std::string, uint64_t>> pick() {
        std::lock_guard<std::mutex> lk(mu_);
        if (ring_.empty()) return std::nullopt;
        const uint64_t now = clock_ ? clock_() : 0;
        const bool cur_ok = has_cur_ && !queues_[cur_].empty();
        const bool expired = cur_ok && (used_ >= slice_quanta_ || (slice_ms_ && clock_ && now - slice_start_ >= slice_ms_));
        std::string next = cur_;
        if (!cur_ok || expired) {
            next = next_with_work(has_cur_ ? cur_ : std::string(), has_cur_);
            if (next.empty()) return std::nullopt;
            if (!has_cur_ || next != cur_ || expired) { used_ = 0; slice_start_ = now; }
            cur_ = next; has_cur_ = true;
        }
        auto & q = queues_[cur_];
        const uint64_t r = q.front();
        q.pop_front();
        used_++;
        std::pair<std::string, uint64_t> out(cur_, r);
        prune();
        return out;
    }

    size_t waiting() const {
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = 0;
        for (auto & kv : queues_) n += kv.second.size();
        return n;
    }

private:
    // first adapter after `from` in ring order (wrapping, `from` itself last) that has work
    std::string next_with_work(const std::string & from, bool have_from) {
        size_t start = 0;
        if (have_from) {
            for (size_t i = 0; i < ring_.size(); i++) if (ring_[i] == from) { start = i + 1; break; }
        }
        for (size_t k = 0; k < ring_.size(); k++) {
            const std::string & a = ring_[(start + k) % ring_.size()];
            if (!queues_[a].empty()) return a;
        }
        return std::string();
    }
    void prune() {
        // drop empty non-current adapters from the ring so it does not grow without bound
        for (size_t i = 0; i < ring_.size();) {
            if (queues_[ring_[i]].empty() && !(has_cur_ && ring_[i] == cur_)) {
                queues_.erase(ring_[i]);
                ring_.erase(ring_.begin() + i);
            } else i++;
        }
    }

    mutable std::mutex                                       mu_;
    size_t                                                   slice_quanta_;
    uint64_t                                                 slice_ms_;
    clock_fn                                                 clock_;
    std::vector<std::string>                                 ring_;
    std::unordered_map<std::string, std::deque<uint64_t>>    queues_;
    std::string                                              cur_;
    bool                                                     has_cur_ = false;
    size_t                                                   used_ = 0;
    uint64_t                                                 slice_start_ = 0;
};
