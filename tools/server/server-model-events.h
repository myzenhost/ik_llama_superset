#pragma once

// Router model lifecycle: why a model stopped, and a bounded ring of recent status changes.
//
// Row S69-ENGREV-TFSERVING-LIFECYCLE-EVENT-RING-2026-10-27-01. Design idea from tensorflow/serving
// core/servable_state_monitor.h (a bounded event log beside the latest state); Apache-2.0, idea only,
// SOURCE-READ: tensorflow/serving core/servable_state_monitor.h:15-272, no code copied.
// Before this the router kept only exit_code for a failed child, so "reaped by keep-alive" and "crashed"
// looked the same. Header-only so tests_cpp can drive it without the router.

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

enum server_stop_reason {
    SERVER_STOP_NONE = 0,   // not stopped by us: the child died on its own (crash) or has not stopped
    SERVER_STOP_IDLE_TTL,   // keep-alive expired
    SERVER_STOP_LRU_EVICT,  // evicted to make room for another model
    SERVER_STOP_MANUAL,     // POST /models/unload
    SERVER_STOP_RELOAD,     // stopped so a preset or num_ctx reload can respawn it
    SERVER_STOP_SHUTDOWN,   // router is exiting
};

static inline const char * server_stop_reason_str(server_stop_reason r) {
    switch (r) {
        case SERVER_STOP_IDLE_TTL:  return "idle-ttl";
        case SERVER_STOP_LRU_EVICT: return "lru-evict";
        case SERVER_STOP_MANUAL:    return "manual";
        case SERVER_STOP_RELOAD:    return "reload";
        case SERVER_STOP_SHUTDOWN:  return "shutdown";
        default:                    return "";
    }
}

// Reason reported once the child has exited. A stop we asked for keeps its reason; otherwise the child
// died by itself: out of memory while loading, any other non-zero exit, or a clean exit nobody asked for.
static inline std::string server_exit_reason(server_stop_reason asked, bool load_oom, int exit_code) {
    if (asked != SERVER_STOP_NONE) {
        return server_stop_reason_str(asked);
    }
    if (load_oom) {
        return "load-oom";
    }
    return exit_code != 0 ? "crash" : "exited";
}

struct server_model_event {
    uint64_t    seq = 0;       // 1-based, never reused, so a client can resume with ?since=<last seq>
    int64_t     ts_ms = 0;     // wall-clock milliseconds since the epoch
    std::string model;
    std::string from;
    std::string to;
    std::string reason;        // set on a transition to unloaded, empty otherwise
    int         exit_code = 0;
};

class server_model_event_ring {
public:
    static constexpr size_t CAPACITY = 256;

    explicit server_model_event_ring(size_t capacity = CAPACITY) : cap(capacity == 0 ? 1 : capacity) {}

    uint64_t record(int64_t ts_ms, const std::string & model, const std::string & from, const std::string & to,
                    const std::string & reason, int exit_code) {
        std::lock_guard<std::mutex> lk(mtx);
        server_model_event ev;
        ev.seq       = ++last_seq;
        ev.ts_ms     = ts_ms;
        ev.model     = model;
        ev.from      = from;
        ev.to        = to;
        ev.reason    = reason;
        ev.exit_code = exit_code;
        events.push_back(std::move(ev));
        while (events.size() > cap) {
            events.pop_front();
        }
        return last_seq;
    }

    // events with seq > since, oldest first; since = 0 returns everything still held
    std::vector<server_model_event> since(uint64_t after) const {
        std::lock_guard<std::mutex> lk(mtx);
        std::vector<server_model_event> out;
        for (const auto & ev : events) {
            if (ev.seq > after) {
                out.push_back(ev);
            }
        }
        return out;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mtx);
        return events.size();
    }

    uint64_t newest_seq() const {
        std::lock_guard<std::mutex> lk(mtx);
        return last_seq;
    }

private:
    size_t cap;
    mutable std::mutex mtx;
    std::deque<server_model_event> events;
    uint64_t last_seq = 0;
};
