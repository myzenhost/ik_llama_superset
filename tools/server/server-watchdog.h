#pragma once

// Forward-progress watchdog for the server decode loop.
//
// Row S69-ENGREV-SGLANG-FORWARD-WATCHDOG-2026-10-27-01. SOURCE-READ: SGLang (Apache-2.0) python/sglang/srt/utils/watchdog.py:124-164 and
// server_arguments.mdx watchdog / soft-watchdog flags, as specified in temp/research/findings/S69_ENGINE_REVIEW_sglang2_DEEP_2026-10-27.md.
// Behaviour taken: a heartbeat counter (forward_ct) must advance within a timeout while a batch is active; SOFT = report only, HARD = the
// process ends after a grace period so a supervisor restarts it. Written from that description, no code copied.
//
// This is the pure state machine: every method takes the clock as an argument, so tests drive it without sleeping. The server owns one
// instance and a poller thread that calls poll() about once a second (tools/server/server-context.cpp, --forward-watchdog-*).
//
// Episode rule: a stall is the time since the last beat while active. A soft/hard verdict is returned ONCE per episode; the next beat()
// (or end()) closes the episode, so a slow but alive decode logs once per stall and not once per poll.

#include <atomic>
#include <cstdint>

class server_fwd_watchdog {
public:
    enum class verdict { ok, soft_fire, hard_fire };

    // 0 (or negative) disables that level
    void configure(int64_t soft_ms, int64_t hard_ms, int64_t grace_ms = 5000) {
        soft_ms_  = soft_ms  > 0 ? soft_ms  : 0;
        hard_ms_  = hard_ms  > 0 ? hard_ms  : 0;
        grace_ms_ = grace_ms > 0 ? grace_ms : 0;
    }

    bool enabled() const { return soft_ms_ > 0 || hard_ms_ > 0; }

    int64_t grace_ms() const { return grace_ms_; }

    // a batch is about to run
    void begin(int64_t now_ms) {
        last_beat_ms_.store(now_ms, std::memory_order_relaxed);
        soft_fired_.store(false, std::memory_order_relaxed);
        hard_fired_.store(false, std::memory_order_relaxed);
        active_.store(true, std::memory_order_release);
    }

    // a forward pass returned: progress
    void beat(int64_t now_ms) {
        forward_ct_.fetch_add(1, std::memory_order_relaxed);
        last_beat_ms_.store(now_ms, std::memory_order_relaxed);
        soft_fired_.store(false, std::memory_order_relaxed);
        hard_fired_.store(false, std::memory_order_relaxed);
        stalled_.store(false, std::memory_order_relaxed);
    }

    // no batch in flight: an idle server never fires
    void end() {
        active_.store(false, std::memory_order_release);
        soft_fired_.store(false, std::memory_order_relaxed);
        hard_fired_.store(false, std::memory_order_relaxed);
        stalled_.store(false, std::memory_order_relaxed);
    }

    // called by the poller thread
    verdict poll(int64_t now_ms) {
        if (!enabled() || !active_.load(std::memory_order_acquire)) {
            return verdict::ok;
        }
        const int64_t idle = now_ms - last_beat_ms_.load(std::memory_order_relaxed);
        if (hard_ms_ > 0 && idle >= hard_ms_ && !hard_fired_.exchange(true)) {
            stalled_.store(true, std::memory_order_relaxed);
            return verdict::hard_fire;
        }
        if (soft_ms_ > 0 && idle >= soft_ms_ && !soft_fired_.exchange(true)) {
            stalled_.store(true, std::memory_order_relaxed);
            return verdict::soft_fire;
        }
        return verdict::ok;
    }

    // true from a fired verdict until progress resumes (for /health style reporting)
    bool     stalled()    const { return stalled_.load(std::memory_order_relaxed); }
    uint64_t forward_ct() const { return forward_ct_.load(std::memory_order_relaxed); }
    int64_t  stall_ms(int64_t now_ms) const {
        return active_.load(std::memory_order_acquire) ? now_ms - last_beat_ms_.load(std::memory_order_relaxed) : 0;
    }

private:
    int64_t soft_ms_  = 0;
    int64_t hard_ms_  = 0;
    int64_t grace_ms_ = 5000;

    std::atomic<bool>     active_{false};
    std::atomic<bool>     soft_fired_{false};
    std::atomic<bool>     hard_fired_{false};
    std::atomic<bool>     stalled_{false};
    std::atomic<int64_t>  last_beat_ms_{0};
    std::atomic<uint64_t> forward_ct_{0};
};

// Process-wide, read-only view of the server's watchdog for GET /health, which must not touch the server context (it answers
// while the server sleeps, and the routes deliberately shadow ctx_server there). The context publishes its watchdog when the
// watchdog is enabled and withdraws it before the watchdog is destroyed. Row S69-ENGREV-TRNRT-CHECK-HEALTH-TRUTHFUL-ON-FAILED-
// ENGINE-2026-11-13-01: an orphaned router answered /health for 19 h while every request failed; a decode that is alive but
// stuck (a batch in flight, no step finished for the soft timeout) must not report ok either.
struct server_fwd_watchdog_health {
    std::atomic<const server_fwd_watchdog *> wd{nullptr};

    // > 0: the decode loop is stalled now, for that many ms; 0: healthy, or no watchdog published
    int64_t stalled_ms(int64_t now_ms) const {
        const server_fwd_watchdog * w = wd.load(std::memory_order_acquire);
        return (w != nullptr && w->stalled()) ? (w->stall_ms(now_ms) > 0 ? w->stall_ms(now_ms) : 1) : 0;
    }
};

inline server_fwd_watchdog_health & server_fwd_watchdog_health_global() {
    static server_fwd_watchdog_health h;
    return h;
}
