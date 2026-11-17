// Retry-After arithmetic and the model-load failure cooldown, for the server's 503 / 429 answers.
//
// Header-only and free of llama / ggml / http dependencies, so it is unit-tested with an injected clock
// (tools/server/tests_cpp/test_server_retry.cpp).
//
// Why: a client that is told "unavailable" with no wait hint retries at its own fixed interval, hammering a server that
// is loading, full, or whose model keeps failing to load. LocalAI computes a real duration for each of those cases and
// sends it as Retry-After; this is the same idea built on this server's own state:
//   * busy (no free slot / admission refused): the wait is estimated from how long finished requests actually took,
//     divided by the number of slots, because with N requests in flight the first slot frees after about 1/N of a
//     request's duration;
//   * loading: no progress signal exists, so the wait grows with the time the load has already taken (a load that has run
//     for 40 s is not about to finish in 1 s);
//   * model load failed: the remaining cooldown, exact.
// RESEARCH-DONE: mudler/LocalAI core/http/app.go:69-149 and pkg/model/loader.go:113-216 (MIT) read; description-level,
// no code copied. See docs/THIRD_PARTY_NOTICES_E_SERVING.md.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

// "try again shortly": the HTTP layer answers 503 and sends Retry-After: retry_after_s
struct server_unavailable_error : std::runtime_error {
    int retry_after_s;
    server_unavailable_error(const std::string & what, int retry_after_s)
        : std::runtime_error(what), retry_after_s(retry_after_s < 1 ? 1 : retry_after_s) {}
};

// whole seconds, at least min_s: Retry-After of 0 invites an immediate retry, which is the thing being prevented
static inline int server_retry_after_s(int64_t ms, int min_s = 1, int max_s = 3600) {
    if (ms <= 0) {
        return min_s;
    }
    const int64_t s = (ms + 999) / 1000;
    return (int) std::min<int64_t>(std::max<int64_t>(s, min_s), max_s);
}

// ---------------------------------------------------------------------------------------------------------------
// estimator
// ---------------------------------------------------------------------------------------------------------------

struct server_retry_estimator {
    mutable std::mutex mtx;
    double  ema_ms  = 0.0;
    int64_t n_obs   = 0;
    static constexpr double ALPHA = 0.2;

    // a request that held a slot from start to finish took this long; a zero or negative duration is a clock glitch
    void observe_request_ms(int64_t ms) {
        if (ms <= 0) {
            return;
        }
        std::lock_guard<std::mutex> lk(mtx);
        ema_ms = n_obs == 0 ? (double) ms : ALPHA * (double) ms + (1.0 - ALPHA) * ema_ms;
        n_obs++;
    }

    // Retry-After for "every slot is busy": with n_slots requests in flight the first one to finish frees a slot after
    // about duration / n_slots. No finished request yet -> 1 s, a floor and not an estimate.
    int busy_retry_after_s(int n_slots) const {
        std::lock_guard<std::mutex> lk(mtx);
        if (n_obs == 0) {
            return 1;
        }
        return server_retry_after_s((int64_t) std::llround(ema_ms / (double) std::max(n_slots, 1)), 1, 60);
    }

    // Retry-After while the model is loading: a quarter of the time already spent loading, 1..10 s
    static int loading_retry_after_s(int64_t elapsed_ms) {
        return server_retry_after_s(elapsed_ms / 4, 1, 10);
    }

    int64_t n_observations() const {
        std::lock_guard<std::mutex> lk(mtx);
        return n_obs;
    }
};

// The one estimator and slot count the HTTP layer reads. `inline` (not `static inline`): one object for the whole program.
inline server_retry_estimator & server_retry_global() {
    static server_retry_estimator e;
    return e;
}
inline std::atomic<int> & server_retry_n_slots() {
    static std::atomic<int> n(1);
    return n;
}
// milliseconds since this function was first called (the server calls it at start-up, so it reads as uptime)
inline int64_t server_retry_uptime_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------------------------------------------
// per-model load-failure cooldown
// ---------------------------------------------------------------------------------------------------------------

// A model whose load keeps failing (corrupt file, impossible context) must not respawn a loader process on every
// request that polls it. After the nth consecutive failure the model is refused for min(base * 2^(n-1), max); a success
// clears it. base <= 0 means the feature is not in use (every query answers "free to load"). The clock is injected.
struct server_load_cooldown {
    struct entry {
        int     n_fail = 0;
        int64_t until_ms = 0;
    };

    int64_t base_ms = 10 * 1000;
    int64_t max_ms  = 300 * 1000;

    mutable std::mutex mtx;
    std::map<std::string, entry> models;

    bool enabled() const { return base_ms > 0; }

    // cooldown length after the nth consecutive failure (n >= 1), overflow-safe
    int64_t cooldown_ms(int n_fail) const {
        if (base_ms <= 0 || n_fail <= 0) {
            return 0;
        }
        int64_t v = base_ms;
        for (int i = 1; i < n_fail; i++) {
            if (v >= max_ms) {
                break;
            }
            v *= 2;
        }
        return std::min(v, max_ms);
    }

    // returns the cooldown just armed (0 when the feature is not in use)
    int64_t on_failure(const std::string & model, int64_t now_ms) {
        if (!enabled()) {
            return 0;
        }
        std::lock_guard<std::mutex> lk(mtx);
        entry & e = models[model];
        e.n_fail++;
        const int64_t c = cooldown_ms(e.n_fail);
        e.until_ms = now_ms + c;
        return c;
    }

    void on_success(const std::string & model) {
        std::lock_guard<std::mutex> lk(mtx);
        models.erase(model);
    }

    // an explicit operator action (POST /models/load) bypasses and clears the cooldown
    void clear(const std::string & model) { on_success(model); }

    // ms the model is still refused for; 0 = free to load
    int64_t remaining_ms(const std::string & model, int64_t now_ms) const {
        if (!enabled()) {
            return 0;
        }
        std::lock_guard<std::mutex> lk(mtx);
        auto it = models.find(model);
        if (it == models.end()) {
            return 0;
        }
        return std::max<int64_t>(0, it->second.until_ms - now_ms);
    }

    int n_failures(const std::string & model) const {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = models.find(model);
        return it == models.end() ? 0 : it->second.n_fail;
    }
};
