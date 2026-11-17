// Per-client request rate limit (--rate-limit CALLS/PERIOD, --rate-limit-key addr|key) for the HTTP layer.
//
// Header-only and free of llama / ggml / http dependencies, so it is unit-tested with an injected clock
// (tools/server/tests_cpp/test_server_ratelimit.cpp).
//
// Why: one client (a runaway agent loop, a misconfigured poller, a script retrying without backoff) can fill every
// slot and the request queue, and the admission ceilings (--max-queued-requests) then refuse EVERY client alike. A
// per-client budget refuses the client that is over its share and leaves the others untouched.
//
// Semantics: fixed window per client, anchored at the client's first counted request; at most `calls` requests are
// let through per window, the next ones are answered 429 until the window ends, and Retry-After is the time left in
// that window (whole seconds, at least 1). A refused request does not extend the window.
// SOURCE-READ: mlflow/mlflow@7d2713204b mlflow/gateway/app.py:277-279 (Limiter keyed by get_remote_address),
//   :176-180 (limit string "{calls}/{renewal_period}"), mlflow/gateway/config.py:493-497,560-572 (Limit{calls, key,
//   renewal_period}, validated by limits.parse) -- Apache-2.0;
//   laurentS/slowapi@d3442b2e1 slowapi/extension.py:247 (default strategy "fixed-window"), :76-87 (429 handler),
//   :378-413 (Retry-After = window reset - now, sent only with headers_enabled, which MLflow does not set) -- MIT;
//   alisaifee/limits@0492c4fbe limits/strategies.py:139-191 (fixed window: allowed while count <= amount, reset = the
//   window's expiry), limits/storage/memory.py:89-103 (the expiry is set on the FIRST hit: window anchored there) -- MIT.
//   Behaviour read; own code, no code copied. Unlike MLflow as configured, the 429 here always carries Retry-After.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

struct server_rate_limit {
    // configuration; calls <= 0 or period_ms <= 0 = the feature is not in use (every request passes, nothing is kept)
    int64_t calls     = 0;
    int64_t period_ms = 0;

    // above this many tracked clients, a counted request first drops the clients whose window has ended
    size_t sweep_above = 4096;

    struct window {
        int64_t start_ms = 0;
        int64_t count    = 0;
    };

    struct verdict {
        bool    allowed        = true;
        int64_t limit          = 0;  // calls per window (X-RateLimit-Limit)
        int64_t remaining      = 0;  // calls left in this window after this one (X-RateLimit-Remaining)
        int64_t reset_ms       = 0;  // ms until this client's window ends (X-RateLimit-Reset, and Retry-After when refused)
    };

    bool enabled() const { return calls > 0 && period_ms > 0; }

    // count one request from `client` at `now_ms` and say whether it may proceed
    verdict hit(const std::string & client, int64_t now_ms) {
        verdict v;
        if (!enabled()) {
            return v;
        }
        v.limit = calls;
        std::lock_guard<std::mutex> lk(mtx);
        if (clients.size() > sweep_above) {
            sweep_locked(now_ms);
        }
        window & w = clients[client];
        if (w.count == 0 || now_ms - w.start_ms >= period_ms || now_ms < w.start_ms) {
            // first request of a new window (a clock that went backwards also starts a new one rather than locking out)
            w.start_ms = now_ms;
            w.count    = 0;
        }
        v.reset_ms = std::max<int64_t>(0, w.start_ms + period_ms - now_ms);
        if (w.count >= calls) {
            v.allowed   = false;
            v.remaining = 0;
            return v; // a refused request is not counted and does not move the window
        }
        w.count++;
        v.remaining = calls - w.count;
        return v;
    }

    size_t n_clients() const {
        std::lock_guard<std::mutex> lk(mtx);
        return clients.size();
    }

    // Parse "CALLS/PERIOD": PERIOD is a duration ("1s", "500ms", "1m30s", a bare number = seconds) or a unit word with an
    // optional count ("second", "minute", "hour", "day", "10 minutes"), as in MLflow's renewal_period. `parse_duration`
    // is the server's duration parser (string_parse_duration_ms), injected so this header stays dependency-free.
    static bool parse(const std::string & spec, int64_t & calls_out, int64_t & period_ms_out,
                      const std::function<bool(const std::string &, int64_t &)> & parse_duration) {
        const size_t slash = spec.find('/');
        if (slash == std::string::npos) {
            return false;
        }
        const std::string n_str = trim(spec.substr(0, slash));
        std::string p_str = trim(spec.substr(slash + 1));
        if (n_str.empty() || p_str.empty()) {
            return false;
        }
        for (char c : n_str) {
            if (!std::isdigit((unsigned char) c)) {
                return false;
            }
        }
        const long long n = std::strtoll(n_str.c_str(), nullptr, 10);
        if (n <= 0) {
            return false;
        }

        int64_t period_ms = 0;
        // "<count> <unit word>" or "<unit word>"
        {
            size_t i = 0;
            while (i < p_str.size() && std::isdigit((unsigned char) p_str[i])) {
                i++;
            }
            const std::string count_str = p_str.substr(0, i);
            std::string word = trim(p_str.substr(i));
            for (auto & c : word) {
                c = (char) std::tolower((unsigned char) c);
            }
            if (word.size() > 1 && word.back() == 's' && word != "s" && word != "ms") {
                word.pop_back(); // "minutes" -> "minute"
            }
            int64_t unit_ms = 0;
            if      (word == "second") { unit_ms = 1000; }
            else if (word == "minute") { unit_ms = 60 * 1000; }
            else if (word == "hour")   { unit_ms = 3600 * 1000; }
            else if (word == "day")    { unit_ms = 86400 * 1000LL; }
            if (unit_ms > 0) {
                const long long k = count_str.empty() ? 1 : std::strtoll(count_str.c_str(), nullptr, 10);
                if (k <= 0) {
                    return false;
                }
                period_ms = (int64_t) k * unit_ms;
            }
        }
        if (period_ms == 0) {
            int64_t ms = 0;
            if (!parse_duration(p_str, ms) || ms <= 0) {
                return false;
            }
            period_ms = ms;
        }
        calls_out     = (int64_t) n;
        period_ms_out = period_ms;
        return true;
    }

private:
    mutable std::mutex mtx;
    std::unordered_map<std::string, window> clients;

    void sweep_locked(int64_t now_ms) {
        for (auto it = clients.begin(); it != clients.end();) {
            if (now_ms - it->second.start_ms >= period_ms || now_ms < it->second.start_ms) {
                it = clients.erase(it);
            } else {
                ++it;
            }
        }
    }

    static std::string trim(const std::string & s) {
        const size_t b = s.find_first_not_of(" \t");
        if (b == std::string::npos) {
            return "";
        }
        const size_t e = s.find_last_not_of(" \t");
        return s.substr(b, e - b + 1);
    }
};

// The client a request is counted against. mode "key": the presented API key (hashed, so the table holds no secret),
// falling back to the remote address when the request carries none; mode "addr" (default): the remote address.
static inline std::string server_rate_limit_client(const std::string & mode, const std::string & remote_addr,
                                                   const std::string & presented_key) {
    if (mode == "key" && !presented_key.empty()) {
        return "k:" + std::to_string(std::hash<std::string>{}(presented_key));
    }
    return "a:" + remote_addr;
}
