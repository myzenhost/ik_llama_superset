#pragma once

// Predictive load shedding (--shed-max-latency-ms): an online least-squares fit of how long a request takes to serve as a
// function of its size, and the rule that drops the oldest queued request when it can no longer meet its latency budget.
//
// Behaviour follows BentoML's dynamic-batching dispatcher (Apache-2.0, read for behaviour, no code copied):
//   SOURCE-READ: bentoml@517b343 src/bentoml/_internal/marshal/dispatcher.py:40-96 (Optimizer: service time = a*n + b, fitted by
//   least squares over the last N_KEPT_SAMPLE = 50 samples, the first N_SKIPPED_SAMPLE = 2 skipped as inaccurate, a floored at 1e-6 and
//   b at 0, refit allowed by a token bucket with rate 1/5 per second = at most once per 5 s) and :286-313 (while n > 1 and
//   wait_of_first + a*n + b >= max_latency, drop the OLDEST queued job; a lone job is dropped only once its wait alone reaches the budget).
// Deliberate differences, all forced by serving an LLM instead of a batch:
//   * n is a TOKEN count (prompt + n_predict of the request being judged), not the number of jobs in the batch;
//   * the fit never acts before N_MIN samples exist (BentoML warms the model up with synthetic requests first; there is nothing to
//     synthesise here, and its cold-start constants would be wrong in token units);
//   * when every sample has the same size the slope is not identifiable (lstsq would return the minimum-norm solution); the fit
//     then falls back to a = mean(ms)/mean(tokens), b = 0.
// Pure arithmetic, no server state: tools/server/tests_cpp style unit tests can include it on its own.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>

struct server_shed_fit {
    static constexpr size_t  N_KEPT      = 50;       // samples kept for the fit
    static constexpr size_t  N_SKIP      = 2;        // first samples ignored
    static constexpr size_t  N_MIN       = 5;        // samples needed before the fit is trusted
    static constexpr int64_t REFIT_US    = 5000000;  // at most one refit per this long (BentoML: token bucket 1/5 per second)
    static constexpr double  A_FLOOR_MS  = 1e-3;     // BentoML: max(1e-6 s, a) -> 1e-3 ms

    struct sample { int64_t tokens; double ms; };

    std::deque<sample> samples;
    size_t  n_seen    = 0;      // every add() call, skipped ones included
    int64_t t_refit   = 0;      // time of the last refit (us, caller's clock); 0 = never
    bool    fitted    = false;
    double  a_ms      = 0.0;    // ms per token
    double  b_ms      = 0.0;    // fixed ms per request

    // record one finished request: its size in tokens and the time it took from slot start to done
    void add(int64_t tokens, double ms, int64_t now_us) {
        if (tokens <= 0 || ms < 0) {
            return;
        }
        n_seen++;
        if (n_seen <= N_SKIP) {
            return;
        }
        samples.push_back({tokens, ms});
        while (samples.size() > N_KEPT) {
            samples.pop_front();
        }
        if (samples.size() >= N_MIN && (!fitted || now_us - t_refit >= REFIT_US)) {
            refit(now_us);
        }
    }

    bool ready() const { return fitted; }

    // predicted service time of a request of `tokens` tokens, in ms
    double predict_ms(int64_t tokens) const { return a_ms * (double) tokens + b_ms; }

    void refit(int64_t now_us) {
        const double n = (double) samples.size();
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (const auto & s : samples) {
            const double x = (double) s.tokens;
            sx += x; sy += s.ms; sxx += x * x; sxy += x * s.ms;
        }
        const double det = n * sxx - sx * sx;
        double a, b;
        if (det <= 1e-9 * std::max(1.0, n * sxx)) {
            a = sx > 0 ? sy / sx : 0.0;   // one distinct size: slope not identifiable
            b = 0.0;
        } else {
            a = (n * sxy - sx * sy) / det;
            b = (sy - a * sx) / n;
        }
        a_ms    = std::max(A_FLOOR_MS, a);
        b_ms    = std::max(0.0, b);
        t_refit = now_us;
        fitted  = true;
    }

    // The rule. `n_queued` requests wait; the oldest has waited `wait_ms` and would need `tokens` tokens of work.
    //   n_queued > 1 : drop it when wait + predicted service time reaches the budget; until the fit is trusted the prediction is 0, so
    //                  only the wait counts (BentoML starts from made-up constants instead);
    //   n_queued == 1: drop it only when its wait alone has reached the budget (BentoML's lone-job rule).
    bool should_shed(size_t n_queued, double wait_ms, int64_t tokens, double max_latency_ms) const {
        if (max_latency_ms <= 0 || n_queued == 0) {
            return false;
        }
        if (n_queued > 1 && fitted) {
            return wait_ms + predict_ms(tokens) >= max_latency_ms;
        }
        return wait_ms >= max_latency_ms;
    }
};
