// Latency and step-composition instruments for the server /metrics endpoint.
//
// Header-only and free of llama / ggml / json dependencies, so the arithmetic is unit-tested with no model
// and no server process (tools/server/tests_cpp/test_server_latency.cpp).
//
// What is measured, and why it is here
//   * server_histogram      Prometheus histogram (cumulative le buckets, _sum, _count) for TTFT, inter-token
//                           latency, end-to-end, queue, prefill and decode time. The metric names follow the
//                           vLLM / Ray Serve dashboard vocabulary so existing dashboards read this server.
//   * server_step_stats     every engine step (one llama_decode, or a run of un-synchronised ones closed by the
//                           next synchronised one) is classified by what it carried: decode tokens only, decode
//                           and prefill mixed, or prefill only. The three latency sums PARTITION the measured
//                           step time, so "a long prompt arrived and stalled everyone's decode" shows up as mixed
//                           latency and an inter-token-latency spike instead of being argued from anecdote.
//   * ready vs admitted     per step, how many generations were past their first token (eligible for a decode
//                           slot) against how many actually had a token in the batch. "The scheduler held work
//                           back" is then separable from "there was no work".
//   * server_finish_counter finished requests by reason.
//
// Behaviour taken from ExecuTorch extension/llm/batching/runner.cpp (BSD-3-Clause, Meta), read in full:
// step classification and the partition invariant (:931-1190), TTFT and ITL sum / min / max (:1518-1570),
// completion record (:1727-1760). Description-level; no code copied. See docs/THIRD_PARTY_NOTICES_E_SERVING.md.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// histogram
// ---------------------------------------------------------------------------------------------------------------

struct server_histogram {
    std::vector<double>   bounds; // upper bounds in seconds, strictly ascending; an implicit +Inf bucket follows
    std::vector<uint64_t> counts; // per bucket (NOT cumulative), size bounds.size() + 1
    double   sum   = 0.0;
    uint64_t count = 0;

    server_histogram() = default;
    explicit server_histogram(std::vector<double> b) : bounds(std::move(b)), counts(bounds.size() + 1, 0) {}

    // a negative or non-finite observation is a clock glitch, not a latency: it is dropped, never clamped into the
    // first bucket, so it cannot bend the sum
    bool observe(double seconds) {
        if (counts.empty() || !std::isfinite(seconds) || seconds < 0.0) {
            return false;
        }
        size_t i = 0;
        while (i < bounds.size() && seconds > bounds[i]) { // le semantics: a value equal to the bound is inside it
            i++;
        }
        counts[i]++;
        sum += seconds;
        count++;
        return true;
    }

    // 1 ms .. 60 s: inter-token latency, time to first token, queue, prefill
    static server_histogram fine() {
        return server_histogram({ 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0 });
    }
    // 100 ms .. 30 min: end-to-end and decode time of a whole request
    static server_histogram coarse() {
        return server_histogram({ 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0 });
    }

    // 16 .. 64k tokens: prompt and generation lengths (the unit is tokens, not seconds)
    static server_histogram tokens() {
        return server_histogram({ 16, 64, 256, 1024, 4096, 16384, 65536 });
    }

    // prometheus text exposition, cumulative buckets; name is the full metric name including the prefix
    void render(std::ostream & out, const std::string & name, const std::string & help) const {
        out << "# HELP " << name << " " << help << "\n"
            << "# TYPE " << name << " histogram\n";
        uint64_t cum = 0;
        char buf[64];
        for (size_t i = 0; i < counts.size(); i++) {
            cum += counts[i];
            if (i < bounds.size()) {
                snprintf(buf, sizeof(buf), "%g", bounds[i]);
                out << name << "_bucket{le=\"" << buf << "\"} " << cum << "\n";
            } else {
                out << name << "_bucket{le=\"+Inf\"} " << cum << "\n";
            }
        }
        out << name << "_sum "   << sum   << "\n"
            << name << "_count " << count << "\n";
    }
};

// ---------------------------------------------------------------------------------------------------------------
// step composition
// ---------------------------------------------------------------------------------------------------------------

// A run of engine steps that ends in a synchronised one. Prefill chunks that carry no logits are submitted without
// a synchronise, so their compute time only becomes known when a later step synchronises: the whole run is timed
// from the start of its first step to that sync and classified by everything it carried.
struct server_step_group {
    uint64_t steps          = 0;
    uint64_t decode_tokens  = 0; // tokens of generating requests (incl. speculative draft tokens)
    uint64_t prefill_tokens = 0; // prompt tokens
    int64_t  t_start_us     = 0;

    bool empty() const { return steps == 0; }

    void add_step(int64_t t_step_start_us, uint64_t n_decode, uint64_t n_prefill) {
        if (steps == 0) {
            t_start_us = t_step_start_us;
        }
        steps++;
        decode_tokens  += n_decode;
        prefill_tokens += n_prefill;
    }

    void clear() { *this = {}; }
};

struct server_step_stats {
    enum kind { DECODE_ONLY = 0, MIXED = 1, PREFILL_ONLY = 2, N_KIND = 3 };

    uint64_t n_steps[N_KIND]    = {0, 0, 0}; // engine steps (llama_decode calls) by kind
    uint64_t latency_us[N_KIND] = {0, 0, 0}; // measured time by kind; the three sums partition total_us()
    uint64_t decode_only_tokens = 0;         // decode tokens that rode in decode-only groups

    // per step: generations eligible for a decode slot, against the ones that actually had a token in the batch
    uint64_t ready_total           = 0;
    uint64_t decode_sessions_total = 0;

    // -1 when the group carried nothing (an empty batch is not a step)
    static int classify(uint64_t n_decode, uint64_t n_prefill) {
        if (n_decode > 0 && n_prefill > 0) return MIXED;
        if (n_decode > 0)                  return DECODE_ONLY;
        if (n_prefill > 0)                 return PREFILL_ONLY;
        return -1;
    }

    void add_session_counts(uint64_t ready, uint64_t admitted) {
        ready_total           += ready;
        decode_sessions_total += admitted;
    }

    // close a group at a synchronised step; returns the kind it was booked under, or -1 when nothing was booked.
    // A clock that ran backwards books 0, never a negative.
    int close_group(const server_step_group & g, int64_t t_end_us) {
        if (g.empty()) {
            return -1;
        }
        const int k = classify(g.decode_tokens, g.prefill_tokens);
        if (k < 0) {
            return -1;
        }
        n_steps[k]    += g.steps;
        latency_us[k] += (uint64_t) std::max<int64_t>(0, t_end_us - g.t_start_us);
        if (k == DECODE_ONLY) {
            decode_only_tokens += g.decode_tokens;
        }
        return k;
    }

    uint64_t total_us() const { return latency_us[0] + latency_us[1] + latency_us[2]; }
    uint64_t total_steps() const { return n_steps[0] + n_steps[1] + n_steps[2]; }

    // share of measured step time that carried prefill work next to decode work (the stall the decode sessions felt)
    double mixed_share() const {
        const uint64_t t = total_us();
        return t > 0 ? (double) latency_us[MIXED] / (double) t : 0.0;
    }

    static const char * kind_name(int k) {
        switch (k) {
            case DECODE_ONLY:   return "decode_only";
            case MIXED:         return "mixed";
            case PREFILL_ONLY:  return "prefill_only";
            default:            return "none";
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// finished requests by reason
// ---------------------------------------------------------------------------------------------------------------

struct server_finish_counter {
    std::map<std::string, uint64_t> by_reason;

    void add(const std::string & reason) { by_reason[reason]++; }

    uint64_t total() const {
        uint64_t t = 0;
        for (const auto & kv : by_reason) t += kv.second;
        return t;
    }

    void render(std::ostream & out, const std::string & name, const std::string & help) const {
        if (by_reason.empty()) {
            return;
        }
        out << "# HELP " << name << " " << help << "\n"
            << "# TYPE " << name << " counter\n";
        for (const auto & kv : by_reason) {
            out << name << "{finished_reason=\"" << kv.first << "\"} " << kv.second << "\n";
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// everything /metrics adds, in one copyable value (server_metrics is copied into every metrics result)
// ---------------------------------------------------------------------------------------------------------------

struct server_latency_metrics {
    server_histogram ttft    = server_histogram::fine();   // submit -> first generated token
    server_histogram itl     = server_histogram::fine();   // gap between consecutive token-emitting steps of one request
    server_histogram e2e     = server_histogram::coarse(); // submit -> last token
    server_histogram queue   = server_histogram::fine();   // submit -> a slot started on it
    server_histogram prefill = server_histogram::fine();   // slot start -> first token
    server_histogram decode  = server_histogram::coarse(); // first token -> last token
    server_histogram n_prompt_tokens = server_histogram::tokens(); // prompt length per request (cached + processed)
    server_histogram n_gen_tokens    = server_histogram::tokens(); // generated tokens per request

    server_step_stats      steps;
    server_finish_counter  finished;

    // render only the histograms and counters that have data; a server that never finished a request exports nothing
    void render(std::ostream & out, const std::string & prefix) const {
        auto h = [&](const server_histogram & x, const char * n, const char * help) {
            if (x.count > 0) x.render(out, prefix + n, help);
        };
        h(ttft,    "time_to_first_token_seconds", "Time from request submission to the first generated token");
        h(itl,     "inter_token_latency_seconds", "Time between consecutive token-emitting steps of one request");
        h(e2e,     "e2e_request_latency_seconds", "Time from request submission to its last token");
        h(queue,   "request_queue_time_seconds",  "Time a request waited before a slot started on it");
        h(prefill, "request_prefill_time_seconds","Time from a slot starting on a request to its first token");
        h(decode,  "request_decode_time_seconds", "Time from a request's first token to its last token");
        h(n_prompt_tokens, "request_prompt_tokens",     "Prompt tokens per request, cached ones included");
        h(n_gen_tokens,    "request_generation_tokens", "Generated tokens per request");
        finished.render(out, prefix + "request_success_total", "Finished requests by reason");

        if (steps.total_steps() > 0) {
            out << "# HELP " << prefix << "step_latency_seconds_total Measured engine step time by what the step carried; the three kinds partition the total\n"
                << "# TYPE " << prefix << "step_latency_seconds_total counter\n";
            for (int k = 0; k < server_step_stats::N_KIND; k++) {
                out << prefix << "step_latency_seconds_total{kind=\"" << server_step_stats::kind_name(k) << "\"} "
                    << (double) steps.latency_us[k] / 1e6 << "\n";
            }
            out << "# HELP " << prefix << "steps_total Engine steps (llama_decode calls) by what the step carried\n"
                << "# TYPE " << prefix << "steps_total counter\n";
            for (int k = 0; k < server_step_stats::N_KIND; k++) {
                out << prefix << "steps_total{kind=\"" << server_step_stats::kind_name(k) << "\"} " << steps.n_steps[k] << "\n";
            }
            out << "# HELP " << prefix << "decode_only_tokens_total Decode tokens that rode in decode-only steps\n"
                << "# TYPE " << prefix << "decode_only_tokens_total counter\n"
                << prefix << "decode_only_tokens_total " << steps.decode_only_tokens << "\n";
            out << "# HELP " << prefix << "decode_ready_sessions_total Per step, generations eligible for a decode slot (summed over steps)\n"
                << "# TYPE " << prefix << "decode_ready_sessions_total counter\n"
                << prefix << "decode_ready_sessions_total " << steps.ready_total << "\n";
            out << "# HELP " << prefix << "decode_admitted_sessions_total Per step, generations that had a token in the batch (summed over steps)\n"
                << "# TYPE " << prefix << "decode_admitted_sessions_total counter\n"
                << prefix << "decode_admitted_sessions_total " << steps.decode_sessions_total << "\n";
        }
    }
};

// per-request latency breakdown from a slot's absolute timestamps (all us; 0 = not reached)
struct server_request_times {
    int64_t t_submit       = 0;
    int64_t t_start        = 0; // a slot started on it
    int64_t t_first_token  = 0;
    int64_t t_last_token   = 0;

    // each stage is -1 when either end was never reached; the caller observes only the stages that exist
    double queue_s()   const { return (t_submit > 0 && t_start > 0)       ? (double) (t_start - t_submit) / 1e6 : -1.0; }
    double prefill_s() const { return (t_start > 0 && t_first_token > 0)  ? (double) (t_first_token - t_start) / 1e6 : -1.0; }
    double ttft_s()    const { return (t_submit > 0 && t_first_token > 0) ? (double) (t_first_token - t_submit) / 1e6 : -1.0; }
    double decode_s()  const { return (t_first_token > 0 && t_last_token > 0) ? (double) (t_last_token - t_first_token) / 1e6 : -1.0; }
    double e2e_s()     const { return (t_submit > 0 && t_last_token > 0)  ? (double) (t_last_token - t_submit) / 1e6 : -1.0; }
};
