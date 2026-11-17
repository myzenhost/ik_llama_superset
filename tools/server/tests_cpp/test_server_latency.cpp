// Standalone test for server-latency.h. Build: g++ -std=c++17 -I.. test_server_latency.cpp -o t && ./t
// No model, no server process. Each block carries an opposite-answer control.
#include "server-latency.h"

#include <cstdio>
#include <sstream>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static bool has(const std::string & s, const std::string & sub) { return s.find(sub) != std::string::npos; }

int main() {
    // ---- histogram: le semantics, cumulative rendering, +Inf, sum/count ------------------------------------------
    {
        server_histogram h({0.1, 1.0});
        CHECK(h.observe(0.05));   // bucket 0
        CHECK(h.observe(0.1));    // equal to the bound -> still bucket 0 (le)
        CHECK(h.observe(0.5));    // bucket 1
        CHECK(h.observe(7.0));    // +Inf
        CHECK(h.counts[0] == 2 && h.counts[1] == 1 && h.counts[2] == 1);
        CHECK(h.count == 4);
        CHECK(std::abs(h.sum - 7.65) < 1e-9);

        std::ostringstream o;
        h.render(o, "llamacpp:x_seconds", "help text");
        const std::string s = o.str();
        CHECK(has(s, "# TYPE llamacpp:x_seconds histogram\n"));
        CHECK(has(s, "llamacpp:x_seconds_bucket{le=\"0.1\"} 2\n"));
        CHECK(has(s, "llamacpp:x_seconds_bucket{le=\"1\"} 3\n"));        // cumulative: 2 + 1
        CHECK(has(s, "llamacpp:x_seconds_bucket{le=\"+Inf\"} 4\n"));     // equals _count
        CHECK(has(s, "llamacpp:x_seconds_count 4\n"));
        // control: a non-cumulative renderer would print 1 for le="1"
        CHECK(!has(s, "llamacpp:x_seconds_bucket{le=\"1\"} 1\n"));

        // clock glitches are dropped, not clamped
        CHECK(!h.observe(-0.001));
        CHECK(!h.observe(std::nan("")));
        CHECK(!h.observe(INFINITY));
        CHECK(h.count == 4 && std::abs(h.sum - 7.65) < 1e-9);

        // an unconfigured histogram (no buckets) refuses instead of indexing out of range
        server_histogram none;
        CHECK(!none.observe(1.0));
    }

    // ---- step classification ------------------------------------------------------------------------------------
    {
        CHECK(server_step_stats::classify(5, 0) == server_step_stats::DECODE_ONLY);
        CHECK(server_step_stats::classify(0, 9) == server_step_stats::PREFILL_ONLY);
        CHECK(server_step_stats::classify(3, 9) == server_step_stats::MIXED);
        CHECK(server_step_stats::classify(0, 0) == -1);
    }

    // ---- the three latency sums PARTITION the measured time (the invariant ExecuTorch asserts) ------------------
    {
        server_step_stats st;
        server_step_group g;

        // 1: decode-only, 10 ms, 4 tokens
        g.add_step(1000, 4, 0);
        CHECK(st.close_group(g, 11000) == server_step_stats::DECODE_ONLY);
        g.clear();

        // 2: a prefill chunk with no logits (not synchronised) then a prefill chunk that synchronises: ONE group,
        //    timed from the first chunk's start to the sync, 2 steps, prefill-only
        g.add_step(20000, 0, 512);
        g.add_step(26000, 0, 512);
        CHECK(!g.empty() && g.steps == 2);
        CHECK(st.close_group(g, 40000) == server_step_stats::PREFILL_ONLY);
        g.clear();

        // 3: mixed step, 25 ms: a decode session waited behind a prefill chunk
        g.add_step(50000, 2, 256);
        CHECK(st.close_group(g, 75000) == server_step_stats::MIXED);
        g.clear();

        // 4: decode-only again, 8 ms, 3 tokens
        g.add_step(80000, 3, 0);
        CHECK(st.close_group(g, 88000) == server_step_stats::DECODE_ONLY);
        g.clear();

        CHECK(st.latency_us[server_step_stats::DECODE_ONLY]  == 10000 + 8000);
        CHECK(st.latency_us[server_step_stats::PREFILL_ONLY] == 20000);
        CHECK(st.latency_us[server_step_stats::MIXED]        == 25000);
        CHECK(st.total_us() == 10000 + 20000 + 25000 + 8000);          // partition: nothing lost, nothing counted twice
        CHECK(st.n_steps[server_step_stats::DECODE_ONLY] == 2);
        CHECK(st.n_steps[server_step_stats::PREFILL_ONLY] == 2);       // two chunks in one group
        CHECK(st.n_steps[server_step_stats::MIXED] == 1);
        CHECK(st.total_steps() == 5);
        CHECK(st.decode_only_tokens == 4 + 3);                         // mixed-step decode tokens are NOT decode-only
        CHECK(std::abs(st.mixed_share() - 25000.0 / 63000.0) < 1e-12);

        // control: an empty group books nothing
        server_step_group empty;
        CHECK(st.close_group(empty, 999999) == -1);
        CHECK(st.total_us() == 63000);

        // a clock that ran backwards books 0, never a huge unsigned wrap
        server_step_group back;
        back.add_step(500000, 1, 0);
        CHECK(st.close_group(back, 400000) == server_step_stats::DECODE_ONLY);
        CHECK(st.total_us() == 63000);
    }

    // ---- ready vs admitted ---------------------------------------------------------------------------------------
    {
        server_step_stats st;
        st.add_session_counts(4, 4);   // all four generations got a token
        st.add_session_counts(4, 1);   // the scheduler held three back (one big prefill took the batch)
        CHECK(st.ready_total == 8 && st.decode_sessions_total == 5);
    }

    // ---- request stage times -------------------------------------------------------------------------------------
    {
        server_request_times t;
        t.t_submit = 1000000; t.t_start = 1200000; t.t_first_token = 1500000; t.t_last_token = 3500000;
        CHECK(std::abs(t.queue_s()   - 0.2) < 1e-12);
        CHECK(std::abs(t.prefill_s() - 0.3) < 1e-12);
        CHECK(std::abs(t.ttft_s()    - 0.5) < 1e-12);
        CHECK(std::abs(t.decode_s()  - 2.0) < 1e-12);
        CHECK(std::abs(t.e2e_s()     - 2.5) < 1e-12);
        CHECK(std::abs((t.queue_s() + t.prefill_s()) - t.ttft_s()) < 1e-12);  // ttft = queue + prefill

        // never reached: -1, so the caller skips the observation instead of observing garbage
        server_request_times nt;
        nt.t_submit = 5; nt.t_start = 7;
        CHECK(nt.queue_s() > 0.0 && nt.ttft_s() < 0.0 && nt.decode_s() < 0.0 && nt.e2e_s() < 0.0);
    }

    // ---- full render: exports nothing until there is data; then names the vLLM-style series ----------------------
    {
        server_latency_metrics m;
        std::ostringstream empty;
        m.render(empty, "llamacpp:");
        CHECK(empty.str().empty());                                     // control: an idle server exports no new series

        m.ttft.observe(0.4);
        m.itl.observe(0.02);
        m.n_prompt_tokens.observe(300);   // bucket le=1024
        m.n_gen_tokens.observe(10);       // bucket le=16
        m.finished.add("stop");
        m.finished.add("stop");
        m.finished.add("length");
        server_step_group g;
        g.add_step(0, 2, 128);
        m.steps.close_group(g, 30000);
        m.steps.add_session_counts(3, 2);

        std::ostringstream o;
        m.render(o, "llamacpp:");
        const std::string s = o.str();
        CHECK(has(s, "llamacpp:time_to_first_token_seconds_count 1\n"));
        CHECK(has(s, "llamacpp:inter_token_latency_seconds_count 1\n"));
        CHECK(has(s, "llamacpp:request_prompt_tokens_bucket{le=\"256\"} 0\n"));    // 300 tokens is above 256 ...
        CHECK(has(s, "llamacpp:request_prompt_tokens_bucket{le=\"1024\"} 1\n"));   // ... and inside 1024
        CHECK(has(s, "llamacpp:request_prompt_tokens_sum 300\n"));
        CHECK(has(s, "llamacpp:request_generation_tokens_bucket{le=\"16\"} 1\n"));
        CHECK(!has(s, "llamacpp:e2e_request_latency_seconds"));          // no e2e observation yet -> not exported
        CHECK(has(s, "llamacpp:request_success_total{finished_reason=\"stop\"} 2\n"));
        CHECK(has(s, "llamacpp:request_success_total{finished_reason=\"length\"} 1\n"));
        CHECK(has(s, "llamacpp:steps_total{kind=\"mixed\"} 1\n"));
        CHECK(has(s, "llamacpp:step_latency_seconds_total{kind=\"mixed\"} 0.03\n"));
        CHECK(has(s, "llamacpp:decode_ready_sessions_total 3\n"));
        CHECK(has(s, "llamacpp:decode_admitted_sessions_total 2\n"));
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("test_server_latency: all checks passed\n");
    return 0;
}
