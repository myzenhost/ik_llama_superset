// Standalone test for server-request-log.h. Build: g++ -std=c++17 -I.. test_server_request_log.cpp -o t && ./t
// No model, no server process. Each block carries an opposite-answer control.
#include "server-request-log.h"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static bool has(const std::string & s, const std::string & sub) { return s.find(sub) != std::string::npos; }

static server_request_record rec(const std::string & id, const std::string & model) {
    server_request_record r;
    r.request_id = id;
    r.model      = model;
    r.n_prompt   = 100;
    r.n_gen      = 20;
    r.queue_ms   = 1.5;
    r.prefill_ms = 30;
    r.decode_ms  = 200;
    r.n_draft    = 16;
    r.n_accept   = 11;
    return r;
}

int main() {
    // ---- traceparent / X-Request-Id ingestion --------------------------------------------------------------------
    {
        const std::string good = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        auto a = server_ingest_trace_headers(good, "", "gen1");
        CHECK(a.from_traceparent && a.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736" && a.parent_id == "00f067aa0ba902b7");
        CHECK(a.request_id == a.trace_id);                 // no X-Request-Id: the trace id keys the request
        auto b = server_ingest_trace_headers(good, "req-42", "gen1");
        CHECK(b.request_id == "req-42" && b.trace_id == a.trace_id); // X-Request-Id echoed, trace id kept
        // controls: each malformed shape is rejected and falls back to the generated id
        const char * bad[] = {
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01", // zero trace id
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01", // zero parent id
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", // version ff
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01", // upper-case hex
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",    // missing flags
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-x", // version 00 with an extra field
            "",
        };
        for (const char * t : bad) {
            auto c = server_ingest_trace_headers(t, "", "gen9");
            CHECK(!c.from_traceparent && c.trace_id.empty() && c.request_id == "gen9");
        }
        // a future version with extra fields is read by its first four
        auto f = server_ingest_trace_headers("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra", "", "g");
        CHECK(f.from_traceparent);
        // an injected newline / space / over-long X-Request-Id is not echoed
        CHECK(server_ingest_trace_headers("", "a\nb", "gen").request_id == "gen");
        CHECK(server_ingest_trace_headers("", "a b", "gen").request_id == "gen");
        CHECK(server_ingest_trace_headers("", std::string(129, 'x'), "gen").request_id == "gen");
        CHECK(server_ingest_trace_headers("", std::string(128, 'x'), "gen").request_id == std::string(128, 'x'));
    }

    // ---- 50 mixed requests -> 50 JSONL lines with the stage times ------------------------------------------------
    {
        std::vector<std::string> lines;
        server_request_log_config cfg;
        cfg.path = "x.jsonl";
        server_request_logger lg(cfg, [&](const std::string & l) { lines.push_back(l); });
        for (int i = 0; i < 50; i++) {
            auto r = rec("id" + std::to_string(i), i % 2 ? "m-a" : "m-b");
            r.status = i % 5 == 0 ? "length" : "ok";
            CHECK(lg.log(r));
        }
        CHECK(lines.size() == 50);
        for (const auto & l : lines) {
            CHECK(l.front() == '{' && l.back() == '}');
            CHECK(has(l, "\"queue_ms\":") && has(l, "\"prefill_ms\":") && has(l, "\"decode_ms\":") && has(l, "\"n_accept\":11"));
            CHECK(l.find('\n') == std::string::npos);
        }
        CHECK(has(lines[0], "\"request_id\":\"id0\"") && has(lines[0], "\"status\":\"length\""));
        CHECK(lg.count("m-a", "logged") == 25 && lg.count("m-b", "logged") == 25 && lg.count("m-a", "sampled_out") == 0);
        // a quote or newline in a field cannot break the line
        server_request_record r = rec("q\"1", "m\nx");
        std::vector<std::string> l2;
        server_request_logger lg2(cfg, [&](const std::string & l) { l2.push_back(l); });
        lg2.log(r);
        CHECK(l2.size() == 1 && l2[0].find('\n') == std::string::npos && has(l2[0], "q\\\"1") && has(l2[0], "m\\nx"));
    }

    // ---- per-model sampling: rate 0.1 over 1000 -> 80..120 ; per-model override ----------------------------------
    {
        std::vector<std::string> lines;
        server_request_log_config cfg;
        cfg.path = "x.jsonl";
        cfg.rate = 0.1;
        cfg.model_rate["all"]  = 1.0;
        cfg.model_rate["none"] = 0.0;
        server_request_logger lg(cfg, [&](const std::string & l) { lines.push_back(l); });
        for (int i = 0; i < 1000; i++) {
            lg.log(rec("s" + std::to_string(i), "dflt"));
        }
        CHECK(lines.size() >= 80 && lines.size() <= 120);
        CHECK(lg.count("dflt", "logged") + lg.count("dflt", "sampled_out") == 1000);  // the counter makes the rate auditable
        CHECK(lg.count("dflt", "logged") == lines.size());
        lines.clear();
        for (int i = 0; i < 100; i++) {
            lg.log(rec("a" + std::to_string(i), "all"));
            lg.log(rec("n" + std::to_string(i), "none"));
        }
        CHECK(lines.size() == 100);                                  // "all" kept 100, "none" kept 0
        CHECK(lg.count("none", "sampled_out") == 100 && lg.count("none", "logged") == 0);
        std::ostringstream o;
        lg.render_counter(o);
        CHECK(has(o.str(), "# TYPE llamacpp:request_log_count counter") && has(o.str(), "request_log_count{model=\"none\",status=\"sampled_out\"} 100"));
    }

    // ---- validate-then-apply: a bad update leaves the old rate in force ------------------------------------------
    {
        std::vector<std::string> lines;
        server_request_log_config cfg;
        cfg.path = "x.jsonl";
        cfg.rate = 1.0;
        server_request_logger lg(cfg, [&](const std::string & l) { lines.push_back(l); });
        server_request_log_config bad = cfg;
        bad.rate = 1.5;
        CHECK(!lg.update_config(bad).empty());
        bad.rate = std::nan("");
        CHECK(!lg.update_config(bad).empty());
        bad.rate = 0.5;
        bad.model_rate["m"] = -0.1;
        CHECK(!lg.update_config(bad).empty());
        CHECK(lg.config().rate == 1.0 && lg.config().model_rate.empty());   // the old config is still in force
        for (int i = 0; i < 20; i++) {
            lg.log(rec("b" + std::to_string(i), "m"));
        }
        CHECK(lines.size() == 20);
        // control: a valid update does apply
        server_request_log_config ok = cfg;
        ok.rate = 0.0;
        CHECK(lg.update_config(ok).empty());
        lg.log(rec("z", "m"));
        CHECK(lines.size() == 20);
    }

    // ---- identical configs share ONE logger ------------------------------------------------------------------------
    {
        server_request_log_registry reg;
        int made = 0;
        auto make = [&](const server_request_log_config & c) {
            made++;
            return std::make_shared<server_request_logger>(c, nullptr);
        };
        server_request_log_config a;
        a.path = "a.jsonl";
        server_request_log_config b = a;
        server_request_log_config c = a;
        c.rate = 0.5;
        auto la = reg.get(a, make);
        auto lb = reg.get(b, make);
        auto lc = reg.get(c, make);
        CHECK(la == lb && made == 2);   // a and b are identical: one logger; c differs: another
        CHECK(la != lc);
    }

    // ---- in-flight journal -> crash dump, including a kill -9 that tore the last line ----------------------------
    {
        std::string journal;
        server_request_log_config cfg;
        cfg.path = "x.jsonl";
        server_request_logger lg(cfg, nullptr, [&](const std::string & l) { journal += l + "\n"; });
        lg.begin("r1", "t1", "m");
        lg.begin("r2", "t2", "m");
        lg.begin("r3", "", "m2");
        lg.end("r2");
        lg.end("never-began");                      // a stray end writes nothing
        CHECK(lg.n_inflight() == 2);
        auto live = lg.inflight_lines();
        CHECK(live.size() == 2 && has(live[0], "\"request_id\":\"r1\"") && has(live[1], "\"request_id\":\"r3\""));
        auto dump = server_request_log_recover(journal);
        CHECK(dump.size() == 2 && has(dump[0], "\"request_id\":\"r1\"") && has(dump[0], "\"trace_id\":\"t1\"") && has(dump[1], "\"request_id\":\"r3\""));
        CHECK(has(dump[0], "in_flight_at_crash"));
        // control: a request that ended is not listed
        CHECK(!has(dump[0], "r2") && !has(dump[1], "r2"));
        // a torn last line (no newline, cut mid-field) is ignored, not half-parsed
        auto torn = server_request_log_recover(journal + "B\tr4\tt4\tmod");
        CHECK(torn.size() == 2);
        auto whole = server_request_log_recover(journal + "B\tr4\tt4\tmod\n");
        CHECK(whole.size() == 3);
    }

    // ---- file-backed logger: lines land on disk; a "crashed" process's journal becomes <path>.crash.jsonl at the next start ----
    {
        const std::string path = "qw21_reqlog_test.jsonl";
        std::remove(path.c_str());
        std::remove((path + ".inflight").c_str());
        std::remove((path + ".crash.jsonl").c_str());
        server_request_log_config cfg;
        cfg.path = path;
        std::string err;
        {
            auto lg = server_request_log_open_files(cfg, &err);
            CHECK(lg && err.empty());
            lg->begin("done1", "t", "m");
            lg->log(rec("done1", "m"));
            lg->end("done1");
            lg->begin("live1", "tl", "m");   // never ends: the process "dies" here
            lg->begin("live2", "", "m");
        }
        std::ifstream f(path);
        std::string l1;
        std::getline(f, l1);
        CHECK(has(l1, "\"request_id\":\"done1\""));
        auto lg2 = server_request_log_open_files(cfg, &err);   // the next start
        CHECK(lg2 != nullptr);
        std::ifstream cd(path + ".crash.jsonl");
        std::string c1, c2, c3;
        std::getline(cd, c1);
        std::getline(cd, c2);
        std::getline(cd, c3);
        CHECK(has(c1, "live1") && has(c1, "in_flight_at_crash") && has(c2, "live2") && c3.empty() && !has(c1 + c2, "done1"));
        // controls: a bad config opens nothing; model-rate text parses, a malformed one is refused and leaves the map alone
        server_request_log_config badcfg = cfg;
        badcfg.rate = 2.0;
        CHECK(server_request_log_open_files(badcfg, &err) == nullptr && !err.empty());
        std::map<std::string, double> mr;
        CHECK(server_request_log_parse_model_rates("a=0.5,b=1", mr).empty() && mr.size() == 2 && mr["a"] == 0.5);
        CHECK(!server_request_log_parse_model_rates("a=0.5,junk", mr).empty() && mr.size() == 2);
        CHECK(!server_request_log_parse_model_rates("a=x", mr).empty());
        lg2.reset();
        std::remove(path.c_str());
        std::remove((path + ".inflight").c_str());
        std::remove((path + ".crash.jsonl").c_str());
    }

    if (fails == 0) {
        std::printf("all tests OK\n");
    }
    return fails ? 1 : 0;
}
