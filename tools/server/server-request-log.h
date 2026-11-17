// Per-request JSONL log, in-flight journal and trace-id ingestion for the server.
//
// Header-only and free of llama / ggml / json dependencies, so it is unit-tested with no model and no server process
// (tools/server/tests_cpp/test_server_request_log.cpp).
//
// What it does
//   * one JSON line per finished request: ids, model, status, token counts, queue / prefill / decode / ttft / e2e in
//     ms and the speculative accept count (the per-request forensics the aggregate /metrics histograms cannot give);
//   * an in-flight journal: begin() appends a "B" line, end() an "E" line. After a crash (even kill -9, which no
//     signal handler can catch) server_request_log_recover() replays the journal and lists the requests that began
//     and never ended, so the crash dump is written from what was durable, not from process memory;
//   * ids: a W3C `traceparent` header (version-00 shape, non-zero trace and parent id) is ingested and its trace id
//     keys the log line; X-Request-Id is echoed back; both fall back to a generated id, so the 8090 -> 8081 hop
//     shares one key;
//   * sampling (TFServing design): a per-model uniform sampling rate, not all-or-nothing; identical configs share
//     ONE logger through server_request_log_registry; a config update is validated first and applied atomically, so
//     a bad update keeps the old logger and rate; a counter request_log_count{model,status} makes the rate auditable.
//
// SOURCE-READ: sglang b4584ab8 python/sglang/srt/observability/request_metrics_exporter.py:1-223 (one JSONL line per
// finished request, rotating file) and forward_pass_metrics.py:1-221; tensorflow/serving core/request_logger.cc:15-95,
// server_request_logger.cc:15-178, apis logging_config.proto:1-87 (sampling rate per model, one logger per config,
// validate-then-swap update); W3C Trace Context level 1 (traceparent grammar). Apache-2.0 sources, behaviour only,
// no code copied. Rows S69-ENGREV-SGLANG-REQUEST-LOG-METRICS-EXPORT-2026-10-27-01 (absorbs
// S69-ENGREV-TGI-TRACEPARENT-REQUEST-ID-2026-10-27-01).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// ids
// ---------------------------------------------------------------------------------------------------------------

struct server_trace_ids {
    std::string trace_id;   // 32 lower-case hex from a valid traceparent, else empty
    std::string parent_id;  // 16 hex
    std::string request_id; // X-Request-Id (sanitised) > trace id > generated
    bool        from_traceparent = false;
};

namespace server_request_log_detail {
    inline bool is_hex(const std::string & s, size_t b, size_t n) {
        if (s.size() < b + n) {
            return false;
        }
        for (size_t i = b; i < b + n; i++) {
            const char c = s[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
                return false;
            }
        }
        return true;
    }
    inline bool all_zero(const std::string & s, size_t b, size_t n) {
        for (size_t i = b; i < b + n; i++) {
            if (s[i] != '0') {
                return false;
            }
        }
        return true;
    }
    inline std::string lower(std::string s) {
        for (auto & c : s) {
            if (c >= 'A' && c <= 'F') {
                c = (char) (c - 'A' + 'a');
            }
        }
        return s;
    }
    inline std::string json_escape(const std::string & s) {
        std::string o;
        o.reserve(s.size() + 2);
        for (unsigned char c : s) {
            switch (c) {
                case '"':  o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n";  break;
                case '\r': o += "\\r";  break;
                case '\t': o += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char b[8];
                        snprintf(b, sizeof b, "\\u%04x", c);
                        o += b;
                    } else {
                        o += (char) c;
                    }
            }
        }
        return o;
    }
}

// a traceparent is `00-<32 hex>-<16 hex>-<2 hex>`; the all-zero trace or parent id and version ff are invalid, and a
// future version (not 00, not ff) is read by its first four fields. Upper-case hex is invalid per the spec.
inline bool server_parse_traceparent(const std::string & v, std::string & trace_id, std::string & parent_id) {
    using namespace server_request_log_detail;
    if (v.size() < 55) {
        return false;
    }
    if (!is_hex(v, 0, 2) || v.substr(0, 2) == "ff") {
        return false;
    }
    if (v[2] != '-' || v[35] != '-' || v[52] != '-') {
        return false;
    }
    if (!is_hex(v, 3, 32) || all_zero(v, 3, 32) || !is_hex(v, 36, 16) || all_zero(v, 36, 16) || !is_hex(v, 53, 2)) {
        return false;
    }
    if (v.substr(0, 2) == "00" && v.size() != 55) {
        return false; // version 00 has exactly four fields
    }
    if (v.size() > 55 && v[55] != '-') {
        return false;
    }
    trace_id  = v.substr(3, 32);
    parent_id = v.substr(36, 16);
    return true;
}

// an inbound X-Request-Id is echoed only when it is short printable ASCII without a space or control byte; anything
// else is replaced, so a header cannot inject a newline into the log or the response
inline bool server_request_id_ok(const std::string & id) {
    if (id.empty() || id.size() > 128) {
        return false;
    }
    for (unsigned char c : id) {
        if (c <= 0x20 || c >= 0x7f) {
            return false;
        }
    }
    return true;
}

inline server_trace_ids server_ingest_trace_headers(const std::string & traceparent, const std::string & x_request_id, const std::string & generated_id) {
    server_trace_ids r;
    r.from_traceparent = server_parse_traceparent(traceparent, r.trace_id, r.parent_id);
    if (!r.from_traceparent) {
        r.trace_id.clear();
        r.parent_id.clear();
    }
    if (server_request_id_ok(x_request_id)) {
        r.request_id = x_request_id;
    } else if (!r.trace_id.empty()) {
        r.request_id = r.trace_id;
    } else {
        r.request_id = generated_id;
    }
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------------------------------------------

struct server_request_record {
    std::string request_id;
    std::string trace_id;
    std::string model;
    std::string budget_group;  // recursion-tree id (server-budget-group.h); empty = none, then the field is omitted
    std::string status = "ok"; // ok | length | abort | deadline | budget_group
    int64_t  n_prompt      = 0;
    int64_t  n_gen         = 0;
    double   queue_ms      = 0;
    double   prefill_ms    = 0;
    double   decode_ms     = 0;
    double   ttft_ms       = 0;
    double   e2e_ms        = 0;
    int64_t  n_draft       = 0; // speculative tokens proposed
    int64_t  n_accept      = 0; // speculative tokens accepted
    int64_t  t_unix_ms     = 0;

    std::string to_jsonl() const {
        using server_request_log_detail::json_escape;
        char b[512];
        snprintf(b, sizeof b,
            ",\"n_prompt\":%lld,\"n_gen\":%lld,\"queue_ms\":%.3f,\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"ttft_ms\":%.3f,\"e2e_ms\":%.3f,"
            "\"n_draft\":%lld,\"n_accept\":%lld,\"t_unix_ms\":%lld}",
            (long long) n_prompt, (long long) n_gen, queue_ms, prefill_ms, decode_ms, ttft_ms, e2e_ms,
            (long long) n_draft, (long long) n_accept, (long long) t_unix_ms);
        return "{\"request_id\":\"" + json_escape(request_id) + "\",\"trace_id\":\"" + json_escape(trace_id) + "\",\"model\":\"" +
               json_escape(model) + "\",\"status\":\"" + json_escape(status) + "\"" +
               (budget_group.empty() ? std::string() : ",\"budget_group\":\"" + json_escape(budget_group) + "\"") + b;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------------------------------------------

struct server_request_log_config {
    std::string path;                          // JSONL file; empty = the log is off
    double      rate = 1.0;                    // uniform sampling probability for a model without its own entry
    std::map<std::string, double> model_rate;  // per-model override
    std::string key() const {                  // identical configs have the same key (and so share one logger)
        std::ostringstream o;
        o << path << '|' << rate;
        for (const auto & kv : model_rate) {
            o << '|' << kv.first << '=' << kv.second;
        }
        return o.str();
    }
    // empty string = valid
    std::string validate() const {
        auto bad = [](double r) { return !(r >= 0.0 && r <= 1.0); }; // also rejects NaN
        if (bad(rate)) {
            return "request-log rate must be in [0, 1]";
        }
        for (const auto & kv : model_rate) {
            if (kv.first.empty()) {
                return "request-log model name must not be empty";
            }
            if (bad(kv.second)) {
                return "request-log rate for model '" + kv.first + "' must be in [0, 1]";
            }
        }
        return "";
    }
};

// ---------------------------------------------------------------------------------------------------------------
// logger
// ---------------------------------------------------------------------------------------------------------------

class server_request_logger {
public:
    using sink_fn = std::function<void(const std::string &)>; // receives one line WITHOUT the trailing newline

    server_request_logger(const server_request_log_config & cfg, sink_fn line_sink, sink_fn journal_sink = nullptr, uint64_t seed = 0x9E3779B97F4A7C15ull)
        : cfg_(cfg), sink_(std::move(line_sink)), journal_(std::move(journal_sink)), rng_(seed ? seed : 1) {}

    // validate first, then swap: a bad update returns its error and leaves the old config (and rate) in force
    std::string update_config(const server_request_log_config & next) {
        const std::string err = next.validate();
        if (!err.empty()) {
            return err;
        }
        std::lock_guard<std::mutex> lk(mu_);
        cfg_ = next;
        return "";
    }

    server_request_log_config config() const {
        std::lock_guard<std::mutex> lk(mu_);
        return cfg_;
    }

    // a finished request: sampled lines are written, the others are only counted
    bool log(const server_request_record & rec) {
        std::lock_guard<std::mutex> lk(mu_);
        const double rate = rate_for(rec.model);
        const bool take = rate >= 1.0 || (rate > 0.0 && uniform() < rate);
        count_[{rec.model, take ? "logged" : "sampled_out"}]++;
        if (take && sink_) {
            sink_(rec.to_jsonl());
        }
        return take;
    }

    // in-flight journal: every begun request is durable the moment it begins, and an end line retires it
    void begin(const std::string & request_id, const std::string & trace_id, const std::string & model) {
        std::lock_guard<std::mutex> lk(mu_);
        inflight_[request_id] = {trace_id, model};
        if (journal_) {
            journal_("B\t" + server_request_log_detail::json_escape(request_id) + "\t" + server_request_log_detail::json_escape(trace_id) +
                     "\t" + server_request_log_detail::json_escape(model));
        }
    }
    void end(const std::string & request_id) {
        std::lock_guard<std::mutex> lk(mu_);
        if (inflight_.erase(request_id) && journal_) {
            journal_("E\t" + server_request_log_detail::json_escape(request_id));
        }
    }
    size_t n_inflight() const {
        std::lock_guard<std::mutex> lk(mu_);
        return inflight_.size();
    }

    // the in-flight set as JSONL (what a SIGTERM / abort handler writes)
    std::vector<std::string> inflight_lines() const {
        using server_request_log_detail::json_escape;
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::string> v;
        for (const auto & kv : inflight_) {
            v.push_back("{\"request_id\":\"" + json_escape(kv.first) + "\",\"trace_id\":\"" + json_escape(kv.second.first) +
                        "\",\"model\":\"" + json_escape(kv.second.second) + "\",\"status\":\"in_flight\"}");
        }
        return v;
    }

    // request_log_count{model,status}
    uint64_t count(const std::string & model, const std::string & status) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = count_.find({model, status});
        return it == count_.end() ? 0 : it->second;
    }
    void render_counter(std::ostream & out) const {
        std::lock_guard<std::mutex> lk(mu_);
        out << "# HELP llamacpp:request_log_count Finished requests by model and whether the request log kept or sampled them out.\n"
            << "# TYPE llamacpp:request_log_count counter\n";
        for (const auto & kv : count_) {
            out << "llamacpp:request_log_count{model=\"" << server_request_log_detail::json_escape(kv.first.first) << "\",status=\""
                << kv.first.second << "\"} " << kv.second << "\n";
        }
    }

private:
    double rate_for(const std::string & model) const {
        auto it = cfg_.model_rate.find(model);
        return it == cfg_.model_rate.end() ? cfg_.rate : it->second;
    }
    double uniform() { // xorshift64*, 53 bits -> [0,1)
        rng_ ^= rng_ >> 12;
        rng_ ^= rng_ << 25;
        rng_ ^= rng_ >> 27;
        const uint64_t x = rng_ * 0x2545F4914F6CDD1Dull;
        return (double) (x >> 11) / 9007199254740992.0;
    }

    mutable std::mutex mu_;
    server_request_log_config cfg_;
    sink_fn sink_;
    sink_fn journal_;
    uint64_t rng_;
    std::map<std::string, std::pair<std::string, std::string>> inflight_; // id -> (trace id, model)
    std::map<std::pair<std::string, std::string>, uint64_t> count_;
};

// identical configs share ONE logger (one file handle, one rate stream, one counter set)
class server_request_log_registry {
public:
    using factory_fn = std::function<std::shared_ptr<server_request_logger>(const server_request_log_config &)>;
    std::shared_ptr<server_request_logger> get(const server_request_log_config & cfg, const factory_fn & make) {
        std::lock_guard<std::mutex> lk(mu_);
        auto & w = by_key_[cfg.key()];
        if (auto p = w.lock()) {
            return p;
        }
        auto p = make(cfg);
        w = p;
        return p;
    }
private:
    std::mutex mu_;
    std::map<std::string, std::weak_ptr<server_request_logger>> by_key_;
};

// replay a journal (the text of the "B"/"E" lines): the requests that began and never ended, as JSONL crash-dump lines.
// A torn last line (a crash mid-write) is ignored, never half-parsed.
inline std::vector<std::string> server_request_log_recover(const std::string & journal_text) {
    std::vector<std::string> order;
    std::map<std::string, std::string> open; // id -> dump line
    std::istringstream in(journal_text);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    const bool last_complete = !journal_text.empty() && journal_text.back() == '\n';
    if (!lines.empty() && !last_complete) {
        lines.pop_back();
    }
    auto split = [](const std::string & s) {
        std::vector<std::string> f;
        size_t b = 0;
        for (;;) {
            const size_t e = s.find('\t', b);
            f.push_back(s.substr(b, e == std::string::npos ? std::string::npos : e - b));
            if (e == std::string::npos) {
                break;
            }
            b = e + 1;
        }
        return f;
    };
    for (const auto & l : lines) {
        const auto f = split(l);
        if (f.size() == 4 && f[0] == "B") {
            if (!open.count(f[1])) {
                order.push_back(f[1]);
            }
            open[f[1]] = "{\"request_id\":\"" + f[1] + "\",\"trace_id\":\"" + f[2] + "\",\"model\":\"" + f[3] + "\",\"status\":\"in_flight_at_crash\"}";
        } else if (f.size() == 2 && f[0] == "E") {
            open.erase(f[1]);
        }
    }
    std::vector<std::string> out;
    for (const auto & id : order) {
        auto it = open.find(id);
        if (it != open.end()) {
            out.push_back(it->second);
            open.erase(it);
        }
    }
    return out;
}

// "NAME=R,NAME=R" -> map. Returns an error text ("" = ok); on error `out` is left untouched.
inline std::string server_request_log_parse_model_rates(const std::string & s, std::map<std::string, double> & out) {
    std::map<std::string, double> m;
    size_t pos = 0;
    while (pos <= s.size() && !s.empty()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        const std::string item = s.substr(pos, end - pos);
        const size_t eq = item.rfind('=');
        if (eq == std::string::npos || eq == 0) {
            return "expected NAME=R, got '" + item + "'";
        }
        char * endp = nullptr;
        const std::string num = item.substr(eq + 1);
        const double r = std::strtod(num.c_str(), &endp);
        if (num.empty() || endp == nullptr || *endp != '\0') {
            return "bad rate in '" + item + "'";
        }
        m[item.substr(0, eq)] = r; // range is checked by validate()
        pos = end + 1;
    }
    out = std::move(m);
    return "";
}

// File-backed logger: <path> gets the JSONL lines (append, flushed per line), <path>.inflight the begin/end journal.
// On creation an old journal is replayed: whatever was in flight when the previous process died is written to
// <path>.crash.jsonl, and the journal starts empty. Returns null if the files cannot be opened (error in *err).
inline std::shared_ptr<server_request_logger> server_request_log_open_files(const server_request_log_config & cfg, std::string * err = nullptr) {
    const std::string bad = cfg.validate();
    if (!bad.empty() || cfg.path.empty()) {
        if (err) {
            *err = bad.empty() ? "no request-log path" : bad;
        }
        return nullptr;
    }
    const std::string jpath = cfg.path + ".inflight";
    {
        std::ifstream old(jpath, std::ios::binary);
        if (old) {
            std::ostringstream ss;
            ss << old.rdbuf();
            const auto dump = server_request_log_recover(ss.str());
            if (!dump.empty()) {
                std::ofstream cd(cfg.path + ".crash.jsonl", std::ios::app | std::ios::binary);
                for (const auto & l : dump) {
                    cd << l << "\n";
                }
            }
        }
    }
    auto lines   = std::make_shared<std::ofstream>(cfg.path, std::ios::app | std::ios::binary);
    auto journal = std::make_shared<std::ofstream>(jpath, std::ios::trunc | std::ios::binary);
    if (!*lines || !*journal) {
        if (err) {
            *err = "cannot open " + cfg.path + " or " + jpath;
        }
        return nullptr;
    }
    return std::make_shared<server_request_logger>(cfg,
        [lines](const std::string & l) { *lines << l << "\n"; lines->flush(); },
        [journal](const std::string & l) { *journal << l << "\n"; journal->flush(); });
}
