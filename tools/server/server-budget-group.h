// Budget groups: ONE token + wall-time pool shared by every request of a recursion tree (RLM-style agents that fan out sub-calls,
// arXiv 2512.24601). Per-request limits (n_predict, t_max_predict_ms, t_deadline_ms) cannot cap the tree as a whole; this does.
//
// Feature layer: L7 serving / admission (request lifecycle). Composition (rule row S69-ENGINE-MERGE-FEATURES-MUST-COMPOSE-INTERACTION-MATRIX-2026-11-17-01):
//  - keyed like X-Request-Id: same charset check (server_request_id_ok), same header-or-body ingestion, echoed back on the response;
//  - radix / prompt cache: only GENERATED tokens are charged, so a sub-call whose prompt came from the cache costs the pool nothing for it;
//    the pool never changes what the cache keeps or evicts;
//  - request log: the group id is a field of the log line, and a stop by the pool has its own status "budget_group", so one query on
//    the group id sums a tree;
//  - per-request budgets stay: the stricter one wins (they are checked independently on every token).
// Header only, no llama.cpp types, so it is unit tested without a model (tests_cpp/test_server_budget_group.cpp).
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>

enum server_budget_stop {
    SERVER_BUDGET_OK = 0,
    SERVER_BUDGET_TOKENS,   // the shared token pool is used up
    SERVER_BUDGET_WALL,     // the shared wall-time pool ran out (counted from the first request of the group)
};

inline const char * server_budget_stop_str(server_budget_stop s) {
    switch (s) {
        case SERVER_BUDGET_TOKENS: return "tokens";
        case SERVER_BUDGET_WALL:   return "wall";
        default:                   return "";
    }
}

struct server_budget_limits {
    int64_t max_tokens  = 0; // 0 = no token cap
    int64_t max_wall_ms = 0; // 0 = no wall-time cap
    bool any() const { return max_tokens > 0 || max_wall_ms > 0; }
};

struct server_budget_snapshot {
    bool    known       = false;
    int64_t used_tokens = 0;
    int64_t max_tokens  = 0;
    int64_t elapsed_ms  = 0;
    int64_t max_wall_ms = 0;
};

// the group id and its limits as ingested from the request. valid=false: no group asked for, or the id / a number was malformed
// (a malformed group is ignored, never half-applied)
struct server_budget_request {
    bool                 valid = false;
    std::string          id;
    server_budget_limits limits;
};

namespace server_budget_detail {
// strictly digits, at most 12 of them (about 31 years of ms, 1e12 tokens); anything else is rejected
inline bool parse_nonneg(const std::string & s, int64_t & out) {
    if (s.empty() || s.size() > 12) {
        return false;
    }
    int64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') {
            return false;
        }
        v = v * 10 + (c - '0');
    }
    out = v;
    return true;
}
inline bool id_ok(const std::string & id) { // same rule as server_request_id_ok (server-request-log.h), kept local so this header stands alone
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
} // namespace server_budget_detail

// group id: the header wins over the body field (like a proxy-set X-Request-Id wins over a client one). Limits from the headers
// X-Budget-Group-Tokens / X-Budget-Group-Ms or the body fields budget_group_tokens / budget_group_ms. An empty limit string = not given.
inline server_budget_request server_budget_ingest(const std::string & hdr_id, const std::string & body_id,
                                                  const std::string & hdr_tokens, const std::string & body_tokens,
                                                  const std::string & hdr_ms, const std::string & body_ms) {
    server_budget_request r;
    const std::string & id  = !hdr_id.empty() ? hdr_id : body_id;
    const std::string & tok = !hdr_tokens.empty() ? hdr_tokens : body_tokens;
    const std::string & ms  = !hdr_ms.empty() ? hdr_ms : body_ms;
    if (id.empty() || !server_budget_detail::id_ok(id)) {
        return r;
    }
    if (!tok.empty() && !server_budget_detail::parse_nonneg(tok, r.limits.max_tokens)) {
        return server_budget_request{};
    }
    if (!ms.empty() && !server_budget_detail::parse_nonneg(ms, r.limits.max_wall_ms)) {
        return server_budget_request{};
    }
    r.id    = id;
    r.valid = true;
    return r;
}

class server_budget_groups {
public:
    // the first request of a group fixes its limits and its wall-time start; later requests join the same pool and cannot enlarge it
    // (a sub-call must not be able to raise the cap of the tree it belongs to)
    void open(const std::string & id, const server_budget_limits & lim, int64_t now_us) {
        std::lock_guard<std::mutex> lk(mu_);
        purge_locked(now_us);
        auto it = pools_.find(id);
        if (it == pools_.end()) {
            if (pools_.size() >= max_groups_) {
                return; // table full: the group stays uncapped rather than evicting a live tree
            }
            pool p;
            p.lim      = lim;
            p.t_open   = now_us;
            p.t_touch  = now_us;
            pools_[id] = p;
        } else {
            it->second.t_touch = now_us;
        }
    }

    // one generated token of a request in the group; returns why the group is exhausted (OK = keep going). The token that reaches
    // the cap is the last one: a group of max_tokens N produces exactly N tokens in total.
    server_budget_stop charge(const std::string & id, int64_t n_tokens, int64_t now_us) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = pools_.find(id);
        if (it == pools_.end()) {
            return SERVER_BUDGET_OK;
        }
        pool & p = it->second;
        p.used    += n_tokens;
        p.t_touch  = now_us;
        return check_locked(p, now_us);
    }

    // without charging (admission: refuse a sub-call of a tree that is already out of budget)
    server_budget_stop check(const std::string & id, int64_t now_us) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = pools_.find(id);
        return it == pools_.end() ? SERVER_BUDGET_OK : check_locked(it->second, now_us);
    }

    server_budget_snapshot snapshot(const std::string & id, int64_t now_us) {
        std::lock_guard<std::mutex> lk(mu_);
        server_budget_snapshot s;
        auto it = pools_.find(id);
        if (it == pools_.end()) {
            return s;
        }
        s.known       = true;
        s.used_tokens = it->second.used;
        s.max_tokens  = it->second.lim.max_tokens;
        s.elapsed_ms  = (now_us - it->second.t_open) / 1000;
        s.max_wall_ms = it->second.lim.max_wall_ms;
        return s;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu_);
        return pools_.size();
    }

    void set_limits(size_t max_groups, int64_t idle_ttl_us) {
        std::lock_guard<std::mutex> lk(mu_);
        max_groups_ = max_groups;
        idle_ttl_us_ = idle_ttl_us;
    }

private:
    struct pool {
        server_budget_limits lim;
        int64_t used    = 0;
        int64_t t_open  = 0;
        int64_t t_touch = 0;
    };

    static server_budget_stop check_locked(const pool & p, int64_t now_us) {
        if (p.lim.max_tokens > 0 && p.used >= p.lim.max_tokens) {
            return SERVER_BUDGET_TOKENS;
        }
        if (p.lim.max_wall_ms > 0 && (now_us - p.t_open) / 1000 >= p.lim.max_wall_ms) {
            return SERVER_BUDGET_WALL;
        }
        return SERVER_BUDGET_OK;
    }

    // a tree nobody touched for the idle time is finished (or abandoned): drop its pool so the table cannot grow without bound
    void purge_locked(int64_t now_us) {
        for (auto it = pools_.begin(); it != pools_.end();) {
            if (now_us - it->second.t_touch > idle_ttl_us_) {
                it = pools_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::mutex                mu_;
    std::map<std::string, pool> pools_;
    size_t                    max_groups_  = 4096;
    int64_t                   idle_ttl_us_ = 3600ll * 1000000ll;
};
