// One warning per unknown request key: a client typo ("top_P") or a sampler name this server does not read must be visible
// instead of silently ignored, and the request still runs (clients send extra keys, so it is never rejected).
// Row S69-ENGREV-TABBYAPI-SAMPLER-ALIASES-AND-UNKNOWN-KEY-WARN-2026-10-27-01.
//
// SOURCE-READ: TabbyAPI be74bf0a common/sampling.py:428-449 (warn when a request sets a parameter the backend ignores). AGPL-3.0:
// the idea was read for the spec; this key-level, once-per-key design and all code are written here, no code copied.
//
// Header-only apart from the JSON class (common/json.h); tested by tools/server/tests_cpp/test_server_unknown_keys.cpp.
#pragma once

#include "json.h"

#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// Top-level keys of `body` that `known` does not claim, in body order. A non-object body, or a null-valued key, has none to report
// (a null counts as absent, the rule the request schema uses).
inline std::vector<std::string> server_unknown_request_keys(const common_json & body, const std::function<bool(const std::string &)> & known) {
    std::vector<std::string> out;
    if (!body.is_object()) {
        return out;
    }
    for (const auto & it : body.items()) {
        if (it.value().is_null()) {
            continue;
        }
        if (!known(it.key())) {
            out.push_back(it.key());
        }
    }
    return out;
}

// A key is printed back into the log: cap its length and replace control characters so a hostile key cannot forge log lines.
inline std::string server_unknown_key_for_log(const std::string & key) {
    std::string s;
    for (const char c : key) {
        if (s.size() >= 64) {
            s += "...";
            break;
        }
        s += (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) ? '?' : c;
    }
    return s;
}

// Remembers which keys were already reported, so a key is reported once per server run, and bounds the memory a client
// sending endless random keys can cost: after `cap` distinct keys no new one is reported (the caller is told once).
class server_unknown_key_registry {
public:
    explicit server_unknown_key_registry(size_t cap = 256) : cap_(cap) {}

    // From `keys`, returns those not reported before and marks them reported. `cap_hit` is set the first time the cap
    // stops a key from being recorded.
    std::vector<std::string> take_new(const std::vector<std::string> & keys, bool & cap_hit) {
        std::vector<std::string> fresh;
        cap_hit = false;
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto & k : keys) {
            if (seen_.count(k)) {
                continue;
            }
            if (seen_.size() >= cap_) {
                if (!cap_reported_) {
                    cap_reported_ = true;
                    cap_hit       = true;
                }
                continue;
            }
            seen_.insert(k);
            fresh.push_back(k);
        }
        return fresh;
    }

private:
    size_t                          cap_;
    std::mutex                      mu_;
    std::unordered_set<std::string> seen_;
    bool                            cap_reported_ = false;
};
