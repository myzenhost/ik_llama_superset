#pragma once

// Router-mode model failover chain (row S69-MLFLOW-PICK-ROUTER-MODE-FAILOVER-CHAIN-2026-10-15-01).
//
// A model entry may name an ordered list of fallback models (`fallback-model = b,c` in its preset section, or --fallback-model on the
// router command line for every model). When a request cannot be served by the model it asked for -- the model fails to load, is in its
// load cooldown, is being unloaded, or the child answers with a server error before any byte of the answer was handed to the client --
// the next model of the chain serves it. A response that has started is never retried.
//
// SOURCE-READ: mlflow/mlflow@2d09f29 mlflow/gateway/providers/base.py:713-839 (FallbackProvider; Apache-2.0, no code copied):
//   :737-738 max_attempts = min(max_attempts or len(providers), len(providers)) -- the primary counts as an attempt
//   :762-770 the first provider that does not raise wins; every exception moves on to the next one
//   :814-829 streaming: `yielded` flips on the first chunk and `if yielded: raise` (:824-825) -- fall back only before the first chunk
//   :831-839 when every attempt failed, the status of the LAST error is the status of the final error
// Deliberate differences, each stated in the findings file: MLflow falls back on ANY exception, including a client error (HTTP 4xx) that
// every provider would repeat; here only a failure of the model or of the server (load failure, 5xx, 429) fails over, because retrying a
// malformed request on every model of the chain only multiplies the load. The final error is the upstream answer itself, not a re-worded
// "All N fallback attempts failed" message.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// split "b, c ,,b" into {"b","c"}: trimmed, empty entries and repeats dropped, order kept
inline std::vector<std::string> server_failover_split(const std::string & list) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) {
            end = list.size();
        }
        std::string item = list.substr(pos, end - pos);
        const size_t b = item.find_first_not_of(" \t");
        const size_t e = item.find_last_not_of(" \t");
        item = b == std::string::npos ? std::string() : item.substr(b, e - b + 1);
        if (!item.empty()) {
            bool seen = false;
            for (const auto & o : out) {
                seen = seen || o == item;
            }
            if (!seen) {
                out.push_back(item);
            }
        }
        pos = end + 1;
    }
    return out;
}

// The models to try, in order: `primary` first, then each name of `list` that `resolve` knows (it returns the canonical model name, or an
// empty string for an unknown one, which is skipped) and that is not already in the chain. `max_attempts` > 0 caps the chain length
// (MLflow base.py:737-738); 0 means the whole chain.
inline std::vector<std::string> server_failover_chain(
        const std::string & primary,
        const std::string & list,
        const std::function<std::string(const std::string &)> & resolve,
        int32_t max_attempts) {
    std::vector<std::string> chain{primary};
    for (const auto & name : server_failover_split(list)) {
        const std::string canon = resolve ? resolve(name) : name;
        if (canon.empty()) {
            continue;
        }
        bool seen = false;
        for (const auto & c : chain) {
            seen = seen || c == canon;
        }
        if (!seen) {
            chain.push_back(canon);
        }
    }
    if (max_attempts > 0 && (size_t) max_attempts < chain.size()) {
        chain.resize((size_t) max_attempts);
    }
    return chain;
}

// Does this upstream status mean "this model could not serve it, another may"? A 4xx other than 429 is the client's own fault and
// would be repeated by every model of the chain.
inline bool server_failover_status(int status) {
    return status == 429 || status >= 500;
}
