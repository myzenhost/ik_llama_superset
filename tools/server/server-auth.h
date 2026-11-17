// Two-tier API keys (regular + admin), constant-time comparison and hot-reloadable key files.
//
// Header-only and free of llama / ggml / http dependencies (tested by tools/server/tests_cpp/test_server_auth.cpp, which also
// checks that the admin route list below matches the routes server.cpp wraps with mgmt()).
//
// Behaviour (own design; TabbyAPI's auth was read for the idea only):
//   * regular keys open the inference routes; an ADMIN key also opens the management routes (model load/unload, control
//     vectors, LoRA hotswap, slot save/restore/erase, saved prompts, the SQL session store, POST /props) and is accepted
//     everywhere a regular key is;
//   * nothing changes unless an admin key is configured: with only regular keys every valid key opens every route, as before;
//   * keys given as files are re-read when the file changes (checked every few seconds), so an operator can add or pull a
//     key without a restart; a re-read that would leave NO key at all (a truncated file, an editor mid-save) is refused and
//     the previous keys stay, so an I/O accident cannot silently turn authentication off;
//   * the presented key is compared with every configured key in time that depends on the lengths only, never on where the
//     first differing byte is.
// SOURCE-READ: upstream/tabbyAPI/common/auth.py:24-57, :82-119, :185-216 (theroyallab/tabbyAPI @be74bf0a, AGPL-3.0),
// whole file read; behaviour only, spec written in my own words, implemented with my own structure: no code copied.
// SOURCE-NUANCE: (1) the admin key passes the regular check too (superset): kept. (2) A reload that fails (partial write, invalid
// YAML, missing keys) KEEPS the previous keys, because admin_key is mandatory so an empty file fails validation; here there is
// no such schema, so the same protection is an explicit rule: a re-read that leaves no key at all is refused. (3) It has exactly
// one admin key and a fixed 2.0 s poll on mtime alone; here any number of admin keys may be configured so a rotation can overlap
// old and new, and a change is (mtime, size) so a same-second rewrite of different length is seen. (4) An admin key is read from
// X-Admin-Key, X-Api-Key or Authorization: Bearer; a bad admin key answers 401 there, while here a VALID regular key on an admin
// route answers 403 (authenticated, not permitted) and an unknown key 401. (5) Comparison there is plain == / set membership
// (not constant time); here it is constant time. (6) It writes fresh random keys to the file when absent; not adopted: the
// server never mints credentials. (7) Its startup warning for "auth disabled while CORS is *" already exists here (server.cpp).
// See docs/THIRD_PARTY_NOTICES_E_SERVING.md.
#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <string>
#include <utility>
#include <vector>

enum server_auth_level {
    SERVER_AUTH_NONE  = 0, // not a configured key
    SERVER_AUTH_API   = 1, // a regular key
    SERVER_AUTH_ADMIN = 2, // an admin key
};

// equal keys compare equal; the work done depends on max(len a, len b) only. Key length is not treated as secret.
inline bool server_auth_ct_equal(const std::string & a, const std::string & b) {
    const size_t n = std::max(a.size(), b.size());
    unsigned char diff = (unsigned char) (a.size() != b.size());
    for (size_t i = 0; i < n; i++) {
        const unsigned char ca = i < a.size() ? (unsigned char) a[i] : 0;
        const unsigned char cb = i < b.size() ? (unsigned char) b[i] : 0;
        diff |= (unsigned char) (ca ^ cb);
    }
    return diff == 0;
}

// the routes that need the admin key when one is configured. A pattern segment starting with ':' matches any one segment.
// MUST match the paths server.cpp wraps with mgmt(...) (test_server_auth.cpp compares them).
struct server_auth_route {
    const char * method; // "GET" | "POST" | "DELETE"
    const char * pattern;
};
inline const std::vector<server_auth_route> & server_auth_admin_routes() {
    static const std::vector<server_auth_route> routes = {
        {"POST", "/props"}, {"POST", "/lora-adapters"}, {"POST", "/sampler-overrides"},
        {"POST", "/control-vectors/load"}, {"POST", "/control-vectors/unload"}, {"POST", "/control-vectors/apply"},
        {"POST", "/load"}, {"POST", "/save"}, {"POST", "/rename"}, {"POST", "/all"}, {"POST", "/sessions"}, {"POST", "/delete"},
        {"GET",  "/vacuum"},
        {"GET",  "/zstd_get_configs"}, {"POST", "/zstd_incremental_maintenance"},
        {"POST", "/zstd_enable_transparent"}, {"POST", "/zstd_update_transparent"},
        {"POST", "/slots/:id_slot"}, {"POST", "/delete_prompt"}, {"POST", "/rename_prompt"},
        {"POST", "/models"}, {"POST", "/models/load"}, {"POST", "/models/unload"}, {"DELETE", "/models"},
    };
    return routes;
}

inline std::vector<std::string> server_auth_split_path(const std::string & p) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= p.size()) {
        size_t j = p.find('/', i);
        if (j == std::string::npos) j = p.size();
        if (j > i) out.push_back(p.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

// method is as the HTTP layer reports it (upper case); path may carry a query string and the configured api prefix
inline bool server_auth_is_admin_route(const std::string & method, std::string path, const std::string & api_prefix = "") {
    if (const size_t q = path.find('?'); q != std::string::npos) {
        path.resize(q);
    }
    if (!api_prefix.empty() && path.compare(0, api_prefix.size(), api_prefix) == 0) {
        path.erase(0, api_prefix.size());
    }
    const auto segs = server_auth_split_path(path);
    for (const auto & r : server_auth_admin_routes()) {
        if (method != r.method) {
            continue;
        }
        const auto pat = server_auth_split_path(r.pattern);
        if (pat.size() != segs.size()) {
            continue;
        }
        bool ok = true;
        for (size_t i = 0; i < pat.size() && ok; i++) {
            ok = (!pat[i].empty() && pat[i][0] == ':') || pat[i] == segs[i];
        }
        if (ok) {
            return true;
        }
    }
    return false;
}

// --models-register-allow: which Hugging Face repos a runtime registration (router POST /models) may download.
// SOURCE-READ: upstream/torchserve/frontend/archive/src/main/java/org/pytorch/serve/archive/utils/ArchiveUtils.java:75-90
// (validateURL), .../archive/model/ModelArchive.java:46-60 (downloadModel) and docs/configuration.md:314-316 (allowed_urls), pytorch/serve
// @62c4d6a1, Apache-2.0. Own implementation, no code copied.
// SOURCE-NUANCE: (1) a pattern must match the WHOLE name (Matcher.matches), case-insensitively, so "acme/.*" does not admit
// "xacme/..." or "mirror/acme/..."; (2) ".." is refused before any pattern is tried, so an allowed prefix cannot carry a traversal;
// (3) the default list allows everything, so an operator who sets nothing sees no change; (4) TorchServe reads the list from the
// environment only when a second switch allows it (an env var must not WIDEN the list behind the operator's back): this flag
// therefore has no LLAMA_ARG_* environment name.
// Returns "" when `repo` may be registered, else the reason. An empty list allows every repo (today's behaviour). A pattern that
// does not compile refuses; start-up validates each pattern with server_models_register_pattern_ok, so this is only a backstop.
inline std::string server_models_register_refusal(const std::vector<std::string> & patterns, const std::string & repo) {
    if (repo.find("..") != std::string::npos) {
        return "model name must not contain \"..\"";   // Hugging Face repo ids cannot contain it either
    }
    for (const auto & p : patterns) {
        try {
            if (std::regex_match(repo, std::regex(p, std::regex::ECMAScript | std::regex::icase))) {
                return "";
            }
        } catch (const std::regex_error &) {
            return "invalid --models-register-allow pattern: " + p;
        }
    }
    if (patterns.empty()) {
        return "";
    }
    return "model '" + repo + "' does not match any --models-register-allow pattern";
}

// start-up check for one --models-register-allow pattern; false (with err set) when it is empty or not a valid regex
inline bool server_models_register_pattern_ok(const std::string & p, std::string & err) {
    err.clear();
    if (p.empty()) {
        err = "--models-register-allow: empty pattern";
        return false;
    }
    try {
        std::regex re(p, std::regex::ECMAScript | std::regex::icase);
        (void) re;
    } catch (const std::regex_error & e) {
        err = std::string("--models-register-allow: invalid regex '") + p + "': " + e.what();
        return false;
    }
    return true;
}

// "one key per line, # comments, blank lines ignored, CR/LF and surrounding spaces/tabs stripped"; false when unreadable
inline bool server_auth_read_key_file(const std::string & path, std::vector<std::string> & out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t a = 0, b = line.size();
        while (a < b && (line[a] == ' ' || line[a] == '\t' || line[a] == '\r')) a++;
        while (b > a && (line[b - 1] == ' ' || line[b - 1] == '\t' || line[b - 1] == '\r')) b--;
        if (b > a && line[a] != '#') {
            out.push_back(line.substr(a, b - a));
        }
    }
    return true;
}

struct server_auth {
    mutable std::mutex mtx;
    std::vector<std::string> api;   // regular keys
    std::vector<std::string> admin; // admin keys

    bool enabled() const {
        std::lock_guard<std::mutex> lk(mtx);
        return !api.empty() || !admin.empty();
    }
    bool tiers_active() const {
        std::lock_guard<std::mutex> lk(mtx);
        return !admin.empty();
    }

    // checks EVERY configured key (no early exit)
    server_auth_level classify(const std::string & presented) const {
        std::lock_guard<std::mutex> lk(mtx);
        int level = SERVER_AUTH_NONE;
        for (const auto & k : api)   { if (server_auth_ct_equal(presented, k)) level = std::max(level, (int) SERVER_AUTH_API); }
        for (const auto & k : admin) { if (server_auth_ct_equal(presented, k)) level = std::max(level, (int) SERVER_AUTH_ADMIN); }
        return (server_auth_level) level;
    }

    // 0 = allowed, 401 = no valid key, 403 = a valid regular key on a route that needs the admin key
    int verdict(const std::string & presented, bool route_needs_admin) const {
        std::lock_guard<std::mutex> lk(mtx);
        if (api.empty() && admin.empty()) {
            return 0; // authentication not configured
        }
        int level = SERVER_AUTH_NONE;
        for (const auto & k : api)   { if (server_auth_ct_equal(presented, k)) level = std::max(level, (int) SERVER_AUTH_API); }
        for (const auto & k : admin) { if (server_auth_ct_equal(presented, k)) level = std::max(level, (int) SERVER_AUTH_ADMIN); }
        if (level == SERVER_AUTH_NONE) {
            return 401;
        }
        if (route_needs_admin && !admin.empty() && level < SERVER_AUTH_ADMIN) {
            return 403;
        }
        return 0;
    }

    // the same decision for a request that may carry the key in several headers (Authorization, X-Api-Key, X-Admin-Key):
    // the best level among the candidates counts, empty candidates are ignored
    int verdict_multi(const std::vector<std::string> & candidates, bool route_needs_admin) const {
        std::lock_guard<std::mutex> lk(mtx);
        if (api.empty() && admin.empty()) {
            return 0;
        }
        int level = SERVER_AUTH_NONE;
        for (const auto & presented : candidates) {
            if (presented.empty()) {
                continue;
            }
            for (const auto & k : api)   { if (server_auth_ct_equal(presented, k)) level = std::max(level, (int) SERVER_AUTH_API); }
            for (const auto & k : admin) { if (server_auth_ct_equal(presented, k)) level = std::max(level, (int) SERVER_AUTH_ADMIN); }
        }
        if (level == SERVER_AUTH_NONE) {
            return 401;
        }
        if (route_needs_admin && !admin.empty() && level < SERVER_AUTH_ADMIN) {
            return 403;
        }
        return 0;
    }

    // swap in a new key set; refuses (returns false, keeps the old keys) when it would leave no key at all while the old set had some
    bool replace(std::vector<std::string> new_api, std::vector<std::string> new_admin) {
        std::lock_guard<std::mutex> lk(mtx);
        if (new_api.empty() && new_admin.empty() && !(api.empty() && admin.empty())) {
            return false;
        }
        api   = std::move(new_api);
        admin = std::move(new_admin);
        return true;
    }
};

// remembers the (mtime, size) of the key files and says whether any changed since the last look
struct server_auth_file_watch {
    std::vector<std::string> files;
    std::vector<std::pair<int64_t, int64_t>> last; // (mtime ticks, size) per file; (-1, -1) = unreadable

    static std::pair<int64_t, int64_t> stamp(const std::string & f) {
        std::error_code ec;
        const auto t = std::filesystem::last_write_time(f, ec);
        if (ec) return {-1, -1};
        const auto s = std::filesystem::file_size(f, ec);
        if (ec) return {-1, -1};
        return {(int64_t) t.time_since_epoch().count(), (int64_t) s};
    }

    void init(std::vector<std::string> fs) {
        files = std::move(fs);
        last.clear();
        for (const auto & f : files) last.push_back(stamp(f));
    }

    bool changed() {
        bool any = false;
        for (size_t i = 0; i < files.size(); i++) {
            const auto s = stamp(files[i]);
            if (s != last[i]) {
                last[i] = s;
                any = true;
            }
        }
        return any;
    }
};
