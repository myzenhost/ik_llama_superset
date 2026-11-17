#pragma once

// Cross-site request guards for a server that has NO api key (header-only, no server state, no model).
//
// A server on loopback without --api-key can be driven by any web page the user opens:
//   * DNS rebinding: the page's own domain is re-pointed at 127.0.0.1, so its requests arrive with
//     "Host: evil.example" and the browser lets the page READ the answers;
//   * a cross-site POST: <form> / fetch with Content-Type text/plain needs no CORS preflight, so the request
//     is processed (generation, slot save/restore/erase, model management) even though the answer is unreadable.
// Both are refused here. With an api key the check is skipped: a foreign page cannot authenticate, and tunnels
// and reverse proxies keep working. Requests without an Origin header (curl, SDKs, server-side proxies) are
// never affected by the Origin rule.
//
// Spec from Strata serve/server.py (host_allowed, origin_allowed, _foreign_origin), MIT, Strata @82f46a8c8;
// no code copied. Row S69-STRATA-S2-SERVER-DNS-REBINDING-HOST-CHECK-AND-FOREIGN-ORIGIN-POST-REFUSAL-2026-10-07-01.

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace hostguard {

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// "Host: example.com:8080" / "[::1]:8080" / "::1" -> "example.com" / "::1" / "::1" (lower case, no port, no brackets)
inline std::string host_only(const std::string & hostport) {
    std::string h = lower(hostport);
    while (!h.empty() && std::isspace((unsigned char) h.front())) { h.erase(h.begin()); }
    while (!h.empty() && std::isspace((unsigned char) h.back()))  { h.pop_back(); }
    if (!h.empty() && h[0] == '[') {
        const size_t close = h.find(']');
        return close == std::string::npos ? h.substr(1) : h.substr(1, close - 1);
    }
    const size_t first = h.find(':');
    if (first == std::string::npos) {
        return h;
    }
    if (h.find(':', first + 1) != std::string::npos) {
        return h; // bare IPv6 literal, the colons are not a port
    }
    return h.substr(0, first);
}

// An Origin header value ("https://evil.example:8443", "null") -> "host[:port]" ("evil.example:8443"), lower case
inline std::string origin_hostport(const std::string & origin) {
    std::string o = lower(origin);
    const size_t scheme = o.find("://");
    if (scheme != std::string::npos) {
        o = o.substr(scheme + 3);
    }
    const size_t slash = o.find('/');
    if (slash != std::string::npos) {
        o = o.substr(0, slash);
    }
    return o;
}

inline bool is_ip_literal(const std::string & h) {
    if (h.empty()) {
        return false;
    }
    if (h.find(':') != std::string::npos) {
        return true; // IPv6
    }
    int dots = 0;
    for (char c : h) {
        if (c == '.') {
            dots++;
        } else if (!std::isdigit((unsigned char) c)) {
            return false;
        }
    }
    return dots == 3;
}

inline bool is_loopback_name(const std::string & h) {
    if (h == "localhost" || h == "::1" || h == "0:0:0:0:0:0:0:1") {
        return true;
    }
    if (h.size() > 10 && h.compare(h.size() - 10, 10, ".localhost") == 0) {
        return true;
    }
    return h.compare(0, 4, "127.") == 0 && is_ip_literal(h);
}

// exact match, or "*" in the list
inline bool list_has(const std::vector<std::string> & list, const std::string & h) {
    for (const auto & e : list) {
        const std::string le = lower(e);
        if (le == "*" || le == h) {
            return true;
        }
    }
    return false;
}

struct config {
    bool bind_loopback = true;                 // the server listens on loopback only (the rebinding case)
    std::vector<std::string> allowed_hosts;    // extra Host names ("*" = any)
    bool cors_open = false;                    // --cors-origins * given on the command line: the operator opted in
    std::vector<std::string> allowed_origins;  // explicit --cors-origins entries: "host[:port]" or full origins
};

// "" = let the request through, otherwise the reason it is refused (403).
// Only call this for a request with no api key configured.
inline std::string check(const config & c, const std::string & method, const std::string & host_header, const std::string & origin_header) {
    const std::string host = host_only(host_header);

    // 1. Host header: on a loopback-only server a name that is not loopback, not an IP literal and not allowed is a
    //    rebinding page. A missing Host (HTTP/1.0) is let through, as is anything when the server listens beyond loopback.
    if (c.bind_loopback && !host.empty() &&
        !is_loopback_name(host) && !is_ip_literal(host) && !list_has(c.allowed_hosts, host)) {
        return "Host header '" + host_header + "' is not allowed (DNS rebinding guard); add it with --allowed-hosts";
    }

    // 2. A state-changing request from a page of ANOTHER site. GET/HEAD/OPTIONS have no side effects here and OPTIONS is
    //    the browser's own preflight, which the CORS handler answers.
    if (origin_header.empty() || c.cors_open) {
        return "";
    }
    const std::string m = method;
    if (m == "GET" || m == "HEAD" || m == "OPTIONS") {
        return "";
    }
    const std::string oh   = origin_hostport(origin_header);   // host[:port]
    const std::string oh_h = host_only(oh);
    if (oh == lower(host_header) || is_loopback_name(oh_h) || list_has(c.allowed_origins, oh) || list_has(c.allowed_origins, oh_h) ||
        list_has(c.allowed_origins, lower(origin_header))) {
        return "";
    }
    return "cross-site request from Origin '" + origin_header + "' refused; allow it with --cors-origins";
}

} // namespace hostguard
