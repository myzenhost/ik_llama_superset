// OpenAI "reasoning_effort" -> a thinking-token budget, as a fraction of the generation limit.
//
// Header-only and dependency-free (tested by tools/server/tests_cpp/test_server_effort.cpp).
//
// The OpenAI field normally only reaches the chat template as the `reasoning_effort` variable, which only gpt-oss-style
// templates read; for every other thinking model an SDK client that sends effort=low gets unrestricted thinking. With a
// table configured, an effort level also becomes a budget: floor(fraction * generation limit), enforced by the existing
// reasoning-budget sampler. An explicit reasoning_budget_tokens in the request always wins; "none" keeps its meaning
// (thinking off); an unknown level, or no finite generation limit, means no budget (unrestricted, as before).
//
// SOURCE-READ: upstream/koboldcpp/koboldcpp.py:2332-2349 and gpttype_adapter.cpp:2282-2333 (koboldcpp
// @36688de9, AGPL-3.0). Read for behaviour, spec written in my own words, implemented here with my own structure: no code
// copied. The fractions are koboldcpp's convention (minimal 10%, low 25%, medium 50%, high 75%), not an OpenAI one, so
// they are data the operator can change (--reasoning-effort-budget).
// SOURCE-NUANCE: (1) koboldcpp applies the fraction to max_length AFTER its genlimit and context-headroom clamps, i.e.
// to the length it will really generate, not the number the client asked for; here the caller passes the effective limit
// (request limit capped by the server's --n-predict). (2) its conversion is an int truncation, so a tiny limit gives a
// budget of 0, which the sampler reads as "no thinking at all"; this port floors at 1 token for a mapped level, a
// deliberate difference. (3) in koboldcpp the effort-derived budget WINS over thinking_budget_tokens (that field is read
// only when the effort gave -1); here an explicit numeric budget in the request wins over the derived one, because it is
// the more specific instruction. (4) "none" is budget 0 there; here "none" keeps its existing meaning (enable_thinking off).
// (5) the enforcing sampler counts tokens since the latest UNCLOSED think-start, per block, and force-closes with a fixed
// phrase; this server's reasoning-budget sampler already does that, so only the derivation is new.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>

struct server_effort_table {
    std::map<std::string, double> fraction; // lowercase level -> (0, 1]

    bool empty() const { return fraction.empty(); }

    static server_effort_table preset_default() {
        server_effort_table t;
        t.fraction = { {"minimal", 0.10}, {"low", 0.25}, {"medium", 0.50}, {"high", 0.75} };
        return t;
    }

    // "default" -> preset_default(); otherwise "level=fraction,level=fraction" with fractions in (0, 1], or a percentage
    // written with %. Returns false and fills err on a malformed spec; the table is then left empty.
    bool parse(const std::string & spec, std::string & err) {
        fraction.clear();
        if (spec == "default") {
            *this = preset_default();
            return true;
        }
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t eq = item.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 >= item.size()) {
                err = "expected level=fraction, got '" + item + "'";
                fraction.clear();
                return false;
            }
            std::string level = lower(trim(item.substr(0, eq)));
            std::string val   = trim(item.substr(eq + 1));
            const bool pct = !val.empty() && val.back() == '%';
            if (pct) {
                val.pop_back();
            }
            char * end = nullptr;
            double f = std::strtod(val.c_str(), &end);
            if (val.empty() || end == nullptr || *end != '\0' || !std::isfinite(f)) {
                err = "not a number: '" + val + "' for level '" + level + "'";
                fraction.clear();
                return false;
            }
            if (pct) {
                f /= 100.0;
            }
            if (!(f > 0.0 && f <= 1.0)) {
                err = "fraction for '" + level + "' must be in (0, 1], got " + val;
                fraction.clear();
                return false;
            }
            fraction[level] = f;
        }
        if (fraction.empty()) {
            err = "empty table";
            return false;
        }
        return true;
    }

    // budget in tokens for this effort level, or -1 = no budget derived (unknown level, no table, or no finite generation
    // limit). A level that maps to a fraction always yields at least 1 token (see SOURCE-NUANCE 2).
    int budget(const std::string & effort, int effective_limit) const {
        if (fraction.empty() || effective_limit <= 0) {
            return -1;
        }
        auto it = fraction.find(lower(effort));
        if (it == fraction.end()) {
            return -1;
        }
        const long long b = (long long) std::floor(it->second * (double) effective_limit);
        return (int) std::max<long long>(1, b);
    }

    // the generation length the budget is a fraction of: the request's own limit, capped by the server's, each ignored
    // when not finite (<= 0); -1 when neither is finite
    static int effective_limit(int request_limit, int server_limit) {
        if (request_limit > 0 && server_limit > 0) return std::min(request_limit, server_limit);
        if (request_limit > 0) return request_limit;
        if (server_limit  > 0) return server_limit;
        return -1;
    }

private:
    static std::string trim(const std::string & s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char) s[a])) a++;
        while (b > a && std::isspace((unsigned char) s[b - 1])) b--;
        return s.substr(a, b - a);
    }
    static std::string lower(std::string s) {
        for (auto & c : s) c = (char) std::tolower((unsigned char) c);
        return s;
    }
};
