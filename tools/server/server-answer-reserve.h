// Opt-in answer reserve for reasoning models: derive a thinking-token budget from the generation limit so a plain request
// (max_tokens set, no reasoning budget, no reasoning_effort) cannot spend its whole limit inside the think block and come
// back with an empty answer.
//
// Header-only and dependency-free (tested by tools/server/tests_cpp/test_server_answer_reserve.cpp).
// Row S69-REASONING-ANSWER-RESERVE-OPT-IN-2026-10-17-01.
//
// Server flag --reasoning-answer-reserve N|P% (default -1 = off) and request field reasoning_answer_reserve (an integer
// token count, or a string "N" / "P%"; -1 turns a server reserve off for that request). When active and the request has
// no explicit reasoning budget and no effort-derived budget, the reasoning budget is
//     limit - reserve, floored at 0,
// where reserve = N tokens, or ceil(P% of limit) so the answer always gets at least P%. limit is the generation limit the
// effort budget also uses (server_effort_table::effective_limit: the request's max_completion_tokens / max_tokens /
// n_predict, capped by the server's --n-predict). With no finite limit the reserve derives nothing. The existing
// reasoning-budget sampler (common/reasoning-budget.cpp) forces the close tag, with --reasoning-budget-message before it.
//
// Precedence (resolve): request reasoning_budget_tokens > effort budget > reserve-derived > server --reasoning-budget.
//
// RESEARCH-DONE: temp/research/findings/S69_REASONING_OUTPUT_CAP_RESERVE_RESEARCH_2026-10-17.md. No engine read there
// derives the thinking cap from max_tokens: vLLM and TensorRT-LLM take an explicit thinking_token_budget only, OpenAI
// leaves it to the client (status incomplete), Anthropic validates budget_tokens < max_tokens. This is the "room for the
// answer" guarantee Anthropic gets by validation, given to clients that only send max_tokens. No code copied.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>

struct server_answer_reserve {
    int    tokens   = -1;  // reserve in tokens, -1 = not a token reserve
    double fraction = -1;  // reserve as a fraction of the limit in [0, 1], -1 = not a percentage

    bool active() const { return tokens >= 0 || fraction >= 0; }

    void clear() { tokens = -1; fraction = -1; }

    // "" / "-1" / "off" -> off; "N" (N >= 0, whole number) -> tokens; "P%" (0 <= P <= 100) -> fraction.
    // Returns false and fills err on a malformed spec; the reserve is then left off.
    bool parse(const std::string & spec_in, std::string & err) {
        clear();
        const std::string spec = trim(spec_in);
        if (spec.empty() || spec == "-1" || spec == "off") {
            return true;
        }
        std::string val = spec;
        const bool pct = val.back() == '%';
        if (pct) {
            val.pop_back();
        }
        if (val.empty()) {
            err = "expected N or P%, got '" + spec + "'";
            return false;
        }
        char * end = nullptr;
        if (pct) {
            const double p = std::strtod(val.c_str(), &end);
            if (end == nullptr || *end != '\0' || !std::isfinite(p) || p < 0.0 || p > 100.0) {
                err = "percentage must be a number in [0, 100], got '" + spec + "'";
                return false;
            }
            fraction = p / 100.0;
            return true;
        }
        const long long n = std::strtoll(val.c_str(), &end, 10);
        if (end == nullptr || *end != '\0' || n < 0 || n > 0x7fffffffLL) {
            err = "token reserve must be a whole number >= 0 (or -1 for off), got '" + spec + "'";
            return false;
        }
        tokens = (int) n;
        return true;
    }

    // the reasoning budget this reserve leaves inside `limit`, or -1 = nothing derived (reserve off, or no finite limit)
    int budget(int limit) const {
        if (!active() || limit <= 0) {
            return -1;
        }
        long long reserve = tokens >= 0 ? (long long) tokens : (long long) std::ceil(fraction * (double) limit);
        return (int) std::max<long long>(0, (long long) limit - reserve);
    }

    // request budget > effort budget > reserve-derived budget > server default; each -1 = absent. The request value is
    // passed through for anything other than -1, exactly as the server did before this existed.
    static int resolve(int request_budget, int effort_budget, int reserve_budget, int server_default) {
        if (request_budget != -1) return request_budget;
        if (effort_budget  >= 0) return effort_budget;
        if (reserve_budget >= 0) return reserve_budget;
        return server_default;
    }

private:
    static std::string trim(const std::string & s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char) s[a])) a++;
        while (b > a && std::isspace((unsigned char) s[b - 1])) b--;
        return s.substr(a, b - a);
    }
};
