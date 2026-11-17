// Rescue of a tool call a model wrote INSIDE its reasoning and then stopped, without ever closing the thinking block.
//
// WHY: a reasoning model sometimes ends its turn while still "thinking", with the call it meant to make as the last thing
// it wrote. The parser files all of that as reasoning, the client gets an empty answer and the turn is lost (measured by
// the Strata project on 12 live strandings). The opposite error is worse: a model that only QUOTES the call format
// ("I must write <tool_call> then <function=...>") must not execute anything, and a call cut by max_tokens must never run.
//
// RULES (own spec, from Strata serve/frontend.py + serve/test_reasoning_rescue.py @82f46a8c8, MIT, behaviour only, no
// code copied; fixture serve/fixtures/rcall_specimens.json, 37 specimens). The format is the Qwen3-Coder XML one:
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
//   - an opener counts only at the START of a line (leading blanks allowed), outside a ``` / ~~~ fence and outside
//     inline code (an odd number of backticks earlier in the paragraph; a blank line ends the paragraph). A fence the
//     text never closes does not count: the turn ended inside it (rcall-live-1017, measured live on Qwen3.5-0.8B)
//   - it must be followed (after blanks) by <function=NAME>; otherwise it is prose naming the format
//   - the call ends at </tool_call> after </function>, found by walking the parameters; a </parameter> closes a value only
//     when the next thing (after blanks) is another <parameter= or </function>, so a value may itself contain
//     </parameter>, </function> or </tool_call> text
//   - a call is ACTED ON only when nothing but blanks or further calls follows it up to the end of the reasoning text;
//     any other text after it means it was a quote
//   - only calls to a DECLARED tool are delivered; an undeclared one (a wrapper such as "tool_call") stays reasoning
//   - the caller decides the rest of the gates: tools offered, the turn ended on its own (not max_tokens), the thinking
//     block never closed (no content, no parsed calls)
// The text of the call is LEFT in reasoning_content: reasoning that was streamed to a client cannot be retracted, so the
// stream and the whole message agree.

#include "chat.h"
#include "json.h"
#include "log.h"

#include <cstring>
#include <string>
#include <vector>

using json = common_json;

namespace {

struct rescued_call {
    std::string name;
    std::vector<std::pair<std::string, std::string>> params;
    size_t end = 0; // one past </tool_call>
};

bool is_blank(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

size_t skip_blank(const std::string & s, size_t i) {
    while (i < s.size() && is_blank(s[i])) {
        i++;
    }
    return i;
}

bool starts_at(const std::string & s, size_t i, const char * lit) {
    return s.compare(i, std::strlen(lit), lit) == 0;
}

// <tool_call> <function=NAME> ... </function> </tool_call> starting exactly at `at`; false when it is not a whole call
bool parse_call(const std::string & s, size_t at, rescued_call & out) {
    static const char TC_OPEN[]  = "<tool_call>";
    static const char TC_CLOSE[] = "</tool_call>";
    static const char FN_OPEN[]  = "<function=";
    static const char FN_CLOSE[] = "</function>";
    static const char PR_OPEN[]  = "<parameter=";
    static const char PR_CLOSE[] = "</parameter>";

    if (!starts_at(s, at, TC_OPEN)) {
        return false;
    }
    size_t i = skip_blank(s, at + std::strlen(TC_OPEN));
    if (!starts_at(s, i, FN_OPEN)) {
        return false;
    }
    i += std::strlen(FN_OPEN);
    const size_t name_end = s.find('>', i);
    if (name_end == std::string::npos) {
        return false;
    }
    out = rescued_call();
    out.name = s.substr(i, name_end - i);
    if (out.name.empty() || out.name.find('<') != std::string::npos) {
        return false;
    }
    i = name_end + 1;

    for (;;) {
        i = skip_blank(s, i);
        if (starts_at(s, i, FN_CLOSE)) {
            i = skip_blank(s, i + std::strlen(FN_CLOSE));
            if (!starts_at(s, i, TC_CLOSE)) {
                return false;
            }
            out.end = i + std::strlen(TC_CLOSE);
            return true;
        }
        if (!starts_at(s, i, PR_OPEN)) {
            return false;
        }
        i += std::strlen(PR_OPEN);
        const size_t key_end = s.find('>', i);
        if (key_end == std::string::npos) {
            return false;
        }
        const std::string key = s.substr(i, key_end - i);
        const size_t      val_begin = key_end + 1;
        // the value ends at a </parameter> that is followed by the next <parameter= or by </function>
        size_t close = val_begin;
        size_t val_end = std::string::npos;
        while ((close = s.find(PR_CLOSE, close)) != std::string::npos) {
            const size_t after = skip_blank(s, close + std::strlen(PR_CLOSE));
            if (starts_at(s, after, PR_OPEN) || starts_at(s, after, FN_CLOSE)) {
                val_end = close;
                break;
            }
            close += std::strlen(PR_CLOSE);
        }
        if (val_end == std::string::npos) {
            return false;
        }
        std::string value = s.substr(val_begin, val_end - val_begin);
        if (!value.empty() && value.front() == '\n') {
            value.erase(0, 1);
        }
        if (!value.empty() && value.back() == '\n') {
            value.pop_back();
        }
        out.params.emplace_back(key, value);
        i = val_end + std::strlen(PR_CLOSE);
    }
}

} // namespace

size_t common_chat_rescue_reasoning_calls(common_chat_msg & msg, const std::vector<common_chat_tool> & tools) {
    if (tools.empty() || !msg.tool_calls.empty() || msg.reasoning_content.empty()) {
        return 0;
    }
    if (skip_blank(msg.content, 0) != msg.content.size()) {
        return 0; // the thinking block was closed and an answer written: the normal parser owns that text
    }
    const std::string & r = msg.reasoning_content;
    const std::vector<size_t> starts = common_chat_reasoning_call_line_starts(r);

    std::vector<rescued_call> delivered;
    size_t resume = 0; // first byte that may still open a candidate
    for (const size_t ls : starts) {
        if (ls < resume) {
            continue;
        }
        size_t p = ls;
        while (p < r.size() && (r[p] == ' ' || r[p] == '\t')) {
            p++;
        }
        rescued_call first;
        if (!parse_call(r, p, first)) {
            continue;
        }
        // the group: this call and any calls right after it, settled only by reaching the end of the text
        std::vector<rescued_call> group{ first };
        size_t tail = first.end;
        bool   settled = false;
        for (;;) {
            tail = skip_blank(r, tail);
            if (tail >= r.size()) {
                settled = true;
                break;
            }
            rescued_call next;
            if (!parse_call(r, tail, next)) {
                break;
            }
            tail = next.end;
            group.push_back(std::move(next));
        }
        if (settled) {
            delivered.clear();
            for (auto & c : group) {
                delivered.push_back(std::move(c));
            }
            break;
        }
        resume = first.end; // quoted: all of it stays reasoning; look for a later, real call
    }

    size_t n = 0;
    for (const auto & c : delivered) {
        bool declared = false;
        for (const auto & t : tools) {
            if (t.name == c.name) {
                declared = true;
                break;
            }
        }
        if (!declared) {
            continue;
        }
        json args = json::object();
        for (const auto & kv : c.params) {
            args[kv.first] = kv.second;
        }
        common_chat_tool_call call;
        call.name      = c.name;
        call.arguments = args.dump();
        msg.tool_calls.push_back(std::move(call));
        n++;
    }
    if (n > 0) {
        // values are text so far: repair them to the declared types (a number, a JSON array) as a parsed call would be
        common_chat_coerce_tool_args(msg, tools);
        LOG_WRN("%s: %zu tool call(s) written inside the reasoning and never closed were delivered as tool calls\n", __func__, n);
    }
    return n;
}

std::vector<size_t> common_chat_reasoning_call_line_starts(const std::string & r) {
    // line starts, with the fence / inline-code state at the beginning of each line
    std::vector<size_t> line_start;
    std::vector<char>   line_in_fence;
    std::vector<char>   line_ticks_odd;
    bool   in_fence = false;
    size_t fence_li = 0; // line of the fence opener still open
    {
        char   fence_ch = 0;
        size_t ticks    = 0;
        size_t pos      = 0;
        while (pos <= r.size()) {
            size_t eol = r.find('\n', pos);
            if (eol == std::string::npos) {
                eol = r.size();
            }
            line_start.push_back(pos);
            line_in_fence.push_back(in_fence ? 1 : 0);
            line_ticks_odd.push_back((ticks % 2) ? 1 : 0);

            size_t p = pos;
            while (p < eol && (r[p] == ' ' || r[p] == '\t')) {
                p++;
            }
            const bool blank = p >= eol || r[p] == '\r';
            if (blank) {
                ticks = 0; // a blank line ends the paragraph, and with it an unclosed inline code span
            } else if (starts_at(r, p, "```") || starts_at(r, p, "~~~")) {
                const char ch = r[p];
                if (!in_fence) {
                    in_fence = true;
                    fence_ch = ch;
                    fence_li = line_start.size() - 1;
                } else if (ch == fence_ch) {
                    in_fence = false;
                }
            } else if (!in_fence) {
                for (size_t k = pos; k < eol; k++) {
                    if (r[k] == '`') {
                        ticks++;
                    }
                }
            }
            if (eol >= r.size()) {
                break;
            }
            pos = eol + 1;
        }
    }

    if (in_fence) {
        // A fence the turn never closed is not a quote: the model stopped inside it, so nothing follows what it wrote
        // there (a quoted call needs text after it, and a closed fence is such text). Measured live 2026-10-17 on
        // Qwen3.5-0.8B: "I'll make the tool call now.\n```\n<tool_call>...</tool_call>" then <|im_end|>; the format
        // parser delivers that call, and counting the open fence would have lost it.
        for (size_t li = fence_li + 1; li < line_start.size(); li++) {
            line_in_fence[li] = 0;
        }
    }

    std::vector<size_t> out;
    for (size_t li = 0; li < line_start.size(); li++) {
        if (!line_in_fence[li] && !line_ticks_odd[li]) {
            out.push_back(line_start[li]);
        }
    }
    return out;
}
