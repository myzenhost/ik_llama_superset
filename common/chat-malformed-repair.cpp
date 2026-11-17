// Repair of a tool call the model wrote in a slightly malformed form, on the FINAL parse of a turn the model ended itself.
//
// WHY: the PEG parsers here accept exactly the call grammar of each model's chat template. A real model drifts from it in
// small, repeatable ways (a closing tag dropped, a string left open, a value written bare or in single quotes, the wrapper
// token of a call missing). The strict parser then either throws ("does not match the expected format", the whole turn is
// lost) or parses the call with the stray markup INSIDE an argument (the tool runs with a corrupt value). Ollama keeps a
// hand-written repair path per model for these families (measured 2026-10-27, tests/test-chat.cpp
// test_ollama_malformed_tolerance: 26 of 38 specimens failed here before this file).
//
// SPEC (own, written from the behaviour table of Ollama's parsers, MIT, read at f067033f; NO CODE COPIED):
//   model/parsers/glm46.go:464-600 (repairGLM46XML) + glm46.go:143-201 (finalize)  -> GLM-4.x XML calls
//   model/parsers/gemma4.go:336-394, :508-697                                       -> Gemma 4 dict calls
//   model/parsers/lfm2.go:154-161                                                   -> LFM2 bare call
// NUANCES THE DESCRIPTION DID NOT SAY (found by reading the source and by the specimens):
//   - the GLM-4.6 template writes a newline after every tag, GLM-4.7-Flash writes none; each template's grammar accepts only
//     its own form, so a repair that renders one dialect fails on the other model. The candidates are therefore rendered in
//     BOTH dialects and the real parser is the judge (the first that yields a clean call wins).
//   - the repair NEVER touches a turn the strict parser handled cleanly: it runs only when the strict parse threw, returned
//     no call although the text holds a call-shaped region, or returned a call whose arguments are not valid JSON or still
//     contain call markup. A well-formed call cannot change.
//   - a region followed by prose is a quote of the format, not a call (GLM, LFM2); a region inside a ``` fence is never
//     acted on. Only calls to a DECLARED tool are rebuilt.
//   - a Gemma 4 value that is not delimited is a string unless the tool schema types the parameter as other than string and
//     the value is a number/boolean/null literal; nested objects/arrays are not repaired (the region is left as written).
//   - a raw (undelimited) Gemma 4 value ends at a comma only when what follows is `name:` with name a declared parameter of
//     the tool, so "hello, world: x" stays one string.
//   - the arguments of a call whose closing marker never came are the arguments as far as they were written: the caller
//     (common_chat_peg_parse) keeps its drop_incomplete_tool_calls gate for turns the SERVER cut.
//
// The result is a list of candidate texts for the caller to parse with the ordinary parser; this file never builds a message.

#include "chat.h"
#include "json.h"

#include <cstring>
#include <set>
#include <string>
#include <vector>

using json = common_json;

namespace {

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::string trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return std::string();
    }
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool blank(const std::string & s) {
    return s.find_first_not_of(" \t\r\n") == std::string::npos;
}

bool at(const std::string & s, size_t i, const char * lit) {
    return s.compare(i, std::strlen(lit), lit) == 0;
}

const common_chat_tool * find_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & t : tools) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

json tool_properties(const common_chat_tool & t) {
    json p = json::parse_no_throw(t.parameters);
    if (!p.is_discarded() && p.is_object() && p.contains("properties") && p.at("properties").is_object()) {
        return p.at("properties");
    }
    return json::object();
}

// an odd number of ``` fences before `pos`: the position is inside a code block, a quote of the format
bool in_fence(const std::string & s, size_t pos) {
    size_t n = 0;
    for (size_t i = s.find("```"); i != std::string::npos && i < pos; i = s.find("```", i + 3)) {
        n++;
    }
    return (n % 2) == 1;
}

// ---- GLM-4.x: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call> ----------------------------------

enum glm_tag { G_TEXT, G_KO, G_KC, G_VO, G_VC };

struct glm_tok {
    glm_tag     tag;
    std::string text;
};

const char * const GLM_TC_OPEN  = "<tool_call>";
const char * const GLM_TC_CLOSE = "</tool_call>";

std::vector<glm_tok> glm_tokenize(const std::string & body) {
    static const struct { const char * lit; glm_tag tag; } TAGS[] = {
        { "<arg_key>", G_KO }, { "</arg_key>", G_KC }, { "<arg_value>", G_VO }, { "</arg_value>", G_VC },
    };
    std::vector<glm_tok> out;
    std::string          cur;
    size_t               i = 0;
    while (i < body.size()) {
        bool hit = false;
        if (body[i] == '<') {
            for (const auto & t : TAGS) {
                if (at(body, i, t.lit)) {
                    if (!cur.empty()) {
                        out.push_back({ G_TEXT, cur });
                        cur.clear();
                    }
                    out.push_back({ t.tag, std::string() });
                    i += std::strlen(t.lit);
                    hit = true;
                    break;
                }
            }
        }
        if (!hit) {
            cur.push_back(body[i++]);
        }
    }
    if (!cur.empty()) {
        out.push_back({ G_TEXT, cur });
    }
    return out;
}

// exactly (KO text KC ws VO text VC ws)* : the grammar the template writes
bool glm_regular(const std::vector<glm_tok> & toks) {
    size_t i = 0;
    auto ws_only = [&]() {
        if (i < toks.size() && toks[i].tag == G_TEXT && blank(toks[i].text)) {
            i++;
        }
    };
    auto text_opt = [&]() {
        if (i < toks.size() && toks[i].tag == G_TEXT) {
            i++;
        }
    };
    ws_only();
    while (i < toks.size()) {
        if (toks[i].tag != G_KO) { return false; }
        i++;
        text_opt();
        if (i >= toks.size() || toks[i].tag != G_KC) { return false; }
        i++;
        ws_only();
        if (i >= toks.size() || toks[i].tag != G_VO) { return false; }
        i++;
        text_opt();
        if (i >= toks.size() || toks[i].tag != G_VC) { return false; }
        i++;
        ws_only();
    }
    return true;
}

// the key/value pairs a malformed call still carries (the tag soup read leniently; see the header for the rules)
std::vector<std::pair<std::string, std::string>> glm_pairs(const std::vector<glm_tok> & toks, const std::string & pre_key) {
    std::vector<std::pair<std::string, std::string>> out;
    bool        in_val = false;
    std::string buf    = pre_key;
    std::string key;
    auto finish = [&]() {
        if (!trim(key).empty()) {
            out.emplace_back(trim(key), trim(buf));
        }
        key.clear();
        buf.clear();
        in_val = false;
    };
    for (const auto & t : toks) {
        switch (t.tag) {
            case G_TEXT:
                buf += t.text;
                break;
            case G_KO:
                if (in_val) { finish(); } else { buf.clear(); }
                break;
            case G_KC:
                if (!in_val) { key = buf; buf.clear(); in_val = true; }
                break;
            case G_VO:
                if (!in_val) {
                    if (!blank(buf)) { key = buf; }   // the key was never closed
                    buf.clear();
                    in_val = true;
                } else if (blank(buf)) {
                    buf.clear();
                }
                break;
            case G_VC:
                if (in_val) { finish(); }
                break;
        }
    }
    if (in_val) {
        finish();
    }
    return out;
}

std::string glm_render(const std::string & name, const std::vector<std::pair<std::string, std::string>> & kv, bool newline) {
    const char * nl = newline ? "\n" : "";
    std::string  r  = std::string(GLM_TC_OPEN) + name + nl;
    for (const auto & p : kv) {
        r += "<arg_key>" + p.first + "</arg_key>" + nl + "<arg_value>" + p.second + "</arg_value>" + nl;
    }
    return r + GLM_TC_CLOSE;
}

struct glm_fix {
    size_t                                           begin = 0;
    size_t                                           end   = 0;   // one past the replaced span
    std::string                                      name;
    std::vector<std::pair<std::string, std::string>> kv;
    bool                                             newline = true;
};

std::vector<glm_fix> glm_find(const std::string & text, const std::vector<common_chat_tool> & tools) {
    std::vector<glm_fix> fixes;
    // every region first, so that "followed by prose" can look at what comes after the last one
    struct reg { size_t begin, end; bool irregular; glm_fix fix; };
    std::vector<reg> regs;
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(GLM_TC_OPEN, pos);
        if (b == std::string::npos) { break; }
        size_t i = b + std::strlen(GLM_TC_OPEN);
        while (i < text.size() && is_ws(text[i])) { i++; }
        size_t j = i;
        while (j < text.size() && !is_ws(text[j]) && text[j] != '<') { j++; }
        const std::string name = text.substr(i, j - i);
        pos = b + 1;
        if (name.empty() || !find_tool(tools, name) || in_fence(text, b)) { continue; }
        const size_t e1 = text.find(GLM_TC_CLOSE, j);
        const size_t e2 = text.find(GLM_TC_OPEN, j);
        const bool   terminated = e1 != std::string::npos && (e2 == std::string::npos || e1 < e2);
        const size_t body_end   = terminated ? e1 : (e2 != std::string::npos ? e2 : text.size());
        const size_t reg_end    = terminated ? e1 + std::strlen(GLM_TC_CLOSE) : body_end;
        // a name followed by blanks and then a key text ("get_weather city</arg_key>"): the key text starts the body
        std::string pre_key;
        size_t body_start = j;
        {
            size_t k = j;
            while (k < body_end && is_ws(text[k])) { k++; }
            if (k < body_end && text[k] != '<') {
                size_t m = k;
                while (m < body_end && text[m] != '<') { m++; }
                pre_key    = text.substr(k, m - k);
                body_start = m;
            }
        }
        const std::string body = text.substr(body_start, body_end - body_start);
        const auto        toks = glm_tokenize(body);
        reg r;
        r.begin     = b;
        r.end       = reg_end;
        r.irregular = !terminated || !pre_key.empty() || !glm_regular(toks);
        r.fix.begin = b;
        r.fix.end   = reg_end;
        r.fix.name  = name;
        r.fix.kv    = glm_pairs(toks, pre_key);
        // dialect of the original: a newline between the name and the end of the body
        const size_t nl_at = text.find('\n', j);
        r.fix.newline = nl_at != std::string::npos && nl_at < body_end;
        regs.push_back(std::move(r));
        pos = reg_end;
    }
    for (size_t k = 0; k < regs.size(); k++) {
        if (!regs[k].irregular) { continue; }
        if (k + 1 == regs.size() && !blank(text.substr(regs[k].end))) { continue; }   // followed by prose: a quote
        fixes.push_back(regs[k].fix);
    }
    return fixes;
}

// ---- Gemma 4: <|tool_call>call:NAME{key:<|"|>value<|"|>,key:5}<tool_call|> ------------------------------------------------

const char * const GEM_OPEN  = "<|tool_call>call:";
const char * const GEM_CLOSE = "<tool_call|>";
const char * const GEM_RESP  = "<|tool_response>";
const char * const GEM_NEXT  = "<|tool_call>";
const char * const GEM_Q     = "<|\"|>";

bool ident_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
}

// at `p` (the character after a value): optional dangling <|"|>, blanks, then "," + declared-parameter + ":" or the end
bool gem_boundary(const std::string & b, size_t p, const std::set<std::string> & params) {
    if (at(b, p, GEM_Q)) { p += std::strlen(GEM_Q); }
    while (p < b.size() && is_ws(b[p])) { p++; }
    if (p >= b.size()) { return true; }
    if (b[p] != ',') { return false; }
    p++;
    while (p < b.size() && is_ws(b[p])) { p++; }
    size_t q = p;
    while (q < b.size() && ident_char(b[q])) { q++; }
    if (q == p || q >= b.size() || b[q] != ':') { return false; }
    return params.empty() || params.count(b.substr(p, q - p)) > 0;
}

bool gem_literal(const std::string & v) {
    if (v == "true" || v == "false" || v == "null") { return true; }
    if (v.empty()) { return false; }
    size_t i = (v[0] == '-') ? 1 : 0;
    bool   digit = false, dot = false;
    for (; i < v.size(); i++) {
        if (v[i] >= '0' && v[i] <= '9') { digit = true; }
        else if (v[i] == '.' && !dot)  { dot = true; }
        else { return false; }
    }
    return digit;
}

// the properties of a body "key:value,key:value" read leniently; false = leave the region as written (nested value, no key)
bool gem_props(const std::string & body, const common_chat_tool & tool, std::vector<std::pair<std::string, std::string>> & out) {
    const json props = tool_properties(tool);
    std::set<std::string> names;
    for (const auto & entry : props.items()) { names.insert(entry.key()); }
    size_t i = 0;
    const size_t n = body.size();
    while (true) {
        while (i < n && (is_ws(body[i]) || body[i] == ',')) { i++; }
        if (i >= n) { break; }
        size_t k = i;
        while (k < n && ident_char(body[k])) { k++; }
        if (k == i || k >= n || body[k] != ':') { return false; }
        const std::string key = body.substr(i, k - i);
        i = k + 1;
        size_t v = i;
        while (v < n && (body[v] == ' ' || body[v] == '\t')) { v++; }
        std::string val;
        bool        bare = false;
        if (at(body, v, GEM_Q)) {
            const size_t s = v + std::strlen(GEM_Q);
            const size_t c = body.find(GEM_Q, s);
            if (c == std::string::npos) { val = body.substr(s); i = n; }
            else                        { val = body.substr(s, c - s); i = c + std::strlen(GEM_Q); }
        } else if (v < n && body[v] == '\'') {
            const size_t s = v + 1;
            size_t       c = s;
            bool         found = false;
            for (; c < n; c++) {
                if (body[c] == '\'' && gem_boundary(body, c + 1, names)) { found = true; break; }
            }
            if (found) { val = body.substr(s, c - s); i = c + 1; if (at(body, i, GEM_Q)) { i += std::strlen(GEM_Q); } }
            else       { val = body.substr(s); i = n; }
        } else if (v < n && (body[v] == '{' || body[v] == '[')) {
            return false;
        } else {
            size_t c = i;
            for (; c < n; c++) {
                if (body[c] == ',' && gem_boundary(body, c, names)) { break; }
            }
            val = body.substr(i, c - i);
            i   = c;
            const std::string t = trim(val);
            std::string type;
            if (props.contains(key) && props.at(key).is_object() && props.at(key).contains("type") && props.at(key).at("type").is_string()) {
                type = props.at(key).at("type").get<std::string>();
            }
            if (type != "string" && gem_literal(t)) { val = t; bare = true; }
        }
        out.emplace_back(key, bare ? val : std::string(GEM_Q) + val + GEM_Q);
    }
    return !out.empty() || trim(body).empty();
}

struct gem_fix {
    size_t      begin = 0;
    size_t      end   = 0;   // one past the replaced span
    std::string text;
};

std::vector<gem_fix> gem_find(const std::string & text, const std::vector<common_chat_tool> & tools) {
    std::vector<gem_fix> fixes;
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(GEM_OPEN, pos);
        if (b == std::string::npos) { break; }
        pos = b + 1;
        const size_t i = b + std::strlen(GEM_OPEN);
        size_t       j = i;
        while (j < text.size() && text[j] != '{' && !is_ws(text[j]) && text[j] != '<') { j++; }
        const std::string name = text.substr(i, j - i);
        const common_chat_tool * tool = find_tool(tools, name);
        if (!tool || j >= text.size() || text[j] != '{' || in_fence(text, b)) { continue; }
        const size_t body_start = j + 1;
        size_t term     = text.size();
        size_t term_len = 0;
        const char * const lits[] = { GEM_CLOSE, GEM_RESP, GEM_NEXT };
        // The call ends at the first terminator that is NOT inside a <|"|>...<|"|> string: a value may hold the literal
        // "<tool_call|>" (the 2026-11-15 end-literal-in-args test; the first find() cut the argument short).
        for (size_t s = body_start; s < text.size();) {
            if (at(text, s, GEM_Q)) {
                const size_t c = text.find(GEM_Q, s + std::strlen(GEM_Q));
                if (c == std::string::npos) { break; }   // unterminated string: fall back to the first terminator below
                s = c + std::strlen(GEM_Q);
                continue;
            }
            bool hit = false;
            for (const char * lit : lits) {
                if (at(text, s, lit)) { term = s; term_len = (lit == GEM_CLOSE) ? std::strlen(GEM_CLOSE) : 0; hit = true; break; }
            }
            if (hit) { break; }
            s++;
        }
        if (term == text.size()) {
            for (const char * lit : lits) {
                const size_t p = text.find(lit, body_start);
                if (p != std::string::npos && p < term) { term = p; term_len = (lit == GEM_CLOSE) ? std::strlen(GEM_CLOSE) : 0; }
            }
        }
        std::string body = text.substr(body_start, term - body_start);
        while (!body.empty() && is_ws(body.back())) { body.pop_back(); }
        if (!body.empty() && body.back() == '}') { body.pop_back(); }
        std::vector<std::pair<std::string, std::string>> kv;
        if (!gem_props(body, *tool, kv)) { continue; }
        std::string r = std::string(GEM_OPEN) + name + "{";
        for (size_t k = 0; k < kv.size(); k++) {
            r += (k ? "," : "") + kv[k].first + ":" + kv[k].second;
        }
        r += std::string("}") + GEM_CLOSE;
        fixes.push_back({ b, term + term_len, r });
        pos = term + term_len;
    }
    return fixes;
}

} // namespace

std::vector<std::string> common_chat_malformed_call_candidates(common_chat_format format, const std::string & text,
                                                               const std::vector<common_chat_tool> & tools) {
    std::vector<std::string> out;
    if (tools.empty() || text.empty()) {
        return out;
    }
    if (format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
        const auto fixes = gem_find(text, tools);
        if (!fixes.empty()) {
            std::string r;
            size_t      p = 0;
            for (const auto & f : fixes) { r += text.substr(p, f.begin - p) + f.text; p = f.end; }
            out.push_back(r + text.substr(p));
        }
        return out;
    }
    if (format != COMMON_CHAT_FORMAT_PEG_NATIVE) {
        return out;
    }
    const auto fixes = glm_find(text, tools);
    if (!fixes.empty()) {
        const bool nl_first = fixes.front().newline;
        for (const bool nl : { nl_first, !nl_first }) {
            std::string r;
            size_t      p = 0;
            for (const auto & f : fixes) { r += text.substr(p, f.begin - p) + glm_render(f.name, f.kv, nl); p = f.end; }
            out.push_back(r + text.substr(p));
        }
    }
    // LFM2: the whole reply is one bare "[name(args)]" with no <|tool_call_start|> wrapper
    const std::string t = trim(text);
    if (t.size() > 4 && t.front() == '[' && t.back() == ']' && t.find("<|tool_call_start|>") == std::string::npos) {
        size_t j = 1;
        while (j < t.size() && ident_char(t[j])) { j++; }
        if (j < t.size() && t[j] == '(' && find_tool(tools, t.substr(1, j - 1)) && !in_fence(text, text.find(t))) {
            out.push_back("<|tool_call_start|>" + t + "<|tool_call_end|>");
        }
    }
    return out;
}

// A parsed call whose arguments are not a JSON value, or still hold call markup.
bool common_chat_calls_look_corrupt(const common_chat_msg & msg) {
    static const char * const MARKS[] = { "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "</tool_call>",
                                           "<tool_call>", "<tool_call|>", "<|tool_call>", "<|\"|>", "<|tool_response>" };
    for (const auto & tc : msg.tool_calls) {
        if (!tc.arguments.empty() && json::parse_no_throw(tc.arguments).is_discarded()) {
            return true;
        }
        for (const char * m : MARKS) {
            if (tc.arguments.find(m) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}
