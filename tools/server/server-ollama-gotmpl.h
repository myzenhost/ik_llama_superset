#pragma once
// Go text/template (the dialect used by Ollama Modelfile TEMPLATE layers) -> Jinja translator, for the subset real Ollama
// templates use. Header-only, no dependency beyond the standard library, so tests/test-ollama-gotmpl.cpp can drive it.
//
// SOURCE-READ (ollama/ollama @ eab97e9f92b9a25c2d52d2cc6c1b1c99bd9fae21, MIT; behaviour re-implemented in C++, no Go code copied):
//   template/template.go:120-143   function map: json, currentDate, yesterdayDate, toTypeScriptType (only json/currentDate/yesterdayDate map here)
//   template/template.go:145-168   Parse: a template naming neither .Messages nor .Response gets "{{ .Response }}" appended
//   template/template.go:257-350   Execute: .Messages flavour (System = system messages, Messages = collated, Response = "") vs the
//                                  legacy System/Prompt/Response flavour (one execution per user/assistant turn, then the LAST turn is
//                                  rendered from the template cut after the first node that names .Response)
//   template/template.go:352-390   collate: consecutive same-role messages are merged with "\n\n" (tool messages stay separate)
//   template/*.gotmpl (20 files)  the constructs actually used: range/if/else if/with, $x := / $x =, eq/ne/and/or/not/len/slice/printf,
//                                  continue, $.Field, field chains .Function.Parameters.Properties
//   Go text/template grammar: {{- / -}} trim markers (strip ALL adjacent whitespace, same as Jinja's {%- / -%}), "" and `` strings,
//                                  pipelines, nested field chains, variables declared with := and assigned with = (function-wide)
//
// Anything outside the subset yields ok=false with a message naming the construct: the caller keeps NO template rather than a wrong one.
#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ollama_gotmpl {

struct expr;
using expr_p = std::shared_ptr<expr>;

struct expr {
    enum kind_t { FIELD, VAR, STR, NUM, BOOL, NIL, CALL, DOT } kind = DOT;
    std::string text;               // string / number / bool literal, variable name, function name
    std::vector<std::string> chain; // FIELD / VAR / DOT: .A.B (after the base)
    bool root = false;              // FIELD written as $.Name
    std::vector<expr_p> args;       // CALL
    expr_p base;                    // (pipeline).Field.Chain, or the receiver of a chain on a CALL
};

enum node_kind { N_TEXT, N_OUT, N_IF, N_RANGE, N_WITH, N_SET, N_CONTINUE, N_BREAK };

struct node;
using node_p = std::shared_ptr<node>;
struct node {
    node_kind kind = N_TEXT;
    std::string text;
    bool ltrim = false, rtrim = false;
    expr_p e;                               // output / condition / range target / set value
    std::string var1, var2;                 // range vars ($idx, $val) or set target
    bool declare = false;                   // := vs =
    std::vector<node_p> body, other;        // then / else
    bool ltrim_end = false, rtrim_end = false; // trim markers of the closing {{ end }} / {{ else }}
    bool ltrim_else = false, rtrim_else = false;
};

struct parse_error { std::string msg; };

// ------------------------------------------------------------------ lexing of one action body
struct tok { enum t { IDENT, FIELD, VAR, STR, NUM, LP, RP, PIPE, DECL, ASSIGN, COMMA, END } k; std::string s; bool adj = false; };

inline std::vector<tok> lex_action(const std::string & s) {
    std::vector<tok> out;
    size_t i = 0;
    auto is_id = [](char c) { return std::isalnum((unsigned char) c) || c == '_'; };
    while (i < s.size()) {
        char c = s[i];
        if (std::isspace((unsigned char) c)) { i++; continue; }
        if (c == '(') { out.push_back({tok::LP, "("}); i++; continue; }
        if (c == ')') { out.push_back({tok::RP, ")"}); i++; continue; }
        if (c == '|') { out.push_back({tok::PIPE, "|"}); i++; continue; }
        if (c == ',') { out.push_back({tok::COMMA, ","}); i++; continue; }
        if (c == ':' && i + 1 < s.size() && s[i + 1] == '=') { out.push_back({tok::DECL, ":="}); i += 2; continue; }
        if (c == '=') { out.push_back({tok::ASSIGN, "="}); i++; continue; }
        if (c == '"') {
            std::string v; i++;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    char n = s[i + 1];
                    v += n == 'n' ? '\n' : n == 't' ? '\t' : n == 'r' ? '\r' : n; // \" \\ and the rest map to themselves
                    i += 2;
                } else { v += s[i++]; }
            }
            if (i >= s.size()) throw parse_error{"unterminated string"};
            i++;
            out.push_back({tok::STR, v});
            continue;
        }
        if (c == '`') {
            size_t e = s.find('`', i + 1);
            if (e == std::string::npos) throw parse_error{"unterminated raw string"};
            out.push_back({tok::STR, s.substr(i + 1, e - i - 1)});
            i = e + 1;
            continue;
        }
        if (c == '$') {
            size_t j = i + 1;
            while (j < s.size() && is_id(s[j])) j++;
            out.push_back({tok::VAR, s.substr(i, j - i)}); // "$" alone is the root
            i = j;
            continue;
        }
        if (c == '.') {
            size_t j = i + 1;
            while (j < s.size() && is_id(s[j])) j++;
            // adj: ".B" glued to the previous token (".A.B", "$x.B") continues its chain; ".B" after a space is a separate operand
            out.push_back({tok::FIELD, s.substr(i, j - i), i > 0 && (is_id(s[i - 1]) || s[i - 1] == '$')});
            i = j;
            continue;
        }
        if (std::isdigit((unsigned char) c) || (c == '-' && i + 1 < s.size() && std::isdigit((unsigned char) s[i + 1]))) {
            size_t j = i + 1;
            while (j < s.size() && (std::isdigit((unsigned char) s[j]) || s[j] == '.')) j++;
            out.push_back({tok::NUM, s.substr(i, j - i)});
            i = j;
            continue;
        }
        if (std::isalpha((unsigned char) c) || c == '_') {
            size_t j = i;
            while (j < s.size() && is_id(s[j])) j++;
            out.push_back({tok::IDENT, s.substr(i, j - i)});
            i = j;
            continue;
        }
        throw parse_error{std::string("unsupported character '") + c + "' in action"};
    }
    out.push_back({tok::END, ""});
    return out;
}

// ------------------------------------------------------------------ expression parser
struct eparser {
    std::vector<tok> t;
    size_t p = 0;
    const tok & cur() const { return t[p]; }

    expr_p operand() {
        const tok & c = cur();
        expr_p e = std::make_shared<expr>();
        if (c.k == tok::LP) {
            p++;
            expr_p inner = pipeline();
            if (cur().k != tok::RP) throw parse_error{"expected )"};
            p++;
            e = inner;
        } else if (c.k == tok::STR)  { e->kind = expr::STR; e->text = c.s; p++; }
        else if (c.k == tok::NUM)    { e->kind = expr::NUM; e->text = c.s; p++; }
        else if (c.k == tok::FIELD)  {
            p++;
            if (c.s == ".") { e->kind = expr::DOT; }
            else { e->kind = expr::FIELD; e->chain.push_back(c.s.substr(1)); }
            // ".A.B" lexes as FIELD(".A") FIELD(".B") back to back: the chain continues
            while (cur().k == tok::FIELD && cur().adj && cur().s != ".") { e->chain.push_back(cur().s.substr(1)); p++; }
        }
        else if (c.k == tok::VAR) {
            p++;
            if (c.s == "$") { e->kind = expr::DOT; e->root = true; } else { e->kind = expr::VAR; e->text = c.s; }
            while (cur().k == tok::FIELD && cur().adj && cur().s != ".") { e->chain.push_back(cur().s.substr(1)); p++; }
            if (c.s == "$" && !e->chain.empty()) { e->kind = expr::FIELD; e->root = true; }
        }
        else if (c.k == tok::IDENT) {
            if (c.s == "true" || c.s == "false") { e->kind = expr::BOOL; e->text = c.s; p++; }
            else if (c.s == "nil")               { e->kind = expr::NIL; p++; }
            else                                  { e->kind = expr::CALL; e->text = c.s; p++; } // function name; args added by command()
        }
        else throw parse_error{"unexpected token '" + c.s + "'"};
        return e;
    }

    expr_p command() {
        expr_p head = operand();
        if (head->kind == expr::CALL && head->args.empty()) {
            // identifier: a function call consumes the remaining operands of the command
            while (cur().k != tok::END && cur().k != tok::PIPE && cur().k != tok::RP && cur().k != tok::COMMA) {
                head->args.push_back(operand());
            }
        }
        return head;
    }

    expr_p pipeline() {
        expr_p e = command();
        while (cur().k == tok::PIPE) {
            p++;
            expr_p f = command();
            if (f->kind != expr::CALL) throw parse_error{"pipeline target is not a function"};
            f->args.push_back(e);
            e = f;
        }
        return e;
    }
};

inline expr_p parse_pipeline(const std::vector<tok> & toks, size_t from, size_t to_excl_end = std::string::npos) {
    eparser ep;
    ep.t.assign(toks.begin() + from, toks.end());
    (void) to_excl_end;
    expr_p e = ep.pipeline();
    if (ep.cur().k != tok::END) throw parse_error{"trailing tokens in pipeline"};
    return e;
}

// ------------------------------------------------------------------ statement parser
struct raw_item { bool is_text; std::string s; bool ltrim = false, rtrim = false; };

// split into text and action items; "{{- " / " -}}" are trim markers only when the dash is adjacent to the delimiter AND separated by space
inline std::vector<raw_item> split_items(const std::string & src) {
    std::vector<raw_item> out;
    size_t i = 0;
    while (i < src.size()) {
        size_t a = src.find("{{", i);
        if (a == std::string::npos) { out.push_back({true, src.substr(i)}); break; }
        if (a > i) out.push_back({true, src.substr(i, a - i)});
        size_t j = a + 2;
        raw_item it{false, "", false, false};
        if (j + 1 < src.size() && src[j] == '-' && std::isspace((unsigned char) src[j + 1])) { it.ltrim = true; j++; }
        size_t body_start = j;
        // find the closing }} outside strings / comments
        size_t k = j;
        bool closed = false;
        while (k < src.size()) {
            if (src.compare(k, 2, "/*") == 0) {
                size_t e = src.find("*/", k + 2);
                if (e == std::string::npos) throw parse_error{"unterminated comment"};
                k = e + 2; continue;
            }
            if (src[k] == '"') {
                k++;
                while (k < src.size() && src[k] != '"') k += src[k] == '\\' ? 2 : 1;
                k++; continue;
            }
            if (src[k] == '`') { size_t e = src.find('`', k + 1); if (e == std::string::npos) throw parse_error{"unterminated raw string"}; k = e + 1; continue; }
            if (src.compare(k, 2, "}}") == 0) { closed = true; break; }
            k++;
        }
        if (!closed) throw parse_error{"unclosed action"};
        std::string body = src.substr(body_start, k - body_start);
        if (body.size() >= 2 && body.back() == '-' && std::isspace((unsigned char) body[body.size() - 2])) { it.rtrim = true; body.pop_back(); }
        it.s = body;
        out.push_back(it);
        i = k + 2;
    }
    return out;
}

inline std::string trim_ws(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) b++;
    while (e > b && std::isspace((unsigned char) s[e - 1])) e--;
    return s.substr(b, e - b);
}

struct stmt_parser {
    std::vector<raw_item> items;
    size_t pos = 0;
    bool uses_messages = false, uses_response = false;
    std::set<std::string> used_fields;

    // returns the terminator action ("end" / "else" / "else if ..." text) or "" at EOF
    struct term { std::string kw; std::string rest; bool ltrim = false, rtrim = false; bool eof = true; };

    static bool starts_kw(const std::string & b, const char * kw, std::string & rest) {
        size_t n = std::string(kw).size();
        if (b.compare(0, n, kw) != 0) return false;
        if (b.size() > n && !std::isspace((unsigned char) b[n]) && b[n] != '(') return false;
        rest = trim_ws(b.substr(n));
        return true;
    }

    void note(const expr_p & e) {
        if (!e) return;
        if (e->kind == expr::FIELD && !e->chain.empty()) {
            std::string f = e->chain[0];
            for (auto & c : f) c = (char) std::tolower((unsigned char) c);
            used_fields.insert(f);
            if (f == "messages") uses_messages = true;
            if (f == "response") uses_response = true;
        }
        for (auto & a : e->args) note(a);
        note(e->base);
    }

    std::vector<node_p> parse_list(term & t) {
        std::vector<node_p> list;
        t = term{};
        while (pos < items.size()) {
            raw_item & it = items[pos++];
            if (it.is_text) {
                auto n = std::make_shared<node>(); n->kind = N_TEXT; n->text = it.s; list.push_back(n); continue;
            }
            std::string b = trim_ws(it.s), rest;
            if (b.compare(0, 2, "/*") == 0 && b.size() >= 4 && b.compare(b.size() - 2, 2, "*/") == 0) {
                auto n = std::make_shared<node>(); n->kind = N_TEXT; n->ltrim = it.ltrim; n->rtrim = it.rtrim; list.push_back(n); continue; // comment: only its trim markers act
            }
            if (b == "end") { t = term{"end", "", it.ltrim, it.rtrim, false}; return list; }
            if (starts_kw(b, "else", rest)) { t = term{"else", rest, it.ltrim, it.rtrim, false}; return list; }
            auto n = std::make_shared<node>();
            n->ltrim = it.ltrim; n->rtrim = it.rtrim;
            if (starts_kw(b, "if", rest) || starts_kw(b, "with", rest)) {
                n->kind = b.compare(0, 2, "if") == 0 ? N_IF : N_WITH;
                n->e = parse_pipeline(lex_action(rest), 0); note(n->e);
                parse_branches(*n);
            } else if (starts_kw(b, "range", rest)) {
                n->kind = N_RANGE;
                auto toks = lex_action(rest);
                size_t from = 0;
                for (size_t k = 0; k < toks.size(); k++) {
                    if (toks[k].k == tok::DECL) {
                        // range $a := X  |  range $a, $b := X
                        if (k == 1 && toks[0].k == tok::VAR) { n->var2 = toks[0].s; }
                        else if (k == 3 && toks[0].k == tok::VAR && toks[1].k == tok::COMMA && toks[2].k == tok::VAR) { n->var1 = toks[0].s; n->var2 = toks[2].s; }
                        else throw parse_error{"unsupported range variable list"};
                        from = k + 1; break;
                    }
                }
                n->e = parse_pipeline(toks, from); note(n->e);
                parse_branches(*n);
            } else if (starts_kw(b, "continue", rest)) { n->kind = N_CONTINUE; }
            else if (starts_kw(b, "break", rest))      { n->kind = N_BREAK; }
            else if (starts_kw(b, "define", rest) || starts_kw(b, "template", rest) || starts_kw(b, "block", rest)) {
                throw parse_error{"{{ define/template/block }} is not supported"};
            } else {
                auto toks = lex_action(b);
                if (toks.size() > 2 && toks[0].k == tok::VAR && (toks[1].k == tok::DECL || toks[1].k == tok::ASSIGN)) {
                    n->kind = N_SET; n->var1 = toks[0].s; n->declare = toks[1].k == tok::DECL;
                    n->e = parse_pipeline(toks, 2); note(n->e);
                } else {
                    n->kind = N_OUT; n->e = parse_pipeline(toks, 0); note(n->e);
                }
            }
            list.push_back(n);
        }
        return list;
    }

    // after an if/with/range header: body, optional else / else if chain, end
    void parse_branches(node & n) {
        term t;
        n.body = parse_list(t);
        if (t.eof) throw parse_error{"missing {{ end }}"};
        if (t.kw == "else") {
            n.ltrim_else = t.ltrim; n.rtrim_else = t.rtrim;
            if (!t.rest.empty()) {
                std::string rest;
                if (!starts_kw(t.rest, "if", rest) || n.kind != N_IF) throw parse_error{"unsupported else clause"};
                // "else if X" == else { if X ... } sharing the one {{ end }}
                auto inner = std::make_shared<node>();
                inner->kind = N_IF; inner->ltrim = t.ltrim; inner->rtrim = t.rtrim;
                inner->e = parse_pipeline(lex_action(rest), 0); note(inner->e);
                parse_branches(*inner);
                inner->ltrim = n.ltrim_else; inner->rtrim = n.rtrim_else;
                n.other.push_back(inner);
                n.ltrim_end = inner->ltrim_end; n.rtrim_end = inner->rtrim_end;
                return;
            }
            term t2;
            n.other = parse_list(t2);
            if (t2.eof || t2.kw != "end") throw parse_error{"missing {{ end }}"};
            n.ltrim_end = t2.ltrim; n.rtrim_end = t2.rtrim;
            return;
        }
        n.ltrim_end = t.ltrim; n.rtrim_end = t.rtrim;
    }
};

// ------------------------------------------------------------------ Jinja emitter
inline std::string snake(const std::string & f) {
    std::string o;
    for (size_t i = 0; i < f.size(); i++) {
        if (std::isupper((unsigned char) f[i]) && i > 0 && !std::isupper((unsigned char) f[i - 1])) o += '_';
        o += (char) std::tolower((unsigned char) f[i]);
    }
    return o;
}

inline std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += c; } else if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t"; else if (c == '\r') o += "\\r"; else o += c; }
    return o + "\"";
}

struct emitter {
    bool messages_flavour = true;
    std::string dot = "";         // Jinja expression for "."; "" = root
    int counter = 0;
    std::string cut_marker;       // legacy last-turn render: set once the first .Response node was emitted
    bool cut = false;
    bool truncate_mode = false;

    std::string root_field(const std::string & f) {
        if (messages_flavour) {
            if (f == "Messages") return "_c.msgs";
            if (f == "System")   return "system";
            if (f == "Tools")    return "tools";
            if (f == "Response") return "\"\"";
            if (f == "Think")    return "(enable_thinking | default(false))";
            if (f == "IsThinkSet") return "(enable_thinking is defined)";
            if (f == "Prompt" || f == "Suffix") throw parse_error{"." + f + " in a .Messages template"};
            throw parse_error{"unknown top-level field ." + f};
        }
        if (f == "System" || f == "Prompt" || f == "Response") return "T" + f;
        if (f == "Think")    return "(enable_thinking | default(false))";
        if (f == "IsThinkSet") return "(enable_thinking is defined)";
        throw parse_error{"unknown top-level field ." + f + " (legacy template)"};
    }

    std::string field_chain(const std::string & base, const std::vector<std::string> & chain, size_t from) {
        std::string o = base;
        for (size_t i = from; i < chain.size(); i++) o += "." + snake(chain[i]);
        return o;
    }

    std::string fmt_printf(const std::vector<expr_p> & a) {
        if (a.empty() || a[0]->kind != expr::STR) throw parse_error{"printf needs a literal format string"};
        const std::string & f = a[0]->text;
        std::string out; size_t arg = 1, i = 0, lit_start = 0;
        auto flush = [&](size_t end) { if (end > lit_start) { out += (out.empty() ? "" : " ~ ") + jstr(f.substr(lit_start, end - lit_start)); } };
        while (i < f.size()) {
            if (f[i] != '%') { i++; continue; }
            if (i + 1 >= f.size()) throw parse_error{"dangling % in printf"};
            if (f[i + 1] == '%') { flush(i); out += (out.empty() ? "" : " ~ ") + jstr("%"); i += 2; lit_start = i; continue; }
            char v = f[i + 1];
            if (v != 's' && v != 'd' && v != 'v') throw parse_error{std::string("printf verb %") + v + " is not supported"};
            if (arg >= a.size()) throw parse_error{"printf: too few arguments"};
            flush(i);
            out += (out.empty() ? "" : " ~ ") + ex(a[arg++]);
            i += 2; lit_start = i;
        }
        flush(f.size());
        if (out.empty()) out = "\"\"";
        return "(" + out + ")";
    }

    std::string call(const expr & e) {
        const std::string & fn = e.text;
        auto A = [&](size_t i) { if (i >= e.args.size()) throw parse_error{fn + ": too few arguments"}; return ex(e.args[i]); };
        auto cmp = [&](const char * op) { if (e.args.size() != 2) throw parse_error{fn + " needs 2 arguments"}; return "(" + A(0) + " " + op + " " + A(1) + ")"; };
        if (fn == "eq") {
            if (e.args.size() < 2) throw parse_error{"eq needs 2+ arguments"};
            std::string o;
            for (size_t i = 1; i < e.args.size(); i++) o += (o.empty() ? "" : " or ") + std::string("(") + A(0) + " == " + A(i) + ")";
            return "(" + o + ")";
        }
        if (fn == "ne") return cmp("!=");
        if (fn == "lt") return cmp("<");
        if (fn == "le") return cmp("<=");
        if (fn == "gt") return cmp(">");
        if (fn == "ge") return cmp(">=");
        if (fn == "and" || fn == "or") {
            if (e.args.size() < 2) throw parse_error{fn + " needs 2+ arguments"};
            std::string o;
            for (size_t i = 0; i < e.args.size(); i++) o += (o.empty() ? "" : " " + fn + " ") + A(i);
            return "(" + o + ")";
        }
        if (fn == "not")   return "(not " + A(0) + ")";
        if (fn == "len")   return "(" + A(0) + " | length)";
        if (fn == "json")  return "(" + A(0) + " | tojson)";
        if (fn == "index") { if (e.args.size() != 2) throw parse_error{"index supports one subscript"}; return A(0) + "[" + A(1) + "]"; }
        if (fn == "slice") {
            if (e.args.size() == 2) return A(0) + "[" + A(1) + ":]";
            if (e.args.size() == 3) return A(0) + "[" + A(1) + ":" + A(2) + "]";
            throw parse_error{"slice supports 1 or 2 indices"};
        }
        if (fn == "printf") return fmt_printf(e.args);
        if (fn == "currentDate")   return "strftime_now(\"%Y-%m-%d\")";
        throw parse_error{"function '" + fn + "' is not supported"};
    }

    std::string ex(const expr_p & p) {
        const expr & e = *p;
        switch (e.kind) {
            case expr::STR:  return jstr(e.text);
            case expr::NUM:  return e.text;
            case expr::BOOL: return e.text == "true" ? "true" : "false";
            case expr::NIL:  return "none";
            case expr::CALL: return call(e);
            case expr::VAR:  return field_chain("_v." + e.text.substr(1), e.chain, 0);
            case expr::DOT: {
                if (e.root) { throw parse_error{"bare $ is not supported"}; }
                if (dot.empty()) throw parse_error{"bare . at top level"};
                return dot;
            }
            case expr::FIELD: {
                const bool at_root = e.root || dot.empty();
                if (at_root) return field_chain(root_field(e.chain[0]), e.chain, 1);
                return field_chain(dot, e.chain, 0);
            }
        }
        throw parse_error{"bad expression"};
    }

    static bool has_response(const expr_p & e) {
        if (!e) return false;
        if (e->kind == expr::FIELD && !e->chain.empty() && e->chain[0] == "Response") return true;
        for (auto & a : e->args) if (has_response(a)) return true;
        return false;
    }

    // several statements as one Go action: only the first carries the left trim marker, only the last the right one
    static std::string tags(bool ltrim, bool rtrim, const std::vector<std::string> & stmts) {
        std::string o;
        for (size_t i = 0; i < stmts.size(); i++) {
            const bool first = i == 0, last = i + 1 == stmts.size();
            o += first ? (ltrim ? "{%- " : "{{ \"\" }}{% ") : "{% ";
            o += stmts[i];
            o += last ? (rtrim ? " -%}" : " %}{{ \"\" }}") : " %}";
        }
        return o;
    }

    // The fork's Jinja lexer defaults to trim_blocks + lstrip_blocks (HF chat-template behaviour): it drops the newline after every
    // block tag and the indentation before one. Go's templates have neither, so an UNTRIMMED tag is fenced with an empty
    // expression on the side where the lexer would eat whitespace; a trim-marked side already strips everything, as Go does.
    static std::string blk(bool lt, bool rt, const std::string & stmt) {
        return std::string(lt ? "{%- " : "{{ \"\" }}{% ") + stmt + (rt ? " -%}" : " %}{{ \"\" }}");
    }

    // text with literal Jinja delimiters defused
    static std::string text_out(const std::string & s) {
        std::string o;
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '{' && i + 1 < s.size() && (s[i + 1] == '{' || s[i + 1] == '%' || s[i + 1] == '#')) { o += "{{ \"{\" }}"; }
            else o += s[i];
        }
        return o;
    }

    // `pending_ltrim` / the node's own markers: Go trims whitespace of the ADJACENT text, which Jinja's {%- -%} does the same way.
    std::string list(const std::vector<node_p> & v) {
        std::string o;
        for (auto & n : v) {
            if (cut) break;
            o += one(*n);
        }
        return o;
    }

    std::string one(const node & n) {
        std::string o;
        switch (n.kind) {
            case N_TEXT:
                if (n.text.empty() && (n.ltrim || n.rtrim)) { return std::string("{#") + (n.ltrim ? "-" : "") + " c " + (n.rtrim ? "-" : "") + "#}"; }
                return text_out(n.text);
            case N_OUT: {
                std::string v = ex(n.e);
                if (truncate_mode && has_response(n.e)) cut = true;
                return std::string("{{") + (n.ltrim ? "- " : " ") + v + (n.rtrim ? " -}}" : " }}");
            }
            case N_SET: {
                std::string v = ex(n.e);
                if (truncate_mode && has_response(n.e)) cut = true;
                return blk(n.ltrim, n.rtrim, "set _v." + n.var1.substr(1) + " = " + v);
            }
            case N_CONTINUE: return blk(n.ltrim, n.rtrim, "continue");
            case N_BREAK:    return blk(n.ltrim, n.rtrim, "break");
            case N_IF: {
                std::string c = ex(n.e);
                const bool resp = truncate_mode && has_response(n.e);
                o += blk(n.ltrim, n.rtrim, "if " + c);
                if (resp) { cut = true; return o + "{% endif %}"; }
                o += list(n.body);
                const bool was_cut = cut;
                if (!n.other.empty()) {
                    // a chained "else if" is a single nested N_IF whose own markers are the else-markers
                    if (n.other.size() == 1 && n.other[0]->kind == N_IF && n.other[0]->ltrim == n.ltrim_else && n.other[0]->rtrim == n.rtrim_else && !was_cut) {
                        o += chain_elif(*n.other[0]);
                        return o;
                    }
                    if (!was_cut) {
                        o += blk(n.ltrim_else, n.rtrim_else, "else");
                        o += list(n.other);
                    }
                }
                return o + blk(n.ltrim_end, n.rtrim_end, "endif");
            }
            case N_WITH: {
                const std::string w = "_v.w" + std::to_string(counter++);
                std::string v = ex(n.e);
                o += tags(n.ltrim, n.rtrim, { "set " + w + " = " + v, "if " + w });
                std::string saved = dot; dot = w;
                o += list(n.body);
                const bool was_cut = cut;
                if (!n.other.empty() && !was_cut) { dot = saved; o += blk(n.ltrim_else, n.rtrim_else, "else") + list(n.other); }
                dot = saved;
                return o + blk(n.ltrim_end, n.rtrim_end, "endif");
            }
            case N_RANGE: {
                std::string src = ex(n.e);
                const std::string it = "r" + std::to_string(counter++);
                // two variables: a map yields (key, value), a slice (index, value). The target's last field decides.
                const bool two = !n.var1.empty();
                bool is_map = false;
                if (n.e->kind == expr::FIELD || n.e->kind == expr::VAR) {
                    const std::string last = n.e->chain.empty() ? "" : n.e->chain.back();
                    is_map = last == "Properties" || last == "Arguments";
                }
                std::vector<std::string> st;
                if (two && is_map)  st.push_back("for " + it + "_k, " + it + " in " + src + " | dictsort");
                else                st.push_back("for " + it + " in " + src);
                if (two) st.push_back("set _v." + n.var1.substr(1) + " = " + (is_map ? it + "_k" : "loop.index0"));
                if (!n.var2.empty()) st.push_back("set _v." + n.var2.substr(1) + " = " + it);
                o += tags(n.ltrim, n.rtrim, st);
                std::string saved = dot; dot = it;
                o += list(n.body);
                const bool was_cut = cut;
                dot = saved;
                if (!n.other.empty() && !was_cut) o += blk(n.ltrim_else, n.rtrim_else, "else") + list(n.other);
                return o + blk(n.ltrim_end, n.rtrim_end, "endfor");
            }
        }
        return o;
    }

    // "{% elif %}" continuation of an if whose else branch is exactly one if with the else-markers
    std::string chain_elif(const node & n) {
        std::string o = blk(n.ltrim, n.rtrim, "elif " + ex(n.e));
        o += list(n.body);
        if (!n.other.empty() && !cut) {
            if (n.other.size() == 1 && n.other[0]->kind == N_IF && n.other[0]->ltrim == n.ltrim_else && n.other[0]->rtrim == n.rtrim_else) {
                return o + chain_elif(*n.other[0]);
            }
            o += blk(n.ltrim_else, n.rtrim_else, "else") + list(n.other);
        }
        return o + blk(n.ltrim_end, n.rtrim_end, "endif");
    }
};

// ------------------------------------------------------------------ top level
// Jinja prelude for the .Messages flavour: template.go:352-390 collate (merge consecutive same-role non-tool messages with "\n\n";
// System = every system message joined with "\n\n").
inline const char * k_collate_prelude =
    "{%- set _v = namespace() -%}"
    "{%- set _c = namespace(sys=[], msgs=[]) -%}"
    "{%- for _m in messages -%}"
    "{%- if _m.role == \"system\" -%}{%- set _c.sys = _c.sys + [_m.content] -%}{%- endif -%}"
    "{%- if _c.msgs and _c.msgs[-1].role == _m.role and _m.role != \"tool\" -%}"
    "{%- set _c.msgs = _c.msgs[:-1] + [{\"role\": _m.role, \"content\": _c.msgs[-1].content ~ \"\\n\\n\" ~ _m.content}] -%}"
    "{%- else -%}{%- set _c.msgs = _c.msgs + [_m] -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- set system = _c.sys | join(\"\\n\\n\") -%}";

inline bool translate(const std::string & go_tmpl, std::string & jinja, std::string & err) {
    try {
        stmt_parser sp;
        sp.items = split_items(go_tmpl);
        stmt_parser::term t;
        std::vector<node_p> tree = sp.parse_list(t);
        if (!t.eof) throw parse_error{"unexpected {{ " + t.kw + " }}"};
        // template.go:159: neither .Messages nor .Response named -> "{{ .Response }}" is appended
        if (!sp.uses_messages && !sp.uses_response) {
            auto n = std::make_shared<node>(); n->kind = N_OUT;
            n->e = std::make_shared<expr>(); n->e->kind = expr::FIELD; n->e->chain.push_back("Response");
            tree.push_back(n);
            sp.uses_response = true;
        }
        if (sp.uses_messages) {
            emitter em; em.messages_flavour = true;
            jinja = std::string(k_collate_prelude) + em.list(tree) + "{{ \"\" }}"; // the lexer drops a final newline of the source; a trailing expression keeps it
            return true;
        }
        // legacy System/Prompt/Response flavour (template.go:284-346)
        emitter full; full.messages_flavour = false;
        std::string body = full.list(tree);
        emitter last; last.messages_flavour = false; last.truncate_mode = true;
        std::string tail = last.list(tree);
        std::string o = std::string(k_collate_prelude);
        // the macro header must not eat the body's leading whitespace, so it is not trimmed on the inside
        o += "{%- macro _turn(TSystem, TPrompt, TResponse) %}{{ \"\" }}";
        o += body + "{%- endmacro -%}";
        o += "{%- macro _last(TSystem, TPrompt, TResponse) %}{{ \"\" }}" + tail + "{%- endmacro -%}";
        o += "{%- set _s = namespace(system=\"\", prompt=\"\", response=\"\", out=\"\") -%}"
             "{%- for _m in _c.msgs -%}"
             "{%- if _m.role == \"system\" -%}"
               "{%- if _s.prompt or _s.response -%}{%- set _s.out = _s.out ~ _turn(_s.system, _s.prompt, _s.response) -%}{%- set _s.system = \"\" -%}{%- set _s.prompt = \"\" -%}{%- set _s.response = \"\" -%}{%- endif -%}"
               "{%- set _s.system = _m.content -%}"
             "{%- elif _m.role == \"user\" -%}"
               "{%- if _s.response -%}{%- set _s.out = _s.out ~ _turn(_s.system, _s.prompt, _s.response) -%}{%- set _s.system = \"\" -%}{%- set _s.prompt = \"\" -%}{%- set _s.response = \"\" -%}{%- endif -%}"
               "{%- set _s.prompt = _m.content -%}"
             "{%- elif _m.role == \"assistant\" -%}{%- set _s.response = _m.content -%}"
             "{%- endif -%}"
             "{%- endfor -%}"
             "{{- _s.out ~ _last(_s.system, _s.prompt, _s.response) -}}{{ \"\" }}";
        jinja = o;
        return true;
    } catch (const parse_error & e) {
        err = e.msg;
        return false;
    }
}

} // namespace ollama_gotmpl
