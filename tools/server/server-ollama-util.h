#pragma once
// Pure helpers for the Ollama-native API shim (server-ollama.cpp): model-name parsing, Modelfile parsing,
// Modelfile PARAMETER coercion, and the Options{} -> server sampling-field mapping.
// Header-only and dependency-light (nlohmann json only) so tests/test-ollama-util.cpp can drive it directly.
//
// SOURCE-READ (ollama/ollama @ e3cddc3e897d8414a60a46e23f5ef3a99be2eb81, MIT; designs re-implemented in C++, no Go code copied):
//   types/model/name.go:100-228,290-345   name grammar, defaults, validation, DisplayShortest
//   parser/parser.go:487-710              Modelfile state machine, quoting rules
//   parser/parser.go:58-188               command semantics (last-wins / append / validation)
//   api/types.go:574-605,1176-1205        Options struct + defaults
//   api/types.go:1264-1334                PARAMETER coercion (FormatParams)
//   server/routes.go:126-160              precedence: defaults < modelfile PARAMETER < request options
#include "json.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <string>
#include <vector>

namespace ollama_util {

// ------------------------------------------------------------------ model names
struct model_name {
    std::string scheme, host, ns, model, tag;
};

inline const char * k_default_host      = "registry.ollama.ai";
inline const char * k_default_namespace = "library";
inline const char * k_default_tag       = "latest";

// name.go ParseNameBare: ":" splits a tag only when it is right of the last "/" (so host:port survives),
// then "/" splits model, namespace, host(+scheme://). An empty promised part becomes "!MISSING!" (invalid).
inline model_name parse_name_bare(std::string s) {
    model_name n;
    auto cut_last = [](std::string & str, char sep, std::string & after) {
        size_t i = str.rfind(sep);
        if (i == std::string::npos) return false;
        after = str.substr(i + 1);
        str   = str.substr(0, i);
        if (str.empty())   str   = "!MISSING!";
        if (after.empty()) after = "!MISSING!";
        return true;
    };
    size_t lc = s.rfind(':'), ls = s.rfind('/');
    if (lc != std::string::npos && (ls == std::string::npos || lc > ls)) {
        cut_last(s, ':', n.tag);
    }
    if (!cut_last(s, '/', n.model)) { n.model = s; return n; }
    if (!cut_last(s, '/', n.ns))    { n.ns    = s; return n; }
    size_t p = s.find("://");
    if (p != std::string::npos) { n.scheme = s.substr(0, p); n.host = s.substr(p + 3); }
    else                        { n.host = s; }
    return n;
}

inline model_name parse_name(const std::string & s) {
    model_name n = parse_name_bare(s);
    if (n.host.empty())   n.host   = k_default_host;
    if (n.ns.empty())     n.ns     = k_default_namespace;
    if (n.tag.empty())    n.tag    = k_default_tag;
    if (n.scheme.empty()) n.scheme = "https";
    return n;
}

inline bool is_alnum_us(char c) { return std::isalnum((unsigned char) c) || c == '_'; }

// kind: 'h' host, 'n' namespace, 'm' model, 't' tag
inline bool valid_part(char kind, const std::string & s) {
    const size_t max = kind == 'h' ? 350 : 80;
    if (s.empty() || s.size() > max) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (i == 0) { if (!is_alnum_us(c)) return false; continue; }
        if (c == '_' || c == '-') continue;
        if (c == '.') { if (kind == 'n') return false; continue; }
        if (c == ':') { if (kind != 'h') return false; continue; }
        if (!is_alnum_us(c)) return false;
    }
    return true;
}

inline bool valid(const model_name & n) {
    return valid_part('h', n.host) && valid_part('n', n.ns) && valid_part('m', n.model) && valid_part('t', n.tag);
}

inline std::string lower(std::string s) { for (auto & c : s) c = (char) std::tolower((unsigned char) c); return s; }

// name.go DisplayShortest: drop the default host, and the default namespace when the host is the default.
inline std::string display_shortest(const model_name & n) {
    std::string out;
    if (lower(n.host) != k_default_host) out += n.host + "/" + n.ns + "/";
    else if (lower(n.ns) != k_default_namespace) out += n.ns + "/";
    return out + n.model + ":" + n.tag;
}

// Canonical comparison key: two spellings of one model ("m", "m:latest", "library/m",
// "registry.ollama.ai/library/m:latest") compare equal; case-insensitive like Name.EqualFold.
inline std::string canonical_key(const std::string & s) {
    model_name n = parse_name(s);
    return lower(n.host + "/" + n.ns + "/" + n.model + ":" + n.tag);
}

// ------------------------------------------------------------------ OLLAMA_* environment
// SOURCE-READ (ollama @ eab97e9f, MIT): envconfig/config.go:177-190 BoolWithDefault (an UNPARSEABLE value counts as true),
//   :216-222 OLLAMA_FLASH_ATTENTION / OLLAMA_KV_CACHE_TYPE, :230 OLLAMA_CONTEXT_LENGTH (0 = unset), :261-273 Uint (bad value -> default),
//   :277-279 OLLAMA_NUM_PARALLEL (default 1) / OLLAMA_MAX_LOADED_MODELS (0 = auto), :126-143 OLLAMA_KEEP_ALIVE (Go duration, else bare
//   integer seconds, negative = forever), llm/server.go:116 kv type lower-cased, llm/llama_server.go:397-399 the one type applies to K and V.
// envconfig/config.go:126-143: a Go duration ("1h2m3s", "5m0s", "-1m") or a bare integer count of seconds; negative = forever (-1 here).
// Returns false for anything else ("1d", "1w", "???", " "), where ollama falls back to its 5 minute default.
inline bool keep_alive_ms(const std::string & s, int64_t & out_ms) {
    if (s.empty()) return false;
    size_t i = 0;
    bool neg = false;
    if (s[0] == '-' || s[0] == '+') { neg = s[0] == '-'; i = 1; }
    if (i >= s.size()) return false;
    bool all_digits = true;
    for (size_t k = i; k < s.size(); ++k) all_digits = all_digits && std::isdigit((unsigned char) s[k]);
    double total_ms = 0;
    if (all_digits) {
        total_ms = std::strtod(s.c_str() + i, nullptr) * 1000.0; // time.ParseDuration("60") fails, then ParseInt(60) * Second
    } else if (s.compare(i, std::string::npos, "0") == 0) {
        total_ms = 0;
    } else {
        while (i < s.size()) {
            size_t j = i;
            while (j < s.size() && (std::isdigit((unsigned char) s[j]) || s[j] == '.')) ++j;
            if (j == i) return false;
            const double v = std::strtod(s.substr(i, j - i).c_str(), nullptr);
            size_t u = j;
            while (u < s.size() && !std::isdigit((unsigned char) s[u]) && s[u] != '.') ++u;
            const std::string unit = s.substr(j, u - j);
            double mul;
            if      (unit == "ns") mul = 1e-6;
            else if (unit == "us" || unit == "\xC2\xB5s" || unit == "\xCE\xBCs") mul = 1e-3;
            else if (unit == "ms") mul = 1;
            else if (unit == "s")  mul = 1000;
            else if (unit == "m")  mul = 60000;
            else if (unit == "h")  mul = 3600000;
            else return false;
            total_ms += v * mul;
            i = u;
        }
    }
    out_ms = neg && total_ms > 0 ? -1 : (int64_t) total_ms; // "-0" is 0; any negative duration means forever
    return true;
}

struct env_overrides {
    bool        has_kv = false;     std::string kv_type;
    bool        has_flash = false;  bool flash = true;
    bool        has_ctx = false;    uint64_t ctx = 0;
    bool        has_parallel = false; uint64_t parallel = 1;
    bool        has_max_models = false; uint64_t max_models = 0;
    bool        has_keep_alive = false; std::string keep_alive;   // raw; the caller parses it with the fork's duration parser
    std::vector<std::string> warnings;
};

template <typename Getter>
inline env_overrides read_env(Getter get) {
    env_overrides o;
    auto val = [&](const char * k) { const char * v = get(k); return std::string(v ? v : ""); };
    auto uintv = [&](const char * k, bool & has, uint64_t & out) {
        const std::string s = val(k);
        if (s.empty()) return;
        char * end = nullptr;
        errno = 0;
        unsigned long long n = std::strtoull(s.c_str(), &end, 10);
        if (errno || end == s.c_str() || *end || s[0] == '-') { o.warnings.push_back(std::string(k) + "=" + s + " is not an unsigned integer, ignored"); return; }
        has = true; out = n;
    };
    if (std::string t = val("OLLAMA_KV_CACHE_TYPE"); !t.empty()) { o.has_kv = true; o.kv_type = lower(t); }
    if (std::string f = val("OLLAMA_FLASH_ATTENTION"); !f.empty()) {
        o.has_flash = true;
        // strconv.ParseBool accepts 1,t,T,TRUE,true,True / 0,f,F,FALSE,false,False; anything else -> true
        if (f == "0" || f == "f" || f == "F" || f == "FALSE" || f == "false" || f == "False") o.flash = false; else o.flash = true;
    }
    uintv("OLLAMA_CONTEXT_LENGTH", o.has_ctx, o.ctx);
    if (o.has_ctx && o.ctx == 0) o.has_ctx = false;
    uintv("OLLAMA_NUM_PARALLEL", o.has_parallel, o.parallel);
    if (o.has_parallel && o.parallel == 0) { o.has_parallel = false; }
    uintv("OLLAMA_MAX_LOADED_MODELS", o.has_max_models, o.max_models);
    if (o.has_max_models && o.max_models == 0) o.has_max_models = false; // 0 = ollama's automatic choice
    if (std::string k = val("OLLAMA_KEEP_ALIVE"); !k.empty()) { o.has_keep_alive = true; o.keep_alive = k; }
    return o;
}

// ------------------------------------------------------------------ Modelfile
struct modelfile_cmd { std::string name, args; };

// parser.go ParseFile. Differences from the Go original, stated: bytes instead of runes (every byte >= 0x80 counts as
// printable; Go's strconv.IsPrint rejects a few exotic code points), TrimSpace is ASCII whitespace only.
inline bool modelfile_unquote(const std::string & s, std::string & out) {
    if (s.size() >= 3 && s.compare(0, 3, "\"\"\"") == 0) {
        if (s.size() >= 6 && s.compare(s.size() - 3, 3, "\"\"\"") == 0) { out = s.substr(3, s.size() - 6); return true; }
        return false;
    }
    if (!s.empty() && s[0] == '"') {
        if (s.size() >= 2 && s.back() == '"') { out = s.substr(1, s.size() - 2); return true; }
        return false;
    }
    out = s;
    return true;
}

inline std::string modelfile_trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) ++a;
    while (b > a && std::isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline bool modelfile_valid_command(const std::string & lc) {
    static const char * k[] = { "from", "license", "template", "system", "adapter", "draft", "renderer", "parser",
                                "parameter", "message", "requires", "capability" };
    for (auto * c : k) if (lc == c) return true;
    return false;
}

inline bool parse_modelfile(const std::string & text_in, std::vector<modelfile_cmd> & cmds, std::string & err) {
    enum st { NIL, NAME, VALUE, PARAM, MESSAGE, COMMENT };
    std::string text = text_in;
    if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3);
    auto is_nl    = [](char c) { return c == '\r' || c == '\n'; };
    auto is_sp    = [](char c) { return c == ' ' || c == '\t'; };
    auto is_alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    auto is_num   = [](char c) { return c >= '0' && c <= '9'; };
    auto is_print = [](char c) { unsigned char u = (unsigned char) c; return u >= 0x20 && u != 0x7f; };

    cmds.clear();
    err.clear();
    st curr = NIL;
    int line = 1;
    std::string b, role;
    modelfile_cmd cmd;
    auto perr = [&](const std::string & m) { err = "(line " + std::to_string(line) + "): " + m; return false; };
    static const std::string k_cmd_err = "command must be one of \"from\", \"license\", \"template\", \"system\", \"adapter\", "
        "\"draft\", \"renderer\", \"parser\", \"parameter\", \"message\", \"requires\", or \"capability\"";

    for (char ch : text) {
        if (is_nl(ch)) ++line;
        st next = curr;
        bool keep = true; // whether ch is delivered to the buffer (parseRuneForState returns r vs 0)
        switch (curr) {
            case NIL:
                if (ch == '#')                 { next = COMMENT; keep = false; }
                else if (is_sp(ch) || is_nl(ch)) { next = NIL;     keep = false; }
                else                             { next = NAME; }
                break;
            case NAME:
                if (is_alpha(ch))   next = NAME;
                else if (is_sp(ch)) { next = VALUE; keep = false; }
                else return perr(k_cmd_err);
                break;
            case VALUE:
                next = (is_nl(ch) || is_sp(ch)) ? NIL : VALUE;
                break;
            case PARAM:
                if (is_alpha(ch) || is_num(ch) || ch == '_') next = PARAM;
                else if (is_sp(ch)) { next = VALUE; keep = false; }
                else { err = "unexpected EOF: " + b; return false; }
                break;
            case MESSAGE:
                if (is_alpha(ch))   next = MESSAGE;
                else if (is_sp(ch)) { next = VALUE; keep = false; }
                else { err = "unexpected EOF: " + b; return false; }
                break;
            case COMMENT:
                next = is_nl(ch) ? NIL : COMMENT;
                keep = false;
                break;
        }
        if (next != curr) {
            switch (curr) {
                case NAME: {
                    std::string lc = lower(b);
                    if (!modelfile_valid_command(lc)) return perr(k_cmd_err);
                    if (lc == "from")           cmd.name = "model";
                    else if (lc == "parameter") next = PARAM;
                    else if (lc == "message")   { next = MESSAGE; cmd.name = lc; }
                    else                        cmd.name = lc;
                    break;
                }
                case PARAM:   cmd.name = b; break;
                case MESSAGE:
                    if (b != "system" && b != "user" && b != "assistant") {
                        return perr("message role must be one of \"system\", \"user\", or \"assistant\"");
                    }
                    role = b;
                    break;
                case VALUE: {
                    std::string s;
                    if (!modelfile_unquote(modelfile_trim(b), s) || is_sp(ch)) { b.push_back(ch); continue; }
                    if (!role.empty()) { s = role + ": " + s; role.clear(); }
                    cmd.args = s;
                    cmds.push_back(cmd);
                    break;
                }
                default: break;
            }
            b.clear();
            curr = next;
        }
        if (keep && is_print(ch)) b.push_back(ch);
    }
    switch (curr) {
        case COMMENT: case NIL: break;
        case VALUE: {
            std::string s;
            if (!modelfile_unquote(modelfile_trim(b), s)) { err = "unexpected EOF"; return false; }
            if (!role.empty()) s = role + ": " + s;
            cmd.args = s;
            cmds.push_back(cmd);
            break;
        }
        default: err = "unexpected EOF"; return false;
    }
    for (auto & c : cmds) if (c.name == "model") return true;
    err = "no FROM line";
    return false;
}

// ------------------------------------------------------------------ PARAMETER / Options
using json = common_json;

// api/types.go:1264-1334: name -> value type. 'f' float32, 'i' int, 'b' bool, 's' string list (stop).
inline char option_type(const std::string & k) {
    static const struct { const char * n; char t; } tab[] = {
        {"num_keep",'i'},{"seed",'i'},{"num_predict",'i'},{"top_k",'i'},{"top_p",'f'},{"min_p",'f'},{"typical_p",'f'},
        {"repeat_last_n",'i'},{"temperature",'f'},{"repeat_penalty",'f'},{"presence_penalty",'f'},{"frequency_penalty",'f'},
        {"stop",'s'},{"num_ctx",'i'},{"num_batch",'i'},{"num_gpu",'i'},{"main_gpu",'i'},{"use_mmap",'b'},{"num_thread",'i'},
        {"draft_num_predict",'i'},
    };
    for (auto & e : tab) if (k == e.n) return e.t;
    return 0;
}

inline bool deprecated_parameter(const std::string & k) {
    static const char * d[] = { "penalize_newline", "low_vram", "f16_kv", "logits_all", "vocab_only", "use_mlock",
                                "mirostat", "mirostat_tau", "mirostat_eta" };
    for (auto * x : d) if (k == x) return true;
    return false;
}

// Builds the params object from Modelfile PARAMETER commands (create/manifest.go semantics: last wins per key,
// `stop` accumulates). Coercion strictness follows FormatParams: "0.5" for an int is an error.
inline bool modelfile_params(const std::vector<modelfile_cmd> & cmds, json & out, std::string & err,
                             std::vector<std::string> * warnings = nullptr) {
    out = json::object();
    for (auto & c : cmds) {
        if (c.name == "model" || c.name == "system" || c.name == "template" || c.name == "license" ||
            c.name == "message" || c.name == "adapter" || c.name == "draft" || c.name == "renderer" ||
            c.name == "parser" || c.name == "requires" || c.name == "capability") continue;
        if (deprecated_parameter(c.name)) {
            if (warnings) warnings->push_back("parameter " + c.name + " is deprecated");
            continue;
        }
        char t = option_type(c.name);
        if (!t) { err = "unknown parameter '" + c.name + "'"; return false; }
        if (c.name == "typical_p") {
            err = "typical_p is deprecated and cannot be set as a model parameter; pass it as a request option instead";
            return false;
        }
        const std::string & v = c.args;
        errno = 0;
        char * end = nullptr;
        if (t == 'f') {
            double d = std::strtod(v.c_str(), &end);
            if (v.empty() || *end || errno) { err = "invalid float value [" + v + "]"; return false; }
            out[c.name] = d;
        } else if (t == 'i') {
            long long x = std::strtoll(v.c_str(), &end, 10);
            if (v.empty() || *end || errno) { err = "invalid int value [" + v + "]"; return false; }
            out[c.name] = x;
        } else if (t == 'b') {
            std::string l = lower(v);
            if (v == "1" || l == "t" || l == "true") out[c.name] = true;
            else if (v == "0" || l == "f" || l == "false") out[c.name] = false;
            else { err = "invalid bool value [" + v + "]"; return false; }
        } else {
            if (!out.contains("stop")) out["stop"] = json::array();
            out["stop"].push_back(v);
        }
    }
    return true;
}

// Options{} (request options layered over Modelfile defaults) -> the field names server-schema.cpp reads.
// Runner options (num_ctx, num_batch, num_gpu, num_thread, main_gpu, use_mmap, draft_num_predict) are load-time
// in this server and are deliberately not mapped (see the num_ctx note in server-ollama.cpp).
inline void apply_options(const json & defaults, const json & req_options, json & target) {
    json eff = defaults.is_object() ? defaults : json::object();
    if (req_options.is_object()) {
        for (auto it = req_options.begin(); it != req_options.end(); ++it) {
            if (!it.value().is_null()) eff[it.key()] = it.value();
        }
    }
    static const struct { const char * from; const char * to; } map[] = {
        {"temperature","temperature"},{"top_p","top_p"},{"top_k","top_k"},{"min_p","min_p"},{"typical_p","typical_p"},
        {"repeat_last_n","repeat_last_n"},{"repeat_penalty","repeat_penalty"},{"presence_penalty","presence_penalty"},
        {"frequency_penalty","frequency_penalty"},{"seed","seed"},{"num_predict","n_predict"},{"num_keep","n_keep"},
    };
    for (auto & m : map) {
        if (eff.contains(m.from) && eff.at(m.from).is_number()) target[m.to] = eff.at(m.from);
    }
    if (eff.contains("stop") && eff.at("stop").is_array()) target["stop"] = eff.at("stop");
}

// Re-emits a Modelfile (parser.go:441-456, 677-687) for /api/show.
inline std::string modelfile_quote(const std::string & s) {
    bool nl = s.find('\n') != std::string::npos;
    if (nl || (!s.empty() && (s.front() == ' ' || s.back() == ' '))) {
        return s.find('"') != std::string::npos ? "\"\"\"" + s + "\"\"\"" : "\"" + s + "\"";
    }
    return s;
}

inline std::string modelfile_format(const std::vector<modelfile_cmd> & cmds) {
    std::string out;
    for (auto & c : cmds) {
        if (c.name == "model")          out += "FROM " + c.args + "\n";
        else if (c.name == "message") {
            size_t p = c.args.find(": ");
            out += "MESSAGE " + (p == std::string::npos ? c.args : c.args.substr(0, p) + " " + modelfile_quote(c.args.substr(p + 2))) + "\n";
        } else if (option_type(c.name) || deprecated_parameter(c.name)) out += "PARAMETER " + c.name + " " + modelfile_quote(c.args) + "\n";
        else {
            std::string u = c.name; for (auto & ch : u) ch = (char) std::toupper((unsigned char) ch);
            out += u + " " + modelfile_quote(c.args) + "\n";
        }
    }
    return out;
}

// ---- media tags for /api/generate with raw or suffix (row S69-ENGREV-OLLAMA-SHIM-RAW-AND-SUFFIX-IMAGES-2026-11-07-01)
// SOURCE-READ: tmp/ollama_src_1107/server_routes.go:578-581,585-602,703 and
//   tmp/_tca_ollama_llama_server_0930.go:1707-1722 (ollama/ollama, MIT; behaviour only, no code copied).
// SOURCE-NUANCE: ollama never rejects images with raw or suffix: media IDs are 0..n-1 (routes.go:578-581), raw and suffix text is NOT
//   tagged for the client (the chat path prefixes "[img-N]" itself, generate raw/suffix does not), the runner swaps only the FIRST
//   "[img-N]" per image and sends every image even when its tag is missing, so an untagged image is a llama-server error, not a drop.
// Swaps the first "[img-i]" found in `parts` (searched in order, the way one rendered prompt reads) for `marker`, for i in 0..n_images-1.
// Returns how many of the n_images tags were found; a tag appearing twice keeps its second occurrence as plain text.
inline size_t swap_media_tags(const std::vector<std::string *> & parts, size_t n_images, const std::string & marker) {
    size_t found = 0;
    for (size_t i = 0; i < n_images; ++i) {
        const std::string tag = "[img-" + std::to_string(i) + "]";
        for (std::string * p : parts) {
            const size_t at = p->find(tag);
            if (at != std::string::npos) {
                p->replace(at, tag.size(), marker);
                ++found;
                break;
            }
        }
    }
    return found;
}

// The FIM layout of the fork's /infill (format_prompt_infill, server-common.cpp) as TEXT, for a request that carries media: the token form
// cannot hold an image chunk, so the multimodal tokeniser parses these special-token pieces back to ids. Absent pieces are "".
// Order: extra (rep + project, sep + file + text ...), sep + "filename", then pre prefix suf suffix mid (spm: suf suffix pre prefix mid).
// The BOS comes from the tokeniser (add_special); the n_batch trimming of the token form is NOT applied (a media chunk cannot be cut).
struct fim_pieces {
    std::string pre, suf, mid, rep, sep;
    bool        spm = false;
};
struct fim_extra_chunk {
    std::string filename, text;
};
inline std::string fim_text(const fim_pieces & p, const std::string & prefix, const std::string & suffix,
                            const std::vector<fim_extra_chunk> & extra) {
    std::string out;
    if (!p.rep.empty()) {
        out += p.rep + "myproject\n";
    }
    for (const auto & c : extra) {
        out += !p.sep.empty() ? p.sep + c.filename + "\n" : std::string("\n\n--- snippet ---\n\n");
        out += c.text;
    }
    if (!p.sep.empty()) {
        out += p.sep + "filename\n";
    }
    const std::string a = p.pre + prefix;
    const std::string b = p.suf + suffix;
    out += p.spm ? b + a : a + b;
    out += p.mid;
    return out;
}

} // namespace ollama_util
