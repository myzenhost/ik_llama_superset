#include "server-ollama.h"
#include "server-context.h"
#include "server-common.h"
#include "json.h"
#include "server-ollama-util.h"
#include "server-ollama-store.h"

#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// See server-ollama.h for the full design note, SOURCE-READ citations and documented scope cuts.
//
// SOURCE-READ: https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md (MIT, fetched
//   2026-09-27, read in full via browser get_page_text -- every request/response field named below
//   -- options, format, think, images, raw, keep_alive, and the final `done` object shape with
//   total_duration/load_duration/prompt_eval_count/prompt_eval_duration/eval_count/eval_duration --
//   comes from this document).
// SOURCE-READ: https://raw.githubusercontent.com/mudler/LocalAI/master/core/schema/ollama.go (MIT,
//   fetched 2026-09-27, read in full) -- struct field names (OllamaOptions, OllamaChatRequest/
//   Response, OllamaGenerateRequest/Response, OllamaEmbedRequest/Response, OllamaShowResponse,
//   OllamaModelDetails, OllamaModelEntry, OllamaPsEntry).
// SOURCE-READ: https://raw.githubusercontent.com/mudler/LocalAI/master/core/http/routes/ollama.go
//   (MIT, fetched 2026-09-27, read in full) -- route table (/api/chat, /api/generate, /api/embed +
//   /api/embeddings, /api/tags, /api/show, /api/ps, /api/version), confirms /api/create|pull|push|
//   copy|delete|blobs are NOT registered there either (LocalAI does not implement ollama's registry
//   CRUD subset), matching this row's own scope decision independently.
// SOURCE-READ: https://raw.githubusercontent.com/mudler/LocalAI/master/core/http/endpoints/ollama/chat.go
//   (MIT, fetched 2026-09-27, read in full) -- non-stream and NDJSON-stream response construction,
//   the `done`-object field set actually populated in practice (TotalDuration/PromptEvalCount/
//   EvalCount only -- LoadDuration/PromptEvalDuration/EvalDuration are declared in the schema but
//   never set by this reference implementation either).
// SOURCE-READ: https://raw.githubusercontent.com/mudler/LocalAI/master/core/http/endpoints/ollama/generate.go
//   (MIT, fetched 2026-09-27, read in full) -- empty-prompt "load" short-circuit, system+prompt to
//   messages conversion, `raw` field meaning.
// SOURCE-READ: https://raw.githubusercontent.com/mudler/LocalAI/master/core/http/endpoints/ollama/embed.go
//   (MIT, fetched 2026-09-27, read in full) -- request/response field mapping for /api/embed.
// SOURCE-READ: https://raw.githubusercontent.com/mudler/LocalAI/master/core/http/endpoints/ollama/models.go
//   (MIT, fetched 2026-09-27, read in full) -- /api/tags, /api/show, /api/ps, /api/version response
//   shapes, including the "digest = sha256(name), not sha256(weights)" precedent this file follows.
// SOURCE-READ: https://raw.githubusercontent.com/mudler/LocalAI/master/core/http/endpoints/ollama/helpers.go
//   (MIT, fetched 2026-09-27, read in full) -- writeNDJSON/ollamaError/applyOllamaOptions/
//   ollamaMessagesToOpenAI, including the num_ctx-ceiling security note (issue
//   mudler/LocalAI#11022: an unauthenticated client must never be able to RAISE num_ctx, since that
//   drives KV-cache allocation and can OOM the host) -- cited in apply_ollama_options below, where
//   this fork goes one step further by not wiring num_ctx to anything at all (no such per-request
//   knob exists in this fork's schema to even cap).
// SOURCE-READ: https://developers.openai.com/api/docs/guides/embeddings (redirected from
//   platform.openai.com/docs/guides/embeddings, fetched 2026-09-27, read in full) -- confirms
//   `dimensions` passthrough in handle_embed needs no extra translation, since both wire protocols
//   spell the field identically and this fork's own /v1/embeddings already implements it (row
//   S69-SERVING-SURFACE-EMBED-DIMENSIONS-2026-09-27-01, TASK A of this same session).
// SOURCE-READ: tools/server/server-schema.cpp:46,91,99,132,151,154,210,561 (this fork, own code,
//   grep-verified 2026-09-27) -- exact accepted field names for the OAI-shaped request this shim
//   builds (n_predict/alias max_tokens, top_k, min_p, repeat_penalty, frequency_penalty,
//   presence_penalty, seed, stop).
// SOURCE-READ: tools/server/server-common.cpp:1248-1260,1423-1430,1695-1712 (this fork, own code,
//   read in full) -- response_format parsing rules, reasoning_effort parsing rules, and the exact
//   SSE framing (format_oai_sse) this shim's parse_sse_events() reverses.
// SOURCE-READ: tools/server/server-context.cpp:5120-5164,5299-5459,5585-5587,6697-6793 (this fork,
//   own code, read in full) -- get_meta()/get_model_info() field provenance, the streaming setup in
//   handle_completions_impl this shim wraps from the outside, and handle_embeddings_impl (this
//   session's own TASK A).
// Licence note: this .cpp/.h pair is original C++ written from the specs above, not a translation
// of the Go source line-by-line -- "no code copied" applies to the MIT sources too even though MIT
// would permit literal copying, because Go and this fork's C++ handler shapes are different enough
// (task-queue-based vs. direct-call) that a literal port was never applicable; what was carried
// over is the WIRE PROTOCOL SHAPE (field names, NDJSON framing, done-object contents) and the
// specific security lesson about num_ctx, both cited above at the point each is used.
//
// COMPATIBILITY-INVESTIGATED: the design below (call the EXISTING PUBLIC server_routes handlers
// with a translated request, translate the response back) was picked over two alternatives after
// reading server-context.h/.cpp: a second, forked completions pipeline for ollama requests (the row
// spec explicitly requires reusing the pipeline, not forking it -- two copies would drift), and a
// new "skip templating / emit ollama JSON" mode flag added inside handle_completions_impl itself
// (that method is kept `private` in server-context.h on purpose, and its region of
// server-context.cpp is a sibling task's concurrently-edited forbidden zone this session, so a
// change there would need review this task cannot get). The chosen design touches zero lines of
// the shared completion internals: the actual sampling/streaming code path is reused verbatim, and
// it is the same "translate at the edge" shape LocalAI's own port uses against ITS pipeline
// (SOURCE-READ core/http/endpoints/ollama/*.go above).

namespace {

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string iso8601_utc(std::time_t t) {
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

std::string iso8601_now() {
    return iso8601_utc(std::time(nullptr));
}

// Minimal SHA-256 (FIPS 180-4), written directly from the published specification -- not adapted
// from any particular codebase, so there is no attribution obligation (the algorithm itself is a
// public standard, not anyone's copyrightable expression). Used only to produce a stable, unique
// "digest" string for /api/tags and /api/ps, matching the *shape* ollama clients expect
// ("sha256:<64 hex chars>"); it hashes the model's identifying name, not its weights, exactly like
// LocalAI's own reference port does (core/http/endpoints/ollama/models.go: `sha256.Sum256([]byte(name))`,
// SOURCE-READ above) -- neither implementation hashes actual model content, since llama.cpp has no
// existing "content hash of this GGUF" primitive to call instead.
std::string sha256_hex(const std::string & input) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    };
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    };

    std::string msg = input;
    uint64_t bit_len = (uint64_t) msg.size() * 8;
    msg.push_back((char) 0x80);
    while (msg.size() % 64 != 56) {
        msg.push_back((char) 0x00);
    }
    for (int i = 7; i >= 0; i--) {
        msg.push_back((char) ((bit_len >> (i * 8)) & 0xff));
    }

    auto rotr = [](uint32_t x, uint32_t n) -> uint32_t { return (x >> n) | (x << (32 - n)); };

    for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            const auto * p = (const unsigned char *) msg.data() + chunk + i * 4;
            w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }

        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            hh = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

    static const char * hexd = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 8; i++) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            out.push_back(hexd[(h[i] >> shift) & 0xf]);
        }
    }
    return out;
}

server_http_res_ptr ollama_error(int status, const std::string & message) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data = safe_json_to_str(json {{ "error", message }});
    return res;
}

// ollama's "model" field carries an optional ":tag" suffix (default "latest") that this
// single-model-per-process server has no concept of; strip it for lookups, same as LocalAI's
// ShowModelEndpoint (`strings.Split(name, ":")[0]`, SOURCE-READ above).
std::string strip_tag(const std::string & name) {
    auto pos = name.find(':');
    return pos == std::string::npos ? name : name.substr(0, pos);
}

std::string with_latest_tag(const std::string & name) {
    return name.find(':') == std::string::npos ? name + ":latest" : name;
}

// Applies ollama's `options{}` object onto an OAI-shaped request body being built. Field names on
// the right are exactly what tools/server/server-schema.cpp's request parser reads (SOURCE-READ
// above: n_predict/alias max_tokens, top_k, min_p, repeat_penalty, frequency_penalty,
// presence_penalty, seed, stop; temperature/top_p are standard OAI fields the same parser also
// reads). `num_ctx` is NOT an OAI field and is not mapped here: the schema has no per-request context-resize field. In router mode
// handle_chat_or_generate passes it on as "ollama_num_ctx" and the router reloads the child (server-models.cpp apply_num_ctx, capped at
// the model train ctx and refused with 400 when it cannot be had); a single-model server logs one WARN that it was not applied.
// LocalAI treats a positive num_ctx as a ceiling it never lets a client raise (core/http/endpoints/ollama/helpers.go, SOURCE-READ above,
// its issue #11022, a client-triggered OOM): the router here clamps to the train ctx and refuses when the memory check fails.
// Request options layered over the Modelfile PARAMETER defaults; field map and precedence live in
// server-ollama-util.h (ollama_util::apply_options, SOURCE-READ api/types.go:574-605, server/routes.go:126-160).
void apply_ollama_options(const json & defaults, const json & options, json & target) {
    ollama_util::apply_options(defaults, options, target);
}

// options.num_ctx over the Modelfile PARAMETER default, same precedence as apply_options (request wins, null ignored).
// 0 = not set (ollama: auto-sized). SOURCE-READ ollama api/types.go Options.NumCtx; the router clamps it (sched.go:204-220).
int64_t ollama_requested_num_ctx(const json & defaults, const json & options) {
    json v;
    if (options.is_object() && options.contains("num_ctx") && !options.at("num_ctx").is_null()) {
        v = options.at("num_ctx");
    } else if (defaults.is_object() && defaults.contains("num_ctx") && !defaults.at("num_ctx").is_null()) {
        v = defaults.at("num_ctx");
    }
    if (v.is_number_integer()) {
        return std::max<int64_t>(v.get<int64_t>(), 0);
    }
    if (v.is_number_float()) {
        return std::max<int64_t>((int64_t) v.get<double>(), 0);
    }
    return 0;
}

// ollama's `think` is bool or "low"/"medium"/"high"/"max"; our schema's `reasoning_effort` is a
// free-form string, "none" disabling it (server-common.cpp:1423-1430, SOURCE-READ above). There is
// no single correct mapping from a bool to an effort level -- ollama's own semantics are "reasoning
// on/off", not a level -- so `true` maps to "medium" as a reasonable default level, documented here
// rather than silently guessed elsewhere.
//
// Validation (api/types.go:1208-1220 ValidateLegacyThinking, types/model/thinking.go:137-148 UnmarshalJSON): a string other than
// low/medium/high/max, or a value that is neither bool nor string, is refused with ollama's own message instead of being passed on
// to the chat template as a free-form effort. Returns "" when fine, else the 400 message.
std::string apply_ollama_think(const json & body, json & target) {
    if (!body.contains("think") || body.at("think").is_null()) {
        return "";
    }
    const json & think = body.at("think");
    if (think.is_boolean()) {
        target["reasoning_effort"] = think.get<bool>() ? "medium" : "none";
    } else if (think.is_string()) {
        const std::string v = think.get<std::string>();
        if (v != "low" && v != "medium" && v != "high" && v != "max") {
            return "invalid think value: \"" + v + "\" (must be \"high\", \"medium\", \"low\", \"max\", true, or false)";
        }
        target["reasoning_effort"] = v;
    } else {
        return "think must be a boolean or string";
    }
    return "";
}

// ollama's `format` is either the literal string "json" or a JSON Schema object; our schema's
// `response_format` follows the OAI shape (server-common.cpp:1248-1260, SOURCE-READ above: type
// must be "text" or "json_object", with an optional schema under response_format.json_schema.schema).
// Returns an error message (400) for a format that is neither "json" nor a schema object, as ollama does
// (llm/llama_server.go:1689 `invalid format: %q; expected "json" or a valid JSON Schema object`); "" and null mean unset.
// `raw_target` selects the /completion field names (json_schema / grammar-free) instead of OAI response_format.
std::string apply_ollama_format(const json & body, json & target, bool raw_target = false) {
    if (!body.contains("format") || body.at("format").is_null()) {
        return "";
    }
    const json & format = body.at("format");
    if (format.is_string() && format.get<std::string>().empty()) {
        return "";
    }
    if (format.is_string() && format.get<std::string>() == "json") {
        if (raw_target) {
            target["json_schema"] = json {{ "type", "object" }};
        } else {
            target["response_format"] = json {{ "type", "json_object" }};
        }
        return "";
    }
    if (format.is_object()) {
        if (raw_target) {
            target["json_schema"] = format;
        } else {
            target["response_format"] = json {
                { "type", "json_schema" },
                { "json_schema", json {
                    { "name", "response" },
                    { "schema", format },
                }},
            };
        }
        return "";
    }
    return "invalid format: " + format.dump() + "; expected \"json\" or a valid JSON Schema object";
}

// HARNESS-CONSIDERED: this is JSON-string-vs-object reshaping of an already-produced tool call's
// arguments payload, not the tool-invocation loop itself (which stays entirely in the caller/agent
// harness on the other side of this HTTP boundary, unmodified). Nothing about tool-use, agentic
// looping, reasoning or state is disabled here; parse_tool_arguments only prevents one malformed
// argument string from turning an otherwise-successful completion into a 500.
// OAI's `function.arguments` is a JSON-encoded STRING (SOURCE-READ common/chat.cpp:245-250,
// grep-verified 2026-09-27: `{"arguments", json(tool_call.arguments)}` where tool_call.arguments is
// a std::string -- wrapping a string in json() makes a JSON string, not the parsed value). ollama's
// own wire shape uses a JSON OBJECT for the same field (PARENT REVIEW 2026-09-27). A model can
// legitimately emit malformed JSON for tool arguments (PRIOR-ART-NONE: no fork test corpus of
// malformed tool-call arguments exists to characterize how often this happens, so this is a
// defensive fallback, not a measured-necessary one); rather than let that throw and drop the whole
// response, an unparseable string is reported back as `{"_raw": "<original string>"}` so the caller
// can see something is wrong instead of getting a 500 for an otherwise-successful completion.
json parse_tool_arguments(const std::string & args_str) {
    if (args_str.empty()) {
        return json::object();
    }
    try {
        json parsed = json::parse(args_str);
        return parsed.is_object() ? parsed : json {{ "_raw", args_str }};
    } catch (const std::exception &) {
        return json {{ "_raw", args_str }};
    }
}

json translate_tool_calls(const json & oai_tool_calls) {
    json out = json::array();
    if (!oai_tool_calls.is_array()) {
        return out;
    }
    for (const auto & tc : oai_tool_calls) {
        if (!tc.contains("function")) {
            continue;
        }
        const json & fn = tc.at("function");
        std::string name = json_value(fn, "name", std::string(""));
        std::string args_str = json_value(fn, "arguments", std::string(""));
        json entry = json {
            { "function", json {
                { "name", name },
                { "arguments", parse_tool_arguments(args_str) },
                { "index", (int) out.size() },
            }},
        };
        if (tc.contains("id") && tc.at("id").is_string()) {
            entry["id"] = tc.at("id");
        }
        out.push_back(entry);
    }
    return out;
}

// Converts ollama chat messages (role/content/images[]/tool_calls) to the OAI message shape this
// fork's chat pipeline already parses. Base64 images become data-URI image_url content parts,
// following LocalAI's own conversion (core/http/endpoints/ollama/helpers.go
// ollamaMessagesToOpenAI, SOURCE-READ above: `fmt.Sprintf("data:image/png;base64,%s", img)`) --
// ollama does not send a mime type for inline images, so "image/png" is asserted the same way
// there, not detected here.
json ollama_messages_to_oai(const json & messages) {
    json out = json::array();
    if (!messages.is_array()) {
        return out;
    }
    // ids handed out to assistant tool_calls, so a following role:"tool" message can name its call
    // (ollama carries tool_name / tool_call_id on the tool message, api/types.go:209-211)
    std::vector<std::pair<std::string, std::string>> issued; // {function name, id}
    int next_id = 0;
    for (const auto & msg : messages) {
        std::string role    = ollama_util::lower(json_value(msg, "role", std::string("user"))); // types.go:222 lowercases
        std::string content = json_value(msg, "content", std::string(""));
        json images = msg.contains("images") ? msg.at("images") : json::array();

        json m = json::object();
        m["role"] = role;
        if (images.is_array() && !images.empty()) {
            json parts = json::array();
            if (!content.empty()) {
                parts.push_back(json {{ "type", "text" }, { "text", content }});
            }
            for (const auto & img : images) {
                if (!img.is_string()) {
                    continue;
                }
                parts.push_back(json {
                    { "type", "image_url" },
                    { "image_url", json {{ "url", "data:image/png;base64," + img.get<std::string>() }} },
                });
            }
            m["content"] = parts;
        } else {
            m["content"] = content;
        }
        if (role == "assistant") {
            std::string thinking = json_value(msg, "thinking", std::string(""));
            if (!thinking.empty()) {
                m["reasoning_content"] = thinking;
            }
            if (msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty()) {
                json tcs = json::array();
                for (const auto & tc : msg.at("tool_calls")) {
                    if (!tc.contains("function") || !tc.at("function").is_object()) {
                        continue;
                    }
                    const json & fn = tc.at("function");
                    std::string name = json_value(fn, "name", std::string(""));
                    std::string id   = json_value(tc, "id", std::string(""));
                    if (id.empty()) {
                        id = "call_" + std::to_string(next_id++);
                    }
                    issued.emplace_back(name, id);
                    // the fork's chat parser needs type:"function" (common/chat.cpp:435-438) and takes the
                    // arguments as an object (chat.cpp:452-456), which is how ollama sends them
                    tcs.push_back(json {
                        { "type", "function" }, { "id", id },
                        { "function", json {{ "name", name },
                                            { "arguments", fn.contains("arguments") ? fn.at("arguments") : json::object() }} },
                    });
                }
                if (!tcs.empty()) {
                    m["tool_calls"] = tcs;
                }
            }
        } else if (role == "tool") {
            std::string tool_name = json_value(msg, "tool_name", std::string(""));
            std::string id        = json_value(msg, "tool_call_id", std::string(""));
            if (id.empty()) {
                // newest unanswered call of that name, else the newest call
                for (auto it = issued.rbegin(); it != issued.rend(); ++it) {
                    if (tool_name.empty() || it->first == tool_name) { id = it->second; break; }
                }
            }
            if (!id.empty())        m["tool_call_id"] = id;
            if (!tool_name.empty()) m["name"] = tool_name;
        }
        out.push_back(m);
    }
    return out;
}

// Parses one httplib SSE payload (as produced by format_oai_sse / server-common.cpp:1695, SOURCE-
// READ above) into zero or more JSON event bodies. A single call to the inner handler's `next()`
// can carry more than one "data: ...\n\n" block (format_oai_sse emits one block per array element
// when the underlying task result serializes to a JSON array), and a "[DONE]" sentinel or a bare
// ":\n\n" keep-alive comment carry no JSON at all -- both are reported back via the output
// parameters instead of being silently dropped, so the caller can tell "no event this tick" from
// "stream just ended."
std::vector<json> parse_sse_events(const std::string & text, bool & saw_done) {
    std::vector<json> events;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find("\n\n", pos);
        std::string block = (end == std::string::npos) ? text.substr(pos) : text.substr(pos, end - pos);
        pos = (end == std::string::npos) ? text.size() : end + 2;
        if (block.empty()) {
            continue;
        }
        std::istringstream iss(block);
        std::string line;
        std::string data_line;
        bool have_data = false;
        while (std::getline(iss, line)) {
            if (line.rfind("data:", 0) == 0) {
                data_line = line.substr(5);
                if (!data_line.empty() && data_line.front() == ' ') {
                    data_line.erase(0, 1);
                }
                have_data = true;
                break;
            }
        }
        if (!have_data || data_line.empty()) {
            continue; // e.g. a bare ping comment line, nothing to translate
        }
        if (data_line == "[DONE]") {
            saw_done = true;
            continue;
        }
        try {
            events.push_back(json::parse(data_line));
        } catch (const std::exception &) {
            // not JSON we understand (should not happen for our own SSE output); skip it rather
            // than crash the stream over one malformed chunk
        }
    }
    return events;
}

// keep_alive 0 means "unload now" (SOURCE-READ ollama 3fbc609be4b3 server/routes.go:480 generate, :3098 chat: `req.KeepAlive != nil &&
// req.KeepAlive.Duration == 0`): a number 0, or a duration string that parses to 0 ("0", "0s", "0m").
bool keep_alive_is_zero(const json & body) {
    if (!body.contains("keep_alive") || body.at("keep_alive").is_null()) {
        return false;
    }
    const json & ka = body.at("keep_alive");
    if (ka.is_number()) {
        return ka.get<double>() == 0.0;
    }
    if (ka.is_string()) {
        int64_t ms = -1;
        return ollama_util::keep_alive_ms(ka.get<std::string>(), ms) && ms == 0;
    }
    return false;
}

// Converts the fork's per-token probability entries (completion_token_output::probs_vector_to_json, server-task.cpp:463-488:
// {id, token, bytes, logprob, top_logprobs:[{id, token, bytes, logprob}]}) -- the same entries ride under OAI chat
// choices[].logprobs.content and /completion completion_probabilities -- into ollama's []Logprob (SOURCE-READ api/types.go:502-521:
// {token, logprob, bytes?, top_logprobs?}). top_n trims the alternatives to what the client asked for: the fork is asked for at least
// one alternative because its per-token readout is gated on n_probs > 0 (server-context.cpp:3494), while ollama's top_logprobs 0
// means "the chosen token only".
json to_ollama_logprobs(const json & entries, int top_n) {
    json out = json::array();
    if (!entries.is_array()) {
        return out;
    }
    auto one = [](const json & e) {
        json lp = json {{ "token", json_value(e, "token", std::string("")) }, { "logprob", json_value(e, "logprob", 0.0) }};
        if (e.contains("bytes") && e.at("bytes").is_array() && !e.at("bytes").empty()) {
            lp["bytes"] = e.at("bytes");
        }
        return lp;
    };
    for (const auto & e : entries) {
        json lp = one(e);
        if (top_n > 0 && e.contains("top_logprobs") && e.at("top_logprobs").is_array()) {
            json tops = json::array();
            for (const auto & t : e.at("top_logprobs")) {
                if ((int) tops.size() >= top_n) {
                    break;
                }
                tops.push_back(one(t));
            }
            if (!tops.empty()) {
                lp["top_logprobs"] = tops;
            }
        }
        out.push_back(lp);
    }
    return out;
}

// prompt_eval_cached_count (api/types.go:566, routes.go:771,3363): OAI usage.prompt_tokens_details.cached_tokens
// (server-task.cpp:591), else the timings cache_n (server-common.cpp:169) that /completion and /infill report. -1 = not reported.
int cached_prompt_tokens(const json & ev) {
    if (ev.contains("usage") && ev.at("usage").is_object() && ev.at("usage").contains("prompt_tokens_details")) {
        const json & d = ev.at("usage").at("prompt_tokens_details");
        if (d.is_object() && d.contains("cached_tokens") && d.at("cached_tokens").is_number_integer()) {
            return d.at("cached_tokens").get<int>();
        }
    }
    if (ev.contains("timings") && ev.at("timings").is_object() && ev.at("timings").contains("cache_n") &&
        ev.at("timings").at("cache_n").is_number_integer()) {
        return ev.at("timings").at("cache_n").get<int>();
    }
    return -1;
}

// Per-token probability entries of one response or SSE event, whichever shape it has.
json logprob_entries(const json & ev, bool completion_shape) {
    if (completion_shape) {
        return ev.contains("completion_probabilities") ? ev.at("completion_probabilities") : json(nullptr);
    }
    if (ev.contains("choices") && ev.at("choices").is_array() && !ev.at("choices").empty()) {
        const json & c = ev.at("choices").at(0);
        if (c.contains("logprobs") && c.at("logprobs").is_object() && c.at("logprobs").contains("content")) {
            return c.at("logprobs").at("content");
        }
    }
    return json(nullptr);
}

// The inner handler's error body ({"error": {type, message, ...}}, server-common.cpp:120-146) names a context overflow by type.
bool is_exceed_context_error(const std::string & data) {
    try {
        const json err = json::parse(data);
        return err.contains("error") && err.at("error").is_object() &&
               json_value(err.at("error"), "type", std::string("")) == "exceed_context_size_error";
    } catch (const std::exception &) {
        return false;
    }
}

} // namespace

struct server_ollama_state;
static bool ollama_name_rejected(const std::shared_ptr<server_ollama_state> & st, const std::string & requested, const json & model_info);

struct server_ollama_state {
    json params = json::object();       // Modelfile PARAMETERs (coerced)
    std::string system;                 // Modelfile SYSTEM
    json messages = json::array();      // Modelfile MESSAGEs as {role, content}
    std::string modelfile_text;         // re-emitted Modelfile for /api/show
    std::string template_text;          // Modelfile TEMPLATE (Go template; shown, not applied)
    json license = json::array();
    bool strict_names = false;
    std::string store_dir;              // Ollama model store (--ollama-models-dir); empty = none
    bool store_write = false;           // --ollama-store-write
    bool router_mode = false;           // server_ollama_config::router_mode
    std::atomic<bool> warned_num_ctx{false}; // the "num_ctx not applied" WARN is logged once per process
};

void server_routes_ollama::configure(const server_ollama_config & cfg) {
    auto st = std::make_shared<server_ollama_state>();
    st->strict_names = cfg.strict_names;
    st->store_dir    = cfg.store_dir;
    st->store_write  = cfg.store_write;
    st->router_mode  = cfg.router_mode;
    if (!cfg.store_model.empty()) {
        // the store model's layers are the defaults a Modelfile then overrides (server/images.go:818-860 layer meaning)
        ollama_util::store_model sm;
        std::string serr;
        if (!ollama_util::store_resolve(cfg.store_dir, cfg.store_model, sm, serr)) {
            throw std::runtime_error(serr);
        }
        st->params   = sm.params;
        st->system   = sm.system_text;
        st->template_text = sm.template_text;
        for (auto & m : sm.messages) {
            st->messages.push_back(json {{ "role", json_value(m, "role", std::string("")) }, { "content", json_value(m, "content", std::string("")) }});
        }
        for (auto & l : sm.licenses) st->license.push_back(l);
        SRV_INF("ollama store: %s -> system=%s, %d params, %d messages, %d licenses, template=%s\n", cfg.store_model.c_str(),
                st->system.empty() ? "no" : "yes", (int) st->params.size(), (int) st->messages.size(), (int) st->license.size(),
                st->template_text.empty() ? "no" : "yes");
    }
    if (!cfg.modelfile_path.empty()) {
        std::ifstream in(cfg.modelfile_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot read " + cfg.modelfile_path);
        }
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<ollama_util::modelfile_cmd> cmds;
        std::string err;
        if (!ollama_util::parse_modelfile(text, cmds, err)) {
            throw std::runtime_error(cfg.modelfile_path + ": " + err);
        }
        std::vector<std::string> warnings;
        if (!ollama_util::modelfile_params(cmds, st->params, err, &warnings)) {
            throw std::runtime_error(cfg.modelfile_path + ": " + err);
        }
        for (auto & w : warnings) {
            SRV_WRN("modelfile: %s\n", w.c_str());
        }
        for (auto & c : cmds) {
            if (c.name == "system")        st->system = c.args;
            else if (c.name == "template") st->template_text = c.args;
            else if (c.name == "license")  st->license.push_back(c.args);
            else if (c.name == "message") {
                size_t p = c.args.find(": ");
                st->messages.push_back(json {{ "role", c.args.substr(0, p) }, { "content", p == std::string::npos ? "" : c.args.substr(p + 2) }});
            }
        }
        if (!st->template_text.empty()) {
            SRV_WRN("%s", "modelfile: TEMPLATE is a Go template and is not applied; the model's Jinja chat template is used\n");
        }
        st->modelfile_text = ollama_util::modelfile_format(cmds);
        // always-on evidence that the Modelfile reached the request path's defaults
        SRV_INF("modelfile: loaded %s (%d parameters, %d messages, system=%s)\n", cfg.modelfile_path.c_str(),
                (int) st->params.size(), (int) st->messages.size(), st->system.empty() ? "no" : "yes");
    }
    state_ = st;
}

// --ollama-strict-model-names: ollama answers 404 for a name it does not hold (server/routes.go:362). The loaded model counts as
// "held" under every spelling that normalises to the same host/namespace/model:tag, plus the bare id and its basename.
static bool ollama_name_rejected(const std::shared_ptr<server_ollama_state> & st, const std::string & requested, const json & model_info) {
    if (!st || !st->strict_names || requested.empty()) {
        return false;
    }
    const std::string id = json_value(model_info, "id", std::string(""));
    return ollama_util::canonical_key(requested) != ollama_util::canonical_key(id);
}

// ---- router mode: per-model information comes from the router's model list, never from routes.get_model_info() -------------------------
// The router's own server_context loads no model, so its `meta` is a null pointer and routes.get_model_info() dereferences it (0xC0000005,
// read 0x38). MEASURED 2026-11-08 on the first router request to /api/embed, /api/tags, /api/show and /api/ps (row
// S69-ENGREV-OLLAMA-ROUTER-NULL-META-EMBED-SHOW-2026-11-08-01; /api/chat was fixed the same way in 9086926b3). The router already knows what the
// shim needs: GET /models (get_router_models) lists every model with its status, the GGUF header (?info=) and, for a running child, the child's
// own get_model_info() payload (loaded_info: "meta" with n_ctx, n_params, sizes ...). These helpers turn one list entry into the same
// {id, aliases, meta} shape get_model_info() returns, so ollama_model_entry / the show and ps bodies are written once for both modes.
static const std::function<bool()> ollama_never_stop = []() { return false; };

static json router_models_list(const server_routes & routes, bool full_info) {
    json out = json::array();
    if (!routes.get_models) {
        return out;
    }
    server_http_req r { std::map<std::string, std::string> {{ "info", full_info ? "full" : "1" }}, {}, "/models", "", "", {}, ollama_never_stop };
    try {
        server_http_res_ptr res = routes.get_models(r);
        if (res && res->status == 200) {
            const json parsed = json::parse(res->data);
            if (parsed.contains("data") && parsed.at("data").is_array()) {
                for (const auto & e : parsed.at("data")) {
                    const std::string st = e.contains("status") && e.at("status").is_object() ? json_value(e.at("status"), "value", std::string("")) : "";
                    if (st == "incomplete" || st == "downloading") continue; // no files to describe yet; ollama lists only models it holds
                    out.push_back(e);
                }
            }
        }
    } catch (const std::exception & e) {
        SRV_WRN("ollama: router model list unavailable (%s)\n", e.what());
    }
    return out;
}

// A router hands back the child's answer as a proxied stream: the body comes through next(), `data` stays empty (a single-model handler
// fills `data`). Collect it so the non-streaming callers below can parse it either way. A response that is not a stream is left alone.
static void ollama_drain_body(server_http_res_ptr & res) {
    if (!res || !res->next) {
        return;
    }
    std::string all, chunk;
    for (;;) {
        chunk.clear();
        const bool more = res->next(chunk);
        all += chunk;
        if (!more) break;
    }
    res->data += all;
    res->next = nullptr;
}

// one router list entry -> get_model_info()-shaped json (+ "loaded", + "gguf" for the show body)
static json router_entry_to_info(const json & e) {
    json meta = e.contains("meta") && e.at("meta").is_object() ? e.at("meta") : json::object();  // present only for a running child
    const json gguf = e.contains("gguf") && e.at("gguf").is_object() ? e.at("gguf") : json::object();
    auto fill = [&](const char * key, const json & v) { if (!meta.contains(key) || meta.at(key).is_null()) meta[key] = v; };
    fill("n_ctx_train", gguf.contains("context_length") ? gguf.at("context_length") : json_value(e, "context_length", (int64_t) 0));
    fill("n_embd",      json_value(gguf, "embedding_length", (int64_t) 0));
    fill("arch",        json_value(gguf, "architecture", std::string("")));
    fill("size",        json_value(gguf, "file_bytes", (int64_t) 0));
    const std::string st = e.contains("status") && e.at("status").is_object() ? json_value(e.at("status"), "value", std::string("")) : "";
    return json {
        { "id",      json_value(e, "id", std::string("")) },
        { "aliases", e.contains("aliases") ? e.at("aliases") : json::array() },
        { "meta",    meta },
        { "loaded",  st == "loaded" },
        { "gguf",    gguf },
    };
}

// the list entry a request name resolves to (id or alias, ollama spelling variants folded), null json when the router holds no such model
static json router_find_info(const server_routes & routes, const std::string & name, bool full_info) {
    const std::string want = ollama_util::canonical_key(name);
    for (const auto & e : router_models_list(routes, full_info)) {
        if (json_value(e, "id", std::string("")) == name || ollama_util::canonical_key(json_value(e, "id", std::string(""))) == want) {
            return router_entry_to_info(e);
        }
        if (e.contains("aliases") && e.at("aliases").is_array()) {
            for (const auto & al : e.at("aliases")) {
                if (al.is_string() && (al.get<std::string>() == name || ollama_util::canonical_key(al.get<std::string>()) == want)) {
                    return router_entry_to_info(e);
                }
            }
        }
    }
    return nullptr;
}

// Swaps the "[img-N]" tags in `parts` for the server's media marker. Every image must have its tag: ollama forwards the untagged image
// and llama-server then fails on the count, so the same refusal is made here up front with a message that says why.
static bool media_tags_to_marker(const json & images, const std::vector<std::string *> & parts, std::string & err) {
    for (const auto & img : images) {
        if (!img.is_string()) {
            err = "images must be base64 strings";
            return false;
        }
    }
    const size_t found = ollama_util::swap_media_tags(parts, images.size(), get_media_marker());
    if (found != images.size()) {
        err = "the prompt has " + std::to_string(found) + " of the " + std::to_string(images.size()) +
              " [img-N] tags needed to place the images (write [img-0], [img-1], ... where each image goes)";
        return false;
    }
    return true;
}

static bool raw_prompt_peek(const json & body, bool is_generate) {
    return is_generate && json_value(body, "raw", false) && !json_value(body, "prompt", std::string("")).empty();
}

server_routes_ollama::server_routes_ollama(server_routes & routes_) : routes(routes_) {
    post_api_chat     = [this](const server_http_req & req) { return handle_chat_or_generate(req, false); };
    post_api_generate = [this](const server_http_req & req) { return handle_chat_or_generate(req, true); };
    post_api_embed    = [this](const server_http_req & req) { return handle_embed(req); };
    post_api_embeddings_legacy = [this](const server_http_req & req) { return handle_embed(req, /*legacy=*/true); };
    get_api_tags      = [this](const server_http_req & req) { return handle_tags(req); };
    post_api_show     = [this](const server_http_req & req) { return handle_show(req); };
    get_api_ps        = [this](const server_http_req & req) { return handle_ps(req); };
    get_api_version   = [this](const server_http_req & req) { return handle_version(req); };
    post_api_copy     = [this](const server_http_req & req) { return handle_copy(req); };
    del_api_delete    = [this](const server_http_req & req) { return handle_delete(req); };
}

server_http_res_ptr server_routes_ollama::handle_chat_or_generate(const server_http_req & req, bool is_generate) const {
    const int64_t t_start = now_ns();
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & e) {
        return ollama_error(400, std::string("invalid JSON body: ") + e.what());
    }

    std::string model = json_value(body, "model", std::string(""));
    if (ollama_name_rejected(state_, model, state_ && state_->strict_names ? routes.get_model_info() : json::object())) {
        return ollama_error(404, "model '" + model + "' not found");
    }
    bool stream = true; // ollama defaults stream=true when the field is absent, per docs/api.md
    if (body.contains("stream") && body.at("stream").is_boolean()) {
        stream = body.at("stream").get<bool>();
    }

    json oai_body = json {{ "model", model }, { "stream", stream }};
    if (stream && !raw_prompt_peek(body, is_generate)) {
        // Without this the inner stream never emits its usage chunk (server-task.cpp:534 gates it on
        // include_usage), so the final ollama chunk reported prompt_eval_count 0 on every streamed request
        // (MEASURED 2026-09-27, test_ollama_chat_stream) and eval_count fell back to a per-chunk estimate.
        oai_body["stream_options"] = json {{ "include_usage", true }};
    }

    // SOURCE-READ: tmp/ollama_src_1107/server_routes.go (ollama/ollama 3fbc609be4b3, MIT; no code copied -- the Go
    //   handlers were read for behaviour and the shim below is written against the fork's own handler shapes).
    // SOURCE-NUANCE: top_logprobs is range-checked (0..20) BEFORE the model lookup and independently of logprobs (routes.go:331,373,
    //   3043,3087); unload needs an EMPTY prompt/messages AND keep_alive exactly 0 (routes.go:480,3098) -- keep_alive 0 on a real
    //   prompt still generates; suffix renders Prompt+Suffix with NO system message (routes.go:596-598); chat truncation keeps the
    //   system messages of the dropped head and never drops the last message (prompt.go:35-74).
    // logprobs / top_logprobs (api/types.go:126-132,178-184; routes.go:331,3043: top_logprobs outside 0..20 is a 400 whether or not
    // logprobs is set).
    int top_logprobs = 0;
    if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        if (!body.at("top_logprobs").is_number_integer()) {
            return ollama_error(400, "top_logprobs must be between 0 and 20");
        }
        top_logprobs = body.at("top_logprobs").get<int>();
        if (top_logprobs < 0 || top_logprobs > 20) {
            return ollama_error(400, "top_logprobs must be between 0 and 20");
        }
    }
    const bool want_logprobs = body.contains("logprobs") && body.at("logprobs").is_boolean() && body.at("logprobs").get<bool>();

    // Load / unload replies (routes.go:480-491 generate unload, :563-571 generate load, :3098-3107 chat unload, :3241-3248 chat load).
    // "unload" is only said when it happened. The request used to be answered right here with "load" and never reached the router, so
    // router mode neither loaded the model on an empty request nor unloaded it on keep_alive 0 (MEASURED 2026-11-08: /api/ps unchanged,
    // the SERVER_ENV_ROUTER_PORT test below is true only inside a CHILD, never in the router process that runs this handler).
    //   router mode : the router's own load / unload handlers (server-models.cpp ollama_model_load / ollama_model_unload) load the child
    //                 (applying options.num_ctx and keep_alive) or expire it now, and the reply waits for the child to be gone.
    //   single model: keep_alive 0 puts the model to sleep (queue request_sleep, needs --sleep-idle-seconds >= 0; the next request wakes it);
    //                 without sleeping a server cannot unload its own model, so it says "load" and logs why. An empty request wakes a
    //                 sleeping model.
    const bool ka_zero = keep_alive_is_zero(body);
    auto load_reply = [&](bool chat) -> server_http_res_ptr {
        bool unload = false;
        if (state_ && state_->router_mode) {
            const auto & handler = ka_zero ? routes.ollama_model_unload : routes.ollama_model_load;
            if (handler) {
                json cb = json {{ "model", model }};
                if (!ka_zero) {
                    if (body.contains("keep_alive") && !body.at("keep_alive").is_null()) {
                        cb["keep_alive"] = body.at("keep_alive");
                    }
                    const int64_t want_ctx = ollama_requested_num_ctx(state_->params, body.contains("options") ? body.at("options") : json::object());
                    if (want_ctx > 0) {
                        cb["ollama_num_ctx"] = want_ctx;
                    }
                }
                server_http_req cr { {}, {}, ka_zero ? "/models/unload" : "/models/load", "", safe_json_to_str(cb), {}, req.should_stop };
                server_http_res_ptr rr = handler(cr);
                if (rr == nullptr) {
                    return nullptr; // connection closed
                }
                ollama_drain_body(rr);
                if (rr->status != 200) {
                    std::string msg = rr->data;
                    try {
                        const json e = json::parse(rr->data);
                        if (e.contains("error")) {
                            msg = e.at("error").is_object() ? json_value(e.at("error"), "message", msg) : e.at("error").dump();
                        }
                    } catch (const std::exception &) {}
                    return ollama_error(rr->status, msg);
                }
                unload = ka_zero;
            } else if (ka_zero) {
                SRV_WRN("%s", "ollama: keep_alive 0 asks for an unload, but this router has no unload handler; the model stays loaded\n");
            }
        } else if (ka_zero) {
            if (routes.model_sleep_enabled()) {
                routes.sleep_model(15000);
                unload = true;
            } else {
                SRV_WRN("%s", "ollama: keep_alive 0 asks for an unload, but this server cannot unload its own model (start it with "
                              "--sleep-idle-seconds N to let keep_alive 0 put it to sleep, or run llama-server in router mode); the model stays loaded\n");
            }
        } else {
            routes.wake_model(); // an empty request loads: a sleeping model wakes, an awake one is already loaded
        }
        json resp = json {
            { "model", model }, { "created_at", iso8601_now() }, { "done", true }, { "done_reason", unload ? "unload" : "load" },
        };
        if (chat) {
            resp["message"] = json {{ "role", "assistant" }, { "content", "" }};
        } else {
            resp["response"] = "";
        }
        auto res = std::make_unique<server_http_res>();
        if (stream) {
            res->content_type = "application/x-ndjson";
        }
        res->data = safe_json_to_str(resp);
        return res;
    };
    if (!is_generate) {
        const bool no_messages = !body.contains("messages") || body.at("messages").is_null() ||
                                 (body.at("messages").is_array() && body.at("messages").empty());
        if (no_messages) {
            return load_reply(true);
        }
    }

    bool raw = false;
    bool infill = false; // /api/generate with suffix -> the fork's /infill (fill-in-the-middle)
    json images = json::array();
    if (is_generate) {
        std::string prompt = json_value(body, "prompt", std::string(""));
        if (prompt.empty()) {
            // matches ollama/LocalAI: an empty prompt just (pre)loads (or, with keep_alive 0, unloads) the model and returns
            // immediately, it is not an error
            return load_reply(false);
        }
        json messages = json::array();
        std::string system = json_value(body, "system", std::string(""));
        const std::string suffix = json_value(body, "suffix", std::string(""));
        if (body.contains("images") && body.at("images").is_array()) {
            images = body.at("images");
        }
        raw = json_value(body, "raw", false);
        if (raw) {
            // raw:true sends the prompt byte-for-byte with no chat template (docs/api.md; server/routes.go:498-501 rejects
            // template/system/context in raw mode). It goes through /completion, which tokenises the string as given.
            if (!system.empty() || body.contains("template") || body.contains("context")) {
                return ollama_error(400, "raw mode does not support template, system, or context");
            }
            if (images.empty()) {
                oai_body = json {{ "model", model }, { "stream", stream }, { "prompt", prompt }};
            } else {
                // raw + images (routes.go:578-581 builds the media, 585 skips the template for raw): the prompt is sent as written, each
                // "[img-N]" in it standing for image N, as a {prompt_string, multimodal_data} prompt -- the shape /completion already
                // tokenises with the multimodal tokeniser (tokenize_input_subprompt). SOURCE-READ: see ollama_util::swap_media_tags.
                std::string err_media;
                if (!media_tags_to_marker(images, { &prompt }, err_media)) {
                    return ollama_error(400, err_media);
                }
                oai_body = json {{ "model", model }, { "stream", stream },
                                 { "prompt", json {{ "prompt_string", prompt }, { "multimodal_data", images }} }};
            }
        } else if (!suffix.empty()) {
            // suffix = insert/FIM (routes.go:512-513 requires the insert capability, :596-598 renders Prompt + Suffix with the insert
            // template and no system message). The fork's /infill builds the FIM prompt from the vocab's FIM tokens and answers
            // "Infill is not supported by this model" (501) when they are missing, the counterpart of ollama's capability check.
            infill = true;
            std::string suffix_text = suffix;
            oai_body = json {{ "model", model }, { "stream", stream }, { "input_prefix", prompt }, { "input_suffix", suffix_text }, { "prompt", "" }};
            if (!images.empty()) {
                // suffix + images (routes.go:578-581 media, 596-598 Prompt + Suffix through the insert template): "[img-N]" tags written in the
                // prompt or the suffix mark where image N goes; /infill gets the images as multimodal_data and builds the FIM prompt as text.
                std::string err_media;
                if (!media_tags_to_marker(images, { &prompt, &suffix_text }, err_media)) {
                    return ollama_error(400, err_media);
                }
                oai_body["input_prefix"]    = prompt;
                oai_body["input_suffix"]    = suffix_text;
                oai_body["multimodal_data"] = images;
            }
        } else {
            if (system.empty() && state_) {
                system = state_->system;
            }
            if (!system.empty()) {
                messages.push_back(json {{ "role", "system" }, { "content", system }});
            }
            if (state_) {
                for (const auto & m : state_->messages) messages.push_back(m);
            }
            if (images.empty()) {
                messages.push_back(json {{ "role", "user" }, { "content", prompt }});
            } else {
                // images on generate (routes.go:578-581 media for the prompt) were dropped here before; reuse the chat converter so
                // both endpoints produce the same data-URI image parts
                messages.push_back(ollama_messages_to_oai(json::array({ json {{ "role", "user" }, { "content", prompt }, { "images", images }} })).at(0));
            }
            oai_body["messages"] = messages;
        }
    } else {
        json chat_msgs = ollama_messages_to_oai(json_value(body, "messages", json::array()));
        if (state_ && (!state_->system.empty() || !state_->messages.empty())) {
            // Modelfile SYSTEM applies when the conversation has no system message of its own; MESSAGE seeds go first
            json seeded = json::array();
            bool has_system = false;
            for (const auto & m : chat_msgs) has_system = has_system || json_value(m, "role", std::string("")) == "system";
            if (!state_->system.empty() && !has_system) seeded.push_back(json {{ "role", "system" }, { "content", state_->system }});
            for (const auto & m : state_->messages) seeded.push_back(m);
            for (const auto & m : chat_msgs) seeded.push_back(m);
            chat_msgs = seeded;
        }
        oai_body["messages"] = chat_msgs;
        // ollama's `tools` array is already OpenAI function-schema-shaped (PARENT REVIEW
        // 2026-09-27, confirmed by reading server-common.cpp:1221-1233 oaicompat_chat_params_parse:
        // it reads `tools` as a plain JSON array with no ollama-specific wrapper), so this passes
        // through unchanged rather than being translated field-by-field. If the server was not
        // started with --jinja, that parser itself throws "tools param requires --jinja flag",
        // which surfaces to the client as a 400 through the existing error-forwarding path below --
        // not a silent drop.
        if (body.contains("tools") && body.at("tools").is_array() && !body.at("tools").empty()) {
            oai_body["tools"] = body.at("tools");
        }
    }

    apply_ollama_options(state_ ? state_->params : json::object(),
                         body.contains("options") ? body.at("options") : json::object(), oai_body);
    {
        // options.num_ctx (row S69-ENGREV-OLLAMA-ROUTER-PER-REQUEST-NUM-CTX-RELOAD-2026-10-27-01): a router reloads the child at that
        // context (proxy_post -> apply_num_ctx), a single-model server cannot resize its context and says so once instead of dropping it.
        const int64_t want_ctx = ollama_requested_num_ctx(state_ ? state_->params : json::object(),
                                                          body.contains("options") ? body.at("options") : json::object());
        if (want_ctx > 0) {
            if (state_ && state_->router_mode) {
                oai_body["ollama_num_ctx"] = want_ctx;
            } else if (state_ && !state_->warned_num_ctx.exchange(true)) {
                const int64_t have = json_value(routes.get_model_info().value("meta", json::object()), "n_ctx", (int64_t) 0);
                if (have != want_ctx) {
                    SRV_WRN("ollama: options.num_ctx=%lld was not applied: this server is not a router, its context is fixed at %lld per slot "
                            "(start it with -c, or run llama-server in router mode to get a per-request reload)\n",
                            (long long) want_ctx, (long long) have);
                } else {
                    state_->warned_num_ctx.store(false); // matches the loaded context, nothing to warn about yet
                }
            }
        }
    }
    if (body.contains("keep_alive") && !body.at("keep_alive").is_null()) {
        // router mode: proxy_post (server-models.cpp) validates and applies it per request, like ollama's keep_alive
        oai_body["keep_alive"] = body.at("keep_alive");
    }
    // raw and infill both answer in the /completion shape (infill is handle_completions_impl with TASK_RESPONSE_TYPE_NONE,
    // server-context.cpp /infill handler), so one flag selects the parser for both
    const bool completion_shape = raw || infill;
    if (!completion_shape) {
        const std::string terr = apply_ollama_think(body, oai_body);
        if (!terr.empty()) {
            return ollama_error(400, terr);
        }
    }
    {
        std::string ferr = apply_ollama_format(body, oai_body, completion_shape);
        if (!ferr.empty()) {
            return ollama_error(400, ferr);
        }
    }
    if (want_logprobs) {
        // at least one alternative: the fork computes per-token probabilities only when n_probs > 0 (server-context.cpp:3494);
        // to_ollama_logprobs trims back to top_logprobs. The OAI chat parser maps top_logprobs to n_probs (server-common.cpp:2241-2245).
        if (completion_shape) {
            oai_body["n_probs"] = std::max(top_logprobs, 1);
        } else {
            oai_body["logprobs"]     = true;
            oai_body["top_logprobs"] = std::max(top_logprobs, 1);
        }
    }

    // SOURCE-READ tools/server/server-stream.h:62-79 + server-context.cpp:5207-5220 (this fork,
    // read in full after a real crash forced the question): server_res_spipe::set_req() stores a
    // RAW POINTER to the handler's `req` parameter (`res->set_req(&req)`), read back later by
    // should_stop()/conn_alive() from the STREAMING response's deferred `next()` lambda -- i.e.
    // AFTER the handler that constructed the request has already returned. For every other route
    // this is safe because httplib's own `req` object lives for the whole connection. Building
    // `inner_req` as a plain local here would make that pointer dangle the moment this function
    // returns, exactly the crash this session hit (0xC0000005 in a second, streaming /api/chat
    // call, isolated with repro_crash.py to be ABSENT when the same two requests go straight to
    // /v1/chat/completions with no shim involved -- i.e. this shim's own bug, not a pre-existing
    // one). Fix: heap-allocate `inner_req` and keep it alive for exactly as long as the streaming
    // response needs it, via `stream_state` below (non-streaming responses just let the shared_ptr
    // go out of scope normally at the end of this function).
    std::shared_ptr<server_http_req> inner_req;
    server_http_res_ptr inner_res;
    auto call_inner = [&]() {
        inner_req = std::shared_ptr<server_http_req>(new server_http_req {
            /* params */       {},
            /* headers */      {},
            /* path */         infill ? "/infill" : (raw ? "/completion" : "/v1/chat/completions"),
            /* query_string */ "",
            /* body */         safe_json_to_str(oai_body),
            /* files */        {},
            /* should_stop */  req.should_stop,
        });
        inner_res = infill ? routes.post_infill(*inner_req) : (raw ? routes.post_completions(*inner_req) : routes.post_chat_completions(*inner_req));
    };
    call_inner();
    if (inner_res == nullptr) {
        return nullptr; // connection closed, matches inner handler's own contract
    }

    // Chat `truncate` (api/types.go:166-168, default true; routes.go:3290; prompt.go:35-74 chatPrompt): when the conversation does not
    // fit, drop messages from the front -- keeping the system messages of the dropped head and always keeping the last message --
    // until it fits. Ollama measures by tokenising the rendered template; here the inner handler measures (it answers
    // exceed_context_size_error, server-context.cpp:7546-7566, before any decode) and the shim retries with a shorter head.
    // Each cut starts the kept part at a user message so strict Jinja templates (alternating roles) still render. truncate:false
    // keeps the error. The cut is reported, never silent: WARN log + "truncated_messages" in the reply (rule 1.13).
    int truncated_messages = 0;
    if (!is_generate && inner_res->status == 400 && is_exceed_context_error(inner_res->data) &&
        !(body.contains("truncate") && body.at("truncate").is_boolean() && !body.at("truncate").get<bool>())) {
        const json full = oai_body.at("messages");
        const size_t last = full.empty() ? 0 : full.size() - 1;
        auto role_of = [&](size_t i) { return json_value(full.at(i), "role", std::string("")); };
        size_t start = 0;
        while (inner_res && inner_res->status == 400 && is_exceed_context_error(inner_res->data) && start < last) {
            ++start;
            while (start < last && role_of(start) != "user") {
                ++start;
            }
            json kept = json::array();
            for (size_t i = 0; i < start; ++i) {
                if (role_of(i) == "system") kept.push_back(full.at(i));
            }
            for (size_t i = start; i < full.size(); ++i) kept.push_back(full.at(i));
            truncated_messages = (int) (full.size() - kept.size());
            oai_body["messages"] = kept;
            call_inner();
        }
        if (inner_res == nullptr) {
            return nullptr;
        }
        if (truncated_messages > 0) {
            SRV_WRN("ollama chat: context exceeded, dropped the %d oldest non-system message(s) of %d (truncate=true)%s\n",
                    truncated_messages, (int) full.size(), inner_res->status == 200 ? "" : "; still too large");
        }
    }

    if (inner_res->status != 200) {
        std::string message = inner_res->status >= 500 ? "model inference failed" : "invalid request";
        try {
            json err = json::parse(inner_res->data);
            if (err.contains("error")) {
                const json & e = err.at("error");
                message = e.is_object() ? json_value(e, "message", message) : (e.is_string() ? e.get<std::string>() : message);
            }
        } catch (const std::exception &) {
            // keep the generic message
        }
        return ollama_error(inner_res->status, message);
    }

    if (!inner_res->is_stream()) {
        json oai_resp;
        try {
            oai_resp = json::parse(inner_res->data);
        } catch (const std::exception & e) {
            return ollama_error(500, std::string("failed to parse inner response: ") + e.what());
        }
        std::string content, thinking, finish;
        json tool_calls = json::array();
        int prompt_tokens = 0, completion_tokens = 0;
        if (completion_shape) {
            // /completion and /infill shape: {content, stop_type, tokens_evaluated, tokens_predicted, timings}
            content = json_value(oai_resp, "content", std::string(""));
            finish  = json_value(oai_resp, "stop_type", std::string("")) == "limit" ? "length" : "stop";
            prompt_tokens     = json_value(oai_resp, "tokens_evaluated", 0);
            completion_tokens = json_value(oai_resp, "tokens_predicted", 0);
        } else if (oai_resp.contains("choices") && oai_resp.at("choices").is_array() && !oai_resp.at("choices").empty()) {
            const json & choice = oai_resp.at("choices").at(0);
            finish = json_value(choice, "finish_reason", std::string(""));
            if (choice.contains("message")) {
                content  = json_value(choice.at("message"), "content", std::string(""));
                thinking = json_value(choice.at("message"), "reasoning_content", std::string(""));
                if (!is_generate && choice.at("message").contains("tool_calls")) {
                    tool_calls = translate_tool_calls(choice.at("message").at("tool_calls"));
                }
            }
        }
        if (!completion_shape && oai_resp.contains("usage")) {
            prompt_tokens     = json_value(oai_resp.at("usage"), "prompt_tokens", 0);
            completion_tokens = json_value(oai_resp.at("usage"), "completion_tokens", 0);
        }
        const int cached_tokens = cached_prompt_tokens(oai_resp);
        // PARENT REVIEW 2026-09-27: fill real sub-phase durations from the pipeline's own
        // `timings` object (server-task.cpp:487-489, SOURCE-READ above: `res["timings"] =
        // stats.to_json()`; fields prompt_ms/predicted_ms, SOURCE-READ server-common.cpp:71-77)
        // instead of leaving them at 0 like LocalAI's reference port does (SOURCE-READ
        // https://raw.githubusercontent.com/mudler/LocalAI/master/core/http/endpoints/ollama/chat.go
        // above: PromptEvalDuration/EvalDuration/LoadDuration are never assigned in that file --
        // TotalDuration/PromptEvalCount/EvalCount are the only fields ever set there). load_duration
        // stays 0: the model is already loaded before any request reaches this handler (no
        // per-request load phase exists in this fork's single-model-per-process design), so 0 is
        // the TRUE value here, not a placeholder.
        int64_t prompt_eval_duration_ns = 0, eval_duration_ns = 0;
        if (oai_resp.contains("timings")) {
            const json & timings = oai_resp.at("timings");
            prompt_eval_duration_ns = (int64_t) (json_value(timings, "prompt_ms", 0.0) * 1.0e6);
            eval_duration_ns        = (int64_t) (json_value(timings, "predicted_ms", 0.0) * 1.0e6);
        }

        json resp = json {
            { "model",       model },
            { "created_at",  iso8601_now() },
            { "done",        true },
            // ollama only ever reports "stop" or "length" here (llm/server.go:260-270); a tool call is a normal stop
            { "done_reason", finish == "length" ? "length" : "stop" },
            { "total_duration", now_ns() - t_start },
            { "load_duration", 0 },
            { "prompt_eval_count", prompt_tokens },
            { "prompt_eval_duration", prompt_eval_duration_ns },
            { "eval_count",        completion_tokens },
            { "eval_duration",     eval_duration_ns },
        };
        if (is_generate) {
            resp["response"] = content;
            if (!thinking.empty()) resp["thinking"] = thinking;
        } else {
            json message = json {{ "role", "assistant" }, { "content", content }};
            if (!thinking.empty()) message["thinking"] = thinking;
            if (!tool_calls.empty()) {
                message["tool_calls"] = tool_calls;
            }
            resp["message"] = message;
        }
        if (cached_tokens >= 0) {
            resp["prompt_eval_cached_count"] = cached_tokens;
        }
        if (want_logprobs) {
            const json lp = to_ollama_logprobs(logprob_entries(oai_resp, completion_shape), top_logprobs);
            if (!lp.empty()) {
                resp["logprobs"] = lp;
            }
        }
        if (truncated_messages > 0) {
            resp["truncated_messages"] = truncated_messages; // fork extension: the chat head the truncate option dropped
        }

        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str(resp);
        return res;
    }

    // streaming: wrap the inner SSE stream and re-emit as ollama NDJSON
    auto res = std::make_unique<server_http_res>();
    res->content_type = "application/x-ndjson";
    res->headers["Cache-Control"] = "no-cache";

    struct tool_call_acc {
        std::string id;
        std::string name;
        std::string arguments; // concatenated across chunks, per OAI streaming convention
    };
    struct stream_state {
        // Keeps the request object `inner->should_stop`/`conn_alive` reads through (via a raw
        // pointer stashed by server_res_spipe::set_req(), see the SOURCE-READ note above) alive for
        // exactly as long as `inner` itself might still be read from -- this field's declaration
        // ORDER matters: members are destroyed in REVERSE declaration order, so `inner` (destroyed
        // first) is gone before `inner_req` (destroyed second), never the other way around.
        std::shared_ptr<server_http_req> inner_req;
        server_http_res_ptr inner;
        std::string model;
        bool is_generate;
        int64_t t0 = now_ns();
        bool raw = false;
        std::string finish;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        double prompt_ms = 0.0;
        double predicted_ms = 0.0;
        std::map<size_t, tool_call_acc> tool_calls; // keyed by OAI's delta.tool_calls[].index
        bool finished = false;
        bool want_logprobs = false;
        int top_logprobs = 0;
        int cached_tokens = -1;
        int truncated_messages = 0;
    };
    auto state = std::make_shared<stream_state>();
    state->inner_req   = inner_req;
    state->inner       = std::move(inner_res);
    state->model       = model;
    state->is_generate = is_generate;
    state->raw         = completion_shape; // selects the /completion-shape event parser (raw and infill)
    state->t0          = t_start;
    state->want_logprobs      = want_logprobs;
    state->top_logprobs       = top_logprobs;
    state->truncated_messages = truncated_messages;

    // first chunk was already produced eagerly into inner->data by the shared streaming setup
    // (server-context.cpp handle_completions_impl always primes the first SSE event into `data`
    // before returning); drain it the same way the inner next() does, through the SAME parser, so
    // this shim has exactly one code path for "a chunk arrived" regardless of which call produced it.
    res->next = [state](std::string & output) -> bool {
        output.clear();

        auto emit_from_text = [&](const std::string & text) -> bool {
            bool saw_done = false;
            for (const auto & ev : parse_sse_events(text, saw_done)) {
                std::string delta, think_delta;
                bool has_usage = false;
                if (state->raw) {
                    // /completion stream: {content, stop?, stop_type?, tokens_evaluated?, tokens_predicted?, timings?}
                    delta = json_value(ev, "content", std::string(""));
                    if (json_value(ev, "stop", false)) {
                        state->finish = json_value(ev, "stop_type", std::string("")) == "limit" ? "length" : "stop";
                        state->prompt_tokens     = json_value(ev, "tokens_evaluated", state->prompt_tokens);
                        state->completion_tokens = json_value(ev, "tokens_predicted", state->completion_tokens);
                        has_usage = true;
                    }
                } else if (ev.contains("choices") && ev.at("choices").is_array() && !ev.at("choices").empty()) {
                    const json & choice = ev.at("choices").at(0);
                    if (choice.contains("delta")) {
                        delta       = json_value(choice.at("delta"), "content", std::string(""));
                        think_delta = json_value(choice.at("delta"), "reasoning_content", std::string(""));
                    }
                    if (choice.contains("finish_reason") && choice.at("finish_reason").is_string()) {
                        state->finish = choice.at("finish_reason").get<std::string>();
                    }
                }
                if (ev.contains("usage") && ev.at("usage").is_object()) {
                    state->prompt_tokens     = json_value(ev.at("usage"), "prompt_tokens", state->prompt_tokens);
                    state->completion_tokens = json_value(ev.at("usage"), "completion_tokens", state->completion_tokens);
                    has_usage = true;
                }
                if (ev.contains("timings") && ev.at("timings").is_object()) {
                    // PARENT REVIEW 2026-09-27: same timings object as the non-stream path
                    // (server-task.cpp:548-550, SOURCE-READ above: `deltas.back()["timings"] =
                    // stats.to_json()` -- attached to whichever SSE event is last in the final
                    // batch), captured here so the final ollama "done" chunk below can report real
                    // prompt_eval_duration/eval_duration instead of 0.
                    const json & timings = ev.at("timings");
                    state->prompt_ms    = json_value(timings, "prompt_ms", state->prompt_ms);
                    state->predicted_ms = json_value(timings, "predicted_ms", state->predicted_ms);
                }
                if (!state->is_generate && ev.contains("choices") && ev.at("choices").is_array() && !ev.at("choices").empty()) {
                    const json & choice = ev.at("choices").at(0);
                    if (choice.contains("delta") && choice.at("delta").contains("tool_calls")) {
                        // Accumulate incremental tool-call fragments (index/id/function.name/
                        // function.arguments partial string chunks, SOURCE-READ
                        // server-chat.cpp:607-635 server_chat_msg_diff_to_json_oaicompat above).
                        // ollama does not stream tool calls line-by-line (PARENT REVIEW
                        // 2026-09-27); they are assembled here and only emitted, complete, in the
                        // final "done" chunk below.
                        for (const auto & tc : choice.at("delta").at("tool_calls")) {
                            size_t idx = json_value(tc, "index", (size_t) 0);
                            auto & acc = state->tool_calls[idx];
                            if (tc.contains("id")) {
                                acc.id = json_value(tc, "id", acc.id);
                            }
                            if (tc.contains("function")) {
                                const json & fn = tc.at("function");
                                if (fn.contains("name")) {
                                    acc.name = json_value(fn, "name", acc.name); // set once, not streamed piecewise
                                }
                                if (fn.contains("arguments")) {
                                    acc.arguments += json_value(fn, "arguments", std::string(""));
                                }
                            }
                        }
                    }
                }
                {
                    const int c = cached_prompt_tokens(ev);
                    if (c >= 0) state->cached_tokens = c;
                }
                // per-token logprobs ride on the event that carries the token (server-task.cpp:1417 completion_probabilities on a
                // /completion partial, :1499 choices[0].logprobs on the last delta of a chat batch); ollama streams them with the
                // chunk, and sends a logprob-only chunk while content is held back (routes.go:3426-3432)
                json lp = json::array();
                if (state->want_logprobs) {
                    lp = to_ollama_logprobs(logprob_entries(ev, state->raw), state->top_logprobs);
                }
                if (!delta.empty() || !think_delta.empty() || !lp.empty()) {
                    if (!delta.empty() || !think_delta.empty()) {
                        state->completion_tokens += has_usage ? 0 : 1; // fallback estimate when no usage block arrives
                    }
                    json chunk = json {
                        { "model", state->model }, { "created_at", iso8601_now() }, { "done", false },
                    };
                    if (state->is_generate) {
                        chunk["response"] = delta;
                        if (!think_delta.empty()) chunk["thinking"] = think_delta;
                    } else {
                        chunk["message"] = json {{ "role", "assistant" }, { "content", delta }};
                        if (!think_delta.empty()) chunk["message"]["thinking"] = think_delta;
                    }
                    if (!lp.empty()) chunk["logprobs"] = lp;
                    output += safe_json_to_str(chunk);
                    output += "\n";
                }
            }
            return saw_done;
        };

        if (!state->inner->data.empty()) {
            std::string first = std::move(state->inner->data);
            state->inner->data.clear();
            emit_from_text(first);
            if (!output.empty()) {
                return true;
            }
            // fall through: the primed first chunk carried no delta (e.g. role-only chunk);
            // keep pulling until we have something or the stream ends
        }

        while (state->inner->next) {
            std::string inner_output;
            bool has_more = state->inner->next(inner_output);
            bool saw_done = emit_from_text(inner_output);
            if (!output.empty()) {
                return true;
            }
            if (!has_more || saw_done) {
                break;
            }
            // no translatable content this tick (e.g. a ping); loop and pull the next one
        }

        if (state->finished) {
            return false;
        }
        state->finished = true;
        int64_t total_ns = now_ns() - state->t0;
        json assembled_tool_calls = json::array();
        for (const auto & kv : state->tool_calls) {
            const tool_call_acc & acc = kv.second;
            json tc = json {
                { "function", json {
                    { "name", acc.name },
                    { "arguments", parse_tool_arguments(acc.arguments) },
                    { "index", (int) assembled_tool_calls.size() },
                }},
            };
            if (!acc.id.empty()) tc["id"] = acc.id;
            assembled_tool_calls.push_back(tc);
        }
        json final_chunk = json {
            { "model", state->model }, { "created_at", iso8601_now() },
            { "done", true },
            { "done_reason", state->finish == "length" ? "length" : "stop" },
            { "total_duration", total_ns },
            { "load_duration", 0 },
            { "prompt_eval_count", state->prompt_tokens },
            { "prompt_eval_duration", (int64_t) (state->prompt_ms * 1.0e6) },
            { "eval_count", state->completion_tokens },
            { "eval_duration", (int64_t) (state->predicted_ms * 1.0e6) },
        };
        if (state->is_generate) {
            final_chunk["response"] = "";
        } else {
            json message = json {{ "role", "assistant" }, { "content", "" }};
            if (!assembled_tool_calls.empty()) {
                message["tool_calls"] = assembled_tool_calls;
            }
            final_chunk["message"] = message;
        }
        if (state->cached_tokens >= 0) {
            final_chunk["prompt_eval_cached_count"] = state->cached_tokens;
        }
        if (state->truncated_messages > 0) {
            final_chunk["truncated_messages"] = state->truncated_messages; // fork extension, see the truncate block above
        }
        output = safe_json_to_str(final_chunk);
        output += "\n";
        return true;
    };

    return res;
}

server_http_res_ptr server_routes_ollama::handle_embed(const server_http_req & req, bool legacy) const {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & e) {
        return ollama_error(400, std::string("invalid JSON body: ") + e.what());
    }

    std::string model = json_value(body, "model", std::string(""));
    if (ollama_name_rejected(state_, model, state_ && state_->strict_names ? routes.get_model_info() : json::object())) {
        return ollama_error(404, "model '" + model + "' not found");
    }
    // ollama's /api/embed accepts both "input" and "prompt" as aliases for the same field
    // (schema.OllamaEmbedRequest, GetInputStrings(), SOURCE-READ above, input takes precedence);
    // our own handler already accepts "input" verbatim.
    json input = body.contains("input") ? body.at("input") : json_value(body, "prompt", json(nullptr));
    if (input.is_null()) {
        return ollama_error(400, "input is required");
    }

    // "truncate" (api/types.go EmbedRequest.Truncate, default true; server/routes.go:1181-1250): an input longer than the context is cut
    // to fit, or refused with 400 "the input length exceeds the context length" when truncate is false. Here the limit is the physical
    // batch (an embedding request cannot be split), capped by the context; BOS/EOS the tokenizer adds are budgeted, not cut.
    {
        const bool truncate = !body.contains("truncate") || body.at("truncate").is_null() || json_value(body, "truncate", true);
        // Router mode (routes.get_model_info() would read the router's null meta, see router_models_list): the tokenizer calls below are
        // proxied to the named model (it must be in the body) and load it, so its context is read from the router's model list AFTER the
        // first tokenize succeeded; an unknown model fails that tokenize and the embeddings call then returns the router's own error.
        const bool router = state_ && state_->router_mode;
        int64_t limit = 0;
        bool limit_known = false;
        auto ensure_limit = [&]() {
            if (limit_known) return;
            limit_known = true;
            const json minfo = router ? router_find_info(routes, model, false) : routes.get_model_info();
            if (!minfo.is_object() || !minfo.contains("meta")) return;
            limit = json_value(minfo.at("meta"), "n_ctx", (int64_t) 0);
            const int64_t ub = json_value(minfo.at("meta"), "n_ubatch", (int64_t) 0);
            if (ub > 0 && (limit <= 0 || ub < limit)) limit = ub;
        };
        if (!router) ensure_limit();
        auto tokenize = [&](const std::string & text, bool special, std::vector<int64_t> & toks) {
            json tb {{ "content", text }, { "add_special", special }};
            if (router) tb["model"] = model;
            server_http_req tr { {}, {}, "/tokenize", "", safe_json_to_str(tb), {}, req.should_stop };
            server_http_res_ptr r;
            try { r = routes.post_tokenize(tr); ollama_drain_body(r); } catch (const std::exception &) { return false; }
            if (!r || r->status != 200) return false;
            try { const json parsed = json::parse(r->data); const json & arr = parsed.at("tokens"); for (size_t i = 0; i < arr.size(); ++i) toks.push_back(arr[i].get<int64_t>()); } catch (const std::exception &) { return false; }
            return true;
        };
        auto fit = [&](const std::string & text, std::string & out, std::string & err) {
            out = text;
            std::vector<int64_t> full;
            const bool tok_ok = tokenize(text, true, full);
            if (tok_ok) ensure_limit();
            if (limit <= 0 || !tok_ok || (int64_t) full.size() <= limit) return true;
            if (!truncate) { err = "the input length exceeds the context length"; return false; }
            std::vector<int64_t> bare;
            if (!tokenize(text, false, bare)) return true;
            int64_t budget = limit - ((int64_t) full.size() - (int64_t) bare.size()); // specials the tokenizer adds
            if (budget <= 0) { err = "input after truncation exceeds maximum context length"; return false; }
            // detokenize + retokenize is not an identity (a SPM tokenizer adds a space prefix back), so verify the fit and shave the overshoot
            for (int attempt = 0; attempt < 8; ++attempt) {
                std::vector<int64_t> cut_toks(bare.begin(), bare.begin() + std::min<size_t>((size_t) budget, bare.size()));
                json toks = json::array();
                for (auto t : cut_toks) toks.push_back(t);
                json db {{ "tokens", toks }};
                if (router) db["model"] = model;
                server_http_req dr { {}, {}, "/detokenize", "", safe_json_to_str(db), {}, req.should_stop };
                server_http_res_ptr r;
                try { r = routes.post_detokenize(dr); ollama_drain_body(r); } catch (const std::exception &) { return true; }
                if (!r || r->status != 200) return true;
                try { const json parsed = json::parse(r->data); out = parsed.at("content").get<std::string>(); } catch (const std::exception &) { return true; }
                std::vector<int64_t> again;
                if (!tokenize(out, true, again) || (int64_t) again.size() <= limit) break;
                budget -= (int64_t) again.size() - limit;
                if (budget <= 0) { err = "input after truncation exceeds maximum context length"; return false; }
            }
            SRV_INF("ollama embed: input of %d tokens truncated to %d\n", (int) full.size(), (int) budget);
            return true;
        };
        std::string err, cut;
        if (input.is_string()) {
            if (!fit(input.get<std::string>(), cut, err)) return ollama_error(400, err);
            input = cut;
        } else if (input.is_array()) {
            for (size_t i = 0; i < input.size(); ++i) {
                if (!input[i].is_string()) continue; // token-id arrays pass through untouched
                if (!fit(input[i].get<std::string>(), cut, err)) return ollama_error(400, err);
                input[i] = cut;
            }
        }
    }

    json oai_body = json {{ "model", model }, { "input", input }};
    // "dimensions" (Matryoshka truncation, row S69-SERVING-SURFACE-EMBED-DIMENSIONS-2026-09-27-01)
    // is documented identically by ollama's /api/embed and our own OAI-compatible endpoint, so it
    // passes straight through with no translation -- both wire protocols spell it "dimensions".
    if (body.contains("dimensions")) {
        oai_body["dimensions"] = body.at("dimensions");
    }
    if (state_ && state_->router_mode) {
        // options.num_ctx on /api/embed, same router reload as /api/chat (proxy_post -> apply_num_ctx); ollama honours it on embed too
        const int64_t want_ctx = ollama_requested_num_ctx(state_->params, body.contains("options") ? body.at("options") : json::object());
        if (want_ctx > 0) {
            oai_body["ollama_num_ctx"] = want_ctx;
        }
    }

    int64_t t0 = now_ns();
    server_http_req inner_req {
        {}, {}, "/v1/embeddings", "", safe_json_to_str(oai_body), {}, req.should_stop,
    };
    server_http_res_ptr inner_res = routes.post_embeddings_oai(inner_req);
    ollama_drain_body(inner_res); // router: a proxied answer arrives through next(), not in data
    if (inner_res == nullptr) {
        return nullptr;
    }
    if (inner_res->status != 200) {
        std::string message = "embedding failed";
        try {
            json err = json::parse(inner_res->data);
            if (err.contains("error")) {
                const json & e = err.at("error");
                message = e.is_object() ? json_value(e, "message", message) : message;
            }
        } catch (const std::exception &) {}
        return ollama_error(inner_res->status, message);
    }

    json oai_resp;
    try {
        oai_resp = json::parse(inner_res->data);
    } catch (const std::exception & e) {
        return ollama_error(500, std::string("failed to parse inner response: ") + e.what());
    }

    json embeddings = json::array();
    if (oai_resp.contains("data") && oai_resp.at("data").is_array()) {
        for (const auto & item : oai_resp.at("data")) {
            embeddings.push_back(json_value(item, "embedding", json::array()));
        }
    }
    int prompt_tokens = 0;
    if (oai_resp.contains("usage")) {
        prompt_tokens = json_value(oai_resp.at("usage"), "prompt_tokens", 0);
    }

    json resp;
    if (legacy) {
        // older /api/embeddings: exactly one flat vector under "embedding" (RESEARCH-DONE: ollama docs/api.md
        // "Generate Embedding", fetched 2026-09-27). A client written for it reads resp["embedding"][i]; the
        // /api/embed shape made that a KeyError. It takes a single "prompt" string, so an array input gets a
        // clear 400 rather than an answer holding only its first vector.
        if (embeddings.size() != 1) {
            return ollama_error(400, "/api/embeddings takes a single \"prompt\" string; use /api/embed for a batch");
        }
        resp = json {{ "embedding", embeddings[0] }};
    } else {
        resp = json {
            { "model", model },
            { "embeddings", embeddings },
            { "total_duration", now_ns() - t0 },
            { "load_duration", 0 },
            { "prompt_eval_count", prompt_tokens },
        };
    }
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(resp);
    return res;
}

// Builds one ollama model-list-style entry from this (single-model) server's own metadata, via the
// PUBLIC server_routes::get_model_info() accessor (server-context.cpp:5585, SOURCE-READ above) --
// no private state is touched. "family"/architecture is not surfaced by that accessor today (out
// of scope to extend it for this shim; the meta struct's `model_ftype` string is the closest
// available field and is used for quantization_level, matching ollama's own field intent).
static json ollama_model_entry(const json & model_info) {
    std::string name = with_latest_tag(json_value(model_info, "id", std::string("model")));
    uint64_t size = 0;
    std::string ftype, arch;
    if (model_info.contains("meta")) {
        const json & meta = model_info.at("meta");
        size  = json_value(meta, "size", (uint64_t) 0);
        ftype = json_value(meta, "ftype", std::string(""));
        arch  = json_value(meta, "arch", std::string(""));
    }
    // PARENT REVIEW 2026-09-27: family/architecture is now real (llama_model_meta_val_str via the
    // new server_context_meta.model_arch field, server-context.cpp:5127-5134, SOURCE-READ above),
    // not the empty string this file shipped with initially.
    json families = arch.empty() ? json::array() : json::array({ arch });
    return json {
        { "name", name },
        { "model", name },
        { "modified_at", iso8601_now() },
        { "size", size },
        { "digest", "sha256:" + sha256_hex(name) },
        { "details", json {
            { "format", "gguf" },
            { "family", arch },
            { "families", families },
            { "parameter_size", "" },
            { "quantization_level", ftype },
        }},
    };
}

server_http_res_ptr server_routes_ollama::handle_tags(const server_http_req &) const {
    // router: every model the router holds (as `ollama list` shows what is installed), loaded or not; single-model: the one loaded model
    json resp = json {{ "models", json::array() }};
    std::set<std::string> held;
    if (state_ && state_->router_mode) {
        for (const auto & e : router_models_list(routes, false)) {
            const json info = router_entry_to_info(e);
            resp["models"].push_back(ollama_model_entry(info));
            held.insert(ollama_util::canonical_key(json_value(info, "id", std::string(""))));
        }
    } else {
        const json info = routes.get_model_info();
        resp["models"].push_back(ollama_model_entry(info));
        held.insert(ollama_util::canonical_key(json_value(info, "id", std::string(""))));
    }
    if (state_ && !state_->store_dir.empty()) {
        // models held by the Ollama store, as `ollama list` shows them (server/routes.go ListHandler): name, size = sum of layer sizes,
        // digest = sha256 of the manifest file, modified_at = the manifest's mtime. The loaded model keeps its richer entry.
        for (const auto & nm : ollama_util::store_list(state_->store_dir)) {
            if (held.count(ollama_util::canonical_key(nm))) continue;
            ollama_util::store_model sm;
            std::string err;
            uint64_t size = 0;
            std::string digest, mtime = iso8601_now();
            if (ollama_util::store_resolve(state_->store_dir, nm, sm, err)) {
                size = (uint64_t) sm.total_size;
                std::string text;
                if (ollama_util::store_read_file(sm.manifest_path, text)) digest = "sha256:" + sha256_hex(text);
                std::error_code ec;
                auto ft = std::filesystem::last_write_time(sm.manifest_path, ec);
                if (!ec) {
                    auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(ft - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                    mtime = iso8601_utc(std::chrono::system_clock::to_time_t(sys));
                }
            }
            resp["models"].push_back(json {
                { "name", nm }, { "model", nm }, { "modified_at", mtime }, { "size", size }, { "digest", digest },
                { "details", json {{ "format", "gguf" }, { "family", "" }, { "families", json::array() }, { "parameter_size", "" }, { "quantization_level", "" }}},
            });
        }
    }
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(resp);
    return res;
}

server_http_res_ptr server_routes_ollama::handle_show(const server_http_req & req) const {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        body = json::object();
    }
    std::string name = json_value(body, "name", json_value(body, "model", std::string("")));
    // single-model server: any non-empty name resolves to the one loaded model, same as ollama's
    // single-alias behavior when only one model is configured; an empty request name is a 400.
    if (name.empty()) {
        return ollama_error(400, "name is required");
    }
    const bool router = state_ && state_->router_mode;
    // router: the named model's entry in the router's list (routes.get_model_info() would read the router's null meta); an unknown name is
    // ollama's 404, and a model that is not loaded is described from its GGUF header without loading it
    json model_info = router ? router_find_info(routes, name, /*full_info=*/true) : routes.get_model_info();
    if (router && !model_info.is_object()) {
        return ollama_error(404, "model '" + name + "' not found");
    }
    std::string our_name = json_value(model_info, "id", std::string(""));
    (void) strip_tag(name); // no per-tag config to look up against; documented, see handle_tags
    if (ollama_name_rejected(state_, name, model_info)) {
        return ollama_error(404, "model '" + name + "' not found");
    }

    // /props carries the chat template, template capabilities and modalities (same in-process call as handle_version)
    json props = json::object();
    if (!router) {
        server_http_req inner_req { {}, {}, "/props", "", "{}", {}, [](){ return false; } };
        server_http_res_ptr pr = routes.get_props(inner_req);
        if (pr != nullptr && pr->status == 200) {
            try { props = json::parse(pr->data); } catch (const std::exception &) {}
        }
    } else {
        // the router's own /props is the router's (no template). A loaded child's /props is proxied (autoload=false: show never loads a
        // model); an unloaded model has no capability probe, its template text comes from the GGUF header (info=full) and capabilities
        // stay at "completion"
        if (json_value(model_info, "loaded", false)) {
            try {
                server_http_req inner_req { std::map<std::string, std::string> {{ "model", our_name }, { "autoload", "false" }}, {}, "/props", "", "", {}, ollama_never_stop };
                server_http_res_ptr pr = routes.get_props(inner_req);
                ollama_drain_body(pr);
                if (pr != nullptr && pr->status == 200) props = json::parse(pr->data);
            } catch (const std::exception &) {}
        }
        if (!props.contains("chat_template") && model_info.contains("gguf") && model_info.at("gguf").contains("chat_template")) {
            props["chat_template"] = model_info.at("gguf").at("chat_template");
        }
    }

    uint64_t n_ctx_train = 0, n_params = 0, n_embd = 0;
    std::string ftype, arch;
    if (model_info.contains("meta")) {
        const json & meta = model_info.at("meta");
        n_ctx_train = json_value(meta, "n_ctx_train", (uint64_t) 0);
        n_params    = json_value(meta, "n_params", (uint64_t) 0);
        n_embd      = json_value(meta, "n_embd", (uint64_t) 0);
        ftype       = json_value(meta, "ftype", std::string(""));
        arch        = json_value(meta, "arch", std::string(""));
    }
    json families = arch.empty() ? json::array() : json::array({ arch });

    // clients (ollama-python, Open WebUI, Continue) gate features on this list (api/types.go:747-784)
    json caps = json::array({ "completion" });
    const json tcaps = props.contains("chat_template_caps") ? props.at("chat_template_caps") : json::object();
    if (json_value(tcaps, "supports_tools", false)) caps.push_back("tools");
    if (props.contains("modalities") && json_value(props.at("modalities"), "vision", false)) caps.push_back("vision");
    if (json_value(tcaps, "supports_reasoning_effort", false) || json_value(tcaps, "supports_preserve_reasoning", false)) caps.push_back("thinking");

    // "7.2B" style, as ollama prints it (format.HumanNumber)
    std::string psize;
    if (n_params) {
        char buf[32];
        double v = (double) n_params;
        if      (v >= 1e12) std::snprintf(buf, sizeof buf, "%.1fT", v / 1e12);
        else if (v >= 1e9)  std::snprintf(buf, sizeof buf, "%.1fB", v / 1e9);
        else if (v >= 1e6)  std::snprintf(buf, sizeof buf, "%.0fM", v / 1e6);
        else                std::snprintf(buf, sizeof buf, "%.0fK", v / 1e3);
        psize = buf;
    }

    std::string params_text;
    if (state_) {
        for (auto it = state_->params.begin(); it != state_->params.end(); ++it) {
            if (it.value().is_array()) {
                for (const auto & v : it.value()) params_text += it.key() + "    " + v.get<std::string>() + "\n";
            } else {
                params_text += it.key() + "    " + it.value().dump() + "\n";
            }
        }
    }
    json model_info_out = json {
        { "general.architecture", arch },
        { "general.parameter_count", n_params },
        { "general.context_length", n_ctx_train },
    };
    if (!arch.empty()) {
        model_info_out[arch + ".context_length"]   = n_ctx_train;
        model_info_out[arch + ".embedding_length"] = n_embd;
    }
    json resp = json {
        { "modelfile", state_ && !state_->modelfile_text.empty() ? state_->modelfile_text : "FROM " + our_name },
        { "parameters", params_text },
        { "template", state_ && !state_->template_text.empty() ? state_->template_text : json_value(props, "chat_template", std::string("")) },
        { "details", json {
            { "parent_model", "" },
            { "format", "gguf" },
            { "family", arch },
            { "families", families },
            { "parameter_size", psize },
            { "quantization_level", ftype },
        }},
        { "model_info", model_info_out },
        { "capabilities", caps },
    };
    if (state_ && !state_->system.empty()) resp["system"] = state_->system;
    if (state_ && !state_->license.empty()) {
        std::string lic;
        for (const auto & l : state_->license) lic += (lic.empty() ? "" : "\n") + l.get<std::string>();
        resp["license"] = lic;
    }
    if (state_ && !state_->messages.empty()) resp["messages"] = state_->messages;
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(resp);
    return res;
}

server_http_res_ptr server_routes_ollama::handle_ps(const server_http_req &) const {
    // single-model, non-router server: the loaded model is always "running" for as long as the
    // process is; there is no idle-unload timer to report an accurate expires_at for, so it is set
    // far in the future rather than asserting a specific TTL this server does not implement.
    // router: the models whose child is running (the router's list; routes.get_model_info() would read the router's null meta);
    // single-model: the one loaded model
    std::vector<json> infos;
    if (state_ && state_->router_mode) {
        for (const auto & e : router_models_list(routes, false)) {
            json info = router_entry_to_info(e);
            if (json_value(info, "loaded", false)) infos.push_back(std::move(info));
        }
    } else if (!routes.model_is_sleeping()) { // a model put to sleep (keep_alive 0 with --sleep-idle-seconds) is not loaded
        infos.push_back(routes.get_model_info());
    }
    json resp = json {{ "models", json::array() }};
    for (const auto & minfo : infos) {
        json m = ollama_model_entry(minfo);
        // size = every byte the loaded model holds (weights + KV cache + compute buffers), size_vram = the part in device buffers, both
        // from the backend's own allocation record (server/routes.go:2712-2760 SizeVRAM / Size; sched.go loadedModels)
        const json meta = minfo.contains("meta") ? minfo.at("meta") : json::object();
        const uint64_t host = json_value(meta, "mem_host_bytes", (uint64_t) 0), dev = json_value(meta, "mem_device_bytes", (uint64_t) 0);
        if (host + dev > 0) {
            m["size"]      = host + dev;
            m["size_vram"] = dev;
        }
        m["expires_at"] = "2099-01-01T00:00:00Z";
        m["context_length"] = json_value(meta, "n_ctx", (uint64_t) 0);
        m.erase("modified_at");
        resp["models"].push_back(std::move(m));
    }
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(resp);
    return res;
}

// POST /api/copy {source, destination}: a new manifest name for the same blobs (server/routes.go:2064-2112: 400 for an invalid
// name, 404 for a missing source, {"status":"success"}). Gated by --ollama-store-write: it edits the user's store.
server_http_res_ptr server_routes_ollama::handle_copy(const server_http_req & req) const {
    if (!state_ || state_->store_dir.empty()) return ollama_error(404, "no Ollama model store configured (--ollama-models-dir)");
    if (!state_->store_write)                 return ollama_error(403, "the Ollama model store is read-only; start with --ollama-store-write to allow /api/copy");
    json body;
    try { body = json::parse(req.body); } catch (const std::exception &) { return ollama_error(400, "missing request body"); }
    const std::string src = json_value(body, "source", std::string("")), dst = json_value(body, "destination", std::string(""));
    if (!ollama_util::valid(ollama_util::parse_name(src))) return ollama_error(400, "source \"" + src + "\" is invalid");
    if (!ollama_util::valid(ollama_util::parse_name(dst))) return ollama_error(400, "destination \"" + dst + "\" is invalid");
    std::string err;
    if (!ollama_util::store_copy(state_->store_dir, src, dst, err)) {
        return ollama_error(err.find("not found") != std::string::npos ? 404 : 500, err.find("not found") != std::string::npos ? "model \"" + src + "\" not found" : err);
    }
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(json {{ "status", "success" }});
    return res;
}

// DELETE /api/delete {model}: remove the manifest, then every blob nothing else references (server/routes.go:1629-1669:
// 400 invalid name, 404 not found, 200 empty body). Gated by --ollama-store-write.
server_http_res_ptr server_routes_ollama::handle_delete(const server_http_req & req) const {
    if (!state_ || state_->store_dir.empty()) return ollama_error(404, "no Ollama model store configured (--ollama-models-dir)");
    if (!state_->store_write)                 return ollama_error(403, "the Ollama model store is read-only; start with --ollama-store-write to allow /api/delete");
    json body;
    try { body = json::parse(req.body); } catch (const std::exception &) { return ollama_error(400, "missing request body"); }
    const std::string name = json_value(body, "model", json_value(body, "name", std::string("")));
    if (!ollama_util::valid(ollama_util::parse_name(name))) return ollama_error(400, "name \"" + name + "\" is invalid");
    std::vector<std::string> removed;
    std::string err;
    if (!ollama_util::store_remove(state_->store_dir, name, removed, err)) return ollama_error(404, "model '" + name + "' not found");
    SRV_INF("ollama store: deleted %s (%d unreferenced blobs removed)\n", name.c_str(), (int) removed.size());
    auto res = std::make_unique<server_http_res>();
    res->data = "";
    return res;
}

server_http_res_ptr server_routes_ollama::handle_version(const server_http_req &) const {
    // ollama clients generally just display this string; this fork's own build identifier
    // (llama_build_info(), surfaced publicly via /props's "build_info") is not a semver string but
    // is a real, truthful identifier of what is actually running -- preferred over inventing a fake
    // ollama version number this server does not correspond to.
    json body_req = json::object();
    server_http_req inner_req { {}, {}, "/props", "", safe_json_to_str(body_req), {}, [](){ return false; } };
    std::string version = "unknown";
    server_http_res_ptr props_res = routes.get_props(inner_req);
    if (props_res != nullptr && props_res->status == 200) {
        try {
            json props = json::parse(props_res->data);
            version = json_value(props, "build_info", version);
        } catch (const std::exception &) {}
    }
    json resp = json {{ "version", version }};
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(resp);
    return res;
}
