// TGI native API compatibility routes. Spec, sources and nuances: server-tgi.h.
// Row S69-TGI-PICK-NATIVE-GENERATE-ROUTES-2026-10-18-01.

#include "server-tgi.h"
#include "server-context.h"
#include "server-common.h"
#include "json.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int k_tgi_max_stop_sequences = 4; // launcher/src/main.rs:883 default
constexpr int k_tgi_max_top_n_tokens   = 5; // launcher/src/main.rs:671 default (max_top_n_tokens)

server_http_res_ptr tgi_error(int status, const std::string & message, const std::string & type) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data   = safe_json_to_str(json {{ "error", message }, { "error_type", type }});
    return res;
}

std::string tgi_sse(const json & ev) {
    return "data:" + safe_json_to_str(ev) + "\n\n";
}

// A stream answers errors the way TGI does: HTTP 200 and one SSE event carrying the error body
// (server.rs:2428-2437, generate_stream_internal yields Err as an event).
server_http_res_ptr tgi_stream_error(const std::string & message, const std::string & type) {
    auto res = std::make_unique<server_http_res>();
    res->content_type = "text/event-stream";
    auto sent = std::make_shared<bool>(false);
    const std::string payload = tgi_sse(json {{ "error", message }, { "error_type", type }});
    res->next = [sent, payload](std::string & out) -> bool {
        out.clear();
        if (*sent) {
            return false;
        }
        *sent = true;
        out = payload;
        return true;
    };
    return res;
}

// Typed optional reads. TGI's serde refuses a wrongly typed field with 422; so does this layer.
bool opt_number(const json & p, const char * key, bool & has, double & v, std::string & err) {
    has = false;
    if (!p.contains(key) || p.at(key).is_null()) {
        return true;
    }
    if (!p.at(key).is_number()) {
        err = std::string("`") + key + "` must be a number";
        return false;
    }
    has = true;
    v   = p.at(key).get<double>();
    return true;
}

bool opt_bool(const json & p, const char * key, bool def, bool & v, std::string & err) {
    v = def;
    if (!p.contains(key) || p.at(key).is_null()) {
        return true;
    }
    if (!p.at(key).is_boolean()) {
        err = std::string("`") + key + "` must be a boolean";
        return false;
    }
    v = p.at(key).get<bool>();
    return true;
}

// parses "data: {...}\n\n" blocks of the inner /completion stream
std::vector<json> parse_events(const std::string & text) {
    std::vector<json> out;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("data:", 0) != 0) {
            continue;
        }
        std::string d = line.substr(5);
        if (!d.empty() && d.front() == ' ') {
            d.erase(0, 1);
        }
        if (d.empty() || d == "[DONE]") {
            continue;
        }
        json ev = json::parse_no_throw(d);
        if (!ev.is_discarded()) {
            out.push_back(ev);
        }
    }
    return out;
}

json tgi_token(const json & prob, bool special) {
    return json {
        { "id",      json_value(prob, "id", 0) },
        { "text",    json_value(prob, "token", std::string("")) },
        { "logprob", json_value(prob, "logprob", 0.0) },
        { "special", special },
    };
}

json tgi_top_tokens(const json & prob, int n) {
    json out = json::array();
    if (n <= 0 || !prob.contains("top_logprobs") || !prob.at("top_logprobs").is_array()) {
        return out;
    }
    const json & tl = prob.at("top_logprobs");
    for (size_t i = 0; i < tl.size() && (int) i < n; ++i) {
        out.push_back(tgi_token(tl.at(i), false));
    }
    return out;
}

int inner_status_to_tgi(int status, std::string & type) {
    if (status == 503) {
        type = "overloaded";
        return 429;
    }
    if (status >= 500) {
        type = "generation";
        return 424;
    }
    type = "validation";
    return 422;
}

std::string inner_error_message(const std::string & data) {
    json err = json::parse_no_throw(data);
    if (!err.is_discarded() && err.contains("error")) {
        const json & e = err.at("error");
        if (e.is_object()) {
            return json_value(e, "message", std::string("invalid request"));
        }
        if (e.is_string()) {
            return e.get<std::string>();
        }
    }
    return "invalid request";
}

} // namespace

std::string server_tgi_finish_reason(const std::string & stop_type) {
    if (stop_type == "eos") {
        return "eos_token";
    }
    if (stop_type == "word") {
        return "stop_sequence";
    }
    return "length"; // limit, and the server's own limits (deadline, repetition) end the text by length
}

std::string server_tgi_translate(const json & body, bool compat_route, bool stream_route, int n_ctx,
                                 int max_best_of, server_tgi_request & out) {
    std::string err;
    if (!body.is_object()) {
        return "request body must be a JSON object";
    }
    if (!body.contains("inputs") || !body.at("inputs").is_string()) {
        return "`inputs` must be a string";
    }
    out.inputs = body.at("inputs").get<std::string>();
    if (out.inputs.empty()) {
        return "`inputs` cannot be empty";
    }
    const bool has_params = body.contains("parameters") && !body.at("parameters").is_null();
    const json p = has_params ? body.at("parameters") : json::object();
    if (!p.is_object()) {
        return "`parameters` must be an object";
    }

    out.stream = stream_route;
    if (compat_route && !opt_bool(body, "stream", false, out.stream, err)) {
        return err;
    }

    // lib.rs default_parameters(): do_sample is TRUE when "parameters" is absent; serde gives FALSE when present.
    bool do_sample = !has_params;
    if (!opt_bool(p, "do_sample", do_sample, do_sample, err)) return err;

    double best_of = 1, temperature = 1, repetition_penalty = 1, frequency_penalty = 0, top_k = 0, top_p = 1,
           typical_p = 1, max_new_tokens = -1, seed = 0, top_n_tokens = 0, truncate = 0;
    bool h_best_of, h_temp, h_rep, h_freq, h_top_k, h_top_p, h_typ, h_max, h_seed, h_topn, h_trunc;
    if (!opt_number(p, "best_of", h_best_of, best_of, err))                       return err;
    if (!opt_number(p, "temperature", h_temp, temperature, err))                  return err;
    if (!opt_number(p, "repetition_penalty", h_rep, repetition_penalty, err))     return err;
    if (!opt_number(p, "frequency_penalty", h_freq, frequency_penalty, err))      return err;
    if (!opt_number(p, "top_k", h_top_k, top_k, err))                             return err;
    if (!opt_number(p, "top_p", h_top_p, top_p, err))                             return err;
    if (!opt_number(p, "typical_p", h_typ, typical_p, err))                       return err;
    if (!opt_number(p, "max_new_tokens", h_max, max_new_tokens, err))             return err;
    if (!opt_number(p, "seed", h_seed, seed, err))                                return err;
    if (!opt_number(p, "top_n_tokens", h_topn, top_n_tokens, err))                return err;
    if (!opt_number(p, "truncate", h_trunc, truncate, err))                       return err;
    bool return_full_text = compat_route; // server.rs:134-136 (+ :1800-1806: text-generation or unknown tag -> true)
    if (!opt_bool(p, "return_full_text", return_full_text, return_full_text, err)) return err;
    bool details = false, decoder_input_details = false, watermark = false;
    if (!opt_bool(p, "details", false, details, err))                             return err;
    if (!opt_bool(p, "decoder_input_details", false, decoder_input_details, err)) return err;
    if (!opt_bool(p, "watermark", false, watermark, err))                         return err;

    // validation.rs:229-330, in TGI's order, with its messages (:963-1029)
    if (h_best_of && best_of != 1 && out.stream) {
        return "`best_of` != 1 is not supported when streaming tokens"; // server.rs:543-548
    }
    if (h_best_of && (best_of < 1 || best_of != (double) (long long) best_of)) {
        return "`best_of` must be > 0 and <= " + std::to_string(max_best_of) + ". Given: " +
               std::to_string((long long) best_of);
    }
    const int n_best_of = h_best_of ? (int) best_of : 1;
    const bool sampling = do_sample || h_temp || h_top_k || h_top_p || h_typ;
    if (n_best_of > 1 && !sampling) {
        return "you must use sampling when `best_of` is > 1";
    }
    if (h_temp && temperature <= 0) {
        return "`temperature` must be strictly positive";
    }
    if (h_rep && repetition_penalty <= 0) {
        return "`repetition_penalty` must be strictly positive";
    }
    if (h_freq && (frequency_penalty < -2.0 || frequency_penalty > 2.0)) {
        return "`frequency_penalty` must be >= -2.0 and <= 2.0";
    }
    if (h_freq && frequency_penalty != 0.0) {
        return "`frequency_penalty` != 0 is not supported by this server: TGI's frequency penalty is length-normalised "
               "(count / sequence length), a different quantity from this server's count-based frequency_penalty "
               "(use /completion for that)";
    }
    if (h_top_p && (top_p <= 0 || top_p >= 1)) {
        return "`top_p` must be > 0.0 and < 1.0";
    }
    if (h_typ && (typical_p <= 0 || typical_p >= 1)) {
        return "`typical_p` must be > 0.0 and < 1.0";
    }
    if (h_top_k && top_k <= 0) {
        return "`top_k` must be strictly positive";
    }
    if (h_max && max_new_tokens <= 0) {
        return "`max_new_tokens` must be strictly positive";
    }
    json stop = json::array();
    if (p.contains("stop") && !p.at("stop").is_null()) {
        if (!p.at("stop").is_array()) {
            return "`stop` must be an array of strings";
        }
        for (size_t i = 0; i < p.at("stop").size(); ++i) {
            if (!p.at("stop").at(i).is_string()) {
                return "`stop` must be an array of strings";
            }
            stop.push_back(p.at("stop").at(i));
        }
        if ((int) stop.size() > k_tgi_max_stop_sequences) {
            return "`stop` supports up to " + std::to_string(k_tgi_max_stop_sequences) + " stop sequences. Given: " +
                   std::to_string(stop.size());
        }
    }
    if (h_topn && (top_n_tokens < 0 || top_n_tokens > k_tgi_max_top_n_tokens)) {
        return "`top_n_tokens` must be >= 0 and <= " + std::to_string(k_tgi_max_top_n_tokens) + ". Given: " +
               std::to_string((long long) top_n_tokens);
    }
    if (h_trunc && (truncate <= 0 || truncate > n_ctx)) {
        return "`truncate` must be strictly positive and less than " + std::to_string(n_ctx) + ". Given: " +
               std::to_string((long long) truncate);
    }
    if (decoder_input_details && out.stream) {
        return "`decoder_input_details` == true is not supported when streaming tokens";
    }
    if (n_best_of > 1 && h_seed) {
        return "`seed` must not be set when `best_of` > 1";
    }
    // infer/mod.rs:327 validate_best_of runs after request validation
    if (n_best_of > 1 && max_best_of <= 1) {
        return "`best_of` != 1 is not allowed for this endpoint";
    }
    if (n_best_of > max_best_of) {
        return "`best_of` must be > 0 and <= " + std::to_string(max_best_of) + ". Given: " + std::to_string(n_best_of);
    }
    // adapter_id (validation.rs:225, lib.rs:433; LoRAX names an adapter the same way): resolved to this server's per-request
    // LoRA fields by handle(), which can see the loaded adapters. Row S69-ENGREV-TGI-ADAPTER-ID-TO-LORA-2026-10-27-01.
    if (p.contains("adapter_id") && !p.at("adapter_id").is_null()) {
        if (!p.at("adapter_id").is_string()) {
            return "`adapter_id` must be a string";
        }
        out.adapter_id = p.at("adapter_id").get<std::string>(); // "" = the base model, as in LoRAX
    }

    json c = json::object();
    c["prompt"]    = out.inputs;
    c["stream"]    = out.stream;
    c["n_predict"] = h_max ? (int) max_new_tokens : -1;
    // TGI applies only what it is asked for (validation.rs: top_k 0, top_p 1, typical_p 1 when absent)
    c["top_k"]     = h_top_k ? (int) top_k : 0;
    c["top_p"]     = h_top_p ? top_p : 1.0;
    c["typical_p"] = h_typ ? typical_p : 1.0;
    c["min_p"]     = 0.0;
    c["mirostat"]  = 0;
    c["dry_multiplier"]  = 0.0;
    c["xtc_probability"] = 0.0;
    c["frequency_penalty"] = 0.0;
    c["presence_penalty"]  = 0.0;
    c["repeat_penalty"]    = h_rep ? repetition_penalty : 1.0;
    c["repeat_last_n"]     = n_ctx > 0 ? n_ctx : 0; // HF repetition penalty covers the whole sequence
    c["temperature"]       = sampling ? (h_temp ? temperature : 1.0) : 0.0; // 0 = greedy here
    if (sampling) {
        // TGI draws a random seed when none is given and reports it (validation.rs:294-302); do the same so
        // details.seed is the seed actually used.
        uint32_t s = 0;
        if (h_seed) {
            s = (uint32_t) (uint64_t) seed;
        } else {
            std::random_device rd;
            s = rd();
        }
        c["seed"] = s;
    }
    if (!stop.empty()) {
        c["stop"] = stop;
    }
    out.top_n_tokens = (int) top_n_tokens;
    out.details      = details || decoder_input_details; // server.rs:310
    out.decoder_input_details = decoder_input_details;
    out.best_of      = n_best_of;
    // token id/text/logprob come from the probabilities the server reports; a stream always carries them, and
    // best_of ranks by them
    const int n_probs = std::max(out.top_n_tokens, (out.stream || out.details || n_best_of > 1) ? 1 : 0);
    if (n_probs > 0) {
        c["n_probs"] = n_probs;
    }
    if (n_best_of > 1) {
        c["n_cmpl"] = n_best_of;
    } else if (decoder_input_details) {
        c["prompt_logprobs"] = true; // with n_cmpl > 1 it is refused, so handle() scores the prompt separately
    }
    if (watermark) {
        c["watermark"]        = true;
        c["watermark_scheme"] = "kgw"; // TGI's watermark is Kirchenbauer et al. (lib.rs:409-411)
    }
    if (p.contains("grammar") && !p.at("grammar").is_null()) {
        const json & g = p.at("grammar");
        const std::string type = json_value(g, "type", std::string(""));
        if (!g.is_object() || !g.contains("value")) {
            return "grammar is not valid: expected {\"type\": ..., \"value\": ...}";
        }
        if (type == "json" || type == "json_object") {
            c["json_schema"] = g.at("value");
        } else if (type == "json_schema") {
            if (!g.at("value").is_object() || !g.at("value").contains("schema")) {
                return "grammar is not valid: json_schema needs a \"schema\"";
            }
            c["json_schema"] = g.at("value").at("schema");
        } else if (type == "regex") {
            if (!g.at("value").is_string()) {
                return "grammar is not valid: regex value must be a string";
            }
            c["regex"] = g.at("value");
        } else {
            return "grammar is not valid: unknown type \"" + type + "\"";
        }
    }
    out.truncate         = h_trunc ? (int) truncate : 0;
    out.return_full_text = return_full_text;
    out.sampling         = sampling;
    out.completion       = c;
    return "";
}

server_routes_tgi::server_routes_tgi(server_routes & routes) : routes(routes) {
    post_generate        = [this](const server_http_req & req) { return handle(req, false, false); };
    post_generate_stream = [this](const server_http_req & req) { return handle(req, false, true); };
    post_compat          = [this](const server_http_req & req) { return handle(req, true, false); };
    get_info             = [this](const server_http_req & req) { return handle_info(req); };
}

json server_routes_tgi::props(const server_http_req & req) const {
    server_http_res_ptr r = routes.get_props(req);
    if (r == nullptr || r->status != 200) {
        return json::object();
    }
    json p = json::parse_no_throw(r->data);
    return p.is_discarded() ? json::object() : p;
}

server_http_res_ptr server_routes_tgi::handle_info(const server_http_req & req) const {
    const json p = props(req);
    const json gs = p.contains("default_generation_settings") ? p.at("default_generation_settings") : json::object();
    const int n_ctx = json_value(gs, "n_ctx", 0);
    std::string model_id = json_value(p, "model_alias", std::string(""));
    if (model_id.empty()) {
        model_id = json_value(p, "model_path", std::string(""));
    }
    // lib.rs:261-300 field set; the router fields describe THIS server
    json info = json {
        { "model_id",                model_id },
        { "model_sha",               nullptr },
        { "model_pipeline_tag",      "text-generation" },
        { "max_concurrent_requests", json_value(p, "total_slots", 1) },
        { "max_best_of",             std::max(1, json_value(p, "total_slots", 1)) }, // n_cmpl children need slots
        { "max_stop_sequences",      k_tgi_max_stop_sequences },
        { "max_input_tokens",        n_ctx > 0 ? n_ctx - 1 : 0 },
        { "max_total_tokens",        n_ctx },
        { "validation_workers",      1 },
        { "max_client_batch_size",   1 },
        { "router",                  "llama-server (TGI-compatible routes)" },
        { "version",                 json_value(p, "build_info", std::string("")) },
        { "sha",                     nullptr },
        { "docker_label",            nullptr },
    };
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(info);
    return res;
}

// file name without directory and extension: "/x/y/legal-v2.gguf" -> "legal-v2"
static std::string adapter_stem(const std::string & path) {
    const size_t b = path.find_last_of("/\\");
    const std::string f = b == std::string::npos ? path : path.substr(b + 1);
    const size_t dot = f.rfind('.');
    return dot == std::string::npos || dot == 0 ? f : f.substr(0, dot);
}

// adapter_id -> this server's per-request LoRA fields (server-schema.cpp "lora" / "lora_adapter"). Returns "" or a 422 message.
std::string server_routes_tgi::resolve_adapter(const server_http_req & req, const common_json & props, const std::string & adapter_id,
                                               common_json & completion) const {
    server_http_req list_req { {}, {}, "/lora-adapters", "", "", {}, req.should_stop };
    server_http_res_ptr lr = routes.get_lora_adapters(list_req);
    if (lr != nullptr && lr->status == 200) {
        const json list = json::parse_no_throw(lr->data);
        if (!list.is_discarded() && list.is_array()) {
            for (const auto & e : list) {
                if (!e.is_object() || !e.contains("id") || !e.contains("path") || !e.at("path").is_string()) {
                    continue;
                }
                const std::string path = e.at("path").get<std::string>();
                if (adapter_id == path || adapter_id == adapter_stem(path)) {
                    completion["lora"] = json::array({ json { { "id", e.at("id") }, { "scale", 1.0 } } }); // the others stay at 0
                    return "";
                }
            }
        }
    }
    if (json_value(props, "lora_pool", false)) {
        completion["lora_adapter"] = adapter_id; // loaded on demand from --lora-pool-dir; the pool answers not-found itself
        return "";
    }
    return "`adapter_id` `" + adapter_id + "` is not a loaded LoRA adapter (this server has no --lora-pool-dir)";
}

server_http_res_ptr server_routes_tgi::handle(const server_http_req & req, bool compat_route, bool stream_route) const {
    json body = json::parse_no_throw(req.body);
    const bool stream_hint = stream_route || (compat_route && !body.is_discarded() && body.is_object() &&
                                              json_value(body, "stream", false));
    if (body.is_discarded()) {
        return stream_hint ? tgi_stream_error("Input validation error: invalid JSON body", "validation")
                           : tgi_error(422, "Input validation error: invalid JSON body", "validation");
    }
    const json pr = props(req);
    const int n_ctx = pr.contains("default_generation_settings")
                    ? json_value(pr.at("default_generation_settings"), "n_ctx", 0) : 0;

    server_tgi_request tr;
    const int max_best_of = std::max(1, json_value(pr, "total_slots", 1));
    const std::string verr = server_tgi_translate(body, compat_route, stream_route, n_ctx, max_best_of, tr);
    if (!verr.empty()) {
        return stream_hint ? tgi_stream_error("Input validation error: " + verr, "validation")
                           : tgi_error(422, "Input validation error: " + verr, "validation");
    }

    if (!tr.adapter_id.empty()) {
        const std::string aerr = resolve_adapter(req, pr, tr.adapter_id, tr.completion);
        if (!aerr.empty()) {
            return stream_hint ? tgi_stream_error("Input validation error: " + aerr, "validation")
                               : tgi_error(422, "Input validation error: " + aerr, "validation");
        }
    }

    // decoder_input_details: the prompt goes to /completion as the very ids reported in details.prefill
    json prefill_ids  = json::array();
    json prefill_text = json::array();
    if (tr.truncate > 0 || tr.decoder_input_details) {
        // TGI truncates the encoded input from the left to `truncate` tokens (validation.rs tokenize + truncate)
        server_http_req tok_req {
            {}, {}, "/tokenize", "", safe_json_to_str(json {{ "content", tr.inputs }, { "add_special", true },
                                                            { "with_pieces", tr.decoder_input_details }}), {},
            req.should_stop,
        };
        server_http_res_ptr tok = routes.post_tokenize(tok_req);
        if (tok == nullptr || tok->status != 200) {
            return tgi_error(422, "Input validation error: tokenizer error", "validation");
        }
        json tj = json::parse_no_throw(tok->data);
        if (tj.is_discarded() || !tj.contains("tokens") || !tj.at("tokens").is_array()) {
            return tgi_error(422, "Input validation error: tokenizer error", "validation");
        }
        const json & all = tj.at("tokens");
        const size_t n = all.size();
        const size_t first = tr.truncate > 0 && n > (size_t) tr.truncate ? n - (size_t) tr.truncate : 0;
        for (size_t i = first; i < n; ++i) {
            const json & t = all.at(i);
            if (!t.is_object()) {
                prefill_ids.push_back(t);
                prefill_text.push_back("");
                continue;
            }
            prefill_ids.push_back(t.at("id"));
            const json & piece = t.at("piece");
            std::string text;
            if (piece.is_string()) {
                text = piece.get<std::string>();
            } else {
                for (const auto & b : piece) { // /tokenize sends a piece that is not valid UTF-8 as its bytes
                    text.push_back((char) b.get<int>());
                }
            }
            prefill_text.push_back(text);
        }
        tr.completion["prompt"] = prefill_ids;
    }

    // The inner streaming response keeps a raw pointer to its request (server-ollama.cpp:690-704 explains the crash
    // this avoids), so the request lives on the heap for as long as the response does.
    auto inner_req = std::shared_ptr<server_http_req>(new server_http_req {
        {}, {}, "/completion", "", safe_json_to_str(tr.completion), {}, req.should_stop,
    });
    server_http_res_ptr inner = routes.post_completions(*inner_req);
    if (inner == nullptr) {
        return nullptr;
    }
    if (inner->status != 200) {
        std::string type;
        const int status = inner_status_to_tgi(inner->status, type);
        const std::string prefix = type == "validation" ? "Input validation error: "
                                 : type == "generation" ? "Request failed during generation: " : "";
        const std::string msg = type == "overloaded" ? "Model is overloaded" : prefix + inner_error_message(inner->data);
        return tr.stream ? tgi_stream_error(msg, type) : tgi_error(status, msg, type);
    }

    const std::string prompt_prefix = tr.return_full_text ? tr.inputs : std::string();

    if (!inner->is_stream()) {
        json all = json::parse_no_throw(inner->data);
        if (all.is_discarded() || (all.is_array() && all.empty())) {
            return tgi_error(500, "Incomplete generation", "incomplete_generation");
        }
        if (!all.is_array()) {
            all = json::array({ all }); // n_cmpl > 1 answers one element per sequence, a single request an object
        }

        // details.prefill, shared by every sequence: [{id, text, logprob}], the first token has no context (null)
        json prefill = json::array();
        if (tr.decoder_input_details) {
            json scored = all.at(0);
            if (tr.best_of > 1) {
                // prompt_logprobs refuses n_cmpl > 1: score the same ids in a one-token greedy request
                server_http_req sc_req {
                    {}, {}, "/completion", "", safe_json_to_str(json {
                        { "prompt", prefill_ids }, { "n_predict", 1 }, { "temperature", 0.0 }, { "prompt_logprobs", true },
                    }), {}, req.should_stop,
                };
                server_http_res_ptr sc = routes.post_completions(sc_req);
                if (sc == nullptr || sc->status != 200 || sc->is_stream()) {
                    return tgi_error(424, "Request failed during generation: prefill scoring failed", "generation");
                }
                scored = json::parse_no_throw(sc->data);
            }
            const json lps = scored.is_object() && scored.contains("prompt_logprobs")
                           ? json_value(scored.at("prompt_logprobs"), "logprobs", json::array()) : json::array();
            if (!lps.is_array() || lps.size() != prefill_ids.size()) {
                return tgi_error(424, "Request failed during generation: prefill logprobs for " +
                                      std::to_string(lps.is_array() ? lps.size() : 0) + " of " +
                                      std::to_string(prefill_ids.size()) + " prompt tokens", "generation");
            }
            for (size_t i = 0; i < prefill_ids.size(); ++i) {
                prefill.push_back(json {
                    { "id", prefill_ids.at(i) }, { "text", prefill_text.at(i) },
                    { "logprob", i == 0 ? json(nullptr) : lps.at(i) },
                });
            }
        }

        struct tgi_seq {
            std::string text;
            std::string stop_type;
            json        tokens = json::array();
            json        top    = json::array();
            int         generated = 0;
            json        seed      = nullptr;
            double      mean_logprob = -INFINITY;
        };
        std::vector<tgi_seq> seqs;
        for (const json & r : all) {
            tgi_seq s;
            s.text      = json_value(r, "content", std::string(""));
            s.stop_type = json_value(r, "stop_type", std::string(""));
            s.generated = json_value(r, "tokens_predicted", 0);
            if (r.contains("completion_probabilities") && r.at("completion_probabilities").is_array()) {
                const json & cp = r.at("completion_probabilities");
                double sum = 0.0;
                for (size_t i = 0; i < cp.size(); ++i) {
                    const bool last_eos = (i + 1 == cp.size()) && s.stop_type == "eos";
                    s.tokens.push_back(tgi_token(cp.at(i), last_eos));
                    if (tr.top_n_tokens > 0) {
                        s.top.push_back(tgi_top_tokens(cp.at(i), tr.top_n_tokens));
                    }
                    const json & lp = s.tokens.back().at("logprob");
                    sum += lp.is_number() ? lp.get<double>() : -INFINITY;
                }
                if (!cp.empty()) {
                    s.mean_logprob = sum / (double) cp.size(); // infer/mod.rs:338-344 ranks by the MEAN
                }
            }
            if (tr.sampling) {
                // each n_cmpl child samples with its own seed (server-task.h add_child); report the one it used
                const json gs = r.contains("generation_settings") ? r.at("generation_settings") : json::object();
                s.seed = gs.contains("seed") ? gs.at("seed")
                       : (tr.completion.contains("seed") ? tr.completion.at("seed") : json(nullptr));
            }
            seqs.push_back(std::move(s));
        }
        size_t best = 0;
        for (size_t i = 1; i < seqs.size(); ++i) {
            if (seqs[i].mean_logprob > seqs[best].mean_logprob) {
                best = i;
            }
        }
        const tgi_seq & b = seqs[best];
        json resp = json {{ "generated_text", prompt_prefix + b.text }};
        if (tr.details) {
            json d = json {
                { "finish_reason",    server_tgi_finish_reason(b.stop_type) },
                { "generated_tokens", b.generated },
                { "seed",             b.seed },
                { "prefill",          prefill },
                { "tokens",           b.tokens },
            };
            if (tr.top_n_tokens > 0) {
                d["top_tokens"] = b.top;
            }
            if (tr.best_of > 1) {
                json others = json::array();
                for (size_t i = 0; i < seqs.size(); ++i) {
                    if (i == best) {
                        continue;
                    }
                    json o = json {
                        { "generated_text",   prompt_prefix + seqs[i].text },
                        { "finish_reason",    server_tgi_finish_reason(seqs[i].stop_type) },
                        { "generated_tokens", seqs[i].generated },
                        { "seed",             seqs[i].seed },
                        { "prefill",          prefill },
                        { "tokens",           seqs[i].tokens },
                    };
                    if (tr.top_n_tokens > 0) {
                        o["top_tokens"] = seqs[i].top;
                    }
                    others.push_back(o);
                }
                d["best_of_sequences"] = others;
            }
            resp["details"] = d;
        }
        auto res = std::make_unique<server_http_res>();
        res->headers["x-prompt-tokens"]    = std::to_string(json_value(all.at(best), "tokens_evaluated", 0));
        res->headers["x-generated-tokens"] = std::to_string(b.generated);
        if (compat_route) {
            json arr = json::array();
            arr.push_back(resp); // server.rs:146-149 wraps the compat answer in a one-element array
            res->data = safe_json_to_str(arr);
        } else {
            res->data = safe_json_to_str(resp);
        }
        return res;
    }

    // streaming: one TGI event per generated token. The last token's event must carry generated_text (and
    // details), so each token event is held until the next token, or the end, decides whether it is the last.
    struct stream_state {
        std::shared_ptr<server_http_req> inner_req; // declared first: destroyed after `inner`
        server_http_res_ptr inner;
        server_tgi_request tr;
        std::string prefix;
        std::string text;
        json held = nullptr;    // the pending token event
        int index = 0;
        bool done = false;
    };
    auto st = std::make_shared<stream_state>();
    st->inner_req = inner_req;
    st->inner     = std::move(inner);
    st->tr        = tr;
    st->prefix    = prompt_prefix;

    auto res = std::make_unique<server_http_res>();
    res->content_type = "text/event-stream";
    res->headers["Cache-Control"]     = "no-cache";
    res->headers["X-Accel-Buffering"] = "no";
    res->next = [st](std::string & output) -> bool {
        output.clear();
        if (st->done) {
            return false;
        }
        auto on_event = [&](const json & ev) {
            if (ev.contains("error")) {
                std::string type;
                inner_status_to_tgi(500, type);
                output += tgi_sse(json {{ "error", "Request failed during generation: " + inner_error_message(safe_json_to_str(ev)) },
                                        { "error_type", type }});
                st->done = true;
                return;
            }
            st->text += json_value(ev, "content", std::string(""));
            if (ev.contains("completion_probabilities") && ev.at("completion_probabilities").is_array()) {
                const json & cp = ev.at("completion_probabilities");
                for (size_t i = 0; i < cp.size(); ++i) {
                    if (!st->held.is_null()) {
                        output += tgi_sse(st->held);
                    }
                    st->index += 1;
                    st->held = json {
                        { "index",          st->index },
                        { "token",          tgi_token(cp.at(i), false) },
                        { "generated_text", nullptr },
                        { "details",        nullptr },
                    };
                    if (st->tr.top_n_tokens > 0) {
                        st->held["top_tokens"] = tgi_top_tokens(cp.at(i), st->tr.top_n_tokens);
                    }
                }
            }
            if (json_value(ev, "stop", false)) {
                const std::string stop_type = json_value(ev, "stop_type", std::string(""));
                if (st->held.is_null()) {
                    st->index += 1;
                    st->held = json {
                        { "index",          st->index },
                        { "token",          json {{ "id", 0 }, { "text", "" }, { "logprob", 0.0 }, { "special", false }} },
                        { "generated_text", nullptr },
                        { "details",        nullptr },
                    };
                }
                if (stop_type == "eos") {
                    st->held["token"]["special"] = true;
                }
                st->held["generated_text"] = st->prefix + st->text;
                if (st->tr.details) {
                    json d = json {
                        { "finish_reason",    server_tgi_finish_reason(stop_type) },
                        { "generated_tokens", json_value(ev, "tokens_predicted", 0) },
                        { "seed",             nullptr },
                        { "input_length",     json_value(ev, "tokens_evaluated", 0) },
                    };
                    if (st->tr.sampling && st->tr.completion.contains("seed")) {
                        d["seed"] = st->tr.completion.at("seed");
                    }
                    st->held["details"] = d;
                }
                output += tgi_sse(st->held);
                st->held = nullptr;
                st->done = true;
            }
        };

        if (!st->inner->data.empty()) {
            // the first chunk is primed into data by handle_completions_impl before the handler returns
            std::string first = std::move(st->inner->data);
            st->inner->data.clear();
            for (const auto & ev : parse_events(first)) {
                if (!st->done) on_event(ev);
            }
            if (!output.empty()) {
                return true;
            }
        }
        while (!st->done && st->inner->next) {
            std::string chunk;
            const bool more = st->inner->next(chunk);
            for (const auto & ev : parse_events(chunk)) {
                if (!st->done) on_event(ev);
            }
            if (!output.empty()) {
                return true;
            }
            if (!more) {
                break;
            }
        }
        if (!st->done) {
            // the inner stream ended without a final event
            output = tgi_sse(json {{ "error", "Incomplete generation stream" }, { "error_type", "incomplete_generation" }});
            st->done = true;
            return true;
        }
        return !output.empty();
    };
    return res;
}
