// Server-side compaction for the OpenAI Responses API: POST /v1/responses/compact, and the compaction_trigger input item
// of POST /v1/responses (the shape current Codex clients send when they compact remotely).
//
// RESEARCH-DONE: tmp/qwork_qworker-fork27/ollama_responses_compact.go, 984 lines read whole (Ollama
// openai/responses_compact.go, MIT; row S69-OLLAMA-PICK-CODEX-RESPONSES-COMPACTION-ENDPOINT-2026-09-29-01).
// This is a re-write on this server's own json wrapper and Responses item shapes, not a copy of the Go code:
//   * the compaction item's encrypted_content is NOT encrypted: it is a versioned JSON payload {type, version, summary,
//     retained[]} that this server reads back on the next request (the client treats it as opaque);
//   * the model is asked for ONE forced create_summary(summary, retain_item_ids[]) call over the transcript; a reply
//     that is not exactly one valid call is refused (and retried once with the validation error in the prompt);
//   * items the model must not paraphrase (active tool state, standalone tool outputs) are always retained, and a
//     retained tool call drags its output along (and the reverse);
//   * when the transcript does not fit the context, the oldest removable items (completed tool pairs, assistant
//     messages) are trimmed, ~20% of the serialized text per round, and the summary is told how many were omitted;
//   * on the next request the newest compaction item expands into a synthetic function_call / function_call_output
//     pair (the summary) followed by the retained items, in their original order.
// One variation on purpose: image bytes are not sent to the summarising model (each item says how many it carried);
// retained items keep them verbatim, so a retained image still reaches the next turn.
//
// Header-only and free of the server: tests link llama-common only (tests_cpp/test_server_responses_compact.cpp).
#pragma once

#include "json.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace respcompact {

using json = common_json;

constexpr const char * PAYLOAD_TYPE      = "llama_compaction";
constexpr int          PAYLOAD_VERSION   = 1;
constexpr const char * SUMMARY_TOOL      = "create_summary";
constexpr const char * SUMMARY_CALL_NAME = "llama_compaction_summary";
constexpr const char * CALL_PREFIX       = "call_llama_compaction_";

inline std::string str_of(const json & j, const char * key) {
    if (j.is_object() && j.contains(key) && j.at(key).is_string()) {
        return j.at(key).get<std::string>();
    }
    return std::string();
}

inline bool bool_of(const json & j, const char * key) {
    return j.is_object() && j.contains(key) && j.at(key).is_boolean() && j.at(key) == true;
}

inline std::string trim_ws(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) { b++; }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) { e--; }
    return s.substr(b, e - b);
}

// the item type, with the Responses shorthand {role, content} read as a message ("" when it is not an object)
inline std::string item_type(const json & it) {
    if (!it.is_object()) {
        return std::string();
    }
    std::string t = str_of(it, "type");
    if (t.empty() && it.contains("role")) {
        return "message";
    }
    return t;
}

// 64-bit FNV-1a, 12 hex digits: a stable id suffix for the synthetic call, not a security hash
inline std::string hash12(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%012llx", (unsigned long long) (h & 0xffffffffffffULL));
    return buf;
}

// the request's input as an array of items: a string is one user message
inline bool input_items(const json & body, json & items, std::string & err) {
    if (!body.is_object() || !body.contains("input") || body.at("input").is_null()) {
        err = "input is required";
        return false;
    }
    const json & in = body.at("input");
    if (in.is_string()) {
        json msg = json::object();
        msg["type"]    = "message";
        msg["role"]    = "user";
        msg["content"] = in.get<std::string>();
        items = json::array();
        items.push_back(msg);
        return true;
    }
    if (!in.is_array()) {
        err = "input must be a string or array";
        return false;
    }
    items = in;
    return true;
}

// ---- expansion of a compaction item -----------------------------------------------------------------------------

// Decodes the payload of a compaction item. The strict checks mirror the reference: a foreign blob is refused by name.
inline bool decode_payload(const json & item, json & payload, std::string & err) {
    const std::string enc = str_of(item, "encrypted_content");
    if (item_type(item) != "compaction" || enc.empty()) {
        err = "invalid compaction item";
        return false;
    }
    json p = json::parse_no_throw(enc);
    if (p.is_discarded() || !p.is_object()) {
        err = "unsupported compaction item: encrypted_content is not a llama.cpp payload";
        return false;
    }
    if (str_of(p, "type") != PAYLOAD_TYPE || !p.contains("version") || !p.at("version").is_number_integer() ||
        !(p.at("version") == PAYLOAD_VERSION)) {
        err = "unsupported compaction payload type or version";
        return false;
    }
    if (trim_ws(str_of(p, "summary")).empty()) {
        err = "compaction payload has an empty summary";
        return false;
    }
    if (!p.contains("retained") || !p.at("retained").is_array()) {
        err = "compaction payload has no retained list";
        return false;
    }
    payload = p;
    return true;
}

// The summary as the synthetic call/output pair every Responses-aware template already knows how to render.
inline void payload_to_items(const json & payload, json & out) {
    const std::string id = std::string(CALL_PREFIX) + hash12(payload.dump());
    json args = json::object();
    args["version"] = PAYLOAD_VERSION;
    json call = json::object();
    call["type"]      = "function_call";
    call["call_id"]   = id;
    call["name"]      = SUMMARY_CALL_NAME;
    call["arguments"] = args.dump();
    json result = json::object();
    result["type"]    = "function_call_output";
    result["call_id"] = id;
    result["output"]  = str_of(payload, "summary");
    out.push_back(call);
    out.push_back(result);
    for (const json & r : payload.at("retained")) {
        out.push_back(r);
    }
}

// Replaces the NEWEST compaction item by its summary and retained items; everything before it is dropped (summarised).
// `changed` is false (and `out` a copy) when there is none. Items around the boundary keep their order.
inline bool expand_items(const json & items, json & out, bool & changed, std::string & err) {
    changed = false;
    int boundary = -1;
    int i = 0;
    for (const json & it : items) {
        if (item_type(it) == "compaction") {
            boundary = i;
        }
        i++;
    }
    if (boundary < 0) {
        out = items;
        return true;
    }
    json payload;
    if (!decode_payload(items.at((size_t) boundary), payload, err)) {
        return false;
    }
    out = json::array();
    i = 0;
    for (const json & it : items) {
        if (i == boundary) {
            payload_to_items(payload, out);
        } else if (i > boundary) {   // items before the newest compaction are what it summarised
            out.push_back(it);
        }
        i++;
    }
    changed = true;
    return true;
}

// Rewrites body["input"] in place. Returns false with `err` for a bad compaction item.
inline bool expand_body(json & body, bool & changed, std::string & err) {
    changed = false;
    if (!body.is_object() || !body.contains("input") || !body.at("input").is_array()) {
        return true;
    }
    json out;
    if (!expand_items(body.at("input"), out, changed, err)) {
        return false;
    }
    if (changed) {
        body["input"] = out;
    }
    return true;
}

// ---- the plan ---------------------------------------------------------------------------------------------------

struct entry {
    std::string ref;
    std::string kind; // message | function_call | function_call_output | reasoning | web_search_call
    std::string call_id;
    bool        standalone = false; // a tool output with no call_id (it names its tool instead)
    bool        assistant  = false;
    json        item;
};

struct group {
    std::string call_ref;
    std::string result_ref; // "" while the call has no output
};

struct plan {
    std::string           model;
    bool                  stream  = false;
    int                   omitted = 0;
    std::vector<entry>    items;
    std::vector<group>    groups;
    std::set<std::string> forced;
    json                  tools = json::array(); // [{name, description}], metadata only
};

// image payloads are not sent to the summarising model; returns a copy with each image_url replaced and counted
inline json strip_images(const json & j, int & n_images) {
    if (j.is_object()) {
        json o = json::object();
        for (const auto & kv : j.items()) {
            if (kv.key() == "image_url" && (kv.value().is_string() || kv.value().is_object())) {
                n_images++;
                o[kv.key()] = "[image omitted]";
            } else {
                o[kv.key()] = strip_images(kv.value(), n_images);
            }
        }
        return o;
    }
    if (j.is_array()) {
        json a = json::array();
        for (const json & v : j) {
            a.push_back(strip_images(v, n_images));
        }
        return a;
    }
    return j;
}

// {ref, type, item, image_count}: what the summarising model sees for one entry, and what the trimmer measures
inline json wire_of(const entry & e) {
    int n = 0;
    json w = json::object();
    w["ref"]  = e.ref;
    w["type"] = e.kind;
    w["item"] = strip_images(e.item, n);
    if (n > 0) {
        w["image_count"] = n;
    }
    return w;
}

inline void collect_tools(const json & tools, const std::string & prefix, json & out) {
    if (!tools.is_array()) {
        return;
    }
    for (const json & t : tools) {
        if (!t.is_object()) {
            continue;
        }
        std::string name = trim_ws(str_of(t, "name"));
        const std::string type = str_of(t, "type");
        if (name.empty() && (type == "tool_search" || type == "web_search")) {
            name = type;
        }
        if (!prefix.empty() && !name.empty()) {
            name = prefix + "." + name;
        }
        if (type == "namespace") {
            if (t.contains("tools")) {
                collect_tools(t.at("tools"), name, out);
            }
            continue;
        }
        if (name.empty()) {
            continue;
        }
        json m = json::object();
        m["name"] = name;
        if (!str_of(t, "description").empty()) {
            m["description"] = str_of(t, "description");
        }
        out.push_back(m);
    }
}

inline bool classify(const json & it, entry & e, std::string & err) {
    const std::string t = item_type(it);
    e.item = it;
    if (t == "message") {
        e.kind      = "message";
        e.assistant = str_of(it, "role") == "assistant";
        return true;
    }
    if (t == "function_call" || t == "tool_search_call") {
        e.kind      = "function_call";
        e.call_id   = str_of(it, "call_id");
        e.assistant = true;
        return true;
    }
    if (t == "function_call_output" || t == "tool_search_output") {
        e.kind       = "function_call_output";
        e.call_id    = str_of(it, "call_id");
        e.standalone = e.call_id.empty() && !trim_ws(str_of(it, "name")).empty();
        return true;
    }
    if (t == "reasoning") {
        e.kind      = "reasoning";
        e.assistant = true;
        return true;
    }
    if (t == "web_search_call") {
        e.kind      = "web_search_call";
        e.assistant = true;
        return true;
    }
    if (t == "compaction" || t == "compaction_trigger") {
        err = "compaction control item in the wrong place";
        return false;
    }
    err = "unsupported compaction input type '" + t + "'";
    return false;
}

// Pairs tool calls with outputs and decides which items can never be dropped: standalone outputs, and every call that
// is still "active" (no output yet, or nothing assistant-side after its output).
inline bool analyze(plan & p, std::string & err) {
    struct pending { size_t call_idx; int result_idx; group g; };
    std::map<std::string, size_t> by_call;
    std::set<std::string> ignored;
    std::vector<pending> ordered;

    for (size_t i = 0; i < p.items.size(); i++) {
        const entry & e = p.items[i];
        if (e.kind == "function_call") {
            if (e.call_id.empty()) {
                err = e.ref + ": function call is missing call_id";
                return false;
            }
            if (str_of(e.item, "name") == SUMMARY_CALL_NAME && e.call_id.rfind(CALL_PREFIX, 0) == 0) {
                ignored.insert(e.call_id);
                continue;
            }
            if (by_call.count(e.call_id)) {
                err = e.ref + ": duplicate function call_id '" + e.call_id + "'";
                return false;
            }
            by_call[e.call_id] = ordered.size();
            ordered.push_back({ i, -1, group{ e.ref, "" } });
        } else if (e.kind == "function_call_output") {
            if (e.standalone) {
                p.forced.insert(e.ref);
                continue;
            }
            if (ignored.count(e.call_id)) {
                continue;
            }
            auto it = by_call.find(e.call_id);
            if (it == by_call.end()) {
                err = e.ref + ": function output has no matching call '" + e.call_id + "'";
                return false;
            }
            pending & g = ordered[it->second];
            if (g.result_idx >= 0) {
                err = e.ref + ": duplicate function output for call '" + e.call_id + "'";
                return false;
            }
            g.result_idx   = (int) i;
            g.g.result_ref = e.ref;
        }
    }
    for (const pending & g : ordered) {
        p.groups.push_back(g.g);
        bool active = g.result_idx < 0;
        if (!active) {
            active = true;
            for (size_t k = (size_t) g.result_idx + 1; k < p.items.size(); k++) {
                const entry & e = p.items[k];
                if (e.kind == "reasoning" || e.kind == "function_call" || (e.kind == "message" && e.assistant)) {
                    active = false;
                    break;
                }
            }
        }
        if (active) {
            p.forced.insert(g.g.call_ref);
            if (!g.g.result_ref.empty()) {
                p.forced.insert(g.g.result_ref);
            }
        }
    }
    return true;
}

inline bool build_plan(const json & body, const json & raw_items, plan & p, std::string & err) {
    if (!raw_items.is_array() || raw_items.size() == 0) {
        err = "compaction input is empty";
        return false;
    }
    p = plan();
    p.model = trim_ws(str_of(body, "model"));
    if (p.model.empty()) {
        err = "model is required";
        return false;
    }
    p.stream = bool_of(body, "stream");
    int i = 0;
    for (const json & it : raw_items) {
        entry e;
        char ref[24];
        std::snprintf(ref, sizeof(ref), "item_%06d", i + 1);
        e.ref = ref;
        std::string why;
        if (!classify(it, e, why)) {
            err = "input[" + std::to_string(i) + "]: " + why;
            return false;
        }
        p.items.push_back(std::move(e));
        i++;
    }
    if (!analyze(p, err)) {
        return false;
    }
    if (body.contains("tools")) {
        collect_tools(body.at("tools"), "", p.tools);
    }
    return true;
}

// POST /v1/responses with a final {"type":"compaction_trigger"} item. requested=false leaves ordinary requests alone.
inline bool prepare_triggered(const json & body, plan & p, bool & requested, std::string & err) {
    requested = false;
    json items;
    if (!input_items(body, items, err)) {
        // not a compaction request as far as we can tell: the ordinary path reports its own error
        err.clear();
        return true;
    }
    if (items.size() == 0 || item_type(items.back()) != "compaction_trigger") {
        return true;
    }
    requested = true;
    if (!bool_of(body, "stream")) {
        err = "compaction_trigger requires stream=true";
        return false;
    }
    json head = json::array();
    for (size_t i = 0; i + 1 < items.size(); i++) {
        if (item_type(items.at(i)) == "compaction_trigger") {
            err = "compaction_trigger must be the final and only compaction trigger";
            return false;
        }
        head.push_back(items.at(i));
    }
    json expanded;
    bool changed = false;
    if (!expand_items(head, expanded, changed, err)) {
        return false;
    }
    return build_plan(body, expanded, p, err);
}

// POST /v1/responses/compact
inline bool prepare_standalone(const json & body, plan & p, std::string & err) {
    json items;
    if (!input_items(body, items, err)) {
        return false;
    }
    for (const json & it : items) {
        if (item_type(it) == "compaction_trigger") {
            err = "compaction_trigger is only valid on POST /v1/responses";
            return false;
        }
    }
    json expanded;
    bool changed = false;
    if (!expand_items(items, expanded, changed, err)) {
        return false;
    }
    return build_plan(body, expanded, p, err);
}

// ---- trimming for a context overflow ------------------------------------------------------------------------------

// Removes the oldest removable items, aiming at 20% of the serialized transcript text. Protected: forced items, the last
// item, user/system messages. Completed tool pairs go together. Returns how many items went (0 = nothing removable).
inline int trim_for_context(plan & p) {
    if (p.items.empty()) {
        return 0;
    }
    std::map<std::string, size_t> sizes;
    size_t total = 0;
    for (const entry & e : p.items) {
        const size_t n = wire_of(e).dump().size();
        sizes[e.ref] = n;
        total += n;
    }
    std::set<std::string> protect = p.forced;
    protect.insert(p.items.back().ref);
    std::map<std::string, std::string> peers;
    for (const group & g : p.groups) {
        if (!g.result_ref.empty()) {
            peers[g.call_ref]   = g.result_ref;
            peers[g.result_ref] = g.call_ref;
        }
    }
    std::set<std::string> removed;
    size_t removed_bytes = 0;
    for (const entry & e : p.items) {
        if (removed_bytes >= (total + 4) / 5) {
            break;
        }
        if (protect.count(e.ref) || removed.count(e.ref)) {
            continue;
        }
        const auto pit = peers.find(e.ref);
        const std::string peer = pit == peers.end() ? std::string() : pit->second;
        if (e.kind == "function_call" || e.kind == "function_call_output") {
            if (peer.empty() || protect.count(peer)) {
                continue;
            }
        } else if (!e.assistant) {
            continue;
        }
        removed.insert(e.ref);
        removed_bytes += sizes[e.ref];
        if (!peer.empty()) {
            removed.insert(peer);
            removed_bytes += sizes[peer];
        }
    }
    if (removed.empty()) {
        return 0;
    }
    p.items.erase(std::remove_if(p.items.begin(), p.items.end(), [&](const entry & e) { return removed.count(e.ref) > 0; }), p.items.end());
    p.groups.erase(std::remove_if(p.groups.begin(), p.groups.end(), [&](const group & g) { return removed.count(g.call_ref) > 0; }), p.groups.end());
    p.omitted += (int) removed.size();
    return (int) removed.size();
}

// ---- the summarising request ---------------------------------------------------------------------------------------

inline std::string omission_notice(int n) {
    return "Compaction warning: " + std::to_string(n) +
           " older transcript items were omitted before summarization because the model's context limit was exceeded.";
}

// A non-streaming Responses request that forces one create_summary call. `repair` is the validation error of the
// previous reply ("" on the first try).
inline json summary_request(const plan & p, const std::string & repair) {
    std::string prompt =
        "Summarize the conversation for another coding agent. Preserve the goal, decisions, constraints, repository state, "
        "changed files, test results, failures, active work, and next actions. After optional tool metadata, each following "
        "user message is one ordered transcript item as JSON: ref, type, the source item, and image_count when it carried "
        "images. Use retain_item_ids for exact source items, including images, that cannot safely be paraphrased. Tool calls "
        "and results are execution state; do not invent or edit them. Call create_summary exactly once.";
    if (p.omitted > 0) {
        prompt += " " + omission_notice(p.omitted);
    }
    if (!repair.empty()) {
        prompt += " Your previous create_summary call was invalid: " + repair + ". Return one corrected create_summary call.";
    }
    json input = json::array();
    {
        json sys = json::object();
        sys["type"]    = "message";
        sys["role"]    = "system";
        sys["content"] = prompt;
        input.push_back(sys);
    }
    if (p.tools.size() > 0) {
        json meta = json::object();
        meta["tools"] = p.tools;
        json m = json::object();
        m["type"]    = "message";
        m["role"]    = "user";
        m["content"] = meta.dump();
        input.push_back(m);
    }
    for (const entry & e : p.items) {
        json m = json::object();
        m["type"]    = "message";
        m["role"]    = "user";
        m["content"] = wire_of(e).dump();
        input.push_back(m);
    }
    json str_t = json::object();
    str_t["type"] = "string";
    json arr_t = json::object();
    arr_t["type"]  = "array";
    arr_t["items"] = str_t;
    json props = json::object();
    props["summary"]         = str_t;
    props["retain_item_ids"] = arr_t;
    json params = json::object();
    params["type"]                 = "object";
    params["properties"]           = props;
    params["required"]             = json::array({ "summary", "retain_item_ids" });
    params["additionalProperties"] = false;
    json tool = json::object();
    tool["type"]        = "function";
    tool["name"]        = SUMMARY_TOOL;
    tool["description"] = "Return the compact summary and the exact input item references that must remain verbatim.";
    tool["parameters"]  = params;
    json choice = json::object();
    choice["type"] = "function";
    choice["name"] = SUMMARY_TOOL;

    json req = json::object();
    req["model"]               = p.model;
    req["input"]               = input;
    req["tools"]               = json::array();
    req["tools"].push_back(tool);
    req["tool_choice"]         = choice;
    req["parallel_tool_calls"] = false;
    req["store"]               = false;
    req["stream"]              = false;
    return req;
}

// ---- validating the reply -----------------------------------------------------------------------------------------

// Reads the create_summary call out of a Responses object and builds the compaction item. `usage` is the reply's usage
// (null when absent). On failure `err` is what the repair prompt shows the model.
inline bool complete(const plan & p, const json & response, json & compaction_item, json & usage, std::string & err) {
    if (!response.is_object() || !response.contains("output") || !response.at("output").is_array()) {
        err = "invalid summary response";
        return false;
    }
    std::vector<const json *> calls;
    for (const json & o : response.at("output")) {
        if (item_type(o) == "function_call" && str_of(o, "name") == SUMMARY_TOOL) {
            calls.push_back(&o);
        }
    }
    if (calls.size() != 1) {
        err = std::string("expected one ") + SUMMARY_TOOL + " call, got " + std::to_string(calls.size());
        return false;
    }
    const json & call = *calls[0];
    if (!call.contains("arguments") || !call.at("arguments").is_string()) {
        err = std::string("invalid ") + SUMMARY_TOOL + " arguments: expected a JSON string";
        return false;
    }
    json sel = json::parse_no_throw(call.at("arguments").get<std::string>());
    if (sel.is_discarded() || !sel.is_object()) {
        err = std::string("invalid ") + SUMMARY_TOOL + " arguments: not a JSON object";
        return false;
    }
    if (!sel.contains("summary") || !sel.at("summary").is_string()) {
        err = std::string("invalid ") + SUMMARY_TOOL + " arguments: summary must be a string";
        return false;
    }
    const std::string summary = trim_ws(sel.at("summary").get<std::string>());
    if (summary.empty()) {
        err = std::string(SUMMARY_TOOL) + " returned an empty summary";
        return false;
    }
    std::set<std::string> known;
    for (const entry & e : p.items) {
        known.insert(e.ref);
    }
    std::set<std::string> selected;
    if (sel.contains("retain_item_ids")) {
        if (!sel.at("retain_item_ids").is_array()) {
            err = std::string("invalid ") + SUMMARY_TOOL + " arguments: retain_item_ids must be an array";
            return false;
        }
        for (const json & r : sel.at("retain_item_ids")) {
            const std::string ref = r.is_string() ? r.get<std::string>() : std::string();
            if (!known.count(ref)) {
                err = std::string(SUMMARY_TOOL) + " selected unknown item '" + ref + "'";
                return false;
            }
            if (!selected.insert(ref).second) {
                err = std::string(SUMMARY_TOOL) + " selected duplicate item '" + ref + "'";
                return false;
            }
        }
    }
    for (const std::string & f : p.forced) {
        selected.insert(f);
    }
    for (const group & g : p.groups) {
        if (selected.count(g.call_ref) || (!g.result_ref.empty() && selected.count(g.result_ref))) {
            selected.insert(g.call_ref);
            if (!g.result_ref.empty()) {
                selected.insert(g.result_ref);
            }
        }
    }
    json retained = json::array();
    for (const entry & e : p.items) {
        if (selected.count(e.ref)) {
            retained.push_back(e.item);
        }
    }
    json payload = json::object();
    payload["type"]     = PAYLOAD_TYPE;
    payload["version"]  = PAYLOAD_VERSION;
    payload["summary"]  = p.omitted > 0 ? omission_notice(p.omitted) + "\n\n" + summary : summary;
    payload["retained"] = retained;

    compaction_item = json::object();
    compaction_item["type"]              = "compaction";
    compaction_item["encrypted_content"] = payload.dump();
    usage = response.contains("usage") ? response.at("usage") : json(nullptr);
    return true;
}

// ---- responses -----------------------------------------------------------------------------------------------------

// body of POST /v1/responses/compact
inline json compacted_response(const std::string & id, const json & item, const json & usage) {
    json out = json::object();
    out["id"]         = id;
    out["object"]     = "response.compaction";
    out["created_at"] = (int64_t) std::time(nullptr);
    out["output"]     = json::array();
    out["output"].push_back(item);
    out["usage"]      = usage;
    return out;
}

inline json response_object(const std::string & id, const std::string & model, const char * status, const json & output, const json & usage) {
    json r = json::object();
    r["id"]         = id;
    r["object"]     = "response";
    r["created_at"] = (int64_t) std::time(nullptr);
    r["status"]     = status;
    r["model"]      = model;
    r["output"]     = output;
    r["usage"]      = usage;
    return r;
}

// The Codex v2 stream: it reads output_item.done and needs exactly one compaction item before completed.
inline std::string render_sse(const std::string & id, const std::string & model, const json & item, const json & usage) {
    std::vector<std::pair<std::string, json>> ev;
    json created  = response_object(id, model, "in_progress", json::array(), json(nullptr));
    json done_out = json::array();
    done_out.push_back(item);
    json completed = response_object(id, model, "completed", done_out, usage);
    completed["completed_at"] = (int64_t) std::time(nullptr);

    auto resp_ev = [](const json & r) { json d = json::object(); d["response"] = r; return d; };
    auto item_ev = [&]() { json d = json::object(); d["output_index"] = 0; d["item"] = item; return d; };
    ev.emplace_back("response.created",           resp_ev(created));
    ev.emplace_back("response.in_progress",       resp_ev(created));
    ev.emplace_back("response.output_item.added", item_ev());
    ev.emplace_back("response.output_item.done",  item_ev());
    ev.emplace_back("response.completed",         resp_ev(completed));

    std::string out;
    int seq = 0;
    for (auto & e : ev) {
        e.second["type"]            = e.first;
        e.second["sequence_number"] = seq++;
        out += "event: " + e.first + "\ndata: " + e.second.dump() + "\n\n";
    }
    return out;
}

} // namespace respcompact
