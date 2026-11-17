// Operator sampler-override layers: server-side values for request fields, with "default", "force" and "additive" modes.
// RESEARCH-DONE: tabbyAPI common/sampling.py + endpoints/core/router.py read for behaviour (see SOURCE-READ below).
//
// A layer is a JSON object  { "<request field>": { "override": <value>, "force": bool, "additive": bool }, ... }.
// Two layers exist: global (--sampler-overrides, or POST /sampler-overrides) and model (--sampler-overrides-model, or the
// same route); a key in the model layer replaces the same key of the global layer. For one request field:
//   default  (neither flag)   the value is used only when the request does not carry the field (null counts as absent,
//                             the same rule the request schema uses);
//   force                     the value always replaces what the request sent, aliases of the field included;
//   additive (list values)    the operator's list is placed IN FRONT of the request's list, no de-duplication; when the
//                             request carries no list the operator's list is used as it is. Force beats additive.
// The operator can therefore pin or cap what a client may do (force max_tokens, force a stop list, force a banned list)
// while every field left alone stays in the client's hands.
//
// Header-only apart from the JSON class (common/json.h); tested by tools/server/tests_cpp/test_server_overrides.cpp.
//
// SOURCE-READ: upstream/tabbyAPI/common/sampling.py:332-348, 452-473, 476-505, 566-646, 681-706 and
// endpoints/core/router.py:548-603 (theroyallab/tabbyAPI be74bf0a, AGPL-3.0). Read for behaviour, spec written in my own
// words, implemented here with my own structure: no code copied.
// SOURCE-NUANCE (what the row's description did not say):
//   (1) tabbyAPI applies force/additive only when the override value is TRUTHY (`if override:`): a forced 0, false or empty
//       list is silently NOT applied, so "force temperature 0" does nothing there. Here any non-null value applies; a forced
//       0 is a forced 0.
//   (2) additive puts the operator's list BEFORE the request's list (override + request), which matters for stop-string
//       order and for logit_bias-style lists where the first entry wins; mirrored here.
//   (3) its switch route stores a raw dict with no validation (a malformed entry then fails on every later request); here a
//       layer is validated whole before it replaces the old one, and a layer that fails validation changes nothing.
//   (4) unknown field names are skipped with a warning at load, not per request (row's test asked for this) - kept.
//   (5) its list route is open to every key and shows forced values; the preset-name listing is admin-only. Here GET shows
//       the layers to any authorised key (a client may want to know what is pinned) and only POST needs the admin tier.
//   (6) a request field has aliases (max_tokens / n_predict); forcing must remove every alias or the schema would read the
//       alias the client sent instead. Handled through the alias groups the request schema defines.
//   (7) fields that the HTTP handler reads BEFORE the schema (stream, id_slot, prompt, messages, use_beam_search,
//       message_delimiters, length_penalty) cannot be overridden: forcing them would split the handler's idea of the
//       request from the task's.
#pragma once

#include "json.h"

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct server_override_entry {
    common_json value;
    bool        force    = false;
    bool        additive = false;
};

// canonical request field name -> entry
using server_override_layer = std::map<std::string, server_override_entry>;

// every accepted request-field name -> the group of names that mean the same field (group[0] is canonical; the request
// schema reads the first name of the group that is present and non-null)
using server_override_groups = std::map<std::string, std::vector<std::string>>;

inline const std::set<std::string> & server_override_barred_fields() {
    static const std::set<std::string> s = {
        "stream", "id_slot", "prompt", "messages", "use_beam_search", "message_delimiters", "length_penalty",
    };
    return s;
}

// Validate and canonicalise a layer spec. On success fills `out` (canonical names) and returns true; on a malformed spec
// returns false with `err`, and `out` is left empty. `warnings` collects entries that were skipped or are inert.
inline bool server_override_parse(const common_json & spec, const server_override_groups & groups, const std::string & where,
                                  server_override_layer & out, std::string & err, std::vector<std::string> & warnings) {
    out.clear();
    if (spec.is_null()) {
        return true; // null empties the layer
    }
    if (!spec.is_object()) {
        err = "sampler overrides in " + where + " must be a JSON object of {field: {override, force, additive}}";
        return false;
    }
    server_override_layer tmp;
    for (auto it = spec.begin(); it != spec.end(); ++it) {
        const std::string name = it.key();
        const common_json & v  = it.value();
        if (!v.is_object() || !v.contains("override")) {
            err = "sampler override \"" + name + "\" in " + where + " must be an object with an \"override\" key (and optionally \"force\" / \"additive\")";
            return false;
        }
        if (v.at("override").is_null()) {
            err = "sampler override \"" + name + "\" in " + where + " has a null \"override\"; remove the entry instead";
            return false;
        }
        for (const char * flag : {"force", "additive"}) {
            if (v.contains(flag) && !v.at(flag).is_null() && !v.at(flag).is_boolean()) {
                err = std::string("\"") + flag + "\" of sampler override \"" + name + "\" in " + where + " must be true or false";
                return false;
            }
        }
        if (server_override_barred_fields().count(name)) {
            err = "sampler override \"" + name + "\" in " + where + " is not allowed: the HTTP handler reads that field before the sampling schema";
            return false;
        }
        const auto g = groups.find(name);
        if (g == groups.end()) {
            warnings.push_back("skipping unknown sampler override key \"" + name + "\" in " + where);
            continue;
        }
        const std::string & canon = g->second.front();
        if (tmp.count(canon)) {
            err = "sampler override \"" + name + "\" in " + where + " names the same field as another entry (" + canon + "); keep one";
            return false;
        }
        server_override_entry e;
        e.value    = v.at("override");
        e.force    = v.contains("force")    && v.at("force").is_boolean()    && v.at("force").template get<bool>();
        e.additive = v.contains("additive") && v.at("additive").is_boolean() && v.at("additive").template get<bool>();
        if (e.additive && !e.force && !e.value.is_array()) {
            warnings.push_back("sampler override \"" + name + "\" in " + where + " is additive but its value is not a list; it acts as a default");
        }
        tmp[canon] = std::move(e);
    }
    out = std::move(tmp);
    return true;
}

// the layers as JSON, for GET /sampler-overrides and for logging
inline common_json server_override_layer_to_json(const server_override_layer & layer) {
    common_json o = common_json::object();
    for (const auto & kv : layer) {
        common_json e = common_json::object();
        e["override"] = kv.second.value;
        e["force"]    = kv.second.force;
        e["additive"] = kv.second.additive;
        o[kv.first]   = e;
    }
    return o;
}

// effective layer: global entries, replaced key by key by the model layer
inline server_override_layer server_override_merge(const server_override_layer & global, const server_override_layer & model) {
    server_override_layer eff = global;
    for (const auto & kv : model) {
        eff[kv.first] = kv.second;
    }
    return eff;
}

// How one request field ended up: "forced", "additive", "default", or "request" (left as the client sent it).
using server_override_trace = std::vector<std::pair<std::string, std::string>>;

// Apply an already-merged layer to a request body. The body is returned by value; a non-object body is returned as it is.
inline common_json server_override_apply(const common_json & body, const server_override_layer & eff,
                                         const server_override_groups & groups, server_override_trace * trace = nullptr) {
    if (!body.is_object() || eff.empty()) {
        return body;
    }
    common_json out = body;
    for (const auto & kv : eff) {
        const std::string & canon = kv.first;
        const auto g = groups.find(canon);
        if (g == groups.end()) {
            continue; // the layer was parsed against other groups; nothing to do for a name the schema does not know
        }
        const std::vector<std::string> & names = g->second;
        // the name the schema would read: the first of the group that is present and non-null
        const std::string * used = nullptr;
        for (const auto & n : names) {
            if (out.contains(n) && !out.at(n).is_null()) {
                used = &n;
                break;
            }
        }
        const server_override_entry & e = kv.second;
        const char * how = "request";
        if (e.force) {
            for (const auto & n : names) {
                if (out.contains(n)) {
                    out.erase(n);
                }
            }
            out[canon] = e.value;
            how = "forced";
        } else if (used == nullptr) {
            for (const auto & n : names) {
                if (out.contains(n)) {
                    out.erase(n); // a null-valued alias would only be noise
                }
            }
            out[canon] = e.value;
            how = "default";
        } else if (e.additive && e.value.is_array() && out.at(*used).is_array()) {
            common_json merged = common_json::array();
            merged.insert(e.value);
            merged.insert(out.at(*used));
            out[*used] = merged;
            how = "additive";
        }
        if (trace) {
            trace->emplace_back(canon, how);
        }
    }
    return out;
}

// Process-wide state: the two layers behind one lock. Reads copy the merged layer out, so a request never holds the lock
// while it is being parsed.
class server_override_state {
  public:
    void set_global(server_override_layer l) {
        std::lock_guard<std::mutex> lk(mtx);
        global = std::move(l);
    }
    void set_model(server_override_layer l) {
        std::lock_guard<std::mutex> lk(mtx);
        model = std::move(l);
    }
    server_override_layer effective() const {
        std::lock_guard<std::mutex> lk(mtx);
        return server_override_merge(global, model);
    }
    bool empty() const {
        std::lock_guard<std::mutex> lk(mtx);
        return global.empty() && model.empty();
    }
    common_json to_json() const {
        server_override_layer g, m;
        {
            std::lock_guard<std::mutex> lk(mtx);
            g = global;
            m = model;
        }
        common_json o = common_json::object();
        o["global"]    = server_override_layer_to_json(g);
        o["model"]     = server_override_layer_to_json(m);
        o["effective"] = server_override_layer_to_json(server_override_merge(g, m));
        return o;
    }

  private:
    mutable std::mutex    mtx;
    server_override_layer global;
    server_override_layer model;
};

inline server_override_state & server_override_instance() {
    static server_override_state s;
    return s;
}
