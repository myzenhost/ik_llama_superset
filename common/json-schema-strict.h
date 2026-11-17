#pragma once

// Strict JSON-schema check: lists the constraints json_schema_to_grammar() would drop or weaken.
//
// Row S69-SGLANG-PICK-STRICT-JSON-SCHEMA-CHECK-2026-10-18-01 (programme row S69-ENGINE-MERGE-SGLANG-2026-10-14-01).
// json_schema_to_grammar() accepts every schema and silently ignores what it cannot express, so a client asking for
// {"type": "integer", "multipleOf": 7} gets any integer and no signal. SGLang refuses such schemas per backend.
//
// SOURCE-READ (rule 1.87; sgl-project/sglang, Apache-2.0, sha b4584ab83884ff40a4e5a77f0ea4ef98c038d3b9):
//   python/sglang/srt/constrained/json_schema_validation.py:94-139 walks only schema-bearing keywords (never
//   instance data such as enum/const/default values), reporting a JSON pointer per subschema; :167-200 (xgrammar)
//   and :203-260 (outlines) are per-BACKEND lists of what that backend drops, plus "constraints X, Y cannot be
//   enforced together" when a string carries more than one of format / pattern / minLength-maxLength;
//   xgrammar_backend.py:346-370 turns it into an invalid-grammar request error.
// SOURCE-NUANCE: the list is a property of the backend, not of JSON Schema, so SGLang's own lists are not reusable
// here; this file's list is derived from THIS fork's converter, common/json-schema-to-grammar.cpp visit()
// (:981-1156), read in full: $ref returns before any sibling keyword is read (:986); oneOf is built as anyOf
// (:989-995, exclusivity lost) and both drop sibling keywords; pattern wins over format, and format over
// minLength/maxLength (:1107-1122 order); only date, time, date-time and uuid[1-5] formats exist (:249-256,
// :1110); minLength/maxLength need type "string" exactly (:1117); numeric bounds are read only for type "integer"
// (:1123), with minimum shadowing exclusiveMinimum and maximum shadowing exclusiveMaximum (:1126-1135 else-if);
// the tuple form ignores minItems/maxItems, and items shadows prefixItems (:1087-1099); required is read only
// when properties or a non-true additionalProperties is present (:1016-1026); allOf is read only for untyped,
// object or string schemas and only its $ref / properties / enum parts, every property of a component counted as
// required (:1038-1085). Nothing here reads keywords such as not, if/then/else, contains, uniqueItems, multipleOf,
// dependent*, propertyNames, patternProperties, unevaluated*, min/maxProperties.
// Fresh C++; no Python code copied.

#include "json.h"

#include <set>
#include <string>
#include <vector>

namespace json_schema_strict_detail {

inline std::string escape_pointer(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '~') {
            out += "~0";
        } else if (c == '/') {
            out += "~1";
        } else {
            out += c;
        }
    }
    return out;
}

inline const std::set<std::string> & never_read() {
    static const std::set<std::string> k = {
        "not", "if", "then", "else", "contains", "minContains", "maxContains", "uniqueItems", "multipleOf",
        "dependentRequired", "dependentSchemas", "dependencies", "propertyNames", "patternProperties",
        "unevaluatedItems", "unevaluatedProperties", "minProperties", "maxProperties",
    };
    return k;
}

// keywords that never constrain the output (annotations / schema plumbing)
inline const std::set<std::string> & annotations() {
    static const std::set<std::string> k = {
        "title", "description", "default", "examples", "$comment", "$id", "$schema", "$anchor", "$dynamicAnchor",
        "deprecated", "readOnly", "writeOnly", "contentEncoding", "contentMediaType", "$defs", "definitions",
    };
    return k;
}

inline std::string join(const std::vector<std::string> & v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        out += (i ? ", " : "") + v[i];
    }
    return out;
}

inline bool is_supported_format(const std::string & f) {
    if (f == "date" || f == "time" || f == "date-time" || f == "uuid") {
        return true;
    }
    return f.size() == 5 && f.compare(0, 4, "uuid") == 0 && f[4] >= '1' && f[4] <= '5';
}

// Row S69-ENGREV-LOCALAI2-SCHEMA-ALLOF-REF-CYCLE-DEPTH-GUARD-2026-10-27-01: the walker recurses once per schema level on a
// client-sized tree, so it stops at the same cap as the converter (common/json-schema-to-grammar.cpp MAX_SCHEMA_DEPTH, 256;
// SOURCE-READ mudler/LocalAI pkg/functions/grammars/json_schema.go:13-19, MIT, behaviour only). Keep the two in step.
inline constexpr int max_depth = 256;

struct walker {
    std::vector<std::string> out;
    int  depth          = 0;      // schema levels on the visit() stack
    bool depth_reported = false;  // one message, however many branches hit the cap

    void add(const std::string & ptr, const std::string & why) { out.push_back("JSON Schema at " + ptr + ": " + why); }

    void children(const common_json & s, const std::string & ptr) {
        static const char * single[] = { "additionalProperties", "items", "not", "if", "then", "else", "contains",
                                         "propertyNames", "unevaluatedItems", "unevaluatedProperties", "additionalItems" };
        for (const char * k : single) {
            if (!s.contains(k)) continue;
            const common_json & c = s.at(k);
            if (c.is_object()) {
                visit(c, ptr + "/" + k);
            } else if (std::string(k) == "items" && c.is_array()) {
                for (size_t i = 0; i < c.size(); ++i) visit(c.at(i), ptr + "/items/" + std::to_string(i));
            }
        }
        static const char * arrays[] = { "allOf", "anyOf", "oneOf", "prefixItems" };
        for (const char * k : arrays) {
            if (s.contains(k) && s.at(k).is_array()) {
                const common_json & c = s.at(k);
                for (size_t i = 0; i < c.size(); ++i) visit(c.at(i), ptr + "/" + k + "/" + std::to_string(i));
            }
        }
        static const char * maps[] = { "properties", "$defs", "definitions", "patternProperties", "dependentSchemas" };
        for (const char * k : maps) {
            if (s.contains(k) && s.at(k).is_object()) {
                for (const auto & [name, c] : s.at(k).items()) {
                    if (c.is_object()) visit(c, ptr + "/" + k + "/" + escape_pointer(name));
                }
            }
        }
    }

    // keywords other than annotations and `allowed`
    static std::vector<std::string> siblings(const common_json & s, const std::set<std::string> & allowed) {
        std::vector<std::string> v;
        for (const auto & [k, _] : s.items()) {
            if (!allowed.count(k) && !annotations().count(k)) v.push_back(k);
        }
        return v;
    }

    void visit(const common_json & s, const std::string & ptr) {
        if (!s.is_object()) {
            return;
        }
        if (depth >= max_depth) {
            if (!depth_reported) {
                depth_reported = true;
                add(ptr, "schema nesting exceeds maximum depth (" + std::to_string(max_depth) + "); nothing below this point was checked");
            }
            return;
        }
        struct level { int & d; explicit level(int & x) : d(x) { ++d; } ~level() { --d; } } scope(depth);
        children(s, ptr);

        std::vector<std::string> dropped;
        for (const auto & [k, _] : s.items()) {
            if (never_read().count(k)) dropped.push_back(k);
        }
        if (!dropped.empty()) {
            add(ptr, "keyword(s) " + join(dropped) + " would be ignored");
        }

        if (s.contains("$ref")) {
            auto sib = siblings(s, { "$ref" });
            if (!sib.empty()) add(ptr, "keyword(s) " + join(sib) + " next to $ref would be ignored");
            return; // visit() returns at the $ref before reading anything else
        }
        if (s.contains("oneOf")) {
            add(ptr, "oneOf would be enforced as anyOf (exclusivity is not checked)");
        }
        if (s.contains("oneOf") || s.contains("anyOf")) {
            auto sib = siblings(s, { "oneOf", "anyOf" });
            if (!sib.empty()) add(ptr, "keyword(s) " + join(sib) + " next to anyOf/oneOf would be ignored");
            return;
        }
        if (s.contains("const") || s.contains("enum")) {
            return; // exact values: nothing else can be lost
        }

        std::vector<std::string> types;
        if (s.contains("type") && s.at("type").is_array()) {
            for (size_t i = 0; i < s.at("type").size(); ++i) {
                if (s.at("type").at(i).is_string()) types.push_back(s.at("type").at(i).get<std::string>());
            }
        } else if (s.contains("type") && s.at("type").is_string()) {
            types.push_back(s.at("type").get<std::string>());
        } else {
            types.push_back(""); // untyped
        }
        for (const auto & t : types) {
            typed(s, ptr, t);
        }
    }

    void typed(const common_json & s, const std::string & ptr, const std::string & t) {
        const std::string tn = t.empty() ? "an untyped schema" : "type \"" + t + "\"";
        const bool obj_path = (t.empty() || t == "object") &&
                              (s.contains("properties") ||
                               (s.contains("additionalProperties") && !(s.at("additionalProperties").is_boolean() &&
                                                                        s.at("additionalProperties").get<bool>())));
        if ((t.empty() || t == "object") && s.contains("required") && !obj_path) {
            add(ptr, "required is only enforced when properties (or a non-true additionalProperties) is present");
        }
        if (obj_path) {
            if (s.contains("allOf")) add(ptr, "allOf next to properties would be ignored");
            return;
        }
        if (s.contains("allOf")) {
            if (!(t.empty() || t == "object" || t == "string")) {
                add(ptr, "allOf is not read for " + tn);
            } else if (s.at("allOf").is_array()) {
                const common_json & all = s.at("allOf");
                for (size_t i = 0; i < all.size(); ++i) {
                    const common_json & c = all.at(i);
                    const std::string cp = ptr + "/allOf/" + std::to_string(i);
                    if (!c.is_object() || c.contains("anyOf") || c.contains("$ref") || c.contains("enum")) continue;
                    if (!c.contains("properties")) {
                        add(cp, "allOf part without $ref / properties / enum would be ignored");
                        continue;
                    }
                    std::set<std::string> req;
                    if (c.contains("required") && c.at("required").is_array()) {
                        for (size_t j = 0; j < c.at("required").size(); ++j) {
                            if (c.at("required").at(j).is_string()) req.insert(c.at("required").at(j).get<std::string>());
                        }
                    }
                    for (const auto & [name, _] : c.at("properties").items()) {
                        if (!req.count(name)) {
                            add(cp, "allOf part: optional property \"" + name + "\" would be required");
                            break;
                        }
                    }
                }
            }
            return;
        }
        if (t.empty() || t == "array") {
            const bool has_items = s.contains("items") || s.contains("prefixItems");
            if (s.contains("items") && s.contains("prefixItems")) {
                add(ptr, "prefixItems would be ignored when items is also present");
            }
            const common_json * it = s.contains("items") ? &s.at("items") : (s.contains("prefixItems") ? &s.at("prefixItems") : nullptr);
            if (it && it->is_array() && (s.contains("minItems") || s.contains("maxItems"))) {
                add(ptr, "minItems/maxItems would be ignored for a tuple (array-valued items / prefixItems)");
            }
            if (!has_items && t == "array" && (s.contains("minItems") || s.contains("maxItems"))) {
                add(ptr, "minItems/maxItems are only enforced when items is present");
            }
            if (has_items) return;
        }
        if (t.empty() || t == "string") {
            // visit() order (:1107-1122): pattern, then a supported format, then minLength/maxLength (type "string" only);
            // the first one present is the whole rule
            const bool fmt     = s.contains("format");
            const bool fmt_ok  = fmt && s.at("format").is_string() && is_supported_format(s.at("format").get<std::string>());
            const bool lengths = s.contains("minLength") || s.contains("maxLength");
            std::vector<std::string> groups;
            if (s.contains("pattern")) groups.push_back("pattern");
            if (fmt_ok) groups.push_back("format");
            if (lengths && t == "string") groups.push_back("minLength/maxLength");
            if (groups.size() > 1) {
                add(ptr, "constraints " + join(groups) + " cannot be enforced together (only " + groups[0] + " is)");
            }
            if (fmt && !fmt_ok) {
                add(ptr, "string format " + s.at("format").dump() + " is not implemented (date, time, date-time, uuid only)");
            }
            if (t.empty() && lengths && !s.contains("pattern") && !fmt_ok) {
                add(ptr, "minLength/maxLength need \"type\": \"string\"");
            }
        }
        const bool bounds = s.contains("minimum") || s.contains("maximum") || s.contains("exclusiveMinimum") ||
                            s.contains("exclusiveMaximum");
        if (bounds && t != "integer" && t != "string" && t != "array" && t != "object" && t != "boolean" && t != "null") {
            add(ptr, "minimum/maximum bounds are only enforced for \"type\": \"integer\" (not " + tn + ")");
        }
        if (bounds && t == "integer") {
            if (s.contains("minimum") && s.contains("exclusiveMinimum")) add(ptr, "exclusiveMinimum is ignored when minimum is present");
            if (s.contains("maximum") && s.contains("exclusiveMaximum")) add(ptr, "exclusiveMaximum is ignored when maximum is present");
            for (const char * k : { "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum" }) {
                if (s.contains(k) && !s.at(k).is_number_integer()) {
                    add(ptr, std::string(k) + " must be an integer for \"type\": \"integer\" (a fraction would be truncated)");
                }
            }
        }
    }
};

} // namespace json_schema_strict_detail

// One message per dropped or weakened constraint, each naming the JSON pointer of its subschema; empty = the
// grammar enforces everything the schema asks for (as far as the list above knows).
inline std::vector<std::string> json_schema_unsupported(const common_json & schema) {
    json_schema_strict_detail::walker w;
    w.visit(schema, "#");
    return w.out;
}
