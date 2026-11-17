// Row S69-SGLANG-PICK-STRICT-JSON-SCHEMA-CHECK-2026-10-18-01: common/json-schema-strict.h lists exactly the
// constraints json_schema_to_grammar() drops or weakens (list derived from json-schema-to-grammar.cpp visit()),
// with a JSON pointer per subschema, and stays silent for schemas the converter enforces in full.

#include "json-schema-strict.h"
#include "json-schema-to-grammar.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<std::string> check(const char * s) {
    return json_schema_unsupported(common_json::parse(s));
}

static bool has(const std::vector<std::string> & v, const std::string & needle) {
    for (const auto & m : v) {
        if (m.find(needle) != std::string::npos) return true;
    }
    return false;
}

static void expect_clean(const char * s) {
    auto v = check(s);
    if (!v.empty()) {
        std::fprintf(stderr, "expected clean: %s\n  got: %s\n", s, v[0].c_str());
    }
    assert(v.empty());
    // and the converter accepts it
    (void) json_schema_to_grammar(common_json::parse(s));
}

static void expect(const char * s, const std::string & needle) {
    auto v = check(s);
    if (!has(v, needle)) {
        std::fprintf(stderr, "expected '%s' for %s; got %zu message(s)%s%s\n", needle.c_str(), s, v.size(),
                     v.empty() ? "" : ": ", v.empty() ? "" : v[0].c_str());
    }
    assert(has(v, needle));
}

int main() {
    // enforced in full: no messages
    expect_clean(R"({"type": "object", "properties": {"a": {"type": "integer", "minimum": 0, "maximum": 9},
                     "b": {"type": "string", "minLength": 1, "maxLength": 4}, "c": {"enum": [1, 2]},
                     "d": {"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 3}},
                     "required": ["a"], "additionalProperties": false, "title": "t", "description": "x"})");
    expect_clean(R"({"type": "string", "format": "date-time"})");
    expect_clean(R"({"type": "string", "pattern": "^[a-z]+$"})");
    expect_clean(R"({"anyOf": [{"type": "string"}, {"type": "null"}]})");
    expect_clean(R"({"$defs": {"x": {"type": "integer"}}, "$ref": "#/$defs/x"})");
    expect_clean(R"({"type": "string", "format": "uuid4"})");
    // instance data is never walked as a schema
    expect_clean(R"({"const": {"not": 1, "multipleOf": 2}})");

    // never read by the converter
    expect(R"({"type": "integer", "multipleOf": 7})", "#: keyword(s) multipleOf would be ignored");
    expect(R"({"type": "array", "items": {"type": "integer"}, "uniqueItems": true})", "uniqueItems");
    expect(R"({"not": {"type": "string"}})", "not");
    expect(R"({"type": "object", "properties": {"a": {"type": "string"}}, "minProperties": 1})", "minProperties");
    // nested: pointer names the subschema, with ~1 escaping for '/'
    expect(R"({"type": "object", "properties": {"a/b": {"type": "integer", "multipleOf": 3}}})",
           "#/properties/a~1b: keyword(s) multipleOf");
    expect(R"({"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string", "format": "email"}}}})",
           "#/items/properties/x: string format \"email\" is not implemented");
    // weakened, not dropped
    expect(R"({"oneOf": [{"type": "integer"}, {"type": "number"}]})", "oneOf would be enforced as anyOf");
    expect(R"({"$ref": "#/$defs/x", "maxLength": 3, "$defs": {"x": {"type": "string"}}})", "maxLength next to $ref");
    expect(R"({"type": "number", "minimum": 0.5})", "only enforced for \"type\": \"integer\"");
    expect(R"({"type": "integer", "minimum": 0, "exclusiveMinimum": 0})", "exclusiveMinimum is ignored");
    expect(R"({"type": "integer", "minimum": 0.5})", "must be an integer");
    expect(R"({"type": "string", "pattern": "^a+$", "maxLength": 3})", "cannot be enforced together");
    expect(R"({"type": "string", "format": "hostname"})", "not implemented");
    expect(R"({"type": "array", "prefixItems": [{"type": "integer"}], "minItems": 2})", "ignored for a tuple");
    expect(R"({"type": "object", "required": ["a"]})", "required is only enforced");
    expect(R"({"allOf": [{"type": "object", "properties": {"a": {"type": "string"}}}]})", "optional property \"a\" would be required");
    expect(R"({"type": "integer", "allOf": [{"minimum": 3}]})", "allOf is not read for type \"integer\"");

    // Row S69-ENGREV-LOCALAI2-SCHEMA-ALLOF-REF-CYCLE-DEPTH-GUARD-2026-10-27-01: the walker recurses once per schema
    // level; a client schema nested past the cap is reported (one message, never a stack overflow), a shallower one
    // is still walked to the bottom.
    {
        auto nest = [](int layers) {
            std::string s;
            for (int i = 0; i < layers; i++) s += R"({"type":"array","items":)";
            s += R"({"type":"integer","multipleOf":3})";
            for (int i = 0; i < layers; i++) s += "}";
            return s;
        };
        auto shallow = check(nest(32).c_str());
        assert(shallow.size() == 1 && has(shallow, "keyword(s) multipleOf"));  // reached the innermost subschema
        auto deep = check(nest(400).c_str());
        assert(has(deep, "maximum depth"));
        auto absurd = check(nest(20000).c_str());
        assert(has(absurd, "maximum depth"));
    }

    std::printf("test-json-schema-strict: all passed\n");
    return 0;
}
