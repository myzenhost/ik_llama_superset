#include "tests.h"

#include "json-schema.h"
#include "json-schema-to-grammar.h"

void test_json_serialization(testing &t) {
    auto original = build_peg_parser([](common_peg_parser_builder & p) {
        return "<tool_call>" + p.json() + "</tool_call>";
    });

    auto json_serialized = original.to_json().dump();

    t.test("compare before/after", [&](testing &t) {
        auto deserialized = common_peg_arena::from_json(common_json::parse(json_serialized));

        // Test complex JSON
        std::string input = R"({"name": "test", "values": [1, 2, 3], "nested": {"a": true}})";
        common_peg_parse_context ctx1(input);
        common_peg_parse_context ctx2(input);

        auto result1 = original.parse(ctx1);
        auto result2 = deserialized.parse(ctx2);

        t.assert_equal("both_succeed", result1.success(), result2.success());
        t.assert_equal("same_end_pos", result1.end, result2.end);
    });

    t.bench("deserialize", [&]() {
        auto deserialized = common_peg_arena::from_json(common_json::parse(json_serialized));
    }, 100);

    // row S69-ENGREV-UPSTREAM-MERGE-PEG-SCHEMA-SUBNODE-SAVE-2026-11-04-01:
    // a schema parser bound to a sub-schema node must survive save/load.
    t.test("schema sub-node save/load", [](testing &t) {
        auto schema = common_json::parse(R"({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"array","items":{"type":"integer"}}},"required":["a","b"]})");
        auto doc = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(schema));
        auto * obj = dynamic_cast<const common_chat_schema_object *>(doc->root.get());
        t.assert_equal("root is object", true, obj != nullptr);
        if (!obj) { return; }

        auto gbnf_of = [](const common_peg_arena & a) {
            return build_grammar([&](const common_grammar_builder & b) { a.build_grammar(b); });
        };

        std::vector<std::string> gbnfs;
        for (size_t i = 0; i < obj->properties.size(); i++) {
            const common_chat_schema * node = obj->properties[i].schema.get();
            auto original = build_peg_parser([&](common_peg_parser_builder & p) {
                return p.schema(p.json(), "sub", doc, *node);
            });
            std::string saved;
            try {
                saved = original.save();
            } catch (const std::exception & e) {
                t.assert_equal("save does not throw (" + obj->properties[i].name + "): " + e.what(), true, false);
                continue;
            }
            common_peg_arena loaded;
            loaded.load(saved);
            const std::string g1 = gbnf_of(original);
            const std::string g2 = gbnf_of(loaded);
            t.assert_equal("grammar identical after load (" + obj->properties[i].name + ")", g1, g2);
            // the "rules" map is unordered, so compare the parsers array, which carries schema_node
            t.assert_equal("save is stable (" + obj->properties[i].name + ")",
                common_json::parse(saved)["parsers"].dump(), common_json::parse(loaded.save())["parsers"].dump());
            gbnfs.push_back(g2);
        }
        // the two sub-nodes are different schemas: a loader that fell back to the root would collapse them
        if (gbnfs.size() == 2) {
            t.assert_equal("sub-node grammars differ", true, gbnfs[0] != gbnfs[1]);
        }
    });
}
