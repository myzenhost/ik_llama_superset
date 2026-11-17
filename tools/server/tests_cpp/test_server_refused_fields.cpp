// Unit test for server-refused-fields.h: output-changing request fields the server does not implement are refused by name.
// Model-free; links llama-common for the JSON class.
// RESEARCH-DONE: behaviour spec from Ferrox crates/frink-server/src/unimplemented_fields.rs (see the SOURCE-READ in the header).
#include "server-refused-fields.h"

#include "json.h"

#include <cstdio>
#include <set>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // the table pins the exact set of refused names: adding or removing one is a deliberate edit of this list too
    const std::set<std::string> expect = {
        "truncate_prompt_tokens", "prompt_embeds", "guided_json", "guided_regex", "guided_choice", "guided_grammar",
        "guided_decoding_backend", "structured_outputs", "logits_processors",
    };
    std::set<std::string> have;
    for (const auto & e : server_refused_field_table()) {
        have.insert(e.name);
    }
    CHECK(have == expect);

    // every name in the table is refused, with the name in the message, whatever the value
    for (const auto & name : expect) {
        for (const char * val : {"1", "\"x\"", "{}", "[]", "true", "0"}) {
            common_json body = common_json::parse(std::string("{\"prompt\":\"hi\",\"") + name + "\":" + val + "}");
            const auto r = server_refuse_unimplemented_field(body);
            CHECK(r.refused);
            CHECK(r.name == name);
            CHECK(r.message.find("`" + name + "`") != std::string::npos);
            CHECK(r.message.find("not implemented") != std::string::npos);
        }
    }

    // null is absent; an empty body and a body of implemented fields pass
    for (const auto & name : expect) {
        common_json body = common_json::parse(std::string("{\"") + name + "\":null}");
        CHECK(!server_refuse_unimplemented_field(body).refused);
    }
    CHECK(!server_refuse_unimplemented_field(common_json::parse("{}")).refused);
    CHECK(!server_refuse_unimplemented_field(common_json::parse(
        "{\"prompt\":\"hi\",\"n\":1,\"echo\":false,\"prompt_logprobs\":true,\"allowed_token_ids\":[1,2],\"bad_words\":[\"x\"],"
        "\"use_beam_search\":false,\"json_schema\":{},\"response_format\":{\"type\":\"json_object\"},\"cache_salt\":\"s\"}")).refused);
    CHECK(!server_refuse_unimplemented_field(common_json::parse("[1,2]")).refused); // not an object

    // the hint is appended when the table has one, and not when it has none
    CHECK(server_refuse_unimplemented_field(common_json::parse("{\"guided_json\":{}}")).message.find("json_schema") != std::string::npos);
    CHECK(server_refuse_unimplemented_field(common_json::parse("{\"prompt_embeds\":\"AA==\"}")).message.find("; ") == std::string::npos);

    // best_of and suffix are NOT in this table (/v1/completions refuses them itself; the TGI route serves best_of)
    CHECK(!server_refuse_unimplemented_field(common_json::parse("{\"best_of\":2,\"suffix\":\"x\"}")).refused);

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("test-server-refused-fields: ok\n");
    return 0;
}
