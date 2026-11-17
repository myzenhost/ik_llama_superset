// Unit test for server-overrides.h (operator sampler-override layers). Model-free; links llama-common for the JSON class.
// RESEARCH-DONE: behaviour spec from tabbyAPI common/sampling.py (see server-overrides.h SOURCE-READ).
#include "server-overrides.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static server_override_groups make_groups() {
    server_override_groups g;
    const std::vector<std::vector<std::string>> fields = {
        {"n_predict", "max_tokens", "max_completion_tokens"},
        {"temperature"},
        {"top_k"},
        {"stop"},
        {"logit_bias"},
    };
    for (const auto & f : fields) {
        for (const auto & n : f) {
            g[n] = f;
        }
    }
    return g;
}

static server_override_layer parse_ok(const char * text, const server_override_groups & g, std::vector<std::string> * warn = nullptr) {
    server_override_layer l;
    std::string err;
    std::vector<std::string> w;
    const bool ok = server_override_parse(common_json::parse(text), g, "test", l, err, w);
    if (!ok) {
        std::printf("unexpected parse error: %s\n", err.c_str());
    }
    CHECK(ok);
    if (warn) {
        *warn = w;
    }
    return l;
}

static bool parse_fails(const char * text, const server_override_groups & g, std::string * why = nullptr) {
    server_override_layer l;
    std::string err;
    std::vector<std::string> w;
    const bool ok = server_override_parse(common_json::parse(text), g, "test", l, err, w);
    CHECK(l.empty()); // a refused layer leaves nothing behind
    if (why) {
        *why = err;
    }
    return !ok;
}

int main() {
    const auto groups = make_groups();

    // ---- control: no layer, body untouched ----------------------------------------------------------------------------
    {
        const auto body = common_json::parse(R"({"max_tokens":64,"temperature":0.2})");
        const auto out  = server_override_apply(body, server_override_layer{}, groups);
        CHECK(out.dump() == body.dump());
    }

    // ---- default: only when the request omits the field (null counts as omitted) --------------------------------------
    {
        const auto l = parse_ok(R"({"temperature":{"override":0.7},"top_k":{"override":40}})", groups);
        server_override_trace tr;
        const auto out = server_override_apply(common_json::parse(R"({"temperature":0.2,"top_k":null})"), l, groups, &tr);
        CHECK(out.at("temperature").template get<double>() == 0.2); // the request keeps its own value
        CHECK(out.at("top_k").template get<int>() == 40);           // null -> default fills in
        bool saw_req = false, saw_def = false;
        for (const auto & t : tr) {
            if (t.first == "temperature" && t.second == "request") saw_req = true;
            if (t.first == "top_k" && t.second == "default") saw_def = true;
        }
        CHECK(saw_req && saw_def);
        const auto out2 = server_override_apply(common_json::parse("{}"), l, groups);
        CHECK(out2.at("temperature").template get<double>() == 0.7);
    }

    // ---- force: replaces the request value, and every alias of the field ---------------------------------------------
    {
        const auto l = parse_ok(R"({"n_predict":{"override":16,"force":true}})", groups);
        // the client uses an alias: the forced value must win, and the alias must be gone or the schema would read it
        const auto out = server_override_apply(common_json::parse(R"({"max_tokens":64,"max_completion_tokens":99})"), l, groups);
        CHECK(out.at("n_predict").template get<int>() == 16);
        CHECK(!out.contains("max_tokens"));
        CHECK(!out.contains("max_completion_tokens"));
        // same with the canonical name in the request
        const auto out2 = server_override_apply(common_json::parse(R"({"n_predict":64})"), l, groups);
        CHECK(out2.at("n_predict").template get<int>() == 16);
        // an operator may key the entry by an alias; it is stored under the canonical name
        const auto l2 = parse_ok(R"({"max_tokens":{"override":8,"force":true}})", groups);
        CHECK(l2.count("n_predict") == 1 && l2.count("max_tokens") == 0);
    }

    // ---- force of a falsy value is applied (tabbyAPI silently ignores it: SOURCE-NUANCE 1) ----------------------------
    {
        const auto l = parse_ok(R"({"temperature":{"override":0,"force":true}})", groups);
        const auto out = server_override_apply(common_json::parse(R"({"temperature":0.9})"), l, groups);
        CHECK(out.at("temperature").template get<double>() == 0.0);
    }

    // ---- additive: operator list first, then the request's, no de-duplication ----------------------------------------
    {
        const auto l = parse_ok(R"({"stop":{"override":["###","X"],"additive":true}})", groups);
        server_override_trace tr;
        const auto out = server_override_apply(common_json::parse(R"({"stop":["Y","X"]})"), l, groups, &tr);
        CHECK(out.at("stop").dump() == R"(["###","X","Y","X"])");
        CHECK(tr.size() == 1 && tr[0].second == "additive");
        // no list in the request: the operator's list is used as it is
        const auto out2 = server_override_apply(common_json::parse("{}"), l, groups);
        CHECK(out2.at("stop").dump() == R"(["###","X"])");
        // a request value that is not a list is left alone (not merged, not replaced)
        const auto out3 = server_override_apply(common_json::parse(R"({"stop":"Y"})"), l, groups);
        CHECK(out3.at("stop").dump() == R"("Y")");
    }

    // ---- force beats additive for the same key ------------------------------------------------------------------------
    {
        const auto l = parse_ok(R"({"stop":{"override":["A"],"force":true,"additive":true}})", groups);
        const auto out = server_override_apply(common_json::parse(R"({"stop":["Y"]})"), l, groups);
        CHECK(out.at("stop").dump() == R"(["A"])");
    }

    // ---- layers: the model layer replaces the same key of the global layer --------------------------------------------
    {
        server_override_state st;
        st.set_global(parse_ok(R"({"temperature":{"override":1.0,"force":true},"top_k":{"override":50}})", groups));
        st.set_model(parse_ok(R"({"temperature":{"override":0.6,"force":true}})", groups));
        const auto eff = st.effective();
        CHECK(eff.at("temperature").value.template get<double>() == 0.6);
        CHECK(eff.at("top_k").value.template get<int>() == 50);
        const auto out = server_override_apply(common_json::parse(R"({"temperature":2})"), eff, groups);
        CHECK(out.at("temperature").template get<double>() == 0.6);
        CHECK(!st.empty());
        st.set_model({});
        CHECK(st.effective().at("temperature").value.template get<double>() == 1.0); // the model layer left with the model
        st.set_global({});
        CHECK(st.empty());
        const auto js = st.to_json();
        CHECK(js.contains("global") && js.contains("model") && js.contains("effective"));
    }

    // ---- validation: unknown key warns at load, malformed layers are refused whole -----------------------------------------
    {
        std::vector<std::string> w;
        const auto l = parse_ok(R"({"temperature":{"override":0.5},"not_a_field":{"override":1}})", groups, &w);
        CHECK(l.size() == 1 && w.size() == 1);
        CHECK(w[0].find("not_a_field") != std::string::npos);
        CHECK(parse_fails(R"([1,2])", groups));
        CHECK(parse_fails(R"({"temperature":0.5})", groups));                                   // not an entry object
        CHECK(parse_fails(R"({"temperature":{"force":true}})", groups));                        // no override key
        CHECK(parse_fails(R"({"temperature":{"override":null}})", groups));                     // null override
        CHECK(parse_fails(R"({"temperature":{"override":1,"force":"yes"}})", groups));          // flag not a bool
        std::string why;
        CHECK(parse_fails(R"({"stream":{"override":true,"force":true}})", groups, &why));       // read before the schema
        CHECK(why.find("stream") != std::string::npos);
        CHECK(parse_fails(R"({"max_tokens":{"override":8},"n_predict":{"override":9}})", groups)); // two names, one field
        // a valid entry next to a broken one: the whole layer is refused, nothing is half-applied
        CHECK(parse_fails(R"({"temperature":{"override":0.5},"top_k":{"override":null}})", groups));
        // null empties a layer
        server_override_layer l2;
        std::string err;
        std::vector<std::string> w2;
        CHECK(server_override_parse(common_json::parse("null"), groups, "test", l2, err, w2) && l2.empty());
        // additive on a non-list value is inert: it behaves as a default and says so
        std::vector<std::string> w3;
        parse_ok(R"({"temperature":{"override":0.5,"additive":true}})", groups, &w3);
        CHECK(w3.size() == 1);
    }

    // ---- a non-object body is returned as it is ------------------------------------------------------------------------
    {
        const auto l = parse_ok(R"({"temperature":{"override":0.7}})", groups);
        const auto out = server_override_apply(common_json::parse("[1]"), l, groups);
        CHECK(out.is_array());
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("All tests passed\n");
    return 0;
}
