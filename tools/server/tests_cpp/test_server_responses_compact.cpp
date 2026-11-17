// Unit test for server-responses-compact.h (Responses compaction). Model-free; links llama-common for the JSON class.
// RESEARCH-DONE: behaviour spec from Ollama openai/responses_compact.go (see the header).
#include "server-responses-compact.h"

#include <cstdio>

using respcompact::json;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static json msg(const char * role, const char * text) {
    json m = json::object();
    m["type"]    = "message";
    m["role"]    = role;
    m["content"] = text;
    return m;
}

static json call(const char * id, const char * name) {
    json c = json::object();
    c["type"]      = "function_call";
    c["call_id"]   = id;
    c["name"]      = name;
    c["arguments"] = "{}";
    return c;
}

static json output(const char * id, const char * text) {
    json c = json::object();
    c["type"]    = "function_call_output";
    c["call_id"] = id;
    c["output"]  = text;
    return c;
}

// a Responses object whose single output is a create_summary call
static json summary_response(const std::string & args) {
    json c = call("c1", "create_summary");
    c["arguments"] = args;
    json r = json::object();
    r["output"] = json::array();
    r["output"].push_back(c);
    json u = json::object();
    u["input_tokens"] = 7;
    r["usage"] = u;
    return r;
}

static respcompact::plan make_plan(const json & items, std::string & err) {
    json body = json::object();
    body["model"] = "m";
    body["input"] = items;
    respcompact::plan p;
    bool ok = respcompact::prepare_standalone(body, p, err);
    CHECK(ok);
    return p;
}

int main() {
    // ---- plan: refs, active tool state is forced ----
    {
        json items = json::array();
        items.push_back(msg("user", "hello"));                 // item_000001
        items.push_back(call("a", "run"));                     // 2  finished pair, then assistant text
        items.push_back(output("a", "ok"));                    // 3
        items.push_back(msg("assistant", "done a"));           // 4
        items.push_back(call("b", "run"));                     // 5  no output yet: active
        std::string err;
        respcompact::plan p = make_plan(items, err);
        CHECK(p.items.size() == 5);
        CHECK(p.items[0].ref == "item_000001");
        CHECK(p.forced.count("item_000005") == 1);
        CHECK(p.forced.count("item_000002") == 0);
        CHECK(p.groups.size() == 2);
    }

    // ---- prepare: validation ----
    {
        respcompact::plan p;
        std::string err;
        json body = json::object();
        CHECK(!respcompact::prepare_standalone(body, p, err));          // no input
        body["input"] = "hi";
        CHECK(!respcompact::prepare_standalone(body, p, err));          // no model
        CHECK(err.find("model") != std::string::npos);
        body["model"] = "m";
        CHECK(respcompact::prepare_standalone(body, p, err));           // string input = one user message
        CHECK(p.items.size() == 1);

        json trig = json::object();
        trig["type"] = "compaction_trigger";
        json items = json::array();
        items.push_back(trig);
        body["input"] = items;
        CHECK(!respcompact::prepare_standalone(body, p, err));          // trigger is not valid on /compact
        CHECK(err.find("compaction_trigger") != std::string::npos);
    }

    // ---- trigger form ----
    {
        json trig = json::object();
        trig["type"] = "compaction_trigger";
        json items = json::array();
        items.push_back(msg("user", "hello"));
        json body = json::object();
        body["model"] = "m";
        body["input"] = items;
        respcompact::plan p;
        bool requested = true;
        std::string err;
        CHECK(respcompact::prepare_triggered(body, p, requested, err)); // ordinary request: left alone
        CHECK(!requested);

        items.push_back(trig);
        body["input"] = items;
        CHECK(!respcompact::prepare_triggered(body, p, requested, err)); // trigger without stream
        CHECK(requested);
        CHECK(err.find("stream") != std::string::npos);

        body["stream"] = true;
        err.clear();
        CHECK(respcompact::prepare_triggered(body, p, requested, err));
        CHECK(requested);
        CHECK(p.items.size() == 1);

        json items2 = json::array();
        items2.push_back(trig);
        items2.push_back(msg("user", "x"));
        items2.push_back(trig);
        body["input"] = items2;
        CHECK(!respcompact::prepare_triggered(body, p, requested, err)); // a second trigger
    }

    // ---- complete: validation and the retained set ----
    {
        json items = json::array();
        items.push_back(msg("user", "hello"));
        items.push_back(call("a", "run"));
        items.push_back(output("a", "ok"));
        items.push_back(msg("assistant", "done"));
        items.push_back(call("b", "run"));
        std::string err;
        respcompact::plan p = make_plan(items, err);
        json item, usage;

        CHECK(!respcompact::complete(p, json::object(), item, usage, err));
        json empty = json::object();
        empty["output"] = json::array();
        CHECK(!respcompact::complete(p, empty, item, usage, err));
        CHECK(err.find("expected one") != std::string::npos);
        CHECK(!respcompact::complete(p, summary_response("not json"), item, usage, err));
        CHECK(!respcompact::complete(p, summary_response("{\"summary\":\"  \"}"), item, usage, err));
        CHECK(!respcompact::complete(p, summary_response("{\"summary\":\"s\",\"retain_item_ids\":[\"item_999999\"]}"), item, usage, err));
        CHECK(err.find("unknown") != std::string::npos);
        CHECK(!respcompact::complete(p, summary_response("{\"summary\":\"s\",\"retain_item_ids\":[\"item_000001\",\"item_000001\"]}"), item, usage, err));
        CHECK(err.find("duplicate") != std::string::npos);

        // retaining the finished call drags its output; the active call is always kept
        CHECK(respcompact::complete(p, summary_response("{\"summary\":\"the summary\",\"retain_item_ids\":[\"item_000002\"]}"), item, usage, err));
        CHECK(item["type"] == "compaction");
        json payload = json::parse(item["encrypted_content"].get<std::string>());
        CHECK(payload["type"] == "llama_compaction");
        CHECK(payload["summary"] == "the summary");
        CHECK(payload["retained"].size() == 3);   // call a, output a, active call b
        CHECK(usage["input_tokens"] == 7);

        // no model-chosen ids still keeps the forced active state
        CHECK(respcompact::complete(p, summary_response("{\"summary\":\"s\"}"), item, usage, err));
        payload = json::parse(item["encrypted_content"].get<std::string>());
        CHECK(payload["retained"].size() == 1);
    }

    // ---- expansion of the newest compaction item ----
    {
        json items = json::array();
        items.push_back(msg("user", "hello"));
        std::string err;
        respcompact::plan p = make_plan(json::array({ msg("user", "q") }), err);
        // build an item through complete() so the round trip is real
        items = json::array();
        items.push_back(msg("user", "q"));
        items.push_back(call("k", "run"));
        p = make_plan(items, err);
        json item, usage;
        CHECK(respcompact::complete(p, summary_response("{\"summary\":\"S1\",\"retain_item_ids\":[\"item_000001\"]}"), item, usage, err));

        json in = json::array();
        in.push_back(msg("user", "old"));
        in.push_back(item);
        in.push_back(msg("user", "after"));
        json out;
        bool changed = false;
        CHECK(respcompact::expand_items(in, out, changed, err));
        CHECK(changed);
        // [summary call, summary output, retained x2 (user q + active call k), user after]
        CHECK(out.size() == 5);
        CHECK(out[0]["type"] == "function_call");
        CHECK(out[0]["name"] == "llama_compaction_summary");
        CHECK(out[1]["type"] == "function_call_output");
        CHECK(out[1]["output"] == "S1");
        CHECK(out[0]["call_id"] == out[1]["call_id"]);
        CHECK(out[4]["content"] == "after");

        // the synthetic pair is recognised on a later round and never counted as a user tool call
        respcompact::plan p2 = make_plan(out, err);
        CHECK(p2.groups.size() == 1);   // only the retained call k

        // a foreign blob is refused by name
        json bad = json::object();
        bad["type"] = "compaction";
        bad["encrypted_content"] = "gAAAA-not-ours";
        json in2 = json::array();
        in2.push_back(bad);
        CHECK(!respcompact::expand_items(in2, out, changed, err));
        CHECK(err.find("unsupported") != std::string::npos);

        // two compaction items: only the newest expands; everything before it (older item, "mid") is dropped
        // result = summary call + output + 2 retained = 4
        json in3 = json::array();
        in3.push_back(item);
        in3.push_back(msg("user", "mid"));
        in3.push_back(item);
        CHECK(respcompact::expand_items(in3, out, changed, err));
        CHECK(out.size() == 4);

        // no compaction item: untouched
        json in4 = json::array();
        in4.push_back(msg("user", "plain"));
        CHECK(respcompact::expand_items(in4, out, changed, err));
        CHECK(!changed);
        CHECK(out.size() == 1);
    }

    // ---- trimming ----
    {
        json items = json::array();
        items.push_back(msg("user", "first"));
        for (int i = 0; i < 6; i++) {
            std::string id = "t" + std::to_string(i);
            items.push_back(call(id.c_str(), "run"));
            items.push_back(output(id.c_str(), "output output output output output"));
            items.push_back(msg("assistant", "assistant text assistant text assistant text"));
        }
        items.push_back(msg("user", "last"));
        std::string err;
        respcompact::plan p = make_plan(items, err);
        const size_t before = p.items.size();
        const int n = respcompact::trim_for_context(p);
        CHECK(n > 0);
        CHECK(p.items.size() == before - (size_t) n);
        CHECK(p.omitted == n);
        CHECK(p.items.front().kind == "message");           // the first user message stays
        CHECK(p.items.back().item["content"] == "last");    // so does the last item
        // every remaining call still has its output (pairs go together)
        for (const auto & g : p.groups) {
            CHECK(!g.result_ref.empty());
        }
        // the summary prompt names the omission
        json req = respcompact::summary_request(p, "");
        CHECK(req["input"][0]["content"].get<std::string>().find("omitted") != std::string::npos);
    }

    // ---- the summarising request ----
    {
        json items = json::array();
        items.push_back(msg("user", "hello"));
        std::string err;
        respcompact::plan p = make_plan(items, err);
        json req = respcompact::summary_request(p, "");
        CHECK(req["stream"] == false);
        CHECK(req["tool_choice"]["name"] == "create_summary");
        CHECK(req["tools"][0]["name"] == "create_summary");
        CHECK(req["input"].size() == 2);                    // system prompt + one transcript item
        json req2 = respcompact::summary_request(p, "bad ids");
        CHECK(req2["input"][0]["content"].get<std::string>().find("bad ids") != std::string::npos);
    }

    // ---- responses ----
    {
        json item = json::object();
        item["type"] = "compaction";
        item["encrypted_content"] = "x";
        json usage = json::object();
        usage["total_tokens"] = 3;
        json r = respcompact::compacted_response("resp_1", item, usage);
        CHECK(r["object"] == "response.compaction");
        CHECK(r["output"].size() == 1);
        const std::string sse = respcompact::render_sse("resp_1", "m", item, usage);
        CHECK(sse.find("event: response.created\n") == 0);
        CHECK(sse.find("event: response.output_item.done") != std::string::npos);
        CHECK(sse.find("event: response.completed") != std::string::npos);
        size_t n = 0;
        for (size_t pos = sse.find("\"type\":\"compaction\""); pos != std::string::npos; pos = sse.find("\"type\":\"compaction\"", pos + 1)) {
            n++;
        }
        CHECK(n == 3);  // added, done, completed.output
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("test-server-responses-compact: all passed\n");
    return 0;
}
