// SCOUT-NA: measurement probe inside the fork tree, no new tool.
// Stack-depth probe for the JSON-schema depth cap (row S69-ENGREV-UPSTREAM-MERGE-SCHEMA-DEPTH-CAP-MEASURE-2026-11-04-01).
// This executable compiles common/json-schema.cpp and common/json-schema-to-grammar.cpp itself with
// COMMON_CHAT_SCHEMA_MAX_DEPTH_VALUE raised (tests/CMakeLists.txt), so the cap does NOT stop the recursion and the real
// stack use at a given depth is measured. One run = one (kind, depth, where): a stack overflow kills the process
// (0xC00000FD), the driver tests/probe-schema-depth.py reads the exit code.
//   probe-schema-depth <array|object|anyof> <depth> <main|thread> <schema|tmpl>
// exit 0 = completed (prints OK or REFUSED:<msg>), exit 3 = a non-invalid_argument C++ exception, crash = overflow.
#include "chat.h"
#include "json-schema-to-grammar.h"
#include "json-schema.h"
#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

static std::string nest(const std::string & kind, int n) {
    std::string s;
    if (kind == "array") {
        for (int i = 0; i < n; i++) { s += R"({"type":"array","items":)"; }
        s += R"({"type":"integer"})";
        for (int i = 0; i < n; i++) { s += "}"; }
    } else if (kind == "anyof") {
        for (int i = 0; i < n; i++) { s += R"({"anyOf":[{"type":"integer"},)"; }
        s += R"({"type":"string"})";
        for (int i = 0; i < n; i++) { s += "]}"; }
    } else {
        for (int i = 0; i < n; i++) { s += R"({"type":"object","properties":{"a":)"; }
        s += R"({"type":"integer"})";
        for (int i = 0; i < n; i++) { s += R"(},"required":["a"],"additionalProperties":false})"; }
    }
    return s;
}

static std::string read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int g_rc = 0;

static void run(const std::string & kind, int depth, const std::string & stage) {
    try {
        const std::string schema = nest(kind, depth);
        if (stage == "schema") {
            auto doc = common_chat_schema_from_json(common_json::parse(schema));
            std::string g1 = json_schema_to_grammar(doc);
            std::string g2 = json_schema_to_grammar(common_json::parse(schema), true);
            printf("OK grammar_bytes=%zu/%zu\n", g1.size(), g2.size());
        } else {
            auto tmpls = common_chat_templates_ptr(common_chat_templates_init(nullptr, read_file("models/templates/Qwen3.5-4B.jinja")));
            common_chat_templates_inputs in;
            common_chat_msg m;
            m.role = "user";
            m.content = "hi";
            in.messages = { m };
            in.tools    = { { "f", "", R"({"type":"object","properties":{"x":)" + schema + "}}" } };
            auto p = common_chat_templates_apply(tmpls.get(), in);
            printf("OK prompt_bytes=%zu grammar_bytes=%zu\n", p.prompt.str().size(), p.grammar.size());
        }
    } catch (const std::invalid_argument & e) {
        printf("REFUSED:%s\n", e.what());
    } catch (const std::exception & e) {
        printf("EXC:%s\n", e.what());
        g_rc = 3;
    }
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: kind depth main|thread schema|tmpl\n"); return 2; }
    const std::string kind = argv[1], where = argv[3], stage = argv[4];
    const int depth = atoi(argv[2]);
    if (where == "thread") {
        // std::thread on MSVC takes the executable's PE stack reserve (1 MiB default) -- the same as an httplib pool worker
        std::thread t([&] { run(kind, depth, stage); });
        t.join();
    } else {
        run(kind, depth, stage);
    }
    return g_rc;
}
