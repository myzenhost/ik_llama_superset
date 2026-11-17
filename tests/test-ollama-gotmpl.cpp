// Differential test: Go template -> Jinja translator (tools/server/server-ollama-gotmpl.h) against ollama's OWN template engine.
// Fixture tests/ollama-engine/gotmpl_oracle.json holds {template, messages, tools, expected} produced by ollama's template.Execute
// (generator: tests/ollama-engine/gen_gotmpl_oracle.go). The translated Jinja is rendered with the fork's Jinja runtime and must match
// ollama's output byte for byte. usage: test-ollama-gotmpl <gotmpl_oracle.json>
#include "server-ollama-gotmpl.h"
#include "json.h"
using json = common_json;
#include "jinja/runtime.h"
#include "jinja/parser.h"
#include "jinja/lexer.h"
#include "jinja/utils.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::string render(const std::string & tmpl, const json & vars) {
    jinja::lexer lexer;
    auto lexer_res = lexer.tokenize(tmpl);
    jinja::program ast = jinja::parse_from_tokens(lexer_res);
    jinja::context ctx(tmpl);
    jinja::global_from_json(ctx, vars, true);
    jinja::runtime runtime(ctx);
    return runtime.gather_string_parts(runtime.execute(ast))->as_string().str();
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: test-ollama-gotmpl gotmpl_oracle.json\n"); return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::printf("cannot open %s\n", argv[1]); return 2; }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    json cases = json::parse(text);
    int n = 0, pass = 0, refused = 0, shown = 0;
    for (size_t i = 0; i < cases.size(); i++) {
        const json & c = cases[i];
        const std::string name = c.value("name", std::string());
        std::string jinja, err;
        ++n;
        if (!ollama_gotmpl::translate(c.value("template", std::string()), jinja, err)) {
            ++refused;
            CHECK(false, "%s: translator refused: %s", name.c_str(), err.c_str());
            continue;
        }
        json vars = json::object();
        vars["messages"] = c.at("messages");
        if (c.contains("tools")) vars["tools"] = c.at("tools");
        std::string got;
        try { got = render(jinja, vars); } catch (const std::exception & e) { got = std::string("<<EXC ") + e.what() + ">>"; }
        const std::string want = c.value("expected", std::string());
        if (got == want) { ++pass; continue; }
        ++g_fail;
        if (shown++ < 12) {
            std::printf("MISMATCH %s\n  want: %s\n  got : %s\n", name.c_str(), json(want).dump().c_str(), json(got).dump().substr(0, 400).c_str());
            if (shown <= 3) std::printf("  jinja: %s\n", jinja.substr(0, 1500).c_str());
        }
    }
    std::printf("oracle cases: %d, identical: %d, refused: %d\n", n, pass, refused);

    // constructs outside the subset are refused with a reason (never mistranslated)
    const char * bad[] = { "{{ define \"x\" }}y{{ end }}", "{{ template \"x\" . }}", "{{ foo .Content }}", "{{ range .Messages }}{{ .Content", "{{ if .System }}x" };
    for (const char * b : bad) {
        std::string j, e;
        CHECK(!ollama_gotmpl::translate(b, j, e) && !e.empty(), "must refuse: %s", b);
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
