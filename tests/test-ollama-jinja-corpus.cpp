// Differential test of the fork's Jinja engine against real Python Jinja2 on the chat templates ollama ships as renderer test data
// (gemma4 e2b/31b, glimmer, nemotron 3.5 / 3-nano, qwen 3.8) over a grid of conversations (user, system, multi-turn, tool declarations,
// tool-call round trips, parallel calls, reasoning history, unicode; enable_thinking on / explicitly off, developer turns, and
// their combinations with tools -- the thinking/developer rows of ollama's gemma4_reference_test.go case table). Fixture: tests/ollama-engine/jinja_corpus_oracle.json
// (generator gen_jinja_corpus_oracle.py). usage: test-ollama-jinja-corpus <fixture.json> [--list]
#include "json.h"
#include "jinja/runtime.h"
#include "jinja/parser.h"
#include "jinja/lexer.h"
#include "jinja/utils.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
using json = common_json;

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
    if (argc < 2) { std::printf("usage: test-ollama-jinja-corpus fixture.json\n"); return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    json fx = json::parse(text);
    const json & cases = fx.at("cases");
    int n = 0, same = 0, bad = 0;
    for (size_t i = 0; i < cases.size(); i++) {
        const json & c = cases[i];
        if (c.contains("oracle_error")) continue;
        ++n;
        const std::string name = c.value("name", std::string());
        std::string got;
        try { got = render(fx.at("templates").at(c.value("template", std::string())), c.at("vars")); } catch (const std::exception & e) { got = std::string("<<EXC ") + e.what() + ">>"; }
        std::string want = c.value("expected", std::string());
        // glimmer prints strftime_now("%Y-%m-%d"): the oracle froze one date, the engine reports today's
        for (std::string * str : { &got, &want }) {
            const size_t p = str->find("Current date: ");
            if (p != std::string::npos && str->size() >= p + 24) str->replace(p + 14, 10, "DATE");
        }
        if (got == want) { ++same; continue; }
        ++bad;
        if (bad <= 6) std::printf("MISMATCH %s\n  want: %s\n  got : %s\n", name.c_str(), json(want).dump().substr(0, 500).c_str(), json(got).dump().substr(0, 500).c_str());
        else std::printf("MISMATCH %s\n", name.c_str());
    }
    std::printf("corpus cases: %d, identical to Python Jinja2: %d, different: %d\n", n, same, bad);
    std::printf(bad ? "FAILED (%d)\n" : "PASS\n", bad);
    return bad ? 1 : 0;
}
