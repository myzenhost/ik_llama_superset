// Probe: what does it cost to build the grammar sampler for a request, i.e. what would SGLang's compiled-grammar cache save here?
// (SGLang constrained/grammar_manager.py:118-175 keys a compiled grammar by (type, text) and reuses it; this fork re-parses the GBNF for every
// request: common/sampling.cpp:133 -> llama_grammar_init_impl.)
// Measures json_schema_to_grammar() and llama_sampler_init_grammar() for a schema with N properties (some nested), median of 200 builds.
// Usage: probe-grammar-parse-cost <model.gguf> [n_properties ...]     (any model with a vocabulary; tests/make_tiny_llama_gguf.py)
#include "llama.h"
#include "json-schema-to-grammar.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static std::string make_schema(int n) {
    std::string s = "{\"type\":\"object\",\"properties\":{";
    for (int i = 0; i < n; i++) {
        if (i) s += ",";
        s += "\"field_" + std::to_string(i) + "\":";
        switch (i % 4) {
            case 0: s += "{\"type\":\"string\",\"maxLength\":40}"; break;
            case 1: s += "{\"type\":\"integer\",\"minimum\":0,\"maximum\":100000}"; break;
            case 2: s += "{\"enum\":[\"alpha\",\"beta\",\"gamma\",\"delta\"]}"; break;
            default: s += "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"},\"b\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"maxItems\":8}},\"required\":[\"a\"]}"; break;
        }
    }
    s += "},\"required\":[";
    for (int i = 0; i < n; i++) { if (i) s += ","; s += "\"field_" + std::to_string(i) + "\""; }
    s += "]}";
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: %s model.gguf [n_properties ...]\n", argv[0]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only   = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::printf("cannot load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<int> sizes;
    for (int i = 2; i < argc; i++) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = { 4, 16, 64, 256 };
    for (int n : sizes) {
        const std::string schema_text = make_schema(n);
        std::vector<double> t_conv, t_init;
        std::string gbnf;
        for (int rep = 0; rep < 200; rep++) {
            auto t0 = clk::now();
            gbnf = json_schema_to_grammar(common_json::parse(schema_text));
            auto t1 = clk::now();
            llama_sampler * smpl = llama_sampler_init_grammar(vocab, gbnf.c_str(), "root");
            auto t2 = clk::now();
            if (!smpl) { std::printf("grammar did not build for n=%d\n", n); return 1; }
            llama_sampler_free(smpl);
            t_conv.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            t_init.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::sort(t_conv.begin(), t_conv.end());
        std::sort(t_init.begin(), t_init.end());
        std::printf("properties %4d  schema %6zu B  gbnf %7zu B  json_schema_to_grammar median %8.3f ms (min %8.3f)   llama_sampler_init_grammar median %8.3f ms (min %8.3f)\n",
                    n, schema_text.size(), gbnf.size(), t_conv[t_conv.size() / 2], t_conv.front(), t_init[t_init.size() / 2], t_init.front());
    }
    llama_model_free(model);
    return 0;
}
