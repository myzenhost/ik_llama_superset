// Parallel constrained decisions (tools/parallel-decision/decision-engine.*, served as POST /v1/decision).
//
//   test-decision-engine [model.gguf]
//
// Part A needs no model: the schema compiler and the answer assembly (enum / boolean / integer / number, the
// numeric aggregates, every input the compiler must refuse).
// Part B runs when a GGUF is given (argv[1] / env DECISION_TEST_MODEL; tests/make_tiny_llama_gguf.py <tree> out.gguf 2048 bytes
// writes a random-weight llama whose vocab is one token per ASCII character): the engine, which forks one KV sequence per scored
// path from a shared trunk and decodes them in one batch, is compared with an independent reference that decodes every complete
// candidate text alone on a fresh context and multiplies the restricted-softmax probabilities along it.
//
// Engine idea and algorithm: thecodacus/llama.cpp branch parallel-decision @ b9244f893 / ad129b08d (MIT).
#include "decision-engine.h"

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (cond) {                                                   \
            std::printf("  ok:   %s\n", #cond);                       \
        } else {                                                      \
            std::printf("  FAIL: %s  (", #cond);                      \
            std::printf(__VA_ARGS__);                                 \
            std::printf(")\n");                                       \
            ++n_fail;                                                 \
        }                                                             \
    } while (0)

template <typename F>
static bool throws(F && f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

using namespace llama_decision;

// ------------------------------------------------------------------------------------------------ part A

static void part_a() {
    std::printf("part A: schema compiler and assembly\n");
    const auto schema = common_json::parse(R"({
        "fire":  {"type": "boolean", "description": "Is it on fire?"},
        "prio":  {"type": "enum", "choices": ["low", "medium", "high"], "description": "Priority."},
        "count": {"type": "integer", "minimum": 0, "maximum": 3, "description": "How many?", "aggregate": "median"},
        "level": {"type": "number", "minimum": 0, "maximum": 1, "step": 0.25, "description": "Level.", "aggregate": "mean"}
    })");
    const auto cs = compile_schema(schema, "Be brief.");
    CHECK(cs.specs.size() == 4 && cs.inputs.size() == 4, "got %zu specs, %zu inputs", cs.specs.size(), cs.inputs.size());
    CHECK(cs.inputs[0].suffix == "  \"fire\": " && cs.inputs[0].candidates == (std::vector<std::string>{ "true", "false" }),
          "boolean: suffix '%s'", cs.inputs[0].suffix.c_str());
    // the values' common leading characters (here the opening quote) move into the suffix; only the rest is scored
    CHECK(cs.inputs[1].suffix == "  \"prio\": \"" &&
          cs.inputs[1].candidates == (std::vector<std::string>{ "low\"", "medium\"", "high\"" }), "enum: suffix '%s'", cs.inputs[1].suffix.c_str());
    CHECK(cs.inputs[2].candidates == (std::vector<std::string>{ "0", "1", "2", "3" }), "integer candidates");
    CHECK(cs.inputs[3].candidates == (std::vector<std::string>{ "0.00", "0.25", "0.50", "0.75", "1.00" }), "number grid is fixed-width");
    CHECK(cs.system_text.find("Allowed values: true, false") != std::string::npos && cs.system_text.find("Be brief.") != std::string::npos,
          "system text carries the catalogue and the instructions");

    // JSON Schema form: string+enum, integer min/max, number multipleOf
    const auto js = compile_schema(common_json::parse(R"({"properties": {
        "a": {"type": "string", "enum": ["x", "y"]}, "b": {"type": "number", "minimum": 0, "maximum": 0.5, "multipleOf": 0.5}}})"), "");
    CHECK(js.specs.size() == 2 && js.specs[1].encoded == (std::vector<std::string>{ "0.0", "0.5" }), "json-schema form");

    CHECK(throws([&] { compile_schema(common_json::parse(R"({"a": {"type": "boolean"}})"), ""); }), "compact field without description is refused");
    CHECK(throws([&] { compile_schema(common_json::parse(R"({"a": {"type": "integer", "minimum": 0, "maximum": 300, "description": "d"}})"), ""); }),
          "integer range over 255 values is refused");
    CHECK(throws([&] { compile_schema(common_json::parse(R"({"a": {"type": "enum", "choices": ["x", "x"], "description": "d"}})"), ""); }),
          "duplicate allowed values are refused");
    CHECK(throws([&] { compile_schema(common_json::parse(R"({"a": {"type": "string", "description": "d"}})"), ""); }), "free-text field is refused");
    CHECK(throws([&] { compile_schema(common_json::parse(R"({"a": {"type": "enum", "choices": ["x"], "description": "d", "aggregate": "mean"}})"), ""); }),
          "aggregate mean on a non-numeric field is refused");
    CHECK(throws([&] { compile_schema(common_json::parse(R"({})"), ""); }), "empty schema is refused");

    // assemble(): counts 0..3 with p = {.1,.2,.3,.4} -> median is the first value whose cumulative p reaches .5
    result r;
    field_result f0; f0.winner = 0; f0.probs = { 0.9f, 0.1f };                       f0.tree = true;
    field_result f1; f1.winner = 2; f1.probs = { 0.2f, 0.3f, 0.5f };                 f1.tree = true;
    field_result f2; f2.winner = 3; f2.probs = { 0.1f, 0.2f, 0.3f, 0.4f };           f2.tree = true;
    field_result f3; f3.winner = 4; f3.probs = { 0.0f, 0.0f, 0.5f, 0.0f, 0.5f };     f3.tree = true;
    r.fields = { f0, f1, f2, f3 };
    const auto out = assemble(cs, r);
    const auto & d = out.at("decision");
    CHECK(d.at("fire").get<bool>() == true && d.at("prio").get<std::string>() == "high", "mode picks the winner");
    CHECK(d.at("count").get<long long>() == 2, "median of {.1,.2,.3,.4} over 0..3 is 2, got %lld", d.at("count").get<long long>());
    CHECK(std::fabs(d.at("level").get<double>() - 0.75) < 1e-9, "mean of {.5,1.0} at p .5/.5 is .75, a grid value, got %g",
          d.at("level").get<double>());
    CHECK(std::fabs(out.at("fields").at("prio").at("probability").get<double>() - 0.5) < 1e-6, "probability is the value's own");
    CHECK(out.at("fields").at("count").contains("interval_p10_p90"), "numeric fields carry the p10-p90 interval");

    field_result bad; bad.winner = -1;
    r.fields[0] = bad;
    CHECK(throws([&] { assemble(cs, r); }), "a field without a winner throws");
}

// ------------------------------------------------------------------------------------------------ part B

struct ref_ctx {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
};

static std::vector<float> all_logits(llama_context * ctx, const tokens_t & toks, int n_vocab) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], (llama_pos) i, { 0 }, true);
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        std::printf("reference decode failed: %d\n", rc);
        std::exit(2);
    }
    std::vector<float> out((size_t) toks.size() * n_vocab);
    for (size_t i = 0; i < toks.size(); ++i) {
        std::copy_n(llama_get_logits_ith(ctx, (int) i), n_vocab, out.begin() + i * n_vocab);
    }
    return out;
}

// Exact constrained distribution of one field, computed without forks, batching or trunks: every candidate's complete text is decoded
// alone; at each depth where the candidates that agree so far disagree on the next token, take the softmax over just those tokens.
static std::vector<double> reference_probs(llama_context * rctx, const llama_vocab * vocab, const tokens_t & prefix, const field_input & in) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<tokens_t> seqs;
    for (const auto & c : in.candidates) {
        seqs.push_back(common_tokenize(vocab, in.suffix + c + "\n", false, true));
    }
    std::vector<std::vector<float>> logits;
    for (const auto & s : seqs) {
        tokens_t full = prefix;
        full.insert(full.end(), s.begin(), s.end());
        logits.push_back(all_logits(rctx, full, n_vocab));
    }
    std::vector<double> lp(seqs.size(), 0.0);
    size_t depth = 0;
    for (bool more = true; more; ++depth) {
        more = false;
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (depth >= seqs[i].size()) {
                continue;
            }
            more = true;
            // candidates sharing seqs[i][0..depth)
            std::vector<llama_token> opts;
            for (size_t j = 0; j < seqs.size(); ++j) {
                if (seqs[j].size() > depth && std::equal(seqs[i].begin(), seqs[i].begin() + depth, seqs[j].begin())) {
                    if (std::find(opts.begin(), opts.end(), seqs[j][depth]) == opts.end()) {
                        opts.push_back(seqs[j][depth]);
                    }
                }
            }
            if (opts.size() < 2) {
                continue; // a forced token: probability 1
            }
            const float * lg = logits[i].data() + (prefix.size() + depth - 1) * n_vocab;
            double mx = -1e30, z = 0;
            for (llama_token t : opts) {
                mx = std::max(mx, (double) lg[t]);
            }
            for (llama_token t : opts) {
                z += std::exp(lg[t] - mx);
            }
            lp[i] += (lg[seqs[i][depth]] - mx) - std::log(z);
        }
    }
    double mx = *std::max_element(lp.begin(), lp.end()), z = 0;
    for (double x : lp) {
        z += std::exp(x - mx);
    }
    std::vector<double> p;
    for (double x : lp) {
        p.push_back(std::exp(x - mx) / z);
    }
    return p;
}

static double max_diff(const std::vector<float> & a, const std::vector<double> & b) {
    if (a.size() != b.size()) {
        return 1e9;
    }
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static void part_b(const char * model_path) {
    std::printf("part B: engine vs independent reference on %s\n", model_path);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::printf("FAIL: cannot load %s\n", model_path);
        ++n_fail;
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_seq = 24;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = n_seq + 1; cp.kv_unified = true;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_context_params rp = llama_context_default_params();
    rp.n_ctx = 1024; rp.n_batch = 512; rp.n_ubatch = 512; rp.n_threads = 2; rp.n_threads_batch = 2;
    llama_context * rctx = llama_init_from_model(model, rp);
    if (!ctx || !rctx) {
        std::printf("FAIL: cannot create contexts\n");
        ++n_fail;
        return;
    }

    // seq 0 stays outside the engine's range [1, 1 + n_seq): a decision must not touch it
    llama_memory_t mem = llama_get_memory(ctx);
    {
        tokens_t sentinel = common_tokenize(vocab, "sentinel", false, true);
        llama_batch b = llama_batch_init((int) sentinel.size(), 0, 1);
        for (size_t i = 0; i < sentinel.size(); ++i) {
            common_batch_add(b, sentinel[i], (llama_pos) i, { 0 }, false);
        }
        llama_decode(ctx, b);
        llama_batch_free(b);
    }
    const llama_pos sentinel_end = llama_memory_seq_pos_max(mem, 0);
    const size_t sentinel_len = (size_t) sentinel_end + 1;

    engine eng(ctx, 1, n_seq);

    const std::string shared = "SYS:pick.\n";
    const std::string context = "abc";
    // three fields: a binary one, a 4-way one whose values share a prefix, and a 5-way one with uneven token lengths
    std::vector<field_input> fields = {
        { "fire: ", { "yes", "no" } },
        { "kind: ", { "red", "rose", "rust", "blue" } },
        { "n: ", { "1", "22", "333", "4", "55" } },
    };
    options opt;
    opt.mode = "tree";

    const auto r = eng.decide(shared, context, fields, opt);
    CHECK(r.fields.size() == fields.size(), "one result per field");
    CHECK(sentinel_end == llama_memory_seq_pos_max(mem, 0), "engine left a sequence outside its range untouched");

    tokens_t prefix = common_tokenize(vocab, shared, true, true);
    {
        tokens_t c = common_tokenize(vocab, context, false, true);
        prefix.insert(prefix.end(), c.begin(), c.end());
    }
    for (size_t f = 0; f < fields.size(); ++f) {
        const auto ref = reference_probs(rctx, vocab, prefix, fields[f]);
        const double d = max_diff(r.fields[f].probs, ref);
        double s = 0;
        for (float p : r.fields[f].probs) { s += p; }
        std::printf("  field %zu: engine/reference max |dp| = %.2e, sum p = %.6f, winner %d\n", f, d, s, r.fields[f].winner);
        CHECK(d < 2e-3, "field %zu probabilities match the reference (max |dp| %.2e)", f, d);
        CHECK(std::fabs(s - 1.0) < 1e-4, "field %zu probabilities sum to 1 (%.6f)", f, s);
        const int best = (int) (std::max_element(ref.begin(), ref.end()) - ref.begin());
        const bool near_tie = [&] { auto c = ref; std::sort(c.rbegin(), c.rend()); return c.size() > 1 && c[0] - c[1] < 4e-3; }();
        CHECK(r.fields[f].winner == best || near_tie, "field %zu winner %d vs reference %d", f, r.fields[f].winner, best);
    }

    // KV hygiene: every trunk and branch cell is gone; only the sentinel (seq 0) and the cached prefix (seq 1) remain.
    // (llama_memory_seq_pos_max reports 0, not -1, for an empty sequence on this fork's legacy KV path, so count cells instead.)
    CHECK(llama_get_kv_cache_used_cells(ctx) == (int) (sentinel_len + r.shared_tokens), "used KV cells %d == sentinel %d + cached prefix %zu",
          (int) llama_get_kv_cache_used_cells(ctx), (int) sentinel_len, r.shared_tokens);

    // prefix cache: the second identical call reuses the cached prefix and gives the same numbers; allow_cache=false rebuilds it
    const auto r2 = eng.decide(shared, context, fields, opt);
    CHECK(r2.cache_hit && !r.cache_hit, "second call hits the cached prefix (first %d, second %d)", r.cache_hit, r2.cache_hit);
    double dd = 0;
    for (size_t f = 0; f < fields.size(); ++f) {
        for (size_t i = 0; i < r.fields[f].probs.size(); ++i) {
            dd = std::max(dd, (double) std::fabs(r.fields[f].probs[i] - r2.fields[f].probs[i]));
        }
    }
    CHECK(dd < 1e-3, "cached and fresh prefix give the same probabilities (%.2e)", dd);
    options nocache = opt;
    nocache.allow_cache = false;
    CHECK(!eng.decide(shared, context, fields, nocache).cache_hit, "allow_cache=false never reports a hit");

    // batch: three contexts decided together equal three decisions made alone
    const std::vector<std::string> contexts = { "abc", "zzzz", "q" };
    const auto b = eng.decide_batch(shared, contexts, fields, opt);
    CHECK(b.items.size() == 3, "one item per context");
    for (size_t i = 0; i < contexts.size() && i < b.items.size(); ++i) {
        const auto alone = eng.decide(shared, contexts[i], fields, opt);
        double m = 0;
        for (size_t f = 0; f < fields.size(); ++f) {
            for (size_t k = 0; k < alone.fields[f].probs.size(); ++k) {
                m = std::max(m, (double) std::fabs(alone.fields[f].probs[k] - b.items[i].fields[f].probs[k]));
            }
        }
        CHECK(m < 2e-3, "context %zu: batched == alone (max |dp| %.2e)", i, m);
    }
    // the three contexts really differ, so a batch that mixed them up could not pass the check above
    CHECK(std::fabs(b.items[0].fields[1].probs[0] - b.items[1].fields[1].probs[0]) > 1e-5, "contexts give different distributions");

    // greedy walk: a valid winner and a path score in (0, 1]
    options greedy = opt;
    greedy.mode = "greedy";
    const auto g = eng.decide(shared, context, fields, greedy);
    for (size_t f = 0; f < fields.size(); ++f) {
        CHECK(!g.fields[f].tree && g.fields[f].winner >= 0 && g.fields[f].winner < (int) fields[f].candidates.size() &&
                  g.fields[f].path_score > 0.0f && g.fields[f].path_score <= 1.0f + 1e-6f,
              "greedy field %zu winner %d score %g", f, g.fields[f].winner, g.fields[f].path_score);
    }

    // refusals
    CHECK(throws([&] { engine bad(ctx, 1, 2); }), "an engine needs at least 3 sequences");
    CHECK(throws([&] { eng.decide(shared, context, { { "x: ", { "ab", "ab" } } }, opt); }), "colliding allowed values are refused");
    CHECK(throws([&] { eng.decide(shared, "", fields, opt); }), "an empty context is refused");
    CHECK(throws([&] { options o = opt; o.mode = "bogus"; eng.decide(shared, context, fields, o); }), "an unknown mode is refused");

    llama_free(rctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    part_a();
    const char * model = argc > 1 ? argv[1] : std::getenv("DECISION_TEST_MODEL");
    if (model && *model) {
        part_b(model);
    } else {
        std::printf("part B skipped: no model (argv[1] / DECISION_TEST_MODEL)\n");
    }
    std::printf(n_fail ? "FAILED: %d check(s)\n" : "all checks passed\n", n_fail);
    return n_fail ? 1 : 0;
}
