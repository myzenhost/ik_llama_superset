// llama_set_output_vocab_subset: the output projection multiplied by a few rows of the output weight (CTranslate2's vocabulary map,
// Decoder::update_output_layer / Dense::select_weights, src/layers/decoder.cc:72-135 + common.cc:313-335, MIT, behaviour only).
//
//   test-output-vocab-subset <model.gguf>        (no model path: SKIP, exit 0, so it can sit in a suite on a machine without a GGUF)
//
// One context, one fixed token sequence (a prompt batch, then three single-token steps, so the cached-graph path is exercised too), decoded
//   (1) with the full projection                         -> FULL[k]  for every step k
//   (2) with subset S1 set                               -> SUB1[k]: the S1 entries equal FULL[k] (exactly: same rows, same kernel), every other entry
//                                                           is -INFINITY, and llama_output_vocab_subset_used() == |S1| at every step
//   (3) with another subset S2 of the same size          -> same, for S2 (the partial weight and the cached graph are replaced)
//   (4) with the subset cleared                          -> FULL again, bit for bit, used == 0
// Controls: FULL's entries are not constant and not -inf (a flat or empty "reference" proves nothing); S1 and S2 differ; invalid ids are refused
// and leave the active subset alone; a subset covering the whole vocabulary switches the restriction off.
// MUTATION: gathering row ids[i] + 1 instead of ids[i] in llama_set_output_vocab_subset makes (2) and (3) fail.
#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char * what, const std::string & detail = "") {
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) {
        g_failures++;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("SKIP: no model path given\n");
        return 0;
    }
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        printf("FAIL: could not load %s\n", argv[1]);
        return 1;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 64;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("FAIL: no context\n");
        return 1;
    }
    const int n_vocab = llama_n_vocab(model);
    printf("n_vocab %d\n", n_vocab);
    check(n_vocab >= 64, "the vocabulary is large enough for a subset test");

    std::vector<llama_token> prompt = common_tokenize(ctx, "Once upon a time", true, true);
    check(!prompt.empty(), "the prompt tokenizes");
    const llama_token steps[3] = { 10 % n_vocab, 20 % n_vocab, 30 % n_vocab };

    auto run = [&](std::vector<std::vector<float>> & out, std::vector<int> & used) {
        llama_memory_clear(llama_get_memory(ctx), true);
        out.clear();
        used.clear();
        if (llama_decode(ctx, llama_batch_get_one(prompt.data(), (int32_t) prompt.size()))) {
            return false;
        }
        const float * l = llama_get_logits_ith(ctx, -1);
        out.emplace_back(l, l + n_vocab);
        used.push_back(llama_output_vocab_subset_used(ctx));
        for (int k = 0; k < 3; ++k) {
            llama_token t = steps[k];
            if (llama_decode(ctx, llama_batch_get_one(&t, 1))) {
                return false;
            }
            l = llama_get_logits_ith(ctx, -1);
            out.emplace_back(l, l + n_vocab);
            used.push_back(llama_output_vocab_subset_used(ctx));
        }
        return true;
    };

    std::vector<std::vector<float>> full, sub;
    std::vector<int> used_full, used_sub;
    check(run(full, used_full), "decode with the full projection");
    bool full_ok = true;
    for (auto & row : full) {
        float lo = row[0], hi = row[0];
        for (float v : row) {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            full_ok = full_ok && std::isfinite(v);
        }
        full_ok = full_ok && hi - lo > 1e-3f;
    }
    check(full_ok, "control: the full logits are finite and not constant");
    check(used_full == std::vector<int>(4, 0), "no subset: used() == 0");

    // subsets: spread over the vocabulary, including both ends
    auto make = [&](int off) {
        std::vector<llama_token> s;
        for (int i = 0; i < 7; ++i) {
            s.push_back((llama_token) ((off + i * (n_vocab / 7)) % n_vocab));
        }
        s.push_back(0);
        s.push_back(n_vocab - 1);
        return s;
    };
    std::vector<llama_token> s1 = make(3), s2 = make(11);
    auto uniq = [](std::vector<llama_token> v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    };
    std::vector<llama_token> u1 = uniq(s1), u2 = uniq(s2);
    check(u1 != u2, "control: the two subsets differ");

    auto verify = [&](const char * name, const std::vector<llama_token> & s, const std::vector<llama_token> & u) {
        check(llama_set_output_vocab_subset(ctx, s.data(), (int32_t) s.size()), (std::string("set ") + name).c_str());
        bool run_ok = run(sub, used_sub);
        check(run_ok, (std::string("decode with ") + name).c_str());
        if (!run_ok) {
            return;
        }
        check(used_sub == std::vector<int>(4, (int) u.size()), (std::string(name) + ": the graph really multiplied by the partial weight at every step").c_str(),
              std::to_string(used_sub[0]) + "," + std::to_string(used_sub[1]) + "," + std::to_string(used_sub[2]) + "," + std::to_string(used_sub[3]));
        double worst = 0;
        bool outside_ok = true, inside_ok = true;
        for (size_t k = 0; k < sub.size(); ++k) {
            for (int j = 0; j < n_vocab; ++j) {
                const bool in = std::binary_search(u.begin(), u.end(), (llama_token) j);
                if (in) {
                    const double d = std::fabs((double) sub[k][j] - (double) full[k][j]);
                    worst = std::max(worst, d);
                    inside_ok = inside_ok && d <= 1e-5 * std::max(1.0, (double) std::fabs(full[k][j]));
                } else {
                    outside_ok = outside_ok && std::isinf(sub[k][j]) && sub[k][j] < 0;
                }
            }
        }
        check(inside_ok, (std::string(name) + ": the subset logits equal the full projection's").c_str(), "max abs diff " + std::to_string(worst));
        check(outside_ok, (std::string(name) + ": every other logit is -INFINITY").c_str());
    };
    verify("S1", s1, u1);
    verify("S2", s2, u2);

    // refusals leave the active subset alone
    {
        const llama_token bad[2] = { 1, n_vocab };
        check(!llama_set_output_vocab_subset(ctx, bad, 2), "an id outside the vocabulary is refused");
        check(run(sub, used_sub) && used_sub[0] == (int) u2.size(), "... and the previous subset is still active");
    }
    // clearing restores the full projection bit for bit
    check(llama_set_output_vocab_subset(ctx, nullptr, 0), "clear the subset");
    check(run(sub, used_sub), "decode after clearing");
    bool same = used_sub == std::vector<int>(4, 0);
    for (size_t k = 0; same && k < sub.size(); ++k) {
        same = std::memcmp(sub[k].data(), full[k].data(), sizeof(float) * n_vocab) == 0;
    }
    check(same, "after clearing the logits are bit-identical to the first full run and used() == 0");
    // a subset covering the whole vocabulary is the full projection
    {
        std::vector<llama_token> all(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            all[i] = i;
        }
        check(llama_set_output_vocab_subset(ctx, all.data(), n_vocab), "the whole vocabulary as a subset is accepted");
        check(run(sub, used_sub) && used_sub == std::vector<int>(4, 0), "... and means the full projection");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("%s (%d failure(s))\n", g_failures ? "FAILED" : "all checks passed", g_failures);
    return g_failures ? 1 : 0;
}
