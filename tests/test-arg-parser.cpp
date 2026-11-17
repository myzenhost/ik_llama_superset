#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;
        base.n_ubatch = 512;

        {
            const auto draft = common_base_params_to_speculative(base);
            assert(draft.n_outputs_max == 4);
            assert(draft.n_outputs_max_per_seq == 1);
            assert(draft.n_ubatch == 512); // inherits the target by default
        }

        base.speculative.draft.n_ubatch = 8;
        {
            const auto draft = common_base_params_to_speculative(base);
            assert(draft.n_ubatch == 8);
        }
    }

    {
        // fork: the draft-model flags common_base_params_to_speculative used to ignore -- each one is written where arg.cpp writes it
        // (row S69-FORK-SPEC-DRAFT-FLAGS-PARSED-NEVER-APPLIED-2026-11-05-01; before the fix every assert below but the path failed)
        static const char * pat = "blk\\.0\\.";
        common_params base;
        base.n_ubatch       = 512;
        base.devices        = "TARGET0";
        base.moe_cache_size = 1024;
        base.cpuparams.n_threads = 9;
        base.speculative.model                 = "draft.gguf";   // -md
        base.speculative.n_ubatch              = 16;             // --spec-draft-n-ubatch
        base.speculative.devices               = "DRAFT0";       // -devd
        base.speculative.cpuparams.n_threads   = 3;              // --spec-draft-threads
        base.speculative.tensor_buft_overrides = { { pat, nullptr }, { nullptr, nullptr } }; // --spec-draft-override-tensor
        const auto draft = common_base_params_to_speculative(base);
        assert(draft.model.path == "draft.gguf");
        assert(draft.n_ubatch == 16);
        assert(draft.devices == "DRAFT0");
        assert(draft.cpuparams.n_threads == 3);
        assert(draft.tensor_buft_overrides.size() == 2 && draft.tensor_buft_overrides[0].pattern == pat);
        assert(draft.moe_cache_size == 0); // the MoE cache serves the target context only

        // control: unset draft flags inherit the target (an empty override list keeps the target's)
        common_params plain = base;
        plain.speculative.n_ubatch = 0;
        plain.speculative.devices.clear();
        plain.speculative.tensor_buft_overrides.clear();
        plain.tensor_buft_overrides = { { pat, nullptr }, { nullptr, nullptr } };
        const auto inherit = common_base_params_to_speculative(plain);
        assert(inherit.n_ubatch == 512);
        assert(inherit.devices == "TARGET0");
        assert(inherit.tensor_buft_overrides.size() == 2);

        // upstream's spelling folds into the same fields
        common_params up;
        up.speculative.draft.mparams.path = "d2.gguf";
        up.speculative.draft.n_max        = 7;
        up.speculative.draft.p_min        = 0.5f;
        const auto folded = common_base_params_to_speculative(up);
        assert(folded.model.path == "d2.gguf");
        assert(folded.speculative.n_max == 7);
        assert(folded.speculative.p_min == 0.5f && folded.speculative.p_min_explicit);
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        // --lazy-mode (upstream spelling, common/arg.cpp:2705 @ 95887577a): default auto, on = --defer-ple, off
        common_params lz;
        assert(lz.defer_ple == false && lz.defer_ple_auto == true);   // upstream default is auto
        argv = {"binary_name", "-m", "abc.gguf", "--load-mode", "none", "--lazy-mode", "on"};   // the Qwen3.8 video line
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), lz, LLAMA_EXAMPLE_COMMON));
        assert(lz.defer_ple == true && lz.defer_ple_auto == false && lz.load_mode == LLAMA_LOAD_MODE_NONE);
        argv = {"binary_name", "-m", "abc.gguf", "-lzm", "off"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), lz, LLAMA_EXAMPLE_COMMON));
        assert(lz.defer_ple == false && lz.defer_ple_auto == false);
        argv = {"binary_name", "-m", "abc.gguf", "--lazy-mode", "auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), lz, LLAMA_EXAMPLE_COMMON));
        assert(lz.defer_ple == false && lz.defer_ple_auto == true);
        argv = {"binary_name", "--lazy-mode", "sometimes"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), lz, LLAMA_EXAMPLE_COMMON));
    }

    {
        // nexa-sdk CLI spellings (behaviour port, BSD-3-Clause source, no code copied):
        // --grammar-path = --grammar-file, --grammar-string = --grammar, --stop-file, --token-file
        auto write_tmp = [](const char * name, const char * text) {
            FILE * f = fopen(name, "wb");
            assert(f);
            fputs(text, f);
            fclose(f);
        };
        write_tmp("test-arg-parser-gram.tmp", "root ::= \"yes\" | \"no\"");
        write_tmp("test-arg-parser-stop.tmp", "STOP_A\n\nSTOP_B\r\n");
        write_tmp("test-arg-parser-tok.tmp", "1 15043\n  29892\t3186\n");

        common_params gp;
        argv = {"binary_name", "-m", "abc.gguf", "--grammar-path", "test-arg-parser-gram.tmp"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), gp, LLAMA_EXAMPLE_COMMON));
        assert(gp.sampling.grammar.grammar == "root ::= \"yes\" | \"no\"");

        common_params gs;
        argv = {"binary_name", "-m", "abc.gguf", "--grammar-string", "root ::= \"x\""};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), gs, LLAMA_EXAMPLE_COMMON));
        assert(gs.sampling.grammar.grammar == "root ::= \"x\"");

        common_params sf;
        argv = {"binary_name", "-m", "abc.gguf", "--stop-file", "test-arg-parser-stop.tmp"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), sf, LLAMA_EXAMPLE_COMPLETION));
        assert(sf.antiprompt.size() == 2 && sf.antiprompt[0] == "STOP_A" && sf.antiprompt[1] == "STOP_B");

        common_params tf;
        argv = {"binary_name", "-m", "abc.gguf", "--token-file", "test-arg-parser-tok.tmp"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), tf, LLAMA_EXAMPLE_COMPLETION));
        assert(tf.prompt_token_ids.size() == 4 && tf.prompt_token_ids[0] == 1 && tf.prompt_token_ids[3] == 3186);

        write_tmp("test-arg-parser-tok.tmp", "1 notanumber\n");
        common_params tb;
        argv = {"binary_name", "-m", "abc.gguf", "--token-file", "test-arg-parser-tok.tmp"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), tb, LLAMA_EXAMPLE_COMPLETION));

        remove("test-arg-parser-gram.tmp");
        remove("test-arg-parser-stop.tmp");
        remove("test-arg-parser-tok.tmp");
    }

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        // fork semantics: DRY keeps -1 = context size (default, and accepted on the CLI) because this
        // fork's sampler implements it; upstream's clamps it to 0 (ggml-org/llama.cpp#25388) and so
        // moved to default 64 + reject -1. See the --dry-penalty-last-n handler in common/arg.cpp.
        assert(penalty_params.sampling.dry_penalty_last_n == -1);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "-m", "abc.gguf", "--dry-penalty-last-n", "-1"};   // a successful COMMON parse needs -m
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
        assert(penalty_params.sampling.dry_penalty_last_n == -1);

        argv = {"binary_name", "--dry-penalty-last-n", "-2"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        // --moe-cache-mib: a MiB count, or 'auto' / -1 (--fit sizes the cache; unslothai/llama.cpp PR 251)
        {
            common_params mc;
            argv = {"binary_name", "-m", "abc.gguf", "--moe-cache-mib", "64"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mc, LLAMA_EXAMPLE_COMMON));
            assert(mc.moe_cache_size == 64u*1024*1024 && !mc.moe_cache_auto);
            argv = {"binary_name", "-m", "abc.gguf", "--moe-cache-mib", "auto"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mc, LLAMA_EXAMPLE_COMMON));
            assert(mc.moe_cache_auto && mc.moe_cache_size == 0);
            argv = {"binary_name", "-m", "abc.gguf", "--moe-cache-mib", "-1"};
            common_params mc2;
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mc2, LLAMA_EXAMPLE_COMMON));
            assert(mc2.moe_cache_auto);
            argv = {"binary_name", "-m", "abc.gguf", "--moe-cache-mib", "-5"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), mc2, LLAMA_EXAMPLE_COMMON));
            argv = {"binary_name", "-m", "abc.gguf", "--moe-cache-mib", "autox"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), mc2, LLAMA_EXAMPLE_COMMON));
            // the draft context never inherits 'auto'
            common_params base_auto;
            base_auto.moe_cache_auto = true;
            assert(!common_base_params_to_speculative(base_auto).moe_cache_auto);
        }

        // router keep_alive: ollama's duration grammar, negative = forever (-1)
        int64_t ms = 0;
        assert(string_parse_duration_ms("5m", ms)     && ms == 300000);
        assert(string_parse_duration_ms("1h30m", ms)  && ms == 5400000);
        assert(string_parse_duration_ms("1.5s", ms)   && ms == 1500);
        assert(string_parse_duration_ms("250ms", ms)  && ms == 250);
        assert(string_parse_duration_ms("0", ms)      && ms == 0);
        assert(string_parse_duration_ms("-1s", ms)    && ms == -1);
        assert(!string_parse_duration_ms("300", ms));  // Go rejects a bare number
        assert(!string_parse_duration_ms("5x", ms));
        assert(!string_parse_duration_ms("m", ms));
        assert(!string_parse_duration_ms("", ms));
        assert(string_parse_duration_ms("300", ms, true) && ms == 300000); // OLLAMA_KEEP_ALIVE fallback
        assert(string_parse_duration_ms("-1", ms, true)  && ms == -1);

        common_params server_params;
        assert(server_params.models_keep_alive_ms == -1);
        argv = {"binary_name", "--models-keep-alive", "5m"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), server_params, LLAMA_EXAMPLE_SERVER));
        assert(server_params.models_keep_alive_ms == 300000);
        argv = {"binary_name", "--models-keep-alive", "30"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), server_params, LLAMA_EXAMPLE_SERVER));
        assert(server_params.models_keep_alive_ms == 30000);
        argv = {"binary_name", "--models-keep-alive", "soon"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), server_params, LLAMA_EXAMPLE_SERVER));

        // RESEARCH-DONE: mudler/LocalAI pkg/model/loader.go:113-216 (MIT) and pytorch/serve README read; both switches are opt-in/off by default
        // DISABLE-REVERSIBLE: --disable-management-routes (default off) is the named switch under test below
        // --model-load-cooldown / --model-load-cooldown-max (router): defaults, parsing, refusals
        {
            common_params cp;
            assert(cp.models_load_cooldown_ms == 10000 && cp.models_load_cooldown_max_ms == 300000);
            argv = {"binary_name", "--model-load-cooldown", "30s", "--model-load-cooldown-max", "2m"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), cp, LLAMA_EXAMPLE_SERVER));
            assert(cp.models_load_cooldown_ms == 30000 && cp.models_load_cooldown_max_ms == 120000);
            argv = {"binary_name", "--model-load-cooldown", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), cp, LLAMA_EXAMPLE_SERVER));
            assert(cp.models_load_cooldown_ms == 0);                        // 0 = feature not in use
            argv = {"binary_name", "--model-load-cooldown", "-5"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), cp, LLAMA_EXAMPLE_SERVER));
            argv = {"binary_name", "--model-load-cooldown-max", "0"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), cp, LLAMA_EXAMPLE_SERVER));
        }

        // --disable-management-routes / --reasoning-effort-budget
        {
            common_params cp;
            assert(!cp.server_disable_management_routes && cp.reasoning_effort_budget.empty());
            argv = {"binary_name", "--disable-management-routes", "--reasoning-effort-budget", "low=0.2"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), cp, LLAMA_EXAMPLE_SERVER));
            assert(cp.server_disable_management_routes && cp.reasoning_effort_budget == "low=0.2");
        }

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.n_max == 123);

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
