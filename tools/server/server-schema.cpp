#include "server-schema.h"
#include "server-loop-recovery.h"
#include "token-ban.h"
#include "server-unknown-keys.h"

#include "json-schema-to-grammar.h"
#include "json-schema-strict.h"
#include "watermark.h"

#include <regex>

#include <unordered_set>

namespace server_schema {

// defined with the eval() implementations below
static bool has_value(const json & data, const char * n);

// Token-id ban lists as TabbyAPI / ooba send them: an integer array, or a comma-separated string "128,330".
// SOURCE-READ: TabbyAPI be74bf0a common/sampling.py:64-73 (banned_tokens: List[int] | str, alias custom_token_bans). AGPL-3.0:
// the shape was read for the spec, the code below is written here, no code copied.
static bool token_id_list_shape(const json & v) {
    if (v.is_array()) {
        if (v.empty()) {
            return false;
        }
        for (const auto & el : v) {
            if (!el.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        bool digit = false;
        for (const char c : s) {
            if (c >= '0' && c <= '9') { digit = true; continue; }
            if (c != ',' && c != ' ' && c != '-') { return false; }
        }
        return digit;
    }
    return false;
}

// Bans each id outright (logit -inf), the way the existing token-level bans do. Ids outside the vocabulary are dropped, as
// logit_bias drops its own. Appended to the bias list: this runs after the logit_bias handler, which clears it.
static void ban_token_ids(field_eval_context & ctx, const json & v, const char * who) {
    GGML_ASSERT(ctx.vocab != nullptr);
    std::vector<long long> ids;
    if (v.is_array()) {
        for (const auto & el : v) {
            if (!el.is_number_integer()) {
                throw std::runtime_error(std::string("Error: '") + who + "' must be an array of integer token ids or a comma-separated string");
            }
            ids.push_back(el.get<long long>());
        }
    } else if (v.is_string()) {
        const std::string s = v.get<std::string>();
        size_t pos = 0;
        while (pos <= s.size()) {
            size_t end = s.find(',', pos);
            if (end == std::string::npos) {
                end = s.size();
            }
            std::string part = s.substr(pos, end - pos);
            part.erase(0, part.find_first_not_of(' '));
            part.erase(part.find_last_not_of(' ') == std::string::npos ? 0 : part.find_last_not_of(' ') + 1);
            if (!part.empty()) {
                char * stop = nullptr;
                const long long id = strtoll(part.c_str(), &stop, 10);
                if (*stop != 0) {
                    throw std::runtime_error(std::string("Error: '") + who + "' string must hold comma-separated integer token ids");
                }
                ids.push_back(id);
            }
            pos = end + 1;
        }
    } else {
        throw std::runtime_error(std::string("Error: '") + who + "' must be an array of integer token ids or a comma-separated string");
    }
    const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
    for (const long long id : ids) {
        if (id >= 0 && id < n_vocab) {
            ctx.params.sampling.logit_bias.push_back({ (llama_token) id, -INFINITY });
        }
    }
}

//
// llama.cpp-specific completion schema
//

std::vector<std::unique_ptr<field>> make_llama_cmpl_schema(const common_params & params_base, task_params & params) {
    std::vector<std::unique_ptr<field>> fields;
    auto add = [&](field * f) {
        fields.emplace_back(f);
    };

    add((new field_bool("verbose", params.verbose))
        ->set_desc("Include __verbose field in the response with additional debug information"));

    add((new field_bool("timings_per_token", params.timings_per_token))
        ->set_desc("Include prompt processing and text generation speed information in each response"));

    add((new field_bool("stream", params.stream))
        ->set_desc("Allows receiving each predicted token in real-time instead of waiting for the completion to finish"));

    add((new field_nested("stream_options"))
        ->add_subfield((new field_bool("include_usage", params.include_usage))
            ->set_desc("Whether to include usage information in the stream"))
        ->set_desc("Additional options for streaming responses"));

    add((new field_bool("cache_prompt", params.cache_prompt))
        ->set_desc("Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests"));

    add((new field_bool("return_tokens", params.return_tokens))
        ->set_desc("Return the raw generated token ids in the `tokens` field"));

    add((new field_bool("return_prompt_token_ids", params.return_prompt_token_ids))
        ->set_desc("Return the prompt as token ids in the `prompt_token_ids` field of the final reply (after chat templating; media positions "
                   "are null). Not available with stream"));

    add((new field_bool("skip_special_tokens", params.skip_special_tokens))
        ->set_desc("Default true. false: special/control tokens are rendered as text in this request's output (per-request counterpart of --special)"));

    add((new field_bool("return_progress", params.return_progress))
        ->set_desc("Include prompt processing progress events in stream mode"));

    add((new field_bool("token_healing", params.token_healing))
        ->set_desc("Regenerate the prompt's last token under a prefix constraint: the first sampled token must start with the text of the removed token. Not available with grammar / json_schema, speculative decoding or media inputs"));

    add((new field_num("sse_ping_interval", params.sse_ping_interval))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Interval in seconds between SSE comment pings emitted while the stream stays silent, -1 disables pings"));

    add((new field_num("n_predict", params.n_predict))
        ->set_hard_limits(-1, INT32_MAX)
        ->add_alias("max_completion_tokens")
        ->add_alias("max_tokens")
        ->add_alias("max_length") // Kobold / ooba / TabbyAPI name
        ->set_desc("Set the maximum number of tokens to predict. When 0, no tokens will be generated but the prompt is evaluated into the cache"));

    add((new field_num("n_indent", params.n_indent))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks"));

    add((new field_num("n_keep", params.n_keep))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Specify the number of tokens from the initial prompt to retain when context size is exceeded. Use -1 to retain all tokens from the prompt"));

    add((new field_num("n_discard", params.n_discard))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Number of tokens after n_keep that may be discarded when shifting context (0 = half context)"));

    add((new field_num("n_cmpl", params.n_cmpl))
        ->set_hard_limits(1, params_base.n_parallel)
        ->add_alias("n") // alias "n" as fallback (OpenAI completions API)
        ->set_desc("Number of completions to generate. If the input has multiple prompts, total outputs will be N prompts times n_cmpl"));

    add((new field_num("n_cache_reuse", params.n_cache_reuse))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Min chunk size to attempt reusing from the cache via KV shifting. See --cache-reuse arg"));

    add((new field_num("t_max_prompt_ms", params.t_max_prompt_ms))
        ->set_hard_limits(-1, std::numeric_limits<int64_t>::max())
        ->set_desc("Set a time limit in milliseconds for the prompt phase (measured from the moment a slot starts the request). A request whose "
                   "prompt is not read by then is answered HTTP 503 and its slot freed (-1 or 0 = no limit)"));

    add((new field_num("t_deadline_ms", params.t_deadline_ms))
        ->set_hard_limits(0, 86400000)
        ->set_desc("One end-to-end budget in milliseconds, counted from the arrival of the request: queue wait + prompt + generation "
                   "(0 = none; the default is the server's --request-deadline-ms). Still queued or reading its prompt when it runs out: "
                   "HTTP 503. Generating: stopped, and the response reports finish_reason \"deadline\" (stop_type \"deadline\")"));

    add((new field_num("t_max_predict_ms", params.t_max_predict_ms))
        ->set_hard_limits(-1, std::numeric_limits<int64_t>::max())
        ->set_desc("Set a time limit in milliseconds for the prediction phase. The timeout triggers if generation exceeds this time (measured since the first token) and a newline has been generated. Useful for FIM applications"));

    add((new field_json("response_fields"))
        ->set_desc("A list of response fields to return. Missing fields are omitted without error. Fields with a slash are unnested (e.g. generation_settings/n_predict moves n_predict to the root)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.response_fields = json_value(data, "response_fields", std::vector<std::string>());
        }));


    //
    // Sampling params
    //

    add((new field_num("top_k", params.sampling.top_k))
        ->set_limits(0, INT32_MAX)
        ->set_desc("Limit the next token selection to the K most probable tokens (0 = disabled)"));

    add((new field_num("top_p", params.sampling.top_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Limit the next token selection to a subset of tokens with cumulative probability above threshold P (1.0 = disabled)"));

    add((new field_num("min_p", params.sampling.min_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("The minimum probability for a token to be considered, relative to the probability of the most likely token (0 = disabled)"));

    add((new field_num("top_n_sigma", params.sampling.top_n_sigma))
        ->set_desc("Keep tokens within n standard deviations of the top token logit (< 0 = disabled)"));

    add((new field_num("xtc_probability", params.sampling.xtc_probability))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Set the chance for token removal via XTC sampler (0 = disabled)"));

    add((new field_num("xtc_threshold", params.sampling.xtc_threshold))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Set a minimum probability threshold for tokens to be removed via XTC sampler (> 0.5 disables XTC)"));

    add((new field_num("typical_p", params.sampling.typical_p))
        ->add_alias("typical") // TabbyAPI name
        // ->set_limits(0.0f, 1.0f) // what's the valid range?
        ->set_desc("Enable locally typical sampling with parameter p (1.0 = disabled)"));

    add((new field_num("temperature", params.sampling.temp))
        ->set_limits(0.0f, std::numeric_limits<float>::infinity())
        ->set_desc("Adjust the randomness of the generated text (0 = greedy)"));

    add((new field_num("dynatemp_range", params.sampling.dynatemp_range))
        ->set_desc("Dynamic temperature range. The final temperature will be in [temperature - range, temperature + range] (0 = disabled)"));

    add((new field_num("dynatemp_exponent", params.sampling.dynatemp_exponent))
        ->add_alias("temp_exponent") // TabbyAPI name
        ->set_desc("Dynamic temperature exponent, controls how entropy maps to temperature"));

    add((new field_num("repeat_last_n", params.sampling.penalty_last_n))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Last n tokens to consider for penalizing repetition (0 = disabled)"));

    // Hard lower bound > 0: a positive logit is divided by the penalty, so 0 gave +inf and a negative value flipped signs, with a
    // 200 answer. SOURCE-READ: vllm/sampling_params.py (repetition_penalty must be a finite number > 0)
    add((new field_num("repeat_penalty", params.sampling.penalty_repeat))
        ->set_hard_limits(1e-6f, 1e6f)
        ->add_alias("rep_pen") // Kobold / ooba / SillyTavern name
        ->set_desc("Control the repetition of token sequences in the generated text (1.0 = disabled)"));

    // vLLM and the HuggingFace generation API spell this parameter "repetition_penalty";
    // llama.cpp spells the SAME parameter "repeat_penalty". Not two algorithms -- MEASURED by
    // reading llama_sample_repetition_penalties_impl (src/llama-sampling.cpp:526), which applies
    // `logit > 0 ? logit / p : logit * p`, exactly the HuggingFace formula.
    //
    // Registered as an alias onto the same field because the alternative is the worst kind of API
    // mismatch: a vLLM-shaped client sends repetition_penalty, the request SUCCEEDS, the penalty
    // is never applied, and nothing reports it.
    //
    // If a client sends BOTH names in one request, the one applied last wins. Acceptable for an
    // alias -- the two names denote one parameter, so sending two different values is already a
    // client-side bug -- and said out loud here rather than left to be discovered.
    add((new field_num("repetition_penalty", params.sampling.penalty_repeat))
        ->set_hard_limits(1e-6f, 1e6f)
        ->set_desc("Alias of repeat_penalty under vLLM's and HuggingFace's name for the same "
                   "parameter. If both are sent, the last one applied wins"));

    add((new field_num("frequency_penalty", params.sampling.penalty_freq))
        ->set_desc("Repeat alpha frequency penalty (0 = disabled)"));

    add((new field_num("presence_penalty", params.sampling.penalty_present))
        ->set_desc("Repeat alpha presence penalty (0 = disabled)"));

    add((new field_bool("penalize_prompt", params.sampling.penalize_prompt))
        ->set_desc("Whether prompt tokens count towards the repeat, presence, frequency and DRY penalties. When false, "
                   "only generated tokens are penalized and repeat_last_n counts generated tokens"));

    add((new field_str("penalty_scope"))
        ->set_desc("Which tokens frequency_penalty and presence_penalty count: \"all\" (default; upstream "
                   "llama.cpp: every token in the repeat_last_n window, prompt included) or \"generated\" "
                   "(OpenAI/vLLM: only tokens generated for this request). repeat_penalty always uses the window")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const std::string v = data.at("penalty_scope").get<std::string>();
            if (v == "all") {
                ctx.params.sampling.penalty_scope_generated = false;
            } else if (v == "generated") {
                ctx.params.sampling.penalty_scope_generated = true;
            } else {
                throw std::runtime_error("Error: \"penalty_scope\" must be \"all\" or \"generated\", got \"" + v + "\"");
            }
        }));

    add((new field_num("dry_multiplier", params.sampling.dry_multiplier))
        ->set_desc("Set the DRY (Don't Repeat Yourself) repetition penalty multiplier (0 = disabled)"));

    add((new field_num("dry_base", params.sampling.dry_base))
        ->set_desc("Set the DRY repetition penalty base value (must be >= 1.0, any values < 1.0 will be replaced with the default value)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            float v = data.at("dry_base").get<float>();
            ctx.params.sampling.dry_base = (v < 1.0f) ? params_base.sampling.dry_base : v;
        }));

    add((new field_num("dry_allowed_length", params.sampling.dry_allowed_length))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Tokens that extend repetition beyond this length receive exponentially increasing penalty: multiplier * base ^ (sequence_length - allowed_length)"));

    add((new field_num("dry_penalty_last_n", params.sampling.dry_penalty_last_n))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("How many tokens to scan for repetitions (0 = disabled)"));

    add((new field_num("mirostat", params.sampling.mirostat))
        ->set_limits(0, 2)
        ->add_alias("mirostat_mode") // ooba / TabbyAPI name
        ->set_desc("Enable Mirostat sampling, controlling perplexity during text generation (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"));

    add((new field_num("mirostat_tau", params.sampling.mirostat_tau))
        ->set_desc("Set the Mirostat target entropy, parameter tau"));

    add((new field_num("mirostat_eta", params.sampling.mirostat_eta))
        ->set_desc("Set the Mirostat learning rate, parameter eta"));

    add((new field_num("adaptive_target", params.sampling.adaptive_target))
        ->set_limits(-std::numeric_limits<float>::max(), 1.0f)
        ->set_desc("Adaptive-p target probability (valid range 0.0 to 1.0; negative = disabled). Needs a truncation sampler: a request that "
                   "sets it while top_k, top_p, min_p, tfs_z, typical_p, top_a, epsilon_cutoff, eta_cutoff and top_n_sigma are all neutral is "
                   "refused with HTTP 400, because it would sample from the whole vocabulary tail"));

    add((new field_num("fsd_alpha", params.sampling.fsd_alpha))
        ->set_limits(-std::numeric_limits<float>::max(), 1.0f)
        ->set_desc("FSD (Frustratingly Simple Decoding, arXiv:2305.12675): weight of the anti-LM n-gram penalty in score = (1-alpha)*p - alpha*penalty "
                   "(valid range 0.0 to 1.0; negative = disabled). When set, FSD picks the token instead of the random draw"));

    add((new field_num("fsd_pool_size", params.sampling.fsd_pool_size))
        ->set_hard_limits(1, INT32_MAX)
        ->set_desc("FSD: candidates that enter the softmax"));

    add((new field_num("fsd_k", params.sampling.fsd_k))
        ->set_hard_limits(1, INT32_MAX)
        ->set_desc("FSD: candidates kept after the softmax"));

    add((new field_num("fsd_n", params.sampling.fsd_n))
        ->set_hard_limits(1, 16)
        ->set_desc("FSD: n-gram order of the anti-LM"));

    add((new field_num("fsd_beta", params.sampling.fsd_beta))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("FSD: back-off weight of the n-gram interpolation"));

    add((new field_num("fsd_random_first_n", params.sampling.fsd_random_first_n))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Randomized FSD: for the first N sampling steps a coin flip chooses FSD or ordinary top-p/top-k sampling (0 = plain FSD)"));

    add((new field_num("top_a", params.sampling.top_a))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("Top-a: drop tokens with probability below top_a * p_max^2 (0 = disabled)"));

    add((new field_num("epsilon_cutoff", params.sampling.epsilon_cutoff))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("Epsilon sampling: drop tokens with probability below epsilon_cutoff (0 = disabled)"));

    add((new field_num("eta_cutoff", params.sampling.eta_cutoff))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("Eta sampling: drop tokens with probability below min(eta, sqrt(eta)*exp(-entropy)) (0 = disabled)"));

    add((new field_num("no_repeat_ngram_size", params.sampling.no_repeat_ngram_size))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Never generate a token completing an n-gram of this size already in prompt+output (0 = disabled)"));

    add((new field_num("no_repeat_ngram_last_n", params.sampling.no_repeat_ngram_last_n))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("History searched by no_repeat_ngram_size (-1 = context size)"));

    add((new field_num("smoothing_factor", params.sampling.smoothing_factor))
        ->set_hard_limits(0.0f, std::numeric_limits<float>::max())
        ->set_desc("Quadratic (smooth) sampling strength, applied at the temperature step (0 = disabled)"));

    add((new field_num("repeat_penalty_slope", params.sampling.penalty_slope))
        ->set_hard_limits(0.0f, 1.0f)
        ->add_alias("rep_pen_slope") // koboldcpp's name for the same setting
        ->set_desc("Repeat penalty for tokens seen only in the older half of the window: 1 + (repeat_penalty - 1) * slope (1 = uniform)"));

    // Penalty fade (row S69-EXLLAMAV3-PICK-PENALTY-FADE-2026-10-18-01; common/sampling-penalty-fade.h). TabbyAPI's
    // name for the same setting is repetition_decay (backends/exllamav3/sampler.py:57-69, read for the name only).
    add((new field_num("repeat_decay", params.sampling.penalty_decay))
        ->set_hard_limits(0, INT32_MAX)
        ->add_alias("repetition_decay")
        ->set_desc("Penalty fade: past the repeat_last_n window, the repeat/frequency/presence penalties fall linearly "
                   "to 0 over this many more tokens (0 = hard window edge)"));

    // Skew sampling (row S69-APHRODITE-PICK-SKEW-SAMPLING-2026-10-18-01; common/sampling-skew.h). Same field name as
    // Aphrodite's SamplingParams.skew (aphrodite/sampling_params.py:405).
    add((new field_num("skew", params.sampling.skew))
        ->set_hard_limits(-10.0f, 10.0f)
        ->set_desc("Skew sampling: reshape the final random draw toward the most likely tokens (< 0) or the less likely "
                   "ones (> 0), over the candidates the other samplers kept (0 = off; greedy decoding is unaffected)"));

    add((new field_num("smoothing_curve", params.sampling.smoothing_curve))
        ->set_desc("Smooth sampling curve: 1 = quadratic, other values add a cubic term"));

    add((new field_num("adaptive_decay", params.sampling.adaptive_decay))
        ->set_hard_limits(0.0f, 0.99f)
        ->set_desc("EMA decay for adaptive sampling; history approximates 1/(1-decay) tokens"));

    add((new field_str("adaptive_untruncated"))
        ->set_desc("What adaptive_target does when no truncating sampler (top_k, top_p, min_p, ...) is in front of it: "
                   "\"floor\" (default) applies min_p = adaptive_floor first, \"refuse\" answers HTTP 400, \"allow\" "
                   "samples from the whole vocabulary as before. The response's generation_settings tell which happened")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const std::string name = data.at("adaptive_untruncated").get<std::string>();
            const int32_t mode = common_adaptive_untruncated_from_name(name.c_str());
            if (mode < 0) {
                throw std::runtime_error("Error: \"adaptive_untruncated\" must be floor, refuse or allow, got \"" + name + "\"");
            }
            ctx.params.sampling.adaptive_untruncated = mode;
        }));

    add((new field_num("adaptive_floor", params.sampling.adaptive_floor))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("The min_p floor used by adaptive_untruncated \"floor\" (default 0.02)"));

    // seed is uint32_t; field_num uses int32_t so use a handler
    add((new field_num("seed", params.sampling.seed))
        ->set_desc("Set the random number generator (RNG) seed (-1 = random)"));

    add((new field_num("n_probs", params.sampling.n_probs))
        ->add_alias("logprobs") // use "logprobs" if "n_probs" wasn't provided
        ->set_desc("If greater than 0, output the probabilities of top N tokens for each generated token"));

    add((new field_num("min_keep", params.sampling.min_keep))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("If greater than 0, force samplers to return at least N possible tokens"));

    add((new field_bool("backend_sampling", params.sampling.backend_sampling))
        ->set_desc("Use backend sampling instead of llama.cpp sampling"));

    // Classifier-Free Guidance (CFG, koboldcpp-compatible field names/gate: negative_prompt +
    // cfg_scale). https://arxiv.org/abs/2306.17806.
    // SOURCE-READ: koboldcpp gpttype_adapter.cpp:4471-4475,5813,6155-6185,6681,7002-7004 and
    // expose.h:101-102,227,238 (AGPL-3.0, no code copied -- behaviour only: a second context
    // evaluates negative_prompt and its logits are blended in by guidance_scale, gated on
    // guidance_scale != 1.0f). Spec + row: S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01
    // (store.get, read in full before this edit).
    // cfg_scale == 1.0 (the struct default) or an empty negative_prompt (the struct default) is a
    // TRUE no-op -- this schema layer only stores the values; server-context.cpp gates slot-level
    // activation on both being non-default before it reserves any guidance sequence or spends any
    // extra decode step.
    add((new field_num("cfg_scale", params.sampling.cfg_scale))
        ->set_desc("Classifier-Free Guidance scale (1.0 = disabled, matches koboldcpp's guidance_scale gate). "
                   "Requires negative_prompt to be set"));

    add((new field_str("negative_prompt"))
        ->set_desc("Classifier-Free Guidance negative prompt. Ignored unless cfg_scale != 1.0")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.sampling.cfg_negative_prompt = data.at("negative_prompt").get<std::string>();
        }));

    add((new field_bool("watermark", params.sampling.watermark))
        ->set_desc("Watermark this completion with the server's Gumbel-max key (--watermark-key). The key "
                   "itself is never accepted from or returned to a request. Same scheme and PRF as vLLM's "
                   "watermarking, so vLLM's detector reads it with the same key"));

    add((new field_str("watermark_scheme"))
        ->set_desc("Watermark scheme for this request: \"gumbel\" (default; vLLM-compatible), \"kgw\" (TGI's green "
                   "list) or \"synthid\" (SynthID-Text tournament). Only the scheme is selectable; the key stays "
                   "server-side. Pass the same scheme to /watermark/detect")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const std::string name = data.at("watermark_scheme").get<std::string>();
            const int32_t s = common_watermark::scheme_from_name(name.c_str());
            if (s < 0) {
                throw std::runtime_error("Error: \"watermark_scheme\" must be gumbel, kgw or synthid, got \"" + name + "\"");
            }
            ctx.params.sampling.watermark_scheme = s;
        }));

    add((new field_bool("post_sampling_probs", params.post_sampling_probs))
        ->set_desc("Return probabilities of top n_probs tokens after applying the sampling chain"));

    //
    // Speculative decoding params
    //

    // Per-request draft-model limits (ENABLED 2026-09-28). Upstream keeps these under `#if 0` and names
    // params.speculative.draft.*, a sub-struct this fork's common_params_speculative does not have -- the fork's
    // fields are flat. They start as the server-wide values (task params copy params_base.speculative), so a
    // request that omits them behaves exactly as before. They reach the draft through the per-sequence
    // common_speculative_draft_params (n_max caps the slot's draft; n_min/p_min override the draft impl's own).
    // Draft-model speculation only: the other speculative types ignore them.
    add((new field_num("speculative.n_max", params.speculative.n_max))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Maximum number of tokens to draft during speculative decoding (draft model)"));

    add((new field_num("speculative.n_min", params.speculative.n_min))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Minimum number of draft tokens to use; shorter drafts are discarded (draft model)"));

    add((new field_num("speculative.p_min", params.speculative.p_min))
        ->set_desc("Minimum top-candidate probability for a drafted token; 0 disables the gate (draft model)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const float v = data.at("speculative.p_min").get<float>();
            if (v < 0.0f || v > 1.0f) {
                throw std::invalid_argument("Value must be between 0 <= value <= 1, but got " + std::to_string(v));
            }
            ctx.params.speculative.p_min          = v;
            ctx.params.speculative.p_min_explicit = true; // an explicit request value always gates
        }));

    // speculative method / n-gram sizing would reconfigure shared per-server speculative state, so they stay off
#if 0

    add((new field_str("speculative.type"))
        ->set_desc("Speculative decoding method (for debugging and research purposes)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.speculative.types = { common_speculative_type_from_name(data.at("speculative.type").get<std::string>()) };
        }));

    add((new field_num("speculative.ngram_size_n", params.speculative.ngram_simple.size_n))
        ->set_desc("Ngram size for lookup in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_size_m", params.speculative.ngram_simple.size_m))
        ->set_desc("Mgram size for speculative tokens in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_min_hits", params.speculative.ngram_simple.min_hits))
        ->set_desc("Minimum hits at ngram lookup for mgram to be proposed"));
#endif

    add((new field_json("lora"))
        ->set_desc("A list of LoRA adapters to apply to this request. Each entry must have `id` and `scale` fields. Adapters not listed default to scale 0.0")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & lora = data.at("lora");
            // one shape check for every route that takes "lora" (also /v1/embeddings and /rerank): server-common.cpp
            validate_lora_request(lora, params_base.lora_adapters.size());
            ctx.params.lora = parse_lora_request(lora);
        }));

    add((new field_bool("asr_stats", params.asr_stats))
        ->set_desc("/v1/audio/transcriptions only: add the decode statistics of the attempt (asr_stats: n_tokens, avg_logprob, tail_entropy) to the reply. "
                   "Needs n_probs > 0. Set by the --asr-quality retry loop"));

    add((new field_str("lora_adapter"))
        ->set_desc("Name of an adapter in the server's --lora-pool-dir (loaded on demand from <dir>/<name>.gguf and kept in the pool). "
                   "Ignored when the server runs without --lora-pool-dir. Allowed characters: letters, digits, '.', '_', '-'")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("lora_adapter");
            if (!v.is_string()) {
                throw std::runtime_error("Error: 'lora_adapter' must be a string");
            }
            ctx.params.lora_adapter = v.get<std::string>();
        }));

    add((new field_num("lora_adapter_scale", params.lora_adapter_scale))
        ->set_desc("Scale of the adapter named by lora_adapter (default 1.0)"));

    // sequence breakers for DRY
    // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
    // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39
    add((new field_json("dry_sequence_breakers"))
        ->set_desc("Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (ctx.params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }

            // each distinct breaker costs a full vocabulary scan at sampler init, so dedupe before counting
            std::unordered_set<std::string> seen;
            std::vector<std::string> unique_breakers;

            for (const auto & breaker : ctx.params.sampling.dry_sequence_breakers) {
                if (seen.insert(breaker).second) {
                    unique_breakers.push_back(breaker);
                }
            }

            if (unique_breakers.size() > LLAMA_DRY_MAX_SEQ_BREAKERS) {
                throw std::runtime_error(string_format(
                    "Error: dry_sequence_breakers must contain at most %d distinct entries", LLAMA_DRY_MAX_SEQ_BREAKERS));
            }

            ctx.params.sampling.dry_sequence_breakers = std::move(unique_breakers);
        }));

    // read by the json_schema handler below (from the request, defaulting to --json-schema-strict)
    add((new field_bool("json_schema_strict", params.json_schema_strict))
        ->set_desc("true: refuse (400) a json_schema whose constraints the grammar cannot enforce (multipleOf, uniqueItems, not, "
                   "unknown string formats, number bounds, ...), naming each by its JSON pointer. false: accept and constrain what "
                   "can be constrained. Default: the server's --json-schema-strict (off)"));

    // handle both "json_schema" and "grammar"
    add((new field_json("json_schema"))
        ->add_alias("grammar")
        ->set_desc("Set a JSON schema (json_schema) or GBNF grammar string (grammar) for constrained generation. json_schema takes precedence if both are provided")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            auto & params = ctx.params;
            if (data.contains("json_schema") && !data.contains("grammar")) {
                try {
                    auto schema                  = json_value(data, "json_schema", json::object());
                    if (schema.is_object() && schema.empty()) {
                        // an empty schema means any object
                        schema["type"] = "object";
                    }
                    SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
                    // strict check (row S69-SGLANG-PICK-STRICT-JSON-SCHEMA-CHECK-2026-10-18-01): the converter below accepts any
                    // schema and drops what it cannot express; list what it would drop, refuse when strict
                    const std::vector<std::string> dropped = json_schema_unsupported(schema);
                    if (!dropped.empty()) {
                        const bool strict = json_value(data, "json_schema_strict", params_base.json_schema_strict);
                        std::string all;
                        for (const auto & m : dropped) {
                            all += (all.empty() ? "" : "; ") + m;
                        }
                        if (strict) {
                            throw std::runtime_error("cannot be enforced (json_schema_strict): " + all);
                        }
                        SRV_WRN("json_schema: %zu constraint(s) will not be enforced: %s\n", dropped.size(), all.c_str());
                    }
                    std::string grammar_str      = json_schema_to_grammar(schema);
                    SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
                    params.sampling.grammar      = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str)};
                } catch (const std::exception & e) {
                    throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
                }
            } else {
                std::string grammar_str = json_value(data, "grammar", std::string());
                if (!grammar_str.empty()) {
                    // grammar_type key is set by the server when converting chat template grammars
                    std::string grammar_type = json_value(data, "grammar_type", std::string());
                    if (grammar_type == "tool_calls") {
                        params.sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str)};
                    } else {
                        // explicit grammar from the user (API field "grammar")
                        params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str)};
                    }
                    SRV_DBG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(params.sampling.grammar).c_str());
                }
            }
        }));

    // Regex-constrained decoding (row S69-SERVER-REGEX-CONSTRAINED-DECODING-2026-09-27-01; TGI GrammarType::Regex
    // parity, text-generation-inference backends/v3/src/queue.rs:522-546, Apache-2.0). The whole output must FULLY
    // match the regex. Converted by regex_to_grammar (common/json-schema-to-grammar.h), which REFUSES a construct GBNF
    // cannot express instead of silently constraining nothing.
    add((new field_str("regex"))
        ->set_desc("Constrain the output to fully match this regular expression (subset: literals, ., classes, groups, "
                   "alternation, * + ? {m,n}). Unsupported constructs such as lookaround are rejected. "
                   "Mutually exclusive with grammar / json_schema")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const std::string regex = json_value(data, "regex", std::string());
            if (regex.empty()) {
                return;
            }
            if (data.contains("grammar") || data.contains("json_schema")) {
                throw std::runtime_error("\"regex\" cannot be combined with \"grammar\" or \"json_schema\"");
            }
            std::string err;
            std::string gbnf = regex_to_grammar(regex, err);
            if (gbnf.empty()) {
                throw std::runtime_error("\"regex\": " + (err.empty() ? std::string("conversion failed") : err));
            }
            SRV_DBG("Regex %s -> grammar: %s\n", regex.c_str(), gbnf.c_str());
            ctx.params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(gbnf)};
        }));

    // SOURCE-READ: vllm/sampling_params.py:99-138 (StructuredOutputsParams.choice, exactly one constraint kind),
    // vllm/v1/structured_output/utils.py:438-455 (choice_as_grammar: alternation of quoted literals, escaping \\ " \n \r \t and
    // control characters). Apache-2.0, own implementation. Row S69-ENGINE-MERGE-PROGRAMME-ALL-ENGINES-2026-09-09-01.
    add((new field_json("choice"))
        ->set_desc("Constrain the output to exactly one of these strings (non-empty array of non-empty strings). "
                   "Mutually exclusive with grammar / json_schema / regex. Same name as vLLM's structured_outputs.choice")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & choice = data.at("choice");
            if (!choice.is_array() || choice.empty()) {
                throw std::runtime_error("\"choice\" must be a non-empty array of non-empty strings");
            }
            if (data.contains("grammar") || data.contains("json_schema") || data.contains("regex")) {
                throw std::runtime_error("\"choice\" cannot be combined with \"grammar\", \"json_schema\" or \"regex\"");
            }
            std::string gbnf = "root ::= ";
            for (size_t i = 0; i < choice.size(); ++i) {
                if (!choice[i].is_string() || choice[i].get<std::string>().empty()) {
                    throw std::runtime_error("\"choice\" must be a non-empty array of non-empty strings");
                }
                if (i > 0) {
                    gbnf += " | ";
                }
                gbnf += '"';
                for (unsigned char c : choice[i].get<std::string>()) {
                    if (c == '\\' || c == '"') {
                        gbnf += '\\';
                        gbnf += (char) c;
                    } else if (c == '\n') {
                        gbnf += "\\n";
                    } else if (c == '\r') {
                        gbnf += "\\r";
                    } else if (c == '\t') {
                        gbnf += "\\t";
                    } else if (c < 0x20 || c == 0x7f) {
                        static const char * hex = "0123456789ABCDEF";
                        gbnf += "\\x";
                        gbnf += hex[c >> 4];
                        gbnf += hex[c & 15];
                    } else {
                        gbnf += (char) c;
                    }
                }
                gbnf += '"';
            }
            gbnf += "\n";
            SRV_INF("choice constraint: %zu alternatives -> grammar\n", choice.size());
            ctx.params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(gbnf)};
        }));

    add((new field_bool("grammar_lazy", params.sampling.grammar_lazy))
        ->set_desc("Whether to apply grammar constraints lazily, only when triggered (instead of at every step)"));

    // MLC-LLM batch_jump_forward.cc, idea only (Apache-2.0); algorithm in server-jump-forward.h
    add((new field_num("grammar_jump_forward", params.grammar_jump_forward))
        ->set_hard_limits(-1, 4096)
        ->set_desc("For a user-supplied \"grammar\" (GBNF) only: when the grammar leaves no choice about the next text, append up to this many "
                   "tokens spelling it and decode them in one batch (fewer decode calls, same text). -1 = the server's --grammar-jump-forward, "
                   "0 = off. Skipped with n_probs / logprobs, banned strings, classifier-free guidance or speculative decoding"));

    add((new field_str("grammar_continue_prefix"))
        ->set_desc("For a user-supplied \"grammar\" (GBNF) only: the exact text already generated "
                   "under this SAME grammar in a prior call. Tokenized and replayed through a fresh "
                   "grammar sampler before generation starts, so the grammar's parse position "
                   "continues from where it left off instead of restarting at the grammar's root. "
                   "Matches koboldcpp's grammar_retain_state intent, but scoped per request instead "
                   "of one global. SOURCE-READ: upstream/koboldcpp/gpttype_adapter.cpp "
                   "lines 6035-6046, AGPL-3.0, no code copied -- read for behaviour, reimplemented "
                   "from this spec. No effect on output-format/tool-calls grammars, which are "
                   "already fast-forwarded via the server's own generation_prompt.")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.sampling.grammar_continue_prefix = data.at("grammar_continue_prefix").get<std::string>();
        }));

    //
    // Chat parser params
    //

    add((new field_str("reasoning_format"))
        ->set_desc("Reasoning format for chain-of-thought models")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            auto reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
            ctx.params.chat_parser_params.reasoning_format = reasoning_format;
            ctx.params.reasoning_format = reasoning_format; // upstream's task param
            ctx.params.chat_parser_params.reasoning_in_content = ctx.params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        }));

    add((new field_str("generation_prompt"))
        ->set_desc("Generation prompt appended to the chat template output")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            std::string s = data.at("generation_prompt").get<std::string>();
            ctx.params.chat_parser_params.generation_prompt = common_chat_input(s);
            ctx.params.sampling.generation_prompt = s;
        }));

    add((new field_bool("parse_tool_calls", params.chat_parser_params.parse_tool_calls))
        ->set_desc("Whether to parse tool calls from the generated output"));

    add((new field_bool("tool_arg_coercion", params.chat_parser_params.coerce_tool_args))
        ->set_desc("Repair parsed tool-call argument values to the types the tool schemas declare (\"5\" -> 5 for an integer, "
                   "\"true\" -> true for a boolean, a JSON-encoded string -> array/object). Non-streaming responses only: "
                   "text already streamed cannot be retracted. Server default: --tool-arg-coercion"));

    add((new field_json("reasoning_call_rescue"))
        ->set_desc("Chat endpoints with `tools` (default: true). While the model is inside a thinking block, a call-looking tag is "
                   "reasoning text, never a call: a quote of the call format, or a call cut by max_tokens, is not executed. When "
                   "the model ends its turn on its own still inside the block and the reasoning ends with whole "
                   "<tool_call><function=NAME> blocks for offered tools, those are returned as tool calls (the text stays in "
                   "reasoning_content). false = the format parser's own behaviour.")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const bool on = data.at("reasoning_call_rescue").get<bool>();
            ctx.params.chat_parser_params.gate_calls_in_reasoning = on;
            ctx.params.chat_parser_params.rescue_reasoning_calls  = on;
        }));

    add((new field_bool("malformed_call_repair", params.chat_parser_params.repair_malformed_calls))
        ->set_desc("Chat endpoints with `tools` (default: true). A tool call the model wrote in a slightly malformed form (GLM-4.x "
                   "XML with a closing tag missing, Gemma 4 with an unclosed, single-quoted or bare value, an LFM2 call without "
                   "its wrapper tokens) is rebuilt and returned as a tool call instead of failing the turn or running with "
                   "markup inside an argument. Applies to the final parse of a turn the model ended itself, non-streaming "
                   "responses only; a call the strict parser read cleanly is never changed. false = the strict parser's own "
                   "behaviour."));

    add((new field_json("withhold_incomplete_tool_calls"))
        ->set_desc("Chat endpoints with `tools` (default: true). A tool call the output ended inside (cut by max_tokens, the "
                   "context or a deadline before its closing marker) is not returned as a tool call: its arguments are "
                   "truncated. finish_reason stays \"length\". When streaming, each call is sent once it is complete (its "
                   "arguments arrive in one delta). A turn the model ends itself (end of generation, a stop word) is parsed "
                   "as before. false = upstream behaviour: arguments stream as they are generated and a cut call is "
                   "returned.")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.drop_incomplete_tool_calls = data.at("withhold_incomplete_tool_calls").get<bool>();
        }));

    add((new field_json("block_level_thinking_end"))
        ->set_desc("Chat endpoints (default: on when the chat template itself writes a line break after the end-of-thinking tag). "
                   "The end tag closes the thinking block only where a block ends: before a line break or a new tag, outside "
                   "backticks and outside a fenced code block. The same tag in the middle of a sentence, in `quotes` or in a "
                   "fence is the model talking about the tag and stays reasoning. false = the first literal tag ends the "
                   "thinking.")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.block_level_thinking_end = data.at("block_level_thinking_end").get<bool>();
        }));

    add((new field_json("error_on_empty_thinking_reply"))
        ->set_desc("Chat endpoints (default: true). A turn the model ends on its own inside or right after its reasoning, with "
                   "an empty reply (text blank, tool calls absent), is answered with a retryable error (type server_error, "
                   "message starting \"invalid_thinking_block\") instead of finish_reason \"stop\" with an empty message. A "
                   "turn cut by max_tokens or the context is unchanged. false = the empty reply is returned as before.")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.error_on_empty_thinking_reply = data.at("error_on_empty_thinking_reply").get<bool>();
        }));

    add((new field_str("thinking_start_tag"))
        ->set_desc("Set by the server from the chat template: the string that opens a thinking block")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.thinking_start_tag = data.at("thinking_start_tag").get<std::string>();
        }));

    add((new field_json("thinking_end_tags"))
        ->set_desc("Set by the server from the chat template: the strings that close a thinking block")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.thinking_end_tags.clear();
            for (const auto & t : data.at("thinking_end_tags")) {
                ctx.params.chat_parser_params.thinking_end_tags.push_back(t.get<std::string>());
            }
        }));

    add((new field_json("tool_arg_schemas"))
        ->set_desc("The request's tools (OpenAI format), set by the server for tool_arg_coercion")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.tools = common_chat_tools_parse_oaicompat(data.at("tool_arg_schemas"));
        }));

    add((new field_str("chat_parser"))
        ->set_desc("Chat parser configuration string")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.chat_parser_params.parser.load(data.at("chat_parser").get<std::string>());
        }));

    add((new field_json("continue_final_message"))
        ->set_desc("Whether to continue the final message of the chat template")
        ->set_handler([&](field_eval_context &, const json & data) {
            common_chat_continuation_parse(data.at("continue_final_message"));
        }));

    add((new field_json("echo"))
        ->set_desc("Whether to include the continued assistant message in the output")
        ->set_handler([&](field_eval_context &, const json & data) {
            data.at("echo").get<bool>();
        }));

    //
    // Token-level fields (require vocab)
    //

    add((new field_json("preserved_tokens"))
        ->set_desc("List of token strings that must not be split during tokenization")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            common_sampling_add_preserved_tokens(ctx.params.sampling, ctx.vocab, data.at("preserved_tokens").get<std::vector<std::string>>());
        }));

    add((new field_json("grammar_triggers"))
        ->set_desc("List of strings or patterns that trigger grammar-constrained generation")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            std::vector<common_grammar_trigger> triggers;
            for (const auto & t : data.at("grammar_triggers")) {
                triggers.push_back(server_grammar_trigger(t).value);
            }
            common_sampling_add_grammar_triggers(ctx.params.sampling, ctx.vocab, std::move(triggers));
        }));

    add((new field_bool("reasoning_control", params.sampling.reasoning_control))
        ->set_desc("Create the budget sampler on demand so reasoning can be ended at runtime"));

    add((new field_num("reasoning_budget_tokens", params.sampling.reasoning_budget_tokens))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Number of tokens in the reasoning budget (-1 = disabled)"));

    add((new field_str("reasoning_budget_start_tag"))
        ->set_desc("Token string marking the start of the reasoning budget section")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.reasoning_budget_start = common_tokenize(ctx.vocab, data.at("reasoning_budget_start_tag").get<std::string>(), false, true);
        }));

    add((new field_json("reasoning_budget_end_tags"))
        ->add_alias("reasoning_budget_end_tag")
        ->set_desc("Token strings marking the end of the reasoning budget section; the first is forced when the budget expires")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.reasoning_budget_end.clear();
            if (data.contains("reasoning_budget_end_tags")) {
                for (const auto & t : data.at("reasoning_budget_end_tags")) {
                    std::string tag = t.get<std::string>();
                    if (!tag.empty()) {
                        ctx.params.sampling.reasoning_budget_end.push_back(common_tokenize(ctx.vocab, tag, false, true));
                    }
                }
            } else if (data.contains("reasoning_budget_end_tag")) {
                std::string tag = data.at("reasoning_budget_end_tag").get<std::string>();
                if (!tag.empty()) {
                    ctx.params.sampling.reasoning_budget_end.push_back(common_tokenize(ctx.vocab, tag, false, true));
                }
            }
        }));

    add((new field_str("reasoning_budget_message"))
        ->set_desc("Message to prepend to the reasoning budget end tag when forcing it")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            if (!ctx.params.sampling.reasoning_budget_end.empty()) {
                llama_tokens end_tag = ctx.params.sampling.reasoning_budget_end.front();
                std::string message = json_value(data, "reasoning_budget_message", std::string());
                if (!message.empty()) {
                    llama_tokens message_tokens = common_tokenize(ctx.vocab, message, false, true);
                    end_tag.insert(end_tag.begin(), message_tokens.begin(), message_tokens.end());
                }
                ctx.params.sampling.reasoning_budget_forced = std::move(end_tag);
            }
        }));

    add((new field_json("suppress_sequences"))
        ->set_desc("Ban token SEQUENCES (CTranslate2 suppress_sequences). An array of sequences; a sequence is an array of token ids or a "
                   "string (tokenized without special tokens). A one-token sequence is banned at every step; a longer one bans its LAST token "
                   "whenever the prompt + output so far ends with all the tokens before it. Token-level and immediate, unlike banned_strings, "
                   "which waits for the text and rewinds")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.suppress_sequences.clear();
            const auto & seqs = data.at("suppress_sequences");
            if (!seqs.is_array()) {
                throw std::runtime_error("Error: \"suppress_sequences\" must be an array of token-id arrays or strings");
            }
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            for (const auto & el : seqs) {
                std::vector<llama_token> seq;
                if (el.is_string()) {
                    seq = common_tokenize(ctx.vocab, el.get<std::string>(), false, false);
                } else if (el.is_array()) {
                    for (const auto & t : el) {
                        if (!t.is_number_integer()) {
                            throw std::runtime_error("Error: \"suppress_sequences\" sequences must hold integer token ids");
                        }
                        const llama_token tok = t.get<llama_token>();
                        if (tok < 0 || tok >= n_vocab) {
                            throw std::runtime_error("Error: \"suppress_sequences\" holds token id " + std::to_string(tok) + " outside [0, " + std::to_string(n_vocab) + ")");
                        }
                        seq.push_back(tok);
                    }
                } else {
                    throw std::runtime_error("Error: \"suppress_sequences\" must be an array of token-id arrays or strings");
                }
                if (!seq.empty()) {
                    ctx.params.sampling.suppress_sequences.push_back(std::move(seq));
                }
            }
        }));

    add((new field_json("allowed_token_ids"))
        ->add_alias("allowed_tokens") // TabbyAPI name
        ->set_alias_to_canonical()
        ->set_desc("Restrict generation to these token ids only. Every other token is masked to "
                   "-inf. A hard constraint, unlike logit_bias. Same name and array shape as "
                   "vLLM's SamplingParams.allowed_token_ids")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.allowed_token_ids.clear();
            const auto & allowed = data.at("allowed_token_ids");
            if (!allowed.is_array()) {
                return;
            }
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            for (const auto & el : allowed) {
                if (!el.is_number_integer()) {
                    continue;
                }
                // Range-filtered here exactly as logit_bias filters its own ids a few lines
                // below, rather than inventing a different rule for the same kind of input.
                const llama_token tok = el.get<llama_token>();
                if (tok >= 0 && tok < n_vocab) {
                    ctx.params.sampling.allowed_token_ids.push_back(tok);
                }
            }
        }));

    add((new field_json("logit_bias"))
        ->set_desc("Modify the likelihood of specific tokens. Accepts an array of [token, bias] pairs or an object mapping token to bias. Use false as bias to ban a token")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.logit_bias.clear();
            const auto & logit_bias = data.at("logit_bias");
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            auto parse_bias = [](const json & v, float & bias) -> bool {
                if (v.is_number())                        { bias = v.get<float>(); return true; }
                if (v.is_boolean() && !v.get<bool>())     { bias = -INFINITY;      return true; }
                return false;
            };
            if (logit_bias.is_array()) {
                for (const auto & el : logit_bias) {
                    if (!el.is_array() || el.size() != 2) continue;
                    float bias;
                    if (!parse_bias(el[1], bias)) continue;
                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) ctx.params.sampling.logit_bias.push_back({tok, bias});
                    } else if (el[0].is_string()) {
                        for (auto tok : common_tokenize(ctx.vocab, el[0].get<std::string>(), false))
                            ctx.params.sampling.logit_bias.push_back({tok, bias});
                    }
                }
            } else if (logit_bias.is_object()) {
                for (const auto & el : logit_bias.items()) {
                    float bias;
                    if (!parse_bias(el.value(), bias)) continue;
                    char * end;
                    llama_token tok = strtol(el.key().c_str(), &end, 10);
                    if (*end == 0) {
                        if (tok >= 0 && tok < n_vocab) ctx.params.sampling.logit_bias.push_back({tok, bias});
                    } else {
                        for (auto t : common_tokenize(ctx.vocab, el.key(), false))
                            ctx.params.sampling.logit_bias.push_back({t, bias});
                    }
                }
            }
        }));

    add((new field_bool("ignore_eos", params.sampling.ignore_eos))
        ->add_alias("ban_eos_token") // ooba / SillyTavern / TabbyAPI name
        ->set_alias_to_canonical()
        ->set_desc("Ignore the end-of-sequence token and continue generating")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.logit_bias_eog != nullptr);
            ctx.params.sampling.ignore_eos = data.at("ignore_eos").get<bool>();
            if (ctx.params.sampling.ignore_eos && ctx.logit_bias_eog) {
                ctx.params.sampling.logit_bias.insert(
                    ctx.params.sampling.logit_bias.end(),
                    ctx.logit_bias_eog->begin(), ctx.logit_bias_eog->end());
            }
        }));

    add((new field_json("stop"))
        ->add_alias("stop_sequence") // Kobold name
        ->set_alias_to_canonical()
        ->set_desc("Specify stopping strings. Generation stops when one is produced, and the string is not included in the output")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.antiprompt.clear();
            const auto & stop = data.at("stop");
            if (stop.is_array()) {
                for (const auto & word : stop) {
                    if (!word.empty()) ctx.params.antiprompt.push_back(word);
                }
            } else if (stop.is_string()) {
                ctx.params.antiprompt.push_back(stop.get<std::string>());
            }
            // fall back to CLI defaults if the request provided no effective stop strings
            if (ctx.params.antiprompt.empty()) {
                ctx.params.antiprompt = params_base.antiprompt;
            }
        }));

    //
    // banned strings (ik_llama.cpp): rewind and resample instead of emitting a banned string
    //

    auto string_list = [](const json & v, const char * name) {
        std::vector<std::string> out;
        if (!v.is_array()) {
            throw std::runtime_error(std::string(name) + " must be an array of strings");
        }
        for (const auto & el : v) {
            if (!el.is_string()) {
                throw std::runtime_error(std::string(name) + " must be an array of strings");
            }
            auto s = el.get<std::string>();
            if (!s.empty()) {
                out.push_back(std::move(s));
            }
        }
        return out;
    };

    add((new field_json("banned_strings"))
        ->set_desc("Strings that must not appear in the output (case-insensitive). When one is generated, the "
                   "server rewinds to where it starts, bans that token at that position and resamples")
        ->set_handler([string_list](field_eval_context & ctx, const json & data) {
            ctx.params.ban_phrases.clear();
            for (auto & s : string_list(data.at("banned_strings"), "banned_strings")) {
                ctx.params.ban_phrases.push_back(string_lower(s));
            }
        }));

    // vLLM's SamplingParams.bad_words (SOURCE-READ:
    // portable/envs/vllm/Lib/site-packages/vllm/sampling_params.py:299-303) is a list[str] of
    // words that must not be generated. vLLM's own enforcement (SOURCE-READ:
    // portable/envs/vllm/Lib/site-packages/vllm/v1/sample/ops/bad_words.py:9-26) is TOKEN-level:
    // each bad word is tokenized twice (with and without a leading space, to catch mid-sentence
    // vs. sentence-initial spelling per sampling_params.py:573-595 update_from_tokenizer), and at
    // every decode step the LAST token id of a sequence whose prefix matches the tail of the
    // tokens generated so far is masked to -inf BEFORE sampling -- so the exact tokenization the
    // model would have used to spell the word can never be chosen.
    //
    // Our banned_strings (server-context.cpp: rewind-and-resample, added 017d29130) is TEXT-level:
    // it lets the model generate freely, decodes the running text, and on a case-insensitive
    // substring match it rewinds the KV cache to where the match started, bans that token at that
    // position with banned_bias and resamples. This is NOT identical to vLLM's semantics: ours
    // reacts after generation and catches the phrase under ANY tokenization that spells it
    // (a strict superset on that axis), while vLLM's is a pure pre-sampling mask restricted to
    // the two tokenizations it precomputed (a strict superset on the "never even sampled" axis,
    // since ours can emit the phrase transiently before the rewind fires -- observable only via
    // e.g. streaming deltas, not in the final /completion response). Per task row
    // S69-VLLM-SAMPLING-DELTA-5-GAPS-2026-09-14-01 ("2026-09-26" addendum): this field is
    // accepted under vLLM's own name and mapped onto the existing banned_strings mechanism as an
    // ALIAS, not a byte-for-byte semantic port. Registered after banned_strings: if a request
    // sends both, whichever field this loop reaches last (i.e. bad_words) wins, since both write
    // the same ctx.params.ban_phrases vector.
    add((new field_json("bad_words"))
        ->set_desc("Alias for banned_strings, under vLLM's SamplingParams.bad_words name. Word-level, "
                   "case-insensitive, rewind-and-resample -- see banned_strings for exact semantics. "
                   "Not a byte-for-byte port of vLLM's token-level bad_words (see source comment).")
        ->set_handler([string_list](field_eval_context & ctx, const json & data) {
            ctx.params.ban_phrases.clear();
            for (auto & s : string_list(data.at("bad_words"), "bad_words")) {
                ctx.params.ban_phrases.push_back(string_lower(s));
            }
        }));

    add((new field_json("banned_regex"))
        ->set_desc("ECMAScript regular expressions that must not match the output; handled like banned_strings")
        ->set_handler([string_list](field_eval_context & ctx, const json & data) {
            ctx.params.ban_regex = string_list(data.at("banned_regex"), "banned_regex");
            for (const auto & s : ctx.params.ban_regex) {
                std::regex re(s); (void) re; // a bad pattern throws here, so the request fails instead of every token
            }
        }));

    add((new field_json("banned_regex_case_insensitive"))
        ->set_desc("Case-insensitive ECMAScript regular expressions; handled like banned_regex")
        ->set_handler([string_list](field_eval_context & ctx, const json & data) {
            ctx.params.ban_regex_ci = string_list(data.at("banned_regex_case_insensitive"), "banned_regex_case_insensitive");
            for (const auto & s : ctx.params.ban_regex_ci) {
                std::regex re(s, std::regex_constants::icase); (void) re;
            }
        }));

    add((new field_num("banned_n", params.banned_n))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Tokens banned at the rewind position: 1 = the first token of the match, N = its first N, -1 = all"));

    add((new field_num("banned_bias", params.banned_bias))
        ->set_desc("Logit bias applied to a banned token at its position (default -999)"));

    add((new field_num("rewind_count_max", params.rewind_count_max))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Maximum rewinds before a banned match is let through: -1 = max(20, 2 x patterns), 0 = unlimited"));

    add((new field_num("banbuffer_size", params.banbuffer_size))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Tokens held back before sending while bans are active (0 = longest pattern length + 1)"));

    // stop controls. SOURCE-READ: upstream/vllm/vllm/sampling_params.py:269-280 (fields),
    // :506-509 (stop_token_ids de-duplicated, order kept), :618-625 (min_tokens >= 0 and <= max_tokens),
    // upstream/vllm/vllm/v1/sample/logits_processor/builtin.py:165-330 (MinTokensLogitsProcessor masks
    // EOS and the request's stop_token_ids until min_tokens outputs exist). Apache-2.0; behaviour reimplemented, no code copied.
    add((new field_num("min_tokens", params.min_tokens))
        ->set_hard_limits(0, INT32_MAX)
        ->add_alias("min_length") // Kobold / ooba / TabbyAPI name
        ->set_desc("The end-of-generation tokens and stop_token_ids are masked until this many tokens have been generated. "
                   "Also applied under a grammar; if the grammar then allows nothing else (for example it can only end) the masked "
                   "tokens are restored so the request still terminates. Must not exceed n_predict"));

    // prompt scoring. SOURCE-READ: upstream/vllm/vllm/sampling_params.py:289 (field) and :553 (prefix cache not
    // read for such a request) for the contract; the readout itself is written against this server's batch/slot code.
    // Apache-2.0, no code copied.
    add((new field_bool("prompt_logprobs", params.prompt_logprobs))
        ->set_desc("Return, for every prompt token after the first, the log-probability the model gave it given the tokens "
                   "before it, and whether it was the model's top choice (response field prompt_logprobs: logprobs[], is_greedy[]). "
                   "The prompt is evaluated without any cache reuse. Use max_tokens 1; not supported with media or n > 1"));

    add((new field_num("logprob_start_len", params.logprob_start_len))
        ->set_hard_limits(0, 1 << 30)
        ->set_desc("With prompt_logprobs: score only the prompt tokens from this index on (earlier entries are null). Tokens before it can then "
                   "come from the slot or prompt cache (all but the last one before the index), which is what makes scoring many "
                   "continuations of one long prompt cheap. 0 = score the whole prompt with no cache reuse"));

    // SOURCE-READ: upstream/vllm/vllm/sampling_params.py:313 (field) and :524, and
    // vllm/v1/engine/detokenizer.py:87,:107 (the stop string is cut from the output unless this is set). Apache-2.0, no code copied.
    add((new field_bool("include_stop_str_in_output", params.include_stop_str_in_output))
        ->set_desc("When generation ends on one of the \"stop\" strings, keep that string at the end of the returned text "
                   "(default false: it is cut off)"));

    // SOURCE-READ: upstream/sglang/python/sglang/srt/sampling/sampling_params.py:339-351 (MAX_STOP_REGEX_LEN = 256
    // pattern length cap). Apache-2.0, behaviour reimplemented, no code copied.
    add((new field_json("stop_regex"))
        ->set_desc("Stop when one of these regular expressions (ECMAScript, at most 256 characters each, at most 16 of them, none "
                   "may match the empty string) matches the generated text. Only the last 512 bytes are searched, and while "
                   "streaming the last 512 bytes are held back. The matched text is cut from the output unless "
                   "include_stop_str_in_output is set; stop_type is \"word\"")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("stop_regex");
            std::vector<std::string> pats;
            if (v.is_string()) {
                pats.push_back(v.get<std::string>());
            } else if (v.is_array()) {
                for (const auto & p : v) {
                    if (!p.is_string()) {
                        throw std::runtime_error("stop_regex must be a string or an array of strings");
                    }
                    pats.push_back(p.get<std::string>());
                }
            } else if (!v.is_null()) {
                throw std::runtime_error("stop_regex must be a string or an array of strings");
            }
            if (pats.size() > 16) {
                throw std::runtime_error("stop_regex: at most 16 patterns are allowed");
            }
            ctx.params.stop_regex.clear();
            ctx.params.stop_regex_compiled.clear();
            for (const std::string & p : pats) {
                if (p.empty() || p.size() > task_params::STOP_REGEX_MAX_PATTERN) {
                    throw std::runtime_error("stop_regex: each pattern must be 1 to " +
                                             std::to_string(task_params::STOP_REGEX_MAX_PATTERN) + " characters long");
                }
                std::regex re;
                try {
                    re = std::regex(p, std::regex::ECMAScript);
                } catch (const std::regex_error & e) {
                    throw std::runtime_error("stop_regex: invalid pattern \"" + p + "\": " + e.what());
                }
                if (std::regex_search(std::string(), re)) {
                    throw std::runtime_error("stop_regex: pattern \"" + p + "\" matches the empty string");
                }
                ctx.params.stop_regex.push_back(p);
                ctx.params.stop_regex_compiled.push_back(std::move(re));
            }
        }));

    // SOURCE-READ: upstream/vllm/vllm/sampling_params.py:297-309 (logprob_token_ids, at most
    // MAX_LOGPROB_TOKEN_IDS = 128, every id inside the vocabulary). Apache-2.0, no code copied.
    add((new field_json("logprob_token_ids"))
        ->set_desc("Also return, at every generated position, the log-probability of each of these token ids (at most 128) taken "
                   "from the raw distribution before sampling: response completion_probabilities[].token_id_logprobs. "
                   "Turns on n_probs 1 when n_probs is 0. Not available with backend sampling")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.logprob_token_ids.clear();
            const auto & ids = data.at("logprob_token_ids");
            if (!ids.is_array()) {
                throw std::runtime_error("logprob_token_ids must be an array of token ids");
            }
            if (ids.size() > 128) {
                throw std::runtime_error("logprob_token_ids: at most 128 token ids are allowed");
            }
            const int32_t n_vocab_ids = llama_vocab_n_tokens(ctx.vocab);
            for (const auto & el : ids) {
                if (!el.is_number_integer()) {
                    throw std::runtime_error("logprob_token_ids must be an array of integer token ids");
                }
                const int64_t tok = el.get<int64_t>();
                if (tok < 0 || tok >= n_vocab_ids) {
                    throw std::runtime_error("logprob_token_ids: token id " + std::to_string(tok) + " is outside the vocabulary");
                }
                if (std::find(ctx.params.logprob_token_ids.begin(), ctx.params.logprob_token_ids.end(), (llama_token) tok) ==
                        ctx.params.logprob_token_ids.end()) {
                    ctx.params.logprob_token_ids.push_back((llama_token) tok);
                }
            }
        }));

    // SOURCE-READ: upstream/vllm/vllm/sampling_params.py:165-191 (RepetitionDetectionParams and its checks) and
    // vllm/v1/core/sched/utils.py:28-62,129-138. Apache-2.0, no code copied. Prior art with the same observable contract: rtp-llm
    // OnlineRepetitionTracker, exllamav3/TabbyAPI loop_detect_window, Aphrodite repetition_detection.
    add((new field_json("repetition_detection"))
        ->set_desc("End the generation early when the tail of its output is a pattern of max_pattern_size tokens or fewer "
                   "(at least min_pattern_size, default 1) repeated min_count times (>= 2) back to back. The response reports "
                   "stop_type \"repetition\" with finish_reason \"stop\". Only generated tokens count; min_tokens is honoured. "
                   "Give max_pattern_size 0 (or null) to leave it off")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.rep_max_pattern_size = 0;
            ctx.params.rep_min_pattern_size = 1;
            ctx.params.rep_min_count        = 0;
            ctx.params.rep_span_min_len     = 0;
            ctx.params.rep_span_min_count   = 0;
            ctx.params.rep_min_dup_tokens   = 0;
            ctx.params.rep_token_guard      = false;
            ctx.params.rep_tool_begin       = -1;
            ctx.params.rep_tool_end         = -1;
            ctx.params.rep_tool_min_count   = 0;
            ctx.params.rep_cov_threshold    = 0.0f;
            ctx.params.rep_cov_window       = 2000;
            ctx.params.rep_cov_every        = 512;
            const auto & rd = data.at("repetition_detection");
            if (rd.is_null()) {
                return;
            }
            // coverage_threshold / coverage_window / coverage_every: the word-level coverage detector (server-loop-recovery.h),
            // run only while the reasoning block is open
            if (rd.is_object() && rd.contains("coverage_threshold") && !rd.at("coverage_threshold").is_null()) {
                if (!rd.at("coverage_threshold").is_number()) {
                    throw std::runtime_error("repetition_detection.coverage_threshold must be a number");
                }
                const double th = rd.at("coverage_threshold").get<double>();
                if (!(th >= 0.0 && th <= 1.0)) {
                    throw std::runtime_error("repetition_detection.coverage_threshold must be 0..1 (0 = off)");
                }
                ctx.params.rep_cov_threshold = (float) th;
            }
            for (const char * key : {"coverage_window", "coverage_every"}) {
                if (rd.is_object() && rd.contains(key) && !rd.at(key).is_null()) {
                    if (!rd.at(key).is_number_integer()) {
                        throw std::runtime_error(std::string("repetition_detection.") + key + " must be an integer");
                    }
                    const int64_t v = rd.at(key).get<int64_t>();
                    if (v < 1 || v > 30000) {
                        throw std::runtime_error(std::string("repetition_detection.") + key + " must be 1..30000");
                    }
                    (std::string(key) == "coverage_window" ? ctx.params.rep_cov_window : ctx.params.rep_cov_every) = (int32_t) v;
                }
            }
            if (!rd.is_object()) {
                throw std::runtime_error("repetition_detection must be an object with max_pattern_size, min_pattern_size and min_count");
            }
            // token_guard: model-free periodic-tail / n-gram / low-entropy loop detectors (server-repetition.h)
            if (rd.contains("token_guard") && !rd.at("token_guard").is_null()) {
                if (!rd.at("token_guard").is_boolean()) {
                    throw std::runtime_error("repetition_detection.token_guard must be a boolean");
                }
                ctx.params.rep_token_guard = rd.at("token_guard").get<bool>();
            }
            // min_duplicate_tokens: floor on (pattern length x copies) for the back-to-back check
            if (rd.contains("min_duplicate_tokens") && !rd.at("min_duplicate_tokens").is_null()) {
                if (!rd.at("min_duplicate_tokens").is_number_integer()) {
                    throw std::runtime_error("repetition_detection.min_duplicate_tokens must be an integer");
                }
                const int64_t md = rd.at("min_duplicate_tokens").get<int64_t>();
                if (md < 0 || md > 65536) {
                    throw std::runtime_error("repetition_detection.min_duplicate_tokens must be 0..65536 (0 = no floor)");
                }
                ctx.params.rep_min_dup_tokens = (int32_t) md;
            }
            auto get_int = [&](const char * key, int64_t dflt) -> int64_t {
                if (!rd.contains(key) || rd.at(key).is_null()) {
                    return dflt;
                }
                if (!rd.at(key).is_number_integer()) {
                    throw std::runtime_error(std::string("repetition_detection.") + key + " must be an integer");
                }
                return rd.at(key).get<int64_t>();
            };
            const int64_t max_p = get_int("max_pattern_size", 0);
            const int64_t min_p = get_int("min_pattern_size", 0);
            const int64_t cnt   = get_int("min_count", 0);
            const int64_t span_len = get_int("span_min_len", 0);
            const int64_t span_cnt = get_int("span_min_count", 0);
            if (max_p < 0 || min_p < 0 || min_p > max_p) {
                throw std::runtime_error("repetition_detection: max_pattern_size and min_pattern_size must be >= 0 with min_pattern_size <= max_pattern_size");
            }
            if (max_p > 0 && cnt < 2) {
                throw std::runtime_error("repetition_detection: min_count must be >= 2 when max_pattern_size is above 0");
            }
            if (max_p > 4096 || cnt > 4096) {
                throw std::runtime_error("repetition_detection: max_pattern_size and min_count are limited to 4096");
            }
            if (span_len < 0 || span_cnt < 0 || span_len > 1024 || span_cnt > 4096) {
                throw std::runtime_error("repetition_detection: span_min_len must be 0..1024 and span_min_count 0..4096");
            }
            if (span_cnt > 0 && (span_len < 1 || span_cnt < 2)) {
                throw std::runtime_error("repetition_detection: span_min_count needs span_min_len >= 1 and span_min_count >= 2");
            }
            if (span_cnt > 0) {
                ctx.params.rep_span_min_len   = (int32_t) span_len;
                ctx.params.rep_span_min_count = (int32_t) span_cnt;
            }
            if (rd.contains("tool_call_markers") && !rd.at("tool_call_markers").is_null()) {
                const auto & tm = rd.at("tool_call_markers");
                if (!tm.is_object()) {
                    throw std::runtime_error("repetition_detection.tool_call_markers must be an object with begin, end (token ids) and min_count");
                }
                auto tm_int = [&](const char * key, int64_t dflt) -> int64_t {
                    if (!tm.contains(key) || tm.at(key).is_null()) {
                        return dflt;
                    }
                    if (!tm.at(key).is_number_integer()) {
                        throw std::runtime_error(std::string("repetition_detection.tool_call_markers.") + key + " must be an integer");
                    }
                    return tm.at(key).get<int64_t>();
                };
                const int64_t tb = tm_int("begin", -1);
                const int64_t te = tm_int("end", -1);
                const int64_t tc = tm_int("min_count", 0);
                if (tb < 0 || te < 0 || tb == te) {
                    throw std::runtime_error("repetition_detection.tool_call_markers: begin and end must be two different token ids >= 0 (use /tokenize to find them)");
                }
                if (tc < 2 || tc > 4096) {
                    throw std::runtime_error("repetition_detection.tool_call_markers: min_count must be 2..4096");
                }
                ctx.params.rep_tool_begin     = (int32_t) tb;
                ctx.params.rep_tool_end       = (int32_t) te;
                ctx.params.rep_tool_min_count = (int32_t) tc;
            }
            if (max_p > 0) {
                ctx.params.rep_max_pattern_size = (int32_t) max_p;
                ctx.params.rep_min_pattern_size = (int32_t) std::max<int64_t>(1, min_p);
                ctx.params.rep_min_count        = (int32_t) cnt;
            }
        }));

    // Strata serve/server.py @82f46a8c8 (MIT) reasoning_loop_recovery false|"stop"|"recover"; spec only, see server-loop-recovery.h
    add((new field_json("reasoning_loop_recovery"))
        ->set_desc("What a loop found while the model is still reasoning does: false / \"off\" (repetition_detection as configured), "
                   "\"stop\" (also run the word-level coverage detector, 12-word phrases seen 3 times covering >= 25% of the last "
                   "2000 words, and end the reply) or true / \"recover\" (on the FIRST loop, resume once: the high-effort sentence in "
                   "the first system turn is replaced by the low-effort one on token ids, every token generated so far is kept, and "
                   "sampling continues with temperature >= 1.0 and presence_penalty >= 1.5; the client's top_p, top_k, seed and "
                   "output limit are not changed). Falls back to stop when the sentence is not in the first turn as whole tokens, "
                   "for media prompts, speculative decoding, guidance, banned strings, LoRA or backend sampling. "
                   "timings.reasoning_recoveries reports how many retries ran")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("reasoning_loop_recovery");
            int32_t m = server_loop_recovery::MODE_OFF;
            if (v.is_null()) {
                return; // keep the server default
            }
            if (v.is_boolean()) {
                m = v.get<bool>() ? server_loop_recovery::MODE_RECOVER : server_loop_recovery::MODE_OFF;
            } else if (!v.is_string() || !server_loop_recovery::parse_mode(v.get<std::string>(), m)) {
                throw std::runtime_error("reasoning_loop_recovery must be false, true, \"off\", \"stop\" or \"recover\"");
            }
            ctx.params.rlr_mode = m;
        }));

    add((new field_str("reasoning_effort_high_text"))
        ->set_desc("The high-effort sentence reasoning_loop_recovery \"recover\" replaces (default --reasoning-effort-high-text, else the "
                   "Qwen3.8 template's xhigh sentence)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.rlr_high_text = data.at("reasoning_effort_high_text").get<std::string>();
        }));

    add((new field_str("reasoning_effort_low_text"))
        ->set_desc("The low-effort sentence reasoning_loop_recovery \"recover\" puts in its place (default --reasoning-effort-low-text, "
                   "else the Qwen3.8 template's low sentence)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.rlr_low_text = data.at("reasoning_effort_low_text").get<std::string>();
        }));

    add((new field_bool("reasoning_loop_user_has_sentence", params.rlr_user_has_sentence))
        ->set_desc("Set by the server for chat requests: a user message holds the high-effort sentence and the first message is not "
                   "a system message, so the splice could land in user text and recover falls back to stop"));

    // SOURCE-READ: portable/envs/lm_eval/Lib/site-packages/lm_eval/models/openai_completions.py:61-122 (what a loglikelihood client sends
    // and parses; MIT, contract only).
    add((new field_bool("echo", params.echo))
        ->set_desc("OpenAI completions echo: the returned text starts with the prompt. With logprobs also given, the prompt tokens are "
                   "scored and choices[].logprobs carries tokens / token_logprobs / top_logprobs / text_offset with one entry per "
                   "prompt token (the first is null) followed by the generated ones. Not available with stream"));

    add((new field_num("priority", params.priority))
        ->set_hard_limits(-1000000, 1000000)
        ->set_desc("Scheduling priority when every slot is busy: waiting requests start lowest value first, then in arrival order "
                   "(default 0). A running request is never interrupted"));

    add((new field_str("cache_salt"))
        ->set_desc("Cache namespace: requests with different cache_salt values never share KV cache (slot reuse, host prompt cache, disk "
                   "tier, radix cache), so one tenant cannot learn another's prompts from cache timing. Empty = the shared namespace. "
                   "Same field name as vLLM's and SGLang's cache_salt")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.cache_salt = data.at("cache_salt").get<std::string>();
        }));

    add((new field_str("extra_key"))
        ->set_desc("Second cache namespace string, combined with cache_salt (SGLang's name for a caller-defined request class)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.extra_key = data.at("extra_key").get<std::string>();
        }));

    add((new field_num("max_queue_wait_ms", params.max_queue_wait_ms))
        ->set_hard_limits(0, 86400000)
        ->set_desc("When every slot is busy, stop waiting after this many milliseconds and answer HTTP 503 (0 = wait as long as it "
                   "takes; the default is the server's --max-queue-wait-ms). A request that already has a slot is never stopped"));

    add((new field_num("stream_interval", params.stream_interval))
        ->set_hard_limits(1, 4096)
        ->set_desc("Streaming: send a chunk only once this many generated tokens are unsent (1 = a chunk per token). The first token and the "
                   "last chunk are always sent, so the text is the same as with 1. Ignored when the request asks for per-token data "
                   "(n_probs, return_tokens) or sets stop_regex. The default is the server's --stream-interval (SGLang's stream_interval)"));

    add((new field_json("stop_token_ids"))
        ->set_desc("Token ids that end generation when sampled (like the end-of-sequence token); the token is not part of the output")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.stop_token_ids.clear();
            const auto & ids = data.at("stop_token_ids");
            if (!ids.is_array()) {
                throw std::runtime_error("stop_token_ids must be an array of token ids");
            }
            const int32_t n_vocab_ids = llama_vocab_n_tokens(ctx.vocab);
            for (const auto & el : ids) {
                if (!el.is_number_integer()) {
                    throw std::runtime_error("stop_token_ids must be an array of integer token ids");
                }
                const int64_t tok = el.get<int64_t>();
                if (tok < 0 || tok >= n_vocab_ids) {
                    throw std::runtime_error("stop_token_ids: token id " + std::to_string(tok) + " is outside the vocabulary");
                }
                if (std::find(ctx.params.stop_token_ids.begin(), ctx.params.stop_token_ids.end(), (llama_token) tok) ==
                        ctx.params.stop_token_ids.end()) {
                    ctx.params.stop_token_ids.push_back((llama_token) tok);
                }
            }
        }));

    // allowlist sampler (fork, pre-merge examples/server/server-context.cpp:1759-1769): the rule sets are
    // server-wide; a request may replace the always-allowed pieces and the keyword delay
    add((new field_json("allowlist_piece_array"))
        ->set_desc("Strings whose tokens are always allowed by the allowlist sampler (replaces --allowlist-pieces)")
        ->set_handler([string_list](field_eval_context & ctx, const json & data) {
            ctx.params.allow_pieces = string_list(data.at("allowlist_piece_array"), "allowlist_piece_array");
        }));

    add((new field_json("allowlist_keyword_delay"))
        ->set_desc("Tokens generated before allowlist keywords are looked for (replaces --allowlist-keyword-delay)")
        ->set_handler([](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("allowlist_keyword_delay");
            if (!v.is_number_integer() || v.get<int64_t>() < 0) {
                throw std::invalid_argument("allowlist_keyword_delay must be a non-negative integer");
            }
            ctx.params.allow_kw_delay = (size_t) v.get<int64_t>();
        }));

    // expiring logit bias (ik_llama.cpp #1731/#1914, SOURCE-READ examples/server/server-context.cpp @32cddbfce:1940-1960): an array
    // of strings, one rule per line, in the --expiring-logit-bias-file format. The rules are ADDED to the ones the server was
    // started with (the file's stages come first), as ik does. ik printed a bad line to stderr and carried on without the
    // rules; here a line that does not parse fails the request (HTTP 400), so a request is never silently half-applied.
    add((new field_json("expiring_logit_bias"))
        ->set_desc("Rules that bias tokens or sampler parameters for a limited number of generated tokens (format of "
                   "--expiring-logit-bias-file), as an array of strings, one rule per element. Added to the server's own rules")
        ->set_handler([](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("expiring_logit_bias");
            if (!v.is_array()) {
                throw std::runtime_error("expiring_logit_bias must be an array of strings, one rule per element");
            }
            std::string content;
            for (const auto & el : v) {
                if (!el.is_string()) {
                    throw std::runtime_error("expiring_logit_bias must be an array of strings, one rule per element");
                }
                content += el.get<std::string>() + "\n";
            }
            if (v.empty()) {
                return; // nothing to add; argparse would still push an empty stage and switch the machinery on
            }
            auto & sparams = ctx.params.sampling;
            const auto before = sparams.elb_params;
            try {
                argparse_expiring_logit_bias(content, sparams);
            } catch (const std::exception & e) {
                sparams.elb_params = before;
                throw std::runtime_error(std::string("expiring_logit_bias: ") + e.what());
            }
            // comment-only / blank input leaves one stage with nothing in it; keep the sampler on its old path then
            bool any = false;
            for (const auto & st : sparams.elb_params) {
                any = any || !st.entries.empty() || !st.exitword.empty();
            }
            if (!any) {
                sparams.elb_params.clear();
            }
        }));

    add((new field_bool("saturate_predict", params.saturate_predict))
        ->set_desc("If true, rewound tokens do not count against n_predict"));

    // koboldcpp "banned_tokens" (behaviour spec docs/agent-reports/koboldcpp-spec.md, feature A; AGPL source read for the spec only, no code copied).
    // Honoured only with --token-ban-substr; without it the field is accepted and ignored, and the request runs exactly as if it were absent.
    // Registered AFTER banned_strings / bad_words: those handlers rebuild ban_phrases, this one appends to it.
    add((new field_json("banned_tokens"))
        ->set_desc("koboldcpp-style token ban (honoured only with --token-ban-substr): a list of strings. A short string that is a single token bans every "
                   "vocabulary entry containing it (ASCII case-insensitive substring of the token's text); any other string is banned as a phrase, like banned_strings")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("banned_tokens");
            // TabbyAPI's banned_tokens holds token IDS (an integer array, or "1,2,3"). Those shapes were a 400 here, so taking them
            // changes no existing request. SOURCE-READ: TabbyAPI be74bf0a common/sampling.py:64-73 (type List[int] | str).
            if (token_id_list_shape(v)) {
                ban_token_ids(ctx, v, "banned_tokens");
                return;
            }
            if (!v.is_array()) {
                throw std::runtime_error("Error: 'banned_tokens' must be an array of strings");
            }
            std::vector<std::string> words;
            for (const auto & el : v) {
                if (!el.is_string()) {
                    throw std::runtime_error("Error: 'banned_tokens' must be an array of strings");
                }
                words.push_back(el.get<std::string>());
            }
            if (!params_base.token_ban_substr) {
                return;
            }
            GGML_ASSERT(ctx.vocab != nullptr);
            std::vector<std::string> token_patterns;
            size_t budget = 0;
            for (const auto & w : words) {
                const std::string lower = token_ban_lower(w);
                if (lower.empty()) {
                    continue;
                }
                const size_t n_tok = common_tokenize(ctx.vocab, lower, false, false).size();
                switch (token_ban_classify(w, n_tok)) {
                    case token_ban_kind::DROP:
                        break;
                    case token_ban_kind::TOKEN:
                        token_patterns.push_back(w);
                        break;
                    case token_ban_kind::PHRASE:
                        ctx.params.ban_phrases.push_back(lower);
                        budget = std::max(budget, token_ban_phrase_budget(n_tok));
                        break;
                }
            }
            if (budget > 0) {
                ctx.params.banbuffer_size = std::max<int32_t>(ctx.params.banbuffer_size, (int32_t) budget);
            }
            if (!token_patterns.empty()) {
                // the vocabulary is scanned once per request that carries a single-token ban (no cache: the spec rebuilds the list per request)
                const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
                std::vector<std::string> pieces((size_t) n_vocab);
                for (llama_token tok = 0; tok < n_vocab; ++tok) {
                    pieces[(size_t) tok] = common_token_to_piece(ctx.vocab, tok, false);
                }
                for (const int id : token_ban_scan(pieces, token_patterns)) {
                    ctx.params.sampling.logit_bias.push_back({ (llama_token) id, -INFINITY });
                }
            }
        }));

    add((new field_bool("tool_call_fix", params.sampling.tool_call_fix))
        ->set_desc("Keep the reply from opening with an empty array: every vocab token whose decoded text contains ']' is "
                   "masked for the first 3 generated tokens while the output holds at most one '[', and is free afterwards. "
                   "Guards a freeform (non-grammar) tool-call parser against an empty tool-call list. Matches koboldcpp's "
                   "tool_call_fix (gpttype_adapter.cpp:159, :5783, :7023-7030, SOURCE-READ, AGPL-3.0, no code copied); "
                   "irrelevant when the request's chat format already constrains tool calls with a grammar.")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.sampling.tool_call_fix = data.at("tool_call_fix").get<bool>();
            ctx.params.tool_call_fix_ids.clear();
            if (ctx.params.sampling.tool_call_fix) {
                GGML_ASSERT(ctx.vocab != nullptr);
                const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
                for (llama_token tok = 0; tok < n_vocab; ++tok) {
                    const std::string piece = common_token_to_piece(ctx.vocab, tok, false);
                    if (piece.find(']') != std::string::npos) {
                        ctx.params.tool_call_fix_ids.push_back(tok);
                    }
                }
            }
        }));

    // ---- request-name compatibility with Kobold / ooba / SillyTavern / TabbyAPI (row S69-ENGREV-TABBYAPI-SAMPLER-ALIASES-AND-UNKNOWN-KEY-WARN-2026-10-27-01)
    // SOURCE-READ: TabbyAPI be74bf0a common/sampling.py:44 (max_length), :52 (min_length), :60 (stop_sequence), :70 (custom_token_bans),
    // :77 (allowed_tokens), :125 (typical), :161 (rep_pen), :169-176 (penalty_range aliases), :230 (mirostat_mode), :249 (ban_eos_token),
    // :283/:291/:299 (dynatemp_high / dynatemp_low / temp_exponent). AGPL-3.0: names and ranges read for the spec, the code is written
    // here, no code copied. The plain renames are add_alias() on the fields above; the three below need their own handling.

    // penalty_range: the same setting as repeat_last_n, but TabbyAPI/ooba/Kobold send -1 for "whole context", which repeat_last_n's hard
    // bound refuses. The sampler already reads a negative penalty_last_n as the context size (common/sampling.cpp), so the range is
    // widened to -1 for these names only; repeat_last_n keeps its bound. If both a repeat_last_n and one of these are sent, this one wins.
    add((new field_num("penalty_range", params.sampling.penalty_last_n))
        ->set_hard_limits(-1, INT32_MAX)
        ->add_alias("repetition_range")
        ->add_alias("repetition_penalty_range")
        ->add_alias("rep_pen_range")
        ->set_desc("Alias family of repeat_last_n under the Kobold / ooba / TabbyAPI names (penalty_range, repetition_range, "
                   "repetition_penalty_range, rep_pen_range): tokens of history the repeat penalty looks at, -1 = whole context, 0 = off"));

    // dynatemp_low / dynatemp_high (TabbyAPI: min_temp / max_temp) describe the temperature RANGE; this server takes a base temperature and a
    // half-width. low < high maps to temperature = (high + low) / 2, dynatemp_range = (high - low) / 2. low == high is a static temperature
    // (dynamic temperature off, temperature untouched). One of the pair alone is ignored with a warning, as TabbyAPI needs both.
    auto dynatemp_pair = [](field_eval_context & ctx, const json & data) {
        float lo = 0.0f, hi = 0.0f;
        auto pick = [&](const char * a, const char * b, float & out) -> bool {
            for (const char * n : { a, b }) {
                if (has_value(data, n)) {
                    if (!data.at(n).is_number()) {
                        throw std::runtime_error(std::string("Error: '") + n + "' must be a number");
                    }
                    out = data.at(n).get<float>();
                    return true;
                }
            }
            return false;
        };
        const bool has_lo = pick("dynatemp_low",  "min_temp", lo);
        const bool has_hi = pick("dynatemp_high", "max_temp", hi);
        if (!has_lo || !has_hi) {
            SRV_WRN("%s", "dynatemp_low/min_temp and dynatemp_high/max_temp must be sent together; the lone one is ignored\n");
            return;
        }
        if (lo < 0.0f || hi < lo) {
            throw std::runtime_error("Error: dynatemp_low/min_temp must be >= 0 and <= dynatemp_high/max_temp");
        }
        if (hi > lo) {
            ctx.params.sampling.temp          = (hi + lo) * 0.5f;
            ctx.params.sampling.dynatemp_range = (hi - lo) * 0.5f;
        } else {
            ctx.params.sampling.dynatemp_range = 0.0f;
        }
    };
    add((new field_json("dynatemp_low"))
        ->add_alias("min_temp")
        ->set_desc("Lower end of the dynamic temperature range (TabbyAPI min_temp). With dynatemp_high: temperature = (high + low) / 2 and "
                   "dynatemp_range = (high - low) / 2; equal values mean a static temperature")
        ->set_handler(dynatemp_pair));
    add((new field_json("dynatemp_high"))
        ->add_alias("max_temp")
        ->set_desc("Upper end of the dynamic temperature range (TabbyAPI max_temp); see dynatemp_low")
        ->set_handler(dynatemp_pair));

    // custom_token_bans: ooba's name for a token-id ban list (TabbyAPI alias of banned_tokens, which holds ids there). Appended to the bias
    // list, so it is registered after logit_bias, which clears it. Out-of-vocabulary ids are dropped.
    add((new field_json("custom_token_bans"))
        ->set_desc("Token ids to ban outright (ooba / TabbyAPI): an integer array or a comma-separated string. Same effect as logit_bias -inf")
        ->set_handler([](field_eval_context & ctx, const json & data) {
            ban_token_ids(ctx, data.at("custom_token_bans"), "custom_token_bans");
        }));

    add((new field_json("samplers"))
        ->set_desc("The order in which samplers are applied. An array of sampler type names, or a single string of sampler chars")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & samplers = data.at("samplers");
            if (samplers.is_array()) {
                // ADAPTED: the field is samplers_sequence here and holds llama_sampler_type,
                // not common_sampler_type, so the llama_sampling_* parsers are the matching pair.
                // allow_alt_names=true keeps the permissive behaviour a public API wants -- a
                // client sending a well-known alias should not get a silently empty chain.
                ctx.params.sampling.samplers_sequence = llama_sampling_types_from_names(samplers.get<std::vector<std::string>>(), true);
            } else if (samplers.is_string()) {
                ctx.params.sampling.samplers_sequence = llama_sampling_types_from_chars(samplers.get<std::string>());
            }
        }));

    return fields;
}

const server_override_groups & override_groups() {
    static const server_override_groups groups = [] {
        server_override_groups g;
        common_params dummy_base;
        task_params   dummy_params;
        const auto schema = make_llama_cmpl_schema(dummy_base, dummy_params);
        for (const auto & f : schema) {
            std::vector<std::string> names;
            for (const char * n : f->name) {
                names.emplace_back(n);
            }
            for (const auto & n : names) {
                g[n] = names;
            }
        }
        return g;
    }();
    return groups;
}

// Top-level request keys that the routes, the OpenAI / Anthropic / Responses / Ollama converters or the HTTP handlers read themselves,
// or that those converters add to the body before the schema sees it. They are not sampler fields, so the schema does not list them,
// but they are not typos either. A key missing from this list costs one log line per server run, nothing more.
static bool request_key_is_envelope(const std::string & k) {
    static const std::unordered_set<std::string> envelope = {
        // prompt / chat envelope (OpenAI chat + completions, Anthropic, Responses, Ollama, TGI)
        "model", "messages", "prompt", "input", "inputs", "system", "instructions", "tools", "tool_choice", "parallel_tool_calls",
        "functions", "function_call", "response_format", "user", "store", "metadata", "service_tier", "modalities", "audio",
        "prediction", "previous_response_id", "truncation", "text", "include", "reasoning", "thinking", "think", "options",
        "format", "raw", "template", "context", "images", "keep_alive", "parameters", "suffix", "best_of", "echo",
        "stop_sequences", "max_output_tokens", "top_logprobs", "reasoning_effort", "thinking_budget_tokens", "extra_params",
        "chat_template_kwargs", "add_generation_prompt", "add_special", "parse_special", "tool_prompt",
        // request handling read by the HTTP handlers / the slot launcher
        "id_slot", "cache_pin_token", "message_delimiters", "use_beam_search", "length_penalty", "beam_mode", "beam_num_hypotheses",
        "beam_patience", "beam_prefix", "beam_prefix_bias_beta", "input_prefix", "input_suffix", "input_extra", "tokens", "documents",
        "ollama_num_ctx", "decision_budget", "decision_window", "decision_window_agg", "decision_window_logits", "questions",
        // keys the converters add to the body (server-common.cpp oaicompat_*_params_parse, server-chat.cpp)
        "plain_text_spans", "tool_arg_schemas", "tool_arg_coercion", "reasoning_call_rescue", "malformed_call_repair",
        "withhold_incomplete_tool_calls", "block_level_thinking_end", "error_on_empty_thinking_reply", "reasoning_control",
        "reasoning_loop_user_has_sentence", "reasoning_answer_reserve", "responses_reasoning_blob", "responses_reasoning_summary", "responses_tool_map",
        "__metadata_user_id", "reasoning_budget_tokens",
    };
    return envelope.count(k) > 0;
}

// Warn, once per key per server run, about each top-level request key that neither a schema field (or alias) nor the envelope claims.
// Never rejects: the request is evaluated exactly as before.
static void warn_unknown_request_keys(const json & data) {
    static server_unknown_key_registry registry;
    const auto & groups = override_groups();
    const std::vector<std::string> unknown = server_unknown_request_keys(data, [&](const std::string & k) {
        return groups.count(k) > 0 || request_key_is_envelope(k);
    });
    if (unknown.empty()) {
        return;
    }
    bool cap_hit = false;
    for (const auto & k : registry.take_new(unknown, cap_hit)) {
        SRV_WRN("request key '%s' is not a known sampling parameter or alias and was ignored (reported once per key per server run)\n",
                server_unknown_key_for_log(k).c_str());
    }
    if (cap_hit) {
        SRV_WRN("%s", "too many distinct unknown request keys; further ones are not reported\n");
    }
}

task_params eval_llama_cmpl_schema(
                const llama_vocab * vocab,
                const common_params & params_base,
                const std::vector<llama_logit_bias> & logit_bias_eog,
                const json & data_in) {
    warn_unknown_request_keys(data_in);
    // operator sampler overrides (--sampler-overrides): applied to the request body before the schema reads it
    json overridden;
    const json * data_ptr = &data_in;
    if (!server_override_instance().empty()) {
        overridden = server_override_apply(data_in, server_override_instance().effective(), override_groups());
        data_ptr   = &overridden;
    }
    const json & data = *data_ptr;

    task_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
    params.sampling      = params_base.sampling;
    params.speculative   = params_base.speculative;
    params.n_keep        = params_base.n_keep;
    params.n_predict     = params_base.n_predict;
    params.n_cache_reuse = params_base.n_cache_reuse;
    params.max_queue_wait_ms = params_base.max_queue_wait_ms;
    params.stream_interval = params_base.stream_interval;
    params.t_deadline_ms = params_base.request_deadline_ms;
    params.cache_prompt  = params_base.cache_prompt;
    params.antiprompt    = params_base.antiprompt;
    params.sse_ping_interval = params_base.sse_ping_interval;

    // --reasoning-loop-recovery / --reasoning-effort-*-text; the request fields override them
    server_loop_recovery::parse_mode(params_base.reasoning_loop_recovery, params.rlr_mode);
    params.rlr_high_text = params_base.reasoning_effort_high_text;
    params.rlr_low_text  = params_base.reasoning_effort_low_text;

    // --banned-string-file / --banned-n; a request's banned_strings replaces the list
    for (const auto & s : params_base.ban_phrases) {
        if (!s.empty()) {
            params.ban_phrases.push_back(string_lower(s));
        }
    }
    params.banned_n    = params_base.banned_n;
    params.banned_bias = params_base.ban_phrases_bias;

    // allowlist sampler: request fields allowlist_piece_array / allowlist_keyword_delay override these
    params.allow_pieces   = params_base.allow_pieces;
    params.allow_kw_delay = params_base.allow_kw_delay;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose       = params_base.verbosity > 9;

    params.reasoning_format = params_base.reasoning_format;

    // create context and schema
    field_eval_context ctx(params);
    ctx.vocab          = vocab;
    ctx.logit_bias_eog = &logit_bias_eog;

    auto schema = make_llama_cmpl_schema(params_base, params);

    // eval all fields in the schema
    for (const auto & f : schema) {
        f->eval(ctx, data);
    }

    // post-processing
    {
        // if "reasoning_format" is not provided, its handler will not be called, we will need to handle it here
        // the parser's reasoning format follows the task param (server default, or the request's via the handler)
        params.chat_parser_params.reasoning_format = params.reasoning_format;
        auto reasoning_format = params.chat_parser_params.reasoning_format;
        params.chat_parser_params.reasoning_in_content = params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);

        // tool-call argument type repair rewrites the arguments on the final parse; a streaming client has already been sent
        // the raw prefix, and a rewritten tail would no longer extend it (compute_diffs would throw), so it is off when streaming
        if (params.stream) {
            params.chat_parser_params.coerce_tool_args = false;
            // a rebuilt call rewrites the tail of the text the same way: off when streaming
            params.chat_parser_params.repair_malformed_calls = false;
        }
    }

    // Fade and slope reshape the same window and the sampler applies only the fade: refuse the pair rather than drop one.
    if (params.sampling.penalty_decay > 0 && params.sampling.penalty_slope > 0.0f && params.sampling.penalty_slope < 1.0f) {
        throw std::runtime_error("Error: \"repeat_decay\" cannot be combined with \"repeat_penalty_slope\" < 1");
    }
    // Skew acts in the plain temperature draw only (common/sampling.cpp); mirostat, adaptive-p, FSD and the gumbel /
    // synthid watermarks make their own draw, so the setting would be dropped without a word. Refuse instead.
    if (params.sampling.skew != 0.0f && params.sampling.temp > 0.0f &&
        (params.sampling.mirostat != 0 || params.sampling.adaptive_target >= 0.0f || params.sampling.fsd_alpha >= 0.0f ||
         (params.sampling.watermark && params.sampling.watermark_scheme != common_watermark::SCHEME_KGW))) {
        throw std::runtime_error("Error: \"skew\" cannot be combined with mirostat, adaptive_target, fsd_alpha or a "
                                 "gumbel/synthid watermark (they make their own draw)");
    }
    // A watermark request on a server with no key would otherwise sample normally and return text the
    // caller believes is marked. Refuse it instead.
    if (params.sampling.watermark && !params.sampling.watermark_key_set) {
        throw std::runtime_error("Error: \"watermark\" requested but the server has no --watermark-key");
    }
    // Same hazard, other route: mirostat and adaptive-p draw in their own branches of
    // llama_sampling_sample_impl, which no watermark scheme reaches, so the text would come back unmarked.
    if (params.sampling.watermark && params.sampling.temp > 0.0f &&
        (params.sampling.mirostat != 0 || params.sampling.adaptive_target >= 0.0f)) {
        throw std::runtime_error("Error: \"watermark\" cannot be combined with mirostat or adaptive_target sampling");
    }
    // adaptive_target with nothing truncating in front of it samples the whole-vocabulary tail. MEASURED twice, independently:
    //  - Llama-3.2-1B (CPU, 24 requests x 96 tokens per arm, temp/research/findings/S69_ADAPTIVE_P_UNTRUNCATED_REFUSE_VS_FLOOR_MEASURED_2026-09-30.md):
    //    99.8% of drawn tokens below 1e-3, 1 in 10 still a tail token behind a min_p 0.002 floor, none behind 0.02;
    //  - gemma-3-270m (262k vocab, CPU, 400 tokens/row, commit 87fc1fb5e, share below 1e-4): untruncated + adaptive 0.3/0.5/0.7 = 99.8/98.8/96.8%,
    //    top_k 40 + adaptive <= 2.5%, min_p 0.002 + adaptive <= 0.2%, default chain 0.0%.
    // What to do about it is the request's choice (adaptive_untruncated): "floor" (default, applied in the sampler), "refuse" (HTTP 400,
    // the behaviour 87fc1fb5e made unconditional) or "allow". A top_k at or above the vocab size is neutral (87fc1fb5e's nuance), so the
    // vocab size goes into the params before the judgement. Row: S69-KOBOLDCPP-PICK-ADAPTIVE-P-UNTRUNCATED-GUARD-2026-09-29-01.
    params.sampling.n_vocab_hint = ctx.vocab ? llama_vocab_n_tokens(ctx.vocab) : 0;
    {
        const std::string refusal = common_sampling_adaptive_p_refusal(params.sampling);
        if (!refusal.empty()) {
            throw std::runtime_error("Error: " + refusal);
        }
    }

    // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01): refuse
    // rather than silently ignore. Before this check existed, cfg_scale/negative_prompt were
    // accepted and did nothing -- the same "field accepted and ignored" defect class the
    // embeddings `dimensions` serving audit already filed.
    //
    // Now keyed on params_base.n_cfg_guidance_slots: server-context.cpp's launch_slot_with_task
    // claims a guidance seq id from a pool sized by this flag and actually wires it through
    // decode (safe-subset only -- see the three additional refusals there for speculative
    // decoding / banned strings / prompt-checkpointing, which desync a guidance seq via rewind
    // paths not yet mirrored; row S69-SERVER-CFG-GUIDANCE-SEQ-MIRROR-REWINDS-2026-09-27-01 owns
    // lifting those). A flag of 0 (the default) keeps this identical to the original defect fix.
    if ((params.sampling.cfg_scale != 1.0f || !params.sampling.cfg_negative_prompt.empty())
            && params_base.n_cfg_guidance_slots <= 0) {
        throw std::runtime_error("Error: classifier-free guidance requires --cfg-guidance-slots; not enabled on this server");
    }

    // debugging
    {
        auto budget = params.sampling.reasoning_budget_tokens;
        SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu seqs, forced=%zu toks\n",
                budget, params.sampling.generation_prompt.c_str(),
                params.sampling.reasoning_budget_start.size(),
                params.sampling.reasoning_budget_end.size(),
                params.sampling.reasoning_budget_forced.size());
    }

    return params;
}

//
// TTS schema
//

std::vector<std::unique_ptr<field>> make_tts_schema(server_task & task) {
    std::vector<std::unique_ptr<field>> fields;
    auto add = [&](field * f) {
        fields.emplace_back(f);
    };

    add((new field_num("top_k", task.tts_inp.data.top_k))
        ->set_limits(0, INT32_MAX)
        ->set_desc("Top-k for the acoustic code predictor, 0 to use the model default"));

    add((new field_num("top_p", task.tts_inp.data.top_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Top-p for the acoustic code predictor, 0.0 to use the model default"));

    add((new field_num("seed", task.tts_inp.data.seed))
        ->set_desc("RNG seed for the backbone sampler and the codec/vocoder (-1 = random)"));

    add((new field_num("n_predict", task.params.n_predict))
        ->set_hard_limits(-1, INT32_MAX)
        ->add_alias("max_tokens")
        ->set_desc("Max number of audio frames to generate"));

    add((new field_num("repeat_penalty", task.params.sampling.penalty_repeat))
        ->set_desc("Repetition penalty applied to the backbone sampler over the whole generation"));

    add((new field_bool("stream", task.params.stream))
        ->set_desc("Stream the audio as it becomes available instead of waiting for the full generation"));

    return fields;
}

void eval_tts_schema(const common_params & params_base, server_task & task, const json & data) {
    // baseline defaults, individual requests can override them
    // TODO @ngxson : change the default values based on model
    task.params.sampling = params_base.sampling;
    task.params.sampling.penalty_repeat = 1.05f;
    task.params.n_predict = -1;
    task.tts_inp.data.top_k = 0;
    task.tts_inp.data.top_p = 0.0f;
    task.tts_inp.data.seed  = params_base.sampling.seed;

    field_eval_context ctx(task.params);

    for (const auto & f : make_tts_schema(task)) {
        f->eval(ctx, data);
    }
}

//
// eval() implementations
//

static void handle_with_catch(const char * name, std::function<void()> func) {
    try {
        func();
    } catch (const std::exception & e) {
        throw std::invalid_argument(string_format("Field '%s': %s", name, e.what()));
    }
}

// treat a null value as absent so clients can send null to request the server default
static bool has_value(const json & data, const char * n) {
    return data.contains(n) && !data.at(n).is_null();
}

// Hand a custom handler the body it expects: when the request used an alias of an alias_to_canonical field, a copy of
// the body with the value stored under the canonical name. Otherwise the body itself (no copy).
static void call_handler(const field & f, field_eval_context & ctx, const json & data, const char * matched) {
    if (f.alias_to_canonical && std::string(matched) != f.name[0]) {
        json copy = data;
        copy[f.name[0]] = data.at(matched);
        f.custom_handler(ctx, copy);
    } else {
        f.custom_handler(ctx, data);
    }
}

template <typename T>
void field_num<T>::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                if (custom_handler) {
                    call_handler(*this, ctx, data, n);
                } else if (!is_hard_limit) {
                    val = std::max(min, std::min(max, data.at(n).template get<T>()));
                } else {
                    T tmp = data.at(n).template get<T>();
                    if (tmp < min || tmp > max) {
                        throw std::invalid_argument(std::string("Value must be between ") + std::to_string(min) + " <= value <= " + std::to_string(max) + ", but got " + std::to_string(tmp));
                    }
                    val = tmp;
                }
            });
            return;
        }
    }
}

void field_str::eval(field_eval_context & ctx, const json & data) {
    GGML_ASSERT(custom_handler);
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                call_handler(*this, ctx, data, n);
            });
            return;
        }
    }
}

void field_bool::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                if (custom_handler) {
                    call_handler(*this, ctx, data, n);
                } else {
                    val = data.at(n).get<bool>();
                }
            });
            return;
        }
    }
}

void field_json::eval(field_eval_context & ctx, const json & data) {
    GGML_ASSERT(custom_handler);
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                call_handler(*this, ctx, data, n);
            });
            return;
        }
    }
}

void field_nested::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (data.contains(n) && data.at(n).is_object()) {
            for (auto & f : subfields) {
                f->eval(ctx, data.at(n));
            }
            return;
        }
    }
}

} // namespace server_schema
