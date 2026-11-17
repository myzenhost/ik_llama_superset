#include "server-task.h"
#include "server-reasoning-blob.h"

#include "asr-fallback.h"
#include "server-asr-format.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"
#include "watermark.h"

#include <cmath>
#include <sstream>
#include <unordered_set>

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    // ADAPTED to this fork's names, 2026-09-14. Two distinct enums coexist here BY DESIGN
    // (sampling.h:141 says so): common_sampler_type, and llama_sampler_type, which is what
    // samplers_sequence actually holds -- so the stringifier must be llama_sampling_type_to_str,
    // not common_sampler_type_to_str. Using the latter would not have compiled; naming it here so
    // a future reader does not "simplify" one into the other.
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers_sequence.size());
    for (const auto & sampler : sampling.samplers_sequence) {
        samplers.emplace_back(llama_sampling_type_to_str(sampler));
    }

    // ADAPTED: upstream read speculative.types, a flat vector this fork's
    // common_params_speculative does not carry. Built from the resolved stage chain instead --
    // the same translation common/arg.cpp:4323 and common/speculative.cpp:1724 already make,
    // and for the same reason: get_resolved_stages() captures which types the caller asked for,
    // including the single-.type legacy path via its own fallback.
    std::vector<common_speculative_type> spec_types;
    for (const auto & st : speculative.get_resolved_stages()) {
        spec_types.push_back(st.type);
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typical_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"penalty_scope",             sampling.penalty_scope_generated ? "generated" : "all"},
            {"penalize_prompt",           sampling.penalize_prompt},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"fsd_alpha",                sampling.fsd_alpha},
            {"fsd_pool_size",            sampling.fsd_pool_size},
            {"fsd_k",                    sampling.fsd_k},
            {"fsd_n",                    sampling.fsd_n},
            {"fsd_beta",                 sampling.fsd_beta},
            {"fsd_random_first_n",       sampling.fsd_random_first_n},
            {"adaptive_untruncated",      common_adaptive_untruncated_name(sampling.adaptive_untruncated)},
            {"adaptive_floor",            sampling.adaptive_floor},
            {"adaptive_p_floor_applied",  sampling.adaptive_untruncated == COMMON_ADAPTIVE_UNTRUNCATED_FLOOR && common_sampling_adaptive_p_untruncated(sampling)},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_format)},
            {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
            {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
            {"generation_prompt",         sampling.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(spec_types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typical_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"penalty_scope",             sampling.penalty_scope_generated ? "generated" : "all"},
        {"penalize_prompt",           sampling.penalize_prompt},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"fsd_alpha",                sampling.fsd_alpha},
        {"fsd_pool_size",            sampling.fsd_pool_size},
        {"fsd_k",                    sampling.fsd_k},
        {"fsd_n",                    sampling.fsd_n},
        {"fsd_beta",                 sampling.fsd_beta},
        {"fsd_random_first_n",       sampling.fsd_random_first_n},
        {"adaptive_untruncated",      common_adaptive_untruncated_name(sampling.adaptive_untruncated)},
        {"adaptive_floor",            sampling.adaptive_floor},
        {"adaptive_p_floor_applied",  sampling.adaptive_untruncated == COMMON_ADAPTIVE_UNTRUNCATED_FLOOR && common_sampling_adaptive_p_untruncated(sampling)},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"allowed_token_ids",         sampling.allowed_token_ids},
        {"suppress_sequences",        sampling.suppress_sequences},
        {"watermark",                 sampling.watermark},
        {"watermark_scheme",          common_watermark::scheme_name(sampling.watermark_scheme)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_continue_prefix",   sampling.grammar_continue_prefix},
        {"grammar_jump_forward",      grammar_jump_forward},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_format)},
        {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
        {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
        {"generation_prompt",         sampling.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(spec_types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
        {"top_a",                     sampling.top_a},
        {"smoothing_factor",          sampling.smoothing_factor},
        {"epsilon_cutoff",            sampling.epsilon_cutoff},
        {"eta_cutoff",                sampling.eta_cutoff},
        {"no_repeat_ngram_size",      sampling.no_repeat_ngram_size},
        {"no_repeat_ngram_last_n",    sampling.no_repeat_ngram_last_n},
        {"smoothing_curve",           sampling.smoothing_curve},
        {"repeat_penalty_slope",      sampling.penalty_slope},
        {"repeat_decay",              sampling.penalty_decay},
        {"skew",                      sampling.skew},
        {"banned_strings",            ban_phrases},
        {"bad_words",                 ban_phrases}, // alias of banned_strings, vLLM's field name (SamplingParams.bad_words)
        {"banned_regex",              ban_regex},
        {"banned_regex_case_insensitive", ban_regex_ci},
        {"banned_n",                  banned_n},
        {"banned_bias",               banned_bias},
        {"rewind_count_max",          rewind_count_max},
        {"banbuffer_size",            banbuffer_size},
        {"saturate_predict",          saturate_predict},
        {"min_tokens",                min_tokens},
        {"stop_token_ids",            stop_token_ids},
        {"logprob_token_ids",         logprob_token_ids},
        {"priority",                  priority},
        {"cache_salt",                cache_salt.empty() ? json(nullptr) : json("<set>")}, // never echo the salt itself
        {"extra_key",                 extra_key.empty()  ? json(nullptr) : json("<set>")},
        {"max_queue_wait_ms",         max_queue_wait_ms},
        {"stream_interval",           stream_interval},
        {"t_deadline_ms",            t_deadline_ms},
        {"t_max_prompt_ms",           t_max_prompt_ms},
        {"repetition_detection",      json {{"max_pattern_size", rep_max_pattern_size}, {"min_pattern_size", rep_min_pattern_size}, {"min_count", rep_min_count}, {"span_min_len", rep_span_min_len}, {"span_min_count", rep_span_min_count}, {"min_duplicate_tokens", rep_min_dup_tokens}, {"token_guard", rep_token_guard},{"tool_call_markers", json {{"begin", rep_tool_begin}, {"end", rep_tool_end}, {"min_count", rep_tool_min_count}}}, {"coverage_threshold", rep_cov_threshold}, {"coverage_window", rep_cov_window}, {"coverage_every", rep_cov_every}}},
        {"reasoning_loop_recovery",   rlr_mode == 2 ? "recover" : rlr_mode == 1 ? "stop" : "off"},
        {"prompt_logprobs",           prompt_logprobs},
        {"return_prompt_token_ids",   return_prompt_token_ids},
        {"skip_special_tokens",       skip_special_tokens},
        {"logprob_start_len",         logprob_start_len},
        {"include_stop_str_in_output", include_stop_str_in_output},
        {"stop_regex",                stop_regex},
        {"tool_call_fix",             sampling.tool_call_fix},
        {"expiring_logit_bias_stages", sampling.elb_params.size()},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(common_chat_session session)
    : chat_session(std::move(session))
    , chat_msg(chat_session.msg())
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
}

json task_result_state::oai_resp_failed_events(const std::string & resp_id, const json & error) {
    // error is a format_error_response() body: {code, message, type}; the Responses API wants {code: string, message}
    const std::string message = error.is_object() ? json_value(error, "message", std::string("generation failed")) : std::string("generation failed");
    const std::string code    = error.is_object() ? json_value(error, "type", std::string("server_error")) : std::string("server_error");
    return json::array({ json {
        {"event", "response.failed"},
        {"data", json {
            {"type", "response.failed"},
            {"response", json {
                {"id",     resp_id},
                {"object", "response"},
                {"status", "failed"},
                {"error",  json {{"code", code}, {"message", message}}},
            }},
        }},
    }});
}

bool task_result_state::oai_resp_tool_is_custom(const json & tool_map, const std::string & flat_name) {
    return tool_map.is_object() && tool_map.contains(flat_name) && json_value(tool_map.at(flat_name), "custom", false);
}

json task_result_state::oai_resp_tool_call_item(const json & tool_map, const std::string & id, const std::string & flat_name,
                                                const std::string & arguments, const char * status) {
    std::string name = flat_name;
    std::string ns;
    bool custom = false;
    if (tool_map.is_object() && tool_map.contains(flat_name)) {
        const json & e = tool_map.at(flat_name);
        name   = json_value(e, "name", flat_name);
        ns     = json_value(e, "namespace", std::string());
        custom = json_value(e, "custom", false);
    }
    json item;
    if (custom) {
        // the model wrote {"input": "<text>"}; the client gets the text itself
        std::string input = arguments;
        try {
            const json parsed = json::parse(arguments);
            if (parsed.is_object() && parsed.contains("input") && parsed.at("input").is_string()) {
                input = parsed.at("input").get<std::string>();
            }
        } catch (const std::exception &) {
            // partial or non-JSON arguments: keep the raw text
        }
        item = json {
            {"id",      "fc_" + id},
            {"type",    "custom_tool_call"},
            {"status",  status},
            {"call_id", "call_" + id},
            {"name",    name},
            {"input",   input},
        };
    } else {
        item = json {
            {"id",        "fc_" + id},
            {"type",      "function_call"},
            {"status",    status},
            {"arguments", arguments},
            {"call_id",   "call_" + id},
            {"name",      name},
        };
    }
    if (!ns.empty()) {
        item["namespace"] = ns;
    }
    return item;
}

json task_result_state::oai_resp_idle_events() {
    const json response = json {
        {"id",     oai_resp_id},
        {"object", "response"},
        {"status", "in_progress"},
    };
    json events = json::array();
    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {{"type", "response.created"}, {"response", response}}},
        });
        oai_resp_created = true;
    }
    events.push_back(json {
        {"event", "response.in_progress"},
        {"data", json {{"type", "response.in_progress"}, {"response", response}}},
    });
    return events;
}

common_chat_msg task_result_state::update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    auto msg_prv_copy = chat_msg;
    common_chat_msg new_msg;
    try {
        new_msg = is_partial ? chat_session.feed(added) : chat_session.finish(added);
    } catch (const std::exception & e) {
        // An output the chat parser rejects (an undeclared tool, a second call with parallel calls off, text that
        // misses the expected format) must not fail the request: surface the raw text as content so the client still
        // gets what the model generated. SOURCE-READ: ik_llama.cpp 801ed7c40 (#2642)
        // examples/server/server-context.cpp:776-794; MIT.
        SRV_WRN("chat parse failed (%s), returning the raw generated text as content\n", e.what());
        new_msg = common_chat_msg();
        new_msg.role    = "assistant";
        new_msg.content = chat_session.accumulated_input().text;
    }
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        case STOP_TYPE_REPETITION: return "repetition";
        case STOP_TYPE_DEADLINE:   return "deadline";
        case STOP_TYPE_GRAMMAR_ERROR: return "grammar_error";
        case STOP_TYPE_BUDGET_GROUP:  return "budget_group";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        json entry = json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        };
        if (!p.id_logprobs.empty()) {
            // always the raw (pre-sampling) distribution, whatever post_sampling_probs says
            json ids = json::array();
            for (const auto & il : p.id_logprobs) {
                ids.push_back(json {{"id", il.first}, {"logprob", il.second}});
            }
            entry["token_id_logprobs"] = std::move(ids);
        }
        out.push_back(std::move(entry));
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    json res = to_json_by_type();
    if (return_prompt_token_ids && !stream && res.is_object()) {
        json ids = json::array();
        for (const llama_token t : prompt_token_ids) {
            ids.push_back(t == LLAMA_TOKEN_NULL ? json(nullptr) : json(t));
        }
        res["prompt_token_ids"] = std::move(ids);
    }
    return res;
}

json server_task_result_cmpl_final::to_json_by_type() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            if (stream && !asr_flush.empty()) {
                // the deltas held these bytes as a possible language tag and the stream ended undecided: send them before done
                json events = json::array();
                events.push_back(json { {"type", "transcript.text.delta"}, {"delta", asr_flush} });
                events.push_back(to_json_oaicompat_asr());
                return events;
            }
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::prompt_logprobs_json() const {
    json lp = json::array();
    for (const float v : prompt_logprobs) {
        lp.push_back(std::isnan(v) ? json(nullptr) : json(v));
    }
    json greedy = json::array();
    for (const uint8_t g : prompt_greedy) {
        greedy.push_back(g != 0);
    }
    return json { {"logprobs", lp}, {"is_greedy", greedy} };
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content.text},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (stop == STOP_TYPE_GRAMMAR_ERROR) {
        res["grammar_error"] = grammar_error;
    }
    if (n_preempt_requeues > 0) {
        // the KV pool ran out while this request ran; it was requeued and recomputed from its prompt (--kv-preempt-requeue)
        res["preempt_requeues"] = n_preempt_requeues;
    }
    if (n_prompt_truncated > 0) {
        // --auto-truncate cut the middle of the prompt to fit the slot context; say so instead of answering silently
        res["prompt_truncated"] = true;
        res["prompt_tokens_dropped"] = n_prompt_truncated;
    }
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    if (!prompt_logprobs.empty()) {
        res["prompt_logprobs"] = prompt_logprobs_json();
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    json usage = json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        // OpenAI field: the share of completion_tokens the model spent inside its thinking block (0 without one)
        {"completion_tokens_details", json { {"reasoning_tokens", n_reasoning_tokens} }},
    };
    if (n_prompt_truncated > 0) {
        // --auto-truncate: extension fields beside the OpenAI ones
        usage["prompt_truncated"] = true;
        usage["prompt_tokens_dropped"] = n_prompt_truncated;
    }
    return usage;
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
        if (echo_prompt_scored()) {
            // legacy OpenAI completions arrays (what lm-eval's completions client reads): one entry per prompt token, then one per
            // generated token. The first prompt token has no context, so its log-probability is null. top_logprobs holds the top
            // token of each position; for the prompt positions that is the model's own top choice, which lm-eval compares with the
            // token's log-probability to decide "greedy".
            json tokens_j = json::array(), token_lp = json::array(), top_lp = json::array(), offsets = json::array();
            size_t offset = 0;
            auto add_entry = [&](std::string piece, json lp, json top) {
                piece.resize(validate_utf8(piece));
                offsets.push_back(offset);
                offset += piece.size();
                tokens_j.push_back(std::move(piece));
                token_lp.push_back(std::move(lp));
                top_lp.push_back(std::move(top));
            };
            auto finite_or_null = [](float v) { return std::isnan(v) ? json(nullptr) : json(v); };
            for (size_t i = 0; i < prompt_pieces.size(); ++i) {
                json top = json(nullptr);
                if (i > 0 && i < prompt_arg.size() && prompt_arg[i] >= 0 && !std::isnan(prompt_arg_lp[i])) {
                    std::string k = prompt_arg_pieces[i];
                    k.resize(validate_utf8(k));
                    top = json::object({{ k, prompt_arg_lp[i] }});
                }
                add_entry(prompt_pieces[i], i == 0 ? json(nullptr) : finite_or_null(prompt_logprobs[i]), std::move(top));
            }
            for (const auto & g : probs_output) {
                json top = json::object();
                for (const auto & p : g.probs) {
                    std::string k(p.txt);
                    k.resize(validate_utf8(k));
                    top[k] = completion_token_output::logarithm(p.prob);
                }
                add_entry(g.text_to_send, completion_token_output::logarithm(g.prob), std::move(top));
            }
            logprobs["tokens"]         = std::move(tokens_j);
            logprobs["token_logprobs"] = std::move(token_lp);
            logprobs["top_logprobs"]   = std::move(top_lp);
            logprobs["text_offset"]    = std::move(offsets);
        }
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS || stop == STOP_TYPE_REPETITION) {
        finish_reason = "stop";
    } else if (stop == STOP_TYPE_DEADLINE) {
        finish_reason = "deadline";
    } else if (stop == STOP_TYPE_GRAMMAR_ERROR) {
        finish_reason = "grammar_error";
    } else if (stop == STOP_TYPE_BUDGET_GROUP) {
        finish_reason = "budget_group";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          echo ? prompt + content.text : content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

// Row S69-FASTLLM-THINKING-BOUNDARY-MENTION-AND-EMPTY-REPLY-C8-2026-10-27-01 (d). The model ended the turn on its own while
// the generation prompt had opened thinking, and the reply carries neither text nor a tool call: the reasoning ran out into
// EOS. "stop" with an empty message reads as "done" to an agent client, so the turn is lost without a sign. It is answered
// with a retryable error instead (request field error_on_empty_thinking_reply, default on). A turn cut by max_tokens, the
// context or a deadline keeps finish_reason "length".
std::unique_ptr<server_task_result> server_task_result_cmpl_final::to_retryable_error() {
    if (!empty_thinking_reply) {
        return nullptr;
    }
    auto err      = std::make_unique<server_task_result_error>();
    err->id       = id;
    err->id_slot  = id_slot;
    err->index    = index;
    err->err_type = ERROR_TYPE_SERVER;
    err->err_msg  = "invalid_thinking_block: the model stopped after its reasoning without an answer or a tool call (retry the request)";
    SRV_WRN("%s", "empty reply after thinking: returning a retryable error instead of an empty finish\n");
    return err;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS || stop == STOP_TYPE_REPETITION) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    } else if (stop == STOP_TYPE_DEADLINE) {
        finish_reason = "deadline";
    } else if (stop == STOP_TYPE_GRAMMAR_ERROR) {
        finish_reason = "grammar_error";
    } else if (stop == STOP_TYPE_BUDGET_GROUP) {
        finish_reason = "budget_group";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };
    if (stop == STOP_TYPE_GRAMMAR_ERROR) {
        choice["grammar_error"] = grammar_error; // the first llguidance / xgrammar matcher error
    }

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS || stop == STOP_TYPE_REPETITION) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    } else if (stop == STOP_TYPE_DEADLINE) {
        finish_reason = "deadline";
    } else if (stop == STOP_TYPE_GRAMMAR_ERROR) {
        finish_reason = "grammar_error";
    } else if (stop == STOP_TYPE_BUDGET_GROUP) {
        finish_reason = "budget_group";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", oai_resp_reasoning_summary ? json::array({ json {{"type", "summary_text"}, {"text", msg.reasoning_content}} }) : json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", oai_resp_reasoning_blob ? server_reasoning_blob::encode(msg.reasoning_content) : std::string()},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
            // commentary when a tool call follows, final_answer at a terminal message (spec: sglang serving_responses.py:1915-1936)
            {"phase",  oaicompat_msg.tool_calls.empty() ? "final_answer" : "commentary"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(task_result_state::oai_resp_tool_call_item(oai_resp_tool_map, tool_call.id, tool_call.name,
                                                                    tool_call.arguments, "completed"));
    }

    // the answer was cut by max_output_tokens (or the context): "incomplete", not "completed"
    const bool incomplete = stop == STOP_TYPE_LIMIT || stop == STOP_TYPE_DEADLINE || stop == STOP_TYPE_GRAMMAR_ERROR || stop == STOP_TYPE_BUDGET_GROUP;
    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       incomplete ? "incomplete" : "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
            {"output_tokens_details", json { {"reasoning_tokens", n_reasoning_tokens} }},
        }},
    };

    if (incomplete) {
        res["incomplete_details"] = json {{"reason", "max_output_tokens"}};
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;
    size_t output_index = 0;

    if (oaicompat_msg.reasoning_content != "") {
        const size_t item_output_index = output_index++;
        const json summary_part = json {{"type", "summary_text"}, {"text", oaicompat_msg.reasoning_content}};
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", oai_resp_reasoning_summary ? json::array({summary_part}) : json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", oai_resp_reasoning_blob ? server_reasoning_blob::encode(oaicompat_msg.reasoning_content) : std::string()},
        };

        if (oai_resp_reasoning_summary) {
            // summary requested: summary_text.done then summary_part.done (sglang serving_responses.py:2076-2124)
            server_sent_events.push_back(json {
                {"event", "response.reasoning_summary_text.done"},
                {"data", json {
                    {"type",          "response.reasoning_summary_text.done"},
                    {"item_id",       oai_resp_reasoning_id},
                    {"output_index",  item_output_index},
                    {"summary_index", 0},
                    {"text",          oaicompat_msg.reasoning_content},
                }}
            });
            server_sent_events.push_back(json {
                {"event", "response.reasoning_summary_part.done"},
                {"data", json {
                    {"type",          "response.reasoning_summary_part.done"},
                    {"item_id",       oai_resp_reasoning_id},
                    {"output_index",  item_output_index},
                    {"summary_index", 0},
                    {"part",          summary_part},
                }}
            });
        } else {
            server_sent_events.push_back(json {
                {"event", "response.reasoning_text.done"},
                {"data", json {
                    {"type",          "response.reasoning_text.done"},
                    {"item_id",       oai_resp_reasoning_id},
                    {"output_index",  item_output_index},
                    {"content_index", 0},
                    {"text",          oaicompat_msg.reasoning_content},
                }}
            });
        }

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        const size_t item_output_index = output_index++;
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"},
            {"phase",   oaicompat_msg.tool_calls.empty() ? "final_answer" : "commentary"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const size_t item_output_index = output_index++;
        const json output_item = task_result_state::oai_resp_tool_call_item(oai_resp_tool_map, tool_call.id, tool_call.name,
                                                                            tool_call.arguments, "completed");
        if (output_item.at("type") == "custom_tool_call") {
            // a custom tool's input is not streamed piece by piece (the model writes it inside a JSON string): it goes out whole
            server_sent_events.push_back(json {
                {"event", "response.custom_tool_call_input.delta"},
                {"data", json {
                    {"type",    "response.custom_tool_call_input.delta"},
                    {"item_id", output_item.at("id")},
                    {"delta",   output_item.at("input")},
                }}
            });
            server_sent_events.push_back(json {
                {"event", "response.custom_tool_call_input.done"},
                {"data", json {
                    {"type",    "response.custom_tool_call_input.done"},
                    {"item_id", output_item.at("id")},
                    {"input",   output_item.at("input")},
                }}
            });
        }
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    // cut by max_output_tokens (or the context): the stream ends with response.incomplete, as the official API does
    const bool incomplete = stop == STOP_TYPE_LIMIT || stop == STOP_TYPE_DEADLINE || stop == STOP_TYPE_GRAMMAR_ERROR || stop == STOP_TYPE_BUDGET_GROUP;
    const char * final_event = incomplete ? "response.incomplete" : "response.completed";
    json final_response = json {
        {"id",         oai_resp_id},
        {"object",     "response"},
        {"created_at", (int64_t) std::time(0)},
        {"status",     incomplete ? "incomplete" : "completed"},
        {"model",      oaicompat_model},
        {"output",     output},
        {"usage",      json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
            {"output_tokens_details", json { {"reasoning_tokens", n_reasoning_tokens} }},
        }}
    };
    if (incomplete) {
        final_response["incomplete_details"] = json {{"reason", "max_output_tokens"}};
    }
    server_sent_events.push_back(json {
        {"event", final_event},
        {"data", json {
            {"type",     final_event},
            {"response", final_response},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    // a model language tag ("language English<asr_text>", Qwen3-ASR) is output framing, not speech: strip it first, then a
    // balanced outer quote pair (both server-asr-format.h). Every response_format renders from this "text".
    const asrfmt::language_tagged tag = asrfmt::strip_language_tag(oaicompat_msg.content);
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  asrfmt::strip_wrapping_quotes(tag.text)},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    if (!tag.language.empty()) {
        // the language the model itself detected; verbose_json reports it when the request named none (server-context.cpp)
        event["language"] = tag.language;
    }
    // One pass over the sampled-token log-probabilities serves both consumers below.
    double sum_lp = 0.0;
    std::vector<int32_t> toks;
    for (const auto & p : probs_output) {
        sum_lp += (double) completion_token_output::logarithm(p.prob);
        toks.push_back((int32_t) p.tok);
    }
    // mean log-probability of the chosen tokens, only when the request asked for probabilities (verbose_json does, and so does the
    // --asr-quality retry loop); the plain json body is left exactly as it was otherwise.
    if (!probs_output.empty()) {
        event["avg_logprob"] = sum_lp / (double) probs_output.size();
        // no_speech_prob proxy: end-of-generation probability at the first generated position (completion_token_output::eog_prob)
        if (probs_output.front().eog_prob >= 0.0f) {
            event["no_speech_prob"] = probs_output.front().eog_prob;
        }
    }
    if (asr_stats) {
        // --asr-quality: what the retry loop gates on. avg_logprob = mean log-probability of the sampled tokens (the
        // distribution before the sampler chain, as n_probs reports it); tail_entropy = entropy (nats) of the token
        // histogram of the last 32 tokens
        event["asr_stats"] = json {
            {"n_tokens",     toks.size()},
            {"avg_logprob",  toks.empty() ? 0.0 : sum_lp / (double) toks.size()},
            {"tail_entropy", toks.empty() ? 1e9 : asr_tail_entropy(toks, 32)},
        };
    }
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS || stop == STOP_TYPE_REPETITION) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS || stop == STOP_TYPE_REPETITION) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    if (res_type == TASK_RESPONSE_TYPE_OAI_ASR) {
        asr_delta = state.asr_tag.push(content.text);
    }

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;
    oai_resp_fc_custom     = state.oai_resp_fc_custom;
    oai_resp_tool_map      = state.oai_resp_tool_map;
    oai_resp_reasoning_summary = state.oai_resp_reasoning_summary;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id     = diff.tool_call_delta.id;
            state.oai_resp_fc_custom = task_result_state::oai_resp_tool_is_custom(state.oai_resp_tool_map, diff.tool_call_delta.name);
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content.text},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", 0},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                if (oai_resp_reasoning_summary) {
                    events.push_back(json {
                        {"event", "response.reasoning_summary_part.added"},
                        {"data", json {
                            {"type",          "response.reasoning_summary_part.added"},
                            {"item_id",       oai_resp_reasoning_id},
                            {"output_index",  0},
                            {"summary_index", 0},
                            {"part",          json {{"type", "summary_text"}, {"text", ""}}},
                        }},
                    });
                }
                thinking_block_started = true;
            }
            if (oai_resp_reasoning_summary) {
                events.push_back(json {
                    {"event", "response.reasoning_summary_text.delta"},
                    {"data", json {
                        {"type",          "response.reasoning_summary_text.delta"},
                        {"item_id",       oai_resp_reasoning_id},
                        {"output_index",  0},
                        {"summary_index", 0},
                        {"delta",         diff.reasoning_content_delta},
                    }},
                });
            } else {
                events.push_back(json {
                    {"event", "response.reasoning_text.delta"},
                    {"data", json {
                        {"type",    "response.reasoning_text.delta"},
                        {"output_index", 0},
                        {"content_index", 0},
                        {"delta",   diff.reasoning_content_delta},
                        {"item_id", oai_resp_reasoning_id},
                    }},
                });
            }
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"content_index", 0},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"output_index", anthropic_has_reasoning ? 1 : 0},
                    {"content_index", 0},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                    {"item", task_result_state::oai_resp_tool_call_item(oai_resp_tool_map, diff.tool_call_delta.id,
                                                                        diff.tool_call_delta.name, "", "in_progress")},
                }},
            });
            oai_resp_fc_id     = diff.tool_call_delta.id;
            oai_resp_fc_custom = task_result_state::oai_resp_tool_is_custom(oai_resp_tool_map, diff.tool_call_delta.name);
        }

        // a custom tool's input is written inside a JSON string by the model, so it cannot be streamed raw:
        // it goes out whole in response.custom_tool_call_input.done
        if (!diff.tool_call_delta.arguments.empty() && !oai_resp_fc_custom) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                    {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    // update() ran the text through task_result_state::asr_tag: a held piece (possibly a language tag) sends no event
    // (format_oai_sse writes nothing for an empty array)
    json events = json::array();
    if (!asr_delta.empty()) {
        events.push_back(json {
            {"type", "transcript.text.delta"},
            {"delta", asr_delta},
        });
    }
    return events;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_tts
//
json server_task_result_tts::to_json() {
    return json {
        {"sample_rate", sample_rate},
        {"n_bytes",     audio.size()},
        {"final",       final},
    };
}

//
// server_task_result_decision
//
json server_task_result_decision::to_json() {
    return json {
        {"index",            index},
        {"scores",           scores},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "grammar_jump_forward_tokens_total",
            "Tokens appended without their own sampling step because a grammar forced their text (--grammar-jump-forward)",
            (double) metrics.n_jump_forward_tokens
        }, {
            "requests_preempted_total",
            "In-flight requests preempted (cells freed; answered 503 or, with --kv-preempt-requeue, requeued) because the --kv-unified KV pool ran out of cells",
            (double) metrics.n_kv_preempted
        }, {
            "requests_preempt_requeued_total",
            "Preempted requests put back in the queue to be recomputed instead of answered 503 (--kv-preempt-requeue)",
            (double) metrics.n_kv_requeued
        }, {
            "kv_preempt_discarded_cells_total",
            "KV cells held by preemption victims at the moment they were evicted: the computed work a recompute has to repeat (--kv-preempt-victim picks who pays)",
            (double) metrics.n_kv_preempt_cells
        }, {
            "kv_cache_idle_slot_purges_total",
            "Idle slots whose cached prompt was purged to free cells under --kv-unified",
            (double) metrics.n_kv_idle_purged
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        }, {
            "kv_cache_tokens_active",
            "KV tokens held by slots that are processing a request",
            (double) n_kv_tokens_active
        }, {
            "kv_cache_tokens_held",
            "KV tokens held by all slots, idle slots included (kept for prompt reuse)",
            (double) n_kv_tokens_held
        }, {
            "kv_cache_tokens_capacity",
            "KV token capacity across all slots",
            (double) n_kv_tokens_capacity
        }, {
            "kv_cache_usage_ratio",
            "Active KV tokens over capacity (0..1)",
            n_kv_tokens_capacity > 0 ? (double) n_kv_tokens_active / (double) n_kv_tokens_capacity : 0.0
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    if (has_radix) {
        add_items("counter", std::vector<metric_item>{
            { "radix_hits_total", "Prefix lookups in the radix cache that reused at least one token", (double) radix_hits },
            { "radix_misses_total", "Prefix lookups in the radix cache that reused nothing", (double) radix_misses },
            { "radix_hit_tokens_total", "Tokens reused through radix-cache hits", (double) radix_hit_tokens },
            { "radix_evictions_total", "Idle radix-cache leaves evicted to make room", (double) radix_evictions },
        });
        add_items("gauge", std::vector<metric_item>{
            { "radix_nodes", "Nodes held by the radix cache", (double) radix_nodes },
        });
    }

    if (has_mmproj) {
        add_items("counter", std::vector<metric_item>{
            { "mmproj_encodes_total", "Image chunks that really ran through the vision tower", (double) mmproj_encodes },
            { "mmproj_embd_cache_hits_total", "Images served from the vision-encoder output cache (--mmproj-embd-cache)", (double) mmproj_hits },
            { "mmproj_embd_cache_misses_total", "Image lookups the cache could not serve", (double) mmproj_misses },
            { "mmproj_embd_cache_evictions_total", "Cached encoder outputs dropped to make room", (double) mmproj_evictions },
            { "mmproj_embd_cache_not_stored_total", "Encoder outputs the cache could not keep (over budget, or only the same request's entries in the way)", (double) mmproj_not_stored },
        });
        add_items("gauge", std::vector<metric_item>{
            { "mmproj_embd_cache_bytes", "Bytes held by the vision-encoder output cache", (double) mmproj_bytes },
            { "mmproj_embd_cache_entries", "Entries held by the vision-encoder output cache", (double) mmproj_entries },
            { "mmproj_embd_cache_budget_bytes", "Byte budget of the vision-encoder output cache (0 = off)", (double) mmproj_budget },
        });
    }

    // per-device memory gauges, same figures as GET /memory (ik_llama.cpp 55813a5d7 render_memory_metrics)
    if (memory.is_array()) {
        static const char * keys[] = { "model", "context", "compute", "mmproj", "total", "free" };
        static const char * helps[] = {
            "Memory allocated for the model weights, in bytes.",
            "Memory allocated for the KV cache, in bytes.",
            "Memory allocated for compute buffers, in bytes.",
            "Memory allocated for the multimodal projector, in bytes.",
            "Total memory of the device, in bytes.",
            "Free memory of the device, in bytes.",
        };
        for (int k = 0; k < 6; ++k) {
            const std::string name = std::string("memory_") + keys[k] + "_bytes";
            bool described = false;
            for (const json & entry : memory) {
                if (!entry.contains(keys[k])) {
                    continue;
                }
                if (!described) {
                    prometheus << "# HELP llamacpp:" << name << " " << helps[k] << "\n"
                               << "# TYPE llamacpp:" << name << " gauge\n";
                    described = true;
                }
                std::string device = entry.at("name").get<std::string>();
                string_replace_all(device, "\\", "\\\\");
                string_replace_all(device, "\"", "\\\"");
                string_replace_all(device, "\n", "\\n");
                prometheus << "llamacpp:" << name << "{device=\"" << device << "\"} " << entry.at(keys[k]).dump() << "\n";
            }
        }
    }

    // latency histograms, step composition (decode-only / mixed / prefill-only), finish reasons: server-latency.h
    metrics.lat.render(prometheus, "llamacpp:");
    if (metrics.reqlog) {
        metrics.reqlog->render_counter(prometheus);   // --request-log: llamacpp:request_log_count{model,status}
    }

    // NVMe tier of the prompt cache (--cache-disk-dir): only present when the tier is on, so a server without the flag
    // exports exactly the metrics it always did. written_bytes is the SSD-wear counter.
    if (disk.enabled) {
        add_items("counter", {
            { "prompt_cache_disk_spill_total",            "Prompt states moved from the RAM cache to the disk tier", (double) disk.n_spilled },
            { "prompt_cache_disk_written_bytes_total",    "File bytes written to the disk tier (SSD wear)",          (double) disk.bytes_written },
            { "prompt_cache_disk_restore_total",          "Prompt states read back from the disk tier and used",     (double) disk.n_restored },
            { "prompt_cache_disk_read_bytes_total",       "File bytes read back from the disk tier",                 (double) disk.bytes_read },
            { "prompt_cache_disk_restore_failed_total",   "Spilled prompt states that could not be read back and were dropped", (double) disk.n_restore_failed },
            { "prompt_cache_disk_checksum_failed_total",  "Blobs whose checksum did not match when read back",       (double) disk.n_checksum_failed },
            { "prompt_cache_disk_io_failed_total",        "Disk tier reads or writes that failed on I/O",            (double) disk.n_io_failed },
            { "prompt_cache_disk_refused_cap_total",      "Spills refused because the disk tier was full",           (double) disk.n_refused_cap },
            { "prompt_cache_disk_refused_budget_total",   "Spills refused because the write budget was spent",       (double) disk.n_refused_budget },
            { "prompt_cache_disk_refused_space_total",    "Spills refused by the free-space guard because the volume was (nearly) full", (double) disk.n_refused_space },
            { "prompt_cache_disk_dropped_small_total",    "Prompt states evicted without a spill, below the minimum token count", (double) disk.n_dropped_small },
            { "prompt_cache_disk_dropped_refused_total",  "Prompt states evicted without a spill because the tier refused them", (double) disk.n_dropped_refused },
            { "prompt_cache_disk_evicted_total",          "Spilled prompt states removed to make room on disk",      (double) disk.n_disk_evicted },
            { "prompt_cache_disk_ckpt_spill_total",       "Context checkpoints of spilled prompt states written to the disk tier", (double) disk.n_ckpt_spilled },
            { "prompt_cache_disk_ckpt_refused_total",     "Context checkpoints the disk tier refused (kept in memory)", (double) disk.n_ckpt_refused },
            { "prompt_cache_disk_ckpt_restore_total",     "Context checkpoints read back from the disk tier and used", (double) disk.n_ckpt_restored },
            { "prompt_cache_disk_ckpt_restore_failed_total", "Disk-tier checkpoints that failed their checks (dropped, the prompt is recomputed)", (double) disk.n_ckpt_restore_failed },
            { "prompt_cache_disk_ckpt_adopted_total",     "Slot checkpoints held in a --ctx-ckpt-spill-dir file, copied into a cache entry", (double) disk.n_ckpt_adopted },
            { "prompt_cache_disk_ckpt_slot_spill_total",  "A slot's live checkpoints moved to the disk tier (--ctx-ckpt-spill-dir with --cache-disk-dir)", (double) disk.n_ckpt_slot_spilled },
            { "prompt_cache_disk_ckpt_slot_refused_total", "Slot checkpoints the disk tier refused (kept in the slot's memory)", (double) disk.n_ckpt_slot_refused },
            { "prompt_cache_disk_dropped_slow_total",     "Prompt states evicted without a spill because restoring them would take longer than recomputing", (double) disk.n_dropped_slow },
            { "prompt_cache_disk_ckpt_dropped_slow_total", "Checkpoints kept in memory because restoring them would take longer than recomputing", (double) disk.n_ckpt_dropped_slow },
            { "prompt_cache_disk_restore_skipped_total",  "Spilled prompt states that matched a request but were not restored: recomputing the matched prefix is quicker", (double) disk.n_restore_skipped },
            { "prompt_cache_disk_ckpt_written_bytes_total", "File bytes of checkpoint blobs written to the disk tier (part of written_bytes)", (double) disk.bytes_written_ckpt },
            { "prompt_cache_disk_ckpt_read_bytes_total",  "File bytes of checkpoint blobs read back from the disk tier (part of read_bytes)", (double) disk.bytes_read_ckpt },
        });
        add_items("gauge", {
            { "prompt_cache_disk_used_bytes",  "Bytes currently in the disk tier",           (double) disk.used_bytes },
            { "prompt_cache_disk_limit_bytes", "Size cap of the disk tier",                  (double) disk.limit_bytes },
            { "prompt_cache_disk_entries",     "Prompt states currently held in the disk tier", (double) disk.n_entries },
            { "prompt_cache_disk_prefill_tokens_per_second", "Measured prompt processing speed used by the restore-versus-recompute rule (0 = not measured yet)", disk.prefill_tps },
            { "prompt_cache_disk_restore_bytes_per_second",  "Measured restore speed used by the restore-versus-recompute rule (0 = unknown)", disk.restore_Bps },
            { "prompt_cache_disk_min_speedup",               "The rule's margin: a restore must be this many times quicker than recomputing (0 = rule off)", disk.min_speedup },
            { "prompt_cache_ram_bytes",        "Bytes the host prompt cache holds in memory", (double) disk.ram_bytes },
            { "prompt_cache_ram_ckpt_bytes",   "Of prompt_cache_ram_bytes: context checkpoints", (double) disk.ram_ckpt_bytes },
        });
    }

    // --spec-async diagnostic counters: only present when the async drafter is on
    if (metrics.spec_async_on) {
        for (const auto & it : common_speculative_async_stats_items(metrics.spec_async)) {
            const std::string full = "llamacpp:spec_async_" + it.name;
            prometheus << "# HELP " << full << " " << it.help << "\n"
                       << "# TYPE " << full << (it.gauge ? " gauge\n" : " counter\n")
                       << full << " " << it.value << "\n";
        }
    }

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    // state bytes per entry plus each checkpoint object once (see size_ckpt)
    size_t res = size_ckpt();

    for (const auto & state : states) {
        res += state.data.size();
    }

    return res;
}

void server_prompt_cache::touch(std::list<server_prompt_cache_state>::iterator it) {
    it->n_hits++;
    states.splice(states.end(), states, it);
}

bool server_prompt_cache::evict_one() {
    // HARNESS-CONSIDERED: this is eviction inside the engine's own prompt cache; it removes cached bytes, not a model capability.
    // The NVMe tier added here EXTENDS the reusable context (long-context state is kept on disk instead of being discarded).
    if (states.empty()) {
        return false;
    }

    // The victim is chosen among the entries whose bytes are held in memory: a spilled entry costs no memory (its bytes are
    // in the NVMe tier), so removing one frees none and the callers' loops would not progress. Without the tier no entry is
    // ever spilled, so the first entry is always the first one in memory and this is the choice made before the tier existed.
    auto first_resident = states.end();
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (!it->spilled() && !it->pinned) {
            first_resident = it;
            break;
        }
    }
    if (first_resident == states.end()) {
        // every unpinned entry is already on disk but its checkpoints still occupy memory: free the oldest unpinned one completely
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (!it->pinned) {
                SRV_WRN(" - removing oldest spilled cache entry (checkpoints only: %.3f MiB)\n", it->size() / (1024.0 * 1024.0));
                ctr.n_disk_evicted++;
                drop_state(it);
                return true;
            }
        }
        return false; // only pinned entries are left in memory
    }

    auto victim = first_resident;
    if (slru) {
        // protected segment over 80 percent of the byte limit: shrink it first, oldest protected entry
        size_t protected_size = 0;
        for (const auto & s : states) {
            if (s.n_hits > 0) {
                protected_size += s.size();
            }
        }
        const bool shrink_protected = limit_size > 0 && protected_size * 5 > limit_size * 4;
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (!it->spilled() && !it->pinned && (it->n_hits > 0) == shrink_protected) {
                victim = it;
                break;
            }
        }
    }

    // NVMe tier: write the bytes to disk and keep the entry (tokens, checkpoints) in place so a later prefix match finds it.
    if (spill_state(victim)) {
        return true;
    }

    SRV_WRN(" - removing %s cache entry (size = %.3f MiB, hits = %u)\n", victim == states.begin() ? "oldest" : "probation",
            victim->size() / (1024.0 * 1024.0), (unsigned) victim->n_hits);
    drop_state(victim);
    return true;
}

size_t server_prompt_cache::pinned_size() const {
    size_t res = 0;
    for (const auto & state : states) {
        if (state.pinned) {
            res += state.size();
        }
    }
    return res;
}

size_t server_prompt_cache::n_pinned() const {
    size_t n = 0;
    for (const auto & state : states) {
        n += state.pinned ? 1 : 0;
    }
    return n;
}

size_t server_prompt_cache::pin_prefix_of(const server_tokens & tokens, uint64_t lora_fp) {
    size_t n = 0;
    for (auto & state : states) {
        const size_t len = state.prompt.tokens.size();
        if (len > 0 && state.prompt.lora_fp == lora_fp && (size_t) state.prompt.tokens.get_common_prefix(tokens) == len && !state.pinned) {
            state.pinned = true;
            n++;
        }
    }
    return n;
}

bool server_prompt_cache::has_pinned_prefix_of(const server_tokens & tokens, uint64_t lora_fp) const {
    for (const auto & state : states) {
        const size_t len = state.prompt.tokens.size();
        if (state.pinned && len > 0 && state.prompt.lora_fp == lora_fp && (size_t) state.prompt.tokens.get_common_prefix(tokens) == len) {
            return true;
        }
    }
    return false;
}

size_t server_prompt_cache::drop_pinned() {
    size_t n = 0;
    for (auto it = states.begin(); it != states.end();) {
        if (it->pinned) {
            it->pinned = false;
            it = drop_state(it);
            n++;
        } else {
            ++it;
        }
    }
    return n;
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::drop_state(std::list<server_prompt_cache_state>::iterator it) {
    if (it->spilled() && disk) {
        disk->erase(it->disk_id);
    }
    // let go of the entry's share of each checkpoint; the last owner (this entry, if the slot dropped its share earlier) also
    // deletes a spill file the object still names
    for (auto & ck : it->prompt.checkpoints) {
        server_ckpt_release(ck);
    }
    return states.erase(it);
}

bool server_prompt_cache::enable_disk(const std::string & dir, size_t limit_bytes, size_t min_tokens, size_t write_budget_bytes_per_hour,
                                      std::string & err, common_disk_tier::clock_fn clock) {
    if (dir.empty() || limit_bytes == 0) {
        err = dir.empty() ? "no directory" : "size cap is 0";
        return false;
    }
    auto tier = std::make_shared<common_disk_tier>(dir, limit_bytes, write_budget_bytes_per_hour, std::move(clock));
    int n_stale = 0;
    if (!tier->init(n_stale, err)) {
        return false;
    }
    SRV_INF("prompt cache disk tier %s ready: cap %.1f MiB, min %zu tokens, write budget %s, %d stale file(s) removed\n",
            dir.c_str(), limit_bytes / (1024.0 * 1024.0), min_tokens,
            write_budget_bytes_per_hour == 0 ? "unlimited" : (std::to_string(write_budget_bytes_per_hour / (1024 * 1024)) + " MiB/h").c_str(),
            n_stale);
    disk            = std::move(tier);
    disk_min_tokens = min_tokens;
    return true;
}

bool server_prompt_cache::spill_state(std::list<server_prompt_cache_state>::iterator it) {
    if (!disk || it->spilled() || it->data.main.empty()) {
        return false;
    }
    if ((size_t) it->prompt.n_tokens() < disk_min_tokens) {
        ctr.n_dropped_small++;
        return false;
    }

    const size_t n_main = it->data.main.size();
    const size_t n_drft = it->data.drft.size();

    // a state that takes longer to read back than to recompute is not worth the write (nor the wear)
    if (!restore_pays((size_t) it->prompt.n_tokens(), n_main + n_drft)) {
        ctr.n_dropped_slow++;
        return false;
    }

    // Make room by removing the oldest spilled entries -- but only if the write budget will accept this one, otherwise the
    // older (already paid for) entries would be thrown away for nothing.
    make_disk_room(disk->file_size_for(n_main, n_drft), it);

    // put() refuses (and counts why) when the cap, the budget or the free-space guard does not allow it (/metrics refused_{cap,budget,space}_total)
    const uint64_t id = disk->put(it->data.main, it->data.drft, it->prompt.n_tokens());
    if (id == 0) {
        ctr.n_dropped_refused++;
        SRV_WRN(" - could not spill prompt of %d tokens (%.3f MiB) to disk: tier full, over the write budget, volume low on free space or an I/O error\n",
                it->prompt.n_tokens(), (n_main + n_drft) / (1024.0 * 1024.0));
        return false;
    }

    SRV_TRC(" - spilled prompt of %d tokens to disk (%.3f MiB)\n", it->prompt.n_tokens(), (n_main + n_drft) / (1024.0 * 1024.0));
    it->disk_id      = id;
    it->disk_n_bytes = n_main + n_drft;
    std::vector<uint8_t>().swap(it->data.main);
    std::vector<uint8_t>().swap(it->data.drft);
    ctr.n_spilled++;

    // The entry's context checkpoints go the same way (row S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01).
    // HARNESS-CONSIDERED: this moves cached bytes between memory tiers inside the engine's own prompt cache; it removes no model
    // capability. Before this the checkpoints stayed in memory and size() counted them, so a hybrid/SWA model's spilled entry still
    // cost most of its RAM. Each checkpoint is its own blob, written after the main state so that state has first call on the
    // write budget; a checkpoint the tier refuses stays in memory and the entry is still usable. The blob is reference counted
    // (common_disk_blob): the slot that restores this entry holds copies of the checkpoints, and a blob lives until the last copy
    // is released.
    for (auto & ckp : it->prompt.checkpoints) {
        auto & ck = *ckp;
        if (ck.empty()) {
            continue; // already spilled, or its data is in a --ctx-ckpt-spill-dir file (alloc() adopts those when the tier is on)
        }
        if (ckp.use_count() > 1) {
            // shared with a live slot (or another entry): the bytes stay resident for that owner whatever this entry does, so
            // spilling would only move them and make the slot reload them; it stays in memory like a refused one
            continue;
        }
        if (!restore_pays((size_t) std::max<int64_t>(ck.n_tokens, 0), ck.tier_payload_a() + ck.tier_payload_b())) {
            ctr.n_ckpt_dropped_slow++;
            continue; // stays in memory, like a refused one
        }
        make_disk_room(disk->file_size_for(ck.tier_payload_a(), ck.tier_payload_b()), it);
        if (ck.spill_tier(disk)) {
            ctr.n_ckpt_spilled++;
        } else {
            ctr.n_ckpt_refused++;
            SRV_WRN(" - could not spill a context checkpoint (%.3f MiB) to disk: tier full, over the write budget, volume low on free space or an I/O error; it stays in memory\n",
                    (ck.tier_payload_a() + ck.tier_payload_b()) / (1024.0 * 1024.0));
        }
    }
    return true;
}

bool server_prompt_cache::make_disk_room(size_t file_bytes, std::list<server_prompt_cache_state>::iterator keep) {
    if (!disk->budget_allows_bytes(file_bytes)) {
        return false;
    }
    while (!disk->fits_bytes(file_bytes)) {
        auto old = states.end();
        for (auto s = states.begin(); s != states.end(); ++s) {
            if (s != keep && s->spilled()) {
                old = s;
                break;
            }
        }
        if (old == states.end()) {
            return false;
        }
        SRV_TRC(" - disk tier full: removing oldest spilled entry (%.3f MiB on disk)\n", old->disk_n_bytes / (1024.0 * 1024.0));
        ctr.n_disk_evicted++;
        drop_state(old);
    }
    return true;
}

void server_prompt_cache::note_prefill(size_t n_tokens, double seconds) {
    // small batches are dominated by fixed costs and say nothing about the steady prefill speed
    if (n_tokens < 32 || seconds <= 0) {
        return;
    }
    const double r = (double) n_tokens / seconds;
    prefill_tps_ = prefill_tps_ > 0 ? 0.7 * prefill_tps_ + 0.3 * r : r;
}

void server_prompt_cache::note_restore(size_t bytes, double seconds) {
    if (bytes == 0 || seconds <= 0) {
        return;
    }
    const double r = (double) bytes / seconds;
    restore_Bps_ = restore_Bps_ > 0 ? 0.7 * restore_Bps_ + 0.3 * r : r;
}

double server_prompt_cache::restore_Bps() const {
    if (restore_Bps_ > 0) {
        return restore_Bps_;
    }
    if (disk) {
        const auto & s = disk->stats();
        if (s.write_seconds > 0 && s.bytes_written > 0) {
            return (double) s.bytes_written / s.write_seconds;
        }
    }
    return 0;
}

bool server_prompt_cache::restore_pays(size_t tokens_saved, size_t bytes) const {
    if (min_speedup <= 0) {
        return true;
    }
    const double r = restore_Bps();
    const double p = prefill_tps_;
    if (r <= 0 || p <= 0) {
        return true; // a rate is unknown: no opinion
    }
    return ((double) tokens_saved / p) >= min_speedup * ((double) bytes / r);
}

bool server_prompt_cache::spill_slot_checkpoint(common_prompt_checkpoint & ck) {
    if (!disk || ck.empty()) {
        return false;
    }
    // older cached entries make room: they are a second copy of something recomputable, a live slot's checkpoint is not
    make_disk_room(disk->file_size_for(ck.tier_payload_a(), ck.tier_payload_b()), states.end());
    if (ck.spill_tier(disk)) {
        ctr.n_ckpt_slot_spilled++;
        return true;
    }
    ctr.n_ckpt_slot_refused++;
    return false;
}

// Unique-bytes accounting (BeeLlama-style checkpoint sharing; row S69-DRIFT-1110-SERVER-CKPT-REWIND-REFCOUNT-2026-11-10-01).
// Entries share checkpoint OBJECTS (server_prompt_checkpoint_ptr), e.g. a slot's prompt cached twice along a diverging
// continuation holds the same early checkpoints in both entries. Summing ckpt->size() per entry counted a shared object once per
// entry, so --cache-ram evicted or refused earlier than the real footprint required. Each object is counted once.
size_t server_prompt_cache::size_ckpt() const {
    size_t res = 0;
    std::unordered_set<const common_prompt_checkpoint *> seen;
    for (const auto & state : states) {
        for (const auto & ckpt : state.prompt.checkpoints) {
            if (seen.insert(ckpt.get()).second) {
                res += ckpt->size();
            }
        }
    }
    return res;
}

bool server_prompt_cache::read_spilled(std::list<server_prompt_cache_state>::iterator it, server_prompt_data & out) {
    if (!it->spilled() || !disk) {
        return false;
    }
    if (!disk->get(it->disk_id, it->prompt.n_tokens(), out.main, out.drft) || out.main.size() + out.drft.size() != it->disk_n_bytes) {
        SRV_WRN(" - cached prompt of %d tokens could not be read back from disk, dropping it\n", it->prompt.n_tokens());
        ctr.n_restore_failed++;
        out.main.clear();
        out.drft.clear();
        drop_state(it); // `it` is invalid from here
        return false;
    }
    ctr.n_restored++;
    return true;
}

size_t server_prompt_cache::size_disk() const {
    return disk ? disk->used_bytes() : 0;
}

size_t server_prompt_cache::n_spilled() const {
    size_t res = 0;
    for (const auto & state : states) {
        res += state.spilled() ? 1 : 0;
    }
    return res;
}

server_prompt_cache_disk_stats server_prompt_cache::disk_stats() const {
    server_prompt_cache_disk_stats r;
    if (!disk) {
        return r;
    }
    const auto & s = disk->stats();
    r.enabled           = true;
    r.n_spilled         = ctr.n_spilled;
    r.n_restored        = ctr.n_restored;
    r.n_restore_failed  = ctr.n_restore_failed;
    r.n_dropped_small   = ctr.n_dropped_small;
    r.n_dropped_refused = ctr.n_dropped_refused;
    r.n_disk_evicted    = ctr.n_disk_evicted;
    r.bytes_written     = s.bytes_written;
    r.bytes_read        = s.bytes_read;
    r.n_refused_cap     = s.n_refused_cap;
    r.n_refused_budget  = s.n_refused_budget;
    r.n_refused_space   = s.n_refused_space;
    r.n_io_failed       = s.n_put_failed + s.n_get_failed;
    r.n_checksum_failed = s.n_checksum_failed;
    r.used_bytes        = s.used_bytes;
    r.limit_bytes       = disk->limit_bytes();
    r.n_entries         = n_spilled();
    r.n_ckpt_spilled         = ctr.n_ckpt_spilled;
    r.n_ckpt_refused         = ctr.n_ckpt_refused;
    r.n_ckpt_restored        = ctr.n_ckpt_restored;
    r.n_ckpt_restore_failed  = ctr.n_ckpt_restore_failed;
    r.n_ckpt_adopted         = ctr.n_ckpt_adopted;
    r.n_ckpt_slot_spilled    = ctr.n_ckpt_slot_spilled;
    r.n_ckpt_slot_refused    = ctr.n_ckpt_slot_refused;
    r.n_dropped_slow         = ctr.n_dropped_slow;
    r.n_ckpt_dropped_slow    = ctr.n_ckpt_dropped_slow;
    r.n_restore_skipped      = ctr.n_restore_skipped;
    r.prefill_tps            = prefill_tps_;
    r.restore_Bps            = restore_Bps();
    r.min_speedup            = min_speedup;
    r.bytes_written_ckpt     = s.bytes_written_ckpt;
    r.bytes_read_ckpt        = s.bytes_read_ckpt;
    r.ram_bytes              = size();
    r.ram_ckpt_bytes         = size_ckpt();
    return r;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

size_t server_prompt_cache::n_tokens_resident() const {
    size_t res = 0;

    for (const auto & state : states) {
        if (!state.spilled()) {
            res += state.prompt.n_tokens();
        }
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    // (only an entry computed under the SAME LoRA configuration contains it: the same tokens under another adapter
    // configuration are a different KV and must be stored as their own entry)
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->prompt.lora_fp != prompt.lora_fp) {
            continue;
        }

        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    // a checkpoint object some entry already holds is shared, not copied: it adds no bytes (unique-bytes accounting)
    size_t checkpoints_size = 0;
    {
        std::unordered_set<const common_prompt_checkpoint *> held;
        for (const auto & state : states) {
            for (const auto & c : state.prompt.checkpoints) {
                held.insert(c.get());
            }
        }
        for (const auto & ckpt : prompt.checkpoints) {
            if (held.insert(ckpt.get()).second) {
                checkpoints_size += ckpt->size();
            }
        }
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    // (again only within the same LoRA configuration)
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.lora_fp == prompt.lora_fp ? it->prompt.tokens.get_common_prefix(prompt.tokens) : -1;

        if (len >= 0 && len == (int) it->prompt.tokens.size() && !it->pinned) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = drop_state(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            if (!evict_one()) {
                break;
            }
        }
        if (size() + state_size_new > limit_size) {
            SRV_WRN(" - pinned prefixes hold %.3f MiB of the %.3f MiB limit, no room for this %.3f MiB state, skipping\n",
                    pinned_size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), state_size_new / (1024.0 * 1024.0));
            return nullptr;
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
            /*.lora_fp     =*/ prompt.lora_fp,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    // With the NVMe tier on, the entry must own its checkpoints' bytes. A checkpoint the slot spilled to a --ctx-ckpt-spill-dir file
    // was copied above together with its spill_path, i.e. two owners of one file: the slot deleting it (a dropped or replaced
    // checkpoint, prompt.clear()) would leave this entry pointing at nothing, and this entry's copy deleting it would take the
    // slot's data. Read the file into the entry's copy and let go of the path; a file that cannot be read costs the copy, not the
    // entry. (Tier off: the copy is kept as it always was.)
    if (disk) {
        auto & cks = states.back().prompt.checkpoints; // (shared with the slot's list: see server_prompt_checkpoint_ptr)
        for (auto c = cks.begin(); c != cks.end();) {
            if ((*c)->data_tgt.empty() && !(*c)->spill_path.empty() && !(*c)->disk) {
                // the object is SHARED with the slot, and adopting rewrites it (reads the file, lets go of the path): the slot's
                // spill file would be orphaned. So the entry takes a private copy (what the deep copy used to be, made only
                // for this rare case) and adopts on that; the slot keeps the original and its file.
                auto priv = std::make_shared<common_prompt_checkpoint>(**c);
                if (priv->adopt_spill_file()) {
                    *c = std::move(priv);
                    ctr.n_ckpt_adopted++;
                    ++c;
                } else {
                    SRV_WRN("%s", " - a context checkpoint spilled by the slot could not be read, not keeping it in the cache entry\n");
                    c = cks.erase(c);
                }
            } else {
                ++c;
            }
        }
    }

    return &states.back();
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::find_best(const server_prompt & prompt, const server_tokens & tokens_new, const llama_context * ctx_tgt, const thinking_tokens & think_tokens, uint64_t lora_fp) {
    // --reasoning-tokens (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-
    // 27-01, item 2; SOURCE-READ: examples/server/server-task.cpp:1153-1200 @ 35f0cac75): compare
    // cached and incoming prompts with reasoning spans stripped, same reasoning as
    // get_available_slot's LCP similarity in server-context.cpp.
    const server_tokens tokens_new_ex = tokens_new.get_tokens_exclude_think(ctx_tgt, think_tokens);

    // A slot prompt held under another LoRA configuration is not a baseline to beat: launch_slot_with_task drops it
    // (or, for aLoRA -> aLoRA, keeps only its pre-invocation part), so any entry under the requested configuration
    // that shares a prefix is an improvement.
    const bool own_usable = prompt.lora_fp == lora_fp;

    const int lcp_best = own_usable ? prompt.tokens.get_tokens_exclude_think(ctx_tgt, think_tokens).get_common_prefix(tokens_new_ex) : 0;

    float f_keep_best = (own_usable && prompt.tokens.size() > 0) ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new_ex.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        // KV computed under another adapter configuration is not this request's KV, whatever the tokens say
        if (it->prompt.lora_fp != lora_fp) {
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_tokens_exclude_think(ctx_tgt, think_tokens).get_common_prefix(tokens_new_ex);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new_ex.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    return it_best;
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, const thinking_tokens & think_tokens, uint64_t lora_fp, std::vector<uint8_t> * spec_out, bool * hit) {
    // NVMe tier: when the best match was spilled, read it back first. A blob that cannot be read (missing, truncated, failed
    // checksum) is dropped by read_spilled() and the search is repeated, so the request falls back to the next best entry or
    // to a plain prefill -- never to bytes that were not verified. Every failure removes one entry, so this ends.
    server_prompt_data from_disk;
    auto it_best = states.end();
    int64_t t_restore0 = ggml_time_us();
    for (;;) {
        it_best = find_best(prompt, tokens_new, ctx_tgt, think_tokens, lora_fp);
        if (it_best == states.end() || !it_best->spilled()) {
            break;
        }
        // restore-versus-recompute: reading this entry back must beat recomputing the prefix it shares with the request
        if (!restore_pays((size_t) it_best->prompt.tokens.get_common_prefix(tokens_new), it_best->disk_n_bytes)) {
            SRV_TRC(" - spilled prompt of %d tokens matches, but recomputing the shared prefix is quicker than reading %.3f MiB\n",
                    it_best->prompt.n_tokens(), it_best->disk_n_bytes / (1024.0 * 1024.0));
            ctr.n_restore_skipped++;
            it_best = states.end();
            break;
        }
        t_restore0 = ggml_time_us();   // the restore clock starts at the read, not at the search
        if (read_spilled(it_best, from_disk)) {
            break;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt (%d tokens) for lora fingerprint %016llx%s\n", (int) it_best->prompt.tokens.size(), (unsigned long long) lora_fp,
                it_best->spilled() ? " (restored from disk)" : "");

        // NON-CONSUMING LOAD (S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-NCONSUMER-FOLLOWUP-2026-09-27-01).
        // Upstream moved the entry into the slot and erased it, so a saved prefix could serve ONE
        // consumer: with cross-slot publish-on-prefill, a third concurrent request sharing the same
        // preamble found nothing and re-ran the whole prefix. The entry is now kept and moved to the
        // BACK of the list -- `states` is evicted from the front (alloc/update pop_front), so this
        // makes eviction least-recently-USED rather than least-recently-saved, and the size/token
        // limits remain the only thing that ever removes an entry. Design source: nano-vLLM's block
        // manager keeps a freed block's hash until the block is reallocated
        // (upstream/nano-vllm/nanovllm/engine/block_manager.py:47-56, MIT; no code
        // copied). Matching here was already token-verified (get_common_prefix), so a kept entry can
        // never be served for a prompt it does not actually prefix.
        // Cost: the served state now exists both in the slot's KV and in host RAM, bounded by the
        // existing cache limit (--cache-ram), which is what that limit is for.
        {
            const auto & data = it_best->spilled() ? from_disk.main : it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }
        }

        {
            const auto & data = it_best->spilled() ? from_disk.drft : it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }
            }
        }

        if (it_best->spilled()) {
            // the measured cost of a restore: read + checksum + loading the state into the context, per byte
            note_restore(it_best->disk_n_bytes, (ggml_time_us() - t_restore0) / 1e6);
        }

        prompt = it_best->prompt.clone();

        if (hit) {
            *hit = true;
        }
        if (spec_out) {
            *spec_out = it_best->data.spec;   // resident even when main/drft were spilled
        }

        touch(it_best);
    }

    return true;
}

size_t server_prompt_cache::remove_prefixes_of(const server_tokens & tokens) {
    // Why ERASE must purge here (MEASURED 2026-09-28, temp/_slotab_0928_post.txt): publish-on-prefill copies every
    // slot's prompt into this cache at DONE_PROMPT (server-context.cpp, the cache_publish_on_prefill loop), and the
    // cache is non-consuming, so after `POST /slots/N?action=erase` the very prompt that was erased was served back to
    // the next request (test_slot_erase: prompt_n 1 instead of 21). Upstream's erase leaves nothing to restore because
    // upstream only saves a slot's prompt when the slot is reassigned. An entry published from a slot is a prefix of
    // that slot's tokens (the prompt, before any generated tokens), so "is a prefix of the erased tokens" is exactly
    // the set that could have come from it. An identical prefix another slot published is removed too -- for that
    // slot this is only a cache miss, never a correctness change.
    if (tokens.size() == 0) {
        return 0;
    }
    size_t n_removed = 0;
    for (auto it = states.begin(); it != states.end();) {
        const size_t len = it->prompt.tokens.size();
        if (len > 0 && (size_t) it->prompt.tokens.get_common_prefix(tokens) == len) {
            SRV_TRC(" - erase: removing cached prompt with length %zu\n", len);
            it = drop_state(it);
            ++n_removed;
        } else {
            ++it;
        }
    }
    return n_removed;
}

size_t server_prompt_cache::clear_all() {
    size_t n_removed = 0;
    for (auto it = states.begin(); it != states.end(); ++n_removed) {
        it = drop_state(it); // drop_state also erases a spilled entry's disk file
    }
    return n_removed;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            if (!evict_one()) {
                break;
            }
        }
    }

    // average size per token (of the entries held in memory: the token limit, like the byte limit, bounds memory, and a
    // spilled entry is on disk; without the NVMe tier these are all the entries)
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens_resident())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens_resident() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached\n", limit_tokens, limit_tokens_cur);

            if (!evict_one()) {
                break;
            }
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB%s\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0),
                state.spilled() ? " (+ state on disk)" : "");
    }
}
