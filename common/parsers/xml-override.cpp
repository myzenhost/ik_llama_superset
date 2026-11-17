#include "parsers.h"

#include <cctype>

// Operator-defined XML-style tool-call format (--tool-call-format-json / --tool-call-format-preset), see
// common_chat_xml_tool_format in chat.h for the grammar of the format and common/chat-xml-format.cpp for where the idea comes from
// (mudler/LocalAI, MIT). The parser and the grammar are built from the struct for every offered tool, for any template: the
// override replaces the specialised handler and the autoparser. Reasoning is handled like the Qwen3 handler does (an optional
// <think> block, ended by </think> or by the first tool marker) when the template mentions <think>.
// Fork parser, moved from common/chat.cpp into the upstream parsers layout in the 2026-11-04 merge and ported to the schema
// document API: parameters come from foreach_parameter (common_chat_schema_property + its document), a string-capable parameter
// takes the raw text (param.schema->may_be_string(), as upstream's qwen3-coder parser decides it), any other one a JSON value
// checked against the parameter's schema node; rule names are by tool / parameter index (upstream's convention).
common_chat_params common_chat_params_init_xml_override(const common_chat_template &          tmpl,
                                                        const autoparser::generation_params & inputs,
                                                        const common_chat_xml_tool_format &   fmt) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;

    const bool supports_reasoning = tmpl.source().find("<think>") != std::string::npos;
    data.supports_thinking        = supports_reasoning;

    // the first thing the model writes when it starts a tool call
    const std::string first_marker = fmt.scope_start.empty() ? fmt.tool_start : fmt.scope_start;

    data.preserved_tokens = { first_marker, fmt.tool_end };
    if (!fmt.scope_end.empty()) {
        data.preserved_tokens.push_back(fmt.scope_end);
    }
    if (supports_reasoning) {
        data.thinking_start_tag = "<think>";
        data.thinking_end_tags  = { "</think>", first_marker };
        data.preserved_tokens.insert(data.preserved_tokens.end(), { "<think>", "</think>" });
    }

    // the part of the generation prompt before a prefilled "<think>": the parser consumes the rest as the optional reasoning block
    std::string gen_prefix = data.generation_prompt;
    if (supports_reasoning) {
        const size_t t = gen_prefix.rfind("<think>");
        if (t != std::string::npos && gen_prefix.find_first_not_of(" \t\r\n", t + 7) == std::string::npos) {
            gen_prefix.resize(t);
        }
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = gen_prefix;
        if (supports_reasoning) {
            data.generation_prompt += "<think>\n" + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += "\n</think>\n\n";
            }
        }
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    // Text after the calls is content. Without the tail the parse simply stopped after the last call and everything the model
    // wrote after it was silently lost (measured: "...</minimax:tool_call>\nDone." came back without "Done."), and a
    // quoted call followed by prose could not be told from a real one. The grammar is built from the same parser WITHOUT
    // the tail (capture_tail false), so what a constrained model may generate is unchanged.
    bool capture_tail = true;
    auto build_parser = [&]() { return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(gen_prefix);

        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", first_marker })) +
                                   (p.literal("</think>") | p.peek(p.literal(first_marker))));
        }

        // Response format parser
        if (has_response_format) {
            return generation_prompt + (reasoning << p.content(p.schema(p.json(), "response-format", inputs.json_schema)));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // a string value is the raw text up to val_end (the ac() keeps the generated grammar from running past it)
            auto raw_string_value = p.rule("xml-override-string-value",
                p.ac(p.tool_arg_string_value(p.until(fmt.val_end)) + p.tool_arg_close(p.literal(fmt.val_end)), fmt.val_end));

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");

                std::vector<common_peg_parser> required_args;
                std::vector<common_peg_parser> optional_args;
                foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                    auto rule_name = "tool-" + std::to_string(tool_index) + "-arg-" + std::to_string(param_index);

                    auto key = p.literal(fmt.key_start) + p.tool_arg_name(p.literal(param.name)) + p.literal(fmt.key_val_sep);
                    if (fmt.key_val_sep2.has_value()) {
                        key = key + p.space() + p.literal(*fmt.key_val_sep2);
                    }
                    auto arg_open = p.tool_arg_open(key);

                    common_peg_parser arg_value = p.eps();
                    if (param.schema->may_be_string()) {
                        arg_value = fmt.raw_argval.has_value() && !*fmt.raw_argval ?
                            p.tool_arg_json_value(p.json_string()) + p.tool_arg_close(p.literal(fmt.val_end)) :
                            raw_string_value;
                    } else {
                        // Whitespace between a JSON value and val_end is skipped, but NOT when val_end itself starts with whitespace
                        // (qwen3-coder: "\n</parameter>"): the greedy space() would eat that first character and the marker
                        // could never match, while the generated grammar would still accept the text. Then the marker is exact.
                        auto close = p.tool_arg_close(p.literal(fmt.val_end));
                        const bool val_end_starts_with_ws = !fmt.val_end.empty() && std::isspace((unsigned char) fmt.val_end[0]) != 0;
                        arg_value  = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) +
                                     (val_end_starts_with_ws ? close : p.choice({ close, p.space() + close }));
                    }

                    (param.required ? required_args : optional_args).push_back(p.rule(rule_name, p.tool_arg(arg_open + arg_value) + p.space()));
                });

                // every required parameter exactly once, in any order (the model does not always keep the schema's order);
                // optional ones after them. Without this a grammar-constrained model may repeat one parameter forever.
                auto args = p.permute("tool-" + std::to_string(tool_index) + "-args", required_args);
                if (!optional_args.empty()) {
                    args = args + p.zero_or_more(p.choice(optional_args));
                }

                auto open = p.literal(fmt.tool_start) + p.tool_name(p.literal(name));
                if (!fmt.tool_sep.empty()) {
                    open = open + p.literal(fmt.tool_sep);
                }

                auto func = p.tool(p.tool_open(open) + p.space() + p.tool_args(args) + p.tool_close(p.literal(fmt.tool_end)));

                tool_choice |= p.rule("tool-" + std::to_string(tool_index), func);
            });

            const bool parallel = inputs.parallel_tool_calls;
            auto       calls    = parallel ? p.one_or_more(tool_choice + p.space()) : tool_choice + p.space();
            auto       block    = fmt.scope_start.empty() ?
                calls :
                p.literal(fmt.scope_start) + p.space() + calls + p.literal(fmt.scope_end) + p.space();

            auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat(block, min_calls, parallel ? -1 : 1));

            if (capture_tail) {
                return generation_prompt + (reasoning << p.content(p.until(first_marker)) << tool_calls << p.content(p.rest()));
            }
            return generation_prompt + (reasoning << p.content(p.until(first_marker)) << tool_calls);
        }

        // Content only parser
        return generation_prompt + (reasoning << p.content(p.rest()));
    }); };

    data.parser = build_parser();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            capture_tail = false;
            build_parser().build_grammar(builder, data.grammar_lazy);
        });

        if (data.grammar_lazy) {
            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, first_marker });
        }
    }

    return data;
}
