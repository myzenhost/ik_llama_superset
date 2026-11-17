#include "parsers.h"

// OLMo-3 (allenai/Olmo-3*-Instruct) format:
// - no reasoning block
// - tool calls: one block <function_calls>name(k=<JSON>, k2=<JSON>)\nname2(...)</function_calls>, one call per line;
//   the template renders `key ~ '=' ~ (value | tojson)` joined by ", ", so values are JSON literals (the parser
//   also takes Python-cased True/False/None and single-quoted strings, as a model may drift into them)
// - tool results come back as role "environment"
// SOURCE-READ: allenai/Olmo-3.1-32B-Instruct chat_template.jinja (models/templates/allenai-Olmo-3.1-32B-Instruct.jinja) for the
// wire format; tabbyAPI endpoints/OAI/utils/toolcall_formats/olmo3.py:1-121 (AGPL-3.0) for the tolerances, behaviour only, no code copied.
// Fork parser, moved from common/chat.cpp into the upstream parsers layout in the 2026-11-04 merge (arena parser, the prompt is a
// common_chat_prompt -- append() instead of +=, no resolve_refs pass).
common_chat_params common_chat_params_init_olmo3(const common_chat_template &          tmpl,
                                                 const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string CALLS_BEGIN = "<function_calls>";
    const std::string CALLS_END   = "</function_calls>";
    const std::string GEN_PREFIX  = "<|im_start|>assistant\n";

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = { CALLS_BEGIN, CALLS_END };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|im_start|>assistant"   },
        { COMMON_CHAT_ROLE_TOOL,      "<|im_start|>environment" },
        { COMMON_CHAT_ROLE_USER,      "<|im_start|>user"        },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|im_start|>system"      },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PREFIX;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(data.generation_prompt);

        // Response format parser
        if (has_response_format) {
            return generation_prompt + p.content(p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto calls = p.python_style_tool_call_lines(inputs.tools, inputs.parallel_tool_calls, /* allow_json_literals = */ true);

            // The closing tag is optional: a stream cut off by the token limit or by EOS right after the last call still
            // yields the complete calls inside it.
            auto block = p.literal(CALLS_BEGIN) + p.space() + calls + p.space() + p.optional(p.literal(CALLS_END));

            auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat(block, min_calls, 1));

            return generation_prompt + p.content(p.until(CALLS_BEGIN)) + tool_calls;
        }

        // Content only parser
        return generation_prompt + p.content(p.rest());
    });

    data.parser = parser;

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        if (data.grammar_lazy) {
            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, CALLS_BEGIN });
        }
    }

    return data;
}
