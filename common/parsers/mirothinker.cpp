#include "parsers.h"

// MiroThinker - uses MCP style toolcalling:
//   <think>{reasoning}</think> content
//   <use_mcp_tool>
//   <server_name>{server_name}</server_name>
//   <tool_name>{tool_name}</tool_name>
//   <arguments>
//   {json_args}
//   </arguments>
//   (more calls in the same form)
//   </use_mcp_tool>
// Fork parser, moved from common/chat.cpp into the upstream parsers layout in the 2026-11-04 merge (arena parser, indexed
// foreach_function, no resolve_refs pass, common_chat_template_direct_apply_impl).
common_chat_params common_chat_params_init_mirothinker(const common_chat_template &          tmpl,
                                                       const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tags  = { "</think>" };
    data.preserved_tokens   = {
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto reasoning = extract_reasoning ? p.optional("<think>" + p.reasoning(p.until("</think>")) + "</think>") : p.eps();

        // Tool call markers
        const std::string SECTION_BEGIN = "<use_mcp_tool>";
        const std::string SECTION_END   = "</use_mcp_tool>";
        const std::string CALL_BEGIN    = "<server_name>";
        const std::string ARGS_BEGIN    = "<arguments>";
        const std::string CALL_END      = "</arguments>";

        auto end = p.end();

        // Content only parser (no tools)
        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return reasoning + p.content(p.rest()) + end;
        }

        // Build tool call parsers for each available function
        // Function name format is: <tool_name>{tool_name}</tool_name>
        // We need to match: {what_ever}</server_name>{spaces}<tool_name>{tool_name}</tool_name>
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](size_t, const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            auto tool_parser = p.tool(
                p.tool_open(
                    p.until("</server_name>") +
                    p.literal("</server_name>") +
                    p.space() +
                    p.literal("<tool_name>") +
                    p.tool_name(p.literal(name)) +
                    p.literal(ARGS_BEGIN)
                ) + p.space() +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)) +
                p.space() + p.tool_close(p.literal(CALL_END))
            );

            tool_choice |= p.rule("tool-" + name, tool_parser);
        });

        // Tool calls section: <use_mcp_tool> tool_calls </use_mcp_tool>
        auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
        auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
        auto tool_calls = p.trigger_rule("tool-calls",
            p.literal(SECTION_BEGIN) + p.space() +
            p.rule("tool-call", p.repeat(CALL_BEGIN + tool_choice, min_calls, max_calls) +
                p.space() + p.literal(SECTION_END))
        );

        auto content_before_tools = p.content(p.until(SECTION_BEGIN));

        return reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser;

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<use_mcp_tool>" }
        };
    }

    return data;
}
