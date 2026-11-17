#include "parsers.h"

// FunctionGemma (google/functiongemma-270m-it): <start_function_call>call:NAME{key:value,...}<end_function_call>, strings wrapped in
// <escape>...<escape>. SOURCE-READ ollama model/parsers/functiongemma.go:1-80 and functiongemma_test.go:14-140 (MIT; the format, no code).
// Fork parser, moved from common/chat.cpp into the upstream parsers layout in the 2026-11-04 merge: the parser is now stored as
// the arena itself (common_chat_params::parser), foreach_function passes the tool index, and the grammar builder resolves
// schema refs on its own (no resolve_refs pass).
common_chat_params common_chat_params_init_functiongemma(const common_chat_template &          tmpl,
                                                         const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_FUNCTIONGEMMA;
    data.preserved_tokens  = { "<start_function_call>", "<end_function_call>" };

    auto has_tools       = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto include_grammar = has_tools;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.optional(p.literal("<start_of_turn>model\n")));
        if (!has_tools) {
            return start + p.content(p.rest());
        }
        add_gemma_value_rules(p, "<escape>");
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](size_t, const json & tool) {
            std::string name = tool.at("function").at("name");
            tool_choice |= p.rule("tool-" + name, p.tool(p.sequence({
                p.tool_open(p.tool_name(p.literal(name)) + p.peek(p.literal("{"))),
                p.tool_args(p.ref("gemma4-dict")),
            })));
        });
        auto tool_call = p.trigger_rule("tool-call", p.repeat(
            "<start_function_call>call:" + tool_choice + "<end_function_call>",
            /* min = */ inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0,
            /* max = */ inputs.parallel_tool_calls ? -1 : 1
        ));
        auto content = p.rule("content", p.content(p.until("<start_function_call>")));
        return start + content + tool_call + p.optional(p.space());
    });
    data.parser = parser;

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });
        data.grammar_triggers = { { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<start_function_call>" } };
    }
    return data;
}
