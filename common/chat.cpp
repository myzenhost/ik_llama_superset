#include "chat.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "log.h"
#include "parsers/parsers.h"
#include "sampling.h"

#include "jinja/value.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "peg-parser.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <map>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = common_json;

static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto               time       = std::chrono::system_clock::to_time_t(now);
    auto               local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static json safe_args_parse(const std::string & to_parse) {
    std::string stripped = to_parse;
    if (to_parse.at(0) == '"' && to_parse.at(to_parse.length() - 1) == '"') {
        stripped = to_parse.substr(1, to_parse.length() - 1);
    }
    try {
        return json::parse(stripped);
    } catch (const common_json_error & e) {
        return stripped;
    }
}

static std::string string_diff(const std::string & last, const std::string & current) {
    if (last.empty()) {
        return current;
    }
    if (!string_starts_with(current, last)) {
        if (string_starts_with(last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    return current.substr(last.size());
}

static bool has_content_or_tool_calls(const common_chat_msg & msg) {
    return !msg.content.empty() || !msg.tool_calls.empty();
}

std::string common_chat_msg::render_content(const std::string & delimiter) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    if (!content.empty()) {
        return content;
    }

    std::string text;
    for (const auto & part : content_parts) {
        if (part.type == "text") {
            if (!text.empty()) {
                text += delimiter;
            }
            text += part.text;
        }
    }
    return text;
}

common_chat_role common_chat_role_from_string(const std::string & role) {
    if (role == "system")    { return COMMON_CHAT_ROLE_SYSTEM;    }
    if (role == "assistant") { return COMMON_CHAT_ROLE_ASSISTANT; }
    if (role == "user")      { return COMMON_CHAT_ROLE_USER;      }
    if (role == "tool")      { return COMMON_CHAT_ROLE_TOOL;      }
    return COMMON_CHAT_ROLE_UNKNOWN;
}

const char * common_chat_role_to_string(common_chat_role role) {
    switch (role) {
        case COMMON_CHAT_ROLE_SYSTEM:    return "system";
        case COMMON_CHAT_ROLE_ASSISTANT: return "assistant";
        case COMMON_CHAT_ROLE_USER:      return "user";
        case COMMON_CHAT_ROLE_TOOL:      return "tool";
        case COMMON_CHAT_ROLE_UNKNOWN:   return "";
    }
    return "";
}

common_chat_msg_delimiters common_chat_msg_delimiters_parse(const json & delimiters) {
    common_chat_msg_delimiters result;

    if (!delimiters.is_array()) {
        return result;
    }

    result.delimiters.reserve(delimiters.size());
    for (const auto & d : delimiters) {
        if (!d.is_object()) {
            continue;
        }
        result.delimiters.push_back({
            common_chat_role_from_string(d.value("role", std::string())),
            d.value("delimiter", std::string()),
        });
    }

    return result;
}

void common_chat_msg_delimiters::tokenize(const llama_vocab * vocab) {
    for (auto & d : delimiters) {
        d.tokens = common_tokenize(vocab, d.delimiter, false, true);
    }
}

common_chat_msg_spans common_chat_msg_delimiters::split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips) const {
    std::vector<std::pair<common_chat_role, size_t>> matches;

    auto skip = skips.begin();
    for (size_t i = 0; i < tokens.size();) {
        if (skip != skips.end() && i == skip->first) {
            i += skip->second;
            ++skip;
            continue;
        }
        for (const auto & d : delimiters) {
            if (i + d.tokens.size() > tokens.size()) {
                continue;
            }
            if (std::equal(d.tokens.begin(), d.tokens.end(), tokens.begin() + i)) {
                matches.emplace_back(d.role, i);
                break;
            }
        }
        i++;
    }

    matches.emplace_back(COMMON_CHAT_ROLE_UNKNOWN, tokens.size());

    common_chat_msg_spans spans;
    for (size_t i = 0; i + 1 < matches.size(); i++) {
        const auto & curr = matches[i];
        const auto & next = matches[i + 1];
        spans.add(curr.first, curr.second, next.second - curr.second);
    }

    return spans;
}

json common_chat_msg::to_json_oaicompat(bool concat_typed_text) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    json jmsg {
        {"role", role},
    };
    if (!content.empty()) {
        jmsg["content"] = content;
    } else if (!content_parts.empty()) {
        if (concat_typed_text || contains_media()) {
            std::string text;
            bool last_was_media_marker = false;
            // join parts with newline, do not add newline before or after media markers
            for (const auto & part : content_parts) {
                bool add_new_line = true;
                if (part.type == "text") {
                    add_new_line = !last_was_media_marker && !text.empty();
                    last_was_media_marker = false;
                } else if (part.type == "media_marker") {
                    add_new_line = false;
                    last_was_media_marker = true;
                } else {
                    LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                    continue;
                }

                if (add_new_line) {
                    text += '\n';
                }

                text += part.text;
            }
            jmsg["content"] = text;
        } else {
            auto & parts = jmsg["content"] = json::array();
            for (const auto & part : content_parts) {
                parts.push_back({
                    {"type", part.type},
                    {"text", part.text},
                });
            }
        }
    } else {
        jmsg["content"] = "";
    }
    if (!reasoning_content.empty()) {
        jmsg["reasoning_content"] = reasoning_content;
    }
    if (!tool_name.empty()) {
        jmsg["name"] = tool_name;
    }
    if (!tool_call_id.empty()) {
        jmsg["tool_call_id"] = tool_call_id;
    }
    if (!tool_calls.empty()) {
        jmsg["tool_calls"] = json::array();
        auto & jtool_calls = jmsg["tool_calls"];
        for (const auto & tool_call : tool_calls) {
            json tc {
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", json(tool_call.arguments)},
                }},
            };
            if (!tool_call.id.empty()) {
                tc["id"] = tool_call.id;
            }
            // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
            // We only generate a random id for the ones that don't generate one by themselves
            // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
            // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            jtool_calls.push_back(tc);
        }
    }

    return jmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & msg_prv,
                                                                      const common_chat_msg & msg_new) {
    std::vector<common_chat_msg_diff> diffs;
    if (msg_new.tool_calls.size() > msg_prv.tool_calls.size()) {
        diffs.reserve(msg_new.tool_calls.size() - msg_prv.tool_calls.size() + 3);
    } else {
        diffs.reserve(3);
    }

    // Migrated prefix: text streamed as content that a later parse state classifies as reasoning (the
    // closing thinking tag arrives after the text). The client already has those characters, so only the
    // part of the reasoning beyond that prefix is new, and the emptied content is not a change to report.
    // Covers tests/test-chat-auto-parser.cpp "migrated prefix emits no duplicate diff".
    const bool migrated_prefix = msg_prv.reasoning_content.empty() && !msg_prv.content.empty() &&
                                 msg_new.content.empty() &&
                                 string_starts_with(msg_new.reasoning_content, msg_prv.content);
    if (migrated_prefix) {
        const std::string tail = msg_new.reasoning_content.substr(msg_prv.content.size());
        if (!tail.empty()) {
            auto & diff                  = diffs.emplace_back();
            diff.reasoning_content_delta = tail;
        }
    } else {
        // Note: these string comparisons cost more as the message grows.
        if (msg_prv.reasoning_content != msg_new.reasoning_content) {
            auto & diff                  = diffs.emplace_back();
            diff.reasoning_content_delta = string_diff(msg_prv.reasoning_content, msg_new.reasoning_content);
        }
        if (msg_prv.content != msg_new.content) {
            auto & diff        = diffs.emplace_back();
            diff.content_delta = string_diff(msg_prv.content, msg_new.content);
        }
    }

    if (msg_new.tool_calls.size() < msg_prv.tool_calls.size()) {
        std::string err = "Invalid diff: now finding less tool calls!\n";
        err += "  Previous (" + std::to_string(msg_prv.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_prv.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current (" + std::to_string(msg_new.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_new.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current msg text content:\n" + msg_new.content + "\n";
        throw std::runtime_error(err);
    }

    if (!msg_prv.tool_calls.empty()) {
        const auto   idx  = msg_prv.tool_calls.size() - 1;
        const auto & pref = msg_prv.tool_calls[idx];
        const auto & newf = msg_new.tool_calls[idx];
        // Allow tool name to change during incremental parsing:
        // - empty -> non-empty (initial discovery)
        // - prefix -> longer string (name grows as more input is parsed)
        if (pref.name != newf.name && !pref.name.empty() && !newf.name.empty()) {
            // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
            bool is_prefix = (newf.name.rfind(pref.name, 0) == 0);
            if (!is_prefix) {
                LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", pref.name.c_str(), newf.name.c_str());
                throw std::runtime_error("Invalid diff: tool call mismatch!");
            }
        }
        const auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id || pref.name != newf.name) {
            auto & diff          = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id || pref.name != newf.name) {
                diff.tool_call_delta.id   = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = msg_prv.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        auto & diff          = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
    }

    return diffs;
}

using chat_template_caps = jinja::caps;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template;  // Model had builtin template or template overridden was specified.
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice) {
    if (tool_choice == "auto") {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    if (tool_choice == "none") {
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    if (tool_choice == "required") {
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    }
    throw std::invalid_argument("Invalid tool_choice: " + tool_choice);
}

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates) {
    common_chat_templates_inputs inputs;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "test";
    inputs.messages = { msg };
    inputs.enable_thinking = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    auto params = common_chat_templates_apply(chat_templates, inputs);
    return params.supports_thinking;
}

std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {
        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content    = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text" && type != "media_marker") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " +
                                                content.dump() +
                                                " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name           = fc.at("name");
                    const auto & args = fc.at("arguments");
                    if (args.is_string()) {
                        tc.arguments = args;
                    } else {
                        tc.arguments = args.dump();
                    }
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument(
                    "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & "
                    "https://github.com/ggml-org/llama.cpp/issues/12279)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id");
            }
            {
                json extra = json::object();
                for (const auto & [k, v] : message.items()) {
                    if (k != "role" && k != "content" && k != "tool_calls" && k != "reasoning_content" &&
                        k != "name" && k != "tool_call_id") {
                        extra[k] = v;
                    }
                }
                if (has_tool_calls) {
                    // per tool call: keys besides type/function/id, and function keys besides name/arguments
                    json tc_extra = json::array();
                    bool any      = false;
                    for (const auto & tool_call : message.at("tool_calls")) {
                        json e = json::object();
                        for (const auto & [k, v] : tool_call.items()) {
                            if (k != "type" && k != "function" && k != "id") {
                                e[k] = v;
                            }
                        }
                        for (const auto & [k, v] : tool_call.at("function").items()) {
                            if (k != "name" && k != "arguments") {
                                e["function"][k] = v;
                            }
                        }
                        any = any || !e.empty();
                        tc_extra.push_back(e);
                    }
                    if (any) {
                        extra["__tool_call_extras"] = tc_extra;
                    }
                }
                if (!extra.empty()) {
                    msg.extra_json = extra.dump();
                }
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        // @ngxson : disable otherwise it's bloating the API response
        // printf("%s\n", std::string("; messages = ") + messages.dump(2));
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }

    return msgs;
}

struct messages_inp_normalizer {
    const jinja::caps & caps;

    messages_inp_normalizer(const jinja::caps & c) : caps(c) {}

    // handle supports_string_content / supports_typed_content
    // if string=true and array=false, convert array to string
    // if string=false and array=true, convert string to array
    // if both are true, do nothing
    json normalize(const json & messages) {
        bool only_string = caps.supports_string_content && !caps.supports_typed_content;
        bool only_typed  = !caps.supports_string_content && caps.supports_typed_content;
        if ((!only_string && !only_typed) || !messages.is_array()) {
            return messages;
        }
        json normalized = json::array();
        for (const auto & msg : messages) {
            json copy = msg;
            if (copy.contains("content")) {
                json & it = copy.at("content");
                if (only_typed && it.is_string()) {
                    it = json::array({
                        json{
                            {"type", "text"},
                            {"text", it.get<std::string>()},
                        }
                    });
                } else if (only_string && it.is_array()) {
                    it = concat_content_parts(it);
                }
            }
            normalized.push_back(std::move(copy));
        }
        return normalized;
    }

    // join parts with newline, do not add newline before or after media markers
    static std::string concat_content_parts(const json & parts) {
        std::string text;
        bool last_was_media_marker = false;
        for (const auto & part : parts) {
            std::string type = part.value("type", "");
            bool add_new_line = true;
            if (type == "text") {
                add_new_line = !last_was_media_marker && !text.empty();
                last_was_media_marker = false;
            } else if (type == "media_marker") {
                add_new_line = false;
                last_was_media_marker = true;
            } else {
                LOG_WRN("Ignoring content part type: %s\n", type.c_str());
                continue;
            }

            if (add_new_line) {
                text += '\n';
            }

            text += part.value("text", "");
        }
        return text;
    }
};

static json render_message_to_json(const std::vector<common_chat_msg> & msgs, const jinja::caps & c) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\n", __func__);
    }

    json messages = json::array();
    for (const auto & msg : msgs) {
        messages.push_back(msg.to_json_oaicompat(/* concat_typed_text= */ false));
    }
    return messages_inp_normalizer(c).normalize(messages);
}

// DeepSeek V4.1 reads non-OpenAI message keys and joins content blocks with "\n\n" around images,
// so keep media markers as separate text parts and restore the extra keys
static json render_message_to_json_deepseek_v4_1(const std::vector<common_chat_msg> & msgs, const jinja::caps & c) {
    json messages = json::array();
    for (const auto & msg : msgs) {
        json jmsg;
        if (msg.contains_media() && msg.content.empty()) {
            common_chat_msg typed = msg;
            for (auto & part : typed.content_parts) {
                part.type = "text";
            }
            jmsg = typed.to_json_oaicompat(/* concat_typed_text= */ false);
        } else {
            jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ false);
        }
        if (!msg.extra_json.empty()) {
            const json extra = json::parse(msg.extra_json);
            for (const auto & [k, v] : extra.items()) {
                if (k == "__tool_call_extras") {
                    if (!jmsg.contains("tool_calls")) {
                        continue;
                    }
                    auto & tcs = jmsg.at("tool_calls");
                    for (size_t i = 0; i < tcs.size() && i < v.size(); i++) {
                        for (const auto & [tk, tv] : v[i].items()) {
                            if (tk == "function") {
                                for (const auto & [fk, fv] : tv.items()) {
                                    tcs[i]["function"][fk] = fv;
                                }
                            } else {
                                tcs[i][tk] = tv;
                            }
                        }
                    }
                } else if (!jmsg.contains(k)) {
                    jmsg[k] = v;
                }
            }
        }
        messages.push_back(std::move(jmsg));
    }
    return messages_inp_normalizer(c).normalize(messages);
}

// DEPRECATED: only used in tests
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    jinja::caps c;
    c.supports_string_content = true;
    c.supports_typed_content = !concat_typed_text;
    return render_message_to_json(msgs, c);
}

json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function", {
                { "name", tool.name },
                { "description", tool.description },
                { "parameters", json::parse(tool.parameters) },
            }},
        });
    }
    return result;
}

json common_chat_tool_parameters(const json & function) {
    if (function.contains("parameters")) {
        const auto & params = function.at("parameters");
        if (!params.is_null() && !(params.is_object() && params.empty())) {
            return params;
        }
    }
    return json{{"type", "object"}, {"properties", json::object()}};
}

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.value("description", ""),
                    /* .parameters = */ function.value("parameters", json::object()).dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

common_chat_continuation common_chat_continuation_parse(const common_json & value) {
    if (value.is_null()) {
        return COMMON_CHAT_CONTINUATION_NONE;
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? COMMON_CHAT_CONTINUATION_AUTO : COMMON_CHAT_CONTINUATION_NONE;
    }
    if (value.is_string()) {
        auto value_str = value.get<std::string>();
        if (value_str == "reasoning_content") {
            return COMMON_CHAT_CONTINUATION_REASONING;
        }
        if (value_str == "content") {
            return COMMON_CHAT_CONTINUATION_CONTENT;
        }
    }
    throw std::invalid_argument("Invalid continue_final_message: expected a boolean, \"content\" or \"reasoning_content\"");
}

bool common_chat_verify_template(const std::string & tmpl, bool use_jinja) {
    if (use_jinja) {
        try {
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = "test";

            auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);

            common_chat_templates_inputs inputs;
            inputs.messages = { msg };

            common_chat_templates_apply(tmpls.get(), inputs);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to apply template: %s\n", __func__, e.what());
            return false;
        }
    }
    llama_chat_message chat[] = {
        { "user", "test" }
    };
    const int res = llama_chat_apply_template(tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja,
                                      const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos   = tmpls->add_bos;
    inputs.add_eos   = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages              = past_msg;
        auto & extra = inputs.messages.emplace_back();
        extra.role = new_msg.role;
        inputs.add_generation_prompt = false;
        fmt_past_msg                 = common_chat_templates_apply(tmpls, inputs).prompt.str();
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg             = common_chat_templates_apply(tmpls, inputs).prompt.str();
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja            = use_jinja;
    inputs.add_bos              = tmpls->add_bos;
    inputs.add_eos              = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg         = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system", "You are a helpful assistant");
    add_simple_msg("user", "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user", "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt.str();
}

#define CHATML_TEMPLATE_SRC                                                               \
    "{%- for message in messages -%}\n"                                                   \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n"                                                                    \
    "{%- if add_generation_prompt -%}\n"                                                  \
    "  {{- '<|im_start|>assistant\n' -}}\n"                                               \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

// The default user prompt is an explicit verbatim instruction, not upstream's "Transcribe audio to text". MEASURED
// 2026-09-02 (fork f4a868e3d) and again 2026-10-17 on Voxtral-Mini-3B-2507 Q4_K_M + mmproj, CPU: with the short prompt
// the instruction-tuned model answers conversationally, HTTP 200 carrying "I'm unable to transcribe audio directly...";
// with the instruction below it returns the transcript. f4a868e3d fixed this in examples/server/, which the build no
// longer compiles once the server moved to tools/server/ and to this common preset, so the refusal came back.
// The caller can still override it with the OpenAI `prompt` field.
//
// The system string comes from fork commit ec6520161 (lost with examples/server). The ASR route always sends it
// (server-chat.cpp convert_transcriptions_to_chatcmpl), so a template's default_system_message cannot fire. Voxtral's
// Devstral template otherwise adds a ~1.2k-token coding persona (1429 prompt tokens vs 213).
//
// The user string is NOT ec6520161's. Its longer "verbatim and in full, from the beginning ... no quotation marks and no
// commentary" wording was chosen on the old examples/server. Re-measured 2026-10-17 on this tools/server:
// 7 cells, temperature 0, the librispeech fixture plus the 17.4 s test-2.mp3, with the opening = "New York Times ... July
// 21st 1969" before the word "newsprint". Script/data: temp/arex1014/_exsrv_1017/asr_prompt_sweep.py, asr_sweep_after.json.
//
//   cell                               ptok  fixture overlap / quote   test-2 opening / quote
//   no system + this user (before)     1429  1.00  clean               4/4   wrapped
//   "" + this user                      209  1.00  preamble+quoted     0/4   wrapped
//   "" + ec6520161 user                 230  0.30  "Transcript unavailable"  0/4
//   "Transcribe audio." + ec6520161     234  1.00  clean               0/4   wrapped   <- dropped the opening here
//   "Transcribe audio." + this user     213  1.00  clean               4/4   wrapped   <- chosen
//   "Transcribe audio." + "Transcribe this audio."   202  1.00  "Here's the transcription:" preamble   4/4
//
// "Wrapped" is a balanced pair of outer quotes. The ASR route removes it (asrfmt::strip_wrapping_quotes in
// server-task.cpp). An EMPTY system string behaved worse than a short one, so the short one stays.
common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates) {
    common_chat_prompt_preset asr_preset;
    asr_preset.system = "Transcribe audio.";
    asr_preset.user   = "Transcribe the speech in this audio verbatim. Output only the words spoken.";

    if (chat_templates && chat_templates->template_default && is_lfm2_template(chat_templates->template_default->source())) {
        asr_preset.system = "Perform ASR.";
        asr_preset.user   = "";
    }

    return asr_preset;
}

std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant) {
    if (!variant.empty()) {
        if (variant == "tool_use") {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source();
            }
            return "";
        }
        LOG_DBG("%s: unknown template variant: %s\n", __func__, variant.c_str());
    }
    return tmpls->template_default->source();
}

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override,
                                                     const std::string &        eos_token_override) {
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src  = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }

    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (default_template_src.find("<|channel|>") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("in message.content or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
                           "\"<|channel|>final<|message|>\" in message.content %}",
                           "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (default_template_src.find("[TOOL_CALLS]") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("if (message['content'] is none or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if (message['content'] is none or message['content'] == '' or "
                           "message['content']|length == 0) and (message['tool_calls'] is not defined or "
                           "message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
                           "{%- if false %}");
    }

    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool        add_bos   = false;
    bool        add_eos   = false;
    if (model) {
        const auto * vocab     = llama_model_get_vocab(model);
        const auto   get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos ||
                    template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN(
                        "common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't "
                          "work as intended.\n",
                        name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
        add_bos   = llama_vocab_get_add_bos(vocab);
        add_eos   = llama_vocab_get_add_eos(vocab);
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos               = add_bos;
    tmpls->add_eos               = add_eos;
    try {
        tmpls->template_default = std::make_unique<common_chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        LOG_ERR("%s: failed to initialize chat template\n", __func__);
        LOG_ERR("%s: please consider disabling jinja via --no-jinja, or using another chat template\n", __func__);
        throw e;
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<common_chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

const char * common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            return "Content-only";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE:
            return "peg-simple";
        case COMMON_CHAT_FORMAT_PEG_NATIVE:
            return "peg-native";
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return "peg-gemma4";
        case COMMON_CHAT_FORMAT_PEG_MINIMAX_M3:
            return "peg-minimax-m3";
        case COMMON_CHAT_FORMAT_PEG_K2_HORIZON:
            return "peg-k2-horizon";
        case COMMON_CHAT_FORMAT_PEG_FUNCTIONGEMMA:
            return "peg-functiongemma";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:
            return "none";
        case COMMON_REASONING_FORMAT_AUTO:
            return "auto";
        case COMMON_REASONING_FORMAT_DEEPSEEK:
            return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY:
            return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

common_reasoning_format common_reasoning_format_from_name(const std::string & format) {
    if (format == "none") {
        return COMMON_REASONING_FORMAT_NONE;
    }
    if (format == "auto") {
        return COMMON_REASONING_FORMAT_AUTO;
    }
    if (format == "deepseek") {
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    if (format == "deepseek-legacy") {
        return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    }
    throw std::runtime_error("Unknown reasoning format: " + format);
}

common_chat_prompt common_chat_template_direct_apply_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override,
    const std::optional<json> & tools_override,
    const std::optional<json> & additional_context) {
    jinja::context ctx(tmpl.source());

    // messages_override is already built for this template, do not touch its content parts
    json inp = json{
        {"messages", messages_override.has_value()
            ? *messages_override
            : messages_inp_normalizer(tmpl.original_caps()).normalize(inputs.messages)},
        {"bos_token", tmpl.bos_token()},
        {"eos_token", tmpl.eos_token()},
        {"enable_thinking", inputs.enable_thinking},
    };
    // openPangu's chat template gates reasoning on a `thinking` variable rather than the
    // ecosystem-standard `enable_thinking`, so the normal toggle never reaches it. Bridge the
    // standard control to it here so reasoning works on and off through `enable_thinking`. An
    // explicit `thinking` chat_template_kwarg still wins (merged via extra_context below).
    if (tmpl.source().find("<|pangu_text_start|>") != std::string::npos) {
        inp["thinking"] = inputs.enable_thinking;
    }
    if (tools_override.has_value() || !inputs.tools.empty()) {
        inp["tools"] = tools_override.has_value() ? *tools_override : inputs.tools;
    } else if (tmpl.source().find("tools is none") != std::string::npos || tmpl.source().find("tools is not none") != std::string::npos) {
        // A template that tests `tools is none` / `tools is not none` was written for a caller that always passes the variable
        // (Hugging Face's apply_chat_template passes tools=None when there are none). Left undefined, `tools is none` is false, the
        // template takes its "tools given" branch and `tools | tojson` fails on an undefined value: OLMo-3-7B-Instruct's own template
        // made the server refuse to start. Defining it as null restores the caller's contract; templates that never compare with
        // none are not touched.
        inp["tools"] = json(nullptr);
    }
    if (inputs.extra_context.is_object()) {
        // TODO: do we need to merge, or replacing is fine?
        for (const auto & [k, v] : inputs.extra_context.items()) {
            inp[k] = v;
        }
    }
    if (additional_context.has_value()) {
        // TODO: merge properly instead of overwriting (matching old behavior)
        for (const auto & [k, v] : additional_context->items()) {
            inp[k] = v;
        }
    }
    if (inputs.add_generation_prompt) {
        inp["add_generation_prompt"] = true;
    }
    if (inp.contains("preserve_reasoning") && inp["preserve_reasoning"].is_boolean()) {
        bool enabled = inp["preserve_reasoning"].get<bool>();
        jinja::caps_apply_preserve_reasoning(ctx, enabled);
    }
    if (inp.contains("reasoning_effort") && inp["reasoning_effort"].is_string() && !inp["reasoning_effort"].empty()) {
        std::string reasoning_effort = inp["reasoning_effort"].get<std::string>();
        jinja::caps_apply_reasoning_effort(ctx, reasoning_effort);
    }

    jinja::global_from_json(ctx, inp, inputs.mark_input);

    // bos/eos come from the model, not from the user
    ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(tmpl.bos_token()));
    ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(tmpl.eos_token()));

    // render
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    auto parts = jinja::runtime::gather_string_parts(results);

    common_chat_prompt result;
    for (auto & part : parts->val_str.parts) {
        result.parts.push_back({std::move(part.val), part.is_input});
    }

    // TODO: improve this later
    if (inputs.add_bos && !result.parts.empty()) {
        auto & first = result.parts.front();
        if (!first.is_input && string_starts_with(first.text, tmpl.bos_token())) {
            first.text.erase(0, tmpl.bos_token().size());
        }
    }
    if (inputs.add_eos && !result.parts.empty()) {
        auto & last = result.parts.back();
        if (!last.is_input && string_ends_with(last.text, tmpl.eos_token())) {
            last.text.resize(last.text.size() - tmpl.eos_token().size());
        }
    }
    return result;
}

common_chat_prompt common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

std::string common_chat_template_generation_prompt_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override,
    const std::optional<json> & tools_override,
    const std::optional<json> & additional_context) {

    autoparser::generation_params params = inputs;
    params.add_generation_prompt = false;
    params.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    std::string no_gen_prompt    = common_chat_template_direct_apply_impl(tmpl, params, messages_override, tools_override, additional_context).str();
    params.add_generation_prompt = true;
    std::string gen_prompt       = common_chat_template_direct_apply_impl(tmpl, params, messages_override, tools_override, additional_context).str();

    size_t prefix_len = 0;
    size_t min_size = std::min(no_gen_prompt.size(), gen_prompt.size());
    while (prefix_len < min_size && no_gen_prompt[prefix_len] == gen_prompt[prefix_len]) {
        prefix_len++;
    }
    return gen_prompt.substr(prefix_len);
}

std::string common_chat_template_generation_prompt(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_generation_prompt_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

namespace workaround {

// true when the template source compares a role against the literal 'developer' / "developer",
// i.e. it renders that role itself and must not have it rewritten to system
static bool template_has_developer_branch(const std::string & src) {
    return src.find("'developer'") != std::string::npos ||
           src.find("\"developer\"") != std::string::npos;
}

static void map_developer_role_to_system(json & messages) {
    for (auto & message : messages) {
        if (message.contains("role")) {
            if (message["role"] == "developer") {
                message["role"] = "system";
            }
        }
    }
}


// if first message is system and template does not support it, merge it with next message
static void system_message_not_supported(json & messages) {
    if (!messages.empty() && messages.front().at("role") == "system") {
        if (messages.size() > 1) {
            LOG_DBG("Merging system prompt into next message\n");
            auto & first_msg = messages.front();
            auto & second_msg = messages[1];
            second_msg["content"] = first_msg.at("content").get<std::string>()
                + "\n" + second_msg.at("content").get<std::string>();
            messages.erase(0);
        } else {
            LOG_WRN("Removing system prompt due to template not supporting system role\n");
            messages.erase(0);
        }
    }
}

static void requires_non_null_content(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls") && !message.contains("content")) {
            message["content"] = "";
        }
    }
}

static void func_args_not_string(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls")) {
            for (auto & tool_call : message["tool_calls"]) {
                if (tool_call.contains("function") && tool_call["function"].contains("arguments")) {
                    auto & args = tool_call["function"]["arguments"];
                    if (args.is_string()) {
                        const std::string raw = args.get<std::string>();
                        try {
                            args = json::parse(raw);
                        } catch (const std::exception & e) {
                            // These strings come from the client's replayed history, not from
                            // our own parser. A turn truncated at max_tokens or carrying an
                            // unquoted number would otherwise wedge the whole conversation
                            // behind an HTTP 500 on every later request. Keep the model's own
                            // text visible to the template, as the official DeepSeek-V4
                            // encoder does, and say so once.
                            LOG_WRN("%s: tool call arguments in history are not valid JSON (%s); passing them through as {\"arguments\": <raw text>}\n",
                                    __func__, e.what());
                            args = json::object({{"arguments", raw}});
                        }
                    }
                }
            }
        }
    }
}

// Trim leading/trailing whitespace from message contents before rendering. This
// has to run on the messages (not on the rendered JSON) because templates with
// string-only content caps concatenate typed content parts into a single string
// during rendering, after which the per-part whitespace can no longer be reached.
// Both the plain string content and the text of typed content parts are trimmed.
static void trim_all_content(std::vector<common_chat_msg> & messages) {
    for (auto & message : messages) {
        message.content           = trim_whitespace(message.content);
        message.reasoning_content = trim_whitespace(message.reasoning_content);
        for (auto & part : message.content_parts) {
            if (part.type == "text") {
                part.text = trim_whitespace(part.text);
            }
        }
    }
}

}

static json common_chat_extra_context() {
    json ctx = json::object();
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::string datetime_str = format_time(now, "%b %d %Y");
    std::string date_str = format_time(now, "%d %b %Y");
    ctx["datetime"] = datetime_str;
    ctx["date_string"] = date_str;
    return ctx;
}

static common_chat_params common_chat_params_init_lfm2_tokens(const common_chat_template & tmpl, const autoparser::generation_params & inputs) {
    return common_chat_params_init_lfm2(tmpl, inputs, /* tool_list_tokens = */ true);
}

static common_chat_params common_chat_params_init_lfm2_5(const common_chat_template & tmpl, const autoparser::generation_params & inputs) {
    return common_chat_params_init_lfm2(tmpl, inputs, /* tool_list_tokens = */ false);
}

// Older gemma4 templates need their tool responses rewritten before rendering
static common_chat_params common_chat_params_init_gemma4_legacy(const common_chat_template & tmpl, const autoparser::generation_params & inputs) {
    auto adjusted = inputs;
    workaround::convert_tool_responses_gemma4(adjusted.messages);
    return common_chat_params_init_gemma4(tmpl, adjusted);
}

// Pick the dedicated handler for a template from its source, or null for the autoparser.
// Order matters: the first match wins, and later checks assume the earlier ones did not match.
static common_chat_params_init_fn common_chat_template_detect_params_init(const std::string & src) {
    // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
    // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
    if (src.find("[SYSTEM_PROMPT]") != std::string::npos && src.find("[TOOL_CALLS]") != std::string::npos &&
        src.find("[ARGS]") != std::string::npos && src.find("[CALL_ID]") == std::string::npos) {
        LOG_INF("Using specialized template: Ministral/Magistral Large 3\n");
        return common_chat_params_init_ministral_3;
    }

    // LLM-jp-4.1 - GPT-OSS dialect (spaces after special tokens, <|end|>-separated parallel calls)
    if (src.find("chat_format=llm-jp-harmony-v1") != std::string::npos) {
        LOG_INF("Using specialized template: LLM-jp Harmony v1\n");
        return common_chat_params_init_llm_jp_harmony;
    }

    // Harmony XML tool calls, checked before the GPT-OSS JSON calls
    if (src.find("<|channel|>") != std::string::npos && src.find("<|constrain|>xml") != std::string::npos &&
        src.find("rfl:parameter") != std::string::npos) {
        LOG_INF("Using specialized template: Harmony XML\n");
        return common_chat_params_init_harmony_xml;
    }

    // GPT-OSS - has unique channel-based structure that needs dedicated handler
    if (src.find("<|channel|>") != std::string::npos) {
        LOG_INF("Using specialized template: GPT-OSS\n");
        return common_chat_params_init_gpt_oss;
    }

    // Muse Spark shares the Muse recipient protocol and adds a commentary recipient.
    if (src.find("to=commentary") != std::string::npos && src.find("<|eom|>") != std::string::npos) {
        LOG_INF("Using specialized template: Muse Spark\n");
        return common_chat_params_init_muse_spark;
    }

    // Muse Glimmer format using " to=<recipient>" recipients and <|eom|>/<|eot|> message terminators.
    if (src.find("<atem:function_calls>") != std::string::npos && src.find("<|eom|>") != std::string::npos) {
        LOG_INF("Using specialized template: Muse Glimmer\n");
        return common_chat_params_init_muse_glimmer;
    }

    // Functionary v3.2 - uses recipient-based format with >>>recipient\n{content}
    // Detection: template has ">>>all" for content and ">>>" prefix for tool calls
    if (src.find(">>>all") != std::string::npos && src.find(">>>${recipient}") != std::string::npos) {
        LOG_INF("Using specialized template: Functionary v3.2\n");
        return common_chat_params_init_functionary_v3_2;
    }

    // Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
    // Detection: template has "<|tool_calls_section_begin|>" and "functions." prefix in tool call IDs
    if (src.find("<|tool_calls_section_begin|>") != std::string::npos &&
        src.find("<|tool_call_begin|>") != std::string::npos) {
        LOG_INF("Using specialized template: Kimi K2 Thinking\n");
        return common_chat_params_init_kimi_k2;
    }

    // Kimi K3 - the <|open|>/<|close|>/<|end_of_msg|> markers are unique to it
    if (src.find("<|open|>") != std::string::npos && src.find("<|close|>") != std::string::npos &&
        src.find("<|end_of_msg|>") != std::string::npos) {
        LOG_INF("Using specialized template: Kimi K3\n");
        return common_chat_params_init_kimi_k3;
    }

    // K2 Horizon - <|ifm|im_start|> turns, <ifm|think*> reasoning picked by reasoning_effort and
    // <ifm|tool_calls> sections; the three think tag pairs defeat the autoparser's reasoning detection
    // (fork: also the abenzerps template, recognised by its <ifm|think> tag, ik_llama #2466)
    if ((src.find("<|ifm|im_start|>") != std::string::npos || src.find("<ifm|think>") != std::string::npos) &&
        src.find("<ifm|tool_calls>") != std::string::npos) {
        LOG_INF("Using specialized template: K2 Horizon\n");
        return common_chat_params_init_k2_horizon;
    }

    // Ling 3.0 / Bailing V3 - <role>X</role> sections with <arg_key>/<arg_value> tagged
    // tool calls. <role> sections are unique to this family among the tagged-arg templates.
    if (src.find("<role>ASSISTANT</role>") != std::string::npos &&
        src.find("<arg_key>") != std::string::npos) {
        LOG_INF("Using specialized template: Ling 3.0 (Bailing V3)\n");
        return common_chat_params_init_ling3;
    }

    // Cohere2 MoE / North Code - marker-wrapped format with <|START_TEXT|> content and
    // <|START_ACTION|> JSON tool calls. <|START_TEXT|> is unique to this template (the older
    // Command-R templates use <|START_RESPONSE|>).
    if (src.find("<|START_TEXT|>") != std::string::npos &&
        src.find("<|START_ACTION|>") != std::string::npos) {
        LOG_INF("Using specialized template: Cohere2 MoE\n");
        return common_chat_params_init_cohere2moe;
    }

    // Inkling / TML typed content blocks: this marker combination is unique to the template
    // (unslothai/llama.cpp G25731 = ggml-org/llama.cpp #25731, common/chat.cpp detection hunk)
    if (src.find("<|content_thinking|>") != std::string::npos &&
        src.find("<|content_text|>") != std::string::npos &&
        src.find("<|message_model|>") != std::string::npos) {
        LOG_INF("Using specialized template: Inkling\n");
        return common_chat_params_init_inkling;
    }

    // MiroThinker - uses MCP style toolcalling <use_mcp_tool> ... </use_mcp_tool> (fork)
    if (src.find("</use_mcp_tool>") != std::string::npos &&
        src.find("</server_name>") != std::string::npos) {
        LOG_INF("Using specialized template: MiroThinker\n");
        return common_chat_params_init_mirothinker;
    }

    if (is_lfm2_template(src)) {
        LOG_INF("Using specialized template: LFM2\n");
        return common_chat_params_init_lfm2_tokens;
    }

    // LFM2.5 format detection: template uses plain "List of tools: [...]" with no special tokens
    if (src.find("List of tools: [") != std::string::npos &&
        src.find("<|tool_list_start|>") == std::string::npos) {
        LOG_INF("Using specialized template: LFM2.5\n");
        return common_chat_params_init_lfm2_5;
    }

    // GigaChatV3 format detection
    if (src.find("<|role_sep|>") != std::string::npos &&
        src.find("<|message_sep|>") != std::string::npos &&
        src.find("<|function_call|>") == std::string::npos) {
        LOG_INF("Using specialized template: GigaChatV3\n");
        return common_chat_params_init_gigachat_v3;
    }

    // MiniMax-M3: the namespace token "]<]minimax[>[" collides with the autoparser's
    // markup delimiters, so detect the template and use a dedicated parser.
    if (src.find("]<]minimax[>[") != std::string::npos &&
        src.find("<tool_call>") != std::string::npos &&
        src.find("<invoke name=") != std::string::npos) {
        LOG_INF("Using specialized template: MiniMax-M3\n");
        return common_chat_params_init_minimax_m3;
    }

    // DeepSeek V3.2/V4 format detection: template defines dsml_token and uses it for tool calls.
    // The template source contains the token as a variable assignment, not as a literal in markup.
    // V3.2 names the tool call block "function_calls", V4 names it "tool_calls".
    // V4.1 tag names have a leading space and its template does not name the dsml_token variable.
    if ((src.find("dsml_token") != std::string::npos &&
         src.find("DSML") != std::string::npos &&
         (src.find("function_calls") != std::string::npos ||
          src.find("tool_calls") != std::string::npos ||
          src.find("' calls>") != std::string::npos ||   // fork: V4.1 also by its spaced tags built from dsml_token
          src.find("' invoke") != std::string::npos)) ||
        src.find("｜DSML｜ calls>") != std::string::npos) {
        LOG_INF("Using specialized template: DeepSeek V3.2/V4/V4.1\n");
        return common_chat_params_init_deepseek_v3_2;
    }

    // FunctionGemma: <start_function_call>call:NAME{...}<end_function_call> (fork)
    if (src.find("'<start_function_call>call:'") != std::string::npos) {
        LOG_INF("Using specialized template: FunctionGemma\n");
        return common_chat_params_init_functiongemma;
    }

    // Gemma4 format detection
    if (src.find("'<|tool_call>call:'") != std::string::npos) {
        LOG_INF("Using specialized template: Gemma4\n");
        if (src.find("{#- OpenAI Chat Completions:") == std::string::npos) {
            LOG_WRN("%s: detected an outdated gemma4 chat template, applying compatibility workarounds. "
                    "Consider updating to the official template.\n", __func__);
            return common_chat_params_init_gemma4_legacy;
        }
        return common_chat_params_init_gemma4;
    }

    // MiniCPM5 - XML tool calls with <function name="..."><param name="...">...</param></function>
    if (src.find("Tool usage guidelines:") != std::string::npos &&
        src.find("<function name=\"") != std::string::npos &&
        src.find("<param name=\"") != std::string::npos) {
        LOG_INF("Using specialized template: MiniCPM5\n");
        return common_chat_params_init_minicpm5;
    }

    // TranslateGemma - user content must follow a custom schema with language codes
    if (src.find("[source_lang_code]") != std::string::npos &&
        src.find("[target_lang_code]") != std::string::npos) {
        LOG_INF("Using specialized template: TranslateGemma\n");
        return common_chat_params_init_translate_gemma;
    }

    // OLMo-3 Instruct: <function_calls>name(k=<JSON>)</function_calls> with the tools rendered into <functions> as JSON (fork).
    // `tools | tojson` is required: the OLMo-3 Think templates carry the same tags but never render `tools`.
    if (src.find("<function_calls>") != std::string::npos &&
        src.find("</function_calls>") != std::string::npos &&
        src.find("<functions>") != std::string::npos &&
        src.find("tools | tojson") != std::string::npos &&
        src.find("dsml_token") == std::string::npos &&
        src.find("<atem:") == std::string::npos) {
        LOG_INF("Using specialized template: OLMo-3\n");
        return common_chat_params_init_olmo3;
    }

    // Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash
    if (src.find("<tool_call>") != std::string::npos &&
        src.find("<function=") != std::string::npos &&
        src.find("<parameter=") != std::string::npos &&
        // Exclude models that don't use \n between tags
        src.find("'<tool_call><function=' ~ tool_call.name ~ '>'") == std::string::npos) {
        LOG_INF("Using specialized template: Qwen3-Coder\n");
        return common_chat_params_init_qwen3_coder;
    }

    return nullptr;
}

common_chat_template::common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token) {
    jinja::lexer lexer;
    auto lexer_res = lexer.tokenize(src);
    this->prog = jinja::parse_from_tokens(lexer_res);

    this->src = lexer_res.source;
    this->bos_tok = bos_token;
    this->eos_tok = eos_token;

    this->caps = jinja::caps_get(prog);
    // LOG_INF("%s: caps:\n%s\n", __func__, this->caps.to_string().c_str());

    this->params_init = common_chat_template_detect_params_init(this->src);
    if (this->params_init) {
        return;
    }

    // No specialized handler matched: say so at load (row TABBYAPI-CHAT-HANDLER-LOADLOG).
    LOG_INF("Using chat handler: autoparser (no specialized template matched)\n");

    // The analysis depends only on the template, so run it once here instead of on every apply.
    // A failure is kept for apply to report, so a bad template still loads like it did before.
    try {
        analysis = std::make_unique<autoparser::autoparser>();
        analysis->analyze_template(*this);
    } catch (const std::exception & e) {
        analysis.reset();
        analysis_error = e.what();
    }
}

common_chat_template::~common_chat_template() = default;
common_chat_template::common_chat_template(common_chat_template &&) = default;
common_chat_template & common_chat_template::operator=(common_chat_template &&) = default;

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const autoparser::generation_params & params) {
    if (!tmpl.params_init) {
        return std::nullopt;
    }
    return tmpl.params_init(tmpl, params);
}

// Row S69-ENGREV-XINFERENCE-SYSTEM-FIRST-NORMALISE-2026-10-27-01. Templates of the Qwen3 family end a render with
// raise_exception('System message must be at the beginning') when a system message is not message 0, which reached the client
// as HTTP 500 (a pipeline stage or a client that interleaves a system reminder). Xinference detects such templates by that
// marker text, case-insensitive (xinference/model/llm/llm_family.py:52-71, sha 0bb96a8a, Apache-2.0; idea only, no code
// copied) and stops at a 400; we fold the later system messages into message 0 instead, so the turn still runs.
static bool common_chat_template_requires_system_first(const std::string & src) {
    static const char * marker = "system message must be at the beginning";
    std::string lower(src.size(), '\0');
    std::transform(src.begin(), src.end(), lower.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return lower.find(marker) != std::string::npos;
}

// Returns true and fills `out` when `in` has a system message that is not message 0 (or no system message in front of a
// later one): every system message becomes one message 0 (text joined with a blank line), the other messages keep their order.
// Returns false, out untouched, when the input is already well ordered. A system message with a non-text part cannot be
// merged: that is a client error, reported as std::invalid_argument naming the index (the server answers 400, never 500).
static bool common_chat_normalise_system_first(const std::vector<common_chat_msg> & in, std::vector<common_chat_msg> & out) {
    bool misplaced = false;
    for (size_t i = 1; i < in.size(); i++) {
        if (in[i].role == "system") {
            misplaced = true;
            break;
        }
    }
    if (!misplaced) {
        return false;
    }
    std::string merged;
    bool        have_system = false;
    std::vector<common_chat_msg> rest;
    std::string roles;
    for (size_t i = 0; i < in.size(); i++) {
        const common_chat_msg & m = in[i];
        roles += (i == 0 ? "" : ",") + m.role;
        if (m.role != "system") {
            rest.push_back(m);
            continue;
        }
        std::string text = m.content;
        if (!m.content_parts.empty()) {
            text.clear();
            for (const auto & part : m.content_parts) {
                if (part.type != "text") {
                    throw std::invalid_argument("system message " + std::to_string(i) +
                                                " has a non-text content part and cannot be moved to the start of the conversation");
                }
                text += (text.empty() ? "" : "\n\n") + part.text;
            }
        }
        merged += (have_system ? "\n\n" : "") + text;
        have_system = true;
    }
    LOG_INF("%s: template requires the system message first; merged the system messages into message 0 (role sequence: %s)\n",
            __func__, roles.c_str());
    out.clear();
    common_chat_msg sys;
    sys.role    = "system";
    sys.content = merged;
    out.push_back(std::move(sys));
    for (auto & m : rest) {
        out.push_back(std::move(m));
    }
    return true;
}

static common_chat_params common_chat_templates_apply_jinja(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    autoparser::generation_params params;
    params.tools = common_chat_tools_to_json_oaicompat(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    const auto & src             = tmpl.source();
    const auto & caps            = tmpl.original_caps();
    std::vector<common_chat_msg>        trimmed_messages;
    std::vector<common_chat_msg>        system_first_messages;
    const std::vector<common_chat_msg> * messages_to_render = &inputs.messages;
    if (common_chat_template_requires_system_first(src) &&
        common_chat_normalise_system_first(inputs.messages, system_first_messages)) {
        messages_to_render = &system_first_messages;
    }
    if (src.find("You have access to the following functions in JSONSchema format") != std::string::npos) {
        // StepFun: trim message contents (including typed content parts) before rendering,
        // otherwise leftover whitespace drives the model into reasoning loops (issue #24181)
        trimmed_messages   = *messages_to_render;
        workaround::trim_all_content(trimmed_messages);
        messages_to_render = &trimmed_messages;
    }
    params.messages              = src.find("｜DSML｜ calls>") != std::string::npos ?
                                       render_message_to_json_deepseek_v4_1(*messages_to_render, tmpl.original_caps()) :
                                       render_message_to_json(*messages_to_render, tmpl.original_caps());
    params.tool_choice           = inputs.tool_choice;
    params.reasoning_format      = inputs.reasoning_format;
    params.enable_thinking       = inputs.enable_thinking;
    params.grammar               = inputs.grammar;
    params.now                   = inputs.now;
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.add_bos               = tmpls->add_bos;
    params.add_eos               = tmpls->add_eos;

    params.continue_final_message = inputs.continue_final_message;
    if (params.continue_final_message != COMMON_CHAT_CONTINUATION_NONE) {
        params.add_generation_prompt = false;

        if (!inputs.messages.empty()) {
            // Render messages[:-1] and store continuation message separately
            params.continue_msg = inputs.messages.back();
            params.messages.erase(params.messages.size() - 1);
        }

        if (params.continue_final_message == COMMON_CHAT_CONTINUATION_AUTO && !inputs.messages.empty()) {
            // Resolve based on message content
            params.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
            if (!params.continue_msg.reasoning_content.empty() &&
                params.continue_msg.content.empty() &&
                params.continue_msg.content_parts.empty()) {
                params.continue_final_message = COMMON_CHAT_CONTINUATION_REASONING;
            }
        }
    }

    if (src.find("<|channel|>") == std::string::npos &&
        !workaround::template_has_developer_branch(src)) {
        // map developer to system for all models except for GPT-OSS and any template that
        // names a 'developer' role itself (SOURCE-READ: RunanywhereAI/trnrt@ef581f5
        // parsers.py:248-251, idea only, no code copied)
        workaround::map_developer_role_to_system(params.messages);
    }

    if (!tmpl.original_caps().supports_system_role) {
        workaround::system_message_not_supported(params.messages);
    }

    if (tmpl.original_caps().supports_tool_calls) {
        // some templates will require the content field in tool call messages
        // to still be non-null, this puts an empty string everywhere where the
        // content field is null
        workaround::requires_non_null_content(params.messages);
    }

    if (tmpl.original_caps().supports_object_arguments) {
        workaround::func_args_not_string(params.messages);
    }

    // ADAPTED 2026-09-13: the function is common_chat_template_generation_prompt (singular
    // "template"), declared at chat.h:370 with exactly this (common_chat_template &,
    // autoparser::generation_params &) signature -- the compiler named it as the near-miss.
    // There is no "templates" plural form in this tree.
    //
    // This fork computes it here as the FALLBACK for a handler that leaves its own empty (upstream computes none here: each handler
    // renders its own). Rendered on the messages as they are, it can throw where a specialized handler would have rewritten them
    // first -- TranslateGemma's template raises on plain string user content, and the handler's whole job is to convert it to its
    // content schema (ctest test-chat test_translate_gemma: an uncaught jinja exception -> 0xC0000409). So a failure is kept and
    // rethrown only where the fallback is actually used.
    std::exception_ptr generation_prompt_error;
    try {
        params.generation_prompt = common_chat_template_generation_prompt(tmpl, params);
    } catch (...) {
        generation_prompt_error = std::current_exception();
    }
    auto fallback_generation_prompt = [&]() -> const std::string & {
        if (generation_prompt_error) {
            std::rethrow_exception(generation_prompt_error);
        }
        return params.generation_prompt;
    };

    params.extra_context = common_chat_extra_context();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }
    if (!params.grammar.empty() && !params.json_schema.is_null()) {
        throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
    }
    params.parallel_tool_calls = inputs.parallel_tool_calls;

    if (params.tools.is_array()) {
        if (params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !params.grammar.empty()) {
            throw std::runtime_error("Cannot specify grammar with tools");
        }
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN(
                "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    if (inputs.force_pure_content) {
        LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
        // Create the result structure
        common_chat_params data;
        auto params_copy               = params;
        params_copy.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
        data.prompt                    = common_chat_template_direct_apply_impl(tmpl, params_copy);
        data.generation_prompt         = common_chat_template_generation_prompt_impl(tmpl, params);
        data.format                    = COMMON_CHAT_FORMAT_PEG_NATIVE;
        data.parser                    = build_chat_peg_parser([&data](common_chat_peg_builder &p) {
            return p.literal(data.generation_prompt) << p.content(p.rest());
        });
        return data;
    }

    // Operator override of the XML tool-call format: wins over the specialised handlers and the autoparser, but only when this
    // request actually offers tools (without them there is nothing to parse and the template's own handling stays).
    if (!inputs.tool_call_format.empty() && params.tools.is_array() && !params.tools.empty() &&
        params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        LOG_DBG("Using operator tool-call format override\n");
        auto result = common_chat_params_init_xml_override(tmpl, params, common_chat_xml_tool_format_parse(inputs.tool_call_format));
        if (result.generation_prompt.empty()) {
            result.generation_prompt = fallback_generation_prompt();
        }
        return result;
    }

    if (auto result = common_chat_try_specialized_template(tmpl, params)) {
        // Keep the handler's own generation_prompt: on continue_final_message the specialized handlers
        // (qwen3_coder :1212, ministral :1098, ...) append the prefilled reasoning/content to it, and the
        // unconditional overwrite with params.generation_prompt (from :3809) threw that away -- test-chat
        // got "<|im_start|>assistant\n<think>\n" where "...I'm thinking\n</think>\n\nHello, " was expected.
        // Upstream returns *result unmodified (engines/llama.cpp common/chat.cpp:1331-1333); the fallback
        // below only fills a prompt a handler left empty.
        if (result->generation_prompt.empty()) {
            result->generation_prompt = fallback_generation_prompt();
        }
        return *result;
    }

    // the autoparser path below renders the unrewritten messages itself, so a template that could not render them fails here
    if (generation_prompt_error) {
        std::rethrow_exception(generation_prompt_error);
    }

    if (!tmpl.analysis) {
        throw std::invalid_argument("Unable to generate parser for this template. Automatic parser generation failed: " + tmpl.analysis_error);
    }

    try {
        LOG_DBG("%s: using differential autoparser\n", __func__);
        const auto & autoparser = *tmpl.analysis;
        auto auto_params = autoparser::peg_generator::generate_parser(tmpl, params, autoparser);

        common_chat_msg_delimiters delimiters;
        if (!autoparser.assistant_start.empty()) {
            delimiters.add(COMMON_CHAT_ROLE_ASSISTANT, autoparser.assistant_start);
        }
        if (!autoparser.user_start.empty()) {
            delimiters.add(COMMON_CHAT_ROLE_USER, autoparser.user_start);
        }

        auto_params.message_delimiters = std::move(delimiters);

        auto_params.supports_thinking = autoparser.reasoning.mode != autoparser::reasoning_mode::NONE;
        if (auto_params.supports_thinking) {
            auto_params.thinking_start_tag = trim_whitespace(autoparser.reasoning.start);
            auto end_tag = trim_whitespace(autoparser.reasoning.end);
            if (!end_tag.empty()) {
                auto_params.thinking_end_tags = {std::move(end_tag)};
            }
        }
        const auto & arena = auto_params.parser;
        LOG_DBG("%s: generated parser:\n%s\n\nparser generation prompt: %s\n", __func__, arena.dump(arena.root()).c_str(), auto_params.generation_prompt.c_str());
        return auto_params;
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("Unable to generate parser for this template. Automatic parser generation failed: ") + e.what());
    }
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(const struct common_chat_templates *        tmpls,
                                                             const struct common_chat_templates_inputs & inputs) {
    size_t                          alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text" && part.type != "media_marker") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
                ;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg     = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({ msg.role.c_str(), content.c_str() });
        size_t msg_size = msg.role.size() + content.size();
        alloc_size += msg_size + (msg_size / 4);  // == msg_size * 1.25 but avoiding float ops
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t      res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt,
                                                 buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported, try using --jinja");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(),
                                        buf.size());
    }

    // for safety, we check the result again
    if (res < 0 || (size_t) res > buf.size()) {
        throw std::runtime_error("failed to apply chat template, try using --jinja");
    }

    common_chat_params params;
    params.prompt = common_chat_prompt(std::string(buf.data(), res));
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                               const struct common_chat_templates_inputs & inputs) {
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja ? common_chat_templates_apply_jinja(tmpls, inputs) :
                              common_chat_templates_apply_legacy(tmpls, inputs);
}

std::string common_chat_prompt::str() const {
    if (parts.size() == 1) {
        return parts[0].text;
    }
    size_t len = 0;
    for (const auto & part : parts) {
        len += part.text.size();
    }
    std::string out;
    out.reserve(len);
    for (const auto & part : parts) {
        out += part.text;
    }
    return out;
}

void common_chat_prompt::append(const std::string & text, bool is_input) {
    if (text.empty()) {
        return;
    }
    if (!parts.empty() && parts.back().is_input == is_input) {
        parts.back().text += text;
    } else {
        parts.push_back({text, is_input});
    }
}

void common_chat_prompt::replace_last(const std::string & from, const std::string & to) {
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        if (it->is_input) {
            continue;
        }
        size_t pos = it->text.rfind(from);
        if (pos != std::string::npos) {
            it->text.replace(pos, from.size(), to);
            return;
        }
    }
}

void common_chat_input::append(const std::string & piece, llama_token token) {
    if (piece.empty()) {
        return;
    }
    tokens.push_back(token);
    tokens.resize(tokens.size() + piece.size() - 1, LLAMA_TOKEN_NULL);
    text += piece;
}

void common_chat_input::append(const common_chat_input & chunk) {
    tokens.insert(tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
    text += chunk.text;
}

void common_chat_input::truncate(size_t pos) {
    if (pos < text.size()) {
        text.erase(pos);
        tokens.resize(pos);
    }
}

common_chat_input common_chat_input::substr(size_t pos, size_t n) const {
    common_chat_input out;
    out.text = text.substr(pos, n);
    out.tokens.assign(tokens.begin() + pos, tokens.begin() + pos + out.size());
    return out;
}

void common_chat_input::prepend(const std::string & prefix) {
    tokens.insert(tokens.begin(), prefix.size(), LLAMA_TOKEN_NULL);
    text = prefix + text;
}

void common_chat_input::prepend(const common_chat_input & prefix) {
    tokens.insert(tokens.begin(), prefix.tokens.begin(), prefix.tokens.end());
    text = prefix.text + text;
}

common_chat_input common_chat_input_tokenize(const llama_vocab * vocab, const std::string & text) {
    common_chat_input input;
    auto tokens = common_tokenize(vocab, text, false, true);
    for (size_t i = 0; i < tokens.size(); i++) {
        std::string piece = common_token_to_piece(vocab, tokens[i], true);
        if (i == 0 && std::isspace(piece[0]) && !std::isspace(text[0])) {
            // Some tokenizers will add a space before the first special token, need to exclude
            continue;
        }
        input.append(piece, tokens[i]);
    }
    if (input.text != text) {
        // the pieces do not give back the same text, keep the text without tokens
        return common_chat_input(text);
    }
    return input;
}

common_chat_msg common_chat_parse(const common_chat_input &         input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    return common_chat_peg_parse(params.parser, input, is_partial, params);
}

// Replace ill-formed UTF-8 in `in` with U+FFFD, one per "maximal subpart" of an ill-formed sequence (Unicode 15.0
// section 3.9, "U+FFFD Substitution of Maximal Subparts"; the same granularity as the WHATWG UTF-8 decoder), using the
// well-formed byte sequences of table 3-7 (so overlong forms, UTF-16 surrogates and code points above U+10FFFF are
// ill-formed as well, which is what a strict JSON writer demands).
//
// keep_incomplete_tail: a sequence that is well-formed so far but cut off by the end of `in` is left untouched,
// because while a message is still streaming the next token may complete it.
//
// Returns false (and leaves `out` alone) when `in` needed no repair, so the common case costs one scan, no copy.
//
// WHY HERE: a model can emit non-UTF-8 bytes (measured 2026-10-01: E5 A1 followed by 'o' from a PQ2_0 model). Every
// PEG text parser (until/rest/string content) returns FAIL on them, and the non-partial path below then throws, which
// the server answers with HTTP 500 and the whole response is lost. Row
// S69-SERVER-COMPLETION-500-ON-CONTENT-ONLY-PARSE-OF-GARBAGE-OUTPUT-2026-10-01-01.
static bool chat_utf8_repair(const std::string & in, bool keep_incomplete_tail, std::string & out, size_t & n_replaced) {
    static const char REPLACEMENT[] = "\xEF\xBF\xBD";
    const size_t      n             = in.size();
    bool              repaired      = false;
    size_t            i             = 0;
    size_t            copied        = 0;  // bytes of `in` already represented in `out` (only meaningful once repaired)

    auto flush_until = [&](size_t end) {
        out.append(in, copied, end - copied);
        copied = end;
    };

    while (i < n) {
        const unsigned char c = (unsigned char) in[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        size_t        need = 0;
        unsigned char lo   = 0x80;  // allowed range of the SECOND byte; later continuation bytes are always 80..BF
        unsigned char hi   = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1;
        } else if (c == 0xE0) {
            need = 2;
            lo   = 0xA0;
        } else if (c >= 0xE1 && c <= 0xEC) {
            need = 2;
        } else if (c == 0xED) {
            need = 2;
            hi   = 0x9F;
        } else if (c >= 0xEE && c <= 0xEF) {
            need = 2;
        } else if (c == 0xF0) {
            need = 3;
            lo   = 0x90;
        } else if (c >= 0xF1 && c <= 0xF3) {
            need = 3;
        } else if (c == 0xF4) {
            need = 3;
            hi   = 0x8F;
        }
        // else: 80..C1 and F5..FF never start a sequence (need stays 0)

        size_t j   = i + 1;
        size_t got = 0;
        while (got < need && j < n) {
            const unsigned char d = (unsigned char) in[j];
            const unsigned char l = got == 0 ? lo : 0x80;
            const unsigned char h = got == 0 ? hi : 0xBF;
            if (d < l || d > h) {
                break;
            }
            j++;
            got++;
        }
        if (need > 0 && got == need) {
            i = j;  // a complete, well-formed sequence
            continue;
        }
        if (need > 0 && j >= n && keep_incomplete_tail) {
            break;  // well-formed so far, but the input ends inside it: wait for more bytes
        }
        // ill-formed (or cut off in a final message): replace the maximal subpart [i, j)
        if (!repaired) {
            repaired = true;
            out.clear();
            out.reserve(n + 8);
        }
        flush_until(i);
        out += REPLACEMENT;
        copied = j;
        n_replaced++;
        i = j;
    }
    if (repaired) {
        flush_until(n);
    }
    return repaired;
}

// Does this parse start INSIDE a thinking block that the generation prompt opened, with the call gate asked for?
static bool chat_thinking_left_open(const common_chat_parser_params & params) {
    // reasoning_format none / reasoning_in_content: the client asked for the thinking text inline in content, which this
    // gate (it files the text as reasoning_content) would change
    if (!params.gate_calls_in_reasoning || params.tools.empty() || !params.parse_tool_calls || params.is_continuation ||
        params.reasoning_format == COMMON_REASONING_FORMAT_NONE || params.reasoning_in_content ||
        params.thinking_start_tag.empty() || params.thinking_end_tags.empty()) {
        return false;
    }
    return common_chat_generation_opens_thinking(params);
}

bool common_chat_generation_opens_thinking(const common_chat_parser_params & params) {
    if (params.thinking_start_tag.empty()) {
        return false;
    }
    std::string gp = params.generation_prompt.text;
    while (!gp.empty() && (gp.back() == '\n' || gp.back() == ' ' || gp.back() == '\r' || gp.back() == '\t')) {
        gp.pop_back();
    }
    const std::string & st = params.thinking_start_tag;
    std::string st_trim = st;
    while (!st_trim.empty() && (st_trim.back() == '\n' || st_trim.back() == ' ')) {
        st_trim.pop_back();
    }
    return !st_trim.empty() && gp.size() >= st_trim.size() && gp.compare(gp.size() - st_trim.size(), st_trim.size(), st_trim) == 0;
}

// The tag that closes the thinking block: thinking_end_tags[0] without the blank some handlers put in front of it for the
// reasoning-budget sampler's forced message (upstream parsers/qwen3-coder.cpp: { "\n</think>", "</think>", "<tool_call>" }). The
// format parsers end the reasoning at the tag itself and keep that blank as reasoning, so the gate and the mention mask must too.
static std::string chat_thinking_close_tag(const common_chat_parser_params & params) {
    if (params.thinking_end_tags.empty()) {
        return "";
    }
    const std::string & t = params.thinking_end_tags[0];
    const size_t        a = t.find_first_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : t.substr(a);
}

// What the format parser itself puts in front of the reasoning text: the blank that follows the start tag in the generation
// prompt ("<think>\n") is reasoning for some parsers (the autoparser's optspace(start)) and skipped by others (p.space()).
// Probed on a one-letter reasoning, so the gated reasoning starts exactly as the parser's own would.
static std::string chat_reasoning_lead(const common_peg_arena & src_parser, const common_chat_parser_params & inner) {
    common_chat_parser_params q = inner;
    q.quiet_unparsed            = true;
    try {
        const common_chat_msg m = common_chat_peg_parse(src_parser, common_chat_input("x"), true, q);
        const std::string & r   = m.reasoning_content;
        if (!r.empty() && r.back() == 'x' && r.find_first_not_of(" \t\r\n", 0) == r.size() - 1) {
            return r.substr(0, r.size() - 1);
        }
    } catch (const std::exception &) {
    }
    return "";
}

// The rescue for any call format: the template's own parser decides what a whole call is. A candidate is a line of the
// reasoning that starts outside code with a non-alphanumeric character (a tag or a bracket, never prose). It is given to
// the format parser as if the model had closed the thinking block right before it ("x" + end tag + the line and all
// after it: a grammar that keeps an unclosed block open to the end would file it as reasoning again) or, failing that,
// as if written right after the start tag. It is accepted when the parser finds only tool calls in it (no content, no
// reasoning but the placeholder) and prose after the calls cannot vanish (the parse fails or returns it as text), so a
// quoted call followed by prose is refused. The first accepted candidate wins: it is the earliest group of calls that
// runs to the end. The text stays in reasoning_content (it was streamed). Returns the number of calls added.
static size_t chat_rescue_native_format_calls(const common_peg_arena & src_parser, const std::string & text,
                                              const common_chat_parser_params & inner, common_chat_msg & msg) {
    if (src_parser.empty() || inner.tools.empty() || !msg.tool_calls.empty()) {
        return 0;
    }
    common_chat_parser_params q = inner;
    q.quiet_unparsed            = true;
    q.coerce_tool_args          = false;
    auto blank = [](const std::string & s) { return s.find_first_not_of(" \t\r\n") == std::string::npos; };
    auto trim  = [](const std::string & s) {
        const size_t a = s.find_first_not_of(" \t\r\n");
        const size_t b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    const std::string closer = chat_thinking_close_tag(inner);
    // the two ways a candidate is put to the parser, and the placeholder reasoning each one leaves
    std::vector<std::pair<std::string, std::string>> forms;
    if (!closer.empty()) {
        forms.emplace_back("x" + closer, "x");
    }
    forms.emplace_back("", "");
    static const char probe[] = "and then some prose";
    for (const size_t ls : common_chat_reasoning_call_line_starts(text)) {
        const size_t p = text.find_first_not_of(" \t", ls);
        if (p == std::string::npos || p >= text.size()) {
            continue;
        }
        const unsigned char c = (unsigned char) text[p];
        if (c == '\n' || c == '\r' || (c < 0x80 && std::isalnum(c))) {
            continue;
        }
        const std::string tail = text.substr(p);
        for (const auto & form : forms) {
            common_chat_msg m;
            try {
                m = common_chat_peg_parse(src_parser, common_chat_input(form.first + tail), false, q);
            } catch (const std::exception &) {
                continue;
            }
            if (m.tool_calls.empty() || !blank(m.content) || trim(m.reasoning_content) != form.second) {
                continue;
            }
            // prose after the calls must show: either the parse fails, or the prose comes back as content / reasoning
            // (and the test above then refuses a quoted call). A parser that silently DROPS trailing text would also
            // accept a quoted call followed by prose, so the rescue stays off for it.
            bool anchored = true;
            try {
                const common_chat_msg pm = common_chat_peg_parse(src_parser, common_chat_input(form.first + tail + "\n" + probe), false, q);
                anchored = pm.content.find(probe) != std::string::npos || pm.reasoning_content.find(probe) != std::string::npos;
            } catch (const std::exception &) {
            }
            if (!anchored) {
                // This parser drops text after the calls (the Qwen3-Coder handler and the --tool-call-format override
                // did, as upstream's common/parsers/qwen3-coder.cpp does, until both captured the tail as content), so
                // its result cannot tell a quoted call followed by prose from a real one. Then every line
                // from the candidate to the end must be markup: a line that starts with a letter or digit is prose (or a
                // bare value) and the candidate is refused, the safe direction.
                bool prose = false;
                for (size_t ln = 0; ln < tail.size() && !prose;) {
                    size_t eol = tail.find('\n', ln);
                    if (eol == std::string::npos) {
                        eol = tail.size();
                    }
                    const size_t f = tail.find_first_not_of(" \t\r", ln);
                    if (f != std::string::npos && f < eol) {
                        const unsigned char fc = (unsigned char) tail[f];
                        prose = fc < 0x80 && std::isalnum(fc);
                    }
                    ln = eol + 1;
                }
                if (prose) {
                    continue;
                }
            }
            size_t n = 0;
            for (auto & tc : m.tool_calls) {
                const bool declared = std::any_of(inner.tools.begin(), inner.tools.end(),
                                                  [&](const common_chat_tool & t) { return t.name == tc.name; });
                if (declared) {
                    msg.tool_calls.push_back(std::move(tc));
                    n++;
                }
            }
            if (n > 0) {
                common_chat_coerce_tool_args(msg, inner.tools);
                LOG_WRN("%s: %zu tool call(s) in the %s format written inside the reasoning and never closed were delivered "
                        "as tool calls\n", __func__, n, common_chat_format_name(inner.format));
            }
            return n;
        }
    }
    return 0;
}

// A parse that starts inside an open thinking block. Everything up to the first end-of-thinking tag is reasoning TEXT, whatever
// it contains; the format parser sees only a placeholder for it, so a quoted "<tool_call>" cannot end the reasoning or run.
// While the block is still open (no end tag yet) the whole text is reasoning, and a call written there is delivered only by
// the rescue (common_chat_rescue_reasoning_calls for the Qwen3-Coder XML call, else the template's own format) on the FINAL
// parse of a turn that ended on its own.
static common_chat_msg chat_parse_open_thinking(const common_peg_arena & src_parser, const std::string & input,
                                                bool is_partial, const common_chat_parser_params & params) {
    common_chat_parser_params inner = params;
    inner.gate_calls_in_reasoning   = false;
    inner.rescue_reasoning_calls    = false;

    const std::string lead = chat_reasoning_lead(src_parser, inner);

    // thinking_end_tags[0] is the tag that really closes the block; later entries are call openers that also end thinking
    // for the budget sampler (e.g. "<tool_call>") and are exactly what must NOT end it here. Bare tag: chat_thinking_close_tag.
    const std::string end_tag = chat_thinking_close_tag(params);

    // A turn that OPENS with a call opener the handler itself lists as ending the thinking block (upstream parsers/ling3.cpp: "a call
    // can be emitted before the think block is closed") has no reasoning text a quoted call could hide in, so the format parser
    // decides it, as upstream does (test-chat gate audit: inclusionai-ling-3.0-flash). While a stream has only a prefix of such an
    // opener, nothing is shown yet, so the message only ever grows.
    // Only when the handler itself states it (call_opener_ends_reasoning, parsers/ling3.cpp): the operator override lists its call
    // marker only so the budget sampler ends thinking there, and under the gate a quoted call followed by prose stays refused for it
    // (test-chat: "a format override that drops trailing text still refuses quotes").
    if (params.call_opener_ends_reasoning) {
        const size_t first = input.find_first_not_of(" \t\r\n");
        for (size_t k = 1; first != std::string::npos && k < params.thinking_end_tags.size(); ++k) {
            const std::string & opener = params.thinking_end_tags[k];
            if (opener.empty() || opener == end_tag) {
                continue;
            }
            if (input.compare(first, opener.size(), opener) == 0) {
                return common_chat_peg_parse(src_parser, common_chat_input(input), is_partial, inner);
            }
            if (is_partial && input.size() - first < opener.size() && opener.compare(0, input.size() - first, input, first) == 0) {
                common_chat_msg held;
                held.role = "assistant";
                return held;
            }
        }
    }

    const size_t pos = end_tag.empty() ? std::string::npos : input.find(end_tag);
    if (pos != std::string::npos) {
        common_chat_msg msg = common_chat_peg_parse(src_parser, common_chat_input("x" + input.substr(pos)), is_partial, inner);
        msg.reasoning_content = lead + input.substr(0, pos);
        return msg;
    }

    std::string text = input;
    if (is_partial && !end_tag.empty()) {
        // a cut-off end tag ("</thi") is not reasoning: hold it back until the next token shows what it is
        size_t hold = 0;
        for (size_t k = std::min(end_tag.size() - 1, text.size()); k > 0; k--) {
            if (text.compare(text.size() - k, k, end_tag, 0, k) == 0) {
                hold = k;
                break;
            }
        }
        text.resize(text.size() - hold);
    }
    common_chat_msg msg;
    msg.role              = "assistant";
    msg.reasoning_content = lead + text;
    if (!is_partial && params.rescue_reasoning_calls) {
        if (common_chat_rescue_reasoning_calls(msg, params.tools) == 0) {
            chat_rescue_native_format_calls(src_parser, text, inner, msg);
        }
    }
    return msg;
}

// A tool call the text ended inside: its own closing marker (or the rest of its arguments) never arrived. The parse is
// lenient, so the input running out inside a call is NEED_MORE_INPUT, not a failure (common/peg-parser.cpp, the
// literal parser), and the mappers emit a call as soon as its name is parsed, with the arguments as far as they got.
// Only the LAST call can be cut: nothing after the cut point was parsed. Removes that call from msg when its arguments
// are not yet a complete JSON value, and returns true.
//
// Why the arguments and not the AST: a sequence stops at its first NEED_MORE child and drops that child's nodes, so a
// call cut just before its closer has no partial tool-tagged node at all, only partial RULE nodes (Seed-OSS, GLM,
// MiniMax-M2 measured with the AST dump); and a call whose closer is optional (Kimi-K2: tool_close(optional(...)))
// sits in a partial "tool" node for one step after its JSON is already whole, so an AST rule withdrew a call a later
// stream step brought back (compute_diffs: "now finding less tool calls"). The arguments say what matters to the
// client: tagged formats get their closing brace only at the call's own close (common_chat_peg_mapper, is_tool_close),
// JSON formats are whole once their object closes, and anything short of that is not something a client can run.
// Empty arguments are a call cut before its arguments began (a whole call has at least "{}" in every PEG mapper).
static bool chat_drop_incomplete_tool_call(common_chat_msg & msg) {
    if (msg.tool_calls.empty()) {
        return false;
    }
    const std::string & args = msg.tool_calls.back().arguments;
    if (!args.empty() && !json::parse_no_throw(args).is_discarded()) {
        return false;
    }
    msg.tool_calls.pop_back();
    return true;
}

// block_level_thinking_end: which occurrences of the end-of-thinking tag, in front of the first real one, only MENTION it.
// Returns `input` with each mention rewritten to a sentinel (the tag with a unit separator after its first character) that
// no parser matches, so the first tag left over is the real end; common_chat_unmask_thinking_end puts the tag back.
// An occurrence is a mention when it touches a backtick, sits in a fenced block that is still open ("```" opened an odd
// number of times before it), or is followed on its line by more text (not a line break and not "<"). An occurrence at the
// very end of the text has nothing after it yet and counts as the real end: a stream that later turns out to continue
// moves it into the reasoning, and the reasoning stays a prefix-extension of what was sent.
static const char * const CHAT_THINKING_END_SENTINEL_MARK = "\x1d";

static std::string chat_mask_mentioned_thinking_end(const std::string & input, const std::string & tag) {
    if (tag.empty() || input.find(tag) == std::string::npos) {
        return input;
    }
    const std::string sentinel = tag.substr(0, 1) + CHAT_THINKING_END_SENTINEL_MARK + tag.substr(1);
    std::string       out;
    size_t            from = 0;
    for (size_t pos = input.find(tag); pos != std::string::npos; pos = input.find(tag, from)) {
        const size_t after = pos + tag.size();
        size_t fences = 0;
        for (size_t f = input.find("```"); f != std::string::npos && f + 3 <= pos; f = input.find("```", f + 3)) {
            fences++;
        }
        bool mention = (pos > 0 && input[pos - 1] == '`') || (after < input.size() && input[after] == '`') || (fences % 2 == 1);
        if (!mention) {
            size_t q = after;
            while (q < input.size() && (input[q] == ' ' || input[q] == '\t')) {
                q++;
            }
            mention = q < input.size() && input[q] != '\n' && input[q] != '\r' && input[q] != '<';
        }
        if (!mention) {
            break; // the real end of the thinking block
        }
        out.append(input, from, pos - from);
        out += sentinel;
        from = after;
    }
    out.append(input, from, std::string::npos);
    return out;
}

static void chat_unmask_thinking_end(common_chat_msg & msg, const std::string & tag) {
    const std::string sentinel = tag.substr(0, 1) + CHAT_THINKING_END_SENTINEL_MARK + tag.substr(1);
    string_replace_all(msg.content, sentinel, tag);
    string_replace_all(msg.reasoning_content, sentinel, tag);
    for (auto & part : msg.content_parts) {
        string_replace_all(part.text, sentinel, tag);
    }
    for (auto & tc : msg.tool_calls) {
        string_replace_all(tc.name, sentinel, tag);
        string_replace_all(tc.arguments, sentinel, tag);
    }
}

bool common_chat_is_empty_thinking_reply(const common_chat_msg & msg, const common_chat_parser_params & params) {
    return params.error_on_empty_thinking_reply && common_chat_generation_opens_thinking(params) &&
           params.reasoning_format != COMMON_REASONING_FORMAT_NONE && !params.reasoning_in_content &&
           msg.content.find_first_not_of(" \t\r\n") == std::string::npos && msg.content_parts.empty() && msg.tool_calls.empty();
}

common_chat_msg common_chat_peg_parse(const common_peg_arena &          src_parser,
                                      const common_chat_input &         raw_input,
                                      bool                              is_partial,
                                      const common_chat_parser_params & params) {
    // repair_malformed_calls: parse strictly first; only a parse that threw, found no call, or read a call into a corrupt
    // argument is offered rebuilt candidates (common/chat-malformed-repair.cpp). The strict result stands otherwise.
    if (params.repair_malformed_calls && !is_partial && !params.tools.empty() && params.parse_tool_calls) {
        common_chat_parser_params strict = params;
        strict.repair_malformed_calls    = false;
        common_chat_msg    strict_msg;
        std::exception_ptr strict_err;
        try {
            strict_msg = common_chat_peg_parse(src_parser, raw_input, false, strict);
        } catch (...) {
            strict_err = std::current_exception();
        }
        if (strict_err || strict_msg.tool_calls.empty() || common_chat_calls_look_corrupt(strict_msg)) {
            common_chat_parser_params probe = strict;
            probe.quiet_unparsed            = true;
            for (const std::string & cand : common_chat_malformed_call_candidates(params.format, raw_input.text, params.tools)) {
                try {
                    common_chat_msg m = common_chat_peg_parse(src_parser, common_chat_input(cand), false, probe);
                    if (!m.tool_calls.empty() && !common_chat_calls_look_corrupt(m)) {
                        LOG_WRN("%s: rebuilt a malformed %s tool call (%zu call(s) returned)\n", __func__,
                                common_chat_format_name(params.format), m.tool_calls.size());
                        return m;
                    }
                } catch (const std::exception &) {
                    // this candidate did not parse either: try the next, then the strict result
                }
            }
        }
        if (strict_err) {
            std::rethrow_exception(strict_err);
        }
        return strict_msg;
    }

    // Never let a model's invalid UTF-8 turn into an exception (HTTP 500) or into invalid JSON downstream.
    // a repaired text is parsed as plain text: its per-byte tokens no longer line up once bytes are replaced
    std::string utf8_clean;
    size_t      n_utf8_replaced = 0;
    const bool  utf8_repaired   = chat_utf8_repair(raw_input.text, /* keep_incomplete_tail = */ is_partial, utf8_clean, n_utf8_replaced);
    const common_chat_input input = utf8_repaired ? common_chat_input(utf8_clean) : raw_input;
    if (utf8_repaired && !is_partial) {
        LOG_WRN("%s: replaced %zu invalid UTF-8 sequence(s) in the model output with U+FFFD\n", __func__, n_utf8_replaced);
    }

    // the end tag written as a mention is not the end of the thinking block (block_level_thinking_end)
    if (params.block_level_thinking_end && !chat_thinking_close_tag(params).empty() && common_chat_generation_opens_thinking(params)) {
        const std::string   tag    = chat_thinking_close_tag(params);
        const std::string   masked = chat_mask_mentioned_thinking_end(input.text, tag);
        if (masked != input.text) {
            common_chat_parser_params inner = params;
            inner.block_level_thinking_end  = false;
            // a same-length mask keeps the per-byte tokens aligned; otherwise the masked text is parsed as plain text
            common_chat_input masked_in = input;
            if (masked.size() == input.text.size()) {
                masked_in.text = masked;
            } else {
                masked_in = common_chat_input(masked);
            }
            common_chat_msg msg = common_chat_peg_parse(src_parser, masked_in, is_partial, inner);
            chat_unmask_thinking_end(msg, tag);
            return msg;
        }
    }

    if (chat_thinking_left_open(params)) {
        return chat_parse_open_thinking(src_parser, input.text, is_partial, params);
    }

    // both branches must be lvalues, a temporary here would copy the arena on every call
    static const common_peg_arena content_only =
        build_chat_peg_parser([](common_chat_peg_builder & p) { return p.content(p.rest()) + p.end(); });
    const common_peg_arena & parser = src_parser.empty() ? content_only : src_parser;

    if (src_parser.empty()) {
        LOG_DBG("No parser definition detected, assuming pure content parser.");
    }

    common_chat_input effective_input = input;
    effective_input.prepend(params.generation_prompt);

    //LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(params.format), effective_input.text.c_str());

    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
    if (params.debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }

    common_peg_parse_context ctx(std::move(effective_input.text), std::move(effective_input.tokens), flags);
    auto result = parser.parse(ctx);

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial && result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            common_chat_msg msg;
            msg.role = "assistant";
            std::unique_ptr<common_chat_peg_mapper> mapper;
            if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4 || params.format == COMMON_CHAT_FORMAT_PEG_FUNCTIONGEMMA) {
                mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
            } else if (params.format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
                mapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(msg);
            } else {
                mapper = std::make_unique<common_chat_peg_mapper>(msg);
            }
            mapper->from_ast(ctx.ast, result);
            if (params.drop_incomplete_tool_calls) {
                chat_drop_incomplete_tool_call(msg);
            }

            if (ctx.is_debug()) {
                fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", ctx.ast.dump().c_str());
                fflush(stderr);
            }
            return msg;
        }
        if (!params.quiet_unparsed) {
            LOG_WRN("%s: unparsed %s output: %s\n", __func__, common_chat_format_name(params.format), ctx.input.substr(result.end).c_str());
        }
        LOG_DBG("%s: full %s output triggering error:\n=== BEGIN ===\n%s\n=== END ===\n", __func__, common_chat_format_name(params.format), ctx.input.c_str());
        throw std::runtime_error(std::string("The model produced output that does not match the expected ") + common_chat_format_name(params.format) + " format");
    }

    common_chat_msg msg;
    msg.role = "assistant";

    std::unique_ptr<common_chat_peg_mapper> mapper;
    if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4 || params.format == COMMON_CHAT_FORMAT_PEG_FUNCTIONGEMMA) {
        mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
    } else if (params.format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
        mapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(msg);
    } else {
        mapper = std::make_unique<common_chat_peg_mapper>(msg);
    }
    mapper->from_ast(ctx.ast, result);
    if (params.drop_incomplete_tool_calls && chat_drop_incomplete_tool_call(msg) && !is_partial) {
        // the turn stopped (max_tokens, context, deadline, ...) inside this call: it is not returned as a call to run
        LOG_WRN("%s: a %s tool call cut off before its end was withheld (it would have run with truncated arguments)\n",
                __func__, common_chat_format_name(params.format));
    }

    if (ctx.is_debug()) {
        fprintf(stderr, "\nAST for %s parse:\n%s\n", is_partial ? "partial" : "full", ctx.ast.dump().c_str());
        fflush(stderr);
    }

    if (!is_partial) {
        // only on the final parse: streamed prefixes of the arguments have already left the building
        if (params.coerce_tool_args && !params.tools.empty() && !msg.tool_calls.empty()) {
            common_chat_coerce_tool_args(msg, params.tools);
        }
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({ msg }).at(0).dump().c_str());
    }
    return msg;
}

common_chat_session::common_chat_session(const common_chat_templates *        tmpls,
                                         const llama_vocab *                  vocab,
                                         const common_chat_templates_inputs & inputs,
                                         const common_chat_session_params &   params) {
    auto applied = common_chat_templates_apply(tmpls, inputs);

    templated    = true;
    prompt_parts = std::move(applied.prompt);
    result.role  = "assistant";

    grammar_text           = std::move(applied.grammar);
    grammar_lazy           = applied.grammar_lazy;
    stops                  = std::move(applied.additional_stops);
    generation_prompt_text = applied.generation_prompt;
    thinking_start         = std::move(applied.thinking_start_tag);
    thinking_ends          = std::move(applied.thinking_end_tags);
    parser_params.call_opener_ends_reasoning = applied.call_opener_ends_reasoning;

    parser_params.format            = applied.format;
    parser_params.generation_prompt = vocab ? common_chat_input_tokenize(vocab, applied.generation_prompt)
                                            : common_chat_input(applied.generation_prompt);
    parser_params.debug             = params.debug;
    parser_params.parser            = std::move(applied.parser);
    // the fork's parser reads the thinking tags (open-thinking parse, block_level_thinking_end, the reasoning-call gate) and the
    // continuation flags; the session holds them (moved into thinking_start / thinking_ends above), so its parser gets them too
    parser_params.thinking_start_tag = thinking_start;
    parser_params.thinking_end_tags  = thinking_ends;
    parser_params.is_continuation    = inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE;
    parser_params.echo               = params.echo;

    delimiters = std::move(applied.message_delimiters);

    if (vocab) {
        common_params_sampling resolved;
        resolved.grammar_lazy = applied.grammar_lazy;
        common_sampling_add_preserved_tokens(resolved, vocab, applied.preserved_tokens);
        common_sampling_add_grammar_triggers(resolved, vocab, std::move(applied.grammar_triggers));
        preserved_tokens = std::move(resolved.preserved_tokens);
        grammar_triggers = std::move(resolved.grammar_triggers);

        delimiters.tokenize(vocab);
    } else {
        grammar_triggers = std::move(applied.grammar_triggers);
    }

    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !params.echo) {
        // start from the prefill so it is not emitted as part of the first delta
        result = common_chat_parse(input, true, parser_params);
    }
}

void common_chat_session::apply_sampling(common_params_sampling & sampling) const {
    if (!templated) {
        return;
    }
    if (!grammar_text.empty()) {
        sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, grammar_text};
    }
    sampling.grammar_lazy      = grammar_lazy;
    sampling.generation_prompt = generation_prompt_text;
    sampling.preserved_tokens.insert(preserved_tokens.begin(), preserved_tokens.end());
    sampling.grammar_triggers.insert(sampling.grammar_triggers.end(), grammar_triggers.begin(), grammar_triggers.end());
}

void common_chat_session::apply_parser_options(const common_chat_parser_params & opts) {
    parser_params.reasoning_format              = opts.reasoning_format;
    parser_params.reasoning_in_content          = opts.reasoning_in_content;
    parser_params.parse_tool_calls              = opts.parse_tool_calls;
    parser_params.debug                         = parser_params.debug || opts.debug;
    parser_params.coerce_tool_args              = opts.coerce_tool_args;
    parser_params.rescue_reasoning_calls        = opts.rescue_reasoning_calls;
    parser_params.gate_calls_in_reasoning       = opts.gate_calls_in_reasoning;
    parser_params.quiet_unparsed                = opts.quiet_unparsed;
    parser_params.repair_malformed_calls        = opts.repair_malformed_calls;
    parser_params.drop_incomplete_tool_calls    = opts.drop_incomplete_tool_calls;
    parser_params.block_level_thinking_end      = opts.block_level_thinking_end;
    parser_params.error_on_empty_thinking_reply = opts.error_on_empty_thinking_reply;
    parser_params.tools                         = opts.tools;
    if (!templated) {
        // no chat template (a raw /completion): the parse configuration is the request's own, as before the session
        parser_params.format             = opts.format;
        parser_params.generation_prompt  = opts.generation_prompt;
        parser_params.thinking_start_tag = opts.thinking_start_tag;
        parser_params.thinking_end_tags  = opts.thinking_end_tags;
        parser_params.is_continuation    = opts.is_continuation;
        parser_params.echo               = opts.echo;
        parser_params.parser             = opts.parser;
    }
}

const common_chat_msg & common_chat_session::feed(const common_chat_input & chunk) {
    GGML_ASSERT(!finished && "feed() after finish()");
    input.append(chunk);
    auto msg = common_chat_parse(input, true, parser_params);
    if (!msg.empty()) {
        result = std::move(msg);
    }
    return result;
}

const common_chat_msg & common_chat_session::finish(const common_chat_input & chunk) {
    GGML_ASSERT(!finished && "finish() called twice");
    finished = true;
    input.append(chunk);
    auto msg = common_chat_parse(input, false, parser_params);
    if (!msg.empty()) {
        result = std::move(msg);
    }
    return result;
}

std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates) {
    GGML_ASSERT(chat_templates != nullptr);
    GGML_ASSERT(chat_templates->template_default != nullptr);
    if (chat_templates->template_tool_use != nullptr) {
        // take the more expressive template when available
        return chat_templates->template_tool_use->caps.to_map();
    }
    return chat_templates->template_default->caps.to_map();
}
