#include "server-chat.h"
#include "server-reasoning-blob.h"
#include "server-common.h"
#include "server-asr-format.h"
#include "server-asr-prompt.h"

#include <algorithm>
#include <sstream>

static bool exists_and_is_array_msg(const json & j, const char * key) {
    return j.contains(key) && j.at(key).is_array();
}

json server_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list

        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (exists_and_is_string(item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                // #responses_create-input-input_item_list-item-input_message
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                // #responses_create-input-input_item_list-item-output_message
                auto chatcmpl_content = json::array();

                // Handle both string content and array content
                if (item.contains("content") && item.at("content").is_string()) {
                    // String content - convert to text content part
                    chatcmpl_content.push_back({
                        {"text", item.at("content")},
                        {"type", "text"},
                    });
                } else if (exists_and_is_array(item, "content")) {
                    // Array content - process each item
                    for (const auto & output_text : item.at("content")) {
                        const std::string type = json_value(output_text, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!exists_and_is_string(output_text, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            chatcmpl_content.push_back({
                                {"refusal", output_text.at("refusal")},
                                {"type", "refusal"},
                            });
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(chatcmpl_content);
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                ((item.at("type") == "function_call" && item.contains("arguments") &&
                    (item.at("arguments").is_string() || item.at("arguments").is_object() || item.at("arguments").is_array())) ||
                 item.at("type") == "custom_tool_call")
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call
                // Codex also sends custom_tool_call (free-form `input`) and, for tools of a namespace, a `namespace` field.
                // The model saw those tools as `namespace.name` and as {"input": "<text>"} (see the tools conversion below), so the
                // history is written back the same way. Arguments that arrive as a JSON value instead of a string are kept as text.
                std::string call_name = item.at("name");
                if (exists_and_is_string(item, "namespace") && !item.at("namespace").get<std::string>().empty()) {
                    call_name = item.at("namespace").get<std::string>() + "." + call_name;
                }
                std::string call_args;
                if (item.at("type") == "custom_tool_call") {
                    call_args = json {{"input", exists_and_is_string(item, "input") ? item.at("input") : json("")}}.dump();
                } else if (item.at("arguments").is_string()) {
                    call_args = item.at("arguments").get<std::string>();
                } else {
                    call_args = item.at("arguments").dump();
                }
                json tool_call = {
                    {"function", json {
                        {"arguments", call_args},
                        {"name",      call_name},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                (item.at("type") == "function_call_output" || item.at("type") == "custom_tool_call_output")
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type")) {
                            throw std::invalid_argument("Output of tool call missing 'type' field");
                        }
                        const auto type = chatcmpl_output.at("type");
                        if (type != "input_text" && type != "input_image") {
                            throw std::invalid_argument("Output of tool call should be 'Input text' or 'Input image'");
                        }
                        if (type == "input_text") {
                            chatcmpl_output["type"] = "text";
                        } else if (type == "input_image") {
                            if (!chatcmpl_output.contains("image_url")) {
                                throw std::invalid_argument("'image_url' is required");
                            }
                            chatcmpl_output["type"] = "image_url";
                            chatcmpl_output["image_url"] = json {
                                {"url", chatcmpl_output.at("image_url")}
                            };
                        }
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning

                // HARNESS-CONSIDERED: reasoning text is still replayed whenever the item has any; only an item with nothing readable
                // (a summary only, or an opaque encrypted_content from another server) is skipped instead of failing the whole
                // conversation with a 400, which lost the entire turn. An item with no readable text but with a blob this
                // server issued (include=["reasoning.encrypted_content"], server-reasoning-blob.h) is replayed from the blob.
                std::string reasoning_text;
                if (exists_and_is_array(item, "content") && !item.at("content").empty() &&
                    exists_and_is_string(item.at("content")[0], "text")) {
                    reasoning_text = item.at("content")[0].at("text").get<std::string>();
                }
                if (reasoning_text.empty() && exists_and_is_string(item, "encrypted_content") &&
                    !server_reasoning_blob::decode(item.at("encrypted_content").get<std::string>(), reasoning_text)) {
                    reasoning_text.clear();
                }
                if (reasoning_text.empty()) {
                    continue;
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = reasoning_text;
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", reasoning_text},
                    });
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    // A client may send the results of parallel calls in the order they finished; the template places a run of tool
    // messages by position, so each run is put back in the order of the calls of the assistant turn before it.
    for (size_t i = 0; i < chatcmpl_messages.size();) {
        if (chatcmpl_messages[i].value("role", "") != "tool") {
            i++;
            continue;
        }
        size_t j = i;
        while (j < chatcmpl_messages.size() && chatcmpl_messages[j].value("role", "") == "tool") {
            j++;
        }
        if (i > 0 && j - i > 1 && exists_and_is_array_msg(chatcmpl_messages[i - 1], "tool_calls")) {
            std::vector<std::string> order;
            for (const auto & tc : chatcmpl_messages[i - 1].at("tool_calls")) {
                order.push_back(json_value(tc, "id", std::string()));
            }
            auto rank = [&order](const json & m) {
                const std::string id = json_value(m, "tool_call_id", std::string());
                for (size_t k = 0; k < order.size(); k++) {
                    if (order[k] == id) {
                        return k;
                    }
                }
                return order.size();
            };
            std::stable_sort(chatcmpl_messages.begin() + i, chatcmpl_messages.begin() + j,
                [&rank](const json & a, const json & b) { return rank(a) < rank(b); });
        }
        i = j;
    }

    // Codex sends `instructions` and then its own developer message: several leading system/developer messages become one,
    // so the start of the prompt does not move when one of them changes (and the prompt cache holds across turns). Only
    // text-only ones are merged.
    {
        auto lead_text = [](const json & m, std::string & out) -> bool {
            const std::string role = m.value("role", "");
            if ((role != "system" && role != "developer") || !m.contains("content")) {
                return false;
            }
            const json & c = m.at("content");
            if (c.is_string()) {
                out = c.get<std::string>();
                return true;
            }
            if (!c.is_array()) {
                return false;
            }
            out.clear();
            for (const auto & part : c) {
                if (json_value(part, "type", std::string()) != "text" || !part.contains("text") || !part.at("text").is_string()) {
                    return false;
                }
                out += (out.empty() ? "" : "\n") + part.at("text").get<std::string>();
            }
            return true;
        };
        size_t lead = 0;
        std::string merged;
        while (lead < chatcmpl_messages.size()) {
            std::string t;
            if (!lead_text(chatcmpl_messages[lead], t)) {
                break;
            }
            merged += (lead == 0 ? "" : "\n\n") + t;
            lead++;
        }
        if (lead > 1) {
            chatcmpl_messages.erase(chatcmpl_messages.begin() + 1, chatcmpl_messages.begin() + lead);
            chatcmpl_messages[0] = json {{"role", "system"}, {"content", merged}};
        }
    }

    chatcmpl_body["messages"] = chatcmpl_messages;

    if (response_body.contains("tools")) {
        if (!response_body.at("tools").is_array()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        std::vector<json> chatcmpl_tools;
        // flat tool name the model sees -> how to report its call back to the client (namespace, original name, custom or not)
        json tool_map = json::object();
        bool tool_map_needed = false;

        // one Responses function/custom tool -> one chat-completions function tool named `namespace.name` (or `name`)
        auto add_tool = [&](json resp_tool, const std::string & ns, const std::string & ns_description) {
            const std::string type = json_value(resp_tool, "type", std::string("function"));
            const std::string name = json_value(resp_tool, "name", std::string());
            if (name.empty()) {
                throw std::invalid_argument("a tool needs a 'name'");
            }
            const std::string flat = ns.empty() ? name : ns + "." + name;
            std::string description = json_value(resp_tool, "description", std::string());
            if (!ns_description.empty()) {
                description = ns_description + (description.empty() ? "" : "\n\n" + description);
            }
            json function = json::object();
            if (type == "custom") {
                // free-form text input; a grammar the client gives is shown to the model as text (it is not enforced)
                if (resp_tool.contains("format") && resp_tool.at("format").is_object()) {
                    const json & format = resp_tool.at("format");
                    if (json_value(format, "type", std::string()) == "grammar" && format.contains("definition")) {
                        description += "\n\nThe input must follow this " + json_value(format, "syntax", std::string()) +
                                       " grammar:\n" + json_value(format, "definition", std::string());
                    }
                }
                function = json {
                    {"name",        flat},
                    {"description", description},
                    {"parameters",  json {
                        {"type",       "object"},
                        {"properties", json {{"input", json {{"type", "string"}, {"description", "the raw input"}}}}},
                        {"required",   json::array({"input"})},
                    }},
                };
            } else {
                resp_tool.erase("type");
                resp_tool.erase("namespace");
                resp_tool["name"] = flat;
                resp_tool["description"] = description;
                if (!resp_tool.contains("strict")) {
                    resp_tool["strict"] = true;
                }
                function = resp_tool;
            }
            chatcmpl_tools.push_back(json {{"type", "function"}, {"function", function}});

            const json entry = json {{"namespace", ns}, {"name", name}, {"custom", type == "custom"}};
            tool_map[flat] = entry;
            if (!ns.empty()) {
                // models trained on Codex's MCP tools write `namespace__name`; map that spelling back as well
                if (!tool_map.contains(ns + "__" + name)) {
                    tool_map[ns + "__" + name] = entry;
                }
            }
            tool_map_needed = tool_map_needed || !ns.empty() || type == "custom";
        };

        for (const json & resp_tool : response_body.at("tools")) {
            const std::string type = json_value(resp_tool, "type", std::string());
            if (type == "function" || type == "custom") {
                add_tool(resp_tool, "", "");
            } else if (type == "namespace") {
                const std::string ns = json_value(resp_tool, "name", std::string());
                if (ns.empty() || !resp_tool.contains("tools") || !resp_tool.at("tools").is_array()) {
                    // a namespace with no member tools offers nothing to call
                    SRV_WRN("namespace tool '%s' has no 'tools' array, skipped\n", ns.c_str());
                    continue;
                }
                for (json member : resp_tool.at("tools")) {
                    const std::string mtype = json_value(member, "type", std::string("function"));
                    if (mtype != "function" && mtype != "custom") {
                        throw std::invalid_argument("a 'namespace' tool holds function or custom tools");
                    }
                    member["type"] = mtype;
                    add_tool(member, ns, json_value(resp_tool, "description", std::string()));
                }
            } else {
                // Hosted Responses tools (web search, file search, ...) cannot be run by this server.
                SRV_WRN("unsupported Responses tool type '%s' skipped\n", type.c_str());
            }
        }
        chatcmpl_body.erase("tools");
        if (!chatcmpl_tools.empty()) {
            chatcmpl_body["tools"] = chatcmpl_tools;
        }
        if (tool_map_needed) {
            chatcmpl_body["responses_tool_map"] = tool_map;
        }
    }

    // Responses forces one tool as the flat {"type":"function","name":X}; Chat Completions wants it nested under "function".
    if (response_body.contains("tool_choice") && response_body.at("tool_choice").is_object()) {
        const json & tc = response_body.at("tool_choice");
        if (tc.contains("name") && tc.at("name").is_string() && !tc.contains("function")) {
            json nested = json::object();
            nested["type"]     = "function";
            nested["function"] = json::object();
            nested["function"]["name"] = tc.at("name").get<std::string>();
            chatcmpl_body["tool_choice"] = nested;
        }
    }

    // include=["reasoning.encrypted_content"]: reasoning items carry a blob the client can send back (server-reasoning-blob.h)
    if (response_body.contains("include") && response_body.at("include").is_array()) {
        for (const auto & inc : response_body.at("include")) {
            if (inc.is_string() && inc.get<std::string>() == server_reasoning_blob::include_key()) {
                chatcmpl_body["responses_reasoning_blob"] = true;
            }
        }
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    if (response_body.contains("reasoning")) {
        // Only "effort" is handled so far
        const json & reasoning = response_body.at("reasoning");
        if (reasoning.contains("effort")) {
            chatcmpl_body["reasoning_effort"] = reasoning.at("effort");
        }
        // reasoning.summary ("auto"/"concise"/"detailed"): the reasoning text is streamed as summary events
        // (SGLang serving_responses.py:2076-2124 spec; a Codex client asks for it on every turn)
        if (reasoning.contains("summary") && reasoning.at("summary").is_string()) {
            const std::string s = reasoning.at("summary").get<std::string>();
            if (!s.empty() && s != "none") {
                chatcmpl_body["responses_reasoning_summary"] = true;
            }
        }
        chatcmpl_body.erase("reasoning");
    }

    // llama.cpp params (e.g. seed, min_p) go under extra_params and are merged into the converted body
    if (response_body.contains("extra_params")) {
        chatcmpl_body.erase("extra_params");
        if (response_body.at("extra_params").is_object()) {
            for (const auto & [key, value] : response_body.at("extra_params").items()) {
                chatcmpl_body[key] = value;
            }
        }
    }

    return chatcmpl_body;
}

// Edits the cch section of an "x-anthropic-billing-header" system prompt.
// Does nothing to any other prompt.
//
// This is a claude message with a "cch=ef01a" attribute that breaks prefix caching.
// The cch stamp is a whitebox end-to-end integrity hint. It's not meaningful as a
// system prompt data, particularly to llama.cpp, but its presence means the prefix
// cache will not get past it: It changes on each request.
//
// Reference: https://github.com/ggml-org/llama.cpp/pull/21793
// Example header:
// ```
// x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code, Anthropic's official CLI for Claude.
//                                                                            ^^^^^
// ```
static void normalize_anthropic_billing_header(std::string & system_text) {
    if (system_text.rfind("x-anthropic-billing-header:", 0) != 0) {
        return;
    }

    const size_t header_prefix_length = strlen("x-anthropic-billing-header:");
    const size_t cch_length = 5;
    const size_t index_cch = system_text.find("cch=", header_prefix_length);
    if (index_cch == std::string::npos) {
        return;
    }

    const size_t index_replace = index_cch + 4;
    if (index_replace + cch_length < system_text.length() && system_text[index_replace + cch_length] == ';') {
        for (size_t i = 0; i < cch_length; ++i) {
            system_text[index_replace + i] = 'f';
        }
    } else {
        LOG_ERR("anthropic string not as expected: %s", system_text.c_str());
    }
}

json server_chat_convert_anthropic_to_oai(const json & body) {
    json oai_body;

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
            normalize_anthropic_billing_header(system_content);
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    auto system_text = json_value(block, "text", std::string());
                    normalize_anthropic_billing_header(system_text);
                    system_content += system_text;
                }
            }
        }

        oai_messages.push_back({
            {"role", "system"},
            {"content", system_content}
        });
    }

    // Convert messages
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    const json & messages = body.at("messages");
    if (messages.is_array()) {
        for (const auto & msg : messages) {
            std::string role = json_value(msg, "role", std::string());

            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                oai_messages.push_back(msg);
                continue;
            }

            const json & content = msg.at("content");

            if (content.is_string()) {
                oai_messages.push_back(msg);
                continue;
            }

            if (!content.is_array()) {
                oai_messages.push_back(msg);
                continue;
            }

            json tool_calls = json::array();
            json converted_content = json::array();
            json tool_results = json::array();
            std::string reasoning_content;
            bool has_tool_calls = false;

            for (const auto & block : content) {
                std::string type = json_value(block, "type", std::string());

                if (type == "text") {
                    converted_content.push_back(block);
                } else if (type == "thinking") {
                    reasoning_content += json_value(block, "thinking", std::string());
                } else if (type == "image") {
                    json source = json_value(block, "source", json::object());
                    std::string source_type = json_value(source, "type", std::string());

                    if (source_type == "base64") {
                        std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                        std::string data = json_value(source, "data", std::string());
                        std::ostringstream ss;
                        ss << "data:" << media_type << ";base64," << data;

                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", ss.str()}
                            }}
                        });
                    } else if (source_type == "url") {
                        std::string url = json_value(source, "url", std::string());
                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", url}
                            }}
                        });
                    }
                } else if (type == "tool_use") {
                    tool_calls.push_back({
                        {"id", json_value(block, "id", std::string())},
                        {"type", "function"},
                        {"function", {
                            {"name", json_value(block, "name", std::string())},
                            {"arguments", json_value(block, "input", json::object()).dump()}
                        }}
                    });
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                    auto result_content = json_value(block, "content", json());
                    if (result_content.is_string()) {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", result_content.get<std::string>()}
                        });
                    } else if (result_content.is_array()) {
                        // Single-pass: build both text and content_parts, decide format at the end
                        std::string result_text;
                        json content_parts = json::array();
                        bool has_images = false;

                        for (const auto & c : result_content) {
                            std::string c_type = json_value(c, "type", std::string());
                            if (c_type == "text") {
                                std::string text = json_value(c, "text", std::string());
                                result_text += text;
                                content_parts.push_back({
                                    {"type", "text"},
                                    {"text", text}
                                });
                            } else if (c_type == "image") {
                                has_images = true;
                                json source = json_value(c, "source", json::object());
                                std::string source_type = json_value(source, "type", std::string());
                                if (source_type == "base64") {
                                    std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                                    std::string data = json_value(source, "data", std::string());
                                    std::string url = "data:" + media_type + ";base64," + data;
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", url}}}
                                    });
                                } else if (source_type == "url") {
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", json_value(source, "url", std::string())}}}
                                    });
                                }
                            }
                        }

                        if (!has_images) {
                            // Text-only: collapse to a plain string for maximum compatibility
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", result_text}
                            });
                        } else {
                            // Mixed or image-only: use array content parts (OpenAI multimodal tool format)
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", content_parts}
                            });
                        }
                    } else {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", ""}
                        });
                    }
                }
            }

            if (!converted_content.empty() || has_tool_calls || !reasoning_content.empty()) {
                json new_msg = {{"role", role}};
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                } else if (has_tool_calls || !reasoning_content.empty()) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                if (!reasoning_content.empty()) {
                    new_msg["reasoning_content"] = reasoning_content;
                }
                oai_messages.push_back(new_msg);
            }

            for (const auto & tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
    }

    oai_body["messages"] = oai_messages;

    // Convert tools
    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (tools.is_array()) {
            json oai_tools = json::array();
            for (const auto & tool : tools) {
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                });
            }
            oai_body["tools"] = oai_tools;
        }
    }

    // Convert tool_choice. Anthropic {"type": "tool", "name": X} forces ONE tool; it used to become "required",
    // which let the model call ANY tool (upstream maps it the same way while its README documents the named form).
    // It now becomes OpenAI's named form, which oaicompat_chat_params_parse turns into "only X offered, a call
    // required". "none" is passed through; before, it fell to the default "auto".
    if (body.contains("tool_choice")) {
        const json & tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_body["tool_choice"] = "auto";
            } else if (type == "none") {
                oai_body["tool_choice"] = "none";
            } else if (type == "any") {
                oai_body["tool_choice"] = "required";
            } else if (type == "tool") {
                const std::string name = json_value(tc, "name", std::string());
                if (name.empty()) {
                    throw std::invalid_argument("tool_choice {\"type\": \"tool\"} requires \"name\"");
                }
                oai_body["tool_choice"] = json{{"type", "function"}, {"function", {{"name", name}}}};
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        oai_body["stop"] = body.at("stop_sequences");
    }

    // Handle max_tokens (required in Anthropic, but we're permissive)
    if (body.contains("max_tokens")) {
        oai_body["max_tokens"] = body.at("max_tokens");
    } else {
        oai_body["max_tokens"] = 4096;
    }

    // Pass through common params
    for (const auto & key : {"temperature", "top_p", "top_k", "stream", "chat_template_kwargs", "id_slot"}) {
        if (body.contains(key)) {
            oai_body[key] = body.at(key);
        }
    }

    // Handle Anthropic-specific thinking param
    if (body.contains("thinking")) {
        json thinking = json_value(body, "thinking", json::object());
        std::string thinking_type = json_value(thinking, "type", std::string());
        if (thinking_type == "enabled") {
            int budget_tokens = json_value(thinking, "budget_tokens", 10000);
            oai_body["thinking_budget_tokens"] = budget_tokens;
        }
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            oai_body["__metadata_user_id"] = user_id;
        }
    }

    // llama.cpp params (e.g. seed, min_p) go under extra_params and are merged into the converted body
    if (body.contains("extra_params") && body.at("extra_params").is_object()) {
        for (const auto & [key, value] : body.at("extra_params").items()) {
            oai_body[key] = value;
        }
    }

    return oai_body;
}

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({ tool_call });
    }
    return delta;
}

json convert_transcriptions_to_chatcmpl(
        const json & inp_body,
        const common_chat_templates * tmpls,
        const std::map<std::string, uploaded_file> & in_files,
        std::vector<raw_buffer> & out_files) {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clear();
    auto it = in_files.find("file");
    if (it != in_files.end()) {
        out_files.push_back(it->second.data);
    } else {
        throw std::invalid_argument("No input file found for transcription");
    }

    // handle input data
    std::string prompt          = json_value(inp_body, "prompt", std::string());
    std::string language        = json_value(inp_body, "language", std::string());
    std::string response_format = json_value(inp_body, "response_format", std::string("json"));
    {
        asrfmt::format fmt;
        if (!asrfmt::parse(response_format, fmt)) {
            throw std::invalid_argument("response_format '" + response_format + "' is not supported for transcription (json, text, srt, vtt, verbose_json)");
        }
        if (fmt != asrfmt::format::json && json_value(inp_body, "stream", std::string("false")) == "true") {
            throw std::invalid_argument("stream=true is only supported with response_format 'json'");
        }
    }
    const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls);
    prompt = asr_compose_user_prompt(preset.user, prompt, language);
    prompt += get_media_marker();

    // ALWAYS send a system message. Leaving it out does not mean "no system prompt": a template with a
    // default_system_message substitutes its own. Voxtral-Mini-3B-2507's GGUF ships an Unsloth DEVSTRAL template whose
    // default is a ~1.2k-token coding persona, so every transcription carried it and the reply came back with a preamble,
    // wrapper quotes, and on a 17 s clip a summary that dropped the opening. Measured by fork commit ec6520161 in
    // examples/server, which the build no longer compiles; ported here.
    json messages = json::array();
    messages.push_back({{"role", "system"}, {"content", preset.system}});
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json chatcmpl_body = inp_body; // copy all fields
    chatcmpl_body["messages"] = messages;

    // because input from form-data, everything is string, we need to correct the types here
    std::string stream = json_value(inp_body, "stream", std::string("false"));
    chatcmpl_body["stream"] = stream == "true";

    // text / srt / vtt / verbose_json are rendered from the finished transcript (server-context.cpp, post_transcriptions, via asrfmt::render).
    // The body below goes to the chat parser, which wants response_format to be an OBJECT (server-common.cpp: "response_format must be an
    // object"), so a transcription format string must not travel in it; the renderer reads the format from the original request body.
    {
        asrfmt::format fmt = asrfmt::format::json;
        asrfmt::parse(response_format, fmt);
        chatcmpl_body.erase("response_format"); // every value, json included: none of them is a chat response_format object
        if (fmt == asrfmt::format::verbose_json) {
            // avg_logprob of the verbose_json segments is the mean log-probability of the chosen tokens
            chatcmpl_body["logprobs"]     = true;
            chatcmpl_body["top_logprobs"] = 1;
        }
    }

    if (inp_body.contains("max_tokens")) {
        std::string inp = inp_body["max_tokens"].get<std::string>();
        chatcmpl_body["max_tokens"] = std::stoul(inp);
    }

    // OpenAI's transcription temperature defaults to 0 (the API reference: "Defaults to 0"). Without a value here the
    // request fell through to the server's general sampling default, so a request that omitted it was sampled while
    // verbose_json reported 0. A form field arrives as a string; a JSON number is taken as is.
    if (inp_body.contains("temperature")) {
        const json & t = inp_body.at("temperature");
        chatcmpl_body["temperature"] = t.is_number() ? t.get<float>() : std::stof(t.get<std::string>());
    } else {
        chatcmpl_body["temperature"] = 0.0f;
    }

    return chatcmpl_body;
}
