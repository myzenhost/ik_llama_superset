// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include "peg-parser.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"

#include "json.h"

#include <chrono>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

using chat_template_caps = jinja::caps;

struct common_chat_templates;

namespace autoparser {
struct generation_params;
struct autoparser;
}  // namespace autoparser

struct common_chat_params;
struct common_chat_template;

// Builds the prompt and parser for a template that has a dedicated handler (see common/parsers)
using common_chat_params_init_fn = common_chat_params (*)(const common_chat_template & tmpl,
                                                          const autoparser::generation_params & inputs);

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const common_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;

    // TODO @ngxson : no known chat templates support reasoning_content in content parts yet
    //                this can be useful for models with interleaved thinking (like Kimi-K2)
    //                if you see any templates explicitly support this, please ping me
    // std::string reasoning_content;

    bool operator==(const common_chat_msg_content_part & other) const {
        return type == other.type && text == other.text;
    }
};

struct common_chat_template {
    jinja::program prog;
    std::string bos_tok;
    std::string eos_tok;
    std::string src;
    chat_template_caps caps;
    // Dedicated handler picked once from the source, null when the differential autoparser is used
    common_chat_params_init_fn params_init = nullptr;

    // Differential analysis, run once here when there is no dedicated handler. Null when there
    // is one, or when the analysis failed, in which case analysis_error says why.
    std::unique_ptr<autoparser::autoparser> analysis;
    std::string analysis_error;

    common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token);

    // autoparser is incomplete here, so these are defined where it is complete
    ~common_chat_template();
    common_chat_template(common_chat_template &&);
    common_chat_template & operator=(common_chat_template &&);

    const std::string & source() const { return src; }
    const std::string & bos_token() const { return bos_tok; }
    const std::string & eos_token() const { return eos_tok; }

    chat_template_caps original_caps() const {
        return caps;
    }
};

struct common_chat_msg {
    std::string                               role;
    std::string                               content;
    std::vector<common_chat_msg_content_part> content_parts;
    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               tool_name;
    std::string                               tool_call_id;
    // non-OpenAI message keys as a JSON object string, only for templates that read them (DeepSeek V4.1)
    std::string                               extra_json;

    common_json to_json_oaicompat(bool concat_typed_text = false) const;

    std::string render_content(const std::string & delimiter = "\n\n") const;

    bool empty() const {
        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() &&
               tool_name.empty() && tool_call_id.empty();
    }

    bool contains_media() const {
        for (const auto & part : content_parts) {
            if (part.type == "media_marker") {
                return true;
            }
        }
        return false;
    }

    void set_tool_call_ids(std::vector<std::string> &           ids_cache,
                           const std::function<std::string()> & gen_tool_call_id) {
        for (auto i = 0u; i < tool_calls.size(); i++) {
            if (ids_cache.size() <= i) {
                auto id = tool_calls[i].id;
                if (id.empty()) {
                    id = gen_tool_call_id();
                }
                ids_cache.push_back(id);
            }
            tool_calls[i].id = ids_cache[i];
        }
    }

    bool operator==(const common_chat_msg & other) const {
        return role == other.role && content == other.content && content_parts == other.content_parts &&
               tool_calls == other.tool_calls && reasoning_content == other.reasoning_content &&
               tool_name == other.tool_name && tool_call_id == other.tool_call_id;
    }

    bool operator!=(const common_chat_msg & other) const { return !(*this == other); }
};

struct common_chat_msg_diff {
    std::string           reasoning_content_delta;
    std::string           content_delta;
    size_t                tool_call_index = std::string::npos;
    common_chat_tool_call tool_call_delta;

    static std::vector<common_chat_msg_diff> compute_diffs(const common_chat_msg & msg_prv,
                                                           const common_chat_msg & msg_new);

    bool operator==(const common_chat_msg_diff & other) const {
        return content_delta == other.content_delta && tool_call_index == other.tool_call_index &&
               tool_call_delta == other.tool_call_delta;
    }
};

enum common_chat_role {
    COMMON_CHAT_ROLE_UNKNOWN,
    COMMON_CHAT_ROLE_SYSTEM,
    COMMON_CHAT_ROLE_ASSISTANT,
    COMMON_CHAT_ROLE_USER,
    COMMON_CHAT_ROLE_TOOL
};

common_chat_role common_chat_role_from_string(const std::string & role);
const char *     common_chat_role_to_string(common_chat_role role);

struct common_chat_msg_span {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::size_t pos = 0;
    std::size_t len = 0;

    bool valid() const {
        return role != COMMON_CHAT_ROLE_UNKNOWN;
    }
};

struct common_chat_msg_spans {
    std::vector<common_chat_msg_span> spans;

    void add(common_chat_role role, size_t pos, size_t len) {
        spans.push_back({ role, pos, len });
    }

    bool is_user_start(int32_t pos) const {
        for (auto it = spans.begin(); it != spans.end(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER && pos == (int32_t) it->pos) {
                return true;
            }
        }
        return false;
    }

    int32_t last_user_message_pos() const {
        for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER) {
                return (int32_t) it->pos;
            }
        }
        return -1;
    }
};

struct common_chat_msg_delimiter {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::string      delimiter;
    llama_tokens     tokens = {};
};

struct common_chat_msg_delimiters {
    std::vector<common_chat_msg_delimiter> delimiters;

    common_chat_msg_delimiters() = default;
    common_chat_msg_delimiters(std::initializer_list<common_chat_msg_delimiter> delims) : delimiters(delims) {}

    void add(common_chat_role role, const std::string & delimiter) {
        delimiters.push_back({ role, delimiter });
    }

    void tokenize(const llama_vocab * vocab);

    // split tokens into message spans. skips maps a start index to a length of a region to jump over without matching
    common_chat_msg_spans split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips = {}) const;
};

// the request field "message_delimiters": [{"role": ..., "delimiter": ...}, ...] (fork, restored 2026-11-04: the merge had
// dropped the definition while llama-server still parses the field)
common_chat_msg_delimiters common_chat_msg_delimiters_parse(const common_json & delimiters);

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,

    // These are intended to be parsed by the PEG parser
    COMMON_CHAT_FORMAT_PEG_SIMPLE,
    COMMON_CHAT_FORMAT_PEG_NATIVE,
    COMMON_CHAT_FORMAT_PEG_GEMMA4,
    COMMON_CHAT_FORMAT_PEG_MINIMAX_M3,
    COMMON_CHAT_FORMAT_PEG_K2_HORIZON,
    COMMON_CHAT_FORMAT_PEG_FUNCTIONGEMMA,

    COMMON_CHAT_FORMAT_COUNT,  // Not a format, just the # formats
};


// Continuation method provided via `continue_final_message`
enum common_chat_continuation {
    COMMON_CHAT_CONTINUATION_NONE,
    COMMON_CHAT_CONTINUATION_AUTO,
    COMMON_CHAT_CONTINUATION_REASONING,
    COMMON_CHAT_CONTINUATION_CONTENT,
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg>          messages;
    std::string                           grammar;
    std::string                           json_schema;
    bool                                  add_generation_prompt  = true;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    bool                                  use_jinja              = true;
    // Parameters below only supported when use_jinja is true
    std::vector<common_chat_tool>         tools;
    common_chat_tool_choice               tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool                                  parallel_tool_calls = false;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool enable_thinking"
    bool                                  enable_thinking     = true;
    std::chrono::system_clock::time_point now                 = std::chrono::system_clock::now();
    std::map<std::string, std::string>    chat_template_kwargs;
    bool                                  add_bos = false;
    bool                                  add_eos = false;
    bool                                  force_pure_content = false;
    // Operator override of the XML-style tool-call format (--tool-call-format-json / --tool-call-format-preset): the JSON
    // text of a common_chat_xml_tool_format. When set and tools are offered, it replaces both the specialised handler and the
    // autoparser for this request. Validate it once with common_chat_xml_tool_format_parse (the CLI does).
    std::string                           tool_call_format;
};

// Shape of an XML-style tool call, as the operator states it:
//
//   [scope_start] tool_start NAME tool_sep ( key_start KEY key_val_sep [key_val_sep2] VALUE val_end )* tool_end ... [scope_end]
//
// e.g. MiniMax-M2  <minimax:tool_call><invoke name="NAME"><parameter name="KEY">VALUE</parameter></invoke></minimax:tool_call>
//      GLM-4.5     <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>   (key_val_sep2 = <arg_value>)
//
// Each field is a literal marker. Whitespace is skipped between structural tokens (after scope_start, after tool_sep, after
// val_end, after tool_end) but NOT between key_val_sep and the value: a newline that belongs there goes into the literal
// (qwen3-coder: key_val_sep ">\n", val_end "\n</parameter>"). Because the skip is greedy, whitespace at the edge of a marker is
// trimmed when the format is parsed, except the START of val_end and the END of key_val_sep / key_val_sep2. Each parameter the tool's schema marks required must appear exactly
// once, in any order; optional ones follow.
// A string-typed parameter takes the raw text up to val_end (raw_argval = false: a JSON string instead); any other type takes a
// JSON value checked against the parameter's schema. Several calls are accepted when the request allows parallel tool calls,
// inside one scope or in one scope each.
struct common_chat_xml_tool_format {
    std::string scope_start;   // optional wrapper opening; scope_start and scope_end are both set or both empty
    std::string tool_start;    // required
    std::string tool_sep;      // after the function name; may be empty
    std::string key_start;     // required
    std::string key_val_sep;   // required
    std::optional<std::string> key_val_sep2;  // second separator when whitespace may sit between two markers (GLM-4.5)
    std::string val_end;       // required
    std::string tool_end;      // required
    std::string scope_end;
    std::optional<bool> raw_argval;           // string params: true/unset = raw text up to val_end, false = JSON string
};

// Parse + validate the JSON text of a common_chat_xml_tool_format. Throws std::invalid_argument naming the field: an unknown
// key, a wrong type, a missing/empty required field, an unpaired scope, or a key the legacy struct has but this build does not
// honour (last_val_end, last_tool_end, trim_raw_argval, allow_toolcall_in_think) is an error, never silently ignored.
common_chat_xml_tool_format common_chat_xml_tool_format_parse(const std::string & json_text);

// The shipped presets (qwen3-coder, minimax-m2, glm-4.5, seed-oss) as JSON text; throws std::invalid_argument on an unknown name.
std::string              common_chat_xml_tool_format_preset_json(const std::string & name);
std::vector<std::string> common_chat_xml_tool_format_preset_names();
struct common_chat_prompt_part {
    std::string text;
    bool        is_input = false; // from user input, special tokens inside must not be parsed
};

// rendered chat template, split into template parts and user input parts
struct common_chat_prompt {
    std::vector<common_chat_prompt_part> parts;

    common_chat_prompt() = default;

    // single part of template text
    explicit common_chat_prompt(std::string text) { parts.push_back({std::move(text), false}); }

    std::string str() const;

    void append(const std::string & text, bool is_input = false);

    // replace the last occurrence of `from`, user input parts are skipped
    void replace_last(const std::string & from, const std::string & to);
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_prompt                  prompt;
    std::string                         grammar;
    bool                                grammar_lazy         = false;
    std::string                         generation_prompt;
    bool                                supports_thinking    = false;
    std::string                         thinking_start_tag;  // e.g., "<think>"
    std::vector<std::string>            thinking_end_tags;   // e.g., "</think>"
    // the handler's own statement that a call opener it lists in thinking_end_tags[k >= 1] really starts a call before the
    // thinking block is closed (parsers/ling3.cpp); the reasoning-call gate then lets the format parser decide a turn that
    // opens with it. Not set for an opener listed only so the budget sampler ends thinking there (the operator override).
    bool                                call_opener_ends_reasoning = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
    common_peg_arena                    parser;
    common_chat_msg_delimiters          message_delimiters;
};

struct common_chat_input {
    std::string              text;
    std::vector<llama_token> tokens;

    common_chat_input() = default;

    // plain text, with no tokens
    explicit common_chat_input(std::string text) : text(std::move(text)), tokens(this->text.size(), LLAMA_TOKEN_NULL) {}

    size_t size() const { return text.size(); }
    bool empty() const { return text.empty(); }

    void append(const std::string & piece, llama_token token);
    void append(const common_chat_input & chunk);

    void prepend(const std::string & prefix);
    void prepend(const common_chat_input & prefix);

    void truncate(size_t pos);

    common_chat_input substr(size_t pos, size_t n = std::string::npos) const;
};

common_chat_input common_chat_input_tokenize(const llama_vocab * vocab, const std::string & text);

// per-message parsing syntax
// should be derived from common_chat_params
struct common_chat_parser_params {
    common_chat_format      format               = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format     = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool parse_reasoning"
    // Whether reasoning_content should be inlined in the content (e.g. for reasoning_format=deepseek in stream mode)
    bool                    reasoning_in_content = false;
    common_chat_input       generation_prompt;   // upstream type (merge 2026-11-04): text + per-byte tokens
    bool                    parse_tool_calls     = true;
    bool                    is_continuation      = false;
    bool                    echo                 = false;  // Include assistant prefilled msg in output
    bool                    debug                = false;  // Enable debug output for PEG parser
    // Repair the argument types of parsed tool calls against `tools` (common_chat_coerce_tool_args). Applied to a FINAL
    // (non-partial) parse only: text already streamed to a client cannot be retracted, so do not enable this for a
    // streamed response (the server turns it off when stream=true).
    bool                    coerce_tool_args     = false;
    // Deliver a tool call the model wrote inside reasoning it never closed (common_chat_rescue_reasoning_calls). FINAL parse
    // only, and only when the caller knows the turn ended on its own: the server clears it after a max_tokens cut.
    bool                    rescue_reasoning_calls = false;
    // While the generation prompt left the thinking block OPEN, a call-looking tag in the text that follows is reasoning
    // until the thinking block is closed: the format parser would otherwise end the reasoning at the first such tag and
    // act on a mere quote of the call format (or on a call cut by max_tokens). Needs thinking_start_tag/_end_tags and tools.
    // HARNESS-CONSIDERED: this ADDS a gate on when a tool call is delivered (the server turns it on per request); the struct
    // default is off only so other callers of the parser keep their behaviour.
    bool                    gate_calls_in_reasoning = false;
    // A parse that is only a probe (the reasoning-call rescue trying candidate positions): a text the format does not match
    // throws as usual but is not logged as "unparsed output". HARNESS-CONSIDERED: log-only switch, removes no capability.
    bool                    quiet_unparsed          = false;
    // Rebuild a tool call the model wrote in a slightly malformed form (GLM-4.x XML with a tag missing, Gemma 4 with an unclosed
    // or single-quoted or bare value, an LFM2 call without its wrapper tokens) when the strict parse threw, found no call, or
    // returned a call whose arguments are not JSON or still hold call markup. FINAL parse only, and only for a turn the model
    // ended itself (the server clears it after a max_tokens cut, and it is off when streaming: a rewritten tail would not extend
    // the text already sent). A call the strict parser handled cleanly is never touched. (common/chat-malformed-repair.cpp)
    // HARNESS-CONSIDERED: this ADDS a way for a call to be delivered; the server turns it on per request (malformed_call_repair).
    bool                    repair_malformed_calls  = false;
    // Leave out of the message a tool call whose own closing marker the text never reached (the input ended inside it):
    // the last call, when its arguments are not yet a complete JSON value (the mappers close them only at the call's end).
    // On a partial parse this holds the call back until it is complete, so a stream sends it whole; on a final parse it
    // withholds a call cut by max_tokens / the context / a deadline, which would otherwise be returned with truncated
    // arguments. The parse stays lenient (as upstream): a FINAL parse without this flag still returns such a call. The
    // server sets it per request (withhold_incomplete_tool_calls, default on) and clears it on the final parse of a turn
    // the model ended itself. HARNESS-CONSIDERED: it decides which calls reach the client, which only the engine knows.
    bool                    drop_incomplete_tool_calls = false;
    // The end-of-thinking tag closes the thinking block only where a block ends: right before a line break or a new tag
    // ("<"), outside backticks and outside a fenced code block. The same tag written in the middle of a sentence, inside
    // `quotes` or in a fence is the model TALKING ABOUT the tag: it stays reasoning and the search for the real end goes on.
    // Without this the first literal tag ends the thinking, the rest of the sentence is sent as the answer and the real
    // end tag lands in the content (fastllm PR #765, class (c); behaviour only, no code copied). Only while the generation
    // prompt left the thinking block open; a format whose template does not put a line break after the tag must leave it off
    // (the server derives the default from the template). HARNESS-CONSIDERED: this only moves where reasoning ends, it
    // removes no capability, and the struct default is off so other callers of the parser keep their behaviour.
    bool                    block_level_thinking_end = false;
    // The turn ended on its own while the generation prompt had opened thinking, and the reply has neither text nor a tool
    // call (the model stopped inside or right after its reasoning). Read by the server, see common_chat_is_empty_thinking_reply.
    bool                    error_on_empty_thinking_reply = false;
    std::string             thinking_start_tag;                  // e.g. "<think>"
    std::vector<std::string> thinking_end_tags;                  // e.g. "</think>"
    // common_chat_params::call_opener_ends_reasoning: the handler's own statement that a call opener it lists in thinking_end_tags[k >= 1] really starts a call before the
    // thinking block is closed (parsers/ling3.cpp); the reasoning-call gate then lets the format parser decide a turn that
    // opens with it. Not set for an opener listed only so the budget sampler ends thinking there (the operator override).
    bool                     call_opener_ends_reasoning = false;
    std::vector<common_chat_tool> tools;               // the request's tools, used by coerce_tool_args and the rescue
    common_peg_arena        parser               = {};
    common_chat_parser_params() = default;
    common_chat_parser_params(const common_chat_params & chat_params) {
        format  = chat_params.format;
        generation_prompt = common_chat_input(chat_params.generation_prompt);
        thinking_start_tag = chat_params.thinking_start_tag;
        thinking_end_tags  = chat_params.thinking_end_tags;
        call_opener_ends_reasoning = chat_params.call_opener_ends_reasoning;
    }
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool common_chat_verify_template(const std::string & tmpl, bool use_jinja);

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter {
    void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override = "",
                                                     const std::string &        eos_token_override = "");

bool        common_chat_templates_was_explicit(const struct common_chat_templates * tmpls);
std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant = "");

struct common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                                      const struct common_chat_templates_inputs & inputs);

// Format single message, while taking into account the position of that message in chat history
std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja,
                                      const std::map<std::string, std::string> & chat_template_kwargs = {});

// Returns an example of formatted chat
std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs);

const char *    common_chat_format_name(common_chat_format format);
common_chat_msg common_chat_parse(const common_chat_input & input, bool is_partial, const common_chat_parser_params & params);
common_chat_msg common_chat_peg_parse(const common_peg_arena & src_parser, const common_chat_input & input, bool is_partial, const common_chat_parser_params & params);

struct common_chat_session_params {
    bool echo  = false; // include the assistant prefill in the output when continuing a message
    bool debug = false; // enable debug output for the PEG parser
};

class common_chat_session {
  public:
    common_chat_session() { result.role = "assistant"; }

    common_chat_session(const common_chat_templates *        tmpls,
                        const llama_vocab *                  vocab,
                        const common_chat_templates_inputs & inputs,
                        const common_chat_session_params &   params = {});

    const common_chat_prompt & prompt() const { return prompt_parts; }
    common_chat_format       format()   const { return parser_params.format; }
    const common_chat_msg &  msg()      const { return result; }
    const common_peg_arena & parser()   const { return parser_params.parser; }

    const std::string &              grammar()            const { return grammar_text; }
    const std::string &              generation_prompt()  const { return generation_prompt_text; }
    const std::string &              thinking_start_tag() const { return thinking_start; }
    const std::vector<std::string> & thinking_end_tags()  const { return thinking_ends; }
    const std::vector<std::string> & additional_stops()   const { return stops; }

    const common_chat_msg_delimiters & message_delimiters() const { return delimiters; }

    void apply_sampling(common_params_sampling & sampling) const;

    // fork (merge 2026-11-04): the per-request parse options the server fills on its task params (tools/server/server-schema.cpp)
    // -- reasoning format, tool-call parsing, the coerce / rescue / gate / quiet / repair / withhold / block-level / empty-reply
    // options and the request's tools -- applied onto this session's parser. The template's part (format, parser, generation
    // prompt, thinking tags, continuation) stays the session's.
    void apply_parser_options(const common_chat_parser_params & opts);

    // the parser params, for per-turn adjustments (the server clears rescue / repair / withhold after a max_tokens cut)
    common_chat_parser_params &       parser_options()       { return parser_params; }
    const common_chat_parser_params & parser_options() const { return parser_params; }

    bool has_template() const { return templated; }

    // everything fed so far (fork: the raw text a server returns when the format parser rejects the output)
    const common_chat_input & accumulated_input() const { return input; }

    const common_chat_msg & feed(const common_chat_input & chunk);

    const common_chat_msg & finish(const common_chat_input & chunk = {});

  private:
    common_chat_prompt                  prompt_parts;
    std::string                         grammar_text;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::set<llama_token>               preserved_tokens;
    std::vector<std::string>            stops;
    std::string                         generation_prompt_text;
    std::string                         thinking_start;
    std::vector<std::string>            thinking_ends;

    common_chat_parser_params  parser_params;
    common_chat_msg_delimiters delimiters;
    common_chat_input          input;
    common_chat_msg            result;
    bool                       templated = false;
    bool                       finished  = false;
};

// Did the generation prompt leave a thinking block open (it ends with the thinking start tag)? Needs thinking_start_tag.
bool common_chat_generation_opens_thinking(const common_chat_parser_params & params);

// A turn that ended on its own inside or right after its reasoning with an empty reply: the generation prompt opened
// thinking and `msg` carries neither text nor a tool call. When error_on_empty_thinking_reply is set the server answers it
// with a retryable error, because an empty reply with finish_reason stop reads as "the model is done" to an agent client
// (fastllm PR #765, class (d); behaviour only, no code copied).
bool common_chat_is_empty_thinking_reply(const common_chat_msg & msg, const common_chat_parser_params & params);

// Repair the argument values of the tool calls in `msg` so they match the types the request's `tools` declare:
// a model that writes {"n":"5","flag":"true","ids":"[1,2]"} for integer/boolean/array parameters gets 5, true, [1,2].
// Only declared types are touched, a value that cannot be repaired keeps its form, arguments that are not a complete JSON
// object are skipped. Returns the number of tool calls whose arguments changed. (common/chat-tool-args.cpp)
size_t common_chat_coerce_tool_args(common_chat_msg & msg, const std::vector<common_chat_tool> & tools);

// A turn that ended while still inside reasoning (no content, no parsed call) but whose reasoning ends with whole
// Qwen3-Coder-style calls (<tool_call><function=NAME>...): move those calls, for DECLARED tools only, into msg.tool_calls.
// Text that merely quotes the format (mid-line, in a code fence or inline code, followed by more prose) is never acted on.
// reasoning_content is left unchanged. Returns the number of calls added. The finish reason is the caller's gate: do not
// call this after a max_tokens cut. (common/chat-reasoning-rescue.cpp)
size_t common_chat_rescue_reasoning_calls(common_chat_msg & msg, const std::vector<common_chat_tool> & tools);

// Candidate texts for a malformed tool call (see repair_malformed_calls): `text` with each malformed call region of the given
// format rebuilt in the form the format parser accepts. GLM comes in both whitespace dialects (the caller tries each in
// order); empty when the text holds nothing to repair. Calls to tools not in `tools` and regions in a ``` fence are never
// rebuilt. (common/chat-malformed-repair.cpp)
std::vector<std::string> common_chat_malformed_call_candidates(common_chat_format format, const std::string & text,
                                                               const std::vector<common_chat_tool> & tools);

// True when a parsed call has arguments that are not a JSON value or still contain call markup (a strict parse that read a
// malformed call into a corrupt argument). (common/chat-malformed-repair.cpp)
bool common_chat_calls_look_corrupt(const common_chat_msg & msg);

// The line starts of `r` where a call may begin by the rescue's rules: outside a ``` / ~~~ fence and outside an inline code
// span left open earlier in the paragraph. (common/chat-reasoning-rescue.cpp)
std::vector<size_t> common_chat_reasoning_call_line_starts(const std::string & r);

// used by arg and server
const char *            common_reasoning_format_name(common_reasoning_format format);
common_reasoning_format common_reasoning_format_from_name(const std::string & format);

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice);

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const common_json & messages);

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const common_json & tools);

common_chat_continuation common_chat_continuation_parse(const common_json & value);

// DEPRECATED: only used in tests
common_json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

common_json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);

// The parameters schema of a function tool. A tool without parameters, or with an empty {}, takes zero arguments.
common_json common_chat_tool_parameters(const common_json & function);

// get template caps, useful for reporting to server /props endpoint
std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates);

common_chat_prompt common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs);

std::string common_chat_template_generation_prompt(
    const common_chat_template &          tmpl,
    const autoparser::generation_params & inputs);

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const autoparser::generation_params & params);


// specialized per-task preset
struct common_chat_prompt_preset {
    std::string system;
    std::string user;
};

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates);
