// Operator-supplied shape of an XML-style tool call: parsing, validation and the shipped presets.
//
// WHY: the specialised handlers in chat.cpp hard-code one XML dialect per model family (Qwen3-Coder, MiniMax-M3, MiniCPM5 ...)
// and everything else goes through the template-diff autoparser. A misdetected or brand-new XML-style model therefore needs a
// code change and a rebuild. `--tool-call-format-json '{...}'` / `--tool-call-format-preset NAME` let the operator say "this model
// emits <call name=...><p k=...>" instead; chat.cpp builds the PEG parser and the grammar from the result.
//
// IDEA from mudler/LocalAI pkg/functions/parse.go:64-125,186-213 (MIT): FunctionsConfig.xml_format / xml_format_preset, whose field
// set mirrors the old llama.cpp xml_tool_call_format. The field names are kept so a LocalAI config can be pasted; the semantics
// below are this fork's PEG ones (see the comment on common_chat_xml_tool_format in chat.h).
//   SOURCE-READ: upstream/LocalAI/pkg/functions/parse.go:64-125,186-213, peg_integration.go:60-110
// The presets are the literals of the fork's former (now unbuilt) common/chat-parser.cpp handlers, re-expressed for the PEG layer.

#include "chat.h"
#include "json.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;

namespace {

struct xml_preset {
    const char * name;
    const char * json;
};

// Whitespace between structural tokens is skipped by the parser, so a literal states only the marker itself, except where
// the marker is followed by value text: there the newline that belongs to the format is part of the literal.
const xml_preset XML_PRESETS[] = {
    { "qwen3-coder",
      R"({"scope_start":"<tool_call>","tool_start":"<function=","tool_sep":">","key_start":"<parameter=","key_val_sep":">\n",)"
      R"("val_end":"\n</parameter>","tool_end":"</function>","scope_end":"</tool_call>"})" },
    { "minimax-m2",
      R"({"scope_start":"<minimax:tool_call>","tool_start":"<invoke name=\"","tool_sep":"\">","key_start":"<parameter name=\"",)"
      R"("key_val_sep":"\">","val_end":"</parameter>","tool_end":"</invoke>","scope_end":"</minimax:tool_call>"})" },
    { "glm-4.5",
      R"({"tool_start":"<tool_call>","key_start":"<arg_key>","key_val_sep":"</arg_key>","key_val_sep2":"<arg_value>",)"
      R"("val_end":"</arg_value>","tool_end":"</tool_call>"})" },
    { "seed-oss",
      R"({"scope_start":"<seed:tool_call>","tool_start":"<function=","tool_sep":">","key_start":"<parameter=","key_val_sep":">",)"
      R"("val_end":"</parameter>","tool_end":"</function>","scope_end":"</seed:tool_call>"})" },
};

[[noreturn]] void fail(const std::string & what) {
    throw std::invalid_argument("tool-call format: " + what);
}

}  // namespace

std::vector<std::string> common_chat_xml_tool_format_preset_names() {
    std::vector<std::string> names;
    for (const auto & p : XML_PRESETS) {
        names.push_back(p.name);
    }
    return names;
}

std::string common_chat_xml_tool_format_preset_json(const std::string & name) {
    for (const auto & p : XML_PRESETS) {
        if (name == p.name) {
            return p.json;
        }
    }
    std::string known;
    for (const auto & n : common_chat_xml_tool_format_preset_names()) {
        known += (known.empty() ? "" : ", ") + n;
    }
    fail("unknown preset '" + name + "' (known: " + known + ")");
}

common_chat_xml_tool_format common_chat_xml_tool_format_parse(const std::string & text) {
    json j = json::parse_no_throw(text);
    if (j.is_discarded()) {
        fail("not valid JSON");
    }
    if (!j.is_object()) {
        fail("expected a JSON object");
    }

    // keys the LocalAI/legacy struct has but this implementation does not honour: refuse them by name, never ignore them
    static const std::set<std::string> NOT_SUPPORTED = { "last_val_end", "last_tool_end", "trim_raw_argval", "allow_toolcall_in_think" };
    static const std::set<std::string> STRING_KEYS   = { "scope_start", "tool_start", "tool_sep", "key_start", "key_val_sep",
                                                         "key_val_sep2", "val_end", "tool_end", "scope_end" };

    common_chat_xml_tool_format f;
    for (const auto & entry : j.items()) {
        const std::string key = entry.key();
        const json &      val = entry.value();
        if (NOT_SUPPORTED.count(key)) {
            fail("key '" + key + "' is not supported by this server (the PEG layer has no equivalent yet)");
        }
        if (key == "raw_argval") {
            if (!val.is_boolean()) {
                fail("'raw_argval' must be true or false");
            }
            f.raw_argval = val.get<bool>();
            continue;
        }
        if (!STRING_KEYS.count(key)) {
            std::string known;
            for (const auto & k : STRING_KEYS) {
                known += k + ", ";
            }
            fail("unknown key '" + key + "' (known: " + known + "raw_argval)");
        }
        if (!val.is_string()) {
            fail("'" + key + "' must be a string");
        }
        const std::string s = val.get<std::string>();
        if (key == "scope_start")  { f.scope_start = s; }
        if (key == "tool_start")   { f.tool_start = s; }
        if (key == "tool_sep")     { f.tool_sep = s; }
        if (key == "key_start")    { f.key_start = s; }
        if (key == "key_val_sep")  { f.key_val_sep = s; }
        if (key == "key_val_sep2") { f.key_val_sep2 = s; }
        if (key == "val_end")      { f.val_end = s; }
        if (key == "tool_end")     { f.tool_end = s; }
        if (key == "scope_end")    { f.scope_end = s; }
    }

    // Whitespace that sits where the parser already skips whitespace must not be part of a marker: the PEG space() is greedy and
    // would eat the first character of a marker that begins with it, so the marker would never match (while the generated grammar,
    // which has no such greed, would accept the text: parser and grammar would disagree). Only the two places where whitespace is
    // meaningful keep it: the START of val_end (the value ends before it: "\n</parameter>") and the END of key_val_sep /
    // key_val_sep2 (the value starts after it: ">\n").
    auto trim = [](std::string & s, bool left, bool right) {
        const char * ws = " \t\r\n";
        if (right) {
            const size_t e = s.find_last_not_of(ws);
            s.erase(e == std::string::npos ? 0 : e + 1);
        }
        if (left) {
            s.erase(0, std::min(s.find_first_not_of(ws), s.size()));
        }
    };
    trim(f.scope_start, true, true);
    trim(f.scope_end, true, true);
    trim(f.tool_start, true, true);
    trim(f.tool_sep, true, true);
    trim(f.key_start, true, true);
    trim(f.tool_end, true, true);
    trim(f.key_val_sep, true, false);
    if (f.key_val_sep2.has_value()) {
        trim(*f.key_val_sep2, true, false);
    }
    trim(f.val_end, false, true);

    // the delimiters the parser cannot work without: a missing one would match everything or nothing
    const std::vector<std::pair<const char *, const std::string *>> required = {
        { "tool_start", &f.tool_start }, { "key_start", &f.key_start }, { "key_val_sep", &f.key_val_sep },
        { "val_end", &f.val_end },       { "tool_end", &f.tool_end },
    };
    for (const auto & r : required) {
        if (r.second->empty()) {
            fail(std::string("required field '") + r.first + "' is missing or empty");
        }
    }
    if (f.key_val_sep2.has_value() && f.key_val_sep2->empty()) {
        fail("'key_val_sep2' is present but empty");
    }
    if (f.scope_start.empty() != f.scope_end.empty()) {
        fail(f.scope_start.empty() ? "'scope_end' is set but 'scope_start' is empty" : "'scope_start' is set but 'scope_end' is empty");
    }
    return f;
}
