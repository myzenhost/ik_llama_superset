// Post-parse repair of tool-call argument types against the request's own tool schemas.
//
// WHY: small and quantised models often write {"n":"5","flag":"true","ids":"[1,2]"} for a schema that says
// integer / boolean / array. Strict clients (OpenAI SDK pydantic models, MCP servers) then fail the call. The
// PEG parsers already pick a string or a value grammar by the property type for the XML-style formats, but the
// JSON-style formats return whatever types the model wrote, and nothing repaired them after the parse.
//
// IDEA from koboldcpp coerce_tool_argtypes (koboldcpp.py:3716-3823 @36688de9, AGPL-3.0): behaviour only, read for
// what it converts, rewritten here from the spec in the findings file; no code copied.
//   SOURCE-READ: upstream/koboldcpp/koboldcpp.py:3716-3823
//
// RULES (own spec; deliberately narrower than the source where a guess could corrupt a value):
//   - only an argument whose property declares a type is touched; unknown tools / properties / arguments are left as is
//   - the type is the "type" (a string, or an array: first non-"null"), else the first typed non-null anyOf/oneOf branch;
//     a local "$ref" (#/$defs/X, #/definitions/X) is followed
//   - integer: JSON string holding an integer, or an integral number written as a float ("5.0"), becomes an integer
//   - number : JSON string holding a decimal number becomes a number (integer-valued text stays an integer)
//   - boolean: the strings true/false/yes/no/1/0 (any case) and the numbers 1/0 become a boolean
//   - string : a number or boolean becomes its JSON text; objects, arrays and null are never turned into strings
//   - array  : a string holding a JSON array becomes that array; any other JSON scalar/object (or a non-JSON, non-empty
//              string) is wrapped as a one-element array; items are repaired against "items"
//   - object : a string holding a JSON object becomes that object; members are repaired against "properties"
//   - null stays null, and a value that cannot be repaired keeps its original form
//   - arguments that are not a complete JSON object (a streamed prefix) are skipped

#include "chat.h"
#include "json.h"
#include "log.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using json = common_json;

namespace {

constexpr int COERCE_MAX_DEPTH = 8;

struct typed_schema {
    std::string  type;
    const json * node = nullptr;
};

// follow "#/a/b" inside root; nullptr when it does not resolve
const json * resolve_local_ref(const json & root, const std::string & ref) {
    if (ref.size() < 2 || ref[0] != '#' || ref[1] != '/') {
        return nullptr;
    }
    const json * cur = &root;
    size_t       pos = 2;
    while (pos <= ref.size()) {
        const size_t      end  = ref.find('/', pos);
        const std::string part = ref.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (!cur->is_object() || !cur->contains(part)) {
            return nullptr;
        }
        cur = &cur->at(part);
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    return cur;
}

typed_schema pick_schema(const json & schema, const json & root, int depth) {
    if (depth > COERCE_MAX_DEPTH || !schema.is_object()) {
        return {};
    }
    const json * node = &schema;
    if (schema.contains("$ref") && schema.at("$ref").is_string()) {
        node = resolve_local_ref(root, schema.at("$ref").get<std::string>());
        if (node == nullptr || !node->is_object()) {
            return {};
        }
    }
    if (node->contains("type")) {
        const json & t = node->at("type");
        if (t.is_string()) {
            return { t.get<std::string>(), node };
        }
        if (t.is_array()) {
            for (const auto & el : t) {
                if (el.is_string() && el.get<std::string>() != "null") {
                    return { el.get<std::string>(), node };
                }
            }
        }
        return {};
    }
    for (const char * combiner : { "anyOf", "oneOf" }) {
        if (node->contains(combiner) && node->at(combiner).is_array()) {
            for (const auto & opt : node->at(combiner)) {
                auto r = pick_schema(opt, root, depth + 1);
                if (!r.type.empty() && r.type != "null") {
                    return r;
                }
            }
        }
    }
    return {};
}

std::string trim_ws(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) {
        b++;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) {
        e--;
    }
    return s.substr(b, e - b);
}

// digits, sign, '.', 'e': keeps strtod from accepting "0x10", "inf", "nan"
bool looks_decimal(const std::string & s) {
    bool digit = false;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            digit = true;
        } else if (c != '+' && c != '-' && c != '.' && c != 'e' && c != 'E') {
            return false;
        }
    }
    return digit;
}

bool parse_int_text(const std::string & s, long long & out) {
    if (!looks_decimal(s) || s.find_first_of(".eE") != std::string::npos) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    out = std::strtoll(s.c_str(), &end, 10);
    return errno == 0 && end != nullptr && *end == '\0';
}

bool parse_double_text(const std::string & s, double & out) {
    if (!looks_decimal(s)) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return errno == 0 && end != nullptr && *end == '\0' && std::isfinite(out);
}

bool integral_double(double d) {
    return std::isfinite(d) && std::floor(d) == d && std::fabs(d) < 9.0e18;
}

bool coerce_value(json & v, const json & schema, const json & root, int depth);

void coerce_object_members(json & obj, const json & node, const json & root, int depth, bool & changed) {
    if (!obj.is_object() || !node.contains("properties") || !node.at("properties").is_object()) {
        return;
    }
    const json & props = node.at("properties");
    for (const auto & entry : obj.items()) {
        if (props.contains(entry.key())) {
            changed |= coerce_value(entry.value(), props.at(entry.key()), root, depth + 1);
        }
    }
}

bool coerce_value(json & v, const json & schema, const json & root, int depth) {
    if (depth > COERCE_MAX_DEPTH || v.is_null()) {
        return false;
    }
    const auto ts = pick_schema(schema, root, 0);
    if (ts.type.empty() || ts.node == nullptr) {
        return false;
    }
    const std::string & t       = ts.type;
    bool                changed = false;

    if (t == "integer") {
        if (v.is_number_float()) {
            const double d = v.get<double>();
            if (integral_double(d)) {
                v       = (long long) d;
                changed = true;
            }
        } else if (v.is_string()) {
            const std::string s = trim_ws(v.get<std::string>());
            long long         i = 0;
            double            d = 0;
            if (parse_int_text(s, i)) {
                v       = i;
                changed = true;
            } else if (parse_double_text(s, d) && integral_double(d)) {
                v       = (long long) d;
                changed = true;
            }
        }
    } else if (t == "number") {
        if (v.is_string()) {
            const std::string s = trim_ws(v.get<std::string>());
            long long         i = 0;
            double            d = 0;
            if (parse_int_text(s, i)) {
                v       = i;
                changed = true;
            } else if (parse_double_text(s, d)) {
                v       = d;
                changed = true;
            }
        }
    } else if (t == "boolean") {
        if (v.is_string()) {
            std::string s = trim_ws(v.get<std::string>());
            for (auto & c : s) {
                c = (char) std::tolower((unsigned char) c);
            }
            if (s == "true" || s == "yes" || s == "1") {
                v       = true;
                changed = true;
            } else if (s == "false" || s == "no" || s == "0") {
                v       = false;
                changed = true;
            }
        } else if (v.is_number_integer()) {
            const long long i = v.get<long long>();
            if (i == 0 || i == 1) {
                v       = (i == 1);
                changed = true;
            }
        }
    } else if (t == "string") {
        if (v.is_number() || v.is_boolean()) {
            v       = v.dump();
            changed = true;
        }
    } else if (t == "array") {
        if (v.is_string()) {
            const std::string s = trim_ws(v.get<std::string>());
            if (!s.empty()) {
                json parsed = json::parse_no_throw(s);
                if (parsed.is_discarded()) {
                    v = json::array({ v.get<std::string>() });
                } else if (parsed.is_array()) {
                    v = parsed;
                } else if (!parsed.is_null()) {
                    v = json::array({ parsed });
                }
                changed = v.is_array();
            }
        } else if (v.is_number() || v.is_boolean() || v.is_object()) {
            v       = json::array({ v });
            changed = true;
        }
        if (v.is_array() && ts.node->contains("items")) {
            for (auto & el : v) {
                changed |= coerce_value(el, ts.node->at("items"), root, depth + 1);
            }
        }
    } else if (t == "object") {
        if (v.is_string()) {
            json parsed = json::parse_no_throw(trim_ws(v.get<std::string>()));
            if (!parsed.is_discarded() && parsed.is_object()) {
                v       = parsed;
                changed = true;
            }
        }
        coerce_object_members(v, *ts.node, root, depth, changed);
    }
    return changed;
}

}  // namespace

size_t common_chat_coerce_tool_args(common_chat_msg & msg, const std::vector<common_chat_tool> & tools) {
    size_t n_repaired = 0;
    for (auto & call : msg.tool_calls) {
        const common_chat_tool * tool = nullptr;
        for (const auto & t : tools) {
            if (t.name == call.name) {
                tool = &t;
                break;
            }
        }
        if (tool == nullptr || tool->parameters.empty() || call.arguments.empty()) {
            continue;
        }
        json params = json::parse_no_throw(tool->parameters);
        json args   = json::parse_no_throw(call.arguments);
        if (params.is_discarded() || !params.is_object() || args.is_discarded() || !args.is_object()) {
            continue;
        }
        bool changed = false;
        coerce_object_members(args, params, params, 0, changed);
        if (changed) {
            call.arguments = args.dump_safe();
            n_repaired++;
        }
    }
    if (n_repaired > 0) {
        LOG_DBG("%s: repaired argument types of %zu tool call(s)\n", __func__, n_repaired);
    }
    return n_repaired;
}
