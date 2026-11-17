// GATE-CLAIM-OK: a render-parity test of the fork's Jinja engine against stored Jinja2 output; it asserts nothing about any host.
//
// Row S69-OLLAMA-PICK-GEMMA4-RENDERER-GOLDEN-VECTORS-2026-09-29-01.
// Renders each case of tests/gemma4-golden/cases.json with the fork's own Jinja engine (common_chat_template_direct_apply, the same call the server makes
// for the template-defined prompt) and compares it byte for byte with what real Jinja2 (3.1.6, Hugging Face's environment settings) rendered for the same
// template and inputs, see tests/gemma4-golden/gen_gemma4_golden.py. The oracle is Jinja2: when the two disagree the ENGINE is the thing under test.
//
// KNOWN_DIFFS lists cases where the engine is known to differ, each with the row that tracks it. The test fails when
//   (a) a case that is not listed disagrees  -> a new defect, or
//   (b) a listed case now AGREES               -> the entry is stale, remove it (so a fixed engine cannot leave a dead excuse behind).
//
// Usage: test-chat-gemma4-golden [path/to/cases.json]   (default tests/gemma4-golden/cases.json, run from the source root)
#include "chat-auto-parser.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

using json = common_json;

namespace {
// "<template key>/<case name>" -> tracking row. Filled from the first measured run; empty = every case agrees.
const std::map<std::string, std::string> KNOWN_DIFFS = {
    // KNOWN-DIFFS-BEGIN
    // (empty since the 2026-11-05 upstream merge) The eight former entries (system_content_parts, tool_response_content_parts,
    // user_parts_image_text on e2b / 31b_new / 31b_fork) were ONE mechanism: the capability probe called these templates "string content
    // only", so typed parts were flattened before the template ran. With upstream's chat code the probe reports typed_content=1 for all three
    // templates and the engine matches Jinja2 on every case (ctest 2026-11-05: 93 renders, 85 agree + these 8 = all agree, 0 differences).
    // Row S69-E-SERVING-GEMMA4-TYPED-CONTENT-CAPS-MEASURE-2026-10-06-01 carries the measurement.
    // KNOWN-DIFFS-END
};

std::string slurp(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string first_diff(const std::string & a, const std::string & b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) {
        ++i;
    }
    auto cut = [&](const std::string & s) {
        const size_t from = i > 30 ? i - 30 : 0;
        return s.substr(from, 80);
    };
    char head[64];
    std::snprintf(head, sizeof(head), "first difference at byte %zu (engine %zu bytes, oracle %zu): ", i, a.size(), b.size());
    std::string o = head;
    o += "engine=[" + cut(a) + "] oracle=[" + cut(b) + "]";
    for (char & c : o) {
        if (c == '\n') {
            c = '~';
        }
    }
    return o;
}
} // namespace

int main(int argc, char ** argv) {
    const std::string cases_path = argc > 1 ? argv[1] : "tests/gemma4-golden/cases.json";
    const std::string dir = cases_path.substr(0, cases_path.find_last_of("/\\") + 1);
    const json doc = json::parse(slurp(cases_path));

    int n_total = 0, n_ok = 0, n_known = 0, n_new = 0, n_stale = 0;
    std::set<std::string> seen_known;

    for (const auto & [key, rel] : doc.at("templates").items()) {
        const std::string src = slurp(dir + rel.get<std::string>());
        common_chat_template tmpl(src, "<bos>", "<eos>");
        std::printf("template %-9s capability probe: string_content=%d typed_content=%d\n", key.c_str(),
                    (int) tmpl.original_caps().supports_string_content, (int) tmpl.original_caps().supports_typed_content);
        for (const auto & c : doc.at("cases")) {
            const std::string name = c.at("name").get<std::string>();
            const std::string id = key + "/" + name;
            const json & exp = c.at("expected").at(key);
            autoparser::generation_params p;
            p.messages              = c.at("messages");
            p.tools                 = c.at("tools");
            p.add_generation_prompt = c.at("add_generation_prompt").get<bool>();
            p.enable_thinking       = c.at("enable_thinking").get<bool>();
            p.mark_input            = false;
            p.extra_context         = json::object();
            if (c.contains("preserve_thinking") && c.at("preserve_thinking").is_boolean()) {
                p.extra_context["preserve_thinking"] = c.at("preserve_thinking");
            }
            std::string got;
            bool threw = false;
            try {
                got = common_chat_template_direct_apply(tmpl, p).str();
            } catch (const std::exception & e) {
                got = std::string("EXCEPTION: ") + e.what();
                threw = true;
            }
            ++n_total;
            bool agree;
            std::string why;
            if (exp.contains("error")) {
                agree = threw; // the oracle raised (raise_exception): the engine must raise too
                why = "oracle raised [" + exp.at("error").get<std::string>() + "], engine " + (threw ? "raised" : "rendered: " + got);
            } else {
                agree = !threw && got == exp.at("text").get<std::string>();
                why = threw ? got : first_diff(got, exp.at("text").get<std::string>());
            }
            const bool known = KNOWN_DIFFS.count(id) > 0;
            if (known) {
                seen_known.insert(id);
            }
            if (agree && !known) {
                ++n_ok;
            } else if (agree && known) {
                ++n_stale;
                std::printf("STALE  %s: now agrees with Jinja2, remove it from KNOWN_DIFFS (%s)\n", id.c_str(), KNOWN_DIFFS.at(id).c_str());
            } else if (!agree && known) {
                ++n_known;
                std::printf("known  %s (%s)\n", id.c_str(), KNOWN_DIFFS.at(id).c_str());
            } else {
                ++n_new;
                std::printf("DIFF   %s: %s\n", id.c_str(), why.c_str());
            }
        }
    }
    for (const auto & [id, row] : KNOWN_DIFFS) {
        if (!seen_known.count(id)) {
            ++n_stale;
            std::printf("STALE  %s names a case that does not exist (%s)\n", id.c_str(), row.c_str());
        }
    }
    std::printf("gemma4 golden: %d renders, %d agree with Jinja2, %d known differences, %d NEW differences, %d stale entries\n", n_total, n_ok, n_known, n_new, n_stale);
    return (n_new || n_stale) ? 1 : 0;
}
