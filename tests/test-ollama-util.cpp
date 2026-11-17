// Unit test for tools/server/server-ollama-util.h. Expected values are FACTS taken from ollama's own test tables
// (types/model/name_test.go, parser/parser_test.go @ e3cddc3e, MIT), not code.
#include "server-ollama-util.h"
#include <cstdio>
#include <map>

using namespace ollama_util;
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::string join(const std::vector<modelfile_cmd> & v) {
    std::string s;
    for (auto & c : v) s += "[" + c.name + "|" + c.args + "]";
    return s;
}
static std::string parse(const std::string & in) {
    std::vector<modelfile_cmd> v; std::string err;
    return parse_modelfile(in, v, err) ? join(v) : "ERR:" + err;
}

int main() {
    // ---- names (name_test.go TestParseNameParts / TestDisplayShortest)
    CHECK(canonical_key("model") == "registry.ollama.ai/library/model:latest", "bare model");
    CHECK(canonical_key("model:latest") == canonical_key("library/model"), "tag/namespace defaults");
    CHECK(canonical_key("registry.ollama.ai/library/model:latest") == canonical_key("MODEL"), "case-insensitive full form");
    CHECK(canonical_key("model:7b") != canonical_key("model"), "different tag differs");
    { auto n = parse_name("host:5000/ns/m"); CHECK(n.host == "host:5000" && n.ns == "ns" && n.model == "m" && n.tag == "latest", "host:port kept"); }
    { auto n = parse_name("scheme://host:port/namespace/model:tag"); CHECK(n.scheme == "scheme" && n.host == "host:port" && n.tag == "tag", "scheme"); }
    { auto n = parse_name("namespace/model"); CHECK(n.host == "registry.ollama.ai" && n.ns == "namespace", "ns only"); }
    CHECK(valid(parse_name("dolphin-mistral:7b-v2.6-dpo-laser-q6_K")), "valid tag with dots");
    CHECK(!valid(parse_name("a/b/c/d:e")), "too many parts is invalid");
    CHECK(!valid(parse_name("model:")), "empty tag invalid");
    CHECK(!valid(parse_name("na.me/model")), "dot in namespace invalid");
    CHECK(!valid(parse_name(std::string(81, 'a'))), "model > 80 invalid");
    CHECK(display_shortest(parse_name_bare("registry.ollama.ai/library/model:latest")) == "model:latest", "shortest 1");
    CHECK(display_shortest(parse_name_bare("registry.ollama.ai/namespace/model:tag")) == "namespace/model:tag", "shortest 2");
    CHECK(display_shortest(parse_name_bare("host/library/model:tag")) == "host/library/model:tag", "shortest 3");

    // ---- Modelfile (parser_test.go)
    CHECK(parse("FROM \"FOO  BAR  \"") == "[model|FOO  BAR  ]", "quoted FROM keeps spaces: %s", parse("FROM \"FOO  BAR  \"").c_str());
    CHECK(parse("FROM \"FOO BAR\"\nPARAMETER param1 value1") == "[model|FOO BAR][param1|value1]", "FROM+PARAMETER");
    CHECK(parse("FROM     FOOO BAR    ") == "[model|FOOO BAR]", "unquoted FROM trims: %s", parse("FROM     FOOO BAR    ").c_str());
    CHECK(parse("FROM /what/is/the path ") == "[model|/what/is/the path]", "path with space");
    CHECK(parse("FROM llama3:7b-instruct-q4_K_M") == "[model|llama3:7b-instruct-q4_K_M]", "tag");
    CHECK(parse("").rfind("ERR:", 0) == 0 && parse("").find("no FROM line") != std::string::npos, "empty -> no FROM");
    CHECK(parse("PARAMETER param1 value1").find("no FROM line") != std::string::npos, "param only -> no FROM");
    CHECK(parse("PARAMETER param1 value1\nFROM foo") == "[param1|value1][model|foo]", "FROM may follow");
    CHECK(parse("PARAMETER what the \nFROM lemons make lemonade ") == "[what|the][model|lemons make lemonade]", "trailing spaces");
    CHECK(parse("FROM foo\nPARAMETER param1\n").rfind("ERR:", 0) == 0, "PARAMETER missing value is an error");
    { std::string e = parse("\nFROM foo\nBADCOMMAND param1 value1\n"); CHECK(e.find("(line 3)") != std::string::npos, "bad command line number: %s", e.c_str()); }
    CHECK(parse("\nFROM foo\nSYSTEM \"\"\"\nThis is a\nmultiline system.\n\"\"\"\n\t\t\t") == "[model|foo][system|\nThis is a\nmultiline system.\n]", "multiline system: %s", parse("\nFROM foo\nSYSTEM \"\"\"\nThis is a\nmultiline system.\n\"\"\"\n\t\t\t").c_str());
    CHECK(parse("FROM foo\nSYSTEM \"\"\"This is a multiline system.\"\"\"") == "[model|foo][system|This is a multiline system.]", "single-line triple quotes");
    CHECK(parse("FROM foo\nSYSTEM \"\"\"This is a multiline system.\"\"\n").rfind("ERR:", 0) == 0, "unterminated triple quote");
    CHECK(parse("FROM foo\nSYSTEM \"\n").rfind("ERR:", 0) == 0, "unterminated quote");
    CHECK(parse("# comment\nFROM foo\n") == "[model|foo]", "comment");
    CHECK(parse("FROM foo\nPARAMETER stop \"### User: \"") == "[model|foo][stop|### User: ]", "quoted stop keeps trailing space");
    CHECK(parse("FROM foo\nPARAMETER stop ### User: ") == "[model|foo][stop|### User:]", "unquoted stop trimmed");
    CHECK(parse("FROM foo\nPARAMETER stop \"\"\"### User:\n\"\"\"") == "[model|foo][stop|### User:\n]", "triple-quote stop with newline");
    CHECK(parse("\xEF\xBB\xBF" "FROM foo") == "[model|foo]", "BOM");
    CHECK(parse("FROM foo\nMESSAGE user hello there\nMESSAGE assistant \"hi\"") == "[model|foo][message|user: hello there][message|assistant: hi]", "messages: %s", parse("FROM foo\nMESSAGE user hello there\nMESSAGE assistant \"hi\"").c_str());
    CHECK(parse("FROM foo\nMESSAGE robot hi").rfind("ERR:", 0) == 0, "bad role");
    CHECK(parse("FROM foo\nsystem be brief") == "[model|foo][system|be brief]", "keywords are case-insensitive");

    // ---- params coercion
    {
        std::vector<modelfile_cmd> v; std::string err; json p;
        CHECK(parse_modelfile("FROM x\nPARAMETER temperature 0.25\nPARAMETER top_k 7\nPARAMETER stop a\nPARAMETER stop \"b c\"\nPARAMETER mirostat 2\nSYSTEM hi", v, err), "%s", err.c_str());
        std::vector<std::string> warn;
        CHECK(modelfile_params(v, p, err, &warn), "%s", err.c_str());
        CHECK(p["temperature"] == 0.25 && p["top_k"] == 7 && p["stop"].size() == 2 && p["stop"][1] == "b c", "params: %s", p.dump().c_str());
        CHECK(!p.contains("mirostat") && warn.size() == 1, "deprecated dropped with warning");
        CHECK(parse_modelfile("FROM x\nPARAMETER top_k 0.5", v, err) && !modelfile_params(v, p, err) && err == "invalid int value [0.5]", "int strict: %s", err.c_str());
        CHECK(parse_modelfile("FROM x\nPARAMETER bogus 1", v, err) && !modelfile_params(v, p, err) && err == "unknown parameter 'bogus'", "unknown");
        CHECK(parse_modelfile("FROM x\nPARAMETER typical_p 0.9", v, err) && !modelfile_params(v, p, err), "typical_p rejected");
        CHECK(parse_modelfile("FROM x\nPARAMETER use_mmap maybe", v, err) && !modelfile_params(v, p, err) && err == "invalid bool value [maybe]", "bool");
    }

    // ---- options precedence + mapping
    {
        json defaults = {{"temperature", 0.1}, {"top_k", 5}, {"stop", json::array({"x"})}};
        json req = {{"temperature", 0.9}, {"min_p", 0.2}, {"typical_p", 0.9}, {"repeat_last_n", 8}, {"presence_penalty", 0.5},
                    {"frequency_penalty", 0.25}, {"num_keep", 2}, {"num_predict", 3}, {"num_ctx", 99999}, {"seed", 1}, {"top_p", nullptr}};
        json t = json::object();
        apply_options(defaults, req, t);
        CHECK(t["temperature"] == 0.9, "request overrides modelfile");
        CHECK(t["top_k"] == 5, "modelfile default survives");
        CHECK(t["min_p"] == 0.2 && t["typical_p"] == 0.9 && t["repeat_last_n"] == 8, "min_p/typical_p/repeat_last_n mapped");
        CHECK(t["presence_penalty"] == 0.5 && t["frequency_penalty"] == 0.25, "penalties mapped");
        CHECK(t["n_keep"] == 2 && t["n_predict"] == 3 && t["seed"] == 1, "n_keep/n_predict/seed");
        CHECK(!t.contains("num_ctx") && !t.contains("top_p"), "num_ctx never mapped; null skipped");
        CHECK(t["stop"].size() == 1, "stop from defaults");
        json t2 = json::object();
        apply_options(json::object(), json {{"stop", json::array({"a", "b"})}}, t2);
        CHECK(t2["stop"].size() == 2, "request stop replaces");
    }
    // ---- OLLAMA_* environment (envconfig/config_test.go TestBool / TestUint / TestKeepAlive tables)
    {
        auto env = [](std::initializer_list<std::pair<const char *, const char *>> kv) {
            std::map<std::string, std::string> m(kv.begin(), kv.end());
            return read_env([m](const char * k) -> const char * { auto it = m.find(k); return it == m.end() ? nullptr : it->second.c_str(); });
        };
        auto fa = [&](const char * v) { auto e = env({{"OLLAMA_FLASH_ATTENTION", v}}); return e.has_flash ? (e.flash ? 1 : 0) : -1; };
        CHECK(fa("true") == 1 && fa("1") == 1 && fa("false") == 0 && fa("0") == 0, "flash bool");
        CHECK(fa("random") == 1 && fa("something") == 1, "invalid bool counts as true (TestBool)");
        CHECK(fa("") == -1, "unset flash");
        CHECK(env({{"OLLAMA_KV_CACHE_TYPE", "Q8_0"}}).kv_type == "q8_0", "kv type lower-cased (llm/server.go:116)");
        auto up = [&](const char * v) { auto e = env({{"OLLAMA_NUM_PARALLEL", v}}); return e.has_parallel ? (int) e.parallel : -1; };
        CHECK(up("4") == 4 && up("1337") == 1337, "uint ok");
        CHECK(up("-1") == -1 && up("0x10") == -1 && up("0o10") == -1 && up("string") == -1 && up("") == -1, "invalid uint -> default (TestUint)");
        CHECK(up("0") == -1, "NUM_PARALLEL=0 keeps the default");
        CHECK(!env({{"OLLAMA_CONTEXT_LENGTH", "0"}}).has_ctx && env({{"OLLAMA_CONTEXT_LENGTH", "8192"}}).ctx == 8192, "context length");
        CHECK(!env({{"OLLAMA_MAX_LOADED_MODELS", "0"}}).has_max_models && env({{"OLLAMA_MAX_LOADED_MODELS", "3"}}).max_models == 3, "max models");
        CHECK(!env({{"OLLAMA_NUM_PARALLEL", "-1"}}).warnings.empty(), "invalid uint warns");
        auto ka = [](const char * v) { int64_t ms = -7; return keep_alive_ms(v, ms) ? ms : -7; };
        CHECK(ka("1s") == 1000 && ka("1m") == 60000 && ka("1h") == 3600000 && ka("5m0s") == 300000, "keep_alive durations");
        CHECK(ka("1h2m3s") == 3723000 && ka("0") == 0 && ka("60") == 60000 && ka("120") == 120000 && ka("3600") == 3600000, "keep_alive ints");
        CHECK(ka("-0") == 0 && ka("-1") == -1 && ka("-1m") == -1, "keep_alive negative = forever, -0 = 0");
        CHECK(ka(" ") == -7 && ka("???") == -7 && ka("1d") == -7 && ka("1y") == -7 && ka("1w") == -7 && ka("") == -7, "keep_alive invalid -> default");
        CHECK(ka("1.5s") == 1500 && ka("250ms") == 250, "fractions and ms");
    }
    // ---- format round trip
    {
        std::vector<modelfile_cmd> v; std::string err;
        parse_modelfile("FROM x\nPARAMETER stop \"a \"\nSYSTEM \"\"\"two\nlines\"\"\"\nMESSAGE user hi", v, err);
        CHECK(modelfile_format(v) == "FROM x\nPARAMETER stop \"a \"\nSYSTEM \"two\nlines\"\nMESSAGE user hi\n", "format: %s", modelfile_format(v).c_str());
    }
    // ---- media tags for generate raw / suffix (server_routes.go:578-581 media IDs 0..n-1; llama_server.go:1707-1722 first-tag swap)
    {
        const std::string M = "<M>";
        std::string a = "see [img-0] and [img-1] then [img-0] again";
        CHECK(swap_media_tags({ &a }, 2, M) == 2 && a == "see <M> and <M> then [img-0] again", "swap first occurrence only: %s", a.c_str());
        std::string b = "no tags here";
        CHECK(swap_media_tags({ &b }, 1, M) == 0 && b == "no tags here", "untagged image counts 0");
        std::string c = "pre [img-1] x";
        CHECK(swap_media_tags({ &c }, 2, M) == 1 && c == "pre <M> x", "img-0 missing, img-1 found: %s", c.c_str());
        std::string pre = "def f(): # [img-0]\n", suf = "\nreturn [img-1]";
        CHECK(swap_media_tags({ &pre, &suf }, 2, M) == 2 && pre == "def f(): # <M>\n" && suf == "\nreturn <M>", "prefix then suffix searched in order");
        std::string both_in_suf = "[img-0]";
        std::string empty_pre = "x";
        CHECK(swap_media_tags({ &empty_pre, &both_in_suf }, 1, M) == 1 && both_in_suf == "<M>" && empty_pre == "x", "tag only in the second part");
        std::string d = "[img-10] [img-1]";
        CHECK(swap_media_tags({ &d }, 2, M) == 1 && d == "[img-10] <M>", "[img-1] must not match inside [img-10]: %s", d.c_str());
        // ---- FIM text (format_prompt_infill layout)
        fim_pieces fp; fp.pre = "<P>"; fp.suf = "<S>"; fp.mid = "<Mid>";
        CHECK(fim_text(fp, "a", "b", {}) == "<P>a<S>b<Mid>", "plain PSM: %s", fim_text(fp, "a", "b", {}).c_str());
        fp.spm = true;
        CHECK(fim_text(fp, "a", "b", {}) == "<S>b<P>a<Mid>", "SPM order");
        fp.spm = false; fp.rep = "<R>"; fp.sep = "<F>";
        CHECK(fim_text(fp, "a", "b", { { "x.c", "int x;" } }) == "<R>myproject\n<F>x.c\nint x;<F>filename\n<P>a<S>b<Mid>", "repo-level extra: %s",
              fim_text(fp, "a", "b", { { "x.c", "int x;" } }).c_str());
        fp.rep.clear(); fp.sep.clear();
        CHECK(fim_text(fp, "a", "b", { { "x.c", "T" } }) == "\n\n--- snippet ---\n\nT<P>a<S>b<Mid>", "no sep token: snippet separator");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
