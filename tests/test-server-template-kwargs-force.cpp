// --chat-template-kwargs-force merge (tools/server/server-template-kwargs-force.h).
// Unmeasured here: an end-to-end /apply-template render needs a GGUF whose template reads enable_thinking.
#include "server-template-kwargs-force.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); std::exit(1); } \
                         std::printf("  ok:   %s\n", #cond); } while (0)

int main() {
    using kw = std::map<std::string, std::string>;

    // request sent enable_thinking=true, server pins false: force wins
    kw merged = {{"enable_thinking", "true"}, {"x", "1"}};
    server_template_kwargs_apply_force(merged, {{"enable_thinking", "false"}});
    CHECK(merged["enable_thinking"] == "false");
    CHECK(merged["x"] == "1");

    // key only in force is added
    kw added;
    server_template_kwargs_apply_force(added, {{"reasoning_effort", "\"low\""}});
    CHECK(added["reasoning_effort"] == "\"low\"");

    // empty force changes nothing (default semantics unchanged)
    kw same = {{"a", "1"}};
    server_template_kwargs_apply_force(same, {});
    CHECK(same.size() == 1 && same["a"] == "1");

    std::printf("PASSED\n");
    return 0;
}
