// "%xgrammar <kind> [any_whitespace] [max_whitespace=N]" header parsing (works without the XGrammar library).
// Row S69-ENGREV-SGLANG-XGRAMMAR-MAX-WHITESPACE-2026-10-27-01.
#include "sampling.h"

#include <cstdio>
#include <string>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
}

int main() {
    std::string kind, err;
    bool any_ws = true;
    int  max_ws = 0;

    check(common_xgrammar_parse_header("%xgrammar json_schema", kind, any_ws, max_ws, err), "plain header parses");
    check(kind == "json_schema" && !any_ws && max_ws == -1, "plain: fixed separators, no cap");

    check(common_xgrammar_parse_header("%xgrammar json any_whitespace", kind, any_ws, max_ws, err), "any_whitespace parses");
    check(kind == "json" && any_ws && max_ws == -1, "any_whitespace: uncapped");

    check(common_xgrammar_parse_header("%xgrammar json_schema max_whitespace=4", kind, any_ws, max_ws, err), "max_whitespace parses");
    check(any_ws && max_ws == 4, "max_whitespace=4 caps at 4 and implies any_whitespace");

    err.clear();
    check(!common_xgrammar_parse_header("%xgrammar json max_whitespace=0", kind, any_ws, max_ws, err), "max_whitespace=0 rejected");
    check(err.find("max_whitespace") != std::string::npos, "error names max_whitespace");
    err.clear();
    check(!common_xgrammar_parse_header("%xgrammar json max_whitespace=x", kind, any_ws, max_ws, err), "non-numeric rejected");
    err.clear();
    check(!common_xgrammar_parse_header("%xgrammar json bogus", kind, any_ws, max_ws, err), "unknown option rejected");
    check(err.find("bogus") != std::string::npos, "error names the unknown option");

    if (g_fail) {
        return 1;
    }
    printf("test-xgrammar-header: OK\n");
    return 0;
}
