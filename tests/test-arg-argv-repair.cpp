// common_params_parse() on Windows swaps argv for the UTF-8 form of the process command line so that
// non-ASCII arguments survive the ANSI code page. It used to swap whenever the argument COUNT matched,
// which also discarded an argv the caller had rewritten one-for-one (llama-tts-outetts maps --out-file
// to --output): the rewritten flag vanished and the raw one was refused.
//
// This program is driven by tests/test-arg-argv-repair.py, which checks both directions:
//   plain   : argv passed through unchanged; prints the hex of params.prompt, which must be the UTF-8
//             bytes of a non-ASCII -p value (the repair still happens).
//   rewrite : env ARGV_REPAIR_REWRITE=1 replaces every "--prompt-alias" with "-p" before parsing; the
//             parse must succeed and use the rewritten argv.
#include "arg.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::vector<char *> args(argv, argv + argc);
    const char * mode = std::getenv("ARGV_REPAIR_REWRITE");
    static char short_prompt[] = "-p";
    if (mode && std::strcmp(mode, "1") == 0) {
        for (auto & a : args) {
            if (std::strcmp(a, "--prompt-alias") == 0) {
                a = short_prompt;
            }
        }
    }
    args.push_back(nullptr);

    common_params params;
    if (!common_params_parse(argc, args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        printf("PARSE_FAILED\n");
        return 2;
    }
    printf("PROMPT_HEX=");
    for (unsigned char c : params.prompt) {
        printf("%02x", c);
    }
    printf("\n");
    return 0;
}
