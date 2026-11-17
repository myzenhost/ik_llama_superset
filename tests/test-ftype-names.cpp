// Every llama_ftype declared in include/llama.h must have a name in llama_model_ftype_name.
//
// A missing case falls to the default and a model of that type prints "unknown, may not work" at load, which
// reads as a broken file. The test parses llama.h at run time, so a new ftype added there without a name fails
// here without anyone having to remember this list.
//
// Thireus/ik_llama.cpp 77eff8798 (ik_llama #2596, MIT) added the names for IQ4_KS_R16, MXFP4_R8, PQ2_0_R8 and
// PTQ1_0_R8; this fork had MXFP4_R8 only. SOURCE-READ: git show 77eff8798 -- src/llama-model.cpp.
// Row S69-DECLINE-COEXISTENCE-CHALLENGE-BATCH-2026-10-17-01 (Thireus 77eff8798 remainder).

#include "llama.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <fstream>
#include <regex>
#include <string>

#ifndef LLAMA_H_PATH
#    error "LLAMA_H_PATH must name include/llama.h"
#endif

int main() {
    std::ifstream in(LLAMA_H_PATH);
    if (!in) {
        printf("FAIL  cannot open %s\n", LLAMA_H_PATH);
        return 1;
    }
    const std::regex re(R"(^\s*(LLAMA_FTYPE_[A-Z0-9_]+)\s*=\s*(\d+)\s*,)");
    // An out-of-range value BELOW LLAMA_FTYPE_GUESSED (1024): a value with that bit set reads as
    // "unknown, may not work (guessed)" and would match nothing. Checked below not to be a real ftype.
    const int out_of_range = 1000;
    const std::string unknown = llama_model_ftype_name((llama_ftype) out_of_range);
    auto is_unknown = [](const std::string & s) { return s.find("unknown") != std::string::npos; };
    bool out_of_range_declared = false;

    int n_seen = 0, n_fail = 0;
    std::string line;
    while (std::getline(in, line)) {
        std::smatch m;
        if (!std::regex_search(line, m, re)) {
            continue;
        }
        const std::string id = m[1];
        if (id == "LLAMA_FTYPE_GUESSED") {
            continue;   // a loader flag bit, not a file type
        }
        ++n_seen;
        const int value = std::stoi(m[2]);
        out_of_range_declared |= value == out_of_range;
        const std::string name = llama_model_ftype_name((llama_ftype) value);
        if (is_unknown(name)) {
            ++n_fail;
            printf("FAIL  %s = %s has no name (prints \"%s\")\n", id.c_str(), std::string(m[2]).c_str(), name.c_str());
        }
    }

    // controls: the parse must find the enum (a broken regex would pass vacuously), and an out-of-range value
    // must still read as unknown (a default that names everything would pass vacuously too)
    int n_ctl = 0;
    if (n_seen < 50) {
        printf("FAIL  control: parsed only %d ftypes from %s\n", n_seen, LLAMA_H_PATH);
        ++n_ctl;
    }
    if (out_of_range_declared) {
        printf("FAIL  control: %d is a declared ftype; pick another out-of-range value\n", out_of_range);
        ++n_ctl;
    }
    if (!is_unknown(unknown)) {
        printf("FAIL  control: out-of-range ftype reads \"%s\", not unknown\n", unknown.c_str());
        ++n_ctl;
    }
    if (is_unknown(llama_model_ftype_name(LLAMA_FTYPE_MOSTLY_Q8_0))) {
        printf("FAIL  control: Q8_0 has no name\n");
        ++n_ctl;
    }

    printf("%d ftypes parsed, %d without a name, %d control failure(s)\n", n_seen, n_fail, n_ctl);
    printf("%s\n", n_fail + n_ctl ? "FAILED" : "PASSED");
    return n_fail + n_ctl ? 1 : 0;
}
