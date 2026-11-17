// The Hugging Face token also lives in a file: huggingface_hub (and `hf auth login`) write
// ~/.cache/huggingface/token, so a user logged in there needs no HF_TOKEN. Lookup order:
//   1 HF_TOKEN_PATH (file)   2 $HF_HOME/token   3 ~/.cache/huggingface/token
// The first file that holds a non-blank token wins; whitespace around it is trimmed.
// Row S69-ENGREV-RAMALAMA-HF-FETCH-HARDENING-2026-10-27-01 (c); idea from containers/ramalama
// a2870d20882f (MIT), behaviour only, own code.

#include "arg.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond) do {                                  \
    if (!(cond)) { printf("  FAIL: %s\n", #cond); g_fail++; } \
    else         { printf("  ok:   %s\n", #cond); }          \
} while (0)

static void set_env(const char * k, const std::string & v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());
#else
    if (v.empty()) { unsetenv(k); } else { setenv(k, v.c_str(), 1); }
#endif
}

static void put(const fs::path & p, const std::string & s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << s;
}

int main() {
    const fs::path root = fs::temp_directory_path() / ("test-hf-token-file-" + std::to_string((long long) std::rand()));
    fs::create_directories(root);

    set_env("HF_TOKEN_PATH", "");
    set_env("HF_HOME", "");

    // A: HF_HOME/token, whitespace trimmed
    put(root / "home" / "token", "  hf_homefile \r\n");
    set_env("HF_HOME", (root / "home").string());
    CHECK(common_hf_token_from_files() == "hf_homefile");

    // B: HF_TOKEN_PATH wins over HF_HOME
    put(root / "explicit", "hf_explicit\n");
    set_env("HF_TOKEN_PATH", (root / "explicit").string());
    CHECK(common_hf_token_from_files() == "hf_explicit");

    // C: a blank file falls through to the next candidate
    put(root / "explicit", " \n");
    CHECK(common_hf_token_from_files() == "hf_homefile");

    // D: nothing readable at the env locations -> no crash (the real home may or may not hold a token)
    set_env("HF_TOKEN_PATH", (root / "missing").string());
    set_env("HF_HOME", (root / "nohome").string());
    (void) common_hf_token_from_files();
    CHECK(true);

    set_env("HF_TOKEN_PATH", "");
    set_env("HF_HOME", "");
    std::error_code ec;
    fs::remove_all(root, ec);

    printf(g_fail ? "FAIL -- %d failure(s)\n" : "PASS -- 0 failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
