// finalize_file: when a symlink cannot be made (Windows without the privilege, some mounts) it used to
// MOVE the blob into snapshots/, leaving blobs/<oid> empty and the cache layout unlike huggingface_hub's.
// A hard link keeps both the blob and the snapshot entry and shares the bytes. Order: symlink, hard
// link, rename, copy. Row S69-ENGREV-RAMALAMA-HF-FETCH-HARDENING-2026-10-27-01 (d); idea from
// containers/ramalama a2870d20882f (MIT), behaviour only, own code.

#include "hf-cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond) do {                                  \
    if (!(cond)) { printf("  FAIL: %s\n", #cond); g_fail++; } \
    else         { printf("  ok:   %s\n", #cond); }          \
} while (0)

static std::string slurp(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main() {
    const fs::path root = fs::temp_directory_path() / ("test-hf-finalize-hardlink-" + std::to_string((long long) std::rand()));
    fs::create_directories(root);
#ifdef _WIN32
    _putenv_s("LLAMA_CACHE", root.string().c_str());
#else
    setenv("LLAMA_CACHE", root.string().c_str(), 1);
#endif

    const std::string oid = "0123456789abcdef0123456789abcdef01234567";
    const fs::path repo = root / "models--o--r";
    const fs::path blob = repo / "blobs" / oid;
    const fs::path fin  = repo / "snapshots" / "c0ffee" / "m.gguf";
    fs::create_directories(blob.parent_path());
    { std::ofstream(blob, std::ios::binary) << "GGUFDATA"; }

    hf_cache::hf_file f;
    f.path       = "m.gguf";
    f.local_path = blob.string();
    f.final_path = fin.string();
    f.oid        = oid;
    f.repo_id    = "o/r";

    hf_cache::finalize_file(f);

    std::error_code ec;
    CHECK(fs::exists(fin, ec));
    CHECK(slurp(fin) == "GGUFDATA");
    CHECK(fs::exists(blob, ec));          // the blob must survive (link, not move)
    CHECK(slurp(blob) == "GGUFDATA");

    fs::remove_all(root, ec);
    printf(g_fail ? "FAIL -- %d failure(s)\n" : "PASS -- 0 failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
