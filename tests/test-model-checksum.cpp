// --model-checksum: common_verify_model_checksum compares a file's SHA-256 with the expected digest and names both on mismatch.
// Row S69-ENGREV-SGLANG-MODEL-CHECKSUM-2026-10-27-01.
#include "common.h"

#include <cstdio>
#include <fstream>
#include <string>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
}

int main() {
    const std::string path = "test-model-checksum.tmp";
    {
        std::ofstream f(path, std::ios::binary);
        f << "abc";
    }
    // FIPS 180-2 test vector for "abc"
    const std::string good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    std::string up = good;
    for (char & c : up) { if (c >= 'a' && c <= 'f') c = (char) (c - 'a' + 'A'); }
    const std::string bad  = std::string(64, '0');

    std::string err;
    check(common_verify_model_checksum(path, good, err), "matching digest accepted");
    err.clear();
    check(common_verify_model_checksum(path, up, err), "upper-case digest accepted");

    err.clear();
    check(!common_verify_model_checksum(path, bad, err), "wrong digest rejected");
    check(err.find(bad)  != std::string::npos, "error names the expected digest");
    check(err.find(good) != std::string::npos, "error names the actual digest");

    err.clear();
    check(!common_verify_model_checksum(path, "xyz", err), "non-hex expected value rejected");
    err.clear();
    check(!common_verify_model_checksum("no-such-model-file.gguf", good, err), "unreadable file rejected");
    check(err.find("cannot read") != std::string::npos, "unreadable file says so");

    std::remove(path.c_str());
    if (g_fail) {
        return 1;
    }
    printf("test-model-checksum: OK\n");
    return 0;
}
