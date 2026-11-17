// probe-disk-tier-flag: the real CLI parser reaches the disk tier. --cache-disk-io-chunk MiB goes through common_params_parse into
// params.cache_disk_io_chunk_mib, common_cache_disk_io_chunk_bytes() (the function llama-server calls) turns it into the tier's chunk size,
// and a put/get on that tier then fires the pipelined path. Default OFF is checked too. Row S69-LINK-STORAGE-OFFLOAD-1014.
#include "arg.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s (line %d)\n", what, __LINE__); } } while (0)

static bool parse(std::vector<std::string> args, common_params & p) {
    std::vector<char *> argv;
    for (auto & s : args) { argv.push_back(s.data()); }
    return common_params_parse((int) argv.size(), argv.data(), p, LLAMA_EXAMPLE_SERVER);
}

int main() {
    {
        common_params p;
        CHECK(parse({ "prog" }, p), "parse with no flags");
        CHECK(p.cache_disk_io_chunk_mib == 0 && common_cache_disk_io_chunk_bytes(p) == 0, "default is OFF (0)");
    }
    common_params p;
    CHECK(parse({ "prog", "--cache-disk-dir", "x", "--cache-disk-io-chunk", "1" }, p), "parse --cache-disk-io-chunk 1");
    CHECK(p.cache_disk_io_chunk_mib == 1, "flag value reaches params");
    CHECK(common_cache_disk_io_chunk_bytes(p) == 1024u * 1024u, "params -> chunk bytes");

    const auto dir = std::filesystem::temp_directory_path() / "probe-disk-tier-flag";
    std::filesystem::remove_all(dir);
    common_disk_tier tier(dir.string(), 64u << 20, 0);
    int n = 0; std::string err;
    CHECK(tier.init(n, err), "tier init");
    tier.set_io_chunk_bytes(common_cache_disk_io_chunk_bytes(p));
    std::vector<uint8_t> a(3u << 20), b(1u << 20);
    for (size_t i = 0; i < a.size(); i++) { a[i] = (uint8_t) (i * 13 + 1); }
    for (size_t i = 0; i < b.size(); i++) { b[i] = (uint8_t) (i * 7 + 3); }
    const uint64_t id = tier.put(a, b, 42);
    std::vector<uint8_t> ra, rb;
    CHECK(id != 0 && tier.get(id, 42, ra, rb) && ra == a && rb == b, "round-trip through the flag-configured tier");
    CHECK(tier.stats().n_pipelined_put == 1 && tier.stats().n_pipelined_get == 1 && tier.stats().n_io_chunks == 8,
          "pipelined path FIRED (1 put + 1 get, 4 chunks each)");
    std::filesystem::remove_all(dir);
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
