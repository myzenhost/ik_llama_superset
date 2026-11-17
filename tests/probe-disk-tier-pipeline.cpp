// probe-disk-tier-pipeline: --cache-disk-io-chunk, the pipelined (read/checksum overlap) path of common_disk_tier.
// No model: payloads are patterned bytes. Prints "N checks, M failed", exits non-zero on a failure. Links only disk-tier.cpp.
// Row S69-LINK-STORAGE-OFFLOAD-1014. Idea from DeepSpeed csrc/aio/common/deepspeed_aio_common.cpp:192-233 (I/O kept in flight while
// other work proceeds), applied to the tier's checksum stage; no code copied.
#include "disk-tier.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_checks = 0;
static int g_fails  = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s (line %d)\n", what, __LINE__); } } while (0)

static std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) { v[i] = (uint8_t) (seed + i * 31 + (i >> 8) * 7); }
    return v;
}

static std::vector<char> slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void flip_byte(const std::string & path, std::streamoff at) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(at); char c = 0; f.read(&c, 1); c = (char) (c ^ 0x5a);
    f.seekp(at); f.write(&c, 1);
}

static size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

int main() {
    const auto root = fs::temp_directory_path() / "probe-disk-tier-pipeline";
    fs::remove_all(root);

    // ---- the incremental checksum equals the one-shot one for every length, split and chunk ------------------------------
    {
        bool all_equal = true;
        const size_t lens[]   = { 0, 1, 7, 8, 31, 32, 33, 63, 64, 65, 100, 1000, 4097 };
        const size_t chunks[] = { 1, 7, 32, 33, 64, 1000, 5000 };
        for (size_t na : lens) for (size_t nb : lens) for (size_t c : chunks) {
            auto a = pattern(na, 3), b = pattern(nb, 9);
            const uint64_t one = common_disk_tier::checksum(b.data(), b.size(), common_disk_tier::checksum(a.data(), a.size(), 12345));
            const uint64_t chk = common_disk_tier::checksum_chunked(a.data(), a.size(), b.data(), b.size(), 12345, c);
            if (one != chk) { all_equal = false; }
        }
        CHECK(all_equal, "checksum_chunked == one-shot checksum for all lengths/splits/chunks");
    }

    const size_t CH = 64 * 1024;
    auto a = pattern(3 * CH + 1234, 5);   // 4 chunks
    auto b = pattern(2 * CH + 17, 11);    // 3 chunks
    const size_t chunks_expected = ceil_div(a.size(), CH) + ceil_div(b.size(), CH);

    // ---- flag OFF: serial path, counters stay 0 ----------------------------------------------------------------------------
    common_disk_tier t_off((root / "off").string(), 64u << 20, 0);
    int n_stale = 0; std::string err;
    CHECK(t_off.init(n_stale, err), "init off");
    const uint64_t id_off = t_off.put(a, b, 77);
    CHECK(id_off != 0, "serial put");
    {
        std::vector<uint8_t> ra, rb;
        CHECK(t_off.get(id_off, 77, ra, rb) && ra == a && rb == b, "serial get round-trips");
    }
    CHECK(t_off.stats().n_pipelined_put == 0 && t_off.stats().n_pipelined_get == 0 && t_off.stats().n_io_chunks == 0,
          "flag off: no pipelined counters move");

    // ---- flag ON: the pipelined path FIRES, the file is byte-identical, the data round-trips --------------------------------
    common_disk_tier t_on((root / "on").string(), 64u << 20, 0);
    CHECK(t_on.init(n_stale, err), "init on");
    t_on.set_io_chunk_bytes(CH);
    const uint64_t id_on = t_on.put(a, b, 77);
    CHECK(id_on != 0, "pipelined put");
    CHECK(t_on.stats().n_pipelined_put == 1, "pipelined put fired (counter)");
    CHECK(t_on.stats().n_io_chunks == chunks_expected, "pipelined put moved the expected number of chunks");
    {
        const auto f_off = slurp(t_off.path_of(id_off)), f_on = slurp(t_on.path_of(id_on));
        CHECK(!f_off.empty() && f_off == f_on, "file written by the pipelined put is byte-identical to the serial one");
    }
    {
        std::vector<uint8_t> ra, rb;
        CHECK(t_on.get(id_on, 77, ra, rb) && ra == a && rb == b, "pipelined get round-trips");
        CHECK(t_on.stats().n_pipelined_get == 1, "pipelined get fired (counter)");
        CHECK(t_on.stats().n_io_chunks == 2 * chunks_expected, "pipelined get moved the expected number of chunks");
    }
    // cross-compat: serial get of the pipelined file, pipelined get of the serial file
    {
        t_on.set_io_chunk_bytes(0);
        std::vector<uint8_t> ra, rb;
        CHECK(t_on.get(id_on, 77, ra, rb) && ra == a && rb == b, "serial get reads the pipelined file");
        t_on.set_io_chunk_bytes(CH);
        t_off.set_io_chunk_bytes(CH);
        ra.clear(); rb.clear();
        CHECK(t_off.get(id_off, 77, ra, rb) && ra == a && rb == b, "pipelined get reads the serial file");
    }
    // a chunk size that is not a multiple of the 32-byte checksum stripe: the stream must carry bytes across chunks, and the file must
    // still verify against the serial reader
    {
        t_on.set_io_chunk_bytes(CH + 5);
        const uint64_t id = t_on.put(a, b, 31);
        std::vector<uint8_t> ra, rb;
        CHECK(id != 0 && t_on.get(id, 31, ra, rb) && ra == a && rb == b, "odd chunk size: pipelined round-trip");
        t_on.set_io_chunk_bytes(0);
        ra.clear(); rb.clear();
        CHECK(t_on.get(id, 31, ra, rb) && ra == a && rb == b, "odd chunk size: serial get verifies the pipelined file's checksum");
        t_on.set_io_chunk_bytes(CH);
    }
    // wrong tag is a miss, not data
    {
        std::vector<uint8_t> ra, rb;
        const uint64_t id = t_on.put(a, b, 5);
        CHECK(id != 0 && !t_on.get(id, 6, ra, rb) && ra.empty() && rb.empty(), "wrong tag: miss on the pipelined path");
    }

    // ---- small payload and empty b are not pipelined (or still correct) ---------------------------------------------------
    {
        const uint64_t before = t_on.stats().n_pipelined_put;
        auto s = pattern(1000, 1);
        const uint64_t id = t_on.put(s, {}, 1);
        std::vector<uint8_t> ra, rb;
        CHECK(id != 0 && t_on.stats().n_pipelined_put == before, "payload below one chunk stays serial");
        CHECK(t_on.get(id, 1, ra, rb) && ra == s && rb.empty(), "small payload round-trips");
        auto big = pattern(5 * CH, 2);
        const uint64_t id2 = t_on.put(big, {}, 2);
        ra.clear();
        CHECK(id2 != 0 && t_on.get(id2, 2, ra, rb) && ra == big && rb.empty(), "pipelined with empty b round-trips");
    }

    // ---- corruption is still caught on the pipelined path (a byte flipped in `a`, in `b`, and in the last chunk) -------------
    for (int which = 0; which < 3; which++) {
        const uint64_t id = t_on.put(a, b, 9);
        const std::streamoff header = 4 + 4 + 8 + 8 + 8 + 8;
        const std::streamoff at = which == 0 ? header + 10 : which == 1 ? header + (std::streamoff) a.size() + 5
                                                                        : header + (std::streamoff) (a.size() + b.size()) - 1;
        flip_byte(t_on.path_of(id), at);
        const uint64_t cs_before = t_on.stats().n_checksum_failed;
        std::vector<uint8_t> ra, rb;
        CHECK(!t_on.get(id, 9, ra, rb), "corrupted blob is a miss (pipelined)");
        CHECK(t_on.stats().n_checksum_failed == cs_before + 1, "corruption counted as a checksum failure (pipelined)");
        CHECK(!t_on.has(id) && ra.empty() && rb.empty(), "corrupted blob deleted, nothing handed out (pipelined)");
    }

    // ---- truncation is still caught -------------------------------------------------------------------------------------------
    {
        const uint64_t id = t_on.put(a, b, 3);
        fs::resize_file(t_on.path_of(id), fs::file_size(t_on.path_of(id)) - 100);
        std::vector<uint8_t> ra, rb;
        CHECK(!t_on.get(id, 3, ra, rb) && !t_on.has(id), "truncated blob is a miss (pipelined)");
    }

    fs::remove_all(root);
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
