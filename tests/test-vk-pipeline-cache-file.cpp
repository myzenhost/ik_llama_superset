// test-vk-pipeline-cache-file.cpp -- ggml/src/ggml-vulkan/pipeline-cache-file.h (no Vulkan, no GPU).
// Row S69-ncnn-PICK-VULKAN-PIPELINE-CACHE-2026-09-29-01.
//
// The spec-addition test the row asks for: corrupt one header byte and one blob byte and expect a clean cold
// start (a non-ok status, never a crash and never stale reuse). Each non-ok status is checked by NAME where it
// is determined, so a loader that failed everything for the wrong reason would read differently from one that
// fails the right things. Control: the unmodified file must round-trip to status::ok with a byte-identical payload.

#include "../ggml/src/ggml-vulkan/pipeline-cache-file.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>

using namespace ggml_vk_pc;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CHECK_ST(got, want) do { if ((got) != (want)) { std::fprintf(stderr, "FAIL %s:%d: status %s, wanted %s\n", __FILE__, __LINE__, status_str(got), status_str(want)); ++g_fail; } } while (0)

static identity make_id() {
    identity id;
    id.vendor_id = 0x10de; id.device_id = 0x2b85; id.api_version = (1u << 22) | (4u << 12) | 357u;
    id.driver_version = 0x24a0c000; id.driver_id = 4;
    id.device_name_hash = hash_name("NVIDIA GeForce RTX 5090"); id.driver_name_hash = hash_name("NVIDIA");
    for (int i = 0; i < 16; ++i) id.uuid[i] = (uint8_t)(0xA0 + i);
    return id;
}

// a conformant blob: 32-byte VkPipelineCacheHeaderVersionOne followed by opaque driver data
static std::vector<uint8_t> make_blob(const identity & id, size_t total) {
    std::vector<uint8_t> b(total);
    std::mt19937 rng(5);
    for (auto & x : b) x = (uint8_t) rng();
    const uint32_t hs = 32, hv = 1;
    std::memcpy(b.data(), &hs, 4); std::memcpy(b.data() + 4, &hv, 4);
    std::memcpy(b.data() + 8, &id.vendor_id, 4); std::memcpy(b.data() + 12, &id.device_id, 4);
    std::memcpy(b.data() + 16, id.uuid, 16);
    return b;
}

static void test_roundtrip_and_failures() {
    const identity id = make_id();
    const auto blob = make_blob(id, 5000);
    const auto file = pack(id, blob.data(), blob.size());
    const uint8_t * p = nullptr; size_t pn = 0;

    CHECK_ST(unpack(id, file.data(), file.size(), &p, &pn), status::ok);          // control: untouched file loads
    CHECK(pn == blob.size() && std::memcmp(p, blob.data(), pn) == 0);

    // one flipped byte anywhere in the HEADER region gives a non-ok status (there is no ignored byte: padding and
    // reserved words must be zero)
    for (size_t i = 0; i < sizeof(file_header); ++i) {
        auto f = file; f[i] ^= 0x01;
        const status s = unpack(id, f.data(), f.size(), &p, &pn);
        CHECK(s != status::ok && s != status::empty && p == nullptr && pn == 0);
    }
    CHECK(sizeof(file_header) == 8 + 4*4 + 5*4 + 2*4 + 16 + 4 + 8 + 8 + 8*4);    // no implicit padding
    // one flipped byte anywhere in the BLOB is caught as damage -- it never reaches the driver
    for (size_t i = 0; i < blob.size(); i += 97) {
        auto f = file; f[sizeof(file_header) + i] ^= 0x80;
        const status s = unpack(id, f.data(), f.size(), &p, &pn);
        CHECK(s == status::bad_hash && p == nullptr);
    }
    // damage whose hash was REPAIRED (a writer that hashed after corrupting) is still caught by the Vulkan header check
    {
        auto bad = blob; bad[8] ^= 0xFF; // vendorID inside the blob
        const auto f = pack(id, bad.data(), bad.size());
        CHECK_ST(unpack(id, f.data(), f.size(), &p, &pn), status::bad_vk_header);
        auto bad2 = blob; bad2[20] ^= 0xFF; // uuid byte
        const auto f2 = pack(id, bad2.data(), bad2.size());
        CHECK_ST(unpack(id, f2.data(), f2.size(), &p, &pn), status::bad_vk_header);
        auto bad3 = blob; bad3[4] = 2;      // header version
        const auto f3 = pack(id, bad3.data(), bad3.size());
        CHECK_ST(unpack(id, f3.data(), f3.size(), &p, &pn), status::bad_vk_header);
        auto bad4 = blob; bad4[0] = 8;      // header size below 32
        const auto f4 = pack(id, bad4.data(), bad4.size());
        CHECK_ST(unpack(id, f4.data(), f4.size(), &p, &pn), status::bad_vk_header);
    }
    // truncation at several lengths
    for (size_t cut : {size_t(0), size_t(1), sizeof(file_header) - 1, sizeof(file_header), sizeof(file_header) + 100, file.size() - 1}) {
        const status s = unpack(id, file.data(), cut, &p, &pn);
        CHECK(s != status::ok && s != status::empty);
    }
    CHECK_ST(unpack(id, file.data(), 10, &p, &pn), status::file_too_small);
    CHECK_ST(unpack(id, file.data(), file.size() - 1, &p, &pn), status::size_mismatch);
    { auto f = file; f.push_back(0); CHECK_ST(unpack(id, f.data(), f.size(), &p, &pn), status::size_mismatch); } // trailing junk
    { const auto f = pack(id, nullptr, 0); CHECK_ST(unpack(id, f.data(), f.size(), &p, &pn), status::empty); CHECK(pn == 0); } // cold start
    { std::vector<uint8_t> big(1); CHECK_ST(unpack(id, big.data(), MAX_FILE_BYTES + 1, &p, &pn), status::file_too_large); }  // size is judged before reading
}

static void test_identity_each_field() {
    const identity id = make_id();
    const auto blob = make_blob(id, 300);
    const auto file = pack(id, blob.data(), blob.size());
    const uint8_t * p; size_t pn;
    struct Mut { const char * what; void (*fn)(identity &); };
    const Mut muts[] = {
        {"vendor",         [](identity & i) { i.vendor_id ^= 1; }},
        {"device",         [](identity & i) { i.device_id ^= 1; }},
        {"api version",    [](identity & i) { i.api_version += 1; }},
        {"driver version", [](identity & i) { i.driver_version += 1; }},   // a driver update that keeps the UUID
        {"driver id",      [](identity & i) { i.driver_id += 1; }},
        {"device name",    [](identity & i) { i.device_name_hash ^= 1; }},
        {"driver name",    [](identity & i) { i.driver_name_hash ^= 1; }},
        {"uuid",           [](identity & i) { i.uuid[15] ^= 1; }},
    };
    for (const auto & m : muts) {
        identity other = id; m.fn(other);
        const status s = unpack(other, file.data(), file.size(), &p, &pn);
        if (s != status::identity_mismatch) { std::fprintf(stderr, "FAIL identity change '%s' gave %s\n", m.what, status_str(s)); ++g_fail; }
    }
}

static void test_adreno_gate() {
    identity a = make_id();
    a.vendor_id = 0x5143; a.api_version = (1u << 22) | 65u;
    CHECK(driver_blocks_cache(a));
    a.api_version = (1u << 22) | 66u;
    CHECK(!driver_blocks_cache(a));
    a.api_version = (1u << 22) | (1u << 12);   // 1.1.0
    CHECK(!driver_blocks_cache(a));
    identity n = make_id(); n.api_version = (1u << 22) | 10u;
    CHECK(!driver_blocks_cache(n));            // only Qualcomm is gated
}

static void test_files_and_paths() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "ggml_vk_pc_test_dir";
    std::error_code ec;
    fs::remove_all(dir, ec);
    const identity id = make_id();
    const auto blob = make_blob(id, 4096);
    const auto file = pack(id, blob.data(), blob.size());

    const std::string target = (dir / "sub" / "cache.bin").generic_string();
    CHECK(write_atomic(target, file.data(), file.size()));                 // creates the directories
    std::vector<uint8_t> back; bool big = true;
    CHECK(read_file(target, back, &big) && !big && back == file);
    // replacing an existing file works and leaves no temp file behind
    const auto blob2 = make_blob(id, 1000);
    const auto file2 = pack(id, blob2.data(), blob2.size());
    CHECK(write_atomic(target, file2.data(), file2.size()));
    CHECK(read_file(target, back, &big) && back == file2);
    int n_files = 0;
    for (auto & e : fs::directory_iterator(dir / "sub")) { (void) e; ++n_files; }
    CHECK(n_files == 1);
    CHECK(!read_file((dir / "nope.bin").generic_string(), back, &big));    // a missing file is a clean false
    CHECK(!write_atomic((dir / "sub" / "cache.bin" / "x.bin").generic_string(), file.data(), file.size())); // a path through a regular file

    // resolve_path
    const std::string def = default_cache_dir("C:/Users/x/AppData/Local", "/home/x/.cache", "/home/x");
    CHECK(resolve_path(nullptr, id, def).empty());
    CHECK(resolve_path("", id, def).empty());
    CHECK(resolve_path("0", id, def).empty());
    CHECK(resolve_path("1", id, def) == def + "/" + default_file_name(id));
    CHECK(resolve_path("1", id, "").empty());                               // no default dir -> not used
    CHECK(resolve_path((dir / "sub").generic_string().c_str(), id, def) == (dir / "sub").generic_string() + "/" + default_file_name(id)); // existing dir
    CHECK(resolve_path((dir / "newdir/").generic_string().c_str(), id, def) == (dir / "newdir/").generic_string() + default_file_name(id)); // trailing slash
    CHECK(resolve_path("some/file.bin", id, def) == "some/file.bin");       // a file path is taken as is
    identity other = id; other.device_id = 0x1234;
    CHECK(default_file_name(id) != default_file_name(other));               // file names differ per device
    identity other2 = id; other2.uuid[0] ^= 1;
    CHECK(default_file_name(id) != default_file_name(other2));
    fs::remove_all(dir, ec);
}

static void test_hashes() {
    CHECK(fnv1a32("", 0) == 2166136261u);
    CHECK(fnv1a32("a", 1) == 0xe40c292cu);                 // published FNV-1a 32 test vector
    CHECK(fnv1a64("a", 1) == 0xaf63dc4c8601ec8cull);       // published FNV-1a 64 test vector
    CHECK(hash_name(nullptr) == 0);
}

int main() {
    test_hashes();
    test_roundtrip_and_failures();
    test_identity_each_field();
    test_adreno_gate();
    test_files_and_paths();
    if (g_fail) { std::fprintf(stderr, "test-vk-pipeline-cache-file: %d FAILURES\n", g_fail); return 1; }
    std::printf("test-vk-pipeline-cache-file: PASS\n");
    return 0;
}
