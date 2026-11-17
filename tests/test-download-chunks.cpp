// Parallel range download with a bitmap of finished ranges (row S69-ENGINE-MERGE-NEXA-2026-09-27-01; behaviour from NEXA
// SDK's chunked downloader, BSD-3-Clause, reimplemented, no code copied).
//
// Planner and bitmap are checked directly. The download is checked against a real cpp-httplib server on the loopback
// that serves a 4 MiB body with ranges, so the real client, the worker threads and the resume logic run with no network.
//   P  planner tiles [0,total) without gap or overlap, grows the range size past 128 pieces, empty for 0 bytes
//   B  bitmap round-trips; another file size, another range size or a damaged body loads as all-zero
//   A  4 connections, 512 KiB ranges      -> file intact, 8 range GETs, no sidecar left        (the feature)
//   R  seeded partial: ranges 0-3 done    -> only ranges 4-7 are requested, file intact        (the resume)
//   S  stale bitmap (other range size)    -> restarts from scratch, 8 range GETs, file intact  (no glue of foreign bytes)
//   N  server says Accept-Ranges: none    -> single stream, 0 range GETs                       (the fallback)
//   D  chunk_concurrency 1 (default)      -> single stream, 0 range GETs                       (opt-in is real)

#include "download.h"
#include "download-chunk.h"
#include "http.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        printf("  FAIL: %s -- ", #cond);                        \
        printf(__VA_ARGS__);                                    \
        printf("\n");                                           \
        g_fail++;                                               \
    } else {                                                    \
        printf("  ok:   %s\n", #cond);                          \
    }                                                           \
} while (0)

static const size_t BODY_SIZE = 4u << 20;
static const size_t CHUNK     = 512u << 10;

static std::string make_body() {
    std::string b(BODY_SIZE, '\0');
    uint32_t x = 777u;
    for (size_t i = 0; i < b.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        b[i] = (char) (x >> 24);
    }
    return b;
}

static std::string read_all(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main() {
    printf("planner\n");
    {
        const auto plan = common_chunk_plan(10, 4);
        CHECK(plan.size() == 3 && plan[0].offset == 0 && plan[0].len == 4 && plan[2].offset == 8 && plan[2].len == 2,
              "10 bytes in 4s -> 4+4+2");
        size_t next = 0;
        bool tiled = true;
        const auto big = common_chunk_plan((size_t) 5 << 30, (size_t) 16 << 20);
        for (const auto & c : big) { tiled = tiled && c.offset == next && c.len > 0; next += c.len; }
        CHECK(tiled && next == ((size_t) 5 << 30), "5 GiB tiles with no gap and no overlap");
        CHECK(big.size() <= COMMON_CHUNK_MAX_COUNT, "5 GiB in 16 MiB asks 320 pieces, capped to %zu", big.size());
        CHECK(common_chunk_plan(0, 4).empty(), "0 bytes -> no ranges");
        CHECK(common_chunk_plan(100, 0).size() == 1, "chunk size 0 means the 16 MiB default");
    }

    printf("bitmap\n");
    const fs::path dir = fs::temp_directory_path() / ("llama-dl-chunks-" + std::to_string(std::rand()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        const std::string bm = (dir / "bm.chunks").string();
        std::vector<char> in = {1, 0, 1, 1, 0}, out;
        CHECK(common_chunk_bitmap_save(bm, 50, 10, in), "saved");
        CHECK(common_chunk_bitmap_load(bm, 50, 10, 5, out) && out == in, "round-trips");
        CHECK(!common_chunk_bitmap_load(bm, 51, 10, 5, out) && out == std::vector<char>(5, 0), "another file size -> all zero");
        CHECK(!common_chunk_bitmap_load(bm, 50, 11, 5, out), "another range size -> all zero");
        CHECK(!common_chunk_bitmap_load((dir / "none").string(), 50, 10, 5, out), "missing file -> all zero");
        { std::ofstream f(bm, std::ios::trunc); f << "llama-chunks-v1 50 10 5\n10x11\n"; }
        CHECK(!common_chunk_bitmap_load(bm, 50, 10, 5, out) && out == std::vector<char>(5, 0), "damaged body -> all zero");
    }

    const std::string good = make_body();
    std::atomic<int> n_get{0};
    std::atomic<int> n_range_get{0};
    std::mutex       mtx;
    std::vector<long> offsets;
    bool accept_ranges = true;

    httplib::Server server;
    server.Get("/f/model.bin", [&](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Accept-Ranges", accept_ranges ? "bytes" : "none");
        if (req.method != "HEAD") {
            n_get++;
            const std::string range = req.get_header_value("Range");
            if (!range.empty()) {
                n_range_get++;
                std::lock_guard<std::mutex> lk(mtx);
                offsets.push_back(std::strtol(range.c_str() + std::string("bytes=").size(), nullptr, 10));
            }
        }
        // the whole body: cpp-httplib slices it for the Range header and answers 206 with Content-Range
        res.set_content(good, "application/octet-stream");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        printf("cannot bind a loopback port\n");
        return 2;
    }
    std::thread th([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/f/model.bin";

    auto run = [&](const char * name, int conns) {
        n_get = 0;
        n_range_get = 0;
        offsets.clear();
        const fs::path out = dir / name;
        common_download_opts opts;
        opts.chunk_concurrency = conns;
        opts.chunk_size        = CHUNK;
        const int rc = common_download_file_single(url, out.string(), opts, /*skip_etag=*/true);
        struct res_t { int rc; fs::path out; } r{rc, out};
        return r;
    };

    auto seed = [&](const char * name, size_t n_done, size_t bm_chunk) {
        const fs::path out = dir / name;
        fs::remove(out);
        const std::string part = out.string() + ".downloadInProgress";
        std::string body(BODY_SIZE, 'Z');
        body.replace(0, n_done * CHUNK, good, 0, n_done * CHUNK);
        { std::ofstream f(part, std::ios::binary | std::ios::trunc); f << body; }
        std::vector<char> done(BODY_SIZE / bm_chunk, 0);
        for (size_t i = 0; i < n_done && i < done.size(); ++i) { done[i] = 1; }
        common_chunk_bitmap_save(part + COMMON_CHUNK_SIDECAR_SUFFIX, BODY_SIZE, bm_chunk, done);
    };

    printf("case A: 4 connections, 512 KiB ranges\n");
    {
        auto r = run("a.bin", 4);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out) && read_all(r.out) == good, "bytes intact");
        CHECK(n_range_get == 8, "8 range GETs, got %d", (int) n_range_get);
        CHECK(!fs::exists(r.out.string() + ".downloadInProgress.chunks"), "no bitmap sidecar left");
        CHECK(!fs::exists(r.out.string() + ".downloadInProgress"), "no partial left");
    }

    printf("case R: partial holds ranges 0-3, bitmap says so\n");
    {
        seed("r.bin", 4, CHUNK);
        auto r = run("r.bin", 4);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out) && read_all(r.out) == good, "bytes intact");
        CHECK(n_range_get == 4, "only the 4 missing ranges are fetched, got %d", (int) n_range_get);
        long lowest = 0x7fffffffL; // long is 32-bit on Windows
        for (long o : offsets) { lowest = o < lowest ? o : lowest; }
        CHECK(lowest == (long) (4 * CHUNK), "lowest requested offset is %ld", lowest);
    }

    printf("case S: bitmap written for another range size\n");
    {
        seed("s.bin", 4, CHUNK / 2);
        auto r = run("s.bin", 4);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out) && read_all(r.out) == good, "foreign bytes are not glued in");
        CHECK(n_range_get == 8, "restart: 8 range GETs, got %d", (int) n_range_get);
    }

    printf("case N: server announces Accept-Ranges: none\n");
    {
        accept_ranges = false;
        auto r = run("n.bin", 4);
        CHECK(r.rc == 200 && fs::exists(r.out) && read_all(r.out) == good, "single stream completes");
        CHECK(n_range_get == 0, "no range request, got %d", (int) n_range_get);
        accept_ranges = true;
    }

    printf("case D: default concurrency 1\n");
    {
        auto r = run("d.bin", 1);
        CHECK(r.rc == 200 && fs::exists(r.out) && read_all(r.out) == good, "single stream completes");
        CHECK(n_range_get == 0, "no range request, got %d", (int) n_range_get);
    }

    server.stop();
    th.join();
    fs::remove_all(dir);

    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
