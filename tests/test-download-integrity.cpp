// A downloaded model file must have the bytes the registry announced before it becomes visible.
//
// common_download_file_single used to rename the finished temp file into place without comparing it with
// anything: a flipped byte, or a transfer that ended short with a clean 200, became the model and the pull
// reported success. Row S69-RAMALAMA-PICK-DOWNLOAD-SHA256-SIZE-VERIFY-2026-09-29-01 (idea from containers/ramalama
// model_store/store.py:236-242 and PR #2917, MIT; reimplemented, no code copied).
//
// A real cpp-httplib server on the loopback serves a 4 MiB body, so the real client, retry loop and range-resume run
// against it with no network. Every case has a partner showing what changed the result:
//   A  correct body, correct digest            -> file appears, bytes intact           (control, passes before and after)
//   B  one flipped byte, correct digest        -> no file, exactly 2 GETs              (before: corrupt file installed)
//   C  flipped byte only on the FIRST response -> file appears after one re-download   (the retry is real)
//   D  announced 4 MiB, first 200 carries 3 MiB-> range-resume completes the file       (before: 3 MiB file installed)
//   E  no digest, correct body                 -> file appears, no hashing required     (the check is opt-in)
//   F  stale partial, 206 starting at byte 0   -> partial discarded, file restarts from 0
//   G  stale partial, server ignores Range     -> a 200 to a Range request drops the partial, file completes
//                                                (row S69-ENGREV-RAMALAMA-DOWNLOAD-RESUME-STALE-PARTIAL-2026-10-27-01)

#include "download.h"
#include "http.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

static std::string make_body() {
    std::string b(BODY_SIZE, '\0');
    uint32_t x = 12345u;
    for (size_t i = 0; i < b.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        b[i] = (char) (x >> 24);
    }
    return b;
}

static std::string sha256_hex(const std::string & s) {
    sha256_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<const unsigned char *>(s.data()), s.size());
    unsigned char d[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, d);
    static const char * hex = "0123456789abcdef";
    std::string out;
    for (unsigned char b : d) { out.push_back(hex[b >> 4]); out.push_back(hex[b & 15]); }
    return out;
}

static std::string read_all(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// what the server does for the file /f/<name>
enum serve_mode { SERVE_GOOD, SERVE_FLIPPED, SERVE_FLIPPED_ONCE, SERVE_SHORT_FIRST, SERVE_BAD_CONTENT_RANGE, SERVE_IGNORE_RANGE };

struct served {
    std::string good;
    std::string flipped;
    std::atomic<int> n_get{0};
    std::atomic<int> n_range_get{0};
    std::atomic<long> last_range_start{-1};
    serve_mode mode = SERVE_GOOD;
    size_t short_len = 0;
};

int main() {
    served s;
    s.good = make_body();
    s.flipped = s.good;
    s.flipped[BODY_SIZE / 2] = (char) (s.flipped[BODY_SIZE / 2] ^ 0x01);
    const std::string good_hex = sha256_hex(s.good);

    httplib::Server server;
    // cpp-httplib routes HEAD to the GET handler and drops the body, keeping Content-Length
    server.Get("/f/model.bin", [&](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Accept-Ranges", "bytes");
        if (req.method == "HEAD") {
            // announces the full length whatever the GET answers
            res.set_content(s.good, "application/octet-stream");
            return;
        }
        s.n_get++;
        const std::string range = req.get_header_value("Range");
        if (!range.empty()) {
            s.n_range_get++;
            const long start = std::strtol(range.c_str() + std::string("bytes=").size(), nullptr, 10);
            s.last_range_start = start;
            if (s.mode == SERVE_BAD_CONTENT_RANGE) {
                // a server that answers 206 but slices from byte 0 whatever was asked
                res.status = 206;
                res.set_header("Content-Range", "bytes 0-" + std::to_string(BODY_SIZE - 1) + "/" + std::to_string(BODY_SIZE));
                res.set_content(s.good, "application/octet-stream");
                return;
            }
            if (s.mode == SERVE_IGNORE_RANGE) {
                // a server that ignores Range: it answers 200 with the whole body. cpp-httplib slices a body for the
                // parsed ranges on its own, so the parsed list is emptied first (the request object is not const).
                const_cast<httplib::Request &>(req).ranges.clear();
                res.set_content(s.good, "application/octet-stream");
                return;
            }
            // the whole body: cpp-httplib itself slices it for the Range header and answers 206 with Content-Range
            res.set_content(s.good, "application/octet-stream");
            return;
        }
        switch (s.mode) {
            case SERVE_GOOD:
            case SERVE_IGNORE_RANGE:
            case SERVE_BAD_CONTENT_RANGE: res.set_content(s.good, "application/octet-stream"); break;
            case SERVE_FLIPPED:       res.set_content(s.flipped, "application/octet-stream"); break;
            case SERVE_FLIPPED_ONCE:  res.set_content(s.n_get == 1 ? s.flipped : s.good, "application/octet-stream"); break;
            case SERVE_SHORT_FIRST:   res.set_content(s.good.substr(0, s.short_len), "application/octet-stream"); break;
        }
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        printf("cannot bind a loopback port\n");
        return 2;
    }
    std::thread th([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    const fs::path dir = fs::temp_directory_path() / ("llama-dl-integrity-" + std::to_string(port));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/f/model.bin";

    auto run = [&](const char * name, serve_mode mode, const std::string & expected, size_t short_len = 0,
                   size_t seed_len = 0) {
        s.mode = mode;
        s.short_len = short_len;
        s.n_get = 0;
        s.n_range_get = 0;
        s.last_range_start = -1;
        const fs::path out = dir / name;
        fs::remove(out);
        fs::remove(out.string() + ".downloadInProgress");
        if (seed_len > 0) {
            // a leftover partial from an earlier run; its bytes are NOT the start of the served file
            std::ofstream seed(out.string() + ".downloadInProgress", std::ios::binary);
            seed << std::string(seed_len, 'Z');
        }
        common_download_opts opts;
        opts.expected_sha256 = expected;
        const int rc = common_download_file_single(url, out.string(), opts, /*skip_etag=*/true);
        struct res_t { int rc; fs::path out; } r{rc, out};
        return r;
    };

    printf("case A: correct body, correct digest\n");
    {
        auto r = run("a.bin", SERVE_GOOD, good_hex);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out), "file installed");
        CHECK(fs::exists(r.out) && read_all(r.out) == s.good, "bytes intact");
        CHECK(s.n_get == 1, "one GET, got %d", (int) s.n_get);
    }

    printf("case B: one flipped byte, correct announced digest\n");
    {
        auto r = run("b.bin", SERVE_FLIPPED, good_hex);
        CHECK(r.rc == -1, "the download fails (rc %d)", r.rc);
        CHECK(!fs::exists(r.out), "NO file at the final path");
        CHECK(!fs::exists(r.out.string() + ".downloadInProgress"), "no partial file left to be resumed");
        CHECK(s.n_get == 2, "exactly one re-download, then give up: %d GETs", (int) s.n_get);
    }

    printf("case C: flipped byte only on the first response\n");
    {
        auto r = run("c.bin", SERVE_FLIPPED_ONCE, good_hex);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out) && read_all(r.out) == s.good, "the re-downloaded file is the good one");
        CHECK(s.n_get == 2, "two GETs, got %d", (int) s.n_get);
    }

    printf("case D: announced 4 MiB, first 200 response carries 3 MiB, no digest\n");
    {
        auto r = run("d.bin", SERVE_SHORT_FIRST, "", 3u << 20);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out) && read_all(r.out) == s.good, "the file was completed to the full 4 MiB and is intact");
        CHECK(s.n_range_get == 1, "one range request, got %d", (int) s.n_range_get);
        CHECK(s.last_range_start == (long) (3u << 20), "it resumed at byte %ld", (long) s.last_range_start);
    }

    printf("case E: no digest, correct body\n");
    {
        auto r = run("e.bin", SERVE_GOOD, "");
        CHECK(r.rc == 200 && fs::exists(r.out) && read_all(r.out) == s.good, "installed without any hashing requirement");
    }

    printf("case F: 1 MiB stale partial, server answers 206 with a Content-Range that starts at 0\n");
    {
        auto r = run("f.bin", SERVE_BAD_CONTENT_RANGE, "", 0, 1u << 20);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out) && read_all(r.out) == s.good,
              "the stale partial is discarded and the file restarts from 0 (size %ld)",
              fs::exists(r.out) ? (long) fs::file_size(r.out) : -1L);
        CHECK(!fs::exists(r.out.string() + ".downloadInProgress"), "no partial left");
    }

    printf("case G: 1 MiB stale partial, server ignores Range and answers 200 with the whole body\n");
    {
        auto r = run("g.bin", SERVE_IGNORE_RANGE, "", 0, 1u << 20);
        CHECK(r.rc == 200, "rc %d", r.rc);
        CHECK(fs::exists(r.out) && read_all(r.out) == s.good,
              "the partial is dropped on the 200 and the file completes from byte 0 (size %ld)",
              fs::exists(r.out) ? (long) fs::file_size(r.out) : -1L);
        CHECK(!fs::exists(r.out.string() + ".downloadInProgress"), "no partial left");
        CHECK(s.n_range_get >= 1, "the first attempt did send a Range request (%d)", (int) s.n_range_get);
    }

    printf("helpers\n");
    CHECK(common_is_sha256_hex(good_hex), "a 64-hex string is a sha256");
    CHECK(!common_is_sha256_hex(good_hex.substr(0, 40)), "a 40-hex git blob id is not");
    CHECK(!common_is_sha256_hex(std::string(63, 'a') + "z"), "non-hex characters are not");
    {
        std::string hex;
        CHECK(common_file_sha256_hex((dir / "a.bin").string(), hex) && hex == good_hex, "file hash equals the in-memory hash");
        CHECK(!common_file_sha256_hex((dir / "missing.bin").string(), hex), "a missing file is reported, not hashed as empty");
    }

    server.stop();
    th.join();
    fs::remove_all(dir);

    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
