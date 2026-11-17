// HF tree API pagination: the Hub answers /tree/<commit>?recursive=true in pages and names the next one in a
// `Link: <url>; rel="next"` header. get_repo_files must follow it, or a repo with more files than one page
// silently loses the rest (e.g. the later shards of a split GGUF). A loopback httplib server answers the API;
// NO real network runs.
// Pins: (1) 3 pages are all listed, in order; (2) a single page with no Link header still works;
// (3) a next link to a DIFFERENT host is not followed (the bearer token must not leave the endpoint);
// (4) a Link loop ends instead of spinning.

#include "common.h"
#include "download.h"
#include "hf-cache.h"
#include "http.h"
#include "log.h"

#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#define REQUIRE(x) do {                                                         \
    if (!(x)) {                                                                 \
        fprintf(stderr, "%s:%d: REQUIRE(%s) failed\n", __FILE__, __LINE__, #x); \
        std::abort();                                                           \
    }                                                                           \
} while (0)

static const std::string C_MAIN = std::string(40, 'a');
static int g_port = 0;

static std::string base() { return "http://127.0.0.1:" + std::to_string(g_port); }

static common_json page_of(const std::string & name) {
    auto files = common_json::array();
    files.push_back({{"type", "file"}, {"path", name}, {"size", 1}, {"oid", std::string(40, '1')}});
    return files;
}

static void serve(httplib::Server & server) {
    server.Get(R"(/api/models/(.+)/refs)", [](const httplib::Request &, httplib::Response & res) {
        common_json j = {
            {"branches", common_json::array({ common_json{{"name", "main"}, {"targetCommit", C_MAIN}} })},
            {"tags", common_json::array()},
        };
        res.set_content(j.dump(), "application/json");
    });
    // repo "test/paged": 3 pages; "test/single": 1 page; "test/foreign": next link on another host;
    // "test/loop": page 1 links to itself
    server.Get(R"(/api/models/(test)/(paged|single|foreign|loop)/tree/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string repo   = req.matches[2];
        const std::string commit = req.matches[3];
        const std::string url    = base() + "/api/models/test/" + repo + "/tree/" + commit;
        const std::string cursor = req.has_param("cursor") ? req.get_param_value("cursor") : "";
        if (repo == "single") {
            res.set_content(page_of("one.gguf").dump(), "application/json");
        } else if (repo == "paged") {
            if (cursor.empty()) {
                res.set_header("Link", "<" + url + "?recursive=true&cursor=2>; rel=\"next\"");
                res.set_content(page_of("p1.gguf").dump(), "application/json");
            } else if (cursor == "2") {
                res.set_header("Link", "<" + url + "?recursive=true&cursor=3>; rel=\"next\", <" + url + ">; rel=\"first\"");
                res.set_content(page_of("p2.gguf").dump(), "application/json");
            } else {
                res.set_content(page_of("p3.gguf").dump(), "application/json");
            }
        } else if (repo == "foreign") {
            res.set_header("Link", "<http://example.invalid/api/models/test/foreign/tree/x?cursor=2>; rel=\"next\"");
            res.set_content(page_of("f1.gguf").dump(), "application/json");
        } else {
            res.set_header("Link", "<" + url + "?recursive=true>; rel=\"next\"");
            res.set_content(page_of("l1.gguf").dump(), "application/json");
        }
    });
}

static std::string joined(const std::string & repo) {
    std::string out;
    for (const auto & f : hf_cache::get_repo_files(repo, "")) {
        out += (out.empty() ? "" : ",") + f.path;
    }
    return out;
}

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    common_log_pause(common_log_main());

    httplib::Server server;
    serve(server);
    g_port = server.bind_to_any_port("127.0.0.1");

    std::filesystem::path cache_dir = std::filesystem::temp_directory_path() /
                                      ("test-hf-tree-pagination-cache-" + std::to_string(g_port));
    std::filesystem::remove_all(cache_dir);
    common_set_env("LLAMA_CACHE", cache_dir.string());

    std::thread server_thread([&server] { server.listen_after_bind(); });
    server.wait_until_ready();
    common_set_env("MODEL_ENDPOINT", base() + "/");

    const std::string paged = joined("test/paged");
    fprintf(stderr, "paged   -> '%s'\n", paged.c_str());
    REQUIRE(paged == "p1.gguf,p2.gguf,p3.gguf");
    REQUIRE(joined("test/single") == "one.gguf");
    REQUIRE(joined("test/foreign") == "f1.gguf"); // foreign host link not followed
    REQUIRE(joined("test/loop") == "l1.gguf");    // a repeated page ends the walk

    server.stop();
    server_thread.join();
    std::filesystem::remove_all(cache_dir);
    printf("test-hf-tree-pagination: all tests OK (loopback server only)\n");
    return 0;
}
