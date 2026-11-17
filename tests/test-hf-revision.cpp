// --hf-revision: a Hugging Face repo can be pinned to a branch, a tag or a 40-hex commit.
// A loopback httplib server answers the HF API (/refs and /tree/<commit>) and records every request,
// so the URL the client builds is asserted exactly; NO real download and no external network runs.
// Pins: (1) no revision = main's commit; (2) branch, tag and commit each resolve to their own commit and
// the tree request carries THAT commit; (3) a revision that names nothing yields no plan (never main);
// (4) the flags --hf-revision / --spec-draft-hf-revision fill common_params_model::hf_revision.

#include "arg.h"
#include "common.h"
#include "download.h"
#include "http.h"
#include "log.h"

#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define REQUIRE(x) do {                                                         \
    if (!(x)) {                                                                 \
        fprintf(stderr, "%s:%d: REQUIRE(%s) failed\n", __FILE__, __LINE__, #x); \
        std::abort();                                                           \
    }                                                                           \
} while (0)

#define REQUIRE_EQ(actual, expected) do {                                       \
    if (!((actual) == (expected))) {                                            \
        fprintf(stderr, "%s:%d: REQUIRE_EQ(%s, %s) failed\n  actual:   '%s'\n  expected: '%s'\n", \
                __FILE__, __LINE__, #actual, #expected,                         \
                std::string(actual).c_str(), std::string(expected).c_str());    \
        std::abort();                                                           \
    }                                                                           \
} while (0)

static const std::string C_MAIN = std::string(40, 'a');
static const std::string C_DEV  = std::string(40, 'b');
static const std::string C_TAG  = std::string(40, 'c');
static const std::string C_RAW  = std::string(40, 'd'); // a commit the refs listing does not mention

static std::mutex               g_mu;
static std::vector<std::string> g_requests; // request paths, in order

static std::string file_for(const std::string & commit) {
    // each commit ships a differently named file, so the plan shows which commit was listed
    return "model-" + commit.substr(0, 1) + "-Q4_K_M.gguf";
}

static void serve(httplib::Server & server) {
    server.Get(R"(/api/models/(.+)/refs)", [](const httplib::Request & req, httplib::Response & res) {
        { std::lock_guard<std::mutex> lk(g_mu); g_requests.push_back(req.path); }
        common_json j = {
            {"branches", common_json::array({
                common_json{{"name", "main"}, {"targetCommit", C_MAIN}},
                common_json{{"name", "dev"},  {"targetCommit", C_DEV}},
            })},
            {"tags", common_json::array({ common_json{{"name", "v1.0"}, {"targetCommit", C_TAG}} })},
        };
        res.set_content(j.dump(), "application/json");
    });
    server.Get(R"(/api/models/(.+)/tree/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        { std::lock_guard<std::mutex> lk(g_mu); g_requests.push_back(req.path); }
        const std::string commit = req.matches[2];
        auto files = common_json::array();
        files.push_back({{"type", "file"}, {"path", file_for(commit)}, {"size", 1}, {"oid", std::string(40, '1')}});
        res.set_content(files.dump(), "application/json");
    });
}

static std::string last_tree_request() {
    std::lock_guard<std::mutex> lk(g_mu);
    for (size_t i = g_requests.size(); i-- > 0;) {
        if (g_requests[i].find("/tree/") != std::string::npos) {
            return g_requests[i];
        }
    }
    return "";
}

static bool saw_refs_request() {
    std::lock_guard<std::mutex> lk(g_mu);
    for (const auto & r : g_requests) {
        if (r.size() >= 5 && r.compare(r.size() - 5, 5, "/refs") == 0) {
            return true;
        }
    }
    return false;
}

// resolves one repo at a revision, returns the primary file's path ("" = no plan)
static std::string plan_file(const std::string & revision) {
    { std::lock_guard<std::mutex> lk(g_mu); g_requests.clear(); }
    common_params_model m;
    m.hf_repo     = "test/repo";
    m.hf_revision = revision;
    common_download_opts opts;
    auto plan = common_download_get_hf_plan(m, opts);
    return plan.primary.path;
}

static void test_resolution() {
    // 1. no revision: main
    REQUIRE_EQ(plan_file(""), file_for(C_MAIN));
    REQUIRE_EQ(last_tree_request(), "/api/models/test/repo/tree/" + C_MAIN);

    // 2. a non-main branch, a tag, and a raw commit
    REQUIRE_EQ(plan_file("dev"), file_for(C_DEV));
    REQUIRE_EQ(last_tree_request(), "/api/models/test/repo/tree/" + C_DEV);

    REQUIRE_EQ(plan_file("v1.0"), file_for(C_TAG));
    REQUIRE_EQ(last_tree_request(), "/api/models/test/repo/tree/" + C_TAG);

    REQUIRE_EQ(plan_file(C_RAW), file_for(C_RAW));
    REQUIRE_EQ(last_tree_request(), "/api/models/test/repo/tree/" + C_RAW);
    REQUIRE(!saw_refs_request()); // a commit needs no refs lookup

    // 3. negative control: a revision that names nothing must NOT resolve to main
    REQUIRE_EQ(plan_file("no-such-rev"), "");
    REQUIRE_EQ(last_tree_request(), ""); // the tree was never requested, so no main files were listed
}

static void test_flags() {
    common_params params;
    std::vector<std::string> args = { "prog", "-hf", "test/repo:Q4_K_M", "--hf-revision", "v1.0",
                                      "--spec-draft-hf", "test/draft", "--spec-draft-hf-revision", "dev" };
    std::vector<char *> argv;
    for (auto & a : args) { argv.push_back(a.data()); }
    REQUIRE(common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER));
    REQUIRE_EQ(params.model.hf_revision, "v1.0");
    REQUIRE_EQ(params.speculative.mparams_dft.hf_revision, "dev");
    REQUIRE_EQ(params.model.hf_repo, "test/repo:Q4_K_M");
}

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    common_log_pause(common_log_main());

    httplib::Server server;
    serve(server);
    int port = server.bind_to_any_port("127.0.0.1");

    std::filesystem::path cache_dir = std::filesystem::temp_directory_path() /
                                      ("test-hf-revision-cache-" + std::to_string(port));
    std::filesystem::remove_all(cache_dir);
    common_set_env("LLAMA_CACHE", cache_dir.string());

    std::thread server_thread([&server] { server.listen_after_bind(); });
    server.wait_until_ready();
    common_set_env("MODEL_ENDPOINT", "http://127.0.0.1:" + std::to_string(port) + "/");

    test_resolution();
    test_flags();

    server.stop();
    server_thread.join();
    std::filesystem::remove_all(cache_dir);
    printf("test-hf-revision: all tests OK (no real download ran; loopback server only)\n");
    return 0;
}
