// Unit test for tools/server/server-ollama-store.h. Layout facts (manifests/<host>/<ns>/<model>/<tag>, blobs/sha256-<hex>,
// layer media types, manifest-list runner selection) come from ollama manifest/paths.go, manifest.go, layer.go (MIT) -- no code.
#include "server-ollama-store.h"
#include <cstdio>
#include <random>

using namespace ollama_util;
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::string hex(char c) { return std::string(64, c); }
static void put(const fs::path & p, const std::string & s) { fs::create_directories(p.parent_path()); std::ofstream(p, std::ios::binary) << s; }
static void blob(const std::string & dir, char c, const std::string & body) { put(fs::path(dir) / "blobs" / ("sha256-" + hex(c)), body); }
static json layer(const char * mt, char c) { return json {{"mediaType", mt}, {"digest", "sha256:" + hex(c)}, {"size", 1}}; }

int main() {
    const std::string dir = (fs::temp_directory_path() / ("ollama-store-" + std::to_string(std::random_device{}()))).string();
    fs::create_directories(dir);
    blob(dir, 'a', "GGUFmodel"); blob(dir, 'b', "mmproj"); blob(dir, 'c', "{{ .Prompt }}"); blob(dir, 'd', "be terse");
    blob(dir, 'e', R"({"temperature":0.3,"stop":["<x>"]})"); blob(dir, 'f', "MIT"); blob(dir, '0', "{}"); blob(dir, '1', "lora");
    json m = {{"schemaVersion", 2}, {"mediaType", "application/vnd.docker.distribution.manifest.v2+json"},
              {"config", layer("application/vnd.docker.container.image.v1+json", '0')},
              {"layers", json::array({layer("application/vnd.ollama.image.model", 'a'), layer("application/vnd.ollama.image.projector", 'b'),
                                      layer("application/vnd.ollama.image.template", 'c'), layer("application/vnd.ollama.image.system", 'd'),
                                      layer("application/vnd.ollama.image.params", 'e'), layer("application/vnd.ollama.image.license", 'f'),
                                      layer("application/vnd.ollama.image.adapter", '1')})}};
    put(fs::path(dir) / "manifests/registry.ollama.ai/library/tiny/latest", m.dump());
    put(fs::path(dir) / "manifests/registry.ollama.ai/acme/tiny/7b", m.dump());

    store_model sm; std::string err;
    CHECK(store_resolve(dir, "tiny", sm, err), "resolve bare: %s", err.c_str());
    CHECK(fs::path(sm.model_blob) == fs::path(dir) / "blobs" / ("sha256-" + hex('a')), "model blob path %s", sm.model_blob.c_str());
    CHECK(sm.projector_blob.size() && sm.adapter_blobs.size() == 1, "projector + adapter");
    CHECK(sm.template_text == "{{ .Prompt }}" && sm.system_text == "be terse" && sm.licenses.size() == 1, "text layers");
    CHECK(sm.params.value("temperature", 0.0) > 0.29 && sm.params["stop"].size() == 1, "params layer");
    CHECK(store_resolve(dir, "REGISTRY.ollama.ai/library/tiny:latest", sm, err), "resolve full/case: %s", err.c_str());
    CHECK(store_resolve(dir, "acme/tiny:7b", sm, err), "resolve namespaced: %s", err.c_str());
    // default public host is tried under both spellings: a store written as ollama.com still answers a bare name
    put(fs::path(dir) / "manifests/ollama.com/library/other/latest", m.dump());
    CHECK(store_resolve(dir, "other", sm, err), "ollama.com spelling: %s", err.c_str());
    CHECK(!store_resolve(dir, "missing", sm, err) && err.find("not found") != std::string::npos, "missing: %s", err.c_str());
    CHECK(!store_resolve(dir, "a/b/c/d:e", sm, err), "invalid name rejected");
    // path traversal through a digest must be refused, not followed
    { json bad = m; bad["layers"][0]["digest"] = "sha256:../../etc/passwd"; put(fs::path(dir) / "manifests/registry.ollama.ai/library/bad/latest", bad.dump());
      CHECK(!store_resolve(dir, "bad", sm, err) && err.find("invalid digest") != std::string::npos, "traversal digest: %s", err.c_str()); }
    CHECK(!store_valid_digest("sha256:" + hex('a') + "0") && store_valid_digest("sha256-" + hex('A')) && !store_valid_digest("sha256:" + std::string(63, 'a')), "digest shape");
    // missing model blob
    { json nb = m; nb["layers"][0]["digest"] = "sha256:" + hex('9'); put(fs::path(dir) / "manifests/registry.ollama.ai/library/nob/latest", nb.dump());
      CHECK(!store_resolve(dir, "nob", sm, err) && err.find("blob missing") != std::string::npos, "missing blob: %s", err.c_str()); }
    // v2 manifest list: picks the llamacpp child (a bare digest reference stored as a blob), ignores mlx
    { blob(dir, '7', m.dump());
      json list = {{"schemaVersion", 2}, {"mediaType", "application/vnd.ollama.manifest.list.v2+json"},
                   {"manifests", json::array({ json {{"mediaType", "application/vnd.docker.distribution.manifest.v2+json"}, {"runner", "mlx"}, {"digest", "sha256:" + hex('8')}},
                                               json {{"mediaType", "application/vnd.docker.distribution.manifest.v2+json"}, {"runner", "llamacpp"}, {"digest", "sha256:" + hex('7')}} })}};
      put(fs::path(dir) / "manifests-v2/ollama.com/library/listed/latest", list.dump());
      CHECK(store_resolve(dir, "listed", sm, err) && sm.model_blob.find(hex('a')) != std::string::npos, "list->llamacpp child: %s", err.c_str());
      json mlxonly = list; mlxonly["manifests"].erase(1);
      put(fs::path(dir) / "manifests-v2/ollama.com/library/mlxonly/latest", mlxonly.dump());
      CHECK(!store_resolve(dir, "mlxonly", sm, err) && err.find("no compatible") != std::string::npos, "mlx only refused: %s", err.c_str()); }
    // listing
    { auto v = store_list(dir); auto has = [&](const char * s) { return std::find(v.begin(), v.end(), s) != v.end(); };
      CHECK(has("tiny:latest") && has("acme/tiny:7b") && has("listed:latest"), "list has %d names", (int) v.size()); }
    // copy + delete (blob GC keeps blobs another manifest still uses)
    CHECK(store_copy(dir, "tiny", "tiny-copy:v1", err), "copy: %s", err.c_str());
    CHECK(store_resolve(dir, "tiny-copy:v1", sm, err), "copy resolves: %s", err.c_str());
    CHECK(!store_copy(dir, "nope", "x", err), "copy of missing fails");
    { std::vector<std::string> gone;
      CHECK(store_remove(dir, "tiny-copy:v1", gone, err) && gone.empty(), "delete copy keeps shared blobs (%d removed)", (int) gone.size());
      CHECK(!store_resolve(dir, "tiny-copy:v1", sm, err), "copy gone");
      CHECK(fs::exists(dir + "/blobs/sha256-" + hex('a')), "shared blob survives"); }
    { std::vector<std::string> gone; blob(dir, '5', "private");
      json solo = m; solo["layers"][0]["digest"] = "sha256:" + hex('5'); put(fs::path(dir) / "manifests/registry.ollama.ai/library/solo/latest", solo.dump());
      CHECK(store_remove(dir, "solo", gone, err), "delete solo: %s", err.c_str());
      CHECK(gone.size() == 1 && !fs::exists(dir + "/blobs/sha256-" + hex('5')), "private blob collected (%d)", (int) gone.size());
      CHECK(!store_remove(dir, "solo", gone, err), "second delete fails"); }
    fs::remove_all(dir);
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
