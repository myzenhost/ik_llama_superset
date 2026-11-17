// Router loaded-set snapshot helpers (tools/server/server-models-snapshot.h, --models-snapshot FILE).
// Row S69-TORCHSERVE-PICK-ROUTER-LOADED-SET-SNAPSHOT-2026-10-18-01. TorchServe is Apache-2.0; behaviour only, no code copied.
// Needs no model. The served path (router writes the file, restart restores it) is tests/test-server-models-snapshot.py.

#include "server-models-snapshot.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

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

using strv = std::vector<std::string>;

static std::string join(const strv & v) {
    std::string s;
    for (const auto & x : v) {
        s += (s.empty() ? "" : ",") + x;
    }
    return "[" + s + "]";
}

int main() {
    std::string err;
    strv names;

    printf("1. parse\n");
    CHECK(server_models_snapshot_parse("# llama-server models snapshot v1\na\n\n  b \n# note\na\nc\r\n", names, err), "%s", err.c_str());
    CHECK(names == (strv{"a", "b", "c"}), "got %s (trim, comment, dedupe, CRLF)", join(names).c_str());
    // negative controls: another format, and an empty file, are refused rather than read as a list
    CHECK(!server_models_snapshot_parse("a\nb\n", names, err) && names.empty(), "a list without the header must be refused");
    CHECK(!server_models_snapshot_parse("", names, err), "an empty file has no header");
    CHECK(!server_models_snapshot_parse("# llama-server models snapshot v2\na\n", names, err), "a later version is not v1");
    CHECK(server_models_snapshot_parse("\n# llama-server models snapshot v1\n", names, err) && names.empty(), "header only = empty list");

    printf("2. serialize round trip\n");
    {
        const strv in = {"m1", "org/m2:Q4_K_M"};
        CHECK(server_models_snapshot_parse(server_models_snapshot_serialize(in), names, err) && names == in, "got %s", join(names).c_str());
        CHECK(server_models_snapshot_parse(server_models_snapshot_serialize({}), names, err) && names.empty(), "empty set round trip");
    }

    printf("3. merge\n");
    const std::set<std::string> known = {"a", "b", "c", "d", "e"};
    {
        // control: no snapshot -> exactly the preset list (the behaviour without the flag)
        auto p = server_models_snapshot_merge({"a"}, {}, known, {"a"}, 4);
        CHECK(p.load == (strv{"a"}) && p.restored.empty(), "got %s", join(p.load).c_str());
    }
    {
        auto p = server_models_snapshot_merge({"a"}, {"b", "a", "zz", "c"}, known, {"a"}, 4);
        CHECK(p.load == (strv{"a", "b", "c"}), "presets first, then restored: %s", join(p.load).c_str());
        CHECK(p.restored == (strv{"b", "c"}), "restored %s", join(p.restored).c_str());
        CHECK(p.skipped_unknown == (strv{"zz"}), "unknown %s", join(p.skipped_unknown).c_str());
    }
    {
        // explicit preset wins: d's preset says load-on-startup = false, the snapshot says load it
        auto p = server_models_snapshot_merge({}, {"d", "e"}, known, {"d"}, 4);
        CHECK(p.load == (strv{"e"}), "got %s", join(p.load).c_str());
        CHECK(p.skipped_explicit == (strv{"d"}), "explicit %s", join(p.skipped_explicit).c_str());
    }
    {
        // cap: presets count toward models_max, restored entries stop at it
        auto p = server_models_snapshot_merge({"a", "b"}, {"c", "d", "e"}, known, {"a", "b"}, 3);
        CHECK(p.load == (strv{"a", "b", "c"}), "got %s", join(p.load).c_str());
        CHECK(p.skipped_cap == (strv{"d", "e"}), "cap %s", join(p.skipped_cap).c_str());
        auto u = server_models_snapshot_merge({"a"}, {"b", "c", "d", "e"}, known, {"a"}, 0);
        CHECK(u.load.size() == 5 && u.skipped_cap.empty(), "models_max 0 = unlimited, got %s", join(u.load).c_str());
    }

    printf("4. file write / read\n");
    {
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / "llama-test-models-snapshot";
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
        const std::string path = (dir / "snap.txt").string();

        CHECK(server_models_snapshot_read(path, names, err) && names.empty(), "a missing file is a first start, not an error: %s", err.c_str());
        CHECK(server_models_snapshot_write(path, {"a", "b"}, err), "%s", err.c_str());
        CHECK(server_models_snapshot_read(path, names, err) && names == (strv{"a", "b"}), "got %s", join(names).c_str());
        // the rename must replace an existing file (std::rename on Windows does not)
        CHECK(server_models_snapshot_write(path, {"c"}, err), "overwrite: %s", err.c_str());
        CHECK(server_models_snapshot_read(path, names, err) && names == (strv{"c"}), "after overwrite got %s", join(names).c_str());
        CHECK(!fs::exists(path + ".tmp"), "no .tmp left behind");
        // a foreign file at the path is refused, and the error names why
        { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << "not a snapshot\n"; }
        CHECK(!server_models_snapshot_read(path, names, err) && names.empty() && !err.empty(), "foreign file refused: %s", err.c_str());
        // an unwritable target directory reports an error instead of claiming success
        CHECK(!server_models_snapshot_write((dir / "no-such-dir" / "snap.txt").string(), {"a"}, err) && !err.empty(), "unwritable path: %s", err.c_str());
        fs::remove_all(dir, ec);
    }

    if (g_fail == 0) {
        printf("ALL PASS\n");
    } else {
        printf("%d FAILED\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
