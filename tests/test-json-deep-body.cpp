// The JSON layer must survive a deeply nested document: a request body is client-sized, and the server parses it before
// any schema or template code (which have their own depth caps) can look at it.
// Row S69-ENGREV-LOCALAI2-REQUEST-BODY-JSON-DEPTH-CAP-PARSER-COPY-2026-10-28-01. Needs no model.
//
// HISTORY: on 2026-10-28 (MSVC 14.51 Release, 1 MiB stack) a 5000-level nested member FOLLOWED BY a sibling key overflowed
// the stack inside the parse (rc 0xC00000FD): the ordered object grew its storage and copied the deep member recursively.
// MEASURED 2026-11-13 on canonical ef429b9c0, same toolchain: that no longer happens. Parse survives 5,000,000 levels,
// copy and dump survive 20,000, and a live CPU llama-server answers 200 to bodies 100,000 levels deep (4 shapes that skip
// the deep value) or a specific 400 from an existing cap (3 shapes that consume it), with no server death. So no body
// depth cap was added: it would only refuse requests that work. THIS TEST pins the property instead, so a change of the
// JSON backend that brings the recursion back fails here rather than in a served process.
//
// Shape: {"deep": {"a": {"a": ... 1}}, "k0": 0, ..., "k39": 39} -- the siblings come AFTER the deep member, so the
// object's storage grows while the deep member is already in it. Each step runs on the main thread and on a worker
// thread (a worker serves requests and has the executable's stack reserve).

#include "json.h"

#include <cstdio>
#include <string>
#include <thread>

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

static std::string body_with_siblings(int depth) {
    std::string s = "{\"deep\":";
    for (int i = 1; i < depth; i++) s += "{\"a\":";
    s += "1";
    for (int i = 1; i < depth; i++) s += "}";
    for (int i = 0; i < 40; i++) s += ",\"k" + std::to_string(i) + "\":" + std::to_string(i);
    s += "}";
    return s;
}

// parse; optionally copy and dump; returns the number of top-level keys seen at the last step (41 when all is well)
static size_t run(int depth, bool copy_and_dump) {
    const std::string body = body_with_siblings(depth);
    common_json j = common_json::parse(body);
    size_t keys = j.size();
    if (copy_and_dump) {
        common_json c = j;
        const std::string text = c.dump();
        if (text.size() < body.size() / 2) {
            return 0; // a dump that lost the deep member
        }
        common_json again = common_json::parse(text);
        keys = again.size();
    }
    return keys;
}

static size_t run_on_worker(int depth, bool copy_and_dump) {
    size_t keys = 0;
    std::thread t([&] { keys = run(depth, copy_and_dump); });
    t.join();
    return keys;
}

int main() {
    // control: a shallow document goes through every step
    CHECK(run(3, true) == 41, "shallow document: %zu keys", run(3, true));

    // the 2026-10-28 crash shape and depth, then far past it
    for (int depth : {5000, 20000}) {
        printf("depth %d\n", depth);
        CHECK(run(depth, false) == 41, "parse on the main thread");
        CHECK(run(depth, true) == 41, "parse + copy + dump + reparse on the main thread");
        CHECK(run_on_worker(depth, false) == 41, "parse on a worker thread");
        CHECK(run_on_worker(depth, true) == 41, "parse + copy + dump + reparse on a worker thread");
    }
    printf("depth 200000\n");
    CHECK(run(200000, false) == 41, "parse on the main thread");
    CHECK(run_on_worker(200000, false) == 41, "parse on a worker thread");

    if (g_fail) { printf("test-json-deep-body: %d FAILED\n", g_fail); return 1; }
    printf("test-json-deep-body: OK\n");
    return 0;
}
