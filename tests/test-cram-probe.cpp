// test-cram-probe: builds FAKE sysfs trees in a temp dir and checks llama_cram_probe() counts/flags,
// plus controls (empty tree, ordinary-nodes-only tree must not be flagged).
#include "llama-cram-probe.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#if defined(__linux__)
#include <unistd.h>   // mkdtemp; only the sysfs branch below needs it
#endif

namespace fs = std::filesystem;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static void touch(const fs::path & p, const std::string & content = "") {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << content;
}

int main() {
#if defined(_WIN32)
    // Windows route: no fake tree exists for the Win32 NUMA calls, so check the real host for sanity and for the
    // control that a plain host is never flagged as having private/compressed memory.
    {
        auto r = llama_cram_probe();
        std::printf("windows:  avail=%d total=%d memonly=%d priv=%d | %s\n", r.available, r.node_count_total,
                    r.node_count_memory_only, r.node_count_private_or_compressed, r.detail.c_str());
        CHECK(r.node_count_total >= 1);
        CHECK(r.node_count_memory_only >= 0 && r.node_count_memory_only <= r.node_count_total);
        CHECK(r.node_count_private_or_compressed == 0);
        CHECK(!r.available);
        CHECK(r.detail.find("windows nodes=") == 0);
    }
    std::printf(g_fail ? "test-cram-probe: FAILED (%d)\n" : "test-cram-probe: PASSED\n", g_fail);
    return g_fail ? 1 : 0;
#elif !defined(__linux__)
    std::printf("SKIPPED: no probe route for this OS\n");
    return 0;
#else
    char tmpl[] = "/tmp/cram-probe-XXXXXX";
    if (!mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 2; }
    const fs::path base = tmpl;

    // CONTROL 1: empty tree -> available=false, zero counts.
    {
        fs::path root = base / "empty";
        fs::create_directories(root);
        auto r = llama_cram_probe(root.c_str());
        std::printf("empty:    avail=%d total=%d memonly=%d priv=%d | %s\n", r.available, r.node_count_total,
                    r.node_count_memory_only, r.node_count_private_or_compressed, r.detail.c_str());
        CHECK(!r.available && r.node_count_total == 0 && r.node_count_private_or_compressed == 0);
    }
    // CONTROL 2: only ordinary nodes (one cpu node, one memory-only node, junk entries) -> not flagged.
    {
        fs::path root = base / "ordinary";
        fs::path n = root / "devices/system/node";
        touch(n / "node0/cpulist", "0-15\n");
        touch(n / "node1/cpulist", "\n");           // memory-only (whitespace only)
        touch(n / "node1/meminfo", "x");
        touch(n / "possible", "0-1");               // not a node dir
        fs::create_directories(n / "nodeX");        // 'nodeX' is not node<digits>
        fs::create_directories(root / "bus/cxl/devices/mem0"); // cxl mem without compress file
        touch(root / "bus/cxl/devices/mem0/serial", "1");
        auto r = llama_cram_probe(root.c_str());
        std::printf("ordinary: avail=%d total=%d memonly=%d priv=%d | %s\n", r.available, r.node_count_total,
                    r.node_count_memory_only, r.node_count_private_or_compressed, r.detail.c_str());
        CHECK(r.node_count_total == 2);
        CHECK(r.node_count_memory_only == 1);
        CHECK(r.node_count_private_or_compressed == 0);
        CHECK(!r.available);
    }
    // Positive: node2 has 'private', node3 has 'has_private', kernel/mm/cram dir, cxl mem1 with compress file.
    {
        fs::path root = base / "cram";
        fs::path n = root / "devices/system/node";
        touch(n / "node0/cpulist", "0-3");
        touch(n / "node1/cpulist", "");
        touch(n / "node2/private", "1");
        touch(n / "node3/has_private", "1");
        fs::create_directories(root / "kernel/mm/cram");
        touch(root / "bus/cxl/devices/mem0/serial", "1");
        touch(root / "bus/cxl/devices/mem1/compression_ratio", "2.0");
        touch(root / "bus/cxl/devices/port1/compress_x", "1"); // not mem*: must be ignored
        auto r = llama_cram_probe(root.c_str());
        std::printf("cram:     avail=%d total=%d memonly=%d priv=%d | %s\n", r.available, r.node_count_total,
                    r.node_count_memory_only, r.node_count_private_or_compressed, r.detail.c_str());
        CHECK(r.available);
        CHECK(r.node_count_total == 4);
        CHECK(r.node_count_memory_only == 3);   // node1 (empty), node2, node3 (no cpulist)
        CHECK(r.node_count_private_or_compressed == 4); // node2 + node3 + cram dir + cxl mem1
    }
    // Informational: the real /sys of this machine.
    {
        auto r = llama_cram_probe();
        std::printf("real /sys: avail=%d total=%d memonly=%d priv=%d | %s\n", r.available, r.node_count_total,
                    r.node_count_memory_only, r.node_count_private_or_compressed, r.detail.c_str());
    }
    fs::remove_all(base);
    std::printf(g_fail ? "test-cram-probe: FAILED (%d)\n" : "test-cram-probe: PASSED\n", g_fail);
    return g_fail ? 1 : 0;
#endif
}
