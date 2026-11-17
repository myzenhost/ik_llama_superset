// llama-cram-probe.cpp -- see llama-cram-probe.h.
//
// IMPORTANT: the exact sysfs names exposed by the unmerged CRAM series are UNKNOWN to the author of
// this file (no kernel with CRAM was available, and the sources were not consulted online). The checks
// below ('has_private'/'private' file in a node dir, <root>/kernel/mm/cram, a CXL 'mem*' device with a
// file whose name contains 'compress') are BEST-EFFORT GUESSES and may never match real hardware.
// A negative result therefore means "no known signal", not "no compressed memory".
#include "llama-cram-probe.h"

#if defined(__linux__)
#include <filesystem>
#include <fstream>
#include <cctype>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

static bool cram_has_content(const fs::path & p) {
    std::ifstream f(p);
    if (!f) {
        return false;
    }
    char c;
    while (f.get(c)) {
        if (!std::isspace((unsigned char) c)) {
            return true;
        }
    }
    return false;
}

llama_cram_probe_result llama_cram_probe(const char * sysfs_root) {
    llama_cram_probe_result r;
    const fs::path root(sysfs_root ? sysfs_root : "/sys");
    std::error_code ec;

    const fs::path node_dir = root / "devices" / "system" / "node";
    std::string priv_nodes, cxl_devs;
    if (fs::is_directory(node_dir, ec)) {
        for (fs::directory_iterator it(node_dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.size() < 5 || name.compare(0, 4, "node") != 0) {
                continue;
            }
            bool digits = true;
            for (size_t i = 4; i < name.size(); i++) {
                digits = digits && std::isdigit((unsigned char) name[i]);
            }
            std::error_code ec2;
            if (!digits || !fs::is_directory(it->path(), ec2)) {
                continue;
            }
            r.node_count_total++;
            if (!cram_has_content(it->path() / "cpulist")) {
                r.node_count_memory_only++;
            }
            if (fs::exists(it->path() / "has_private", ec2) || fs::exists(it->path() / "private", ec2)) {
                r.node_count_private_or_compressed++;
                priv_nodes += " " + name;
            }
        }
    }

    bool cram_dir = false;
    if (fs::is_directory(root / "kernel" / "mm" / "cram", ec)) {
        cram_dir = true;
        r.node_count_private_or_compressed++;
    }

    const fs::path cxl_dir = root / "bus" / "cxl" / "devices";
    int cxl_hits = 0;
    if (fs::is_directory(cxl_dir, ec)) {
        for (fs::directory_iterator it(cxl_dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            std::error_code ec2;
            if (name.compare(0, 3, "mem") != 0 || !fs::is_directory(it->path(), ec2)) { // follows symlinks
                continue;
            }
            bool found = false;
            for (fs::directory_iterator d(it->path(), ec2), dend; !ec2 && d != dend; d.increment(ec2)) {
                if (d->path().filename().string().find("compress") != std::string::npos) {
                    found = true;
                    break;
                }
            }
            if (found) {
                cxl_hits++;
                r.node_count_private_or_compressed++;
                cxl_devs += " " + name;
            }
        }
    }

    r.available = r.node_count_private_or_compressed > 0;
    r.detail = "nodes=" + std::to_string(r.node_count_total) +
               " memory_only=" + std::to_string(r.node_count_memory_only) +
               " private_nodes=[" + priv_nodes + " ]" +
               " kernel/mm/cram=" + (cram_dir ? "yes" : "no") +
               " cxl_compress_devs=[" + cxl_devs + " ]";
    if (r.node_count_total == 0) {
        r.detail += " (no node directories found under " + node_dir.string() + ")";
    }
    return r;
}

#elif defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <string>

// Windows route. Counts NUMA nodes through GetNumaHighestNodeNumber / GetNumaNodeProcessorMaskEx and treats a node
// with an empty processor mask as memory-only (this is how CPU-less CXL / HBM nodes show up). Windows documents no
// interface that marks a node as private or inline-compressed, so node_count_private_or_compressed stays 0 and
// available stays false: as on the sysfs route, false means "no known signal", not "no compressed memory".
// The sysfs_root argument is ignored.
llama_cram_probe_result llama_cram_probe(const char * /*sysfs_root*/) {
    llama_cram_probe_result r;
    ULONG highest = 0;
    if (!GetNumaHighestNodeNumber(&highest)) {
        r.detail = "GetNumaHighestNodeNumber failed, error " + std::to_string((unsigned long) GetLastError());
        return r;
    }
    std::string mem_only;
    unsigned long long avail_total = 0;
    for (ULONG n = 0; n <= highest; n++) {
        GROUP_AFFINITY ga = {};
        if (!GetNumaNodeProcessorMaskEx((USHORT) n, &ga)) {
            continue; // node number not present
        }
        r.node_count_total++;
        if (ga.Mask == 0) {
            r.node_count_memory_only++;
            mem_only += " node" + std::to_string((unsigned long) n);
        }
        ULONGLONG avail = 0;
        if (GetNumaAvailableMemoryNodeEx((USHORT) n, &avail)) {
            avail_total += avail;
        }
    }
    r.detail = "windows nodes=" + std::to_string(r.node_count_total) +
               " memory_only=" + std::to_string(r.node_count_memory_only) + " [" + mem_only + " ]" +
               " available_bytes=" + std::to_string(avail_total) +
               " (no Windows interface for private/compressed nodes is known)";
    return r;
}

#else // neither Linux nor Windows

llama_cram_probe_result llama_cram_probe(const char * /*sysfs_root*/) {
    llama_cram_probe_result r;
    r.available = false;
    r.detail = "unsupported OS";
    return r;
}

#endif
