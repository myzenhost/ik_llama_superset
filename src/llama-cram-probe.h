// llama-cram-probe.h -- gated-OFF, read-only capability probe for compressed / private memory tiers.
//
// Background: Meta's CRAM (Compressed RAM, Gregory Price's Linux RFC: an N_MEMORY_PRIVATE NUMA node
// plus mm/cram.c) keeps cold pages compressed by CXL inline-compression hardware, maps them
// read-only, and promotes a page back to DRAM on a write fault. It is not in mainline.
//
// STATUS: called at model load by common_init_from_params() when --cram-probe is given (log only; changes nothing).
#pragma once

#include "llama.h" // LLAMA_API

#include <string>

struct llama_cram_probe_result {
    // true only if at least one private/compressed signal was found (node_count_private_or_compressed > 0).
    // false on an empty/unreadable tree, on a tree of ordinary nodes, on Windows (nodes are counted, but no
    // private/compressed signal is known there) and on any other OS.
    bool available = false;
    int  node_count_total                = 0; // node<N> directories under <root>/devices/system/node
    int  node_count_memory_only          = 0; // nodes whose cpulist is missing or has no content
    // Number of signals found: one per node with a has_private/private file, +1 if <root>/kernel/mm/cram
    // is a directory, +1 per <root>/bus/cxl/devices/mem* device that has a file with "compress" in its name.
    int  node_count_private_or_compressed = 0;
    std::string detail;
};

// Read-only: never writes under sysfs_root. sysfs_root == nullptr means "/sys".
LLAMA_API llama_cram_probe_result llama_cram_probe(const char * sysfs_root = "/sys");
