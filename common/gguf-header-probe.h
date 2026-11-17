#pragma once

// Read general.file_type out of the head of a GGUF without the rest of the file, so a remote file whose
// name carries no quant tag can still be matched against `-hf repo:TAG`. The key sits behind the tokenizer
// arrays (several MiB into a real model), so callers feed growing prefixes until the scan is final.
//
// SOURCE-READ: nexa-sdk sdk/model-manager/crates/core/src/gguf.rs:1-359 (BSD-3-Clause, HEAD 7a0ff49c4e;
// read in full 2026-11-16, no code copied): appending ranged reads of 2 MiB up to an 8 MiB cap, a
// Found / Absent / NeedMore walk of the KV section, array skipping with a depth bound. The tag table
// here is this fork's own: it is generated from the LLAMA_FTYPE_MOSTLY_* enumerators in include/llama.h,
// whose numbers differ from mainline's for several ftypes.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class common_gguf_scan_status {
    found,      // general.file_type was read
    absent,     // final: not a GGUF, bad version, malformed, or the key is not in the KV section
    need_more,  // the prefix ends before the answer; feed more bytes
};

struct common_gguf_scan_result {
    common_gguf_scan_status status = common_gguf_scan_status::need_more;
    uint32_t                file_type = 0;  // valid when status == found
};

// Walk the KV section of a GGUF prefix (v1..v3, little-endian) looking for general.file_type (uint32).
common_gguf_scan_result common_gguf_scan_file_type(const uint8_t * data, size_t size);

// Filename-style quant tag for a llama_ftype value ("Q4_K_M", "IQ4_KS", "F32", ...), or "" when unknown.
std::string common_ftype_tag(uint32_t file_type);

// True when `tag` (any case) is the tag of some llama_ftype, i.e. a filename tag that names a quant.
bool common_ftype_tag_is_known(const std::string & tag);

// Fetch [offset, offset+len) of the remote file and APPEND it to out; false on any transport failure.
using common_gguf_range_fetch = std::function<bool(uint64_t offset, uint64_t len, std::vector<uint8_t> & out)>;

constexpr uint64_t COMMON_GGUF_PROBE_CHUNK = 2ull * 1024 * 1024;  // bytes per ranged read
constexpr uint64_t COMMON_GGUF_PROBE_MAX   = 8ull * 1024 * 1024;  // read cap; callers may pass another

// Probe a remote file's quant tag with appending ranged reads. file_size 0 = unknown (the cap alone bounds it).
// Returns the tag ("" when absent, unknown, past the cap, or on a transport failure) and how many bytes were read.
std::string common_gguf_probe_tag(const common_gguf_range_fetch & fetch, uint64_t file_size, uint64_t * bytes_read = nullptr,
                                  uint64_t chunk = COMMON_GGUF_PROBE_CHUNK, uint64_t cap = COMMON_GGUF_PROBE_MAX);
