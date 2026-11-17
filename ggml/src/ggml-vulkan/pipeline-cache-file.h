#pragma once

// On-disk container for a Vulkan VkPipelineCache blob (ggml-vulkan persistent pipeline cache).
// RESEARCH-DONE: github.com/Tencent/ncnn src/pipelinecache.cpp @ c6b351b5 (BSD-3-Clause) and src/gpu.cpp read
// 2026-10-05; Vulkan spec VkPipelineCacheHeaderVersionOne (vkCreatePipelineCache / vkGetPipelineCacheData).
//
// WHY. ggml-vulkan used to pass VK_NULL_HANDLE as the pipelineCache of every vkCreateComputePipelines call
// (ggml-vulkan.cpp, pipeline creation), so every process start recompiled every SPIR-V module it touched from
// scratch. Some drivers keep their own implicit disk cache; many (Android Mali/Adreno, MoltenVK, older
// Mesa configurations) do not. The blob a VkPipelineCache hands back is only meaningful on the device, driver and
// driver version that produced it, so the file wraps it in an identity header and refuses anything that does
// not match -- a stale or damaged cache must produce a clean cold start, never a crash and never silent reuse.
//
// This header has no Vulkan dependency: the device identity is passed in as plain integers, which keeps the
// whole format (and every refusal path) testable on a machine with no GPU (tests/test-vk-pipeline-cache-file.cpp).
//
// DESIGN REFERENCE (structure and the list of checks; own file format and code): Tencent/ncnn
// src/pipelinecache.cpp @ c6b351b5 (BSD-3-Clause), read in full: validate_vk_pipeline_cache_data (32-byte
// VkPipelineCacheHeaderVersionOne check), cache_file_header / validate_cache_file_header, replace_file
// (write temp file, then replace), the 256 MiB load limit, and create_vk_pipeline_cache with the Adreno
// bug flag.
//
// NUANCES THE ROW TEXT DID NOT SAY (found by reading pipelinecache.cpp and gpu.cpp):
//   1. TWO identity checks, not one. The Vulkan-defined 32-byte header inside the blob (vendorID, deviceID,
//      pipelineCacheUUID) is checked AND the file's own header (api version, driver version, driverID, device name
//      hash, driver name hash, endianness, pointer size). The driver's header alone does not cover a driver
//      update that keeps the UUID, and the UUID alone does not cover a blob truncated by a crashed writer.
//   2. A blob-integrity hash is stored and checked BEFORE the blob is handed to the driver: drivers are not
//      required to survive corrupted initial data, and the point of the file check is that they never see it.
//   3. Saving writes a temporary file and then replaces the target, so a crash mid-write leaves the old file
//      (or none), never a half file. On Windows the replacement is a rename-over (MoveFileEx semantics).
//   4. Loading refuses files larger than 256 MiB without reading them.
//   5. An empty blob is valid input to vkCreatePipelineCache (it means "start cold") and is therefore not an
//      error here; it is reported as status::empty so the caller can log a cold start.
//   6. ncnn skips the cache entirely on one driver family: Qualcomm Adreno (vendor 0x5143) with a Vulkan API
//      version below 1.0.66, whose old driver "cannot share created pipeline properly" (gpu.cpp bug flag
//      bug_corrupted_online_pipeline_cache). Reproduced as ggml_vk_pc::driver_blocks_cache().
//   7. The cache file is per device: the default file name carries vendor id, device id and the first bytes of
//      the cache UUID, so two GPUs (or one GPU under two driver versions that changed the UUID) never overwrite
//      each other's file.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace ggml_vk_pc {

constexpr uint64_t FILE_MAGIC        = 0x3148435056474747ull; // "GGGVPCH1" little endian
constexpr uint32_t FILE_VERSION      = 1;
constexpr uint32_t FILE_ENDIAN_MARK  = 0x01020304u;
constexpr size_t   MAX_FILE_BYTES    = size_t(256) << 20;
constexpr size_t   VK_HEADER_V1_SIZE = 32; // VkPipelineCacheHeaderVersionOne

inline uint32_t fnv1a32(const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
    return h;
}

inline uint64_t fnv1a64(const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// What a cache blob is tied to.
struct identity {
    uint32_t vendor_id      = 0;
    uint32_t device_id      = 0;
    uint32_t api_version    = 0;
    uint32_t driver_version = 0;
    uint32_t driver_id      = 0;
    uint32_t device_name_hash = 0; // fnv1a32 of the device name (hash_name)
    uint32_t driver_name_hash = 0;
    uint8_t  uuid[16]       = {};  // VkPhysicalDeviceProperties::pipelineCacheUUID
};

inline uint32_t hash_name(const char * s) { return s ? fnv1a32(s, std::strlen(s)) : 0u; }

#pragma pack(push, 8)
struct file_header {
    uint64_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t endian;
    uint32_t pointer_size;
    uint32_t vendor_id, device_id, api_version, driver_version, driver_id;
    uint32_t device_name_hash, driver_name_hash;
    uint8_t  uuid[16];
    uint32_t pad0;          // explicit so no implicit padding exists; must be zero
    uint64_t blob_size;
    uint64_t blob_hash;
    uint32_t reserved[8];
};
#pragma pack(pop)

enum class status {
    ok,
    empty,             // valid file carrying an empty blob: cold start
    file_too_small,
    file_too_large,
    bad_magic,
    bad_version,
    bad_header_size,
    bad_endian,
    bad_pointer_size,
    identity_mismatch, // vendor/device/api/driver version/driver id/name hashes/uuid differ
    size_mismatch,     // blob_size does not match the bytes present
    bad_hash,          // blob damaged
    bad_vk_header,     // the 32-byte Vulkan header inside the blob does not describe THIS device
};

inline const char * status_str(status s) {
    switch (s) {
        case status::ok:                return "ok";
        case status::empty:             return "empty";
        case status::file_too_small:    return "file too small";
        case status::file_too_large:    return "file too large";
        case status::bad_magic:         return "bad magic";
        case status::bad_version:       return "unsupported file version";
        case status::bad_header_size:   return "bad header size";
        case status::bad_endian:        return "different endianness";
        case status::bad_pointer_size:  return "different pointer size";
        case status::identity_mismatch: return "device/driver identity differs";
        case status::size_mismatch:     return "blob size does not match file size";
        case status::bad_hash:          return "blob hash mismatch (damaged file)";
        case status::bad_vk_header:     return "Vulkan cache header does not match this device";
    }
    return "?";
}

// Old Qualcomm Adreno drivers cannot share a created pipeline through a cache; the cache is not used at all there.
inline bool driver_blocks_cache(const identity & id) {
    // VK_MAKE_VERSION(1, 0, 66) = (1 << 22) | (0 << 12) | 66
    return id.vendor_id == 0x5143u && id.api_version < ((1u << 22) | 66u);
}

inline uint32_t rd32(const uint8_t * p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

// The 32-byte VkPipelineCacheHeaderVersionOne at the start of every conformant blob:
//   u32 headerSize, u32 headerVersion (1), u32 vendorID, u32 deviceID, u8 pipelineCacheUUID[16].
inline bool vk_blob_header_matches(const identity & id, const uint8_t * blob, size_t n) {
    if (n < VK_HEADER_V1_SIZE) return false;
    const uint32_t header_size = rd32(blob), header_version = rd32(blob + 4);
    if (header_size < VK_HEADER_V1_SIZE || header_size > n) return false;
    if (header_version != 1u) return false;
    if (rd32(blob + 8)  != id.vendor_id) return false;
    if (rd32(blob + 12) != id.device_id) return false;
    return std::memcmp(blob + 16, id.uuid, 16) == 0;
}

inline std::vector<uint8_t> pack(const identity & id, const void * blob, size_t n) {
    file_header h;
    std::memset(&h, 0, sizeof(h));
    h.magic = FILE_MAGIC; h.version = FILE_VERSION; h.header_size = (uint32_t) sizeof(file_header);
    h.endian = FILE_ENDIAN_MARK; h.pointer_size = (uint32_t) sizeof(void *);
    h.vendor_id = id.vendor_id; h.device_id = id.device_id; h.api_version = id.api_version;
    h.driver_version = id.driver_version; h.driver_id = id.driver_id;
    h.device_name_hash = id.device_name_hash; h.driver_name_hash = id.driver_name_hash;
    std::memcpy(h.uuid, id.uuid, 16);
    h.blob_size = n;
    h.blob_hash = fnv1a64(blob, n);
    std::vector<uint8_t> out(sizeof(h) + n);
    std::memcpy(out.data(), &h, sizeof(h));
    if (n) std::memcpy(out.data() + sizeof(h), blob, n);
    return out;
}

// Validates `file` against the current device. On ok/empty, *blob/*blob_n describe the payload INSIDE `file`.
inline status unpack(const identity & id, const uint8_t * file, size_t n, const uint8_t ** blob, size_t * blob_n) {
    *blob = nullptr; *blob_n = 0;
    if (n > MAX_FILE_BYTES)        return status::file_too_large;
    if (n < sizeof(file_header))   return status::file_too_small;
    file_header h;
    std::memcpy(&h, file, sizeof(h));
    if (h.magic != FILE_MAGIC)              return status::bad_magic;
    if (h.version != FILE_VERSION)          return status::bad_version;
    if (h.header_size != sizeof(file_header)) return status::bad_header_size;
    if (h.pad0 != 0) return status::bad_header_size;
    for (uint32_t r : h.reserved) { if (r != 0) return status::bad_header_size; } // a future version would bump `version`
    if (h.endian != FILE_ENDIAN_MARK)       return status::bad_endian;
    if (h.pointer_size != sizeof(void *))   return status::bad_pointer_size;
    if (h.vendor_id != id.vendor_id || h.device_id != id.device_id || h.api_version != id.api_version ||
        h.driver_version != id.driver_version || h.driver_id != id.driver_id ||
        h.device_name_hash != id.device_name_hash || h.driver_name_hash != id.driver_name_hash ||
        std::memcmp(h.uuid, id.uuid, 16) != 0) {
        return status::identity_mismatch;
    }
    if (h.blob_size != n - sizeof(file_header)) return status::size_mismatch;
    const uint8_t * b = file + sizeof(file_header);
    if (fnv1a64(b, (size_t) h.blob_size) != h.blob_hash) return status::bad_hash;
    if (h.blob_size == 0) return status::empty;
    if (!vk_blob_header_matches(id, b, (size_t) h.blob_size)) return status::bad_vk_header;
    *blob = b; *blob_n = (size_t) h.blob_size;
    return status::ok;
}

// ---- files ---------------------------------------------------------------------------------------------------------

inline std::filesystem::path path_from_utf8(const std::string & s) {
    return std::filesystem::u8path(s);
}

// Reads at most MAX_FILE_BYTES; a larger file is reported as `too_large` without being read.
inline bool read_file(const std::string & path, std::vector<uint8_t> & out, bool * too_large) {
    if (too_large) *too_large = false;
    std::error_code ec;
    const auto p = path_from_utf8(path);
    const auto sz = std::filesystem::file_size(p, ec);
    if (ec) return false;
    if (sz > MAX_FILE_BYTES) { if (too_large) *too_large = true; return false; }
    FILE * f = std::fopen(p.string().c_str(), "rb");
    if (!f) return false;
    out.resize((size_t) sz);
    const size_t got = sz ? std::fread(out.data(), 1, (size_t) sz, f) : 0;
    std::fclose(f);
    if (got != (size_t) sz) { out.clear(); return false; }
    return true;
}

// Writes `data` next to `path` under a temporary name and replaces `path` with it (nuance 3).
inline bool write_atomic(const std::string & path, const void * data, size_t n) {
    std::error_code ec;
    const auto p = path_from_utf8(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    const auto tmp = std::filesystem::path(p.string() + ".tmp" + std::to_string((unsigned long long) fnv1a64(path.data(), path.size())));
    FILE * f = std::fopen(tmp.string().c_str(), "wb");
    if (!f) return false;
    const size_t put = n ? std::fwrite(data, 1, n, f) : 0;
    const bool flushed = std::fflush(f) == 0;
    std::fclose(f);
    if (put != n || !flushed) { std::filesystem::remove(tmp, ec); return false; }
    std::filesystem::rename(tmp, p, ec); // replaces an existing file on Windows and POSIX
    if (ec) { std::filesystem::remove(tmp, ec); return false; }
    return true;
}

// ---- where the file lives ----------------------------------------------------------------------------------------

inline std::string hex8(uint32_t v) {
    char b[9]; std::snprintf(b, sizeof(b), "%08x", v); return b;
}

// vendor-device-uuid8 .bin  (nuance 7)
inline std::string default_file_name(const identity & id) {
    uint32_t u; std::memcpy(&u, id.uuid, 4);
    return "pipeline-cache-" + hex8(id.vendor_id) + "-" + hex8(id.device_id) + "-" + hex8(u) + ".bin";
}

// Per-user cache directory for ggml-vulkan, or "" when no home can be found. getenv values are passed in so the
// logic is testable.
inline std::string default_cache_dir(const char * local_app_data, const char * xdg_cache_home, const char * home) {
    auto nonempty = [](const char * s) { return s && *s; };
#if defined(_WIN32)
    (void) xdg_cache_home; (void) home;
    if (nonempty(local_app_data)) return std::string(local_app_data) + "/ggml/vulkan";
#elif defined(__APPLE__)
    (void) local_app_data; (void) xdg_cache_home;
    if (nonempty(home)) return std::string(home) + "/Library/Caches/ggml/vulkan";
#else
    (void) local_app_data;
    if (nonempty(xdg_cache_home)) return std::string(xdg_cache_home) + "/ggml/vulkan";
    if (nonempty(home)) return std::string(home) + "/.cache/ggml/vulkan";
#endif
    return "";
}

// Interprets the GGML_VK_PIPELINE_CACHE setting:
//   unset / "" / "0"        -> "" (not used)
//   "1"                     -> <default dir>/<default file name>   ("" if no default dir exists)
//   an existing directory, or a value ending in '/' or '\\'  -> <value>/<default file name>
//   anything else           -> the value is the file itself
inline std::string resolve_path(const char * setting, const identity & id, const std::string & default_dir) {
    if (!setting || !*setting || std::string(setting) == "0") return "";
    const std::string s(setting);
    const std::string name = default_file_name(id);
    if (s == "1") return default_dir.empty() ? "" : default_dir + "/" + name;
    std::error_code ec;
    const char last = s.back();
    if (last == '/' || last == '\\' || std::filesystem::is_directory(path_from_utf8(s), ec)) {
        return (last == '/' || last == '\\') ? s + name : s + "/" + name;
    }
    return s;
}

} // namespace ggml_vk_pc
