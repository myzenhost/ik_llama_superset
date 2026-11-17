#include "nvrtc_cache.h"

// Reuses the vendored public-domain SHA-256 core directly (Igor Pavlov, 2010, public domain --
// vendor/hash/sha256/{sha256.h,sha256.c,LICENSE}) rather than depending on the vendor::hash CMake
// alias target, which is only defined by add_subdirectory(vendor) -- and that subdirectory is
// processed AFTER add_subdirectory(ggml) at the top level, so `vendor::hash` does not exist yet
// at ggml-cuda-dl's configure time. No code copied: the .c file is compiled as-is into this
// target (see ggml-cuda-dl/CMakeLists.txt), not duplicated.
// sha256.h has no extern "C" guard of its own (it's a plain C header with no __cplusplus check),
// so included as-is from this .cpp its declarations would get C++ name mangling while sha256.c
// (compiled as C -- see ggml-cuda-dl/CMakeLists.txt) exports plain C symbols. Wrap the include.
extern "C" {
#include "sha256/sha256.h"
}

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

namespace ggml_cuda_dl {

namespace {

// Cache file format: a tiny self-describing container so a truncated/corrupted entry is DETECTED
// (hash mismatch), not silently trusted -- "corrupt cache entries are recompiled, not trusted".
//   [8 bytes magic "GGCUBIN2"] [8 bytes little-endian payload length] [32 bytes sha256(payload)] [payload]
// payload (format 2):
//   [4 bytes LE count N] N x ([4 bytes LE length L] [L bytes lowered name]) [cubin bytes ...]
// FORMAT 2 (2026-11-11, row S69-TEST-CUDA-DL-LAUNCH-ACCESS-VIOLATION-NO-OUTPUT-2026-11-11-01): format 1 stored
// the cubin ONLY, so a cache hit returned ok=true with EMPTY lowered_names -- every caller that looks a
// kernel up by its lowered name (test-cuda-dl-launch: cr.lowered_names[0]) read past an empty vector on
// the second and every later run (MEASURED: 0xC0000005 at a null+0x18 read = MSVC std::string _Myres).
// The cache key carries kCacheFormat too, so a format-1 file is never even looked up (plain miss).
constexpr char kMagic[8] = { 'G', 'G', 'C', 'U', 'B', 'I', 'N', '2' };
constexpr int kCacheFormat = 2;

void put_u32(std::vector<char> & out, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
    }
}

bool get_u32(const std::vector<char> & in, size_t & pos, uint32_t & v) {
    if (pos + 4 > in.size()) {
        return false;
    }
    v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<uint32_t>(static_cast<unsigned char>(in[pos + i])) << (8 * i);
    }
    pos += 4;
    return true;
}

// payload <-> (lowered names, cubin). decode returns false on any inconsistency (treated as corrupt).
std::vector<char> encode_payload(const std::vector<std::string> & names, const std::vector<char> & cubin) {
    std::vector<char> out;
    put_u32(out, static_cast<uint32_t>(names.size()));
    for (const auto & n : names) {
        put_u32(out, static_cast<uint32_t>(n.size()));
        out.insert(out.end(), n.begin(), n.end());
    }
    out.insert(out.end(), cubin.begin(), cubin.end());
    return out;
}

bool decode_payload(const std::vector<char> & payload, std::vector<std::string> & names, std::vector<char> & cubin) {
    size_t pos = 0;
    uint32_t count = 0;
    if (!get_u32(payload, pos, count)) {
        return false;
    }
    names.clear();
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t len = 0;
        if (!get_u32(payload, pos, len) || pos + len > payload.size()) {
            return false;
        }
        names.emplace_back(payload.data() + pos, payload.data() + pos + len);
        pos += len;
    }
    if (pos >= payload.size()) {
        return false; // no cubin bytes: never a valid entry
    }
    cubin.assign(payload.begin() + static_cast<std::ptrdiff_t>(pos), payload.end());
    return true;
}

CacheStats g_stats;

void sha256_raw(const void * data, size_t len, unsigned char out[32]) {
    sha256_hash(out, reinterpret_cast<const unsigned char *>(data), len);
}

std::string sha256_hex(const void * data, size_t len) {
    unsigned char digest[32];
    sha256_raw(data, len, digest);
    static const char hexd[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 32; ++i) {
        out[2 * i]     = hexd[digest[i] >> 4];
        out[2 * i + 1] = hexd[digest[i] & 0xF];
    }
    return out;
}

} // namespace

CacheStats & cache_stats_for_test() {
    return g_stats;
}

std::string cache_dir() {
    std::string base;
#ifdef _WIN32
    if (const char * local = std::getenv("LOCALAPPDATA")) {
        base = local;
    } else if (const char * temp = std::getenv("TEMP")) {
        base = temp;
    } else {
        base = ".";
    }
    fs::path p = fs::path(base) / "ggml" / "nvrtc-cache";
#else
    if (const char * xdg = std::getenv("XDG_CACHE_HOME")) {
        base = xdg;
    } else if (const char * home = std::getenv("HOME")) {
        base = std::string(home) + "/.cache";
    } else {
        base = "/tmp";
    }
    fs::path p = fs::path(base) / "ggml" / "nvrtc-cache";
#endif
    std::error_code ec;
    fs::create_directories(p, ec); // best-effort; a failure here just means every call is a miss
    return p.string();
}

int driver_version() {
    int v = -1;
    if (const DriverApi * d = driver()) {
        if (!d->cuDriverGetVersion || d->cuDriverGetVersion(&v) != CUDA_SUCCESS_DL) {
            v = -1;
        }
    }
    g_stats.driver_version_queries++;
    g_stats.last_driver_version = v;
    return v;
}

std::string cache_key(const std::string & source, const std::string & program_name,
                       const std::vector<std::string> & name_expressions, const CompileOptions & opts) {
    return cache_key(source, program_name, name_expressions, opts, driver_version());
}

std::string cache_key(const std::string & source, const std::string & program_name,
                       const std::vector<std::string> & name_expressions, const CompileOptions & opts,
                       int drv_version) {
    std::ostringstream blob;
    blob << "cache_format=" << kCacheFormat << "\n";
    blob << "src_len=" << source.size() << "\n" << source << "\n";
    blob << "name=" << program_name << "\n";
    blob << "arch=" << opts.arch_major << "." << opts.arch_minor << "\n";
    for (const auto & ne : name_expressions) {
        blob << "ne:" << ne << "\n";
    }
    for (const auto & o : opts.extra_options) {
        blob << "opt:" << o << "\n";
    }
    int nvmaj = -1, nvmin = -1;
    if (const NvrtcApi * n = nvrtc()) {
        if (n->nvrtcVersion && n->nvrtcVersion(&nvmaj, &nvmin) != NVRTC_SUCCESS_DL) {
            nvmaj = nvmin = -1;
        }
    }
    blob << "nvrtc_version=" << nvmaj << "." << nvmin << "\n";
    // The driver's bundled backend produces the final SASS (see nvrtc_cache.h), so a driver-only
    // upgrade must miss rather than serve a cubin compiled by the old driver.
    blob << "driver_version=" << drv_version << "\n";
    std::string s = blob.str();
    return sha256_hex(s.data(), s.size());
}

namespace {

bool read_cache_entry(const fs::path & path, std::vector<char> & out_payload) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (raw.size() < sizeof(kMagic) + 8 + 32) {
        return false; // too small to be a valid entry -- corrupt/truncated
    }
    if (std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0) {
        return false;
    }
    uint64_t payload_len = 0;
    std::memcpy(&payload_len, raw.data() + sizeof(kMagic), 8);
    const char * stored_digest = raw.data() + sizeof(kMagic) + 8;
    size_t header_size = sizeof(kMagic) + 8 + 32;
    if (raw.size() != header_size + payload_len) {
        return false; // length field disagrees with actual file size -- corrupt/truncated
    }
    const char * payload_ptr = raw.data() + header_size;
    unsigned char recomputed[32];
    sha256_raw(payload_ptr, payload_len, recomputed);
    if (std::memcmp(stored_digest, recomputed, 32) != 0) {
        return false; // hash mismatch -- corrupted, do not trust
    }
    out_payload.assign(payload_ptr, payload_ptr + payload_len);
    return true;
}

void write_cache_entry(const fs::path & path, const std::vector<char> & payload) {
    unsigned char digest[32];
    sha256_raw(payload.data(), payload.size(), digest);

    std::random_device rd;
    fs::path tmp = path;
    tmp += ".tmp" + std::to_string(rd());

    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return; // best-effort cache; a write failure just means next call recompiles
        }
        f.write(kMagic, sizeof(kMagic));
        uint64_t len = payload.size();
        f.write(reinterpret_cast<const char *>(&len), 8);
        f.write(reinterpret_cast<const char *>(digest), 32);
        if (!payload.empty()) {
            f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec); // atomic on the same filesystem; reader never sees a partial file
    if (ec) {
        fs::remove(tmp, ec);
    }
}

} // namespace

CompileResult compile(const std::string & source, const std::string & program_name,
                       const std::vector<std::string> & name_expressions, const CompileOptions & opts) {
    CompileResult result;

    std::string key = cache_key(source, program_name, name_expressions, opts, driver_version());
    fs::path entry = fs::path(cache_dir()) / (key + ".cubin");

    std::vector<char> cached;
    std::vector<std::string> cached_names;
    std::vector<char> cached_cubin;
    // A hit must carry one lowered name per requested expression, like a cold compile does
    // (callers index lowered_names in parallel with name_expressions).
    if (read_cache_entry(entry, cached) && decode_payload(cached, cached_names, cached_cubin) &&
        cached_names.size() == name_expressions.size()) {
        result.ok = true;
        result.from_cache = true;
        result.cubin = std::move(cached_cubin);
        result.lowered_names = std::move(cached_names);
        g_stats.hits++;
        return result;
    }
    bool entry_existed = fs::exists(entry);
    if (entry_existed) {
        g_stats.corrupt_recompiles++; // it existed but failed verification -- corrupt, not a plain miss
    } else {
        g_stats.misses++;
    }

    const NvrtcApi * n = nvrtc();
    if (!n) {
        result.error = "NVRTC not available: " + ensure_loaded().summary;
        return result;
    }

    nvrtcProgram prog = nullptr;
    nvrtcResult rc = n->nvrtcCreateProgram(&prog, source.c_str(), program_name.c_str(), 0, nullptr, nullptr);
    if (rc != NVRTC_SUCCESS_DL) {
        result.error = "nvrtcCreateProgram failed: " + nvrtc_error_string(rc);
        return result;
    }

    for (const auto & ne : name_expressions) {
        n->nvrtcAddNameExpression(prog, ne.c_str());
    }

    std::string arch_opt = "--gpu-architecture=sm_" + std::to_string(opts.arch_major) +
                            std::to_string(opts.arch_minor);
    std::vector<std::string> opt_storage;
    opt_storage.push_back(arch_opt);
    for (const auto & o : opts.extra_options) {
        opt_storage.push_back(o);
    }
    std::vector<const char *> opt_ptrs;
    for (const auto & o : opt_storage) {
        opt_ptrs.push_back(o.c_str());
    }

    rc = n->nvrtcCompileProgram(prog, static_cast<int>(opt_ptrs.size()), opt_ptrs.data());

    size_t log_size = 0;
    if (n->nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS_DL && log_size > 1) {
        result.log.resize(log_size);
        n->nvrtcGetProgramLog(prog, result.log.data());
        if (!result.log.empty() && result.log.back() == '\0') {
            result.log.pop_back();
        }
    }

    if (rc != NVRTC_SUCCESS_DL) {
        result.error = "nvrtcCompileProgram failed: " + nvrtc_error_string(rc) +
                        (result.log.empty() ? "" : ("\n" + result.log));
        n->nvrtcDestroyProgram(&prog);
        return result;
    }

    for (const auto & ne : name_expressions) {
        const char * lowered = nullptr;
        if (n->nvrtcGetLoweredName(prog, ne.c_str(), &lowered) == NVRTC_SUCCESS_DL && lowered) {
            result.lowered_names.emplace_back(lowered);
        } else {
            result.lowered_names.emplace_back();
        }
    }

    size_t cubin_size = 0;
    if (n->nvrtcGetCUBINSize(prog, &cubin_size) != NVRTC_SUCCESS_DL || cubin_size == 0) {
        result.error = "nvrtcGetCUBINSize failed or empty cubin";
        n->nvrtcDestroyProgram(&prog);
        return result;
    }
    result.cubin.resize(cubin_size);
    if (n->nvrtcGetCUBIN(prog, result.cubin.data()) != NVRTC_SUCCESS_DL) {
        result.error = "nvrtcGetCUBIN failed";
        n->nvrtcDestroyProgram(&prog);
        return result;
    }

    n->nvrtcDestroyProgram(&prog);

    result.ok = true;
    write_cache_entry(entry, encode_payload(result.lowered_names, result.cubin));
    return result;
}

} // namespace ggml_cuda_dl
