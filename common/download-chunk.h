#pragma once

// Chunk planner and resume bitmap for the parallel range download in download.cpp.
//
// Behaviour taken from NEXA SDK's chunked parallel downloader (BSD-3-Clause): a large file is cut into fixed-size
// byte ranges, several connections fetch ranges at once, and a persisted bitmap of finished ranges lets an
// interrupted download continue where it stopped. Described and reimplemented here, no code copied.
// Row S69-ENGINE-MERGE-NEXA-2026-09-27-01.

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct common_chunk {
    size_t offset = 0;
    size_t len    = 0;
};

static constexpr size_t COMMON_CHUNK_DEFAULT_SIZE = (size_t) 16 << 20; // 16 MiB
static constexpr size_t COMMON_CHUNK_MAX_COUNT    = 128;
static constexpr const char * COMMON_CHUNK_SIDECAR_SUFFIX = ".chunks"; // appended to the partial file's path

// chunk size actually used: the requested one, grown when the file would need more than max_chunks pieces
inline size_t common_chunk_effective_size(size_t total, size_t chunk_size, size_t max_chunks = COMMON_CHUNK_MAX_COUNT) {
    if (total == 0) {
        return 0;
    }
    if (chunk_size == 0) {
        chunk_size = COMMON_CHUNK_DEFAULT_SIZE;
    }
    if (max_chunks == 0) {
        max_chunks = 1;
    }
    const size_t n = (total + chunk_size - 1) / chunk_size;
    if (n > max_chunks) {
        chunk_size = (total + max_chunks - 1) / max_chunks;
    }
    return chunk_size;
}

// tiling of [0, total): consecutive ranges, last one short, no gap and no overlap
inline std::vector<common_chunk> common_chunk_plan(size_t total, size_t chunk_size, size_t max_chunks = COMMON_CHUNK_MAX_COUNT) {
    std::vector<common_chunk> out;
    const size_t cs = common_chunk_effective_size(total, chunk_size, max_chunks);
    for (size_t off = 0; cs > 0 && off < total; off += cs) {
        out.push_back({off, off + cs <= total ? cs : total - off});
    }
    return out;
}

// bitmap sidecar: "llama-chunks-v1 <total> <chunk_size> <n>\n" then n characters, '1' = range finished
inline bool common_chunk_bitmap_save(const std::string & path, size_t total, size_t chunk_size, const std::vector<char> & done) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        f << "llama-chunks-v1 " << total << " " << chunk_size << " " << done.size() << "\n";
        for (char c : done) {
            f << (c ? '1' : '0');
        }
        f << "\n";
        if (!f) {
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

// loads the bitmap; a missing file, another file size/chunk size, or a damaged body gives all-zero (false)
inline bool common_chunk_bitmap_load(const std::string & path, size_t total, size_t chunk_size, size_t n, std::vector<char> & done) {
    done.assign(n, 0);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::string magic;
    size_t t = 0, cs = 0, cnt = 0;
    if (!(f >> magic >> t >> cs >> cnt) || magic != "llama-chunks-v1" || t != total || cs != chunk_size || cnt != n) {
        return false;
    }
    std::string bits;
    if (!(f >> bits) || bits.size() != n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (bits[i] != '0' && bits[i] != '1') {
            done.assign(n, 0);
            return false;
        }
        done[i] = bits[i] == '1';
    }
    return true;
}

// bytes covered by the finished ranges
inline size_t common_chunk_done_bytes(const std::vector<common_chunk> & plan, const std::vector<char> & done) {
    size_t sum = 0;
    for (size_t i = 0; i < plan.size() && i < done.size(); ++i) {
        if (done[i]) {
            sum += plan[i].len;
        }
    }
    return sum;
}
