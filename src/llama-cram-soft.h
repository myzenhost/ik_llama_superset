// llama-cram-soft.h -- READ-ONLY COMPRESSED BLOCK STORE (software analogue of Meta's CRAM tier).
//
// The input byte span is cut into fixed-size blocks (default 64 KiB). Each block is stored either RAW or
// compressed; a block that does not shrink by at least 5% is stored RAW (no capacity "lie"). Reads decode
// blocks on demand into a bounded LRU "hot window" of decoded blocks. There is no write path: promote()
// hands back a mutable decoded COPY.
//
// Codec: neither lz4 nor zstd has headers/libs usable in this tree (only runtime .so.1 files, no headers,
// nothing vendored), so the codec is hand-written: optional byte-plane split (stride 1 or 2, e.g. the two
// bytes of an f16), then per plane the smaller of {raw, zero-friendly run-length, canonical Huffman}.
// Each block carries a CRC32 of its stored bytes, checked on decode (disable with verify=false).
//
// STATUS: used by common_prompt_checkpoint::cram_compress() (--ctx-ckpt-cram-soft), see common/common.cpp.
#pragma once

#include "llama.h" // LLAMA_API

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

enum llama_cram_codec {
    LLAMA_CRAM_CODEC_RAW = 0,        // every block stored raw (control)
    LLAMA_CRAM_CODEC_PLANE_RLE = 1,  // byte planes, each raw or RLE
    LLAMA_CRAM_CODEC_PLANE_HUF = 2,  // byte planes, each raw, RLE or Huffman (smallest wins)
};

struct llama_cram_soft_params {
    size_t            block_size = 64 * 1024;
    llama_cram_codec  codec      = LLAMA_CRAM_CODEC_PLANE_HUF;
    int               stride     = 1;   // byte-plane count: 1 or 2
    size_t            hot_blocks = 16;  // total decoded blocks kept (split across shards)
    size_t            shards     = 1;   // hot-window shards (one mutex each); hot_blocks/shards per shard, min 1
    bool              verify     = true; // check per-block CRC32 on decode; mismatch throws
};

struct llama_cram_soft_stats {
    uint64_t compressed_bytes = 0; // sum of stored block payload sizes (RAW blocks count at raw size)
    uint64_t raw_bytes        = 0;
    uint64_t meta_bytes       = 0; // per-block table
    uint64_t blocks           = 0;
    uint64_t raw_blocks       = 0; // blocks that fell back to RAW
    uint64_t hits             = 0;
    uint64_t misses           = 0;
    uint64_t decode_ns        = 0;
    uint64_t crc_failures     = 0;
};

class LLAMA_API llama_cram_soft_store {
public:
    llama_cram_soft_store(const void * data, size_t size, const llama_cram_soft_params & p = {});
    llama_cram_soft_store(const llama_cram_soft_store &) = delete;
    llama_cram_soft_store & operator=(const llama_cram_soft_store &) = delete;

    size_t size() const { return size_; }
    size_t block_size() const { return params_.block_size; }
    size_t n_blocks() const { return meta_.size(); }

    // Copies up to len bytes starting at offset into dst; returns bytes copied (0 past the end).
    // Thread-safe. Throws std::runtime_error on checksum mismatch or undecodable block.
    size_t read(size_t offset, size_t len, void * dst);

    // Mutable decoded copy of [offset, offset+len) -- the "promote to DRAM" operation.
    std::vector<uint8_t> promote(size_t offset, size_t len);

    // Decode one block into dst (block_len(blk) bytes) bypassing the hot window and every lock/allocation:
    // no mutex, no heap use. Returns false on checksum mismatch or malformed data (never throws). Used by the
    // signal-handler fault test.
    bool decode_block_nolock(size_t blk, uint8_t * dst) const noexcept;
    size_t block_len(size_t blk) const;

    llama_cram_soft_stats stats() const;

    // Test hooks.
    std::vector<size_t> hot_blocks_mru_first() const; // shard 0 only meaningful when shards==1
    uint8_t * debug_stored_bytes_of_block(size_t blk, size_t * len); // lets a test corrupt storage

private:
    struct block_meta { uint64_t off; uint32_t csize; uint32_t crc; bool raw; };
    struct entry { std::list<size_t>::iterator it; std::shared_ptr<std::vector<uint8_t>> data; };
    struct shard {
        std::mutex mu;
        std::list<size_t> lru; // front = most recent
        std::unordered_map<size_t, entry> map;
    };

    std::shared_ptr<std::vector<uint8_t>> get_block(size_t blk);

    llama_cram_soft_params params_;
    size_t size_ = 0;
    std::vector<block_meta> meta_;
    std::vector<uint8_t> store_;
    std::vector<std::unique_ptr<shard>> shards_;
    size_t shard_cap_ = 1;
    mutable llama_cram_soft_stats st_;
    std::atomic<uint64_t> hits_{0}, misses_{0}, decode_ns_{0}, crc_fail_{0};
};
