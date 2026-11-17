// common_disk_tier: a directory of checksummed blobs with a hard size cap and an SSD write budget.
//
// It is the NVMe tier (T2) of the fork's tier manager (temp/research/findings/S69_TIERED_MEMORY_AIDAPTIV_2026-10-05.md,
// section "Design"). The first user is server_prompt_cache (--cache-disk-dir): when the host-RAM cache must evict a prompt
// state it is written here instead of being dropped, and read back when a request's prefix matches it.
//
// What it guarantees, each with a test in tests/probe-disk-tier.cpp:
//   * a blob is either fully on disk and readable or not there at all: it is written to <name>.tmp and renamed, and a
//     failed or short write leaves nothing behind;
//   * every read is verified against the checksum, the lengths and the caller's tag before any byte is handed out; a blob
//     that fails is deleted and reported as a miss, so a corrupt file can never be fed to the model;
//   * the sum of the file sizes never exceeds the cap (put() refuses, the caller decides what to evict);
//   * SSD wear is a measured software budget: bytes_written is counted and an optional budget in bytes per hour (a token
//     bucket) refuses writes beyond it. This is the part Phison aiDAPTIV+ buys as high-endurance flash hardware.
//   * the tier never eats the last free space of the volume: put() asks the file system for the free space every time
//     (other processes fill the disk too) and refuses the write, counted in n_refused_space, when the volume has less
//     than 10 GiB free or when the write would take the tier past a quarter of the space it could have (see the free-space
//     guard below). Policy from lmstudio-ai/mlx-engine @86f20eb4 (MIT) prompt_cache/disk_budget.py:27-29,82-91, rewritten.
//   * a server owns its directory: init() removes the pcache-*/pckpt-* .bin and .tmp files a crashed earlier run left, and the
//     destructor removes everything this object wrote. Nothing outlives the process (persistence across restarts is the
//     next slice, see the findings file).
//
// Not thread-safe: server_prompt_cache is only touched on the server's main loop thread.
//
// Design sources read for this component (ideas only, no code copied; every one of them is Apache-2.0 or MIT):
//   LMCache paper arXiv 2510.09665 s5.1-5.3 (large chunks, zero extra copies), SGLang HiCache docs hicache_design.html
//   ("write_through_selective": back up only what has shown reuse), Dynamo KVBM design (G3 disk pool with dedup by
//   sequence hash), ik_llama.cpp PR 2377 (block files with a model fingerprint, atomic rename, self-healing on corruption).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct common_disk_tier_stats {
    uint64_t n_put                = 0; // blobs written
    uint64_t bytes_written        = 0; // file bytes written, headers included (the endurance counter)
    uint64_t n_get                = 0; // blobs read back and verified
    uint64_t bytes_read           = 0;
    uint64_t n_put_ckpt           = 0; // of n_put / bytes_written: the blobs that hold a context checkpoint (is_ckpt = true)
    uint64_t bytes_written_ckpt   = 0;
    uint64_t n_get_ckpt           = 0; // of n_get / bytes_read: checkpoint blobs
    uint64_t bytes_read_ckpt      = 0;
    double   write_seconds        = 0; // wall time spent inside put() (write, flush, size check): bytes_written / write_seconds is the measured write speed
    double   read_seconds         = 0; // wall time spent inside a successful get() (read + checksum): bytes_read / read_seconds is the measured read speed
    uint64_t n_refused_cap        = 0; // put() refused: it would exceed the size cap
    uint64_t n_refused_budget     = 0; // put() refused: the write budget is spent
    uint64_t n_refused_space      = 0; // put() refused: the volume is (nearly) full -- the free-space guard
    uint64_t n_put_failed         = 0; // put() hit an I/O error (nothing was left behind)
    uint64_t n_get_failed         = 0; // get() found the blob missing, truncated or foreign (it was deleted)
    uint64_t n_checksum_failed    = 0; // get() found the right length but the wrong bytes (it was deleted)
    uint64_t n_erased             = 0; // blobs removed by erase()
    uint64_t n_pipelined_put      = 0; // put() calls that overlapped the checksum with the file write (--cache-disk-io-chunk)
    uint64_t n_pipelined_get      = 0; // get() calls that overlapped the file read with the checksum
    uint64_t n_io_chunks          = 0; // chunks moved by those pipelined calls
    size_t   used_bytes           = 0; // current sum of file sizes
    size_t   n_blobs              = 0; // current blob count
};

class common_disk_tier {
public:
    // seconds on any monotonic clock; injectable so the budget can be tested without sleeping
    using clock_fn = std::function<double()>;
    // bytes free on the volume that holds `dir` (what an unprivileged writer may use), or a negative value when it cannot be
    // told. Injectable so the free-space guard can be tested without filling a disk; the default is std::filesystem::space().
    using free_space_fn = std::function<int64_t(const std::string & dir)>;

    // Free-space guard defaults (mlx-engine disk_budget.py:27-29,82-91): below MIN_FREE_BYTES free the tier writes nothing;
    // otherwise the tier may hold at most 1/FREE_SPACE_DIVISOR of the space it could have (free now + what it already holds),
    // and never more than limit_bytes. With an empty tier that is min(limit_bytes, free / 4).
    static constexpr int64_t DEFAULT_MIN_FREE_BYTES    = 10ll << 30; // 10 GiB
    static constexpr int64_t DEFAULT_FREE_SPACE_DIVISOR = 4;

    // limit_bytes: hard cap on the sum of file sizes (0 = no tier: every put() is refused)
    // write_budget_bytes_per_hour: 0 = unlimited. Otherwise a token bucket that refills continuously and holds at most
    // one hour of budget, so a burst is allowed but the long-run average is capped.
    common_disk_tier(std::string dir, size_t limit_bytes, size_t write_budget_bytes_per_hour, clock_fn clock = nullptr);
    ~common_disk_tier();

    common_disk_tier(const common_disk_tier &) = delete;
    common_disk_tier & operator=(const common_disk_tier &) = delete;

    // create the directory (with parents) and remove stale pcache-* files. False + err when the directory is unusable.
    bool init(int & n_stale_removed, std::string & err);

    // Write two byte vectors as one blob. `tag` is stored and must be presented again to get(); the caller uses it for
    // something that identifies the owner (the token count). Returns a non-zero blob id, or 0 when refused or failed
    // (see the counters for which). `a` must be non-empty.
    // is_ckpt: the blob holds a context checkpoint (row S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01). It shares the cap,
    // the write budget and the checksum with every other blob; the only differences are the file name (pckpt-<id>.bin, so a
    // test or an operator can tell the two kinds apart) and the *_ckpt counters.
    uint64_t put(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, int64_t tag, bool is_ckpt = false);

    // Read a blob back. False when it is unknown, missing, truncated, carries another tag or fails the checksum; in every
    // such case the blob is deleted (the caller treats it as a cache miss and recomputes).
    bool get(uint64_t id, int64_t tag, std::vector<uint8_t> & a, std::vector<uint8_t> & b);

    void erase(uint64_t id);

    // ---- async restore support (row S69-ENGREV-RTPLLM-ASYNC-DISK-RESTORE-2026-10-27-01; see common_disk_loader below) ----------
    // A read that can run on a worker thread while the main loop keeps decoding. The tier itself stays single-threaded: the worker
    // calls only the pure static read_blob_file() on a snapshot taken by describe() on the main thread, and the main thread books the
    // outcome with finish_async_get(). Behaviour identical to get(): same header/tag/length/checksum verification, same bookkeeping
    // (a blob that fails is deleted and counted).
    enum class read_status { OK, IO, CHECKSUM };
    // path and file size of blob `id`; false when unknown (main thread only)
    bool describe(uint64_t id, std::string & path, size_t & file_bytes, bool & ckpt) const;
    // PURE: touches no tier state, safe on any thread. Reads, verifies and returns the two payloads.
    static read_status read_blob_file(const std::string & path, size_t file_bytes, int64_t tag, std::vector<uint8_t> & a, std::vector<uint8_t> & b);
    // main thread: book what read_blob_file() returned for blob `id` exactly as get() would have (stats on OK, delete + count on failure)
    void finish_async_get(uint64_t id, read_status st, double seconds);

    // Overlap file I/O with the checksum, in chunks of this many bytes (0 = off, the serial path). Only payloads larger than one
    // chunk are pipelined. The file written is byte-identical either way. Row S69-LINK-STORAGE-OFFLOAD-1014 (DeepSpeed aio
    // overlap, csrc/aio/common/deepspeed_aio_common.cpp:192-233, applied to the checksum stage).
    void   set_io_chunk_bytes(size_t n) { io_chunk_bytes_ = n; }
    size_t io_chunk_bytes() const { return io_chunk_bytes_; }

    // Free-space guard (row S69-MLX-PICK-DISK-TIER-FREE-SPACE-GUARD-2026-10-05-01). put() re-checks it on every write.
    //   set_free_space_fn: replace the free-space query (nullptr restores std::filesystem::space); a test hook.
    //   set_free_space_policy: min_free_bytes <= 0 turns the floor off, divisor <= 0 turns the fraction cap off.
    // When the query cannot tell (it returns a value below zero) the guard stays out of the way: put() behaves as before.
    void set_free_space_fn(free_space_fn fn);
    void set_free_space_policy(int64_t min_free_bytes, int64_t divisor) { min_free_bytes_ = min_free_bytes; free_divisor_ = divisor; }
    // would the free-space guard accept FILE bytes right now? (asks the file system; does not count a refusal)
    bool space_allows_bytes(size_t file_bytes);

    // checksum(b, checksum(a, seed)) computed through the incremental stream in pieces of `chunk` bytes (test hook: must equal the one-shot).
    static uint64_t checksum_chunked(const void * a, size_t na, const void * b, size_t nb, uint64_t seed, size_t chunk);

    // would a blob with these payload sizes fit under the cap right now? / is the budget able to pay for it?
    bool   fits(size_t payload_a, size_t payload_b) const;
    bool   budget_allows(size_t payload_a, size_t payload_b);
    // the same two questions for a number of FILE bytes (several blobs asked about at once)
    bool   fits_bytes(size_t file_bytes) const;
    bool   budget_allows_bytes(size_t file_bytes);

    size_t file_size_for(size_t payload_a, size_t payload_b) const;
    size_t limit_bytes() const { return limit_bytes_; }
    size_t used_bytes() const { return stats_.used_bytes; }
    bool   has(uint64_t id) const { return blobs_.count(id) != 0; }
    const common_disk_tier_stats & stats() const { return stats_; }
    const std::string & dir() const { return dir_; }
    std::string path_of(uint64_t id) const; // for tests

    // 64-bit non-cryptographic checksum (corruption detection, not security). Own design: four multiply-rotate lanes over
    // 32-byte stripes so it runs at memory speed, a byte-wise tail, and a final avalanche that also mixes in the length.
    static uint64_t checksum(const void * data, size_t n, uint64_t seed = 0);

private:
    struct blob {
        size_t file_bytes = 0;
        bool   ckpt       = false;
    };

    std::string path_for(uint64_t id, bool ckpt) const;
    void refill_budget();
    void drop_blob(uint64_t id, bool count_as_erase);

    std::string dir_;
    size_t      limit_bytes_;
    size_t      budget_per_hour_;
    clock_fn    clock_;
    size_t      io_chunk_bytes_ = 0;

    free_space_fn free_space_;
    int64_t     min_free_bytes_      = DEFAULT_MIN_FREE_BYTES;
    int64_t     free_divisor_        = DEFAULT_FREE_SPACE_DIVISOR;
    bool        space_guard_logged_  = false; // the first refusal is logged, the rest only counted

    double      bucket_      = 0.0; // bytes of write budget available
    double      bucket_time_ = 0.0;

    uint64_t    next_id_ = 1;
    std::unordered_map<uint64_t, blob> blobs_;
    common_disk_tier_stats stats_;
};

// Asynchronous restore with load-join dedup and a per-task deadline (design from rtp-llm BlockTreeLoader.cc:277-357: async load,
// LoadJoinRegistry, per-task deadline; Apache-2.0, clean-room). start() launches the blob read on a worker thread, or JOINS the read
// already in flight for the same blob so N identical requests cost one read. The caller (the server main loop) polls; a task past its
// deadline reports TIMEOUT (the request falls back to recompute) while the read still finishes and is booked. Main thread only for
// every method; the worker thread runs only common_disk_tier::read_blob_file().
// STATUS: the primitive. Wiring it into server_prompt_cache::load / the slot state machine ("loading" parked slot) is the separate
// DEFERRED-ON-PREREQ part of the row (needs a slot LOADING state in update_slots()).
class common_disk_loader {
public:
    enum class state { NONE, LOADING, READY, FAILED, TIMEOUT };
    using clock_fn = std::function<double()>;

    explicit common_disk_loader(std::shared_ptr<common_disk_tier> tier, clock_fn clock = nullptr);
    ~common_disk_loader(); // waits for every read in flight

    // false: the blob is unknown to the tier (nothing started). true: started, or joined an existing task (n_joined()++)
    bool  start(uint64_t id, int64_t tag, double deadline_seconds);
    state poll(uint64_t id);                                       // books a finished read into the tier on first sight
    bool  take(uint64_t id, std::vector<uint8_t> & a, std::vector<uint8_t> & b); // READY only: moves the payload out, forgets the task
    void  forget(uint64_t id);                                     // drop a task (a still-running read is detached and booked by reap())
    void  reap();                                                  // book + discard finished abandoned reads

    uint64_t n_started() const { return n_started_; }
    uint64_t n_joined()  const { return n_joined_; }
    uint64_t n_timeout() const { return n_timeout_; }

private:
    struct task;
    std::shared_ptr<common_disk_tier> tier_;
    clock_fn clock_;
    std::unordered_map<uint64_t, std::shared_ptr<task>> tasks_;
    std::vector<std::shared_ptr<task>> abandoned_;
    uint64_t n_started_ = 0, n_joined_ = 0, n_timeout_ = 0;
};

// A counted reference to one blob of a tier. Context checkpoints are copied freely (server_prompt::clone(), the slot's own list
// and the cache entry's list are separate copies of the same checkpoint), so "who deletes the file" cannot be answered by
// ownership of one copy: every copy holds a shared_ptr to the same common_disk_blob, and the blob is erased from the tier when
// the LAST copy goes away. A copy can therefore never be left pointing at a file somebody else deleted, and a file never outlives
// the last checkpoint that could read it. The handle keeps the tier alive (shared_ptr), so releasing it is always safe.
// Not thread-safe, like the tier: the server touches checkpoints only on its main loop thread.
class common_disk_blob {
public:
    common_disk_blob(std::shared_ptr<common_disk_tier> tier, uint64_t id) : tier_(std::move(tier)), id_(id) {}
    ~common_disk_blob() {
        if (tier_ && id_ != 0) {
            tier_->erase(id_);
        }
    }
    common_disk_blob(const common_disk_blob &) = delete;
    common_disk_blob & operator=(const common_disk_blob &) = delete;

    uint64_t id() const { return id_; }
    const std::shared_ptr<common_disk_tier> & tier() const { return tier_; }

private:
    std::shared_ptr<common_disk_tier> tier_;
    uint64_t id_;
};
