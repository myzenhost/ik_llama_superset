#include "disk-tier.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

// ---- file layout (host byte order; the file never outlives the process that wrote it) ----------------------------------
//   u32 magic "PCD1", u32 version, i64 tag, u64 len_a, u64 len_b, u64 checksum, then len_a bytes of `a`, len_b bytes of `b`.
// checksum = checksum(b, checksum(a, tag)): chained, so swapping the two payloads or changing the tag is detected too.
static constexpr uint32_t DISK_TIER_MAGIC   = 0x31444350; // "PCD1"
static constexpr uint32_t DISK_TIER_VERSION = 1;
static constexpr size_t   DISK_TIER_HEADER  = 4 + 4 + 8 + 8 + 8 + 8;

static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

static inline uint64_t load64(const uint8_t * p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint64_t common_disk_tier::checksum(const void * data, size_t n, uint64_t seed) {
    // two odd 64-bit constants with well-spread bits (golden-ratio derived); any such pair works for corruption detection
    constexpr uint64_t K1 = 0x9E3779B97F4A7C15ull;
    constexpr uint64_t K2 = 0xC2B2AE3D27D4EB4Full;

    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t lane[4] = { seed + K1, seed + K2, seed ^ K1, seed - K2 };

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        for (int k = 0; k < 4; k++) {
            lane[k] = rotl64(lane[k] ^ (load64(p + i + 8 * k) * K2), 29) * K1;
        }
    }

    uint64_t tail = seed ^ (uint64_t) n;
    for (; i + 8 <= n; i += 8) {
        tail = rotl64(tail ^ (load64(p + i) * K2), 31) * K1;
    }
    for (; i < n; i++) {
        tail = (tail ^ p[i]) * 0x100000001B3ull;
        tail = rotl64(tail, 13);
    }

    uint64_t h = rotl64(lane[0], 1) + rotl64(lane[1], 7) + rotl64(lane[2], 12) + rotl64(lane[3], 18);
    h ^= tail * K2;
    // final avalanche
    h ^= h >> 33; h *= K2;
    h ^= h >> 29; h *= K1;
    h ^= h >> 32;
    return h;
}

// Incremental form of common_disk_tier::checksum(): feeding the same bytes in any number of pieces gives the same value as the one-shot call.
// It exists so that a read can be checksummed while the next chunk is still being read from the file (--cache-disk-io-chunk).
struct disk_tier_csum_stream {
    static constexpr uint64_t K1 = 0x9E3779B97F4A7C15ull;
    static constexpr uint64_t K2 = 0xC2B2AE3D27D4EB4Full;

    uint64_t lane[4];
    uint8_t  pend[32];
    size_t   npend = 0;
    uint64_t seed;
    uint64_t total = 0;

    explicit disk_tier_csum_stream(uint64_t seed_) : seed(seed_) {
        lane[0] = seed + K1; lane[1] = seed + K2; lane[2] = seed ^ K1; lane[3] = seed - K2;
    }

    void stripe(const uint8_t * p) {
        for (int k = 0; k < 4; k++) {
            lane[k] = rotl64(lane[k] ^ (load64(p + 8 * k) * K2), 29) * K1;
        }
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        total += n;
        if (npend > 0) {
            const size_t take = std::min(n, 32 - npend);
            std::memcpy(pend + npend, p, take);
            npend += take; p += take; n -= take;
            if (npend < 32) {
                return;
            }
            stripe(pend);
            npend = 0;
        }
        for (; n >= 32; p += 32, n -= 32) {
            stripe(p);
        }
        if (n > 0) {
            std::memcpy(pend, p, n);
            npend = n;
        }
    }

    uint64_t finish() {
        uint64_t tail = seed ^ total;
        size_t i = 0;
        for (; i + 8 <= npend; i += 8) {
            tail = rotl64(tail ^ (load64(pend + i) * K2), 31) * K1;
        }
        for (; i < npend; i++) {
            tail = (tail ^ pend[i]) * 0x100000001B3ull;
            tail = rotl64(tail, 13);
        }
        uint64_t h = rotl64(lane[0], 1) + rotl64(lane[1], 7) + rotl64(lane[2], 12) + rotl64(lane[3], 18);
        h ^= tail * K2;
        h ^= h >> 33; h *= K2;
        h ^= h >> 29; h *= K1;
        h ^= h >> 32;
        return h;
    }
};

uint64_t common_disk_tier::checksum_chunked(const void * a, size_t na, const void * b, size_t nb, uint64_t seed, size_t chunk) {
    chunk = std::max<size_t>(chunk, 1);
    disk_tier_csum_stream sa(seed);
    for (size_t o = 0; o < na; o += chunk) {
        sa.update(static_cast<const uint8_t *>(a) + o, std::min(chunk, na - o));
    }
    disk_tier_csum_stream sb(sa.finish());
    for (size_t o = 0; o < nb; o += chunk) {
        sb.update(static_cast<const uint8_t *>(b) + o, std::min(chunk, nb - o));
    }
    return sb.finish();
}

static double steady_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Bytes an unprivileged writer may still put on the volume holding `dir` (space_info::available, the same quantity as
// Python's shutil.disk_usage().free on POSIX, which mlx-engine's disk_budget.py:84 reads), or -1 when it cannot be told.
static int64_t fs_free_space(const std::string & dir) {
    std::error_code ec;
    const fs::space_info si = fs::space(dir, ec);
    if (ec || si.available == static_cast<std::uintmax_t>(-1)) {
        return -1;
    }
    return si.available > (std::uintmax_t) INT64_MAX ? INT64_MAX : (int64_t) si.available;
}

common_disk_tier::common_disk_tier(std::string dir, size_t limit_bytes, size_t write_budget_bytes_per_hour, clock_fn clock)
    : dir_(std::move(dir)), limit_bytes_(limit_bytes), budget_per_hour_(write_budget_bytes_per_hour),
      clock_(clock ? std::move(clock) : clock_fn(steady_seconds)), free_space_(fs_free_space) {
    bucket_      = (double) budget_per_hour_; // start full: the first hour may use a full hour of budget
    bucket_time_ = clock_();
}

void common_disk_tier::set_free_space_fn(free_space_fn fn) {
    free_space_ = fn ? std::move(fn) : free_space_fn(fs_free_space);
}

// The free-space guard. Policy (mlx-engine disk_budget.py:82-91, provisional stage): under min_free_bytes_ free, write
// nothing; otherwise the store may be at most free/divisor. mlx-engine samples free space once, when it computes the budget,
// with its store (nearly) empty; this runs on every put(), when the tier's own blobs already occupy part of what was free,
// so the fraction is taken of free + used (the space the tier could have), which equals free/divisor for an empty tier.
// The configured cap (limit_bytes_) is applied by fits_bytes(); this only ever lowers it.
bool common_disk_tier::space_allows_bytes(size_t file_bytes) {
    if (min_free_bytes_ <= 0 && free_divisor_ <= 0) {
        return true;
    }
    const int64_t free_now = free_space_ ? free_space_(dir_) : -1;
    if (free_now < 0) {
        return true; // unknown: no opinion, the cap and the budget still apply
    }
    if (min_free_bytes_ > 0 && free_now < min_free_bytes_) {
        return false;
    }
    if (free_divisor_ > 0) {
        const uint64_t could_have = (uint64_t) free_now + (uint64_t) stats_.used_bytes;
        const uint64_t allowance  = could_have / (uint64_t) free_divisor_;
        if ((uint64_t) stats_.used_bytes + (uint64_t) file_bytes > allowance) {
            return false;
        }
    }
    return true;
}

common_disk_tier::~common_disk_tier() {
    std::error_code ec;
    for (const auto & kv : blobs_) {
        fs::remove(path_of(kv.first), ec);
    }
}

std::string common_disk_tier::path_for(uint64_t id, bool ckpt) const {
    return (fs::path(dir_) / ((ckpt ? "pckpt-" : "pcache-") + std::to_string(id) + ".bin")).string();
}

std::string common_disk_tier::path_of(uint64_t id) const {
    auto it = blobs_.find(id);
    return path_for(id, it != blobs_.end() && it->second.ckpt);
}

bool common_disk_tier::init(int & n_stale_removed, std::string & err) {
    n_stale_removed = 0;
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (ec) {
        err = ec.message();
        return false;
    }
    for (const auto & entry : fs::directory_iterator(dir_, ec)) {
        const std::string name = entry.path().filename().string();
        const bool ours = (name.rfind("pcache-", 0) == 0 || name.rfind("pckpt-", 0) == 0) &&
            ((name.size() > 4 && name.compare(name.size() - 4, 4, ".bin") == 0) ||
             (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0));
        if (ours) {
            std::error_code ec2;
            if (fs::remove(entry.path(), ec2)) {
                n_stale_removed++;
            }
        }
    }
    if (ec) {
        err = ec.message();
        return false;
    }
    return true;
}

size_t common_disk_tier::file_size_for(size_t payload_a, size_t payload_b) const {
    return DISK_TIER_HEADER + payload_a + payload_b;
}

bool common_disk_tier::fits(size_t payload_a, size_t payload_b) const {
    return fits_bytes(file_size_for(payload_a, payload_b));
}

bool common_disk_tier::fits_bytes(size_t file_bytes) const {
    return limit_bytes_ > 0 && stats_.used_bytes + file_bytes <= limit_bytes_;
}

void common_disk_tier::refill_budget() {
    if (budget_per_hour_ == 0) {
        return;
    }
    const double now = clock_();
    const double dt  = std::max(0.0, now - bucket_time_);
    bucket_time_ = now;
    bucket_ = std::min((double) budget_per_hour_, bucket_ + dt * (double) budget_per_hour_ / 3600.0);
}

bool common_disk_tier::budget_allows(size_t payload_a, size_t payload_b) {
    return budget_allows_bytes(file_size_for(payload_a, payload_b));
}

bool common_disk_tier::budget_allows_bytes(size_t file_bytes) {
    if (budget_per_hour_ == 0) {
        return true;
    }
    refill_budget();
    return bucket_ >= (double) file_bytes;
}

void common_disk_tier::drop_blob(uint64_t id, bool count_as_erase) {
    auto it = blobs_.find(id);
    if (it == blobs_.end()) {
        return;
    }
    std::error_code ec;
    fs::remove(path_of(id), ec);
    stats_.used_bytes -= std::min(stats_.used_bytes, it->second.file_bytes);
    blobs_.erase(it);
    stats_.n_blobs = blobs_.size();
    if (count_as_erase) {
        stats_.n_erased++;
    }
}

void common_disk_tier::erase(uint64_t id) {
    drop_blob(id, true);
}

uint64_t common_disk_tier::put(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, int64_t tag, bool is_ckpt) {
    if (a.empty()) {
        return 0;
    }
    if (!fits(a.size(), b.size())) {
        stats_.n_refused_cap++;
        return 0;
    }
    // re-checked on every write: other processes fill the volume too (row S69-MLX-PICK-DISK-TIER-FREE-SPACE-GUARD-2026-10-05-01)
    if (!space_allows_bytes(file_size_for(a.size(), b.size()))) {
        stats_.n_refused_space++;
        if (!space_guard_logged_) {
            space_guard_logged_ = true;
            const int64_t free_now = free_space_ ? free_space_(dir_) : -1;
            std::fprintf(stderr, "common_disk_tier: %s: free-space guard engaged (%lld bytes free, floor %lld bytes, tier holds %llu bytes "
                         "and may hold 1/%lld of free + used); writes are refused and counted in n_refused_space, this message is printed once\n",
                         dir_.c_str(), (long long) free_now, (long long) min_free_bytes_, (unsigned long long) stats_.used_bytes, (long long) free_divisor_);
        }
        return 0;
    }
    if (!budget_allows(a.size(), b.size())) {
        stats_.n_refused_budget++;
        return 0;
    }

    const size_t   file_bytes = file_size_for(a.size(), b.size());
    const uint64_t id         = next_id_++;
    const std::string path    = path_for(id, is_ckpt);
    const std::string tmp     = path.substr(0, path.size() - 4) + ".tmp";

    // the attempt is paid for whether or not it succeeds: a failed write can still have worn the flash
    if (budget_per_hour_ > 0) {
        bucket_ = std::max(0.0, bucket_ - (double) file_bytes);
    }

    const double t_io0 = steady_seconds();
    const bool pipelined = io_chunk_bytes_ > 0 && a.size() + b.size() > io_chunk_bytes_;

    bool ok = false;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (f) {
            const uint32_t magic   = DISK_TIER_MAGIC;
            const uint32_t version = DISK_TIER_VERSION;
            const uint64_t la = a.size(), lb = b.size();
            uint64_t h = 0;
            auto write_header = [&]() {
                f.write(reinterpret_cast<const char *>(&magic),   sizeof(magic));
                f.write(reinterpret_cast<const char *>(&version), sizeof(version));
                f.write(reinterpret_cast<const char *>(&tag),     sizeof(tag));
                f.write(reinterpret_cast<const char *>(&la),      sizeof(la));
                f.write(reinterpret_cast<const char *>(&lb),      sizeof(lb));
                f.write(reinterpret_cast<const char *>(&h),       sizeof(h));
            };
            if (pipelined) {
                // The checksum goes into the header, which comes first in the file, so the header is written with a zero
                // placeholder, the payload is written by a second thread while this thread checksums the same (read-only)
                // buffers, and the header is rewritten in place once the checksum is known. The file bytes are identical to
                // the serial path's.
                write_header();
                std::atomic<size_t> n_chunks{0};
                std::thread writer([&]() {
                    const std::vector<uint8_t> * parts[2] = { &a, &b };
                    for (const auto * v : parts) {
                        for (size_t o = 0; o < v->size(); o += io_chunk_bytes_) {
                            f.write(reinterpret_cast<const char *>(v->data() + o), (std::streamsize) std::min(io_chunk_bytes_, v->size() - o));
                            n_chunks.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
                h = checksum_chunked(a.data(), a.size(), b.data(), b.size(), (uint64_t) tag, io_chunk_bytes_);
                writer.join();
                f.flush();
                if (f.good()) {
                    f.seekp(0);
                    write_header();
                    f.seekp(0, std::ios::end);
                }
                stats_.n_pipelined_put++;
                stats_.n_io_chunks += n_chunks.load();
            } else {
                h = checksum(b.data(), b.size(), checksum(a.data(), a.size(), (uint64_t) tag));
                write_header();
                f.write(reinterpret_cast<const char *>(a.data()), (std::streamsize) a.size());
                if (!b.empty()) {
                    f.write(reinterpret_cast<const char *>(b.data()), (std::streamsize) b.size());
                }
            }
            f.flush();
            ok = f.good();
        }
    } // closed here: Windows cannot rename or delete a file that is still open

    std::error_code ec;
    if (ok) {
        // verify what reached the disk before the caller frees the only in-memory copy
        const auto on_disk = fs::file_size(tmp, ec);
        ok = !ec && (size_t) on_disk == file_bytes;
    }
    if (ok) {
        fs::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        std::error_code ec2;
        fs::remove(tmp, ec2);
        fs::remove(path, ec2);
        stats_.n_put_failed++;
        return 0;
    }

    stats_.write_seconds += steady_seconds() - t_io0;
    blobs_[id].file_bytes = file_bytes;
    blobs_[id].ckpt       = is_ckpt;
    stats_.used_bytes    += file_bytes;
    stats_.bytes_written += file_bytes;
    stats_.n_put++;
    if (is_ckpt) {
        stats_.bytes_written_ckpt += file_bytes;
        stats_.n_put_ckpt++;
    }
    stats_.n_blobs = blobs_.size();
    return id;
}

bool common_disk_tier::get(uint64_t id, int64_t tag, std::vector<uint8_t> & a, std::vector<uint8_t> & b) {
    auto it = blobs_.find(id);
    if (it == blobs_.end()) {
        stats_.n_get_failed++;
        return false;
    }

    auto fail = [&](bool checksum_failure) {
        if (checksum_failure) {
            stats_.n_checksum_failed++;
        } else {
            stats_.n_get_failed++;
        }
        a.clear();
        b.clear();
        drop_blob(id, false);
        return false;
    };

    const double t_io0 = steady_seconds();
    const std::string path = path_of(id);
    std::error_code ec;
    const auto on_disk = fs::file_size(path, ec);
    if (ec || (size_t) on_disk != it->second.file_bytes || (size_t) on_disk < DISK_TIER_HEADER) {
        return fail(false);
    }

    std::vector<uint8_t> ra, rb;
    uint64_t h_file = 0;
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return fail(false);
        }
        uint32_t magic = 0, version = 0;
        int64_t  ftag  = 0;
        uint64_t la = 0, lb = 0;
        f.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
        f.read(reinterpret_cast<char *>(&version), sizeof(version));
        f.read(reinterpret_cast<char *>(&ftag),    sizeof(ftag));
        f.read(reinterpret_cast<char *>(&la),      sizeof(la));
        f.read(reinterpret_cast<char *>(&lb),      sizeof(lb));
        f.read(reinterpret_cast<char *>(&h_file),  sizeof(h_file));
        // lengths come from the file: believe them only after they add up to the size the file really has
        if (!f || magic != DISK_TIER_MAGIC || version != DISK_TIER_VERSION || ftag != tag ||
                la == 0 || la > on_disk || lb > on_disk || DISK_TIER_HEADER + la + lb != (uint64_t) on_disk) {
            f.close();
            return fail(false);
        }
        ra.resize((size_t) la);
        rb.resize((size_t) lb);
        if (io_chunk_bytes_ > 0 && la + lb > io_chunk_bytes_) {
            // Read/checksum overlap: a reader thread fills ra then rb chunk by chunk and publishes how many payload bytes are
            // in place; this thread checksums every chunk as soon as it is published, so the checksum of chunk i runs while chunk
            // i+1 is being read. The destination vectors are the staging buffers (no bounce copy).
            std::mutex mu;
            std::condition_variable cv;
            size_t avail = 0;
            bool   read_failed = false, finished = false;
            std::thread reader([&]() {
                bool good = true;
                for (int part = 0; part < 2 && good; part++) {
                    std::vector<uint8_t> & v = part == 0 ? ra : rb;
                    for (size_t o = 0; o < v.size() && good; o += io_chunk_bytes_) {
                        const size_t n = std::min(io_chunk_bytes_, v.size() - o);
                        f.read(reinterpret_cast<char *>(v.data() + o), (std::streamsize) n);
                        good = (bool) f;
                        std::lock_guard<std::mutex> lk(mu);
                        if (good) { avail += n; } else { read_failed = true; }
                        cv.notify_one();
                    }
                }
                std::lock_guard<std::mutex> lk(mu);
                finished = true;
                cv.notify_one();
            });
            disk_tier_csum_stream sa((uint64_t) tag);
            uint64_t ha = 0, hb = 0;
            size_t consumed = 0, n_chunks = 0;
            bool failed = false;
            for (int part = 0; part < 2 && !failed; part++) {
                const std::vector<uint8_t> & v = part == 0 ? ra : rb;
                disk_tier_csum_stream * st = &sa;
                disk_tier_csum_stream sb(0);
                if (part == 1) { sb = disk_tier_csum_stream(ha); st = &sb; }
                for (size_t o = 0; o < v.size(); o += io_chunk_bytes_) {
                    const size_t n = std::min(io_chunk_bytes_, v.size() - o);
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&]() { return avail >= consumed + n || read_failed; });
                        if (avail < consumed + n) { failed = true; break; }
                    }
                    st->update(v.data() + o, n);
                    consumed += n;
                    n_chunks++;
                }
                if (failed) { break; }
                if (part == 0) { ha = sa.finish(); } else { hb = sb.finish(); }
            }
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return finished; });
            }
            reader.join();
            if (failed) {
                f.close();
                return fail(false);
            }
            stats_.n_pipelined_get++;
            stats_.n_io_chunks += n_chunks;
            if (hb != h_file) {
                f.close();
                return fail(true);
            }
            stats_.read_seconds += steady_seconds() - t_io0;
            stats_.n_get++;
            stats_.bytes_read += on_disk;
            if (it->second.ckpt) {
                stats_.n_get_ckpt++;
                stats_.bytes_read_ckpt += on_disk;
            }
            a = std::move(ra);
            b = std::move(rb);
            return true;
        }
        f.read(reinterpret_cast<char *>(ra.data()), (std::streamsize) la);
        if (lb > 0) {
            f.read(reinterpret_cast<char *>(rb.data()), (std::streamsize) lb);
        }
        if (!f) {
            f.close();
            return fail(false);
        }
    }

    if (checksum(rb.data(), rb.size(), checksum(ra.data(), ra.size(), (uint64_t) tag)) != h_file) {
        return fail(true);
    }

    stats_.read_seconds += steady_seconds() - t_io0;
    stats_.n_get++;
    stats_.bytes_read += on_disk;
    if (it->second.ckpt) {
        stats_.n_get_ckpt++;
        stats_.bytes_read_ckpt += on_disk;
    }
    a = std::move(ra);
    b = std::move(rb);
    return true; // the blob stays: the entry that owns it is still in the cache and may be hit again
}

// ---- async restore (row S69-ENGREV-RTPLLM-ASYNC-DISK-RESTORE-2026-10-27-01) -------------------------------------------------
bool common_disk_tier::describe(uint64_t id, std::string & path, size_t & file_bytes, bool & ckpt) const {
    auto it = blobs_.find(id);
    if (it == blobs_.end()) {
        return false;
    }
    path       = path_of(id);
    file_bytes = it->second.file_bytes;
    ckpt       = it->second.ckpt;
    return true;
}

common_disk_tier::read_status common_disk_tier::read_blob_file(const std::string & path, size_t file_bytes, int64_t tag,
        std::vector<uint8_t> & a, std::vector<uint8_t> & b) {
    a.clear();
    b.clear();
    std::error_code ec;
    const auto on_disk = fs::file_size(path, ec);
    if (ec || (size_t) on_disk != file_bytes || (size_t) on_disk < DISK_TIER_HEADER) {
        return read_status::IO;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return read_status::IO;
    }
    uint32_t magic = 0, version = 0;
    int64_t  ftag  = 0;
    uint64_t la = 0, lb = 0, h_file = 0;
    f.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    f.read(reinterpret_cast<char *>(&ftag),    sizeof(ftag));
    f.read(reinterpret_cast<char *>(&la),      sizeof(la));
    f.read(reinterpret_cast<char *>(&lb),      sizeof(lb));
    f.read(reinterpret_cast<char *>(&h_file),  sizeof(h_file));
    if (!f || magic != DISK_TIER_MAGIC || version != DISK_TIER_VERSION || ftag != tag ||
            la == 0 || la > on_disk || lb > on_disk || DISK_TIER_HEADER + la + lb != (uint64_t) on_disk) {
        return read_status::IO;
    }
    std::vector<uint8_t> ra((size_t) la), rb((size_t) lb);
    f.read(reinterpret_cast<char *>(ra.data()), (std::streamsize) la);
    if (lb > 0) {
        f.read(reinterpret_cast<char *>(rb.data()), (std::streamsize) lb);
    }
    if (!f) {
        return read_status::IO;
    }
    if (checksum(rb.data(), rb.size(), checksum(ra.data(), ra.size(), (uint64_t) tag)) != h_file) {
        return read_status::CHECKSUM;
    }
    a = std::move(ra);
    b = std::move(rb);
    return read_status::OK;
}

void common_disk_tier::finish_async_get(uint64_t id, read_status st, double seconds) {
    auto it = blobs_.find(id);
    if (it == blobs_.end()) {
        return; // dropped while the read was in flight: nothing to book
    }
    if (st == read_status::OK) {
        const size_t fb = it->second.file_bytes;
        stats_.read_seconds += seconds;
        stats_.n_get++;
        stats_.bytes_read += fb;
        if (it->second.ckpt) {
            stats_.n_get_ckpt++;
            stats_.bytes_read_ckpt += fb;
        }
        return;
    }
    if (st == read_status::CHECKSUM) {
        stats_.n_checksum_failed++;
    } else {
        stats_.n_get_failed++;
    }
    drop_blob(id, false);
}

struct common_disk_loader::task {
    uint64_t id = 0;
    double   deadline = 0.0; // absolute, in clock_ seconds
    std::atomic<bool> done{false};
    bool     booked = false;
    double   seconds = 0.0;
    common_disk_tier::read_status st = common_disk_tier::read_status::IO;
    std::vector<uint8_t> a, b;
    std::thread th;
    ~task() { if (th.joinable()) { th.join(); } }
};

common_disk_loader::common_disk_loader(std::shared_ptr<common_disk_tier> tier, clock_fn clock)
    : tier_(std::move(tier)), clock_(clock ? std::move(clock) : clock_fn(steady_seconds)) {}

common_disk_loader::~common_disk_loader() {
    tasks_.clear();     // task dtor joins
    abandoned_.clear();
}

bool common_disk_loader::start(uint64_t id, int64_t tag, double deadline_seconds) {
    auto it = tasks_.find(id);
    if (it != tasks_.end()) {
        n_joined_++;
        return true; // join the read in flight (the earlier deadline stays: a joiner never extends a task)
    }
    std::string path;
    size_t bytes = 0;
    bool ckpt = false;
    if (!tier_ || !tier_->describe(id, path, bytes, ckpt)) {
        return false;
    }
    auto t = std::make_shared<task>();
    t->id = id;
    t->deadline = clock_() + deadline_seconds;
    task * raw = t.get();
    t->th = std::thread([raw, path, bytes, tag]() {
        const double t0 = steady_seconds();
        raw->st = common_disk_tier::read_blob_file(path, bytes, tag, raw->a, raw->b);
        raw->seconds = steady_seconds() - t0;
        raw->done.store(true, std::memory_order_release);
    });
    tasks_[id] = std::move(t);
    n_started_++;
    return true;
}

common_disk_loader::state common_disk_loader::poll(uint64_t id) {
    auto it = tasks_.find(id);
    if (it == tasks_.end()) {
        return state::NONE;
    }
    task & t = *it->second;
    if (t.done.load(std::memory_order_acquire)) {
        if (!t.booked) {
            if (t.th.joinable()) { t.th.join(); }
            tier_->finish_async_get(id, t.st, t.seconds);
            t.booked = true;
        }
        return t.st == common_disk_tier::read_status::OK ? state::READY : state::FAILED;
    }
    if (clock_() > t.deadline) {
        n_timeout_++;
        // the request gives up and recomputes; the read still completes and is booked by reap()
        abandoned_.push_back(it->second);
        tasks_.erase(it);
        return state::TIMEOUT;
    }
    return state::LOADING;
}

bool common_disk_loader::take(uint64_t id, std::vector<uint8_t> & a, std::vector<uint8_t> & b) {
    if (poll(id) != state::READY) {
        return false;
    }
    auto it = tasks_.find(id);
    a = std::move(it->second->a);
    b = std::move(it->second->b);
    tasks_.erase(it);
    return true;
}

void common_disk_loader::forget(uint64_t id) {
    auto it = tasks_.find(id);
    if (it == tasks_.end()) {
        return;
    }
    abandoned_.push_back(it->second);
    tasks_.erase(it);
}

void common_disk_loader::reap() {
    for (auto it = abandoned_.begin(); it != abandoned_.end();) {
        task & t = **it;
        if (t.done.load(std::memory_order_acquire)) {
            if (t.th.joinable()) { t.th.join(); }
            if (!t.booked) {
                tier_->finish_async_get(t.id, t.st, t.seconds);
                t.booked = true;
            }
            it = abandoned_.erase(it);
        } else {
            ++it;
        }
    }
}
