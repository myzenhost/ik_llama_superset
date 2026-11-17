#pragma once

// Generic kernel-variant autotuner: candidate budgets, interleaved successive halving, noise-robust
// scoring and a salted on-disk result cache.  Host-only C++ with NO CUDA dependency, so the whole
// selection logic is unit-tested on the CPU (tests/test-cuda-autotune.cpp) and the CUDA side only
// supplies a timing callback.
//
// SOURCE-READ: exllamav3 exllamav3/exllamav3_ext/libtorch/coop_autotune.cu (MIT), read in full (675
// lines).  Rewritten here for ggml; no code copied.  NUANCES THE ROW DESCRIPTION DID NOT SAY:
//   N1. The search budget SHRINKS with problem size (a big GEMM costs more per sample, so fewer
//       candidates / rounds / repeats); a tuning pass must never cost more than a few launches' worth
//       of the work it is tuning.
//   N2. Candidates are not timed in a fixed order: each round visits them in a rotated order with a
//       stride coprime to the candidate count, so slow clock ramp / thermal drift / L2 state does not
//       systematically favour whichever candidate happens to go first.
//   N3. One UNTIMED warm-up launch per candidate (first-launch cost: lazy module load, attribute
//       set, shared-memory carve-out change) before any sample is taken.
//   N4. An L2-thrash write of ~2x the L2 size runs before every sample so the operands are cold,
//       which is what a real model layer sees (weights are streamed once per token).
//   N5. A sample is a quartile-trimmed mean, not a min or a plain mean: the min rewards a lucky
//       outlier, the mean is dragged by a stall.
//   N6. Three-stage successive halving (keep 6, then 3, then 1) with growing round counts, so most
//       of the timing budget goes to the last few finalists, not to hopeless candidates.
//   N7. The cache key is SALTED with a version number so a kernel change invalidates old results,
//       the file starts with magic + format + record size, and a reader tolerates a SECOND header
//       embedded mid-file (two processes that both found the file empty both write a header).
//   N8. A cached record is only trusted if its candidate is still in the current candidate set.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace ggml_cuda_at {

// ---- budgets (N1) ---------------------------------------------------------------------------

struct budget {
    int repeats;     // launches averaged inside one timed sample
    int max_rounds;  // upper bound on interleaved rounds per stage
    int max_cands;   // candidates considered at all
};

// `work` is any monotone measure of the problem (flops or output elements); tiers are in the same unit.
inline budget budget_for(double work) {
    if (work < 1e6) { return { 8, 32, 7 }; }
    if (work < 1e7) { return { 5, 12, 5 }; }
    if (work < 1e8) { return { 4,  8, 4 }; }
    if (work < 2e8) { return { 3,  5, 3 }; }
    return { 2, 2, 2 };
}

// ---- interleaved visit order (N2) -----------------------------------------------------------

inline int gcd_int(int a, int b) {
    while (b) { const int t = a % b; a = b; b = t; }
    return a;
}

// The order in which candidate slots 0..n-1 are visited in `round`; always a permutation of 0..n-1.
inline std::vector<int> visit_order(int n, int round) {
    std::vector<int> out;
    if (n <= 0) { return out; }
    int step = 2 * round + 1;
    while (gcd_int(step, n) != 1) { step += 2; }   // n == 1 is always coprime
    // splitmix64 of the round number: a plain LCG start (a*round + c) % n is constant for every n sharing a factor with
    // both a and c (here 3 and 5), which would pin the first-visited slot -- the exact bias this ordering exists to remove.
    uint64_t z = uint64_t(round) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const uint64_t start = z % uint64_t(n);
    out.reserve(n);
    uint64_t cur = start;
    for (int i = 0; i < n; i++) {
        out.push_back(int(cur));
        cur = (cur + uint64_t(step)) % uint64_t(n);
    }
    return out;
}

// ---- scoring (N5) ---------------------------------------------------------------------------

// Mean of the samples after dropping the lowest and highest quartile (at least one value is kept).
inline double trimmed_mean(std::vector<double> v) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    const size_t n    = v.size();
    const size_t drop = n / 4;
    double s = 0.0;
    size_t c = 0;
    for (size_t i = drop; i < n - drop; i++) { s += v[i]; c++; }
    return s / double(c);
}

// ---- candidate expansion for "how many SMs / what concurrency" style variants -----------------

struct sm_candidate {
    int num_sms;
    int concurrency;
    bool operator==(const sm_candidate & o) const { return num_sms == o.num_sms && concurrency == o.concurrency; }
};

// Persistent-kernel grid sizes to try for a problem of `total_tiles` work tiles on a device with
// `max_sms` SMs: from the smallest grid that still covers the tiles up to 85% of the device in steps
// of 2, plus the full device; concurrency = how many independent launches fit side by side.
inline std::vector<sm_candidate> expand_sm_candidates(int total_tiles, int max_sms, int max_concurrency) {
    std::vector<sm_candidate> out;
    if (max_sms <= 0 || total_tiles <= 0) { return out; }
    const int min_sms = std::max(1, std::min(max_sms, total_tiles));
    const int top     = std::max(min_sms, (max_sms * 85) / 100);
    auto add = [&](int s) {
        s = std::min(s, max_sms);
        sm_candidate c { s, std::max(1, std::min(max_sms / s, std::max(1, max_concurrency))) };
        for (const auto & e : out) { if (e == c) { return; } }
        out.push_back(c);
    };
    for (int s = min_sms; s <= top; s += 2) { add(s); }
    add(max_sms);
    return out;
}

// ---- roofline early stop (row S69-ENGREV-BURN-ROOFLINE-STOP-2026-10-27-01) --------------------
//
// SOURCE-READ: Burn tracel-ai/burn @ 3bf6e039 (Apache-2.0/MIT) crates/burn-backend/src/cubecl/autotune.rs:11-44 and
// crates/burn-cubecl/src/kernel/matmul/tune/bounds.rs:13-38, read per the row; the design is ported, no code copied.  A per-shape
// analytic lower bound on the kernel time is registered with the tuning round, and the round ENDS at the first stage whose best
// candidate runs within a level-dependent fraction of that bound.  The exact Burn fractions live in cubecl-std (not read), so the
// fractions below are our own.  Level `full` registers nothing: the search is exactly what it was before.

struct roofline_cost {
    double flops;   // arithmetic work of one launch
    double bytes;   // bytes that must cross the memory interface (operands + result)
};

struct device_peaks {
    double flops_per_s;   // peak arithmetic rate for the kernel's datatype
    double bytes_per_s;   // peak memory bandwidth
    double launch_ms;     // fixed per-launch overhead floor
};

enum class level { full = 0, balanced = 1, fast = 2 };

// Fraction of the bound a candidate must reach to end the search; 0 means "never stop early" (`full`).
inline double accept_fraction(level l) {
    switch (l) {
        case level::balanced: return 0.85;
        case level::fast:     return 0.70;
        default:              return 0.0;
    }
}

// "full" | "balanced" | "fast" (anything else, or null, is `full`: an unknown spelling must never switch the search off).
inline level parse_level(const char * s) {
    if (!s) { return level::full; }
    if (std::strcmp(s, "balanced") == 0) { return level::balanced; }
    if (std::strcmp(s, "fast") == 0)     { return level::fast; }
    return level::full;
}

// t_min = max(flops / peak_flops, bytes / peak_bw) + launch overhead, in milliseconds; 0 when the inputs are unusable.
inline double roofline_min_ms(const roofline_cost & c, const device_peaks & p) {
    if (!(p.flops_per_s > 0.0) || !(p.bytes_per_s > 0.0) || !(c.flops >= 0.0) || !(c.bytes >= 0.0)) { return 0.0; }
    const double t = std::max(c.flops / p.flops_per_s, c.bytes / p.bytes_per_s) * 1e3 + std::max(0.0, p.launch_ms);
    return t;
}

// DDR-style memory: 2 transfers per clock.  Clock in kHz (cudaDevAttrMemoryClockRate), bus width in bits.
inline double peak_bytes_per_s(int mem_clock_khz, int bus_bits) {
    if (mem_clock_khz <= 0 || bus_bits <= 0) { return 0.0; }
    return 2.0 * double(mem_clock_khz) * 1e3 * double(bus_bits) / 8.0;
}

inline double peak_flops_per_s(int num_sms, int clock_khz, double flops_per_clk_per_sm) {
    if (num_sms <= 0 || clock_khz <= 0 || !(flops_per_clk_per_sm > 0.0)) { return 0.0; }
    return double(num_sms) * double(clock_khz) * 1e3 * flops_per_clk_per_sm;
}

struct bound {
    double t_min_ms     = 0.0;   // analytic lower bound for one launch
    double accept_frac  = 0.0;   // stop when t_min_ms / best_ms >= accept_frac
    bool   active() const { return t_min_ms > 0.0 && accept_frac > 0.0; }
};

inline bound make_bound(const roofline_cost & c, const device_peaks & p, level l) {
    bound b;
    b.accept_frac = accept_fraction(l);
    b.t_min_ms    = b.accept_frac > 0.0 ? roofline_min_ms(c, p) : 0.0;
    return b;
}

// A measured time far BELOW the bound means the peaks are wrong (or one lucky sample), not that the kernel beat physics:
// such a reading is not trusted as a reason to stop.
constexpr double ROOFLINE_IMPLAUSIBLE_FRAC = 2.0;

// ---- successive halving (N3, N4, N6) ----------------------------------------------------------

struct hooks {
    // Time `repeats` back-to-back launches of candidate slot `idx`, return milliseconds PER LAUNCH.
    std::function<double(int idx, int repeats)> time_ms;
    std::function<void(int idx)>                warmup;  // one untimed launch (optional)
    std::function<void()>                       flush;   // thrash L2 (optional)
};

struct result {
    int    best;        // slot index of the winner, -1 if none
    double best_ms;     // its trimmed mean
    int    samples;     // number of timed samples taken in total
    int    cand_count;  // candidates actually considered
    int    stages      = 0;      // halving stages that ran
    bool   early_stop  = false;  // ended at the roofline bound before the last stage
    double frac        = 0.0;    // achieved fraction of the bound for the winner (0 when no bound was given)
};

inline result select_best(int n_cands, const budget & b, const hooks & h, const bound * rb = nullptr) {
    result r { -1, 0.0, 0, 0 };
    if (n_cands <= 0 || !h.time_ms) { return r; }
    const int n0 = std::min(n_cands, std::max(1, b.max_cands));
    r.cand_count = n0;
    if (n0 == 1) { r.best = 0; return r; }

    std::vector<int> active(n0);
    for (int i = 0; i < n0; i++) { active[i] = i; }
    std::vector<std::vector<double>> samples(n0);

    if (h.warmup) { for (int i : active) { h.warmup(i); } }

    static const int keep[3]   = { 6, 3, 1 };
    static const int rounds[3] = { 2, 8, 14 };
    int round_no = 0;
    for (int stage = 0; stage < 3 && active.size() > 1; stage++) {
        const int nr = std::min(rounds[stage], std::max(1, b.max_rounds));
        for (int rd = 0; rd < nr; rd++, round_no++) {
            const std::vector<int> order = visit_order(int(active.size()), round_no);
            for (int slot : order) {
                const int idx = active[slot];
                if (h.flush) { h.flush(); }
                samples[idx].push_back(h.time_ms(idx, std::max(1, b.repeats)));
                r.samples++;
            }
        }
        std::vector<std::pair<double, int>> rank;
        for (int idx : active) { rank.push_back({ trimmed_mean(samples[idx]), idx }); }
        std::stable_sort(rank.begin(), rank.end(), [](const std::pair<double, int> & a, const std::pair<double, int> & c) { return a.first < c.first; });
        const size_t k = std::min<size_t>(size_t(keep[stage]), rank.size());
        active.clear();
        for (size_t i = 0; i < k; i++) { active.push_back(rank[i].second); }
        r.stages = stage + 1;
        // roofline bound: the stage leader already runs within the accepted fraction of the analytic minimum -> done
        if (rb && rb->active() && stage < 2 && !rank.empty() && rank.front().first > 0.0) {
            const double frac = rb->t_min_ms / rank.front().first;
            if (frac >= rb->accept_frac && frac <= ROOFLINE_IMPLAUSIBLE_FRAC) {
                active.assign(1, rank.front().second);
                r.early_stop = true;
                break;
            }
        }
    }
    r.best    = active.front();
    r.best_ms = trimmed_mean(samples[r.best]);
    if (rb && rb->active() && r.best_ms > 0.0) { r.frac = rb->t_min_ms / r.best_ms; }
    return r;
}

// ---- salted keys (N7) -------------------------------------------------------------------------

inline uint64_t fnv1a64(const void * data, size_t n, uint64_t h = 14695981039346656037ull) {
    const unsigned char * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct key_builder {
    uint64_t h;
    explicit key_builder(uint32_t version_salt) : h(14695981039346656037ull) {
        const uint64_t s = 0x9E3779B97F4A7C15ull ^ uint64_t(version_salt);
        h = fnv1a64(&s, sizeof(s), h);
    }
    key_builder & add(int64_t v) { h = fnv1a64(&v, sizeof(v), h); return *this; }
    key_builder & add(const std::string & s) { add(int64_t(s.size())); h = fnv1a64(s.data(), s.size(), h); return *this; }
    uint64_t done() const { return h; }
};

// ---- disk cache (N7, N8) ----------------------------------------------------------------------

#pragma pack(push, 1)
struct file_header {            // exactly one record slot (16 bytes)
    char     magic[8];          // "GGMLATUN"
    uint32_t format;
    uint32_t record_size;
};
struct record {
    uint64_t key;
    int32_t  cand;              // candidate VALUE (e.g. mmq_x), not a slot index: stable across candidate-list changes
    uint32_t check;             // fnv of key+cand, catches torn writes
};
#pragma pack(pop)
static_assert(sizeof(file_header) == 16 && sizeof(record) == 16, "header must occupy one record slot");

constexpr uint32_t FILE_FORMAT = 1;

inline uint32_t record_check(uint64_t key, int32_t cand) {
    uint64_t h = fnv1a64(&key, sizeof(key));
    h = fnv1a64(&cand, sizeof(cand), h);
    return uint32_t(h ^ (h >> 32));
}

inline file_header make_header() {
    file_header h {};
    std::memcpy(h.magic, "GGMLATUN", 8);
    h.format      = FILE_FORMAT;
    h.record_size = uint32_t(sizeof(record));
    return h;
}

inline bool is_header(const unsigned char * p) {
    file_header h;
    std::memcpy(&h, p, sizeof(h));
    return std::memcmp(h.magic, "GGMLATUN", 8) == 0;
}

// Parse a whole file image.  Returns false when the leading header is absent or foreign (the caller
// then treats the cache as empty and rewrites it).  Later records win over earlier ones for a key.
inline bool parse(const std::vector<unsigned char> & img, std::map<uint64_t, int32_t> & out, int * skipped_headers = nullptr, int * bad_records = nullptr) {
    if (skipped_headers) { *skipped_headers = 0; }
    if (bad_records)     { *bad_records = 0; }
    if (img.size() < sizeof(file_header) || !is_header(img.data())) { return false; }
    file_header h;
    std::memcpy(&h, img.data(), sizeof(h));
    if (h.format != FILE_FORMAT || h.record_size != sizeof(record)) { return false; }
    for (size_t off = sizeof(file_header); off + sizeof(record) <= img.size(); off += sizeof(record)) {
        if (is_header(img.data() + off)) {   // second header from a concurrent first writer
            if (skipped_headers) { (*skipped_headers)++; }
            continue;
        }
        record rec;
        std::memcpy(&rec, img.data() + off, sizeof(rec));
        if (rec.check != record_check(rec.key, rec.cand)) {
            if (bad_records) { (*bad_records)++; }
            continue;
        }
        out[rec.key] = rec.cand;
    }
    return true;
}

inline std::vector<unsigned char> read_all(const std::string & path) {
    std::vector<unsigned char> img;
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { return img; }
    unsigned char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) { img.insert(img.end(), buf, buf + n); }
    std::fclose(f);
    return img;
}

// Atomically append `n` bytes: one write() on an O_APPEND descriptor (POSIX) or one WriteFile() on a handle opened with
// FILE_APPEND_DATA only (Windows).  A stdio "ab" append is NOT enough on Windows: the CRT seeks to the end and then writes,
// so two writers can land on the same offset and one record is lost (measured: 337 of 400 records survived 4 threads).
// `was_empty` reports whether the file had no bytes when the write happened (best effort; the loser of a race writes a
// second header, which parse() tolerates).
inline bool append_atomic(const std::string & path, const void * data, size_t n, bool * was_empty = nullptr) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { return false; }
    LARGE_INTEGER sz;
    sz.QuadPart = 1;
    if (!GetFileSizeEx(h, &sz)) { sz.QuadPart = 1; }
    if (was_empty) { *was_empty = sz.QuadPart == 0; }
    DWORD wrote = 0;
    const bool ok = WriteFile(h, data, DWORD(n), &wrote, nullptr) && wrote == n;
    CloseHandle(h);
    return ok;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) { return false; }
    if (was_empty) { struct stat st; *was_empty = (fstat(fd, &st) == 0) && st.st_size == 0; }
    const ssize_t w = ::write(fd, data, n);
    ::close(fd);
    return w == ssize_t(n);
#endif
}

// Append one record; the first writer of an empty file puts the header in the SAME write, so a reader never sees a
// header-less file that has records.  Concurrent first writers may both add a header: parse() skips the second one.
inline bool append_record(const std::string & path, uint64_t key, int32_t cand) {
    record rec { key, cand, record_check(key, cand) };
    bool empty = false;
    unsigned char buf[sizeof(file_header) + sizeof(record)];
    // the emptiness probe and the write are two steps; to keep the write single, build both layouts up front
    const file_header hd = make_header();
    std::memcpy(buf, &hd, sizeof(hd));
    std::memcpy(buf + sizeof(hd), &rec, sizeof(rec));
    // probe size first (cheap), then write the right slice in ONE call
    {
        std::FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) { empty = true; }
        else { std::fseek(f, 0, SEEK_END); empty = std::ftell(f) <= 0; std::fclose(f); }
    }
    if (empty) { return append_atomic(path, buf, sizeof(buf)); }
    return append_atomic(path, buf + sizeof(hd), sizeof(rec));
}

// ---- the cache object --------------------------------------------------------------------------

class cache {
public:
    explicit cache(std::string path = "") : path_(std::move(path)) { load(); }

    void load() {
        map_.clear();
        if (path_.empty()) { return; }
        parse(read_all(path_), map_, &headers_skipped, &records_dropped);
    }

    // Returns the cached candidate VALUE if present AND still one of `valid` (N8).
    bool lookup(uint64_t key, const std::vector<int> & valid, int & cand) const {
        auto it = map_.find(key);
        if (it == map_.end()) { return false; }
        if (std::find(valid.begin(), valid.end(), it->second) == valid.end()) { return false; }
        cand = it->second;
        return true;
    }

    void store(uint64_t key, int cand) {
        map_[key] = cand;
        if (!path_.empty()) { append_record(path_, key, cand); }
    }

    size_t size() const { return map_.size(); }

    int headers_skipped  = 0;
    int records_dropped = 0;

private:
    std::string                 path_;
    std::map<uint64_t, int32_t> map_;
};

}  // namespace ggml_cuda_at
