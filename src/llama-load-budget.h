// llama-load-budget.h -- host-RAM budget for parallel tensor loads.
//
// llama_model_loader::load_all_data loads tensors on 8 worker threads. On the --split-mode graph path every worker
// reads a WHOLE tensor into host memory before uploading it (a per-thread buffer on Windows, file pages through a
// mapping elsewhere), so 8 large tensors in flight can need more host RAM than the machine has free. This header
// bounds the bytes in flight by what the OS says can be allocated right now.
//
// Behaviour from ikawrakow/ik_llama.cpp PR #2125 (MIT; src/llama-model-loader.cpp allocatable_host_memory() and
// struct llama_memory_budget, diff read in full): Linux /proc/meminfo MemAvailable, tightened by every cgroup v2
// memory.max on the path from the leaf cgroup to the root (charged = memory.current minus reclaimable page cache,
// memory.stat `file` minus `shmem`), budget = 2/3 of that, and a byte semaphore that always admits a tensor when
// nothing else is in flight. Written for this loader; no code copied. Differences from the PR, on purpose:
//   - Windows is covered (GlobalMemoryStatusEx ullAvailPhys); the PR is __linux__ && GGML_USE_CUDA only.
//   - a FAILED probe means UNLIMITED (the loader's behaviour before this header); the PR's budget 0 serialises.
//   - a cgroup level that cannot be read is skipped instead of discarding the whole probe, a level charged above
//     its limit gives 0 available (the PR returns "unknown"), and missing file/shmem lines count as 0 reclaimable.
// Pure functions + one OS probe, so tests/test-load-budget.cpp drives every branch without CUDA or a model.
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#include "ggml-host-memory.h"   // ggml/src: llama links it PRIVATE (src/CMakeLists.txt:542); the test adds it

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace llama_load_budget {

constexpr uint64_t UNLIMITED = UINT64_MAX;

// The /proc/meminfo + cgroup v2 parsers and the leaf-to-root walk live in ggml/src/ggml-host-memory.h (moved there
// unchanged on 2026-10-20 so the CPU device's get_memory and this budget share one implementation).
using ggml_host_mem::parse_u64;
using ggml_host_mem::line_value;
using ggml_host_mem::meminfo_available;
using ggml_host_mem::cgroup_v2_path;
using ggml_host_mem::cgroup_available;
using ggml_host_mem::read_fn;
using ggml_host_mem::linux_allocatable_bytes;

// Bytes the OS can allocate to this process now; nullopt when the platform or the probe gives no answer.
inline std::optional<uint64_t> host_allocatable_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX st = {};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return (uint64_t) st.ullAvailPhys;
    }
    return std::nullopt;
#elif defined(__linux__)
    return linux_allocatable_bytes(ggml_host_mem::read_file);
#else
    return std::nullopt;
#endif
}

// 2/3 of what can be allocated (headroom for everything else the process and the OS do meanwhile).
inline uint64_t budget_bytes(std::optional<uint64_t> allocatable) {
    if (!allocatable) {
        return UNLIMITED;
    }
    return *allocatable / 3 * 2;
}

// May a load of n bytes start while `used` bytes are in flight? Nothing in flight always admits: a tensor larger than
// the whole budget still loads (alone), so a short budget caps the parallelism at 1 and never deadlocks.
inline bool admit(uint64_t used, uint64_t n, uint64_t budget) {
    if (budget == UNLIMITED || used == 0) {
        return true;
    }
    return n <= budget && used <= budget - n;
}

// Loads of n bytes that can be in flight together under `budget`, in [1, n_workers].
inline int max_parallel(uint64_t budget, uint64_t n, int n_workers) {
    if (n_workers < 1) {
        return 1;
    }
    if (budget == UNLIMITED || n == 0) {
        return n_workers;
    }
    const uint64_t k = budget / n;
    return (int) std::max<uint64_t>(1, std::min<uint64_t>(k, (uint64_t) n_workers));
}

// Byte semaphore over admit().
class semaphore {
public:
    explicit semaphore(uint64_t budget = UNLIMITED) : budget_(budget) {}

    void set_budget(uint64_t b) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            budget_ = b;
        }
        cv_.notify_all();
    }

    void acquire(uint64_t n) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return admit(used_, n, budget_); });
        used_ += n;
        peak_ = std::max(peak_, ++holders_);
    }

    void release(uint64_t n) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            used_ -= n;
            --holders_;
        }
        cv_.notify_all();
    }

    uint64_t budget() const { std::lock_guard<std::mutex> lk(mu_); return budget_; }
    uint64_t used() const { std::lock_guard<std::mutex> lk(mu_); return used_; }
    int peak_holders() const { std::lock_guard<std::mutex> lk(mu_); return peak_; }

    // RAII: holds n bytes of the budget for the guard's lifetime.
    struct guard {
        semaphore & s;
        uint64_t    n;
        guard(semaphore & s_, uint64_t n_) : s(s_), n(n_) { s.acquire(n); }
        ~guard() { s.release(n); }
        guard(const guard &) = delete;
        guard & operator=(const guard &) = delete;
    };

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    uint64_t                budget_;
    uint64_t                used_    = 0;
    int                     holders_ = 0;
    int                     peak_    = 0;
};

} // namespace llama_load_budget
