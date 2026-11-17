#include "ggml-moe-prefetch.h"

#if defined(__linux__)

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// chunk granularity; big enough to amortize syscall cost,
// small enough to spread one expert (~MBs) across several workers.
static constexpr size_t GGML_MOE_PREFETCH_CHUNK = 2u*1024u*1024u;
// lookahead jobs are dropped beyond this queue depth so a stalled consumer
// cannot accumulate unbounded work.
static constexpr size_t GGML_MOE_PREFETCH_MAX_QUEUE = 65536;

// experts are faulted in with MADV_POPULATE_READ, which brings pages into the
// page cache AND this mm's page tables, so consumers take no minor faults.
// Unsupported kernels leave the engine off (the callers fall back to
// madvise(MADV_WILLNEED) hints)
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

namespace {

struct mapping_entry {
    uintptr_t base;
    size_t    size;
};

struct ticket {
    std::atomic<int> pending{0};
    uint64_t         epoch = 0;
    bool             track_reads = false;
    // chunks the workers actually read from storage this epoch (i.e. pages
    // that were cold before the sweep); guarded by prefetch_pool::mtx.
    // Used by ggml_moe_prefetch_cold() to deactivate streaming traffic.
    std::vector<std::pair<uintptr_t, uint32_t>> read_chunks;
};

struct job {
    uintptr_t addr;
    uint32_t  len;
    std::shared_ptr<ticket> tk;
};

struct prefetch_pool {
    std::mutex               mtx;
    std::condition_variable  cv_work;   // workers sleep here
    std::condition_variable  cv_done;   // waiters sleep here
    std::deque<job>          queue;
    std::vector<std::thread> workers;
    bool                     shutdown = false;

    std::unordered_map<const void *, std::shared_ptr<ticket>> tickets;

    // cumulative observability counters (GGML_MOE_PREFETCH_DEBUG)
    std::atomic<uint64_t> n_jobs{0};
    std::atomic<uint64_t> n_skipped{0};
    std::atomic<uint64_t> bytes_populated{0};
    std::atomic<uint64_t> n_colded{0};

    ~prefetch_pool() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            shutdown = true;
            // complete outstanding tickets so no waiter blocks forever
            for (auto & j : queue) {
                if (j.tk) {
                    j.tk->pending.fetch_sub(1, std::memory_order_acq_rel);
                }
            }
            queue.clear();
            cv_work.notify_all();
            cv_done.notify_all();
        }
        for (auto & w : workers) {
            if (w.joinable()) w.join();
        }
        workers.clear();
        if (getenv("GGML_MOE_PREFETCH_DEBUG") && n_jobs.load() > 0) {
            fprintf(stderr, "%s: jobs=%llu skipped_resident=%llu bytes_populated=%.2f GiB colded=%llu\n", __func__,
                    (unsigned long long) n_jobs.load(), (unsigned long long) n_skipped.load(),
                    (double) bytes_populated.load()/(1024.0*1024.0*1024.0),
                    (unsigned long long) n_colded.load());
        }
    }

    void start(int n_threads) {
        stop();
        std::lock_guard<std::mutex> lock(mtx);
        shutdown = false;
        workers.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([this] { run(); });
        }
    }

    static bool chunk_resident(uintptr_t addr, size_t len) {
        const long page = sysconf(_SC_PAGESIZE);
        const uintptr_t astart = addr & ~(uintptr_t)(page - 1);
        const size_t    alen   = (addr + len) - astart;
        const size_t    npages = (alen + page - 1)/page;
        if (npages > GGML_MOE_PREFETCH_CHUNK/4096 + 2) {
            return false;
        }
        unsigned char vec[GGML_MOE_PREFETCH_CHUNK/4096 + 2];
        if (mincore((void *)astart, alen, vec) != 0) {
            return false;
        }
        for (size_t i = 0; i < npages; ++i) {
            if (!(vec[i] & 1)) return false;
        }
        return true;
    }

    void run() {
        const long page = sysconf(_SC_PAGESIZE);
        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_work.wait(lock, [this] { return shutdown || !queue.empty(); });
                if (shutdown) return;
                j = std::move(queue.front());
                queue.pop_front();
            }
            n_jobs.fetch_add(1, std::memory_order_relaxed);
            bool done_read = false;
            // skip when every page is already resident; keeps the drain rate
            // high when the cache is warm
            if (chunk_resident(j.addr, j.len)) {
                n_skipped.fetch_add(1, std::memory_order_relaxed);
            } else {
                const uintptr_t astart = j.addr & ~(uintptr_t)(page - 1);
                const size_t    alen   = ((j.addr + j.len + page - 1) & ~(uintptr_t)(page - 1)) - astart;
                if (madvise((void *)astart, alen, MADV_POPULATE_READ) == 0) {
                    bytes_populated.fetch_add(alen, std::memory_order_relaxed);
                    done_read = true;
                } // on failure the fault path takes over
            }
            if (j.tk) {
                const int left = j.tk->pending.fetch_sub(1, std::memory_order_acq_rel);
                std::lock_guard<std::mutex> lock(mtx);
                if (done_read && j.tk->track_reads) {
                    j.tk->read_chunks.emplace_back(j.addr, j.len);
                }
                if (left == 1) {
                    cv_done.notify_all();
                }
            }
        }
    }
};

struct prefetch_state {
    std::mutex                 reg_mtx;
    std::vector<mapping_entry> mappings;

    std::atomic<uint64_t> epoch{1};

    std::mutex                     pool_mtx;
    std::shared_ptr<prefetch_pool> pool;
};

static prefetch_state & state() {
    static prefetch_state s;
    return s;
}

// true when [p, p+len) lies inside a registered mmap
static bool is_mapped(const void * p, size_t len) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.reg_mtx);
    const uintptr_t a = (uintptr_t)p;
    for (const auto & m : s.mappings) {
        if (a >= m.base && a + len <= m.base + m.size) return true;
    }
    return false;
}

// collect [offset, offset+len) ranges of the selected experts of weight tensor w
static void collect_selected_ranges(const ggml_tensor * w, const ggml_tensor * ids,
        std::vector<std::pair<size_t, size_t>> & ranges) {
    const int64_t n_as   = w->ne[2];
    const size_t  stride = w->nb[2];
    const size_t  wbytes = ggml_nbytes(w);

    if (n_as <= 1 || stride == 0) {
        ranges.emplace_back(0, wbytes);
        return;
    }

    std::vector<uint32_t> seen((n_as + 31)/32, 0);
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const int32_t id = *(const int32_t *)((const char *)ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
            if (id < 0 || id >= n_as) continue; // ids may hold -1 sentinels (ggml_top_k_thresh)
            seen[id >> 5] |= 1u << (id & 31);
        }
    }
    // coalesce consecutive experts into single ranges
    int64_t id = 0;
    while (id < n_as) {
        while (id < n_as && !(seen[id >> 5] & (1u << (id & 31)))) ++id;
        if (id >= n_as) break;
        int64_t first = id;
        while (id < n_as && (seen[id >> 5] & (1u << (id & 31)))) ++id;
        const size_t off = (size_t)first*stride;
        const size_t len = std::min<size_t>((size_t)(id - first)*stride, wbytes - off);
        if (off < wbytes && len > 0) {
            ranges.emplace_back(off, len);
        }
    }
}

// enqueue ranges of tensor w; returns false when the engine is off or w is not mmap-registered
static bool enqueue_ranges(const ggml_tensor * w, const std::vector<std::pair<size_t, size_t>> & ranges, bool urgent, bool track_reads) {
    auto & s = state();
    std::shared_ptr<prefetch_pool> pool_sp;
    {
        std::lock_guard<std::mutex> plock(s.pool_mtx);
        pool_sp = s.pool;
    }
    if (!pool_sp) return false;

    if (!is_mapped(w->data, ggml_nbytes(w))) return false;

    const uint64_t cur_epoch = s.epoch.load(std::memory_order_relaxed);

    prefetch_pool & pool = *pool_sp;
    std::lock_guard<std::mutex> lock(pool.mtx);
    if (pool.shutdown) return false;

    auto & tk = pool.tickets[w];
    if (!tk) tk = std::make_shared<ticket>();
    if (tk->epoch == cur_epoch) {
        return true; // already enqueued for this scheduler pass
    }
    if (!urgent && pool.queue.size() > GGML_MOE_PREFETCH_MAX_QUEUE) {
        return true; // drop lookahead under backpressure; do not mark the epoch
    }
    tk->epoch = cur_epoch;
    tk->track_reads = track_reads;
    tk->read_chunks.clear();

    std::vector<job> jobs;
    for (const auto & r : ranges) {
        for (size_t o = r.first; o < r.first + r.second; o += GGML_MOE_PREFETCH_CHUNK) {
            const size_t len = std::min(GGML_MOE_PREFETCH_CHUNK, r.first + r.second - o);
            jobs.push_back({(uintptr_t)w->data + o, (uint32_t)len, tk});
        }
    }
    if (jobs.empty()) return true;

    tk->pending.fetch_add((int)jobs.size(), std::memory_order_acq_rel);
    if (urgent) {
        pool.queue.insert(pool.queue.begin(), std::make_move_iterator(jobs.begin()), std::make_move_iterator(jobs.end()));
    } else {
        pool.queue.insert(pool.queue.end(), std::make_move_iterator(jobs.begin()), std::make_move_iterator(jobs.end()));
    }
    pool.cv_work.notify_all();
    return true;
}

static void node_weights_and_ids(const ggml_tensor * node, const ggml_tensor * & w0, const ggml_tensor * & w1, const ggml_tensor * & ids) {
    w0 = nullptr; w1 = nullptr; ids = nullptr;
    if (node->op == GGML_OP_MUL_MAT_ID) {
        w0  = node->src[0];
        ids = node->src[2];
    } else if (node->op == GGML_OP_MOE_FUSED_UP_GATE) {
        w0  = node->src[0];
        w1  = node->src[1]; // NULL when up/gate are packed into one tensor
        ids = node->src[3];
    }
}

// legacy fallback; hints kernel readahead for the selected experts when the
// read pool is unavailable
static void legacy_madvise(const ggml_tensor * w, const ggml_tensor * ids) {
    if (!w || !w->data || !ids || !ids->data) return;
    std::vector<std::pair<size_t, size_t>> ranges;
    collect_selected_ranges(w, ids, ranges);
    const uintptr_t page_mask = (uintptr_t)sysconf(_SC_PAGESIZE) - 1;
    const char * base = (const char *)w->data;
    for (const auto & r : ranges) {
        const uintptr_t start  = (uintptr_t)(base + r.first);
        const uintptr_t astart = start & ~page_mask;
        (void) madvise((void *)astart, (size_t)(start + r.second - astart), MADV_WILLNEED);
    }
}

} // namespace

void ggml_moe_prefetch_register_mapping(const void * base, size_t size) {
    if (!base || size == 0) return;
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.reg_mtx);
    for (const auto & m : s.mappings) {
        if (m.base == (uintptr_t)base) return; // contexts may re-register the same model
    }
    s.mappings.push_back({(uintptr_t)base, size});
}

void ggml_moe_prefetch_unregister_mapping(const void * base) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.reg_mtx);
    // a queued job may still point into this range; its madvise then fails
    // (ENOMEM once unmapped) and the worker skips the chunk
    s.mappings.erase(std::remove_if(s.mappings.begin(), s.mappings.end(),
                [base](const mapping_entry & m) { return m.base == (uintptr_t)base; }),
            s.mappings.end());
}

static bool populate_read_supported() {
    const long page = sysconf(_SC_PAGESIZE);
    void * p = mmap(nullptr, (size_t)page, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    const bool ok = madvise(p, (size_t)page, MADV_POPULATE_READ) == 0;
    munmap(p, (size_t)page);
    return ok;
}

void ggml_moe_prefetch_set_n_threads(int n_threads) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.pool_mtx);
    if (n_threads <= 0) {
        s.pool.reset();
        return;
    }
    if (!populate_read_supported()) {
        fprintf(stderr, "%s: MADV_POPULATE_READ not supported; MoE prefetch disabled\n", __func__);
        s.pool.reset();
        return;
    }
    if (s.pool && s.pool->workers.size() == (size_t)n_threads) {
        return;
    }
    s.pool.reset();
    s.pool = std::make_shared<prefetch_pool>();
    s.pool->start(n_threads);
}

bool ggml_moe_prefetch_enabled(void) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.pool_mtx);
    return s.pool != nullptr;
}

void ggml_moe_prefetch_new_epoch(void) {
    state().epoch.fetch_add(1, std::memory_order_relaxed);
}

void ggml_moe_prefetch_node(const struct ggml_tensor * node) {
    const ggml_tensor * w0; const ggml_tensor * w1; const ggml_tensor * ids;
    node_weights_and_ids(node, w0, w1, ids);
    if (!w0 || !ids || !ids->data) return;

    std::vector<std::pair<size_t, size_t>> ranges;
    collect_selected_ranges(w0, ids, ranges);
    enqueue_ranges(w0, ranges, /*urgent =*/ true, /*track_reads =*/ false);
    if (w1) {
        ranges.clear();
        collect_selected_ranges(w1, ids, ranges);
        enqueue_ranges(w1, ranges, /*urgent =*/ true, /*track_reads =*/ false);
    }
}

void ggml_moe_prefetch_tensor(const struct ggml_tensor * w) {
    if (!w || !w->data) return;
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.emplace_back(0, ggml_nbytes(w));
    enqueue_ranges(w, ranges, /*urgent =*/ false, /*track_reads =*/ true);
}

#ifndef MADV_COLD
#define MADV_COLD 20
#endif
void ggml_moe_prefetch_cold(const struct ggml_tensor * w) {
    auto & s = state();
    std::shared_ptr<prefetch_pool> pool;
    {
        std::lock_guard<std::mutex> plock(s.pool_mtx);
        pool = s.pool;
    }
    if (!pool) return;
    std::vector<std::pair<uintptr_t, uint32_t>> chunks;
    {
        std::lock_guard<std::mutex> lock(pool->mtx);
        auto it = pool->tickets.find(w);
        if (it == pool->tickets.end() || !it->second->track_reads) return;
        chunks.swap(it->second->read_chunks);
    }
    const uintptr_t page_mask = (uintptr_t)sysconf(_SC_PAGESIZE) - 1;
    for (const auto & c : chunks) {
        const uintptr_t astart = c.first & ~page_mask;
        (void) madvise((void *)astart, (size_t)(c.first + c.second - astart), MADV_COLD);
    }
    pool->n_colded.fetch_add(chunks.size(), std::memory_order_relaxed);
}

void ggml_moe_prefetch_wait(const struct ggml_tensor * w) {
    auto & s = state();
    std::shared_ptr<prefetch_pool> pool;
    {
        std::lock_guard<std::mutex> plock(s.pool_mtx);
        pool = s.pool;
    }
    if (!pool) return;
    std::shared_ptr<ticket> tk;
    {
        std::lock_guard<std::mutex> lock(pool->mtx);
        auto it = pool->tickets.find(w);
        if (it == pool->tickets.end()) return;
        tk = it->second;
    }
    if (tk->pending.load(std::memory_order_acquire) <= 0) return;
    // the shared_ptr keeps the pool alive; shutdown wakes cv_done
    std::unique_lock<std::mutex> lock(pool->mtx);
    pool->cv_done.wait(lock, [&] {
        return pool->shutdown || tk->pending.load(std::memory_order_acquire) <= 0;
    });
}

void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor * node, int ith) {
    if (ith != 0) return;
    const ggml_tensor * w0; const ggml_tensor * w1; const ggml_tensor * ids;
    node_weights_and_ids(node, w0, w1, ids);
    if (!w0 || !ids || !ids->data) return;

    if (!ggml_moe_prefetch_enabled()) {
        legacy_madvise(w0, ids);
        if (w1) legacy_madvise(w1, ids);
        return;
    }
    // fire-and-forget enqueue, idempotent within the current epoch; a no-op
    // when the scheduler hook already covered this node (the self-enqueue
    // handles pure-CPU graphs). We deliberately do not wait; the compute
    // threads' demand faults overlap with the workers' populates, which run ahead
    // in the same expert-index order.
    ggml_moe_prefetch_node(node);
}

#elif defined(_WIN32)

// Windows arm: no worker pool (enabled() stays false, so the kernel hook is the only entry). At MoE matmul start thread 0 reads the routed expert ids
// and hands the selected expert ranges of each registered mmap weight to PrefetchVirtualMemory, a one-call asynchronous read-ahead hint. Hints only: a
// failed or missing call costs the demand faults it would have avoided. Read from Strata (MIT) expert warm path; no code copied.
// Row S69-STRATA-S1-WINDOWS-EXPERT-WARM-ARM-2026-10-07-01.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace {

struct win_mapping { uintptr_t base; size_t size; };

struct win_state {
    std::mutex               mtx;
    std::vector<win_mapping> mappings;
    // observability for tests: what the hook actually handed to PrefetchVirtualMemory (a direct observation, so a test
    // does not have to infer it from page-fault timing, which flips under host load)
    uint64_t                 n_calls = 0;   // PrefetchVirtualMemory invocations
    uint64_t                 n_ok    = 0;   // of those, returned TRUE
    std::vector<std::pair<size_t, size_t>> last_ranges; // [offset, len) of the most recent call, relative to the weight
};

win_state & wstate() {
    static win_state s;
    return s;
}

bool win_is_mapped(const void * p, size_t len) {
    auto & s = wstate();
    std::lock_guard<std::mutex> lock(s.mtx);
    const uintptr_t a = (uintptr_t) p;
    for (const auto & m : s.mappings) {
        if (a >= m.base && a + len <= m.base + m.size) return true;
    }
    return false;
}

// [offset, len) byte ranges of the experts named in ids, consecutive experts coalesced (same selection rule as the Linux arm)
void win_selected_ranges(const ggml_tensor * w, const ggml_tensor * ids, std::vector<std::pair<size_t, size_t>> & ranges) {
    const int64_t n_as   = w->ne[2];
    const size_t  stride = w->nb[2];
    const size_t  wbytes = ggml_nbytes(w);
    if (n_as <= 1 || stride == 0) {
        ranges.emplace_back(0, wbytes);
        return;
    }
    std::vector<uint32_t> seen((n_as + 31)/32, 0);
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const int32_t id = *(const int32_t *)((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
            if (id < 0 || id >= n_as) continue; // -1 sentinels from ggml_top_k_thresh
            seen[id >> 5] |= 1u << (id & 31);
        }
    }
    int64_t id = 0;
    while (id < n_as) {
        while (id < n_as && !(seen[id >> 5] & (1u << (id & 31)))) ++id;
        if (id >= n_as) break;
        const int64_t first = id;
        while (id < n_as && (seen[id >> 5] & (1u << (id & 31)))) ++id;
        const size_t off = (size_t) first*stride;
        const size_t len = std::min<size_t>((size_t)(id - first)*stride, wbytes - off);
        if (off < wbytes && len > 0) ranges.emplace_back(off, len);
    }
}

// Not gated on _WIN32_WINNT (mingw-w64 defaults it below 0x602): resolved at run time, absent on old Windows -> hint skipped.
typedef BOOL (WINAPI * prefetch_fn)(HANDLE, ULONG_PTR, void *, ULONG);
struct win_range_entry { PVOID VirtualAddress; SIZE_T NumberOfBytes; }; // layout of WIN32_MEMORY_RANGE_ENTRY

prefetch_fn win_prefetch_fn() {
    static prefetch_fn fn = (prefetch_fn)(void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    return fn;
}

void win_warm(const ggml_tensor * w, const ggml_tensor * ids) {
    prefetch_fn fn = win_prefetch_fn();
    if (!fn || !w || !w->data || !ids || !ids->data) return;
    if (!win_is_mapped(w->data, ggml_nbytes(w))) return;
    std::vector<std::pair<size_t, size_t>> ranges;
    win_selected_ranges(w, ids, ranges);
    if (ranges.empty()) return;
    std::vector<win_range_entry> entries;
    entries.reserve(ranges.size());
    for (const auto & r : ranges) {
        entries.push_back({ (PVOID)((char *) w->data + r.first), (SIZE_T) r.second });
    }
    const BOOL ok = fn(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0);
    auto & s = wstate();
    std::lock_guard<std::mutex> lock(s.mtx);
    s.n_calls++;
    if (ok) s.n_ok++;
    s.last_ranges = ranges;
}

} // namespace

uint64_t ggml_moe_prefetch_win_calls(uint64_t * n_ok) {
    auto & s = wstate();
    std::lock_guard<std::mutex> lock(s.mtx);
    if (n_ok) *n_ok = s.n_ok;
    return s.n_calls;
}

size_t ggml_moe_prefetch_win_last_ranges(size_t * offs, size_t * lens, size_t cap) {
    auto & s = wstate();
    std::lock_guard<std::mutex> lock(s.mtx);
    const size_t n = s.last_ranges.size();
    for (size_t i = 0; i < n && i < cap; ++i) {
        offs[i] = s.last_ranges[i].first;
        lens[i] = s.last_ranges[i].second;
    }
    return n;
}

void ggml_moe_prefetch_win_reset_stats(void) {
    auto & s = wstate();
    std::lock_guard<std::mutex> lock(s.mtx);
    s.n_calls = 0;
    s.n_ok = 0;
    s.last_ranges.clear();
}

void ggml_moe_prefetch_register_mapping(const void * base, size_t size) {
    if (!base || size == 0) return;
    auto & s = wstate();
    std::lock_guard<std::mutex> lock(s.mtx);
    for (const auto & m : s.mappings) {
        if (m.base == (uintptr_t) base) return;
    }
    s.mappings.push_back({ (uintptr_t) base, size });
}

void ggml_moe_prefetch_unregister_mapping(const void * base) {
    auto & s = wstate();
    std::lock_guard<std::mutex> lock(s.mtx);
    s.mappings.erase(std::remove_if(s.mappings.begin(), s.mappings.end(),
                [base](const win_mapping & m) { return m.base == (uintptr_t) base; }), s.mappings.end());
}

void ggml_moe_prefetch_set_n_threads(int) {}
bool ggml_moe_prefetch_enabled(void) { return false; }
void ggml_moe_prefetch_new_epoch(void) {}
void ggml_moe_prefetch_node(const struct ggml_tensor *) {}
void ggml_moe_prefetch_tensor(const struct ggml_tensor *) {}
void ggml_moe_prefetch_wait(const struct ggml_tensor *) {}
void ggml_moe_prefetch_cold(const struct ggml_tensor *) {}

void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor * node, int ith) {
    if (ith != 0) return;
    const ggml_tensor * w0 = nullptr; const ggml_tensor * w1 = nullptr; const ggml_tensor * ids = nullptr;
    if (node->op == GGML_OP_MUL_MAT_ID) {
        w0 = node->src[0]; ids = node->src[2];
    } else if (node->op == GGML_OP_MOE_FUSED_UP_GATE) {
        w0 = node->src[0]; w1 = node->src[1]; ids = node->src[3];
    }
    if (!w0 || !ids || !ids->data) return;
    win_warm(w0, ids);
    if (w1) win_warm(w1, ids);
}

#else // neither Linux nor Windows

void ggml_moe_prefetch_register_mapping(const void *, size_t) {}
void ggml_moe_prefetch_unregister_mapping(const void *) {}
void ggml_moe_prefetch_set_n_threads(int) {}
bool ggml_moe_prefetch_enabled(void) { return false; }
void ggml_moe_prefetch_new_epoch(void) {}
void ggml_moe_prefetch_node(const struct ggml_tensor *) {}
void ggml_moe_prefetch_tensor(const struct ggml_tensor *) {}
void ggml_moe_prefetch_wait(const struct ggml_tensor *) {}
void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor *, int) {}
void ggml_moe_prefetch_cold(const struct ggml_tensor *) {}

#endif
