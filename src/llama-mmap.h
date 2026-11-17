#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <cstdio>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define LLAMA_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;

    // fork-only: get_path()/clone() -- used by src/llama-model-loader.cpp, src/llama-reload.cpp
    const std::string & get_path() const;
    std::unique_ptr<llama_file> clone() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false, bool use_thp = false,
               const ranges & lazy_ranges = {});  // lazy_ranges: upstream lazy loading
    ~llama_mmap();

    size_t size() const;
    void * addr() const;
    bool is_anonymous() const;

    void dontneed_fragment(size_t first, size_t last);

    void random_fragment(size_t first, size_t last);

    // Selectively warm [first,last); deferred/VRAM-bound ranges stay cold.
    void prefetch_fragment(size_t first, size_t last);

    void unmap_fragment(size_t first, size_t last);

    // Windows: trim the COMPLETE interior pages of [first, last) of this read-only file view out of the process working
    // set so the OS can reuse them as standby memory; a later touch faults them back in and the bytes stay identical.
    // Other platforms do nothing here. Read from Strata (MIT) FileExpertSource::release(), no code copied.
    // Row S69-STRATA-S1-WINDOWS-WORKINGSET-RELEASE-2026-10-07-01.
    // CAPABILITY-GENUINELY-NA: a memory-residency hint on weight pages, not a model capability.
    void release_fragment(size_t first, size_t last);

    // ADOPTED from unslothai/llama.cpp PR #137 ("llama: batched readahead for lazily read
    // gather tables"). true if [ptr, ptr + len) is inside this mapping.
    bool contains(const void * ptr, size_t len) const;

    // Start reading the given rows of a tensor in this mapping, in ONE batch.
    // A sparse gather over an mmap'd table costs one synchronous major fault per row,
    // and on Windows nothing else in this fork issues a random-access readahead hint --
    // ggml-moe-prefetch.cpp is #if defined(__linux__) and stubs out entirely here.
    // Hints only: this never changes results, and a failed hint costs only the fault it
    // would have avoided.
    void prefetch_rows(const void * base, size_t stride, size_t row_size,
                       const int32_t * rows, size_t n_rows) const;

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Counters of the row-readahead hints issued so far (process wide). Test hooks: they let a test prove
// that a decode REACHED llama_mmap::prefetch_rows and that the OS call behind it ran (and did not fail),
// which a hint that "never changes results" cannot show any other way.
uint64_t llama_mmap_prefetch_hint_calls();   // prefetch_rows calls that produced at least one range
uint64_t llama_mmap_prefetch_hint_ranges();  // page runs those calls produced
uint64_t llama_mmap_prefetch_os_calls();     // PrefetchVirtualMemory / posix_madvise calls issued
uint64_t llama_mmap_prefetch_os_failed();    // ... that returned failure

// Load-time readahead calls issued by the llama_mmap constructor when prefetch > 0 (process wide).
uint64_t llama_mmap_load_prefetch_os_calls();
struct llama_memory_range {
    const void * addr;
    size_t size;
};

using llama_memory_ranges = std::vector<llama_memory_range>;

// Prefetch the host pages covering these memory ranges.
void llama_prefetch(llama_memory_ranges mr);

size_t llama_path_max();
