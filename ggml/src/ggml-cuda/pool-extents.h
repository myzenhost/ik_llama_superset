#pragma once

// Extent bookkeeping for the CUDA/HIP VMM buffer pool (ggml_cuda_pool_vmm in ggml-cuda.cu).
//
// WHY. The VMM pool used to be a pure bump allocator: free() asserted that the pointer being
// released was exactly the most recent live allocation (strict LIFO). The legacy pool next to it
// accepts frees in any order. Because the VMM pool is the PREFERRED pool whenever the driver
// supports virtual memory, the strictness sat on the common path: any caller that holds two
// ggml_cuda_pool_alloc objects across an interleaved lifetime aborts the process with a
// GGML_ASSERT instead of degrading. This class keeps the cheap bump behaviour for the LIFO case
// and adds the missing free-list for the rest.
//
// MODEL. Offsets are relative to the start of the reserved virtual range. [0, top) is the part of
// the range that has ever been handed out; inside it every byte is either live or on the free map.
// [top, capacity) is untouched. Invariants (checked by the unit test, tests/test-cuda-pool-extents.cpp):
//   I1 free extents never overlap each other and never touch (adjacent ones are merged);
//   I2 no free extent ends at `top` (such an extent is given back to the bump region instead);
//   I3 every free extent lies inside [0, top).
// Under a strictly LIFO workload no free extent ever exists, so addresses are identical to the old
// bump allocator and nothing changes for the existing callers.
//
// Pure C++17, no CUDA/HIP types, so it is compiled and tested on a CPU-only build.
// Design reference (behaviour only, no code copied): mlx/backend/cuda/allocator.cpp keeps freed
// buffers in a free list that any later allocation of a matching size may reuse in any free order
// (MIT, mlx commit 64ea011cb65f); the legacy ggml_cuda_pool_leg best-fit scan in ggml-cuda.cu is the
// in-tree precedent for the best-fit choice made here.

#include <cstddef>
#include <iterator>
#include <map>

struct ggml_cuda_pool_extents {
    size_t top = 0;                       // bump pointer: end of the part of the range ever handed out
    std::map<size_t, size_t> free_map;    // offset -> size of each free extent below `top`

    // Allocate `size` bytes inside a range that is `capacity` bytes long.
    // Best fit among the free extents first (smallest leftover, lowest offset on ties), otherwise bump.
    // Returns false when neither fits; the caller then grows the range by shortfall() and retries.
    bool alloc(size_t size, size_t capacity, size_t * offset) {
        auto best = free_map.end();
        for (auto it = free_map.begin(); it != free_map.end(); ++it) {
            if (it->second >= size && (best == free_map.end() || it->second < best->second)) {
                best = it;
                if (it->second == size) {
                    break;
                }
            }
        }
        if (best != free_map.end()) {
            const size_t off  = best->first;
            const size_t left = best->second - size;
            free_map.erase(best);
            if (left > 0) {
                free_map.emplace(off + size, left);   // remainder stays below `top`, still not touching it (I2)
            }
            *offset = off;
            return true;
        }
        if (size <= capacity && top <= capacity - size) {
            *offset = top;
            top += size;
            return true;
        }
        return false;
    }

    // Bytes the range must grow by so that alloc(size, ...) can succeed through the bump region.
    // Only meaningful after alloc() returned false.
    size_t shortfall(size_t size, size_t capacity) const {
        const size_t need = top + size;
        return need > capacity ? need - capacity : 0;
    }

    // Release [offset, offset+size). Returns false (and changes nothing) on a block that cannot be
    // a live allocation of this pool: outside [0, top) or overlapping a free extent (double free).
    bool release(size_t offset, size_t size) {
        if (size == 0) {
            return offset <= top;
        }
        if (offset > top || size > top - offset) {
            return false;
        }
        // first free extent starting at or after `offset`, and the one before it
        auto next = free_map.lower_bound(offset);
        if (next != free_map.end() && next->first < offset + size) {
            return false;
        }
        auto prev = next;
        bool has_prev = false;
        if (next != free_map.begin()) {
            prev = std::prev(next);
            has_prev = true;
            if (prev->first + prev->second > offset) {
                return false;
            }
        }
        size_t start = offset;
        size_t len   = size;
        if (has_prev && prev->first + prev->second == offset) {      // merge with the extent just below
            start = prev->first;
            len  += prev->second;
            free_map.erase(prev);
        }
        if (next != free_map.end() && next->first == offset + size) { // merge with the extent just above
            len += next->second;
            free_map.erase(next);
        }
        if (start + len == top) {                                     // I2: give the tail back to the bump region
            top = start;
        } else {
            free_map.emplace(start, len);
        }
        return true;
    }

    // Bytes currently live (handed out and not released).
    size_t used() const {
        size_t free_bytes = 0;
        for (const auto & e : free_map) {
            free_bytes += e.second;
        }
        return top - free_bytes;
    }
};
