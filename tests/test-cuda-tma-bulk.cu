// 1D TMA (cp.async.bulk + mbarrier complete_tx) helpers in ggml-cuda/cp-async.cuh: do they deliver the SAME shared
// memory tile as the existing per-thread cp.async loader, for the layout the FA kernels use?
//
// GATE-CLAIM-OK: this test names a GPU because it runs on one; it states no availability claim. With no device it
// prints SKIP and exits 77.
//
// WHY. Row S69-SM120-PICK-TMA-WARPSPEC-CLUSTER-FA-AND-MMQ-2026-10-20-01: sm_120a accepts TMA, mbarrier and clusters
// (wgmma/tcgen05 are rejected), and no fork kernel used TMA. Before any FA kernel switches its K/V loads, the helpers
// themselves must be shown correct on the layout those kernels use: nbatch rows of D2 half2, global row stride
// stride_KV, shared row stride stride_tile = D2 + 4 (16 bytes of padding per row against bank conflicts).
//
// ARMS (all write the shared tile back to global so the host compares bytes):
//   REF   per-thread cp_async_cg_16 + cp_async_wait_all + __syncthreads (the current loader's mechanism)
//   TMA   one thread per row issues tma_bulk_g2s(row); thread 0 arrives with expect_tx = nbatch*D2*4 bytes;
//         all threads tma_mbar_wait(phase). TWO phases into the SAME buffer: phase 0 loads tile A, the block reads
//         it out, then fence.proxy.async + phase 1 loads tile B over it. Phase 1 checks the parity flip and the
//         generic-read -> async-write ordering; a stale phase-0 tile in the phase-1 output is a FAIL.
//   FALLBACK  src offset by 4 bytes (not 16-aligned): tma_bulk_ok() must return false, the kernel takes the
//         cp_async_cg_16 path... which needs 16-byte alignment too, so the fallback is the synchronous copy.
//   NEG   TMA arm with each row's src shifted by one row: the comparator MUST report a mismatch.
// Shapes: D in {64, 128, 256, 576}, nbatch in {32, 64}, stride_KV in {D2, D2+8 (padded view), 3*D2 (strided)}.
// HANG GUARD: the host polls cudaStreamQuery for up to 10 s; a barrier that never completes prints HANG, exit 3.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/cp-async.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum Arm { ARM_REF = 0, ARM_TMA = 1, ARM_NEG = 2 };

// out layout: [2 phases][nbatch][D2] half2 (REF/NEG write phase 1 only for tile B, phase 0 for tile A)
template <int arm>
static __global__ void k_load(const half2 * __restrict__ A, const half2 * __restrict__ B, half2 * __restrict__ out,
                              const int D2, const int stride_KV, const int nbatch, int * __restrict__ path) {
    extern __shared__ half2 tile[];
    const int stride_tile = D2 + 4;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const unsigned int tile32 = ggml_cuda_cvta_generic_to_shared(tile);

#ifdef TMA_BULK_AVAILABLE
    __shared__ alignas(8) unsigned long long bar_storage;
    const unsigned int bar = ggml_cuda_cvta_generic_to_shared(&bar_storage);
    const unsigned int row_bytes = D2 * sizeof(half2);
    const bool tma_rows_ok = tma_bulk_ok(A, tile32, row_bytes) && tma_bulk_ok(B, tile32, row_bytes) &&
                             (stride_KV * sizeof(half2)) % 16 == 0 && (stride_tile * sizeof(half2)) % 16 == 0;
    if (arm != ARM_REF && tid == 0) {
        tma_mbar_init(bar, 1);
        tma_mbar_init_fence();
    }
    __syncthreads();
    unsigned int phase = 0;
#endif

    for (int ph = 0; ph < 2; ++ph) {
        const half2 * src = ph == 0 ? A : B;
        bool used_tma = false;
#ifdef TMA_BULK_AVAILABLE
        if (arm != ARM_REF && tma_rows_ok) {
            used_tma = true;
            if (tid == 0) {
                tma_fence_proxy_async();   // previous phase's generic reads of `tile` precede these async writes
                tma_mbar_arrive_expect_tx(bar, nbatch * row_bytes);
            }
            for (int i = tid; i < nbatch; i += nthr) {
                const int srow = arm == ARM_NEG ? (i + 1) % nbatch : i;
                tma_bulk_g2s(tile32 + i * stride_tile * sizeof(half2), src + srow * stride_KV, row_bytes, bar);
            }
            tma_mbar_wait(bar, phase);
            phase ^= 1;
        }
#endif
        if (!used_tma) {
            const bool aligned = ((uintptr_t) src % 16) == 0 && (stride_KV * sizeof(half2)) % 16 == 0;
            if (arm == ARM_REF && aligned) {
                const int chunks = D2 / 4;   // 16 bytes = 4 half2
                for (int c = tid; c < nbatch * chunks; c += nthr) {
                    const int i = c / chunks, k = c % chunks;
                    cp_async_cg_16<64>(tile32 + i * stride_tile * sizeof(half2) + k * 16, src + i * stride_KV + k * 4);
                }
                cp_async_wait_all();
            } else {
                for (int c = tid; c < nbatch * D2; c += nthr) {
                    const int i = c / D2, k = c % D2;
                    const int srow = arm == ARM_NEG ? (i + 1) % nbatch : i;
                    tile[i * stride_tile + k] = src[srow * stride_KV + k];
                }
            }
            __syncthreads();
        }
        if (tid == 0) path[ph] = used_tma ? 1 : 0;
        for (int c = tid; c < nbatch * D2; c += nthr) {
            const int i = c / D2, k = c % D2;
            out[(ph * nbatch + i) * D2 + k] = tile[i * stride_tile + k];
        }
        __syncthreads();   // all generic reads of `tile` done before the next phase overwrites it
    }
}

static bool wait_or_hang(const char * what) {
    // A launch that never started is not a hang and not a wrong tile. MEASURED 2026-10-20: under a 3 GB job commit
    // cap the lazily loaded TMA instantiation failed to launch ("out of memory" with 30 GB VRAM free); unchecked, that
    // read as "TMA BAD" on all 32 cases, and an earlier run of the same binary read as HANG.
    const cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { printf("LAUNCH ERROR in %s: %s\n", what, cudaGetErrorString(le)); exit(2); }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const cudaError_t e = cudaStreamQuery(0);
        if (e == cudaSuccess) return true;
        if (e != cudaErrorNotReady) { printf("CUDA ERROR in %s: %s\n", what, cudaGetErrorString(e)); exit(2); }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
            printf("HANG in %s (barrier never completed in 10 s)\n", what);
            fflush(stdout);
            exit(3);
        }
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("SKIP: no CUDA device\n"); return 77; }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("device %s cc %d.%d\n", prop.name, prop.major, prop.minor);

    const int Ds[] = { 64, 128, 256, 576 };
    const int NBs[] = { 32, 64 };
    int failed = 0, cases = 0, tma_cases = 0, neg_caught = 0, neg_cases = 0, fb_cases = 0;

    for (int D : Ds) for (int nb : NBs) for (int sk = 0; sk < 4; ++sk) {
        const int D2 = D / 2;
        // sk 0: packed, 1: padded view (+8 half2 = 32 B), 2: strided 3*D2, 3: misaligned base (+1 half2 = 4 B)
        const int stride_KV = sk == 1 ? D2 + 8 : sk == 2 ? 3 * D2 : D2;
        const int base_off = sk == 3 ? 1 : 0;
        const size_t n = (size_t) nb * stride_KV + 16;
        std::vector<half2> hA(n), hB(n);
        for (size_t i = 0; i < n; ++i) {
            hA[i] = make_half2(__float2half((float) ((i * 7) % 1000) * 0.01f), __float2half((float) i * 1e-3f));
            hB[i] = make_half2(__float2half(-(float) ((i * 13) % 997) * 0.01f), __float2half(-(float) i * 2e-3f));
        }
        half2 *dA, *dB, *dRef, *dT, *dN;
        int *dPath;
        const size_t out_n = 2 * (size_t) nb * D2;
        cudaMalloc(&dA, n * sizeof(half2)); cudaMalloc(&dB, n * sizeof(half2));
        cudaMalloc(&dRef, out_n * sizeof(half2)); cudaMalloc(&dT, out_n * sizeof(half2)); cudaMalloc(&dN, out_n * sizeof(half2));
        cudaMalloc(&dPath, 2 * sizeof(int));
        cudaMemcpy(dA, hA.data(), n * sizeof(half2), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), n * sizeof(half2), cudaMemcpyHostToDevice);
        const size_t smem = (size_t) nb * (D2 + 4) * sizeof(half2);
        cudaFuncSetAttribute(k_load<ARM_REF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(k_load<ARM_TMA>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(k_load<ARM_NEG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);

        int path[2] = { -1, -1 };
        k_load<ARM_REF><<<1, 128, smem>>>(dA + base_off, dB + base_off, dRef, D2, stride_KV, nb, dPath);
        wait_or_hang("REF");
        k_load<ARM_TMA><<<1, 128, smem>>>(dA + base_off, dB + base_off, dT, D2, stride_KV, nb, dPath);
        wait_or_hang("TMA");
        cudaMemcpy(path, dPath, sizeof(path), cudaMemcpyDeviceToHost);
        k_load<ARM_NEG><<<1, 128, smem>>>(dA + base_off, dB + base_off, dN, D2, stride_KV, nb, dPath);
        wait_or_hang("NEG");

        std::vector<half2> r(out_n), t(out_n), g(out_n);
        cudaMemcpy(r.data(), dRef, out_n * sizeof(half2), cudaMemcpyDeviceToHost);
        cudaMemcpy(t.data(), dT, out_n * sizeof(half2), cudaMemcpyDeviceToHost);
        cudaMemcpy(g.data(), dN, out_n * sizeof(half2), cudaMemcpyDeviceToHost);

        // host oracle: what the tile must hold, independent of every device path
        std::vector<half2> want(out_n);
        for (int ph = 0; ph < 2; ++ph) for (int i = 0; i < nb; ++i) for (int k = 0; k < D2; ++k)
            want[(ph * nb + i) * D2 + k] = (ph == 0 ? hA : hB)[base_off + i * stride_KV + k];

        const bool ref_ok = memcmp(r.data(), want.data(), out_n * sizeof(half2)) == 0;
        const bool tma_ok = memcmp(t.data(), want.data(), out_n * sizeof(half2)) == 0;
        const bool neg_differs = memcmp(g.data(), want.data(), out_n * sizeof(half2)) != 0;
        const bool expect_tma = sk != 3;   // misaligned base must fall back
        const bool path_ok = expect_tma ? (path[0] == 1 && path[1] == 1) : (path[0] == 0 && path[1] == 0);
        const bool ok = ref_ok && tma_ok && neg_differs && path_ok;
        ++cases; if (expect_tma) ++tma_cases; else ++fb_cases;
        ++neg_cases; if (neg_differs) ++neg_caught;
        if (!ok) ++failed;
        printf("D=%3d nb=%2d stride_KV=%4d base_off=%d  REF %s  TMA %s (path %d/%d, want %s)  NEG %s  %s\n",
               D, nb, stride_KV, base_off, ref_ok ? "ok" : "BAD", tma_ok ? "ok" : "BAD", path[0], path[1],
               expect_tma ? "tma" : "fallback", neg_differs ? "caught" : "MISSED", ok ? "ok" : "FAIL");
        cudaFree(dA); cudaFree(dB); cudaFree(dRef); cudaFree(dT); cudaFree(dN); cudaFree(dPath);
    }
    printf("cases %d (tma %d, fallback %d), failed %d, negative control caught %d/%d\n",
           cases, tma_cases, fb_cases, failed, neg_caught, neg_cases);
    return failed ? 1 : 0;
}
