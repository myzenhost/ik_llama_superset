// Batched row readahead for a gather over an mmap'd table (llama_mmap::prefetch_rows, llama_model::prefetch_rows)
// and the proof that a decode actually REACHES it.
//
//  part A  llama_mmap::prefetch_rows: rows are merged to page runs (one OS call for the whole batch), an
//          out-of-range or negative row is ignored rather than turned into a hint outside the mapping, and the
//          OS call succeeds on this host (PrefetchVirtualMemory on Windows, posix_madvise elsewhere).
//  part B  llama_model::prefetch_rows: no-op for a tensor that is not in a mapping, a hint for one that is.
//  part C  a real tiny gemma4 with a per-layer token-embedding table, decoded from an mmap'd file: the decode
//          issues hints (the call site in llama_set_inputs is reached), issues none without mmap (control),
//          and the logits are identical either way (hints only).
//
// usage: test-mmap-row-prefetch [gemma4_ple.gguf]    (part C is skipped when no file is given; see
//        tests/make_tiny_gemma4_ple_gguf.py)
// Row: S69-UNSLOTH-ADOPTABLE-BRANCHES-NOT-YET-PORTED-01.
// SOURCE-READ: tmp/_unsl-branches_g3.out (git show 4e1865e3 of the unslothai/llama.cpp remote, MIT)
#include "llama.h"

#include "../src/llama-mmap.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static size_t page_size() { SYSTEM_INFO si; GetSystemInfo(&si); return (size_t) si.dwPageSize; }
#else
#include <unistd.h>
static size_t page_size() { return (size_t) sysconf(_SC_PAGESIZE); }
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

// llama_model::prefetch_rows is detected, not named, so this source also builds against a tree that lacks it:
// that is how the "before" run is made (such a tree has the mmap-level hint and no caller of it)
template <class M>
static auto model_prefetch(M * m, const ggml_tensor * t, const int32_t * rows, size_t n, int) -> decltype(m->prefetch_rows(t, rows, n), bool()) {
    m->prefetch_rows(t, rows, n);
    return true;
}
template <class M>
static bool model_prefetch(M *, const ggml_tensor *, const int32_t *, size_t, long) {
    return false;
}

struct counters {
    uint64_t calls, ranges, os, fail;
    static counters now() {
        return { llama_mmap_prefetch_hint_calls(), llama_mmap_prefetch_hint_ranges(), llama_mmap_prefetch_os_calls(), llama_mmap_prefetch_os_failed() };
    }
};

static void part_a() {
    printf("--- part A: llama_mmap::prefetch_rows\n");
    const char * path = "test-mmap-row-prefetch.tmp";
    const size_t ps   = page_size();
    const size_t size = 256 * ps;
    {
        FILE * f = fopen(path, "wb");
        std::vector<char> z(size, 1);
        fwrite(z.data(), 1, z.size(), f);
        fclose(f);
    }
    {
        llama_file file(path, "rb");
        llama_mmap map(&file, 0, false, false);
        const char * base = (const char *) map.addr();
        const size_t stride = 1024, row = 512;   // 4 rows per 4 KiB page; row index r starts at byte r*stride

        CHECK(map.contains(base, size), "the mapping must contain its own range");
        CHECK(!map.contains(base + size - 1, 2), "a range that runs past the end must not be contained");

        // rows 0..2 are on page 0 (stride 1024, 4 KiB pages): ONE page run, ONE OS call
        {
            const int32_t rows[] = { 0, 1, 2, 0, 2 };
            const counters c0 = counters::now();
            map.prefetch_rows(base, stride, row, rows, 5);
            const counters c1 = counters::now();
            CHECK(c1.calls - c0.calls == 1 && c1.ranges - c0.ranges == 1, "5 rows on one page = 1 range, saw %llu calls %llu ranges",
                    (unsigned long long) (c1.calls - c0.calls), (unsigned long long) (c1.ranges - c0.ranges));
            CHECK(c1.os - c0.os >= 1, "the OS readahead call must have been issued");
            CHECK(c1.fail - c0.fail == 0, "the OS readahead call must not have failed");
        }
        // pages 4,5,6 are adjacent, page 40 is not: 2 runs, still ONE OS call on Windows (PrefetchVirtualMemory takes all)
        {
            const size_t per_page = ps / stride;
            const int32_t rows[] = { (int32_t) (4 * per_page), (int32_t) (5 * per_page), (int32_t) (6 * per_page), (int32_t) (40 * per_page) };
            const counters c0 = counters::now();
            map.prefetch_rows(base, stride, row, rows, 4);
            const counters c1 = counters::now();
            CHECK(c1.ranges - c0.ranges == 2, "adjacent pages merge: 4 rows on pages 4,5,6,40 = 2 runs, saw %llu", (unsigned long long) (c1.ranges - c0.ranges));
#ifdef _WIN32
            CHECK(c1.os - c0.os == 1, "Windows batches every run into ONE PrefetchVirtualMemory call, saw %llu", (unsigned long long) (c1.os - c0.os));
#endif
            CHECK(c1.fail - c0.fail == 0, "no OS failure");
        }
        // a negative row and a row past the end of the mapping are dropped, not hinted
        {
            const int32_t rows[] = { -1, (int32_t) (size / stride) + 5, 2000000000 };
            const counters c0 = counters::now();
            map.prefetch_rows(base, stride, row, rows, 3);
            const counters c1 = counters::now();
            CHECK(c1.calls == c0.calls && c1.os == c0.os, "bad row indices must produce no hint at all");
        }
        // a row that straddles a page boundary pulls in both pages
        {
            const int32_t rows[] = { 3 };   // bytes 3072..3583: one page; use a wide row instead
            const size_t wide = ps + 100;
            const counters c0 = counters::now();
            map.prefetch_rows(base, ps * 2, wide, rows, 1);   // row 3 -> byte 6*ps, wide spans 2 pages
            const counters c1 = counters::now();
            CHECK(c1.ranges - c0.ranges == 1, "one wide row spanning two pages = 1 run");
        }
    }
    remove(path);
}

static std::vector<float> decode_logits(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 16; cp.n_ubatch = 16; cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::vector<float> res;
    if (!ctx) {
        return res;
    }
    const int T = 6;
    const llama_token toks[T] = { 3, 5, 7, 9, 11, 13 };
    llama_batch b = llama_batch_init(T, 0, 1);
    b.n_tokens = T;
    for (int i = 0; i < T; i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    if (llama_decode(ctx, b) == 0) {
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * lg = llama_get_logits(ctx);
        if (lg) {
            res.assign(lg, lg + (size_t) T * nv);
        }
    }
    llama_batch_free(b);
    llama_free(ctx);
    return res;
}

static void part_bc(const char * gguf) {
    printf("--- part B/C: llama_model::prefetch_rows + a real gemma4 decode (%s)\n", gguf);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = true;
    mp.defer_ple    = true;
    llama_model * model = llama_model_load_from_file(gguf, mp);
    CHECK(model != nullptr, "tiny gemma4 must load from the mmap'd file");
    if (!model) { return; }
    CHECK(model->tok_embd_per_layer != nullptr, "the model must have a per-layer token-embedding table");
    if (!model->tok_embd_per_layer) { llama_model_free(model); return; }

    // part B: a tensor that is in the mapping gets a hint, one that is not gets nothing
    {
        const int32_t rows[] = { 1, 2, 3 };
        const counters c0 = counters::now();
        const bool have = model_prefetch(model, model->tok_embd_per_layer, rows, 3, 0);
        const counters c1 = counters::now();
        CHECK(have, "llama_model::prefetch_rows must exist");
        CHECK(c1.calls - c0.calls == 1, "a mapped table must receive the hint, saw %llu", (unsigned long long) (c1.calls - c0.calls));

        float heap[64] = {};
        ggml_tensor fake = *model->tok_embd_per_layer;
        fake.data = heap;
        model_prefetch(model, &fake, rows, 3, 0);
        const counters c2 = counters::now();
        CHECK(c2.calls == c1.calls, "a tensor outside every mapping must be a no-op");

        model_prefetch(model, model->tok_embd_per_layer, nullptr, 3, 0);
        model_prefetch(model, model->tok_embd_per_layer, rows, 0, 0);
        model_prefetch(model, nullptr, rows, 3, 0);
        CHECK(counters::now().calls == c1.calls, "null rows / zero rows / null tensor must be no-ops");
    }

    // part C: the decode reaches it
    const counters d0 = counters::now();
    const std::vector<float> lg_mmap = decode_logits(model);
    const counters d1 = counters::now();
    CHECK(!lg_mmap.empty(), "tiny gemma4 must decode");
    CHECK(d1.calls - d0.calls >= 1, "a gemma4 decode must issue the per-layer-embedding readahead, saw %llu hint calls",
            (unsigned long long) (d1.calls - d0.calls));
    CHECK(d1.fail == d0.fail, "no OS readahead failure during the decode");
    llama_model_free(model);

    // control: without mmap there is no mapping to hint, so a decode issues none
    llama_model_params mp2 = llama_model_default_params();
    mp2.n_gpu_layers = 0;
    mp2.use_mmap     = false;
    llama_model * model2 = llama_model_load_from_file(gguf, mp2);
    CHECK(model2 != nullptr, "tiny gemma4 must load without mmap");
    if (model2) {
        const counters e0 = counters::now();
        const std::vector<float> lg_nommap = decode_logits(model2);
        const counters e1 = counters::now();
        CHECK(e1.calls == e0.calls, "no mmap = no hint, saw %llu", (unsigned long long) (e1.calls - e0.calls));
        CHECK(!lg_nommap.empty() && lg_nommap.size() == lg_mmap.size() &&
              memcmp(lg_nommap.data(), lg_mmap.data(), lg_mmap.size() * sizeof(float)) == 0,
              "hints only: logits with and without mmap readahead must be bit identical");
        llama_model_free(model2);
    }
}

int main(int argc, char ** argv) {
    llama_backend_init();
    part_a();
    if (argc > 1) {
        part_bc(argv[1]);
    } else {
        printf("--- part B/C skipped (no gguf given)\n");
    }
    llama_backend_free();
    printf(g_fail ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
