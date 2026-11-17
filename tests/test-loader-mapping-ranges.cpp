// Per-run mmap buffers: a context that holds tensors on both sides of a LARGE foreign tensor must map each
// contiguous run, not one span over the foreign tensor too (llama_model_loader::get_mapping_ranges, consumed
// by llama.cpp's mmap buffer creation and by load_all_data's per-buffer lookup).
// SOURCE-READ: tmp/_unsl-branches_g2.out (git show 258345ef of the unslothai/llama.cpp remote, MIT)
//
// File layout (I8 tensors), in file order:
//   a1 256K | a2 256K | m 8M | d1 256K | b 40M | c1 256K | c2 256K
// Contexts under test (a context holds only the tensors named):
//   {a1,a2,c1,c2}      gap = 8M+256K+40M  > 32M  -> 2 runs  (one span would map m, d1 and b too)
//   {a1,a2}            adjacent                  -> 1 run   (mapping each tensor alone would be 2)
//   {a1,d1}            gap = 256K+8M      < 32M  -> 1 run   (small gaps are merged, no buffer for padding)
//   {d1,c1}            gap = 40M          > 32M  -> 2 runs
//   {a1,a2,m,d1,b,c1,c2} (everything)             -> 1 run
// Then the real thing: init_mappings (a real mmap of the file), one CPU buffer per run, load_all_data, and every
// tensor must resolve into the buffer that CONTAINS it and point at the right bytes of the file.
//
// usage: test-loader-mapping-ranges     (writes a ~49 MiB temp file in the cwd and removes it)
// Row: S69-UNSLOTH-ADOPTABLE-BRANCHES-NOT-YET-PORTED-01.
#include "../src/llama-model-loader.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

struct spec { const char * name; size_t bytes; };
static const size_t KiB = 1024, MiB = 1024 * 1024;
static const spec specs[] = {
    { "a1", 256 * KiB }, { "a2", 256 * KiB }, { "m", 8 * MiB }, { "d1", 256 * KiB },
    { "b", 40 * MiB }, { "c1", 256 * KiB }, { "c2", 256 * KiB },
};

static uint8_t pattern(int tensor_id, size_t i) { return (uint8_t) (tensor_id * 37 + i * 7 + (i >> 9)); }

static void write_file(const char * path) {
    ggml_init_params ip = { /*mem_size*/ 50 * MiB + 1 * MiB, nullptr, /*no_alloc*/ false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    int id = 0;
    for (const spec & s : specs) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) s.bytes);
        ggml_set_name(t, s.name);
        uint8_t * d = (uint8_t *) t->data;
        for (size_t i = 0; i < s.bytes; i++) { d[i] = pattern(id, i); }
        gguf_add_tensor(g, t);
        id++;
    }
    gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
}

// a no_alloc context holding just `names`
static ggml_context * make_ctx(const std::vector<std::string> & names) {
    ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead() * 16 + 4096, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    for (const std::string & n : names) {
        size_t bytes = 0;
        for (const spec & s : specs) { if (n == s.name) { bytes = s.bytes; } }
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) bytes);
        ggml_set_name(t, n.c_str());
    }
    return ctx;
}

int main() {
    const char * path = "test-loader-mapping-ranges.tmp.gguf";
    write_file(path);

    {
        llama_model_loader ml(path, /*ncmoe*/ 0, /*use_mmap*/ true, /*check_tensors*/ false, false, false, false, false, false,
                              nullptr, nullptr);
        ml.init_mappings(/*prefetch*/ false);

        struct cs { const char * label; std::vector<std::string> names; size_t want_runs; };
        const cs cases[] = {
            { "{a1,a2,c1,c2}  (large foreign gap)",      { "a1", "a2", "c1", "c2" },                         2 },
            { "{a1,a2}        (adjacent)",               { "a1", "a2" },                                     1 },
            { "{a1,d1}        (8M gap, merged)",         { "a1", "d1" },                                     1 },
            { "{d1,c1}        (40M gap)",                { "d1", "c1" },                                     2 },
            { "everything",                              { "a1", "a2", "m", "d1", "b", "c1", "c2" },         1 },
        };
        for (const cs & c : cases) {
            ggml_context * ctx = make_ctx(c.names);
            std::vector<std::pair<size_t, size_t>> runs;
            void * addr = nullptr;
            ml.get_mapping_ranges(runs, &addr, 0, ctx);
            CHECK(runs.size() == c.want_runs, "%s: wanted %zu run(s), got %zu", c.label, c.want_runs, runs.size());
            // never mapped alone: fewer runs than tensors whenever tensors are adjacent
            if (c.names.size() > 1 && c.want_runs < c.names.size()) {
                CHECK(runs.size() < c.names.size(), "%s: must not be one run per tensor", c.label);
            }
            ggml_free(ctx);
        }

        // the single-span function the runs replace: its span covers the foreign tensors, the runs do not
        {
            ggml_context * ctx = make_ctx({ "a1", "a2", "c1", "c2" });
            size_t first = 0, last = 0; void * addr = nullptr;
            ml.get_mapping_range(&first, &last, &addr, 0, ctx);
            std::vector<std::pair<size_t, size_t>> runs;
            ml.get_mapping_ranges(runs, &addr, 0, ctx);
            size_t run_bytes = 0;
            for (auto & r : runs) { run_bytes += r.second - r.first; }
            printf("info  single span = %.2f MiB, runs = %.2f MiB for 1 MiB of own tensors\n", (last - first) / 1048576.0, run_bytes / 1048576.0);
            CHECK(last - first > 48 * MiB, "the single span must cover the foreign tensors (control), got %.2f MiB", (last - first) / 1048576.0);
            CHECK(run_bytes < 2 * MiB, "the runs must map only the context's own bytes, got %.2f MiB", run_bytes / 1048576.0);
            ggml_free(ctx);
        }

        // end to end on a real mapping: buffers per run, load_all_data resolves each tensor into its run
        {
            ggml_context * ctx = make_ctx({ "a1", "a2", "c1", "c2" });
            std::vector<std::pair<size_t, size_t>> runs;
            void * addr = nullptr;
            ml.get_mapping_ranges(runs, &addr, 0, ctx);
            llama_buf_map bufs;
            for (auto & r : runs) {
                ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr((char *) addr + r.first, r.second - r.first);
                bufs[0].push_back(buf);
            }
            CHECK(bufs[0].size() == 2, "two buffers for two runs");
            const bool ok = ml.load_all_data(ctx, nullptr, bufs, nullptr, nullptr, nullptr);
            CHECK(ok, "load_all_data must complete");
            int id_of[] = { 0, 1, 5, 6 };
            int k = 0;
            for (const char * n : { "a1", "a2", "c1", "c2" }) {
                ggml_tensor * t = ggml_get_tensor(ctx, n);
                const auto * w = ml.get_weight(n);
                CHECK(t && w && t->data == (uint8_t *) addr + w->offs, "%s must point straight into the file mapping", n);
                bool match = t && t->data;
                const size_t nb = t ? ggml_nbytes(t) : 0;
                for (size_t i = 0; match && i < nb; i += 4099) { match = ((uint8_t *) t->data)[i] == pattern(id_of[k], i); }
                CHECK(match, "%s must hold the file's bytes", n);
                // the buffer that owns it is the one for ITS run
                bool owned = false;
                if (t && t->buffer) {
                    for (size_t b = 0; b < bufs[0].size(); b++) {
                        if (t->buffer == bufs[0][b]) { owned = (b == (k < 2 ? 0u : 1u)); }
                    }
                }
                CHECK(owned, "%s must be allocated in the buffer of its own run", n);
                k++;
            }
            for (auto b : bufs[0]) { ggml_backend_buffer_free(b); }
            ggml_free(ctx);
        }
    }

    remove(path);
    printf(g_fail ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
