// llama_model_size / llama_model_n_params must count each weight of the FILE once.
// A tied head (no output.weight in the file) is materialised as a second tensor object carrying
// token_embd.weight's name (TENSOR_DUPLICATED); the old sum over tensors_by_name counted it twice
// (llama-bench: 761.80 MiB / 751.63 M vs upstream 603.87 MiB / 595.78 M for Qwen3-0.6B Q8_0).
// Row: S69-FORK-REPORTS-EXTRA-PARAMS-AND-SIZE-MERGED-VIEWS-2026-09-26-01. Test written here for this fork.
//
// usage: test-model-count-tied <dir with the ggufs from tests/make_tiny_shared_pair_gguf.py>
// Uses target.gguf (untied control: output.weight in the file) and target_tied.gguf (tied head).
// Expected values are read from the GGUF itself (sum of ggml_nelements / ggml_nbytes over its tensors).
// exit 0 = every check passed. CPU only, tiny F32 models, milliseconds.
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <string>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

static bool file_counts(const std::string & path, uint64_t & n_params, uint64_t & n_bytes, bool & has_output) {
    struct ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc =*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        return false;
    }
    n_params = 0; n_bytes = 0; has_output = false;
    for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        n_params += ggml_nelements(t);
        n_bytes  += ggml_nbytes(t);
        if (std::string(ggml_get_name(t)) == "output.weight") {
            has_output = true;
        }
    }
    gguf_free(g);
    ggml_free(meta);
    return true;
}

static void check_one(const std::string & path, bool expect_output) {
    uint64_t np = 0, nb = 0; bool has_out = false;
    CHECK(file_counts(path, np, nb, has_out), "gguf must read: %s", path.c_str());
    CHECK(has_out == expect_output, "fixture shape: output.weight %s in %s", expect_output ? "expected" : "absent expected", path.c_str());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = true;
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    CHECK(m != nullptr, "model must load: %s", path.c_str());
    if (!m) {
        return;
    }
    printf("info  %s: file %llu params %llu bytes | model %llu params %llu bytes\n", path.c_str(),
           (unsigned long long) np, (unsigned long long) nb,
           (unsigned long long) llama_model_n_params(m), (unsigned long long) llama_model_size(m));
    CHECK(llama_model_n_params(m) == np, "n_params must equal the distinct weights of the file");
    CHECK(llama_model_size(m)     == nb, "size must equal the distinct weight bytes of the file");
    llama_model_free(m);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf dir>\n", argv[0]);
        return 64;
    }
    const std::string dir = std::string(argv[1]) + "/";
    llama_backend_init();
    check_one(dir + "target.gguf",      true);   // control: untied, was already right
    check_one(dir + "target_tied.gguf", false);  // tied: was counted twice for token_embd
    llama_backend_free();
    printf(g_fail ? "FAILED %d\n" : "All tests passed\n", g_fail);
    return g_fail ? 1 : 0;
}
