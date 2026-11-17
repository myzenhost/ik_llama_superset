// --fit must know how many bytes the --lora adapters will take.
//
// common_init_result runs common_fit_params BEFORE the adapters are loaded, and the adapter tensors land in the
// buffer type of the base tensor they patch, so on GPU-resident layers their bytes were VRAM the fit never accounted
// for: a tight --fit plus --lora could OOM at adapter load or leave less KV room than planned. The fit target now
// carries the adapter bytes, the same way it carries --moe-cache-mib and (in mtmd-cli) the mmproj.
// Row S69-ENGREV-PETALS-LORA-ADAPTER-FIT-RESERVE-2026-10-27-01 (idea from petals/petals _choose_num_blocks, MIT;
// own code, no code copied).
//
// Model-free: synthetic adapter GGUFs written with gguf_init_empty.
//   A  one adapter, known tensor bytes          -> the summer returns exactly the tensor bytes
//   B  two adapters                             -> the sum
//   C  scale 0.0                                -> still counted (the adapter is loaded and resident anyway)
//   D  missing file                             -> contributes 0, no crash
//   E  CPU-only (no offload)                    -> reserve is 0
//   F  offload                                  -> reserve equals the summed bytes
//   G  no adapters                              -> 0

#include "common.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        printf("  FAIL: %s -- ", #cond);                        \
        printf(__VA_ARGS__);                                    \
        printf("\n");                                           \
        g_fail++;                                               \
    } else {                                                    \
        printf("  ok:   %s\n", #cond);                          \
    }                                                           \
} while (0)

// writes a GGUF with n_tensors F32 tensors of rows x cols; returns the sum of their data bytes
static size_t write_adapter(const std::string & path, int n_tensors, int64_t rows, int64_t cols) {
    struct ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_str(g, "general.type", "adapter");
    size_t total = 0;
    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        const std::string name = "blk." + std::to_string(i) + ".attn_q.weight.lora_a";
        ggml_set_name(t, name.c_str());
        ggml_set_zero(t);
        gguf_add_tensor(g, t);
        total += ggml_nbytes(t);
    }
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    return total;
}

static common_adapter_lora_info info(const std::string & path, float scale) {
    common_adapter_lora_info a;
    a.path  = path;
    a.scale = scale;
    return a;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "llama-lora-fit-reserve";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string p1 = (dir / "a1.gguf").string();
    const std::string p2 = (dir / "a2.gguf").string();
    const size_t b1 = write_adapter(p1, 3, 16, 64);   // 3 * 4096 bytes
    const size_t b2 = write_adapter(p2, 5, 32, 128);  // 5 * 16384 bytes

    printf("case A: one adapter\n");
    CHECK(b1 == 3u * 4096u, "synthetic adapter bytes %zu", b1);
    CHECK(common_lora_adapters_tensor_bytes({ info(p1, 1.0f) }) == b1, "summer returns %zu",
          common_lora_adapters_tensor_bytes({ info(p1, 1.0f) }));

    printf("case B: two adapters\n");
    CHECK(common_lora_adapters_tensor_bytes({ info(p1, 1.0f), info(p2, 0.5f) }) == b1 + b2, "sum");

    printf("case C: scale 0 is still resident\n");
    CHECK(common_lora_adapters_tensor_bytes({ info(p1, 0.0f), info(p2, 0.0f) }) == b1 + b2, "scale 0 counted");

    printf("case D: missing adapter file\n");
    CHECK(common_lora_adapters_tensor_bytes({ info((dir / "nope.gguf").string(), 1.0f), info(p1, 1.0f) }) == b1,
          "the missing one adds 0, the other still counts");

    printf("case E: CPU only\n");
    CHECK(common_lora_fit_reserve_bytes({ info(p1, 1.0f) }, /*gpu_offload=*/false) == 0, "no device, no reserve");

    printf("case F: offload\n");
    CHECK(common_lora_fit_reserve_bytes({ info(p1, 1.0f), info(p2, 1.0f) }, true) == b1 + b2, "reserve = summed bytes");

    printf("case G: no adapters\n");
    CHECK(common_lora_adapters_tensor_bytes({}) == 0 && common_lora_fit_reserve_bytes({}, true) == 0, "empty is 0");

    fs::remove_all(dir);
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
