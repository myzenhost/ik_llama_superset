#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Deterministic FLASH_ATTN_EXT probe; banded-API-free so the same source builds on base and final trees.
int main(int argc, char ** argv) {
    const std::string backend_kind = argc > 1 ? argv[1] : "cpu";
    const std::string output_path = argc > 2 ? argv[2] : "";

    // GATE-CLAIM-OK: design note about a SHAPE this fork's CUDA path declines, measured by running
    // the binary on the card (RTX 5090 reported, 32606 MiB, device initialised fine). Nothing is
    // claimed unavailable -- the device works; it is this tensor shape that is refused.
    //
    // OPTIONAL argv[3]: KV length. Upstream's fixed 47 is kept as the DEFAULT so a hash taken here
    // stays comparable with one taken on an unsloth or upstream build. It is overridable because
    // this fork's CUDA backend declines that shape outright:
    //   ggml_cuda_fattn_is_supported (fattn.cu:202) returns false unless
    //   K->ne[1] % FATTN_KQ_STRIDE == 0, and FATTN_KQ_STRIDE is 256 (fattn-common.cuh:16).
    // 47 % 256 != 0, so at the default the CUDA arm honestly prints "ordinary flash attention is
    // unsupported" and exits 2 without ever entering a kernel. That check is deliberate and is
    // documented in fattn.cu: the same condition is asserted inside three separate kernels, and
    // asserting it there KILLS the process (0xC0000409) on a shape ggml could have routed around.
    // So reporting unsupported is correct behaviour, not a defect -- but it also means the CUDA
    // arm measures nothing at the default. Pass a multiple of 256 to exercise the CUDA kernels:
    //   test-flash-attn-generic-hash cuda out.bin 256
    const int64_t nkv = argc > 3 ? std::atoll(argv[3]) : 47;

    ggml_backend_load_all();
    const enum ggml_backend_dev_type wanted = backend_kind == "cuda" ?
        GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_dev_t chosen = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == wanted) {
            chosen = dev;
            break;
        }
    }
    if (!chosen) {
        fprintf(stderr, "requested backend is unavailable: %s\n", backend_kind.c_str());
        return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(chosen, nullptr);
    if (!backend) {
        return 2;
    }

    constexpr int64_t d = 64;
    constexpr int64_t nq = 33;
    constexpr int64_t hq = 8;
    constexpr int64_t hkv = 2;
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return 2;
    }

    // FORK ADAPTATION, measured not guessed. Upstream builds this mask at exactly [nkv, nq]; on
    // this fork that aborts before producing a hash:
    //   ggml.c:11181: GGML_ASSERT(mask->ne[1] >= GGML_PAD(q->ne[1], GGML_KQ_MASK_PAD) && ...) failed
    // ik_llama keeps the padded-mask precondition upstream has since relaxed, and its pad is 64,
    // not upstream's 32 (ggml.h:2619). Padding is the correct fix rather than a workaround: the
    // extra rows are query slots that do not exist, they are filled with -INFINITY below, and the
    // kernel reads only the first q->ne[1] of them -- so the hash covers the same 33 real queries
    // either way and stays comparable across builds.
    constexpr int64_t nq_pad = GGML_PAD(nq, GGML_KQ_MASK_PAD);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, hq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nkv, hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nkv, hkv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, nq_pad, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/float(d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (!ggml_backend_supports_op(backend, out)) {
        fprintf(stderr, "ordinary flash attention is unsupported on %s\n",
                ggml_backend_dev_description(chosen));
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_f32(ggml_nelements(k));
    std::vector<float> v_f32(ggml_nelements(v));
    std::vector<ggml_fp16_t> k_data(k_f32.size());
    std::vector<ggml_fp16_t> v_data(v_f32.size());
    std::vector<ggml_fp16_t> mask(ggml_nelements(m));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.20f*std::sin(0.017f*float(i) + 0.11f);
    }
    for (size_t i = 0; i < k_f32.size(); ++i) {
        k_f32[i] = 0.23f*std::cos(0.013f*float(i) - 0.29f);
        v_f32[i] = 0.31f*std::sin(0.019f*float(i) + 0.71f);
    }
    ggml_fp32_to_fp16_row(k_f32.data(), k_data.data(), k_data.size());
    ggml_fp32_to_fp16_row(v_f32.data(), v_data.data(), v_data.size());
    for (int64_t iq = 0; iq < nq_pad; ++iq) {
        for (int64_t ik = 0; ik < nkv; ++ik) {
            // rows >= nq are padding for the fork's GGML_KQ_MASK_PAD precondition: no query lives
            // there, so every key is masked out.
            const int64_t dist = iq + (nkv - nq) - ik;
            const bool visible = iq < nq && dist >= 0 && dist < 29;
            mask[iq*nkv + ik] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(q_data[0]));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size()*sizeof(k_data[0]));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size()*sizeof(v_data[0]));
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(mask[0]));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph failed with status %d\n", int(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(result[0]));
    uint64_t fnv = UINT64_C(1469598103934665603);
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(result.data());
    for (size_t i = 0; i < result.size()*sizeof(result[0]); ++i) {
        fnv ^= bytes[i];
        fnv *= UINT64_C(1099511628211);
    }
    if (!output_path.empty()) {
        FILE * fp = fopen(output_path.c_str(), "wb");
        if (!fp || fwrite(result.data(), sizeof(result[0]), result.size(), fp) != result.size()) {
            fprintf(stderr, "failed to write %s\n", output_path.c_str());
            if (fp) {
                fclose(fp);
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
        fclose(fp);
    }
    // nkv is printed because it is now an argument: two hashes taken at different KV lengths are
    // different graphs, and a comparison that does not show it would silently compare nothing.
    printf("generic_hash backend=%s device=%s nkv=%lld bytes=%zu fnv1a64=%016llx\n",
           backend_kind.c_str(), ggml_backend_dev_description(chosen),
           (long long) nkv, result.size()*sizeof(result[0]), (unsigned long long) fnv);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
