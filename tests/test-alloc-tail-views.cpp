// ggml_backend_alloc_ctx_tensors_from_buft() must initialise EVERY view in the context, also when the buffer type's max_size
// forces a split and the context ends in views only.
//
// WHY. The split loop (ggml/src/ggml-alloc.c, ggml_backend_alloc_ctx_tensors_from_buft_impl) cut a new range at ANY tensor once
// cur_buf_size + this_size > max_size -- views included, whose this_size is 0. A tensor larger than max_size followed by views
// therefore closed its range at the first view and opened a new one of size 0; "allocate remaining tensors" is skipped when that
// size is 0, so the views never got ggml_backend_view_init(): buffer == NULL, and ggml_backend_tensor_get() on them asserts
// "tensor not allocated". Upstream llama.cpp PR #25584 (MIT); carried by llamafile as
// llama.cpp.patches/patches/ggml_src_ggml-alloc.c.patch (llamafile a3ccf098; the symptom there is llama-server on Vulkan,
// max_size 1 GiB, dying on the second request at --parallel 4 when slot state save reads the KV stream views, upstream #29221).
//
// HOW. A buffer type that is the CPU one with get_max_size = 4096 bytes, so the split is reached on any host.
//   case A (the bug): t0 = 8 KiB f32 (over max_size, first in the context), then two views of t0.
//   case B (control, same as before the fix): t0 2 KiB, t1 2 KiB, t2 2 KiB (forces one split), views of t0 and t2 interleaved.
//   case C (control): no split at all (CPU max_size), views must also be initialised.
// For every case: every non-view tensor has data, every view has buffer != NULL and data == view_src->data + view_offs, and the
// bytes written through the parent read back through the view. Exit 0 pass, 1 fail.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void check(const char * what, bool ok) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

static size_t small_max_size(ggml_backend_buffer_type_t) { return 4096; }

static ggml_backend_buffer_type_t small_cpu_buft() {
    static ggml_backend_buffer_type buft;
    static bool init = false;
    if (!init) {
        ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
        buft = *cpu;
        buft.iface.get_max_size = small_max_size;
        init = true;
    }
    return &buft;
}

static void verify(const char * label, ggml_context * ctx, ggml_backend_buffer_t buf) {
    char what[256];
    std::snprintf(what, sizeof(what), "%s: a buffer was returned", label);
    check(what, buf != nullptr);
    if (!buf) return;
    int n_views = 0, n_views_ok = 0, n_plain = 0, n_plain_ok = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src) {
            ++n_views;
            const bool ok = t->buffer != nullptr && t->data != nullptr &&
                            (char *) t->data == (char *) t->view_src->data + t->view_offs;
            if (ok) ++n_views_ok;
            else std::printf("       view %s: buffer=%p data=%p\n", t->name, (void *) t->buffer, t->data);
        } else {
            ++n_plain;
            if (t->data != nullptr && t->buffer != nullptr) ++n_plain_ok;
        }
    }
    std::snprintf(what, sizeof(what), "%s: %d/%d plain tensors allocated", label, n_plain_ok, n_plain);
    check(what, n_plain_ok == n_plain);
    std::snprintf(what, sizeof(what), "%s: %d/%d views initialised (buffer set, data = parent + offset)", label, n_views_ok, n_views);
    check(what, n_views_ok == n_views);
    // round trip through every initialised view
    bool rt = true;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->view_src || !t->buffer) continue;
        std::vector<float> src(ggml_nelements(t->view_src));
        for (size_t i = 0; i < src.size(); ++i) src[i] = (float) i + 0.25f;
        ggml_backend_tensor_set(t->view_src, src.data(), 0, ggml_nbytes(t->view_src));
        std::vector<float> got(ggml_nelements(t));
        ggml_backend_tensor_get(t, got.data(), 0, ggml_nbytes(t));
        const size_t off = t->view_offs / sizeof(float);
        for (size_t i = 0; i < got.size(); ++i) if (got[i] != src[off + i]) { rt = false; break; }
    }
    std::snprintf(what, sizeof(what), "%s: bytes written to the parent read back through each initialised view", label);
    check(what, rt && n_views_ok > 0);
}

static ggml_context * new_ctx() {
    ggml_init_params ip = { /*mem_size*/ 64 * ggml_tensor_overhead(), /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    return ggml_init(ip);
}

int main() {
    ggml_backend_buffer_type_t sbuft = small_cpu_buft();
    check("control: the test buffer type reports max_size 4096", ggml_backend_buft_get_max_size(sbuft) == 4096);

    {   // case A: oversize tensor first, then only views
        ggml_context * ctx = new_ctx();
        ggml_tensor * t0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048); ggml_set_name(t0, "t0");           // 8 KiB > 4 KiB
        ggml_tensor * v0 = ggml_view_1d(ctx, t0, 256, 0);                ggml_set_name(v0, "t0.view0");
        ggml_tensor * v1 = ggml_view_1d(ctx, t0, 256, 1024 * sizeof(float)); ggml_set_name(v1, "t0.view1");
        (void) v0; (void) v1;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, sbuft);
        verify("A oversize tensor + tail views (split)", ctx, buf);
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    {   // case B: ordinary split with views interleaved
        ggml_context * ctx = new_ctx();
        ggml_tensor * t0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512); ggml_set_name(t0, "t0");             // 2 KiB
        ggml_tensor * v0 = ggml_view_1d(ctx, t0, 64, 64 * sizeof(float)); ggml_set_name(v0, "t0.view");
        ggml_tensor * t1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512); ggml_set_name(t1, "t1");
        ggml_tensor * t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512); ggml_set_name(t2, "t2");             // 3rd 2 KiB -> split
        ggml_tensor * v2 = ggml_view_1d(ctx, t2, 64, 0);                 ggml_set_name(v2, "t2.view");
        (void) v0; (void) t1; (void) v2;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, sbuft);
        verify("B split with interleaved views", ctx, buf);
        check("B: the split produced a multi-buffer", buf && ggml_backend_buffer_get_size(buf) >= 3 * 2048);
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    {   // case C: no split
        ggml_context * ctx = new_ctx();
        ggml_tensor * t0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048); ggml_set_name(t0, "t0");
        ggml_tensor * v0 = ggml_view_1d(ctx, t0, 256, 512 * sizeof(float)); ggml_set_name(v0, "t0.view");
        (void) v0;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        verify("C no split (CPU max_size)", ctx, buf);
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
