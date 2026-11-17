// Regression test for S69-FORK-TEST-QUANTIZE-FNS-ABORTS-AT-TURBO3-HEAP-CORRUPTION-2026-10-07-01.
//
// ROOT CAUSE (reproduced with AddressSanitizer, tests/test-quantize-fns.cpp before the fix): dot_product_error() quantised the
// vec_dot operand into a scratch buffer of 2*n bytes ("2 bytes per element is enough for any type"). The turbo K types use
// vec_dot_type = F32, whose row is 4*n bytes: for n = 4096 the buffer was 8192 B and the write was 16384 B
// ("WRITE of size 16384 ... 0 bytes after 8192-byte region", ggml_cpu_fp32_to_fp32). The heap was corrupted, the allocator
// aborted a few allocations later (0xC0000374 on Windows), right after the turbo2 line.
//
// This test sizes the scratch buffers the way the fixed harness does (from the type traits) and checks, per turbo type:
//   * the scratch bytes the two sides NEED, derived from ggml_row_size(), against the buffer given;
//   * a canary region behind each buffer is untouched after from_float + vec_dot;
//   * the dot product is finite.
// `--old-size` gives the buffers the OLD size (2*n): the test must then FAIL (registered WILL_FAIL), which shows that it can
// detect the defect it guards. Under -fsanitize=address the old size also produces the report above.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const bool old_size = argc > 1 && std::string(argv[1]) == "--old-size";
    const size_t n = 32 * 128;
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; i++) { a[i] = 0.1f + 2 * cosf((float) i); b[i] = 0.1f + 2 * cosf((float) i + 1.0f); }

    int failed = 0;
    for (ggml_type t : { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 }) {
        const auto * cpu = ggml_get_type_traits_cpu(t);
        const ggml_type vt = cpu->vec_dot_type;
        const size_t need_x = ggml_row_size(t, n);
        const size_t need_y = ggml_row_size(vt, n);
        const size_t have_x = old_size ? 2 * n : std::max<size_t>(need_x, 2 * n);
        const size_t have_y = old_size ? 2 * n : std::max<size_t>(need_y, 2 * n);
        printf("%-7s vec_dot_type=%-4s need_x=%zu need_y=%zu have_x=%zu have_y=%zu\n", ggml_type_name(t), ggml_type_name(vt), need_x, need_y, have_x, have_y);
        if (have_x < need_x || have_y < need_y) {
            printf("FAIL %s: scratch buffer smaller than the row it receives (the 2*n sizing of the old harness)\n", ggml_type_name(t));
            failed++;
            continue; // writing would corrupt the heap: that is the defect, do not perform it
        }
        const size_t canary = 256;
        std::vector<uint8_t> qx(have_x + canary, 0xA5), qy(have_y + canary, 0xA5);
        cpu->from_float(a.data(), qx.data(), n);
        ggml_get_type_traits_cpu(vt)->from_float(b.data(), qy.data(), n);
        float r = INFINITY;
        cpu->vec_dot((int) n, &r, 0, qx.data(), 0, qy.data(), 0, 1);
        bool ok = std::isfinite(r);
        for (size_t i = have_x; i < qx.size(); i++) { ok = ok && qx[i] == 0xA5; }
        for (size_t i = have_y; i < qy.size(); i++) { ok = ok && qy[i] == 0xA5; }
        if (!ok) { printf("FAIL %s: canary overwritten or non-finite dot (%g)\n", ggml_type_name(t), r); failed++; }
    }
    printf("test-turbo-vecdot-scratch%s: %s\n", old_size ? " --old-size" : "", failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}
