// test-col2im-3d.cpp: validate GGML_OP_COL2IM_3D, the scatter-add adjoint of GGML_OP_IM2COL_3D.
//
// Two independent checks per geometry, both on the CPU backend:
//   1. col2im_3d(c) against a naive double precision scatter written straight from the
//      definition, on random columns.
//   2. The adjoint identity  <im2col_3d(x), c> == <x, col2im_3d(c)>  for random x and c. It ties
//      the new op to the existing forward op's own indexing, so a layout slip that the naive
//      reference shares with the kernel would still show up.
// The geometries cover overlapping windows (stride < kernel), dilation, padding, stride larger
// than the kernel (gaps), N > 1, IC > 1 and the 1x1x1 degenerate window. Thread counts 1 and 4
// must agree bit for bit: the kernel splits over image planes, so no cell has two writers.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct c3d_case {
    int64_t IW, IH, ID, N, IC, KW, KH, KD;
    int s0, s1, s2, p0, p1, p2, d0, d1, d2;
};

static const c3d_case CASES[] = {
    //  IW IH ID N IC KW KH KD   s0 s1 s2  p0 p1 p2  d0 d1 d2
    {   5, 4, 3, 1, 1, 3, 3, 3,   1, 1, 1,  0, 0, 0,  1, 1, 1 },  // dense overlap, no padding
    {   6, 5, 4, 2, 3, 3, 2, 2,   1, 1, 1,  1, 1, 1,  1, 1, 1 },  // N=2, IC=3, padding
    {   7, 6, 5, 1, 2, 3, 3, 3,   2, 2, 2,  1, 1, 1,  1, 1, 1 },  // stride 2 with overlap
    {   8, 7, 6, 2, 2, 2, 2, 2,   3, 3, 3,  0, 0, 0,  1, 1, 1 },  // stride > kernel, gaps stay zero
    {   9, 8, 7, 1, 2, 3, 3, 3,   1, 2, 1,  2, 1, 2,  2, 1, 2 },  // dilation, mixed strides and padding
    {   4, 4, 4, 1, 1, 1, 1, 1,   1, 1, 1,  0, 0, 0,  1, 1, 1 },  // 1x1x1 window = identity scatter
    {   3, 2, 2, 3, 1, 3, 3, 3,   1, 1, 1,  1, 1, 1,  1, 1, 1 },  // window larger than the image
    {  10, 3, 5, 1, 4, 4, 2, 3,   2, 1, 2,  2, 0, 1,  1, 2, 1 },  // anisotropic everything
};

static uint64_t g_rng = 0x9e3779b97f4a7c15ULL;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

static int64_t out_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins + 2*p - d*(ks - 1) - 1)/s + 1;
}

// Test-local IM2COL_3D builder: the library has no public one yet. Layout ground-truthed against
// ggml_compute_forward_im2col_3d_f32 (ggml-cpu/ops.cpp): op_params = {s0,s1,s2,p0,p1,p2,d0,d1,d2,IC},
// dst ne = {IC*KD*KH*KW, OW, OH, N*OD}. The kernel tensor only supplies its extents.
static struct ggml_tensor * im2col_3d(struct ggml_context * ctx, struct ggml_tensor * kern,
        struct ggml_tensor * img, const c3d_case & c) {
    const int64_t N  = img->ne[3]/c.IC;
    const int64_t OD = out_size(img->ne[2], kern->ne[2], c.s2, c.p2, c.d2);
    const int64_t OH = out_size(img->ne[1], kern->ne[1], c.s1, c.p1, c.d1);
    const int64_t OW = out_size(img->ne[0], kern->ne[0], c.s0, c.p0, c.d0);
    const int64_t ne[4] = { c.IC*kern->ne[2]*kern->ne[1]*kern->ne[0], OW, OH, N*OD };
    struct ggml_tensor * r = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    const int32_t params[10] = { c.s0, c.s1, c.s2, c.p0, c.p1, c.p2, c.d0, c.d1, c.d2, (int32_t) c.IC };
    memcpy(r->op_params, params, sizeof(params));
    r->op     = GGML_OP_IM2COL_3D;
    r->src[0] = kern;
    r->src[1] = img;
    return r;
}

// Naive scatter from the definition, double accumulator.
static std::vector<double> naive_col2im(const float * cols, const c3d_case & c,
        int64_t OW, int64_t OH, int64_t OD) {
    std::vector<double> out((size_t)(c.IW*c.IH*c.ID*c.N*c.IC), 0.0);
    const int64_t KDKHKW = c.KD*c.KH*c.KW;
    for (int64_t n = 0; n < c.N; n++)
    for (int64_t od = 0; od < OD; od++)
    for (int64_t oh = 0; oh < OH; oh++)
    for (int64_t ow = 0; ow < OW; ow++)
    for (int64_t ic = 0; ic < c.IC; ic++)
    for (int64_t kd = 0; kd < c.KD; kd++)
    for (int64_t kh = 0; kh < c.KH; kh++)
    for (int64_t kw = 0; kw < c.KW; kw++) {
        const int64_t iw = ow*c.s0 + kw*c.d0 - c.p0;
        const int64_t ih = oh*c.s1 + kh*c.d1 - c.p1;
        const int64_t id = od*c.s2 + kd*c.d2 - c.p2;
        if (iw < 0 || iw >= c.IW || ih < 0 || ih >= c.IH || id < 0 || id >= c.ID) {
            continue;
        }
        const int64_t row = ((n*OD + od)*OH + oh)*OW + ow;
        const int64_t col = ic*KDKHKW + kd*c.KH*c.KW + kh*c.KW + kw;
        const int64_t dst = (((n*c.IC + ic)*c.ID + id)*c.IH + ih)*c.IW + iw;
        out[dst] += cols[row*(c.IC*KDKHKW) + col];
    }
    return out;
}

static std::vector<float> run_col2im(const c3d_case & c, const std::vector<float> & cols_data, int nth) {
    struct ggml_init_params ip = { (size_t) 256 << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    const int64_t OW = out_size(c.IW, c.KW, c.s0, c.p0, c.d0);
    const int64_t OH = out_size(c.IH, c.KH, c.s1, c.p1, c.d1);
    const int64_t OD = out_size(c.ID, c.KD, c.s2, c.p2, c.d2);
    struct ggml_tensor * cols = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.IC*c.KD*c.KH*c.KW, OW, OH, c.N*OD);
    memcpy(cols->data, cols_data.data(), cols_data.size()*sizeof(float));
    struct ggml_tensor * y = ggml_col2im_3d(ctx, cols, c.IW, c.IH, c.ID, c.N, c.IC, c.KW, c.KH, c.KD,
            c.s0, c.s1, c.s2, c.p0, c.p1, c.p2, c.d0, c.d1, c.d2);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, nth);
    std::vector<float> out((float *) y->data, (float *) y->data + ggml_nelements(y));
    ggml_free(ctx);
    return out;
}

int main(void) {
    int fails = 0;

    for (const c3d_case & c : CASES) {
        const int64_t OW = out_size(c.IW, c.KW, c.s0, c.p0, c.d0);
        const int64_t OH = out_size(c.IH, c.KH, c.s1, c.p1, c.d1);
        const int64_t OD = out_size(c.ID, c.KD, c.s2, c.p2, c.d2);
        const int64_t ncols = c.IC*c.KD*c.KH*c.KW*OW*OH*c.N*OD;
        const int64_t nimg  = c.IW*c.IH*c.ID*c.N*c.IC;

        std::vector<float> cols_data(ncols);
        for (float & v : cols_data) { v = frand(); }

        // Check 1: naive reference, 1 and 4 threads
        const std::vector<double> ref = naive_col2im(cols_data.data(), c, OW, OH, OD);
        const std::vector<float>  y1  = run_col2im(c, cols_data, 1);
        const std::vector<float>  y4  = run_col2im(c, cols_data, 4);

        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < nimg; i++) {
            num += ((double) y1[i] - ref[i])*((double) y1[i] - ref[i]);
            den += ref[i]*ref[i];
        }
        const double nmse = den > 0.0 ? num/den : num;
        const bool same_threads = memcmp(y1.data(), y4.data(), nimg*sizeof(float)) == 0;

        // Check 2: adjoint identity against the existing im2col_3d kernel
        std::vector<float> x(nimg);
        for (float & v : x) { v = frand(); }
        double lhs = 0.0, rhs = 0.0;
        {
            struct ggml_init_params ip = { (size_t) 256 << 20, NULL, false };
            struct ggml_context * ctx = ggml_init(ip);
            struct ggml_tensor * kern = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.KW, c.KH, c.KD, c.IC);
            struct ggml_tensor * img  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.IW, c.IH, c.ID, c.N*c.IC);
            memcpy(img->data, x.data(), nimg*sizeof(float));
            struct ggml_tensor * ic = im2col_3d(ctx, kern, img, c);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, ic);
            ggml_graph_compute_with_ctx(ctx, gf, 2);
            GGML_ASSERT(ggml_nelements(ic) == ncols);
            const float * icd = (const float *) ic->data;
            for (int64_t i = 0; i < ncols; i++) { lhs += (double) icd[i]*cols_data[i]; }
            ggml_free(ctx);
        }
        for (int64_t i = 0; i < nimg; i++) { rhs += (double) x[i]*y1[i]; }
        const double adj_err = std::fabs(lhs - rhs)/(std::fabs(lhs) + 1e-9);

        const bool ok = nmse <= 1e-10 && same_threads && adj_err <= 1e-4;
        if (!ok) { fails++; }
        printf("col2im_3d img=%lldx%lldx%lld N=%lld IC=%lld k=%lldx%lldx%lld s=%d,%d,%d p=%d,%d,%d d=%d,%d,%d: "
               "nmse=%.2e threads_equal=%d adjoint_rel_err=%.2e %s\n",
               (long long) c.IW, (long long) c.IH, (long long) c.ID, (long long) c.N, (long long) c.IC,
               (long long) c.KW, (long long) c.KH, (long long) c.KD,
               c.s0, c.s1, c.s2, c.p0, c.p1, c.p2, c.d0, c.d1, c.d2,
               nmse, same_threads ? 1 : 0, adj_err, ok ? "OK" : "FAIL");
    }

    printf(fails == 0 ? "all col2im_3d checks passed\n" : "%d col2im_3d checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
