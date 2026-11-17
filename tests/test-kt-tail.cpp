// KT "tail" rows: IQ3_KT / IQ4_KT rows whose length is a multiple of 32 but not of QK_K (upstream ik_llama format).
//
// For each type and each row length K this quantises M rows with ggml_quantize_chunk, checks the byte count is exactly
// M*ggml_row_size, dequantises with the type's to_float and compares with the source:
//   * the tail region (the last K % QK_K values) must reconstruct about as well as the full-block head
//     (rel-RMS error at most TAIL_SLACK x the head's), and both must be well below 1 (signal present);
//   * NEGATIVE CONTROL: flipping the tail's scale nibbles/records to zero must make the tail error ~1 --
//     proves the check actually reads the tail bytes rather than, say, zeros matching zeros;
//   * K % QK_K == 0 is covered byte-exactly by test-quantize-fudge-factor (iq3_kt / iq4_kt hashes).

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr double TAIL_SLACK = 1.5;
static constexpr int    QKK = 256;

static double rel_rms(const float * a, const float * b, int n) {
    double e = 0, s = 0;
    for (int i = 0; i < n; ++i) { e += double(a[i] - b[i])*(a[i] - b[i]); s += double(a[i])*a[i]; }
    return s > 0 ? std::sqrt(e/s) : 0.0;
}

int main() {
    ggml_cpu_init();
    const ggml_type types[] = { GGML_TYPE_IQ3_KT, GGML_TYPE_IQ4_KT };
    const int Ks[] = { 2880, 992, 32 + QKK };
    const int M = 8;
    int fail = 0;
    for (ggml_type t : types) {
        const auto * tt = ggml_get_type_traits(t);
        for (int K : Ks) {
            const int nt = (K % QKK)/32, head = K - K % QKK;
            std::mt19937 rng(20260926 + K);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> x((size_t) K*M), y((size_t) K*M), imat((size_t) K, 1.0f);
            for (auto & v : x) v = nd(rng);
            const size_t rs = ggml_row_size(t, K);
            std::vector<uint8_t> q(rs*M + 64, 0xAB);
            const size_t got = ggml_quantize_chunk(t, x.data(), q.data(), 0, M, K, imat.data(), nullptr);
            const bool size_ok = got == rs*M && q[rs*M] == 0xAB;   // exact size, and nothing written past it
            for (int r = 0; r < M; ++r) tt->to_float(q.data() + r*rs, y.data() + (size_t) r*K, K);
            double eh = 0, et = 0;
            for (int r = 0; r < M; ++r) {
                eh += head ? rel_rms(x.data() + (size_t) r*K, y.data() + (size_t) r*K, head) : 0;
                et += rel_rms(x.data() + (size_t) r*K + head, y.data() + (size_t) r*K + head, K - head);
            }
            eh /= M; et /= M;
            // NEGATIVE CONTROL: zero the tail bytes of every row, dequantise again, the tail must lose its signal
            std::vector<uint8_t> qc(q);
            const size_t head_bytes = ggml_row_size(t, head);
            for (int r = 0; r < M; ++r) std::memset(qc.data() + r*rs + head_bytes, 0, rs - head_bytes);
            std::vector<float> yc((size_t) K*M);
            for (int r = 0; r < M; ++r) tt->to_float(qc.data() + r*rs, yc.data() + (size_t) r*K, K);
            double ec = 0;
            for (int r = 0; r < M; ++r) ec += rel_rms(x.data() + (size_t) r*K + head, yc.data() + (size_t) r*K + head, K - head);
            ec /= M;
            const bool ok = size_ok && et < 0.5 && (head == 0 || et <= TAIL_SLACK*eh) && ec > 0.9;
            printf("%-7s K=%5d nt=%d row_size=%zu bytes_ok=%d  head rel_rms %.4f  tail rel_rms %.4f  control(tail zeroed) %.4f  %s\n",
                   ggml_type_name(t), K, nt, rs, size_ok, eh, et, ec, ok ? "PASS" : "FAIL");
            fail += !ok;
        }
    }
    // CPU MUL_MAT with a KT tail weight: ggml_mul_mat(A[K x N], B[K x M]) against a double-precision product of the
    // DEQUANTISED A with B. The activation is quantised to q8 inside the kernel, so the bound is loose (MM_TOL); a tail
    // path that dropped or misread the last nt groups misses by ~ (K%QK_K)/K of the signal, far above it. K%QK_K==0 is the
    // baseline the same bound must also pass.
    const double MM_TOL = 0.03;
    const int Kmm[] = { 2880, 992, 1024 };
    const int Ms[] = { 1, 8 };
    const int N = 16;
    for (ggml_type t : types) {
        const auto * tt = ggml_get_type_traits(t);
        for (int K : Kmm) for (int M : Ms) {
            std::mt19937 rng(7 + K + M);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> a((size_t) K*N), b((size_t) K*M), ad((size_t) K*N), imat((size_t) K, 1.0f);
            for (auto & v : a) v = nd(rng);
            for (auto & v : b) v = nd(rng);
            const size_t rs = ggml_row_size(t, K);
            std::vector<uint8_t> q(rs*N);
            ggml_quantize_chunk(t, a.data(), q.data(), 0, N, K, imat.data(), nullptr);
            for (int r = 0; r < N; ++r) tt->to_float(q.data() + r*rs, ad.data() + (size_t) r*K, K);

            ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * A = ggml_new_tensor_2d(ctx, t, K, N);
            ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            std::memcpy(A->data, q.data(), rs*N);
            std::memcpy(B->data, b.data(), b.size()*sizeof(float));
            ggml_tensor * C = ggml_mul_mat(ctx, A, B);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, C);
            ggml_graph_compute_with_ctx(ctx, gf, 4);
            double e = 0, s = 0;
            for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int k = 0; k < K; ++k) ref += double(ad[(size_t) n*K + k])*b[(size_t) m*K + k];
                const double got = ((const float *) C->data)[(size_t) m*N + n];
                e += (got - ref)*(got - ref); s += ref*ref;
            }
            const double rel = std::sqrt(e/s);
            const bool ok = std::isfinite(rel) && rel < MM_TOL;
            printf("MUL_MAT %-7s K=%5d nt=%d M=%d  rel err vs dequant x f32 %.5f  %s\n", ggml_type_name(t), K, (K % QKK)/32, M, rel, ok ? "PASS" : "FAIL");
            fail += !ok;
            ggml_free(ctx);
        }
    }

    printf("%s -- %d failure(s)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
