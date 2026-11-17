// CPU self-check of GGML_OP_FLASH_ATTN_EXT with kv_rows (ggml_flash_attn_ext_rows).
// Task fork-farows-kvrows-cpu-selfcheck-test.   GATE-CLAIM-OK: documentation only.
//
// Why this exists: test-backend-ops case test_flash_attn_ext_kv_rows compares a device against the CPU backend, but
// ggml_cuda_fattn_is_supported answers false for kv_rows nodes, so on CUDA0 no case runs ("0/0" summary, gate b1113c) and the CPU
// kernel is never compared with anything. This test needs no device: it executes the CPU kv_rows kernel and proves it against
//   (A) a REFERENCE GRAPH: per slice an explicit host-side gather of the listed K/V rows into a contiguous [hs, kv, nh, 1] tensor of the
//       same type, then the plain ggml_flash_attn_ext (no kv_rows) on the slice's q / mask. Same random inputs, so the gather logic is the
//       only difference. NMSE must be under 1e-6 (max abs diff printed).
//   (B) an independent double-precision softmax(QK^T*scale + mask)V computed here from the dequantized cache (NMSE under 5e-4, the
//       test-backend-ops tolerance; the kernel rounds Q to the K vec_dot type).
// Grid: n_slices {1,2,4,7}, mask per slice (with padded -1 list tails) and shared by all slices, K/V types F16, F32, Q8_0, Q4_0, Q4_1, Q5_0,
// Q5_1, IQ4_NL (a type the CPU backend declines for the node is counted SKIP and printed, never silently dropped), several hs / GQA ratios.
// NEGATIVE CONTROL: one entry of a visible row index is replaced in the kv_rows tensor only (reference keeps the original list); check (A)
// must then FAIL (NMSE over 1e-5). If the control does not fail the whole test fails: the check could not see a wrong row.
// Output: "kv_rows cpu self-check: N/N cases pass (N>0), control FAILED-as-required"; exit 0 only then.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct fa_case {
    int64_t   hs, nh, gqa, n_cells, kv, nb, n_slices;
    ggml_type type;
    bool      shared_mask;   // one mask for all slices (no padded list entries) vs one per slice
};

struct fa_result {
    bool   ran = false;       // false = op declined (SKIP)
    double nmse_ref = 0.0, maxabs_ref = 0.0;   // (A)
    double nmse_dbl = 0.0, maxabs_dbl = 0.0;   // (B)
};

static double nmse(const std::vector<float> & a, const std::vector<double> & b, double * maxabs) {
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        num += d*d;
        den += b[i]*b[i];
        mx = std::max(mx, std::fabs(d));
    }
    if (maxabs) *maxabs = mx;
    return den > 0.0 ? num/den : num;
}

// perturb: if >= 0, kv_rows[0][perturb] is replaced by a different cell in the kernel input only
static fa_result run_case(ggml_backend_t backend, const fa_case & c, int64_t perturb, int64_t * perturbed_to) {
    fa_result res;
    const int64_t nq_heads  = c.nh*c.gqa;
    const int64_t mask_rows = GGML_PAD(c.nb, GGML_KQ_MASK_PAD);
    const size_t  nodes     = 8 + 6*c.n_slices;

    ggml_init_params ip = { ggml_tensor_overhead()*(nodes + 16*c.n_slices + 16) + ggml_graph_overhead_custom(nodes + 16, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hs, c.nb, nq_heads, c.n_slices);
    ggml_tensor * k0 = ggml_new_tensor_3d(ctx, c.type, c.hs, c.nh, c.n_cells);
    ggml_tensor * v0 = ggml_new_tensor_3d(ctx, c.type, c.hs, c.nh, c.n_cells);
    ggml_tensor * k  = ggml_permute(ctx, k0, 0, 2, 1, 3);
    ggml_tensor * v  = ggml_permute(ctx, v0, 0, 2, 1, 3);
    ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.kv, mask_rows, 1, c.shared_mask ? 1 : c.n_slices);
    ggml_tensor * r  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.kv, c.n_slices);
    const float scale = 1.0f/sqrtf((float) c.hs);

    ggml_tensor * out = ggml_flash_attn_ext_rows(ctx, q, k, v, m, r, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return res;
    }

    // reference graph: one plain flash_attn_ext per slice on host-gathered K/V
    std::vector<ggml_tensor *> kg(c.n_slices), vg(c.n_slices), qs(c.n_slices), ms(c.n_slices), outs(c.n_slices);
    for (int64_t s = 0; s < c.n_slices; ++s) {
        kg[s] = ggml_new_tensor_4d(ctx, c.type, c.hs, c.kv, c.nh, 1);
        vg[s] = ggml_new_tensor_4d(ctx, c.type, c.hs, c.kv, c.nh, 1);
        qs[s] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hs, c.nb, nq_heads, 1);
        ms[s] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.kv, mask_rows, 1, 1);
        outs[s] = ggml_flash_attn_ext(ctx, qs[s], kg[s], vg[s], ms[s], scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(outs[s], GGML_PREC_F32);
        // kv_rows nodes never reach the iqk kernels (ops.cpp: kv_rows != nullptr -> generic kernel); a plain node would, and the two
        // differ by ~2e-6 NMSE in summation order (measured: hs=64/128 F16/Q8_0). Pin the reference to the generic kernel too, so (A)
        // isolates the row gather and nothing else.
        ggml_flash_attn_ext_set_n_swa(outs[s], GGML_FLASH_ATTN_EXT_IQK_DISABLED);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(0x6B76u + (unsigned) (c.hs*7 + c.kv*3 + c.n_slices*11 + (int) c.type*131 + (c.shared_mask ? 5 : 0)));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    // cache: random f32 -> stored type
    const size_t rowb = ggml_row_size(c.type, c.hs);
    const int64_t n_rows_total = c.nh*c.n_cells;
    auto make_cache = [&](std::vector<uint8_t> & st, std::vector<float> & deq) {
        std::vector<float> f((size_t) (n_rows_total*c.hs));
        for (auto & x : f) x = nd(rng);
        st.assign((size_t) n_rows_total*rowb, 0);
        deq.assign(f.size(), 0.0f);
        if (c.type == GGML_TYPE_F32) {
            memcpy(st.data(), f.data(), st.size());
            deq = f;
        } else if (c.type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) st.data(), f.size());
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) st.data(), deq.data(), f.size());
        } else {
            ggml_quantize_chunk(c.type, f.data(), st.data(), 0, n_rows_total, c.hs, nullptr);
            const ggml_type_traits * tt = ggml_get_type_traits(c.type);
            for (int64_t i = 0; i < n_rows_total; ++i) {
                tt->to_float(st.data() + (size_t) i*rowb, deq.data() + (size_t) i*c.hs, c.hs);
            }
        }
    };
    std::vector<uint8_t> kst, vst;
    std::vector<float>   kdq, vdq;
    make_cache(kst, kdq);
    make_cache(vst, vdq);
    // cache row for (head h, cell n) in layout [hs, nh, n_cells] is row index n*nh + h

    // per slice: distinct random cells, per-slice list tails padded with -1 (never in shared-mask mode)
    std::vector<int32_t> rows((size_t) (c.kv*c.n_slices), -1);
    std::vector<int32_t> order(c.n_cells);
    for (int64_t s = 0; s < c.n_slices; ++s) {
        for (int64_t i = 0; i < c.n_cells; ++i) order[i] = (int32_t) i;
        std::shuffle(order.begin(), order.end(), rng);
        const int64_t n_used = c.shared_mask ? c.kv : std::min<int64_t>(c.n_cells, c.kv - (s*37) % std::max<int64_t>(1, c.kv/2));
        for (int64_t i = 0; i < n_used; ++i) rows[s*c.kv + i] = order[i];
    }

    // mask: per slice (or one shared); -inf on padded entries and a deterministic drop pattern
    std::vector<float> mf((size_t) (c.kv*mask_rows*(c.shared_mask ? 1 : c.n_slices)), 0.0f);
    const int64_t n_mask_slices = c.shared_mask ? 1 : c.n_slices;
    for (int64_t s = 0; s < n_mask_slices; ++s) {
        for (int64_t j = 0; j < c.nb; ++j) {
            for (int64_t i = 0; i < c.kv; ++i) {
                const bool pad  = !c.shared_mask && rows[s*c.kv + i] < 0;
                const bool drop = (i + 3*j + 5*s) % 11 == 0;
                mf[(s*mask_rows + j)*c.kv + i] = (pad || drop) ? -INFINITY : ud(rng);
            }
        }
    }
    std::vector<ggml_fp16_t> mh(mf.size());
    ggml_fp32_to_fp16_row(mf.data(), mh.data(), mf.size());
    ggml_fp16_to_fp32_row(mh.data(), mf.data(), mf.size());   // the f16-rounded mask is what every path sees

    std::vector<float> qd((size_t) ggml_nelements(q));
    for (auto & x : qd) x = nd(rng);

    // kernel inputs
    std::vector<int32_t> rows_k = rows;
    if (perturb >= 0) {
        // replace visible entry `perturb` of slice 0 by a cell that is not in slice 0's list
        std::vector<bool> used(c.n_cells, false);
        for (int64_t i = 0; i < c.kv; ++i) if (rows[i] >= 0) used[rows[i]] = true;
        int32_t other = -1;
        for (int64_t n = 0; n < c.n_cells; ++n) if (!used[n]) { other = (int32_t) n; break; }
        if (other < 0) other = (rows[perturb] + 1) % (int32_t) c.n_cells;
        rows_k[perturb] = other;
        if (perturbed_to) *perturbed_to = other;
    }
    ggml_backend_tensor_set(q,  qd.data(), 0, qd.size()*sizeof(float));
    ggml_backend_tensor_set(k0, kst.data(), 0, kst.size());
    ggml_backend_tensor_set(v0, vst.data(), 0, vst.size());
    ggml_backend_tensor_set(m,  mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(r,  rows_k.data(), 0, rows_k.size()*sizeof(int32_t));

    // reference inputs: gather from the ORIGINAL list; padded entry -> cell 0 (masked with -inf in the mask)
    const size_t gsz = (size_t) (c.nh*c.kv)*rowb;
    for (int64_t s = 0; s < c.n_slices; ++s) {
        std::vector<uint8_t> gk(gsz), gv(gsz);
        for (int64_t h = 0; h < c.nh; ++h) {
            for (int64_t i = 0; i < c.kv; ++i) {
                const int32_t cell = std::max<int32_t>(rows[s*c.kv + i], 0);
                const size_t src = ((size_t) cell*c.nh + h)*rowb;
                const size_t dst = ((size_t) h*c.kv + i)*rowb;
                memcpy(gk.data() + dst, kst.data() + src, rowb);
                memcpy(gv.data() + dst, vst.data() + src, rowb);
            }
        }
        ggml_backend_tensor_set(kg[s], gk.data(), 0, gk.size());
        ggml_backend_tensor_set(vg[s], gv.data(), 0, gv.size());
        ggml_backend_tensor_set(qs[s], qd.data() + (size_t) s*c.hs*c.nb*nq_heads, 0, (size_t) c.hs*c.nb*nq_heads*sizeof(float));
        ggml_backend_tensor_set(ms[s], mh.data() + (size_t) (c.shared_mask ? 0 : s)*c.kv*mask_rows, 0, (size_t) c.kv*mask_rows*sizeof(ggml_fp16_t));
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, nodes + 16, false);
    ggml_build_forward_expand(gf, out);
    for (int64_t s = 0; s < c.n_slices; ++s) ggml_build_forward_expand(gf, outs[s]);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL graph compute failed\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        res.ran = true; res.nmse_ref = res.nmse_dbl = 1e30;
        return res;
    }

    // out layout [hs, nq_heads, nb, n_slices]
    const size_t slice_n = (size_t) c.hs*nq_heads*c.nb;
    std::vector<float> got(slice_n*c.n_slices), ref(got.size());
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
    for (int64_t s = 0; s < c.n_slices; ++s) {
        ggml_backend_tensor_get(outs[s], ref.data() + (size_t) s*slice_n, 0, slice_n*sizeof(float));
    }
    std::vector<double> refd(ref.begin(), ref.end());
    res.nmse_ref = nmse(got, refd, &res.maxabs_ref);

    // (B) double-precision softmax over the visible list entries; a fully masked row is not compared (kernel output undefined)
    std::vector<double> dbl(got.size(), 0.0);
    std::vector<float>  got_cmp;
    std::vector<double> dbl_cmp;
    std::vector<double> sc((size_t) c.kv);
    const int64_t rep = c.gqa;
    for (int64_t s = 0; s < c.n_slices; ++s) {
        for (int64_t j = 0; j < c.nb; ++j) {
            for (int64_t ih = 0; ih < nq_heads; ++ih) {
                const int64_t h = ih / rep;
                double mx = -INFINITY;
                for (int64_t i = 0; i < c.kv; ++i) {
                    const float mv = mf[((c.shared_mask ? 0 : s)*mask_rows + j)*c.kv + i];
                    if (!std::isfinite(mv) || rows[s*c.kv + i] < 0) { sc[i] = -INFINITY; continue; }
                    const float * kr = kdq.data() + ((size_t) rows[s*c.kv + i]*c.nh + h)*c.hs;
                    const float * qr = qd.data() + (((size_t) s*nq_heads + ih)*c.nb + j)*c.hs;
                    double dot = 0.0;
                    for (int64_t d = 0; d < c.hs; ++d) dot += (double) qr[d]*(double) kr[d];
                    sc[i] = dot*scale + (double) mv;
                    mx = std::max(mx, sc[i]);
                }
                if (!std::isfinite(mx)) continue;
                double den = 0.0;
                std::vector<double> acc(c.hs, 0.0);
                for (int64_t i = 0; i < c.kv; ++i) {
                    if (!std::isfinite(sc[i])) continue;
                    const double w = std::exp(sc[i] - mx);
                    den += w;
                    const float * vr = vdq.data() + ((size_t) rows[s*c.kv + i]*c.nh + h)*c.hs;
                    for (int64_t d = 0; d < c.hs; ++d) acc[d] += w*(double) vr[d];
                }
                const size_t base = (size_t) s*slice_n + ((size_t) j*nq_heads + ih)*c.hs;
                for (int64_t d = 0; d < c.hs; ++d) {
                    got_cmp.push_back(got[base + d]);
                    dbl_cmp.push_back(acc[d]/den);
                }
            }
        }
    }
    res.nmse_dbl = nmse(got_cmp, dbl_cmp, &res.maxabs_dbl);
    res.ran = true;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "no CPU backend\n"); return 2; }
    ggml_backend_cpu_set_n_threads(backend, argc > 1 ? atoi(argv[1]) : 4);

    const double TOL_REF = 1e-6, TOL_DBL = 5e-4, TOL_CTL = 1e-5;
    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL };
    const int64_t slices[] = { 1, 2, 4, 7 };
    struct shape { int64_t hs, nh, gqa, n_cells, kv, nb; };
    const shape shapes[] = { { 128, 2, 4, 700, 160, 4 }, { 64, 4, 1, 400, 96, 3 }, { 256, 2, 2, 300, 130, 8 } };

    int n_total = 0, n_pass = 0, n_skip = 0;
    double worst_ref = 0.0, worst_dbl = 0.0, worst_abs = 0.0;
    fa_case first_ok = {};
    bool have_first = false;
    for (ggml_type t : types) {
        for (const shape & sh : shapes) {
            if (sh.hs % ggml_blck_size(t) != 0) continue;
            for (int64_t ns : slices) {
                for (int sm = 0; sm < 2; ++sm) {
                    fa_case c = { sh.hs, sh.nh, sh.gqa, sh.n_cells, sh.kv, sh.nb, ns, t, sm == 1 };
                    fa_result r = run_case(backend, c, -1, nullptr);
                    const std::string nm = std::string(ggml_type_name(t)) + " hs=" + std::to_string(sh.hs) + " nh=" + std::to_string(sh.nh) +
                        " gqa=" + std::to_string(sh.gqa) + " kv=" + std::to_string(sh.kv) + " nb=" + std::to_string(sh.nb) +
                        " slices=" + std::to_string(ns) + (c.shared_mask ? " mask=shared" : " mask=per-slice");
                    if (!r.ran) {
                        ++n_skip;
                        printf("  SKIP %s (CPU declines the node)\n", nm.c_str());
                        continue;
                    }
                    ++n_total;
                    const bool ok = std::isfinite(r.nmse_ref) && r.nmse_ref < TOL_REF && std::isfinite(r.nmse_dbl) && r.nmse_dbl < TOL_DBL;
                    if (ok) {
                        ++n_pass;
                        if (!have_first) { first_ok = c; have_first = true; }
                    }
                    worst_ref = std::max(worst_ref, r.nmse_ref);
                    worst_dbl = std::max(worst_dbl, r.nmse_dbl);
                    worst_abs = std::max(worst_abs, r.maxabs_ref);
                    printf("  %s %-64s nmse_ref=%.3e maxabs_ref=%.3e nmse_dbl=%.3e\n", ok ? "ok  " : "FAIL", nm.c_str(), r.nmse_ref, r.maxabs_ref, r.nmse_dbl);
                }
            }
        }
    }

    // negative control: same inputs as the first passing case, one visible row index replaced in the kernel input only
    bool control_ok = false;
    double ctl_nmse = 0.0;
    if (have_first) {
        // entry 1 of slice 0 is visible for every case (drop pattern hits i=0 for j=0..; entry 1 is visible for most queries)
        int64_t to = -1;
        fa_result r = run_case(backend, first_ok, 1, &to);
        ctl_nmse = r.nmse_ref;
        control_ok = r.ran && r.nmse_ref > TOL_CTL;
        printf("control: kv_rows[0][1] -> cell %lld, nmse_ref=%.3e maxabs_ref=%.3e (must exceed %.0e): %s\n",
               (long long) to, r.nmse_ref, r.maxabs_ref, TOL_CTL, control_ok ? "FAILED-as-required" : "NOT-DETECTED");
    }

    printf("kv_rows cpu self-check: %d/%d cases pass (skipped %d), worst nmse_ref=%.3e (tol %.0e) maxabs_ref=%.3e worst nmse_dbl=%.3e (tol %.0e), control nmse=%.3e %s\n",
           n_pass, n_total, n_skip, worst_ref, TOL_REF, worst_abs, worst_dbl, TOL_DBL, ctl_nmse, control_ok ? "FAILED-as-required" : "NOT-DETECTED");
    ggml_backend_free(backend);
    if (n_total == 0) { fprintf(stderr, "FAIL: no case ran (vacuous)\n"); return 1; }
    if (n_pass != n_total || !control_ok) return 1;
    return 0;
}
