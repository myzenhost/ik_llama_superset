// A CPU op that consumes a quantized weight stored in an OpenVINO weight buffer.
// Row S69-FORK-OPENVINO-QUANT-WEIGHT-READBACK-BY-CPU-OP-GARBAGE-2026-10-27-01.
//
// GATE-CLAIM-OK: "CPU" here names the ggml CPU backend this test schedules beside OPENVINO0; nothing
// claims a device is unavailable.
//
// WHAT FAILED. The OpenVINO weight buffer rewrites a quantized weight into its own extracted layout
// (weights / scales / zero points, ggml-openvino/ggml-quants.cpp process_weight_tensor) inside the
// tensor's own storage. When the scheduler leaves an op that reads that weight on the CPU, it copies the
// weight out with ggml_backend_tensor_get (ggml-backend.cpp ggml_backend_tensor_copy), and the copy
// returned the extracted bytes -- the CPU kernel then multiplied garbage.
//
// WHAT THIS CHECKS, per quant type and per source kind:
//   A. ggml_backend_tensor_get of the whole weight, and of an unaligned sub-range, returns the exact
//      GGUF bytes that were uploaded (memcmp == 0).
//   B. A graph whose consumer OpenVINO declines -- FUSED_UP_GATE with a GELU gate
//      (ggml-openvino.cpp supports_op: only SILU/RELU are translated) -- is placed on the CPU by
//      ggml_backend_sched (asserted, so the test cannot silently measure an OpenVINO node), and its
//      output is bit-identical to the same graph with the weights in a CPU buffer.
//   C. No "reading back quantized weight" error is logged.
// Source kinds: "mmap" (weights uploaded from a read-only file mapping, as llama's default mmap load
// does) and "heap" (uploaded from ordinary memory, as --no-mmap does). Types whose extraction cannot be
// inverted exactly need the mapping or GGML_OPENVINO_KEEP_RAW_WEIGHTS=1; "--heap-keep-raw" runs every
// type from heap with that variable set.
//
// CONTROL: a deliberately corrupted reference byte must make check A report a mismatch, so a PASS is
// not a comparator that cannot fail.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#endif

static int g_readback_errors = 0;

static void log_cb(enum ggml_log_level level, const char * text, void * /*user*/) {
    if (level == GGML_LOG_LEVEL_ERROR && strstr(text, "reading back quantized weight") != nullptr) {
        g_readback_errors++;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// A read-only file mapping of `bytes`, standing in for llama_mmap.
struct file_map {
    void * addr = nullptr;
    size_t size = 0;
    std::string path;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#endif
    bool create(const std::string & p, const std::vector<uint8_t> & bytes) {
        path = p;
        size = bytes.size();
        FILE * f = fopen(p.c_str(), "wb");
        if (!f) {
            return false;
        }
        fwrite(bytes.data(), 1, bytes.size(), f);
        fclose(f);
#ifdef _WIN32
        file = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                           nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return false;
        }
        mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) {
            return false;
        }
        addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        return addr != nullptr;
#else
        int fd = open(p.c_str(), O_RDONLY);
        if (fd < 0) {
            return false;
        }
        void * m = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (m == MAP_FAILED) {
            return false;
        }
        addr = m;
        return true;
#endif
    }
    ~file_map() {
#ifdef _WIN32
        if (addr) {
            UnmapViewOfFile(addr);
        }
        if (mapping) {
            CloseHandle(mapping);
        }
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
        }
#else
        if (addr) {
            munmap(addr, size);
        }
#endif
        if (!path.empty()) {
            remove(path.c_str());
        }
    }
};

struct case_def {
    ggml_type type;
    const char * up_name;    // names steer the OpenVINO requant choice (token_embd/output -> Q8_0_C)
    const char * gate_name;
};

static std::vector<uint8_t> make_raw(ggml_type type, int64_t k, int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> f(k * n);
    for (auto & v : f) {
        v = dist(rng) * (1.0f + 0.5f * std::sin((float) (&v - f.data())));
    }
    std::vector<uint8_t> raw(ggml_row_size(type, k) * n);
    ggml_quantize_chunk(type, f.data(), raw.data(), 0, n, k, nullptr, nullptr);
    return raw;
}

// out = fused_up_gate(up, gate, x, GELU) on `sched` (or a CPU-only sched); returns output, sets on_cpu.
static bool run_graph(ggml_backend_sched_t sched, ggml_backend_t cpu, ggml_tensor * up, ggml_tensor * gate,
                      const std::vector<float> & xin, int64_t k, int64_t t, std::vector<float> & out,
                      bool & on_cpu) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, k, t);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * y = ggml_fused_up_gate(gctx, up, gate, x, GGML_UNARY_OP_GELU);
    ggml_set_name(y, "y");
    ggml_set_output(y);
    if (y->op != GGML_OP_FUSED_UP_GATE) {
        fprintf(stderr, "  ggml_fused_up_gate did not build a FUSED_UP_GATE node (op %s)\n", ggml_op_name(y->op));
        ggml_free(gctx);
        return false;
    }
    ggml_cgraph * g = ggml_new_graph(gctx);
    ggml_build_forward_expand(g, y);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, g)) {
        fprintf(stderr, "  sched alloc failed\n");
        ggml_free(gctx);
        return false;
    }
    ggml_backend_tensor_set(x, xin.data(), 0, xin.size() * sizeof(float));
    on_cpu = ggml_backend_sched_get_tensor_backend(sched, y) == cpu;
    if (ggml_backend_sched_graph_compute(sched, g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  compute failed\n");
        ggml_free(gctx);
        return false;
    }
    out.resize(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_free(gctx);
    return true;
}

static size_t count_diff(const uint8_t * a, const uint8_t * b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) {
        d += a[i] != b[i];
    }
    return d;
}

int main(int argc, char ** argv) {
    bool heap_keep_raw = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--heap-keep-raw") == 0) {
            heap_keep_raw = true;
        }
    }
    if (heap_keep_raw) {
#ifdef _WIN32
        _putenv_s("GGML_OPENVINO_KEEP_RAW_WEIGHTS", "1");
#else
        setenv("GGML_OPENVINO_KEEP_RAW_WEIGHTS", "1", 1);
#endif
    }
    ggml_log_set(log_cb, nullptr);

    ggml_backend_dev_t ov_dev = ggml_backend_dev_by_name("OPENVINO0");
    if (ov_dev == nullptr) {
        printf("SKIP: no OPENVINO0 device\n");
        return 0;
    }
    ggml_backend_t ov = ggml_backend_dev_init(ov_dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ov || !cpu) {
        printf("FAIL: backend init\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_type_t ov_buft = ggml_backend_dev_buffer_type(ov_dev);

    ggml_backend_t both[2] = { ov, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(both, nullptr, 2, 256, false, false);
    ggml_backend_t cpu_only[1] = { cpu };
    ggml_backend_sched_t sched_cpu = ggml_backend_sched_new(cpu_only, nullptr, 1, 256, false, false);

    const int64_t K = 512, N = 64, T = 3;
    const case_def cases[] = {
        { GGML_TYPE_Q8_0, "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight" },
        { GGML_TYPE_Q4_0, "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight" },
        { GGML_TYPE_Q4_1, "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight" },
        { GGML_TYPE_Q5_1, "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight" },
        { GGML_TYPE_Q4_K, "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight" },
        { GGML_TYPE_Q5_K, "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight" },
        { GGML_TYPE_Q6_K, "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight" },
        // requantized by the OpenVINO buffer (Q8_0 -> per-channel Q8_0_C): never invertible
        { GGML_TYPE_Q8_0, "output.weight", "token_embd.weight" },
    };
    // exact inversion is promised only for these layouts; the rest need a mapped source or a kept copy
    auto invertible = [](const case_def & c) {
        return (c.type == GGML_TYPE_Q8_0 || c.type == GGML_TYPE_Q4_0) && strncmp(c.up_name, "blk.", 4) == 0;
    };

    std::vector<float> xin(K * T);
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto & v : xin) {
            v = dist(rng);
        }
    }

    int n_fail = 0, n_pass = 0, n_skip = 0;
    bool control_done = false;
    for (const char * mode : { "mmap", "heap" }) {
        const bool from_map = strcmp(mode, "mmap") == 0;
        if (heap_keep_raw && from_map) {
            continue;
        }
        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
            const case_def & c = cases[ci];
            const std::string label = std::string(mode) + " " + ggml_type_name(c.type) + " " + c.up_name;
            if (!from_map && !heap_keep_raw && !invertible(c)) {
                printf("SKIP %-40s (heap source, non-invertible layout: run --heap-keep-raw)\n", label.c_str());
                n_skip++;
                continue;
            }
            const int errors_before = g_readback_errors;
            std::vector<uint8_t> raw_up = make_raw(c.type, K, N, 100 + (uint32_t) ci);
            std::vector<uint8_t> raw_gate = make_raw(c.type, K, N, 200 + (uint32_t) ci);
            std::vector<uint8_t> file_bytes(raw_up);
            file_bytes.insert(file_bytes.end(), raw_gate.begin(), raw_gate.end());

            file_map fm;
            const uint8_t * src_up = raw_up.data();
            const uint8_t * src_gate = raw_gate.data();
            if (from_map) {
                char p[64];
                snprintf(p, sizeof(p), "ov_readback_%d_%zu.bin", (int) ci, (size_t) file_bytes.size());
                if (!fm.create(p, file_bytes)) {
                    printf("FAIL %-40s could not map a temp file\n", label.c_str());
                    n_fail++;
                    continue;
                }
                src_up = (const uint8_t *) fm.addr;
                src_gate = (const uint8_t *) fm.addr + raw_up.size();
            }

            // the weights in an OpenVINO weight buffer, and the same bytes in a CPU buffer as the reference
            ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
            ggml_context * wctx = ggml_init(ip);
            ggml_context * rctx = ggml_init(ip);
            ggml_tensor * up = ggml_new_tensor_2d(wctx, c.type, K, N);
            ggml_tensor * gate = ggml_new_tensor_2d(wctx, c.type, K, N);
            ggml_set_name(up, c.up_name);
            ggml_set_name(gate, c.gate_name);
            ggml_tensor * rup = ggml_new_tensor_2d(rctx, c.type, K, N);
            ggml_tensor * rgate = ggml_new_tensor_2d(rctx, c.type, K, N);
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ov_buft);
            ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors_from_buft(rctx, ggml_backend_cpu_buffer_type());
            ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);  // before set, as llama does
            ggml_backend_tensor_set(up, src_up, 0, raw_up.size());
            ggml_backend_tensor_set(gate, src_gate, 0, raw_gate.size());
            ggml_backend_tensor_set(rup, raw_up.data(), 0, raw_up.size());
            ggml_backend_tensor_set(rgate, raw_gate.data(), 0, raw_gate.size());

            bool ok = true;
            // A: whole readback + an unaligned sub-range
            std::vector<uint8_t> back(raw_up.size());
            ggml_backend_tensor_get(up, back.data(), 0, back.size());
            const size_t d_whole = count_diff(back.data(), raw_up.data(), back.size());
            const size_t sub_off = ggml_type_size(c.type) * 3 + 5;
            const size_t sub_len = ggml_type_size(c.type) * 7 + 3;
            std::vector<uint8_t> sub(sub_len);
            ggml_backend_tensor_get(gate, sub.data(), sub_off, sub_len);
            const size_t d_sub = count_diff(sub.data(), raw_gate.data() + sub_off, sub_len);
            if (d_whole != 0 || d_sub != 0) {
                ok = false;
            }
            if (!control_done) {
                std::vector<uint8_t> bad(raw_up);
                bad[bad.size() / 2] ^= 0x5a;
                const size_t d_ctl = count_diff(back.data(), bad.data(), back.size());
                printf("CONTROL corrupted reference byte -> %zu byte(s) differ (must be >= 1 when readback is exact): %s\n",
                       d_ctl, d_ctl >= 1 ? "ok" : "COMPARATOR BROKEN");
                if (d_ctl < 1) {
                    n_fail++;
                }
                control_done = true;
            }

            // B: CPU consumer through the scheduler vs. CPU-only reference
            std::vector<float> y_mixed, y_ref;
            bool on_cpu = false, ref_on_cpu = false;
            const bool ran = run_graph(sched, cpu, up, gate, xin, K, T, y_mixed, on_cpu) &&
                             run_graph(sched_cpu, cpu, rup, rgate, xin, K, T, y_ref, ref_on_cpu);
            double max_abs = 0.0;
            size_t n_diff = 0;
            if (ran) {
                for (size_t i = 0; i < y_ref.size(); ++i) {
                    const double d = std::fabs((double) y_mixed[i] - (double) y_ref[i]);
                    if (!(d == 0.0)) {
                        n_diff++;
                    }
                    if (!(d <= max_abs)) {
                        max_abs = std::isnan(d) ? INFINITY : d;
                    }
                }
            }
            if (!ran || !on_cpu || n_diff != 0) {
                ok = false;
            }
            const int new_errors = g_readback_errors - errors_before;
            if (new_errors != 0) {
                ok = false;
            }
            printf("%s %-40s readback whole %zu/%zu bytes differ, sub-range %zu/%zu; consumer on %s; y %zu/%zu differ "
                   "(max |d| %.3g); readback errors logged %d\n",
                   ok ? "PASS" : "FAIL", label.c_str(), d_whole, back.size(), d_sub, sub_len,
                   ran ? (on_cpu ? "CPU" : "NOT-CPU") : "n/a", n_diff, y_ref.size(), max_abs, new_errors);
            ok ? n_pass++ : n_fail++;

            ggml_backend_sched_reset(sched);
            ggml_backend_sched_reset(sched_cpu);
            ggml_backend_buffer_free(wbuf);
            ggml_backend_buffer_free(rbuf);
            ggml_free(wctx);
            ggml_free(rctx);
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_sched_free(sched_cpu);
    ggml_backend_free(ov);
    ggml_backend_free(cpu);
    printf("RESULT: %d pass, %d fail, %d skip\n", n_pass, n_fail, n_skip);
    return n_fail == 0 ? 0 : 1;
}
