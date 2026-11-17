// Model loader: registry devices the #if chain does not cover (HEXAGON, DL plugins, future RKNPU / XDNA) must be nameable
// with -dev, hold layers, get a backend, and coexist with the CPU path. Row S69-NPU-LOADER-EXTRA-BACKEND-ARM-REGISTRY-DEVICES-2026-11-06-01.
// Needs no NPU: a tiny STUB backend registered as device type GPU, name "STUB0", whose buffer type is its own ("STUB0")
// and whose compute delegates to the CPU backend.
//
// Needs a small GGUF: env LLAMA_TEST_TINY_GGUF (e.g. stories15M-q4_0.gguf); exits 77 (SKIPPED) without it.
//   BEFORE the loader arm: `-dev STUB0` logs "STUB0 backend not available." and the model has no STUB0 layers -> FAIL.
//   AFTER: STUB0 holds layers, logits equal the CPU-only run, and with the stub registered but not named the default
//   device selection and the buffer placement are exactly what they were without it.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------- stub backend (delegates to CPU) ----------------------
static ggml_backend_dev_t   g_cpu_dev  = nullptr;
static ggml_backend_dev_t   g_stub_dev = nullptr;
static ggml_backend_buffer_type_t g_stub_buft = nullptr;
static ggml_backend_reg_t   g_stub_reg = nullptr;
static ggml_guid g_stub_guid = { 0x53, 0x54, 0x55, 0x42, 0x30, 0x53, 0x54, 0x55, 0x42, 0x30, 0x53, 0x54, 0x55, 0x42, 0x30, 0x01 };

static const char * stub_buft_name(ggml_backend_buffer_type_t) { return "STUB0"; }
static ggml_backend_buffer_t stub_buft_alloc(ggml_backend_buffer_type_t, size_t size) {
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_type()->iface.alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (buf != nullptr) {
        buf->buft = g_stub_buft;   // the buffer reports its buffer type as STUB0, as a real device's would
    }
    return buf;
}

static const char * stub_dev_name(ggml_backend_dev_t)        { return "STUB0"; }
static const char * stub_dev_desc(ggml_backend_dev_t)        { return "test stub (CPU-backed)"; }
static void stub_dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) { ggml_backend_dev_memory(g_cpu_dev, free, total); }
static enum ggml_backend_dev_type stub_dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void stub_dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * p) {
    p->name = stub_dev_name(dev); p->description = stub_dev_desc(dev); p->type = stub_dev_type(dev);
    stub_dev_memory(dev, &p->memory_free, &p->memory_total);
    p->caps = { false, false, false, false, true };
}

static const char * stub_be_name(ggml_backend_t) { return "STUB0"; }
static void stub_be_free(ggml_backend_t b) {
    ggml_backend_free((ggml_backend_t) b->context);
    delete b;
}
static enum ggml_status stub_be_compute(ggml_backend_t b, struct ggml_cgraph * g) {
    return ggml_backend_graph_compute((ggml_backend_t) b->context, g);
}
static ggml_backend_t stub_dev_init(ggml_backend_dev_t dev, const char *) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_i iface = {};
    iface.get_name      = stub_be_name;
    iface.free          = stub_be_free;
    iface.graph_compute = stub_be_compute;
    return new ggml_backend { &g_stub_guid, iface, dev, cpu };
}
static ggml_backend_buffer_type_t stub_dev_buft(ggml_backend_dev_t) { return g_stub_buft; }
static bool stub_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) { return ggml_backend_dev_supports_op(g_cpu_dev, op); }
static bool stub_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == g_stub_buft || ggml_backend_buft_is_host(buft);
}

static const char * stub_reg_name(ggml_backend_reg_t) { return "STUB"; }
static size_t stub_reg_count(ggml_backend_reg_t)      { return 1; }
static ggml_backend_dev_t stub_reg_get(ggml_backend_reg_t, size_t) { return g_stub_dev; }
static void * stub_reg_proc(ggml_backend_reg_t, const char *) { return nullptr; }

static void register_stub() {
    g_cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    g_stub_buft = new ggml_backend_buffer_type(*ggml_backend_cpu_buffer_type());
    g_stub_buft->iface.get_name     = stub_buft_name;
    g_stub_buft->iface.alloc_buffer = stub_buft_alloc;
    g_stub_reg = new ggml_backend_reg { GGML_BACKEND_API_VERSION, { stub_reg_name, stub_reg_count, stub_reg_get, stub_reg_proc }, nullptr };
    ggml_backend_device_i di = {};
    di.get_name = stub_dev_name; di.get_description = stub_dev_desc; di.get_memory = stub_dev_memory; di.get_type = stub_dev_type;
    di.get_props = stub_dev_props; di.init_backend = stub_dev_init; di.get_buffer_type = stub_dev_buft;
    di.supports_op = stub_dev_supports_op; di.supports_buft = stub_dev_supports_buft;
    g_stub_dev = new ggml_backend_device { di, g_stub_reg, nullptr };
    g_stub_buft->device = g_stub_dev;
    ggml_backend_register(g_stub_reg);
}

// ---------------------------------------------------------------- harness -----------------------------------------------
static std::vector<std::string> g_log;
static void log_cb(ggml_log_level, const char * text, void *) { g_log.push_back(text); }

static bool log_has(const char * needle) {
    for (const auto & l : g_log) {
        if (l.find(needle) != std::string::npos) return true;
    }
    return false;
}
// lines naming a buffer or a device, with the numbers removed so two loads compare textually
static std::vector<std::string> placement_lines() {
    std::vector<std::string> out;
    for (const auto & l : g_log) {
        if (l.find("buffer size") != std::string::npos || l.find("using device") != std::string::npos) {
            std::string s;
            for (char c : l) if (!(c >= '0' && c <= '9') && c != '.') s.push_back(c);
            out.push_back(s);
        }
    }
    return out;
}
static void print_placement() {
    for (const auto & l : g_log) {
        if (l.find("buffer size") != std::string::npos || l.find("using device") != std::string::npos ||
            l.find("backend not available") != std::string::npos || l.find("offloaded") != std::string::npos) {
            fputs(l.c_str(), stdout);
            if (l.empty() || l.back() != '\n') fputc('\n', stdout);
        }
    }
}

struct run_result { bool ok = false; std::vector<float> logits; };

static run_result run(const char * path, const char * dev_spec, int ngl) {
    run_result r;
    g_log.clear();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.device_spec  = dev_spec;
    mp.split_mode   = LLAMA_SPLIT_MODE_LAYER;
    mp.use_mmap     = false;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) { fprintf(stderr, "model load failed (spec=%s)\n", dev_spec ? dev_spec : "<none>"); return r; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) { fprintf(stderr, "context init failed (spec=%s)\n", dev_spec ? dev_spec : "<none>"); llama_model_free(model); return r; }
    std::vector<llama_token> toks(32);
    const std::string prompt = "Once upon a time, there was a little girl named Lily.";
    int n = llama_tokenize(model, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, false);
    if (n <= 0) { llama_free(ctx); llama_model_free(model); return r; }
    toks.resize(n);
    llama_batch b = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, b) != 0) { llama_free(ctx); llama_model_free(model); return r; }
    const float * lg = llama_get_logits_ith(ctx, n - 1);
    r.logits.assign(lg, lg + llama_n_vocab(model));
    r.ok = true;
    llama_free(ctx);
    llama_model_free(model);
    return r;
}

#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

static double max_diff(const std::vector<float> & a, const std::vector<float> & b, int * am_a, int * am_b) {
    double d = 0; *am_a = 0; *am_b = 0;
    for (size_t i = 0; i < a.size(); i++) {
        d = std::fmax(d, std::fabs(a[i] - b[i]));
        if (a[i] > a[*am_a]) *am_a = (int) i;
        if (b[i] > b[*am_b]) *am_b = (int) i;
    }
    return d;
}

int main() {
    const char * path = getenv("LLAMA_TEST_TINY_GGUF");
    if (path == nullptr || !*path) { printf("SKIP: LLAMA_TEST_TINY_GGUF not set\n"); return 77; }
    llama_backend_init();
    llama_log_set(log_cb, nullptr);

    // 0. baselines before the stub exists: CPU-only logits, and the default placement of an -ngl 99 load
    run_result cpu_only = run(path, nullptr, 0);
    CHECK(cpu_only.ok, "cpu-only run");
    run_result base_default = run(path, nullptr, 99);
    CHECK(base_default.ok, "default run");
    const std::vector<std::string> base_lines = placement_lines();

    register_stub();
    CHECK(ggml_backend_dev_by_name("STUB0") != nullptr, "stub did not register");

    // 1. stub registered but NOT named: default device selection and placement are unchanged
    run_result with_stub_unnamed = run(path, nullptr, 99);
    CHECK(with_stub_unnamed.ok, "run with stub registered, not named");
    CHECK(!log_has("STUB0"), "an unnamed extra device took part in the default selection");
    CHECK(placement_lines() == base_lines, "default placement changed when an extra device is merely registered");
    printf("[1] stub registered, not named: placement lines identical to the run without it (%zu lines)\n", base_lines.size());

    // 2. -dev STUB0
    run_result only_stub = run(path, "STUB0", 99);
    print_placement();
    CHECK(!log_has("STUB0 backend not available"), "STUB0 reported as not available");
    CHECK(only_stub.ok, "run -dev STUB0");
    CHECK(log_has("STUB0 model buffer size") || log_has("STUB0 buffer size"), "no STUB0 buffer in the placement log");
    int a1, b1;
    double d1 = max_diff(cpu_only.logits, only_stub.logits, &a1, &b1);
    printf("[2] -dev STUB0: max|logit diff vs CPU-only| = %g, argmax %d vs %d\n", d1, a1, b1);
    CHECK(a1 == b1 && d1 < 2e-3, "logits differ from CPU-only");

    // 3. CPU + STUB0 layer split: both hold layers
    run_result split = run(path, "CPU,STUB0", 99);
    print_placement();
    CHECK(split.ok, "run -dev CPU,STUB0");
    CHECK(log_has("STUB0 model buffer size") || log_has("STUB0 buffer size"), "no STUB0 buffer in the CPU,STUB0 split");
    int a2, b2;
    double d2 = max_diff(cpu_only.logits, split.logits, &a2, &b2);
    printf("[3] -dev CPU,STUB0: max|logit diff vs CPU-only| = %g, argmax %d vs %d\n", d2, a2, b2);
    CHECK(a2 == b2 && d2 < 2e-3, "split logits differ from CPU-only");

    printf("PASS test-loader-extra-backend\n");
    fflush(stdout);
    _Exit(0);
}
