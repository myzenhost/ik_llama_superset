// Three RPC correctness guards ported from upstream ggml-org/llama.cpp (row S69-ENGREV-UPSTREAM-MERGE-RPC-V8-FEATURE-PORT-2026-11-03-01).
// Two in-process rpc servers on the CPU backend, no model, no GPU:
//   floor : #29283 (66963a8bc) get_alloc_size never returns less than ggml_nbytes. GGML_RPC_TEST_ALLOC_SIZE_SHRINK makes the server
//           answer 1 byte (a peer that truncated a stride); the client must still report >= ggml_nbytes, fresh and from its cache.
//   free  : #24292 (60199339b) free_buffer drops the stored graphs. A graph reused through GRAPH_RECOMPUTE after the buffer it
//           points into was freed must MISS on the server (client logs "server had no stored graph" and resends), not run the
//           stale graph through dangling pointers. The second graph computes a different op, so a stale run gives a wrong result.
//   owner : #26500 (a7cc83bba) a graph sent to server B must not carry the remote pointers of a tensor that lives on server A.
//           BEFORE: B cannot resolve A's buffer, drops the connection, the client aborts. AFTER: the tensor is sent host-side.
//   hop   : client-link bytes of a cross-server tensor copy (see case_hop); not a guard, the baseline for a direct server hop.
// usage: test-rpc-guards [floor|free|owner|hop]   (no argument runs all four)
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static bool port_open(int port) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    auto s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short) port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bool ok = connect(s, (sockaddr *) &a, sizeof(a)) == 0;
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return ok;
}

#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

static int g_miss_warnings = 0;

static void log_cb(ggml_log_level, const char * text, void *) {
    if (text != nullptr && strstr(text, "server had no stored graph") != nullptr) {
        g_miss_warnings++;
    }
}

static bool start_server(int port, ggml_backend_dev_t dev) {
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    std::thread([endpoint, dev]() {
        ggml_backend_dev_t devs[1] = { dev };
        ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 2, 1, devs);
    }).detach();
    for (int i = 0; i < 400 && !port_open(port); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return port_open(port);
}

static int case_floor(const std::string & endpoint) {
    set_env("GGML_RPC_TEST_ALLOC_SIZE_SHRINK", "1");
    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint.c_str(), 0);
    CHECK(buft != nullptr, "rpc buffer type");

    ggml_init_params p = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);

    // op that asks the server (MUL_MAT_ID): not cacheable, so this is the fresh-answer return
    ggml_tensor * fresh = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    fresh->op = GGML_OP_MUL_MAT_ID;
    // quantized weight with ne[0] % 512 != 0: asks the server and IS cacheable; ask twice so the second is the cache return
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 96, 8);

    size_t f1 = ggml_backend_buft_get_alloc_size(buft, fresh);
    size_t w1 = ggml_backend_buft_get_alloc_size(buft, weight);
    size_t w2 = ggml_backend_buft_get_alloc_size(buft, weight); // from the client cache
    printf("floor: fresh %zu (nbytes %zu), weight %zu then cached %zu (nbytes %zu)\n", f1, ggml_nbytes(fresh), w1, w2, ggml_nbytes(weight));
    CHECK(f1 >= ggml_nbytes(fresh), "fresh answer %zu below ggml_nbytes %zu", f1, ggml_nbytes(fresh));
    CHECK(w1 >= ggml_nbytes(weight), "weight answer %zu below ggml_nbytes %zu", w1, ggml_nbytes(weight));
    CHECK(w2 >= ggml_nbytes(weight), "cached weight answer %zu below ggml_nbytes %zu", w2, ggml_nbytes(weight));
    ggml_free(ctx);
    set_env("GGML_RPC_TEST_ALLOC_SIZE_SHRINK", "");
    return 0;
}

// build y = a <op> b over fresh tensors in their own RPC buffer; uid is forced equal so the client takes the RECOMPUTE path
struct case_graph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph * g = nullptr;
    ggml_tensor * a = nullptr, * b = nullptr, * y = nullptr;
};

static int build(case_graph & c, ggml_backend_buffer_type_t buft, bool use_mul, const std::vector<float> & av, const std::vector<float> & bv) {
    ggml_init_params p = { 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true };
    c.ctx = ggml_init(p);
    const int64_t n = (int64_t) av.size();
    c.a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n);
    c.b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n);
    ggml_set_input(c.a);
    ggml_set_input(c.b);
    c.y = use_mul ? ggml_mul(c.ctx, c.a, c.b) : ggml_add(c.ctx, c.a, c.b);
    ggml_set_output(c.y);
    c.g = ggml_new_graph_custom(c.ctx, 16, false);
    ggml_build_forward_expand(c.g, c.y);
    c.g->uid = 7;
    c.galloc = ggml_gallocr_new(buft);
    CHECK(ggml_gallocr_alloc_graph(c.galloc, c.g), "gallocr alloc");
    ggml_backend_tensor_set(c.a, av.data(), 0, ggml_nbytes(c.a));
    ggml_backend_tensor_set(c.b, bv.data(), 0, ggml_nbytes(c.b));
    return 0;
}

static int case_free(const std::string & endpoint) {
    ggml_backend_t rpc = ggml_backend_rpc_init(endpoint.c_str(), 0);
    CHECK(rpc != nullptr, "rpc init");
    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint.c_str(), 0);
    const int n = 256;
    std::vector<float> av(n), bv(n);
    for (int i = 0; i < n; i++) { av[i] = 0.5f + (float) (i % 7); bv[i] = 1.5f + (float) (i % 5); }

    case_graph c1;
    if (int rc = build(c1, buft, /*use_mul=*/false, av, bv)) return rc;
    CHECK(ggml_backend_graph_compute(rpc, c1.g) == GGML_STATUS_SUCCESS, "compute 1");
    CHECK(ggml_backend_graph_compute(rpc, c1.g) == GGML_STATUS_SUCCESS, "compute 2 (recompute hit)");
    CHECK(g_miss_warnings == 0, "a recompute with the buffer alive must hit, got %d miss warnings", g_miss_warnings);

    // free the buffer the stored graph points into
    ggml_gallocr_free(c1.galloc);
    ggml_free(c1.ctx);

    // same uid, different graph (MUL, not ADD) in a new buffer: the client asks for RECOMPUTE; the server must MISS
    case_graph c2;
    if (int rc = build(c2, buft, /*use_mul=*/true, av, bv)) return rc;
    CHECK(ggml_backend_graph_compute(rpc, c2.g) == GGML_STATUS_SUCCESS, "compute 3");
    std::vector<float> out(n);
    ggml_backend_tensor_get(c2.y, out.data(), 0, ggml_nbytes(c2.y));
    double maxerr = 0;
    for (int i = 0; i < n; i++) {
        maxerr = std::fmax(maxerr, std::fabs(out[i] - av[i] * bv[i]));
    }
    printf("free: miss warnings after the free = %d, max |result - a*b| = %g\n", g_miss_warnings, maxerr);
    CHECK(g_miss_warnings == 1, "expected exactly 1 server miss after free_buffer, got %d", g_miss_warnings);
    CHECK(maxerr < 1e-6, "result is not a*b: the stale graph ran");
    ggml_gallocr_free(c2.galloc);
    ggml_free(c2.ctx);
    return 0;
}

static int case_owner(const std::string & endpoint_a, const std::string & endpoint_b) {
    ggml_backend_t ba = ggml_backend_rpc_init(endpoint_a.c_str(), 0);
    ggml_backend_t bb = ggml_backend_rpc_init(endpoint_b.c_str(), 0);
    CHECK(ba && bb, "rpc init");
    ggml_init_params p = { 3 * ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, ba);
    CHECK(buf != nullptr, "alloc on A");
    // a remote pointer allocated by server A means nothing to server B
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 1, false);
    g->nodes[0] = t;
    g->n_nodes = 1;
    CHECK(ggml_backend_graph_compute(bb, g) == GGML_STATUS_SUCCESS, "compute on B");
    size_t free_mem, total_mem; // round trip to B: it has finished the graph (or dropped the connection) by now
    ggml_backend_rpc_get_device_memory(endpoint_b.c_str(), 0, &free_mem, &total_mem);
    CHECK(total_mem > 0, "server B no longer answers");
    printf("owner: server B computed a graph holding a tensor of server A and still answers\n");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

// hop : client-link bytes of a tensor copy. Same-server copy is one COPY_TENSOR command (no payload through the client); a copy
//       between buffers of two servers falls back to get_tensor from A + set_tensor to B, so the activation crosses the client
//       link twice. This is the measured baseline a direct server-to-server hop would remove
//       (row S69-PETALS-PICK-RPC-SERVER-TO-SERVER-DIRECT-HOP-2026-10-11-01).
static int case_hop(const std::string & endpoint_a, const std::string & endpoint_b) {
    ggml_backend_t ba = ggml_backend_rpc_init(endpoint_a.c_str(), 0);
    ggml_backend_t bb = ggml_backend_rpc_init(endpoint_b.c_str(), 0);
    CHECK(ba && bb, "rpc init");
    const int64_t n = 64 * 1024; // 256 KiB of f32, one activation
    ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ca = ggml_init(p);
    ggml_context * cb = ggml_init(p);
    ggml_tensor * a1 = ggml_new_tensor_1d(ca, GGML_TYPE_F32, n);
    ggml_tensor * a2 = ggml_new_tensor_1d(ca, GGML_TYPE_F32, n);
    ggml_tensor * b1 = ggml_new_tensor_1d(cb, GGML_TYPE_F32, n);
    ggml_backend_buffer_t bufa = ggml_backend_alloc_ctx_tensors(ca, ba);
    ggml_backend_buffer_t bufb = ggml_backend_alloc_ctx_tensors(cb, bb);
    CHECK(bufa && bufb, "alloc");
    const size_t nbytes = ggml_nbytes(a1);
    std::vector<float> data(n), back(n);
    for (int64_t i = 0; i < n; i++) { data[i] = (float) (i % 251) * 0.25f; }
    ggml_backend_tensor_set(a1, data.data(), 0, nbytes);

    uint64_t sa0 = 0, ra0 = 0, sb0 = 0, rb0 = 0, sa1 = 0, ra1 = 0, sb1 = 0, rb1 = 0;
    CHECK(ggml_backend_rpc_endpoint_bytes(endpoint_a.c_str(), &sa0, &ra0), "no live connection to A");
    CHECK(ggml_backend_rpc_endpoint_bytes(endpoint_b.c_str(), &sb0, &rb0), "no live connection to B");

    // control: both tensors on A, the copy stays on the server
    ggml_backend_tensor_copy(a1, a2);
    CHECK(ggml_backend_rpc_endpoint_bytes(endpoint_a.c_str(), &sa1, &ra1), "A meter");
    const uint64_t same_client_bytes = (sa1 - sa0) + (ra1 - ra0);
    sa0 = sa1; ra0 = ra1;

    // cross-server copy A -> B
    ggml_backend_tensor_copy(a1, b1);
    CHECK(ggml_backend_rpc_endpoint_bytes(endpoint_a.c_str(), &sa1, &ra1), "A meter 2");
    CHECK(ggml_backend_rpc_endpoint_bytes(endpoint_b.c_str(), &sb1, &rb1), "B meter 2");
    const uint64_t a_to_client = ra1 - ra0; // server A -> client
    const uint64_t client_to_b = sb1 - sb0; // client -> server B
    printf("hop: tensor %zu bytes; same-server copy client bytes %llu; cross-server copy: A->client %llu, client->B %llu (client link total %llu = %.2fx tensor)\n",
           nbytes, (unsigned long long) same_client_bytes, (unsigned long long) a_to_client, (unsigned long long) client_to_b,
           (unsigned long long) (a_to_client + client_to_b), (double) (a_to_client + client_to_b) / (double) nbytes);
    CHECK(same_client_bytes < nbytes / 16, "same-server copy moved %llu client bytes, expected only the command", (unsigned long long) same_client_bytes);
    CHECK(a_to_client >= nbytes && client_to_b >= nbytes, "cross-server copy must cross the client twice (%llu, %llu < %zu)",
          (unsigned long long) a_to_client, (unsigned long long) client_to_b, nbytes);
    // the copy is correct: the hop is a traffic question, not a result question
    ggml_backend_tensor_get(b1, back.data(), 0, nbytes);
    CHECK(memcmp(back.data(), data.data(), nbytes) == 0, "copied activation differs");

    ggml_backend_buffer_free(bufa);
    ggml_backend_buffer_free(bufb);
    ggml_free(ca);
    ggml_free(cb);
    return 0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_log_set(log_cb, nullptr);
    const std::string which = argc > 1 ? argv[1] : "all";
    const int port = 42000 + (int) (std::chrono::steady_clock::now().time_since_epoch().count() % 8000);
    const std::string ea = "127.0.0.1:" + std::to_string(port);
    const std::string eb = "127.0.0.1:" + std::to_string(port + 1);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu_dev != nullptr, "no cpu device");
    CHECK(start_server(port, cpu_dev), "server A did not come up");
    CHECK(start_server(port + 1, cpu_dev), "server B did not come up");

    int rc = 0;
    if (which == "all" || which == "floor") rc |= case_floor(ea);
    if (rc == 0 && (which == "all" || which == "free")) rc |= case_free(ea);
    if (rc == 0 && (which == "all" || which == "owner")) rc |= case_owner(ea, eb);
    if (rc == 0 && (which == "all" || which == "hop")) rc |= case_hop(ea, eb);
    if (rc == 0) {
        printf("PASS test-rpc-guards (%s)\n", which.c_str());
    }
    fflush(stdout);
    fflush(stderr);
    _Exit(rc); // the detached server threads never return
}
