// RPC client supports_op must ask the remote backend (RPC_CMD_SUPPORTS_OP) and cache the answer, so an op the remote
// cannot run stays on the local CPU instead of being shipped to the server (which refuses the whole graph).
// Row S69-NPU-RPC-CLIENT-SUPPORTS-OP-QUERY-REMOTE-2026-11-06-01. Needs no NPU: an in-process rpc server on the CPU backend
// whose backend refuses one op through the test-only hook GGML_RPC_TEST_UNSUPPORTED_OP.
//
//   BEFORE (client supports_op always true): the MUL is scheduled on the RPC backend, the server refuses the graph, the
//   connection closes and the client aborts ("Remote RPC server crashed or returned malformed response").
//   AFTER: MUL is placed on the CPU backend, the result equals the host reference, other ops still use RPC.
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

// y = w*x + w with w in the RPC buffer; returns 0 on success. `mul_on_cpu` is the placement the caller expects for MUL.
static int run_case(ggml_backend_t rpc, ggml_backend_t cpu, int64_t n, bool expect_mul_on_cpu, bool compute = true) {
    std::vector<float> wv(n * 4), xv(n * 4), ref(n * 4);
    for (size_t i = 0; i < wv.size(); i++) {
        wv[i] = 0.25f * (float) ((i % 7) + 1);
        xv[i] = 0.5f * (float) ((i % 5) + 1) - 1.0f;
        ref[i] = wv[i] * xv[i] + wv[i];
    }

    ggml_init_params wp = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n, 4);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, rpc);
    CHECK(wbuf != nullptr, "alloc weights on rpc");
    ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));

    ggml_init_params gp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n, 4);
    ggml_set_input(x);
    ggml_tensor * m = ggml_mul(gctx, w, x);
    ggml_tensor * y = ggml_add(gctx, m, w);
    ggml_set_output(y);
    ggml_cgraph * g = ggml_new_graph_custom(gctx, 16, false);
    ggml_build_forward_expand(g, y);

    ggml_backend_t backs[2] = { rpc, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs, nullptr, 2, 16, false, false);
    CHECK(ggml_backend_sched_alloc_graph(sched, g), "sched alloc");
    // placement is decided at alloc time; check it before computing so a wrong placement is reported as such
    ggml_backend_t mb = ggml_backend_sched_get_tensor_backend(sched, m);
    printf("n=%lld: MUL placed on %s, ADD placed on %s\n", (long long) n, ggml_backend_name(mb),
           ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, y)));
    CHECK((mb == cpu) == expect_mul_on_cpu, "MUL on %s, expected %s", ggml_backend_name(mb), expect_mul_on_cpu ? "CPU" : "RPC");
    if (!compute) {
        ggml_backend_sched_free(sched);
        return 0;
    }
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    CHECK(ggml_backend_sched_graph_compute(sched, g) == GGML_STATUS_SUCCESS, "compute");

    std::vector<float> out(n * 4);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    double maxerr = 0;
    for (size_t i = 0; i < out.size(); i++) {
        maxerr = std::fmax(maxerr, std::fabs(out[i] - ref[i]));
    }
    printf("n=%lld: max |rpc-sched result - host reference| = %g\n", (long long) n, maxerr);
    CHECK(maxerr < 1e-6, "result differs from reference");

    ggml_backend_sched_free(sched);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return 0;
}

int main() {
    ggml_backend_load_all();
    const int port = 41000 + (int) (std::chrono::steady_clock::now().time_since_epoch().count() % 8000);
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);

    set_env("GGML_RPC_TEST_UNSUPPORTED_OP", "MUL");

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu_dev != nullptr, "no cpu device");
    std::thread([endpoint, cpu_dev]() {
        ggml_backend_dev_t devs[1] = { cpu_dev };
        ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 2, 1, devs);
    }).detach();
    for (int i = 0; i < 400 && !port_open(port); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    CHECK(port_open(port), "rpc server did not come up on %s", endpoint.c_str());

    ggml_backend_t rpc = ggml_backend_rpc_init(endpoint.c_str(), 0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(rpc && cpu, "backend init");

    // 1. the server's backend refuses MUL: it must stay on the client CPU and the output must still be right
    if (int rc = run_case(rpc, cpu, 8, /*expect_mul_on_cpu=*/true)) return rc;
    // 2. same signature again: answered from the client cache (no second query is observable here; placement must not change)
    if (int rc = run_case(rpc, cpu, 8, true)) return rc;
    // 3. refusal off, new signature (other shape): MUL is offered to the remote again
    set_env("GGML_RPC_TEST_UNSUPPORTED_OP", "");
    if (int rc = run_case(rpc, cpu, 16, /*expect_mul_on_cpu=*/false)) return rc;

    // 4. an OLD server (announces patch 4, does not know RPC_CMD_SUPPORTS_OP): the client must not send the command (the
    //    server would drop the connection) and keeps the pre-query answer "supported", so the MUL goes to the remote.
    {
        set_env("GGML_RPC_TEST_UNSUPPORTED_OP", "MUL");
        set_env("GGML_RPC_TEST_ANNOUNCE_PATCH", "4");
        const int port2 = port + 1;
        const std::string endpoint2 = "127.0.0.1:" + std::to_string(port2);
        std::thread([endpoint2, cpu_dev]() {
            ggml_backend_dev_t devs[1] = { cpu_dev };
            ggml_backend_rpc_start_server(endpoint2.c_str(), nullptr, 2, 1, devs);
        }).detach();
        for (int i = 0; i < 400 && !port_open(port2); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        CHECK(port_open(port2), "second rpc server did not come up");
        ggml_backend_t rpc2 = ggml_backend_rpc_init(endpoint2.c_str(), 0);
        CHECK(rpc2 != nullptr, "backend init 2");
        // the env stays set: the client reconnects (the sockets are weak-cached) and every HELLO must keep announcing 4
        if (int rc = run_case(rpc2, cpu, 8, /*expect_mul_on_cpu=*/false, /*compute=*/false)) return rc;
        set_env("GGML_RPC_TEST_ANNOUNCE_PATCH", "");
    }

    printf("PASS test-rpc-supports-op\n");
    fflush(stdout);
    _Exit(0); // the detached server thread never returns
}
