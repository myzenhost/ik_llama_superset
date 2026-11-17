// RPC TCP transport (ggml/src/ggml-rpc/transport.cpp), three rows of engine-merge bundle fork-engrev-ports-02:
//   S69-ENGREV-DLLAMA-RPC-SERVER-LISTEN-BACKLOG-2026-11-06-01   : listen() queue length was 1, so a burst of simultaneous
//                                                                  connects to one rpc-server loses all but the first.
//   S69-ENGREV-DLLAMA-RPC-SYNC-TIME-AND-BYTES-COUNTERS-2026-11-06-01 : per-socket sent/recv byte counters and recv wait time.
//   S69-ENGREV-DLLAMA-RPC-TCP-TURBO-QUICKACK-2026-11-06-01      : ack-frequency/TCP_QUICKACK is applied; GGML_RPC_TCP_SPIN=1
//                                                                  (bounded poll-spin) moves the same bytes unchanged.
// Internal symbols (socket_t): static-library build only. Loopback only, no model, no GPU.
#include "ggml-rpc/transport.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET raw_fd_t;
static void close_fd(raw_fd_t s) { closesocket(s); }
static void set_nonblock(raw_fd_t s) { u_long m = 1; ioctlsocket(s, FIONBIO, &m); }
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int raw_fd_t;
static void close_fd(raw_fd_t s) { close(s); }
static void set_nonblock(raw_fd_t s) { fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK); }
#endif

#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// N simultaneous non-blocking connects to a listening socket nobody accepts from; returns how many completed within wait_ms.
static int burst_connect(int port, int n, int wait_ms) {
    std::vector<raw_fd_t> fds;
    for (int i = 0; i < n; i++) {
        raw_fd_t s = socket(AF_INET, SOCK_STREAM, 0);
        set_nonblock(s);
        sockaddr_in a = {};
        a.sin_family      = AF_INET;
        a.sin_port        = htons((unsigned short) port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        (void) connect(s, (sockaddr *) &a, sizeof(a)); // completes asynchronously; result read below
        fds.push_back(s);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    int ok = 0;
    for (raw_fd_t s : fds) {
        fd_set w, e;
        FD_ZERO(&w);
        FD_ZERO(&e);
        FD_SET(s, &w);
        FD_SET(s, &e);
        timeval tv = { 0, 0 };
        int r = select((int) s + 1, nullptr, &w, &e, &tv);
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(s, SOL_SOCKET, SO_ERROR, (char *) &err, &len);
        if (r > 0 && FD_ISSET(s, &w) && !FD_ISSET(s, &e) && err == 0) {
            ok++;
        }
    }
    for (raw_fd_t s : fds) {
        close_fd(s);
    }
    return ok;
}

// client sends `n` bytes, server echoes back the first `echo` bytes; returns 0 on success.
static int run_transfer(int port, size_t n, size_t echo, bool check_counters, bool expect_quickack) {
    socket_ptr srv = socket_t::create_server("127.0.0.1", port);
    CHECK(srv != nullptr, "create_server %d", port);

    std::vector<uint8_t> payload(n), got(n), back(echo);
    for (size_t i = 0; i < n; i++) {
        payload[i] = (uint8_t) (i * 131u + 7u);
    }
    std::atomic<bool> srv_ok{false};
    std::atomic<uint64_t> srv_recv{0}, srv_sent{0};
    std::thread th([&] {
        socket_ptr c = srv->accept();
        if (!c) return;
        if (!c->recv_data(got.data(), n)) return;
        if (!c->send_data(got.data(), echo)) return;
        srv_recv = c->recv_bytes();
        srv_sent = c->sent_bytes();
        srv_ok   = true;
    });
    socket_ptr cl = socket_t::connect("127.0.0.1", port);
    CHECK(cl != nullptr, "connect");
    CHECK(cl->send_data(payload.data(), n), "client send");
    CHECK(cl->recv_data(back.data(), echo), "client recv");
    th.join();
    CHECK(srv_ok.load(), "server side failed");
    CHECK(memcmp(got.data(), payload.data(), n) == 0, "payload corrupted on the wire");
    CHECK(memcmp(back.data(), payload.data(), echo) == 0, "echo corrupted");

    if (check_counters) {
        CHECK(cl->sent_bytes() == n, "client sent_bytes=%llu want %zu", (unsigned long long) cl->sent_bytes(), n);
        CHECK(cl->recv_bytes() == echo, "client recv_bytes=%llu want %zu", (unsigned long long) cl->recv_bytes(), echo);
        CHECK(srv_recv.load() == n, "server recv_bytes=%llu want %zu", (unsigned long long) srv_recv.load(), n);
        CHECK(srv_sent.load() == echo, "server sent_bytes=%llu want %zu", (unsigned long long) srv_sent.load(), echo);
        CHECK(cl->sent_bytes() == srv_recv.load(), "client sent != server recv");
        CHECK(cl->recv_wait_ns() > 0, "recv_wait_ns not accumulated");
    }
    if (expect_quickack) {
        CHECK(cl->quickack_active(), "ack-frequency/QUICKACK not applied on the client socket");
    }
    return 0;
}

int main() {
    CHECK(rpc_transport_init(), "rpc_transport_init");

    // 1. listen queue: 16 simultaneous connects against an idle listener must all complete (a queue of 1 drops/stalls most).
    {
        socket_ptr srv = socket_t::create_server("127.0.0.1", 50861);
        CHECK(srv != nullptr, "create_server for the listen-queue case");
        int ok = burst_connect(50861, 16, 400);
        printf("listen-queue burst: %d/16 connects completed in 400 ms\n", ok);
        CHECK(ok == 16, "only %d/16 simultaneous connects completed (listen queue too small)", ok);
    }

    // 2. byte counters + recv wait + QUICKACK/ack-frequency, default (blocking) mode.
    set_env("GGML_RPC_TCP_SPIN", "");
    if (run_transfer(50862, 3 * 1000 * 1000 + 17, 4099, true, true)) return 1;
    printf("counters + quickack: ok\n");

    // 3. spin mode moves the same bytes (identical payload with and without the flag, at the transport level).
    set_env("GGML_RPC_TCP_SPIN", "1");
    if (run_transfer(50863, 3 * 1000 * 1000 + 17, 4099, true, true)) return 1;
    printf("spin mode: ok\n");
    set_env("GGML_RPC_TCP_SPIN", "");

    rpc_transport_shutdown();
    printf("PASS\n");
    return 0;
}
