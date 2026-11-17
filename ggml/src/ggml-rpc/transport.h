#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    // Per-socket wire counters (TCP and RDMA payload bytes handed to / taken from the transport) and the wall time spent
    // inside recv_data (waiting for the peer + copying). Atomic, read at any time. Row S69-ENGREV-DLLAMA-RPC-SYNC-TIME-AND-BYTES-COUNTERS-2026-11-06-01.
    uint64_t sent_bytes() const;
    uint64_t recv_bytes() const;
    uint64_t recv_wait_ns() const;
    // true when the TCP ack-frequency (Windows SIO_TCP_SET_ACK_FREQUENCY) / TCP_QUICKACK (Linux) option took effect on this socket.
    bool quickack_active() const;

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);
    const char * transport_name() const;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
