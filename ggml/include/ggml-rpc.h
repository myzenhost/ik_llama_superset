#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// FORK PROTOCOL IDENTITY. The wire format serializes ggml_op enum VALUES, and this fork's op table is not
// upstream's (upstream pins GGML_OP_COUNT == 101; this fork appends its own ops). A fork peer and an upstream
// peer would therefore decode each other's graphs into the WRONG ops, silently. So the fork speaks major
// 128 + upstream's major (5 -> 133): negotiate_hello() rejects any major mismatch in BOTH directions, which
// turns that silent mismatch into a refused connection. Fork-to-fork RPC is unaffected.
// The op-count guard is KEPT, pinned to the fork's own table: adding an op must still bump
// RPC_PROTO_PATCH_VERSION (and GGML_RPC_FORK_OP_COUNT) -- only silencing it would remove the protection.
// FORK_WIRE_REV counts fork-only wire-layout changes that the upstream major does not see; each one adds 32 to
// the major so an older fork peer is refused instead of decoding the new struct wrongly. Rev 1: rpc_tensor ne/nb
// widened 32 -> 64 bit (ik_llama #2360). The major is a uint8 on the wire: 128 + 5 + 32*rev must stay <= 255.
#define RPC_PROTO_UPSTREAM_MAJOR_VERSION 5
#define RPC_PROTO_FORK_WIRE_REV    1
#define RPC_PROTO_MAJOR_VERSION    (128 + RPC_PROTO_UPSTREAM_MAJOR_VERSION + 32 * RPC_PROTO_FORK_WIRE_REV)
#if RPC_PROTO_MAJOR_VERSION > 255
#error "RPC_PROTO_MAJOR_VERSION no longer fits the uint8 wire field"
#endif
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_SUPPORTS_OP 5  // first patch whose server answers RPC_CMD_SUPPORTS_OP; the client gates the query on it
#define RPC_PROTO_PATCH_VERSION    7   // 7: GGML_OP_COL2IM_3D appended (2026-11-15); 6: GGML_OP_FMA appended (2026-11-10, ik #2794); 5: RPC_CMD_SUPPORTS_OP added (2026-11-06; additive, old clients unaffected); 4: GGML_OP_TOP_K_MASK appended (2026-11-03, upstream merge); 3: GGML_OP_GATED_GROUP_RMS_NORM (2026-10-16)
#define GGML_RPC_FORK_OP_COUNT     140

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == GGML_RPC_FORK_OP_COUNT,
              "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION and GGML_RPC_FORK_OP_COUNT");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

// client-link byte meter of the live connection to endpoint (sent to / received from it); false when no connection is open
GGML_BACKEND_API bool ggml_backend_rpc_endpoint_bytes(const char * endpoint, uint64_t * sent, uint64_t * recv);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
