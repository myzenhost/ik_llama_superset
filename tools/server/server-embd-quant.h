// Binary / int8 quantised output for /v1/embeddings and /embedding.
//
// Header-only and free of llama / ggml / json dependencies, so it is unit-tested standalone
// (tools/server/tests_cpp/test_server_embd_quant.cpp) and end to end by tools/server/tests/unit/test_embedding_quant.py.
//
// Why: a vector database that stores 1 bit per dimension (Hamming distance) or 1 byte per dimension needs the server to
// hand it the packed form; shipping 32-bit floats and packing client side costs 32x (binary) or 4x (int8) the bytes on
// the wire and an extra pass in every client.
//
// Order of operations (the part that is easy to get wrong): the vector is pooled and normalised in float FIRST, cut to
// its Matryoshka head SECOND, and only then packed. Packing before normalising would quantise a vector whose scale
// carries no meaning; packing before the cut would pack bits that the client then throws away.
//
// SOURCE-READ: vespa-engine/vespa@26cb1df (Apache-2.0), model-integration/src/main/java/ai/vespa/embedding/HuggingFaceEmbedder.java
// :210-229 (binaryQuantization: pool + normalise at 8*targetDims in float, then pack, "cannot pack" error when 8*target >
// model output) and vespajlib/src/main/java/com/yahoo/tensor/Tensors.java:58-102 (packBits: a bit is 1 when the value is
// strictly positive, the FIRST value of each group of 8 is the HIGHEST bit, the byte is stored as a signed int8).
// EmbeddingQuantization.java:21-100 (quantization names float | int8 | binary, "binary requires dim % 8 == 0 and the packed
// dimension is dim / 8"). Description-level port, no code copied; the int8 scaling below is this server's own design (Vespa's
// int8 mode at that commit is a pass-through of a provider's already-quantised output, there is no float -> int8 step to read).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

enum server_embd_quant {
    SERVER_EMBD_QUANT_NONE = 0,   // float32 (or base64 of float32): the existing behaviour
    SERVER_EMBD_QUANT_BINARY,     // packed sign bits as signed int8 (-128..127), Vespa pack_bits layout
    SERVER_EMBD_QUANT_UBINARY,    // the same bytes as unsigned (0..255), Cohere's name for it
    SERVER_EMBD_QUANT_INT8,       // round(v * 127) clamped to -127..127
};

// "binary" | "ubinary" | "int8" -> the mode; anything else (including "float" and "base64") -> NONE. The caller has already
// decided which strings are legal; this only maps the quantised ones.
static inline server_embd_quant server_embd_quant_from_name(const std::string & name) {
    if (name == "binary")  { return SERVER_EMBD_QUANT_BINARY;  }
    if (name == "ubinary") { return SERVER_EMBD_QUANT_UBINARY; }
    if (name == "int8")    { return SERVER_EMBD_QUANT_INT8;    }
    return SERVER_EMBD_QUANT_NONE;
}

// binary modes pack 8 dimensions per byte, so the unpacked dimension must be a positive multiple of 8
static inline bool server_embd_quant_dim_ok(server_embd_quant q, int64_t n_dim) {
    if (q == SERVER_EMBD_QUANT_BINARY || q == SERVER_EMBD_QUANT_UBINARY) {
        return n_dim > 0 && n_dim % 8 == 0;
    }
    return n_dim > 0;
}

// Pack sign bits: bit = 1 iff v > 0 (zero and negative-zero and NaN are 0), first value of each group of 8 -> bit 7.
// Packs the first n_dim values only (the Matryoshka head); n_dim must be a multiple of 8 and <= v.size() (checked by the
// caller through server_embd_quant_dim_ok; a violation here returns an empty vector rather than reading out of range).
static inline std::vector<uint8_t> server_embd_pack_bits(const std::vector<float> & v, size_t n_dim) {
    std::vector<uint8_t> out;
    if (n_dim == 0 || n_dim % 8 != 0 || n_dim > v.size()) {
        return out;
    }
    out.assign(n_dim / 8, 0);
    for (size_t i = 0; i < n_dim; i++) {
        if (v[i] > 0.0f) {
            out[i / 8] = (uint8_t) (out[i / 8] | (1u << (7 - (i % 8))));
        }
    }
    return out;
}

// the packed byte as the signed int8 a client of the "binary" mode reads (two's complement: 0x80 -> -128)
static inline int server_embd_as_int8(uint8_t b) {
    return b >= 128 ? (int) b - 256 : (int) b;
}

// int8 quantisation: round-half-away(v * 127) clamped to [-127, 127] (symmetric: -128 is never produced, so negating a
// quantised vector stays representable). The scale 127 maps a unit-length component (|v| <= 1) onto the full range, which
// is exactly the case for the default L2-normalised output; a vector normalised another way (embd_normalize != 2) can have
// components above 1 and those saturate at +-127 rather than wrapping. NaN becomes 0.
static inline std::vector<int> server_embd_to_int8(const std::vector<float> & v, size_t n_dim) {
    std::vector<int> out;
    if (n_dim == 0 || n_dim > v.size()) {
        return out;
    }
    out.reserve(n_dim);
    for (size_t i = 0; i < n_dim; i++) {
        const float x = v[i];
        if (std::isnan(x)) {
            out.push_back(0);
            continue;
        }
        const double s = std::round((double) x * 127.0);
        out.push_back((int) std::max(-127.0, std::min(127.0, s)));
    }
    return out;
}

// One entry point for the handler: the integers to put in the JSON "embedding" array, or empty when the request is not
// quantised (mode NONE) or the dimension is illegal for the mode.
static inline std::vector<int> server_embd_quantize(const std::vector<float> & v, size_t n_dim, server_embd_quant q) {
    std::vector<int> out;
    switch (q) {
        case SERVER_EMBD_QUANT_BINARY:
        case SERVER_EMBD_QUANT_UBINARY: {
            const std::vector<uint8_t> bytes = server_embd_pack_bits(v, n_dim);
            out.reserve(bytes.size());
            for (uint8_t b : bytes) {
                out.push_back(q == SERVER_EMBD_QUANT_BINARY ? server_embd_as_int8(b) : (int) b);
            }
        } break;
        case SERVER_EMBD_QUANT_INT8:
            out = server_embd_to_int8(v, n_dim);
            break;
        case SERVER_EMBD_QUANT_NONE:
            break;
    }
    return out;
}

// Per-token rows (pooling 'none': one vector per input token, the ColBERT-style multi-vector form a late-interaction index
// stores). SOURCE-READ: upstream/vespa/model-integration/src/main/java/ai/vespa/embedding/ColBertEmbedder.java
// (vespa-engine/vespa@f4223f13, Apache-2.0): toBitTensor packs every token row ON ITS OWN from its first 8*dim floats with the
// same bit rule as above (value > 0, first value -> bit 7), embedDocument packs, embedQuery refuses an int8 target.
// SOURCE-NUANCE: the only int8 a token row gets there is the bit-packed one; no row is ever scaled to bytes. That matters here
// because send_embedding normalises POOLED vectors only, so a token row is not unit length and x127 would saturate: only the
// binary modes are accepted per token, int8 is refused (server_embd_quant_rows_ok). Description-level port, no code copied.
static inline bool server_embd_quant_rows_ok(server_embd_quant q) {
    return q == SERVER_EMBD_QUANT_BINARY || q == SERVER_EMBD_QUANT_UBINARY;
}

// every row quantised with server_embd_quantize; empty when the mode is not allowed per token, there are no rows, or ANY row
// cannot be packed (shorter than n_dim, illegal n_dim) -- one bad row fails the whole result instead of returning a ragged matrix
static inline std::vector<std::vector<int>> server_embd_quantize_rows(const std::vector<std::vector<float>> & rows, size_t n_dim, server_embd_quant q) {
    std::vector<std::vector<int>> out;
    if (!server_embd_quant_rows_ok(q) || rows.empty()) {
        return out;
    }
    out.reserve(rows.size());
    for (const auto & r : rows) {
        std::vector<int> one = server_embd_quantize(r, n_dim, q);
        if (one.empty()) {
            return {};
        }
        out.push_back(std::move(one));
    }
    return out;
}
