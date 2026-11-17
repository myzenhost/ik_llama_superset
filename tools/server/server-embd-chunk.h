// Long-text embedding for /v1/embeddings and /embedding: overlapping chunks, one vector per chunk, mean of the chunk
// vectors. Opt-in per request through "long_text_mode"; without it a text longer than one micro-batch keeps failing with
// the existing "input is too large to process" error, byte for byte.
//
// Header-only and free of llama / ggml / json dependencies, so it is unit-tested standalone
// (tools/server/tests_cpp/test_server_embd_chunk.cpp).
//
// Why: an embedding sequence must sit whole inside one micro-batch (n_ubatch) and inside the model's trained context, so a
// document longer than that cannot be embedded at all today. Truncating silently loses the tail; the usual client-side
// workaround (split, embed N times, average) costs N round trips and gets the overlap and the renormalisation wrong half the
// time. The server can do it in one request.
//
// SOURCE-READ: nomic-ai/gpt4all@b666d16db (MIT), gpt4all-backend/src/llamamodel.cpp:1149-1173 (max_len = min(n_ctx_train,
// n_batch) minus prefix and EOS tokens; an error when max_len <= overlap; chunks advance by max_len - overlap; the "truncate"
// mode keeps the first chunk only), :1202-1239 (every chunk vector is L2-normalised on its own before it is summed) and
// :1258-1270 (the sum is divided by the number of chunks, then L2-normalised again). Description-level port, no code copied.
// Row: S69-ENGINE-MERGE-GPT4ALL-2026-09-27-01 (NEW-PICK long-text embedding).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

enum server_embd_long_text {
    SERVER_EMBD_LONG_TEXT_ERROR = 0, // the existing behaviour: refuse a text that does not fit
    SERVER_EMBD_LONG_TEXT_TRUNCATE,  // embed the first chunk only
    SERVER_EMBD_LONG_TEXT_MEAN,      // embed every chunk, average
};

// "error" | "truncate" | "mean" -> true and the mode; anything else -> false
inline bool server_embd_long_text_from_name(const std::string & name, server_embd_long_text & out) {
    if (name == "error")    { out = SERVER_EMBD_LONG_TEXT_ERROR;    return true; }
    if (name == "truncate") { out = SERVER_EMBD_LONG_TEXT_TRUNCATE; return true; }
    if (name == "mean")     { out = SERVER_EMBD_LONG_TEXT_MEAN;     return true; }
    return false;
}

// a half-open range [begin, end) over the BODY tokens (the text tokens between the leading and the trailing special token)
struct server_embd_chunk_range {
    size_t begin;
    size_t end;
};

// Split n_body body tokens into chunks of at most max_body tokens, each starting max_body - overlap after the previous one,
// so neighbours share `overlap` tokens. The last chunk ends exactly at n_body (no chunk made only of overlap). With
// truncate_only only the first chunk is returned. Returns false and fills err when the geometry is impossible
// (max_body <= overlap would never advance).
inline bool server_embd_chunk_plan(size_t n_body, size_t max_body, size_t overlap, bool truncate_only,
                                   std::vector<server_embd_chunk_range> & out, std::string & err) {
    out.clear();
    if (max_body == 0 || max_body <= overlap) {
        err = "long_text chunk length (" + std::to_string(max_body) + " tokens) must be larger than the overlap (" + std::to_string(overlap) + " tokens)";
        return false;
    }
    if (n_body == 0) {
        out.push_back({ 0, 0 });
        return true;
    }
    const size_t stride = max_body - overlap;
    for (size_t b = 0;; b += stride) {
        const size_t e = std::min(b + max_body, n_body);
        out.push_back({ b, e });
        if (e == n_body || truncate_only) {
            break;
        }
    }
    return true;
}

// Mean of chunk vectors that the worker already L2-normalised: the plain average (the caller applies the request's
// normalisation to the result). Vectors of unequal length are a bug upstream: the result is empty then.
inline std::vector<float> server_embd_mean_pool(const std::vector<std::vector<float>> & vecs) {
    if (vecs.empty()) {
        return {};
    }
    const size_t n = vecs[0].size();
    std::vector<double> sum(n, 0.0);
    for (const auto & v : vecs) {
        if (v.size() != n) {
            return {};
        }
        for (size_t i = 0; i < n; i++) {
            sum[i] += v[i];
        }
    }
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) {
        out[i] = (float) (sum[i] / (double) vecs.size());
    }
    return out;
}
