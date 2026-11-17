#include "llama-impl.h"
#include "llama-mmap.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstring>
#include <vector>

// WIRE-66 GROUP 1 (2026-09-08): this file arrived unbuilt (upstream file-split, src/llama.cpp
// carries the fork's own logging + gguf-kv-to-string implementation). Measured collision
// (temp/_adapter_rename_surface.py "BOTH" section): llama_log_set and gguf_kv_to_str are DEFINED
// in built src/llama.cpp already; llama_log_internal / llama_log_callback_default are not in the
// declared-API list but redefine (duplicate-symbol) the same names src/llama.cpp:13734-13775
// already provides against its OWN static state (g_state), which additionally handles
// GGML_USE_METAL/CUDA/CANN backend log-callback propagation that this file's version lacks --
// src/llama.cpp's is strictly more complete, so it stays authoritative. Wiring those four verbatim
// would be a link error (duplicate symbol) for zero gain. They are cut from this translation unit;
// llama_log_set/_internal/_callback_default/gguf_kv_to_str keep resolving to src/llama.cpp exactly
// as before -- unchanged behavior for existing callers (common/, tools/, src/llama.cpp itself).
//
// llama_log_get has NO twin anywhere (surface script: "defined NOWHERE" before this file is wired)
// -- it is the one declared-API symbol this file exists to provide.
//
// replace_all / format / llama_format_tensor_shape (both overloads) are NOT declared in the public
// API surface (internal helpers declared only in llama-impl.h) so they are outside this task's 66,
// but they are USED by already-built src/llama.cpp:7879+ and src/llama-quantize.cpp:1505+,
// src/llama-load-tensors.cpp:2760+, src/llama-model-loader.cpp:590+ with NO definition anywhere in
// the tree before this file is wired -- confirmed by grepping src/*.cpp for their definitions
// (only llama-impl.cpp had one). That is a pre-existing undefined-reference link gap, separate
// from the 66-symbol brief; keeping them wired here closes it as a side effect, not scope creep,
// since excluding them would newly introduce a link failure for callers that already exist.

// RESTORED 2026-09-13, verbatim from this repository's own history: `git show 5bbb857c2^:
// src/llama-impl.cpp` lines 20-26. It was cut by 5bbb857c2 (the WIRE-66 GROUP 1 commit described
// above) as COLLATERAL, not on purpose -- that commit's own message names exactly four symbols it
// cut and why (llama_log_set, llama_log_internal, llama_log_callback_default, gguf_kv_to_str, all
// four defined more completely in src/llama.cpp). time_meas is not one of them and has no twin:
// src/llama-impl.h:226-228 DECLARES it and, until this line, nothing in the tree defined it.
//
// The cost of that was 144 undefined references at link -- 96 to the destructor and 48 to the
// constructor -- across the tools and tests that use it, measured in temp/_fullbuild3.log.
//
// NOTE the near-name: common/common.h:60 declares common_time_meas and common/common.cpp:66
// defines it. Different type, different library, same idea; it is not a substitute and the two
// must not be conflated.
void llama_prefetch_rows(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) {
    if (!tensor || !tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || n_rows == 0) {
        return;
    }

    GGML_ASSERT(ggml_is_matrix(tensor));

    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    const auto * base = (const char *) tensor->data;

    std::vector<llama_memory_range> mr;
    mr.reserve(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        GGML_ASSERT(rows[i] >= 0 && rows[i] < tensor->ne[1]);
        mr.push_back({ base + (size_t) rows[i] * tensor->nb[1], row_bytes });
    }

    llama_prefetch(std::move(mr));
}

time_meas::time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

time_meas::~time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

void llama_log_get(ggml_log_callback * log_callback, void ** user_data) {
    ggml_log_get(log_callback, user_data);
}

void llama_clear_tensor_data(ggml_tensor * t, size_t offset, size_t size) {
    static const std::vector<uint8_t> zeros(1024*1024, 0);

    // not all backend buffers implement ggml_backend_tensor_memset(), so write zeros instead
    // TODO: make this a generic fallback in `ggml_backend_tensor_memset` when `set_tensor` is available
    for (size_t ofs = 0; ofs < size; ofs += zeros.size()) {
        ggml_backend_tensor_set(t, zeros.data(), offset + ofs, std::min(size - ofs, zeros.size()));
    }
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, ne.at(i));
    }
    return buf;
}

std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, t->ne[i]);
    }
    return buf;
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, size_t i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i, size_t max_len) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                const size_t arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::string res = "[";
                size_t j = 0;
                for (; j < arr_n && res.size() <= max_len; j++) {
                    if (j > 0) {
                        res += ", ";
                    }
                    if (arr_type == GGUF_TYPE_STRING) {
                        const char * s = gguf_get_arr_str(ctx_gguf, i, j);
                        const size_t need = res.size() + 2 * strlen(s) + 2;
                        if (res.capacity() < need) {
                            res.reserve(std::max(need, 2 * res.capacity()));
                        }
                        res += '"';
                        for (; *s; s++) {
                            if (*s == '\\' || *s == '"') {
                                res += '\\';
                            }
                            res += *s;
                        }
                        res += '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        res += "???";
                    } else {
                        res += gguf_data_to_str(arr_type, data, j);
                    }
                }
                if (j == arr_n) {
                    res += "]";
                }
                return res;
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}
