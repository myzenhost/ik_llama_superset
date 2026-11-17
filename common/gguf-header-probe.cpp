#include "gguf-header-probe.h"

#include "llama.h"

#include <cctype>
#include <cstring>

// GGUF value type ids (ggml/include/gguf.h enum gguf_type).
enum : uint32_t {
    GT_UINT8 = 0, GT_INT8 = 1, GT_UINT16 = 2, GT_INT16 = 3, GT_UINT32 = 4, GT_INT32 = 5, GT_FLOAT32 = 6, GT_BOOL = 7,
    GT_STRING = 8, GT_ARRAY = 9, GT_UINT64 = 10, GT_INT64 = 11, GT_FLOAT64 = 12,
};

namespace {

constexpr uint32_t MAX_ARRAY_DEPTH = 4;  // nested arrays deeper than this are treated as malformed

struct reader {
    const uint8_t * p;
    size_t          n;
    size_t          off  = 0;
    bool            wide = true;  // v2+ lengths/counts are u64, v1 are u32

    bool take(size_t k, const uint8_t ** out = nullptr) {
        if (k > n - off) {
            return false;  // runs past the prefix: caller decides need_more
        }
        if (out) {
            *out = p + off;
        }
        off += k;
        return true;
    }
    bool u32(uint32_t & v) {
        const uint8_t * b;
        if (!take(4, &b)) {
            return false;
        }
        v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
        return true;
    }
    bool u64(uint64_t & v) {
        uint32_t lo, hi;
        if (!u32(lo) || !u32(hi)) {
            return false;
        }
        v = (uint64_t) lo | ((uint64_t) hi << 32);
        return true;
    }
    bool varlen(uint64_t & v) {
        if (wide) {
            return u64(v);
        }
        uint32_t w;
        if (!u32(w)) {
            return false;
        }
        v = w;
        return true;
    }
    // a length larger than what remains is a "need more" (or an absurd value that will never fit the cap)
    bool str(const uint8_t ** s, uint64_t & len) {
        return varlen(len) && len <= (uint64_t) (n - off) && take((size_t) len, s);
    }
};

int scalar_width(uint32_t ty) {
    switch (ty) {
        case GT_UINT8: case GT_INT8: case GT_BOOL:                   return 1;
        case GT_UINT16: case GT_INT16:                               return 2;
        case GT_UINT32: case GT_INT32: case GT_FLOAT32:              return 4;
        case GT_UINT64: case GT_INT64: case GT_FLOAT64:              return 8;
        default:                                                     return 0;
    }
}

// 1 = skipped, 0 = prefix too short, -1 = malformed (unknown type / too deep)
int skip_value(reader & r, uint32_t ty, uint32_t depth) {
    if (int w = scalar_width(ty)) {
        return r.take((size_t) w) ? 1 : 0;
    }
    if (ty == GT_STRING) {
        const uint8_t * s;
        uint64_t len;
        return r.str(&s, len) ? 1 : 0;
    }
    if (ty != GT_ARRAY || depth >= MAX_ARRAY_DEPTH) {
        return -1;
    }
    uint32_t elem;
    uint64_t count;
    if (!r.u32(elem) || !r.varlen(count)) {
        return 0;
    }
    if (int w = scalar_width(elem)) {
        if (count > (uint64_t) (r.n - r.off) / (uint64_t) w) {
            return 0;
        }
        return r.take((size_t) (count * (uint64_t) w)) ? 1 : 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        int k = skip_value(r, elem, depth + 1);
        if (k != 1) {
            return k;
        }
    }
    return 1;
}

}  // namespace

common_gguf_scan_result common_gguf_scan_file_type(const uint8_t * data, size_t size) {
    static const char KEY[] = "general.file_type";
    const size_t      key_len = sizeof(KEY) - 1;

    common_gguf_scan_result res;
    reader                  r{data, size};
    const uint8_t *         magic;
    if (!r.take(4, &magic)) {
        return res;  // need_more
    }
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        res.status = common_gguf_scan_status::absent;
        return res;
    }
    uint32_t version;
    if (!r.u32(version)) {
        return res;
    }
    if (version < 1 || version > 3) {
        res.status = common_gguf_scan_status::absent;  // also catches a big-endian file (version bytes swapped)
        return res;
    }
    r.wide = version >= 2;
    uint64_t n_tensors, n_kv;
    if (!r.varlen(n_tensors) || !r.varlen(n_kv)) {
        return res;
    }
    for (uint64_t i = 0; i < n_kv; ++i) {
        const uint8_t * key;
        uint64_t        len;
        uint32_t        ty;
        if (!r.str(&key, len) || !r.u32(ty)) {
            return res;
        }
        if (len == key_len && std::memcmp(key, KEY, key_len) == 0 && ty == GT_UINT32) {
            uint32_t v;
            if (!r.u32(v)) {
                return res;
            }
            res.status    = common_gguf_scan_status::found;
            res.file_type = v;
            return res;
        }
        int k = skip_value(r, ty, 0);
        if (k == 0) {
            return res;
        }
        if (k < 0) {
            res.status = common_gguf_scan_status::absent;
            return res;
        }
    }
    res.status = common_gguf_scan_status::absent;  // the whole KV section was read: the key is not there
    return res;
}

std::string common_ftype_tag(uint32_t file_type) {
    switch ((enum llama_ftype) (file_type & ~(uint32_t) LLAMA_FTYPE_GUESSED)) {
        case LLAMA_FTYPE_ALL_F32: return "F32";
#define T(name) case LLAMA_FTYPE_MOSTLY_##name: return #name;
        T(F16) T(BF16) T(Q4_0) T(Q4_1) T(Q5_0) T(Q5_1) T(Q6_0) T(Q8_0) T(Q2_K) T(Q2_K_S) T(Q3_K_S) T(Q3_K_M) T(Q3_K_L)
        T(Q4_K_S) T(Q4_K_M) T(Q5_K_S) T(Q5_K_M) T(Q6_K) T(IQ2_XXS) T(IQ2_XS) T(IQ2_S) T(IQ2_M) T(IQ3_XS) T(IQ3_XXS)
        T(IQ3_S) T(IQ3_M) T(IQ1_S) T(IQ1_M) T(IQ4_NL) T(IQ4_XS) T(Q4_0_4_4) T(Q4_0_4_8) T(Q4_0_8_8) T(TQ1_0) T(TQ2_0)
        T(MXFP4) T(IQ1_XS) T(IQ1_XXS) T(IQ1_XXXS) T(MXFP6_E2M3) T(MXFP6_E3M2) T(MXFP8_E4M3) T(MXFP8_E5M2) T(TQ3_1S)
        T(TQ4_1S) T(Q1_0_G128) T(Q2_0_G128) T(IQ1_BN) T(IQ2_BN) T(IQ2_K) T(IQ3_K) T(IQ4_K) T(IQ5_K) T(IQ6_K) T(IQ4_KS)
        T(IQ3_KL) T(IQ2_KS) T(IQ4_KSS) T(Q8_KV) T(IQ5_KS) T(IQ2_KT) T(IQ3_KT) T(IQ4_KT) T(IQ3_KS) T(IQ2_KL) T(IQ1_KT)
        T(PQ2_0) T(PTQ1_0) T(Q1_0_G128_R8) T(PQ2_0_R8) T(PTQ1_0_R8) T(TQK6) T(TQK7) T(TQ2_T) T(TQP5) T(TQP6) T(TQP7)
        T(NVFP4_KS) T(Q4_0_R8) T(Q8_0_R8) T(Q5_0_R4) T(Q2_K_R4) T(Q3_K_R4) T(Q4_K_R4) T(Q5_K_R4) T(Q6_K_R4)
        T(IQ2_XXS_R4) T(IQ2_XS_R4) T(IQ3_XXS_R4) T(IQ1_S_R4) T(IQ4_NL_R4) T(IQ3_S_R4) T(IQ2_M_R4) T(IQ4_XS_R8)
        T(IQ1_M_R4) T(Q6_0_R4) T(BF16_R16) T(IQ2_BN_R4) T(IQ2_K_R4) T(IQ3_K_R4) T(IQ4_K_R4) T(IQ5_K_R4) T(IQ4_KS_R4)
        T(IQ4_KS_R16) T(IQ3_KS_R16) T(IQ5_KS_R16) T(IQ5_KS_R4) T(MXFP4_R8) T(Q2_0_R8) T(Q8_K_R16) T(Q8_KV_R8)
        T(Q8_K_R8) T(NVFP4) T(Q1_0) T(MXFP4_MOE) T(Q2_0) T(F8_E4M3)
#undef T
        default: return "";
    }
}

bool common_ftype_tag_is_known(const std::string & tag) {
    std::string up = tag;
    for (char & c : up) {
        c = (char) std::toupper((unsigned char) c);
    }
    // every enumerator is below LLAMA_FTYPE_GUESSED (1024); walking the switch is cheaper than a second table to keep in sync
    for (uint32_t v = 0; v < (uint32_t) LLAMA_FTYPE_GUESSED; ++v) {
        const std::string t = common_ftype_tag(v);
        if (!t.empty() && t == up) {
            return true;
        }
    }
    return false;
}

std::string common_gguf_probe_tag(const common_gguf_range_fetch & fetch, uint64_t file_size, uint64_t * bytes_read,
                                  uint64_t chunk, uint64_t cap) {
    const uint64_t       limit = file_size ? (file_size < cap ? file_size : cap) : cap;
    std::vector<uint8_t> buf;
    std::string          tag;
    while ((uint64_t) buf.size() < limit && chunk > 0) {
        const uint64_t have = buf.size();
        const uint64_t want = (limit - have < chunk) ? limit - have : chunk;
        if (!fetch(have, want, buf) || (uint64_t) buf.size() == have) {
            break;  // transport failure or no progress
        }
        auto r = common_gguf_scan_file_type(buf.data(), buf.size());
        if (r.status == common_gguf_scan_status::found) {
            tag = common_ftype_tag(r.file_type);
            break;
        }
        if (r.status == common_gguf_scan_status::absent) {
            break;
        }
    }
    if (bytes_read) {
        *bytes_read = buf.size();
    }
    return tag;
}
