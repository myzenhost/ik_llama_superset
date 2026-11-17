// Dense-layer selection for offload-attention (--offload-attn-dense-layers).
// Row S69-ENGREV-HIPATTENTION-FLAG-SURFACE-2026-10-27-01. SOURCE-READ (spec only, no code copied): HiP Attention
// configs/mixed_landmark_0814_no_extend_qsa.json (Qwen3-235B dense_layers [0,1,2,47,46,45]) and configs/qwq_32b_1m.json
// (24 of 64 layers dense: 0-5, every 4th from 9..57, 59-63); hip_config.py:366 dense_layers is a LIST, not a count.
// Header-only and model-free so a unit test can drive it without a GGUF.
//
// Grammar, applied to one string:
//   "N"          first N layers (N >= 0; the original meaning, unchanged; "0" = none)
//   "-N"         last N layers
//   "a-b,c,d-e"  an explicit set (any token containing a comma, or an a-b range, selects explicit mode;
//                a lone layer index in explicit mode is written "c-c" or as part of a comma list "3,7")
// Layers outside [0, n_layer) are an ERROR (never silently clamped: a typo would quietly un-mask or mask the wrong layer).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace ik_offload {

struct DenseLayers {
    int32_t  first_n   = 0;        // used when !listed
    bool     listed    = false;    // explicit set in `mask`
    uint64_t mask[4]   = {0, 0, 0, 0};  // up to 256 layers

    bool contains(int il) const {
        if (!listed) return il < first_n;
        return il >= 0 && il < 256 && ((mask[il >> 6] >> (il & 63)) & 1ull) != 0;
    }
    // "0-2,45-47" style, or "first N" / "none"; used by the ENGAGED trace so the resolved set is visible.
    std::string describe(int n_layer) const {
        std::string out;
        if (!listed) {
            return first_n > 0 ? "first " + std::to_string(first_n) : std::string("none");
        }
        int il = 0;
        while (il < n_layer) {
            if (!contains(il)) { il++; continue; }
            int j = il;
            while (j + 1 < n_layer && contains(j + 1)) j++;
            if (!out.empty()) out += ",";
            out += (j == il) ? std::to_string(il) : std::to_string(il) + "-" + std::to_string(j);
            il = j + 1;
        }
        return out.empty() ? std::string("none") : out;
    }
};

// Rebuild the set from the flat cparams fields (first_n, listed flag, 256-bit mask).
inline DenseLayers dense_layers_from_cparams(int32_t first_n, bool listed, const uint64_t * mask4) {
    DenseLayers d;
    d.first_n = first_n;
    d.listed  = listed;
    for (int i = 0; i < 4; i++) d.mask[i] = mask4[i];
    return d;
}

namespace dense_detail {
inline bool parse_int(const std::string & s, long & v) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i = 1;
    if (i >= s.size()) return false;
    for (size_t k = i; k < s.size(); k++) if (s[k] < '0' || s[k] > '9') return false;
    if (s.size() - i > 9) return false;
    v = std::strtol(s.c_str(), nullptr, 10);
    return true;
}
} // namespace dense_detail

// Resolve `spec` for a model with n_layer layers. Returns false and fills err on a malformed or out-of-range spec.
inline bool parse_dense_layers(const std::string & spec_in, int n_layer, DenseLayers & out, std::string & err) {
    out = DenseLayers();
    std::string spec;
    for (char c : spec_in) if (c != ' ' && c != '\t') spec += c;
    if (spec.empty()) return true;
    if (n_layer < 1 || n_layer > 256) { err = "n_layer " + std::to_string(n_layer) + " outside 1..256"; return false; }

    long v = 0;
    if (dense_detail::parse_int(spec, v)) {      // "N" / "-N" / "+N"
        if (v >= 0) { out.first_n = (int32_t) v; return true; }
        long last = -v;
        if (last > n_layer) { err = "last-" + std::to_string(last) + " exceeds n_layer=" + std::to_string(n_layer); return false; }
        out.listed = true;
        for (long il = n_layer - last; il < n_layer; il++) out.mask[il >> 6] |= 1ull << (il & 63);
        return true;
    }

    out.listed = true;
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (tok.empty()) { err = "empty item in '" + spec_in + "'"; return false; }
        long a = 0, b = 0;
        size_t dash = tok.find('-', 1);          // a '-' after the first char is a range separator
        bool ok;
        if (dash == std::string::npos) {
            ok = dense_detail::parse_int(tok, a);
            b = a;
        } else {
            ok = dense_detail::parse_int(tok.substr(0, dash), a) && dense_detail::parse_int(tok.substr(dash + 1), b);
        }
        if (!ok) { err = "bad item '" + tok + "' in '" + spec_in + "'"; return false; }
        if (a < 0) a += n_layer;                 // inside a list a negative index counts from the end
        if (b < 0) b += n_layer;
        if (a < 0 || b < 0 || a >= n_layer || b >= n_layer || a > b) {
            err = "item '" + tok + "' outside 0.." + std::to_string(n_layer - 1) + " (or reversed)";
            return false;
        }
        for (long il = a; il <= b; il++) out.mask[il >> 6] |= 1ull << (il & 63);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return true;
}

} // namespace ik_offload
