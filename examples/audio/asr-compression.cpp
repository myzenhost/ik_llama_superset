// asr-compression.cpp -- see asr-compression.h
#include "asr-compression.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef ASR_HAVE_ZLIB
#include <zlib.h>
#endif

bool asr_compression_zlib_available() {
#ifdef ASR_HAVE_ZLIB
    return true;
#else
    return false;
#endif
}

double asr_compression_ratio_zlib(const std::string & text) {
#ifdef ASR_HAVE_ZLIB
    if (text.empty()) {
        return 0.0;
    }
    uLongf cap = compressBound((uLong) text.size());
    std::vector<Bytef> out(cap);
    if (compress2(out.data(), &cap, (const Bytef *) text.data(), (uLong) text.size(), Z_DEFAULT_COMPRESSION) != Z_OK) {
        return -1.0;
    }
    return (double) text.size() / (double) cap;
#else
    (void) text;
    return -1.0;
#endif
}

// SIZE MODEL for the no-zlib fallback. It estimates len(zlib.compress(text)) from the real RFC 1951 coding costs: a greedy LZ77 parse (hash chains, 32 KiB
// window, 64 candidates), then the cheapest of a stored block, the fixed-Huffman block and a dynamic-Huffman block whose header (HLIT/HDIST/HCLEN and the
// run-length coded code lengths) is counted too, plus the 6 framing bytes of the zlib wrapper. The previous estimate charged a flat 14 bits per match and an
// order-0 entropy per literal, which under-counted the size by 17-44 percent on short repetitive strings (measured against python zlib on eight fixtures,
// tests/test-asr-compression.cpp "RFC 1951 size model"), and so overstated the ratio the gate compares with a threshold.
// Behaviour read from RFC 1951 and zlib's deflate.c/trees.c docs (zlib licence); the code is written fresh.
namespace {


static const int LEN_BASE[29]  = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
static const int LEN_EXTRA[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
static const int DIST_BASE[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
static const int DIST_EXTRA[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };

inline int len_class(int len) {
    int c = 28;
    while (c > 0 && LEN_BASE[c] > len) {
        --c;
    }
    return c;
}

inline int dist_class(int dist) {
    int c = 29;
    while (c > 0 && DIST_BASE[c] > dist) {
        --c;
    }
    return c;
}

// Huffman code lengths for `freq` limited to `max_len` bits. A tree with fewer than two used symbols gets dummy symbols, as zlib does, so the code
// stays complete. Length limiting: clamp, then lengthen the cheapest shorter codes until the Kraft sum is back to 1.
inline std::vector<int> huffman_lengths(std::vector<long> freq, int max_len) {
    const size_t n = freq.size();
    int used = 0;
    for (long f : freq) {
        used += f > 0;
    }
    for (size_t s = 0; used < 2 && s < n; ++s) {
        if (freq[s] == 0) {
            freq[s] = 1;
            ++used;
        }
    }
    struct node { long w; int left, right; };
    std::vector<node> nodes;
    std::vector<int> heap;
    for (size_t s = 0; s < n; ++s) {
        nodes.push_back({ freq[s], -1, (int) s });
        if (freq[s] > 0) {
            heap.push_back((int) s);
        }
    }
    auto cmp = [&](int a, int b) { return nodes[a].w != nodes[b].w ? nodes[a].w > nodes[b].w : a > b; }; // min-heap, ties by index (deterministic)
    std::make_heap(heap.begin(), heap.end(), cmp);
    while (heap.size() > 1) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        const int a = heap.back();
        heap.pop_back();
        std::pop_heap(heap.begin(), heap.end(), cmp);
        const int b = heap.back();
        heap.pop_back();
        nodes.push_back({ nodes[a].w + nodes[b].w, a, b });
        heap.push_back((int) nodes.size() - 1);
        std::push_heap(heap.begin(), heap.end(), cmp);
    }
    std::vector<int> len(n, 0);
    // depth of each leaf
    std::vector<std::pair<int, int>> stack;
    stack.push_back({ heap.empty() ? 0 : heap[0], 0 });
    while (!stack.empty()) {
        const auto [id, d] = stack.back();
        stack.pop_back();
        if (id < (int) n) {
            len[id] = std::max(d, 1);
        } else {
            stack.push_back({ nodes[id].left, d + 1 });
            stack.push_back({ nodes[id].right, d + 1 });
        }
    }
    bool over = false;
    for (int & l : len) {
        if (l > max_len) {
            l = max_len;
            over = true;
        }
    }
    if (over) {
        auto kraft = [&]() { double k = 0; for (int l : len) { if (l > 0) { k += std::ldexp(1.0, -l); } } return k; };
        while (kraft() > 1.0 + 1e-12) {
            int pick = -1;
            for (size_t s = 0; s < n; ++s) { // lengthen the shortest code that can still grow: costs the fewest bits
                if (len[s] > 0 && len[s] < max_len && (pick < 0 || len[s] > len[pick])) {
                    pick = (int) s;
                }
            }
            if (pick < 0) {
                break;
            }
            ++len[pick];
        }
    }
    return len;
}

struct lz_token { int lit; int len; int dist; }; // lit >= 0: literal byte; else a match

inline std::vector<lz_token> lz_parse(const std::string & s) {
    const size_t n = s.size();
    std::vector<lz_token> out;
    std::vector<std::vector<int>> chains(1 << 12);
    auto h3 = [&](size_t i) { return (unsigned) ((((unsigned char) s[i] << 8) ^ ((unsigned char) s[i + 1] << 4) ^ (unsigned char) s[i + 2]) & 0xfff); };
    size_t i = 0;
    while (i < n) {
        int best_len = 0, best_dist = 0;
        if (i + 2 < n) {
            const auto & ch = chains[h3(i)];
            for (int k = (int) ch.size() - 1; k >= 0 && k >= (int) ch.size() - 64; --k) {
                const size_t j = (size_t) ch[k];
                if (i - j > 32768) {
                    break;
                }
                int l = 0;
                while (i + l < n && l < 258 && s[j + l] == s[i + l]) {
                    ++l;
                }
                if (l > best_len) {
                    best_len = l;
                    best_dist = (int) (i - j);
                }
            }
        }
        const int adv = best_len >= 3 ? best_len : 1;
        if (best_len >= 3) {
            out.push_back({ -1, best_len, best_dist });
        } else {
            out.push_back({ (unsigned char) s[i], 0, 0 });
        }
        for (int k = 0; k < adv; ++k) {
            if (i + k + 2 < n) {
                chains[h3(i + k)].push_back((int) (i + k));
            }
        }
        i += adv;
    }
    return out;
}

// bits of the run-length coded code-length sequence of a dynamic block header, plus the code-length code's own header
inline size_t dynamic_header_bits(const std::vector<int> & lit_len, const std::vector<int> & dist_len, int hlit, int hdist) {
    std::vector<int> seq(lit_len.begin(), lit_len.begin() + hlit);
    seq.insert(seq.end(), dist_len.begin(), dist_len.begin() + hdist);
    struct sym { int code; int extra_bits; };
    std::vector<sym> rle;
    for (size_t i = 0; i < seq.size();) {
        size_t j = i;
        while (j < seq.size() && seq[j] == seq[i]) {
            ++j;
        }
        size_t run = j - i;
        if (seq[i] == 0) {
            while (run >= 11) { const size_t r = std::min<size_t>(run, 138); rle.push_back({ 18, 7 }); run -= r; }
            if (run >= 3)     { rle.push_back({ 17, 3 }); run = 0; }
            while (run > 0)   { rle.push_back({ 0, 0 }); --run; }
        } else {
            rle.push_back({ seq[i], 0 });
            --run;
            while (run >= 3)  { const size_t r = std::min<size_t>(run, 6); rle.push_back({ 16, 2 }); run -= r; }
            while (run > 0)   { rle.push_back({ seq[i], 0 }); --run; }
        }
        i = j;
    }
    std::vector<long> cf(19, 0);
    for (const auto & r : rle) {
        ++cf[r.code];
    }
    const std::vector<int> cl = huffman_lengths(cf, 7);
    static const int ORDER[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
    int hclen = 19;
    while (hclen > 4 && cl[ORDER[hclen - 1]] == 0) {
        --hclen;
    }
    size_t bits = 5 + 5 + 4 + 3 * (size_t) hclen;
    for (const auto & r : rle) {
        bits += (size_t) cl[r.code] + r.extra_bits;
    }
    return bits;
}



// estimated len(zlib.compress(text)) in bytes
size_t zlib_size_estimate(const std::string & s) {
    const size_t n = s.size();
    if (n == 0) {
        return 8; // 2 header + empty fixed block (1 byte) + 4 adler
    }
    const auto toks = lz_parse(s);
    std::vector<long> lf(286, 0), df(30, 0);
    size_t fixed_bits = 3, extra_bits = 0;
    for (const auto & t : toks) {
        if (t.lit >= 0) {
            ++lf[t.lit];
            fixed_bits += t.lit < 144 ? 8 : 9;
        } else {
            const int lc = len_class(t.len), dc = dist_class(t.dist);
            ++lf[257 + lc];
            ++df[dc];
            fixed_bits += (257 + lc <= 279 ? 7 : 8) + 5;
            extra_bits += (size_t) LEN_EXTRA[lc] + DIST_EXTRA[dc];
        }
    }
    ++lf[256];
    fixed_bits += 7 + extra_bits;

    const std::vector<int> ll = huffman_lengths(lf, 15);
    const std::vector<int> dl = huffman_lengths(df, 15);
    int hlit = 286, hdist = 30;
    while (hlit > 257 && ll[hlit - 1] == 0) {
        --hlit;
    }
    while (hdist > 1 && dl[hdist - 1] == 0) {
        --hdist;
    }
    size_t dyn_bits = 3 + dynamic_header_bits(ll, dl, hlit, hdist) + extra_bits;
    for (int sy = 0; sy < 286; ++sy) {
        dyn_bits += (size_t) lf[sy] * ll[sy];
    }
    for (int sy = 0; sy < 30; ++sy) {
        dyn_bits += (size_t) df[sy] * dl[sy];
    }
    const size_t best_bits = std::min(fixed_bits, dyn_bits);
    const size_t deflate_bytes = std::min<size_t>((best_bits + 7) / 8, n + 5); // the stored block (5 bytes of framing) when raw is smaller
    return 2 + deflate_bytes + 4;
}

} // namespace

double asr_compression_ratio_estimate(const std::string & text) {
    if (text.empty()) {
        return 0.0;
    }
    return (double) text.size() / (double) zlib_size_estimate(text);
}

double asr_compression_ratio(const std::string & text) {
#ifdef ASR_HAVE_ZLIB
    const double r = asr_compression_ratio_zlib(text);
    if (r >= 0.0) {
        return r;
    }
#endif
    return asr_compression_ratio_estimate(text);
}

bool asr_text_repeats(const std::string & text, double thold) {
    return asr_compression_ratio(text) > thold;
}
