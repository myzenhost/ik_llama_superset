// llama-cram-soft.cpp -- see llama-cram-soft.h.
#include "llama-cram-soft.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>

namespace {

// ---------- CRC32 (IEEE, slicing-by-8) ----------
struct crc_tables {
    uint32_t t[8][256];
    crc_tables() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; i++)
            for (int s = 1; s < 8; s++) t[s][i] = (t[s-1][i] >> 8) ^ t[0][t[s-1][i] & 0xff];
    }
};
const crc_tables g_crc;

uint32_t crc32(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    while (n >= 8) {
        uint32_t a, b;
        std::memcpy(&a, p, 4); std::memcpy(&b, p + 4, 4);
        a ^= c;
        c = g_crc.t[7][a & 0xff] ^ g_crc.t[6][(a >> 8) & 0xff] ^ g_crc.t[5][(a >> 16) & 0xff] ^ g_crc.t[4][a >> 24] ^
            g_crc.t[3][b & 0xff] ^ g_crc.t[2][(b >> 8) & 0xff] ^ g_crc.t[1][(b >> 16) & 0xff] ^ g_crc.t[0][b >> 24];
        p += 8; n -= 8;
    }
    while (n--) c = g_crc.t[0][(c ^ *p++) & 0xff] ^ (c >> 8);
    return ~c;
}

// ---------- plane codecs ----------
// Plane encodings: 0 raw, 1 RLE, 2 Huffman. Plane record: [mode u8][payload_len u32 LE][payload].
// RLE: control c<128 -> c+1 literal bytes follow; c>=128 -> byte follows, repeated (c-128)+3 times.
constexpr int HUF_MAX_LEN = 12;

void put32(std::vector<uint8_t> & o, uint32_t v) { for (int i = 0; i < 4; i++) o.push_back((v >> (8*i)) & 0xff); }

void rle_encode(const uint8_t * s, size_t n, std::vector<uint8_t> & o) {
    size_t i = 0, lit_start = 0;
    auto flush_lits = [&](size_t end) {
        while (lit_start < end) {
            size_t c = std::min<size_t>(128, end - lit_start);
            o.push_back((uint8_t)(c - 1));
            o.insert(o.end(), s + lit_start, s + lit_start + c);
            lit_start += c;
        }
    };
    while (i < n) {
        size_t j = i;
        while (j < n && s[j] == s[i] && j - i < 130) j++;
        if (j - i >= 3) {
            flush_lits(i);
            o.push_back((uint8_t)(128 + (j - i - 3)));
            o.push_back(s[i]);
            i = j; lit_start = j;
        } else {
            i++;
        }
    }
    flush_lits(n);
}

// out has room for n elements written at out[k*stride]
bool rle_decode(const uint8_t * p, size_t plen, uint8_t * out, size_t stride, size_t n) {
    size_t k = 0, i = 0;
    while (i < plen) {
        uint8_t c = p[i++];
        if (c < 128) {
            size_t cnt = (size_t) c + 1;
            if (i + cnt > plen || k + cnt > n) return false;
            for (size_t t = 0; t < cnt; t++) out[(k++) * stride] = p[i++];
        } else {
            size_t cnt = (size_t)(c - 128) + 3;
            if (i >= plen || k + cnt > n) return false;
            uint8_t v = p[i++];
            for (size_t t = 0; t < cnt; t++) out[(k++) * stride] = v;
        }
    }
    return k == n;
}

// Huffman lengths (<= HUF_MAX_LEN) from frequencies; returns false if <1 symbol.
bool huf_lengths(const uint32_t * freq_in, uint8_t * len) {
    uint32_t freq[256];
    std::memcpy(freq, freq_in, sizeof(freq));
    int nsym = 0;
    for (int i = 0; i < 256; i++) nsym += freq[i] != 0;
    std::memset(len, 0, 256);
    if (nsym == 0) return false;
    if (nsym == 1) { for (int i = 0; i < 256; i++) if (freq[i]) len[i] = 1; return true; }
    for (;;) {
        struct node { uint64_t w; int l, r; };
        std::vector<node> nodes;
        using pq_t = std::pair<uint64_t,int>;
        std::priority_queue<pq_t, std::vector<pq_t>, std::greater<pq_t>> pq;
        for (int i = 0; i < 256; i++) if (freq[i]) { nodes.push_back({freq[i], -1 - i, 0}); pq.push({freq[i], (int) nodes.size() - 1}); }
        while (pq.size() > 1) {
            auto a = pq.top(); pq.pop(); auto b = pq.top(); pq.pop();
            nodes.push_back({a.first + b.first, a.second, b.second});
            pq.push({a.first + b.first, (int) nodes.size() - 1});
        }
        // depth walk
        std::memset(len, 0, 256);
        int maxl = 0;
        std::vector<std::pair<int,int>> st; st.push_back({pq.top().second, 0});
        while (!st.empty()) {
            auto [id, d] = st.back(); st.pop_back();
            if (nodes[id].l < 0 && nodes[id].r == 0) { len[-1 - nodes[id].l] = (uint8_t) d; maxl = std::max(maxl, d); continue; }
            st.push_back({nodes[id].l, d + 1}); st.push_back({nodes[id].r, d + 1});
        }
        if (maxl <= HUF_MAX_LEN) return true;
        for (int i = 0; i < 256; i++) if (freq[i]) freq[i] = (freq[i] + 1) / 2;
    }
}

// canonical codes: sorted by (len, sym)
void huf_codes(const uint8_t * len, uint16_t * code) {
    uint32_t c = 0;
    for (int l = 1; l <= HUF_MAX_LEN; l++) {
        for (int s = 0; s < 256; s++) if (len[s] == l) code[s] = (uint16_t) c++;
        c <<= 1;
    }
}

bool huf_encode(const uint8_t * s, size_t n, std::vector<uint8_t> & o) {
    uint32_t freq[256] = {0};
    for (size_t i = 0; i < n; i++) freq[s[i]]++;
    uint8_t len[256]; uint16_t code[256] = {0};
    if (!huf_lengths(freq, len)) return false;
    huf_codes(len, code);
    for (int i = 0; i < 256; i += 2) o.push_back((uint8_t)(len[i] | (len[i+1] << 4)));
    std::vector<uint8_t> bits;
    uint64_t acc = 0; int nb = 0;
    for (size_t i = 0; i < n; i++) {
        acc = (acc << len[s[i]]) | code[s[i]]; nb += len[s[i]];
        while (nb >= 8) { bits.push_back((uint8_t)(acc >> (nb - 8))); nb -= 8; }
    }
    if (nb) bits.push_back((uint8_t)(acc << (8 - nb)));
    put32(o, (uint32_t) bits.size());
    o.insert(o.end(), bits.begin(), bits.end());
    return true;
}

bool huf_decode(const uint8_t * p, size_t plen, uint8_t * out, size_t stride, size_t n) {
    if (plen < 132) return false;
    uint8_t len[256]; uint16_t code[256] = {0};
    for (int i = 0; i < 256; i += 2) { len[i] = p[i/2] & 15; len[i+1] = p[i/2] >> 4; }
    // Kraft check so table fill cannot overflow
    uint64_t kraft = 0;
    for (int i = 0; i < 256; i++) if (len[i]) { if (len[i] > HUF_MAX_LEN) return false; kraft += 1u << (HUF_MAX_LEN - len[i]); }
    if (kraft > (1u << HUF_MAX_LEN)) return false;
    huf_codes(len, code);
    uint16_t tab[1 << HUF_MAX_LEN]; // (sym<<4)|len ; 0 = invalid  (sym<<4 fits: 255<<4 | 12 < 65536)
    std::memset(tab, 0, sizeof(tab));
    for (int s = 0; s < 256; s++) if (len[s]) {
        uint32_t base = (uint32_t) code[s] << (HUF_MAX_LEN - len[s]);
        uint32_t cnt = 1u << (HUF_MAX_LEN - len[s]);
        if (base + cnt > (1u << HUF_MAX_LEN)) return false;
        for (uint32_t t = 0; t < cnt; t++) tab[base + t] = (uint16_t)((s << 4) | len[s]);
    }
    uint32_t nbytes = (uint32_t) p[128] | ((uint32_t) p[129] << 8) | ((uint32_t) p[130] << 16) | ((uint32_t) p[131] << 24);
    if (132 + (size_t) nbytes != plen) return false;
    const uint8_t * b = p + 132; const uint8_t * e = b + nbytes;
    uint64_t acc = 0; int nb = 0;
    for (size_t k = 0; k < n; k++) {
        while (nb <= 56 && b < e) { acc = (acc << 8) | *b++; nb += 8; }
        uint32_t peek = nb >= HUF_MAX_LEN ? (uint32_t)(acc >> (nb - HUF_MAX_LEN)) & ((1u << HUF_MAX_LEN) - 1)
                                          : (uint32_t)(acc << (HUF_MAX_LEN - nb)) & ((1u << HUF_MAX_LEN) - 1);
        uint16_t t = tab[peek];
        int l = t & 15;
        if (l == 0 || l > nb) return false;
        out[k * stride] = (uint8_t)(t >> 4);
        nb -= l;
    }
    return true;
}

// Block encode: [stride u8] then `stride` plane records. Returns encoded size.
size_t encode_block(const uint8_t * src, size_t n, llama_cram_codec codec, int stride, std::vector<uint8_t> & out) {
    out.clear();
    out.push_back((uint8_t) stride);
    std::vector<uint8_t> plane, rle, huf;
    for (int p = 0; p < stride; p++) {
        plane.clear();
        for (size_t i = (size_t) p; i < n; i += stride) plane.push_back(src[i]);
        rle.clear(); huf.clear();
        int mode = 0; size_t best = plane.size();
        rle_encode(plane.data(), plane.size(), rle);
        if (rle.size() < best) { best = rle.size(); mode = 1; }
        if (codec == LLAMA_CRAM_CODEC_PLANE_HUF && !plane.empty() && huf_encode(plane.data(), plane.size(), huf) && huf.size() < best) {
            best = huf.size(); mode = 2;
        }
        out.push_back((uint8_t) mode);
        put32(out, (uint32_t) best);
        const std::vector<uint8_t> & sel = mode == 0 ? plane : mode == 1 ? rle : huf;
        out.insert(out.end(), sel.begin(), sel.end());
    }
    return out.size();
}

bool decode_block(const uint8_t * p, size_t plen, uint8_t * dst, size_t n) {
    if (plen < 1) return false;
    size_t stride = p[0];
    if (stride < 1 || stride > 2) return false;
    size_t pos = 1;
    for (size_t pl = 0; pl < stride; pl++) {
        if (pos + 5 > plen) return false;
        int mode = p[pos];
        uint32_t len = (uint32_t) p[pos+1] | ((uint32_t) p[pos+2] << 8) | ((uint32_t) p[pos+3] << 16) | ((uint32_t) p[pos+4] << 24);
        pos += 5;
        if (len > plen - pos) return false;
        size_t cnt = n > pl ? (n - pl + stride - 1) / stride : 0;
        uint8_t * out = dst + pl;
        bool ok;
        if (mode == 0) { ok = len == cnt; if (ok) for (size_t k = 0; k < cnt; k++) out[k * stride] = p[pos + k]; }
        else if (mode == 1) ok = rle_decode(p + pos, len, out, stride, cnt);
        else if (mode == 2) ok = huf_decode(p + pos, len, out, stride, cnt);
        else ok = false;
        if (!ok) return false;
        pos += len;
    }
    return pos == plen;
}

uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

llama_cram_soft_store::llama_cram_soft_store(const void * data, size_t size, const llama_cram_soft_params & p) : params_(p), size_(size) {
    if (params_.block_size == 0) throw std::invalid_argument("block_size must be > 0");
    if (params_.stride < 1 || params_.stride > 2) throw std::invalid_argument("stride must be 1 or 2");
    if (params_.shards == 0) params_.shards = 1;
    shard_cap_ = std::max<size_t>(1, params_.hot_blocks / params_.shards);
    for (size_t i = 0; i < params_.shards; i++) shards_.emplace_back(new shard());

    const uint8_t * src = (const uint8_t *) data;
    std::vector<uint8_t> enc;
    for (size_t off = 0; off < size; off += params_.block_size) {
        size_t n = std::min(params_.block_size, size - off);
        block_meta m{store_.size(), 0, 0, true};
        bool use_raw = true;
        if (params_.codec != LLAMA_CRAM_CODEC_RAW) {
            encode_block(src + off, n, params_.codec, params_.stride, enc);
            // admission rule: must shrink by >= 5%
            if (enc.size() * 100 <= n * 95) use_raw = false;
        }
        if (use_raw) {
            store_.insert(store_.end(), src + off, src + off + n);
            m.csize = (uint32_t) n; m.raw = true;
        } else {
            store_.insert(store_.end(), enc.begin(), enc.end());
            m.csize = (uint32_t) enc.size(); m.raw = false;
        }
        m.crc = crc32(store_.data() + m.off, m.csize);
        meta_.push_back(m);
        st_.compressed_bytes += m.csize;
        st_.raw_blocks += m.raw;
    }
    st_.raw_bytes = size; st_.blocks = meta_.size(); st_.meta_bytes = meta_.size() * sizeof(block_meta);
    store_.shrink_to_fit();
}

size_t llama_cram_soft_store::block_len(size_t blk) const {
    return std::min(params_.block_size, size_ - blk * params_.block_size);
}

bool llama_cram_soft_store::decode_block_nolock(size_t blk, uint8_t * dst) const noexcept {
    if (blk >= meta_.size()) return false;
    const block_meta & m = meta_[blk];
    const uint8_t * s = store_.data() + m.off;
    if (params_.verify && crc32(s, m.csize) != m.crc) return false;
    size_t n = block_len(blk);
    if (m.raw) { if (m.csize != n) return false; std::memcpy(dst, s, n); return true; }
    return decode_block(s, m.csize, dst, n);
}

std::shared_ptr<std::vector<uint8_t>> llama_cram_soft_store::get_block(size_t blk) {
    shard & sh = *shards_[blk % shards_.size()];
    std::lock_guard<std::mutex> lk(sh.mu);
    auto it = sh.map.find(blk);
    if (it != sh.map.end()) {
        sh.lru.splice(sh.lru.begin(), sh.lru, it->second.it);
        hits_++;
        return it->second.data;
    }
    misses_++;
    auto buf = std::make_shared<std::vector<uint8_t>>(block_len(blk));
    uint64_t t0 = now_ns();
    bool ok = decode_block_nolock(blk, buf->data());
    decode_ns_ += now_ns() - t0;
    if (!ok) {
        crc_fail_++;
        throw std::runtime_error("llama_cram_soft: block " + std::to_string(blk) + " failed checksum or decode");
    }
    sh.lru.push_front(blk);
    sh.map[blk] = {sh.lru.begin(), buf};
    while (sh.map.size() > shard_cap_) {
        size_t victim = sh.lru.back();
        sh.lru.pop_back();
        sh.map.erase(victim);
    }
    return buf;
}

size_t llama_cram_soft_store::read(size_t offset, size_t len, void * dst) {
    if (offset >= size_) return 0;
    len = std::min(len, size_ - offset);
    uint8_t * d = (uint8_t *) dst;
    size_t done = 0;
    while (done < len) {
        size_t pos = offset + done;
        size_t blk = pos / params_.block_size, in = pos % params_.block_size;
        auto b = get_block(blk);
        size_t c = std::min(len - done, b->size() - in);
        std::memcpy(d + done, b->data() + in, c);
        done += c;
    }
    return done;
}

std::vector<uint8_t> llama_cram_soft_store::promote(size_t offset, size_t len) {
    if (offset >= size_) return {};
    len = std::min(len, size_ - offset);
    std::vector<uint8_t> v(len);
    read(offset, len, v.data());
    return v;
}

llama_cram_soft_stats llama_cram_soft_store::stats() const {
    llama_cram_soft_stats s = st_;
    s.hits = hits_; s.misses = misses_; s.decode_ns = decode_ns_; s.crc_failures = crc_fail_;
    return s;
}

std::vector<size_t> llama_cram_soft_store::hot_blocks_mru_first() const {
    shard & sh = *shards_[0];
    std::lock_guard<std::mutex> lk(sh.mu);
    return std::vector<size_t>(sh.lru.begin(), sh.lru.end());
}

uint8_t * llama_cram_soft_store::debug_stored_bytes_of_block(size_t blk, size_t * len) {
    if (blk >= meta_.size()) return nullptr;
    if (len) *len = meta_[blk].csize;
    return store_.data() + meta_[blk].off;
}
