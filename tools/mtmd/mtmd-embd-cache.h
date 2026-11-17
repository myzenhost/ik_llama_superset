// Byte-budgeted LRU cache of vision-encoder outputs (--mmproj-embd-cache MiB), keyed by a content hash of the preprocessed image.
//
// Re-running the vision tower for an image that was already encoded is the most expensive repeated step of a multimodal chat: the
// slot KV cache only helps while the same tokens precede the image in the same slot. This cache keeps the encoder OUTPUT (the
// embedding rows) so a returning image skips the tower whatever precedes it, which servers with several slots, an evicted slot, or a
// model class where the prefix/radix cache is gated off need.
//
// Rules:
//   - The key is a 128-bit hash of everything that decides the encoder output (the preprocessed pixels of every entry of the chunk, the
//     per-entry flags and tile geometry, the token count) plus the float count; equal content gives an equal key, so the same image sent
//     as a new upload or under a different URL is a hit and an edited image under the same name is a miss.
//   - The budget is in BYTES of the stored floats, not a count: a high-resolution image and a thumbnail are different sizes.
//   - Least-recently-used entries are evicted to make room, EXCEPT entries the current request already used (its "scope"): a request
//     with more images than the budget holds fills the cache as far as the budget allows and the rest is simply not cached, instead of
//     every image evicting the one before it and turning the whole cache into misses on the next turn.
//   - An entry bigger than the whole budget is not stored and evicts nothing. A budget of 0 stores nothing: today's behaviour.
//
// Header-only, model-free; tested by tests/test-mtmd-embd-cache.cpp.
//
// SOURCE-READ: the vision.py helper under backends/ in the local tabbyAPI clone, upstream/tabbyAPI, lines 19-165
// (theroyallab/tabbyAPI be74bf0a, AGPL-3.0). Read for behaviour, spec in my own words, own structure: no code copied.
// SOURCE-NUANCE (what the row's description did not say):
//   (1) the source keys by a 16-byte blake2b of the image URL and checks URL equality on a hit: an image served from the same URL
//       with new content would be a stale hit there; this port keys by content, so it cannot be stale and needs no URL at all.
//   (2) its put() of a key that is already cached REMOVES the entry first and may then fail to re-store it (an eviction-protected
//       neighbour blocks the room), losing a good entry; here a present key is only refreshed and protected.
//   (3) the source returns shared tensors that the caller keeps using after the call, which is why eviction needs a protected set at
//       all; this port copies the rows out at once, so the protected set exists only to stop one request from thrashing the cache.
//   (4) the source counts deepstack tensors in an entry's size; here the encoder returns ONE vector per chunk that already carries
//       them (n_embd_out covers the deepstack channels), so the size is that vector's.
//   (5) the source defaults the budget to 1024 MB (on); this port defaults to 0 (off) until a model-level number exists.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <vector>

struct mtmd_embd_key {
    uint64_t h1 = 0;
    uint64_t h2 = 0;
    uint64_t n_floats = 0; // part of the key: two different outputs can never share a size by accident

    bool operator<(const mtmd_embd_key & o) const {
        if (h1 != o.h1) return h1 < o.h1;
        if (h2 != o.h2) return h2 < o.h2;
        return n_floats < o.n_floats;
    }
    bool operator==(const mtmd_embd_key & o) const { return h1 == o.h1 && h2 == o.h2 && n_floats == o.n_floats; }
};

// 128-bit content hash: two 64-bit lanes with different constants over 8-byte words, mixed with a splitmix64 finaliser.
class mtmd_embd_hasher {
  public:
    void add_u64(uint64_t v) {
        a = mix(a ^ v);
        b = mix(b + v * 0x9e3779b97f4a7c15ULL + 0x632be59bd9b4e019ULL);
        n_words++;
    }
    void add_bytes(const void * data, size_t len) {
        const unsigned char * p = static_cast<const unsigned char *>(data);
        size_t i = 0;
        for (; i + 8 <= len; i += 8) {
            uint64_t w;
            std::memcpy(&w, p + i, 8);
            add_u64(w);
        }
        uint64_t tail = 0;
        for (size_t k = 0; i + k < len; k++) {
            tail |= (uint64_t) p[i + k] << (8 * k);
        }
        add_u64(tail ^ ((uint64_t) len << 56)); // the length is folded in, so "ab" + "c" and "a" + "bc" differ
    }
    void add_floats(const float * f, size_t n) { add_bytes(f, n * sizeof(float)); }
    mtmd_embd_key finish(uint64_t n_floats) const {
        mtmd_embd_key k;
        k.h1       = mix(a ^ (n_words * 0xff51afd7ed558ccdULL));
        k.h2       = mix(b ^ (n_words * 0xc4ceb9fe1a85ec53ULL));
        k.n_floats = n_floats;
        return k;
    }

  private:
    static uint64_t mix(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    uint64_t a = 0x243f6a8885a308d3ULL;
    uint64_t b = 0x13198a2e03707344ULL;
    uint64_t n_words = 0;
};

struct mtmd_embd_cache_stats {
    uint64_t hits      = 0;
    uint64_t misses    = 0;
    uint64_t evictions = 0;
    uint64_t not_stored  = 0; // put() that stored nothing (over budget, or only protected entries stood in the way)
    uint64_t bytes     = 0;
    uint64_t entries   = 0;
    uint64_t budget    = 0;
};

class mtmd_embd_cache {
  public:
    explicit mtmd_embd_cache(size_t budget_bytes = 0) : budget(budget_bytes) {}

    bool enabled() const {
        std::lock_guard<std::mutex> lk(mtx);
        return budget > 0;
    }

    // changing the budget drops entries, oldest first, until the rest fits
    void set_budget(size_t budget_bytes) {
        std::lock_guard<std::mutex> lk(mtx);
        budget = budget_bytes;
        evict_until(budget, /*respect_protected=*/false);
    }

    // A new request begins: nothing is protected any more.
    void set_scope(uint64_t scope_id) {
        std::lock_guard<std::mutex> lk(mtx);
        if (!has_scope || scope_id != scope) {
            scope     = scope_id;
            has_scope = true;
            protected_keys.clear();
        }
    }

    // On a hit copies the rows into `out`, refreshes the entry and protects it for the current scope.
    bool get(const mtmd_embd_key & key, std::vector<float> & out) {
        std::lock_guard<std::mutex> lk(mtx);
        const auto it = index.find(key);
        if (it == index.end()) {
            st.misses++;
            return false;
        }
        lru.splice(lru.end(), lru, it->second); // most recently used
        out = it->second->rows;
        protected_keys.insert(key);
        st.hits++;
        return true;
    }

    // Stores a copy. Returns false (and stores nothing) when it does not fit: bigger than the budget, or room could only be made by
    // evicting an entry the current request already used.
    bool put(const mtmd_embd_key & key, const float * data, size_t n_floats) {
        std::lock_guard<std::mutex> lk(mtx);
        const size_t bytes = n_floats * sizeof(float);
        const auto it = index.find(key);
        if (it != index.end()) {
            lru.splice(lru.end(), lru, it->second); // already there: refresh only
            protected_keys.insert(key);
            return true;
        }
        if (budget == 0 || bytes > budget) {
            st.not_stored++;
            return false;
        }
        // decide BEFORE evicting: when the entries this request already used leave no room, nothing may be thrown away for nothing
        size_t protected_bytes = 0;
        for (const auto & e : lru) {
            if (protected_keys.count(e.key)) {
                protected_bytes += e.rows.size() * sizeof(float);
            }
        }
        if (protected_bytes + bytes > budget) {
            st.not_stored++;
            return false;
        }
        evict_until(budget - bytes, /*respect_protected=*/true);
        lru.push_back(entry{key, std::vector<float>(data, data + n_floats)});
        index[key] = std::prev(lru.end());
        used += bytes;
        protected_keys.insert(key);
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx);
        lru.clear();
        index.clear();
        protected_keys.clear();
        used = 0;
    }

    mtmd_embd_cache_stats stats() const {
        std::lock_guard<std::mutex> lk(mtx);
        mtmd_embd_cache_stats s = st;
        s.bytes   = used;
        s.entries = lru.size();
        s.budget  = budget;
        return s;
    }

  private:
    struct entry {
        mtmd_embd_key      key;
        std::vector<float> rows;
    };

    // drop least-recently-used entries until at most `target` bytes remain (entries protected for this scope are skipped)
    void evict_until(size_t target, bool respect_protected) {
        for (auto it = lru.begin(); it != lru.end() && used > target;) {
            if (respect_protected && protected_keys.count(it->key)) {
                ++it;
                continue;
            }
            used -= it->rows.size() * sizeof(float);
            index.erase(it->key);
            protected_keys.erase(it->key);
            it = lru.erase(it);
            st.evictions++;
        }
    }

    mutable std::mutex mtx;
    size_t   budget = 0;
    size_t   used   = 0;
    uint64_t scope  = 0;
    bool     has_scope = false;
    std::list<entry> lru; // front = least recently used
    std::map<mtmd_embd_key, std::list<entry>::iterator> index;
    std::set<mtmd_embd_key> protected_keys;
    mtmd_embd_cache_stats st;
};
