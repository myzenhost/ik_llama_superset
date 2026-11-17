// Implementation of server_radix_cache -- see server-radix-cache.h for design/SOURCE-READ notes.

#include "server-radix-cache.h"

#include "log.h"

#include <algorithm>

server_radix_cache::server_radix_cache(llama_context * ctx_, llama_seq_id seq_id_base_, int32_t seq_id_count_)
    : ctx(ctx_), seq_id_base(seq_id_base_), seq_id_count(seq_id_count_) {
    root_for(0, true); // the no-adapter tree, present from the start as the single tree always was
    free_seq_ids.reserve(seq_id_count);
    for (int32_t i = seq_id_count - 1; i >= 0; --i) {
        free_seq_ids.push_back(seq_id_base + i);
    }
}

bool server_radix_evict_policy_parse(const std::string & name, server_radix_evict_policy & out) {
    static const struct { const char * name; server_radix_evict_policy p; } table[] = {
        {"lru", server_radix_evict_policy::LRU},   {"lfu", server_radix_evict_policy::LFU},
        {"fifo", server_radix_evict_policy::FIFO}, {"mru", server_radix_evict_policy::MRU},
        {"filo", server_radix_evict_policy::FILO}, {"slru", server_radix_evict_policy::SLRU},
        {"priority", server_radix_evict_policy::PRIORITY},
    };
    for (const auto & e : table) {
        if (name == e.name) {
            out = e.p;
            return true;
        }
    }
    return false;
}

const char * server_radix_evict_policy_name(server_radix_evict_policy p) {
    switch (p) {
        case server_radix_evict_policy::LRU:      return "lru";
        case server_radix_evict_policy::LFU:      return "lfu";
        case server_radix_evict_policy::FIFO:     return "fifo";
        case server_radix_evict_policy::MRU:      return "mru";
        case server_radix_evict_policy::FILO:     return "filo";
        case server_radix_evict_policy::SLRU:     return "slru";
        case server_radix_evict_policy::PRIORITY: return "priority";
    }
    return "lru";
}

bool server_radix_evict_before(server_radix_evict_policy p, uint32_t slru_hits, const server_radix_evict_info & a, const server_radix_evict_info & b) {
    // each case decides on its primary key, and falls through to the recency / creation tie-break only when that key is equal
    switch (p) {
        case server_radix_evict_policy::LRU:
            break;
        case server_radix_evict_policy::LFU:
            if (a.hits != b.hits) {
                return a.hits < b.hits;
            }
            break;
        case server_radix_evict_policy::FIFO:
            if (a.created != b.created) {
                return a.created < b.created;
            }
            break;
        case server_radix_evict_policy::MRU:
            if (a.last_use != b.last_use) {
                return a.last_use > b.last_use;
            }
            break;
        case server_radix_evict_policy::FILO:
            if (a.created != b.created) {
                return a.created > b.created;
            }
            break;
        case server_radix_evict_policy::SLRU: {
            const bool pa = a.hits >= slru_hits;
            const bool pb = b.hits >= slru_hits;
            if (pa != pb) {
                return !pa; // the probationary node goes first
            }
            break;
        }
        case server_radix_evict_policy::PRIORITY:
            if (a.priority != b.priority) {
                return a.priority > b.priority; // larger value = less important = evicted first
            }
            break;
    }
    if (a.last_use != b.last_use) {
        return a.last_use < b.last_use;
    }
    return a.created < b.created;
}

void server_radix_cache::set_evict_policy(server_radix_evict_policy p, uint32_t slru_hits_) {
    evict_policy = p;
    slru_hits    = slru_hits_ < 1 ? 1 : slru_hits_;
}

static server_radix_evict_info info_of(const server_radix_node * n) {
    return { n->last_use, n->created, n->hit_count, n->priority };
}

server_radix_node * server_radix_cache::root_for(uint64_t lora_fp, bool create) {
    auto it = roots.find(lora_fp);
    if (it != roots.end()) {
        return it->second.get();
    }
    if (!create) {
        return nullptr;
    }
    auto root = std::make_unique<server_radix_node>();
    root->pos_start = 0;
    root->seq_id    = -1; // a root owns no cells of its own
    server_radix_node * ptr = root.get();
    roots.emplace(lora_fp, std::move(root));
    return ptr;
}

static size_t common_len(const llama_tokens & a, const llama_tokens & b, size_t b_off) {
    size_t n = 0;
    const size_t max_n = std::min(a.size(), b.size() - b_off);
    while (n < max_n && a[n] == b[b_off + n]) {
        ++n;
    }
    return n;
}

size_t server_radix_cache::match_prefix(const llama_tokens & tokens, uint64_t lora_fp) {
    locked.clear();

    server_radix_node * cur = root_for(lora_fp, false);
    if (cur == nullptr) {
        stats.misses++; // nothing was ever inserted under this LoRA configuration
        return 0;
    }
    size_t pos = 0;

    while (pos < tokens.size()) {
        server_radix_node * next = nullptr;
        for (auto & c : cur->children) {
            if (!c->tokens.empty() && c->tokens[0] == tokens[pos]) {
                next = c.get();
                break;
            }
        }
        if (next == nullptr) {
            break;
        }

        const size_t l = common_len(next->tokens, tokens, pos);
        if (l < next->tokens.size()) {
            // partial edge match: the tree can only offer `l` more tokens from this edge
            next->ref++;
            next->last_use = ++clock;
            locked.push_back(next);
            pos += l;
            break;
        }

        next->ref++;
        next->last_use = ++clock;
        locked.push_back(next);
        cur = next;
        pos += l;
    }

    if (pos == 0) {
        stats.misses++;
        locked.clear(); // nothing was actually locked (loop broke before any push_back)
        return 0;
    }

    stats.hits++;
    stats.hit_tokens += pos;
    return pos;
}

size_t server_radix_cache::purge_path_of(const llama_tokens & tokens, uint64_t lora_fp) {
    server_radix_node * cur = root_for(lora_fp, false);
    if (cur == nullptr) {
        return 0;
    }
    std::vector<server_radix_node *> path;
    size_t pos = 0;
    while (pos < tokens.size()) {
        server_radix_node * next = nullptr;
        for (auto & c : cur->children) {
            if (!c->tokens.empty() && c->tokens[0] == tokens[pos]) {
                next = c.get();
                break;
            }
        }
        if (next == nullptr) {
            break;
        }
        const size_t l = common_len(next->tokens, tokens, pos);
        path.push_back(next);
        pos += l;
        if (l < next->tokens.size()) {
            break;
        }
        cur = next;
    }
    size_t n_removed = 0;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        server_radix_node * n = *it;
        if (!n->is_leaf() || n->ref != 0) {
            break;
        }
        remove_node(n);
        n_removed++;
    }
    return n_removed;
}

size_t server_radix_cache::peek_prefix(const llama_tokens & tokens, uint64_t lora_fp) const {
    const auto it = roots.find(lora_fp);
    if (it == roots.end()) {
        return 0;
    }
    const server_radix_node * cur = it->second.get();
    size_t pos = 0;
    while (pos < tokens.size()) {
        const server_radix_node * next = nullptr;
        for (const auto & c : cur->children) {
            if (!c->tokens.empty() && c->tokens[0] == tokens[pos]) {
                next = c.get();
                break;
            }
        }
        if (next == nullptr) {
            break;
        }
        const size_t l = common_len(next->tokens, tokens, pos);
        pos += l;
        if (l < next->tokens.size()) {
            break;
        }
        cur = next;
    }
    return pos;
}

void server_radix_cache::commit(llama_seq_id dst_seq, size_t n) {
    llama_memory_t mem = llama_get_memory(ctx);
    for (auto * node : locked) {
        const llama_pos p0 = node->pos_start;
        const llama_pos p1 = std::min<llama_pos>(node->pos_end(), (llama_pos) n);
        if (p1 > p0) {
            llama_memory_seq_cp(mem, node->seq_id, dst_seq, p0, p1);
        }
        node->ref--;
    }
    locked.clear();
}

void server_radix_cache::release() {
    for (auto * node : locked) {
        node->ref--;
    }
    locked.clear();
}

llama_seq_id server_radix_cache::claim_seq_id() {
    if (free_seq_ids.empty()) {
        if (!evict_one_idle_leaf()) {
            return -1;
        }
    }
    if (free_seq_ids.empty()) {
        return -1; // eviction freed a node but not a seq id (shouldn't happen; defensive)
    }
    const llama_seq_id id = free_seq_ids.back();
    free_seq_ids.pop_back();
    return id;
}

bool server_radix_cache::evict_one_idle_leaf() {
    server_radix_node * victim = nullptr;

    // DFS over the whole tree; the reserved pool is sized in the dozens (--radix-cache-slots),
    // never large enough for this linear scan to matter.
    std::vector<server_radix_node *> stack;
    for (auto & r : roots) {
        for (auto & c : r.second->children) {
            stack.push_back(c.get());
        }
    }
    while (!stack.empty()) {
        server_radix_node * n = stack.back();
        stack.pop_back();
        if (n->is_leaf()) {
            if (n->ref == 0 && (victim == nullptr || server_radix_evict_before(evict_policy, slru_hits, info_of(n), info_of(victim)))) {
                victim = n;
            }
        } else {
            for (auto & c : n->children) {
                stack.push_back(c.get());
            }
        }
    }

    if (victim == nullptr) {
        return false;
    }

    // always on (INFO), like the other cache decisions the server logs: the policy that chose this leaf and the numbers it chose on
    LOG_INF("radix cache: evicted idle leaf by policy %s (hits %llu, priority %d, last_use %llu, created %llu, %zu token(s))\n",
            server_radix_evict_policy_name(evict_policy), (unsigned long long) victim->hit_count, (int) victim->priority,
            (unsigned long long) victim->last_use, (unsigned long long) victim->created, victim->tokens.size());
    stats.evict_tokens += victim->tokens.size();
    remove_node(victim);
    stats.evictions++;
    return true;
}

void server_radix_cache::remove_node(server_radix_node * node) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, node->seq_id, -1, -1);
    free_seq_ids.push_back(node->seq_id);

    server_radix_node * parent = node->parent;
    if (parent != nullptr) {
        auto & sibs = parent->children;
        sibs.erase(std::remove_if(sibs.begin(), sibs.end(),
                                   [&](const std::unique_ptr<server_radix_node> & p) { return p.get() == node; }),
                   sibs.end());
    }
}

server_radix_node * server_radix_cache::split(server_radix_node * node, size_t l) {
    llama_memory_t mem = llama_get_memory(ctx);

    const llama_seq_id mid_seq = claim_seq_id();
    if (mid_seq < 0) {
        // Pool exhausted even after eviction (rare: sized by --radix-cache-slots). Do NOT touch
        // `node` or its subtree -- a failed split must be a no-op, never destructive. The caller
        // (insert()) stops walking on a null return, leaving the remaining suffix uncached this
        // time; `node` and everything under it stays exactly as it was.
        return nullptr;
    }

    auto mid = std::make_unique<server_radix_node>();
    mid->tokens    = llama_tokens(node->tokens.begin(), node->tokens.begin() + l);
    mid->pos_start = node->pos_start;
    mid->parent    = node->parent;
    mid->hit_count = node->hit_count; // a split keeps the original's counters (the shared head was reached as often as the whole edge)
    mid->priority  = node->priority;
    mid->last_use  = node->last_use;
    mid->created   = ++clock;
    server_radix_node * mid_ptr = mid.get();
    mid->seq_id = mid_seq;

    // Head range [pos_start, pos_start+l): tag it with mid's fresh id too, then strip node's old
    // id from that range -- both same-stream metadata ops, zero-copy (src/llama-kv-cache.cpp:527-556).
    const llama_pos head_p0 = node->pos_start;
    const llama_pos head_p1 = node->pos_start + (llama_pos) l;
    llama_memory_seq_cp(mem, node->seq_id, mid->seq_id, head_p0, head_p1);
    llama_memory_seq_rm(mem, node->seq_id, head_p0, head_p1);

    // node becomes the tail: shrink its own edge label/pos_start; its seq_id now correctly covers
    // only the remaining range [pos_start+l, pos_end) since the head range was just stripped.
    node->tokens    = llama_tokens(node->tokens.begin() + l, node->tokens.end());
    node->pos_start = head_p1;

    // splice: parent -> mid -> node
    server_radix_node * parent = node->parent;
    auto & sibs = parent->children;
    std::unique_ptr<server_radix_node> node_owned;
    for (auto & s : sibs) {
        if (s.get() == node) {
            node_owned = std::move(s);
            s.reset(); // placeholder, erased below
            break;
        }
    }
    sibs.erase(std::remove_if(sibs.begin(), sibs.end(),
                               [](const std::unique_ptr<server_radix_node> & p) { return p == nullptr; }),
               sibs.end());

    node->parent = mid_ptr;
    mid_ptr->children.push_back(std::move(node_owned));
    sibs.push_back(std::move(mid));

    return mid_ptr;
}

void server_radix_cache::insert(const llama_tokens & tokens, llama_seq_id src_seq, uint64_t lora_fp, int32_t priority) {
    if (tokens.empty()) {
        return;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    server_radix_node * cur = root_for(lora_fp, true);
    size_t pos = 0;
    bool grew = false;

    while (pos < tokens.size()) {
        server_radix_node * match = nullptr;
        for (auto & c : cur->children) {
            if (!c->tokens.empty() && c->tokens[0] == tokens[pos]) {
                match = c.get();
                break;
            }
        }

        if (match == nullptr) {
            const llama_seq_id id = claim_seq_id();
            if (id < 0) {
                // pool exhausted and nothing evictable -- stop here, leave the remaining suffix
                // uncached (pure miss next time, not a correctness issue).
                break;
            }

            auto node = std::make_unique<server_radix_node>();
            node->tokens    = llama_tokens(tokens.begin() + pos, tokens.end());
            node->pos_start = (llama_pos) pos;
            node->seq_id    = id;
            node->parent    = cur;
            node->last_use  = ++clock;
            node->created   = node->last_use;
            node->hit_count = 1;
            node->priority  = priority;

            llama_memory_seq_cp(mem, src_seq, id, (llama_pos) pos, (llama_pos) tokens.size());

            cur->children.push_back(std::move(node));
            grew = true;
            pos = tokens.size();
            break;
        }

        const size_t l = common_len(match->tokens, tokens, pos);
        if (l < match->tokens.size()) {
            server_radix_node * mid = split(match, l);
            if (mid == nullptr) {
                // split couldn't get a seq id even after eviction; stop, leave the remainder
                // uncached this time (match/node subtree is untouched, per split()'s contract).
                break;
            }
            cur = mid;
        } else {
            cur = match;
        }
        cur->last_use = ++clock;
        cur->hit_count++;
        cur->priority = std::min(cur->priority, priority);
        pos += l;
    }

    if (grew) {
        stats.inserts++;
    }
}

size_t server_radix_cache::n_nodes() const {
    size_t n = 0;
    std::vector<server_radix_node *> stack;
    for (auto & r : roots) {
        for (auto & c : r.second->children) {
            stack.push_back(c.get());
        }
    }
    while (!stack.empty()) {
        server_radix_node * cur = stack.back();
        stack.pop_back();
        ++n;
        for (auto & c : cur->children) {
            stack.push_back(c.get());
        }
    }
    return n;
}

void server_radix_cache::clear() {
    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<server_radix_node *> stack;
    for (auto & r : roots) {
        for (auto & c : r.second->children) {
            stack.push_back(c.get());
        }
    }
    while (!stack.empty()) {
        server_radix_node * cur = stack.back();
        stack.pop_back();
        for (auto & c : cur->children) {
            stack.push_back(c.get());
        }
        llama_memory_seq_rm(mem, cur->seq_id, -1, -1);
    }
    roots.clear();
    root_for(0, true);
    free_seq_ids.clear();
    free_seq_ids.reserve(seq_id_count);
    for (int32_t i = seq_id_count - 1; i >= 0; --i) {
        free_seq_ids.push_back(seq_id_base + i);
    }
    locked.clear();
}
