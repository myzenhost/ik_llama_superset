#include "suffix-automaton.h"

#include <algorithm>

common_suffix_automaton::common_suffix_automaton(int max_refresh) : _max_refresh(std::max(0, max_refresh)) {
    clear();
}

void common_suffix_automaton::clear() {
    _text.clear();
    _len.assign(1, 0);
    _link.assign(1, -1);
    _last_end.assign(1, -1);
    _head.assign(1, -1);
    _edges.clear();
    _table.assign(1024, -1);
    _last = 0;
    _match_len = 0;
    _match_end = -1;
    ++_gen;
}

void common_suffix_automaton::advance(cursor & c, llama_token tok) const {
    if (c.gen != _gen) {            // made against another generation of the history: its state index means nothing here
        c.state = 0;
        c.len   = 0;
        c.gen   = _gen;
    }
    while (c.state != 0 && find_edge(c.state, tok) < 0) {
        c.state = _link[c.state];
        c.len   = _len[c.state];
    }
    const int e = find_edge(c.state, tok);
    if (e >= 0) {
        c.state = _edges[e].to;
        c.len++;
    } else {
        c.state = 0;
        c.len   = 0;
    }
}

int common_suffix_automaton::cursor_end(const cursor & c) const {
    if (c.gen != _gen || c.state <= 0 || c.state >= (int) _last_end.size()) {
        return -1;
    }
    return _last_end[c.state];
}

std::vector<llama_token> common_suffix_automaton::draft_after(const cursor & c, int max_tokens, int min_len) const {
    std::vector<llama_token> out;
    const int end = cursor_end(c);
    if (end < 0 || c.len < std::max(1, min_len) || max_tokens <= 0) {
        return out;
    }
    const int n = (int) _text.size();
    for (int src = end + 1; src < n && (int) out.size() < max_tokens; ++src) {
        out.push_back(_text[src]);
    }
    return out;
}

static inline uint64_t sa_hash(int state, llama_token tok) {
    uint64_t h = ((uint64_t) (uint32_t) state << 32) ^ (uint64_t) (uint32_t) tok;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
    return h;
}

int common_suffix_automaton::find_edge(int state, llama_token tok) const {
    const size_t mask = _table.size() - 1;
    for (size_t i = sa_hash(state, tok) & mask;; i = (i + 1) & mask) {
        const int32_t e = _table[i];
        if (e < 0) {
            return -1;
        }
        if (_edges[e].from == state && _edges[e].tok == tok) {
            return e;
        }
    }
}

void common_suffix_automaton::table_insert(int edge_index) {
    const size_t mask = _table.size() - 1;
    size_t i = sa_hash(_edges[edge_index].from, _edges[edge_index].tok) & mask;
    while (_table[i] >= 0) {
        i = (i + 1) & mask;
    }
    _table[i] = edge_index;
}

void common_suffix_automaton::table_rebuild(size_t cap) {
    _table.assign(cap, -1);
    for (size_t e = 0; e < _edges.size(); ++e) {
        table_insert((int) e);
    }
}

void common_suffix_automaton::set_edge(int state, llama_token tok, int to) {
    const int e = find_edge(state, tok);
    if (e >= 0) {
        _edges[e].to = to;     // retarget (clone redirection); the table key (from, tok) is unchanged
        return;
    }
    if ((_edges.size() + 1) * 2 > _table.size()) {
        table_rebuild(_table.size() * 2);
    }
    _edges.push_back({ tok, state, to, _head[state] });
    _head[state] = (int32_t) _edges.size() - 1;
    table_insert((int) _edges.size() - 1);
}

int common_suffix_automaton::new_state(int len, int link, int last_end) {
    _len.push_back(len);
    _link.push_back(link);
    _last_end.push_back(last_end);
    _head.push_back(-1);
    return (int) _len.size() - 1;
}

void common_suffix_automaton::extend(const llama_token * tokens, int n_tokens) {
    if (n_tokens > 0) {
        ++_gen;
    }
    for (int t = 0; t < n_tokens; ++t) {
        const llama_token c = tokens[t];
        const int pos = (int) _text.size();
        _text.push_back(c);

        const int cur = new_state(_len[_last] + 1, 0, pos);
        int p = _last;
        while (p != -1 && find_edge(p, c) < 0) {
            set_edge(p, c, cur);
            p = _link[p];
        }
        if (p == -1) {
            _link[cur] = 0;
        } else {
            const int q = _edges[find_edge(p, c)].to;
            if (_len[p] + 1 == _len[q]) {
                _link[cur] = q;
            } else {
                const int clone = new_state(_len[p] + 1, _link[q], _last_end[q]);
                for (int e = _head[q]; e >= 0; e = _edges[e].next) {
                    // copy before set_edge may grow _edges (the references below are re-read each time)
                    const llama_token et = _edges[e].tok;
                    const int         eto = _edges[e].to;
                    set_edge(clone, et, eto);
                }
                while (p != -1) {
                    const int e = find_edge(p, c);
                    if (e < 0 || _edges[e].to != q) {
                        break;
                    }
                    _edges[e].to = clone;
                    p = _link[p];
                }
                _link[q]   = clone;
                _link[cur] = clone;
            }
        }
        _last = cur;

        // The longest suffix of the history that ended here before is the string of link[cur]; its most recent earlier end position is the
        // value stored BEFORE this position is written into it.
        const int s = _link[cur];
        _match_len = _len[s];
        _match_end = s == 0 ? -1 : _last_end[s];
        int steps = 0;
        for (int x = s; x > 0 && _last_end[x] != pos && steps < _max_refresh; x = _link[x], ++steps) {
            _last_end[x] = pos;
        }
    }
}

std::vector<llama_token> common_suffix_automaton::speculate(int max_tokens, int min_match_len) const {
    std::vector<llama_token> out;
    if (_match_end < 0 || _match_len < std::max(1, min_match_len) || max_tokens <= 0) {
        return out;
    }
    const int n = (int) _text.size();
    out.reserve(max_tokens);
    for (int i = 0; i < max_tokens; ++i) {
        const int src = _match_end + 1 + i;
        out.push_back(src < n ? _text[src] : out[src - n]);     // past the end: copy from our own output (overlapping copy)
    }
    return out;
}

size_t common_suffix_automaton::bytes_used() const {
    return _text.capacity() * sizeof(llama_token) + (_len.capacity() + _link.capacity() + _last_end.capacity() + _head.capacity()) * sizeof(int32_t) +
           _edges.capacity() * sizeof(edge) + _table.capacity() * sizeof(int32_t);
}
