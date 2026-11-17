#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Suffix automaton over a growing token history, for self-speculative ("suffix") drafting.
//
// S69-TRTLLM-PICK-SUFFIXAUTOMATON-2026-09-28-01. Behaviour spec: TensorRT-LLM's suffixAutomaton (Apache-2.0, adapted there from Baseten
// sa_spec): index the generated + prompt tokens online, find the LONGEST suffix of the history that occurred earlier, and propose the
// tokens that followed that earlier occurrence. Written from that description and the textbook construction (Blumer et al. 1985); no code
// copied.
//
// Why next to common_suffix_tree: the tree stores every suffix of the history to a fixed depth, which MEASURED costs about 9.7 KB per token
// of history (1.2 GB at 131072 tokens, 20 us per token to extend; bench_suffix_tree, 2026-10-05). The automaton has at most 2n states and
// 3n edges, and a match is not capped at a depth.
//
//   * extend() is amortised O(1) per token.
//   * The suffix link of the newest state is the longest suffix that already occurred, so the match is known the moment a token is added;
//     last_end[] keeps the most recent end position seen for each state, refreshed along the link chain (bounded by max_refresh, any
//     value kept is still a true occurrence, only possibly not the latest).
//   * speculate() copies from after the earlier occurrence; when the copy runs into the current end of the history it continues from its
//     own output (an LZ77-style overlapping copy), so a periodic history keeps drafting.
class common_suffix_automaton {
public:
    explicit common_suffix_automaton(int max_refresh = 64);

    void clear();
    void extend(const llama_token * tokens, int n_tokens);

    int token_count() const { return (int) _text.size(); }
    int state_count() const { return (int) _len.size(); }

    // length of the longest suffix of the indexed history that also ends earlier (0 = none), and the index of the LAST token of the
    // most recent earlier occurrence that was tracked (-1 = none)
    int match_len() const { return _match_len; }
    int match_end() const { return _match_end; }

    // up to max_tokens draft tokens (empty if the match is shorter than min_match_len)
    std::vector<llama_token> speculate(int max_tokens, int min_match_len) const;

    size_t bytes_used() const;

    // Read-only match cursor for a STATIC reference corpus (S69-ENGREV-SGLANG-NGRAM-EXTERNAL-CORPUS-2026-10-27-01). advance() feeds one
    // context token and keeps the longest suffix of the fed context that is a substring of the indexed history (len), with the automaton
    // state that recognises it. A cursor is only valid for the generation() it was made against; extend()/clear() bump the generation.
    struct cursor {
        int32_t  state = 0;
        int32_t  len   = 0;
        uint64_t gen   = 0;
    };
    uint64_t generation() const { return _gen; }
    void     advance(cursor & c, llama_token tok) const;
    // end index of a recorded occurrence of the cursor's string (-1 = none)
    int      cursor_end(const cursor & c) const;
    // up to max_tokens tokens that followed that occurrence; never past the end of the history (no overlapping copy), empty below min_len
    std::vector<llama_token> draft_after(const cursor & c, int max_tokens, int min_len) const;

private:
    struct edge {
        llama_token tok;
        int32_t     from;
        int32_t     to;
        int32_t     next;   // next edge of the same state, -1 = end (the list is only walked to copy a state when it is cloned)
    };

    int  find_edge(int state, llama_token tok) const;     // edge index or -1
    void set_edge(int state, llama_token tok, int to);    // add or retarget
    void table_insert(int edge_index);
    void table_rebuild(size_t cap);
    int  new_state(int len, int link, int last_end);

    int _max_refresh;
    int _last = 0;
    int _match_len = 0;
    int _match_end = -1;
    uint64_t _gen = 0;

    std::vector<llama_token> _text;
    std::vector<int32_t> _len, _link, _last_end, _head;
    std::vector<edge>    _edges;
    std::vector<int32_t> _table;                           // open addressing: edge index or -1
};
