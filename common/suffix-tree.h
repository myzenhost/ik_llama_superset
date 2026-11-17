#pragma once

#include "llama.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// A trie-based suffix tree for suffix-decoding speculative decoding.
//
// Stores all suffixes (up to max_depth) of the token history.
// Used to find matching patterns in context and generate draft tokens
// by following the most frequent continuation path.
//
// Reference: "Suffix Decoding" (Saxena et al., 2024) — arXiv:2411.04975

struct common_suffix_node {
    int64_t count = 0;
    std::unordered_map<llama_token, std::unique_ptr<common_suffix_node>> children;
};

class common_suffix_tree {
public:
    explicit common_suffix_tree(int max_depth = 64);
    ~common_suffix_tree();

    // Append tokens to the history and insert new suffixes into the trie.
    // Incremental: only processes suffixes that haven't been inserted yet.
    void extend(const llama_token * tokens, int n_tokens);

    void clear();

    // Bound the memory of the tree: once it holds more than `max_nodes` trie nodes (or keeps more than `max_nodes` history tokens) the OLDEST
    // history is dropped and the tree is rebuilt from the most recent tokens, so a long-lived server cannot grow without limit. 0 = unbounded
    // (the behaviour before 2026-11-07, measured at ~9.7 KB per history token at depth 64: common/suffix-automaton.h:16-17). Row
    // S69-ENGREV-SGLANG2-SUFFIX-TREE-UNBOUNDED-2026-10-27-01. SOURCE-READ: sglang kernels/jit/csrc/ngram_corpus/trie.cpp + param.h (a trie
    // with a node capacity and LRU eviction). Apache-2.0, behaviour only, no code copied.
    void   set_max_nodes(size_t max_nodes) { _max_nodes = max_nodes; }
    size_t max_nodes()   const { return _max_nodes; }
    size_t n_nodes()     const { return _n_nodes; }   // trie nodes alive (root excluded)
    size_t n_trims()     const { return _n_trims; }   // times the history was cut back to the cap

    // Generate draft tokens by matching the context in the trie.
    // Tries multiple context lengths and returns the draft with the best score.
    std::vector<llama_token> speculate(
            const llama_token * context, int n_context,
            int max_spec_tokens,
            float min_token_prob  = 0.1f,
            int   min_match_count = 1,
            int   min_match_len   = 5) const;

    // Load an offline corpus to pre-warm the tree before any request.
    // Supported formats (.json or .bin)
    bool load_corpus(
            const std::string & path,
            std::function<std::vector<llama_token>(const std::string &)> tokenize_fn = {});

    int  max_depth()   const { return _max_depth; }
    int  token_count() const { return (int)_tokens.size(); }

private:
    int _max_depth;
    std::unique_ptr<common_suffix_node> _root;
    std::vector<llama_token> _tokens;
    int _n_inserted = 0;
    size_t _max_nodes = 0;
    size_t _n_nodes   = 0;
    size_t _n_trims   = 0;

    void _append(const llama_token * tokens, int n_tokens);
    void _trim();
    void _insert_suffix(int start_pos);
    void _extend_suffix(int start_pos, int old_len, int new_len);
};
