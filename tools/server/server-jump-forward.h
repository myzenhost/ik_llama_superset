#pragma once

// Grammar jump-forward: choose the tokens to force when a grammar leaves no choice about the next text.
// Row S69-MLC-LLM-PICK-GRAMMAR-JUMP-FORWARD-2026-10-15-01.
//
// SOURCE-READ: mlc-ai/mlc-llm cpp/serve/engine_actions/batch_jump_forward.cc (Apache-2.0; copy kept at
// temp/serving-picks-1016/mlc_batch_jumpforward.cc). Lines 69-78: ask the grammar matcher for the string it forces,
// retokenise "last <=10 committed tokens + that string", roll back the tokens that changed. Lines 155-166: if the LAST new
// token is a strict prefix of another token it is dropped (it would usually be rolled back by the next decode, which
// disturbs the distribution). Lines 113-121: skipped when logprobs are asked for. Behaviour reimplemented, no code copied.
//
// This tree does NOT roll back (the cache is append-only here, and a rolled-back sampled token has already been streamed):
// the forced string is tokenised ON ITS OWN and only appended. The cost is that the tokens can differ from what the
// tokenizer would give for the whole text, so a model may see a split it would not have produced itself; the TEXT is
// exactly the grammar's forced text. That is why the feature is off by default and measured against the plain run.
//
// Pure functions over the vocabulary's token texts, no llama.h: tested standalone by tests_cpp/test_server_jump_forward.cpp.

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace server_jump_forward {

struct vocab_index {
    std::vector<std::string>                    pieces;     // text of every token id (control tokens must be passed as "")
    std::unordered_map<std::string, int>        by_piece;   // text -> lowest token id; empty texts are not indexed
    std::unordered_set<std::string>             prefix_of_other; // texts that are a strict prefix of another token's text
    size_t                                      max_len = 0;

    void build(std::vector<std::string> p) {
        pieces = std::move(p);
        by_piece.clear();
        prefix_of_other.clear();
        max_len = 0;
        std::vector<const std::string *> sorted;
        sorted.reserve(pieces.size());
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (pieces[i].empty()) {
                continue;
            }
            by_piece.emplace(pieces[i], (int) i); // emplace keeps the first (= lowest id) on a duplicate text
            max_len = std::max(max_len, pieces[i].size());
            sorted.push_back(&pieces[i]);
        }
        std::sort(sorted.begin(), sorted.end(), [](const std::string * a, const std::string * b) { return *a < *b; });
        // in sorted order every string that has s as a prefix sits right after s (the block s, s+x, ...), so the
        // next DISTINCT text decides whether s is a strict prefix of anything
        for (size_t k = 0; k < sorted.size(); ++k) {
            size_t j = k + 1;
            while (j < sorted.size() && *sorted[j] == *sorted[k]) {
                ++j;
            }
            if (j < sorted.size() && sorted[j]->size() > sorted[k]->size() &&
                    sorted[j]->compare(0, sorted[k]->size(), *sorted[k]) == 0) {
                prefix_of_other.insert(*sorted[k]);
            }
        }
    }

    bool is_prefix_token(int id) const {
        return id >= 0 && (size_t) id < pieces.size() && prefix_of_other.count(pieces[(size_t) id]) != 0;
    }
};

inline std::string join(const vocab_index & vi, const std::vector<int> & toks) {
    std::string s;
    for (int t : toks) {
        if (t >= 0 && (size_t) t < vi.pieces.size()) {
            s += vi.pieces[(size_t) t];
        }
    }
    return s;
}

// Longest-match tokenisation of s over the vocabulary's token texts. False when some position matches no token.
inline bool greedy_tokenize(const vocab_index & vi, const std::string & s, std::vector<int> & out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        const size_t room = std::min(vi.max_len, s.size() - i);
        int found = -1;
        size_t len = 0;
        for (size_t l = room; l >= 1; --l) {
            const auto it = vi.by_piece.find(s.substr(i, l));
            if (it != vi.by_piece.end()) {
                found = it->second;
                len   = l;
                break;
            }
        }
        if (found < 0) {
            return false;
        }
        out.push_back(found);
        i += len;
    }
    return true;
}

struct plan {
    std::vector<int> tokens; // empty = do not jump
    std::string      text;   // the text those tokens spell
};

// forced:    the text the grammar forces (llama_grammar_forced_string)
// canonical: the tokenizer's own tokens for `forced` (may be empty/wrong for a vocabulary that adds a leading space)
// A token list is used only if its text IS the forced text, the tokenizer's list first, the longest-match list otherwise.
// At most max_tokens are kept, then the last token is dropped once if it is a strict prefix of another token (MLC rule),
// so the sampler can still pick the longer token next step.
inline plan make_plan(const vocab_index & vi, const std::string & forced, const std::vector<int> & canonical, size_t max_tokens) {
    plan p;
    if (forced.empty() || max_tokens == 0) {
        return p;
    }
    std::vector<int> toks;
    if (!canonical.empty() && join(vi, canonical) == forced) {
        toks = canonical;
    } else if (!greedy_tokenize(vi, forced, toks)) {
        return p;
    }
    if (toks.size() > max_tokens) {
        toks.resize(max_tokens);
    }
    if (!toks.empty() && vi.is_prefix_token(toks.back())) {
        toks.pop_back();
    }
    if (toks.empty()) {
        return p;
    }
    p.tokens = std::move(toks);
    p.text   = join(vi, p.tokens);
    return p;
}

} // namespace server_jump_forward
