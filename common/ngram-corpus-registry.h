#pragma once

#include "suffix-automaton.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Named external reference corpora for lookup drafting (S69-ENGREV-SGLANG-NGRAM-EXTERNAL-CORPUS-2026-10-27-01).
//
// Behaviour spec: SGLang's ngram speculative corpus (python/sglang/srt/speculative/cpp_ngram/ngram_corpus.py add/remove/list of named
// corpora, ngram_corpus_ffi.cpp, ngram.cpp: one suffix automaton per corpus, a draft budget shared between the live context and the
// corpora, and an incremental match state that is reset when the corpus set changes). Written from that description on top of the fork's
// own common_suffix_automaton; Apache-2.0 reference, no code copied.
//
// What this is: the registry (add / append / remove / list, a total token budget), the budget split, and a matcher that holds a versioned
// incremental match state per corpus. What it is NOT: a speculative proposer. Nothing in common/speculative.cpp or the server calls it yet,
// so there is no CLI flag; wiring it as a --spec-type needs a corpus loader (file of token ids or text + tokenizer) and is a separate step.
class common_ngram_corpus_registry {
public:
    struct info {
        std::string name;
        int         tokens  = 0;
        uint64_t    version = 0;    // bumped on every append to this corpus
        size_t      bytes   = 0;
    };

    // max_total_tokens: 0 = unlimited. add() that would pass it is refused whole (no partial corpus).
    explicit common_ngram_corpus_registry(int max_total_tokens = 0) : _max_total(std::max(0, max_total_tokens)) {}

    bool add(const std::string & name, const llama_token * tokens, int n_tokens) {
        if (name.empty() || tokens == nullptr || n_tokens <= 0) {
            return false;
        }
        if (_max_total > 0 && total_tokens() + n_tokens > _max_total) {
            return false;
        }
        auto it = _corpora.find(name);
        if (it == _corpora.end()) {
            it = _corpora.emplace(name, entry{}).first;
        }
        it->second.sa.extend(tokens, n_tokens);
        it->second.version++;
        ++_version;
        return true;
    }

    bool remove(const std::string & name) {
        if (_corpora.erase(name) == 0) {
            return false;
        }
        ++_version;
        return true;
    }

    std::vector<info> list() const {
        std::vector<info> out;
        for (const auto & kv : _corpora) {
            out.push_back({ kv.first, kv.second.sa.token_count(), kv.second.version, kv.second.sa.bytes_used() });
        }
        return out;
    }

    int total_tokens() const {
        int n = 0;
        for (const auto & kv : _corpora) {
            n += kv.second.sa.token_count();
        }
        return n;
    }

    // changes on every add / remove; a matcher holding state from another version rebuilds it
    uint64_t version() const { return _version; }

    // Split a draft budget between corpora by match length: each corpus with a match gets at least 1 token while the budget lasts, the rest
    // goes in proportion to match length, and the remainder to the longest match. Result is parallel to match_lens, sums to <= total.
    static std::vector<int> split_budget(int total, const std::vector<int> & match_lens) {
        std::vector<int> out(match_lens.size(), 0);
        if (total <= 0) {
            return out;
        }
        std::vector<size_t> live;
        int64_t sum = 0;
        for (size_t i = 0; i < match_lens.size(); ++i) {
            if (match_lens[i] > 0) {
                live.push_back(i);
                sum += match_lens[i];
            }
        }
        if (live.empty()) {
            return out;
        }
        std::stable_sort(live.begin(), live.end(), [&](size_t a, size_t b) { return match_lens[a] > match_lens[b]; });
        if ((int) live.size() > total) {
            live.resize((size_t) total);                       // longest matches keep a slot
            sum = 0;
            for (size_t i : live) {
                sum += match_lens[i];
            }
        }
        int left = total;
        for (size_t i : live) {
            out[i] = 1;
            --left;
        }
        const int pool = left;
        for (size_t i : live) {
            const int share = (int) ((int64_t) pool * match_lens[i] / sum);
            out[i] += share;
            left   -= share;
        }
        out[live[0]] += left;
        return out;
    }

    struct draft_part {
        std::string              corpus;
        int                      match_len = 0;
        std::vector<llama_token> tokens;
    };

    // Per-request incremental match state. It remembers the context it has consumed and one cursor per corpus; the next propose() with a
    // context that extends it only feeds the new tokens. A different context prefix, or a registry whose version changed, rebuilds it.
    class matcher {
    public:
        std::vector<draft_part> propose(const common_ngram_corpus_registry & reg, const llama_token * ctx, int n_ctx, int max_tokens, int min_match) {
            const bool extends = _reg == &reg && _version == reg.version() && _consumed.size() <= (size_t) n_ctx &&
                                 (_consumed.empty() || std::memcmp(_consumed.data(), ctx, _consumed.size() * sizeof(llama_token)) == 0);
            if (!extends) {
                _cursors.clear();
                _consumed.clear();
                _reg     = &reg;
                _version = reg.version();
                ++_rebuilds;
            }
            const int start = (int) _consumed.size();
            for (const auto & kv : reg._corpora) {
                common_suffix_automaton::cursor & c = _cursors[kv.first];
                for (int i = start; i < n_ctx; ++i) {
                    kv.second.sa.advance(c, ctx[i]);
                }
            }
            _fed += n_ctx - start;
            _consumed.assign(ctx, ctx + n_ctx);

            std::vector<std::string> names;
            std::vector<int>         lens;
            for (const auto & kv : reg._corpora) {
                const common_suffix_automaton::cursor & c = _cursors[kv.first];
                names.push_back(kv.first);
                lens.push_back(c.len >= std::max(1, min_match) && kv.second.sa.cursor_end(c) >= 0 ? c.len : 0);
            }
            const std::vector<int> budget = split_budget(max_tokens, lens);
            std::vector<draft_part> out;
            for (size_t i = 0; i < names.size(); ++i) {
                if (budget[i] <= 0) {
                    continue;
                }
                const auto & sa = reg._corpora.at(names[i]).sa;
                draft_part p;
                p.corpus    = names[i];
                p.match_len = lens[i];
                p.tokens    = sa.draft_after(_cursors[names[i]], budget[i], min_match);
                if (!p.tokens.empty()) {
                    out.push_back(std::move(p));
                }
            }
            return out;
        }

        int  rebuilds() const { return _rebuilds; }     // times the state was thrown away
        long fed() const { return _fed; }               // context tokens walked through automata so far (per call, not per corpus)

    private:
        const common_ngram_corpus_registry *                _reg = nullptr;
        uint64_t                                            _version = 0;
        std::vector<llama_token>                            _consumed;
        std::map<std::string, common_suffix_automaton::cursor> _cursors;
        int                                                 _rebuilds = 0;
        long                                                _fed = 0;
    };

private:
    struct entry {
        entry() : sa(64) {}
        common_suffix_automaton sa;
        uint64_t                version = 0;
    };

    int                          _max_total;
    uint64_t                     _version = 0;
    std::map<std::string, entry> _corpora;
};
