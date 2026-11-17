#pragma once
// Vocabulary-substring token ban. Spec: docs/agent-reports/koboldcpp-spec.md, feature A. Standard library only.

#include <cstddef>
#include <string>
#include <vector>

enum class token_ban_kind {
    DROP,    // empty after lowercasing, or the tokenizer produced nothing
    TOKEN,   // short single-token string: expand to every vocab entry containing it
    PHRASE,  // anything else: handled by text-level rewind
};

// ASCII-only lowercase; bytes >= 0x80 are left alone.
std::string token_ban_lower(const std::string & s);

// n_tokens = how many tokens the tokenizer yields for the lowercased string.
token_ban_kind token_ban_classify(const std::string & word, size_t n_tokens);

// Look-behind budget for a phrase of n_tokens tokens.
size_t token_ban_phrase_budget(size_t n_tokens);

// Ascending, duplicate-free ids of every vocab entry whose lowercased text contains any pattern (patterns are
// lowercased here). Empty patterns are ignored; empty entries are never banned.
std::vector<int> token_ban_scan(const std::vector<std::string> & pieces, const std::vector<std::string> & patterns);
