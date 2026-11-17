#include "token-ban.h"

#include <algorithm>

std::string token_ban_lower(const std::string & s) {
    std::string out = s;
    for (char & c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
    }
    return out;
}

token_ban_kind token_ban_classify(const std::string & word, size_t n_tokens) {
    const std::string w = token_ban_lower(word);
    if (w.empty() || n_tokens == 0) {
        return token_ban_kind::DROP;
    }
    if (n_tokens == 1 && w.size() < 12) {
        return token_ban_kind::TOKEN;
    }
    return token_ban_kind::PHRASE;
}

size_t token_ban_phrase_budget(size_t n_tokens) {
    return n_tokens + 3;
}

std::vector<int> token_ban_scan(const std::vector<std::string> & pieces, const std::vector<std::string> & patterns) {
    std::vector<std::string> pats;
    for (const auto & p : patterns) {
        std::string l = token_ban_lower(p);
        if (!l.empty()) {
            pats.push_back(std::move(l));
        }
    }
    std::vector<int> ids;
    if (pats.empty()) {
        return ids;
    }
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (pieces[i].empty()) {
            continue;
        }
        const std::string text = token_ban_lower(pieces[i]);
        for (const auto & p : pats) {
            if (text.find(p) != std::string::npos) {
                ids.push_back((int) i);
                break;
            }
        }
    }
    return ids;
}
