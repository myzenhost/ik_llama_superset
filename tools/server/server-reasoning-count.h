#pragma once

// Counts the generated tokens that belong to a reasoning (thinking) block, for the OpenAI usage fields
// completion_tokens_details.reasoning_tokens (chat completions) and output_tokens_details.reasoning_tokens
// (/v1/responses). Row S69-STRATA-S2-RESPONSES-API-PARITY-CUSTOM-NAMESPACE-TOOLS-INCOMPLETE-FAILED-USAGE-2026-10-07-01.
//
// WHY a separate counter and not the reasoning-budget sampler: that sampler only exists when a budget is set or a lazy
// grammar is active (common/sampling.cpp, "reasoning budget sampler" block), so it cannot be the source of a number the
// server reports on every response.
//
// RULES
//   - the start and end markers are the TOKEN sequences the server already builds for the budget sampler from the
//     chat template's thinking tags (params.sampling.reasoning_budget_start / reasoning_budget_end, set whenever the
//     template has thinking end tags)
//   - the generation prompt is replayed first without counting, so a template that leaves the block open
//     (generation prompt ending in "<think>\n") starts in the open state
//   - a token counts when it is generated while the block is open; the closing marker's tokens count (they are part of
//     the reasoning the model produced), and so do the opening marker's tokens when the model wrote them itself
//   - the block can open and close more than once (interleaved thinking); every block adds up
//   - no markers (a template without thinking) means the count stays 0
// SOURCE-READ: sglang b4584ab python/sglang/srt/managers/schedule_batch.py:2151-2170 (Apache-2.0) counts every output
// token up to and including the end sequence, assuming the block is open from the first token; behaviour read, no code
// copied. This version also handles a closed start, a model-written opener and repeated blocks.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

struct server_reasoning_counter {
    using tokens = std::vector<int32_t>;

    tokens              start;
    std::vector<tokens> ends;
    bool                open    = false;
    int32_t             n       = 0;   // reasoning tokens counted so far
    int32_t             n_gen   = 0;   // generated tokens seen (prefill excluded)

    // the last few tokens with a flag saying whether each was generated (prefill tokens never count)
    std::deque<std::pair<int32_t, bool>> window;
    size_t                               max_len = 0;

    void init(const tokens & start_seq, const std::vector<tokens> & end_seqs, const tokens & prefill) {
        start   = start_seq;
        ends.clear();
        for (const auto & e : end_seqs) {
            if (!e.empty()) {
                ends.push_back(e);
            }
        }
        open    = false;
        n       = 0;
        n_gen   = 0;
        window.clear();
        max_len = start.size();
        for (const auto & e : ends) {
            max_len = std::max(max_len, e.size());
        }
        for (const int32_t t : prefill) {
            step(t, false);
        }
    }

    bool enabled() const {
        return !start.empty() && !ends.empty();
    }

    void accept(int32_t tok) {
        ++n_gen;
        step(tok, true);
    }

  private:
    bool window_ends_with(const tokens & seq) const {
        if (seq.empty() || seq.size() > window.size()) {
            return false;
        }
        const size_t off = window.size() - seq.size();
        for (size_t i = 0; i < seq.size(); ++i) {
            if (window[off + i].first != seq[i]) {
                return false;
            }
        }
        return true;
    }

    void step(int32_t tok, bool generated) {
        if (!enabled()) {
            return;
        }
        window.emplace_back(tok, generated);
        while (window.size() > max_len) {
            window.pop_front();
        }
        if (open) {
            if (generated) {
                ++n;
            }
            for (const auto & e : ends) {
                if (window_ends_with(e)) {
                    open = false;
                    window.clear();   // the closer cannot also be the start of the next opener
                    break;
                }
            }
        } else if (window_ends_with(start)) {
            open = true;
            // the opener's tokens the model generated itself are reasoning too
            for (size_t i = window.size() - start.size(); i < window.size(); ++i) {
                if (window[i].second) {
                    ++n;
                }
            }
            window.clear();
        }
    }
};
