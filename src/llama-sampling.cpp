#include "llama-sampling.h"
#include "llama-vocab.h"
#include "llama-grammar.h"

#include "iqk/iqk_cpu_ops.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <string>

static void llama_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}

void llama_set_rng_seed_impl(struct llama_sampling * smpl, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }

    smpl->rng.seed(seed);
}

static void llama_sort(llama_token_data_array * candidates, int32_t k) {
    if (candidates->sorted || candidates->size < 2) {
        return;
    }
    if (k < 0) {
        k = candidates->size;
    }
    // Equal logits are ordered by the lower token id, so which tokens survive a top-k cut on an exact tie is
    // defined by us, not by the standard library's sort (MinGW/libstdc++ and MSVC order ties differently).
    // Tie-break rule from mistral.rs sampler.rs:563-569 (cmp_desc_by_prob), reimplemented.
    auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };
    if (k <= 1024) { //128) {
        if (k == int(candidates->size)) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
    } else {
        constexpr int   nbuckets     = 128;
        constexpr float bucket_low   = -10.0f;
        constexpr float bucket_high  =  10.0f;
        constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
        constexpr float bucker_inter = -bucket_low * bucket_scale;

        std::vector<int> bucket_idx(candidates->size);
        std::vector<int> histo(nbuckets, 0);

        for (int i = 0; i < (int)candidates->size; ++i) {
            const float val = candidates->data[i].logit;
            int ib = int(bucket_scale * val + bucker_inter); //nbuckets * (val - bucket_low) / (bucket_high - bucket_low);
            ib = std::max(0, std::min(nbuckets-1, ib));
            bucket_idx[i] = ib;
            ++histo[ib];
        }
        int nhave = 0;
        int ib = nbuckets - 1;
        for ( ; ib >= 0; --ib) {
            nhave += histo[ib];
            if (nhave >= k) break;
        }
        std::vector<llama_token_data> tmp_tokens(nhave);
        auto ptr = tmp_tokens.data();
        std::vector<llama_token_data*> bucket_ptrs;
        bucket_ptrs.reserve(nbuckets - ib);
        for (int j = nbuckets - 1; j >= ib; --j) {
            bucket_ptrs.push_back(ptr);
            ptr += histo[j];
        }
        for (int i = 0; i < (int)candidates->size; ++i) {
            int j = bucket_idx[i];
            if (j >= ib) {
                *bucket_ptrs[nbuckets-1-j]++ = candidates->data[i];
            }
        }

        ptr = tmp_tokens.data();
        int ndone = 0;
        for (int j = nbuckets-1; j > ib; --j) {
            std::sort(ptr, ptr + histo[j], comp);
            ptr += histo[j];
            ndone += histo[j];
        }
        std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);

        std::memcpy(candidates->data, tmp_tokens.data(), k*sizeof(llama_token_data));

    }
    candidates->sorted = true;
}

void llama_sample_softmax_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, bool normalize) {
    GGML_ASSERT(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order if necessary
    llama_sort(candidates, -1);

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    if (normalize) {
        for (size_t i = 0; i < candidates->size; ++i) {
            candidates->data[i].p /= cum_sum;
        }
    }


    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_k_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, int32_t k, size_t min_keep) {

    const int64_t t_start_sample_us = ggml_time_us();

    if (k <= 0) {
        k = candidates->size;
    }

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    llama_sort(candidates, k);

    candidates->size = k;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

// top_p over an UNSORTED candidate list without sorting all of it. Only tokens with probability >= p / 1024 are
// collected and sorted (llama_sort's order: logit descending, ties by lower id). Those tokens are a prefix of the full
// sorted order, so when their cumulative probability reaches p (with at least min_keep of them) the nucleus lies inside
// them and is the same token prefix the full sort finds. Lower cutoffs follow, then a full sort (see below). Returns
// false only when it cannot run (sorted input, fewer than 2 tokens, p <= 0, no finite logit); the caller's path runs.
// SOURCE-READ: mlc-ai/mlc-llm 9fa644f54b04983adea4d0168f49fc6af4a893ba cpp/serve/sampler/cpu_sampler.cc:84-162
// (SampleTopPFromProb: filter at top_p / 1024, sort the survivors, fall back to the full set when they do not reach
// top_p; Apache-2.0). Fresh code, no code copied.
// SOURCE-NUANCE: MLC works on probabilities and stops the filter scan early once the remaining mass is below the cutoff;
// here the input is logits, the normalising sum is taken first (in double, index order), and the scan always runs to the
// end (a float early exit can skip a qualifying token). MLC falls back to sorting everything after its one cutoff; here
// two lower cutoffs come first, each tested by a counting pass before anything is copied or sorted. The sorted path of
// llama_sample_top_p_impl normalises with a double sum too, so both paths agree (the double sums differ only in order).
// Row S69-MLC-LLM-PICK-TOPK-TOPP-WITHOUT-FULL-SORT-2026-10-18-01.
static bool llama_sample_top_p_unsorted(llama_token_data_array * candidates, float p, size_t min_keep) {
    const size_t n = candidates->size;
    if (candidates->sorted || n < 2 || !(p > 0.0f)) {
        return false;
    }
    llama_token_data * d = candidates->data;
    float max_l = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        max_l = std::max(max_l, d[i].logit);
    }
    if (!std::isfinite(max_l)) {
        return false;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float e = expf(d[i].logit - max_l);
        d[i].p = e;
        sum += e;
    }
    auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };
    // normalise a sorted run and find the smallest prefix with cumulative probability >= p and >= min_keep tokens
    auto cut_sorted = [&](llama_token_data * s, size_t m) -> size_t {
        float cum_sum = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            s[i].p = (float) (s[i].p / sum);
            cum_sum += s[i].p;
            if (cum_sum >= p && i + 1 >= min_keep) {
                return i + 1;
            }
        }
        return 0;
    };
    // MLC's single cutoff is p / 1024. A flat distribution (nucleus of thousands of tokens) does not reach p there, and
    // re-sorting everything after a failed try cost more than the plain sort (measured, test-sampling-nosort run 5:
    // 21.9 ms vs 18.1 ms). So each cutoff is tried with a counting pass first (no copy, no sort), three cutoffs 32x apart
    // and 1024x apart, and the last resort sorts the whole list in place with the exp terms already computed.
    static thread_local std::vector<llama_token_data> kept;
    for (const double frac : { 1.0 / 1024.0, 1.0 / 32768.0, 1.0 / 33554432.0 }) {
        const double cut   = (double) p * frac * sum;
        double       mass  = 0.0;
        size_t       count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (d[i].p >= cut) {
                mass += d[i].p;
                ++count;
            }
        }
        if (count < min_keep || mass < (double) p * sum) {
            continue; // the nucleus extends below this cutoff
        }
        kept.clear();
        kept.reserve(count);
        for (size_t i = 0; i < n; ++i) {
            if (d[i].p >= cut) {
                kept.push_back(d[i]);
            }
        }
        std::sort(kept.begin(), kept.end(), comp);
        const size_t last = cut_sorted(kept.data(), kept.size());
        if (last == 0) {
            continue; // float rounding of the running sum left it just short of p: try a lower cutoff
        }
        std::copy(kept.begin(), kept.begin() + last, d);
        candidates->size   = last;
        candidates->sorted = true;
        return true;
    }
    std::sort(d, d + n, comp);
    const size_t last = cut_sorted(d, n);
    candidates->size   = last == 0 ? n : last; // never reaching p keeps everything, as the sorted path does
    candidates->sorted = true;
    return true;
}

void llama_sample_top_p_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    if (!candidates->sorted) {
        const int64_t t_start_unsorted_us = ggml_time_us();
        if (llama_sample_top_p_unsorted(candidates, p, min_keep)) {
            if (smpl) {
                smpl->t_sample_us += ggml_time_us() - t_start_unsorted_us;
            }
            return;
        }
    }

    // sort, unnormalised exp terms in p; the normalising sum is taken in double, as llama_sample_top_p_unsorted does, so
    // both paths give the same probabilities and the same cut. A float sum of a 152k vocabulary, largest first, is off
    // by more than 1e-5 relative and moved the nucleus boundary by one token (measured in tests/test-sampling-nosort.cpp:
    // 165 tokens with the float sum, 166 with the exact one).
    llama_sample_softmax_impl(smpl, candidates, /* normalize */ false);

    const int64_t t_start_sample_us = ggml_time_us();

    double sum_e = 0.0;
    for (size_t i = 0; i < candidates->size; ++i) {
        sum_e += candidates->data[i].p;
    }

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p = (float) (candidates->data[i].p / sum_e);
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_min_p_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p <= 0.0f || !candidates->size) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    bool min_p_applied = false;

    // if the candidates aren't sorted, try the unsorted implementation first
    if (!candidates->sorted) {
        std::vector<llama_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < candidates->size; ++i) {
            max_logit = std::max(max_logit, candidates->data[i].logit);
        }
        const float min_logit = max_logit + logf(p); // min logit for p_i >= p * p_max

        for (size_t i = 0; i < candidates->size; ++i) {
            if (candidates->data[i].logit >= min_logit) {
                filtered_tokens.push_back(candidates->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (filtered_tokens.size() >= min_keep) {
            memcpy(candidates->data, filtered_tokens.data(), filtered_tokens.size()*sizeof(llama_token_data));
            candidates->size = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the candidates are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order if needed
        llama_sort(candidates, -1);

        const float min_logit = candidates->data[0].logit + logf(p); // min logit for p_i >= p * p_max
        size_t i = 1; // first token always matches

        for (; i < candidates->size; ++i) {
            if (candidates->data[i].logit < min_logit && i >= min_keep) {
                break; // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        candidates->size = i;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_tail_free_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);
    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = std::abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    if (min_keep < 1) min_keep = 1;
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_typical_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        shifted_scores[i] = fabsf(-logf(candidates->data[i].p) - entropy);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> new_candidates(last_idx);
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates[i] = candidates->data[idx];
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();
    candidates->sorted = false;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_entropy_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val) {
    const int64_t t_start_sample_us = ggml_time_us();

    // no need to do anything if there is only one (or zero) candidates
    if(candidates->size <= 1) {
        return;
    }

    // Calculate maximum possible entropy over the FINITE logits: a candidate masked to -inf (grammar, logit bias)
    // carries no probability, so counting it widens the denominator and squeezes the temperature toward min_temp.
    size_t n_finite = 0;
    for (size_t i = 0; i < candidates->size; ++i) {
        n_finite += std::isfinite(candidates->data[i].logit) ? 1 : 0;
    }
    if (n_finite <= 1) {
        return; // one live candidate: temperature cannot change the outcome (and log(1) = 0 would divide by zero)
    }
    float max_entropy = -logf(1.0f / n_finite);

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    // Calculate entropy of the softmax probabilities
    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float prob = candidates->data[i].p;
        if (prob > 0.0f) { // Ensure no log(0)
            entropy -= prob * logf(prob);
        }
    }

    // Normalize the entropy (max_entropy cannot be 0 here because we checked candidates->size != 1 above)
    float normalized_entropy = entropy / max_entropy;

    // Map the normalized entropy to the desired temperature range using the power function
    float dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent_val);

#ifdef DEBUG
    LLAMA_LOG_INFO("Your text maxtemp value is: %f\n", max_temp);
    LLAMA_LOG_INFO("Entropy: %f\n", entropy);
    LLAMA_LOG_INFO("Max Possible Entropy: %f\n", max_entropy);
    LLAMA_LOG_INFO("Normalized Entropy: %f\n", normalized_entropy);
    LLAMA_LOG_INFO("Exponent: %f\n", exponent_val);
    LLAMA_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", dyn_temp);
#endif

    // Apply the dynamically calculated temperature scaling
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].logit /= dyn_temp;
    }

    // Re-compute softmax probabilities after scaling logits with dynamic temperature
    double max_l_double = candidates->data[0].logit;
    double cum_sum_double = 0.0;
    for (size_t i = 0; i < candidates->size; ++i) {
        double p = exp(candidates->data[i].logit - max_l_double);
        candidates->data[i].p = p; // Store the scaled probability
        cum_sum_double += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum_double; // Re-normalize the probabilities
    }

#ifdef DEBUG
    // Print the updated top 25 probabilities after temperature scaling
    LLAMA_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
    for (size_t i = 0; i < 25 && i < candidates->size; ++i) {
        LLAMA_LOG_INFO("Token %zu: %f%%\n", i + 1, candidates->data[i].p * 100.0f);
    }
#endif

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temp_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].logit /= temp;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_xtc_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float probability, float threshold, size_t min_keep,
                           const struct llama_vocab * vocab) {
    if (probability <= 0 || threshold > 0.5f || candidates->size < 2) {
        return;
    }
    // End-of-generation tokens and tokens whose decoded piece contains a newline are exempt (neither counted as a crossing token nor cut):
    // without this, an EOG token that crosses the threshold together with another token can be removed and the run cannot stop, and
    // paragraph breaks get cut from the candidate list.
    // SOURCE-READ: upstream/exllamav2/exllamav2/generator/sampler.py:277-302 (default XTC mask = every piece containing a
    // newline plus the special/extended tokens, EOS included; MIT, behaviour only, no code copied).
    // Row: S69-EXLLAMAV2-PICK-XTC-EXEMPT-EOG-NEWLINE-2026-09-29-01.
    if (vocab) {
        const int64_t t_start_xtc_us = ggml_time_us();
        if (probability < 1) {
            std::uniform_real_distribution<float> distribution_e(0.0f, 1.0f);
            if (distribution_e(smpl->rng) > probability) return;
        }
        llama_sample_softmax_impl(nullptr, candidates);
        // crossing tokens are the leading run with p >= threshold (candidates sorted by p, descending)
        size_t n_cross = 0;
        while (n_cross < candidates->size && candidates->data[n_cross].p >= threshold) {
            ++n_cross;
        }
        // a token whose decoded piece contains a newline is exempt too (ExLlamaV2's default mask); only the few crossing candidates are decoded
        auto piece_has_newline = [&](llama_token id) {
            char buf[128];
            int32_t n = vocab->token_to_piece(id, buf, (int32_t) sizeof(buf), 0, true);
            if (n < 0) {
                std::vector<char> big((size_t) -n);
                n = vocab->token_to_piece(id, big.data(), (int32_t) big.size(), 0, true);
                return n > 0 && std::memchr(big.data(), '\n', (size_t) n) != nullptr;
            }
            return n > 0 && std::memchr(buf, '\n', (size_t) n) != nullptr;
        };
        std::vector<size_t> cut; // crossing, non-exempt tokens
        for (size_t i = 0; i < n_cross; ++i) {
            const llama_token id = candidates->data[i].id;
            if (!llama_token_is_eog(vocab, id) && !piece_has_newline(id)) {
                cut.push_back(i);
            }
        }
        // like the unguarded path: keep the least probable crossing token, remove the others (needs at least two non-exempt ones)
        if (cut.size() >= 2 && candidates->size - (cut.size() - 1) >= min_keep) {
            size_t w = 0;
            size_t next_cut = 0;
            for (size_t i = 0; i < candidates->size; ++i) {
                if (next_cut + 1 < cut.size() && i == cut[next_cut]) {
                    ++next_cut;      // drop this one
                    continue;
                }
                candidates->data[w++] = candidates->data[i];
            }
            candidates->size = w;
        }
        if (smpl) {
            smpl->t_sample_us += ggml_time_us() - t_start_xtc_us;
            smpl->n_sample++;
        }
        return;
    }
    GGML_ASSERT(smpl);
    const int64_t t_start_sample_us = ggml_time_us();
    if (probability < 1) {
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        float chance = distribution(smpl->rng);
        if (chance > probability) return;
    }

    llama_sample_softmax_impl(nullptr, candidates);

    int pos_last = 0;

    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].p >= threshold) {
            pos_last = i;
        } else break;
    }

    if (candidates->size - pos_last >= min_keep && pos_last > 0) {
        candidates->data += pos_last;
        candidates->size -= pos_last;
    }

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    smpl->n_sample++;

}

void llama_sample_top_n_sigma_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float top_n_sigma) {

    if (top_n_sigma <= 0.0f || candidates->size < 4) {
        // top_n_sigma <= 0: disabled
        // candidates->size < 4: no point in applying the transformation for fewer than 4 logits.
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    double sum1 = 0, sum2 = 0;
    float max = -INFINITY; // seeded below zero: all-negative logits must still find their true maximum
    int count = 0;
    for (int i = 0; i < (int)candidates->size; ++i) {
        if (auto l = candidates->data[i].logit; l != -INFINITY) {
            max = std::max(max, l);
            ++count;
            double dl = l;
            sum1 += dl;
            sum2 += dl*dl;
        }
    }
    if (count < 4) {
        return;
    }
    double dmean = sum1/count;
    double dsigma2 = sum2/count - dmean*dmean;
    if (dsigma2 <= 0) {
        return;
    }
    float sigma = float(sqrt(dsigma2));

    float thresh = max - top_n_sigma*sigma;
    int n_use = 0;
    for (int i = 0; i < (int)candidates->size; ++i) {
        if (candidates->data[i].logit >= thresh) {
            candidates->data[n_use++] = candidates->data[i];
        }
    }
    if (n_use < (int)candidates->size) {
        candidates->size = n_use;
        llama_sample_softmax_impl(nullptr, candidates);
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
    }
}


void llama_sample_repetition_penalties_impl(
        struct llama_sampling * smpl,
       llama_token_data_array * candidates,
            const llama_token * last_tokens,
                       size_t   penalty_last_n,
                       float   penalty_repeat,
                       float   penalty_freq,
                       float   penalty_present) {
    if (penalty_last_n == 0 || (penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_token, int> token_count;
    for (size_t i = 0; i < penalty_last_n; ++i) {
        token_count[last_tokens[i]]++;
    }

    auto penalize = [&](llama_token_data & cand, int count) {
        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (cand.logit <= 0) {
            cand.logit *= penalty_repeat;
        } else {
            cand.logit /= penalty_repeat;
        }

        cand.logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    };

    // ADOPTED from unslothai/llama.cpp branch `penalties-index-by-token-id`, commits e2e842a4 +
    // 3db8cb5b ("sampling : index penalties by token id instead of scanning every candidate" /
    // "require the full identity layout before indexing by token id"). Ported to this fork's
    // pre-refactor entry point (llama_sample_repetition_penalties_impl) -- upstream's
    // llama-sampler.cpp does not exist here, but the inner loop was character-identical.
    //
    // token_count holds at most penalty_last_n entries (typically 64), while candidates is the
    // whole vocabulary (151936 for Qwen3.8). The old loop did one hash probe PER CANDIDATE; this
    // walks the small map instead, turning O(n_vocab) probes into O(penalty_last_n) indexed
    // writes. The guard is one compare per candidate and stops at the first mismatch.
    //
    // SAFETY, and where this deliberately differs from upstream: upstream proves the ids are in
    // range via `cur_p->size == ctx->n_vocab`, a field this fork's sampler does not carry. So the
    // identity check below establishes data[i].id == i, and each map key is ALSO bounds-checked
    // before it indexes. A token id outside the candidate array would otherwise be an
    // out-of-bounds write -- the identity of the array alone does not bound the map's keys.
    bool by_index = true;

    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].id != (llama_token) i) {
            by_index = false;
            break;
        }
    }

    // Apply frequency and presence penalties to the candidates
    if (by_index) {
        for (const auto & it : token_count) {
            if (it.first >= 0 && (size_t) it.first < candidates->size) {
                penalize(candidates->data[it.first], it.second);
            }
        }
    } else {
        for (size_t i = 0; i < candidates->size; ++i) {
            const auto token_iter = token_count.find(candidates->data[i].id);
            if (token_iter == token_count.end()) {
                continue;
            }

            penalize(candidates->data[i], token_iter->second);
        }
    }

    candidates->sorted = false;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_apply_guidance_impl(
        struct llama_sampling * smpl,
                        float * logits,
                        float * logits_guidance,
                        float   scale) {
    GGML_ASSERT(smpl);

    const auto t_start_sample_us = ggml_time_us();
    const auto n_vocab = smpl->n_vocab;

    llama_log_softmax(logits, n_vocab);
    llama_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
              auto & l = logits[i];
        const auto & g = logits_guidance[i];

        l = scale * (l - g) + g;
    }

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
}

// rng: the caller's RNG (common_sampler's per-request seeded one). nullptr keeps the old behaviour, smpl->rng -- the llama_context's
// RNG, which a request seed never reaches, so seeded mirostat requests were not repeatable (MEASURED 2026-11-08,
// _g4_miro_probe_1108.py: 'mirostat 2' x2 differed while every other sampler path, which passes ctx_sampling->rng, was repeatable).
llama_token llama_sample_token_mirostat_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu,
                                             std::mt19937 * rng) {
    GGML_ASSERT(smpl);

    const int32_t n_vocab = float(smpl->n_vocab);

    int64_t t_start_sample_us = ggml_time_us();

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(n_vocab, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sample_top_k_impl((struct llama_sampling *) nullptr, candidates, int(k), 1);
    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    llama_token X = rng ? llama_sample_token_with_rng_impl(smpl, candidates, *rng) : llama_sample_token_impl(smpl, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    return X;
}

llama_token llama_sample_token_mirostat_v2_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, float * mu,
                                                std::mt19937 * rng) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax_impl(smpl, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax_impl(smpl, candidates);

    // Sample the next word X from the remaining words
    llama_token X = rng ? llama_sample_token_with_rng_impl(smpl, candidates, *rng) : llama_sample_token_impl(smpl, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_greedy_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
    }
    return result;
}

llama_token llama_sample_token_with_rng_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, std::mt19937 & rng) {
    GGML_ASSERT(smpl);

    const int64_t t_start_sample_us = ggml_time_us();

    if (candidates->size < 2) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
        return candidates->data[0].id;
    }

    std::vector<float> probs(candidates->size);
    probs[0] = candidates->data[0].logit;
    float max = probs[0];
    for (int j = 1; j < candidates->size; ++j) {
        probs[j] = candidates->data[j].logit;
        max = std::max(max, probs[j]);
    }

    // Every candidate masked to -inf (a grammar/choice this vocabulary cannot satisfy, or a logit_bias that bans the
    // whole list): there is nothing to draw. Return LLAMA_TOKEN_NULL so the caller can fail ONE request; the abort
    // below used to take down the whole llama-server. NaN still reaches the abort -- that is corrupt data, not a
    // request error. Row S69-FORK-UNSATISFIABLE-GRAMMAR-ABORTS-SERVER-SAMPLING-839-2026-10-11-01.
    if (max == -INFINITY) {
        LLAMA_LOG_WARN("%s: all %zu candidates have logit -inf (unsatisfiable grammar/choice?); no token sampled\n",
                __func__, candidates->size);
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
        return LLAMA_TOKEN_NULL;
    }

    float sump = 0;
    for (int j = 0; j < candidates->size; ++j) {
        float p = expf(probs[j] - max);
        candidates->data[j].p = p;
        sump += p;
        probs[j] = sump;
    }
    // Leave the distribution this draw used in the candidates, normalised, as upstream's dist sampler leaves it in cur_p:
    // common_sampler_get_candidates() reads p, and without this it saw whatever an earlier step left (ctest
    // test-speculative-rejection-sampling: every target probability but one read 0). The draw itself is unchanged.
    for (int j = 0; j < candidates->size; ++j) {
        candidates->data[j].p /= sump;
    }
    probs.back() += sump;

    auto r = rng();
    auto p = sump * r / rng.max();
    auto iter = std::upper_bound(probs.begin(), probs.end(), p);
    if (iter == probs.end()) {
        // Log the evidence instead of writing probabilities.txt into whatever directory the process runs in (for
        // llama-server that is the service CWD, and the file was overwritten by the next crash anyway).
        int n_nonfinite = 0;
        for (size_t j = 0; j < candidates->size; ++j) {
            n_nonfinite += !std::isfinite(candidates->data[j].logit);
        }
        LLAMA_LOG_ERROR("=============================== Failed to sample token\n");
        LLAMA_LOG_ERROR("candidates->size = %zu, max = %g, sump = %g, r = %u, non-finite logits = %d\n",
                candidates->size, max, sump, (unsigned) r, n_nonfinite);
        for (size_t j = 0; j < candidates->size && j < 16; ++j) {
            LLAMA_LOG_ERROR("  %zu  id %d  logit %g  cum %g\n", j, candidates->data[j].id, candidates->data[j].logit, probs[j]);
        }
        GGML_ABORT("Fatal error: failed to sample token (non-finite probabilities)");
    }
    GGML_ASSERT(iter != probs.end());
    auto idx = std::distance(probs.begin(), iter);
    auto id  = candidates->data[idx].id;

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    smpl->n_sample++;

    return id;
}

llama_token llama_sample_token_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    return llama_sample_token_with_rng_impl(smpl, candidates, smpl->rng);
}


// DRY

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void get_overlapping_token_sequences(const llama_vocab& vocab, const std::string& str, std::unordered_multimap<llama_token, std::vector<llama_token>>& token_sequences, int max_tail_len = -1) {
    for (llama_token token_id = 0; token_id < (llama_token)vocab.n_tokens(); token_id++) {
        auto word = vocab.detokenize( { token_id }, true);
        if (word.find(str) != std::string::npos) {
            token_sequences.emplace(token_id, std::vector<llama_token>());
        }
        else {
            size_t word_len = word.size(), str_len = str.size();
            size_t pos = -1;
            while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
                bool match = true;
                size_t i;
                for (i = 1; i < str_len && i + pos < word_len; ++i) {
                    if (word[pos + i] != str[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    auto tokenization = vocab.tokenize(str.substr(i), false, false);
                    //std::vector<llama_token> tokenization = llama_tokenize_internal(vocab, str.substr(i), false, false);
                    if (max_tail_len >= 0 && tokenization.size() > (size_t)max_tail_len) {
                        tokenization.resize(max_tail_len);
                    }

                    // Ensure we don't already have a duplicate matching tokenization
                    auto its = token_sequences.equal_range(token_id);
                    bool found = false;
                    for (auto it = its.first; it != its.second; ++it) {
                        if (tokenization == it->second) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        token_sequences.emplace(token_id, tokenization);
                    }
                }
            }
        }
    }
}

static const char* llama_sampler_dry_name(const struct llama_sampler* /*smpl*/) {
    return "dry";
}



// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
void llama_sampler_dry_apply(struct llama_sampler_dry* smpl, llama_token_data_array* cur_p) {
    if (smpl->dry_multiplier == 0.0f || smpl->dry_base < 1.0f || smpl->dry_penalty_last_n == 0) {
        return;
    }

    int32_t effective_dry_penalty_last_n = (smpl->dry_penalty_last_n == -1) ? smpl->total_context_size : std::max(smpl->dry_penalty_last_n, 0);
    int last_n_repeat = std::min(std::min((int)smpl->last_tokens.size(), effective_dry_penalty_last_n), smpl->total_context_size);

    if (last_n_repeat <= smpl->dry_allowed_length) {
        return;
    }

    smpl->dry_repeat_count.assign(last_n_repeat, 0);
    smpl->dry_max_token_repeat.clear();

    // Step 1: Look for restart sequences to limit the maximum repetition length.
    // Work backwards through the context looking for any token that begins a restart sequence.
    //
    // The collection `restart_sequences` is a mapping from a "head" token to all "tail"
    // sequences that together comprise a restart sequence. This allows us to quickly check
    // whether each token is the head of a complete sequence. Most restart sequences are actually
    // a single token, and for these the "tail" is an empty vector.
    //
    // If the token is a "head", test all restart sequences that begin with this token
    // (there will often only be one sequence for each token, but if sequences like 'aaaq1' and
    // 'aaa1' are used as restart strings, both could start with 'aaa' when tokenized). The
    // longest matching sequence (if any) is used to limit the maximum repetition length.
    //
    // Note that in the case case of a short sequence contained in a longer one, this might fail to
    // find the smallest value for `rep_limit`. For example, if 'amniotic' and 'ni' are both used as
    // restart sequences, 'ni' will be found first, and since it's shorter it will fail to suppress
    // 'otic'. This is a minor issue since fully contained restart sequences are likely to be rare.
    //
    // This is theoretically worst-case O(N^2) for arbitrary restart sequences, which is why we
    // have already clamped the maximum tail sequence length when generating `restart_sequences`.
    // With clamping, this scan is O(N) in the context length.

    int rep_limit = last_n_repeat;
    for (int i = 0; i < last_n_repeat; ++i) {
        llama_token token = smpl->last_tokens.rat(i);
        auto its = smpl->dry_processed_breakers.equal_range(token);
        if (its.first == smpl->dry_processed_breakers.end()) {
            continue;
        }
        int longest_match = -1;
        for (auto it = its.first; it != its.second; ++it) {
            // Note that (*it) does not contain the head character, so seq_len will be
            // the restart sequence length minus 1.
            // In the common case of a single-token restart sequence, (*it) will be empty
            // and we will trivially match.
            int seq_len = (int)it->second.size();
            if (seq_len > longest_match && seq_len <= (int)i) {
                bool match = true;
                for (int offset = 0; offset < seq_len; ++offset) {
                    // The -1 when indexing `last_tokens` is because we already matched the head.
                    if (it->second[offset] != smpl->last_tokens.rat(i - offset - 1)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    longest_match = seq_len;
                }
            }
        }
        if (longest_match >= 0) {
            // We found a restart sequence starting `i` tokens from the end and continuing for
            // `longest_match` tokens.
            rep_limit = i - longest_match;
            break;
        }
    }
    if (rep_limit < smpl->dry_allowed_length) {
        return;
    }

    // Step 2: Iterate in reverse over the last N tokens of the context, using the "Z-algorithm" (in
    // the reverse direction) to efficiently compute the positions and lengths of suffixes appearing
    // elsewhere in the context. We limit the suffix length to `rep_limit` to respect restart sequences.
    //
    // This algorithm is not currently documented on Wikipedia, but there is a clear description here:
    // https://ivanyu.me/blog/2014/10/15/z-algorithm/
    //
    // The code below is adapted from the public domain implementation by the same author here:
    // https://github.com/ivanyu/string-algorithms/blob/master/z_algorithm.py
    //
    // Example:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //                    ^
    //   This `3` means that the last three tokens of the context (a b c) also appear here.
    //
    // This step is worst case O(N) since the Z-algorithm is linear, despite the appearance of nested
    // for/while loops. This can be seen by observing that the `lt` and `rt` bounds are set after each
    // repeated suffix is detected (i.e. after each while loop when n > 0). These bound variables
    // ensure that the inner while loops only examine each token in the context once as the outer
    // for loop iterates over the context.

    {
        const int last = last_n_repeat - 1;
        int rt = 0, lt = 0;

        for (int k = 1; k < last_n_repeat; ++k) {
            if (k > rt) {
                // If k is outside the current Z-box, do naive computation.
                int n = 0;
                while (n + k < last_n_repeat && smpl->last_tokens.rat(n) == smpl->last_tokens.rat(n + k)) {
                    ++n;
                }
                smpl->dry_repeat_count[last - k] = std::min(n, rep_limit);
                if (n > 0) {
                    lt = k;
                    rt = k + n - 1;
                }
            }
            else {
                // If k is inside the current Z-box, consider two cases.

                int p = k - lt; // Pair index.
                int right_part_len = rt - k + 1;

                if (smpl->dry_repeat_count[last - p] < right_part_len) {
                    int n = std::min(smpl->dry_repeat_count[last - p], rep_limit);
                    smpl->dry_repeat_count[last - k] = n;
                }
                else {
                    int i = rt + 1;
                    while (i < last_n_repeat && smpl->last_tokens.rat(i) == smpl->last_tokens.rat(i - k)) {
                        i += 1;
                    }

                    int n = std::min(i - k, rep_limit);
                    smpl->dry_repeat_count[last - k] = n;
                    lt = k;
                    rt = i - 1;
                }
            }
        }
    }

    // Step 3: Iterate over dry_repeat_count and last_tokens, examining the maximum repeat length
    // that would be generated by emitting each new token that would extend a sequence.
    //
    // Following the same example as above:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //
    // For each non-zero, look ahead one token. This token, if emitted, would extend the repetition.
    // c: 3 -> 4 (from `a b c` to `a b c c`)
    // b: 1 -> 2 (from `c` to `c b`)
    // y: 2 -> 3 (from `b c` to `b c y`)

    for (int i = 0; i < last_n_repeat - 1; ++i) {
        int repeat_len = smpl->dry_repeat_count[i];
        if (repeat_len >= smpl->dry_allowed_length) {
            // This token ends a repeat, so the next token would continue one.
            // By convention, the value of `repeat_len` only includes the tokens currently
            // in the context, not the new token that would be added.
            llama_token token = smpl->last_tokens.rat(last_n_repeat - 2 - i);
            // Track the maximum sequence ending in this token.
            const auto& it = smpl->dry_max_token_repeat.find(token);
            if (it == smpl->dry_max_token_repeat.end() || it->second < repeat_len) {
                smpl->dry_max_token_repeat[token] = repeat_len;
            }
        }
    }

    // Step 4: Apply logit penalties based on the maximum repeat length for relevant tokens.

    // Prevent floating point overflow in `pow(penalty_base, exponent)` by clamping to `max_exponent`.
    // Compute it from `penalty_base` and the approximate log of `std::numeric_limits<float>::max()`
    const float FLOAT_MAX_LOG = 88.7228391f;
    int max_exponent = 0;
    if (smpl->dry_base > 1.000001f) {
        max_exponent = FLOAT_MAX_LOG / std::log(smpl->dry_base);
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto& af_kvp = smpl->dry_max_token_repeat.find(cur_p->data[i].id);
        if (af_kvp != smpl->dry_max_token_repeat.end()) {
            // Check all sequence breakers starting with this token
            auto range = smpl->dry_processed_breakers.equal_range(cur_p->data[i].id);
            bool is_single_token_breaker = false;

            for (auto it = range.first; it != range.second; ++it) {
                if (it->second.empty()) {
                    is_single_token_breaker = true;
                    break;
                }
            }

            // Apply penalty only if it's not a single-token sequence breaker
            if (!is_single_token_breaker) {
                int repeat_exp = af_kvp->second - smpl->dry_allowed_length;
                if (max_exponent > 0 && repeat_exp > max_exponent) {
                    repeat_exp = max_exponent;
                }
                float penalty = smpl->dry_multiplier * std::pow(smpl->dry_base, repeat_exp);
                cur_p->data[i].logit -= penalty;
            }
        }
    }

    cur_p->sorted = false;
}



static size_t g_dry_cache_hits = 0, g_dry_cache_misses = 0;

void llama_dry_breaker_cache_stats(size_t * hits, size_t * misses) {
    if (hits)   { *hits   = g_dry_cache_hits; }
    if (misses) { *misses = g_dry_cache_misses; }
}

struct llama_sampler_dry* llama_sampler_init_dry_impl(const struct llama_vocab& vocab, int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers) {
    int32_t effective_dry_penalty_last_n = (dry_penalty_last_n == -1) ? context_size : std::max(dry_penalty_last_n, 0);
    std::unordered_multimap<llama_token, std::vector<llama_token>> processed_breakers;
    const int MAX_CHAR_LEN = 40;
    const int MAX_SEQ_LEN = 20;

    const bool dry_enabled = (dry_multiplier != 0.0f && dry_base >= 1.0f && dry_penalty_last_n != 0);

    if (dry_enabled && seq_breakers != nullptr && num_breakers > 0) {
        // cache key: vocab identity + size + a fingerprint of ~256 token texts (guards a recycled vocab address) + the breaker strings
        std::string key = std::to_string((uintptr_t) &vocab) + "/" + std::to_string(vocab.n_tokens()) + "/";
        {
            const int32_t nt = (int32_t) vocab.n_tokens();
            const int32_t step = std::max(1, nt / 256);
            size_t h = 1469598103934665603ULL;
            for (int32_t t = 0; t < nt; t += step) {
                for (unsigned char c : vocab.detokenize({ (llama_token) t }, true)) { h = (h ^ c) * 1099511628211ULL; }
            }
            key += std::to_string(h) + "/";
        }
        for (size_t i = 0; i < num_breakers; ++i) {
            const char * s = seq_breakers[i] ? seq_breakers[i] : "";
            key += std::to_string(std::strlen(s)) + ":" + s + ";";
        }
        static std::mutex cache_mu;
        static std::unordered_map<std::string, std::unordered_multimap<llama_token, std::vector<llama_token>>> cache;
        {
            std::lock_guard<std::mutex> lk(cache_mu);
            auto it = cache.find(key);
            if (it != cache.end()) {
                ++g_dry_cache_hits;
                processed_breakers = it->second;
            }
        }
        if (processed_breakers.empty()) {
        ++g_dry_cache_misses;
        // Process sequence breakers
        for (size_t i = 0; i < num_breakers; ++i) {
            if (seq_breakers[i] == nullptr || std::strlen(seq_breakers[i]) == 0) {
                LLAMA_LOG_WARN("skipping null or empty DRY sequence breaker at index %zu\n", i);
                continue;
            }

            std::string sequence_break(seq_breakers[i]);
            if (sequence_break.empty()) {
                LLAMA_LOG_WARN("skipping empty DRY sequence breaker\n");
                continue;
            }

            if (sequence_break.size() > MAX_CHAR_LEN) {
                LLAMA_LOG_WARN("truncating DRY sequence breaker to %d characters\n", MAX_CHAR_LEN);
                sequence_break.resize(MAX_CHAR_LEN);
            }

            get_overlapping_token_sequences(vocab, sequence_break, processed_breakers, MAX_SEQ_LEN);
        }
        {
            std::lock_guard<std::mutex> lk(cache_mu);
            if (cache.size() >= 32) { cache.clear(); }   // bound memory
            cache[key] = processed_breakers;
        }
        }
    }

    return  new llama_sampler_dry {
            /* .total_context_size     = */ context_size,
            /* .dry_multiplier         = */ dry_multiplier,
            /* .dry_base               = */ dry_base,
            /* .dry_allowed_length     = */ dry_allowed_length,
            /* .dry_penalty_last_n     = */ dry_penalty_last_n,
            /* .dry_processed_breakers = */ std::move(processed_breakers),
            /* .dry_repeat_count       = */ dry_enabled ? std::vector<int>(effective_dry_penalty_last_n, 0) : std::vector<int>{},
            /* .dry_max_token_repeat   = */ {},
            /* .last_tokens            = */ dry_enabled ? ring_buffer<llama_token>(effective_dry_penalty_last_n) : ring_buffer<llama_token>(0),
    };
}


// adaptive p

void llama_review_adaptive_p_impl(llama_sampler_adaptive_p * adapt_p_ctx, const size_t n_unsent, const bool rewind_status) {
    // LLAMA_LOG_DEBUG("%s: n_unsent = %zu, rewind_status = %s\n", __func__, n_unsent, rewind_status ? "true" : "false");
    if (adapt_p_ctx->target < 0.0f) {
        // LLAMA_LOG_DEBUG("%s: sampler disabled, target = %f\n", __func__, adapt_p_ctx->target);
        return;
    }

    auto & history = adapt_p_ctx->history;
    const size_t hsz = history.size();
    const size_t hsz_next = 1 + n_unsent;
    // LLAMA_LOG_DEBUG("%s: hsz = %zu, hsz_next = %zu\n", __func__, hsz, hsz_next);
    if (hsz_next >= hsz >> 1) { return; }   // skip small update

    if (!rewind_status) {
        // sent results, overwrite old history
        // LLAMA_LOG_DEBUG("%s: hsz = %zu, hsz_next = %zu\n", __func__, hsz, hsz_next);
        // LLAMA_LOG_DEBUG("%s: history[hsz-1].first = %f\n", __func__, history[hsz-1].first);
        const size_t hsz_diff = hsz - hsz_next;
        for (int j = 0; j < hsz_next; ++j) {
            history[j] = history[j + hsz_diff];
        }
    }
    history.resize(hsz_next);
    // LLAMA_LOG_DEBUG("%s: history[hsz_next-1].first = %f\n", __func__, history[hsz_next-1].first);
}

llama_token llama_sample_token_adaptive_p_impl(
                  struct llama_sampling * smpl,
                 llama_token_data_array * candidates,
        struct llama_sampler_adaptive_p * adapt_p_ctx,
                           std::mt19937 & rng) {
    if (adapt_p_ctx->cum_orig_prob == 0.f) {
        LLAMA_LOG_ERROR("%s[%d]: falling back to temperature sampling\n", __func__, __LINE__);
        return llama_sample_token_with_rng_impl(smpl, candidates, rng);
    }

    GGML_ASSERT(candidates->size > 0);
    const int64_t t_start_sample_us = ggml_time_us();

    struct llama_sampler_adaptive_p * ctx = adapt_p_ctx;
    ctx->cum_probs.resize(candidates->size);

    // compute cumulative probability distribution
    const float max_logit = ctx->max_xform_logit;
    float cum_prob = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        cum_prob += expf(candidates->data[i].logit - max_logit);
        ctx->cum_probs[i] = cum_prob;
    }
    ctx->cum_probs.back() += 1.0f;  // safety margin in case rng() ~= rng.max()

    // select first token whose cum_prob > target_cum_prob
    const float target_cum_prob = cum_prob * (float)ctx->rng() / (float)ctx->rng.max();
    auto iter = std::upper_bound(ctx->cum_probs.begin(), ctx->cum_probs.end(), target_cum_prob);
    GGML_ASSERT(iter != ctx->cum_probs.end());
    const size_t idx = std::distance(ctx->cum_probs.begin(), iter);
    llama_token id = candidates->data[idx].id;
    GGML_ASSERT(id < int(ctx->orig_prob.size()));

    // update history
    const float update_prob = (ctx->updt_w_cur ? candidates->data[idx].p
                                               : ctx->orig_prob[id])
                              / ctx->cum_orig_prob;
    if (update_prob > 0) {
        const auto [weighted_sum, total_weight] = ctx->history.back();
        ctx->history.push_back({ ctx->decay * weighted_sum + update_prob,
                                 ctx->decay * total_weight + 1.0f });
    }

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    smpl->n_sample++;

    // printf("%s[%d]: idx = %zu, id = %d\n", __func__, __LINE__, idx, id);

    return id;
}

void llama_sample_adaptive_p_impl(struct llama_sampling * ctx, llama_token_data_array * candidates,
        struct llama_sampler_adaptive_p * adapt_p_ctx) {
    if (adapt_p_ctx->target < 0.0f) {
        // LLAMA_LOG_DEBUG("%s: sampler disabled, target = %f\n", __func__, adapt_p_ctx->target);
        llama_sample_softmax_impl(nullptr, candidates);
        return;
    }

    auto t_start = ggml_time_us();

    // incomplete softmax because final division can be fused
    float max_l = candidates->data[0].logit;
    if (!candidates->sorted) {
        for (size_t i = 1; i < candidates->size; ++i) {
            max_l = std::max(max_l, candidates->data[i].logit);
        }
    }
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        const float prob = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = prob;
        cum_sum += prob;
    }

    if (adapt_p_ctx->updt_w_cur) {
        adapt_p_ctx->cum_orig_prob = cum_sum;
    }

    if (adapt_p_ctx->cum_orig_prob == 0.f) {
        LLAMA_LOG_ERROR("%s[%d]: cum_orig_prob=%f is invalid\n", __func__, __LINE__, adapt_p_ctx->cum_orig_prob);
        return;
    }

    // compute adapted target probability
    const float weighted_sum = adapt_p_ctx->history.back().first;
    const float total_weight = adapt_p_ctx->history.back().second;
    const float target = std::clamp(adapt_p_ctx->target, 0.0f, 1.0f);
    const float adapted_target = std::clamp(total_weight == 0.0f
        ? target
        : 2.0f * target - (weighted_sum / total_weight),
        0.0f, 1.0f);

    // transformation constants
    static constexpr float peak_logit_value = 5.0f;
    static constexpr float inv_width = 1.0f / 0.3f;
    static constexpr float sharpness = 10.0f;

    const float fused_target = adapted_target * inv_width;
    const float fused_width = inv_width / cum_sum;

    // quadratic near target for finite differentiation, transitioning to linear decay in tails
    // unbounded negative logits suppress far-from-target tokens after softmax
    float max_logit = -INFINITY;
    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].logit <= -999.f) {
            // probably masked tokens
            continue;
        }
        const float dist = std::abs(candidates->data[i].p * fused_width - fused_target);
        const float logit = peak_logit_value - sharpness * dist * dist / (1.0f + dist);
        candidates->data[i].logit = logit;
        max_logit = std::max(max_logit, logit);
    }
    candidates->sorted = false;
    adapt_p_ctx->max_xform_logit = max_logit;

    ctx->t_sample_us += ggml_time_us() - t_start;
}

void llama_prep_adaptive_p_impl(struct llama_sampling           * smpl,
                                llama_token_data_array          * candidates,
                                struct llama_sampler_adaptive_p * adapt_p_ctx) {
    auto & orig_prob = adapt_p_ctx->orig_prob;
    orig_prob.resize(candidates->size);

    if (adapt_p_ctx->updt_w_cur     // update with current probability, original not needed
        || (adapt_p_ctx->target < 0.0f)) {  // or disabled
        return;
    }

    constexpr float kDelta = 30.0f; //16.6f;
    auto t_start = ggml_time_us();
    if (candidates->size != orig_prob.size() || candidates->sorted) {
        LLAMA_LOG_ERROR("%s: this function must be called before any other sampler has been applied\n", __func__);
        LLAMA_LOG_ERROR("%s: the sampler has been initialized with a vocabulary of %zu, but is being called with %zu candidates\n",
                __func__, orig_prob.size(), candidates->size);
        GGML_ABORT("Bad candidates in adaptive_p sampler");
    }

    float max_logit = -INFINITY;
    for (int j = 0; j < int(candidates->size); ++j) {
        orig_prob[j] = candidates->data[j].logit;
        max_logit = std::max(max_logit, orig_prob[j]);
    }
    adapt_p_ctx->cum_orig_prob = iqk_exp_with_thresh(orig_prob.size(), orig_prob.data(), max_logit, max_logit - kDelta);

    if (smpl) smpl->t_sample_us += ggml_time_us() - t_start;
}

struct llama_sampler_adaptive_p * llama_init_adaptive_p_impl(const float    target,
                                                             const float    decay,
                                                             const bool     updt_w_cur,
                                                             const uint32_t seed) {
    const float clamped_decay = std::clamp(decay, 0.0f, 0.99f);
    auto result = new llama_sampler_adaptive_p {
        /* .target            = */ target,
        /* .decay             = */ clamped_decay,
        /* .updt_w_cur        = */ updt_w_cur,
        /* .rng               = */ std::mt19937(seed),
        /* .history           = */ {},
        /* .orig_prob         = */ {},
        /* .cum_orig_prob     = */ 1.0f,
        /* .max_xform_logit   = */ -INFINITY,
        /* .cum_probs         = */ {},
    };
    result->history.push_back({
        target / (1.0f - clamped_decay),    // weighted_sum
        1.0f / (1.0f - clamped_decay) });   // total_weight
    return result;
}

// REMOVED 2026-09-08 -- the duplicate grammar-sampler block that used to run from here to the end
// of this file (177 lines). src/llama-sampler.cpp owns it now.
//
// BOTH files are named by src/CMakeLists.txt, so both are compiled into libllama, and both defined
// the grammar sampler. Four of those definitions were non-static and genuinely collided:
//     llama_sampler_init_grammar, _lazy, _lazy_patterns   same parameters, DIFFERENT return type
//     llama_sampler_reset                                 identical signature, two bodies
// The first three produced a compile error because include/llama.h:2362 disagreed with them:
//     error: ambiguating new declaration of 'llama_grammar* llama_sampler_init_grammar(...)'
// llama_sampler_reset produced NO diagnostic -- two identical non-static definitions in one
// library, one of which silently wins at link. That one was the more dangerous.
//
// WHY THIS SIDE LOST, decided on evidence rather than on which file looked newer. Three
// independent places agree the symbol returns llama_sampler*: include/llama.h:2362,
// common/sampling.cpp:88 and :91 which feed the result into a sampler chain, and
// src/llama-sampler.cpp:2711 which defines it that way. Only this file said llama_grammar*. The
// fork's own history explains it without blame: at the fork parent 35f0cac75, include/llama.h:1532
// DID declare the grammar-returning form, so this code was correct when it was written and was
// left behind when the merge moved the header.
//
// NOTHING FORK-SPECIFIC WAS LOST. This file's 26 unique symbols -- the legacy llama_sample_*_impl
// API including adaptive_p -- are untouched above; they are why the file exists. Verified before
// deleting: the retained lines contained zero references to any symbol in the removed block, so
// nothing left behind lost a callee.
//
// A note at src/CMakeLists.txt:38 said "Only 2 names collide with llama-sampling.cpp". The measured
// number is 13 names defined in both files and 4 real collisions; the other 9 are `static` in at
// least one file and therefore have internal linkage. The count of 2 appears to have come from
// what the compiler reported rather than from comparing the files.
