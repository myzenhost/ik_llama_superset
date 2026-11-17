// POST /classify: turn the n_cls_out raw outputs of a sequence-classification head into a label distribution.
//
// Header-only and model-free (tested by tools/server/tests_cpp/test_server_classify.cpp). The server worker already pulls
// n_cls_out floats per sequence out of the context (rank pooling); /rerank keeps only the first one. /classify keeps all of
// them, names them with the GGUF class labels and normalises them here.
//
// SOURCE-READ: upstream/lorax/server/lorax_server/models/flash_bert.py:204-217 (LoRAX predibase/lorax a7394ba3,
// Apache-2.0): logits -> softmax -> max -> id2label[argmax], confidence = the max probability. Spec written in my own words
// and implemented with my own structure. DIFFERENCES: (1) LoRAX only returns the winner; this also returns every class
// probability (the row asks for scores:[{label,probability}]). (2) LoRAX always softmaxes; a multi-label head (independent
// yes/no per class) is normalised per class with a sigmoid instead, selected by the request, because softmax would force the
// classes to compete. (3) a single-output head (n_cls_out == 1, the reranker shape) has no distribution to normalise: it
// gets a sigmoid, so the one number is a probability in (0,1) rather than a softmax that is always 1.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

struct server_classify_entry {
    std::string label;
    float       probability = 0.0f;
};

struct server_classify_result {
    std::string                         label;      // the most probable class
    float                               score = 0;  // its probability
    std::vector<server_classify_entry>  scores;     // every class, in model order
};

// logits -> probabilities. softmax (max-shifted, stable) when there are several outputs and the head is single-label;
// element-wise sigmoid when the head is multi-label or has one output. Empty in, empty out.
static inline std::vector<float> server_classify_probs(const std::vector<float> & logits, bool multilabel) {
    std::vector<float> p(logits.size());
    if (logits.empty()) {
        return p;
    }
    if (multilabel || logits.size() == 1) {
        for (size_t i = 0; i < logits.size(); i++) {
            p[i] = 1.0f / (1.0f + std::exp(-logits[i]));
        }
        return p;
    }
    const float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        p[i] = std::exp(logits[i] - mx);
        sum += p[i];
    }
    for (size_t i = 0; i < p.size(); i++) {
        p[i] = (float) (p[i] / sum);
    }
    return p;
}

// label for class i: the GGUF label when there is one, else its index as text.
static inline std::string server_classify_label(const std::vector<std::string> & labels, size_t i) {
    if (i < labels.size() && !labels[i].empty()) {
        return labels[i];
    }
    return std::to_string(i);
}

static inline server_classify_result server_classify_make(const std::vector<float> & logits,
                                                           const std::vector<std::string> & labels,
                                                           bool multilabel) {
    server_classify_result r;
    const std::vector<float> p = server_classify_probs(logits, multilabel);
    size_t best = 0;
    for (size_t i = 0; i < p.size(); i++) {
        r.scores.push_back({ server_classify_label(labels, i), p[i] });
        if (p[i] > p[best]) {
            best = i;
        }
    }
    if (!p.empty()) {
        r.label = r.scores[best].label;
        r.score = p[best];
    }
    return r;
}
