// Speculative-decoding acceptance rules -- see spec-accept.h for the contract and sources.

#include "spec-accept.h"

#include <algorithm>
#include <unordered_map>

static double dist_total(const common_spec_dist & d) {
    double t = 0.0;
    for (float v : d.probs) {
        t += v > 0.0f ? v : 0.0f;
    }
    return t;
}

float common_spec_dist_prob(const common_spec_dist & d, llama_token id) {
    const double t = dist_total(d);
    if (t <= 0.0) {
        return 0.0f;
    }
    for (size_t i = 0; i < d.ids.size(); ++i) {
        if (d.ids[i] == id) {
            return d.probs[i] > 0.0f ? (float) (d.probs[i] / t) : 0.0f;
        }
    }
    return 0.0f;
}

// Draw one id from weights w over ids (w need not be normalised). Returns -1 if every weight is <= 0.
static llama_token draw(const std::vector<llama_token> & ids, const std::vector<double> & w, std::mt19937 & rng) {
    double t = 0.0;
    for (double v : w) {
        t += v > 0.0 ? v : 0.0;
    }
    if (t <= 0.0) {
        return -1;
    }
    std::uniform_real_distribution<double> u(0.0, t);
    const double r = u(rng);
    double acc = 0.0;
    llama_token last = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (w[i] <= 0.0) {
            continue;
        }
        acc += w[i];
        last = ids[i];
        if (r < acc) {
            return ids[i];
        }
    }
    return last; // r landed on the float edge of the total
}

static llama_token draw_from(const common_spec_dist & d, std::mt19937 & rng) {
    std::vector<double> w(d.probs.begin(), d.probs.end());
    return draw(d.ids, w, rng);
}

common_spec_accept_result common_spec_accept_rejection(const common_spec_dist & p, const common_spec_dist & q,
                                                       llama_token x, std::mt19937 & rng) {
    const double px = common_spec_dist_prob(p, x);
    const double qx = common_spec_dist_prob(q, x);

    // qx == 0: x could not have been drawn from the q we were given (e.g. a truncated top-k representation of the
    // drafter's distribution that misses x). The rejection rule is exact only for x ~ q, so fall back to
    // sample-and-match, which is exact for ANY x whatever q is. Treating the ratio as unbounded would bias toward x.
    if (qx <= 0.0) {
        return common_spec_accept_sample_match(p, x, rng);
    }
    const double a = std::min(1.0, px / qx);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (u(rng) < a) {
        return { x, true };
    }

    // Rejected: draw from norm(max(0, p - q)) over the union of both supports.
    std::unordered_map<llama_token, double> qmap;
    const double qt = dist_total(q);
    if (qt > 0.0) {
        for (size_t i = 0; i < q.ids.size(); ++i) {
            if (q.probs[i] > 0.0f) {
                qmap[q.ids[i]] += q.probs[i] / qt;
            }
        }
    }
    const double pt = dist_total(p);
    std::vector<llama_token> ids;
    std::vector<double>      w;
    ids.reserve(p.ids.size());
    w.reserve(p.ids.size());
    for (size_t i = 0; i < p.ids.size(); ++i) {
        const double pi = (pt > 0.0 && p.probs[i] > 0.0f) ? p.probs[i] / pt : 0.0;
        const auto it = qmap.find(p.ids[i]);
        const double qi = it == qmap.end() ? 0.0 : it->second;
        ids.push_back(p.ids[i]);
        w.push_back(std::max(0.0, pi - qi));
    }
    llama_token y = draw(ids, w, rng);
    if (y < 0) {
        // p <= q everywhere means p == q up to rounding, where rejection has probability ~0; fall back to p itself,
        // which is the exact target when the residual is empty.
        y = draw_from(p, rng);
    }
    return { y, y == x };
}

common_spec_accept_result common_spec_accept_relaxed(const common_spec_dist & p, const common_spec_dist & q,
                                                     llama_token x, std::mt19937 & rng, float top_p) {
    // the exact rule runs first and consumes the same random numbers it always did: whatever it accepts stays an exact accept, and a
    // token outside the nucleus gets precisely the exact rule's verdict. Only an exact REJECTION of a nucleus token is overturned.
    const common_spec_accept_result exact = common_spec_accept_rejection(p, q, x, rng);
    if (exact.accepted) {
        return exact;
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        const double pt = dist_total(p);
        if (pt > 0.0) {
            // nucleus of p: descending probability, equal probabilities by lower id; x is in it iff the mass strictly BEFORE x is < top_p
            const double px = common_spec_dist_prob(p, x);
            if (px > 0.0) {
                double before = 0.0;
                for (size_t i = 0; i < p.ids.size(); ++i) {
                    const double pi = p.probs[i] > 0.0f ? p.probs[i] / pt : 0.0;
                    if (pi > px || (pi == px && p.ids[i] < x)) {
                        before += pi;
                    }
                }
                if (before < (double) top_p) {
                    return { x, true, true };
                }
            }
        }
    }
    return exact;
}

common_spec_accept_result common_spec_accept_sample_match(const common_spec_dist & p, llama_token x, std::mt19937 & rng) {
    const llama_token y = draw_from(p, rng);
    return { y, y == x };
}

double common_spec_accept_rate_rejection(const common_spec_dist & p, const common_spec_dist & q) {
    double s = 0.0;
    for (llama_token id : p.ids) {
        s += std::min((double) common_spec_dist_prob(p, id), (double) common_spec_dist_prob(q, id));
    }
    return s;
}

double common_spec_accept_rate_sample_match(const common_spec_dist & p, const common_spec_dist & q) {
    double s = 0.0;
    for (llama_token id : p.ids) {
        s += (double) common_spec_dist_prob(p, id) * common_spec_dist_prob(q, id);
    }
    return s;
}
