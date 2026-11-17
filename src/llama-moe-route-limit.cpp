// llama-moe-route-limit.cpp -- see llama-moe-route-limit.h (design reference: fastllm, Apache-2.0, behaviour only).

#include "llama-moe-route-limit.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <set>
#include <tuple>

double llama_moe_route_makespan(const std::vector<llama_moe_expert_cost> & experts, int32_t limit, int32_t n_gpu) {
    double cpu_total = 0.0;
    std::vector<double> gpu_jobs;
    for (const auto & e : experts) {
        if (e.routes <= 0) {
            continue;
        }
        if (e.routes < limit) {
            cpu_total += e.cpu_us;
        } else {
            gpu_jobs.push_back(e.gpu_us);
        }
    }
    // longest job first onto the least-loaded GPU queue
    std::sort(gpu_jobs.begin(), gpu_jobs.end(), std::greater<double>());
    std::vector<double> load((size_t) std::max<int32_t>(1, n_gpu), 0.0);
    for (double job : gpu_jobs) {
        *std::min_element(load.begin(), load.end()) += job;
    }
    return std::max(cpu_total, *std::max_element(load.begin(), load.end()));
}

int32_t llama_moe_select_route_limit(const std::vector<llama_moe_expert_cost> & experts,
                                     int32_t default_limit, int32_t n_gpu) {
    std::set<int32_t> candidates = {1};
    for (const auto & e : experts) {
        if (e.routes > 0) {
            candidates.insert(e.routes + 1);
        }
    }
    if (candidates.size() == 1) {
        return default_limit; // no active expert
    }
    int32_t best_limit = default_limit;
    double  best_time  = std::numeric_limits<double>::max();
    for (int32_t limit : candidates) { // ascending: a tie keeps the smaller limit (more experts on the GPU)
        const double t = llama_moe_route_makespan(experts, limit, n_gpu);
        if (t < best_time) {
            best_time  = t;
            best_limit = limit;
        }
    }
    return best_limit;
}

double llama_moe_cost_curve::at(int32_t n) const {
    if (n <= 0 || pts.empty()) {
        return 0.0;
    }
    auto it = pts.find(n);
    if (it != pts.end()) {
        return it->second;
    }
    auto upper = pts.upper_bound(n);
    if (upper == pts.begin()) {
        return upper->second; // below the first sample
    }
    if (upper == pts.end()) { // above the last sample: extrapolate with the last segment's slope, never below 0
        auto last = std::prev(upper);
        if (last == pts.begin()) {
            return last->second;
        }
        auto prev = std::prev(last);
        const double slope = (last->second - prev->second) / double(last->first - prev->first);
        return std::max(last->second + slope * double(n - last->first), 0.0);
    }
    auto lower = std::prev(upper);
    const double t = double(n - lower->first) / double(upper->first - lower->first);
    return lower->second + t * (upper->second - lower->second);
}

static int32_t route_step_for(int32_t n, bool is_cpu) {
    int32_t base;
    if      (n <= 16)   base = 1;
    else if (n <= 64)   base = 4;
    else if (n <= 256)  base = 16;
    else if (n <= 1024) base = 64;
    else                base = 128;
    return is_cpu ? std::max(base * 2, 2) : base;
}

std::vector<int32_t> llama_moe_route_sample_points(int32_t max_n, bool is_cpu) {
    std::vector<int32_t> pts;
    if (max_n <= 0) {
        return pts;
    }
    for (int32_t n = 1; n <= max_n; n += route_step_for(n, is_cpu)) {
        pts.push_back(n);
    }
    if (pts.empty() || pts.back() != max_n) {
        pts.push_back(max_n);
    }
    return pts;
}

static double route_measure(const std::function<double(int32_t)> & fn, int32_t n) {
    for (int32_t i = 0; i < LLAMA_MOE_ROUTE_WARMUP_ROUNDS; ++i) {
        (void) fn(n);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < LLAMA_MOE_ROUTE_MEASURE_ROUNDS; ++i) {
        sum += fn(n);
    }
    return sum / double(LLAMA_MOE_ROUTE_MEASURE_ROUNDS);
}

bool llama_moe_route_profile_build(llama_moe_route_profile & out, int32_t max_n,
                                   const std::function<double(int32_t)> & time_cpu_us,
                                   const std::function<double(int32_t)> & time_gpu_us) {
    if (max_n <= 0 || !time_cpu_us) {
        return false;
    }
    llama_moe_route_profile p;
    p.max_n = max_n;
    p.cpu.pts[0] = 0.0;
    p.gpu.pts[0] = 0.0;

    const std::vector<int32_t> cpu_pts = llama_moe_route_sample_points(max_n, true);
    const std::vector<int32_t> gpu_pts = llama_moe_route_sample_points(max_n, false);
    std::set<int32_t> cpu_set(cpu_pts.begin(), cpu_pts.end());
    std::set<int32_t> gpu_set(gpu_pts.begin(), gpu_pts.end());
    std::set<int32_t> all(cpu_set);
    all.insert(gpu_set.begin(), gpu_set.end());

    for (auto it = all.rbegin(); it != all.rend(); ++it) { // largest first, as the reference does
        const int32_t n = *it;
        if (cpu_set.count(n)) {
            p.cpu.pts[n] = route_measure(time_cpu_us, n);
        }
        if (time_gpu_us && gpu_set.count(n)) {
            p.gpu.pts[n] = route_measure(time_gpu_us, n);
        }
    }
    if (!time_gpu_us) {
        p.gpu = p.cpu;
    }
    p.initialized = true;
    out = std::move(p);
    return true;
}

bool llama_moe_route_shape_key::operator<(const llama_moe_route_shape_key & o) const {
    return std::tie(n_embd, n_ff_exp, n_embd_out, n_expert_used, type_up, type_down, gpu_id) <
           std::tie(o.n_embd, o.n_ff_exp, o.n_embd_out, o.n_expert_used, o.type_up, o.type_down, o.gpu_id);
}

bool llama_moe_route_profile_cache::get(const llama_moe_route_shape_key & key, int32_t want_n,
                                        const std::function<double(int32_t)> & time_cpu_us,
                                        const std::function<double(int32_t)> & time_gpu_us,
                                        llama_moe_route_profile & out) {
    const int32_t n = std::min(want_n, LLAMA_MOE_ROUTE_BENCH_MAX_N);
    if (n <= 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    llama_moe_route_profile & p = profiles[key];
    if (!p.initialized || p.max_n < n) {
        if (!llama_moe_route_profile_build(p, n, time_cpu_us, time_gpu_us)) {
            return false;
        }
    }
    out = p;
    return true;
}

size_t llama_moe_route_profile_cache::size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return profiles.size();
}

int32_t llama_moe_plan_route_limit(const llama_moe_route_profile & profile,
                                   const std::vector<int32_t> & routes_per_expert,
                                   const std::vector<char> & has_weights,
                                   int32_t default_limit, int32_t n_gpu) {
    if (!profile.initialized) {
        return default_limit;
    }
    std::vector<llama_moe_expert_cost> costs;
    costs.reserve(routes_per_expert.size());
    for (size_t e = 0; e < routes_per_expert.size(); ++e) {
        const int32_t r = routes_per_expert[e];
        if (r <= 0) {
            continue;
        }
        if (!has_weights.empty() && (e >= has_weights.size() || !has_weights[e])) {
            continue;
        }
        costs.push_back({r, profile.cpu.at(r), profile.gpu.at(r)});
    }
    return llama_moe_select_route_limit(costs, default_limit, n_gpu);
}

int32_t llama_moe_route_limit_parse_override(const char * s) {
    if (!s || *s < '0' || *s > '9') {
        return -1;
    }
    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 1 || v > (1L << 20)) {
        return -1;
    }
    return (int32_t) v;
}
