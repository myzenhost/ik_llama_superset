// llama-moe-route-limit.h -- measured per-layer CPU/GPU expert split for MoE layers whose experts are CPU-resident.
//
// RESEARCH-DONE: github.com/ztxz16/fastllm src/devices/numas/moeexpertpartition.h (58 lines) and
// numasdevice.cpp:5715-5998 read in full 2026-10-05 (clone upstream/fastllm @ a2bf07fd, Apache-2.0).
//
// WHAT. For one MoE layer and one batch, every active expert has a ROUTE COUNT (how many tokens were routed to
// it). An expert with few routes is cheap on the CPU (its weights are already there); an expert with many routes
// amortises a weight upload and runs far faster on the GPU. A single static batch-size threshold (the current
// rule in ggml-cuda.cu: get_op_batch_size(op) >= op_offload_min_batch_size, MUL_MAT_ID using the TOTAL token
// count ne[2]) sends every expert of the layer the same way. This module picks the threshold T per layer and per
// batch: experts with routes < T stay on the CPU, experts with routes >= T go to the GPU, and T minimises the
// layer's makespan = max(sum of CPU times, longest GPU queue) from MEASURED cost curves.
//
// SCOPE (stated so nothing here is read as more than it is). This file is the planner only: the cost curves,
// the exact threshold search, the micro-benchmark driver and the profile cache. It is pure host code with no ggml
// dependency, so it is unit-tested (tests/test-moe-route-limit.cpp). It is NOT yet consumed by the live graph:
// step 0 of the row found that no path splits ONE MUL_MAT_ID across backends. ggml_backend_sched's
// only_active_experts mode (ggml/src/ggml-backend.cpp:1761-1844) stages just the active expert slices to ONE
// backend and that backend runs the whole op. The splice that would run experts below T on the CPU backend and
// the rest on the GPU backend is the open work, filed as its own row; this planner is its decision function.
//
// DESIGN REFERENCE (behaviour only, own code): ztxz16/fastllm @ a2bf07fd (Apache-2.0),
// src/devices/numas/moeexpertpartition.h:19-53 SelectMoeExpertLimit and src/devices/numas/numasdevice.cpp
// :5715-5998 MoeExpertSpeedEstimator. No code copied.
//
// NUANCES THE ROW TEXT DID NOT SAY (found by reading those two files):
//   1. Candidate thresholds are {1} U {routes+1 of every active expert}. T=1 means "everything on the GPU"; the
//      largest candidate means "everything on the CPU". The set is the complete set of distinct partitions, so the
//      search is exact, not a sweep.
//   2. Ties pick the SMALLEST T (the candidate set is ordered ascending and the comparison is strict '<'), i.e. on
//      a tie more experts go to the GPU.
//   3. Experts with routes <= 0 are skipped entirely (they cost nothing on either side).
//   4. The GPU side is a longest-processing-time-first assignment over n_gpu queues (greedy onto the least loaded);
//      the layer time is max(cpu_total, max GPU queue). The CPU is modelled as ONE serial queue.
//   5. When no expert is active the caller's default is returned unchanged; the default is a fallback, never a
//      bound on the optimum (the optimum may lie on either side of it).
//   6. Cost curves are interpolated linearly between samples; past the last sample they are EXTRAPOLATED with the
//      slope of the last two samples and clamped at 0; the curve always carries the point (0 -> 0).
//   7. Sample points are denser at small n (step 1 up to 16, 4 up to 64, 16 up to 256, 64 up to 1024, 128 above)
//      and the CPU curve uses double the step (min 2), because CPU timing is slower to measure; the max point is
//      always sampled. Benchmark N is capped at 512 and at the batch's token count.
//   8. A profile is rebuilt when a later batch needs a larger N than the one it was built for.
//   9. On a host with no accelerator the GPU curve is a copy of the CPU curve (so the planner expresses no
//      preference either way).
//  10. An explicit user override bypasses the search entirely (parse_override below).
//
// The GPU timing callback MUST include whatever the real path pays to run an expert whose weights live on the
// host (staging the expert's slices), otherwise the planner will over-send experts to the GPU.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

struct llama_moe_expert_cost {
    int32_t routes; // tokens routed to this expert in this batch
    double  cpu_us; // time to run the expert on the CPU for `routes` tokens
    double  gpu_us; // time to run the expert on one GPU for `routes` tokens (incl. any weight staging)
};

// Exact threshold search. Returns T such that experts with routes < T run on the CPU and the rest on the GPU,
// minimising max(cpu_total, longest GPU queue). Returns default_limit when no expert has routes > 0.
int32_t llama_moe_select_route_limit(const std::vector<llama_moe_expert_cost> & experts,
                                     int32_t default_limit, int32_t n_gpu);

// Makespan of the partition produced by threshold `limit` (what the search minimises). Exposed for tests/logging.
double llama_moe_route_makespan(const std::vector<llama_moe_expert_cost> & experts, int32_t limit, int32_t n_gpu);

// Piecewise-linear cost curve: n tokens -> microseconds.
struct llama_moe_cost_curve {
    std::map<int32_t, double> pts; // always holds (0 -> 0) once built
    double at(int32_t n) const;
};

// Sample points for a micro-benchmark up to max_n (see nuance 7).
std::vector<int32_t> llama_moe_route_sample_points(int32_t max_n, bool is_cpu);

constexpr int32_t LLAMA_MOE_ROUTE_BENCH_MAX_N    = 512;
constexpr int32_t LLAMA_MOE_ROUTE_WARMUP_ROUNDS  = 2;
constexpr int32_t LLAMA_MOE_ROUTE_MEASURE_ROUNDS = 8;

struct llama_moe_route_profile {
    int32_t              max_n = 0;
    bool                 initialized = false;
    llama_moe_cost_curve cpu;
    llama_moe_cost_curve gpu;
};

// time_*_us(n) runs the expert FFN once for n routed tokens and returns the elapsed microseconds of that one run.
// A null time_gpu_us means "no accelerator": the GPU curve becomes a copy of the CPU curve (nuance 9).
// Each sample point is run LLAMA_MOE_ROUTE_WARMUP_ROUNDS times unmeasured, then LLAMA_MOE_ROUTE_MEASURE_ROUNDS
// times, and the mean is stored. Returns false (profile untouched) when max_n <= 0 or time_cpu_us is null.
bool llama_moe_route_profile_build(llama_moe_route_profile & out, int32_t max_n,
                                   const std::function<double(int32_t)> & time_cpu_us,
                                   const std::function<double(int32_t)> & time_gpu_us);

// What a profile depends on (one expert FFN shape on one device).
struct llama_moe_route_shape_key {
    int32_t n_embd = 0, n_ff_exp = 0, n_embd_out = 0, n_expert_used = 0;
    int32_t type_up = 0, type_down = 0;
    int32_t gpu_id = 0;
    bool operator<(const llama_moe_route_shape_key & o) const;
};

// Thread-safe cache of profiles; rebuilds when a larger N than the stored one is requested (nuance 8).
class llama_moe_route_profile_cache {
public:
    // Returns the profile (copied out) or false when it could not be built.
    bool get(const llama_moe_route_shape_key & key, int32_t want_n,
             const std::function<double(int32_t)> & time_cpu_us,
             const std::function<double(int32_t)> & time_gpu_us,
             llama_moe_route_profile & out);
    size_t size() const;
private:
    mutable std::mutex mtx;
    std::map<llama_moe_route_shape_key, llama_moe_route_profile> profiles;
};

// Plan one layer: routes_per_expert[e] = tokens routed to expert e (0 for inactive). Experts with
// has_weights[e] == 0 are skipped (an empty vector means every expert is present). Returns the threshold, or
// default_limit when the profile is not usable.
int32_t llama_moe_plan_route_limit(const llama_moe_route_profile & profile,
                                   const std::vector<int32_t> & routes_per_expert,
                                   const std::vector<char> & has_weights,
                                   int32_t default_limit, int32_t n_gpu);

// Parses an explicit override such as an environment value: a positive integer in [1, 1<<20] yields it, anything
// else (null, empty, non-numeric, trailing junk, out of range, zero, negative) yields -1 = "no override".
int32_t llama_moe_route_limit_parse_override(const char * s);
