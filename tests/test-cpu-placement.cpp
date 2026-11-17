// --cpu-placement physical: the CPU-list builder and the CLI wiring.
//
//  1. common_cpu_first_cpu_per_core on synthetic topologies (adjacent SMT pairs, split pairs, unsorted input).
//  2. Live host: common_cpu_core_siblings() (Windows GetLogicalProcessorInformationEx / Linux thread_siblings_list)
//     yields one entry per physical core, and that count equals common_cpu_get_num_physical_cores(). On the 3990X
//     (64 cores / 128 logical, two processor groups) that is 64 entries.
//  3. Parse (LLAMA_EXAMPLE_SERVER): --cpu-placement physical sets one mask bit per physical core and strict placement
//     for --threads and --threads-batch, and the mask reaches ggml_threadpool_params; an explicit -C wins in either
//     argument order; a bad value fails the parse; the draft flag touches only the speculative params.
// Exit 0 = all checks passed; 1 = at least one failed (each failure is printed).

#include "arg.h"
#include "common.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        ++g_fail;                                               \
        std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
        std::fprintf(stderr, __VA_ARGS__);                      \
        std::fprintf(stderr, "\n");                             \
    }                                                           \
} while (0)

static int popcount(const bool (&mask)[GGML_MAX_N_THREADS]) {
    int n = 0;
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        n += mask[i] ? 1 : 0;
    }
    return n;
}

static bool parse(std::vector<std::string> args, common_params & params) {
    args.insert(args.begin(), "test-cpu-placement");
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    argv.push_back(nullptr);
    return common_params_parse((int) args.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER);
}

int main() {
    // ---- 1. synthetic topologies -------------------------------------------------------------------------------------
    {
        const auto a = common_cpu_first_cpu_per_core({{0, 1}, {2, 3}});
        CHECK((a == std::vector<int32_t>{0, 2}), "adjacent pairs -> {0,2}, got %zu entries", a.size());
        const auto b = common_cpu_first_cpu_per_core({{0, 2}, {1, 3}});
        CHECK((b == std::vector<int32_t>{0, 1}), "split pairs -> {0,1}, got %zu entries", b.size());
        const auto c = common_cpu_first_cpu_per_core({{3, 1}, {2, 0}});
        CHECK((c == std::vector<int32_t>{0, 1}), "unsorted input -> {0,1}, got %zu entries", c.size());
        const auto d = common_cpu_first_cpu_per_core({{5}, {}, {4}});
        CHECK((d == std::vector<int32_t>{4, 5}), "no-SMT cores and an empty core -> {4,5}, got %zu entries", d.size());

        common_cpu_params p;
        p.place_physical = false;
        CHECK(!common_cpu_apply_physical_placement(p, {0, 2}), "placement off must not touch the mask");
        CHECK(popcount(p.cpumask) == 0 && !p.mask_valid, "placement off left a mask");
        p.place_physical = true;
        p.mask_valid     = true;   // an explicit -C mask
        p.cpumask[7]     = true;
        CHECK(!common_cpu_apply_physical_placement(p, {0, 2}), "an explicit mask must win");
        CHECK(popcount(p.cpumask) == 1 && p.cpumask[7], "explicit mask was changed");

        bool phys = false;
        CHECK(common_cpu_placement_parse("physical", phys) && phys, "parse physical");
        CHECK(common_cpu_placement_parse("none", phys) && !phys, "parse none");
        CHECK(!common_cpu_placement_parse("bogus", phys), "parse bogus must fail");
    }

    // ---- 2. live host --------------------------------------------------------------------------------------------------
    const auto cores    = common_cpu_core_siblings();
    const auto cpus     = common_cpu_first_cpu_per_core(cores);
    const int  n_phys   = common_cpu_get_num_physical_cores();
    int        n_logic  = 0;
    for (const auto & core : cores) {
        n_logic += (int) core.size();
    }
    std::printf("host: %zu physical cores from the topology query, %d logical CPUs in them, "
                "common_cpu_get_num_physical_cores() = %d\n", cores.size(), n_logic, n_phys);
    std::printf("host: first cpu per core:");
    for (size_t i = 0; i < cpus.size(); ++i) {
        std::printf(" %d", cpus[i]);
    }
    std::printf("\n");
#if defined(_WIN32) || defined(__linux__)
    CHECK(!cores.empty(), "the topology query returned no cores on an OS that has an arm for it");
#endif
    if (!cores.empty()) {
        CHECK((int) cpus.size() == (int) cores.size(), "one entry per core: %zu entries vs %zu cores", cpus.size(), cores.size());
        CHECK((int) cpus.size() == n_phys, "entries %zu vs common_cpu_get_num_physical_cores %d", cpus.size(), n_phys);
        std::set<int32_t> seen;
        for (const auto & core : cores) {
            int hits = 0;
            for (int32_t cpu : core) {
                CHECK(seen.insert(cpu).second, "logical cpu %d appears in two cores", cpu);
                for (int32_t chosen : cpus) {
                    hits += chosen == cpu ? 1 : 0;
                }
            }
            CHECK(hits == 1, "a core has %d chosen cpus (want exactly 1)", hits);
        }
    }

    // ---- 3. CLI wiring --------------------------------------------------------------------------------------------------
    {
        common_params params;
        CHECK(parse({"--cpu-placement", "physical"}, params), "parse --cpu-placement physical");
        if (!cores.empty()) {
            CHECK(popcount(params.cpuparams.cpumask) == (int) cpus.size(), "main mask %d bits vs %zu cores",
                  popcount(params.cpuparams.cpumask), cpus.size());
            CHECK(params.cpuparams.mask_valid && params.cpuparams.strict_cpu, "main mask_valid/strict");
            CHECK(popcount(params.cpuparams_batch.cpumask) == (int) cpus.size(), "batch mask %d bits",
                  popcount(params.cpuparams_batch.cpumask));
            CHECK(params.cpuparams_batch.mask_valid && params.cpuparams_batch.strict_cpu, "batch mask_valid/strict");
            for (int32_t cpu : cpus) {
                CHECK(params.cpuparams.cpumask[cpu], "main mask misses cpu %d", cpu);
            }
            const ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(params.cpuparams);
            CHECK(popcount(tpp.cpumask) == (int) cpus.size() && tpp.strict_cpu, "threadpool params: %d bits strict=%d",
                  popcount(tpp.cpumask), (int) tpp.strict_cpu);
        }
    }
    {
        common_params params;
        CHECK(parse({"--cpu-placement", "physical", "-C", "0x3"}, params), "parse physical then -C");
        CHECK(popcount(params.cpuparams.cpumask) == 2 && params.cpuparams.cpumask[0] && params.cpuparams.cpumask[1],
              "physical then -C 0x3: %d bits", popcount(params.cpuparams.cpumask));
        CHECK(!params.cpuparams.strict_cpu, "physical then -C: strict must stay at its default");
    }
    {
        common_params params;
        CHECK(parse({"-C", "0x3", "--cpu-placement", "physical"}, params), "parse -C then physical");
        CHECK(popcount(params.cpuparams.cpumask) == 2, "-C 0x3 then physical: %d bits", popcount(params.cpuparams.cpumask));
        CHECK(!params.cpuparams.strict_cpu, "-C then physical: strict must stay at its default");
    }
    {
        common_params params;
        CHECK(!parse({"--cpu-placement", "bogus"}, params), "--cpu-placement bogus must fail the parse");
    }
    {
        common_params params;
        CHECK(parse({"--cpu-placement", "none"}, params), "parse --cpu-placement none");
        CHECK(popcount(params.cpuparams.cpumask) == 0 && !params.cpuparams.mask_valid, "none must leave the mask empty");
    }
    {
        common_params params;
        CHECK(parse({"--cpu-placement-draft", "physical"}, params), "parse --cpu-placement-draft physical");
        CHECK(popcount(params.cpuparams.cpumask) == 0 && !params.cpuparams.mask_valid, "draft flag touched the main mask");
        CHECK(popcount(params.cpuparams_batch.cpumask) == 0, "draft flag touched the batch mask");
        if (!cores.empty()) {
            CHECK(popcount(params.speculative.cpuparams.cpumask) == (int) cpus.size() && params.speculative.cpuparams.strict_cpu,
                  "draft mask %d bits", popcount(params.speculative.cpuparams.cpumask));
        }
    }

    std::printf("%s: %d failed check(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
