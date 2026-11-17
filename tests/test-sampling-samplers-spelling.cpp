// Upstream's common_params_sampling::samplers spelling on this fork's samplers_sequence chain, and the INFILL step it brought
// (row S69-ENGREV-UPSTREAM-MERGE-SAMPLERS-SPELLING-INFILL-2026-11-05-01). No model needed: common_sampler_init(nullptr, ...).
// Fails before the change: `samplers`, COMMON_SAMPLER_TYPE_INFILL -> llama_sampler_type::INFILL and the resolve helpers did not exist.
#include "sampling.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

int main() {
    using T = llama_sampler_type;

    // 1) unset (the default): samplers_sequence drives the chain, untouched
    {
        common_params_sampling sp;
        CHECK(common_sampler_samplers_unset(sp));
        CHECK(common_sampler_resolved_sequence(sp) == sp.samplers_sequence);
    }
    // 2) set: mapped 1:1; PENALTIES has no position in this chain (penalties run before the queue) and is dropped from the sequence
    {
        common_params_sampling sp;
        sp.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_INFILL, COMMON_SAMPLER_TYPE_TEMPERATURE };
        CHECK(!common_sampler_samplers_unset(sp));
        const std::vector<T> want = { T::TOP_K, T::INFILL, T::TEMPERATURE };
        CHECK(common_sampler_resolved_sequence(sp) == want);
        CHECK(common_sampler_needs_host(sp)); // the backend chain has no vocab handle for infill
        common_params_sampling folded = sp;
        CHECK(common_sampler_resolve_samplers(folded));
        CHECK(folded.samplers_sequence == want);
        CHECK(common_sampler_samplers_unset(folded)); // folded once; a later samplers_sequence write is not overridden
    }
    // 3) an empty list is upstream's "no samplers": an empty chain, not the default one
    {
        common_params_sampling sp;
        sp.samplers.clear();
        CHECK(!common_sampler_samplers_unset(sp));
        CHECK(common_sampler_resolved_sequence(sp).empty());
    }
    // 4) init folds it: gsmpl->params carries the mapped sequence, and INFILL creates its sampler
    {
        common_params_sampling sp;
        sp.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_INFILL };
        common_sampler * s = common_sampler_init(nullptr, sp);
        CHECK(s != nullptr);
        const std::vector<T> want = { T::TOP_K, T::INFILL };
        CHECK(s->params.samplers_sequence == want);
        CHECK(common_sampler_samplers_unset(s->params));
        CHECK(s->infill != nullptr);
        common_params_sampling plain;
        plain.samplers_sequence = { T::TOP_K, T::TEMPERATURE }; // no DRY: it needs a vocab (test-sampling-adaptive-order.cpp)
        common_sampler * c = common_sampler_init(nullptr, plain);
        CHECK(c != nullptr && c->infill == nullptr); // a chain without INFILL has no infill sampler
        common_sampler_clone(s, c);
        CHECK(c->infill != nullptr && c->infill != s->infill);
        common_sampler_free(c);
        common_sampler_free(s);
    }
    // 5) the fork's own spellings reach it too
    {
        const auto by_name = llama_sampling_types_from_names({ "top_k", "infill" }, false);
        CHECK(by_name.size() == 2 && by_name[1] == T::INFILL);
        const auto by_char = llama_sampling_types_from_chars("ki");
        CHECK(by_char.size() == 2 && by_char[1] == T::INFILL);
        CHECK(llama_sampling_type_to_str(T::INFILL) == "infill");
    }
    // 6) control: with `samplers` unset the sequence passes through untouched and no infill sampler exists
    {
        common_params_sampling sp;
        sp.samplers_sequence = { T::TOP_K, T::TOP_P, T::TEMPERATURE }; // no DRY: it needs a vocab
        common_sampler * s = common_sampler_init(nullptr, sp);
        CHECK(s != nullptr && s->infill == nullptr);
        const std::vector<T> want = { T::TOP_K, T::TOP_P, T::TEMPERATURE };
        CHECK(s->params.samplers_sequence == want);
        common_sampler_free(s);
    }
    printf("test-sampling-samplers-spelling: OK\n");
    return 0;
}
