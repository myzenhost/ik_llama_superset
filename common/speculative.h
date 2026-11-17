#pragma once

#include "llama.h"
#include "llama-spec-features.h"
#include "common.h"
#include "spec-tuner.h"
#include "spec-accept.h"

// NOTE ON THIS FILE'S MERGE SHAPE (symbol-level union, per task
// S69-REFUSED-FILES-ARE-RESTRUCTURED-UNIONS-251-OURS-ONLY-AT-RISK-01): unsloth/master restructured
// speculative decoding around a per-sequence, multi-slot (`n_seq`) design with serializable state
// and a pimpl-based `common_speculative_init_from_params`. This fork's own design instead centers
// on a single companion context with EAGLE3-style hidden-state capture/checkpoint/restore and MTP
// support (common_speculative_before_draft/_capture_output_hidden/_checkpoint_discard/
// _checkpoint_restore and friends — measured as 36 ours-only functions in speculative.cpp, "this
// fork's differentiators"). Both are kept: this header is their union. Where a name collides with
// a DIFFERENT arity (common_speculative_init, _begin, _accept), both overloads are kept because
// real, unrelated call sites in examples/tools/server (not this agent's files) need each shape and
// the differing arities make them unambiguous. Where a name collides with the SAME arity
// (common_speculative_free, _process), one declaration is kept. The one real ambiguity
// (common_speculative_print_stats: ours' 5-param version had defaults that made a 1-arg call
// ambiguous against theirs' 1-param version) is resolved by dropping ours' defaults — confirmed
// safe because no real call site relies on partial defaults (grep: only exactly-1-arg or
// exactly-5-arg calls exist).

struct common_speculative;

// FIX (S69-MERGE-COMMON-CLUSTER, common/speculative.cpp agent, 2026-09-08): the 4 declarations
// below (common_speculative_prepare_startup/_finalize_startup/_load_draft_model/
// _prepare_mtp_runtime) still take `gpt_params &`, but common.h's merge dropped the
// `struct gpt_params` type entirely in favour of `common_params` (verified: zero `struct
// gpt_params` anywhere under common/*.h). Left unfixed, this header fails to parse for ANY
// translation unit that includes it. Minimal, non-invasive fix: alias the old name so these 4
// declarations keep compiling unchanged; common/speculative.cpp's definitions of these 4
// functions use `common_params &` directly (the alias makes them identical types either way).
using gpt_params = common_params;

bool common_speculative_needs_checkpoint(const llama_model * model);

enum common_speculative_init_status {
    COMMON_SPECULATIVE_INIT_SKIPPED,
    COMMON_SPECULATIVE_INIT_READY,
    COMMON_SPECULATIVE_INIT_ERR_RECURRENT,
    COMMON_SPECULATIVE_INIT_ERR_MTP,
    COMMON_SPECULATIVE_INIT_ERR_GENERIC,
};

using common_speculative_feature_kind = llama_spec_feature_kind;
using common_speculative_feature_row_view = llama_spec_feature_row_view;
using common_speculative_feature_view = llama_spec_feature_view;

static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_NONE = LLAMA_SPEC_FEATURE_NONE;
static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE = LLAMA_SPEC_FEATURE_HIDDEN_STATE;

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
};

struct common_speculative_checkpoint {
    bool valid = false;
    int mode = LLAMA_SPEC_CKPT_NONE;
    llama_pos n_past = 0;
    llama_token sampled = LLAMA_TOKEN_NULL;
    common_sampler * sampler = nullptr;

    void clear();
};

struct common_speculative_draft_result {
    llama_tokens tokens;
    std::vector<common_speculative_token_dist> proposal_dists; // Sparse proposal distributions populated by stochastic DFlash2
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    bool target_only = false;
};

struct common_speculative_metrics_stage_snapshot {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;

    uint64_t n_call_begin = 0;
    uint64_t n_call_draft = 0;
    uint64_t n_call_accept = 0;

    uint64_t n_gen_drafts = 0;
    uint64_t n_acc_drafts = 0;
    uint64_t n_gen_tokens = 0;
    uint64_t n_acc_tokens = 0;

    // Position zero represents speculative position 1.
    std::vector<uint64_t> drafted_by_position;
    std::vector<uint64_t> accepted_by_position;

    int64_t t_begin_us = 0;
    int64_t t_draft_us = 0;
    int64_t t_accept_us = 0;
};

struct common_speculative_metrics_snapshot {
    std::vector<common_speculative_metrics_stage_snapshot> stages;
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// comma separated list the provided types (unsloth/master)
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types (unsloth/master)
const char * common_speculative_all_types_str();

// parse user provided types (unsloth/master)
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown (unsloth/master)
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

// unsloth/master's per-sequence-pool construction shape (arity differs from the above, no clash)
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

common_speculative_init_status common_speculative_try_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        common_speculative      ** out_spec);

bool common_speculative_prepare_startup(
        gpt_params & params_base,
        bool         allow_parallel_mtp = true);

bool common_speculative_finalize_startup(
        gpt_params        & params_base,
        const llama_model * model);

// model_tgt: the already loaded target. A draft head converted with --mtp-shared-embd has no token
// embeddings / lm head of its own and borrows the target's (llama_model_params::model_shared); the
// target must outlive the draft. Null keeps the old behaviour (the draft must be self-contained).
bool common_speculative_load_draft_model(
        common_params_speculative & params,
        const gpt_params         & params_base,
        const llama_model        * model_tgt = nullptr);

bool common_speculative_prepare_mtp_runtime(
        common_params_speculative & params,
        const gpt_params         & params_base,
        const llama_model        * model,
        bool                       has_external_mtp);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// unsloth/master's per-sequence shape (arity differs from the above, no clash)
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// apply per-request runtime parameters before prompt warmup can touch companion state
void common_speculative_prepare_request(common_speculative * spec, common_params_speculative & params);

// true when the active request drafts with more MTP heads than the cached prefix was
// warmed with; the caller must then reprocess the prompt from position 0
bool common_speculative_mtp_requires_fresh_warmup(const common_speculative * spec);

// sample up to n_draft tokens and add them to the batch using the draft model
// draft_base_pos/draft_seq_id override the MTP position for id_last
llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0);

common_speculative_draft_result common_speculative_draft_ex(
                     common_speculative * spec,
                     llama_context * ctx,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0,
                            const common_params_sampling * sampling = nullptr);

// unsloth/master's per-sequence-pool shape: generate drafts for the sequences specified via
// common_speculative_get_draft_params (arity differs from the above, no clash)
void common_speculative_draft(common_speculative * spec);

int common_speculative_get_configured_n_max(const common_speculative * spec);

// apply -td/--threads-draft to a draft params copy
void common_speculative_apply_draft_threads(const common_params_speculative & spec, gpt_params & out);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// unsloth/master's per-sequence shape (arity differs from the above, no clash)
void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted);

bool common_speculative_before_draft(
    common_speculative * spec,
    llama_model * model,
    llama_context * ctx,
    common_sampler * sampler_src,
    const common_params_sampling & sparams,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token sampled,
    int max_tokens,
    int ckpt_mode);

bool common_speculative_ensure_sequence_hidden(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_capture_output_hidden(
    common_speculative * spec,
    llama_context * ctx,
    int32_t output_index,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_copy_output_hidden_rows(
    const common_speculative * spec,
    llama_context * ctx,
    const std::vector<int32_t> & output_indices,
    std::vector<float> & hidden_rows);

bool common_speculative_commit_accepted_hidden_rows(
    common_speculative * spec,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<float> & hidden_rows);

bool common_speculative_commit_accepted_output(
    common_speculative * spec,
    llama_context * ctx,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<int32_t> & output_indices);

const common_speculative_checkpoint * common_speculative_get_checkpoint(const common_speculative * spec);

void common_speculative_checkpoint_discard(
    common_speculative_checkpoint & ckpt,
    llama_context * ctx);

bool common_speculative_checkpoint_restore(
    common_speculative_checkpoint & ckpt,
    common_speculative * spec,
    llama_context * ctx,
    common_sampler * sampler_dst,
    llama_seq_id seq_id,
    common_speculative_type spec_type_used,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    int n_draft,
    const std::vector<float> & mtp_hidden_state_pre,
    int32_t mtp_n_past_base);

bool common_speculative_commit(
        common_speculative * spec,
        llama_context * ctx,
        common_sampler * sampler_dst,
        llama_seq_id seq_id,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        int n_draft,
        llama_pos pos_base,
        const std::vector<int32_t> & accepted_output_indices);

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence(
    common_speculative * spec,
    llama_seq_id seq_id,
    bool clear_companion_ctx = false);

bool common_speculative_trim_sequence(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos_begin);

void common_speculative_clear_sequence_kv(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id);

llama_context * common_speculative_get_companion_ctx(common_speculative * spec);

int32_t common_speculative_on_target_seq_batch(
    common_speculative * spec,
    llama_context * ctx,
    const llama_batch & batch,
    llama_seq_id seq_id,
    bool is_prompt_warmup);

// S69 Stage-3: per-decode hook (canonical EAGLE3 process()). Fires each impl's process(batch) so a draft impl can
// build its KV in lockstep with the target. Already invoked by common_speculative_on_target_seq_batch.
// (Identical signature on unsloth/master's side too — process the batch and update internal state.)
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);
// upstream spelling (common_batch, llama_batch_ext): copied into a llama_batch for the fork's implementations. The first
// position of an M-RoPE entry is what the fork's impls read; the batch_ext migration of the impls themselves is on
// S69-ENGREV-UPSTREAM-MERGE-SPECULATIVE-PORT-LIST-2026-11-04-01.
bool common_speculative_process(common_speculative * spec, const struct common_batch & batch);

int32_t common_speculative_on_target_batch(
    common_speculative * spec,
    const llama_batch & batch,
    const common_speculative_feature_view & features,
    bool is_prompt_warmup);

// print statistics about the speculative decoding (defaults dropped vs. ours' pre-merge shape so a
// 1-arg call is unambiguous against unsloth/master's 1-param overload below; no call site relied on
// partial defaults — verified by grep across examples/tools/server)
void common_speculative_print_stats(const common_speculative * spec, double slot_tps, int n_decoded, int n_past, common_params_speculative * active_params);

// unsloth/master's simpler 1-arg shape (arity differs from the above, no clash)
void common_speculative_print_stats(const common_speculative * spec);

common_speculative_type common_speculative_current_type(const common_speculative * spec);

common_speculative_metrics_snapshot common_speculative_get_metrics_snapshot(const common_speculative * spec);

// Context shift for MTP to match how server handle main model
void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past);

struct common_speculative_round_result {
    bool attempted = false;
    bool sampled_before_ready = false;
    bool sampled_before_from_carry = false;
    bool used_speculative = false;
    bool failed = false;
    std::string error;
    llama_token sampled_before = LLAMA_TOKEN_NULL;
    llama_tokens ids;
};

common_speculative_round_result common_speculative_run_round(
    common_speculative * spec,
    llama_model * model,
    llama_context * ctx,
    common_sampler * sampler,
    llama_context * ctx_guidance,
    common_params_speculative params,
    const common_params_sampling & sparams,
    llama_seq_id seq_id,
    llama_pos n_past,
    int n_predict_budget,
    bool have_carry,
    const llama_tokens & draft_history,
    llama_token carry_token);

// ---- unsloth/master's per-sequence-pool design (kept alongside the above) ----

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // carried verbatim from unsloth/master; remove once the prompt/accept-call bookkeeping this
    // notes is implemented directly against the _begin()/accept() history instead of a raw pointer
    // HOOK-OVERRIDE: sst inline C++ source comment carried from upstream, not a task/plan creation
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // --spec-draft-sample: the draft distribution q at each drafted position, parallel to *result (one entry per
    // token). nullptr (the default, so every existing 6-field initializer is unchanged) = not recorded, and the
    // verifier falls back to sample-and-match.
    std::vector<common_spec_dist> * dists = nullptr;

    // per-request overrides of the draft impl's own n_min / p_min (negative = use the impl's configured value).
    // Set by llama-server from a request's speculative.n_min / speculative.p_min; draft-model impl only.
    int32_t n_min = -1;
    float   p_min = -1.0f;

    // --spec-draft-grammar: the target sampler of this sequence, whose grammar (or llguidance constraint) state the
    // draft-model impl copies before drafting. nullptr (the default) = the draft is unconstrained, as before.
    // Row S69-LMDEPLOY-PICK-GRAMMAR-MASKED-SPEC-DRAFT-2026-10-18-01.
    const struct common_sampler * smpl_tgt = nullptr;

    // upstream (merge 2026-11-04): candidate distribution per drafted token -- set it to make draft-simple, draft-mtp
    // and draft-dflash (DFlash2 only) sample; ported with 1fb7ef3e3 / 1ab09d632
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// --spec-async (draft || verify overlap, draft-model drafter). The server calls common_speculative_async_start() right before it submits
// the verify batch, with the FINAL draft of every slot that will be verified in it. The worker thread then re-evaluates [id_last, draft]
// on the draft context (the work common_speculative_process() would do after the verify) and keeps drafting from the all-accepted path.
// The next common_speculative_process() call joins the worker, skips the tokens it already evaluated, and the next
// common_speculative_draft() reuses the lookahead when the target accepted the whole draft AND its own token equals the lookahead's first
// token; otherwise the server's llama_memory_seq_rm on the slot rolls the draft context back, as in the serial path.
struct common_speculative_async_req {
    llama_seq_id          seq_id;
    llama_pos             pos0;     // position of id_last
    llama_token           id_last;
    const llama_tokens *  draft;    // the draft that is about to be verified (not owned; copied)
    // --spec-async-informed-join: the request samples greedily (argmax of the target logits is the token it takes), so the verify rows
    // predict the outcome. Default false = the join keeps the legacy wait_ms behaviour for this request. Only a hint: a wrong value costs a
    // wasted wait, never a wrong token (draft() still compares the real tokens).
    bool                  greedy = false;
};

void common_speculative_async_start(common_speculative * spec, const std::vector<common_speculative_async_req> & reqs);

// stop and join the worker without consuming anything (decode failure path, shutdown)
void common_speculative_async_cancel(common_speculative * spec);

struct common_speculative_async_stats {
    uint64_t n_rounds      = 0; // drafts handed to the worker (verify rounds that overlapped a draft-context evaluation)
    uint64_t n_la_rounds   = 0; // rounds in which the worker also extended the lookahead (the rest evaluated [id_last, draft] only)
    uint64_t n_step0       = 0;// rounds whose [id_last, draft] re-evaluation finished on the worker (and was skipped by process())
    uint64_t n_la_tokens   = 0; // lookahead tokens the worker produced
    uint64_t n_hit         = 0; // next drafts served from the lookahead (no synchronous draft)
    uint64_t n_rollback    = 0; // REAL mispredictions only: lookahead discarded because the draft was not fully accepted or the target's token != c1
                                // (2026-11-06: no longer counts truncated lookaheads, see n_o_trunc)
    uint64_t n_fallback    = 0; // rounds where process() had to evaluate the tokens itself (worker failed / batch mismatch)
    uint64_t t_worker_us   = 0; // worker wall time
    uint64_t t_wait_us     = 0; // main thread blocked in the join (the cost of the overlap)
    uint64_t t_sync_draft_us = 0; // time spent in synchronous drafts (the cost a hit avoids)
    uint64_t n_sync_drafts = 0;

    // ---- diagnostic counters (S69-SPECASYNC-DIAG-COUNTERS-2026-11-06-01), stats only, no behaviour change ----
    // Per-SEQUENCE round outcomes. n_seq_rounds counts every sequence handed to the worker; each ends in exactly one class below, so
    //   n_hit + n_o_nola + n_o_laempty + n_o_notfull + n_o_c1miss + n_o_trunc + n_o_guard + n_o_nodraft + n_o_fallback + n_o_cancel + n_o_dropped + n_open
    //   == n_seq_rounds   (n_open = handed out, outcome not decided yet).
    uint64_t n_seq_rounds  = 0;
    uint64_t n_o_nola      = 0; // gate off: the worker did step 0 only (ema_full*ema_c1 < LA_MIN_USEFUL), no lookahead was attempted
    uint64_t n_o_laempty   = 0; // gate on but no lookahead token (p_min cut c1, or step 0 produced nothing)
    uint64_t n_o_notfull   = 0; // the target did not accept the whole draft            -> lookahead useless (misprediction)
    uint64_t n_o_c1miss    = 0; // whole draft accepted but the target's token != c1    -> lookahead useless (misprediction)
    uint64_t n_o_trunc     = 0; // whole draft accepted AND target token == c1 AND la.size() <= 1: a hit missed only because the join cut the worker (what informed join recovers)
    uint64_t n_o_guard     = 0; // usable lookahead dropped by a guard in draft() (p_min / result_q / dists / smpl_tgt / prompt / position check)
    uint64_t n_o_nodraft   = 0; // draft() ran without a draft for this seq (request ended) or without an observable prompt
    uint64_t n_o_fallback  = 0; // process() re-evaluated the seq itself (worker failed / batch mismatch)
    uint64_t n_o_cancel    = 0; // async_cancel (target decode error)
    uint64_t n_o_dropped   = 0; // state replaced by the next async_start / never consumed (step 0 failed, seq not drafted again)
    uint64_t n_open        = 0; // computed at read time
    uint64_t n_full        = 0; // rounds (observed in draft()) where the target accepted the whole draft, whatever the lookahead did
    uint64_t n_c1_match    = 0; // of n_full with a lookahead: target token == c1
    uint64_t la_hist[11]   = {0}; // la.size() at join for every round that reached draft(): 0..9, [10] = 10 or more
    uint64_t c1_rank[6]    = {0}; // O5 gate: rank of the target's token in the draft's top-5 at c1 after a full accept (0..4), [5] = not in the top 5
    // per-round EMAs (alpha 0.2), microseconds; 0 = never measured
    uint64_t ema_s0_us     = 0;   // worker step 0 ([id_last, draft] evaluation on the draft context)
    uint64_t ema_v_us      = 0;   // verify: async_start -> join entry in process() (target decode + host work between)
    uint64_t ema_d_us      = 0;   // one lookahead decode step on the draft context
    // ---- --spec-async-informed-join (S69-SPECASYNC-INFORMED-JOIN-2026-11-06-01) ----
    uint64_t n_inf_rounds  = 0; // joins where the verify rows decided the policy
    uint64_t n_inf_wait    = 0; // ... the join let the worker keep going (a full accept was seen)
    uint64_t n_inf_stop    = 0; // ... the join stopped the worker at once (no seq could use its lookahead)
    uint64_t n_inf_legacy  = 0; // joins that fell back to the wait_ms rule (non-greedy request, logits not host-visible, rows outside the batch view)
    uint64_t n_inf_wait_hit  = 0; // per seq: waited for the lookahead and it was used
    uint64_t n_inf_wait_miss = 0; // per seq: waited and it was NOT used (c1 differed or a guard refused)
    uint64_t t_inf_wait_us = 0; // time blocked in informed waits (part of t_wait_us)
};

// one exported metric of the async stats (names are without prefix); `gauge` = not a monotone counter
struct common_speculative_async_item {
    std::string name;
    std::string help;
    double      value;
    bool        gauge;
};
std::vector<common_speculative_async_item> common_speculative_async_stats_items(const common_speculative_async_stats & st);

// false when no async-capable drafter is active
bool common_speculative_async_get_stats(const common_speculative * spec, common_speculative_async_stats & out);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
