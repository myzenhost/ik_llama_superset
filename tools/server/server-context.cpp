#include "server-context.h"
#include <atomic>
#include <thread>
#include "base64.hpp"
#include "ckpt-dominance.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-responses-compact.h"
#include "server-asr-format.h"
#include "server-asr-vibevoice.h"
#include "server-asr-words.h"
#include "server-decode-fail-latch.h"
#include "server-lora-pool.h"
#include "server-radix-cache.h"
#include "server-lora-fair.h"
#include "server-repetition.h"
#include "server-loop-recovery.h"
#include "server-toolloop.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-shed.h"
#include "server-schema.h"
#include "server-refused-fields.h"
#include "server-draft-schedule.h"
#include "server-embd-quant.h"
#include "server-embd-chunk.h"
#include "server-truncate.h"
#include "server-classify.h"
#include "server-jump-forward.h"
#include "server-stream.h"
#include "server-reasoning-count.h"
#include "server-slot-evict.h"
#include "server-slot-purge.h"
#include "server-slot-victim.h"
#include "server-swa-reuse.h"
#include "server-ollama-util.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "watermark.h"
#include "speculative.h"
#include "token-ban.h"
#include "asr-compression.h"
#include "asr-fallback.h"
#include "asr-diarize.h"
#include "asr-vad-gate.h"
#include "asr-vad-trim.h"
#include "prompt-reuse.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "decision-engine.h"
#include "base64.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <regex>
#include <set>
#include <cstddef>
#include <cinttypes>
#include <deque>
#include <exception>
#include <memory>
#include <atomic>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>
#include <chrono>
#include "server-watchdog.h"
#include "server-kv-audit.h"
#include <ctime>
#include <iomanip>
#include <sstream>

// the speculative-decode checkpoint slot (ik cf0b666df): the rewind checkpoints live in their own buffers
static const llama_ckpt_data server_spec_ckpt = { LLAMA_CHECKPOINT_SPEC, 0 };

// G3 SQL session store (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01).
// SOURCE-READ: examples/server/server.cpp:43-54, 570-571, 1991-2096, 2225-2235 @ 35f0cac75 (local
// worktree copy of the pre-merge fork; MIT-licensed fork code, reimplemented against the same
// vendored sqlite_modern_cpp wrapper already carried in tools/server/sqlite_modern_cpp/ -- no code
// copied verbatim). Only pulled in when this target is built with SQLITE3_MODERN_CPP_SUPPORT
// defined (CMake option LLAMA_SERVER_SQLITE3, default OFF, tools/server/CMakeLists.txt); kept out
// of server-context.h entirely so no other translation unit needs the vendored sqlite3.h.
#ifdef SQLITE3_MODERN_CPP_SUPPORT
#include <sqlite_modern_cpp.h>
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (!params.mmproj.path.empty()) {
        const auto mcaps = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        if (mcaps.gen_audio) {
            // some TTS models decode with embeddings output for the whole batch
            return { params.n_batch, params.n_batch };
        }
    }

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    // /v1/decision scores one output row per branch sequence, all in the same ubatch (0 when the endpoint is off)
    result.total += params.n_seq_decision;

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, ctx, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, ctx, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, ctx, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, ctx, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, ctx, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()

    struct token {
        int32_t id_slot;
        llama_token token;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        bool output;
        bool is_prompt; // for stats tracking
        // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01):
        // true on a CFG-active slot's MAIN token entry when its guidance-token entry immediately
        // follows (server_slot::handle_last_sampled_token adds them back-to-back). Never true on
        // any other entry, including the guidance token itself. update_slots()'s chunking loop
        // reads this so a sub-batch boundary never separates the pair -- coordinator review,
        // 2026-09-27: a silently-unguided sample on a boundary split is the same defect class as
        // the original silently-ignored-field bug, just narrower.
        bool pairs_with_next = false;
        int32_t decision_order = 0;
        int32_t i_embd = -1; // row in embd, -1 if this is a token
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // embedding entries, they can be mixed with tokens if the context supports it
    std::vector<float> embd; // n_embd per row

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, { pos, 0, 0, 0 }, output, is_prompt });
        return true;
    }

    // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
    bool add_embd(int32_t id_slot, const float * embd_in, const llama_pos * pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, { pos[0], pos[1], pos[2], pos[3] }, output, is_prompt });
        tokens.back().i_embd = (int32_t) (embd.size() / n_embd);
        embd.insert(embd.end(), embd_in, embd_in + n_embd);
        return true;
    }

    bool has_embd() const {
        return !embd.empty();
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // Drop every entry at index >= off whose seq id is in `seqs` (entries before `off` were already decoded and
    // are left alone), compact the rest in order (decode() renders each sub-batch). Returns the old->new index map; removed entries
    // map to -1. Used by KV-exhaustion preemption (row S69-SERVER-KV-UNIFIED-EXHAUSTION-FAILS-ALL-SLOTS-2026-09-27-01).
    std::vector<int32_t> remove_seqs_from(int32_t off, const std::vector<int32_t> & seqs) {
        GGML_ASSERT(!has_embd()); // embd batches carry a parallel float array; not compacted here
        GGML_ASSERT(off >= 0 && off <= size());
        std::vector<int32_t> remap(tokens.size(), -1);
        for (int32_t i = 0; i < off; ++i) {
            remap[i] = i;
        }
        int32_t w = off;
        for (int32_t i = off; i < size(); ++i) {
            if (std::find(seqs.begin(), seqs.end(), tokens[i].id_slot) != seqs.end()) {
                continue;
            }
            remap[i]     = w;
            tokens[w++]  = tokens[i];
        }
        tokens.resize(w);
        return remap;
    }

    // remove the entries after the first n
    void truncate(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= size());
        for (int32_t i = n; i < size(); i++) {
            if (tokens[i].i_embd >= 0) {
                embd.resize((size_t) tokens[i].i_embd * n_embd);
                break;
            }
        }
        tokens.resize(n);
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_decision_order(int32_t idx, int32_t order) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].decision_order = order;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (t.i_embd >= 0) {
                view.add_embd({ embd.data() + (size_t) t.i_embd * n_embd, 1, (size_t) n_embd }, t.pos.data(), t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            view.tokens.back().decision_order = t.decision_order;
        }
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    struct tts_ctx {
        mtmd_helper::gen_audio ctx;
        std::vector<float> h_state_prompt; // the first h_state after step_prompt done
        const float * h_state;
        llama_token   sampled;
        int32_t       n_decoded;
        bool          stop;
        bool is_supported() const {
            return ctx.valid();
        }
        void reset() {
            ctx.reset();
            h_state_prompt.clear();
            h_state   = nullptr;
            sampled   = LLAMA_TOKEN_NULL;
            n_decoded = 0;
            stop      = false;
        }
    };
    tts_ctx tts;

    // speculative decoding
    common_speculative * spec;

    // grammar jump-forward (server-jump-forward.h): tokens decided without their own sampling step that go into the NEXT batch
    // in front of `sampled` (the last of them), logits only on `sampled`. Already accepted by the sampler and streamed.
    llama_tokens ff_before;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    // the target's recurrent state for this draft is held on the device (llama_spec_ckpt_save), not in spec_ckpt.data_tgt
    bool spec_ckpt_dev = false;
    bool spec_is_replay = false;
    // --spec-draft-sample: q per drafted position, parallel to spec_draft; empty = verify by sample-and-match
    std::vector<common_spec_dist> spec_dists;
    std::mt19937 spec_rng{std::random_device{}()};
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)
    // --stream-interval / stream_interval > 1: generated tokens whose text is not sent yet, and the text size up to which a chunk may be
    // sent (what interval 1 would already have sent: not past a held-back partial stop string or a split UTF-8 character)
    int32_t n_unsent_tokens = 0;
    size_t  n_sendable_text = 0;

    std::vector<completion_token_output> generated_token_probs;

    // output tokens of this request, kept only when repetition_detection is on (the detector reads the tail)
    std::vector<llama_token> rep_tokens;
    server_repetition::token_loop_guard<llama_token> loop_guard; // repetition_detection.token_guard state (accepted tokens only)
    bool loop_guard_eog = false;

    // generated tokens inside the thinking block, for usage.*_tokens_details.reasoning_tokens (server-reasoning-count.h)
    server_reasoning_counter reasoning_count;

    // reasoning_loop_recovery (server-loop-recovery.h). A loop found inside the reasoning in "recover" mode sets rlr_pending in
    // process_token(); pre_decode() then rewrites this slot's KV with the spliced prompt before the next batch is built.
    bool    rlr_pending      = false;
    int32_t rlr_next_check   = 0;   // stats.n_gen at which the coverage detector runs next
    int32_t rlr_prompt_delta = 0;   // prompt length after the splice minus task->n_tokens()
    llama_tokens rlr_tokens;        // pending: the whole decoded sequence after the splice
    size_t       rlr_pos = 0;       // pending: first index that differs from prompt.tokens
    bool         rlr_backend_smpl = false; // this request samples on the backend chain

    // prompt_logprobs: entry k = log-probability of prompt token k given the tokens before it (NaN for k = 0, and
    // until the row that predicts it has been decoded); prompt_greedy[k] = 1 when token k was the model's top choice
    std::vector<float>   prompt_lp;
    std::vector<uint8_t> prompt_greedy;
    std::vector<int32_t> prompt_arg;     // echo: the top token id at the row that predicts position k (-1 until seen)
    std::vector<float>   prompt_arg_lp;  // echo: that top token's log-probability

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    // initialised here too (reset() sets it per task): ik 8a862fdcd hardening, SOURCE-READ examples/server/server-context.h:91, MIT
    stop_type stop = STOP_TYPE_NONE;

    std::string stopping_word;
    std::string grammar_error; // set when the grammar sampler failed closed (STOP_TYPE_GRAMMAR_ERROR)

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        // MTP boundary hidden row rides with the entry, so a cache hit does not resume drafting from a stale / zero row
        if (spec) {
            common_speculative_get_state(spec, id, cur->data.spec);
        }

        return true;
    }

    // true when the slot holds no prompt, or a cache entry (same LoRA configuration) already contains all of it
    bool prompt_is_cached(const server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return true;
        }
        for (const auto & st : prompt_cache.states) {
            if (st.prompt.lora_fp == prompt.lora_fp &&
                    st.prompt.tokens.get_common_prefix(prompt.tokens) == (int) prompt.tokens.size()) {
                return true;
            }
        }
        return false;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens, const thinking_tokens & think_tokens, uint64_t lora_fp) {
        std::vector<uint8_t> spec_state;
        bool hit = false;
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id, think_tokens, lora_fp, &spec_state, &hit);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        } else if (hit && spec) {
            common_speculative_set_state(spec, id, spec_state);   // empty (old entry) resets the carry state
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    // --lora-pool-dir: name of the pooled adapter the running task holds a pin on ("" = none). --lora-hetero-batch: this
    // slot may share a decode batch with slots running other adapters (the adapters then travel per sequence).
    std::string lora_pool_pin;
    bool        lora_hetero = false;
    int32_t alora_invocation_start = -1;

    // Branch-point (LCP) checkpoint. -1 = none. Otherwise the number of prompt tokens this task SHARES with the slot's cached
    // prompt when the memory could not be rolled back to that length (recurrent / hybrid / SWA state with no checkpoint at or
    // after it): prefill breaks there and takes a checkpoint, so the NEXT request that diverges at the same place restores it
    // instead of re-evaluating the shared prefix. Behaviour spec from Ollama mlxrunner/prefix_cache.go:131-135 (MIT, read, not
    // copied). Set in the task-start block of update_slots(), consumed in the prompt-fill loop.
    int32_t lcp_snapshot_at = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // Classifier-Free Guidance, SAFE SUBSET only (row
    // S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01; the excluded combinations --
    // speculative decoding, banned strings, prompt-checkpointing, context-shift -- are refused in
    // launch_slot_with_task, never silently desynced). id_cfg is a seq id claimed from
    // server_context_impl::cfg_seq_free for the DURATION of one task; -1 means no guidance active
    // (the default, and the only state for any slot on a server with --cfg-guidance-slots 0 --
    // zero extra fields read on that path beyond this one already-existing member).
    llama_seq_id id_cfg      = -1;
    int32_t      n_cfg_past  = 0;  // tokens already decoded into id_cfg's own position sequence
    int32_t      i_batch_cfg = -1; // this decode's batch index of id_cfg's token, for idx_cfg

    // --radix-cache: true once this task's prompt has been published into server_context_impl's
    // radix tree (server_radix_cache::insert()), so the post-decode publish loop
    // (server-context.cpp, alongside the prompt_save() loop) does not re-insert it every tick a
    // slot lingers at SLOT_STATE_DONE_PROMPT. Reset in reset() below, same lifetime as id_cfg.
    bool radix_inserted = false;
    // --radix-cache: token count the prompt-time insert published, so the finished-slot publish (server_radix_gen_publish_len())
    // only re-inserts when generation grew the path. Reset with radix_inserted.
    size_t radix_pub_len = 0;
    // --radix-cache: set by release() (survives its reset()) when this slot ended a request that generated tokens; consumed by the
    // post-decode publish loop, which inserts prompt + generation into the tree. Same shape as pin_pending.
    bool radix_gen_pending = false;
    size_t radix_gen_base  = 0; // radix_pub_len as it stood at release (reset() zeroes radix_pub_len)
    // --radix-cache: this request hit the slot context shift (positions renumbered), so its generation must not be published. Reset in reset().
    bool radix_ctx_shifted = false;
    // --warm-prefix-file: set by release() for a task that carried the pin token; consumed by the post-decode loop in update_slots()
    bool pin_pending = false;
    // The FULL tokenized negative prompt, held from claim time (launch_slot_with_task) until the
    // main prompt's completion round, where it is added into the SAME shared batch as the main
    // prompt's own last token (the "entire prompt has been processed" branch, server-context.cpp)
    // -- never via a separate/out-of-band llama_decode call. MEASURED TWICE why this matters:
    // (1) an earlier one-shot dedicated-decode design left the guidance context's logits
    // overwritten by the main prompt's own later decode before the first sampling call ever ran,
    // tripping the sampling-site invariant check on the first real test
    // ("guidance token was not in the same sub-batch as its main token"); (2) fixing that with a
    // SEPARATE decode call for just the deferred last token still crashed the server
    // (0xC0000005) under concurrent load, because src/llama-context.cpp:2843's output_reserve()
    // resets the shared output/logits bookkeeping on EVERY llama_decode call, and an out-of-band
    // decode for one task clobbers another task's still-unconsumed decode results from the same
    // processing round. The only safe design is to never call llama_decode for this outside the
    // normal update_slots() cycle at all -- hence the whole prompt is held here and folded into
    // the shared batch in one place, once.
    llama_tokens cfg_prompt_tokens;

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    // for realtime transcription, these are the rows from the last audio stream step (size = n_rows * n_embd)
    std::vector<float> inp_embd;

    // realtime transcription, see SERVER_TASK_TYPE_TRANSCRIBE_RT
    mtmd::audio_stream_ptr audio_stream;
    bool audio_wait = false; // the stream needs more audio before it can produce rows
    std::deque<int64_t> audio_commit_rows; // end row of each non-final commit not yet completed

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    // banned strings (ik_llama.cpp, MIT): while the task bans strings, sampled tokens are held in
    // ban_buffer before they are sent. When the held text matches, the slot rewinds to where the
    // match starts, bans the token there at that position, and resamples. ban_smpl[i] is the
    // sampler as it was before ban_buffer[i] was sampled, so a rewind restores grammar, DRY,
    // penalty history, rng and reasoning-budget state exactly; ik's original rolled back only
    // adaptive-p and left the discarded tokens in every other sampler history.
    int32_t ban_n_hold  = 0; // tokens held back; 0 = banning inactive for this task
    int32_t ban_rewinds = 0; // consecutive rewinds since a token was last sent
    std::vector<completion_token_output>     ban_buffer;
    std::vector<common_sampler_ptr>          ban_smpl;
    std::vector<std::regex>                  ban_re;
    std::map<int32_t, std::set<llama_token>> ban_at; // prompt.tokens index -> tokens banned there

    // Models whose memory cannot drop part of a sequence (recurrent / hybrid, SEQ_RM_TYPE_FULL) rewind
    // through a partial-state checkpoint instead, as speculative decoding does. ban_ckpt[i] is the
    // memory holding prompt[0, pos(ban_buffer[i]) - 1) -- the state from which ban_buffer[i]'s
    // predecessor is decoded again -- taken just before that decode (ban_ckpt_take). One partial-state
    // copy per generated token while bans are active; nothing when they are not.
    // HARNESS-BUILD-TASK: S69-FORK-BANNED-STRINGS-ON-FULL-SEQ-RM-MODELS-VIA-CHECKPOINTS-2026-09-26-01
    bool ban_ckpt_on = false;
    bool ban_ckpt_pending_ok = false;
    common_prompt_checkpoint              ban_ckpt_pending;
    std::vector<common_prompt_checkpoint> ban_ckpt;

    // n_tokens: how many prompt tokens the memory holds right now (the one about to be decoded excluded)
    void ban_ckpt_take(int64_t n_tokens) {
        llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
        ban_ckpt_pending.update_pos(n_tokens,
                                    llama_memory_seq_pos_min(mem_tgt, id),
                                    llama_memory_seq_pos_max(mem_tgt, id));
        // an empty memory (a one-token prompt) has nothing to restore; that rewind is let through
        ban_ckpt_pending_ok = ban_ckpt_pending.pos_max >= 0;
        if (ban_ckpt_pending_ok) {
            ban_ckpt_pending.update_tgt(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
    }

    // allowlist sampler (see server_context::allow_build_base): this request's per-stage bias vectors (the
    // server-wide base plus the request's always-allowed pieces) and the stage in force. The sampler reads the
    // current stage through common_sampler::server_biases (common/sampling.cpp:815); empty = allowlist off.
    std::vector<std::vector<float>> allow_biases;
    size_t allow_idx        = 0;
    size_t allow_gen_prev   = 0; // generated_text size before the latest token, where the keyword search starts

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()
    // Classifier-Free Guidance: returns id_cfg's seq id to server_context_impl::cfg_seq_free and
    // seq_rm's its KV. Only invoked when id_cfg >= 0 (the default -1 never calls it).
    std::function<void(llama_seq_id /* id_cfg */)> callback_on_cfg_release;

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        radix_inserted    = false;
        radix_pub_len     = 0;
        radix_ctx_shifted = false;
        spec_is_replay = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        grammar_error.clear();
        n_sent_text    = 0;
        n_unsent_tokens = 0;
        n_sendable_text = 0;
        rep_tokens.clear();
        loop_guard.reset();
        loop_guard_eog = false;
        ff_before.clear();
        rlr_pending      = false;
        rlr_next_check   = 0;
        rlr_prompt_delta = 0;
        rlr_tokens.clear();
        rlr_pos          = 0;
        rlr_backend_smpl = false;

        if (can_speculate()) {
            spec_draft.clear();
            spec_dists.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
            if (spec_ckpt_dev) {
                llama_spec_ckpt_discard(ctx_tgt, server_spec_ckpt);
                spec_ckpt_dev = false;
            }
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        prompt_lp.clear();
        prompt_greedy.clear();
        prompt_arg.clear();
        prompt_arg_lp.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        allow_biases.clear();
        allow_idx      = 0;
        allow_gen_prev = 0;

        ban_n_hold  = 0;
        ban_rewinds = 0;
        ban_buffer.clear();
        ban_smpl.clear();
        ban_re.clear();
        ban_at.clear();
        ban_ckpt_on         = false;
        ban_ckpt_pending_ok = false;
        ban_ckpt_pending.clear();
        ban_ckpt.clear();

        // the sampler's text view points into generated_text; do not leave it pointing at a released slot's string
        if (smpl) {
            smpl->generated_text = std::string_view();
            smpl->playing_text.clear();
            smpl->n_rewind = 0;
            smpl->rewinded_text.clear();
        }

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        lcp_snapshot_at = -1;

        // clear multimodal state
        mbatch.reset();

        tts.reset();

        audio_stream.reset();
        audio_wait = false;
        audio_commit_rows.clear();
        inp_embd.clear();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                // ADAPTED: this fork's common_sampler_accept takes the llama_context
                // (sampling.h:422). ctx_tgt is the enclosing slot's own context -- the same one
                // used by llama_set_sampler a few lines above in reset().
                common_sampler_accept(smpl.get(), ctx_tgt, id, false);
                n_text++;
            }
        }

        // The prompt is in. From the next accept on, the sampler tracks the GENERATED text: it appends each accepted
        // piece to playing_text and advances the expiring-logit-bias schedule (common_sampler_accept, is_decoding).
        // Nothing set this flag in tools/server, so both stayed inert. ik_llama.cpp sets it at the same point (the end
        // of prompt ingestion), SOURCE-READ: ik_llama.cpp examples/server/server-context.cpp @32cddbfce:4380, MIT.
        smpl->is_decoding = true;

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // a decision over a non-causal context reads the whole prompt in one batch
        if (task->type == SERVER_TASK_TYPE_DECISION && !llama_get_causal_attn(ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (task->type == SERVER_TASK_TYPE_DECISION) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        // a joint decision head reads the whole batch
        if (!task->decision.order.empty() || !other_slot.task->decision.order.empty()) {
            return false;
        }

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && (lora_hetero || are_lora_equal(lora, other_slot.lora));
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match.
    // Stateful samplers (mirostat v1/v2, adaptive-p) update their state from the token THEY sample inside the chain; the
    // rejection verifier emits a token drawn from max(0, p - q) instead, so their state would follow a token that was never
    // emitted (review finding m1 on upstream #27694). Those requests verify by sample-and-match, which keeps the chain's own
    // sample. Row S69-ENGREV-UPSTREAM-MERGE-SPECULATIVE-PORT-LIST-2026-11-04-01, plan gap G4.
    bool spec_sampler_is_stateful() const {
        return task && (task->params.sampling.mirostat != 0 || task->params.sampling.adaptive_target >= 0.0f);
    }

    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f && !spec_sampler_is_stateful();
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    // min_tokens (request field): the end-of-generation tokens and stop_token_ids are masked while fewer than
    // min_tokens tokens have been generated. With a grammar, the mask is applied too; if the grammar then leaves no
    // candidate (it can only end, or only produce a masked token) the sampler puts the masked logits back
    // (common_sampler::restore_if_empty, like vLLM's min_tokens processor, builtin.py:254-277).
    bool min_tokens_pending() const {
        return task && task->params.min_tokens > (int32_t) stats.n_gen;
    }

    bool is_stop_token_id(llama_token tok) const {
        if (!task) {
            return false;
        }
        const auto & ids = task->params.stop_token_ids;
        return std::find(ids.begin(), ids.end(), tok) != ids.end();
    }

    // repetition_detection: true when the tail of the output is one pattern (rep_min_pattern_size..rep_max_pattern_size tokens)
    // repeated rep_min_count times back to back. For each length L, the last L tokens must equal the L tokens before them, and so
    // on for rep_min_count copies; the shortest matching L wins. Same test as vLLM v1/core/sched/utils.py:10-62 (SOURCE-READ).
    bool repetition_detected() const {
        if (!task) {
            return false;
        }
        const auto & p = task->params;
        if (p.rep_span_min_count > 0 &&
                server_repetition::noncontiguous_span_repeated(rep_tokens, p.rep_span_min_len, p.rep_span_min_count)) {
            return true; // the same span reappears later, not necessarily back to back
        }
        // tool-call loop: a span can only have just completed when the newest token is the end marker, so the [prompt, output]
        // copy below is built only then. The prompt part is the first task->n_tokens() entries of prompt.tokens (generated tokens
        // are appended to prompt.tokens one decode late, so they are taken from rep_tokens instead).
        if (p.rep_tool_min_count > 0 && !rep_tokens.empty() && rep_tokens.back() == (llama_token) p.rep_tool_end) {
            llama_tokens all = prompt.tokens.get_text_tokens();
            // after a reasoning_loop_recovery splice the prompt part is rlr_prompt_delta tokens longer (or shorter)
            const size_t n_hist = std::min<size_t>((size_t) std::max<int64_t>(0, (int64_t) task->n_tokens() + rlr_prompt_delta), all.size());
            all.resize(n_hist);
            all.insert(all.end(), rep_tokens.begin(), rep_tokens.end());
            if (server_toolloop::toolcall_loop_detected(all, n_hist, (llama_token) p.rep_tool_begin, (llama_token) p.rep_tool_end,
                                                         p.rep_tool_min_count)) {
                return true;
            }
        }
        // token loop guard (periodic tail / n-gram dominance / low entropy), every check_interval accepted tokens
        if (p.rep_token_guard && loop_guard.should_check(reasoning_count.open, loop_guard_eog)) {
            const auto r = loop_guard.check(reasoning_count.open);
            if (r.triggered) {
                SLT_INF(*this, "token loop guard: %s period %d coverage %d score %.3f\n", r.kind.c_str(), (int) r.period, (int) r.coverage, (double) r.score);
                return true;
            }
        }
        if (p.rep_max_pattern_size <= 0) {
            return false;
        }
        return server_repetition::contiguous_pattern_repeated(rep_tokens, p.rep_min_pattern_size, p.rep_max_pattern_size,
                                                              p.rep_min_count, p.rep_min_dup_tokens);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // accepted drafts would bypass the min_tokens mask, which is applied to the logits of the one sampled token
        if (min_tokens_pending()) {
            return 0;
        }

        // banned strings rewind token by token through ban_buffer; accepted drafts would bypass it
        if (ban_n_hold > 0) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        // In-Place TTT target: no draft token may complete a fast-weight chunk, since a rejected draft's update cannot
        // be undone; the sampled token goes at prompt.n_tokens() (row S69-FORK-INPLACE-TTT-2026-10-26-01)
        if (ctx_tgt != nullptr) {
            n_draft_max = std::min(n_draft_max, (int) llama_ttt_draft_cap(llama_get_model(ctx_tgt), prompt.n_tokens()));
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        // the audio stream has no token before the first step, track the position with a dummy token
        const llama_token tok_track = sampled == LLAMA_TOKEN_NULL ? 0 : sampled;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (ban_ckpt_on) {
                ban_ckpt_take(prompt.n_tokens()); // the memory before `sampled` is decoded
            }

            if (!inp_embd.empty()) {
                const int32_t n_rows = (int32_t) inp_embd.size() / batch.n_embd;
                GGML_ASSERT((int32_t) inp_embd.size() == n_rows * batch.n_embd);
                i_batch = batch.size() + n_rows - 1;
                for (int32_t i = 0; i < n_rows; i++) {
                    // text embeddings broadcast the same position across the M-RoPE sections
                    const llama_pos p = prompt.tokens.pos_next();
                    const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                    add_ok &= batch.add_embd(id, inp_embd.data() + (size_t) i * batch.n_embd, pos, i == n_rows - 1, false);
                    if (i < n_rows - 1) {
                        prompt.tokens.push_back(tok_track);
                    }
                }
                if (audio_stream) {
                    inp_embd.clear(); // rows are consumed, the next step produces new ones
                }
            } else if (!ff_before.empty() && id_cfg < 0) {
                // grammar jump-forward: the forced tokens go in first without logits, `sampled` (the last forced one) carries them
                auto pos0 = prompt.tokens.pos_next();
                for (const llama_token t : ff_before) {
                    add_ok &= batch.add(id, t, pos0++, false, false);
                }
                i_batch = batch.size();
                add_ok &= batch.add(id, sampled, pos0++, true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01):
            // replay the just-sampled MAIN token into the guidance seq too, in the SAME decode
            // step -- this is what lets the guidance context's own next-token prediction (read via
            // idx_cfg at the sampling call site) stay one step ahead, matching koboldcpp's
            // "guidance context replays the main stream" behaviour (SOURCE-READ already cited on
            // this row). inp_embd (TTS-style embedding continuation) is out of scope -- id_cfg is
            // never set for those slots in the safe subset, but the guard is explicit here too.
            if (id_cfg >= 0 && inp_embd.empty()) {
                i_batch_cfg = batch.size();
                const bool cfg_added = batch.add(id_cfg, sampled, n_cfg_past++, true, false);
                add_ok &= cfg_added;
                if (cfg_added) {
                    // Mark the MAIN entry (added just above, so it's exactly one before this
                    // guidance entry) so update_slots()'s chunking never splits this pair.
                    GGML_ASSERT(batch.tokens.size() >= 2);
                    batch.tokens[batch.tokens.size() - 2].pairs_with_next = true;
                }
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        if (!ff_before.empty() && spec_draft.empty() && inp_embd.empty() && id_cfg < 0) {
            prompt.tokens.insert(ff_before);
        }
        ff_before.clear();
        prompt.tokens.push_back(tok_track);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            pin_pending = task->params.cache_pin;

            // --radix-cache: queue the finished request's prompt + generation for publishing. Decided here, before reset() clears
            // the per-request state; the insert itself waits for the post-decode phase (reading a slot's KV inline crashed).
            radix_gen_base    = radix_pub_len;
            radix_gen_pending = stats.n_gen > 0 && !radix_ctx_shifted && id_cfg < 0 && !prompt.tokens.has_mtmd &&
                                !task->is_child() && !task->is_stateless() && !audio_stream && task->type != SERVER_TASK_TYPE_TTS;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless() || audio_stream) {
                prompt_clear();
            }

            // mtmd_helper always re-eval the whole prompt
            if (task->type == SERVER_TASK_TYPE_TTS) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            // Classifier-Free Guidance: release AFTER reset() so a leftover id_cfg (e.g. the task
            // was aborted mid-generation) is still returned to the pool -- reset() itself does not
            // touch id_cfg, only this release() path does.
            if (id_cfg >= 0) {
                callback_on_cfg_release(id_cfg);
                id_cfg      = -1;
                n_cfg_past  = 0;
                i_batch_cfg = -1;
            }

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        // stop_regex: a regex has no partial-match mode, so only the full check looks at it, and only at the tail window
        if (is_full_stop) {
            const size_t win  = task_params::STOP_REGEX_WINDOW_BYTES + last_token_size;
            const size_t from = text.size() > win ? text.size() - win : 0;
            for (const std::regex & re : task->params.stop_regex_compiled) {
                std::smatch m;
                try {
                    const auto flags = from > 0 ? std::regex_constants::match_prev_avail : std::regex_constants::match_default;
                    if (!std::regex_search(text.begin() + from, text.end(), m, re, flags) || m.length(0) == 0) {
                        continue;
                    }
                } catch (const std::regex_error &) {
                    continue; // complexity/stack limits of the standard library: treat as no match
                }
                const size_t pos = from + (size_t) m.position(0);
                if (stop_pos == std::string::npos || pos < stop_pos) {
                    stop          = STOP_TYPE_WORD;
                    stopping_word = m.str(0);
                    has_next_token = false;
                    stop_pos      = pos;
                }
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    // the other slot continues from the tokens processed so far
    void copy_prompt_to(server_slot & other) const {
        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.prompt = prompt.clone();
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int encode_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx) {
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);

    mtmd_embd_cache_set_scope(mctx, (uint64_t) slot.task->id); // --mmproj-embd-cache: the request being served
    mbatch.reset(mtmd_batch_init(mctx));
    int32_t res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return 0;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (encode_mtmd_chunk(slot, mbatch, idx) != 0) {
        return -1;
    }

    return try_decode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool add_mtmd_chunk(const server_slot & slot, const mtmd_input_chunk * chunk, const float * embd, server_batch & batch) {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const auto * image    = mtmd_input_chunk_get_tokens_image(chunk);
    const bool   is_mrope = mtmd_decode_use_mrope(slot.mctx);
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd   = batch.n_embd;
    const llama_pos pos_0 = slot.prompt.tokens.pos_next();

    for (size_t i = 0; i < n_tokens; i++) {
        const llama_pos p = pos_0 + (llama_pos) i;
        llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
        if (is_mrope && image) {
            const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = rel.t;
            pos[1] = rel.y;
            pos[2] = rel.x;
            pos[3] = rel.z;
        }
        if (!batch.add_embd(slot.id, embd + i * n_embd, pos, slot.need_embd(), /* is_prompt */ true)) {
            return false;
        }
    }
    return true;
}

//
// server_context_impl (private implementation)
//

// Ported from the pre-merge fork's examples/server/server-common.h (row
// S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G1: runtime
// control vectors). SOURCE-READ: examples/server/server-common.h:148-156 @ 35f0cac75 (local
// worktree copy, pre-merge 279b4bfaf) -- llama_control_vector_data renamed common_control_vector_data
// by the upstream merge (common/common.h:1701), otherwise unchanged.
struct control_vector_container {
    std::string path;
    float       scale;
    int32_t     layer_start;
    int32_t     layer_end;
    common_control_vector_data data;
    bool        applied;
};

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    mtmd_helper_cap mtmd_cap = {};
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    bool has_cap_tts() const {
        return mtmd_cap.audio_gen;
    }

    bool has_cap_chat() const {
        return mtmd_cap.chat;
    }

    server_decision_context decision;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    // created by the first /v1/decision request; owns no llama state, only the cached-prefix bookkeeping
    std::unique_ptr<llama_decision::engine> decision_engine;

    // timestamp_granularities[]=word: the second context + projector that replay a finished transcription for the aligner
    // (server-asr-words.h). Created by the first request that asks for words; never touched by requests that do not.
    std::unique_ptr<asrwords::engine> asr_words_engine;
    std::mutex                        asr_words_mu;

    // budget groups (server-budget-group.h): the shared token / wall-time pools of recursion trees, keyed by task.params.budget_group
    server_budget_groups              budget_groups;

    // --asr-vad: the Silero VAD weights, loaded once by the first load_model() (kept across sleep/resume: they are plain
    // host floats, no llama state) and read-only afterwards, so concurrent transcription requests share it without a lock.
    // nullptr = gate off.
    std::shared_ptr<const asr_silero_model> asr_vad;

    server_state_callback_t callback_state = [](server_state, json) -> void {};
    std::function<void(int)> callback_model_loaded; // see server_context::set_model_loaded_callback

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    // --forward-watchdog-*: a poller thread over server_fwd_watchdog (server-watchdog.h). SOFT logs once per stall; HARD waits the
    // grace period, re-checks that no step finished, then ends the process so a supervisor restarts it. Started on the first decode.
    server_fwd_watchdog fwd_wd;
    std::thread         fwd_wd_thread;
    std::atomic<bool>   fwd_wd_stop{false};
    bool                fwd_wd_started = false;

    static int64_t fwd_wd_now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void fwd_wd_start_once() {
        if (fwd_wd_started) {
            return;
        }
        fwd_wd_started = true;
        fwd_wd.configure((int64_t) params_base.forward_watchdog_soft_s * 1000, (int64_t) params_base.forward_watchdog_hard_s * 1000);
        if (!fwd_wd.enabled()) {
            return;
        }
        server_fwd_watchdog_health_global().wd.store(&fwd_wd, std::memory_order_release);  // GET /health reads stalled() from here
        fwd_wd_thread = std::thread([this]() {
            auto nap = [this](int64_t ms) {
                for (int64_t t = 0; t < ms && !fwd_wd_stop.load(); t += 100) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            };
            while (!fwd_wd_stop.load()) {
                nap(1000);
                const int64_t now = fwd_wd_now_ms();
                const auto v = fwd_wd.poll(now);
                if (v == server_fwd_watchdog::verdict::soft_fire) {
                    SRV_WRN("forward watchdog: no decode step finished for %lld ms (step count %llu); a batch is in flight\n",
                            (long long) fwd_wd.stall_ms(now), (unsigned long long) fwd_wd.forward_ct());
                } else if (v == server_fwd_watchdog::verdict::hard_fire) {
                    const uint64_t ct = fwd_wd.forward_ct();
                    SRV_ERR("forward watchdog: no decode progress for %lld ms; ending the process in %lld ms unless a step finishes\n",
                            (long long) fwd_wd.stall_ms(now), (long long) fwd_wd.grace_ms());
                    nap(fwd_wd.grace_ms());
                    if (!fwd_wd_stop.load() && fwd_wd.forward_ct() == ct && fwd_wd.stalled()) {
                        SRV_ERR("%s", "forward watchdog: still stalled after the grace period, exiting\n");
                        std::_Exit(134);
                    }
                }
            }
        });
    }

    // --kv-audit-interval: per-slot canary chains, kept beside the slots by index
    std::vector<kv_audit_canary> kv_audit_chains;
    uint64_t                     kv_audit_step = 0;

    ~server_context_impl() {
        // withdraw the /health view before the watchdog it points at goes away
        const server_fwd_watchdog * expected = &fwd_wd;
        server_fwd_watchdog_health_global().wd.compare_exchange_strong(expected, nullptr);
        fwd_wd_stop.store(true);
        if (fwd_wd_thread.joinable()) {
            fwd_wd_thread.join();
        }
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    // Per-device memory rows for GET /memory and the memory_*_bytes gauges. ik_llama.cpp 55813a5d7
    // examples/server/server-context.cpp get_memory_data(), adapted: this fork has ONE draft context (ctx_dft: draft model or
    // MTP) plus the speculative companion context, not one per slot. llama_memory_breakdown_rows counts a shared model once.
    json get_memory_data() const {
        std::vector<llama_memory_buft> mmproj;
        if (mctx) {
            mmproj.resize(std::max(0, mtmd_get_memory_usage(mctx, nullptr, 0)));
            mtmd_get_memory_usage(mctx, mmproj.data(), (int32_t) mmproj.size());
        }

        std::vector<const llama_context *> ctxs = { ctx_tgt };
        if (ctx_dft) {
            ctxs.push_back(ctx_dft);
        }
        if (spec) {
            if (const llama_context * ctx_comp = common_speculative_get_companion_ctx(spec.get())) {
                ctxs.push_back(ctx_comp);
            }
        }

        std::vector<llama_memory_row> rows(std::max(0, llama_memory_breakdown_rows(ctxs.data(), (int32_t) ctxs.size(), mmproj.data(), (int32_t) mmproj.size(), nullptr, 0)));
        llama_memory_breakdown_rows(ctxs.data(), (int32_t) ctxs.size(), mmproj.data(), (int32_t) mmproj.size(), rows.data(), (int32_t) rows.size());

        json memory_data = json::array();
        for (const llama_memory_row & r : rows) {
            json entry = {
                { "name",    r.name    },
                { "model",   r.model   },
                { "context", r.context },
                { "compute", r.compute },
            };
            if (mctx) {
                entry["mmproj"] = r.mmproj;
            }
            if (r.is_device) {
                entry["total"] = r.total;
                entry["free"]  = r.free;
            }
            memory_data.push_back(entry);
        }
        return memory_data;
    }

    server_slot_gauges get_slot_gauges() {
        server_slot_gauges g;
        for (server_slot & slot : slots) {
            g.n_kv_tokens_capacity += (uint64_t) std::max(slot.n_ctx, 0);
            g.n_kv_tokens_held     += (uint64_t) std::max(slot.prompt.n_tokens(), 0);
            if (slot.is_processing()) {
                g.n_processing_slots++;
                g.n_kv_tokens_active += (uint64_t) std::max(slot.prompt.n_tokens(), 0);
            }
        }
        return g;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    // The watermark settings, BY VALUE, for the /watermark/detect route. params_base is private and is only
    // assigned at model load (load_model), so a copy of these few fields is safe to take from a route thread;
    // exposing only them keeps the rest of params_base (and write access to the key) out of the routes.
    struct watermark_settings {
        bool     key_set   = false;
        uint64_t key       = 0;
        int32_t  width     = 4;
        int32_t  scheme    = 0;
        float    kgw_gamma = 0.5f;
        int32_t  depth     = 30;
    };
    watermark_settings get_watermark_settings() const {
        return { params_base.sampling.watermark_key_set, params_base.sampling.watermark_key,
                 params_base.sampling.watermark_context_width, params_base.sampling.watermark_scheme,
                 params_base.sampling.watermark_kgw_gamma, params_base.sampling.watermark_synthid_depth };
    }

    // read-only views for the request handlers (the fields themselves stay private)
    bool has_speculative() const { return spec != nullptr; }
    bool is_pin_token(const std::string & token) const { return !pin_token.empty() && token == pin_token; }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // --spec-async: the drafts that go into this update's verify batch, handed to the draft worker right before the target decode
    // (decode(), first chunk). Filled while the drafts are collected, consumed (and cleared) there.
    std::vector<common_speculative_async_req> spec_async_reqs;

    // On-device speculative checkpoint of the target's recurrent state (llama_spec_ckpt_*: per-step or gpu-fallback),
    // or LLAMA_SPEC_CKPT_NONE for the host copy (common_prompt_checkpoint::update_tgt / load_tgt). The host copy moved
    // the whole recurrent state device -> host -> device every verify round: MEASURED 2026-10-18 (GPU-1, nsys,
    // Qwen3.5-0.8B mtp n_max=3) ~33 MB each way per round, 2.70 s of cudaMemcpyAsync API time in a 10.72 s decode.
    // The library checkpoint copies whole state tensors and one cache-cell table, so it is used only when one sequence
    // owns every batch: one slot and no guidance slots. Row S69-HYPERQWEN-CHALLENGE-SPEC-CKPT-HOST-COPY-2026-10-18-01.
    int spec_ckpt_dev_mode   = LLAMA_SPEC_CKPT_NONE;
    int spec_ckpt_dev_tokens = 0;

    common_speculative_ptr spec;

    // --warm-prefix-file state (see server_context::set_pin_token)
    std::string pin_token;
    std::atomic<size_t> pinned_bytes_a{0};
    std::atomic<size_t> pinned_entries_a{0};
    std::atomic<bool>   pin_overflow_a{false};
    std::atomic<size_t> pin_passes_a{0};
    std::atomic<bool>   last_pin_covered_a{false};

    // every token's text with special tokens NOT rendered (a control token is ""), built on first use by token healing
    std::vector<std::string> heal_pieces;
    const std::vector<std::string> & token_pieces() {
        if (heal_pieces.empty()) {
            const int32_t n = llama_vocab_n_tokens(vocab);
            heal_pieces.reserve(n);
            for (int32_t i = 0; i < n; ++i) {
                heal_pieces.push_back(common_token_to_piece(vocab, i, false));
            }
        }
        return heal_pieces;
    }
    server_jump_forward::vocab_index jf_index; // built on first use by grammar jump-forward
    server_draft_schedule  draft_schedule; // --draft-max-by-slots: draft budget by the number of generating slots (empty = no opinion)
    server_lora_fair_pick  lora_fair;      // round-robin over distinct lora configs for the decode batch leader (server-lora-fair.h)

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    // --lora-pool-dir (null = off). Destroyed before the model (destroy()).
    std::unique_ptr<server_lora_pool> lora_pool;

    // Classifier-Free Guidance, SAFE SUBSET (row
    // S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01). cfg_seq_free[i] tracks whether seq id
    // (params_base.n_parallel + i) is free; sized to params_base.n_cfg_guidance_slots at slot
    // setup, so an empty vector (the default, N=0) makes claim_cfg_seq always fail -- consistent
    // with server-schema.cpp's own N<=0 refusal, never reached in that case anyway.
    std::vector<bool> cfg_seq_free;

    bool claim_cfg_seq(server_slot & slot) {
        for (size_t i = 0; i < cfg_seq_free.size(); i++) {
            if (cfg_seq_free[i]) {
                cfg_seq_free[i] = false;
                slot.id_cfg      = params_base.n_parallel + (llama_seq_id) i;
                slot.n_cfg_past  = 0;
                slot.i_batch_cfg = -1;
                return true;
            }
        }
        return false;
    }

    void release_cfg_seq(llama_seq_id id_cfg) {
        const size_t i = (size_t) (id_cfg - params_base.n_parallel);
        GGML_ASSERT(i < cfg_seq_free.size());
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_cfg, -1, -1);
        cfg_seq_free[i] = true;
    }

    // Ported control vectors (group G1). See control_vector_container above.
    std::vector<control_vector_container> control_vectors;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;
    // DISABLE-REVERSIBLE: null is --radix-cache's default-off / gated-off state (server_radix_cache
    // is simply never constructed); re-enable condition is --radix-cache plus kv_unified plus a
    // supported model/memory type, checked once at init() below.
    std::unique_ptr<server_radix_cache>  radix_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    // engine steps not yet closed by a synchronised one (server-latency.h): classified and timed as one group
    server_step_group step_group;

    // --prefill-tbt-target-ms: running estimates, in us. 0 = not measured yet.
    double ema_decode_step_us = 0.0; // one decode-only engine step with the current streams
    double ema_prefill_us_tok = 0.0; // prompt cost per token incl. the attention over the context it already has

    // the prompt tokens all slots together may add to this step when it also carries decode tokens; INT32_MAX = no limit
    int32_t prefill_step_cap(int32_t n_batch, int32_t n_ubatch) const {
        int32_t cap = INT32_MAX;
        if (params_base.prefill_step_budget > 0) {
            cap = params_base.prefill_step_budget;
        }
        if (params_base.prefill_tbt_target_ms > 0) {
            if (ema_prefill_us_tok > 0.0) {
                const double room_us = params_base.prefill_tbt_target_ms * 1000.0 - ema_decode_step_us;
                int32_t n = room_us > 0.0 ? (int32_t) (room_us / ema_prefill_us_tok) : 0;
                n = (n / 32) * 32;
                cap = std::max<int32_t>(32, std::min<int32_t>(n, n_batch));
            } else if (cap == INT32_MAX) {
                cap = std::max<int32_t>(32, n_ubatch); // nothing measured yet: one micro-batch
            }
        }
        return cap;
    }

    // feed the estimates from a closed step group (called only at a synchronised step, where the time is real)
    void note_step_group(const server_step_group & g, int64_t t_end_us) {
        if (params_base.prefill_tbt_target_ms <= 0 || g.empty() || t_end_us <= g.t_start_us) {
            return;
        }
        const double dur = (double) (t_end_us - g.t_start_us);
        const double a = 0.25;
        if (g.prefill_tokens == 0 && g.decode_tokens > 0) {
            const double x = dur / (double) g.steps;
            ema_decode_step_us = ema_decode_step_us > 0.0 ? ema_decode_step_us + a * (x - ema_decode_step_us) : x;
        } else if (g.prefill_tokens >= 32) {
            const double dec_part = g.decode_tokens > 0 ? ema_decode_step_us * (double) g.steps : 0.0;
            const double x = std::max(dur - dec_part, dur * 0.1) / (double) g.prefill_tokens;
            ema_prefill_us_tok = ema_prefill_us_tok > 0.0 ? ema_prefill_us_tok + a * (x - ema_prefill_us_tok) : x;
        }
    }

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    // --slot-evict-lookahead: how many waiting requests the LRU branch of get_available_slot looks at before it evicts
    // an idle slot (server-slot-evict.h); 0 = plain LRU
    int32_t slot_evict_lookahead = 0;

    // fork prompt-cache tuning, restored 2026-09-27 (pre-merge examples/server/server-context.cpp @ 35f0cac75:1088-1098;
    // flags -crs/--cache-ram-similarity, -cram-n-min/--cache-ram-n-min dropped by merge 279b4bfaf).
    float   cache_ram_similarity = 0.5f; // save the slot's prompt to the RAM cache when f_keep falls below this
    int32_t cache_ram_n_min      = 0;    // never SAVE a prompt shorter than this many tokens (loads are unaffected)

    // allowlist sampler (fork; pre-merge examples/server/server-context.cpp @ 35f0cac75:1748-1845, merge 279b4bfaf
    // dropped it). One additive logit-bias vector per rule STAGE, built once: the rule sets and keywords are
    // server-wide flags, so only the always-allowed pieces vary per request (applied on a per-slot copy).
    // A token keeps the highest bias of a rule that covers it only if EVERY codepoint is covered; script
    // common/inherited codepoints (whitespace, punctuation, combining marks) defer to the others; otherwise -INF.
    std::vector<std::vector<float>> allow_base_biases; // [stage][token]
    std::vector<float>              allow_stage_max;   // [stage] highest rule bias, given to allowed pieces
    bool                            allow_base_built = false;

    void allow_build_base() {
        if (allow_base_built) {
            return;
        }
        allow_base_built = true;

        auto rules_all = params_base.allow_rules;
        auto kws       = params_base.allow_kws;
        if (rules_all.empty()) {
            return;
        }
        // one more rule set than keywords: the last set never expires; fewer: the extra keywords do nothing
        if (rules_all.size() > kws.size() + 1) {
            rules_all.resize(kws.size() + 1);
        }

        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<std::string> pieces(n_vocab);
        for (int32_t id = 0; id < n_vocab; ++id) {
            pieces[id] = common_token_to_piece(vocab, id, true);
        }

        allow_base_biases.assign(rules_all.size(), {});
        allow_stage_max.assign(rules_all.size(), -INFINITY);
        std::vector<uint32_t>    cpts;
        std::vector<std::string> scripts;
        for (size_t s = 0; s < rules_all.size(); ++s) {
            const auto & rules = rules_all[s];
            for (const auto & rule : rules) {
                allow_stage_max[s] = std::max(allow_stage_max[s], std::get<3>(rule));
            }
            auto & biases = allow_base_biases[s];
            biases.assign(n_vocab, -INFINITY);
            for (int32_t id = 0; id < n_vocab; ++id) {
                const size_t n_cpt = llama_fill_from_utf8(&pieces[id], &cpts, &scripts);
                float bias = -INFINITY;
                for (size_t j = 0; j < n_cpt; ++j) {
                    bool in_rule = false;
                    for (const auto & rule : rules) {   // earlier rule has priority
                        in_rule = std::get<0>(rule) <= cpts[j] && cpts[j] <= std::get<1>(rule) &&
                                  (std::get<2>(rule) == "*" || std::get<2>(rule) == scripts[j]);
                        if (in_rule) {
                            bias = std::max(bias, std::get<3>(rule));
                            break;
                        }
                    }
                    if (!in_rule) {
                        if (scripts[j] == "common" || scripts[j] == "inherited") {
                            continue; // defer to the other codepoints of the token
                        }
                        bias = -INFINITY;
                        break;
                    }
                }
                biases[id] = bias;
            }
        }
        SRV_INF("allowlist sampler: %zu rule set(s), %zu keyword(s), vocab %d\n",
                rules_all.size(), std::min(kws.size(), rules_all.size()), n_vocab);
    }

    // point the slot's sampler at the stage in force (null once every stage has expired). Called after each
    // stage update AND before each sample, as the old apply_server_biases was (:4305, :4834): a banned-string
    // rewind restores a cloned sampler whose pointer may name an earlier stage.
    static void allow_apply(server_slot & slot) {
        if (slot.allow_biases.empty() || !slot.smpl) {
            return;
        }
        slot.smpl->server_biases = slot.allow_idx < slot.allow_biases.size() ? &slot.allow_biases[slot.allow_idx] : nullptr;
    }

    // Sampler text feed. The sampler (common/sampling.cpp) reads the text generated so far from two fields that only the
    // server can fill: generated_text (a view of the slot's text, which holds every token already passed through
    // process_token) and playing_text (what was sampled but is not in generated_text yet: the banned-string hold-back
    // buffer). Expiring logit bias reads both to decide when a phrase's bias applies and when a stage's exit word has
    // been produced (common_expiring_logit_bias_apply/_accept). Call this before EVERY sample (the sampler's own accept
    // keeps playing_text current only inside one sample-and-accept run); it also zeroes the per-token rewind counters.
    // ik_llama.cpp does the same before each sample, SOURCE-READ: examples/server/server-context.cpp @32cddbfce:4446-4450
    // (speculative accept) and :5011-5016 (process_batch_tokens), MIT. Their token_buffer is our ban_buffer.
    static void sampler_feed_text(server_slot & slot) {
        common_sampler * s = slot.smpl.get();
        if (!s) {
            return;
        }
        s->generated_text = slot.generated.text;
        s->playing_text.clear();
        for (const auto & r : slot.ban_buffer) {
            s->playing_text.append(r.text_to_send);
        }
        s->n_rewind = 0;
        s->rewinded_text.clear();
    }

    // Build the sampler's expiring-logit-bias schedule (one elb_state per stage) from the request's / --expiring-logit-bias-file's
    // parsed rules (common_params_sampling::elb_params, filled by argparse_expiring_logit_bias). Rules whose bias list is empty
    // are expiring SAMPLER-PARAMETER biases: they need no token table, the sampler applies them from the params directly.
    // Port of ik_llama.cpp examples/server/server-context.cpp @32cddbfce:1940-2071 (MIT, SOURCE-READ). Differences, both
    // fixes: (1) the bias list is copied per PHRASE (ik copies it per entry, so a one-token phrase shrank the list for the
    // phrases after it); (2) an empty tokenization is skipped (ik read ids[0] of an empty vector); (3) the "infinite duration"
    // arithmetic is done in unsigned 32-bit (ik's signed INT_MIN + (-j) overflows; the wrapped value is the same).
    void elb_build_states(server_slot & slot) {
        common_sampler * smpl = slot.smpl.get();
        smpl->elb_states.clear();
        smpl->elb_idx        = 0;
        smpl->elb_search_pos = 0;

        const auto & elb_params = smpl->params.elb_params;
        smpl->elb_states.reserve(elb_params.size());

        // 1 state <-> 1 exit word <-> 1+ entries
        for (int32_t i = 0; i < (int32_t) elb_params.size(); ++i) {
            const auto & entries  = elb_params[i].entries;
            const auto & exitword = elb_params[i].exitword;
            const auto & op       = elb_params[i].op;

            if (op == ">>") {
                // jump: the previous states that have no jump word yet also move on to the state after this one when
                // this state's exit word shows up
                for (auto & st : smpl->elb_states) {
                    if (st.jumpword.empty()) {
                        st.jumpword        = exitword;
                        st.jump_idx        = i + 1;
                        st.search_word_len = (int32_t) std::max(st.exitword.length(), st.jumpword.length());
                    }
                }
            }

            common_sampler::elb_state st{};
            st.exitword        = exitword;
            st.search_word_len = (int32_t) exitword.size();
            smpl->elb_states.push_back(std::move(st));

            auto & first_tokens = smpl->elb_states.back().first_tokens;
            auto & other_tokens = smpl->elb_states.back().other_tokens;
            auto & delay        = smpl->elb_states.back().delay;
            auto & max_cond_len = smpl->elb_states.back().max_cond_len;

            // 1 entry <-> 1+ phrases <-> 1+ biases
            for (const auto & entry : entries) {
                if (entry.biases.empty()) {
                    continue; // expiring sampler-parameter bias
                }

                for (const auto & phrase : entry.phrases) {
                    if (phrase.empty()) {
                        continue;
                    }
                    const auto ids = common_tokenize(vocab, phrase, false, true);
                    if (ids.empty()) {
                        continue;
                    }

                    auto biases = entry.biases;
                    if (!entry.is_range) {
                        // extrapolate the last bias over the rest of the phrase
                        biases.resize(ids.size(), biases.back());
                    } else if (ids.size() == 1) {
                        biases[0] = biases.back();
                        biases.resize(1);
                    } else {
                        // interpolate from the first bias to the last over the phrase
                        float bb = biases.back();
                        const float inc = (bb - biases.front()) / (ids.size() - 1);
                        biases.resize(ids.size());
                        for (int32_t j = (int32_t) ids.size() - 1; j >= 0; --j) {
                            biases[j] = bb;
                            bb -= inc;
                        }
                    }

                    const int32_t duration = entry.duration;
                    if (biases[0] != 0.0f) {
                        // cond is the token's own piece: the bias is not applied when the text already ends with it
                        first_tokens.push_back({ ids[0], biases[0], size_t(duration), common_token_to_piece(vocab, ids[0], true) });
                    }

                    // a negative duration is "persistent": effectively unbounded, later tokens of the phrase a little shorter
                    // so that the tables stay ordered by duration
                    const bool    persistent = duration < 0;
                    const uint32_t base      = persistent ? (uint32_t(duration) ^ 0x7FFFFFFFu) : uint32_t(duration);

                    std::string cond;
                    for (int32_t j = 1; j < (int32_t) ids.size(); ++j) {
                        // cond collects the text that must precede ids[j] for its bias to apply
                        cond += common_token_to_piece(vocab, ids[j - 1], true);
                        if (biases[j] == 0.0f) {
                            continue;
                        }
                        const size_t dur = persistent ? size_t(uint32_t(base - uint32_t(j))) : size_t(base + uint32_t(j));
                        if (biases[j] > 0.0f) {
                            delay = std::max(dur, delay);
                        }
                        other_tokens.push_back({ ids[j], biases[j], dur, cond });
                    }
                    max_cond_len = std::max((int32_t) cond.size(), max_cond_len);
                }
            }
        }

        // longest duration first (the sampler's upper_bound over the tables relies on it)
        for (auto & st : smpl->elb_states) {
            std::sort(st.first_tokens.begin(), st.first_tokens.end(), [](const auto & a, const auto & b) {
                return a.duration > b.duration;
            });
            std::sort(st.other_tokens.begin(), st.other_tokens.end(), [](const auto & a, const auto & b) {
                return a.duration > b.duration;
            });
        }
    }

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        lora_pool.reset();
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    // Re-aggregates all active control vectors and updates the model state. Ported from the
    // pre-merge fork (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01,
    // group G1). SOURCE-READ: examples/server/server-context.cpp:3264-3321 @ 35f0cac75 (local
    // worktree copy) -- `ctx`/`model` renamed `ctx_tgt`/`model_tgt` (this struct's own names),
    // `llama_control_vector_data`/`llama_control_vector_load_info` renamed
    // `common_control_vector_*` by the upstream merge; the llama_control_vector_apply() call
    // itself is byte-identical (include/llama.h:1177, kept by the fork, not replaced).
    // Fingerprint of the control vector currently applied (0 = none). Folded into task_lora_fingerprint() so the slot
    // prompt, the host prompt cache and the radix cache never serve KV computed under another steering vector.
    uint64_t cv_fp = 0;

    // set once the --sched-policy callback is registered with queue_tasks (see load_model)
    bool sched_policy_installed = false;

    bool apply_control_vectors_internal() {
        common_control_vector_data combined_cv = { -1, {} };

        // check if we have anything to apply
        bool any_active = false;
        for (const auto & cv : control_vectors) {
            if (cv.scale != 0.0f) {
                any_active = true;
                break;
            }
        }

        if (!any_active) {
            // clear control vectors if nothing is active
            llama_control_vector_apply(ctx_tgt, nullptr, 0, 0, 0, 0);
            cv_fp = 0;
            return true;
        }

        // aggregate control vectors with scaling
        for (auto & cv : control_vectors) {
            if (cv.scale == 0.0f) {
                cv.applied = false;
                continue;
            }

            if (combined_cv.n_embd == -1) {
                combined_cv.n_embd = cv.data.n_embd;
                combined_cv.data.resize(cv.data.data.size(), 0.0f);
            }

            for (size_t i = 0; i < cv.data.data.size(); i++) {
                combined_cv.data[i] += cv.data.data[i] * cv.scale;
            }
            cv.applied = true;
        }

        // apply combined control vector
        if (combined_cv.n_embd != -1 && !combined_cv.data.empty()) {
            int32_t min_layer_start = INT32_MAX;
            int32_t max_layer_end   = 0;

            for (const auto & cv : control_vectors) {
                if (cv.scale != 0.0f) {
                    min_layer_start = std::min(min_layer_start, cv.layer_start);
                    max_layer_end   = std::max(max_layer_end, cv.layer_end);
                }
            }

            int err = llama_control_vector_apply(ctx_tgt,
                                                combined_cv.data.data(),
                                                combined_cv.data.size(),
                                                combined_cv.n_embd,
                                                min_layer_start,
                                                max_layer_end);
            // set even when the apply failed: the context's state is then unknown, and a new key only costs a re-prefill
            cv_fp = control_vector_fingerprint(combined_cv.data.data(), combined_cv.data.size(),
                                               combined_cv.n_embd, min_layer_start, max_layer_end);
            return (err == 0);
        }

        return true;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        if (!params_base.draft_max_by_slots.empty()) {
            std::string err;
            if (!draft_schedule.parse(params_base.draft_max_by_slots, err)) {
                SRV_ERR("invalid --draft-max-by-slots '%s': %s\n", params_base.draft_max_by_slots.c_str(), err.c_str());
                return false;
            }
            draft_schedule.hysteresis = params_base.draft_max_by_slots_hysteresis;
            SRV_INF("draft length by generating slots: %zu range(s) from '%s', hysteresis %d\n", draft_schedule.ranges.size(),
                    params_base.draft_max_by_slots.c_str(), draft_schedule.hysteresis);
        }

        // --asr-vad: load the Silero file once, before the (slow) model load, so a wrong path fails the start at once
        if (!params_base.asr_vad_model.empty() && !asr_vad) {
            auto sm = std::make_shared<asr_silero_model>();
            std::string err;
            if (!asr_silero_load(params_base.asr_vad_model, *sm, &err)) {
                SRV_ERR("--asr-vad: cannot load Silero VAD model '%s': %s\n", params_base.asr_vad_model.c_str(), err.c_str());
                return false;
            }
            SRV_INF("--asr-vad: Silero VAD %s v%s loaded from '%s' (window %d samples at 16 kHz, threshold %.2f); "
                    "/v1/audio/transcriptions returns an empty transcript for an upload with no speech\n",
                    sm->type.c_str(), sm->version.c_str(), params_base.asr_vad_model.c_str(), (int) sm->n_window,
                    (double) params_base.asr_vad_threshold);
            asr_vad = std::move(sm);
        }

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        // ADAPTED: upstream scanned speculative.types, a flat vector this fork does not carry,
        // for COMMON_SPECULATIVE_TYPE_DRAFT_MTP, which this fork spells COMMON_SPECULATIVE_TYPE_MTP
        // (common.h:281). has_stage_type() (common.h:435) already answers exactly this question
        // over the resolved stage chain, so the search is handed to the struct that owns the data
        // rather than reimplemented here.
        const bool spec_mtp = params_base.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            // -tm / --threads-mtmd (restored 2026-09-27): common.h documents n_threads_mtmd as "-1 = use
            // n_threads_batch, then n_threads"; before this the field had no reader and mtmd always got n_threads.
            mparams.n_threads        = params_base.n_threads_mtmd > 0 ? params_base.n_threads_mtmd
                                     : params_base.cpuparams_batch.n_threads > 0 ? params_base.cpuparams_batch.n_threads
                                     : params_base.cpuparams.n_threads;
            // ADAPTED: mtmd's field is `enum llama_flash_attn_type` (tools/mtmd/mtmd.h:97);
            // this fork's common_params carries `bool flash_attn` (common.h:809). The enum's AUTO
            // state is not produced here because the bool has no value meaning it -- widening that
            // field to the tri-state enum is owed work, and fabricating AUTO at this call site
            // would silently change behaviour for every multimodal load.
            mparams.flash_attn_type  = params_base.flash_attn
                                         ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                         : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.embd_cache_bytes = (size_t) std::max(params_base.mmproj_embd_cache_mib, 0) * 1024u * 1024u;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        // on resume params_base already holds the margin from the first load
        if (has_mmproj && params_base.fit_params && !is_resume) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        try {
            decision.init(model_tgt);
        } catch (const std::exception & e) {
            SRV_ERR("failed to init decision model: %s\n", e.what());
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        if (decision.type != COMMON_DECISION_TYPE_NONE) {
            decision.set_serving(params_base.decision_budget, params_base.decision_window_agg, params_base.decision_window, decision_n_fit(), (size_t) llama_model_n_ctx_train(model_tgt));
            SRV_INF("decision budget: %s, one pass takes at most %zu tokens (n_ctx_slot = %d, n_ubatch = %d, n_ctx_train = %d), long states are read in windows (aggregation: %s)\n",
                    params_base.decision_budget.c_str(), decision_n_fit(), n_ctx_slot(), (int) llama_n_ubatch(ctx_tgt),
                    (int) llama_model_n_ctx_train(model_tgt), params_base.decision_window_agg.c_str());
        }

        // Test hook for the router's load-OOM retry (tools/server/server-models.cpp
        // retry_after_load_oom): fail this load the way an allocation failure does, when the context
        // came out larger than the given size. Lets the retry's context step-down and eviction be
        // tested deterministically without exhausting a real device. Value "N" or "N:alias" (only the
        // child serving that alias fails, so another model can load alongside it).
        if (const char * hook = std::getenv("LLAMA_SERVER_TEST_LOAD_OOM_ABOVE_CTX")) {
            const std::string h = hook;
            const size_t colon = h.find(':');
            const int lim = std::atoi(h.substr(0, colon).c_str());
            const bool for_us = colon == std::string::npos || params_base.model_alias.count(h.substr(colon + 1)) > 0;
            if (for_us && n_ctx > lim) {
                SRV_ERR("failed to allocate the KV cache: out of memory (test hook: n_ctx %d > %d)\n", n_ctx, lim);
                return false;
            }
        }

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                // REMOVED 2026-09-14, and the reason belongs in the file rather than only in
                // a commit message. Upstream stored the two contexts into
                // params.speculative.draft, a common_params_speculative_draft sub-struct that is
                // DECLARED in this fork (common.h:473) but never wired in as a `.draft` member --
                // common/speculative.cpp:1709 records that explicitly. Nothing in this tree reads
                // those fields. The contexts already reach the speculative system BY ARGUMENT,
                // through common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt)
                // three lines above, whose result supplies ctx_dft. Re-adding these assignments
                // against the unwired struct would be a no-op that LOOKS like wiring, which is
                // worse than their absence.
                // UPDATE 2026-09-26: the per-sequence DRAFT impl now reads the contexts from FLAT
                // params.speculative.ctx_tgt / ctx_dft, set just before common_speculative_init below.
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                const bool is_mixed = use_mixed_batch();
                if (is_mixed || mmproj_usage.use_non_causal) {
                    const int n_max = is_mixed ? n_ubatch / 2 : n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        SRV_WRN("cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", mmproj_usage.image_max_tokens, n_max, n_ubatch,
                                is_mixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
                        SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();
            // fork: --video-max-frames / --video-max-tokens (were server_set_video_limits globals)
            init_opt.video_params.max_frames        = params_base.video_max_frames;
            init_opt.video_params.max_vision_tokens = params_base.video_max_tokens;

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        mtmd_cap = mtmd_helper_get_capabilities(model_tgt, mctx);

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;
        slot_evict_lookahead   = std::max(0, params_base.slot_evict_lookahead);
        cache_ram_similarity   = params_base.cache_ram_similarity;
        cache_ram_n_min        = params_base.cache_ram_n_min;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01):
        // --cfg-guidance-slots widens n_seq_max, which n_ctx_slot above already reflects. Make the
        // cost VISIBLE rather than hidden -- with kv_unified=false (this server's default),
        // llama-context.cpp divides n_ctx evenly across n_seq_max, so reserving guidance seq ids
        // shrinks every normal slot's own n_ctx_slot too. Coordinator review, 2026-09-27: measured
        // via src/llama-context.cpp:516-530 (n_ctx_seq = kv_unified ? n_ctx : n_ctx/n_seq_max).
        if (params_base.n_cfg_guidance_slots > 0) {
            const int n_ctx_slot_without_cfg = params_base.n_ctx / params_base.n_parallel;
            SRV_WRN("--cfg-guidance-slots %d reserves %d extra seq id(s), raising n_seq_max from %d to %d. "
                    "n_ctx_slot = %d; WITHOUT --cfg-guidance-slots it would be ~%d (kv_unified = '%s'). "
                    "Pass --kv-unified to keep full per-slot context instead of shrinking it.\n",
                    params_base.n_cfg_guidance_slots, params_base.n_cfg_guidance_slots,
                    params_base.n_parallel, params_base.n_parallel + params_base.n_cfg_guidance_slots,
                    n_ctx_slot(), n_ctx_slot_without_cfg, params_base.kv_unified ? "true" : "false");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            // Hand the companion context to the per-sequence speculator: the draft model's context
            // for the DRAFT stage (common_speculative_impl_draft, -md), or the MTP draft context
            // (ctx_type MTP over this target, built above) for the MTP stage (common_speculative_impl_mtp).
            // Each impl drives only the context its own stage asked for: the pool builds DRAFT only
            // for a `draft` stage and MTP only for an `mtp` stage, so an MTP context is never driven
            // as a separate draft model.
            if (ctx_dft != nullptr && (params_base.speculative.has_dft() || spec_mtp)) {
                params_base.speculative.ctx_tgt = ctx_tgt;
                params_base.speculative.ctx_dft = ctx_dft;
            }
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        spec_ckpt_dev_mode   = LLAMA_SPEC_CKPT_NONE;
        spec_ckpt_dev_tokens = 0;
        // FULL as well as RS: a hybrid context with n_rs_seq = 0 (the default) resolves to FULL, and every verify then takes
        // the checkpoint. MEASURED 2026-10-18: with RS alone, Qwen3.5-0.8B MTP printed no checkpoint line in any mode and
        // never took the on-device path (fail-before and pass-after logs identical). A context without recurrent state
        // resolves the mode to NONE in llama_spec_ckpt_init and keeps the host copy.
        if (spec && (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS)) {
            const int req = params_base.speculative.spec_ckpt_mode;
            if (params_base.n_parallel != 1 || params_base.n_cfg_guidance_slots > 0) {
                SRV_INF("speculative checkpoint: host copy (n_slots = %d, guidance slots = %d; the on-device checkpoint "
                        "needs one sequence per batch)\n", params_base.n_parallel, params_base.n_cfg_guidance_slots);
            } else if (req == LLAMA_SPEC_CKPT_CPU || req == LLAMA_SPEC_CKPT_NONE) {
                SRV_INF("%s", "speculative checkpoint: host copy (--spec-ckpt-mode cpu)\n");
            } else {
                const int n_tok = std::max(1, common_speculative_n_max(&params_base.speculative) + 1);
                const int mode  = llama_spec_ckpt_init(ctx_tgt, server_spec_ckpt, req, n_tok);
                llama_spec_ckpt_discard(ctx_tgt, server_spec_ckpt);
                if (mode == LLAMA_SPEC_CKPT_PER_STEP || mode == LLAMA_SPEC_CKPT_GPU_FALLBACK) {
                    spec_ckpt_dev_mode   = mode;
                    spec_ckpt_dev_tokens = n_tok;
                    SRV_INF("speculative checkpoint: on-device (%s, up to %d tokens per verify)\n",
                            mode == LLAMA_SPEC_CKPT_PER_STEP ? "per-step" : "gpu-fallback", n_tok);
                } else {
                    SRV_INF("speculative checkpoint: host copy (the context resolved mode %d)\n", mode);
                }
            }
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            // Say so when speculation was ASKED FOR and could not be built. The per-sequence
            // constructor (common/speculative.cpp common_speculative_init(params, n_seq)) builds
            // ngram_map_k / ngram_map_k4v / ngram_simple / ngram_mod / ngram_cache / suffix / suffix_automaton, draft (with -md) and mtp (with an
            // MTP / NextN head), and names every other requested stage in its own warning; when none of
            // them could be built the draft context loaded above is freed here and every request runs
            // without speculation. Rows S69-FORK-SERVER-PER-SEQ-SPECULATOR-HAS-NO-DRAFT-IMPL-2026-09-26-01
            // (draft), S69-FORK-SERVER-PER-SEQ-SPECULATOR-NO-MTP-NO-DRAFT-POLICY-2026-10-17-01 (mtp, warnings).
            if (params_base.speculative.has_stage_chain()) {
                SRV_WRN("speculative stage chain '%s' requested, but none of its stages could be built by the server's "
                        "per-sequence speculator (see the warnings above) -- serving WITHOUT speculative decoding\n",
                        common_speculative_stage_chain_to_str(params_base.speculative).c_str());
            }
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            if (has_cap_tts()) {
                slot.tts.ctx.init(ctx_tgt, mctx);
            }

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.lora_hetero = params_base.lora_hetero_batch;

            slot.callback_on_release = [this](int id_slot) {
                lora_pool_unpin(slots[id_slot]);
                if (in_preempt_release) {
                    // a slot freed BY a KV-exhaustion preemption must not start a waiting request (for example one just requeued):
                    // the pool is short, so it would allocate again at once and be preempted again (MEASURED 2026-10-27, 4 requests
                    // against a pool that holds one: each victim's release restarted the previous victim and all 4 hit the requeue
                    // limit). Waiting requests start when a request FINISHES and frees its cells for good.
                    return;
                }
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // slots[] owns this object; the callback type is const only because it reads stats
                free_dominated_checkpoints(slots[slot.id]);
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
                metrics_on_request_done(slot);
            };

            slot.callback_on_cfg_release = [this](llama_seq_id id_cfg) {
                release_cfg_seq(id_cfg);
            };

            slot.reset();
        }

        // Classifier-Free Guidance: cfg_seq_free sized AFTER the slots loop above, so
        // params_base.n_parallel (used as the base offset for guidance seq ids) is unambiguous
        // regardless of loop order. An empty vector (N=0, the default) makes claim_cfg_seq always
        // return false, consistent with server-schema.cpp's own N<=0 refusal.
        cfg_seq_free.assign(std::max(0, params_base.n_cfg_guidance_slots), true);

        init_lora_pool();
        server_retry_n_slots().store(std::max(1, (int) slots.size())); // Retry-After for "every slot is busy" divides by this

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (decision.type != COMMON_DECISION_TYPE_NONE && !params_base.cache_ram_explicit) {
            SRV_INF("%s", "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n");
            params_base.cache_ram_mib    = 0;
            params_base.cache_idle_slots = false;
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
            prompt_cache->slru = params_base.cache_ram_slru;
            prompt_cache->min_speedup = std::max(0.0f, params_base.cache_disk_min_speedup);
            prompt_cache->pinned_share = params_base.warm_prefix_share;

            // NVMe tier under the RAM cache (row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01). It is opt-in (--cache-disk-dir) and
            // never the cause of a failed start: an unusable directory is reported and the server runs as it does without the flag.
            // DISABLE-REVERSIBLE: the tier is off unless --cache-disk-dir names a usable directory; fixing the path re-enables it.
            if (!params_base.cache_disk_dir.empty()) {
                std::string err;
                if (params_base.cache_ram_mib < 0) {
                    SRV_WRN("--cache-disk-dir %s is idle: --cache-ram is unlimited, so nothing is ever evicted to disk\n",
                            params_base.cache_disk_dir.c_str());
                }
                const size_t limit_bytes = (size_t) std::max(params_base.cache_disk_mib, 0) * 1024ull * 1024ull;
                const size_t budget      = (size_t) std::max(params_base.cache_disk_write_mib_per_hour, 0) * 1024ull * 1024ull;
                if (!prompt_cache->enable_disk(params_base.cache_disk_dir, limit_bytes,
                                               (size_t) std::max(params_base.cache_disk_min_tokens, 0), budget, err)) {
                    SRV_WRN("prompt cache disk tier is not active: directory %s is not usable (%s)\n",
                            params_base.cache_disk_dir.c_str(), err.c_str());
                } else {
                    if (params_base.cache_disk_io_chunk_mib > 0) {
                        // --cache-disk-io-chunk: overlap the tier's file I/O with its checksum (common/disk-tier.h set_io_chunk_bytes)
                        const size_t chunk = common_cache_disk_io_chunk_bytes(params_base);
                        prompt_cache->disk->set_io_chunk_bytes(chunk);
                        SRV_INF("prompt cache disk tier: pipelined I/O on, chunk %d MiB (states larger than one chunk overlap file I/O and checksum)\n",
                                params_base.cache_disk_io_chunk_mib);
                    }
                    if (params_base.cache_disk_min_free_mib >= 0) {
                        // --cache-disk-min-free: replace the built-in 10 GiB floor; the fraction cap (1/4 of the space) is kept
                        prompt_cache->disk->set_free_space_policy((int64_t) params_base.cache_disk_min_free_mib * 1024 * 1024,
                                                                  common_disk_tier::DEFAULT_FREE_SPACE_DIVISOR);
                    }
                    SRV_INF("prompt cache disk tier: free-space floor %s\n",
                            params_base.cache_disk_min_free_mib < 0 ? "10 GiB (built-in)"
                            : params_base.cache_disk_min_free_mib == 0 ? "off"
                            : (std::to_string(params_base.cache_disk_min_free_mib) + " MiB").c_str());
                }
            }
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
            if (!params_base.cache_disk_dir.empty()) {
                SRV_WRN("--cache-disk-dir %s has no effect: it sits under the RAM prompt cache, which --cache-ram 0 turned off\n",
                        params_base.cache_disk_dir.c_str());
            }
        }

        // GATE-CLAIM-OK: documentation/design comment only, no runtime resource-availability claim
        // -- "unused" below means "not exercised by any request", not "a shared resource is down".
        // DISABLE-REVERSIBLE: --radix-cache is runtime-gated (row's CHALLENGE section) rather than
        // at arg-parse time, because whether it's safe depends on the loaded MODEL, not just the
        // flag. Each why_disabled case below names its own condition; re-enabling means not being
        // that case (recurrent/hybrid/diffusion/SWA), not a fix owed to this code -- those are
        // load-bearing correctness gates (shared position != shared meaning), not missing
        // features. A refusal here leaves the reserved seq ids (already sized into n_seq_max,
        // common.cpp) simply not exercised -- extra KV-sizing cost with no benefit, not a failure.
        if (params_base.radix_cache) {
            std::string why_disabled;
            if (!params_base.kv_unified) {
                why_disabled = "requires --kv-unified (token-granular sharing needs one shared KV stream)";
            } else if (llama_model_is_recurrent(model_tgt)) {
                why_disabled = "model uses recurrent memory (a state vector collapses history -- "
                                "matching absolute position does not mean matching cached meaning)";
            } else if (llama_model_is_hybrid(model_tgt)) {
                why_disabled = "model uses hybrid attention+recurrent memory (same hazard as recurrent)";
            } else if (llama_model_is_diffusion(model_tgt)) {
                why_disabled = "model is a diffusion model (non-causal mask; the prefix-sharing "
                                "assumption behind this cache does not hold)";
            } else if (llama_model_n_swa(model_tgt) > 0) {
                why_disabled = "model uses sliding-window attention (n_swa > 0); a shared prefix's "
                                "cells can be outside a later consumer's own SWA window";
            } else if (params_base.n_radix_cache_slots <= 0) {
                why_disabled = "--radix-cache-slots is 0";
            }

            if (!why_disabled.empty()) {
                SRV_WRN("--radix-cache requested but disabled: %s\n", why_disabled.c_str());
            } else {
                const llama_seq_id seq_id_base = (llama_seq_id) (params_base.n_parallel + params_base.n_cfg_guidance_slots);
                radix_cache = std::make_unique<server_radix_cache>(ctx_tgt, seq_id_base, params_base.n_radix_cache_slots);
                server_radix_evict_policy evict_policy = server_radix_evict_policy::LRU;
                if (!server_radix_evict_policy_parse(params_base.radix_evict_policy, evict_policy)) {
                    SRV_WRN("--radix-evict-policy '%s' is not known, using lru\n", params_base.radix_evict_policy.c_str());
                }
                radix_cache->set_evict_policy(evict_policy, (uint32_t) params_base.radix_evict_slru_hits);
                SRV_INF("radix cache enabled: %d reserved seq id(s) starting at %d, eviction policy %s\n",
                        params_base.n_radix_cache_slots, (int) seq_id_base, server_radix_evict_policy_name(evict_policy));
            }
        }
        // --sched-policy lpm: how much of a waiting request's prompt some cache already holds (a free slot's own leftover prompt, a host
        // prompt cache entry, the radix tree), all under the request's cache namespace. Runs on the loop thread (see server_queue).
        // Installed on the first load only: load_model() also runs when waking from --sleep-idle-seconds, and then the queue loop
        // calls it while holding mutex_tasks, which set_sched_policy() takes too (self-deadlock, the server never answered after
        // waking). The policy comes from params_base and the callback reads this->slots at call time, so a reload changes neither.
        if (!sched_policy_installed) {
            server_sched_params sp;
            if (!server_sched_policy_parse(params_base.sched_policy, sp.policy)) {
                SRV_WRN("--sched-policy '%s' is not known, using fcfs\n", params_base.sched_policy.c_str());
                sp.policy = server_sched_policy::FCFS;
            }
            sp.check_threshold = params_base.sched_prefix_threshold;
            sp.depri_threshold = params_base.sched_prefix_threshold;
            queue_tasks.set_sched_policy(sp, [this](const server_task & t) -> size_t {
                size_t best = 0;
                if (t.tokens.has_mtmd) {
                    return 0; // media prompts are not compared by token
                }
                const uint64_t fp = task_lora_fingerprint(t);
                for (const server_slot & slot : slots) {
                    if (!slot.is_processing() && slot.prompt.lora_fp == fp && !slot.prompt.tokens.has_mtmd) {
                        best = std::max(best, slot.prompt.tokens.get_common_prefix(t.tokens));
                    }
                }
                if (prompt_cache) {
                    for (const auto & st : prompt_cache->states) {
                        if (st.prompt.lora_fp == fp && !st.prompt.tokens.has_mtmd) {
                            best = std::max(best, st.prompt.tokens.get_common_prefix(t.tokens));
                        }
                    }
                }
                if (radix_cache) {
                    best = std::max(best, radix_cache->peek_prefix(t.tokens.get_tokens(), fp));
                }
                return best;
            });
            SRV_INF("scheduling policy %s (prefix threshold %d)\n", server_sched_policy_name(sp.policy), (int) params_base.sched_prefix_threshold);
            sched_policy_installed = true;
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.ctx_checkpoint_spill_dir.empty()) {
            // This server owns the spill dir for its lifetime: remove ckpt-s* files left by a crashed earlier run so
            // orphans never accumulate. RESEARCH-DONE: ik_llama.cpp main examples/server/server-context.cpp:464-476
            // (the same wipe; MIT, source read). Row S69-FLAG-WIRING-CTX-CKPT-SPILL-NOT-PORTED-2026-10-01-01.
            std::error_code ec;
            std::filesystem::create_directories(params_base.ctx_checkpoint_spill_dir, ec);
            int n_wiped = 0;
            if (!ec) {
                for (const auto & entry : std::filesystem::directory_iterator(params_base.ctx_checkpoint_spill_dir, ec)) {
                    if (entry.path().filename().string().rfind("ckpt-s", 0) == 0 && std::filesystem::remove(entry.path(), ec)) {
                        n_wiped++;
                    }
                }
            }
            if (ec) {
                SRV_WRN("context checkpoint spill dir %s is not usable (%s); spilling will fail and checkpoints stay in memory\n",
                        params_base.ctx_checkpoint_spill_dir.c_str(), ec.message().c_str());
            } else {
                SRV_INF("context checkpoint spill dir %s ready (%d stale files removed), live checkpoints per slot = %d\n",
                        params_base.ctx_checkpoint_spill_dir.c_str(), n_wiped, params_base.ctx_checkpoint_ram_live);
            }
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        // every (re)load: the first one and each wake from sleep re-enter here, so anything derived from the loaded context follows
        if (callback_model_loaded) {
            callback_model_loaded(n_ctx_slot());
        }

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });
        // request_sleep() (Ollama keep_alive 0): never put the model to sleep under a working slot
        queue_tasks.on_is_busy([this]() {
            for (const auto & slot : slots) {
                if (slot.is_processing()) {
                    return true;
                }
            }
            return false;
        });

        metrics.init();

        // --request-log: JSONL per finished request, in-flight journal, crash dump of the previous run (server-request-log.h)
        if (!params_base.request_log_path.empty()) {
            server_request_log_config rcfg;
            rcfg.path = params_base.request_log_path;
            rcfg.rate = params_base.request_log_rate;
            std::string rerr = server_request_log_parse_model_rates(params_base.request_log_model_rate, rcfg.model_rate);
            if (rerr.empty()) {
                metrics.reqlog = server_request_log_open_files(rcfg, &rerr);
            }
            if (!metrics.reqlog) {
                SRV_ERR("--request-log disabled: %s\n", rerr.c_str());
            } else {
                SRV_INF("--request-log: %s (rate %.3f)\n", rcfg.path.c_str(), rcfg.rate);
            }
        }

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            server_effort_table effort_table;
            if (!params_base.reasoning_effort_budget.empty()) {
                std::string err;
                if (!effort_table.parse(params_base.reasoning_effort_budget, err)) {
                    SRV_ERR("invalid --reasoning-effort-budget '%s': %s\n", params_base.reasoning_effort_budget.c_str(), err.c_str());
                    return false;
                }
                SRV_INF("reasoning effort -> thinking budget table active (%zu level(s))\n", effort_table.fraction.size());
            }
            server_answer_reserve answer_reserve;
            {
                std::string err;
                if (!answer_reserve.parse(params_base.reasoning_answer_reserve, err)) {
                    SRV_ERR("invalid --reasoning-answer-reserve '%s': %s\n", params_base.reasoning_answer_reserve.c_str(), err.c_str());
                    return false;
                }
                if (answer_reserve.active()) {
                    SRV_INF("reasoning answer reserve active: %s\n", params_base.reasoning_answer_reserve.c_str());
                }
            }

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mtmd_cap.vision,
                /* allow_audio           */ mtmd_cap.audio,
                /* allow_video           */ mtmd_cap.video,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser,
                /* parallel_tool_calls_default */ params_base.parallel_tool_calls,
                /* default_system_prompt */ params_base.system_prompt,
                /* tool_arg_coercion     */ params_base.tool_arg_coercion,
                /* tool_call_format      */ params_base.tool_call_format,
                /* plain_literals        */ {},  // filled just below, once the vocab is known
                /* effort_budget         */ effort_table,
                /* n_predict_default     */ params_base.n_predict,
                /* answer_reserve        */ answer_reserve,
                /* rlr_high_text         */ params_base.reasoning_effort_high_text,
                /* chat_template_kwargs_force */ params_base.force_template_kwargs
            };
            // control-token text inside message text is tokenised as ordinary text (BPE vocabs only)
            try {
                chat_params.plain_literals = server_collect_plain_literals(vocab);
            } catch (const std::exception & e) {
                SRV_ERR("%s: %s\n", __func__, e.what());
                return false;
            }
            SRV_INF("%s: %zu control-token literal(s) neutralised inside message text\n", __func__, chat_params.plain_literals.size());

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                // A BUSY pinned slot is returned untouched; the caller defers the task (is_processing() check
                // after get_available_slot). Before this, the similarity loop below skipped the busy slot, so
                // f_sim_best stayed 0, f_keep = 0 < --cache-ram-similarity set update_cache, and prompt_save() +
                // prompt_load() ran on the slot while it was still generating: a better RAM-cache match for the
                // new request was loaded under the running generation, which then continued on the wrong context.
                // MEASURED 2026-10-18 (tests/test-server-pinned-busy-slot.py, stories15M Q8_0, greedy): the
                // running text diverged at char 33 into the pinned request's prompt. SOURCE-READ: upstream
                // llama.cpp 23b0202a (#30295) tools/server/server-context.cpp:1686-1694.
                // Row S69-LLAMACPP-PICK-PINNED-BUSY-SLOT-UNTOUCHED-2026-10-18-01.
                if (ret->is_processing()) {
                    return ret;
                }

                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // a request in its own cache namespace (cache_salt / extra_key) must not take over a slot whose cache belongs to another
                // configuration: launch_slot_with_task would discard that cache (and, in a multi-tenant server, evict the other tenant's prefix)
                if ((!task.params.cache_salt.empty() || !task.params.extra_key.empty()) && slot.prompt.lora_fp != task_lora_fingerprint(task)) {
                    SLT_TRC(slot, "%s", " - skipping, slot cache belongs to another cache namespace\n");
                    continue;
                }

                // --reasoning-tokens (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-
                // SERVER-2026-09-27-01, item 2; SOURCE-READ: examples/server/server-context.cpp:
                // 1022-1030 @ 35f0cac75): when excluding reasoning tokens, compare the cached and
                // incoming prompts with their <think> spans stripped, so two turns whose visible
                // prefix matches but whose PAST reasoning trace differs don't get scored as
                // dissimilar just because of that trace.
                size_t lcp_len;
                size_t sim_denom;
                if (params_base.think_tokens.exclude) {
                    const server_tokens cache_ex   = tokens.get_tokens_exclude_think(ctx_tgt, params_base.think_tokens);
                    const server_tokens prompt_ex  = task.tokens.get_tokens_exclude_think(ctx_tgt, params_base.think_tokens);
                    lcp_len   = cache_ex.get_common_prefix(prompt_ex);
                    sim_denom = prompt_ex.size();
                } else {
                    lcp_len   = tokens.get_common_prefix(task.tokens);
                    sim_denom = task.tokens.size();
                }
                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float f_sim_cur = sim_denom > 0 ? float(lcp_len) / sim_denom : 0.0f;

                // sim_denom (not task.tokens.size()) is the actual denominator above -- they differ
                // in exclude mode (prompt_ex.size() < task.tokens.size()); print what was compared.
                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, sim_denom, slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr && ret->prompt.tokens.empty()) {
                // A slot pinned by id_slot that holds nothing: f_keep below would be 0/0 = NaN, NaN < threshold is
                // false, update_cache stayed false and the prompt cache was never consulted -- so a pinned request
                // could never reuse another slot's published prefix (the "still goes through cache update logic"
                // promise above did not hold). MEASURED 2026-09-28: cross-slot tests pinning id_slot got cache_n 0.
                // Upstream has the same arithmetic (engines/llama.cpp tools/server/server-context.cpp:1598).
                // An empty slot has nothing to lose, so let the cache load run.
                update_cache = true;
            } else if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache.
                // Threshold = --cache-ram-similarity (fork default 0.5 == the constant this replaced, so the default
                // behaviour is unchanged).
                if (f_keep < cache_ram_similarity) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            // --slot-evict-lookahead N (row S69-NANOVLLM-PICK-QUEUE-AWARE-SLOT-EVICTION-2026-10-18-01; idea from nano-vLLM
            // PR #278, MIT, description read, no code copied): do not evict an idle slot that one of the next N waiting
            // requests would select by LCP similarity, unless every idle slot is such a slot (server-slot-evict.h).
            // Candidates in upstream's order (get_free_slots: empty slots first, then least recently used).
            const std::vector<server_slot *> idle = get_free_slots(slots.size(), -1);
            if (slot_evict_lookahead > 0 && idle.size() > 1 && slot_prompt_similarity > 0.0f) {
                std::vector<int64_t>            t_used(idle.size());
                std::vector<std::vector<float>> sims; // [waiting][candidate]
                for (size_t c = 0; c < idle.size(); ++c) {
                    // an empty slot loses nothing by being reused: it counts as the oldest
                    t_used[c] = idle[c]->prompt.n_tokens() == 0 ? INT64_MIN : idle[c]->t_last_used;
                }
                queue_tasks.peek_waiting((size_t) slot_evict_lookahead, [&](const server_task & w) {
                    std::vector<float> row(idle.size(), 0.0f);
                    // only a completion request that may choose its slot by similarity can retain one
                    if (w.type == SERVER_TASK_TYPE_COMPLETION && w.id_slot == -1 && w.id != task.id && !w.tokens.empty()) {
                        const bool     ns = !w.params.cache_salt.empty() || !w.params.extra_key.empty();
                        const uint64_t fp = ns ? task_lora_fingerprint(w) : 0;
                        for (size_t c = 0; c < idle.size(); ++c) {
                            const auto & toks = idle[c]->prompt.tokens;
                            if (toks.empty() || (ns && idle[c]->prompt.lora_fp != fp)) {
                                continue;
                            }
                            row[c] = float(toks.get_common_prefix(w.tokens)) / w.tokens.size();
                        }
                    }
                    sims.push_back(std::move(row));
                });
                const server_slot_evict_result ev = server_slot_pick_evict(t_used, sims.size(),
                    [&](size_t c, size_t w) { return sims[w][c]; }, slot_prompt_similarity);
                if (ev.pick >= 0) {
                    ret = idle[ev.pick];
                    if (ev.pick != ev.lru_pick) {
                        SLT_INF(*ret, "slot-evict-lookahead: kept slot %d for a waiting request (%zu of %zu idle slot(s) retained, %zu waiting request(s) checked)\n",
                                idle[ev.lru_pick]->id, ev.n_retained, idle.size(), sims.size());
                    }
                }
            }

            if (ret == nullptr && !idle.empty()) {
                ret = idle[0];
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        // Same rule when similarity selection is off (--slot-prompt-similarity 0): an id-pinned EMPTY slot skipped
        // the LRU branch, the only other place update_cache is set, so it never consulted the prompt cache either.
        if (ret && task.id_slot != -1 && ret->prompt.tokens.empty()) {
            update_cache = true;
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache && !task.is_stateless();

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // --cache-ram-n-min gates the SAVE only, as in the fork's server: prompts below the minimum are not
                // stored, but a longer cached prompt may still be loaded for them below.
                if (!ret->prompt.tokens.empty() && (int32_t) ret->prompt.tokens.size() >= cache_ram_n_min) {
                    ret->prompt_save(*prompt_cache);
                } else {
                    SRV_TRC("not saving prompt of %zu tokens (< --cache-ram-n-min %d)\n",
                            ret->prompt.tokens.size(), cache_ram_n_min);
                }

                // The cache is keyed by tokens AND by the LoRA configuration the request will run under: get_available_slot
                // runs before launch_slot_with_task applies the request's adapters, so the fingerprint is derived here from
                // the same inputs. Without it an entry computed under one adapter configuration was restored for another.
                if (!ret->prompt_load(*prompt_cache, task.tokens, params_base.think_tokens, task_lora_fingerprint(task))) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        // purge the least recently used idle slot that holds tokens (server-slot-purge.h), not the lowest slot id
        std::vector<int64_t> t_used;
        std::vector<size_t>  n_tok;
        std::vector<server_slot *> cand;
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }
            cand.push_back(&slot);
            t_used.push_back(slot.t_last_used);
            n_tok.push_back((size_t) slot.prompt.n_tokens());
        }

        const int pick = server_slot_pick_purge(t_used, n_tok);
        if (pick >= 0) {
            server_slot & slot = *cand[pick];
            SRV_WRN("purging slot %d with %zu tokens (least recently used idle slot)\n", slot.id, slot.prompt.tokens.size());

            slot.prompt_clear();

            metrics.n_kv_idle_purged++;
            res = true;
        }

        return res;
    }

    // ---- --lora-pool-dir / --lora-hetero-batch (docs/agent-reports/wire-serving-features.md) ----

    // Position-gated LoRA under speculative decoding (row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01): a draft context built on
    // the TARGET model (MTP) can carry the draft_rows adapters, which are bound to that model. The draft pass marks its rows with
    // llama_set_draft_start (common/speculative.cpp), so the draft rows run the adapter and the verify pass (ctx_tgt, no draft start)
    // runs the base model: acceptance is judged by the base model and the output distribution stays the base model's.
    // A separate draft model (-md) cannot hold an adapter of this model.
    bool dft_gated_lora_ok() const {
        return ctx_dft != nullptr && model_tgt != nullptr && llama_get_model(ctx_dft) == model_tgt;
    }

    bool dft_gated_lora_active(const std::vector<common_adapter_lora_info> & lora) const {
        if (!dft_gated_lora_ok()) {
            return false;
        }
        for (const auto & l : lora) {
            if (l.ptr != nullptr && l.scale != 0.0f && llama_adapter_lora_is_draft_gated(l.ptr)) {
                return true;
            }
        }
        return false;
    }

    // the context-wide set of the draft context: only the draft_rows adapters of `lora`
    void set_dft_gated_lora(const std::vector<common_adapter_lora_info> & lora) {
        if (!dft_gated_lora_ok()) {
            return;
        }
        std::vector<common_adapter_lora_info> gated;
        for (const auto & l : lora) {
            if (l.ptr != nullptr && l.scale != 0.0f && llama_adapter_lora_is_draft_gated(l.ptr)) {
                gated.push_back(l);
            }
        }
        common_set_adapter_lora(ctx_dft, gated);
    }

    void init_lora_pool() {
        if (params_base.lora_hetero_batch) {
            // per-sequence adapters ride on the target context; a draft context on the same model also takes the draft_rows ones
            // (dft_gated_lora_ok); a separate draft model would decode the same sequences with no adapter
            if ((spec || ctx_dft) && !dft_gated_lora_ok()) {
                SRV_ERR("%s", "--lora-hetero-batch is not supported with speculative decoding: ignored (slots with different adapters run in separate batches)\n");
                for (auto & slot : slots) {
                    slot.lora_hetero = false;
                }
            } else {
                SRV_INF("%s", "lora-hetero-batch: slots with different LoRA adapters share one decode batch (per-sequence adapters)\n");
            }
        }
        if (params_base.lora_pool_dir.empty()) {
            return;
        }
        std::error_code ec;
        if (!std::filesystem::is_directory(params_base.lora_pool_dir, ec)) {
            SRV_ERR("--lora-pool-dir '%s' is not a directory: the LoRA pool is off\n", params_base.lora_pool_dir.c_str());
            return;
        }
        lora_pool = std::make_unique<server_lora_pool>(
            model_tgt, params_base.lora_pool_dir,
            (size_t) std::max(0, params_base.lora_pool_max),
            (size_t) std::max(0, params_base.lora_pool_max_mib) * 1024ull * 1024ull,
            [this](llama_adapter_lora * ad) { lora_pool_drop_refs(ad); });
        SRV_INF("lora-pool: dir=%s max_adapters=%d max_mib=%d\n", params_base.lora_pool_dir.c_str(), params_base.lora_pool_max, params_base.lora_pool_max_mib);
    }

    // The pool is about to free `ad`: drop every reference the llama context still holds (the context-wide set is re-applied before every
    // batch, per-sequence lists are re-applied before every batch in hetero mode), so neither the sets nor a cached graph can reach freed weights.
    void lora_pool_drop_refs(llama_adapter_lora * ad) {
        GGML_UNUSED(ad);
        common_set_adapter_lora(ctx_tgt, empty_lora_list);
        if (dft_gated_lora_ok()) {
            common_set_adapter_lora(ctx_dft, empty_lora_list);
        }
        const int n_seq = params_base.n_parallel + std::max(0, params_base.n_cfg_guidance_slots);
        for (int s = 0; s < n_seq; ++s) {
            llama_set_seq_adapters_lora(ctx_tgt, s, nullptr, 0, nullptr);
            if (dft_gated_lora_ok() && s < params_base.n_parallel) { // the draft context has no guidance sequences
                llama_set_seq_adapters_lora(ctx_dft, s, nullptr, 0, nullptr);
            }
        }
    }

    std::vector<common_adapter_lora_info> empty_lora_list;
    uint64_t n_lora_hetero_batches = 0;

    void lora_pool_unpin(server_slot & slot) {
        if (!slot.lora_pool_pin.empty() && lora_pool) {
            lora_pool->release(slot.lora_pool_pin);
        }
        slot.lora_pool_pin.clear();
    }

    // Pins the adapter a request names. False (with the error already sent) when it cannot be made resident.
    bool lora_pool_pin_for_task(server_slot & slot, const server_task & task) {
        lora_pool_unpin(slot);
        if (!lora_pool || task.params.lora_adapter.empty()) {
            return true;
        }
        const auto & st0 = lora_pool->stats();
        const uint64_t ev0 = st0.evictions;
        const auto p = lora_pool->acquire(task.params.lora_adapter);
        const auto & st = lora_pool->stats();
        // always-on observable: one line per pooled request
        SRV_INF("lora-pool: acquire '%s' -> %s%s (serial %llu); resident %zu, acquires %llu, hits %llu, loads %llu, evictions %llu (+%llu), refused %llu, failed %llu\n",
                task.params.lora_adapter.c_str(),
                p.kind == server_lora_pool::result_kind::ok ? "ok" : (p.kind == server_lora_pool::result_kind::no_room ? "REFUSED(no_room)" : "ERROR"),
                p.kind == server_lora_pool::result_kind::ok ? (p.loaded_now ? " loaded" : " hit") : "",
                (unsigned long long) p.serial, lora_pool->resident(),
                (unsigned long long) st.acquires, (unsigned long long) st.hits, (unsigned long long) st.loads,
                (unsigned long long) st.evictions, (unsigned long long) (st.evictions - ev0),
                (unsigned long long) st.refused, (unsigned long long) st.failed);
        if (p.kind != server_lora_pool::result_kind::ok) {
            const bool refused = p.kind == server_lora_pool::result_kind::no_room || p.kind == server_lora_pool::result_kind::over_budget;
            send_error(task, p.error, refused ? ERROR_TYPE_UNAVAILABLE
                           : p.kind == server_lora_pool::result_kind::not_found ? ERROR_TYPE_NOT_FOUND : ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        slot.lora_pool_pin = task.params.lora_adapter;
        return true;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    // The adapter list a task runs with: its own per-request list, or the server-wide one when it names none.
    std::vector<common_adapter_lora_info> task_lora_list(const server_task & task) const {
        auto list = task.params.lora.empty() ? params_base.lora_adapters : construct_lora_list(task.params.lora);
        // the request's pooled adapter (--lora-pool-dir), when it is resident: appended after the server-wide ones
        if (lora_pool && !task.params.lora_adapter.empty()) {
            if (llama_adapter_lora * ad = lora_pool->peek(task.params.lora_adapter)) {
                list.push_back({ lora_pool->path_of(task.params.lora_adapter), task.params.lora_adapter_scale, "", "", ad });
            }
        }
        return list;
    }

    // Fingerprint of the LoRA configuration a task's KV is computed under (0 = no adapter active, always the case on a
    // server with no adapters loaded, which is the fast path below).
    uint64_t task_lora_fingerprint(const server_task & task) const {
        const bool pooled = lora_pool && !task.params.lora_adapter.empty();
        uint64_t lora_fp = params_base.lora_adapters.empty() && !pooled
            ? 0
            : lora_effective_fingerprint(task_lora_list(task), task.tokens);
        if (pooled) {
            // The adapter pointer in the fingerprint is not a stable identity once adapters come and go: a freed adapter's address can be
            // handed to the next one loaded. The load serial is unique per load; 0 (not resident yet) matches nothing that was saved.
            const uint64_t serial = lora_pool->serial_of(task.params.lora_adapter);
            lora_fp = (lora_fp ^ 0x6C6F7261706F6F6CULL) * 1099511628211ULL ^ (serial ? serial : ~0ULL);
            lora_fp = lora_fp ? lora_fp : 1;
        }
        // the control vector is part of the configuration too (0 = none, which leaves lora_fp untouched)
        // cache_salt / extra_key (both empty = none) key the KV like a configuration change
        return kv_namespace_fingerprint(kv_config_fingerprint(lora_fp, cv_fp), task.params.cache_salt, task.params.extra_key);
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // --lora-pool-dir: make the request's adapter resident and pinned BEFORE its adapter list is built; any failure below releases the pin
        if (!lora_pool_pin_for_task(slot, task)) {
            return false;
        }
        const bool launched = launch_slot_with_task_impl(slot, std::move(task));
        if (!launched) {
            lora_pool_unpin(slot);
        }
        return launched;
    }

    bool launch_slot_with_task_impl(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        {
            const auto     task_loras = task_lora_list(task);
            const uint64_t task_fp    = task_lora_fingerprint(task);

            // slot.prompt.lora_fp is the configuration the KV held by this slot was computed under (set here on every
            // launch, carried through prompt cache loads, which only restore an entry with the request's fingerprint).
            // Previously the clear below ran only for a per-request list that differed from slot.lora, so a request
            // without a list, a server-wide LoRA change (POST /lora-adapters) or a prompt-cache restore kept KV that
            // another adapter configuration had produced. Row S69-LORAX-PICK-LORA-CONFIG-IN-PROMPT-CACHE-AND-RADIX-KEY-2026-09-29-01.
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (slot.prompt.lora_fp == task_fp) {
                    SLT_TRC(slot, "keeping cache for lora change with unchanged fingerprint. %zu loras -> %zu loras\n", slot.lora.size(), task_loras.size());
                } else if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task_loras.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            } else if (slot.prompt.lora_fp != task_fp) {
                // same adapter list, but an aLoRA request resolves differently for this prompt (invocation present or not)
                SLT_TRC(slot, "clearing cache: the request runs under another lora fingerprint (%016llx -> %016llx)\n",
                        (unsigned long long) slot.prompt.lora_fp, (unsigned long long) task_fp);
                slot.prompt.clear();
            }
            slot.prompt.lora_fp = task_fp;
        }
        if (!task.params.cache_salt.empty() || !task.params.extra_key.empty()) {
            // always on (not --verbose), on every slot path: the proof that the request ran in its own cache namespace and KV reuse
            // is restricted to requests with the same namespace; the salt itself is never logged
            SLT_INF(slot, "cache namespace active: cache_salt %zu byte(s), extra_key %zu byte(s), fingerprint %016llx\n",
                    task.params.cache_salt.size(), task.params.extra_key.size(), (unsigned long long) task_lora_fingerprint(task));
        }

        // per-sequence adapters carry no aLoRA activation point
        if (slot.lora_hetero && lora_all_alora(slot.lora)) {
            send_error(task, "aLoRA adapters cannot be used with --lora-hetero-batch", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // scan backwards through the prompt tokens to find the last occurrence of the invocation sequence
            // (server-common.cpp; the prompt cache lookup resolves the same search when it fingerprints the request)
            alora_invocation_start = lora_find_alora_invocation_start(lora, task.tokens);

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        // vLLM sampling_params.py:622 refuses min_tokens > max_tokens; with a limit of n_predict the request could
        // never reach its minimum, so say so instead of silently generating n_predict tokens
        if (task.params.n_predict >= 0 && task.params.min_tokens > task.params.n_predict) {
            send_error(task, "min_tokens must be less than or equal to n_predict", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        // logprob_token_ids ride on the n_probs machinery (the per-token readout is gated on n_probs > 0), so ask for the
        // smallest top list when the caller did not
        if (!task.params.logprob_token_ids.empty() && task.params.sampling.n_probs == 0) {
            task.params.sampling.n_probs = 1;
        }

        if (task.params.return_prompt_token_ids && task.params.stream) {
            send_error(task, "return_prompt_token_ids does not support stream", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        // echo: the reply text starts with the prompt, which a streamed reply cannot do; with logprobs asked for too, the prompt tokens
        // have to be scored, so prompt_logprobs comes on with it
        if (task.params.echo) {
            if (task.params.stream) {
                send_error(task, "echo does not support stream", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            if (task.params.sampling.n_probs > 0) {
                task.params.prompt_logprobs = true;
            }
        }

        // prompt_logprobs: every prompt token must be evaluated by THIS request. Any reused prefix (slot cache, radix
        // tree, cache reuse) was not scored here, so its entries would be missing; refuse the combinations that cannot
        // be scored rather than return holes.
        if (task.params.prompt_logprobs) {
            if (task.tokens.has_mtmd) {
                send_error(task, "prompt_logprobs does not support media inputs", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            if (task.params.n_cmpl > 1) {
                send_error(task, "prompt_logprobs does not support n > 1", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            if (task.params.logprob_start_len > 0) {
                if (task.params.logprob_start_len >= (int32_t) task.tokens.size()) {
                    send_error(task, string_format("logprob_start_len (%d) must be smaller than the prompt length (%d tokens)",
                                                    (int) task.params.logprob_start_len, (int) task.tokens.size()), ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                // the tokens before the start are not scored, so they may be reused (capped where the prompt is evaluated, below)
            } else {
                task.params.cache_prompt = false;
            }
            slot.prompt_lp.assign((size_t) task.tokens.size(), std::numeric_limits<float>::quiet_NaN());
            slot.prompt_greedy.assign((size_t) task.tokens.size(), 0);
            slot.prompt_arg.assign((size_t) task.tokens.size(), -1);
            slot.prompt_arg_lp.assign((size_t) task.tokens.size(), std::numeric_limits<float>::quiet_NaN());
        }

        // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01):
        // three additional refusals beyond server-schema.cpp's --cfg-guidance-slots==0 check.
        // Each names a rewind path that mutates slot.id's KV with no knowledge of a guidance
        // sequence -- landing CFG through any of them would silently desync guidance from the
        // main stream instead of failing loudly. Row
        // S69-SERVER-CFG-GUIDANCE-SEQ-MIRROR-REWINDS-2026-09-27-01 owns mirroring these paths and
        // lifting the corresponding refusal. Banned strings are mirrored (ban_rewind) on PART memory,
        // so that refusal is lifted for it; speculative / checkpoint / ctx-shift stay refused.
        if (task.params.sampling.cfg_scale != 1.0f || !task.params.sampling.cfg_negative_prompt.empty()) {
            if (slot.can_speculate()) {
                send_error(task, "Error: classifier-free guidance is not supported together with speculative "
                                  "decoding (a draft model is configured on this server)", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            // HARNESS-BUILD-TASK: S69-SERVER-CFG-GUIDANCE-SEQ-MIRROR-REWINDS-2026-09-27-01 (FULL/RS/SWA guidance mirror)
            if (task.params.has_bans() && (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART || n_swa > 0)) {
                send_error(task, "Error: classifier-free guidance is not supported together with "
                                  "banned_strings/bad_words on this model (only models whose memory can drop a "
                                  "sequence tail without a checkpoint are mirrored)", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            // Mirrors create_checkpoint()'s own root gate (this file, ~4333-4346): when true, the
            // prompt-fill loop MAY create or restore a prompt checkpoint for this task, rewinding
            // slot.id's KV to an earlier position.
            const bool checkpoint_possible = params_base.n_ctx_checkpoints > 0
                && task.type == SERVER_TASK_TYPE_COMPLETION
                && (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                    n_swa > 0);
            if (checkpoint_possible) {
                send_error(task, "Error: classifier-free guidance is not supported on this server/model "
                                  "configuration (prompt context-checkpointing is active: --ctx-checkpoints "
                                  "> 0 with a memory type or SWA setup that checkpoints)", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            // pre_decode()'s context-shift (this file, ~3782-3845) rewinds slot.id's KV the same
            // way -- also not mirrored onto a guidance seq. --no-context-shift is the escape hatch.
            if (params_base.ctx_shift) {
                send_error(task, "Error: classifier-free guidance requires the server to be started with "
                                  "--no-context-shift (context-shift rewinds a slot's KV in a way the "
                                  "guidance sequence does not mirror yet)", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            // Claim a free guidance seq id (server-context.cpp's cfg_seq_free pool, sized by
            // --cfg-guidance-slots). No free id -> refuse rather than silently run unguided.
            if (!claim_cfg_seq(slot)) {
                send_error(task, "Error: classifier-free guidance requested but no --cfg-guidance-slots "
                                  "seq id is free right now", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            // Tokenize and HOLD the negative prompt -- do NOT decode it here. Coordinator review,
            // 2026-09-27: a dedicated out-of-band llama_decode() call (tried first) crashed the
            // server under concurrent load (0xC0000005) because src/llama-context.cpp:2843's
            // output_reserve() resets the shared context's output/logits bookkeeping on EVERY
            // decode call -- an intervening decode for THIS task clobbers another task's
            // still-unconsumed decode results from the same processing round. The whole prompt is
            // folded into the SHARED batch instead, once, at the exact moment the main prompt's
            // own processing completes (the "entire prompt has been processed" branch, below).
            {
                llama_tokens cfg_tokens = common_tokenize(vocab, task.params.sampling.cfg_negative_prompt, true, true);
                const int32_t n_batch_tgt = (int32_t) llama_n_batch(ctx_tgt);
                if (cfg_tokens.empty()) {
                    // koboldcpp's own gate: an empty negative prompt on a no-BOS vocab cannot be
                    // guided against (gpttype_adapter.cpp SOURCE-READ already cited on this row).
                    release_cfg_seq(slot.id_cfg);
                    slot.id_cfg = -1;
                    send_error(task, "Error: classifier-free guidance negative_prompt tokenized to zero "
                                      "tokens (empty prompt on a vocab with no BOS token)", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                if ((int32_t) cfg_tokens.size() > n_batch_tgt) {
                    release_cfg_seq(slot.id_cfg);
                    slot.id_cfg = -1;
                    send_error(task, "Error: classifier-free guidance negative_prompt is longer than "
                                      "--batch-size; not supported in the safe subset", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }

                slot.n_cfg_past = 0;
                slot.cfg_prompt_tokens = std::move(cfg_tokens);
            }
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        if (task.type == SERVER_TASK_TYPE_TTS) {
            GGML_ASSERT(has_cap_tts()); // should already checked in route handler
            if (!slot.tts.is_supported()) {
                slot.tts.ctx.init(ctx_tgt, slot.mctx);
            }
            // mtmd_helper always re-eval the whole prompt
            slot.prompt_clear();
            task.tts_inp.data.seq_id = slot.id;
            if (slot.tts.ctx.set_input(task.tts_inp.get()) != 0) {
                send_error(task, "failed to process TTS prompt", ERROR_TYPE_SERVER);
                return false;
            }
        }

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            // reasoning-token count: same markers the budget sampler gets, and the generation prompt replayed the same
            // way common_sampler_init replays it (a leading space token the tokenizer adds is skipped there too)
            {
                const auto & sp = task.params.sampling;
                std::vector<int32_t> prefill;
                if (!sp.generation_prompt.empty() && !sp.reasoning_budget_start.empty()) {
                    const auto gp = common_tokenize(vocab, sp.generation_prompt, false, true);
                    for (size_t i = 0; i < gp.size(); i++) {
                        const std::string piece = common_token_to_piece(vocab, gp[i], true);
                        if (i == 0 && !piece.empty() && std::isspace((unsigned char) piece[0])
                                && !std::isspace((unsigned char) sp.generation_prompt[0])) {
                            continue;
                        }
                        prefill.push_back(gp[i]);
                    }
                }
                std::vector<std::vector<int32_t>> ends(sp.reasoning_budget_end.begin(), sp.reasoning_budget_end.end());
                slot.reasoning_count.init(
                    std::vector<int32_t>(sp.reasoning_budget_start.begin(), sp.reasoning_budget_start.end()), ends, prefill);
            }

            // reasoning_loop_recovery: observable in timings whenever the request turned any part of it on
            slot.stats.rlr_on    = task.params.rlr_mode != server_loop_recovery::MODE_OFF || task.params.rep_cov_threshold > 0.0f;
            slot.rlr_next_check  = std::max<int32_t>(1, task.params.rep_cov_every);
            if (slot.stats.rlr_on && !slot.reasoning_count.enabled()) {
                // no thinking markers: the reasoning block can never be seen open, so neither half can fire. Say so.
                slot.stats.rlr_note = "the chat template has no thinking markers for this request; reasoning loop detection inactive";
            }

            // allowlist sampler: this request's copy of the per-stage masks, plus its always-allowed pieces at the
            // stage's highest rule bias (old examples/server/server-context.cpp:1774-1779, 1836-1842)
            allow_build_base();
            if (!allow_base_biases.empty()) {
                slot.allow_biases = allow_base_biases;
                std::set<llama_token> piece_tokens;
                for (const auto & p : task.params.allow_pieces) {
                    for (const llama_token t : common_tokenize(vocab, p, false, true)) {
                        piece_tokens.insert(t);
                    }
                }
                for (size_t s = 0; s < slot.allow_biases.size(); ++s) {
                    for (const llama_token t : piece_tokens) {
                        if (t >= 0 && (size_t) t < slot.allow_biases[s].size()) {
                            slot.allow_biases[s][t] = allow_stage_max[s];
                        }
                    }
                }
                slot.allow_idx      = 0;
                slot.allow_gen_prev = 0;
                slot.smpl->server_biases = &slot.allow_biases[0];
            }

            // expiring logit bias: this request's schedule (stages, token tables). Rules come from --expiring-logit-bias-file
            // and/or the request's "expiring_logit_bias" field, both already in task.params.sampling.elb_params. Nothing
            // below runs, and the sampler stays on its old path, when there are none.
            if (!task.params.sampling.elb_params.empty()) {
                elb_build_states(slot);
                SLT_INF(slot, "expiring logit bias: %zu stage(s)\n", slot.smpl->elb_states.size());
            }

            const bool need_pre_sample_logits = (task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs) ||
                                                !task.params.logprob_token_ids.empty();

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // the allowlist biases host logits (common/sampling.cpp:815); a backend sampler never reads them
            use_backend_sampling &= slot.allow_biases.empty();

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // banned strings bias host logits at a position; a backend sampler never reads them
            use_backend_sampling &= !task.params.has_bans();

            // expiring logit bias adds to the host logits (common_expiring_logit_bias_apply); a backend sampler never reads them
            use_backend_sampling &= task.params.sampling.elb_params.empty();

            // the backend chain skips DRY, TFS, adaptive-p, top-a and smoothing; keep the host path
            // when any of them is on rather than dropping it without a word
            if (use_backend_sampling && common_sampler_needs_host(task.params.sampling)) {
                SLT_WRN(slot, "%s", "backend_sampling requested, but an enabled sampler is host-only; sampling on the host\n");
                use_backend_sampling = false;
            }
            // TODO: check verify if this actually works with TTS
            use_backend_sampling &= task.type != SERVER_TASK_TYPE_TTS;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }
            slot.rlr_backend_smpl = use_backend_sampling; // its chain is built once: a recover retry could not raise its temperature

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
            // plan gap G3: the rejection verifier's stream follows the request seed, so a fixed-seed request reproduces;
            // a distinct constant keeps it apart from the target sampler's own dist stream (seeded with the same seed).
            // LLAMA_DEFAULT_SEED keeps it random, as before.
            if (spec) {
                slot.spec_rng.seed(task.params.sampling.seed == LLAMA_DEFAULT_SEED
                        ? std::random_device{}()
                        : (uint32_t) task.params.sampling.seed ^ 0x9E3779B9u);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        if (task.need_sampling() && task.params.has_bans()) {
            // FULL (recurrent / hybrid) memory rewinds through a per-token partial-state checkpoint
            // (server_slot::ban_ckpt); PART and RS remove the discarded tail directly.
            // HARNESS-BUILD-TASK: S69-FORK-BANNED-STRINGS-ON-FULL-SEQ-RM-MODELS-VIA-CHECKPOINTS-2026-09-26-01
            slot.ban_ckpt_on = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
            if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
                ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                // a pattern of L bytes spans at most L tokens that render any text
                size_t n_longest = 0;
                for (const auto & s : task.params.ban_phrases)  { n_longest = std::max(n_longest, s.size()); }
                for (const auto & s : task.params.ban_regex)    { n_longest = std::max(n_longest, s.size()); }
                for (const auto & s : task.params.ban_regex_ci) { n_longest = std::max(n_longest, s.size()); }
                slot.ban_n_hold = task.params.banbuffer_size > 0 ? task.params.banbuffer_size : (int32_t) n_longest + 1;
                for (const auto & s : task.params.ban_regex) {
                    slot.ban_re.emplace_back(s);
                }
                for (const auto & s : task.params.ban_regex_ci) {
                    slot.ban_re.emplace_back(s, std::regex_constants::ECMAScript | std::regex_constants::icase);
                }
            } else {
                // a memory type none of the three paths above knows how to rewind
                // HARNESS-BUILD-TASK: S69-FORK-BANNED-STRINGS-ON-FULL-SEQ-RM-MODELS-VIA-CHECKPOINTS-2026-09-26-01
                SLT_WRN(slot, "banned strings not applied: unhandled memory seq_rm type %d\n", (int) ctx_tgt_seq_rm_type);
            }
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        if (slot.task->type == SERVER_TASK_TYPE_TTS) {
            slot.state = SLOT_STATE_PROCESSING_PROMPT;
        } else {
            slot.state = slot.task->is_child()
                ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
                : SLOT_STATE_STARTED;
        }

        // no prompt, the rows come from the audio stream, which audio controls reach from now on
        if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            slot.audio_stream.reset(mtmd_audio_stream_init(mctx, mtmd_audio_stream_params_default()));
            if (!slot.audio_stream) {
                send_error(slot, "failed to initialize the audio stream", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }
            slot.prompt_clear();
            slot.init_sampler();
            slot.sampled = LLAMA_TOKEN_NULL;
            slot.i_batch = -1;
            slot.stats.update_prompt_start();
            slot.state = SLOT_STATE_GENERATING;
            send_partial_response(slot, {}, false, true);
        }

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    // Grammar jump-forward (row S69-MLC-LLM-PICK-GRAMMAR-JUMP-FORWARD-2026-10-15-01, algorithm in server-jump-forward.h): called right
    // after a sampled token went through process_token(). When the slot's user grammar forces the next text, the tokens spelling it are
    // accepted by the sampler and streamed here, and queued in slot.ff_before so handle_last_sampled_token() decodes them with the
    // sampled token in ONE batch. Returns false when one of them ended the generation (the caller releases the slot as for process_token).
    bool jump_forward(server_slot & slot) {
        slot.ff_before.clear();
        int32_t n_max = slot.task->params.grammar_jump_forward >= 0 ? slot.task->params.grammar_jump_forward : params_base.grammar_jump_forward;
        if (n_max <= 0) {
            return true;
        }
        // cases where a token is not decided by the grammar alone, or where this slot's batch layout is special
        if (slot.can_speculate() || slot.ban_n_hold > 0 || slot.id_cfg >= 0 || !slot.inp_embd.empty() ||
                slot.task->params.sampling.n_probs > 0 || slot.min_tokens_pending() || !slot.allow_biases.empty() ||
                !slot.task->params.tool_call_fix_ids.empty() || slot.task->params.heal_prefix_token != LLAMA_TOKEN_NULL ||
                slot.task->params.sampling.grammar.type != COMMON_GRAMMAR_TYPE_USER) {
            return true;
        }
        // room: the forced tokens and the sampled one are not in slot.prompt yet; keep the slack get_n_draft_max() keeps for a shift
        n_max = std::min<int32_t>(n_max, n_ctx_slot_room(slot));
        // the batch holds max(n_batch, n_parallel) tokens and every slot adds 1 + n_max of them
        n_max = std::min<int32_t>(n_max, (int32_t) (params_base.n_batch / std::max(1, params_base.n_parallel)) - 1);
        if (n_max <= 0) {
            return true;
        }
        const std::string forced = common_sampler_forced_string(slot.smpl.get(), (size_t) n_max * 32);
        if (forced.empty()) {
            return true;
        }
        if (jf_index.pieces.empty()) {
            jf_index.build(token_pieces());
        }
        std::vector<int> canonical;
        for (const llama_token t : common_tokenize(vocab, forced, /* add_special */ false, /* parse_special */ false)) {
            canonical.push_back((int) t);
        }
        const auto plan = server_jump_forward::make_plan(jf_index, forced, canonical, (size_t) n_max);
        if (plan.tokens.empty()) {
            return true;
        }
        SLT_DBG(slot, "grammar jump-forward: %zu token(s) spell '%s'\n", plan.tokens.size(), plan.text.c_str());
        for (const int t : plan.tokens) {
            const llama_token tok = (llama_token) t;
            slot.ff_before.push_back(slot.sampled); // the token before this one is decoded in front of it
            common_sampler_accept(slot.smpl.get(), slot.ctx_tgt, tok, true);
            slot.stats.n_gen += 1;
            slot.stats.n_jump_forward_tokens += 1;
            completion_token_output result;
            result.tok          = tok;
            result.text_to_send = jf_index.pieces[(size_t) t];
            result.prob         = 1.0f;
            if (!process_token(result, slot)) {
                slot.ff_before.clear();
                return false;
            }
        }
        return true;
    }

    // tokens a slot can still take into its context beyond the pending sampled token(s), minus the slack kept for a context shift
    int32_t n_ctx_slot_room(server_slot & slot) {
        return slot.n_ctx - slot.prompt.n_tokens() - 3;
    }

    // reasoning_loop_recovery "recover" (server-loop-recovery.h): a loop detector fired on the token just processed. Returns true
    // when the once-only retry is scheduled (the reply goes on; pre_decode() rewrites the slot before the next batch), false when
    // the caller must stop the reply as before. Every refusal is logged and reported in timings.reasoning_recovery_note.
    // HARNESS-CONSIDERED: the retry splices prompt TOKEN IDS and keeps the KV of the reply in flight; a harness above the HTTP API
    // sees text only and would have to re-encode the prompt, which the spec forbids, so the request lifecycle owns it.
    bool rlr_schedule(server_slot & slot, const char * detector) {
        const auto & tp = slot.task->params;
        if (tp.rlr_mode != server_loop_recovery::MODE_RECOVER) {
            return false;
        }
        auto refuse = [&](const std::string & why) {
            slot.stats.rlr_note = why;
            SLT_WRN(slot, "reasoning loop (%s) at n_gen = %d: not recovered, reply stopped: %s\n", detector, (int) slot.stats.n_gen, why.c_str());
            return false;
        };
        if (slot.rlr_pending || slot.stats.n_reasoning_recoveries > 0) {
            return refuse("a recovery already ran (once only)");
        }
        if (!slot.reasoning_count.open) {
            return refuse("the loop is outside the reasoning block");
        }
        if (slot.prompt.tokens.has_mtmd) {
            return refuse("the prompt has media");
        }
        if (slot.can_speculate() || ctx_dft) {
            return refuse("speculative decoding is on");
        }
        if (slot.id_cfg >= 0) {
            return refuse("classifier-free guidance is on");
        }
        if (tp.has_bans()) {
            return refuse("banned strings are on");
        }
        for (const auto & l : slot.lora) {
            if (l.scale != 0.0f) {
                return refuse("a LoRA adapter is active");
            }
        }
        if (slot.rlr_backend_smpl) {
            return refuse("backend sampling is on");
        }
        if (tp.rlr_user_has_sentence) {
            return refuse("a user message holds the high-effort sentence and there is no system message");
        }

        std::string high = tp.rlr_high_text.empty() ? std::string(server_loop_recovery::QWEN38_EFFORT_HIGH) : tp.rlr_high_text;
        std::string low  = tp.rlr_low_text.empty()  ? std::string(server_loop_recovery::QWEN38_EFFORT_LOW)  : tp.rlr_low_text;

        // the same surrounding text is tokenised onto both sentences, so a tokenizer that glues the final "." to the newlines
        // after it, or a leading space to the first word, is matched as whole tokens either way
        std::vector<server_loop_recovery::splice_variant<llama_token>> vars;
        for (const char * pre : {"", " "}) {
            for (const char * suf : {"", "\n\n", "\n"}) {
                vars.push_back({common_tokenize(vocab, pre + high + suf, false, false),
                                common_tokenize(vocab, pre + low  + suf, false, false)});
            }
        }
        const llama_tokens cur = slot.prompt.tokens.get_text_tokens();
        const size_t n_prompt  = std::min<size_t>(cur.size(), (size_t) std::max<int64_t>(0, (int64_t) slot.task->n_tokens() + slot.rlr_prompt_delta));
        const llama_tokens prompt_only(cur.begin(), cur.begin() + n_prompt);
        const size_t fe = server_loop_recovery::first_turn_end(prompt_only, [&](llama_token t) { return llama_vocab_is_eog(vocab, t); });
        auto sp = server_loop_recovery::effort_splice(cur, fe, vars);
        if (!sp.ok) {
            return refuse("the high-effort sentence is not in the first turn of the prompt as whole tokens");
        }
        // room: the rewritten sequence, the pending sampled token and the context-shift slack must still fit
        if ((int64_t) sp.tokens.size() + 4 > (int64_t) slot.n_ctx) {
            return refuse("no room left in the context for the rewritten prompt");
        }
        slot.rlr_tokens  = std::move(sp.tokens);
        slot.rlr_pos     = sp.pos;
        slot.rlr_pending = true;
        SLT_INF(slot, "reasoning loop (%s) at n_gen = %d: recover scheduled, effort sentence at prompt token %zu (%zu -> %zu tokens)\n",
                detector, (int) slot.stats.n_gen, sp.pos, sp.n_old, sp.n_new);
        return true;
    }

    // Perform a scheduled recovery: remove this slot's sequence from the splice point (or all of it, when the memory cannot
    // remove a suffix: recurrent and hybrid models), decode the rewritten tokens, and raise the sampler floor. Runs in
    // pre_decode(), where no slot is waiting for logits. The already-sampled token is decoded by the normal batch after this.
    void rlr_apply(server_slot & slot) {
        slot.rlr_pending = false;
        const llama_tokens toks = std::move(slot.rlr_tokens);
        slot.rlr_tokens.clear();
        const int32_t n_old = slot.prompt.n_tokens();

        llama_memory_t mem = llama_get_memory(ctx_tgt);
        size_t keep = std::min(slot.rlr_pos, toks.size());
        if (!llama_memory_seq_rm(mem, slot.id, (llama_pos) keep, -1)) {
            llama_memory_seq_rm(mem, slot.id, -1, -1);
            keep = 0;
        }

        const int32_t n_batch = llama_n_batch(ctx_tgt);
        llama_batch b = llama_batch_init(n_batch, 0, 1);
        bool ok = true;
        const int64_t t0 = ggml_time_us();
        for (size_t i = keep; ok && i < toks.size(); ) {
            common_batch_clear(b);
            for (int32_t k = 0; k < n_batch && i < toks.size(); ++k, ++i) {
                common_batch_add(b, toks[i], (llama_pos) i, { slot.id }, false);
            }
            ok = llama_decode(ctx_tgt, b) == 0;
        }
        llama_batch_free(b);

        if (!ok) {
            SLT_ERR(slot, "%s", "reasoning loop recovery: decoding the rewritten prompt failed\n");
            send_error(slot, "reasoning_loop_recovery: decoding the rewritten prompt failed", ERROR_TYPE_SERVER);
            slot.release();
            return;
        }

        // prompt.tokens now mirrors the rewritten memory; checkpoints of the old sequence are void (prompt.clear() drops them)
        slot.prompt.clear();
        slot.prompt.tokens.insert(toks);
        slot.rlr_prompt_delta += (int32_t) toks.size() - n_old;
        slot.rep_tokens.clear(); // the span detectors judge the output after the retry only
        slot.loop_guard.reset();

        // sampler floor: temperature >= 1.0, presence_penalty >= 1.5; nothing else of the client's sampling changes
        auto & sp = slot.smpl->params;
        const float t_old = sp.temp, p_old = sp.penalty_present;
        sp.temp            = std::max(sp.temp, server_loop_recovery::RETRY_MIN_TEMP);
        sp.penalty_present = std::max(sp.penalty_present, server_loop_recovery::RETRY_MIN_PRESENCE);

        slot.stats.n_reasoning_recoveries++;
        slot.stats.rlr_note.clear();
        SLT_INF(slot, "reasoning loop recovery: %d -> %zu tokens re-decoded from %zu in %.1f ms; temp %.2f -> %.2f, presence_penalty %.2f -> %.2f\n",
                n_old, toks.size(), keep, (ggml_time_us() - t0) / 1000.0, t_old, sp.temp, p_old, sp.penalty_present);
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        // EOG is a control token, not content: keep its text out of the generated text even with --special / skip_special_tokens=false
        // (the chat parser failed on the trailing marker and fell back to raw text, which broke tool calls).
        // ik_llama.cpp 762eca65b (#2828); SOURCE-READ ik examples/server/server-context.cpp process_token, same hunk.
        const bool is_eog = llama_vocab_is_eog(vocab, result.tok);
        const std::string token_str = is_eog ? std::string() : result.text_to_send;
        slot.sampled = result.tok;

        // stop_token_ids: ends the generation like an EOS token would, and the token is not part of the output
        if (slot.is_stop_token_id(result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;
            SLT_DBG(slot, "stopped by stop_token_ids, token %d\n", (int) result.tok);
            return false;
        }

        if (slot.task->params.rep_max_pattern_size > 0 || slot.task->params.rep_span_min_count > 0 || slot.task->params.rep_tool_min_count > 0) {
            slot.rep_tokens.push_back(result.tok);
        }

        // every path that keeps a generated token reaches here once (a banned-phrase rewind drops tokens before this)
        slot.reasoning_count.accept(result.tok);
        if (slot.task->params.rep_token_guard) {
            // accepted tokens only: a speculative reject never reaches process_token, so the guard needs no rewind
            slot.loop_guard.accept(result.tok, slot.reasoning_count.open);
            slot.loop_guard_eog = is_eog;
        }

        slot.allow_gen_prev = slot.generated.text.size();
        slot.generated.append(token_str, result.tok);

        // allowlist sampler: advance a rule stage for every keyword now present (old update_allowlist_state,
        // examples/server/server-context.cpp:4675-4692). Search from just before the newest text, so a keyword
        // split across tokens is found once, and not before --allowlist-keyword-delay tokens are generated.
        // As many keywords as rule sets means the LAST keyword ends the allowlist (stage index runs past the end).
        if (!slot.allow_biases.empty()) {
            const auto & kws = params_base.allow_kws;
            const size_t n_kw = std::min(kws.size(), slot.allow_biases.size());
            size_t & idx = slot.allow_idx;
            if ((size_t) slot.stats.n_gen >= slot.task->params.allow_kw_delay && idx < n_kw) {
                const std::string & kw0 = kws[idx];
                const size_t from = slot.allow_gen_prev + 1 > kw0.size() ? slot.allow_gen_prev + 1 - kw0.size() : 0;
                size_t pos = slot.generated.text.find(kw0, from);
                while (pos != std::string::npos) {
                    if (++idx >= n_kw) {
                        break;
                    }
                    pos = slot.generated.text.find(kws[idx], pos + 1);
                }
            }
            allow_apply(slot);
        }
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();
        bool defer_regex_send = false; // stop_regex request: the partial response is sent after the stop checks below
        bool literal_hold     = false; // a "stop" string may still complete: send nothing yet
        // stream_interval > 1: the chunk is decided at the end of this function, after every stop check (a token that turns out to be the last
        // one must flush everything unsent). Per-token data and stop_regex need one chunk per token, so they keep the old path.
        const bool interval_on = slot.task->params.stream && slot.task->params.stream_interval > 1 &&
                                 slot.task->params.sampling.n_probs == 0 && !slot.task->params.return_tokens &&
                                 slot.task->params.stop_regex_compiled.empty();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                // include_stop_str_in_output keeps the matched string itself; anything after it is always cut
                const size_t keep = slot.task->params.include_stop_str_in_output ? slot.stopping_word.size() : 0;
                slot.generated.truncate(pos + stop_pos + keep);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !is_eog) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (!slot.task->params.stop_regex_compiled.empty()) {
                // stop_regex: what to send is decided at the end of this function, when the stop state is final, so that the
                // held-back tail is flushed on a length / EOS stop. text_to_send here is only the tentative unsent text.
                defer_regex_send = true;
                literal_hold     = !send_text;
                result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                slot.add_token(result);
            } else if (interval_on) {
                if (send_text) {
                    slot.n_sendable_text = slot.generated.text.size(); // what interval 1 would send now
                }
                result.text_to_send = "";
                slot.add_token(result);
            } else {
                if (send_text) {
                    // no send the stop word in the response
                    result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                    slot.n_sent_text += result.text_to_send.size();
                    // add the token to slot queue and cache
                } else {
                    result.text_to_send = "";
                }

                slot.add_token(result);
                if (slot.task->params.stream) {
                    send_partial_response(slot, result, false);
                }
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
            // The token ends inside a multi-byte character, so its text waits for the next token -- but the token is
            // generated output all the same. Record it, so completion_probabilities has one entry per generated
            // token, and stream it to a client that asked for per-token ids or probs. Until 2026-10-18 it was
            // dropped: every split character removed one entry, every later entry sat one token off, and a
            // temperature-0 losslessness check read a 0.07-logit near-tie (the first emoji byte token) as a 0.91 gap.
            result.text_to_send = "";
            slot.add_token(result);
            if (slot.task->params.stream && (slot.task->params.sampling.n_probs > 0 || slot.task->params.return_tokens)) {
                send_partial_response(slot, result, false);
            }
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        // repetition_detection: after the limits and after min_tokens, like vLLM's check_stop (utils.py:126-138)
        if (slot.has_next_token && (slot.task->params.rep_max_pattern_size > 0 || slot.task->params.rep_span_min_count > 0 ||
                                    slot.task->params.rep_tool_min_count > 0 || slot.task->params.rep_token_guard) &&
                (int32_t) slot.stats.n_gen >= slot.task->params.min_tokens && slot.repetition_detected()) {
            // reasoning_loop_recovery "recover": a loop inside the reasoning resumes once instead of ending the reply
            if (!rlr_schedule(slot, "repetition_detection")) {
                slot.stop           = STOP_TYPE_REPETITION;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by repetition_detection, n_output = %d\n", (int) slot.rep_tokens.size());
            }
        }

        // reasoning_loop_recovery coverage detector (server-loop-recovery.h): word-level, on the output text, only while the
        // reasoning block is open, every rep_cov_every tokens at a clean UTF-8 boundary, and only before the first recovery
        if (slot.has_next_token && !slot.rlr_pending && !incomplete && slot.stats.n_reasoning_recoveries == 0 &&
                slot.reasoning_count.open && (int32_t) slot.stats.n_gen >= slot.task->params.min_tokens) {
            const auto & tp = slot.task->params;
            const float th  = tp.rep_cov_threshold > 0.0f ? tp.rep_cov_threshold
                            : (tp.rlr_mode != server_loop_recovery::MODE_OFF ? server_loop_recovery::coverage_params().threshold : 0.0f);
            if (th > 0.0f && (int32_t) slot.stats.n_gen >= slot.rlr_next_check) {
                slot.rlr_next_check = (int32_t) slot.stats.n_gen + std::max<int32_t>(1, tp.rep_cov_every);
                server_loop_recovery::coverage_params cp;
                cp.threshold = th;
                cp.window    = tp.rep_cov_window;
                const float cov = server_loop_recovery::reasoning_coverage(slot.generated.text, cp);
                slot.stats.reasoning_coverage = cov;
                if (cov >= th && !rlr_schedule(slot, "coverage")) {
                    slot.stop           = STOP_TYPE_REPETITION;
                    slot.has_next_token = false;
                    SLT_INF(slot, "reasoning loop: coverage %.3f >= %.3f at n_gen = %d, reply stopped\n", cov, th, (int) slot.stats.n_gen);
                }
            }
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        // end-to-end deadline (t_deadline_ms): the budget counted from the request's arrival ran out; stop with finish_reason "deadline".
        // Checked before EOS so a model that ends on its own at the same step still reports "eos". Row S69-TFSERVING-PICK-END-TO-END-DEADLINE-2026-10-15-01.
        if (slot.has_next_token && slot.task->t_deadline_us > 0 && ggml_time_us() >= slot.task->t_deadline_us) {
            slot.stop           = STOP_TYPE_DEADLINE;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by the end-to-end deadline, n_gen = %d, t_deadline_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_deadline_ms);
        }

        // budget group (server-budget-group.h): one generated token is charged to the pool shared by the whole recursion tree; when the
        // pool (tokens or wall time) is spent, this slot ends with finish_reason "budget_group". Checked after EOS/deadline so a request
        // that ends on its own at the same step keeps its own reason. Only generated tokens count: a prompt served from the radix
        // cache costs the pool nothing.
        if (slot.has_next_token && !slot.task->params.budget_group.empty()) {
            const server_budget_stop bs = budget_groups.charge(slot.task->params.budget_group, 1, ggml_time_us());
            if (bs != SERVER_BUDGET_OK) {
                slot.stop           = STOP_TYPE_BUDGET_GROUP;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by the budget group '%s' (%s), n_gen = %d\n", slot.task->params.budget_group.c_str(), server_budget_stop_str(bs), (int) slot.stats.n_gen);
            }
        }

        // fail-closed grammar sampler (llguidance / xgrammar matcher error): the constrained output can no longer be trusted, so the
        // slot ends here with an explicit finish_reason "grammar_error" instead of waiting for the forced EOS.
        // Row S69-ENGREV-APHRODITE-LLG-FAILCLOSED-2026-10-27-01 follow-up.
        if (slot.has_next_token && !is_eog && common_sampler_grammar_failed(slot.smpl.get(), &slot.grammar_error)) {
            slot.stop           = STOP_TYPE_GRAMMAR_ERROR;
            slot.has_next_token = false;
            result.text_to_send.clear();

            SLT_ERR(slot, "grammar sampler failed closed, ending the slot: %s\n", slot.grammar_error.c_str());
        }

        if (is_eog) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;
            result.text_to_send.clear();

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        // stop_regex: send now. While the generation continues the last STOP_REGEX_WINDOW_BYTES bytes stay back (a regex
        // can match text that started that far behind the end); on a final stop everything left is flushed.
        if (defer_regex_send) {
            const size_t pos_sent = std::min(slot.n_sent_text, slot.generated.text.size());
            size_t end = slot.generated.text.size();
            if (slot.has_next_token) {
                const size_t hold = task_params::STOP_REGEX_WINDOW_BYTES;
                end = end > pos_sent + hold ? end - hold : pos_sent;
                // never cut inside a UTF-8 sequence
                while (end > pos_sent && (static_cast<unsigned char>(slot.generated.text[end]) & 0xC0) == 0x80) {
                    end--;
                }
                if (literal_hold) {
                    end = pos_sent;
                }
            }
            result.text_to_send = slot.generated.text.substr(pos_sent, end - pos_sent);
            slot.n_sent_text += result.text_to_send.size();
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        // stream_interval > 1: send one chunk for the tokens collected so far when N are unsent, for the first token, or when this was the
        // last one. SOURCE-READ: sglang output_streamer.py:413-445 (sends at len % N == 1; the fork counts unsent tokens instead, so a
        // speculative step that adds several tokens cannot skip the boundary). Apache-2.0, no code copied.
        if (interval_on) {
            slot.n_unsent_tokens++;
            const bool final_chunk = !slot.has_next_token;
            if (final_chunk || slot.stats.n_gen <= 1 || slot.n_unsent_tokens >= slot.task->params.stream_interval) {
                const size_t pos_sent = std::min(slot.n_sent_text, slot.generated.text.size());
                const size_t end      = std::max(pos_sent, std::min(slot.n_sendable_text, slot.generated.text.size()));
                if (end > pos_sent || final_chunk) {
                    result.text_to_send = slot.generated.text.substr(pos_sent, end - pos_sent);
                    slot.n_sent_text += result.text_to_send.size();
                    slot.n_unsent_tokens = 0;
                    send_partial_response(slot, result, false);
                }
            }
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    //
    // banned strings
    //
    // Positions are indices into slot.prompt.tokens. The newest buffered token has just been
    // sampled and is not in the prompt yet, so ban_buffer[i] sits at prompt.n_tokens() - B + 1 + i.
    // (HARNESS-CONSIDERED: this block ADDS an output-constraint capability; it switches nothing off)
    //

    static int32_t ban_buffer_pos(const server_slot & slot, size_t i) {
        return slot.prompt.n_tokens() - (int32_t) slot.ban_buffer.size() + 1 + (int32_t) i;
    }

    // before sampling: bias the tokens banned at this position, and remember the sampler state
    void ban_before_sample(server_slot & slot, int tok_idx) {
        const auto it = slot.ban_at.find(slot.prompt.n_tokens());
        if (it != slot.ban_at.end()) {
            float * logits = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
            for (const llama_token tok : it->second) {
                logits[tok] += slot.task->params.banned_bias;
            }
        }
        slot.ban_smpl.emplace_back(common_sampler_clone(slot.smpl.get()));
    }

    // index into ban_buffer of the token where the earliest match starts, or -1
    int32_t ban_find(const server_slot & slot) const {
        std::string text;
        std::vector<size_t> offs;
        offs.reserve(slot.ban_buffer.size());
        for (const auto & r : slot.ban_buffer) {
            offs.push_back(text.size());
            text += r.text_to_send;
        }

        size_t best = std::string::npos;
        if (!slot.task->params.ban_phrases.empty()) {
            const std::string lower = string_lower(text);
            for (const auto & s : slot.task->params.ban_phrases) {
                best = std::min(best, lower.find(s));
            }
        }
        for (const auto & re : slot.ban_re) {
            std::smatch m;
            if (std::regex_search(text, m, re)) {
                best = std::min(best, (size_t) m.position(0));
            }
        }
        if (best == std::string::npos) {
            return -1;
        }
        for (size_t i = 0; i < offs.size(); ++i) {
            const size_t end = i + 1 < offs.size() ? offs[i + 1] : text.size();
            if (best >= offs[i] && best < end) {
                return (int32_t) i;
            }
        }
        return -1;
    }

    // drop ban_buffer[j..] and resample from there with its first token(s) banned at that position;
    // false when the memory cannot take the rewind, and then nothing has changed
    bool ban_rewind(server_slot & slot, int32_t j) {
        const auto & params = slot.task->params;

        const int32_t pos    = ban_buffer_pos(slot, j); // index being resampled
        const int32_t n_keep = pos - 1;                 // prompt[n_keep] is decoded again to get its logits
        if (n_keep < 0 || n_keep >= slot.prompt.n_tokens()) {
            return false;
        }
        const llama_token prev = slot.prompt.tokens[n_keep];
        if (prev == LLAMA_TOKEN_NULL) {
            return false; // a media chunk cannot be decoded again as one token
        }
        const int32_t n_rm = slot.prompt.n_tokens() - n_keep;
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rm > (int32_t) llama_n_rs_seq(slot.ctx_tgt)) {
            return false;
        }
        // Classifier-Free Guidance (row S69-SERVER-CFG-GUIDANCE-SEQ-MIRROR-REWINDS-2026-09-27-01): the guidance
        // sequence replays every sampled main token one-for-one (pre_decode), so dropping the last n_rm main
        // tokens drops the last n_rm guidance cells and `prev` is decoded again into both. That only holds while
        // `prev` is a GENERATED token: when n_keep falls on the last prompt token the guidance sequence would
        // need ITS OWN last prompt token again, not `prev`, so that rewind is refused (match let through).
        // Only the PART (seq_rm) path is mirrored; FULL / RS are refused at launch_slot_with_task.
        if (slot.id_cfg >= 0) {
            if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
                slot.n_cfg_past - n_rm < (int32_t) slot.cfg_prompt_tokens.size()) {
                return false;
            }
        }
        // FULL memory: the checkpoint taken before ban_buffer[j]'s predecessor was decoded must exist
        // and sit exactly at n_keep, or restoring it would leave memory and prompt.tokens disagreeing
        const bool use_ckpt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        if (use_ckpt && (j >= (int32_t) slot.ban_ckpt.size() || slot.ban_ckpt[j].data_tgt.empty() ||
                         slot.ban_ckpt[j].n_tokens != n_keep)) {
            return false;
        }

        auto & bans = slot.ban_at[pos];
        for (int32_t i = j, n = 0; i < (int32_t) slot.ban_buffer.size(); ++i, ++n) {
            if (params.banned_n >= 0 && n >= params.banned_n) {
                break;
            }
            bans.insert(slot.ban_buffer[i].tok);
        }
        // bans recorded further along the branch being discarded belong to that branch
        slot.ban_at.erase(slot.ban_at.upper_bound(pos), slot.ban_at.end());

        SLT_DBG(slot, "banned match at index %d ('%s'): rewinding %d token(s), %zu banned there\n",
                pos, slot.ban_buffer[j].text_to_send.c_str(), n_rm, bans.size());

        const size_t n_drop = slot.ban_buffer.size() - j;

        slot.prompt.tokens.keep_first(n_keep);
        if (use_ckpt) {
            const auto & ck = slot.ban_ckpt[j];
            ck.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            slot.mem.seq_rm(slot.id, ck.pos_max + 1, -1); // the attention half of a hybrid
            slot.ban_ckpt.resize(j);
        } else {
            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
        }
        if (slot.id_cfg >= 0) {
            slot.n_cfg_past -= n_rm;
            if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id_cfg, (llama_pos) slot.n_cfg_past, -1)) {
                SLT_WRN(slot, "guidance sequence %d could not drop its tail from %d; guidance for this request is stale\n",
                        (int) slot.id_cfg, slot.n_cfg_past);
            }
        }
        slot.sampled = prev;

        // what is being dropped, for the sampler's rewind counters (ik: server_context::rewind_context, SOURCE-READ
        // examples/server/server-context.cpp @32cddbfce:4729-4733). They are RECORDED only: this server restores the whole
        // sampler from ban_smpl[j] just below, which already carries the expiring-logit-bias schedule (elb_states / elb_idx /
        // elb_search_pos and the params' addflags / search_posi, all copied by common_sampler_clone). ik instead keeps one live
        // sampler and subtracts the dropped tokens from it in common_sampler_review -> common_expiring_logit_bias_rewind;
        // calling that here as well would rewind a schedule that is already back where it was. The counters are not part of
        // the clone, and sampler_feed_text() zeroes them before the next sample.
        std::string dropped_text;
        for (size_t k = (size_t) j; k < slot.ban_buffer.size(); ++k) {
            dropped_text += slot.ban_buffer[k].text_to_send;
        }

        common_sampler_copy(slot.ban_smpl[j].get(), slot.smpl.get());
        slot.smpl->n_rewind      = (int32_t) n_drop;
        slot.smpl->rewinded_text = std::move(dropped_text);
        slot.ban_smpl.resize(j);
        slot.ban_buffer.resize(j);

        if (params.saturate_predict) {
            slot.stats.n_gen -= std::min<uint64_t>(slot.stats.n_gen, n_drop);
        }
        slot.ban_rewinds++;
        return true;
    }

    // send up to n buffered tokens in order; true when the slot finished (final response sent)
    bool ban_flush(server_slot & slot, size_t n) {
        while (n-- > 0 && !slot.ban_buffer.empty()) {
            const int32_t pos = ban_buffer_pos(slot, 0);
            completion_token_output res = std::move(slot.ban_buffer.front());
            slot.ban_buffer.erase(slot.ban_buffer.begin());
            slot.ban_smpl.erase(slot.ban_smpl.begin());
            if (!slot.ban_ckpt.empty()) {
                slot.ban_ckpt.erase(slot.ban_ckpt.begin());
            }
            // a sent token can no longer be rewound, so neither can anything before it
            slot.ban_at.erase(slot.ban_at.begin(), slot.ban_at.upper_bound(pos));

            const bool more = process_token(res, slot);
            // the n_predict budget is spent when the LAST token is sampled, so while flushing the
            // tokens before it a budget stop is expected and must not drop them
            const bool budget_only = !more && slot.stop == STOP_TYPE_LIMIT && !slot.has_budget();
            if (!more && !(budget_only && !slot.ban_buffer.empty())) {
                slot.print_timings();
                send_final_response(slot);
                slot.release();
                return true;
            }
        }
        return false;
    }

    // buffer a sampled token; rewind on a match, otherwise send what no longer needs holding.
    // true when the slot finished
    bool ban_step(server_slot & slot, completion_token_output && result) {
        const auto & params = slot.task->params;
        const llama_token tok = result.tok;
        slot.ban_buffer.push_back(std::move(result));
        if (slot.ban_ckpt_on) {
            // the state this token's predecessor was decoded from; an empty entry means "let through"
            slot.ban_ckpt.push_back(slot.ban_ckpt_pending_ok ? std::move(slot.ban_ckpt_pending) : common_prompt_checkpoint{});
            slot.ban_ckpt_pending_ok = false;
            slot.ban_ckpt_pending.clear();
        }

        const int32_t j = ban_find(slot);
        if (j >= 0) {
            const int32_t n_pat = (int32_t) (params.ban_phrases.size() + params.ban_regex.size() + params.ban_regex_ci.size());
            const int32_t n_max = params.rewind_count_max == -1 ? std::max(20, 2 * n_pat) : params.rewind_count_max;
            if ((n_max == 0 || slot.ban_rewinds < n_max) && ban_rewind(slot, j)) {
                return false;
            }
            SLT_WRN(slot, "banned match at index %d let through (rewinds = %d)\n", ban_buffer_pos(slot, j), slot.ban_rewinds);
        }

        // the same stop conditions process_token applies, checked now because sending lags sampling
        const bool stopping = llama_vocab_is_eog(vocab, tok) || !slot.has_budget() ||
                              (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx);

        const size_t n_buf  = slot.ban_buffer.size();
        const size_t n_send = stopping ? n_buf : (n_buf >= (size_t) slot.ban_n_hold ? n_buf - slot.ban_n_hold + 1 : 0);
        if (n_send > 0) {
            slot.ban_rewinds = 0;
        }
        if (ban_flush(slot, n_send)) {
            return true;
        }
        if (stopping) {
            // process_token did not stop on the last token although a stop was due -- end it anyway
            slot.print_timings();
            send_final_response(slot);
            slot.release();
            return true;
        }
        // process_token set sampled to the last token it SENT; the next decode needs the newest one
        slot.sampled = tok;
        return false;
    }

    // logprob_token_ids: log-softmax of the requested ids over the row's raw logits, read the way get_token_probabilities()
    // reads them: entry i of the row belongs to token sampled_ids[i] (the fork hands back the identity list 0..n-1 when no
    // backend sampler ran, so a non-null list does NOT mean the row is a reduced candidate set). An id the row does not
    // hold (a reduced backend-sampled row) is left out of the answer instead of being reported as a number.
    void populate_id_logprobs(const server_slot & slot, completion_token_output & result, int idx) const {
        const auto & ids = slot.task->params.logprob_token_ids;
        if (ids.empty()) {
            return;
        }
        const float *       lg          = llama_get_logits_ith(ctx_tgt, idx);
        const llama_token * sampled_ids = llama_get_sampled_candidates_ith(ctx_tgt, idx);
        if (lg == nullptr) {
            return;
        }
        const int n_row = (int) llama_get_sampled_logits_count_ith(ctx_tgt, idx);
        if (n_row <= 0) {
            return;
        }
        std::unordered_map<llama_token, size_t> want;
        want.reserve(ids.size() * 2);
        for (size_t k = 0; k < ids.size(); ++k) {
            want.emplace(ids[k], k);
        }
        std::vector<float> got(ids.size(), std::numeric_limits<float>::quiet_NaN());
        float max_l = lg[0];
        for (int i = 1; i < n_row; ++i) {
            max_l = std::max(max_l, lg[i]);
        }
        double sum = 0.0;
        for (int i = 0; i < n_row; ++i) {
            sum += std::exp((double) (lg[i] - max_l));
            const llama_token tok = sampled_ids ? sampled_ids[i] : (llama_token) i;
            const auto it = want.find(tok);
            if (it != want.end()) {
                got[it->second] = lg[i];
            }
        }
        const double log_z = std::log(sum);
        result.id_logprobs.reserve(ids.size());
        for (size_t k = 0; k < ids.size(); ++k) {
            if (!std::isnan(got[k])) {
                result.id_logprobs.emplace_back(ids[k], (float) ((double) (got[k] - max_l) - log_z));
            }
        }
    }

    // --restrict-output-vocab (CTranslate2's vocabulary map, Decoder::update_output_layer, src/layers/decoder.cc:72-135): when every running request
    // carries allowed_token_ids and none of them reads the raw distribution (n_probs / logprobs / echo / prompt scoring, which are normalised over the
    // WHOLE vocabulary), the output projection of the next decode is multiplied only by the union of the allowed tokens. The sampler masks everything
    // else to -inf anyway (common/sampling.cpp, allowed_token_ids), so the sampled token is unchanged; any other request in the batch switches it off.
    int32_t output_subset_logged = 0;   // last value reported by the decode log line
    void update_output_vocab_subset() {
        if (!params_base.restrict_output_vocab) {
            return;
        }
        std::vector<llama_token> uni;
        bool ok = true;
        int  n_active = 0;
        for (auto & slot : slots) {
            if (!slot.is_processing() || !slot.task) {
                continue;
            }
            n_active++;
            const auto & tp = slot.task->params;
            if (tp.sampling.allowed_token_ids.empty() || tp.sampling.n_probs > 0 || !tp.logprob_token_ids.empty() || tp.prompt_logprobs || tp.echo || tp.n_cmpl > 1) {
                ok = false;
                break;
            }
            uni.insert(uni.end(), tp.sampling.allowed_token_ids.begin(), tp.sampling.allowed_token_ids.end());
        }
        if (!ok || n_active == 0) {
            uni.clear();
        }
        // llama_set_output_vocab_subset ignores an unchanged set and clears on an empty one, so calling it every step costs a sort of a few ids
        if (!llama_set_output_vocab_subset(ctx_tgt, uni.data(), (int32_t) uni.size()) && !uni.empty()) {
            SRV_WRN("%s", "--restrict-output-vocab: this model's output weight cannot be restricted, using the full projection\n");
            params_base.restrict_output_vocab = false;
        }
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        populate_id_logprobs(slot, result, idx);

        if (slot.stats.n_gen == 1) {
            // no_speech_prob proxy (see completion_token_output::eog_prob): one softmax over the raw row, once per request.
            // With backend sampling the row holds only the sampled candidates, so the sum is over those.
            const std::vector<llama_token_data> all = get_token_probabilities(ctx_tgt, idx, 0);
            double s = 0.0;
            for (const auto & t : all) {
                if (llama_vocab_is_eog(vocab, t.id)) {
                    s += t.p;
                }
            }
            result.eog_prob = (float) std::min(1.0, s);
        }

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);

            // The legacy chain leaves .p as exp(logit - max) -- llama_sample_softmax_impl with
            // normalize=false -- or stale from an earlier stage, so reporting .p gave the top token
            // 1.0 and a list summing past 1 (MEASURED 2026-09-26: ',' reported 1.0 where its
            // pre-sampling probability was 0.969). The token is drawn from
            // exp(logit - max) / sum over the final candidates (llama_sample_token_with_rng_impl),
            // so that distribution is what is reported.
            std::vector<std::pair<llama_token, float>> post(cur_p->size);
            float max_l = -INFINITY;
            for (size_t i = 0; i < cur_p->size; i++) {
                max_l = std::max(max_l, cur_p->data[i].logit);
            }
            double sum = 0.0;
            for (size_t i = 0; i < cur_p->size; i++) {
                const float e = std::isfinite(cur_p->data[i].logit) ? expf(cur_p->data[i].logit - max_l) : 0.0f;
                post[i] = { cur_p->data[i].id, e };
                sum += e;
            }
            if (sum > 0.0) {
                for (auto & e : post) {
                    e.second = (float) (e.second / sum);
                }
            }
            std::stable_sort(post.begin(), post.end(), [](const auto & a, const auto & b) { return a.second > b.second; });

            const size_t n_probs = std::min(post.size(), n_probs_request);

            // set probability for sampled token
            for (const auto & e : post) {
                if (e.first == result.tok) {
                    result.prob = e.second;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (post[i].second == 0.0f) {
                    break;
                }

                result.probs.push_back({
                    post[i].first,
                    common_token_to_piece(ctx_tgt, post[i].first, special),
                    post[i].second
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            res->tokens.assign(1, tkn.tok);
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_reasoning_tokens    = slot.reasoning_count.n;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_tts_result(server_slot & slot, int32_t sample_rate, const char * data, size_t data_len, bool final) {
        auto res = std::make_unique<server_task_result_tts>();

        res->id          = slot.task->id;
        res->index       = slot.task->index;
        res->sample_rate = sample_rate;
        res->audio.assign(data, data_len);
        res->final       = final;

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        // stream_interval > 1: a stop that did not come through process_token (so no last chunk was sent) must not lose the text still unsent
        if (slot.task->params.stream && slot.n_unsent_tokens > 0) {
            const size_t pos_sent = std::min(slot.n_sent_text, slot.generated.text.size());
            const size_t end      = std::max(pos_sent, std::min(slot.n_sendable_text, slot.generated.text.size()));
            completion_token_output tail;
            tail.tok  = LLAMA_TOKEN_NULL;
            tail.prob = 0.0f;
            tail.text_to_send = slot.generated.text.substr(pos_sent, end - pos_sent);
            slot.n_sent_text += tail.text_to_send.size();
            slot.n_unsent_tokens = 0;
            send_partial_response(slot, tail, false);
        }

        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        if (slot.task->params.return_prompt_token_ids) {
            res->return_prompt_token_ids = true;
            res->prompt_token_ids.reserve(slot.task->tokens.size());
            for (size_t i = 0; i < slot.task->tokens.size(); ++i) {
                res->prompt_token_ids.push_back(slot.task->tokens[i]); // LLAMA_TOKEN_NULL at a media position
            }
        }
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_reasoning_tokens    = slot.reasoning_count.n;
        res->n_preempt_requeues    = slot.task->n_preempt_requeues;
        res->n_prompt_truncated    = slot.task->n_prompt_truncated;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->grammar_error         = slot.grammar_error;
        res->echo                 = slot.task->params.echo;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;
        res->asr_stats             = slot.task->params.asr_stats;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD && !slot.task->params.include_stop_str_in_output) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        if (slot.task->params.prompt_logprobs) {
            res->prompt_logprobs = std::move(slot.prompt_lp);
            res->prompt_greedy   = std::move(slot.prompt_greedy);
            if (slot.task->params.echo) {
                res->prompt_arg    = std::move(slot.prompt_arg);
                res->prompt_arg_lp = std::move(slot.prompt_arg_lp);
                const auto & ptoks = slot.task->tokens;
                res->prompt_pieces.reserve(ptoks.size());
                res->prompt_arg_pieces.reserve(ptoks.size());
                for (size_t i = 0; i < ptoks.size(); ++i) {
                    res->prompt_pieces.push_back(common_token_to_piece(ctx_tgt, ptoks[i], true));
                    res->prompt_arg_pieces.push_back(i < res->prompt_arg.size() && res->prompt_arg[i] >= 0 ?
                            common_token_to_piece(ctx_tgt, res->prompt_arg[i], true) : std::string());
                }
                res->echo = true;
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    // prompt_logprobs: for every output row of this slot's prompt in the just-decoded (sub-)batch, the log-softmax of the
    // logits at the prompt token that FOLLOWS the row's position, and whether that token was the row's argmax. The row of
    // the last prompt token predicts the first generated token, which is not part of the prompt, so it has no entry.
    // Rows are addressed relative to the rendered view (batch.view), like llama_get_embeddings_ith in send_embedding().
    void extract_prompt_logprobs(server_slot & slot, const common_batch & view) {
        const auto & toks   = slot.task->tokens;
        const int    n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < view.size(); ++i) {
            const auto & e = view.tokens[(size_t) i];
            if (!e.output || e.seq_id != slot.id) {
                continue;
            }
            const int64_t next_idx = (int64_t) e.pos[0] + 1;
            if (next_idx <= 0 || next_idx >= (int64_t) toks.size() || (size_t) next_idx >= slot.prompt_lp.size()) {
                continue;
            }
            const float * lg = llama_get_logits_ith(slot.ctx_tgt, i);
            if (lg == nullptr) {
                throw std::runtime_error(string_format("prompt_logprobs: no logits for prompt row %d", i));
            }
            const llama_token next = toks[(size_t) next_idx];
            float  max_l = lg[0];
            int    arg   = 0;
            for (int t = 1; t < n_vocab; ++t) {
                if (lg[t] > max_l) { max_l = lg[t]; arg = t; }
            }
            double sum = 0.0;
            for (int t = 0; t < n_vocab; ++t) {
                sum += std::exp((double) (lg[t] - max_l));
            }
            slot.prompt_lp[(size_t) next_idx]     = (float) ((double) (lg[next] - max_l) - std::log(sum));
            slot.prompt_greedy[(size_t) next_idx] = arg == next ? 1 : 0;
            slot.prompt_arg[(size_t) next_idx]    = arg;
            // same expression as the token's own value, so when the token IS the top one the two floats are identical
            slot.prompt_arg_lp[(size_t) next_idx] = (float) ((double) (lg[arg] - max_l) - std::log(sum));
        }
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
            if (!decision.label_groups.empty()) {
                std::vector<float> scores;
                size_t i = 0;
                for (const int32_t n : decision.label_groups) {
                    GGML_ASSERT(n > 0 && i + n <= res->scores.size());
                    scores.push_back(*std::max_element(res->scores.begin() + i, res->scores.begin() + i + n));
                    i += n;
                }
                res->scores = std::move(scores);
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };

            // joint head (decision model): the scores are the first rows
            for (int32_t i = 0; i < decision.n_scores; i++) {
                const float * embd = i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
                if (embd == nullptr) {
                    send_error(slot, "failed to get embeddings", ERROR_TYPE_SERVER);
                    return;
                }
                res->scores.push_back(embd[0]);
            }

            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
            GGML_ASSERT(decision.column >= 0 && decision.column < n_embd_out);

            for (const int32_t marker : decision.markers) {
                const float * embd = get_embd(marker);
                if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                    send_error(slot, "failed to get embeddings, the question and its options must fit in one batch", ERROR_TYPE_SERVER);
                    return;
                }
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }
                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
        }

        SLT_DBG(slot, "%s", "sending decision result\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
            const int32_t n_cls = std::max<int32_t>(1, llama_model_n_cls_out(model_tgt));
            res->scores.assign(embd, embd + n_cls);
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
        }

        std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot * a, const server_slot * b) {
            if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) {
                return a->prompt.n_tokens() == 0;
            }
            return a->t_last_used < b->t_last_used;
        });

        free_slots.resize(std::min(free_slots.size(), n_slots_needed));

        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // ctx_checkpoint_eviction (fork field, port row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-
    // EXAMPLES-SERVER-2026-09-27-01, item 3). SOURCE-READ: examples/server/server-context.cpp:3791-
    // 3834 @ 35f0cac75 (evict_checkpoint_by_variance, the fork's own MIT code, adapted here to the
    // split server's common_prompt_checkpoint/std::list types -- no logic changed).
    // Picks the checkpoint whose removal leaves the remaining ones most evenly spaced by pos_max,
    // instead of always dropping the oldest. The first and last checkpoints are never candidates.
    static server_prompt_checkpoints::iterator evict_checkpoint_by_variance(server_prompt_checkpoints & ckpts) {
        auto it = ckpts.begin();
        if (ckpts.size() < 3) {
            return it;
        } else if (ckpts.size() == 3) {
            std::advance(it, 1);
            return it;
        }
        std::vector<int64_t> tokens;
        tokens.reserve(ckpts.size());
        for (const auto & ckpt : ckpts) {
            tokens.push_back((int64_t) ckpt->pos_max);
        }
        // Remove the checkpoint that makes the distribution most even after removal.
        // For each interior checkpoint, compute the variance of gaps that would result
        // if it were removed. Pick the one with the lowest variance (most uniform spacing).
        size_t best_idx = 1;
        const size_t n = tokens.size();
        const size_t start = 1;     // never remove the first
        const size_t end = n - 1;   // never remove the last
        const double max_pos = (double) tokens[n - 1];
        // To avoid a double for loop to compute variance, we only need the min product of two
        // consecutive gaps (see the fork source comment above for the derivation).
        double diff  = (double) (tokens[start]     - tokens[start - 1]);
        double diff2 = (double) (tokens[start + 1] - tokens[start]);
        double best_variance = diff * (diff2 / max_pos);
        for (size_t i = start + 1; i < end; i++) {
            diff  = (double) (tokens[i]     - tokens[i - 1]);
            diff2 = (double) (tokens[i + 1] - tokens[i]);
            const double variance = diff * (diff2 / max_pos);
            if (variance < best_variance) {
                best_variance = variance;
                best_idx = i;
            }
        }
        std::advance(it, best_idx);
        return it;
    }

    // Remove a checkpoint and delete its spill file, if it has one (--ctx-ckpt-spill-dir).
    // The checkpoint may be shared with cache entries / child slots (server_prompt_checkpoint_ptr): only the LAST owner deletes the file.
    static server_prompt_checkpoints::iterator drop_checkpoint(
            server_prompt_checkpoints & ckpts,
            server_prompt_checkpoints::iterator it) {
        server_ckpt_release(*it);
        return ckpts.erase(it);
    }

    // Spill file name for one checkpoint. The counter is process-wide so a name is never reused: slot prompts are
    // copied (server_prompt::clone), and a stale copy that still names a deleted or replaced file must fail to
    // reload (fall back to reprocessing) and never read another entry's data.
    static std::string checkpoint_spill_path(const std::string & dir, int slot_id) {
        static std::atomic<uint64_t> seq{0};
        return (std::filesystem::path(dir) / string_format("ckpt-s%d-%llu.bin", slot_id, (unsigned long long) ++seq)).string();
    }

    // With --ctx-ckpt-spill-dir: keep at most --ctx-ckpt-live-n checkpoints of this slot in memory and spill the rest.
    // The victim follows the eviction policy (variance unless fifo) but is always an in-memory entry and never the newest
    // one, which is the likeliest to be restored next. A failed write stops the loop and leaves the entry in memory.
    void enforce_checkpoint_memory(server_slot & slot) {
        const auto & dir  = params_base.ctx_checkpoint_spill_dir;
        const int    live = params_base.ctx_checkpoint_ram_live;
        // --ctx-ckpt-cram-soft: with no spill dir, cold checkpoints go to the CRAM software tier (compressed, still in RAM) instead
        const bool   use_cram = params_base.ctx_checkpoint_cram_soft && dir.empty();
        if ((dir.empty() && !use_cram) || live < 0) {
            return;
        }

        auto & ckpts = slot.prompt.checkpoints;
        auto n_resident = [&]() {
            int n = 0;
            for (const auto & c : ckpts) {
                if (!c->empty()) {
                    n++;
                }
            }
            return n;
        };

        while (n_resident() > live && ckpts.size() > 1) {
            auto victim = ckpts.begin();
            if (params_base.ctx_checkpoint_eviction != COMMON_CHECKPOINT_EVICTION_FIFO) {
                victim = evict_checkpoint_by_variance(ckpts);
            }
            if ((*victim)->empty() || victim == std::prev(ckpts.end())) {
                // already spilled, or the newest: take the oldest in-memory entry that is not the newest
                victim = ckpts.begin();
                while (victim != std::prev(ckpts.end()) && (*victim)->empty()) {
                    ++victim;
                }
                if (victim == std::prev(ckpts.end())) {
                    break;
                }
            }

            // With the NVMe tier on (--cache-disk-dir) the checkpoint goes there: one size cap, one SSD write budget and a checksum on
            // every read, none of which the per-checkpoint files have. Without the tier it is the file it always was.
            const bool use_tier = !use_cram && prompt_cache && prompt_cache->disk_enabled();
            auto & vck = **victim; // shared with cache entries / child slots: a tier move rewrites the one object (content unchanged)
            const size_t size_before = vck.size();
            const bool spilled_ok = use_cram ? vck.cram_compress()
                                  : use_tier ? prompt_cache->spill_slot_checkpoint(vck)
                                             : vck.spill(checkpoint_spill_path(dir, slot.id));
            if (!spilled_ok) {
                SLT_WRN(slot, "failed to spill context checkpoint (pos_min = %d, pos_max = %d) to %s; keeping it in memory\n",
                        vck.pos_min, vck.pos_max, use_cram ? "the CRAM soft tier" : use_tier ? "the disk tier" : dir.c_str());
                break;
            }
            SLT_TRC(slot, "spilled context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB) to %s\n",
                    vck.pos_min, vck.pos_max, vck.n_tokens, (float) size_before / 1024 / 1024,
                    use_cram ? "the CRAM soft tier" : use_tier ? "the disk tier" : vck.spill_path.c_str());
        }
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    // --ctx-ckpt-dominance: at slot release drop the checkpoints a deeper one on the path dominates (common/ckpt-dominance.h)
    void free_dominated_checkpoints(server_slot & slot) {
        auto & ckpts = slot.prompt.checkpoints;
        if (!params_base.ctx_checkpoint_dominance || ckpts.size() < 2) {
            return;
        }
        std::vector<common_ckpt_dom_entry> ent;
        ent.reserve(ckpts.size());
        for (const auto & c : ckpts) {
            ent.push_back({c->n_tokens, c->anchor, c->branch});
        }
        const auto idx = common_ckpt_dominated(ent);
        size_t pos = 0, next = 0;
        for (auto it = ckpts.begin(); it != ckpts.end(); ++pos) {
            if (next < idx.size() && idx[next] == pos) {
                SLT_TRC(slot, "freeing dominated context checkpoint (n_tokens = %" PRId64 ")\n", (*it)->n_tokens);
                it = drop_checkpoint(ckpts, it);
                ++next;
            } else {
                ++it;
            }
        }
    }

    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max,
                           bool anchor = false, bool branch = false) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            const auto & ck_it = **it;
            if (ck_it.id_task != id_task && last >= 0 && ck_it.n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        ck_it.pos_min, ck_it.pos_max, ck_it.n_tokens, (float) ck_it.size() / 1024 / 1024);

                it = drop_checkpoint(slot.prompt.checkpoints, it);
                continue;
            }

            last = ck_it.n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed. Default eviction policy is variance-based
            // (evict whichever checkpoint leaves the remaining ones most evenly spaced); FIFO (oldest
            // first, ckpts.begin()) is used only when explicitly requested via -ctx-ckpt-e fifo.
            auto victim = slot.prompt.checkpoints.begin();
            if (params_base.ctx_checkpoint_eviction != COMMON_CHECKPOINT_EVICTION_FIFO) {
                victim = evict_checkpoint_by_variance(slot.prompt.checkpoints);
            }
            const auto & cur = **victim;

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            drop_checkpoint(slot.prompt.checkpoints, victim);
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if ((*it)->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", (*it)->n_tokens);
                    it = drop_checkpoint(slot.prompt.checkpoints, it); // a spill file of the replaced one used to be left behind
                } else {
                    ++it;
                }
            }
        }

        auto ckpt_new = std::make_shared<common_prompt_checkpoint>();
        auto & cur = *ckpt_new;

        cur.id_task = id_task;
        cur.anchor  = anchor;
        cur.branch  = branch;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        slot.prompt.checkpoints.push_back(std::move(ckpt_new));

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

        enforce_checkpoint_memory(slot);
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    static bool ckpt_read_buf(std::ifstream & ifs, std::vector<uint8_t> & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    static void ckpt_write_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & ckpt : slot.prompt.checkpoints) {
            const auto & cur = *ckpt;
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        server_prompt_checkpoints checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            auto & cur = *ckpt;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(ckpt));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                checkpoints.pop_front();
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back()->data_dft.empty()) {
            const bool ok = checkpoints.back()->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur->clear_dft();
                }
            }
        }
        slot.prompt.checkpoints = std::move(checkpoints);
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(spec.get(), slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(spec.get(), slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    // POST /v1/decision: answer a finite JSON schema in one batched pass on the main loop thread.
    // Sequence ids: the LAST n_seq_decision ids of the context (after the slots, the cfg-guidance ids and the radix cache's ids).
    // Ported from thecodacus/llama.cpp @ b9244f893 (MIT): tools/server/server-context.cpp handle_decision.
    json handle_decision(const json & body) {
        if (params_base.n_seq_decision < 3) {
            throw std::invalid_argument("decisions are disabled: start the server with --decision-seqs N (N >= 3)");
        }
        // one decision per context; all contexts share the schema, the instructions and the cached prefix
        if (!body.contains("contexts") || !body.at("contexts").is_array() || body.at("contexts").empty() || body.at("contexts").size() > 256) {
            throw std::invalid_argument("\"contexts\" must be an array of 1-256 strings");
        }
        std::vector<std::string> contexts;
        for (const auto & c : body.at("contexts")) {
            if (!c.is_string() || c.get<std::string>().empty()) {
                throw std::invalid_argument("every entry of \"contexts\" must be a non-empty string");
            }
            contexts.push_back(c.get<std::string>());
        }
        if (!body.contains("schema")) {
            throw std::invalid_argument("\"schema\" must be provided");
        }
        if (!decision_engine) {
            const llama_seq_id base = (llama_seq_id) (params_base.n_parallel + params_base.n_cfg_guidance_slots +
                                                      (params_base.radix_cache ? params_base.n_radix_cache_slots : 0));
            decision_engine = std::make_unique<llama_decision::engine>(ctx_tgt, base, params_base.n_seq_decision);
        }
        const auto cs = llama_decision::compile_schema(body.at("schema"), body.value("instructions", std::string()));
        std::string shared;
        std::vector<std::string> dynamic;
        for (const auto & c : contexts) {
            auto [head, tail] = llama_decision::render_prompt(chat_params.tmpls.get(), chat_params.use_jinja, cs.system_text, c);
            if (dynamic.empty()) {
                shared = head;
            } else if (head != shared) {
                throw std::runtime_error("the chat template renders a different prefix per context");
            }
            dynamic.push_back(tail);
        }
        llama_decision::options opt;
        opt.mode        = body.value("mode", std::string("auto"));
        opt.tree_max    = (size_t) body.value("tree_max", 128);
        opt.allow_cache = body.value("cache_prompt", true);

        const auto b = decision_engine->decide_batch(shared, dynamic, cs.inputs, opt);

        size_t context_tokens = 0;
        for (const auto & r : b.items) {
            context_tokens += r.context_tokens;
        }
        SRV_INF("decision: %zu context(s), %zu field(s), %d scored row(s), %d round(s), prefix cache %s, %.1f ms\n",
                b.items.size(), cs.specs.size(), b.rows, b.rounds, b.cache_hit ? "hit" : "miss", b.prefill_ms + b.scoring_ms);
        json usage = json::object();
        usage["prompt_tokens"]  = (long long) (b.shared_tokens + context_tokens);
        usage["cached_tokens"]  = (long long) (b.cache_hit ? b.shared_tokens : 0);
        usage["context_tokens"] = (long long) context_tokens;
        usage["scored_rows"]    = b.rows;
        json timings = json::object();
        timings["prefill_ms"] = b.prefill_ms;
        timings["scoring_ms"] = b.scoring_ms;
        timings["total_ms"]   = b.prefill_ms + b.scoring_ms;
        timings["rounds"]     = b.rounds;
        timings["per_decision_ms"] = (b.prefill_ms + b.scoring_ms) / (double) b.items.size();

        json results = json::array();
        for (const auto & r : b.items) {
            json item = llama_decision::assemble(cs, r);
            json item_usage = json::object();
            item_usage["context_tokens"] = (long long) r.context_tokens;
            item_usage["scored_rows"]    = r.rows;
            item["usage"] = item_usage;
            results.push_back(item);
        }
        json out = json::object();
        out["object"]  = "decision";
        out["results"] = results;
        out["model"]   = model_name;
        out["created"] = (long long) std::time(nullptr);
        out["usage"]   = usage;
        out["timings"] = timings;
        return out;
    }

    // POST /v1/score: log-probabilities of N candidate continuations of ONE prompt, the prompt decoded once. The prompt goes
    // into an idle slot's sequence; each candidate is decoded from the prompt boundary (the previous candidate's cells are
    // removed first), so the prompt costs one pass and the work is prompt + sum(candidate tokens). Returns false when no slot is
    // idle (the caller defers the task). Candidate token 0 is scored by the logits of the last prompt row, saved before the first
    // candidate overwrites the output buffer; token k by the logits of the row before it.
    // SOURCE-READ: LocalAI patch 0001-add-server-task-type-score.patch (MIT, read this session): a new task type that decodes the
    // prefix once and each suffix from the prefix boundary. SOURCE-NUANCE: the fork's own slot/seq layout and logprob routine
    // (same log-softmax as extract_prompt_logprobs) are used; no code copied, written fresh against this server.
    bool handle_score(const json & body, json & out) {
        if (!body.is_object() || !body.contains("prompt") || !body.contains("candidates") || !body.at("candidates").is_array()
                || body.at("candidates").empty() || body.at("candidates").size() > 256) {
            throw std::invalid_argument("\"prompt\" (string or token array) and \"candidates\" (array of 1-256 strings or token arrays) are required");
        }
        server_slot * sc = nullptr;
        for (auto & slot : slots) {
            if (!slot.is_processing()) {
                sc = &slot;
                break;
            }
        }
        if (sc == nullptr) {
            return false;
        }
        const llama_tokens prompt = tokenize_mixed(vocab, body.at("prompt"), true, true);
        if (prompt.empty()) {
            throw std::invalid_argument("\"prompt\" tokenised to nothing");
        }
        std::vector<llama_tokens> cands;
        size_t n_suffix = 0;
        size_t longest  = 0;
        for (const auto & c : body.at("candidates")) {
            llama_tokens t = tokenize_mixed(vocab, c, false, true);
            if (t.empty()) {
                throw std::invalid_argument("every candidate must tokenise to at least one token");
            }
            n_suffix += t.size();
            longest   = std::max(longest, t.size());
            cands.push_back(std::move(t));
        }
        if (prompt.size() + longest > (size_t) sc->n_ctx) {
            throw std::invalid_argument(string_format("prompt (%zu) + longest candidate (%zu) tokens exceed the slot context (%d)",
                                                       prompt.size(), longest, (int) sc->n_ctx));
        }

        llama_memory_t mem     = llama_get_memory(ctx_tgt);
        const int32_t  n_batch = llama_n_batch(ctx_tgt);
        const int      n_vocab = llama_vocab_n_tokens(vocab);
        llama_batch    b       = llama_batch_init(n_batch, 0, 1);
        const llama_seq_id sid = sc->id;
        size_t n_decoded = 0;
        std::vector<float> last_row((size_t) n_vocab);
        bool ok = true;
        std::string fail;

        // decode toks at positions p0.. into sid, calling on_row(index, logits) for every row that has logits: all rows when
        // want_all, else only the last token
        auto decode_span = [&](const llama_tokens & toks, size_t p0, bool want_all, auto && on_row) {
            for (size_t i = 0; ok && i < toks.size(); ) {
                common_batch_clear(b);
                const size_t first = i;
                for (int32_t k = 0; k < n_batch && i < toks.size(); ++k, ++i) {
                    const bool out_row = want_all || i + 1 == toks.size();
                    common_batch_add(b, toks[i], (llama_pos) (p0 + i), { sid }, out_row);
                }
                if (llama_decode(ctx_tgt, b) != 0) {
                    ok   = false;
                    fail = "llama_decode failed while scoring";
                    return;
                }
                n_decoded += i - first;
                for (size_t j = first; j < i; ++j) {
                    if (want_all || j + 1 == toks.size()) {
                        on_row(j, llama_get_logits_ith(ctx_tgt, (int32_t) (j - first)));
                    }
                }
            }
        };
        auto log_prob = [&](const float * lg, llama_token tok) {
            float mx = lg[0];
            for (int t = 1; t < n_vocab; ++t) { mx = std::max(mx, lg[t]); }
            double sum = 0.0;
            for (int t = 0; t < n_vocab; ++t) { sum += std::exp((double) (lg[t] - mx)); }
            return (double) (lg[tok] - mx) - std::log(sum);
        };

        // the slot's cached prompt is replaced: its sequence and checkpoints are dropped
        llama_memory_seq_rm(mem, sid, -1, -1);
        sc->prompt.clear();
        const int64_t t0 = ggml_time_us();
        decode_span(prompt, 0, false, [&](size_t, const float * lg) {
            if (lg != nullptr) {
                std::copy(lg, lg + n_vocab, last_row.begin());
            } else {
                ok   = false;
                fail = "no logits for the last prompt row";
            }
        });

        json results = json::array();
        for (size_t ci = 0; ok && ci < cands.size(); ++ci) {
            const auto & c = cands[ci];
            if (!llama_memory_seq_rm(mem, sid, (llama_pos) prompt.size(), -1)) {
                // memory that rejects a partial removal (recurrent / hybrid): rebuild the prompt; the decoded count shows it
                llama_memory_seq_rm(mem, sid, -1, -1);
                decode_span(prompt, 0, false, [&](size_t, const float *) {});
            }
            std::vector<double> lps;
            lps.reserve(c.size());
            lps.push_back(log_prob(last_row.data(), c[0]));
            decode_span(c, prompt.size(), true, [&](size_t j, const float * lg) {
                if (lg == nullptr) {
                    ok   = false;
                    fail = "no logits for a candidate row";
                    return;
                }
                if (j + 1 < c.size()) {
                    lps.push_back(log_prob(lg, c[j + 1]));
                }
            });
            double total = 0.0;
            json lpj = json::array();
            for (double v : lps) {
                total += v;
                lpj.push_back(v);
            }
            json item = json::object();
            item["index"]          = (long long) ci;
            item["n_tokens"]       = (long long) c.size();
            item["token_logprobs"] = lpj;
            item["logprob_sum"]    = total;
            results.push_back(item);
        }
        llama_batch_free(b);
        llama_memory_seq_rm(mem, sid, -1, -1);
        sc->prompt.clear();
        if (!ok) {
            throw std::runtime_error(fail);
        }
        const double ms = (ggml_time_us() - t0) / 1000.0;
        SRV_INF("score: prompt %zu token(s), %zu candidate(s), %zu candidate token(s), %zu decoded, %.1f ms\n",
                prompt.size(), cands.size(), n_suffix, n_decoded, ms);
        json usage = json::object();
        usage["prompt_tokens"]             = (long long) prompt.size();
        usage["candidate_tokens"]          = (long long) n_suffix;
        usage["n_prompt_tokens_processed"] = (long long) n_decoded;
        out = json::object();
        out["object"]  = "score";
        out["results"] = results;
        out["model"]   = model_name;
        out["usage"]   = usage;
        out["timings"] = json{{"total_ms", ms}};
        return true;
    }

    // POST /v1/logits: prefill the prompt on an idle slot's sequence and return the raw LM-head logits. A row is the top-N
    // [token id, logit] pairs by descending logit (ties: the lower id first), or with top_n 0 the whole vocabulary indexed by token id.
    // last_only keeps the final position's row only. Returns false when no slot is idle (the caller defers the task).
    // SOURCE-READ: engines/nexa-sdk cli/server/handler/logits.go:19-133 (BSD-3-Clause): the request fields, the top_n default of 20
    // and the 400 cases. Behaviour only, written fresh against this server's slots; no code copied.
    bool handle_logits(const json & body, json & out) {
        if (!body.is_object() || (!body.contains("input_ids") && !body.contains("content"))) {
            throw std::invalid_argument("\"input_ids\" (token array) or \"content\" (string) is required");
        }
        const bool last_only = json_value(body, "last_only", false);
        int top_n = 20;
        if (body.contains("top_n") && !body.at("top_n").is_null()) {
            if (!body.at("top_n").is_number_integer()) {
                throw std::invalid_argument("\"top_n\" must be an integer");
            }
            top_n = body.at("top_n").get<int>();
            if (top_n < 0) {
                throw std::invalid_argument("\"top_n\" must be >= 0");
            }
        }
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_tokens prompt;
        if (body.contains("input_ids")) {
            const auto & a = body.at("input_ids");
            if (!a.is_array()) {
                throw std::invalid_argument("\"input_ids\" must be an array of token ids");
            }
            for (const auto & v : a) {
                if (!v.is_number_integer() || v.get<long long>() < 0 || v.get<long long>() >= n_vocab) {
                    throw std::invalid_argument("\"input_ids\" holds an id outside the vocabulary");
                }
                prompt.push_back((llama_token) v.get<long long>());
            }
        } else {
            const json & c = body.at("content");
            if (c.is_string() && c.get<std::string>().empty()) {
                throw std::invalid_argument("the input is empty");
            }
            prompt = tokenize_mixed(vocab, c, true, true);
        }
        if (prompt.empty()) {
            throw std::invalid_argument("the input is empty");
        }
        server_slot * sc = nullptr;
        for (auto & slot : slots) {
            if (!slot.is_processing()) {
                sc = &slot;
                break;
            }
        }
        if (sc == nullptr) {
            return false;
        }
        if (prompt.size() > (size_t) sc->n_ctx) {
            throw std::invalid_argument(string_format("input (%zu tokens) exceeds the slot context (%d)", prompt.size(), (int) sc->n_ctx));
        }
        const int width = (top_n == 0 || top_n > n_vocab) ? n_vocab : top_n;
        const bool full = top_n == 0;

        llama_memory_t mem     = llama_get_memory(ctx_tgt);
        const int32_t  n_batch = llama_n_batch(ctx_tgt);
        llama_batch    b       = llama_batch_init(n_batch, 0, 1);
        const llama_seq_id sid = sc->id;
        bool ok = true;
        std::string fail;
        json rows = json::array();
        std::vector<int> order((size_t) n_vocab);

        llama_memory_seq_rm(mem, sid, -1, -1);
        sc->prompt.clear();
        const int64_t t0 = ggml_time_us();
        for (size_t i = 0; ok && i < prompt.size(); ) {
            common_batch_clear(b);
            const size_t first = i;
            for (int32_t k = 0; k < n_batch && i < prompt.size(); ++k, ++i) {
                common_batch_add(b, prompt[i], (llama_pos) i, { sid }, !last_only || i + 1 == prompt.size());
            }
            if (llama_decode(ctx_tgt, b) != 0) {
                ok   = false;
                fail = "llama_decode failed while reading logits";
                break;
            }
            for (size_t j = first; j < i; ++j) {
                if (last_only && j + 1 != prompt.size()) {
                    continue;
                }
                const float * lg = llama_get_logits_ith(ctx_tgt, (int32_t) (j - first));
                if (lg == nullptr) {
                    ok   = false;
                    fail = "no logits for a prompt row";
                    break;
                }
                json row = json::object();
                if (full) {
                    row["logits"] = std::vector<float>(lg, lg + n_vocab);
                } else {
                    for (int t = 0; t < n_vocab; ++t) { order[(size_t) t] = t; }
                    std::partial_sort(order.begin(), order.begin() + width, order.end(), [lg](int x, int y) {
                        return lg[x] > lg[y] || (lg[x] == lg[y] && x < y);
                    });
                    std::vector<int>   ids(order.begin(), order.begin() + width);
                    std::vector<float> vals;
                    vals.reserve((size_t) width);
                    for (int t : ids) { vals.push_back(lg[t]); }
                    row["token_ids"] = std::move(ids);
                    row["logits"]    = std::move(vals);
                }
                rows.push_back(std::move(row));
            }
        }
        llama_batch_free(b);
        llama_memory_seq_rm(mem, sid, -1, -1);
        sc->prompt.clear();
        if (!ok) {
            throw std::runtime_error(fail);
        }
        const double ms = (ggml_time_us() - t0) / 1000.0;
        SRV_INF("logits: %zu token(s), %zu row(s), top_n %d, %.1f ms\n", prompt.size(), rows.size(), top_n, ms);
        out = json::object();
        out["model"]      = model_name;
        out["n_rows"]     = (long long) rows.size();
        out["vocab_size"] = n_vocab;
        out["top_n"]      = top_n;
        out["rows"]       = std::move(rows);
        return true;
    }

    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET
                && task.type != SERVER_TASK_TYPE_SLOT_PROMPTS && task.type != SERVER_TASK_TYPE_LIST_SAVED_PROMPTS) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_DECISION:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    // end-to-end deadline (t_deadline_ms): a request whose budget ran out before it reached a slot is answered now
                    if (task.t_deadline_us > 0 && ggml_time_us() >= task.t_deadline_us) {
                        send_error(task, "request deadline (t_deadline_ms) exceeded before the request reached a slot; retry later", ERROR_TYPE_UNAVAILABLE);
                        break;
                    }

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (kv_admission_defers(task, *slot)) {
                        // the shared pool would overflow before the in-flight requests finish; a finishing slot
                        // re-offers deferred tasks (callback_on_release), so this is re-checked as cells free up
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (kv_reserve_defers(task)) {
                        // starting this request would eat into the pool share kept back for the running ones; re-checked when a slot is released
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                const bool idle_saved = slot.prompt_save(*prompt_cache);
                                if (idle_saved) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                // BeeLlama-style (Anbeeld/beellama.cpp, MIT, rewritten): free the idle slot's unified-KV cells only when its
                                // prompt is held by the cache (just saved, or an entry already contains it). A failed save (over the cache
                                // limit, no room) used to clear it anyway and the prompt was lost; now it stays resident.
                                if (params_base.kv_unified) {
                                    if (idle_saved || slot.prompt_is_cached(*prompt_cache)) {
                                        // [TAG_IDLE_SLOT_CLEAR]
                                        slot.prompt_clear();
                                    } else {
                                        SLT_WRN(slot, "%s", "idle slot prompt not saved to the cache - keeping it resident\n");
                                    }
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %d on unknown completion id=%s, no live slot\n",
                                (int) task.params.control_action, task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == SERVER_CONTROL_ACTION_REASONING_END) {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else if (task.params.control_action == SERVER_CONTROL_ACTION_CANCEL) {
                        // cancel by completion id: same release the id_target cancel does, found through the cmpl id
                        slot->release();
                        res->success = true;
                    } else if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END) {
                        if (!slot->audio_stream) {
                            res->success = false;
                            res->message = "not a realtime transcription slot";
                        } else {
                            auto * stream = slot->audio_stream.get();
                            if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND) {
                                mtmd_audio_stream_push(stream, task.params.control_audio.data(), task.params.control_audio.size());
                            } else {
                                const int32_t ret = mtmd_audio_stream_flush(stream, task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END);
                                // a non-final commit completes its item once the rows of its audio are decoded
                                if (ret == 0 && task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT) {
                                    slot->audio_commit_rows.push_back(mtmd_audio_stream_get_n_rows_flushed(stream));
                                }
                            }
                            slot->audio_wait = false;
                            res->success = true;
                        }
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_PARALLEL_DECISION:
                {
                    try {
                        auto res  = std::make_unique<server_task_result_parallel_decision>();
                        res->id   = task.id;
                        res->data = handle_decision(task.decision_request);
                        queue_results.send(std::move(res));
                    } catch (const std::invalid_argument & e) {
                        send_error(task, e.what(), ERROR_TYPE_INVALID_REQUEST);
                    } catch (const common_json_error & e) {
                        send_error(task, std::string("invalid decision request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                    } catch (const std::exception & e) {
                        send_error(task, e.what(), ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_SCORE:
                {
                    try {
                        json data;
                        if (!handle_score(task.score_request, data)) {
                            SRV_DBG("score: every slot is processing, defer, id_task = %d\n", task.id);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        auto res  = std::make_unique<server_task_result_parallel_decision>();
                        res->id   = task.id;
                        res->data = std::move(data);
                        queue_results.send(std::move(res));
                    } catch (const std::invalid_argument & e) {
                        send_error(task, e.what(), ERROR_TYPE_INVALID_REQUEST);
                    } catch (const common_json_error & e) {
                        send_error(task, std::string("invalid score request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                    } catch (const std::exception & e) {
                        send_error(task, e.what(), ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_LOGITS:
                {
                    try {
                        json data;
                        if (!handle_logits(task.score_request, data)) {
                            SRV_DBG("logits: every slot is processing, defer, id_task = %d\n", task.id);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        auto res  = std::make_unique<server_task_result_parallel_decision>();
                        res->id   = task.id;
                        res->data = std::move(data);
                        queue_results.send(std::move(res));
                    } catch (const std::invalid_argument & e) {
                        send_error(task, e.what(), ERROR_TYPE_INVALID_REQUEST);
                    } catch (const common_json_error & e) {
                        send_error(task, std::string("invalid logits request: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                    } catch (const std::exception & e) {
                        send_error(task, e.what(), ERROR_TYPE_SERVER);
                    }
                } break;
            case SERVER_TASK_TYPE_HOTSWAP_RELOAD:
                {
                    // llama_reload_changed_tensors rewrites weights in place and resets the cached graphs: never while a
                    // slot is mid-generation (the unbuilt examples/server/server.cpp:806-809 skipped then; this runs on
                    // the main loop, so "no slot processing" holds for the whole reload, not just at a snapshot)
                    int n_busy = 0;
                    for (const server_slot & slot : slots) {
                        n_busy += slot.is_processing() ? 1 : 0;
                    }
                    if (n_busy > 0 && !task.hotswap_if_idle) {
                        SRV_DBG("hotswap: %d slot(s) processing, defer reload, id_task = %d\n", n_busy, task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    auto res = std::make_unique<server_task_result_hotswap>();
                    res->id = task.id;
                    if (n_busy > 0) {
                        SRV_INF("hotswap: skipping tensor reload, %d slot(s) processing\n", n_busy);
                        res->skipped_busy = true;
                    } else if (llama_reload_changed_tensors(ctx_tgt)) {
                        // every KV entry and cached prompt was computed with the previous weights
                        res->reloaded = true;
                        for (server_slot & slot : slots) {
                            res->n_slots_cleared += slot.prompt.n_tokens() > 0 ? 1 : 0;
                            slot.prompt_clear();
                        }
                        if (prompt_cache) {
                            res->n_prompt_cache_dropped = prompt_cache->clear_all();
                        }
                        if (radix_cache) {
                            radix_cache->clear();
                            res->radix_cleared = true;
                        }
                        SRV_INF("hotswap: tensors reloaded; KV cache and cached prompts cleared (%d slot(s), %zu prompt-cache entr%s%s)\n",
                                res->n_slots_cleared, res->n_prompt_cache_dropped, res->n_prompt_cache_dropped == 1 ? "y" : "ies",
                                res->radix_cleared ? ", radix tree" : "");
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    const server_slot_gauges g = get_slot_gauges();
                    SRV_DBG("n_processing_slots = %d\n", g.n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = g.n_processing_slots;
                    res->n_kv_tokens_active   = g.n_kv_tokens_active;
                    res->n_kv_tokens_held     = g.n_kv_tokens_held;
                    res->n_kv_tokens_capacity = g.n_kv_tokens_capacity;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;
                    if (prompt_cache) {
                        res->disk = prompt_cache->disk_stats();
                    }
                    if (radix_cache) {
                        const auto & rs = radix_cache->get_stats();
                        res->has_radix         = true;
                        res->radix_hits        = rs.hits;
                        res->radix_misses      = rs.misses;
                        res->radix_hit_tokens  = rs.hit_tokens;
                        res->radix_evictions   = rs.evictions;
                        res->radix_nodes       = (uint64_t) radix_cache->n_nodes();
                    }
                    if (mctx != nullptr && mtmd_support_vision(mctx)) {
                        mtmd_embd_cache_info mi{};
                        mtmd_embd_cache_get_info(mctx, &mi);
                        res->has_mmproj        = true;
                        res->mmproj_encodes    = mi.encodes;
                        res->mmproj_hits       = mi.hits;
                        res->mmproj_misses     = mi.misses;
                        res->mmproj_evictions  = mi.evictions;
                        res->mmproj_not_stored = mi.not_stored;
                        res->mmproj_bytes      = mi.bytes;
                        res->mmproj_entries    = mi.entries;
                        res->mmproj_budget     = mi.budget;
                    }

                    if (task.metrics_memory) {
                        res->n_layer = llama_model_n_layer(llama_get_model(ctx_tgt));
                        res->memory  = get_memory_data();
                    }

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_PROMPTS:
                {
                    // ported from examples/server/server.cpp:1845-1855 @ 35f0cac75 (list_slot_prompts) --
                    // row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, G2.
                    // lighter-weight than SERVER_TASK_TYPE_SLOT_GET: no --slots gate, no task/params
                    // dump, just the tokens currently held in each slot's KV cache (server_prompt).
                    json slots_data = json::array();

                    for (server_slot & slot : slots) {
                        slots_data.push_back({
                            {"slot_id",     slot.id},
                            {"token_count", slot.prompt.tokens.size()},
                            {"prompt",      slot.prompt.tokens.detokenize(slot.ctx_tgt, true)},
                        });
                    }

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id         = task.id;
                    res->slots_data = std::move(slots_data);

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_LIST_SAVED_PROMPTS:
                {
                    // ported from examples/server/server.cpp:1786-1843 @ 35f0cac75 (list_saved_prompts) --
                    // row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, G2.
                    // Runs on the main loop (not the HTTP thread) because decoding a saved file's
                    // packed server_tokens blob into preview text needs detokenize(ctx_tgt), and
                    // ctx_tgt is private to this class outside thread-safe tokenizer calls.
                    namespace fs = std::filesystem;
                    json response = json::array();

                    try {
                        for (const auto & entry : fs::directory_iterator(params_base.slot_save_path)) {
                            if (!entry.is_regular_file() || entry.file_size() < 12) {
                                continue;
                            }

                            std::ifstream file(entry.path(), std::ios::binary);
                            if (!file) {
                                continue;
                            }

                            // NOTE: unlike the pre-merge fork (a flat llama_token array right after
                            // the header), this split server's /slots save writes a packed
                            // server_tokens blob here (it may encode multimodal chunks too) -- see
                            // SERVER_TASK_TYPE_SLOT_SAVE above. n_packed is a WORD count of that
                            // packed blob, not a literal token count, and must go through
                            // server_tokens::deserialize() to recover the real token count and text.
                            uint32_t magic = 0, version = 0, n_packed = 0;
                            file.read(reinterpret_cast<char *>(&magic),    sizeof(magic));
                            file.read(reinterpret_cast<char *>(&version),  sizeof(version));
                            file.read(reinterpret_cast<char *>(&n_packed), sizeof(n_packed));

                            if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION ||
                                    entry.file_size() < (12 + ((uint64_t) n_packed * sizeof(llama_token)))) {
                                continue;
                            }

                            llama_tokens packed(n_packed);
                            file.read(reinterpret_cast<char *>(packed.data()), packed.size() * sizeof(llama_token));

                            size_t token_count = 0;
                            std::string prompt_preview;
                            try {
                                server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);
                                token_count    = restored.size();
                                prompt_preview = restored.detokenize(ctx_tgt, true);
                            } catch (const std::exception &) {
                                // corrupt/foreign file, or a media chunk this build can't decode -- skip.
                                continue;
                            }

                            const auto ftime = fs::last_write_time(entry.path());
                            const auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                                ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                            const std::time_t c_time = std::chrono::system_clock::to_time_t(system_time);
                            std::tm tm_struct{};
#if defined(_WIN32)
                            localtime_s(&tm_struct, &c_time);
#else
                            localtime_r(&c_time, &tm_struct);
#endif
                            std::ostringstream oss;
                            oss << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S");

                            response.push_back({
                                {"filename",    entry.path().filename().string()},
                                {"filesize",    entry.file_size()},
                                {"mtime",       oss.str()},
                                {"token_count", token_count},
                                {"prompt",      prompt_preview},
                            });
                        }
                    } catch (const std::exception & e) {
                        send_error(task, e.what(), ERROR_TYPE_SERVER);
                        break;
                    }

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id         = task.id;
                    res->slots_data = std::move(response);

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        {   // a slot file saved under another checkpoint/quant of the same arch is garbage with no error: refuse, naming the field
                            char why[512] = {0};
                            if (llama_state_seq_file_identity_check(ctx_tgt, filepath.c_str(), why, sizeof(why)) == 1) {
                                throw std::runtime_error(why);
                            }
                        }
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    // ...and every host-RAM prompt-cache entry this slot's content could have been published as
                    // (server_prompt_cache::remove_prefixes_of explains why: publish-on-prefill + non-consuming
                    // loads otherwise hand the erased prompt straight back to the next request).
                    // --radix-cache nodes are purged just below (nodes shared with other prompts stay).
                    if (prompt_cache && n_erased > 0) {
                        const size_t n_purged = prompt_cache->remove_prefixes_of(slot->prompt.tokens);
                        SRV_INF("slot erase: id %d, %zu tokens, %zu prompt-cache entr%s purged\n", id_slot, n_erased,
                                n_purged, n_purged == 1 ? "y" : "ies");
                    }

                    if (radix_cache && n_erased > 0) {
                        const size_t n_nodes_purged = radix_cache->purge_path_of(slot->prompt.tokens.get_tokens(), slot->prompt.lora_fp);
                        SRV_INF("slot erase: id %d, %zu radix node(s) purged\n", id_slot, n_nodes_purged);
                    }

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
            // Ported from the pre-merge fork (row
            // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G1).
            // SOURCE-READ: examples/server/server-context.cpp:3125-3261 @ 35f0cac75 (local worktree
            // copy) -- ctx/model renamed ctx_tgt/model_tgt, llama_control_vector_load_info/
            // llama_control_vector_load renamed common_control_vector_* by the upstream merge, the
            // old bare-json result shape ported into server_task_result_control_vector /
            // server_task_result_control_vectors_list (server-task.h) to match this split codebase's
            // typed-result convention (mirrors SERVER_TASK_TYPE_GET_LORA / SET_LORA just above).
            case SERVER_TASK_TYPE_GET_CONTROL_VECTORS:
                {
                    auto res = std::make_unique<server_task_result_control_vectors_list>();
                    res->id = task.id;
                    for (size_t i = 0; i < control_vectors.size(); ++i) {
                        auto & cv = control_vectors[i];
                        res->data.push_back({
                            {"id", i},
                            {"path", cv.path},
                            {"scale", cv.scale},
                            {"layer_start", cv.layer_start},
                            {"layer_end", cv.layer_end},
                            {"applied", cv.applied},
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR:
                {
                    const std::string & path  = task.control_vector_load.path;
                    float   scale             = task.control_vector_load.scale;
                    int32_t layer_start       = task.control_vector_load.layer_start;
                    int32_t layer_end         = task.control_vector_load.layer_end == -1
                                                    ? llama_n_layer(model_tgt)
                                                    : task.control_vector_load.layer_end;

                    // check if already loaded
                    int cv_id = -1;
                    for (size_t i = 0; i < control_vectors.size(); i++) {
                        if (control_vectors[i].path == path) {
                            control_vectors[i].scale       = scale;
                            control_vectors[i].layer_start = layer_start;
                            control_vectors[i].layer_end   = layer_end;
                            cv_id = (int) i;
                            break;
                        }
                    }

                    if (cv_id == -1) {
                        control_vector_container new_cv;
                        new_cv.path        = path;
                        new_cv.scale       = scale;
                        new_cv.layer_start = layer_start;
                        new_cv.layer_end   = layer_end;
                        new_cv.applied     = false;

                        // load the control vector data (strength 1.0: scaling happens at apply time)
                        common_control_vector_load_info load_info;
                        load_info.fname    = path;
                        load_info.strength = 1.0f;

                        std::vector<common_control_vector_load_info> load_infos = { load_info };
                        new_cv.data = common_control_vector_load(load_infos, llama_model_n_layer(model_tgt));

                        if (new_cv.data.n_embd == -1) {
                            auto res = std::make_unique<server_task_result_control_vector>();
                            res->id = task.id;
                            res->success = false;
                            res->error = "Failed to load control vector from " + path;
                            queue_results.send(std::move(res));
                            break;
                        }

                        // validate dimension to prevent heap corruption
                        if (new_cv.data.n_embd != llama_model_n_embd(model_tgt)) {
                            auto res = std::make_unique<server_task_result_control_vector>();
                            res->id = task.id;
                            res->success = false;
                            res->error = "Vector dimension mismatch";
                            queue_results.send(std::move(res));
                            break;
                        }

                        control_vectors.push_back(new_cv);
                        cv_id = (int) control_vectors.size() - 1;
                    }

                    // auto-apply control vectors after loading
                    if (!apply_control_vectors_internal()) {
                        auto res = std::make_unique<server_task_result_control_vector>();
                        res->id = task.id;
                        res->success = false;
                        res->error = "Failed to apply control vectors";
                        queue_results.send(std::move(res));
                        break;
                    }

                    auto res = std::make_unique<server_task_result_control_vector>();
                    res->id        = task.id;
                    res->success   = true;
                    res->has_cv_id = true;
                    res->cv_id     = cv_id;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR:
                {
                    int id = task.control_vector_unload_id;

                    if (id < 0 || id >= (int) control_vectors.size()) {
                        auto res = std::make_unique<server_task_result_control_vector>();
                        res->id = task.id;
                        res->success = false;
                        res->error = "Invalid control vector ID";
                        queue_results.send(std::move(res));
                        break;
                    }

                    control_vectors.erase(control_vectors.begin() + id);

                    if (!apply_control_vectors_internal()) {
                        auto res = std::make_unique<server_task_result_control_vector>();
                        res->id = task.id;
                        res->success = false;
                        res->error = "Failed to apply control vectors";
                        queue_results.send(std::move(res));
                        break;
                    }

                    auto res = std::make_unique<server_task_result_control_vector>();
                    res->id = task.id;
                    res->success = true;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_CONTROL_VECTOR:
                {
                    int max_idx = (int) control_vectors.size();
                    bool bad_id = false;
                    // reset all scales first, matching old semantics ("apply" = full replace, not merge)
                    for (auto & cv : control_vectors) {
                        cv.scale = 0.0f;
                    }
                    for (const auto & u : task.control_vector_apply) {
                        if (u.id < 0 || u.id >= max_idx) {
                            bad_id = true;
                            break;
                        }
                        control_vectors[u.id].scale = u.scale;
                        if (u.has_layer_start) {
                            control_vectors[u.id].layer_start = u.layer_start;
                        }
                        if (u.has_layer_end) {
                            control_vectors[u.id].layer_end = u.layer_end;
                        }
                    }
                    if (bad_id) {
                        auto res = std::make_unique<server_task_result_control_vector>();
                        res->id = task.id;
                        res->success = false;
                        res->error = "Invalid control vector id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (!apply_control_vectors_internal()) {
                        auto res = std::make_unique<server_task_result_control_vector>();
                        res->id = task.id;
                        res->success = false;
                        res->error = "Failed to apply control vectors";
                        queue_results.send(std::move(res));
                        break;
                    }

                    auto res = std::make_unique<server_task_result_control_vector>();
                    res->id = task.id;
                    res->success = true;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    // Peak-KV admission (--kv-admission, --kv-unified only). LightLLM's rule, SOURCE-READ at
    // lightllm/server/router/req_queue/chunked_prefill/impl.py:20-32 (Apache-2.0; design only, no code copied):
    // each request holds `held` cells now and will write `left` more. Sorted by `left` DESCENDING, request i is the
    // (i+1)-th still running when the i-th-longest finishes, so the pool peaks at max_i(left_i*(i+1) + cumsum(held)_i)
    // -- finished requests free their cells, which is why this is tighter than sum(held+left). Returns true when
    // admitting `task` would push that peak past the pool AND someone is in flight to finish first (never defers
    // onto an empty server, so nothing can wait forever). A request with no n_predict is projected to its slot's full
    // context. Idle slots' cached cells are not counted: try_clear_idle_slots() purges them under pressure.
    //
    // --kv-admission-decode-tokens N (row S69-ENGREV-TRNRT-ADAPT-KV-RESERVE-ADMISSION-2026-11-13-01): cap every request's
    // projected `left` at N (times its guidance multiplier). SOURCE-READ: RunanywhereAI/trnrt@ef581f5 scheduler.py:440-496
    // and config.py:69-72 (Apache-2.0; design only, no code copied) -- a waiting request is admitted when free + evictable
    // pages cover its prompt plus at most `admission_decode_tokens` (256 there) of decode, on top of what running requests
    // still need. Without the cap a request with no n_predict is projected to the whole slot context, so such requests run
    // one at a time. With it they run side by side; one that does decode past N can exhaust the pool, and
    // preempt_largest_slot() then handles that exactly as it does without admission. 0 = no cap (the rule above, unchanged).
    bool kv_admission_defers(const server_task & task, const server_slot & slot) {
        if (!params_base.kv_unified || !params_base.kv_admission) {
            return false;
        }
        const int64_t decode_cap = params_base.kv_admission_decode_tokens;
        auto capped = [decode_cap](int64_t left) { return decode_cap > 0 ? std::min<int64_t>(left, decode_cap) : left; };
        std::vector<std::pair<int64_t, int64_t>> reqs; // (held, left)
        for (const auto & s : slots) {
            if (!s.is_processing() || !s.task) {
                continue;
            }
            const int64_t mult = s.id_cfg >= 0 ? 2 : 1; // a guidance seq grows in step with the main one
            const int64_t held = std::max<int64_t>(s.prompt.n_tokens(), (int64_t) s.task->n_tokens()) + s.n_cfg_past;
            const int64_t rem  = s.n_remaining();
            const int64_t left = mult * capped(rem >= 0 ? rem : std::max<int64_t>(0, s.n_ctx - held));
            reqs.emplace_back(held, left);
        }
        if (reqs.empty()) {
            return false;
        }
        const int64_t n_new    = 1 + (int64_t) task.child_tasks.size();
        const int64_t held_new = (int64_t) task.n_tokens();
        const int32_t n_pred   = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;
        const int64_t left_new = capped(n_pred > 0 ? n_pred : std::max<int64_t>(0, slot.n_ctx - held_new));
        for (int64_t k = 0; k < n_new; ++k) {
            reqs.emplace_back(held_new, left_new);
        }
        std::sort(reqs.begin(), reqs.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
        int64_t cum = 0;
        int64_t peak = 0;
        for (size_t i = 0; i < reqs.size(); ++i) {
            cum += reqs[i].first;
            peak = std::max<int64_t>(peak, reqs[i].second * (int64_t) (i + 1) + cum);
        }
        const int64_t pool = llama_n_ctx(ctx_tgt);
        if (peak <= pool) {
            return false;
        }
        SRV_DBG("kv-admission: projected peak %lld > pool %lld with %zu in flight, defer task %d\n",
                (long long) peak, (long long) pool, reqs.size() - (size_t) n_new, task.id);
        return true;
    }

    // --shed-max-latency-ms: the online fit of service time against request size (main thread only)
    server_shed_fit shed_fit;

    // --kv-reserve-ratio: hold back a percentage of the shared KV pool from NEW requests, so the requests already running keep room to
    // grow. SOURCE-READ: rtp-llm@95ce33f4 rtp_llm/cpp/cache/CoordinatorCacheManager.cc:74-92 (reserve = ratio * free / 100, taken once
    // at start-up, when the whole pool is free, so it is ratio% of the pool) and :1881-1934 (a new request needs additional blocks +
    // reserve free; only a first allocation pays the reserve, never the growth of a running one). Apache-2.0, no code copied.
    // Differences: the pool is counted in cells held by the running slots (the same accounting kv_admission_defers uses); and rtp-llm
    // answers "permanently rejected" when a request can never fit beside the reserve, which would starve here, so on an idle server
    // the request is always admitted. Row S69-RTP-LLM-PICK-KV-RESERVE-RATIO-2026-10-15-01.
    bool kv_reserve_defers(const server_task & task) {
        if (!params_base.kv_unified || params_base.kv_reserve_ratio <= 0) {
            return false;
        }
        int64_t held  = 0;
        int     n_run = 0;
        for (const auto & s : slots) {
            if (!s.is_processing() || !s.task) {
                continue;
            }
            held += std::max<int64_t>(s.prompt.n_tokens(), (int64_t) s.task->n_tokens()) + s.n_cfg_past;
            n_run++;
        }
        if (n_run == 0) {
            return false; // never defer onto an idle server: nothing would ever free the cells
        }
        const int64_t pool    = llama_n_ctx(ctx_tgt);
        const int64_t reserve = pool * params_base.kv_reserve_ratio / 100;
        const int64_t need    = (1 + (int64_t) task.child_tasks.size()) * (int64_t) task.n_tokens();
        if (pool - held - need >= reserve) {
            return false;
        }
        SRV_DBG("kv-reserve: free %lld - prompt %lld < reserve %lld (%d%% of %lld), %d in flight, defer task %d\n",
                (long long) (pool - held), (long long) need, (long long) reserve, (int) params_base.kv_reserve_ratio, (long long) pool, n_run, task.id);
        return true;
    }

    // KV-exhaustion preemption under --kv-unified (row S69-SERVER-KV-UNIFIED-EXHAUSTION-FAILS-ALL-SLOTS-2026-09-27-01).
    // With one shared pool, a decode that cannot find a single free cell used to error EVERY in-flight slot. Instead,
    // evict the one slot holding the most cells (main seq + its CFG guidance seq): answer it 503 so the client can
    // retry, free its cells, strip its not-yet-decoded entries from the pending batch and remap the other slots'
    // batch indices. Returns false -- leaving the old behaviour -- when there is no one else to save (fewer than 2
    // slots in flight), in per-slot mode (each slot owns its slice; one slot's overflow is not another's problem),
    // or for an embedding batch (not compacted by remove_seqs_from).
    // Design: vLLM also preempts under KV pressure instead of failing the batch, but picks the most recently
    // scheduled request (vllm/v1/core/sched/scheduler.py:759-767, running[-1] or max (priority, arrival)) and
    // re-queues it for recompute (_preempt_request, :1539). Here the victim is, by default, the LARGEST, as upstream's TODO
    // above asks -- one eviction frees the most cells -- and it is failed-retryable (503) rather than recomputed unless
    // --kv-preempt-requeue lets a non-streaming one be recomputed (a streamed victim has already sent tokens a recompute
    // could not take back). --kv-preempt-victim least-progress|newest changes WHO is chosen among equal priorities
    // (server-slot-victim.h; measured in S69_ENGINE_VLLM_W2_2026-10-27.md, wave 3).
    // SOURCE-READ: scheduler.py:759-801,1539; design only, no code copied (Apache-2.0).
    bool in_preempt_release = false; // true only while preempt_largest_slot() releases its victim (--kv-preempt-requeue)

    // true when a preemption of this task can end in a recompute instead of a 503 (--kv-preempt-requeue): a request that has sent nothing
    // to the client (non-streaming completion / infill, not parent / child / CLI) and has requeues left
    bool preempt_can_requeue(const server_task * vt) const {
        return params_base.kv_preempt_requeue && vt != nullptr
            && (vt->type == SERVER_TASK_TYPE_COMPLETION || vt->type == SERVER_TASK_TYPE_INFILL)
            && !vt->params.stream && !vt->is_parent() && !vt->is_child() && !vt->cli
            && vt->n_preempt_requeues < params_base.kv_preempt_requeue_max;
    }

    bool preempt_largest_slot(int32_t off) {
        if (!params_base.kv_unified || batch.has_embd()) {
            return false;
        }
        server_slot * victim   = nullptr;
        int           n_active = 0;
        auto cells = [](const server_slot & s) { return (int64_t) s.prompt.n_tokens() + s.n_cfg_past; };
        // request `priority` (lower = more important): the victim is the slot with the HIGHEST value, like vLLM's
        // max((priority, arrival)) (scheduler.py:759-767); among equal values --kv-preempt-victim decides (server-slot-victim.h):
        // the default 'largest' is the old rule, so with every priority at its default 0 and the flag unset nothing changes
        std::vector<server_victim_cand> cand;
        std::vector<server_slot *>      cand_slot;
        for (auto & s : slots) {
            if (!s.is_processing()) {
                continue;
            }
            ++n_active;
            server_victim_cand c;
            c.priority    = s.task ? s.task->params.priority : 0;
            c.cells       = cells(s);
            c.t_arrival   = s.task ? s.task->t_submit_us : 0;
            c.id          = s.task ? s.task->id : 0;
            c.requeueable = preempt_can_requeue(s.task.get());
            c.remaining   = s.n_remaining() < 0 ? INT64_MAX : (int64_t) s.n_remaining(); // limitless generation: never near-finished
            cand.push_back(c);
            cand_slot.push_back(&s);
        }
        if (n_active < 2) {
            return false;
        }
        victim = cand_slot[(size_t) server_pick_victim(cand, params_base.kv_preempt_victim, params_base.kv_preempt_near_finished)];

        std::vector<int32_t> seqs = { victim->id };
        if (victim->id_cfg >= 0) {
            seqs.push_back(victim->id_cfg);
        }
        const int32_t n_before = batch.size();
        // Owner of every entry BEFORE compaction: a slot's index is remapped only if the entry it points
        // at is really that slot's own. MEASURED 2026-09-27 (test_kv_unified_preempt, 3rd preemption): a
        // slot released by an EARLIER preemption keeps its last i_batch, which can point into the new
        // victim's entries -- remapping it blindly hit the assert below. Such a stale index is left
        // exactly as it was (it is never read for a slot that is not processing).
        std::vector<int32_t> owner(n_before);
        for (int32_t i = 0; i < n_before; ++i) {
            owner[i] = batch.tokens[i].id_slot;
        }
        const auto remap = batch.remove_seqs_from(off, seqs);
        for (auto & s : slots) {
            if (&s == victim || !s.is_processing()) {
                continue;
            }
            auto fix = [&](int32_t & idx) {
                if (idx < off || idx >= n_before) {
                    return; // already decoded (before off) or unset (-1): positions unchanged
                }
                if (owner[idx] != s.id && owner[idx] != s.id_cfg) {
                    return; // stale index from an earlier batch -- not this slot's entry
                }
                GGML_ASSERT(remap[idx] >= 0 && "a surviving slot's own batch entry was removed");
                idx = remap[idx];
            };
            fix(s.i_batch);
            fix(s.i_batch_cfg);
            for (auto & i : s.spec_i_batch) {
                fix(i);
            }
        }

        SRV_WRN("KV cache exhausted (--kv-unified): preempting slot %d (%lld cells, %d tokens decoded, victim policy %s, %d in flight); "
                "removed %d pending batch entries\n", victim->id, (long long) cells(*victim), (int) victim->stats.n_gen,
                params_base.kv_preempt_victim == 1 ? "least-progress" : params_base.kv_preempt_victim == 2 ? "newest" : "largest",
                n_active, n_before - batch.size());
        // --kv-preempt-requeue (row S69-VLLM-W2-RECOMPUTE-PREEMPTION-INSTEAD-OF-503-2026-10-27-01): vLLM's preemption frees the
        // victim's blocks, resets num_computed_tokens to 0 and puts the request at the FRONT of the waiting queue to be
        // recomputed (scheduler.py:1539-1582). Same here, but only for a request that has sent NOTHING to the client: a
        // streaming request already sent its 200 headers / first chunk (a recompute could not take those back), so it keeps
        // the 503. A non-streaming request sends nothing until it finishes, however many tokens it generated, so it can be
        // recomputed from its prompt. Bounded by --kv-preempt-requeue-max so re-admission cannot loop for ever.
        const server_task * vt = victim->task.get();
        const bool can_requeue = preempt_can_requeue(vt);
        metrics.n_kv_preempt_cells += (uint64_t) cells(*victim);

        if (can_requeue) {
            server_task rq = vt->requeue_copy(); // BEFORE release(): release() moves the task out of the slot
            SRV_WRN("KV cache exhausted (--kv-unified): requeueing task %d (requeue %d of max %d) to recompute from its prompt\n",
                    rq.id, rq.n_preempt_requeues, params_base.kv_preempt_requeue_max);
            in_preempt_release = true; // this release must not start a waiting task, see callback_on_release
            victim->release();      // also returns (and clears) its CFG guidance seq
            in_preempt_release = false;
            victim->prompt_clear(); // frees its main-seq cells
            queue_tasks.defer(std::move(rq)); // AFTER release(): started by the next request that FINISHES
            metrics.n_kv_requeued++;
            metrics.n_kv_preempted++;
            return true;
        }

        std::string why;
        if (params_base.kv_preempt_requeue) {
            why = vt == nullptr ? " (--kv-preempt-requeue: not requeued, no task)"
                : vt->params.stream ? " (--kv-preempt-requeue: not requeued, it already started streaming to the client)"
                : vt->n_preempt_requeues >= params_base.kv_preempt_requeue_max ? " (--kv-preempt-requeue: not requeued, the requeue limit was reached)"
                : " (--kv-preempt-requeue: not requeued, this request kind cannot be recomputed)";
        }
        send_error(*victim, string_format("KV cache exhausted (--kv-unified): this request held the most cache "
                   "(%lld cells) and was preempted so the other %d in-flight request(s) can finish; retry it%s",
                   (long long) cells(*victim), n_active - 1, why.c_str()), ERROR_TYPE_UNAVAILABLE);
        // with --kv-preempt-requeue a preemption (even one answered 503) must not start a waiting task into the same shortage;
        // without the flag nothing changes: the release pops the deferred queue as before
        in_preempt_release = params_base.kv_preempt_requeue;
        victim->release();      // also returns (and clears) its CFG guidance seq
        in_preempt_release = false;
        victim->prompt_clear(); // frees its main-seq cells
        metrics.n_kv_preempted++;
        return true;
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // A request that carries the warm-prefix token has just finished: save its slot's state in the host prompt cache and pin it.
    // Called from the post-decode phase of update_slots() AND from its all-idle early return: a slot is released at the END of a
    // tick (after the post-decode phase), so when the server then goes idle no later tick reaches the post-decode phase and the pin
    // would wait for the next request (measured: e2e_warm_prefix_served.py "0 pinned entries" after the last warm-up).
    void pin_finished_warm_slots() {
        if (!prompt_cache) {
            return;
        }
        for (auto & slot : slots) {
            if (!slot.pin_pending || slot.is_processing()) {
                continue;
            }
            slot.pin_pending = false;
            slot.prompt_save(*prompt_cache); // false when the identical prompt is already cached: it is pinned below all the same
            const size_t n_pinned_now = prompt_cache->pin_prefix_of(slot.prompt.tokens, slot.prompt.lora_fp);
            if (prompt_cache->pinned_over_share()) {
                SRV_ERR("pinned prefixes hold %.1f MiB, more than --warm-prefix-share (%.2f) of --cache-ram (%.1f MiB): deleting the pins\n",
                        prompt_cache->pinned_size() / (1024.0 * 1024.0), (double) prompt_cache->pinned_share,
                        prompt_cache->limit_size / (1024.0 * 1024.0));
                prompt_cache->drop_pinned();
                pin_overflow_a.store(true);
            } else {
                SRV_INF("warm prefix pinned: %zu entr%s, %d tokens in the slot, %.1f MiB pinned in total\n", n_pinned_now,
                        n_pinned_now == 1 ? "y" : "ies", slot.prompt.n_tokens(), prompt_cache->pinned_size() / (1024.0 * 1024.0));
            }
            pinned_bytes_a.store(prompt_cache->pinned_size());
            pinned_entries_a.store(prompt_cache->n_pinned());
            last_pin_covered_a.store(!pin_overflow_a.load() && prompt_cache->has_pinned_prefix_of(slot.prompt.tokens, slot.prompt.lora_fp));
            prompt_cache->update();
            pin_passes_a.fetch_add(1); // last: the warm-up thread reads the fields above once this moves
        }
    }

    // fill slot.inp_embd with the next rows of the audio stream
    // returns false if the slot has nothing to decode now
    bool slot_audio_step(server_slot & slot) {
        if (!slot.inp_embd.empty()) {
            return true; // rows from a previous step, not decoded yet
        }

        // every produced row is decoded and its delta queued, the commits ending there are complete
        const int64_t n_rows = mtmd_audio_stream_get_n_rows(slot.audio_stream.get());
        while (!slot.audio_commit_rows.empty() && slot.audio_commit_rows.front() <= n_rows) {
            slot.audio_commit_rows.pop_front();
            auto res = std::make_unique<server_task_result_cmpl_partial>();
            res->id                   = slot.task->id;
            res->index                = slot.task->index;
            res->n_decoded            = slot.stats.n_gen;
            res->n_prompt_tokens      = slot.task->n_tokens();
            res->res_type             = slot.task->params.res_type;
            res->oaicompat_model      = slot.task->params.oaicompat_model;
            res->oaicompat_cmpl_id    = slot.task->params.oaicompat_cmpl_id;
            res->audio_commit_drained = true;
            queue_results.send(std::move(res));
        }

        const int32_t n = mtmd_audio_stream_step(slot.audio_stream.get(), slot.sampled);
        if (n > 0) {
            const float * rows = mtmd_audio_stream_get_output_embd(slot.audio_stream.get());
            slot.inp_embd.assign(rows, rows + (size_t) n * batch.n_embd);
            slot.audio_wait = false;
            return true;
        }
        if (n == 0) {
            slot.audio_wait = true;
            return false;
        }
        if (n == -2) {
            send_error(slot, "audio stream error", ERROR_TYPE_SERVER);
            slot.release();
            return false;
        }

        // end of stream
        slot.stop           = STOP_TYPE_EOS;
        slot.has_next_token = false;
        slot.print_timings();
        send_final_response(slot);
        slot.release();
        return false;
    }

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        // a realtime slot waiting for audio has no work, the next audio control task wakes the loop
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing() && !slot.audio_wait) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                pin_finished_warm_slots();
                metrics_flush_idle();

                return; // skip further processing

            } else {
                // requests that have waited past their max_queue_wait_ms for a slot are answered 503 and dropped
                for (const auto & dropped : queue_tasks.take_expired_deferred(ggml_time_ms(), ggml_time_us())) {
                    if (dropped.reason == server_queue::DROP_DEADLINE) {
                        send_error(dropped.id, "request deadline (t_deadline_ms) exceeded while waiting for a slot; retry later", ERROR_TYPE_UNAVAILABLE);
                    } else {
                        send_error(dropped.id, "no slot became free within max_queue_wait_ms; retry later", ERROR_TYPE_UNAVAILABLE);
                    }
                }

                // --shed-max-latency-ms: drop the oldest queued request while it is predicted to miss its budget (server-shed.h)
                if (params_base.shed_max_latency_ms > 0) {
                    const double max_ms = (double) params_base.shed_max_latency_ms;
                    for (;;) {
                        const int id_shed = queue_tasks.shed_oldest_deferred(ggml_time_us(), (int64_t) slots[0].n_ctx,
                            [&](double wait_ms, int64_t tokens, size_t n_queued) { return shed_fit.should_shed(n_queued, wait_ms, tokens, max_ms); });
                        if (id_shed < 0) {
                            break;
                        }
                        send_error(id_shed, string_format("shed: this request would miss the latency budget --shed-max-latency-ms (%d ms) (measured %.0f ms per 1000 tokens); retry later",
                                   (int) params_base.shed_max_latency_ms, shed_fit.fitted ? shed_fit.a_ms * 1000.0 : 0.0), ERROR_TYPE_UNAVAILABLE);
                    }
                }

                // a request still reading its prompt (or waiting for the parent that reads it) when its end-to-end deadline passes, or
                // whose prompt phase has outlasted t_max_prompt_ms, is answered with an error and its slot freed. A request that is
                // already generating is stopped gracefully instead, in process_token(). Row S69-TFSERVING-PICK-END-TO-END-DEADLINE-2026-10-15-01.
                {
                    const int64_t now_us = ggml_time_us();
                    for (auto & slot : slots) {
                        if (!slot.is_processing() || !slot.task) {
                            continue;
                        }
                        const bool in_prompt = slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_WAIT_OTHER;
                        if (!in_prompt) {
                            continue;
                        }
                        const char * why = nullptr;
                        if (slot.task->t_deadline_us > 0 && now_us >= slot.task->t_deadline_us) {
                            why = "request deadline (t_deadline_ms) exceeded while the prompt was being read; retry later";
                        } else if (slot.state == SLOT_STATE_PROCESSING_PROMPT && slot.task->params.t_max_prompt_ms > 0 && slot.stats.t_start > 0 &&
                                   (now_us - slot.stats.t_start) / 1000 >= slot.task->params.t_max_prompt_ms) {
                            why = "t_max_prompt_ms exceeded while the prompt was being read; retry later";
                        }
                        if (why) {
                            SLT_WRN(slot, "%s\n", why);
                            const int id_parent = slot.task->id;
                            send_error(slot, why, ERROR_TYPE_UNAVAILABLE);
                            slot.release();
                            // children (n > 1) that were waiting for this parent's prompt would wait for ever: answer and free them too
                            for (auto & other : slots) {
                                if (other.is_processing() && other.task && other.state == SLOT_STATE_WAIT_OTHER && other.task->id_parent == id_parent) {
                                    send_error(other, why, ERROR_TYPE_UNAVAILABLE);
                                    other.release();
                                }
                            }
                        }
                    }
                }

                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        // note: TTS slots bypass the shared batch entirely
        try {
            process_tts_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_tts_slots() failed: %s\n", e.what());
            abort_all_slots("process_tts_slots() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            if (slot_batched->lora_hetero) {
                // --lora-hetero-batch: nothing context-wide; every sequence carries its own adapters (slots that are not running, and the
                // guidance sequence of a running slot, are set too so no stale list survives a slot's task)
                common_set_adapter_lora(ctx_tgt, empty_lora_list);
                {
                    // always-on observable: a decode batch whose slots run different adapter sets (what --lora-hetero-batch exists for)
                    std::vector<const server_slot *> reps;
                    int n_run = 0;
                    for (auto & s : slots) {
                        // only slots that have a token in THIS batch (a guidance sequence's tokens belong to its slot)
                        bool in_batch = false;
                        for (const auto & t : batch.tokens) {
                            in_batch = in_batch || t.id_slot == s.id || (s.id_cfg >= 0 && t.id_slot == s.id_cfg);
                        }
                        if (!s.is_processing() || !in_batch) {
                            continue;
                        }
                        n_run++;
                        bool seen = false;
                        for (const auto * r : reps) {
                            seen = seen || are_lora_equal(r->lora, s.lora);
                        }
                        if (!seen) {
                            reps.push_back(&s);
                        }
                    }
                    if (reps.size() > 1) {
                        n_lora_hetero_batches++;
                        SRV_INF("lora-hetero: decode batch of %d slots with %zu distinct adapter sets (heterogeneous batches so far: %llu)\n",
                                n_run, reps.size(), (unsigned long long) n_lora_hetero_batches);
                    }
                }
                for (auto & s : slots) {
                    std::vector<llama_adapter_lora *> ptrs;
                    std::vector<float>                scales;
                    if (s.is_processing()) {
                        for (const auto & l : s.lora) {
                            if (l.scale != 0.0f && l.ptr != nullptr) {
                                ptrs.push_back(l.ptr);
                                scales.push_back(l.scale);
                            }
                        }
                    }
                    llama_set_seq_adapters_lora(ctx_tgt, s.id, ptrs.data(), ptrs.size(), scales.data());
                    if (s.id_cfg >= 0) {
                        llama_set_seq_adapters_lora(ctx_tgt, s.id_cfg, ptrs.data(), ptrs.size(), scales.data());
                    }
                    if (dft_gated_lora_ok()) {
                        // the draft context takes the draft_rows adapters of this slot only (the verify context sees them as base rows)
                        std::vector<llama_adapter_lora *> gptrs;
                        std::vector<float>                gscales;
                        for (size_t k = 0; k < ptrs.size(); ++k) {
                            if (llama_adapter_lora_is_draft_gated(ptrs[k])) {
                                gptrs.push_back(ptrs[k]);
                                gscales.push_back(scales[k]);
                            }
                        }
                        llama_set_seq_adapters_lora(ctx_dft, s.id, gptrs.data(), gptrs.size(), gscales.data());
                    }
                }
            } else {
                common_set_adapter_lora(ctx_tgt, slot_batched->lora);
                set_dft_gated_lora(slot_batched->lora);
            }

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01):
                // never let a sub-batch boundary split a slot's main-token/guidance-run group
                // (server_batch::token::pairs_with_next) -- shrink this chunk so the WHOLE group
                // defers to the next one instead. Coordinator review, 2026-09-27: reading (or
                // skipping) logits from the wrong batch.view for a split group is the same defect
                // class as the original silently-ignored CFG fields bug, just narrower and much
                // harder to notice -- "make it impossible" beats any fallback that still samples.
                // WHILE, not IF: a multi-token CFG-prefill run (the whole negative prompt, folded
                // in at prompt-completion time) can need MULTIPLE shrinks to walk the boundary
                // back before the run's first token -- unlike the steady-state 2-entry
                // main+one-guidance-token pair, where a single decrement always suffices (a
                // guidance-token entry is never itself marked pairs_with_next, so one step always
                // lands on an unmarked entry). A run of N marked entries needs up to N steps.
                while (n_tokens > 0 && off + n_tokens < batch.size() && batch.tokens[off + n_tokens - 1].pairs_with_next) {
                    n_tokens--;
                    if (n_tokens == 0) {
                        // Only reachable if n_batch itself shrank below the group's size (the
                        // decode()-failure retry path below halves n_batch under KV pressure)
                        // while a CFG group is pending right at `off`. Fail this round loudly --
                        // caught below, which aborts the affected slot(s) with a clear error --
                        // rather than call render(off, 0) (GGML_ASSERT, uncatchable process
                        // abort) or sample an unguided token silently.
                        throw std::runtime_error("classifier-free guidance main/guidance group "
                                "does not fit in the current batch size (n_batch shrank while "
                                "retrying after a decode failure)");
                    }
                }

                batch.render(off, n_tokens);
                bool ok = decode(n_batch, off);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void process_tts_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || slot.task->type != SERVER_TASK_TYPE_TTS) {
                return;
            }

            llama_set_embeddings(ctx_tgt, true);

            if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                const int32_t ret = slot.tts.ctx.step_prompt(llama_n_batch(ctx_tgt));
                if (ret < 0) {
                    send_error(slot, "TTS prompt processing failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (ret == 0) {
                    // done prompt, do sample and save h_state_prompt
                    slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                    common_sampler_accept(slot.smpl.get(), ctx_tgt, slot.tts.sampled, true);
                    const float * h_embd = llama_get_embeddings_ith(ctx_tgt, -1);
                    slot.tts.h_state_prompt.assign(h_embd, h_embd + llama_model_n_embd(model_tgt));
                    slot.tts.h_state = slot.tts.h_state_prompt.data();
                    slot.state = SLOT_STATE_GENERATING;
                }
                return;
            }

            const int32_t n_predict = slot.n_predict_max > 0 ? slot.n_predict_max : 512;
            if (slot.tts.stop || slot.tts.n_decoded >= n_predict) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                // generation truly ends here: force out any sub-window remainder still buffered
                if (slot.tts.ctx.flush() != 0 || slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "failed to finalize TTS output", ERROR_TYPE_SERVER);
                } else {
                    send_tts_result(slot, sample_rate, data, data_len, true);
                }
                slot.release();
                return;
            }

            const float * h_state_next = nullptr;
            bool          stop         = false;
            if (slot.tts.ctx.step_gen(slot.tts.sampled, slot.tts.h_state, &h_state_next, &stop) != 0) {
                send_error(slot, "TTS generation failed", ERROR_TYPE_SERVER);
                slot.release();
                return;
            }
            if (h_state_next == nullptr) {
                // end-of-speech without a new frame (e.g. pocket-tts eos head)
                slot.tts.stop = true;
                return;
            }
            slot.tts.h_state = h_state_next;
            slot.tts.n_decoded++;
            slot.tts.stop = stop;

            if (!stop) {
                slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                common_sampler_accept(slot.smpl.get(), ctx_tgt, slot.tts.sampled, true);
            }

            if (slot.task->params.stream) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                if (slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "TTS streaming output failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (data_len > 0) {
                    send_tts_result(slot, sample_rate, data, data_len, false);
                }
            }
        });
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.task && slot.task->type == SERVER_TASK_TYPE_TTS) {
                // TTS slots drive their own decode loop in process_tts_slots(), never enter the shared batch
                return;
            }
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                slot.radix_ctx_shifted = true; // --radix-cache: positions renumbered, do not publish this request's generation

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // reasoning_loop_recovery: rewrite the KV of slots whose recover retry was scheduled by process_token(). Here, because no
        // slot is waiting for logits yet; the sampled token goes into the batch below at the rewritten sequence's end.
        iterate(slots, [&](server_slot & slot) {
            if (slot.rlr_pending && slot.state == SLOT_STATE_GENERATING) {
                rlr_apply(slot);
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;
        spec_async_reqs.clear();

        // --draft-max-by-slots: how many slots will share this batch, counted with the same can_batch_with filter the loop below uses,
        // so the schedule looks up the number of slots that really decode together. The lookup only ever lowers a slot's budget.
        int draft_sched_k = -1;
        if (spec && !draft_schedule.empty()) {
            int n_gen_batch = 0;
            server_slot * first_gen = nullptr;
            iterate(slots, [&](server_slot & slot) {
                if (slot.state != SLOT_STATE_GENERATING) {
                    return;
                }
                if (!first_gen) {
                    first_gen = &slot;
                } else if (!first_gen->can_batch_with(slot)) {
                    return;
                }
                n_gen_batch++;
            });
            draft_sched_k = draft_schedule.lookup_sticky(n_gen_batch);
        }

        // fair group lead: slots whose lora lists differ never share a batch and the first generating slot used to lead every step,
        // so its group starved the others. Pre-select the leader by round-robin over the distinct lora configs (server-lora-fair.h).
        // Only plain generating slots are candidates (no TTS, no audio stream), so the leader always lands in `generating` below.
        {
            std::vector<server_slot *> cand;
            std::vector<uint64_t>      fps;
            iterate(slots, [&](server_slot & slot) {
                if (slot.state != SLOT_STATE_GENERATING || slot.task->type == SERVER_TASK_TYPE_TTS || slot.audio_stream || slot.lora_hetero) {
                    return;
                }
                cand.push_back(&slot);
                fps.push_back(lora_config_fingerprint(slot.lora));
            });
            if (cand.size() > 1) {
                server_slot * lead = cand[lora_fair.pick(fps)];
                if (!slot_batched) {
                    slot_batched = lead;
                }
            }
        }

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || slot.task->type == SERVER_TASK_TYPE_TTS) {
                return;
            }

            if (slot.audio_stream && !slot_audio_step(slot)) {
                return; // waiting for audio, or the stream ended
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                int n_draft_max = slot.get_n_draft_max();
                if (draft_sched_k >= 0) {
                    n_draft_max = std::min(n_draft_max, draft_sched_k);
                }

                // per-request speculative.* (server-schema.cpp): forwarded only when the request CHANGED the value,
                // so an omitted field leaves every draft impl exactly as configured server-wide (including a stage's
                // own n_max, and the explicit-only p_min gate of common_params_speculative::p_min_explicit)
                const auto & sp_req  = slot.task->params.speculative;
                const auto & sp_base = params_base.speculative;
                if (sp_req.n_max != sp_base.n_max && sp_req.n_max >= 0) {
                    n_draft_max = std::min(n_draft_max, (int) sp_req.n_max);
                }
                const int32_t req_n_min = sp_req.n_min != sp_base.n_min ? sp_req.n_min : -1;
                const float   req_p_min = sp_req.p_min_explicit &&
                        (!sp_base.p_min_explicit || sp_req.p_min != sp_base.p_min) ? sp_req.p_min : -1.0f;

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            // an on-device checkpoint holds no host bytes (data_tgt is cleared when it is taken) and is kept
                            // across the replay round; MEASURED pa3 gpufb arm: the host-only check aborted the first replay
                            GGML_ASSERT(slot.spec_ckpt_dev || !slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();
                        slot.spec_dists.clear();

                        const bool spec_reject = slot.use_spec_rejection();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .dists    = */ params_base.speculative.draft_sample ? &slot.spec_dists : nullptr,
                            /* .n_min    = */ req_n_min,
                            /* .p_min    = */ req_p_min,
                            /* .smpl_tgt = */ params_base.speculative.draft_grammar ? slot.smpl.get() : nullptr,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            // --spec-draft-validate: keep only the grammar-valid prefix of the draft (vLLM validate_tokens analogue), so the
            // verify batch carries no token the grammar will refuse. Lossless: the target still decides every token.
            if (params_base.speculative.draft_validate && !draft.empty() && slot.smpl) {
                const size_t n_valid = common_sampler_draft_valid_prefix(slot.smpl.get(), ctx_tgt, draft);
                if (n_valid < draft.size()) {
                    SLT_DBG(slot, "draft cut to its grammar-valid prefix: %zu of %zu tokens\n", n_valid, draft.size());
                    draft.resize(n_valid);
                    if (slot.spec_dists.size() > n_valid) {
                        slot.spec_dists.resize(n_valid);
                    }
                }
            }

            slot.stats.n_draft_tokens += draft.size();

            // --spec-async: this draft is verified in the batch built below; a worker evaluates it on the draft context meanwhile. Only for
            // contexts that roll back partial sequences by themselves (no checkpoint on either side), a plain (non-replay) round, and the
            // greedy draft (no recorded distributions / rejection candidates).
            // A slot with a draft_rows adapter on the draft context stays on the serial draft: the async worker evaluates the verify's
            // draft tokens ahead of their acceptance and is not draft-rows aware.
            if (params_base.speculative.async && !draft.empty() && ctx_dft && !dft_gated_lora_active(slot.lora) &&
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART &&
                    slot.spec_dists.empty() && slot.spec_draft_q.empty() && !slot.spec_is_replay) {
                // greedy hint for --spec-async-informed-join: the target takes the argmax of its row (no sampling randomness, penalty,
                // bias or grammar between the logits and the token); a wrong hint only wastes a wait, it cannot change a token
                const auto & sp_s = slot.task->params.sampling;
                const bool greedy = sp_s.temp <= 0.0f && sp_s.penalty_repeat == 1.0f && sp_s.penalty_freq == 0.0f && sp_s.penalty_present == 0.0f &&
                                    sp_s.logit_bias.empty() && sp_s.grammar.empty();
                spec_async_reqs.push_back({ slot.id, slot.prompt.tokens.pos_next(), slot.sampled, &slot.spec_draft, greedy });
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                // test hook (needs a draft-model server; not covered by test_slot_fault.py): LLAMA_SERVER_TEST_FAULT_DRAFT_SEQ_RM_SLOT=<id> fails this
                // rollback for that slot id only, ONCE per process (a fault that re-fired on every later request would test a wedged slot,
                // not the recovery this path promises)
                static bool  fault_dft_fired = false;
                const char * fault_dft       = std::getenv("LLAMA_SERVER_TEST_FAULT_DRAFT_SEQ_RM_SLOT");
                const bool   inject_dft      = !fault_dft_fired && fault_dft && std::atoi(fault_dft) == slot.id;
                fault_dft_fired             |= inject_dft;
                const bool   dft_rm_ok = inject_dft ? false :
                        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1);
                if (!dft_rm_ok) {
                    // BeeLlama-style owner-only failure (Anbeeld/beellama.cpp, MIT; server-context.cpp slot-failure paths, rewritten): a
                    // draft-context rollback that cannot be done fails THIS slot's request, it does not abort the shared server
                    // (every other slot on the port keeps generating). Row S69-ENGREV-BEELLAMA-OWNER-ONLY-FAILURE-NO-ABORT-2026-10-27-01.
                    SLT_ERR(slot, "failed to remove sequence %d from the draft context (p0 = %d) - failing this slot only\n", slot.id, (int) (ckpt.pos_max + 1));
                    if (!spec_async_reqs.empty() && spec_async_reqs.back().seq_id == slot.id) {
                        spec_async_reqs.pop_back();
                    }
                    generating.erase(std::remove(generating.begin(), generating.end(), &slot), generating.end());
                    if (slot_batched == &slot) {
                        slot_batched = generating.empty() ? nullptr : generating.front();
                    }
                    send_error(slot, "speculative draft context rollback failed", ERROR_TYPE_SERVER);
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                    slot.prompt_clear();
                    slot.release();
                    return;
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    if (slot.spec_ckpt_dev) {
                        llama_spec_ckpt_discard(ctx_tgt, server_spec_ckpt);
                        slot.spec_ckpt_dev = false;
                    }
                    if (spec_ckpt_dev_mode != LLAMA_SPEC_CKPT_NONE &&
                            (int) draft.size() + 1 <= spec_ckpt_dev_tokens) {
                        const int mode = llama_spec_ckpt_init(ctx_tgt, server_spec_ckpt, spec_ckpt_dev_mode, (int) draft.size() + 1);
                        slot.spec_ckpt_dev = mode == spec_ckpt_dev_mode && llama_spec_ckpt_save(ctx_tgt, server_spec_ckpt, slot.id);
                        if (!slot.spec_ckpt_dev) {
                            llama_spec_ckpt_discard(ctx_tgt, server_spec_ckpt);
                            SLT_WRN(slot, "on-device speculative checkpoint failed (mode %d), using the host copy for this draft\n", mode);
                        }
                    }
                    if (slot.spec_ckpt_dev) {
                        ckpt.data_tgt.clear();
                    } else {
                        ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            // --prefill-fair-share: without it the first prompt-processing slot fills the whole batch and every later slot's
            // prompt waits for it to finish. Water-filling: sort the prefilling slots by what they still need, give each in
            // turn min(need, equal share of what is left), so a short prompt is done this step and the long one gets the rest.
            // A single prefilling slot keeps the whole batch. Slots that cannot split their prompt (non-causal embedding) are
            // left alone.
            std::map<int, int32_t> prefill_cap;
            if (params_base.prefill_fair_share) {
                std::vector<std::pair<int32_t, int>> need;
                iterate(slots, [&](server_slot & slot) {
                    if (slot.is_processing() && slot.can_split() &&
                            (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT)) {
                        const int32_t done = slot.state == SLOT_STATE_STARTED ? 0 : (int32_t) slot.prompt.n_tokens();
                        need.emplace_back(std::max<int32_t>(1, (int32_t) slot.task->n_tokens() - done), (int) slot.id);
                    }
                });
                if (need.size() > 1) {
                    std::sort(need.begin(), need.end());
                    int32_t budget = std::max<int32_t>(1, n_batch - (int32_t) batch.size());
                    int32_t n_left = (int32_t) need.size();
                    for (const auto & e : need) {
                        const int32_t cap = std::min<int32_t>(e.first, std::max<int32_t>(1, budget / n_left));
                        prefill_cap[e.second] = cap;
                        budget -= cap;
                        --n_left;
                    }
                }
            }

            // VISIT ORDER (--embd-length-order). A sequence that cannot be split must sit whole inside this micro-batch, and the loop below skips a slot
            // whose sequence does not fit and tries the next one, i.e. first-fit over the visit order. Slot ids do not follow the order the requests were
            // queued in (an idle slot is chosen by prompt similarity / least-recent use), so first-fit over slot ids is not "decreasing" even when the queue
            // is. Visiting the unsplittable slots largest first makes it first-fit-DECREASING over what is resident. MEASURED cause of the 4-vs-5 decode
            // result (temp/_e1011_sim.py on the e2e's real token counts 327 15 305 ... sum 1550, 6 slots, n_ubatch 512): visiting by slot id, 5 decodes
            // happen in 1.2% of slot assignments for the arrival order and 5.5% for the longest-first order; visiting largest first gives 4 (the optimum,
            // ceil(1550/512)) in all 4000 assignments for both orders. Other slots keep their id order.
            // --prefill-step-budget / --prefill-tbt-target-ms: decode tokens are already in the batch here; when there are any, the prompt tokens
            // of all slots together stop at the cap, so the step (and with it every stream's next token) is bounded. No decode tokens: whole batch.
            const int32_t n_dec_in_batch = (int32_t) batch.size();
            int32_t step_prompt_cap = INT32_MAX;
            if (n_dec_in_batch > 0 && (params_base.prefill_step_budget > 0 || params_base.prefill_tbt_target_ms > 0)) {
                step_prompt_cap = prefill_step_cap(n_batch, n_ubatch);
            }

            std::vector<server_slot *> visit;
            visit.reserve(slots.size());
            for (auto & s : slots) {
                visit.push_back(&s);
            }
            if (params_base.embd_length_order) {
                const auto pending_whole = [](const server_slot * s) -> int32_t {
                    return s->is_processing() && s->task && (s->state == SLOT_STATE_STARTED || s->state == SLOT_STATE_PROCESSING_PROMPT) && !s->can_split()
                        ? (int32_t) s->task->n_tokens() : -1;
                };
                std::stable_sort(visit.begin(), visit.end(), [&](const server_slot * a, const server_slot * b) { return pending_whole(a) > pending_whole(b); });
            }

            iterate(visit, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }
                if (step_prompt_cap != INT32_MAX && (int32_t) batch.size() - n_dec_in_batch >= step_prompt_cap) {
                    return; // the step's prompt budget is spent (--prefill-step-budget)
                }

                if (!slot.is_processing() || slot.task->type == SERVER_TASK_TYPE_TTS) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // no prompt, the rows come from the audio stream
                if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start(slot.task->t_submit_us);
                        if (metrics.reqlog) {
                            metrics.reqlog->begin(request_log_key(*slot.task), slot.task->params.trace_id, model_name);
                        }

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // the outputs of a decision are read from one batch
                            const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;
                            if (n_decision_first >= 0 && slot.task->n_tokens() - n_decision_first > n_batch) {
                                send_error(slot,
                                           string_format("the question and its options (%d tokens) are too large to process. "
                                                         "increase the batch size (current batch size: %d)",
                                                         slot.task->n_tokens() - n_decision_first, n_batch),
                                           ERROR_TYPE_INVALID_REQUEST);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // --prompt-reuse-plan (koboldcpp-style fast-forward guards, common/prompt-reuse.h): the planner may only LOWER the
                                // reuse the longest common prefix allows; prompts with mtmd chunks keep the plain prefix (their tokens are not text)
                                if (params_base.prompt_reuse_plan && n_past > 0 && !slot.prompt.tokens.has_mtmd && !input_tokens.has_mtmd) {
                                    prompt_reuse_opts ro;
                                    ro.require_full_subset = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                                    ro.min_to_proceed      = (size_t) std::max(0, params_base.prompt_reuse_min);
                                    ro.min_input_to_keep   = (size_t) std::max(0, params_base.prompt_reuse_keep_input);
                                    const llama_tokens & cached = slot.prompt.tokens.get_tokens();
                                    const llama_tokens & prompt = input_tokens.get_tokens();
                                    const size_t plan = prompt_reuse_plan(std::vector<int>(cached.begin(), cached.end()),
                                                                          std::vector<int>(prompt.begin(), prompt.end()), ro);
                                    // always-on observable: one line per planned reuse
                                    SLT_INF(slot, "prompt-reuse-plan: common prefix %d -> planner reuses %zu (min %d, keep-input %d, full-subset %d)\n",
                                            n_past, std::min<size_t>(plan, (size_t) n_past), params_base.prompt_reuse_min,
                                            params_base.prompt_reuse_keep_input, (int) ro.require_full_subset);
                                    n_past = (int) std::min<size_t>(plan, (size_t) n_past);
                                }

                                // --radix-cache: only tried when this slot has NO leftover cache of
                                // its own (n_past == 0 AND prompt.tokens empty) -- a fresh/idle
                                // slot picking up a brand-new request, the common case an idle-slot
                                // dispatch produces. Deliberately NOT tried when the slot already
                                // has a shorter-but-nonzero own match: interleaving a radix hit with
                                // cache_prompt's own reuse/checkpoint/alora machinery below (which
                                // all assume prompt.tokens already holds exactly what n_past claims)
                                // was left alone rather than risk desyncing it -- filed as a known
                                // gap, not silently dropped (see server-radix-cache.h's top comment).
                                if (radix_cache && n_past == 0 && slot.prompt.tokens.size() == 0 &&
                                    !input_tokens.has_mtmd) {
                                    const llama_tokens & full = input_tokens.get_tokens();
                                    const size_t radix_n = radix_cache->match_prefix(full, slot.prompt.lora_fp);
                                    if (radix_n > 0) {
                                        radix_cache->commit(slot.id, radix_n);
                                        slot.prompt.tokens = server_tokens(
                                            llama_tokens(full.begin(), full.begin() + radix_n), false);
                                        n_past = (int) radix_n;
                                        SLT_DBG(slot, "radix-cache hit: %zu prefix token(s) reused via zero-copy seq_cp\n", radix_n);
                                    } else {
                                        radix_cache->release();
                                    }
                                }

                                // prompt_logprobs with logprob_start_len: tokens before the start were not scored by this request so they may be
                                // reused, but the row of token start - 1 produces the score of token start, so that token is evaluated again
                                if (slot.task->params.prompt_logprobs && slot.task->params.logprob_start_len > 0) {
                                    n_past = std::min(n_past, (int) slot.task->params.logprob_start_len - 1);
                                    // always on: the proof that part of the prompt came from a cache while the rest was scored
                                    SLT_INF(slot, "prompt_logprobs from position %d: %d cached token(s) reused, %d prompt token(s) evaluated\n",
                                            (int) slot.task->params.logprob_start_len, n_past, slot.task->n_tokens() - n_past);
                                }

                                // the children start from the shared prefix, do not go past it
                                if (slot.task->n_tokens_shared > 0) {
                                    n_past = std::min(n_past, slot.task->n_tokens_shared);
                                }

                                // the outputs of a decision are read from the batch, so its first position must be evaluated
                                if (n_decision_first >= 0) {
                                    n_past = std::min(n_past, n_decision_first);
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                // chunk shifting would place cached tokens past the start without evaluating them, leaving holes in the scores
                                const auto n_cache_reuse = slot.task->params.prompt_logprobs ? 0 : slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // number of cached tokens shared with the new prompt, before any checkpoint is restored
                            const int n_prefix = n_past;

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                            // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval.
                            // The old threshold (pos_eval - n_swa) asked for two more cells than the window, so a SWA state
                            // read back from a slot file or the RAM prompt cache (which holds only its window) was always
                            // refused. Upstream llama.cpp 61ad80bf (#30580, MIT); row
                            // S69-LLAMACPP-PICK-SWA-REUSE-THRESHOLD-OFF-BY-TWO-2026-10-18-01.
                            // a state or a checkpoint is useful when its pos_min is below this threshold
                            // (tools/server/server-swa-reuse.h, unit test test-server-swa-reuse)
                            const auto pos_min_thold = server_swa_reuse_pos_min_thold(pos_next, has_new_tokens, n_swa);

                            // tokens this prompt shares with the cached one, before any checkpoint fallback shortens n_past
                            const int n_past_shared = n_past;
                            slot.lcp_snapshot_at = -1;

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                // test hook (tools/server/tests/unit/test_slot_fault.py): LLAMA_SERVER_TEST_FAULT_POS_MIN_SLOT=<id> makes this
                                // slot see an empty sequence here, ONCE per process. MEASURED (gate b1115a, 2026-11-15): with a re-firing
                                // fault the "slot 0 serves again" request failed too -- the reset slot legitimately got its 21-token prefix
                                // back from the prompt cache (n_past = 21, prompt.tokens.size() = 21), so the injected -1 hit it again.
                                static bool  fault_pm_fired = false;
                                const char * fault_pm       = std::getenv("LLAMA_SERVER_TEST_FAULT_POS_MIN_SLOT");
                                const bool   inject_pm      = !fault_pm_fired && fault_pm && std::atoi(fault_pm) == slot.id;
                                fault_pm_fired             |= inject_pm;
                                const auto pos_min = inject_pm ? (llama_pos) -1 :
                                        llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    // BeeLlama-style owner-only failure (Anbeeld/beellama.cpp, MIT, rewritten): the slot's bookkeeping says it holds
                                    // n_past > 0 tokens but its sequence is empty (upstream llama.cpp #13833 "should not happen"). That is one slot's
                                    // state, so fail and reset THIS slot instead of aborting the shared server. Row
                                    // S69-ENGREV-BEELLAMA-OWNER-ONLY-FAILURE-NO-ABORT-2026-10-27-01.
                                    // CAPABILITY-GENUINELY-NA: error path for a corrupt slot, no model capability is disabled; it narrows an abort to one request.
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d - failing this slot only\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    send_error(slot, "slot memory out of sync with its prompt (pos_min == -1 with n_past > 0); the slot was reset, retry the request", ERROR_TYPE_SERVER);
                                    slot.prompt_clear();
                                    if (ctx_dft) {
                                        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                                    }
                                    slot.release();
                                    return;
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                // --swa-compress: a state that passes the threshold may still sit below the compacted
                                // window's rewind floor, where seq_rm refuses and the server aborts (server-swa-reuse.h)
                                const bool rewind_ok = server_swa_reuse_rewind_ok(pos_next, has_new_tokens,
                                        llama_kv_cache_swa_rewind_floor(ctx_tgt),
                                        llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) + 1);
                                if (!rewind_ok && pos_min < pos_min_thold) {
                                    SLT_WRN(slot, "the compacted SWA window (pos_min = %d) cannot be rewound to %d; treating the state as unusable\n",
                                            pos_min, pos_next - (has_new_tokens ? 0 : 1));
                                }

                                if (pos_min >= pos_min_thold || !rewind_ok) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & ckpt_p) {
                                            const auto & cur = *ckpt_p;
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset && (*it)->spilled()) {
                                        // the data is on disk (--ctx-ckpt-spill-dir file, or a blob of the --cache-disk-dir tier): bring
                                        // it back, or reprocess the prompt as if there were no checkpoint (what happens without the
                                        // spill feature). A tier blob is checksummed by the tier: a bad one is dropped there and lands here.
                                        // the checkpoint may be shared with a cache entry that holds it spilled: reload into THIS slot's own copy
                                        server_ckpt_private(*it);
                                        const bool from_tier = (*it)->disk != nullptr;
                                        const std::string where = from_tier ? std::string("the disk tier") : (*it)->spill_path;
                                        const int64_t t_load = ggml_time_us();
                                        const bool reloaded = (*it)->reload();
                                        if (from_tier && prompt_cache) {
                                            prompt_cache->note_ckpt_reload(reloaded);
                                            if (reloaded) {
                                                prompt_cache->note_restore((*it)->size(), (ggml_time_us() - t_load) / 1e6);
                                            }
                                        }
                                        if (!reloaded) {
                                            SLT_WRN(slot, "could not reload the spilled context checkpoint (pos_min = %d, pos_max = %d) from %s; reprocessing the prompt\n",
                                                    (*it)->pos_min, (*it)->pos_max, where.c_str());
                                            do_reset = true;
                                        } else {
                                            SLT_WRN(slot, "reloaded context checkpoint from spill disk took %.2f ms (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ")\n",
                                                    (ggml_time_us() - t_load) / 1000.0, (*it)->pos_min, (*it)->pos_max, (*it)->n_tokens);
                                        }
                                    }

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        if (!(*it)->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                                            !(*it)->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                            if ((*it)->id_task != -1) {
                                                GGML_ABORT("failed to restore context checkpoint\n");
                                            }
                                            // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                            SLT_WRN(slot, "%s", "failed to load context checkpoint restored from a slot file\n");
                                            do_reset = true;
                                        }
                                    }

                                    if (!do_reset) {
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, (*it)->data_spec);

                                        pos_next = std::min(pos_next, std::max((*it)->pos_min + 1, (*it)->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) (*it)->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", (*it)->pos_min, (*it)->pos_max, (*it)->n_tokens, n_past, (float) (*it)->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }

                                    // Branch-point checkpoint: the state could not be rewound to the shared length (n_past_shared) and the
                                    // nearest checkpoint left less than that. This request re-evaluates the tokens between, so take a
                                    // checkpoint when the prefill reaches the shared length; a later request that diverges there restores
                                    // it. SOURCE-READ: ollama mlxrunner/prefix_cache.go:131-135 (schedule at the branch point during
                                    // prefill), :313-367 (offsets at/before the cache position or past the prompt end are dropped).
                                    // Not scheduled when the prompt carries media (the offset could fall inside a chunk; ollama moves it
                                    // past the item, here the existing "no checkpoint after mtmd" rule would refuse it anyway).
                                    // DISABLE-REVERSIBLE: LLAMA_SERVER_LCP_CHECKPOINT=0 restores the old behaviour. RESEARCH-DONE: row S69-ENGINE-MERGE-OLLAMA-2026-09-09-01.
                                    static const bool lcp_checkpoint_on = []() {
                                        const char * e = getenv("LLAMA_SERVER_LCP_CHECKPOINT");
                                        return !(e && std::string(e) == "0");
                                    }();
                                    if (lcp_checkpoint_on && params_base.n_ctx_checkpoints > 0 && !input_tokens.has_mtmd &&
                                            n_past < n_past_shared && n_past_shared < (int) slot.task->n_tokens()) {
                                        slot.lcp_snapshot_at = n_past_shared;
                                        SLT_TRC(slot, "branch point at %d tokens (state restored to %d): a checkpoint will be taken there\n",
                                                n_past_shared, n_past);
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = **it;
                                    if (cur.pos_max > pos_next || cur.n_tokens > n_prefix) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = drop_checkpoint(slot.prompt.checkpoints, it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // shared prompt prefix: once it is processed, the children continue from it with their own prompt
                    bool wait_shared = false;
                    if (slot.task->n_tokens_shared > 0) {
                        const bool is_shared_done = slot.prompt.n_tokens() == slot.task->n_tokens_shared;
                        for (auto & other : slots) {
                            if (other.state != SLOT_STATE_WAIT_OTHER || other.task->id_parent != slot.task->id) {
                                continue;
                            }
                            if (is_shared_done) {
                                SLT_TRC(slot, " - copying shared prompt (%d tokens) to child %d\n", slot.prompt.n_tokens(), other.id);
                                slot.copy_prompt_to(other);
                                other.state = SLOT_STATE_STARTED;
                            } else {
                                wait_shared = true;
                            }
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    // TODO: do the same for all models, then remove process_mtmd_chunk()
                    if (use_mixed_batch() && !slot.can_split() && input_tokens.has_mtmd) {
                        if (!add_prompt_mixed(slot)) {
                            return;
                        }
                    }

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        queue_tasks.add_prefill_tokens(n_tokens_out); // --sched-policy hrrn aging
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // this slot's share of the batch under --prefill-fair-share (no entry: no limit beyond the batch itself)
                    int32_t prefill_cap_this = INT32_MAX;
                    {
                        const auto it_cap = prefill_cap.find((int) slot.id);
                        if (it_cap != prefill_cap.end()) {
                            prefill_cap_this = it_cap->second;
                        }
                    }
                    // --prefill-ubatch-chunks: one micro-batch of prompt per step, the cancel task is seen after it (a slot that cannot
                    // split its prompt keeps the whole sequence in one step)
                    if (params_base.prefill_ubatch_chunks && slot.can_split()) {
                        prefill_cap_this = std::min<int32_t>(prefill_cap_this, std::max<int32_t>(1, n_ubatch));
                    }
                    if (step_prompt_cap != INT32_MAX && slot.can_split()) {
                        prefill_cap_this = std::min<int32_t>(prefill_cap_this, std::max<int32_t>(1, step_prompt_cap - ((int32_t) batch.size() - n_dec_in_batch)));
                    }
                    const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch &&
                            (int32_t) (batch.size() - n_tokens_prev) < prefill_cap_this) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // stop at the end of the shared prefix, the children are started from this state
                        if (wait_shared && slot.prompt.n_tokens() == slot.task->n_tokens_shared) {
                            break;
                        }

                        // the outputs of a decision are read from one batch, do not split them
                        if (slot.prompt.n_tokens() == n_decision_first && batch.size() + slot.task->n_tokens() - n_decision_first > n_batch) {
                            break;
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd() || (slot.task->params.prompt_logprobs &&
                                                    slot.prompt.n_tokens() + 1 >= slot.task->params.logprob_start_len),
                            /* is_prompt = */ true);
                        if (!slot.task->decision.order.empty()) {
                            batch.set_decision_order(batch.size() - 1, slot.task->decision.order[slot.prompt.n_tokens()]);
                        }
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back()->n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // end this batch at the branch point so the next one starts there and takes the checkpoint
                        if (do_checkpoint && slot.lcp_snapshot_at > 0 && slot.prompt.n_tokens() == slot.lcp_snapshot_at) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }

                        // banned strings on FULL memory: the first generated token can only be rewound
                        // from the state before the LAST prompt token, so decode that token on its own
                        if (slot.ban_ckpt_on && slot.task->n_tokens() == slot.prompt.n_tokens() + 1) {
                            break;
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    // ctx_checkpoints_tolerance (fork field, port row S69-FORK-SERVER-FEATURES-
                    // STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 3; SOURCE-READ:
                    // examples/server/server-context.cpp:4205 @ 35f0cac75) is a floor, not a
                    // replacement: std::max keeps the existing n_ubatch-based window at its default
                    // (5, always < a realistic n_ubatch) and only widens it when a caller explicitly
                    // asks for more tokens of "near the end" slack than the batch size already gives.
                    const int32_t checkpoint_tolerance = std::max(n_ubatch, params_base.ctx_checkpoints_tolerance);
                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + checkpoint_tolerance;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;
                    // this batch starts exactly at the branch point: the memory holds the shared prefix and nothing else
                    const bool is_lcp_start = slot.lcp_snapshot_at > 0 && n_tokens_start == slot.lcp_snapshot_at;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        // Classifier-Free Guidance: fold the WHOLE held negative prompt
                        // (server_slot::cfg_prompt_tokens's comment explains why it waits until
                        // exactly here, and why never via a separate decode call) into the SAME
                        // shared batch as the main prompt's own last token, in the SAME decode
                        // call, so the guidance logits this produces are still fresh when the
                        // sampling call for the FIRST generated token runs.
                        if (slot.id_cfg >= 0) {
                            // The main token (added just above) is about to be followed by the
                            // whole CFG-prefill run -- mark it so update_slots()'s chunking never
                            // starts a chunk boundary between it and the run.
                            batch.tokens[slot.i_batch].pairs_with_next = true;
                            for (size_t i = 0; i < slot.cfg_prompt_tokens.size(); i++) {
                                const bool is_last_cfg_tok = (i + 1 == slot.cfg_prompt_tokens.size());
                                const int32_t idx = batch.size();
                                const bool cfg_ok = batch.add(slot.id_cfg, slot.cfg_prompt_tokens[i],
                                        slot.n_cfg_past++, /* output = */ is_last_cfg_tok, /* is_prompt = */ true);
                                GGML_ASSERT(cfg_ok && "batch must be large enough to hold the CFG prefill");
                                if (is_last_cfg_tok) {
                                    // The natural end of the atomic run -- logits requested here,
                                    // and nothing after it needs to stay glued to this one.
                                    slot.i_batch_cfg = idx;
                                } else {
                                    // Still inside the run: the chunking guard must not stop
                                    // between this token and the next one either.
                                    batch.tokens[idx].pairs_with_next = true;
                                }
                            }
                        }

                        // the last prompt token was split off above, so the memory still lacks it
                        if (slot.ban_ckpt_on && n_tokens_cur == 1) {
                            slot.ban_ckpt_take(slot.prompt.n_tokens() - 1);
                        }

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end && !is_lcp_start) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end || is_lcp_start ||
                            n_tokens_start > slot.prompt.checkpoints.back()->n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max, near_prompt_end, is_lcp_start);
                    }
                    if (is_lcp_start) {
                        SLT_INF(slot, "branch-point checkpoint at %d prompt tokens: %s\n", (int) n_tokens_start,
                                do_checkpoint ? "created" : "not created");
                        slot.lcp_snapshot_at = -1;   // once per task
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // see https://github.com/ggml-org/llama.cpp/pull/29969
    // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
    bool use_mixed_batch() const {
        return !llama_get_memory(ctx_tgt) || !llama_get_causal_attn(ctx_tgt);
    }

    // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
    // the caller makes sure that it fits in the batch
    // returns false on error, the slot is then released
    bool add_prompt_mixed(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const auto n_tokens_prev = batch.size();

        while (slot.prompt.n_tokens() < slot.task->n_tokens()) {
            const auto cur_token_idx = slot.prompt.n_tokens();
            const llama_token cur_tok = input_tokens[cur_token_idx];

            if (cur_tok != LLAMA_TOKEN_NULL) {
                const bool add_ok = batch.add(slot.id,
                    cur_tok,
                    /* pos       = */ slot.prompt.tokens.pos_next(),
                    /* output    = */ slot.need_embd(),
                    /* is_prompt = */ true);
                GGML_ASSERT(add_ok);
                if (!slot.task->decision.order.empty()) {
                    batch.set_decision_order(batch.size() - 1, slot.task->decision.order[cur_token_idx]);
                }
                slot.prompt.tokens.push_back(cur_tok);
                continue;
            }

            const auto & chunk = input_tokens.find_chunk(cur_token_idx);

            float * embd = slot.mbatch ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            if (!embd) {
                // encode on the worker thread, so we can still handle metrics tasks
                int32_t res = 0;
                queue_tasks.yield_to_queue([&]() {
                    res = encode_mtmd_chunk(slot, slot.mbatch, cur_token_idx);
                });
                embd = res == 0 ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            }

            if (!embd || !add_mtmd_chunk(slot, chunk.get(), embd, batch)) {
                SLT_ERR(slot, "%s", "failed to process mtmd chunk\n");
                // the batch must not keep the entries of a released slot
                batch.truncate(n_tokens_prev);
                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }

            slot.prompt.tokens.push_back_placeholder(chunk.get());
        }

        return true;
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd()) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        update_output_vocab_subset();

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        // --spec-async: hand the drafts of this batch to the draft worker, which evaluates them on the draft context while the target decodes
        if (spec && off == 0 && !spec_async_reqs.empty()) {
            common_speculative_async_start(spec.get(), spec_async_reqs);
            spec_async_reqs.clear();
        }
        fwd_wd_start_once();
        fwd_wd.begin(fwd_wd_now_ms());
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
        fwd_wd.beat(fwd_wd_now_ms()); // a step returned (ok or not): the loop is alive
        fwd_wd.end();
        if (ret != 0 && spec) {
            common_speculative_async_cancel(spec.get()); // no process() call will join the worker on this path
        }

        if (ret == 0 && params_base.restrict_output_vocab) {
            // always-on evidence that the restriction reached the graph: logged when the number of rows the decode multiplied by changes
            const int32_t used = llama_output_vocab_subset_used(ctx_tgt);
            if (used != output_subset_logged) {
                SRV_INF("--restrict-output-vocab: the decode multiplied the output weight by %d of %d vocabulary entries%s\n", used > 0 ? used : llama_vocab_n_tokens(vocab),
                        llama_vocab_n_tokens(vocab), used > 0 ? "" : " (full projection)");
                output_subset_logged = used;
            }
        }

        server_decode_fail_latch_global().note_decode(ret);

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // Upstream's TODO here ("terminate only the largest active slot/sequence and continue with
                    // the rest; need to remove the tokens from the current batch too") -- done for --kv-unified,
                    // where one shared pool makes every in-flight request fail together otherwise. MEASURED
                    // 2026-09-27: 4 of 4 concurrent requests got HTTP 500 where per-slot KV served all 4.
                    if (preempt_largest_slot(off)) {
                        n_batch = llama_n_batch(ctx_tgt);
                        return false; // retry what remains of the batch, now with the victim's cells freed
                    }
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch.view.size(), has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // Cross-slot concurrent prefix sharing (SGLang RadixAttention gap port,
        // S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-2026-09-26-01): publish a slot's just-completed
        // prompt into the shared prompt cache THE MOMENT prefill finishes, instead of only when
        // this slot is later chosen for eviction by get_available_slot(). This lets a DIFFERENT,
        // concurrently-admitted request that shares this prefix (e.g. a common RAG system+context
        // preamble) find and reuse it via the existing prompt_load() path in get_available_slot()
        // WHILE this slot is still generating -- previously that path only ever saw prompts saved
        // from slots that had already gone idle/been overwritten.
        // Placement: this loop runs AFTER the per-token output-processing loop above has finished
        // for this decode step, the same safe post-decode phase the `n_cmpl > 1` fan-out below
        // already uses for its own KV op (`copy_state_to` / `seq_cp`) -- MEASURED: calling
        // prompt_save() (llama_state_seq_get_data_ext) INLINE at the DONE_PROMPT transition inside
        // the per-token output loop instead crashed the server under concurrent load
        // (`GGML_ASSERT(kv_self.cells[kv_self.head].pos == batch.pos[0])`,
        // test_completion_parallel_slots[2-2] and [4-6], reproduced on the patched build, absent
        // on the unpatched build with the identical test) -- the KV cache is not in a safely
        // readable state mid-batch while other slots' outputs in the same batch are still being
        // processed. alloc()'s own exact-duplicate check (server-task.cpp:1741-1748) makes this
        // safe to call every tick a slot remains at DONE_PROMPT, not just once.
        // N-way sharing: server_prompt_cache::load() is NON-consuming (it keeps the entry and moves
        // it to the LRU tail), so any number of concurrent siblings can reuse one published prefix
        // -- S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-NCONSUMER-FOLLOWUP-2026-09-27-01. Before that it
        // erased the entry it served and only ONE sibling could consume a given prefix.
        // COST GATE (S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-2026-09-26-01, parent review finding:
        // benefit was measured, cost was not). MEASURED single-slot overhead (CPU MinGW build-cpu,
        // -t 8, Qwen2.5-0.5B-Instruct-Q8_0, 5 sequential non-sharing requests x3 reps, base
        // c3d1c748 vs this branch): at ~8K-token prompts (n_parallel=1) branch prompt_ms mean
        // 22272.8 vs base 22354.4 (branch -82ms, i.e. WITHIN the ~1800ms rep-to-rep noise band) --
        // publish cost at this size is noise, not signal, so the default stays enabled. At
        // ~3.8K-token prompts the same comparison showed base 9216.7 vs branch 9493.9 (+277ms,
        // +3.0%, on a tighter ~960ms noise band) -- a real but small always-serial cost when
        // n_parallel==1, which is exactly why n_parallel==1 is excluded unconditionally below
        // regardless of the flag: with only one slot, no sibling can ever consume a published
        // prefix, so every publish in that mode is pure waste with zero possible benefit. Full
        // numbers: tmp/_agent_ledger_sglang-cost.md.
        if (prompt_cache && params_base.cache_publish_on_prefill && params_base.n_parallel > 1) {
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.prompt_save(*prompt_cache)) {
                        prompt_cache->update();
                    }
                }
            }
        }

        // --warm-prefix-file: a warm-up task has just ended. Save its slot's state to the host prompt cache and pin it. Done here, in the
        // same post-decode phase as the publish loop above, because reading a slot's KV inline while other slots are mid-batch crashed
        // the server (measured, see the comment on that loop); an idle slot at this point is safe.
        pin_finished_warm_slots();

        // --radix-cache: publish this slot's just-completed PROMPT (never its generation -- see
        // server-radix-cache.h's scope note on why generation is excluded, and this file's
        // slot-context-shift block ~4209-4238 which only ever fires during the GENERATION loop,
        // i.e. strictly after this point for this request) into the on-device radix tree at the
        // same safe post-decode phase prompt_save() above already uses (not inline at the
        // DONE_PROMPT transition -- this file already measured that crash class for prompt_save(),
        // see the comment on that loop). slot.radix_inserted dedups across ticks a slot lingers at
        // DONE_PROMPT (insert() itself has no such dedup).
        if (radix_cache) {
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && !slot.radix_inserted && !slot.prompt.tokens.has_mtmd) {
                    radix_cache->insert(slot.prompt.tokens.get_tokens(), slot.id, slot.prompt.lora_fp, slot.task ? slot.task->params.priority : 0);
                    slot.radix_inserted = true;
                    slot.radix_pub_len  = slot.prompt.tokens.size();
                }
                // A slot that finished its request: publish prompt + generation. If a new task already took the slot the flag is stale.
                if (slot.radix_gen_pending) {
                    slot.radix_gen_pending = false;
                    if (!slot.is_processing()) {
                        const llama_pos pm = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
                        const size_t n_pub = server_radix_gen_publish_len(slot.prompt.tokens.size(), slot.radix_gen_base, pm, false, slot.prompt.tokens.has_mtmd);
                        if (n_pub > 0) {
                            const auto & all_toks = slot.prompt.tokens.get_tokens();
                            radix_cache->insert(llama_tokens(all_toks.begin(), all_toks.begin() + n_pub), slot.id, slot.prompt.lora_fp, 0);
                        }
                    }
                }
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);
                    // children with their own prompt are started at the end of the shared prefix
                    GGML_ASSERT(slot.task->n_tokens_shared == 0);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside the current sub-batch
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        // --kv-audit-interval (debug, log-only): canary chain per slot, KV cell count vs slot tokens (one-sided), non-finite logits.
        // Boundary: the chain covers the host token view of each slot, not K/V bytes (see server-kv-audit.h).
        const bool kv_audit_now = params_base.kv_audit_interval > 0 && (++kv_audit_step % (uint64_t) params_base.kv_audit_interval) == 0;
        if (kv_audit_now) {
            if (kv_audit_chains.size() < slots.size()) {
                kv_audit_chains.resize(slots.size());
            }
            int64_t sum_tokens = 0;
            size_t  k = 0;
            for (auto & slot : slots) {
                const auto & toks = slot.prompt.tokens.get_text_tokens();
                sum_tokens += (int64_t) toks.size();
                const int64_t owner = slot.task ? (int64_t) slot.task->id : -1;
                const int64_t bad = kv_audit_chains[k++].audit(owner, std::vector<int32_t>(toks.begin(), toks.end()));
                if (bad >= 0) {
                    SRV_ERR("%s\n", kv_audit_describe(slot.id, bad, "token no longer matches the stamped canary chain").c_str());
                }
            }
            const auto leak = kv_audit_pool_check((int64_t) llama_get_kv_cache_used_cells(ctx_tgt), sum_tokens, (int64_t) n_batch_tokens);
            if (leak.leak) {
                SRV_ERR("kv-audit: %lld KV cells are in use that no slot accounts for (cells %lld, slot tokens %lld)\n",
                        (long long) leak.unowned, (long long) llama_get_kv_cache_used_cells(ctx_tgt), (long long) sum_tokens);
            }
        }

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special || !slot.task->params.skip_special_tokens ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
                // prompt_logprobs: read this sub-batch's prompt rows while their logits are still in the output buffer
                if (slot.task->params.prompt_logprobs) {
                    extract_prompt_logprobs(slot, batch.view);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_DECISION) {
                    send_decision(slot, batch.view, slot.i_batch - off);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            if (kv_audit_now) {
                const float * arow = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                if (arow) {
                    const int64_t bad = kv_audit_first_nonfinite(arow, (int64_t) llama_vocab_n_tokens(vocab));
                    if (bad >= 0) {
                        SRV_ERR("%s\n", kv_audit_describe(slot.id, (int64_t) slot.prompt.tokens.size(), ("non-finite logit at token id " + std::to_string((long long) bad)).c_str()).c_str());
                    }
                }
            }

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                // the sampler reads the text so far (expiring logit bias); a rewind above has changed ban_buffer
                sampler_feed_text(slot);
                if (slot.ban_n_hold > 0) {
                    ban_before_sample(slot, tok_idx);
                }
                std::vector<std::pair<llama_token, float>> min_tokens_masked;
                if (slot.min_tokens_pending()) {
                    // mask the end-of-generation tokens and this request's stop_token_ids in the raw row, before any sampler
                    float * row = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                    auto mask = [&](llama_token t) {
                        if (std::isfinite(row[t])) {
                            min_tokens_masked.emplace_back(t, row[t]);
                        }
                        row[t] = -INFINITY;
                    };
                    for (const auto & b : params_base.sampling.logit_bias_eog) {
                        mask(b.token);
                    }
                    for (const llama_token t : slot.task->params.stop_token_ids) {
                        mask(t);
                    }
                    if (!slot.task->params.sampling.grammar.empty()) {
                        common_sampler_set_restore_if_empty(slot.smpl.get(), min_tokens_masked);
                    }
                }
                if (!slot.task->params.tool_call_fix_ids.empty() && slot.stats.n_gen < 3 &&
                        std::count(slot.generated.text.begin(), slot.generated.text.end(), '[') <= 1) {
                    // tool_call_fix: no ']' among the first 3 tokens, so the reply cannot open as an empty array
                    float * row = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                    for (const llama_token t : slot.task->params.tool_call_fix_ids) {
                        row[t] = -INFINITY;
                    }
                }
                if (slot.task->params.heal_prefix_token != LLAMA_TOKEN_NULL && slot.stats.n_gen == 0) {
                    // token healing: the first sampled token must begin with the text of the prompt token that was removed.
                    // Masked in the raw row, before any sampler, like min_tokens above.
                    const auto & pieces = token_pieces();
                    const std::string & pre = pieces[slot.task->params.heal_prefix_token];
                    float * row = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                    size_t n_allowed = 0;
                    for (size_t t = 0; t < pieces.size(); ++t) {
                        if (pieces[t].compare(0, pre.size(), pre) == 0) {
                            n_allowed++;
                        } else {
                            row[t] = -INFINITY;
                        }
                    }
                    SLT_DBG(slot, "token healing: prefix '%s', %zu of %zu tokens allowed for the first sample\n", pre.c_str(), n_allowed, pieces.size());
                }
                allow_apply(slot);
                // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01):
                // id_cfg >= 0 -> ctx_cfg is the SAME context (ctx_tgt), idx_cfg is the guidance
                // seq's own batch index from this decode step, SHIFTED by `off` exactly like
                // tok_idx above -- slot.i_batch_cfg is a GLOBAL batch index (set the same way
                // slot.i_batch is, in handle_last_sampled_token), and llama_get_logits_ith reads
                // relative to the just-decoded batch.view, not the whole unchunked batch. Missing
                // this shift was caught before any build: it would have read whatever token
                // happened to sit at that raw offset in the sub-batch -- wrong logits, not a crash,
                // so nothing would have failed loudly.
                // id_cfg == -1 (every slot on a server with --cfg-guidance-slots 0, and every slot
                // outside the safe subset, which is refused before reaching here) passes
                // ctx_cfg=nullptr, byte-identical to the call before this row existed.
                struct llama_context * ctx_cfg = nullptr;
                int idx_cfg = -1;
                if (slot.id_cfg >= 0) {
                    // INVARIANT, enforced above in update_slots()'s chunking loop
                    // (server_batch::token::pairs_with_next): a slot's main token and its
                    // guidance token are never split across sub-batches. This is therefore a
                    // should-never-happen defensive check, not a normal-operation fallback --
                    // coordinator review, 2026-09-27 explicitly rejected a silent unguided sample
                    // as "the same defect class" as the original silently-ignored-fields bug, so
                    // if this invariant is ever violated (a latent bug in the chunking guard),
                    // fail the request loudly (caught by update_slots()'s existing try/catch,
                    // which aborts the affected slot with a clear error) rather than degrade.
                    if (!is_inside_view(slot.i_batch_cfg)) {
                        throw std::runtime_error("classifier-free guidance invariant violated: "
                                "guidance token was not in the same sub-batch as its main token");
                    }
                    ctx_cfg = slot.ctx_tgt;
                    idx_cfg = slot.i_batch_cfg - off;
                }
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx, /* grammar_first= */ false,
                        ctx_cfg, idx_cfg);
                if (!min_tokens_masked.empty()) {
                    common_sampler_set_restore_if_empty(slot.smpl.get(), {});
                }
                // The request's grammar/choice allows no token of this vocabulary: a client error for THIS request.
                // It used to reach GGML_ABORT in the sampler and kill every slot on the server.
                // Row S69-FORK-UNSATISFIABLE-GRAMMAR-ABORTS-SERVER-SAMPLING-839-2026-10-11-01.
                if (id == LLAMA_TOKEN_NULL) {
                    slot.i_batch     = -1;
                    slot.i_batch_cfg = -1;
                    SLT_WRN(slot, "%s", "grammar/choice allows no token of this vocabulary; failing the request\n");
                    send_error(slot, "grammar/choice cannot be satisfied by this model's vocabulary (no allowed token "
                            "at generated position " + std::to_string(slot.stats.n_gen + 1) + ")",
                            ERROR_TYPE_INVALID_REQUEST);
                    slot.release();
                    return;
                }
                // Refuse a token drawn from a row the MODEL corrupted (NaN/+inf in the raw logits, counted before
                // any bias). Every sampler downstream maps a non-finite logit to probability 0 or loses the
                // comparison, so without this the request keeps "succeeding" with garbage and nothing reports it.
                // Same convention as the CFG invariant above: update_slots()'s try/catch aborts this slot with
                // the message. Row S69-mistralrs-PICK-SAMPLER-FAIL-LOUD-NONFINITE-LOGITS-2026-09-28-01.
                if (const int32_t n_bad = common_sampler_n_nonfinite_last(slot.smpl.get()); n_bad > 0) {
                    throw std::runtime_error(string_format(
                            "the model produced %d non-finite logit(s) (NaN/+inf) at generated token %d; "
                            "refusing to sample from a corrupted row", n_bad, (int) slot.stats.n_gen + 1));
                }
            }

            slot.i_batch     = -1;
            slot.i_batch_cfg = -1;

            common_sampler_accept(slot.smpl.get(), slot.ctx_tgt, id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();
            metrics_observe_itl(slot);

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.heal_prefix_token != LLAMA_TOKEN_NULL && slot.stats.n_gen == 1) {
                // the healed token re-states the removed prompt token's text: that text is already in the prompt, drop it from the output
                const std::string & pre = token_pieces()[slot.task->params.heal_prefix_token];
                if (result.text_to_send.compare(0, pre.size(), pre) == 0) {
                    result.text_to_send.erase(0, pre.size());
                }
            }

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            // banned strings: the token is buffered and may be rewound instead of sent
            if (slot.ban_n_hold > 0) {
                if (!ban_step(slot, std::move(result))) {
                    slot.print_timings_tg();
                }
                return;
            }

            if (!process_token(result, slot) || !jump_forward(slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // batch indices of the draft tokens, used to get the token probabilities below
            auto spec_i_batch = std::move(slot.spec_i_batch);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // expiring logit bias: the sampler reads the text so far. Inside the run below it appends each accepted
                // piece to playing_text itself (common_sampler_accept), so this only seeds it. Drafting is off while banned
                // strings are active (get_n_draft_max), so ban_buffer is empty here, and the same call stays correct if not.
                sampler_feed_text(slot);
                // allowlist: one stage for the whole verified run (keywords inside a draft advance it afterwards,
                // as in the old server, which applied the stage once per speculative step at :4305)
                allow_apply(slot);

                // two spellings of one idea, each with the distribution its drafter recorded (row
                // S69-ENGREV-UPSTREAM-MERGE-DRAFT-SAMPLE-EQUIVALENCE-TEST-2026-11-04-01): the fork's --spec-draft-sample q per
                // drafted token (common_spec_dist), and upstream's candidate lists (spec_draft_q, filled by draft-simple /
                // draft-mtp / DFlash2 when the drafter samples). Drafters that fill neither fall back to sample-and-match.
                const bool use_rejection_dist = slot.spec_dists.size() == n_draft && !slot.spec_is_replay &&
                                                !slot.spec_sampler_is_stateful(); // G4, see use_spec_rejection()
                const bool use_rejection      = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (use_rejection_dist) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch,
                                                                           slot.spec_draft, slot.spec_dists, slot.spec_rng);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                }
                SLT_DBG(slot, "spec verify: %s (dists=%zu, draft_q=%zu, n_draft=%zu, replay=%d)\n",
                        !synth_probs.empty() ? "synthetic" : use_rejection_dist ? "rejection (spec-draft-sample)" :
                        (slot.spec_is_replay && slot.use_spec_rejection()) ? "replay" : use_rejection ? "rejection (draft_q)" : "sample-and-match",
                        slot.spec_dists.size(), slot.spec_draft_q.size(), n_draft, (int) slot.spec_is_replay);
                if (trace > 0) {
                    // LLAMA_TRACE: what was drafted, what was accepted, and the raw logits of the first row that did
                    // not accept. A lossy temperature-0 verify is either wrong logits (model / state) or a sampler or
                    // row-index fault; the argmax printed here next to the sampled token tells the two apart.
                    std::string sd, sa;
                    for (llama_token t : slot.spec_draft) sd += std::to_string(t) + " ";
                    for (llama_token t : accepted) sa += std::to_string(t) + " ";
                    SLT_INF(slot, "spec verify ids: draft [%s] accepted+sampled [%s]\n", sd.c_str(), sa.c_str());
                    const size_t r = accepted.size() - 1;
                    if (r < spec_i_batch.size()) {
                        const float * lg = llama_get_logits_ith(slot.ctx_tgt, spec_i_batch[r]);
                        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(slot.ctx_tgt)));
                        if (lg != nullptr) {
                            int a1 = 0, a2 = -1;
                            for (int v = 1; v < n_vocab; ++v) {
                                if (lg[v] > lg[a1]) { a2 = a1; a1 = v; }
                                else if (a2 < 0 || lg[v] > lg[a2]) { a2 = v; }
                            }
                            SLT_INF(slot, "spec verify row %zu (batch idx %d): sampled %d, logits argmax %d (%.4f), 2nd %d (%.4f), draft %d\n",
                                    r, (int) spec_i_batch[r], (int) accepted.back(), a1, (double) lg[a1], a2,
                                    a2 >= 0 ? (double) lg[a2] : 0.0, r < slot.spec_draft.size() ? (int) slot.spec_draft[r] : -1);
                        }
                    }
                }
                slot.spec_dists.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                // the recurrent snapshots are only guaranteed for batches of up to n_rs_seq + 1 tokens, so a longer draft uses its checkpoint
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                bool restored_direct = false;
                if (n_rollback > 0 && use_ckpt_tgt && slot.spec_ckpt_dev) {
                    // on-device checkpoint: per-step restores the state after the last accepted token in place (DIRECT, the
                    // ordinary accept path below then runs as for a context that can roll back by itself); gpu-fallback restores
                    // the pre-draft state and the accepted tokens are replayed as for the host copy below
                    const auto & ckpt = slot.spec_ckpt;
                    const int step = (int) accepted.size() - 1;
                    const llama_pos pm_before = llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id);
                    const llama_spec_ckpt_restore_result res =
                        llama_spec_ckpt_restore_ex(slot.ctx_tgt, server_spec_ckpt, slot.id, ckpt.pos_max + 1, step);
                    if (trace > 0) {
                        SLT_INF(slot, "on-device restore: n_past %d step %d, pos_max before %d after %d, prompt n_tokens %d, n_draft %zu\n",
                                (int) ckpt.pos_max + 1, step, (int) pm_before,
                                (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id),
                                (int) slot.prompt.n_tokens(), slot.spec_draft.size());
                    }
                    if (res == LLAMA_SPEC_CKPT_RESTORE_FAILED) {
                        GGML_ABORT("slot %d: on-device speculative checkpoint restore failed (step %d, pos_max %d)\n",
                                slot.id, step, (int) ckpt.pos_max);
                    }
                    if (res == LLAMA_SPEC_CKPT_RESTORE_DIRECT) {
                        // MEASURED (pa2 trace, Qwen3.5-0.8B MTP): the restore rewinds ctx_tgt only (pos_max 52 -> 50), but the MTP
                        // draft context is a context of its own over the same recurrent cache and still holds the rejected tail,
                        // so the slot.mem.seq_rm below aborted on it ("failed to remove sequence 0 with p0=51"). Treat the draft
                        // context exactly as the host-copy path does: back to its pre-draft copy, the drafter re-evaluates from there.
                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                            if (!llama_memory_seq_rm(llama_get_memory(slot.ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                                GGML_ABORT("slot %d: draft context cannot drop positions from %d after an on-device restore\n",
                                        slot.id, (int) ckpt.pos_max + 1);
                            }
                        }
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (on-device checkpoint, direct)\n", accepted.size() - 1, slot.spec_draft.size());
                        }
                        llama_spec_ckpt_discard(slot.ctx_tgt, server_spec_ckpt);
                        slot.spec_ckpt_dev = false;
                        restored_direct = true;
                    }
                }
                if (n_rollback > 0 && !restored_direct) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint%s)\n", accepted.size() - 1, slot.spec_draft.size(),
                                    slot.spec_ckpt_dev ? ", on-device, replay" : "");
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        // the on-device one was restored above (and is kept until the replay round accepts in full)
                        if (!slot.spec_ckpt_dev) {
                            GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0 && !restored_direct) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                // the round is over without a rollback that needs the saved state (full acceptance, a finished replay, or a
                // rollback the context did by itself): stop the per-step recording and drop the on-device checkpoint
                if (slot.spec_ckpt_dev) {
                    llama_spec_ckpt_discard(slot.ctx_tgt, server_spec_ckpt);
                    slot.spec_ckpt_dev = false;
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();
            metrics_observe_itl(slot);

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        // rope scaling (--rope-scale) extends the context the model reads: the training context times the factor
        int64_t n_train = llama_model_n_ctx_train(model_tgt);
        if (decision.type != COMMON_DECISION_TYPE_NONE && params_base.rope_freq_scale > 0.0f && params_base.rope_freq_scale < 1.0f &&
            params_base.rope_scaling_type != LLAMA_ROPE_SCALING_TYPE_NONE) {
            n_train = (int64_t) (n_train / params_base.rope_freq_scale + 0.5f);
        }
        return (int) std::min<int64_t>(res, n_train);
    }

    // the largest prompt a decision takes in one pass: the slot context, and for a non-causal model also the physical batch
    size_t decision_n_fit() const {
        size_t n_fit = (size_t) n_ctx_slot();
        if (!llama_get_causal_attn(ctx_tgt)) {
            n_fit = std::min(n_fit, (size_t) llama_n_ubatch(ctx_tgt));
        }
        return n_fit;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        if (prompt_cache) {
            // the NVMe tier's restore-versus-recompute rule needs the speed at which this server really processes prompts
            prompt_cache->note_prefill(n_prompt_queued, (ggml_time_us() - t_prompt_start) / 1e6);
        }
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            // Classifier-Free Guidance: a guidance-seq token's id_slot is slot.id_cfg
            // (params_base.n_parallel + i, row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-
            // 2026-09-26-01), NOT an index into `slots` (sized to n_parallel only) -- indexing
            // slots[t.id_slot] for one read/wrote past the vector's end (UB, intermittent
            // SIGSEGV depending on heap layout: row
            // S69-SERVER-CFG-PROMPT-CACHE-REUSE-CRASH-2026-09-28-01, backtrace pinned to this
            // exact line, t.id_slot == 1 with n_parallel == 1). The guidance token's own
            // processing time has nowhere per-slot to attribute anyway (it belongs to the main
            // slot's paired token, already counted via that entry), so just skip it here.
            if (t.id_slot < 0 || (size_t) t.id_slot >= slots.size()) {
                continue;
            }

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);
        queue_tasks.add_prefill_tokens(n_prompt_tokens); // --sched-policy hrrn aging

        // step composition: what this step carried, and how many generations were eligible vs actually in the batch.
        // Counted over the batch actually submitted (batch.view), after any retry with a smaller n_batch.
        {
            uint64_t n_dec_tokens = 0;
            uint64_t n_pre_tokens = 0;
            uint64_t n_admitted   = 0;
            std::vector<uint8_t> seen(slots.size(), 0);
            for (int i = off; i < off + n_tokens; ++i) {
                const auto & t = batch.tokens[i];
                if (t.is_prompt) {
                    n_pre_tokens++;
                    continue;
                }
                n_dec_tokens++;
                if (t.id_slot >= 0 && (size_t) t.id_slot < slots.size() && !seen[t.id_slot]) {
                    seen[t.id_slot] = 1;
                    n_admitted++;
                }
            }
            uint64_t n_ready = 0;
            for (const auto & slot : slots) {
                if (slot.state == SLOT_STATE_GENERATING) {
                    n_ready++;
                }
            }
            metrics.lat.steps.add_session_counts(n_ready, n_admitted);
            step_group.add_step(t_decode_start, n_dec_tokens, n_pre_tokens);
        }

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
            note_step_group(step_group, ggml_time_us());
            metrics.lat.steps.close_group(step_group, ggml_time_us());
            step_group.clear();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            // Classifier-Free Guidance: same out-of-bounds hazard as the loop above -- t.id_slot
            // can be a guidance seq id (>= slots.size()), not a real slot index.
            if (t.id_slot < 0 || (size_t) t.id_slot >= slots.size()) {
                continue;
            }

            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
        metrics.lat.steps.close_group(step_group, ggml_time_us());
        step_group.clear();
    }

    // one observation per token-emitting step: the gap since this request's previous emission
    void metrics_observe_itl(const server_slot & slot) {
        if (slot.stats.itl_last_us >= 0) {
            metrics.lat.itl.observe((double) slot.stats.itl_last_us / 1e6);
        }
    }

    // a generation request is leaving its slot (finished, stopped, cancelled or failed): book its stage times and
    // the reason. Called from callback_on_reset, before the slot's stats are cleared.
    // the key a request goes by in the request log: X-Request-Id / trace id when the client sent one, else the task id
    static std::string request_log_key(const server_task & task) {
        return task.params.request_id.empty() ? "task-" + std::to_string(task.id) : task.params.request_id;
    }

    void metrics_on_request_done(const server_slot & slot) {
        const std::string log_key = (metrics.reqlog && slot.task) ? request_log_key(*slot.task) : std::string();
        if (slot.stats.n_gen == 0 || slot.stats.t_start == 0) {
            if (!log_key.empty()) {
                metrics.reqlog->end(log_key); // retire the in-flight entry of a request that never produced a token
            }
            return; // embedding / rerank / a request that never produced a token
        }
        server_request_times rt;
        rt.t_submit      = slot.stats.t_submit;
        rt.t_start       = slot.stats.t_start;
        rt.t_first_token = slot.stats.t_first_token;
        rt.t_last_token  = slot.stats.t_gen_last;

        auto obs = [](server_histogram & h, double s) {
            if (s >= 0.0) {
                h.observe(s);
            }
        };
        obs(metrics.lat.queue,   rt.queue_s());
        obs(metrics.lat.prefill, rt.prefill_s());
        obs(metrics.lat.ttft,    rt.ttft_s());
        obs(metrics.lat.decode,  rt.decode_s());
        obs(metrics.lat.e2e,     rt.e2e_s());
        metrics.lat.n_prompt_tokens.observe((double) (slot.stats.n_prompt_cached + slot.stats.n_prompt_processed));
        metrics.lat.n_gen_tokens   .observe((double) slot.stats.n_gen);
        // how long a request holds a slot, for the Retry-After of a "no free slot" 503 (server-retry.h)
        server_retry_global().observe_request_ms((slot.stats.t_gen_last - slot.stats.t_start) / 1000);
        // --shed-max-latency-ms: one more (tokens, service ms) point for the fit. Only tokens this request really processed count
        // (a prompt served from the cache cost nothing); a request cut short still measured what it did.
        if (params_base.shed_max_latency_ms > 0) {
            shed_fit.add((int64_t) (slot.stats.n_prompt_processed + slot.stats.n_gen), (double) (slot.stats.t_gen_last - slot.stats.t_start) / 1000.0, ggml_time_us());
        }

        // vLLM vocabulary: stop (the model or a stop string ended it), length (hit the limit), abort (no stop
        // condition was reached: cancelled, disconnected or failed)
        const char * reason = "abort";
        switch (slot.stop) {
            case STOP_TYPE_EOS:
            case STOP_TYPE_WORD:
            case STOP_TYPE_REPETITION: reason = "stop";   break;
            case STOP_TYPE_LIMIT:      reason = "length"; break;
            case STOP_TYPE_DEADLINE:   reason = "deadline"; break;
            case STOP_TYPE_GRAMMAR_ERROR: reason = "grammar_error"; break;
            case STOP_TYPE_BUDGET_GROUP:  reason = "budget_group"; break;
            default: break;
        }
        metrics.lat.finished.add(reason);

        if (!log_key.empty()) {
            server_request_record rec;
            rec.request_id = log_key;
            rec.trace_id   = slot.task->params.trace_id;
            rec.model      = model_name;
            rec.budget_group = slot.task->params.budget_group;
            rec.status     = reason;
            rec.n_prompt   = (int64_t) (slot.stats.n_prompt_cached + slot.stats.n_prompt_processed);
            rec.n_gen      = (int64_t) slot.stats.n_gen;
            rec.queue_ms   = std::max(0.0, rt.queue_s())   * 1e3;
            rec.prefill_ms = std::max(0.0, rt.prefill_s()) * 1e3;
            rec.decode_ms  = std::max(0.0, rt.decode_s())  * 1e3;
            rec.ttft_ms    = std::max(0.0, rt.ttft_s())    * 1e3;
            rec.e2e_ms     = std::max(0.0, rt.e2e_s())     * 1e3;
            rec.n_draft    = (int64_t) slot.stats.n_draft_tokens;
            rec.n_accept   = (int64_t) slot.stats.n_draft_accepted;
            rec.t_unix_ms  = (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            metrics.reqlog->log(rec);
            metrics.reqlog->end(log_key);
        }
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;
        metrics.n_jump_forward_tokens += slot.stats.n_jump_forward_tokens;

        if (spec) {
            // cumulative inside the drafter already: copy, do not add
            metrics.spec_async_on = common_speculative_async_get_stats(spec.get(), metrics.spec_async);
        }

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

// the memory breakdown rows as JSON; bytes are exact, *_mib is bytes / 1048576 rounded down (what the log table prints)
static json memory_breakdown_to_json(const std::vector<common_memory_breakdown_row> & rows) {
    constexpr int64_t MiB = 1024 * 1024;
    auto fill = [&](const common_memory_breakdown_row & r) {
        json j = {
            {"name",          r.name},
            {"model_bytes",   r.model},
            {"context_bytes", r.context},
            {"compute_bytes", r.compute},
            {"self_bytes",    r.self()},
            {"model_mib",     (int64_t) r.model   / MiB},
            {"context_mib",   (int64_t) r.context / MiB},
            {"compute_mib",   (int64_t) r.compute / MiB},
            {"self_mib",      (int64_t) r.self()  / MiB},
        };
        if (r.kind == common_memory_breakdown_row::DEVICE) {
            j["description"]           = r.description;
            j["total_bytes"]           = r.total;
            j["free_bytes_at_load"]    = r.free;
            j["unaccounted_bytes"]     = r.unaccounted;
            j["total_mib"]             = r.total / MiB;
            j["free_mib_at_load"]      = r.free  / MiB;
            j["unaccounted_mib"]       = r.unaccounted / MiB;
        }
        return j;
    };
    json out = {
        {"devices", json::array()},
        {"host",    json::object()},
        {"other",   json::array()},
    };
    for (const auto & r : rows) {
        switch (r.kind) {
            case common_memory_breakdown_row::DEVICE: out["devices"].push_back(fill(r)); break;
            case common_memory_breakdown_row::HOST:   out["host"] = fill(r);              break;
            default:                                  out["other"].push_back(fill(r));    break;
        }
    }
    return out;
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    // "general.architecture" for the ollama-native API shim's /api/tags|show|ps "family" field
    // (row S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01). llama_model_meta_val_str
    // returns -1 when the key is absent; leave model_arch empty in that case rather than embedding
    // a truncated/garbage buffer.
    char arch_buf[128] = {0};
    std::string model_arch;
    if (llama_model_meta_val_str(impl->model_tgt, "general.architecture", arch_buf, sizeof(arch_buf)) >= 0) {
        model_arch = arch_buf;
    }

    uint64_t mem_host = 0, mem_dev = 0;
    common_memory_totals(impl->ctx_tgt, mem_host, mem_dev);

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(impl->model_tgt, impl->has_cap_chat(), impl->has_cap_tts()),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* has_cap_chat            */ impl->has_cap_chat(),
        /* has_cap_tts             */ impl->has_cap_tts(),
        /* has_audio_stream        */ impl->mtmd_cap.audio_stream,
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* mem_host_bytes          */ mem_host,
        /* mem_device_bytes        */ mem_dev,
        /* n_ubatch                */ (int32_t) llama_n_ubatch(impl->ctx_tgt),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
        /* model_arch             */ model_arch,
        /* memory_breakdown       */ memory_breakdown_to_json(common_memory_breakdown_collect(impl->ctx_tgt)),
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    std::unique_ptr<server_admission_ticket> admission; // set only for an admitted request under a ceiling
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = error_http_status(error_data, 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

bool server_admission::try_admit(int64_t prompt_tokens, std::string & why) {
    std::lock_guard<std::mutex> lock(mtx);
    if (max_reqs > 0 && n_reqs >= max_reqs) {
        why = string_format("server busy: %d request(s) in flight, the limit is %d (--max-queued-requests); retry later",
                            n_reqs, max_reqs);
    } else if (max_tokens > 0 && n_tokens > 0 && n_tokens + prompt_tokens > max_tokens) {
        why = string_format("server busy: %lld prompt token(s) already waiting for prefill, this request adds %lld and the "
                            "limit is %lld (--max-queued-tokens); retry later",
                            (long long) n_tokens, (long long) prompt_tokens, (long long) max_tokens);
    } else {
        n_reqs++;
        n_tokens += prompt_tokens;
        n_admitted++;
        return true;
    }
    n_rejected++;
    return false;
}

void server_admission::force_admit(int64_t prompt_tokens) {
    std::lock_guard<std::mutex> lock(mtx);
    n_reqs++;
    n_tokens += prompt_tokens;
    n_admitted++;
    n_evicted++;
    if (n_rejected > 0) {
        n_rejected--; // try_admit() booked this arrival as refused a moment ago; it was not
    }
}

void server_admission::release(int64_t prompt_tokens, bool request) {
    std::lock_guard<std::mutex> lock(mtx);
    n_tokens -= prompt_tokens;
    if (request) {
        n_reqs--;
    }
    GGML_ASSERT(n_tokens >= 0 && n_reqs >= 0 && "admission counters released more than they took");
}

json server_admission::to_json() {
    std::lock_guard<std::mutex> lock(mtx);
    return json {
        {"max_queued_requests", max_reqs},
        {"max_queued_tokens",   max_tokens},
        {"in_flight_requests",  n_reqs},
        {"prefill_tokens",      n_tokens},
        {"admitted",            n_admitted},
        {"rejected",            n_rejected},
        {"evicted",             n_evicted},
    };
}

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

void server_context::set_model_loaded_callback(std::function<void(int n_ctx_slot)> callback) {
    impl->callback_model_loaded = std::move(callback);
}

void server_context::set_pin_token(const std::string & token) {
    impl->pin_token = token;
}

size_t server_context::pinned_bytes() const {
    return impl->pinned_bytes_a.load();
}

size_t server_context::pinned_entries() const {
    return impl->pinned_entries_a.load();
}

bool server_context::pin_overflowed() const {
    return impl->pin_overflow_a.load();
}

size_t server_context::pin_passes() const {
    return impl->pin_passes_a.load();
}

bool server_context::last_pin_covered() const {
    return impl->last_pin_covered_a.load();
}

//
// Beam search, IN-BAND (row S69-VLLM-BEAM-SEARCH-SERVER-PORT-2026-09-27-01).
//
// SOURCE-READ (Apache-2.0; reimplemented from reading, no code copied): vLLM @ 924707f1,
//   vllm/entrypoints/generate/beam_search/online.py:28-229 -- each step, every live beam asks for ONE token with the
//   top 2*width logprobs; a candidate is cum_logprob + token logprob; an EOS candidate finishes (unless ignore_eos);
//   the best `width` non-EOS candidates live on; after max_tokens, finished + live are ranked by
//   cum_logprob / max(len, 1) ** length_penalty with EOS not counted in len (utils.py:137-154).
//   vLLM refuses streaming with beam search (entrypoints/openai/completion/serving.py:130-133); so does this.
//
// WHY IN-BAND. Every step is an ordinary batch of 1-token tasks through the task queue and update_slots. Two hazard
// classes measured on the CFG port (2026-09-27) therefore cannot arise here: an out-of-band llama_decode clobbering
// another slot's pending outputs (output_reserve(), src/llama-context.cpp:2843), and extra seq ids desynced by
// checkpoint / speculative / banned-string rewinds. Prefix reuse is whatever the prompt cache already gives.
//
// Candidate logprobs are the RAW next-token distribution (n_probs with post_sampling_probs = false): temperature,
// penalties and truncation samplers do not reshape them. Invariant: width 1 == greedy decoding, byte for byte
// (tests/unit/test_beam_search.py, and temp/beam_search_oracle.py as the external reference).
struct beam_hyp {
    llama_tokens tokens;          // generated tokens only (EOS included when it ended the hypothesis)
    double       cum_logprob = 0.0;
    bool         eos         = false;
    double       score       = std::nan(""); // ct2 mode: the finalised score; NaN = rank with beam_score()
    bool         diverged    = false;        // ct2 mode: this beam has left the soft prefix
};

static double beam_score(const beam_hyp & h, double length_penalty) {
    if (!std::isnan(h.score)) {
        return h.score;
    }
    const size_t n = h.tokens.size() - (h.eos ? 1 : 0);
    return h.cum_logprob / std::pow((double) std::max<size_t>(n, 1), length_penalty);
}

// One live beam's next-token distribution as the server reports it: the top list, plus the exact answer for the ids asked for.
struct beam_raw {
    std::vector<std::pair<llama_token, double>> top;   // (token, log p) with p > 0, best first
    std::unordered_map<llama_token, double>     ids;   // logprob_token_ids answer
};

// Counters of one search, returned in the reply and logged: they are how a caller (and the tests) see which rules FIRED.
struct beam_stats {
    int steps          = 0;     // model steps (each is one batch of 1-token tasks)
    int registered     = 0;     // hypotheses finished
    int min_masked     = 0;     // steps on which the EOG tokens were removed (min_tokens)
    int bias_steps     = 0;     // steps on which at least one beam was pulled towards the prefix
    bool early_stop    = false; // the search ended before n_predict because enough hypotheses had finished
    int prefix_forced  = 0;     // tokens of a hard prefix that were forced instead of searched
};

// Every EOG token of the vocabulary (what min_tokens removes). Empty with ignore_eos: then nothing ends a hypothesis.
static std::vector<llama_token> beam_eog_ids(const llama_vocab * vocab, bool ignore_eos) {
    std::vector<llama_token> ids;
    if (ignore_eos) {
        return ids;
    }
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n; ++t) {
        if (llama_vocab_is_eog(vocab, t)) {
            ids.push_back(t);
        }
    }
    return ids;
}

// Post one 1-token task per beam (prompt + that beam's tokens) and read back each beam's raw next-token distribution.
// get_response_reader() is private to server_context_impl; the queues it wraps are public.
static bool beam_step_raw(server_context_impl & ctx_server,
                          server_task_type type,
                          const task_params & base,
                          const llama_tokens & prompt,
                          const std::vector<const llama_tokens *> & gens,
                          int n_probs,
                          const std::vector<llama_token> & want_ids,
                          const std::function<bool()> & should_stop,
                          std::vector<beam_raw> & out,
                          bool & terminated,
                          std::string & err) {
    server_response_reader rd(ctx_server.queue_tasks, ctx_server.queue_results, HTTP_POLLING_SECONDS);
    std::vector<server_task> tasks;
    std::unordered_map<int, size_t> id_to_beam;

    for (size_t b = 0; b < gens.size(); ++b) {
        server_task task(type);
        task.id = rd.get_new_id();

        llama_tokens toks = prompt;
        toks.insert(toks.end(), gens[b]->begin(), gens[b]->end());
        task.tokens = server_tokens(toks, false);

        task.params                     = base;
        task.params.stream              = false;
        task.params.n_predict           = 1;
        task.params.n_cmpl              = 1;
        task.params.min_tokens          = 0;     // the beam loop applies min_tokens itself, over the whole hypothesis
        task.params.post_sampling_probs = false;
        task.params.sampling.n_probs    = n_probs;
        task.params.sampling.temp       = 0.0f; // the sampled token is unused; only the raw distribution is read
        task.params.logprob_token_ids   = want_ids;
        task.params.res_type            = TASK_RESPONSE_TYPE_NONE;

        id_to_beam[task.id] = b;
        tasks.push_back(std::move(task));
    }

    rd.post_tasks(std::move(tasks));
    auto all = rd.wait_for_all(should_stop);
    if (all.is_terminated) {
        terminated = true;
        return false;
    }
    if (all.error) {
        err = all.error->to_json().dump();
        return false;
    }

    out.assign(gens.size(), beam_raw());
    for (auto & r : all.results) {
        auto * fin = dynamic_cast<server_task_result_cmpl_final *>(r.get());
        if (fin == nullptr) {
            err = "beam search: a step result is not a final completion result";
            return false;
        }
        auto it = id_to_beam.find(fin->id);
        if (it == id_to_beam.end()) {
            err = "beam search: result for a task id this step did not post";
            return false;
        }
        if (fin->probs_output.empty()) {
            err = "beam search: the step returned no token probabilities";
            return false;
        }
        beam_raw & br = out[it->second];
        for (const auto & p : fin->probs_output[0].probs) {
            if (p.prob > 0.0f) {
                br.top.emplace_back(p.tok, std::log((double) p.prob));
            }
        }
        for (const auto & kv : fin->probs_output[0].id_logprobs) {
            br.ids[kv.first] = (double) kv.second;
        }
        if (!want_ids.empty() && br.ids.size() != want_ids.size()) {
            err = "beam search: the step did not return every requested token log-probability";
            return false;
        }
    }
    return true;
}

// min_tokens, CTranslate2's way (src/decoding.cc:384-408 apply_min_length + :494-498 disable_tokens.apply(), then LogSoftMax):
// while step < min_tokens the EOG tokens' logits are -inf BEFORE the softmax, so the other tokens' probabilities are renormalised.
// The EOG tokens' total mass comes from the exact log-probabilities asked for in logprob_token_ids.
// Returns the distribution to search over and sets log_keep = log(1 - mass(EOG)) (0 when nothing was removed).
static std::vector<std::pair<llama_token, double>> beam_mask_eog(const beam_raw & r,
                                                                 const std::vector<llama_token> & eog,
                                                                 bool mask,
                                                                 double & log_keep) {
    log_keep = 0.0;
    if (!mask || eog.empty()) {
        return r.top;
    }
    double mass = 0.0;
    for (llama_token t : eog) {
        auto it = r.ids.find(t);
        if (it != r.ids.end()) {
            mass += std::exp(it->second);
        }
    }
    if (!(mass < 1.0 - 1e-12)) {
        return {};   // every bit of the probability is an EOG token: nothing can continue the hypothesis
    }
    log_keep = std::log1p(-mass);
    std::vector<std::pair<llama_token, double>> out;
    out.reserve(r.top.size());
    for (const auto & c : r.top) {
        if (std::find(eog.begin(), eog.end(), c.first) == eog.end()) {
            out.emplace_back(c.first, c.second - log_keep);
        }
    }
    return out;
}

// Returns false with `err` set on failure; `terminated` is set when the client went away mid-search.
static bool beam_search_run(server_context_impl & ctx_server,
                            server_task_type type,
                            const task_params & base,
                            const llama_tokens & prompt,
                            int width,
                            int max_tokens,
                            double length_penalty,
                            const std::function<bool()> & should_stop,
                            std::vector<beam_hyp> & out,
                            beam_stats & stats,
                            bool & terminated,
                            std::string & err) {
    terminated = false;
    std::vector<beam_hyp> live(1);
    std::vector<beam_hyp> done;
    const std::vector<llama_token> eog = beam_eog_ids(ctx_server.vocab, base.sampling.ignore_eos);
    const int n_vocab = llama_vocab_n_tokens(ctx_server.vocab);

    for (int step = 0; step < max_tokens && !live.empty(); ++step) {
        const bool mask = base.min_tokens > step && !eog.empty();
        std::vector<const llama_tokens *> gens;
        for (const auto & h : live) {
            gens.push_back(&h.tokens);
        }
        const int n_probs = std::min(2 * width + (mask ? (int) eog.size() : 0), n_vocab);
        std::vector<beam_raw> raws;
        if (!beam_step_raw(ctx_server, type, base, prompt, gens, n_probs, mask ? eog : std::vector<llama_token>(),
                           should_stop, raws, terminated, err)) {
            return false;
        }
        stats.steps++;
        stats.min_masked += mask ? 1 : 0;

        struct cand { double lp; llama_token tok; size_t beam; };
        std::vector<cand> cands;
        for (size_t b = 0; b < live.size(); ++b) {
            const beam_hyp & parent = live[b];
            double log_keep = 0.0;
            for (const auto & c : beam_mask_eog(raws[b], eog, mask, log_keep)) {
                const double lp = parent.cum_logprob + c.second;
                if (!base.sampling.ignore_eos && llama_vocab_is_eog(ctx_server.vocab, c.first)) {
                    beam_hyp h = parent;
                    h.tokens.push_back(c.first);
                    h.cum_logprob = lp;
                    h.eos         = true;
                    done.push_back(std::move(h));
                } else {
                    cands.push_back({lp, c.first, b});
                }
            }
        }

        std::stable_sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.lp > b.lp; });
        std::vector<beam_hyp> next;
        for (size_t i = 0; i < cands.size() && (int) next.size() < width; ++i) {
            beam_hyp h = live[cands[i].beam];
            h.tokens.push_back(cands[i].tok);
            h.cum_logprob = cands[i].lp;
            next.push_back(std::move(h));
        }
        live = std::move(next);
    }

    for (auto & h : live) {
        done.push_back(std::move(h));
    }
    stats.registered = (int) done.size();
    std::stable_sort(done.begin(), done.end(), [length_penalty](const beam_hyp & a, const beam_hyp & b) {
        return beam_score(a, length_penalty) > beam_score(b, length_penalty);
    });
    if ((int) done.size() > width) {
        done.resize(width);
    }
    out = std::move(done);
    return true;
}

//
// Beam search, CTranslate2's loop (beam_mode "ct2"). SOURCE-READ (MIT, OpenNMT/CTranslate2 @ 998bb99f, no code copied):
//   src/decoding.cc:411-713   BeamSearch::search -- the loop below.
//     :433-436 the number of candidates is 2 * beam size; :469-482 only one beam is live at the first step.
//     :497-505 min_tokens: EOS is removed from the logits before the softmax (apply_min_length :384-408).
//     :523-525 a step's scores are the beam's running score plus the log-softmax, flattened across beams; the 2 * width best
//              are taken (sampler :546).
//     :594-640 for the first `width` of those candidates: an EOS (or any candidate on the last step) is REGISTERED as a
//              hypothesis and its slot is refilled by the next NON-EOS candidate of the spare `width`; EOS candidates ranked
//              beyond `width` are dropped, not registered (this is where vLLM's loop differs).
//     :642-650 stopping: last step; or, when length_penalty == 0, the best candidate finished and `num_hypotheses` are
//              registered; or, with a length penalty, round(width * patience) hypotheses are registered (:355-357).
//     :173-186,205-222 a hypothesis's score is cum_logprob / len^length_penalty, len counting the EOS; sorted best first and cut
//              to num_hypotheses.
//   src/decoding.cc:259-323 BiasedDecoder (soft prefix): while a beam still follows the prefix its distribution is
//              p' = (1 - beta) * p, p'[prefix[step]] += beta; :325-355 a beam is "diverged" once it emits another token.
//   src/decoding.cc:21-66 hard prefix (beta == 0): the prefix tokens are forced at score 0, and at the first free step an EOS that
//              is not the best candidate is removed (CTranslate2 issue #277: a bare </s> would win with no length penalty).
//   NUANCES THE DESCRIPTION DID NOT SAY: (1) CT2 indexes the diverged flag by candidate RANK (:325-338 reads
//   beams_diverged_from_prefix[i][k] for candidate k), not by the candidate's parent beam, and does not re-order it when slots are
//   refilled; this port tracks it per lineage (child inherits the parent's flag), which is what the flag is meant to say.
//   (2) A finished slot with no non-EOS spare candidate left KEEPS the EOS candidate as a live beam (:609-618); so does this port.
//   (3) With beta > 0 the prefix token can be outside the model's top list; its exact log-prob is asked for explicitly.
//   (4) The hard prefix is not searched, so the model is not called for those steps; max_tokens counts the prefix tokens (CT2's
//   return_prefix = true) and a prefix as long as max_tokens is refused.
//
static bool beam_search_ct2(server_context_impl & ctx_server,
                            server_task_type type,
                            const task_params & base,
                            const llama_tokens & prompt,
                            int width,
                            int num_hypotheses,
                            int max_tokens,
                            double length_penalty,
                            double patience,
                            const llama_tokens & prefix,
                            double prefix_beta,
                            const std::function<bool()> & should_stop,
                            std::vector<beam_hyp> & out,
                            beam_stats & stats,
                            bool & terminated,
                            std::string & err) {
    terminated = false;
    const bool ignore_eos = base.sampling.ignore_eos;
    const std::vector<llama_token> eog = beam_eog_ids(ctx_server.vocab, ignore_eos);
    const int n_vocab = llama_vocab_n_tokens(ctx_server.vocab);
    auto is_eos = [&](llama_token t) { return !ignore_eos && llama_vocab_is_eog(ctx_server.vocab, t); };

    const bool soft_prefix = !prefix.empty() && prefix_beta > 0.0;
    const bool hard_prefix = !prefix.empty() && !soft_prefix;
    const int  max_cand    = (int) std::lround((double) width * patience);
    const bool early_rule  = length_penalty == 0.0;   // src/decoding.cc:451 allow_early_exit

    std::vector<beam_hyp> live(1);
    std::vector<beam_hyp> done;
    bool top_beam_finished = false;
    int  first_step = 0;
    if (hard_prefix) {
        live[0].tokens = prefix;   // forced at score 0
        first_step = (int) prefix.size();
        stats.prefix_forced = first_step;
    }

    for (int step = first_step; step < max_tokens && !live.empty(); ++step) {
        const bool last = step + 1 == max_tokens;
        const bool mask = base.min_tokens > step && !eog.empty();
        bool want_prefix_tok = false;
        if (soft_prefix && step < (int) prefix.size()) {
            for (const auto & h : live) {
                want_prefix_tok = want_prefix_tok || !h.diverged;
            }
        }
        std::vector<llama_token> want_ids;
        if (mask) {
            want_ids = eog;
        }
        if (want_prefix_tok && std::find(want_ids.begin(), want_ids.end(), prefix[step]) == want_ids.end()) {
            want_ids.push_back(prefix[step]);
        }
        const int n_probs = std::min(2 * width + (mask ? (int) eog.size() : 0), n_vocab);

        std::vector<const llama_tokens *> gens;
        for (const auto & h : live) {
            gens.push_back(&h.tokens);
        }
        std::vector<beam_raw> raws;
        if (!beam_step_raw(ctx_server, type, base, prompt, gens, n_probs, want_ids, should_stop, raws, terminated, err)) {
            return false;
        }
        stats.steps++;
        stats.min_masked += mask ? 1 : 0;

        struct cand { double lp; llama_token tok; size_t beam; };
        std::vector<cand> cands;
        bool biased_any = false;
        for (size_t b = 0; b < live.size(); ++b) {
            double log_keep = 0.0;
            auto dist = beam_mask_eog(raws[b], eog, mask, log_keep);
            if (want_prefix_tok && !live[b].diverged) {
                const llama_token ptok = prefix[step];
                const bool masked_ptok = mask && std::find(eog.begin(), eog.end(), ptok) != eog.end();
                double lp_ptok = -INFINITY;
                if (!masked_ptok) {
                    auto it = raws[b].ids.find(ptok);
                    if (it == raws[b].ids.end()) {
                        err = "beam search: the prefix token's log-probability is missing";
                        return false;
                    }
                    lp_ptok = it->second - log_keep;
                }
                const double lk = std::log1p(-prefix_beta);   // log(1 - beta); -inf at beta = 1
                bool seen = false;
                for (auto & c : dist) {
                    if (c.first == ptok) {
                        seen = true;
                    } else {
                        c.second += lk;
                    }
                }
                const double biased = std::log((1.0 - prefix_beta) * std::exp(lp_ptok) + prefix_beta);
                if (seen) {
                    for (auto & c : dist) {
                        if (c.first == ptok) {
                            c.second = biased;
                        }
                    }
                } else {
                    dist.emplace_back(ptok, biased);
                }
                biased_any = true;
            }
            for (const auto & c : dist) {
                if (std::isfinite(c.second)) {
                    cands.push_back({live[b].cum_logprob + c.second, c.first, b});
                }
            }
        }
        stats.bias_steps += biased_any ? 1 : 0;

        std::stable_sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.lp > b.lp; });
        if ((int) cands.size() > 2 * width) {
            cands.resize(2 * width);
        }
        if (hard_prefix && step == first_step) {
            // :48-57 the first free step: only the best candidate may be an EOS
            for (size_t k = 1; k < cands.size();) {
                if (is_eos(cands[k].tok)) {
                    cands.erase(cands.begin() + k);
                } else {
                    ++k;
                }
            }
        }

        std::vector<beam_hyp> next;
        size_t secondary = (size_t) width;
        for (size_t k = 0; k < cands.size() && (int) k < width; ++k) {
            size_t pick = k;
            if (is_eos(cands[k].tok) || last) {
                if (k == 0) {
                    top_beam_finished = true;
                }
                beam_hyp h = live[cands[k].beam];
                h.tokens.push_back(cands[k].tok);
                h.cum_logprob = cands[k].lp;
                h.eos         = is_eos(cands[k].tok);
                done.push_back(std::move(h));
                for (size_t j = secondary; j < cands.size(); ++j) {
                    if (!is_eos(cands[j].tok)) {
                        pick = j;
                        secondary = j + 1;
                        break;
                    }
                }
            }
            beam_hyp child = live[cands[pick].beam];
            if (soft_prefix) {
                child.diverged = child.diverged || step >= (int) prefix.size() || cands[pick].tok != prefix[step];
            }
            child.tokens.push_back(cands[pick].tok);
            child.cum_logprob = cands[pick].lp;
            next.push_back(std::move(child));
        }

        bool finished = last;
        if (!finished) {
            finished = early_rule ? (top_beam_finished && (int) done.size() >= num_hypotheses)
                                  : (int) done.size() >= max_cand;
            stats.early_stop = finished;
        }
        if (finished) {
            break;
        }
        live = std::move(next);
    }

    stats.registered = (int) done.size();
    for (auto & h : done) {
        h.score = h.cum_logprob / std::pow((double) std::max<size_t>(h.tokens.size(), 1), length_penalty);
    }
    std::stable_sort(done.begin(), done.end(), [](const beam_hyp & a, const beam_hyp & b) { return a.score > b.score; });
    if ((int) done.size() > num_hypotheses) {
        done.resize(num_hypotheses);
    }
    out = std::move(done);
    return true;
}

//
// server_routes
//

// The body of a finished non-stream completion: one object, or for n > 1 the OAI choices merged into the first (OAI
// chat / completions) or a plain array (native API).
static json completions_nonstream_body(std::vector<server_task_result_ptr> & results, task_response_type res_type) {
    json arr = json::array();
    for (auto & r : results) {
        GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(r.get()) != nullptr);
        arr.push_back(r->to_json());
    }
    GGML_ASSERT(!arr.empty() && "empty results");
    if (arr.size() == 1) {
        return arr[0]; // single request: a single object, not an array
    }
    if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
        json & usage   = arr[0]["usage"];
        json & choices = arr[0]["choices"];
        for (size_t i = 1; i < arr.size(); i++) {
            choices.push_back(std::move(arr[i]["choices"][0]));
        }
        // OAI counts generated tokens across all choices (upstream)
        int64_t n_completion_tokens = 0;
        for (auto & item : arr) {
            n_completion_tokens += item["usage"]["completion_tokens"].get<int64_t>();
        }
        usage["completion_tokens"] = n_completion_tokens;
        usage["total_tokens"]      = usage["prompt_tokens"].get<int64_t>() + n_completion_tokens;
        return arr[0];
    }
    return arr;
}

// Non-stream keepalive (opt-in, header X-KoboldCpp-Keepalive -- the name KoboldCpp clients already send). A proxy or
// client with an idle-read timeout (Cloudflare ~100 s, some reverse proxies 60 s) drops a long non-stream request that
// sends nothing until the answer is ready. With the header, a request still running after the interval commits
// "200 + chunked JSON" and writes a whitespace line every interval until the body; JSON allows leading whitespace, so a
// client that parses the whole body sees the same object. A request that finishes (or fails) inside the first interval
// is answered exactly as without the header, real status included; after the commit an error goes in the body as
// {"error": ...}. A write that fails ends the response and the reader cancels the generation.
// Header value: "true" (any case; the only value KoboldCpp acts on) = on, a line every 15 s (KoboldCpp's text interval);
// an integer 1..300 = on, a line every that many seconds; anything else ("false", "0", empty) = off.
// Row S69-ENGINE-REVIEW-1016-KOBOLD-NONSTREAM-KEEPALIVE-01 (SPEC read: koboldcpp.py handle_request / send_json_keepalives,
// AGPL; no code copied).
// value of a request header by case-insensitive name (empty when absent); `lname` is lower case
static std::string request_header_ci(const std::map<std::string, std::string> & headers, const char * lname) {
    const size_t n = strlen(lname);
    for (const auto & [key, value] : headers) {
        if (key.size() != n) {
            continue;
        }
        bool same = true;
        for (size_t i = 0; i < n && same; i++) {
            same = std::tolower((unsigned char) key[i]) == lname[i];
        }
        if (same) {
            return value;
        }
    }
    return std::string();
}

static int nonstream_keepalive_seconds(const std::map<std::string, std::string> & headers) {
    for (const auto & [key, value] : headers) {
        if (key.size() != strlen("x-koboldcpp-keepalive")) {
            continue;
        }
        bool same = true;
        for (size_t i = 0; i < key.size() && same; i++) {
            same = std::tolower((unsigned char) key[i]) == "x-koboldcpp-keepalive"[i];
        }
        if (!same) {
            continue;
        }
        std::string v;
        for (const char c : value) {
            if (!std::isspace((unsigned char) c)) {
                v += (char) std::tolower((unsigned char) c);
            }
        }
        if (v == "true") {
            return 15;
        }
        if (v.empty() || v.size() > 3 || v.find_first_not_of("0123456789") != std::string::npos) {
            return 0;
        }
        const int s = std::stoi(v);
        return s >= 1 && s <= 300 ? s : 0;
    }
    return 0;
}

struct nonstream_keepalive_acc {
    std::vector<server_task_result_ptr> results;
    server_task_result_ptr error;
    bool finished = false;
};

enum class nonstream_wait { done, error, timeout, stopped };

// collect results until all are in, an error arrives, `should_stop` fires, or `seconds` pass (resumable: the results
// received so far stay in `acc`)
static nonstream_wait nonstream_collect(server_response_reader & rd, nonstream_keepalive_acc & acc,
                                        const std::function<bool()> & should_stop, int seconds) {
    if (acc.results.size() != rd.id_tasks.size()) {
        acc.results.resize(rd.id_tasks.size());
    }
    const int64_t deadline = ggml_time_ms() + (int64_t) seconds * 1000;
    while (rd.has_next()) {
        bool timeout = false;
        auto r = rd.next([&]() {
            if (should_stop()) {
                return true;
            }
            if (ggml_time_ms() >= deadline) {
                timeout = true;
                return true;
            }
            return false;
        });
        if (r == nullptr) {
            return timeout ? nonstream_wait::timeout : nonstream_wait::stopped;
        }
        if (r->is_error()) {
            acc.error = std::move(r);
            return nonstream_wait::error;
        }
        const size_t idx = r->index;
        GGML_ASSERT(idx < acc.results.size() && "index out of range");
        GGML_ASSERT(acc.results[idx] == nullptr && "duplicate result received");
        acc.results[idx] = std::move(r);
    }
    return nonstream_wait::done;
}

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    const int64_t t_arrival_us = ggml_time_us(); // t_deadline_ms is counted from here

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_chat()) {
        res->error(format_error_response("this server does not support chat/completions", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    // output-changing fields this server has no effect for are refused by name (server-refused-fields.h), on every generation route
    {
        const auto refused = server_refuse_unimplemented_field(data);
        if (refused.refused) {
            SRV_WRN("refusing request: %s\n", refused.message.c_str());
            res->error(format_error_response(refused.message, ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
    }

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        // Control-token text that arrived inside message text is tokenised as ordinary text (server_chat_params::plain_literals).
        // Needs a plain string prompt; a request that carries media still goes through the multimodal tokeniser below, where the
        // whole string is parsed for specials (open edge, row S69-STRATA-S2-CONTROL-TOKEN-INJECTION-...).
        plain_spans_t plain_spans;
        if (prompt.is_string() && data.contains("plain_text_spans")) {
            plain_spans = plain_spans_from_json(data.at("plain_text_spans"), prompt.get<std::string>().size());
        }

        if (!plain_spans.empty() && files.empty()) {
            inputs.push_back(server_tokens(tokenize_with_plain_spans(ctx_server.vocab, prompt.get<std::string>(), plain_spans, true), false));
        } else if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // --auto-truncate (row S69-ENGREV-SGLANG-AUTO-TRUNCATE-2026-10-27-01): a single text prompt longer than the slot context is
        // cut (first n_keep tokens + the last ones) instead of being refused with HTTP 400; the reply reports it (prompt_truncated).
        // Opt-in flag, default off. Media prompts and batches are left alone: a cut would break image markers / pairing.
        int32_t n_prompt_dropped = 0;
        if (params.auto_truncate && inputs.size() == 1 && !inputs[0].has_mtmd && meta->slot_n_ctx > 0) {
            int64_t n_pred = json_value(data, "n_predict", (int64_t) -1);
            if (n_pred <= 0) {
                n_pred = json_value(data, "max_tokens", (int64_t) -1);
            }
            const int64_t n_keep_req = json_value(data, "n_keep", (int64_t) 0);
            const server_truncate_plan plan = server_truncate_plan_for(inputs[0].size(), (size_t) meta->slot_n_ctx, n_pred, n_keep_req);
            if (plan.truncate) {
                const llama_tokens & all = inputs[0].get_tokens();
                llama_tokens cut(all.begin(), all.begin() + plan.keep_head);
                cut.insert(cut.end(), all.end() - plan.keep_tail, all.end());
                SRV_WRN("--auto-truncate: prompt of %zu tokens cut to %zu (kept %zu head + %zu tail, dropped %zu) to fit n_ctx_slot = %d\n",
                        all.size(), cut.size(), plan.keep_head, plan.keep_tail, plan.n_dropped, (int) meta->slot_n_ctx);
                n_prompt_dropped = (int32_t) plan.n_dropped;
                inputs[0] = server_tokens(cut, false);
            }
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // Admission ceilings (see server_admission): refuse here, after tokenizing (the token ceiling needs the prompt
        // length) and before anything is queued. Beam search below is covered too -- its ticket lives in `res`.
        if (admission->enabled()) {
            int64_t n_prompt = 0;
            for (const auto & in : inputs) {
                n_prompt += (int64_t) in.size();
            }
            std::string why;
            if (!admission->try_admit(n_prompt, why)) {
                // --evict-low-priority: the queue is full, but a request that is only WAITING for a slot and is strictly less important
                // than this arrival gives up its place (503 "evicted"); the arrival takes it. Equal priority never evicts.
                // SOURCE-READ: tensorflow shared_batch_scheduler.h:452-503 (PriorityTaskQueue::AddTask), no code copied.
                bool evicted = false;
                if (params.evict_low_priority) {
                    int32_t new_priority = 0;
                    try {
                        new_priority = json_value(data, "priority", 0);
                    } catch (const std::exception &) {
                        // a malformed priority is rejected by the schema later; here it simply cannot evict
                        new_priority = INT32_MAX;
                    }
                    const int id_victim = ctx_server.queue_tasks.evict_deferred(new_priority);
                    if (id_victim >= 0) {
                        SRV_WRN("evicting queued task %d for a request of priority %d: %s\n", id_victim, (int) new_priority, why.c_str());
                        auto res_victim = std::make_unique<server_task_result_error>();
                        res_victim->id       = id_victim;
                        res_victim->err_type = ERROR_TYPE_UNAVAILABLE;
                        res_victim->err_msg  = "evicted: a request of higher priority took this place in the full queue (--evict-low-priority); retry later";
                        ctx_server.queue_results.send(std::move(res_victim));
                        admission->force_admit(n_prompt);
                        evicted = true;
                    }
                }
                if (!evicted) {
                    SRV_WRN("%s\n", why.c_str());
                    res->error(format_error_response(why, ERROR_TYPE_UNAVAILABLE));
                    return res;
                }
            }
            res->admission = std::make_unique<server_admission_ticket>(admission, n_prompt);
        }

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        // Beam search (vLLM's use_beam_search / n / length_penalty surface) -- see beam_search_run above.
        if (json_value(data, "use_beam_search", false)) {
            auto bad = [&](const std::string & msg) {
                res->error(format_error_response(msg, ERROR_TYPE_INVALID_REQUEST));
                return std::move(res);
            };
            if (json_value(data, "stream", false)) {
                return bad("Streaming is not supported with beam search");
            }
            if (inputs.size() != 1) {
                return bad("beam search takes exactly one prompt");
            }
            if (inputs[0].has_mtmd) {
                return bad("beam search does not support media inputs");
            }
            task_params base = server_schema::eval_llama_cmpl_schema(ctx_server.vocab, params, meta->logit_bias_eog, data);
            if (!base.sampling.grammar.grammar.empty()) {
                // a grammar would be rebuilt from its root on every 1-token step -- refuse rather than mis-constrain
                return bad("beam search does not support grammar / json_schema");
            }
            if (base.sampling.cfg_scale != 1.0f) {
                return bad("beam search does not support cfg_scale");
            }
            if (base.n_predict <= 0) {
                return bad("beam search needs a positive n_predict / max_tokens");
            }
            const int    width          = std::max(1, base.n_cmpl);
            const std::string mode      = json_value(data, "beam_mode", params.beam_mode);
            if (mode != "vllm" && mode != "ct2") {
                return bad("beam_mode must be \"vllm\" or \"ct2\"");
            }
            const bool   ct2            = mode == "ct2";
            // length_penalty: vLLM's loop defaults to 1 (score = logprob / len), CTranslate2's to 0 (no normalisation, early exit)
            const double length_penalty = json_value(data, "length_penalty", ct2 ? 0.0 : 1.0);
            const llama_tokens prompt_tokens = inputs[0].get_text_tokens();
            if (base.min_tokens > base.n_predict) {
                return bad("min_tokens must be less than or equal to n_predict");
            }
            if (!ct2 && (data.contains("beam_patience") || data.contains("beam_prefix") ||
                         data.contains("beam_prefix_bias_beta") || data.contains("beam_num_hypotheses"))) {
                return bad("beam_patience / beam_prefix / beam_prefix_bias_beta / beam_num_hypotheses need beam_mode \"ct2\"");
            }

            std::vector<beam_hyp> beams;
            beam_stats stats;
            bool terminated = false;
            std::string err;
            bool ok = false;
            if (ct2) {
                const double patience = json_value(data, "beam_patience", 1.0);
                if (!(patience > 0.0) || !std::isfinite(patience)) {
                    return bad("beam_patience must be > 0");
                }
                const int num_hyps = json_value(data, "beam_num_hypotheses", width);
                if (num_hyps < 1 || num_hyps > width) {
                    return bad("beam_num_hypotheses must be between 1 and n (the beam width)");
                }
                const double beta = json_value(data, "beam_prefix_bias_beta", 0.0);
                if (!(beta >= 0.0 && beta <= 1.0)) {
                    return bad("beam_prefix_bias_beta must be in [0, 1]");
                }
                llama_tokens prefix;
                const std::string prefix_text = json_value(data, "beam_prefix", std::string());
                if (!prefix_text.empty()) {
                    prefix = common_tokenize(ctx_server.vocab, prefix_text, false, false);
                    if ((int) prefix.size() >= base.n_predict && beta == 0.0) {
                        return bad("beam_prefix is as long as n_predict: nothing is left to search");
                    }
                }
                ok = beam_search_ct2(ctx_server, type, base, prompt_tokens, width, num_hyps, base.n_predict, length_penalty, patience,
                                     prefix, beta, req.should_stop, beams, stats, terminated, err);
            } else {
                ok = beam_search_run(ctx_server, type, base, prompt_tokens, width, base.n_predict, length_penalty,
                                     req.should_stop, beams, stats, terminated, err);
            }
            if (!ok) {
                if (terminated) {
                    return res; // connection is closed
                }
                res->error(format_error_response(err, ERROR_TYPE_SERVER));
                return res;
            }
            SRV_INF("beam search (%s): width %d, %d step(s), %d hypothesis(es) finished, min_tokens masked %d step(s), prefix bias on %d step(s), "
                    "forced prefix %d token(s), early stop %s\n", mode.c_str(), width, stats.steps, stats.registered, stats.min_masked,
                    stats.bias_steps, stats.prefix_forced, stats.early_stop ? "yes" : "no");

            json choices = json::array();
            size_t n_generated = 0;
            for (size_t i = 0; i < beams.size(); ++i) {
                const auto & h = beams[i];
                llama_tokens gen = h.tokens;
                if (h.eos && !gen.empty()) {
                    gen.pop_back();
                }
                n_generated += h.tokens.size();
                const std::string text = tokens_to_str(ctx_server.vocab, gen);
                const char * finish = h.eos ? "stop" : "length";
                json c = {
                    {"index",              i},
                    {"finish_reason",      finish},
                    {"cumulative_logprob", h.cum_logprob},
                    {"score",              beam_score(h, length_penalty)},
                    {"tokens",             gen},
                };
                if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT) {
                    c["message"] = {{"role", "assistant"}, {"content", text}};
                } else {
                    c["text"] = text;
                }
                choices.push_back(std::move(c));
            }
            const json usage = {
                {"prompt_tokens",     prompt_tokens.size()},
                {"completion_tokens", n_generated},
                {"total_tokens",      prompt_tokens.size() + n_generated},
            };
            // ct2 mode reports which of its rules fired; the vllm reply keeps its old shape
            json beam_info = json::object();
            if (ct2) {
                beam_info = {
                    {"mode",          mode},
                    {"steps",         stats.steps},
                    {"registered",    stats.registered},
                    {"min_masked",    stats.min_masked},
                    {"bias_steps",    stats.bias_steps},
                    {"prefix_forced", stats.prefix_forced},
                    {"early_stop",    stats.early_stop},
                };
            }
            json reply;
            if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                reply = {
                    {"id",      completion_id},
                    {"object",  res_type == TASK_RESPONSE_TYPE_OAI_CHAT ? "chat.completion" : "text_completion"},
                    {"created", (int64_t) std::time(nullptr)},
                    {"model",   meta->model_name},
                    {"choices", choices},
                    {"usage",   usage},
                };
            } else {
                reply = {
                    {"content",          choices.empty() ? "" : choices[0].value("text", "")},
                    {"beams",            choices},
                    {"tokens_evaluated", prompt_tokens.size()},
                    {"beam_logprobs",    "raw"},
                    {"usage",            usage},
                };
            }
            if (ct2) {
                reply["beam_search"] = std::move(beam_info);
            }
            res->ok(std::move(reply));
            return res;
        }

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);
            task.n_prompt_truncated = n_prompt_dropped;

            if (task.params.token_healing) {
                // Token healing, behaviour read from exllamav3 generator/job.py (MIT) and tabbyAPI (AGPL, wiring only; no code copied).
                // Refused where the first token cannot be freely constrained: a grammar would be applied to a position that holds
                // the removed token's text, drafted tokens are verified in a batch that bypasses the first-token mask, and a media
                // prompt does not end in a plain text token.
                if (task.tokens.has_mtmd) {
                    throw std::invalid_argument("token_healing does not support media inputs");
                }
                if (!task.params.sampling.grammar.grammar.empty()) {
                    throw std::invalid_argument("token_healing cannot be combined with grammar / json_schema / response_format / tools that build a grammar");
                }
                if (ctx_server.has_speculative()) {
                    throw std::invalid_argument("token_healing cannot be combined with speculative decoding on this server");
                }
                // a single-token prompt is left alone (nothing would remain to condition on); a control token is never dropped,
                // because its text is empty and the model could not be made to write it back
                if (task.tokens.size() > 1) {
                    const llama_token last = task.tokens[task.tokens.size() - 1];
                    if (last != LLAMA_TOKEN_NULL && !llama_vocab_is_control(ctx_server.vocab, last)) {
                        task.tokens.keep_first(task.tokens.size() - 1);
                        task.params.heal_prefix_token = last;
                    }
                }
            }

            task.params.cache_pin = ctx_server.is_pin_token(json_value(data, "cache_pin_token", std::string()));

            // request log (server-request-log.h): ingest the W3C traceparent and echo X-Request-Id, so the 8090 -> 8081 hop shares one key
            {
                const server_trace_ids ids = server_ingest_trace_headers(request_header_ci(req.headers, "traceparent"),
                                                                         request_header_ci(req.headers, "x-request-id"), completion_id);
                task.params.trace_id   = ids.trace_id;
                task.params.request_id = ids.request_id;
                res->headers["X-Request-Id"] = ids.request_id;
            }

            // budget group (server-budget-group.h): X-Budget-Group[-Tokens|-Ms] headers or body fields budget_group[_tokens|_ms]; the first
            // request of a group fixes the pool, later ones join it. A malformed group is ignored (no cap), never half applied.
            {
                const auto body_str = [&](const char * k) {
                    return data.contains(k) && data.at(k).is_string() ? data.at(k).get<std::string>()
                         : data.contains(k) && data.at(k).is_number_integer() ? std::to_string(data.at(k).get<long long>()) : std::string();
                };
                const server_budget_request br = server_budget_ingest(
                    request_header_ci(req.headers, "x-budget-group"), body_str("budget_group"),
                    request_header_ci(req.headers, "x-budget-group-tokens"), body_str("budget_group_tokens"),
                    request_header_ci(req.headers, "x-budget-group-ms"), body_str("budget_group_ms"));
                if (br.valid) {
                    ctx_server.budget_groups.open(br.id, br.limits, ggml_time_us());
                    task.params.budget_group = br.id;
                    res->headers["X-Budget-Group"] = br.id;
                }
            }

            // end-to-end deadline: counted from the arrival of the HTTP request (taken on entry, before tokenizing and admission),
            // so queue wait, prompt and generation all spend the same budget
            if (task.params.t_deadline_ms > 0) {
                task.t_deadline_us = t_arrival_us + task.params.t_deadline_ms * 1000;
            }

            task.apply_chat_session(chat_session);
            // the request's own message_delimiters win over the template's (applied above by the session)
            if (!delimiters.delimiters.empty()) {
                task.params.message_spans = task.tokens.find_message_spans(delimiters);
            }

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;
            if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && data.contains("responses_tool_map") && data.at("responses_tool_map").is_object()) {
                task.params.responses_tool_map = data.at("responses_tool_map");
            }
            if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                task.params.responses_reasoning_blob = json_value(data, "responses_reasoning_blob", false);
                task.params.responses_reasoning_summary = json_value(data, "responses_reasoning_summary", false);
            }

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    // the ASR route parses res->data after this returns, so it never takes the chunked keepalive form
    const int keepalive_s = (!stream && res_type != TASK_RESPONSE_TYPE_OAI_ASR) ? nonstream_keepalive_seconds(req.headers) : 0;

    if (!stream && keepalive_s > 0) {
        auto acc = std::make_shared<nonstream_keepalive_acc>();
        const nonstream_wait w = nonstream_collect(rd, *acc, req.should_stop, keepalive_s);
        if (w == nonstream_wait::stopped) {
            return res; // connection is closed
        }
        if (w == nonstream_wait::error) {
            res->error(acc->error->to_json());
            return res;
        }
        if (w == nonstream_wait::done) {
            res->ok(completions_nonstream_body(acc->results, res_type));
            return res;
        }
        // still running after the first interval: commit 200 and keep the connection busy until the body is ready
        SRV_INF("non-stream keepalive: request still running after %d s, sending whitespace every %d s\n", keepalive_s, keepalive_s);
        res->status       = 200;
        res->content_type = "application/json; charset=utf-8";
        res->headers["X-Keepalive"] = "whitespace";
        res->set_next([res_this = res.get(), acc, res_type, keepalive_s](std::string & output) -> bool {
            if (acc->finished) {
                return false;
            }
            try {
                const nonstream_wait w = nonstream_collect(res_this->rd, *acc, [res_this]() { return res_this->should_stop(); }, keepalive_s);
                switch (w) {
                    case nonstream_wait::timeout:
                        output = std::string(2047, ' ') + "\n";
                        return true;
                    case nonstream_wait::stopped:
                        acc->finished = true;
                        return false;
                    case nonstream_wait::error:
                        acc->finished = true;
                        output = safe_json_to_str({{ "error", acc->error->to_json() }});
                        return false;
                    case nonstream_wait::done:
                        acc->finished = true;
                        output = safe_json_to_str(completions_nonstream_body(acc->results, res_type));
                        return false;
                }
            } catch (const std::exception & e) {
                acc->finished = true;
                output = safe_json_to_str({{ "error", format_error_response(e.what(), ERROR_TYPE_SERVER) }});
            }
            return false;
        });
    } else if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            res->ok(completions_nonstream_body(all_results.results, res_type));
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        if (res->admission) {
            res->admission->end_prefill(); // the first result exists, so this request's prefill is over
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                // /v1/responses: Codex's idle timeout counts events, not SSE comments, so a silent stretch gets a real
                // response.in_progress event at least every 15 s (see task_result_state::oai_resp_idle_events)
                const bool resp_idle_event = res_type == TASK_RESPONSE_TYPE_OAI_RESP && sse_ping_interval > 0 && !rd.states.empty();
                const int32_t ping_s = resp_idle_event ? std::min<int32_t>(sse_ping_interval, 15) : sse_ping_interval;
                auto result = rd.next([&timeout, &start_time, ping_s, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (ping_s > 0 && ggml_time_ms() - start_time > (int64_t)ping_s * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = resp_idle_event ? format_oai_resp_sse(rd.states[0].oai_resp_idle_events()) : std::string(":\n\n");
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !rd.states.empty()) {
                        output = format_oai_resp_sse(task_result_state::oai_resp_failed_events(rd.states[0].oai_resp_id, res_json));
                    } else {
                        output = format_error(res_type, res_json);
                    }
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !res_this->rd.states.empty()) {
                    output = format_oai_resp_sse(task_result_state::oai_resp_failed_events(res_this->rd.states[0].oai_resp_id, error_json));
                } else {
                    output = format_error(res_type, error_json);
                }

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

// G3 SQL session store (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01).
#ifdef SQLITE3_MODERN_CPP_SUPPORT
struct sql_session_db {
    sqlite::database db;
    explicit sql_session_db(const std::string & path) : db(path) {}
};
#endif

namespace {

// SOURCE-READ: examples/server/server.cpp:2035-2037 @ 35f0cac75 -- ported unchanged in spirit.
std::string sql_get_key_string(const json & j) {
    return j.is_string() ? j.get<std::string>() : j.dump();
}

// SOURCE-READ: examples/server/server.cpp:2020-2033 @ 35f0cac75. The fork's own normalize_store_name
// stripped the input to alpha-lowercase and let an unknown result hit sqlite as "no such table"
// (a 500, not a refusal). That is safe against injection (a table name can't be parameterized, and
// the strip leaves no SQL metacharacter able to survive) but not against a request naming a table
// that isn't one of the 3 the schema actually creates -- it just silently produces a different,
// less clear error further downstream. Replaced here with an explicit allowlist against the exact
// 3 tables server_routes::server_routes() creates, so an unknown storeName fails loudly as
// ERROR_TYPE_INVALID_REQUEST at the handler instead of as an sqlite exception message.
bool sql_normalize_store_name(const std::string & store_name_in, std::string & out) {
    if (store_name_in.empty()) {
        out = "sessions";
        return true;
    }
    if (store_name_in == "sessions" || store_name_in == "templates" || store_name_in == "names") {
        out = store_name_in;
        return true;
    }
    return false;
}

} // namespace

std::string server_routes::sql_unavailable_reason() const {
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    if (params.sql_save_file.empty()) {
        return "This server does not support the SQL session store. Start it with `--sql-save-file`";
    }
    return "The SQL session store failed to open: " + sql_db_error;
#else
    return "SQLite3 support was not compiled into this server. Rebuild with -DLLAMA_SERVER_SQLITE3=ON";
#endif
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    admission->max_reqs   = params.max_queued_requests;
    admission->max_tokens = params.max_queued_tokens;
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });

    // G3 SQL session store: opt-in, see server-context.h's sql_db comment. A user who explicitly
    // passed --sql-save-file and got a file we cannot open or migrate needs to know at startup,
    // not discover it as a 500 on their first /save call -- but this is one optional admin feature
    // among many the server offers, so a failure here logs loudly and turns off just this one
    // feature (every G3 route then reports sql_unavailable_reason(), a clear 5xx naming the exact
    // open error) rather than aborting server startup and taking down completions/embeddings/etc.
    // with it. DISABLE-REVERSIBLE: guarded by --sql-save-file itself; the operator fixes the
    // underlying file/permission/disk error this logged and restarts the server to re-enable.
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    if (!params.sql_save_file.empty()) {
        try {
            auto handle = std::make_shared<sql_session_db>(params.sql_save_file);
            handle->db << "CREATE TABLE IF NOT EXISTS sessions (key TEXT PRIMARY KEY, data TEXT)";
            handle->db << "CREATE TABLE IF NOT EXISTS templates (key TEXT PRIMARY KEY, data TEXT)";
            handle->db << "CREATE TABLE IF NOT EXISTS names    (key TEXT PRIMARY KEY, data TEXT)";
            sql_db = handle;
            SRV_INF("SQL session store opened: %s\n", params.sql_save_file.c_str());
        } catch (const std::exception & e) {
            sql_db_error = e.what();
            SRV_ERR("failed to open --sql-save-file '%s': %s\n", params.sql_save_file.c_str(), e.what());
        }
    }
    // G4: load the sqlite-zstd extension into the store's connection. SOURCE-READ: examples/server/server.cpp:570-592
    // @ 35f0cac75 (the pre-merge fork's own code). Loading is enabled ONLY for this one call and disabled again,
    // so no SQL a request can reach is ever able to call load_extension() itself. A failure turns off only G4.
    if (!params.sqlite_zstd_ext_file.empty()) {
        if (!sql_db) {
            sql_zstd_error = "--sqlite-zstd-ext-file needs a working --sql-save-file store";
            SRV_ERR("%s\n", sql_zstd_error.c_str());
        } else {
            sqlite3 * conn = sql_db->db.connection().get();
            sqlite3_enable_load_extension(conn, 1);
            char * errmsg = nullptr;
            const int rc = sqlite3_load_extension(conn, params.sqlite_zstd_ext_file.c_str(), nullptr, &errmsg);
            sqlite3_enable_load_extension(conn, 0);
            if (rc != SQLITE_OK) {
                sql_zstd_error = std::string("failed to load '") + params.sqlite_zstd_ext_file + "': " +
                                 (errmsg ? errmsg : "unknown extension error");
                SRV_ERR("%s\n", sql_zstd_error.c_str());
            } else {
                sql_zstd_loaded = true;
                SRV_INF("sqlite-zstd extension loaded: %s\n", params.sqlite_zstd_ext_file.c_str());
            }
            sqlite3_free(errmsg);
        }
    }
#endif
}

std::string server_routes::sql_zstd_unavailable_reason() const {
    if (!sql_db) {
        return sql_unavailable_reason();
    }
    if (params.sqlite_zstd_ext_file.empty()) {
        return "Transparent zstd compression is not enabled. Start the server with `--sqlite-zstd-ext-file <path to the sqlite-zstd extension>`";
    }
    return "The sqlite-zstd extension is not loaded: " + sql_zstd_error;
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ubatch",    meta.n_ubatch},
            {"mem_host_bytes",   meta.mem_host_bytes},
            {"mem_device_bytes", meta.mem_device_bytes},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
            {"arch",        meta.model_arch},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "lora_pool",                   !params.lora_pool_dir.empty() }, // /generate adapter_id may name a pooled adapter
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
        { "memory",                      meta.memory_breakdown },
    };

    if (!params.server_connect_code.empty()) {
        props["connect_code"] = params.server_connect_code;
    }
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

bool server_routes::model_sleep_enabled() const {
    return queue_tasks.is_sleep_enabled();
}

bool server_routes::model_is_sleeping() const {
    return queue_tasks.is_sleeping();
}

bool server_routes::sleep_model(int64_t wait_ms) const {
    // grace: a slot that is still generating keeps the model awake; the request lapses after 10 minutes (the longest a keep_alive 0
    // sender could reasonably expect the in-flight request to run) so a stale latch never sleeps the model under a later request
    if (!queue_tasks.request_sleep(10LL * 60 * 1000)) {
        return false;
    }
    const int64_t t_end = ggml_time_ms() + wait_ms;
    while (!queue_tasks.is_sleeping() && ggml_time_ms() < t_end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

void server_routes::wake_model() const {
    queue_tasks.wait_until_no_sleep();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req & req) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        if (server_decode_fail_latch_global().is_failed()) {
            res->error(format_error_response("last decode failed (compute error); unhealthy until a decode succeeds", ERROR_TYPE_UNAVAILABLE));
            return res;
        }
        // --forward-watchdog-soft/-hard: a batch in flight and no decode step finished for the soft timeout -> unhealthy (503)
        // until a step finishes, so a router or supervisor stops sending work to a server that is alive but stuck. Reads only
        // the process-wide watchdog view (server-watchdog.h), never the context.
        {
            const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const int64_t st = server_fwd_watchdog_health_global().stalled_ms(now_ms);
            if (st > 0) {
                res->error(format_error_response(string_format("decode stalled: no step finished for %lld ms with a batch in flight "
                                                               "(forward watchdog); unhealthy until a step finishes", (long long) st),
                                                 ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        json body = {{"status", "ok"}};
        // LLAMA_HOTSWAP_ENABLED: the documented /health trigger (docs/development/on-demand-tensor-reload.md) -- reload the
        // tensors that changed on disk, but only when no slot is processing; the reload itself runs on the main loop. Not
        // while sleeping: no model is loaded, and the next load reads the files as they are.
        if (hotswap_on_health.load() && !queue_tasks.is_sleeping()) {
            server_task task(SERVER_TASK_TYPE_HOTSWAP_RELOAD);
            task.id              = res->rd.get_new_id();
            task.hotswap_if_idle = true;
            res->rd.post_task(std::move(task));
            auto result = res->rd.next(req.should_stop);
            if (result && !result->is_error()) {
                body["hotswap"] = result->to_json();
            }
        }
        res->ok(body);
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            tmp.n_processing_slots   = cached_gauges.n_processing_slots;
            tmp.n_kv_tokens_active   = cached_gauges.n_kv_tokens_active;
            tmp.n_kv_tokens_held     = cached_gauges.n_kv_tokens_held;
            tmp.n_kv_tokens_capacity = cached_gauges.n_kv_tokens_capacity;
            tmp.memory               = cached_memory;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                task.metrics_memory = true; // memory_*_bytes gauges need no --memory
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    // GET /memory (--memory): ik_llama.cpp 55813a5d7 handle_memory. 501 without the flag, like /metrics.
    // CAPABILITY-GENUINELY-NA: read-only introspection endpoint; the 501 is the flag-gate, no capability is removed
    this->get_memory = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_memory) {
            res->error(format_error_response("This server does not support memory endpoint. Start it with `--memory`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (queue_tasks.is_sleeping()) {
            res->error(format_error_response("the server is sleeping, no model is loaded", ERROR_TYPE_UNAVAILABLE));
            return res;
        }

        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            task.metrics_memory = true;
            res->rd.post_task(std::move(task), true); // high-priority task
        }
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);
        res->ok(json{
            { "n_layer", res_task->n_layer },
            { "data",    res_task->memory  },
        });
        return res;
    };

    this->get_slots =[this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G2).
    // SOURCE-READ: examples/server/server.cpp:1845-1855 @ 35f0cac75 (list_slot_prompts, local worktree copy).
    // Unlike get_slots this is not gated by --slots / params.endpoint_slots: it reports only the tokens
    // currently held in each slot's KV cache (server_prompt), not the request params or generated text.
    this->get_slots_list = [this](const server_http_req & req) {
        auto res = create_response();
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SLOT_PROMPTS);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task), true); // high-priority, read-only
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);
        res->ok(res_task->to_json());
        return res;
    };

    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G2).
    // SOURCE-READ: examples/server/server.cpp:1786-1843, 1858-1965, 2216-2220 @ 35f0cac75 (local worktree copy).
    // Gated on --slot-save-path like the /slots/:id_slot save/restore/erase actions above. Pure
    // filesystem metadata (saved-prompt-file listing/delete/rename): does not touch model state or
    // `slots`, so unlike SLOT_PROMPTS it runs directly on the HTTP thread, not the task queue --
    // matching how the fork itself ran these handlers (no ctx_server access in the old lambdas either).
    // Runs through the task queue (SERVER_TASK_TYPE_LIST_SAVED_PROMPTS, main loop, server-context.cpp
    // above) rather than directly on the HTTP thread like post_delete_prompt/post_rename_prompt below:
    // decoding each saved file's packed server_tokens blob into preview text needs
    // detokenize(ctx_tgt), and ctx_tgt is private to server_context_impl outside thread-safe
    // tokenizer calls made from the main loop.
    this->get_saved_prompts = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support saved prompts. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_LIST_SAVED_PROMPTS);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);
        res->ok(res_task->to_json());
        return res;
    };

    this->post_delete_prompt = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support saved prompts. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        namespace fs = std::filesystem;
        try {
            const json body = json::parse(req.body);
            const std::string filename = body.at("filename");

            if (!fs_validate_filename(filename)) {
                res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }

            const fs::path file_to_delete = fs::path(params.slot_save_path) / filename;
            if (!fs::exists(file_to_delete) || !fs::is_regular_file(file_to_delete)) {
                res->error(format_error_response("File not found", ERROR_TYPE_NOT_FOUND));
                return res;
            }

            if (!fs::remove(file_to_delete)) {
                res->error(format_error_response("Failed to delete the file", ERROR_TYPE_SERVER));
                return res;
            }

            res->ok({{"status", "deleted"}, {"filename", filename}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
        return res;
    };

    this->post_rename_prompt = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support saved prompts. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        namespace fs = std::filesystem;
        try {
            const json body = json::parse(req.body);
            const std::string old_filename = body.at("old_filename");
            const std::string new_filename = body.at("new_filename");

            if (!fs_validate_filename(old_filename) || !fs_validate_filename(new_filename)) {
                res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }

            const fs::path old_path = fs::path(params.slot_save_path) / old_filename;
            const fs::path new_path = fs::path(params.slot_save_path) / new_filename;

            if (!fs::exists(old_path) || !fs::is_regular_file(old_path)) {
                res->error(format_error_response("Source file not found", ERROR_TYPE_NOT_FOUND));
                return res;
            }
            if (fs::exists(new_path)) {
                res->error(format_error_response("Destination filename already exists", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }

            std::error_code ec;
            fs::rename(old_path, new_path, ec);
            if (ec) {
                res->error(format_error_response("Failed to rename file: " + ec.message(), ERROR_TYPE_SERVER));
                return res;
            }

            res->ok({{"status", "renamed"}, {"old_filename", old_filename}, {"new_filename", new_filename}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
        return res;
    };

    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G3).
    // SOURCE-READ: examples/server/server.cpp:1991-2096, 2225-2235 @ 35f0cac75 (local worktree copy
    // of the pre-merge fork; MIT, reimplemented against the vendored sqlite_modern_cpp wrapper, no
    // code copied verbatim). All 7 routes run directly on the HTTP thread (pure DB I/O, no model or
    // slot access, same as the fork's own db_handler design and this file's own G2 handlers above).
    // Every "key" value round-trips through sql_get_key_string, exactly like the fork: a JSON string
    // is used as-is, anything else is re-dumped to a string, so a caller can key by a JSON literal.
    // Every "storeName" is passed through sql_normalize_store_name's 3-way allowlist before ever
    // reaching a SQL string -- see that function's comment for why this is stricter than the fork's
    // own strip-to-alpha version. The key/data VALUES are always bound as `?` parameters; only the
    // already-allowlisted table name is ever concatenated into a query string.
    this->post_sql_load = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            const json body = json::parse(req.body);
            std::string store;
            if (!sql_normalize_store_name(body.value("storeName", std::string()), store)) {
                res->error(format_error_response("Unknown storeName (must be sessions, templates, or names)", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            std::string data;
            bool found = false;
            sql_db->db << ("SELECT data FROM " + store + " WHERE key = ?") << sql_get_key_string(body.at("key"))
                >> [&](const std::string & d) { data = d; found = true; };
            if (!found) {
                res->error(format_error_response("Key not found", ERROR_TYPE_NOT_FOUND));
                return res;
            }
            json response{{"ok", true}};
            response["result"] = (store == "names") ? json(data) : json::parse(data);
            res->ok(response);
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
#endif
        return res;
    };

    this->post_sql_save = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            const json body = json::parse(req.body);
            std::string store;
            if (!sql_normalize_store_name(body.value("storeName", std::string()), store)) {
                res->error(format_error_response("Unknown storeName (must be sessions, templates, or names)", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            const json & data_field = body.at("data");
            const std::string data = (store == "names") ? data_field.get<std::string>() : data_field.dump();
            sql_db->db << ("INSERT OR REPLACE INTO " + store + " (key, data) VALUES (?, ?)")
                << sql_get_key_string(body.at("key")) << data;
            res->ok({{"ok", true}, {"result", "Data saved successfully"}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
#endif
        return res;
    };

    this->post_sql_rename = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            const json body = json::parse(req.body);
            // Fix vs. the fork's own handle_rename (old server.cpp:2061-2066): an UPDATE against a
            // key that does not exist in `names` affects 0 rows and sqlite raises no error, so the
            // fork's handler always answered "Session renamed successfully" even when nothing was
            // renamed -- a silent no-op reported as success. rows_modified() after the statement
            // runs turns that into a real 404 instead.
            sql_db->db << "UPDATE names SET data = ? WHERE key = ?"
                << body.at("newName").get<std::string>()
                << sql_get_key_string(body.at("key"));
            if (sql_db->db.rows_modified() == 0) {
                res->error(format_error_response("Key not found", ERROR_TYPE_NOT_FOUND));
                return res;
            }
            res->ok({{"ok", true}, {"result", "Session renamed successfully"}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
#endif
        return res;
    };

    this->post_sql_all = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            const json body = !req.body.empty() ? json::parse(req.body) : json::object();
            std::string store;
            if (!sql_normalize_store_name(body.value("storeName", std::string()), store)) {
                res->error(format_error_response("Unknown storeName (must be sessions, templates, or names)", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            json result = json::object();
            // Fix vs. the fork's own handle_all (old server.cpp:2068-2074): it called json::parse()
            // on every row's data unconditionally, including for the `names` store -- whose rows are
            // always a raw display-name string (see post_sql_save above), not JSON. A plain name
            // (anything not itself valid JSON, e.g. any name containing a space) would throw there.
            // Mirrors post_sql_load's own per-store handling instead.
            sql_db->db << ("SELECT key, data FROM " + store) >>
                [&](const std::string & key, const std::string & data) {
                    result[key] = (store == "names") ? json(data) : json::parse(data);
                };
            res->ok({{"ok", true}, {"result", result}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
#endif
        return res;
    };

    this->get_sql_sessions = [this](const server_http_req &) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            json result = json::object();
            sql_db->db << "SELECT key, data FROM names" >> [&](const std::string & key, const std::string & data) {
                result[key] = data;
            };
            res->ok({{"ok", true}, {"result", result}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
#endif
        return res;
    };
    // The fork's own handle_sessions (old server.cpp:2077-2083) is mounted on both GET and POST
    // /sessions and ignores the request body/method either way; kept as two identical handler_t
    // fields (rather than one field mounted twice) purely so router mode above can proxy each verb
    // through its own proxy_get/proxy_post.
    this->post_sql_sessions = this->get_sql_sessions;

    this->post_sql_delete = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            const json body = json::parse(req.body);
            std::string store;
            if (!sql_normalize_store_name(body.value("storeName", std::string()), store)) {
                res->error(format_error_response("Unknown storeName (must be sessions, templates, or names)", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // Fix vs. the fork's own handle_delete (old server.cpp:2085-2089): same silent-no-op
            // shape as post_sql_rename above -- deleting a key that does not exist affected 0 rows
            // and was still reported as "Session deleted successfully".
            sql_db->db << ("DELETE FROM " + store + " WHERE key = ?") << sql_get_key_string(body.at("key"));
            if (sql_db->db.rows_modified() == 0) {
                res->error(format_error_response("Key not found", ERROR_TYPE_NOT_FOUND));
                return res;
            }
            res->ok({{"ok", true}, {"result", "Session deleted successfully"}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
#endif
        return res;
    };

    this->get_sql_vacuum = [this](const server_http_req &) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            sql_db->db << "VACUUM";
            res->ok({{"ok", true}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
#endif
        return res;
    };

    // G4: sqlite-zstd transparent compression (row S69-G4-ZSTD-ROUTES-PORT-USER-APPROVED-2026-09-27-01).
    // SOURCE-READ: examples/server/server.cpp:2097-2127, 2236-2243 @ 35f0cac75 (pre-merge fork, MIT).
    // SECURITY FIX vs that source: handle_zstd_enable (:2119) built its JSON config by concatenating the
    // request's "table" and "column" strings into a JSON literal inside a SQL string literal, and
    // handle_zstd_config_update (:2125) concatenated its patch into SQL the same way -- a request could close
    // the quotes and inject SQL/JSON. Here the config is built with the JSON library and passed as ONE bound
    // `?` parameter, "table" must be one of the 3 store tables (sql_normalize_store_name) and "column" must be
    // "data" (the only compressible column in this schema). Request validation runs BEFORE the "extension
    // loaded" check, so a bad request is a 400 on every build. Also fixed: the source answered json{"ok", true},
    // which is the ARRAY ["ok", true], not an object.
    this->get_sql_zstd_configs = [this](const server_http_req &) {
        auto res = create_response();
        if (!sql_db || !sql_zstd_loaded) {
            res->error(format_error_response(sql_zstd_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            json configs = json::object();
            sql_db->db << "SELECT id, config FROM _zstd_configs"
                >> [&](const std::string & id, const std::string & config) { configs[id] = config; };
            res->ok({{"ok", true}, {"configs", configs}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
#endif
        return res;
    };

    this->post_sql_zstd_maintenance = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res->error(format_error_response(std::string("invalid JSON body: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!body.contains("db_load") || !body.at("db_load").is_number()) {
            res->error(format_error_response("\"db_load\" (number, 0..1 share of the DB's time to spend) is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (body.contains("duration") && !body.at("duration").is_null() && !body.at("duration").is_number()) {
            res->error(format_error_response("\"duration\" must be a number of seconds or null", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!sql_zstd_loaded) {
            res->error(format_error_response(sql_zstd_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            const double db_load = body.at("db_load").get<double>();
            std::string data;
            if (!body.contains("duration") || body.at("duration").is_null()) {
                sql_db->db << "SELECT zstd_incremental_maintenance(?, ?)" << nullptr << db_load >> data;
            } else {
                sql_db->db << "SELECT zstd_incremental_maintenance(?, ?)" << body.at("duration").get<double>() << db_load >> data;
            }
            json response{{"ok", true}};
            response["result"] = data.empty() ? json() : json::parse(data);
            res->ok(response);
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
#endif
        return res;
    };

    this->post_sql_zstd_enable = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res->error(format_error_response(std::string("invalid JSON body: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string table;
        if (!body.contains("table") || !body.at("table").is_string() ||
                !sql_normalize_store_name(body.at("table").get<std::string>(), table) || body.at("table").get<std::string>().empty()) {
            res->error(format_error_response("\"table\" must be one of: sessions, templates, names", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (body.value("column", std::string()) != "data") {
            res->error(format_error_response("\"column\" must be \"data\" (the only compressible column in the store)", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!body.contains("compression_level") || !body.at("compression_level").is_number_integer() ||
                body.at("compression_level").get<int>() < 1 || body.at("compression_level").get<int>() > 22) {
            res->error(format_error_response("\"compression_level\" must be an integer 1..22", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!body.contains("train_dict_samples_ratio") || !body.at("train_dict_samples_ratio").is_number() ||
                body.at("train_dict_samples_ratio").get<double>() <= 0) {
            res->error(format_error_response("\"train_dict_samples_ratio\" must be a positive number", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!sql_zstd_loaded) {
            res->error(format_error_response(sql_zstd_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            // dict_chooser "'a'": one shared dictionary for every row -- the source's fixed choice
            // ("''a''" inside its SQL literal is the SQL expression 'a').
            const json cfg = {
                {"table",                    table},
                {"column",                   "data"},
                {"compression_level",        body.at("compression_level").get<int>()},
                {"dict_chooser",             "'a'"},
                {"train_dict_samples_ratio", body.at("train_dict_samples_ratio").get<double>()},
            };
            sql_db->db << "SELECT zstd_enable_transparent(?)" << cfg.dump();
            res->ok({{"ok", true}, {"config", cfg}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
#endif
        return res;
    };

    this->post_sql_zstd_update = [this](const server_http_req & req) {
        auto res = create_response();
        if (!sql_db) {
            res->error(format_error_response(sql_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res->error(format_error_response(std::string("invalid JSON body: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        json patch = json::object();
        if (body.contains("compression_level")) {
            if (!body.at("compression_level").is_number_integer() ||
                    body.at("compression_level").get<int>() < 1 || body.at("compression_level").get<int>() > 22) {
                res->error(format_error_response("\"compression_level\" must be an integer 1..22", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            patch["compression_level"] = body.at("compression_level").get<int>();
        }
        if (body.contains("train_dict_samples_ratio")) {
            if (!body.at("train_dict_samples_ratio").is_number() || body.at("train_dict_samples_ratio").get<double>() <= 0) {
                res->error(format_error_response("\"train_dict_samples_ratio\" must be a positive number", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            patch["train_dict_samples_ratio"] = body.at("train_dict_samples_ratio").get<double>();
        }
        if (patch.empty()) {
            res->error(format_error_response("nothing to update: give compression_level and/or train_dict_samples_ratio", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!sql_zstd_loaded) {
            res->error(format_error_response(sql_zstd_unavailable_reason(), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        try {
            sql_db->db << "UPDATE _zstd_configs SET config = json_patch(config, ?)" << patch.dump();
            res->ok({{"ok", true}, {"updated", sql_db->db.rows_modified()}, {"patch", patch}});
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
#endif
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        json props;
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            props = cached_props;
        } else {
            props = get_res_props(*meta, params, false);
        }
        props["admission"] = admission->to_json();
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        // fork fix 2026-11-07: these three set the error but did not return, so a missing input_prefix fell through to
        // data.at("input_prefix") below and threw (json out_of_range -> 500) instead of answering the 400 it had just built
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        if (data.contains("multimodal_data") && !(data.at("multimodal_data").is_array() && data.at("multimodal_data").empty())) {
            // The ollama shim's suffix + images (row S69-ENGREV-OLLAMA-SHIM-RAW-AND-SUFFIX-IMAGES-2026-11-07-01): input_prefix / input_suffix
            // carry the media marker where an image goes. The token form of the FIM prompt cannot hold an image chunk, so the same layout is
            // built as text from the vocab's FIM pieces and tokenised with the multimodal tokeniser (special pieces parse back to ids).
            if (!data.at("input_prefix").is_string() || !data.at("input_suffix").is_string()) {
                res->error(format_error_response("\"input_prefix\" and \"input_suffix\" must be strings when \"multimodal_data\" is given", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            if (!data.at("multimodal_data").is_array() || ctx_server.mctx == nullptr) {
                res->error(format_error_response(ctx_server.mctx == nullptr ? "Multimodal data provided, but model does not support multimodal requests."
                                                                           : "\"multimodal_data\" must be an array of base64 strings", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            auto piece = [&](llama_token t) { return t == LLAMA_TOKEN_NULL ? std::string() : common_token_to_piece(ctx_server.vocab, t, true); };
            ollama_util::fim_pieces fp;
            fp.pre = piece(llama_vocab_fim_pre(ctx_server.vocab));
            fp.suf = piece(llama_vocab_fim_suf(ctx_server.vocab));
            fp.mid = piece(llama_vocab_fim_mid(ctx_server.vocab));
            fp.rep = piece(llama_vocab_fim_rep(ctx_server.vocab));
            fp.sep = piece(llama_vocab_fim_sep(ctx_server.vocab));
            fp.spm = params.spm_infill;
            std::vector<ollama_util::fim_extra_chunk> extra;
            for (const auto & chunk : input_extra) {
                extra.push_back({ json_value(chunk, "filename", std::string("tmp")), json_value(chunk, "text", std::string()) });
            }
            json mm_prompt = json::object();
            mm_prompt["prompt_string"]  = ollama_util::fim_text(fp, data.at("input_prefix").get<std::string>() + prompt,
                                                                data.at("input_suffix").get<std::string>(), extra);
            mm_prompt["multimodal_data"] = data.at("multimodal_data");
            data["prompt"] = mm_prompt;
            std::vector<raw_buffer> files; // dummy, the media travels inside the prompt object
            return handle_completions_impl(req, SERVER_TASK_TYPE_INFILL, data, files, TASK_RESPONSE_TYPE_NONE);
        }
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end" && action != "cancel") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action == "cancel" ? SERVER_CONTROL_ACTION_CANCEL : SERVER_CONTROL_ACTION_REASONING_END;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json raw_body = json::parse(req.body);
        {
            // compaction_trigger as the last input item: answer with one compaction item instead of a model turn
            respcompact::plan cplan;
            bool requested = false;
            std::string cerr;
            if (!respcompact::prepare_triggered(raw_body, cplan, requested, cerr)) {
                res->error(format_error_response(cerr, ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            if (requested) {
                json item, usage;
                if (auto failed = run_compaction(req, cplan, item, usage)) {
                    return failed;
                }
                res->status       = 200;
                res->content_type = "text/event-stream";
                res->data         = respcompact::render_sse("resp_" + random_string(), cplan.model, item, usage);
                return res;
            }
            // an earlier compaction item in the input expands into its summary + retained items
            bool changed = false;
            if (!respcompact::expand_body(raw_body, changed, cerr)) {
                res->error(format_error_response(cerr, ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        json body = server_chat_convert_responses_to_chatcmpl(raw_body);
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    // POST /v1/responses/compact: one compaction item for the given input (Codex remote compaction)
    this->post_responses_compact = [this](const server_http_req & req) {
        auto res = create_response();
        json raw_body = json::parse(req.body);
        respcompact::plan cplan;
        std::string cerr;
        if (!respcompact::prepare_standalone(raw_body, cplan, cerr)) {
            res->error(format_error_response(cerr, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        json item, usage;
        if (auto failed = run_compaction(req, cplan, item, usage)) {
            return failed;
        }
        res->ok(respcompact::compacted_response("resp_" + random_string(), item, usage));
        return res;
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        // The model's own transcription prompt (--asr-native-prompt): mistral-common encodes a transcription request as
        // BOS [INST] [BEGIN_AUDIO] <audio> [/INST] lang:<code> [TRANSCRIBE] (tokens/tokenizers/instruct.py,
        // _encode_instruct_transcription; SOURCE-READ, Apache-2.0), not as a chat turn; mtmd writes [BEGIN_AUDIO] at the media
        // marker for a Voxtral projector. Through the chat template (the Unsloth Voxtral GGUF ships Devstral's) the model treats
        // the audio as a message to answer: a transcript can come back wrapped in quotes and silence draws a refusal sentence.
        // OPT-IN (default off), MEASURED 2026-10-17 on Voxtral-Mini-3B Q4_K_M (temp/arex1014/_voxtx_1017/summary.json): the
        // native prompt gives the bare transcript on speech, but on silence and on -50 dB noise it does NOT stop -- it writes
        // invented text, and its first-position end-of-generation probability is ~0 on speech and silence alike. Silence is
        // a voice-activity question, not a prompt one. plain_text_spans index the rendered chat prompt, so they go with it.
        if (params.asr_native_prompt == "on") {
            const json inp = json::parse(req.body);
            if (json_value(inp, "prompt", std::string()).empty()) {
                const auto tt = common_tokenize(ctx_server.vocab, "[TRANSCRIBE]", false, true);
                if (tt.size() == 1 && llama_vocab_is_control(ctx_server.vocab, tt[0])) {
                    const std::string lang = json_value(inp, "language", std::string());
                    body_parsed["prompt"] = std::string("[INST]") + get_media_marker() + "[/INST]" + (lang.empty() ? std::string() : "lang:" + lang) + "[TRANSCRIBE]";
                    body_parsed["plain_text_spans"] = json::array();
                    SRV_DBG("transcription: native prompt '%s'\n", body_parsed["prompt"].get<std::string>().c_str());
                } else {
                    SRV_WRN("%s", "--asr-native-prompt on: the vocabulary has no [TRANSCRIBE] control token; using the chat template\n");
                }
            }
        }
        // --asr-native-prompt vibevoice: VibeVoice-ASR's offline prompt, rendered verbatim and sent without a chat template (a template
        // that appends an assistant header changes the output with no error), plus its RMS -25 dBFS front end on the whole upload
        // (server-asr-vibevoice.h; SOURCE-READ microsoft/VibeVoice, MIT). The request `prompt` becomes the hotword clause.
        // mtmd writes the audio wrapper tokens at the media marker once a VibeVoice projector exists; without one the upload is
        // refused by the has_mtmd check above, so a Voxtral/other projector never sees this prompt by accident.
        if (params.asr_native_prompt == "vibevoice" && ctx_server.mctx != nullptr && !files.empty()) {
            const int32_t sr = mtmd_get_audio_sample_rate(ctx_server.mctx);
            auto wrap = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, files[0].data(), files[0].size(), false, mtmd_helper_init_opt_default());
            if (wrap.bitmap != nullptr && mtmd_bitmap_is_audio(wrap.bitmap) && sr > 0) {
                const size_t  n   = mtmd_bitmap_get_nx(wrap.bitmap);
                const float * src = reinterpret_cast<const float *>(mtmd_bitmap_get_data(wrap.bitmap));
                std::vector<float> pcm(src, src + n);
                const double scalar = asr_vibevoice::rms_normalize(pcm);
                const auto   wav    = asr_vibevoice::wav_f32_mono(pcm, sr);
                files[0].assign(wav.begin(), wav.end());
                const json inp = json::parse(req.body);
                body_parsed["prompt"] = asr_vibevoice::render(n, sr, json_value(inp, "prompt", std::string()), get_media_marker());
                body_parsed["plain_text_spans"] = json::array();
                SRV_DBG("transcription: vibevoice prompt, %zu samples at %d Hz, rms scalar %.4f\n", n, (int) sr, scalar);
            } else {
                SRV_WRN("%s", "--asr-native-prompt vibevoice: upload did not decode as audio; using the chat template\n");
            }
            if (wrap.bitmap != nullptr) {
                mtmd_bitmap_free(wrap.bitmap);
            }
        }
        // response_format other than json: render the finished transcript (convert_transcriptions_to_chatcmpl
        // already refused unknown formats and stream=true with them). json takes the unchanged path below.
        // Applies to the --asr-quality path as well: the format is rendered from whichever attempt was chosen.
        asrfmt::format asr_fmt = asrfmt::format::json;
        {
            const json inp = json::parse(req.body);
            asrfmt::parse(json_value(inp, "response_format", std::string()), asr_fmt);
        }

        // timestamp_granularities[]=word (OpenAI): top-level "words" in verbose_json, times from the aligner behind
        // `llama-audio --word-timestamps`. Every refusal is decided here, before the model runs; a request without the field
        // never touches this block (its response is byte-identical to before).
        bool        want_words = false;
        asrwords::engine * words_engine = nullptr;
        {
            const json inp = json::parse(req.body);
            std::vector<std::string> gv;
            for (const char * key : { "timestamp_granularities[]", "timestamp_granularities" }) {
                if (!inp.contains(key)) { continue; }
                const json & v = inp.at(key);
                if (v.is_string()) {
                    gv.push_back(v.get<std::string>());
                } else if (v.is_array()) {
                    for (const auto & e : v) { if (e.is_string()) { gv.push_back(e.get<std::string>()); } }
                }
            }
            const auto gr = asrwords::parse_granularities(gv);
            if (!gr.bad.empty()) {
                res->error(format_error_response("unknown timestamp_granularities value '" + gr.bad + "' (use word or segment)", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            want_words = gr.word;
            if (want_words) {
                if (asr_fmt != asrfmt::format::verbose_json) {
                    res->error(format_error_response("timestamp_granularities[]=word needs response_format=verbose_json", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                if (files.size() != 1) {
                    res->error(format_error_response("timestamp_granularities[]=word needs exactly one audio file", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                std::string werr;
                {
                    std::lock_guard<std::mutex> lk(ctx_server.asr_words_mu);
                    if (!ctx_server.asr_words_engine && ctx_server.model_tgt != nullptr) {
                        ctx_server.asr_words_engine = std::make_unique<asrwords::engine>(ctx_server.model_tgt, params);
                    }
                    words_engine = ctx_server.asr_words_engine.get();
                }
                if (words_engine == nullptr || !words_engine->ready(werr)) {
                    res->error(format_error_response(werr.empty() ? std::string("word timestamps are not available on this server") : werr, ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
            }
        }

        // --asr-vad (opt-in): run the native Silero VAD over the decoded upload first. MEASURED 2026-10-17 on Voxtral-Mini-3B: a silent or
        // -50 dB clip gets a refusal (chat prompt) or invented text (native prompt), and the first-position EOG probability is 0.0 on both
        // silence and speech, so the model cannot tell silence itself. No speech segment -> an empty transcript at once, the model never
        // runs (whisper.cpp does the same with VAD on: src/whisper.cpp:7945-7953). Speech -> the request continues unchanged (the whole
        // upload, untrimmed, goes to the model). An upload that does not decode as audio here goes on to the model path, handled as before.
        //
        // --asr-vad-trim (opt-in, with --asr-vad): when speech is found, the model is given the speech spans only. The kept samples go
        // back into files[0] as a float32 WAV at the model's rate (so the audio path decodes them unchanged), the upload itself is kept
        // for diarize, and asr_trim_map takes every segment time of the reply back to the upload (examples/audio/asr-vad-trim.h).
        asr_vad_trim_map asr_trim_map;              // empty = this request was not trimmed
        raw_buffer       asr_trim_upload;           // the upload as the client sent it, when trimmed
        double           asr_trim_upload_s = 0.0;   // its length
        json             asr_trim_info;             // the "asr_vad" object of the reply, when trimmed
        if (ctx_server.asr_vad && ctx_server.mctx != nullptr && !files.empty()) {
            const int32_t sr = mtmd_get_audio_sample_rate(ctx_server.mctx);
            auto wrap = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, files[0].data(), files[0].size(), false, mtmd_helper_init_opt_default());
            bool   gated      = false;
            double duration_s = 0.0;
            asr_vad_gate_result g;
            if (wrap.bitmap != nullptr && mtmd_bitmap_is_audio(wrap.bitmap) && sr > 0) {
                const size_t  n   = mtmd_bitmap_get_nx(wrap.bitmap);
                const float * pcm = reinterpret_cast<const float *>(mtmd_bitmap_get_data(wrap.bitmap));
                asr_vad_post_params pp;
                pp.threshold = params.asr_vad_threshold;
                const int64_t t0 = ggml_time_us();
                g = asr_vad_gate(*ctx_server.asr_vad, pcm, n, sr, pp);
                duration_s = (double) n / (double) sr;
                gated = !g.has_speech();
                SRV_INF("asr-vad: %.3f s at %d Hz, %zu Silero window(s), max speech p %.3f, %zu speech span(s) [%.1f ms] -> %s\n",
                        duration_s, (int) sr, g.n_probs, (double) g.max_prob, g.spans.size(), (ggml_time_us() - t0) / 1000.0,
                        gated ? "NO SPEECH: empty transcript, model not run" : "speech: transcribing");
                if (!gated && params.asr_vad_trim) {
                    asr_vad_trim_params tp;
                    tp.pad_ms = params.asr_vad_trim_pad_ms;
                    asr_vad_trim_map cut;
                    const std::vector<float> kept = asr_vad_trim(pcm, n, sr, g.spans, tp, cut);
                    if (!kept.empty() && cut.n_out < (int64_t) n) {
                        const double kept_s = (double) kept.size() / (double) sr;
                        asr_trim_info = {
                            {"speech",          true},
                            {"trimmed",         true},
                            {"speech_segments", g.spans.size()},
                            {"pieces",          cut.pieces.size()},
                            {"audio_s",         duration_s},
                            {"transcribed_s",   kept_s},
                            {"pad_ms",          params.asr_vad_trim_pad_ms},
                            {"max_speech_prob", g.max_prob},
                            {"threshold",       params.asr_vad_threshold},
                        };
                        std::vector<uint8_t> wav = asr_wav_f32_encode(kept.data(), kept.size(), sr);
                        asr_trim_upload   = std::move(files[0]);
                        files[0]          = std::move(wav);
                        asr_trim_upload_s = duration_s;
                        asr_trim_map      = std::move(cut);
                        SRV_INF("asr-vad-trim: %.3f s of %.3f s kept in %zu piece(s) (%zu speech span(s), pad %d ms): the model transcribes the kept audio, "
                                "times are mapped back to the upload\n",
                                kept_s, duration_s, asr_trim_map.pieces.size(), g.spans.size(), (int) params.asr_vad_trim_pad_ms);
                    } else {
                        SRV_INF("asr-vad-trim: nothing to cut (%.3f s, %zu speech span(s)); the upload is transcribed as it is\n", duration_s, g.spans.size());
                    }
                }
            } else {
                SRV_WRN("%s", "asr-vad: upload did not decode as audio here; passing it to the model path unchanged\n");
            }
            if (wrap.bitmap != nullptr) {
                mtmd_bitmap_free(wrap.bitmap);
            }
            if (gated) {
                auto out = create_response();
                const json vad_info = {
                    {"speech",          false},
                    {"speech_segments", 0},
                    {"max_speech_prob", g.max_prob},
                    {"windows",         g.n_probs},
                    {"threshold",       params.asr_vad_threshold},
                };
                if (asr_fmt == asrfmt::format::json) {
                    // the shape of the model path's final event (server_task_result_cmpl_final, TASK_RESPONSE_TYPE_OAI_ASR), nothing decoded;
                    // no_speech_prob 1.0: the VAD found no speech at all
                    const json done = {
                        {"type",  "transcript.text.done"},
                        {"text",  ""},
                        {"usage", json {
                            {"type",         "tokens"},
                            {"input_tokens",  0},
                            {"output_tokens", 0},
                            {"total_tokens",  0},
                            {"input_tokens_details", json { {"cached_tokens", 0} }},
                        }},
                        {"no_speech_prob", 1.0},
                        {"asr_vad",        vad_info},
                    };
                    if (json_value(body_parsed, "stream", false)) {
                        out->status       = 200;
                        out->content_type = "text/event-stream";
                        out->data         = format_oai_sse(done) + "data: [DONE]\n\n";
                    } else {
                        out->ok(done);
                    }
                } else {
                    asrfmt::verbose_extras vx;
                    vx.temperature    = json_value(body, "temperature", 0.0);
                    vx.no_speech_prob = 1.0;   // written on every segment -- there are none
                    vx.words_on       = want_words;   // no speech: an empty, honest word list
                    vx.usage_json     = json { {"type", "tokens"}, {"input_tokens", 0}, {"output_tokens", 0}, {"total_tokens", 0} }.dump();
                    out->status       = 200;
                    out->data         = asrfmt::render(asr_fmt, "", duration_s, json_value(json::parse(req.body), "language", std::string()),
                                                       false, 0.0, asr_fmt == asrfmt::format::verbose_json ? &vx : nullptr);
                    out->content_type = asrfmt::content_type(asr_fmt);
                }
                return out;
            }
        }

        std::unique_ptr<server_res_generator> res_gen;

        // --asr-quality: gate the transcript and decode again at the next temperature of the whisper ladder when it fails
        // (examples/audio/asr-fallback.h; whisper.cpp / faster-whisper semantics). Streaming replies are not gated.
        if (params.asr_quality && !json_value(body_parsed, "stream", false)) {
            asr_fallback_params fp;
            fp.temperature             = json_value(body_parsed, "temperature", 0.0f);
            fp.temperature_inc         = params.asr_temperature_inc;
            fp.check_compression       = true;
            fp.compression_ratio_thold = params.asr_compression_ratio_thold;
            fp.check_logprob           = true;
            fp.logprob_thold           = params.asr_logprob_thold;
            fp.check_no_speech         = false; // an LLM-based decoder has no no-speech head
            fp.check_entropy           = params.asr_entropy_thold > 0.0f;
            fp.entropy_thold           = params.asr_entropy_thold;
            fp.on_exhausted            = asr_fallback_params::EXHAUSTED_BEST_LOGPROB;

            std::vector<std::unique_ptr<server_res_generator>> kept;
            std::unique_ptr<server_res_generator> failed;
            const auto fr = asr_run_fallback(fp, [&](float temperature) {
                asr_decode_attempt att;
                att.temperature = temperature;
                json b = body_parsed;
                b["temperature"] = temperature;
                b["n_probs"]     = 1;
                b["asr_stats"]   = true;
                auto r = handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, b, files, TASK_RESPONSE_TYPE_OAI_ASR, session);
                json j;
                if (r->status == 200) {
                    try {
                        j = json::parse(r->data);
                    } catch (const std::exception &) {
                        j = json();
                    }
                }
                if (!j.is_object() || !j.contains("asr_stats")) {
                    // an error (or a connection that went away): hand that response back unchanged; the empty attempt passes every gate
                    failed = std::move(r);
                    return att;
                }
                const json & st    = j.at("asr_stats");
                att.text           = j.value("text", std::string());
                att.avg_logprob    = st.value("avg_logprob", 0.0);
                att.n_tokens       = (int) st.value("n_tokens", (size_t) 0);
                att.tail_entropy   = st.value("tail_entropy", 1e9);
                kept.push_back(std::move(r));
                return att;
            });
            if (failed) {
                return failed;
            }
            GGML_ASSERT(fr.chosen >= 0 && (size_t) fr.chosen < kept.size());
            json attempts = json::array();
            for (size_t i = 0; i < fr.attempts.size(); ++i) {
                const auto & a = fr.attempts[i];
                const auto & v = fr.verdicts[i];
                attempts.push_back({
                    {"temperature",       a.temperature},
                    {"avg_logprob",       a.avg_logprob},
                    {"tail_entropy",      a.tail_entropy},
                    {"n_tokens",          a.n_tokens},
                    {"compression_ratio", v.compression_ratio},
                    {"reasons",           v.reasons},
                    {"retried",           v.needs_fallback},
                });
                SRV_INF("asr-quality: attempt %zu T=%.2f n_tokens=%d avg_logprob=%.3f tail_entropy=%.3f compression_ratio=%.2f reasons=0x%x -> %s\n",
                        i + 1, (double) a.temperature, a.n_tokens, a.avg_logprob, a.tail_entropy, v.compression_ratio, (unsigned) v.reasons,
                        v.needs_fallback ? "RETRY at the next temperature" : "accepted");
            }
            SRV_INF("asr-quality: %zu attempt(s), chosen attempt %d%s\n", fr.attempts.size(), fr.chosen + 1, fr.exhausted ? " (every temperature failed)" : "");
            auto out = std::move(kept[(size_t) fr.chosen]);
            json j = json::parse(out->data);
            j.erase("asr_stats");
            j["asr_quality"] = json { {"attempts", attempts}, {"chosen", fr.chosen}, {"exhausted", fr.exhausted} };
            out->ok(j);
            res_gen = std::move(out);
        } else {
            res_gen = handle_completions_impl(
                req,
                SERVER_TASK_TYPE_COMPLETION,
                body_parsed,
                files,
                TASK_RESPONSE_TYPE_OAI_ASR, session);
        }
        if (asr_fmt != asrfmt::format::json && res_gen && !res_gen->is_stream() && res_gen->status == 200) {
            try {
                const json done = json::parse(res_gen->data);
                double duration_s = 0.0;
                if (ctx_server.mctx != nullptr && !files.empty()) {
                    auto wrap = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, files[0].data(), files[0].size(), false, mtmd_helper_init_opt_default());
                    if (wrap.bitmap) {
                        if (mtmd_bitmap_is_audio(wrap.bitmap) && mtmd_get_audio_sample_rate(ctx_server.mctx) > 0) {
                            duration_s = (double) mtmd_bitmap_get_nx(wrap.bitmap) / (double) mtmd_get_audio_sample_rate(ctx_server.mctx);
                        }
                        mtmd_bitmap_free(wrap.bitmap);
                    }
                }
                std::string lang = json_value(json::parse(req.body), "language", std::string());
                if (lang.empty() && done.contains("language") && done.at("language").is_string()) {
                    // no language in the request: report the one the model tagged its output with (Qwen3-ASR "language English<asr_text>",
                    // stripped by asrfmt::strip_language_tag). OpenAI's verbose_json writes the full name in lower case ("english").
                    lang = done.at("language").get<std::string>();
                    for (auto & c : lang) { if (c >= 'A' && c <= 'Z') { c = (char) (c - 'A' + 'a'); } }
                }
                // avg_logprob is present when the request carried token probabilities (verbose_json asks for them; so does --asr-quality)
                const bool has_lp = done.contains("avg_logprob") && done.at("avg_logprob").is_number();
                // OpenAI's verbose_json also carries a temperature, a per-segment compression_ratio and usage. The temperature is the one the
                // reply was decoded at: the chosen --asr-quality attempt when the retry loop ran, else the request's own (default 0).
                std::vector<float> diar_l, diar_r;   // the two channels for vx.speaker; they outlive the render call below
                asrfmt::verbose_extras vx;
                if (asr_fmt == asrfmt::format::verbose_json) {
                    // the CONVERTED body: a form field is a string in req.body, and json_value fell back to 0.0 on it (with a
                    // "Wrong type supplied" warning), so verbose_json reported 0 whatever the request asked for
                    vx.temperature = json_value(body, "temperature", 0.0);
                    if (done.contains("asr_quality") && done.at("asr_quality").contains("attempts")) {
                        const auto & aq = done.at("asr_quality");
                        const int chosen = json_value(aq, "chosen", 0);
                        if (chosen >= 0 && (size_t) chosen < aq.at("attempts").size() && aq.at("attempts")[(size_t) chosen].contains("temperature")) {
                            vx.temperature = aq.at("attempts")[(size_t) chosen].at("temperature").get<double>();
                        }
                    }
                    vx.compression_ratio = [](const std::string & s) { return asr_compression_ratio(s); };
                    if (done.contains("no_speech_prob") && done.at("no_speech_prob").is_number()) {
                        vx.no_speech_prob = done.at("no_speech_prob").get<double>();
                    }
                    if (done.contains("usage") && done.at("usage").is_object()) {
                        vx.usage_json = done.at("usage").dump();
                    }
                    if (want_words && words_engine != nullptr) {
                        // replay the finished request on the aligner's own context (server-asr-words.h); a failure is an error, never guessed times
                        std::string werr;
                        std::vector<asrwords::engine::word> ws;
                        asrwords::timing wt;
                        const std::string text = json_value(done, "text", std::string());
                        const std::string prompt = body_parsed.contains("prompt") && body_parsed.at("prompt").is_string() ? body_parsed.at("prompt").get<std::string>() : std::string();
                        if (!text.empty() && !words_engine->align(prompt, files, text, true, ws, &wt, werr)) {
                            auto er = create_response();
                            er->error(format_error_response(werr, ERROR_TYPE_SERVER));
                            return er;
                        }
                        SRV_INF("word timestamps: %zu words, %d audio positions, %d tokens, %d heads; prefill %.0f ms, forced %.0f ms, align %.0f ms, create %.0f ms, ctx %zu bytes\n",
                                ws.size(), wt.n_audio_pos, wt.n_tokens, wt.n_heads, wt.prefill_ms, wt.forced_ms, wt.align_ms, wt.create_ms, words_engine->context_bytes());
                        vx.words_on = true;
                        for (const auto & w : ws) {
                            asrfmt::verbose_extras::word xw;
                            xw.text = w.text; xw.t0 = w.t0; xw.t1 = w.t1;
                            vx.words.push_back(std::move(xw));
                        }
                    }
                    // diarize (whisper.cpp server's form field, examples/server/server.cpp there): a per-segment "speaker" from
                    // the stereo channel energy over the segment's [start, end] (asr-diarize.h). The model hears the mono mix; the
                    // channels are decoded again here, by the same decoder at the same rate, so frame i is the model's sample i.
                    // Only for a 2-channel upload: a mono file gets no "speaker" key and a warning, never an invented label. The
                    // window is the segment's own, so the label is as good as its times ("timestamps": model or interpolated).
                    const json raw = json::parse(req.body);
                    bool diarize = false;
                    if (raw.contains("diarize")) {
                        const json & d = raw.at("diarize");
                        diarize = d.is_boolean() ? d.get<bool>() : (d.is_string() && (d.get<std::string>() == "true" || d.get<std::string>() == "1"));
                    }
                    if (diarize && ctx_server.mctx != nullptr && !files.empty()) {
                        const int32_t sr = mtmd_get_audio_sample_rate(ctx_server.mctx);
                        size_t n = 0;
                        // --asr-vad-trim put a mono cut in files[0]; the two channels are in the upload, and the segment times the
                        // callback gets are already the upload's (asrfmt::time_remap runs before it)
                        const raw_buffer & diar_src = asr_trim_map.empty() ? files[0] : asr_trim_upload;
                        const int32_t ch = mtmd_helper_audio_decode_stereo(diar_src.data(), diar_src.size(), sr, nullptr, 0, &n);
                        std::vector<float> lr(2 * n);
                        size_t got = 0;
                        if (ch == 2 && n > 0 && mtmd_helper_audio_decode_stereo(diar_src.data(), diar_src.size(), sr, lr.data(), n, &got) == 2 && got > 0) {
                            diar_l.resize(got);
                            diar_r.resize(got);
                            for (size_t i = 0; i < got; ++i) { diar_l[i] = lr[2 * i]; diar_r[i] = lr[2 * i + 1]; }
                            vx.speaker = [&diar_l, &diar_r, sr](double t0, double t1) {
                                size_t i0 = (size_t) std::max(0.0, t0 * (double) sr);
                                size_t i1 = (size_t) std::max(0.0, t1 * (double) sr);
                                if (i1 == 0 || i1 > diar_l.size()) { i1 = diar_l.size(); }
                                i0 = std::min(i0, i1);
                                return asr_stereo_speaker(diar_l.data(), diar_r.data(), i0, i1);
                            };
                            SRV_INF("diarize: stereo speaker labels on (%zu frames at %d Hz)\n", got, (int) sr);
                        } else {
                            SRV_WRN("diarize requested but the upload decodes to %d channel(s), not 2; segments carry no speaker\n", (int) ch);
                        }
                    }
                }
                // --asr-vad-trim: duration_s above is the length of what the model heard (files[0] is the cut); the segment times are
                // made on it and then taken back to the upload, whose length is the "duration" the client reads
                asrfmt::time_remap rm;
                if (!asr_trim_map.empty()) {
                    rm.segment    = [&asr_trim_map](double & t0, double & t1) { asr_trim_map.segment_to_original(t0, t1); };
                    // word times come from the aligner on the trimmed signal: midpoint rule, one piece per word (asr-vad-trim.h)
                    rm.word       = [&asr_trim_map](double & t0, double & t1) { asr_trim_map.word_to_original(t0, t1); };
                    rm.duration_s = asr_trim_upload_s;
                    rm.info_json  = asr_trim_info.dump();
                }
                // max_len / split_on_word: whisper.cpp server's subtitle wrapping fields (examples/server/server.cpp there: max_len,
                // split_on_word). Form fields arrive as strings; max_len <= 0 or absent = no wrapping.
                asrfmt::wrap_opts wrap;
                {
                    const json raw = json::parse(req.body);
                    if (raw.contains("max_len")) {
                        const json & v = raw.at("max_len");
                        try {
                            wrap.max_len = v.is_number() ? v.get<int>() : (v.is_string() ? std::stoi(v.get<std::string>()) : 0);
                        } catch (const std::exception &) {
                            wrap.max_len = 0;
                            SRV_WRN("transcription: max_len '%s' is not an integer; no wrapping\n", v.dump().c_str());
                        }
                    }
                    if (raw.contains("split_on_word")) {
                        const json & v = raw.at("split_on_word");
                        wrap.split_on_word = v.is_boolean() ? v.get<bool>() : !(v.is_string() && (v.get<std::string>() == "false" || v.get<std::string>() == "0"));
                    }
                }
                res_gen->data = asrfmt::render(asr_fmt, json_value(done, "text", std::string()), duration_s, lang, has_lp, has_lp ? done.at("avg_logprob").get<double>() : 0.0,
                                               asr_fmt == asrfmt::format::verbose_json ? &vx : nullptr, asr_trim_map.empty() ? nullptr : &rm,
                                               wrap.max_len > 0 ? &wrap : nullptr);
                res_gen->content_type = asrfmt::content_type(asr_fmt);
                SRV_INF("transcription response_format=%s rendered (audio %.3f s%s, %zu bytes)\n",
                        json_value(json::parse(req.body), "response_format", std::string()).c_str(),
                        asr_trim_map.empty() ? duration_s : asr_trim_upload_s, asr_trim_map.empty() ? "" : ", silence trimmed", res_gen->data.size());
            } catch (const std::exception & e) {
                SRV_ERR("transcription response_format rendering failed: %s\n", e.what());
            }
        } else if (!asr_trim_map.empty() && res_gen && !res_gen->is_stream() && res_gen->status == 200) {
            // response_format json, trimmed: the reply says so (the same "asr_vad" object verbose_json carries)
            try {
                json j = json::parse(res_gen->data);
                if (j.is_object()) {
                    j["asr_vad"] = asr_trim_info;
                    res_gen->ok(j);
                }
            } catch (const std::exception & e) {
                SRV_ERR("asr-vad-trim: could not add the asr_vad object to the reply: %s\n", e.what());
            }
        }
        return res_gen;
    };

    // realtime transcription over websocket, modeled after the OpenAI realtime API
    // audio is PCM16 mono at the model sample rate, sent as base64 in input_audio_buffer.append events
    // input_audio_buffer.commit with "final": true ends the input, closing the socket also ends it
    this->ws_realtime = [this](const server_http_req &, server_http_context::ws_conn & conn) {
        auto send_event = [&conn](json ev) {
            ev["event_id"] = "event_" + random_string();
            conn.send(safe_json_to_str(ev));
        };
        auto send_error_event = [&send_event](const std::string & message, const std::string & type) {
            send_event(json {
                {"type", "error"},
                {"error", {
                    {"type",    type},
                    {"code",    nullptr},
                    {"message", message},
                    {"param",   nullptr},
                }},
            });
        };

        if (!meta->has_audio_stream) {
            send_error_event("The current model does not support realtime transcription.", "invalid_request_error");
            conn.close();
            return;
        }

        const std::string session_id = "sess_" + random_string();
        const json session_config = json {
            {"id",     session_id},
            {"object", "realtime.transcription_session"},
            {"type",   "transcription"},
            {"audio",  {{"input", {
                {"format",          {{"type", "audio/pcm"}, {"rate", mtmd_get_audio_sample_rate(ctx_server.mctx)}}},
                {"transcription",   {{"model", meta->model_name}}},
                {"turn_detection",  nullptr},
                {"noise_reduction", nullptr},
            }}}},
        };

        auto res = create_response();
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_TRANSCRIBE_RT);
            task.id = rd.get_new_id();
            task.params.sampling          = params.sampling;
            task.params.stream            = true;
            task.params.oaicompat_cmpl_id = session_id;
            task.params.oaicompat_model   = meta->model_name;
            rd.post_task(std::move(task));
        }

        // the slot answers once its audio stream is live, the session starts there
        {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result || result->is_error()) {
                if (result) {
                    send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                }
                conn.close();
                return;
            }
        }
        send_event(json {{"type", "session.created"}, {"session", session_config}});

        // one item per committed audio buffer, deltas are attributed to the current item
        std::mutex  mutex_item;
        std::string item_id = "item_" + random_string();
        std::string item_prev; // the committed item before item_id, empty for the first one
        std::string item_text;

        // called with mutex_item held
        // per item: the Qwen3-ASR "language X<asr_text>" tag is held out of the deltas and the transcript (server-asr-format.h)
        asrfmt::language_tag_stream item_tag;
        auto send_delta = [&](const std::string & piece) {
            if (piece.empty()) {
                return;
            }
            item_text += piece;
            send_event(json {
                {"type",          "conversation.item.input_audio_transcription.delta"},
                {"item_id",       item_id},
                {"content_index", 0},
                {"delta",         piece},
            });
        };
        auto send_completed = [&]() {
            send_delta(item_tag.flush());
            item_tag.reset();
            send_event(json {
                {"type",          "conversation.item.input_audio_transcription.completed"},
                {"item_id",       item_id},
                {"content_index", 0},
                {"transcript",    item_text},
            });
            item_text.clear();
        };

        auto post_control = [&](server_control_action action, std::vector<float> && audio) {
            server_response_reader ctl(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id                     = ctl.get_new_id();
            task.params.control_action  = action;
            task.params.control_cmpl_id = session_id;
            task.params.control_audio   = std::move(audio);
            ctl.post_task(std::move(task));
            auto result = ctl.next([&conn]() { return !conn.is_open(); });
            auto * control = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
            return control && control->success;
        };

        // client events are read on this thread, the transcript is sent from the thread below
        std::atomic<bool> input_ended{false};
        std::thread reader([&]() {
            std::string msg;
            while (conn.read(msg) != 0) {
                json ev;
                try {
                    ev = json::parse(msg);
                } catch (const std::exception &) {
                    send_error_event("invalid JSON", "invalid_request_error");
                    continue;
                }
                const std::string type = json_value(ev, "type", std::string());
                if (type == "session.update" || type == "transcription_session.update") {
                    send_event(json {{"type", "session.updated"}, {"session", session_config}});
                } else if (type == "input_audio_buffer.append") {
                    std::string pcm;
                    try {
                        pcm = base64::decode(json_value(ev, "audio", std::string()));
                    } catch (const std::exception &) {
                        send_error_event("invalid base64 audio", "invalid_request_error");
                        continue;
                    }
                    std::vector<float> samples(pcm.size() / 2);
                    for (size_t i = 0; i < samples.size(); i++) {
                        int16_t v;
                        memcpy(&v, pcm.data() + 2 * i, sizeof(v));
                        samples[i] = v / 32768.0f;
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_APPEND, std::move(samples))) {
                        break;
                    }
                } else if (type == "input_audio_buffer.commit") {
                    // "final": true ends the input, the last item is completed when the stream drains
                    if (json_value(ev, "final", false)) {
                        input_ended = true;
                        post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
                        continue;
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_COMMIT, {})) {
                        break;
                    }
                    // completed is sent once the rows of the committed audio are decoded (audio_commit_drained below)
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_event(json {
                        {"type",             "input_audio_buffer.committed"},
                        {"previous_item_id", item_prev.empty() ? json(nullptr) : json(item_prev)},
                        {"item_id",          item_id},
                    });
                } else if (type == "input_audio_buffer.clear") {
                    send_event(json {{"type", "input_audio_buffer.cleared"}});
                } else {
                    send_error_event("unknown event type: " + type, "invalid_request_error");
                }
            }
            if (!input_ended.exchange(true)) {
                post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
            }
            // the result loop waits on the socket, nobody reads it from here on
            conn.close();
        });

        while (true) {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result) {
                break; // connection closed
            }
            if (result->is_error()) {
                send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                break;
            }
            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                if (partial->audio_commit_drained) {
                    // complete the committed item and open the next one
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_completed();
                    item_prev = item_id;
                    item_id   = "item_" + random_string();
                } else if (!partial->content.text.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_delta(item_tag.push(partial->content.text));
                }
            } else if (result->is_stop()) {
                std::lock_guard<std::mutex> lock(mutex_item);
                send_completed();
                break;
            }
        }

        input_ended = true;
        conn.close();
        reader.join();
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    // /apply-template plus the tokens of the templated prompt, so a client can budget a chat request or
    // highlight spans in it without a second round trip. Body: the chat-completions body, plus the
    // optional "add_special" (default false, the template already carries BOS), "parse_special"
    // (default true), "with_pieces" and "with_offsets" of /tokenize. Reply: {"prompt", "tokens"}.
    // Shape of text-generation-inference router/src/server.rs chat_tokenize (Apache-2.0); no code copied.
    this->post_chat_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        if (!body.is_object()) {
            throw std::invalid_argument("request body must be a JSON object");
        }
        const bool add_special  = json_value(body, "add_special", false);
        const bool parse_special = json_value(body, "parse_special", true);
        const bool with_pieces  = json_value(body, "with_pieces", false);
        const bool with_offsets = json_value(body, "with_offsets", false);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        const std::string prompt = data.at("prompt").get<std::string>();
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, json(prompt), add_special, parse_special);
        json tokens_json;
        if (with_pieces || with_offsets) {
            tokens_json = tokens_to_json_detailed(ctx_server.vocab, tokens, with_pieces, with_offsets);
        } else {
            tokens_json = tokens;
        }
        res->ok(json{{ "prompt", prompt }, { "tokens", std::move(tokens_json) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_object()) {
            throw std::invalid_argument("request body must be a JSON object");
        }
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);
            const bool with_offsets = json_value(body, "with_offsets", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces || with_offsets) {
                tokens_response = tokens_to_json_detailed(ctx_server.vocab, tokens, with_pieces, with_offsets);
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    // Gumbel watermark detector (common/watermark.h). Body: {"tokens": [...]} -- the GENERATED ids, as
    // returned with "return_tokens" -- or {"content": "..."}, retokenized here without special tokens.
    // Retokenizing text can split differently from the ids that were sampled, which only lowers the
    // score; ids are exact. Uses the server's own key, which is never accepted from the request.
    this->post_watermark_detect = [this](const server_http_req & req) {
        auto res = create_response();
        const auto wm = ctx_server.get_watermark_settings();
        if (!wm.key_set) {
            res->error(format_error_response("This server has no watermark key. Start it with `--watermark-key`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        const json body = json::parse(req.body);
        llama_tokens tokens;
        if (body.contains("tokens")) {
            tokens = body.at("tokens").get<llama_tokens>();
        } else if (body.contains("content")) {
            tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), false, false);
        } else {
            res->error(format_error_response("\"tokens\" or \"content\" is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const int    width  = json_value(body, "context_width", (int) wm.width);
        const bool   dedup  = json_value(body, "deduplicate_contexts", true);
        const double thresh = json_value(body, "p_value_threshold", common_watermark::DEFAULT_P_THRESHOLD);
        // Upper bound: detect() keeps a width-sized context per scored token, so an unbounded width from a
        // request is a memory exhaustion lever. vLLM already warns past 16; 1024 leaves room and stays small.
        if (width < 1 || width > 1024 || !(thresh > 0.0 && thresh < 1.0)) {
            res->error(format_error_response("context_width must be in [1, 1024] and 0 < p_value_threshold < 1", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        // "scheme" defaults to the server's; kgw's gamma and synthid's depth come from the server too, because a
        // detector run with parameters other than the generator's measures nothing.
        const std::string scheme_s = json_value(body, "scheme", std::string(common_watermark::scheme_name(wm.scheme)));
        const int32_t     scheme   = common_watermark::scheme_from_name(scheme_s.c_str());
        if (scheme < 0) {
            res->error(format_error_response("\"scheme\" must be gumbel, kgw or synthid", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        common_watermark::detection d;
        if (scheme == common_watermark::SCHEME_KGW) {
            d = common_watermark::detect_kgw(wm.key, tokens, wm.kgw_gamma, dedup, thresh);
        } else if (scheme == common_watermark::SCHEME_SYNTHID) {
            d = common_watermark::detect_synthid(wm.key, tokens, width, wm.depth, dedup, thresh);
        } else {
            d = common_watermark::detect(wm.key, tokens, width, dedup, thresh);
        }
        res->ok(json{
            {"scheme",            common_watermark::scheme_name(scheme)},
            {"score",             d.score},
            {"p_value",           d.p_value},
            {"num_scored_tokens", d.n_scored},
            {"num_tokens",        tokens.size()},
            {"is_watermarked",    d.is_watermarked},
        });
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        // SOURCE-READ: vllm/entrypoints/serve/tokenize/protocol.py:164-170 (tokens: list[int], ge=0), renderers/base.py:449-455
        // ("Token id N is out of vocabulary" -> 400). Out-of-range ids used to reach id_to_token.at() and answer 500.
        if (!body.is_object() || body.count("tokens") == 0) {
            throw std::invalid_argument("'tokens' is required");
        }
        const json & jtokens = body.at("tokens");
        if (!jtokens.is_array()) {
            throw std::invalid_argument("'tokens' must be an array of token ids");
        }
        const int32_t n_vocab_detok = llama_vocab_n_tokens(ctx_server.vocab);
        llama_tokens tokens;
        for (size_t i = 0; i < jtokens.size(); ++i) {
            const json & t = jtokens[i];
            if (!t.is_number_integer()) {
                throw std::invalid_argument("tokens[" + std::to_string(i) + "] must be an integer token id");
            }
            const int64_t id = t.get<int64_t>();
            if (id < 0 || id >= n_vocab_detok) {
                throw std::invalid_argument("tokens[" + std::to_string(i) + "]=" + std::to_string(id) +
                                            " is outside the vocabulary [0, " + std::to_string(n_vocab_detok) + ")");
            }
            tokens.push_back((llama_token) id);
        }
        const std::string content = tokens_to_str(ctx_server.vocab, tokens);

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_decision = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        server_task task(SERVER_TASK_TYPE_PARALLEL_DECISION);
        task.id               = res->rd.get_new_id();
        task.decision_request = body;
        res->rd.post_task(std::move(task));

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            return res; // the client went away
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_score = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        server_task task(SERVER_TASK_TYPE_SCORE);
        task.id            = res->rd.get_new_id();
        task.score_request = body;
        res->rd.post_task(std::move(task));

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            return res; // the client went away
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_logits = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        server_task task(SERVER_TASK_TYPE_LOGITS);
        task.id            = res->rd.get_new_id();
        task.score_request = body;
        res->rd.post_task(std::move(task));

        auto result = res->rd.next(req.should_stop);
        if (!result) {
            return res; // the client went away
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_classify = [this](const server_http_req & req) {
        return handle_classify_impl(req);
    };
    this->post_rerank = [this](const server_http_req & req) {
        return handle_rerank_impl(req);
    };

    this->post_tts = [this](const server_http_req & req) {
        auto res = create_response();
        res->set_req(&req); // will also set spipe if needed

        if (!ctx_server.has_cap_tts()) {
            res->error(format_error_response("this server does not support audio generation", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const auto info = mtmd_gen_audio_get_info(ctx_server.mctx);

        json body = json::parse(req.body);

        // multipart form fields arrive as strings, correct the types of the numeric and boolean params
        for (const char * k : { "top_k", "n_predict", "max_tokens", "seed" }) {
            if (body.contains(k) && body[k].is_string()) {
                body[k] = std::stoll(body[k].get<std::string>());
            }
        }
        for (const char * k : { "top_p", "repeat_penalty" }) {
            if (body.contains(k) && body[k].is_string()) {
                body[k] = std::stof(body[k].get<std::string>());
            }
        }
        if (body.contains("stream") && body["stream"].is_string()) {
            body["stream"] = body["stream"] == "true";
        }

        std::string prompt = json_value(body, "input", json_value(body, "prompt", std::string()));
        if (prompt.empty()) {
            res->error(format_error_response("\"input\" must be a non-empty string", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string response_format = json_value(body, "response_format", std::string("wav"));

        server_task task(SERVER_TASK_TYPE_TTS);
        server_schema::eval_tts_schema(params, task, body);
        const bool stream = task.params.stream;
        task.tts_inp.set_prompt(prompt);
        task.tts_inp.set_lang(json_value(body, "lang", std::string()));
        task.tts_inp.data.stream    = stream;
        task.tts_inp.data.out_type  = response_format == "pcm"
            ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
            : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
        // note: -1 is clamped to 0 by the sampler, it will disable the penalty
        task.params.sampling.penalty_last_n = task.params.n_predict > 0 ? task.params.n_predict : 512;
        task.params.sampling.seed = task.tts_inp.data.seed; // same seed for the codec/vocoder RNG
        if (task.tts_inp.data.top_k > 0) {
            task.params.sampling.top_k = task.tts_inp.data.top_k;
        }
        if (task.tts_inp.data.top_p > 0) {
            task.params.sampling.top_p = task.tts_inp.data.top_p;
        }

        // speaker reference: either an uploaded form file ("speaker_ref") or a base64 JSON field ("speaker_ref_b64")
        const unsigned char * speaker_ref_data = nullptr;
        size_t speaker_ref_len  = 0;
        std::string speaker_ref_b64_decoded;

        auto speaker_ref_file = req.files.find("speaker_ref");
        if (speaker_ref_file != req.files.end()) {
            speaker_ref_data = speaker_ref_file->second.data.data();
            speaker_ref_len  = speaker_ref_file->second.data.size();
        } else {
            std::string speaker_ref_b64 = json_value(body, "speaker_ref_b64", std::string());
            if (!speaker_ref_b64.empty()) {
                try {
                    speaker_ref_b64_decoded = base64::decode(speaker_ref_b64);
                } catch (const std::exception &) {
                    res->error(format_error_response("\"speaker_ref_b64\" is not valid base64", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                speaker_ref_data = (const unsigned char *) speaker_ref_b64_decoded.data();
                speaker_ref_len  = speaker_ref_b64_decoded.size();
            }
        }

        if (speaker_ref_len > 0) {
            auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, speaker_ref_data, speaker_ref_len, false, ctx_server.init_opt);
            if (!wrapper.bitmap) {
                res->error(format_error_response("failed to decode \"speaker_ref\"", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(wrapper.bitmap));
        } else {
            SRV_WRN("%s", "no speaker reference provided, the model may behave randomly\n");
        }

        auto & rd = res->rd;
        task.id = rd.get_new_id();
        rd.post_task(std::move(task));

        // raw float32 LE mono samples; audio/L16 would imply 16-bit big-endian (RFC 2586)
        const std::string content_type = response_format == "pcm"
            ? "audio/pcm;rate=" + std::to_string(info.sample_rate) + ";encoding=float;bits=32"
            : "audio/wav";

        if (!stream) {
            auto result = rd.next(req.should_stop);
            if (!result) {
                GGML_ASSERT(req.should_stop());
                return res; // connection is closed
            }
            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }
            auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
            GGML_ASSERT(tts_res != nullptr);
            res->status       = 200;
            res->content_type = content_type;
            res->data         = std::move(tts_res->audio);
            return res;
        } else {
            auto first_result = rd.next(req.should_stop);
            if (!first_result) {
                GGML_ASSERT(req.should_stop());
                return res; // connection is closed
            }
            if (first_result->is_error()) {
                res->error(first_result->to_json());
                return res;
            }
            auto * first_tts_res = dynamic_cast<server_task_result_tts *>(first_result.get());
            GGML_ASSERT(first_tts_res != nullptr);

            res->status       = 200;
            res->content_type = content_type;
            res->data          = std::move(first_tts_res->audio);
            bool is_done       = first_tts_res->final;

            res->set_next([res_this = res.get(), is_done](std::string & output) mutable -> bool {
                // flush buffered audio before is_done - the first result can already be final
                if (!res_this->data.empty()) {
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }
                if (is_done) {
                    return false;
                }
                if (res_this->should_stop()) {
                    return false;
                }

                server_response_reader & rd = res_this->rd;
                if (!rd.has_next()) {
                    return false;
                }
                auto result = rd.next([&res_this]() { return res_this->should_stop(); });
                if (!result || result->is_error()) {
                    return false;
                }
                auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
                GGML_ASSERT(tts_res != nullptr);
                output  = std::move(tts_res->audio);
                is_done = tts_res->final;
                return true;
            });
        }

        return res;
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.prepare_state(decision.parse_state(body, files));
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        std::vector<size_t> n_win_variant; // tasks (windows of the state) of each variant, in order
        int32_t n_cut_total = 0;
        size_t  n_win_max   = 1;
        size_t  n_win_dropped = 0;
        std::string win_agg = decision.get_window_agg(questions[0]);
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.mctx, ctx_server.init_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        // a state that does not fit the context is read in windows, each one is a task with the whole head
                        const size_t n_win = std::max<int32_t>(1, task.decision.n_windows);
                        const size_t n_run = decision.get_window_agg(question) == "first" ? 1 : n_win;
                        n_cut_total += task.decision.n_cut;
                        n_win_max = std::max(n_win_max, n_run);
                        n_win_dropped += n_win - n_run; // "first" reads one window only, for comparison
                        tasks.push_back(std::move(task));
                        for (size_t window = 1; window < n_run; window++) {
                            server_task task_w = server_task(SERVER_TASK_TYPE_DECISION);
                            task_w.id = rd.get_new_id();
                            decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task_w, window);
                            tasks.push_back(std::move(task_w));
                        }
                        n_win_variant.push_back(n_run);
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        size_t i_variant = 0;
        json usage_windows = json::object();
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                // the windows of this variant: the logits of each option are combined
                std::vector<std::vector<float>> windows;
                for (size_t w = 0; w < n_win_variant.at(i_variant); w++) {
                    auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[i_result++].get());
                    GGML_ASSERT(result != nullptr);
                    windows.push_back(result->scores);
                    n_tokens += result->n_tokens;
                }
                i_variant++;
                if (json_value(body, "decision_window_logits", false)) {
                    // measurement: the raw logits of every window, to compare the aggregations offline
                    if (!usage_windows.contains(question.id)) {
                        usage_windows[question.id] = json::array();
                    }
                    usage_windows[question.id].push_back(windows);
                }
                scores.push_back(windows.size() == 1 ? windows[0] : server_decision_aggregate(windows, decision.get_window_agg(question)));
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        json usage = json{
            {"input_tokens",  n_tokens},
            {"output_tokens", 0},
        };
        if (n_win_max > 1) {
            // the state was read in windows, nothing was cut
            usage["windows"]            = n_win_max;
            usage["window_aggregation"] = win_agg;
        }
        if (!usage_windows.empty()) {
            usage["window_logits"] = usage_windows;
        }
        if (n_win_dropped > 0) {
            usage["truncated"]       = true; // only with --decision-window-agg first
            usage["windows_dropped"] = n_win_dropped;
        }
        if (n_cut_total > 0) {
            // --decision-budget model|N cuts the question and the options, say so
            usage["truncated"]        = true;
            usage["truncated_tokens"] = n_cut_total;
        }
        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   usage},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    // operator sampler overrides (server-overrides.h): GET shows the layers, POST replaces one or both (admin tier, mgmt()).
    // A layer that fails validation changes nothing.
    this->get_sampler_overrides = [this](const server_http_req &) {
        auto res = create_response(true);
        res->ok(server_override_instance().to_json());
        return res;
    };
    this->post_sampler_overrides = [this](const server_http_req & req) {
        auto res = create_response(true);
        const json body = json::parse_no_throw(req.body);
        if (body.is_discarded() || !body.is_object() || (!body.contains("global") && !body.contains("model"))) {
            res->error(format_error_response("body must be a JSON object with a \"global\" and/or \"model\" layer (an object, or null to empty it)", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const auto & groups = server_schema::override_groups();
        std::vector<std::string> warnings;
        std::string err;
        server_override_layer g, m;
        const bool set_g = body.contains("global");
        const bool set_m = body.contains("model");
        if (set_g && !server_override_parse(body.at("global"), groups, "\"global\"", g, err, warnings)) {
            res->error(format_error_response(err, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (set_m && !server_override_parse(body.at("model"), groups, "\"model\"", m, err, warnings)) {
            res->error(format_error_response(err, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (set_g) { server_override_instance().set_global(std::move(g)); }
        if (set_m) { server_override_instance().set_model(std::move(m)); }
        for (const auto & w : warnings) {
            SRV_WRN("%s\n", w.c_str());
        }
        json out = server_override_instance().to_json();
        json w = json::array();
        for (const auto & s : warnings) {
            w.push_back(s);
        }
        out["warnings"] = w;
        res->ok(out);
        return res;
    };

    // HOOK-OVERRIDE: sst -- this block's "server_task::control_vector_scale_update" (C++ scope
    // resolution syntax, matches the bias-hook's "task:" substring check) is not a task/plan marker.
    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, groups G5: info
    // and (now) G3: SQL session store). SOURCE-READ: examples/server/server.cpp:1973-1988 @
    // c3d1c7480 (both branches of handle_version). "sql" now reports whether sql_db is actually
    // open (compiled in, --sql-save-file given, and the file/schema opened without error) -- the
    // same condition every G3 route itself checks -- rather than the flag's mere presence, so a
    // client can't be told "sql": true while every /save call would 501. "zstd_compression" reports the
    // same way for G4: true only when the sqlite-zstd extension actually loaded into the open store
    // (row S69-G4-ZSTD-ROUTES-PORT-USER-APPROVED-2026-09-27-01) -- the condition every /zstd_* route checks.
    this->get_version = [this](const server_http_req &) {
        auto res = create_response(true);
        res->ok({
            { "version", 4 },
            { "features", { { "sql", sql_db != nullptr }, { "zstd_compression", sql_db != nullptr && sql_zstd_loaded } } },
        });
        return res;
    };

    // SOURCE-READ: examples/server/server.cpp:1068-1088 @ c3d1c7480 (handle_props_simple). The old
    // handler also computed n_past/slot_id from ctx_server.slots but never used them in its
    // response -- dead even in the original, dropped here rather than ported as-is. Built from
    // *meta like get_props/get_res_props just above, not a raw ctx_server read (server_routes has
    // no access to ctx_server's private slot/params_base state, by design -- see server-context.h).
    this->get_props_v1 = [this](const server_http_req &) {
        auto res = create_response();
        res->ok({
            { "model_name", meta->model_name },
            { "model_path", meta->model_path },
            { "modalities", {
                { "vision", meta->has_inp_image },
                { "audio",  meta->has_inp_audio },
            } },
            { "n_ctx", meta->slot_n_ctx },
        });
        return res;
    };

    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G1: runtime
    // control vectors). Routed through the task queue / main loop like SERVER_TASK_TYPE_GET_LORA /
    // SET_LORA just above -- never a raw HTTP-thread read or mutation of server state.
    this->get_control_vectors = [this](const server_http_req & req) {
        auto res = create_response();
        auto & rd = res->rd;
        {
            server_task cv_task(SERVER_TASK_TYPE_GET_CONTROL_VECTORS);
            cv_task.id = rd.get_new_id();
            rd.post_task(std::move(cv_task));
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_control_vectors_list*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_control_vectors_load = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.contains("path") || !body.at("path").is_string()) {
            res->error(format_error_response("\"path\" must be a string", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        // --control-vector-root: when set, only files under that directory can be loaded (a relative path is taken relative to it).
        // Without the flag any readable path is accepted, as before.
        if (!params.control_vector_root.empty()) {
            namespace fs = std::filesystem;
            std::error_code ec;
            const fs::path root = fs::weakly_canonical(fs::path(params.control_vector_root), ec);
            fs::path want = fs::path(body.at("path").get<std::string>());
            if (want.is_relative()) {
                want = root / want;
            }
            want = fs::weakly_canonical(want, ec);
            const fs::path rel = ec ? fs::path() : want.lexically_relative(root);
            const bool inside = !ec && !rel.empty() && *rel.begin() != fs::path("..") && !rel.is_absolute();
            if (!inside) {
                res->error(format_error_response("control vector path is outside --control-vector-root", ERROR_TYPE_PERMISSION));
                return res;
            }
        }

        auto & rd = res->rd;
        {
            server_task cv_task(SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR);
            cv_task.id = rd.get_new_id();
            cv_task.control_vector_load.path        = body.at("path").get<std::string>();
            cv_task.control_vector_load.scale       = json_value(body, "scale", 1.0f);
            cv_task.control_vector_load.layer_start = json_value(body, "layer_start", 1);
            cv_task.control_vector_load.layer_end   = json_value(body, "layer_end", -1);
            rd.post_task(std::move(cv_task));
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_control_vector*>(result.get()) != nullptr);
        bool ok = static_cast<server_task_result_control_vector*>(result.get())->success;
        res->status = ok ? 200 : 400;
        res->data = result->to_json().dump();
        res->content_type = "application/json";
        return res;
    };

    this->post_control_vectors_unload = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.contains("id") || body.at("id").is_null() || !body.at("id").is_number()) {
            res->error(format_error_response("Missing or invalid 'id' field", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task cv_task(SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR);
            cv_task.id = rd.get_new_id();
            cv_task.control_vector_unload_id = body.at("id").get<int>();
            rd.post_task(std::move(cv_task));
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_control_vector*>(result.get()) != nullptr);
        bool ok = static_cast<server_task_result_control_vector*>(result.get())->success;
        res->status = ok ? 200 : 400;
        res->data = result->to_json().dump();
        res->content_type = "application/json";
        return res;
    };

    this->post_control_vectors_apply = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task cv_task(SERVER_TASK_TYPE_SET_CONTROL_VECTOR);
            cv_task.id = rd.get_new_id();
            for (const auto & entry : body) {
                server_task::control_vector_scale_update u;
                u.id    = entry.at("id").get<int>();
                u.scale = entry.at("scale").get<float>();
                if (entry.contains("layer_start")) {
                    u.has_layer_start = true;
                    u.layer_start     = entry.at("layer_start").get<int32_t>();
                }
                if (entry.contains("layer_end")) {
                    u.has_layer_end = true;
                    u.layer_end     = entry.at("layer_end").get<int32_t>();
                }
                cv_task.control_vector_apply.push_back(u);
            }
            rd.post_task(std::move(cv_task));
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        GGML_ASSERT(dynamic_cast<server_task_result_control_vector*>(result.get()) != nullptr);
        bool ok = static_cast<server_task_result_control_vector*>(result.get())->success;
        res->status = ok ? 200 : 400;
        res->data = result->to_json().dump();
        res->content_type = "application/json";
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // Per-request adapters, the same three fields the completion routes take ("lora", "lora_adapter",
    // "lora_adapter_scale"): one embedder plus N adapters on one server instead of N servers. The tasks carry them and
    // the slot launch path applies them exactly as for a completion (task_lora_list / the --lora-pool-dir pin); tasks
    // with different adapter lists never share a batch (can_batch_with compares the lists).
    // Row S69-ENGREV-LORAX-EMBED-RERANK-PER-REQUEST-LORA-2026-10-27-01.
    std::map<int, float> req_lora;
    std::string          req_lora_adapter;
    float                req_lora_adapter_scale = 1.0f;
    try {
        parse_request_lora_fields(body, params.lora_adapters.size(), req_lora, req_lora_adapter, req_lora_adapter_scale);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    // "binary" / "ubinary" / "int8": packed sign bits or scaled bytes (row S69-VESPA-PICK-BINARY-INT8-EMBEDDING-OUTPUT-2026-10-15-01),
    // see server-embd-quant.h. SERVER_EMBD_QUANT_NONE keeps every existing response byte for byte.
    server_embd_quant quant = SERVER_EMBD_QUANT_NONE;
    std::string quant_name;
    if (body.count("encoding_format") != 0) {
        if (!body.at("encoding_format").is_string()) {
            res->error(format_error_response("\"encoding_format\" must be a string", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const std::string format = body.at("encoding_format").get<std::string>();
        if (format == "base64") {
            use_base64 = true;
        } else if (format == "binary" || format == "ubinary" || format == "int8") {
            quant      = server_embd_quant_from_name(format);
            quant_name = format;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float, base64, binary, ubinary or int8", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }
    if (quant != SERVER_EMBD_QUANT_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE && !server_embd_quant_rows_ok(quant)) {
        // pooling 'none' returns one vector per token (only on the non-OAI route, refused above for OAI). Those rows are not
        // normalised, so only the sign-bit modes are defined for them (row S69-VESPA-PICK-PER-TOKEN-BINARY-EMBEDDING-2026-10-18-01)
        res->error(format_error_response("encoding_format '" + quant_name + "' needs a pooled embedding; with pooling type 'none' (one vector per token) only binary or ubinary are accepted", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    // dispatch to the appropriate tokenizer based on the shape of the input
    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // "long_text_mode": what to do with a text that does not fit one micro-batch / the trained context. "error" (default) keeps the
    // existing refusal; "truncate" embeds the first chunk; "mean" embeds overlapping chunks and averages them (server-embd-chunk.h,
    // row S69-ENGINE-MERGE-GPT4ALL-2026-09-27-01). "long_text_overlap" = tokens shared by neighbouring chunks (default 8).
    server_embd_long_text long_text = SERVER_EMBD_LONG_TEXT_ERROR;
    size_t                long_text_overlap = 8;
    if (body.count("long_text_mode") != 0) {
        if (!body.at("long_text_mode").is_string() || !server_embd_long_text_from_name(body.at("long_text_mode").get<std::string>(), long_text)) {
            res->error(format_error_response("\"long_text_mode\" must be one of error, truncate, mean", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }
    if (body.count("long_text_overlap") != 0) {
        if (!body.at("long_text_overlap").is_number_integer() || body.at("long_text_overlap").get<int64_t>() < 0) {
            res->error(format_error_response("\"long_text_overlap\" must be a non-negative integer", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        long_text_overlap = (size_t) body.at("long_text_overlap").get<int64_t>();
    }
    if (long_text != SERVER_EMBD_LONG_TEXT_ERROR && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("\"long_text_mode\" needs a pooled embedding; with pooling type 'none' (one vector per token) a long text cannot be reduced to one vector", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Split every text longer than one micro-batch (or the trained context) into overlapping chunks that each keep the leading
    // special token (BOS/CLS) and the trailing one (EOS/SEP). `owner[i]` = the request input that work item i belongs to.
    // Without long_text_mode nothing is split and every byte of the old path is unchanged.
    const size_t n_inputs = tokenized_prompts.size();
    std::vector<size_t> owner(n_inputs);
    for (size_t i = 0; i < n_inputs; i++) {
        owner[i] = i;
    }
    std::vector<size_t> n_chunks(n_inputs, 1);
    if (long_text != SERVER_EMBD_LONG_TEXT_ERROR) {
        size_t n_max = (size_t) std::max<int32_t>(1, meta->n_ubatch);
        if (meta->model_n_ctx_train > 0) {
            n_max = std::min<size_t>(n_max, (size_t) meta->model_n_ctx_train);
        }
        std::vector<server_tokens> expanded;
        std::vector<size_t>        expanded_owner;
        for (size_t i = 0; i < n_inputs; i++) {
            auto & t = tokenized_prompts[i];
            const llama_tokens text = t.get_text_tokens();
            // media inputs (an image or audio placeholder in the list) cannot be cut by token count: leave them to the existing path
            if (t.size() <= n_max || text.size() != t.size()) {
                expanded.push_back(std::move(t));
                expanded_owner.push_back(i);
                continue;
            }
            const llama_token bos = llama_vocab_bos(ctx_server.vocab);
            const llama_token eos = llama_vocab_eos(ctx_server.vocab);
            const llama_token sep = llama_vocab_sep(ctx_server.vocab);
            const size_t head = (bos != LLAMA_TOKEN_NULL && text.front() == bos) ? 1 : 0;
            const size_t tail = (text.size() > head && ((eos != LLAMA_TOKEN_NULL && text.back() == eos) || (sep != LLAMA_TOKEN_NULL && text.back() == sep))) ? 1 : 0;
            const size_t n_body = text.size() - head - tail;
            const size_t max_body = n_max > head + tail ? n_max - head - tail : 0;
            std::vector<server_embd_chunk_range> ranges;
            std::string                          plan_err;
            if (!server_embd_chunk_plan(n_body, max_body, long_text_overlap, long_text == SERVER_EMBD_LONG_TEXT_TRUNCATE, ranges, plan_err)) {
                res->error(format_error_response(plan_err, ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            n_chunks[i] = ranges.size();
            for (const auto & rg : ranges) {
                llama_tokens piece;
                piece.reserve(head + (rg.end - rg.begin) + tail);
                if (head) { piece.push_back(text.front()); }
                piece.insert(piece.end(), text.begin() + head + rg.begin, text.begin() + head + rg.end);
                if (tail) { piece.push_back(text.back()); }
                expanded.emplace_back(piece, ctx_server.mctx != nullptr);
                expanded_owner.push_back(i);
            }
        }
        tokenized_prompts = std::move(expanded);
        owner             = std::move(expanded_owner);
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // "dimensions": Matryoshka/MRL truncation, as documented by both the OpenAI-compatible
    // embeddings protocol and ollama's /api/embed (row S69-SERVING-SURFACE-EMBED-DIMENSIONS-2026-09-27-01).
    // -1 means "field absent, no truncation, return the model's native size" (unchanged behavior).
    int32_t dimensions = -1;
    if (body.count("dimensions") != 0) {
        const json & dims_val = body.at("dimensions");
        if (!dims_val.is_number_integer()) {
            res->error(format_error_response("\"dimensions\" must be an integer", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        dimensions = dims_val.get<int32_t>();
        const int32_t n_embd_native = llama_model_n_embd_out(ctx_server.model_tgt);
        if (dimensions <= 0 || dimensions > n_embd_native) {
            res->error(format_error_response(
                string_format("\"dimensions\" must be between 1 and the model's native embedding size (%d), got %d",
                    n_embd_native, dimensions),
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // the dimension the quantised vector is cut to (and packed from): the requested head, else the model's native size.
    // A binary mode needs a multiple of 8 -- refused up front with the number the client can fix, before any compute.
    const int32_t quant_dim = dimensions > 0 ? dimensions : llama_model_n_embd_out(ctx_server.model_tgt);
    if (quant != SERVER_EMBD_QUANT_NONE && !server_embd_quant_dim_ok(quant, quant_dim)) {
        res->error(format_error_response(
            string_format("encoding_format '%s' packs 8 dimensions per byte, so the embedding dimension must be a multiple of 8, got %d (set \"dimensions\" to a multiple of 8)",
                quant_name.c_str(), quant_dim),
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    // LENGTH-ORDERED ADMISSION (row S69-infinity-PICK-LENGTH-BUCKET-BATCH-2026-09-28-01; the idea, from infinity's queue that pops near-uniform batches
    // after sorting by length, is first-fit-DECREASING bin packing). An embedding sequence must sit whole inside one micro-batch, and the slot loop packs
    // slots in id order into batches of n_ubatch tokens, so a short sequence queued between two long ones can leave the first batch part empty and push
    // the next long sequence to a second decode. Posting the longest first lets the short ones fill the gaps. Results come back indexed by POSTED order
    // and are put back in REQUEST order below; every sequence is pooled on its own, so the vectors are independent of the order.
    std::vector<size_t> order(tokenized_prompts.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = i;
    }
    if (params.embd_length_order) {
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return tokenized_prompts[a].size() > tokenized_prompts[b].size(); });
    }
    {
        std::vector<server_task> tasks;
        for (size_t k = 0; k < order.size(); k++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[order[k]]);

            // OAI-compat
            task.params.res_type = res_type;
            // a chunk vector is L2-normalised on its own (gpt4all does the same); the request's normalisation is applied to the mean below
            task.params.embd_normalize = n_chunks[owner[order[k]]] > 1 ? 2 : embd_normalize;

            // per-request adapters (parsed above)
            task.params.lora               = req_lora;
            task.params.lora_adapter       = req_lora_adapter;
            task.params.lora_adapter_scale = req_lora_adapter_scale;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    std::string quant_err;
    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        // back to request order: the k-th posted task answers the order[k]-th input
        std::vector<server_task_result_ptr> in_request_order(all_results.results.size());
        for (size_t k = 0; k < all_results.results.size(); k++) {
            in_request_order[order[k]] = std::move(all_results.results[k]);
        }
        // long_text_mode: fold the chunk results of one input into one result (mean of the chunk vectors, then the request's normalisation)
        if (in_request_order.size() != n_inputs) {
            std::vector<server_task_result_ptr> merged(n_inputs);
            std::vector<std::vector<std::vector<float>>> parts(n_inputs);
            std::vector<int32_t> n_tok(n_inputs, 0);
            for (size_t j = 0; j < in_request_order.size(); j++) {
                auto * re = dynamic_cast<server_task_result_embd*>(in_request_order[j].get());
                GGML_ASSERT(re != nullptr && !re->embedding.empty());
                const size_t o = owner[j];
                n_tok[o] += re->n_tokens;
                if (n_chunks[o] > 1) {
                    parts[o].push_back(re->embedding[0]);
                }
                if (!merged[o]) {
                    merged[o] = std::move(in_request_order[j]);
                }
            }
            for (size_t o = 0; o < n_inputs; o++) {
                auto * re = dynamic_cast<server_task_result_embd*>(merged[o].get());
                GGML_ASSERT(re != nullptr);
                re->n_tokens = n_tok[o];
                if (n_chunks[o] > 1) {
                    std::vector<float> mean = server_embd_mean_pool(parts[o]);
                    if (mean.empty()) {
                        res->error(format_error_response("long_text_mode: chunk vectors differ in length", ERROR_TYPE_SERVER));
                        return res;
                    }
                    common_embd_normalize(mean.data(), mean.data(), (int) mean.size(), embd_normalize);
                    re->embedding.assign(1, std::move(mean));
                }
            }
            in_request_order = std::move(merged);
        }
        for (size_t qi = 0; qi < in_request_order.size(); qi++) {
            auto & res = in_request_order[qi];
            auto * res_embd = dynamic_cast<server_task_result_embd*>(res.get());
            GGML_ASSERT(res_embd != nullptr);
            res_embd->index = qi; // the non-OAI shape prints this field: it must name the request position, not the posted one
            if (dimensions > 0) {
                // truncate to the first N dims, then re-normalize IFF the worker normalized in the
                // first place (send_embedding only calls common_embd_normalize when pooling_type !=
                // NONE) -- mirrors that exact condition so an embd_normalize=-1 (no-op) request
                // stays a no-op after truncation too, since common_embd_normalize itself treats -1
                // as "copy unchanged".
                for (auto & vec : res_embd->embedding) {
                    if ((int32_t) vec.size() > dimensions) {
                        vec.resize(dimensions);
                        if (meta->pooling_type != LLAMA_POOLING_TYPE_NONE) {
                            common_embd_normalize(vec.data(), vec.data(), dimensions, embd_normalize);
                        }
                    }
                }
            }
            json one = res->to_json();
            if (quant != SERVER_EMBD_QUANT_NONE && !res_embd->embedding.empty() && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
                // one packed row per token, each cut to the same head as the float rows above (non-OAI shape: a matrix)
                const std::vector<std::vector<int>> qr = server_embd_quantize_rows(res_embd->embedding, (size_t) quant_dim, quant);
                if (qr.empty()) {
                    quant_err = string_format("cannot quantise: a token row has fewer than %d values", (int) quant_dim);
                    break;
                }
                one["embedding"] = qr;
            } else if (quant != SERVER_EMBD_QUANT_NONE && !res_embd->embedding.empty()) {
                // pooled + normalised + (maybe) Matryoshka-cut in float above; quantise last
                const std::vector<int> q = server_embd_quantize(res_embd->embedding[0], (size_t) quant_dim, quant);
                if (q.empty()) {
                    // cannot happen while the worker returns n_embd_out values; reported rather than asserted so one bad vector fails one request
                    quant_err = string_format("cannot quantise: embedding has %d values, need %d", (int) res_embd->embedding[0].size(), (int) quant_dim);
                    break;
                }
                // OAI shape carries the vector itself, the non-OAI shape a one-row matrix
                one["embedding"] = res_type == TASK_RESPONSE_TYPE_OAI_EMBD ? json(q) : json::array({q});
            }
            responses.push_back(std::move(one));
        }
        if (!quant_err.empty()) {
            res->error(format_error_response(quant_err, ERROR_TYPE_SERVER));
            return res;
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64, quant_name)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_rerank_impl(const server_http_req & req) {
    auto res = create_response();
    if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const json body = json::parse(req.body);

    // per-request adapters: same fields and handling as /v1/embeddings above
    // (row S69-ENGREV-LORAX-EMBED-RERANK-PER-REQUEST-LORA-2026-10-27-01)
    std::map<int, float> req_lora;
    std::string          req_lora_adapter;
    float                req_lora_adapter_scale = 1.0f;
    try {
        parse_request_lora_fields(body, params.lora_adapters.size(), req_lora, req_lora_adapter, req_lora_adapter_scale);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    bool is_tei_format = body.contains("texts");

    // expand one rerank input item: plain string or {"text": str|[str], "image": url|data-uri|raw-base64|[urls]}
    auto parse_item = [&](const json & item, std::string & out_text, std::vector<raw_buffer> & out_files) {
        if (item.is_string()) {
            out_text = item.get<std::string>();
            return;
        }
        if (!item.is_object()) {
            throw std::invalid_argument("rerank input must be a string or an object of the form {\"text\": ..., \"image\": ...}");
        }
        if (item.contains("video") || item.contains("fps") || item.contains("max_frames")) {
            throw std::invalid_argument("video/fps/max_frames are not supported for rerank input");
        }
        json text = item.value("text", json());
        if (text.is_string()) {
            out_text += text.get<std::string>();
        } else if (text.is_array()) {
            for (const auto & t : text) {
                if (!t.is_string()) {
                    throw std::invalid_argument("\"text\" array elements must be strings");
                }
                out_text += t.get<std::string>();
            }
        } else if (!text.is_null()) {
            throw std::invalid_argument("\"text\" must be a string or an array of strings");
        }
        json image = item.value("image", json());
        auto add_image = [&](const std::string & url) {
            // marker is appended by format_prompt_rerank, one per decoded file
            try {
                handle_media(out_files, url, params.media_path);
            } catch (const std::runtime_error & e) {
                throw std::invalid_argument(e.what());
            }
        };
        if (image.is_string()) {
            add_image(image.get<std::string>());
        } else if (image.is_array()) {
            for (const auto & u : image) {
                if (!u.is_string()) {
                    throw std::invalid_argument("\"image\" array elements must be strings");
                }
                add_image(u.get<std::string>());
            }
        } else if (!image.is_null()) {
            throw std::invalid_argument("\"image\" must be a string or an array of strings");
        }
    };

    // Parse query
    std::string query_str;
    std::vector<raw_buffer> query_files;
    if (body.count("query") == 1) {
        try {
            parse_item(body.at("query"), query_str, query_files);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Resolve Jina/TEI format ambiguity
    json documents_json = json_value(body, "documents",
        json_value(body, "texts", json::array()));
    if (!documents_json.is_array() || documents_json.empty()) {
        res->error(format_error_response("\"documents\" must be a non-empty array of strings or objects", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Parse documents
    std::vector<std::string>          documents(documents_json.size());
    std::vector<std::vector<raw_buffer>> document_files(documents_json.size());
    for (size_t i = 0; i < documents_json.size(); i++) {
        try {
            parse_item(documents_json[i], documents[i], document_files[i]);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // top_n <= 0 means "all documents" (Infinity batch_handler.py:218; format_response_rerank applies it). The
    // upstream sync of 2026-11-04 brought a rejection of negative values here (#29625), which turned the fork's
    // documented top_n=-1 into HTTP 400 -- unit/test_rerank.py::test_rerank_nonpositive_top_n_returns_all[-1].
    int top_n = json_value(body, "top_n", (int)documents.size());

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        tasks.reserve(documents.size());
        for (size_t i = 0; i < documents.size(); i++) {
            auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query_str, documents[i], query_files, document_files[i], ctx_server.init_opt);
            server_task task = server_task(SERVER_TASK_TYPE_RERANK);
            task.id     = rd.get_new_id();
            task.tokens = std::move(tmp);
            task.params.lora               = req_lora;
            task.params.lora_adapter       = req_lora_adapter;
            task.params.lora_adapter_scale = req_lora_adapter_scale;
            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = format_response_rerank(
        body,
        meta->model_name,
        responses,
        is_tei_format,
        documents,
        top_n);

    res->ok(root);
    return res;
}

// POST /classify and /v1/classify: the whole n_cls_out label distribution of a sequence-classification head, where /rerank
// keeps only output 0. Same gate as /rerank (--reranking, pooling RANK) and the same per-request adapter fields.
// Row S69-ENGREV-LORAX-CLASSIFY-ROUTE-MULTILABEL-2026-10-27-01; the maths is in server-classify.h.
std::unique_ptr<server_res_generator> server_routes::handle_classify_impl(const server_http_req & req) {
    auto res = create_response();
    if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        res->error(format_error_response("This server does not support classification. Start it with `--reranking` and a sequence-classifier model", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const json body = json::parse(req.body);

    std::map<int, float> req_lora;
    std::string          req_lora_adapter;
    float                req_lora_adapter_scale = 1.0f;
    try {
        parse_request_lora_fields(body, params.lora_adapters.size(), req_lora, req_lora_adapter, req_lora_adapter_scale);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // "input" (OpenAI/Infinity shape) or "inputs" (LoRAX/TEI shape): a string or a list of strings
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.count("inputs") != 0) {
        prompt = body.at("inputs");
    } else {
        res->error(format_error_response("\"input\" or \"inputs\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // multi-label head: independent sigmoid per class instead of a softmax over competing classes
    bool multilabel = false;
    for (const char * key : { "multi_label", "multilabel" }) {
        if (body.count(key) != 0) {
            if (!body.at(key).is_boolean()) {
                res->error(format_error_response(std::string("\"") + key + "\" must be a boolean", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            multilabel = body.at(key).get<bool>();
        }
    }

    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt));
        }
    } else {
        tokenized_prompts.push_back(tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    for (const auto & tokens : tokenized_prompts) {
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_RERANK);
            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);
            task.params.lora               = req_lora;
            task.params.lora_adapter       = req_lora_adapter;
            task.params.lora_adapter_scale = req_lora_adapter_scale;
            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    auto all_results = rd.wait_for_all(req.should_stop);
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    }

    // the GGUF class labels, in output order (empty strings when the file carries none: the index is used then)
    const int32_t n_cls = std::max<int32_t>(1, llama_model_n_cls_out(ctx_server.model_tgt));
    std::vector<std::string> labels;
    for (int32_t i = 0; i < n_cls; i++) {
        const char * l = llama_model_cls_label(ctx_server.model_tgt, (uint32_t) i);
        labels.push_back(l ? l : "");
    }

    json data = json::array();
    int  prompt_tokens = 0;
    for (size_t i = 0; i < all_results.results.size(); i++) {
        auto * rr = dynamic_cast<server_task_result_rerank *>(all_results.results[i].get());
        GGML_ASSERT(rr != nullptr);
        prompt_tokens += rr->n_tokens;
        const server_classify_result c = server_classify_make(rr->scores, labels, multilabel);
        json scores = json::array();
        for (const auto & s : c.scores) {
            scores.push_back({ {"label", s.label}, {"probability", s.probability} });
        }
        data.push_back({ {"index", (int) i}, {"label", c.label}, {"score", c.score}, {"scores", scores} });
    }

    res->ok(json {
        {"object", "list"},
        {"model",  meta->model_name},
        {"data",   data},
        {"usage",  { {"prompt_tokens", prompt_tokens}, {"total_tokens", prompt_tokens} }},
    });
    return res;
}

std::unique_ptr<server_res_generator> server_routes::run_compaction(const server_http_req & req, respcompact::plan & plan, json & item, json & usage) {
    // the internal call is always non-stream; a copy without the client's headers keeps the keepalive chunking (header-driven) off
    server_http_req ireq = req;
    ireq.headers.clear();
    std::string repair;
    for (int round = 0; round < 12; round++) {
        json sreq = respcompact::summary_request(plan, repair);
        json body = server_chat_convert_responses_to_chatcmpl(sreq);
        std::vector<raw_buffer> files;
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(ctx_server.vocab, body, meta->chat_params, files, session);
        auto r = handle_completions_impl(ireq, SERVER_TASK_TYPE_COMPLETION, body_parsed, files, TASK_RESPONSE_TYPE_OAI_RESP, session);
        if (r->is_stream()) {
            return r;
        }
        json j;
        try {
            j = json::parse(r->data);
        } catch (const std::exception &) {
            j = json();
        }
        if (r->status != 200) {
            const bool overflow = j.is_object() && j.contains("error") && j.at("error").is_object() &&
                                  json_value(j.at("error"), "type", std::string()) == "exceed_context_size_error";
            if (overflow && respcompact::trim_for_context(plan) > 0) {
                continue; // the transcript did not fit: retry with the oldest removable items trimmed
            }
            return r;
        }
        std::string err;
        if (respcompact::complete(plan, j, item, usage, err)) {
            return nullptr;
        }
        if (!repair.empty()) {
            auto failed = create_response();
            failed->error(format_error_response("compaction failed: " + err, ERROR_TYPE_SERVER));
            return failed;
        }
        repair = err; // one retry with the validation error in the prompt
    }
    auto failed = create_response();
    failed->error(format_error_response("compaction failed: the transcript does not fit the context", ERROR_TYPE_EXCEED_CONTEXT_SIZE));
    return failed;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                {
                    bool changed = false;
                    std::string cerr;
                    if (!respcompact::expand_body(body, changed, cerr)) {
                        res->error(format_error_response(cerr, ERROR_TYPE_INVALID_REQUEST));
                        return res;
                    }
                }
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        // same segmentation as handle_completions_impl, so the count matches what a completion would tokenise to
        plain_spans_t plain_spans;
        if (prompt.is_string() && body_parsed.contains("plain_text_spans")) {
            plain_spans = plain_spans_from_json(body_parsed.at("plain_text_spans"), prompt.get<std::string>().size());
        }
        if (!plain_spans.empty()) {
            n_tokens = tokenize_with_plain_spans(ctx_server.vocab, prompt.get<std::string>(), plain_spans, true).size();
        } else {
            n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
        }
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();
        cached_gauges  = ctx_server.get_slot_gauges();
        cached_memory  = ctx_server.get_memory_data();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
