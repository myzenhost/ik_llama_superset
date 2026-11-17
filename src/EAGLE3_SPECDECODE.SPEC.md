# EAGLE3 speculative decoding — taxonomy, methodology, ik_llama wiring + per-cycle build plan
# Task S69 #33 (EAGLE3 port step 5). Cron-driven, one compilable+verified piece per firing.

## TAXONOMY (where it sits)
Family: speculative decoding (draft-then-verify lossless acceleration).
Sub-family: SEPARATE-draft-model, HIDDEN-STATE-CONDITIONED.
- vs MTP (already wired, common/speculative.cpp:1218 mtp_speculative_gen_draft): MTP is SELF-draft — one
  model, an extra MTP head, fed its own last hidden (llama_get_embeddings_ith). EAGLE3 uses a SEPARATE small
  draft model fed the TARGET's intermediate hidden states.
- vs ngram/lookup draft: those use no model; EAGLE3 uses a trained 1-layer draft head.
ik_llama already has: LLM_ARCH_EAGLE3 (arch+tensors+hparams, #30 done), build_eagle3() graph (#32d done,
llama-build-context.cpp:4336, reads input "eagle3_target_feats" = [3*n_embd_tgt, n_tokens]), the draft-input
setter llama_set_draft_input_hidden_state (#31, llama.cpp:9252). REMAINING = the loop that fills target_feats
and runs the draft (#33) + the -md flag.

## METHODOLOGY (EAGLE3 algorithm, arxiv 2503.01840 EAGLE-3, VERIFY-FROM-REF before each edit)
Per generation step with target T (big) and draft D (EAGLE3 head):
1. T forward on the last accepted token -> logits + capture 3 hidden states h_low (early layer ~2),
   h_mid (~middle), h_high (last pre-norm). Concatenate -> feats = [3*n_embd, 1].
2. Set feats as D's input (llama_set_draft_input_hidden_state) -> D runs build_eagle3 -> proposes draft
   token d1; feed d1 + its own hidden recurrently to propose d2..dK (K = n_draft, tree or linear chain).
3. T verifies the K drafts in ONE forward (batch d1..dK) -> accept the longest prefix whose argmax matches;
   the first mismatch's T-sampled token is the correction. n_accepted in [1, K+1].
4. d2t map (draft-vocab -> target-vocab) applied if vocabularies differ (EAGLE3 uses a reduced draft vocab;
   tensor D2T already loaded #30). Lossless: final tokens are always T-verified.
EAGLE3 key vs EAGLE1/2: uses 3 multi-level features (not just last hidden) + training-time test-time-alignment.

## WIRING (4 parts, exact files — each = one cron cycle, build+verify before next)
A. TARGET 3-LAYER CAPTURE (keystone graph edit — opt-in, default OFF so live serving path is identical):
   - llama-context.h: add `std::vector<float> eagle3_feats; bool eagle3_capture=false; int e3_lo,e3_mid,e3_hi;`
   - llama-build-context.cpp main decoder layer loop: after each layer's `cur` (the residual stream hidden),
     when eagle3_capture && il in {e3_lo,e3_mid,e3_hi}, `cb(cur,"eagle3_h_lo/mid/hi",il)`.
   - llama.cpp eval callback (reuse the snapkv_eval_callback pattern, #22): on those 3 names, copy the last
     token's hidden into lctx.eagle3_feats at the right 1/3 offset. F32 path.
   - GUARD: only when eagle3_capture (set by the spec loop on the TARGET ctx). Default path byte-identical.
B. GETTER: llama.h + llama.cpp `void llama_get_eagle3_target_feats(llama_context*, float* out)` -> copies
   the [3*n_embd] concat for the last token. (out sized 3*n_embd by caller.)
C. DRAFT-GEN: common/speculative.cpp `eagle3_speculative_gen_draft(smpl, ctx_draft, n_draft, feats, ...)`
   mirror mtp_speculative_gen_draft: set feats on ctx_draft, decode build_eagle3, sample dK chain. Returns drafts.
D. SERVER + TYPE: common_speculative_type enum += EAGLE3; common_params_speculative + the -md/--draft-model
   already plumbed (verify); server completion loop: if draft type==EAGLE3, run T(capture on)->getter->C->
   verify-batch->accept. spec-tuner stats reuse.

## PER-CYCLE BUILD PLAN (cron: do ONE, build_sm120.bat, verify it compiles + serving still 200, then next)
1. A (capture + eval-cb + context fields) -> build -> serve a normal model, confirm default path unaffected (200).
2. B (getter) -> build.
3. C (eagle3_speculative_gen_draft) -> build.
4. D (type + server loop + -md) -> build.
5. #34: load Qwen3.5-2B target + Qwen3-8B-EAGLE3 draft (-md), measure tok/s vs no-draft on legal prompts;
   verify OUTPUT IDENTICAL to non-spec (lossless check) — rule 1.12 A/B.
SAFETY GATE: after EACH build, run the deployed binary --version + a tiny serve smoke; if it fails to load,
git-revert that cycle's edit before proceeding. The deployed binary must always serve (live legal serving).
