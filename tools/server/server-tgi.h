#pragma once

#include "server-http.h"

#include "json.h"

#include <string>

struct server_routes;

// Text Generation Inference (TGI) native API compatibility routes.
//
// Row S69-TGI-PICK-NATIVE-GENERATE-ROUTES-2026-10-18-01 (programme row S69-ENGINE-MERGE-TGI-2026-09-27-01).
// TGI clients (huggingface_hub InferenceClient.text_generation, the `text-generation` python client,
// LangChain HuggingFaceTextGenInference, Inference Endpoints tooling) speak TGI's own wire protocol:
// POST /generate, POST /generate_stream (SSE), POST / (compat: {"stream": bool}) and GET /info. This
// server answered all four with 404 (the /v1/* OpenAI routes are a feature match, not a protocol match).
//
// SOURCE-READ (rule 1.87; huggingface/text-generation-inference, Apache-2.0, sha b4adbf2f6e2e7212):
//   router/src/server.rs:126-151   compat_generate: "/" switches on `stream`; the non-stream answer is a
//                                  ONE-ELEMENT ARRAY; return_full_text defaults to true for a
//                                  text-generation (or unknown) pipeline tag (:1800-1806).
//   router/src/server.rs:289-438   generate_internal: details only when details || decoder_input_details;
//                                  return_full_text prepends the prompt; GenerateResponse {generated_text, details}.
//   router/src/server.rs:508-620   generate_stream_internal: one SSE event per token, index 1-based, the LAST
//                                  token's event carries generated_text (+ details when asked); best_of != 1
//                                  and decoder_input_details are refused when streaming.
//   router/src/server.rs:2404-2440 error mapping: validation 422, overloaded 429, generation 424; body
//                                  {"error", "error_type"}; a stream error is one SSE event with that body.
//   router/src/lib.rs:261-460      Info and GenerateParameters field names and defaults.
//   router/src/lib.rs:1337-1520    GenerateRequest, Token {id,text,logprob,special}, Details, StreamResponse,
//                                  FinishReason (length | eos_token | stop_sequence).
//   router/src/validation.rs:229-330 + :963-1029  limits and the exact messages (temperature > 0,
//                                  repetition_penalty > 0, 0 < top_p < 1, 0 < typical_p < 1, top_k > 0,
//                                  max_new_tokens > 0, at most max_stop_sequences (default 4) stops,
//                                  top_n_tokens <= max_top_n_tokens (default 5), inputs non-empty).
//   server/text_generation_server/utils/logits_process.py:151-180 frequency penalty.
//   backends/v3/src/backend.rs:460-474 a Prefill event exists only when prefill logprobs were asked for.
// SOURCE-NUANCE: TGI's "/" route answers a one-element ARRAY when not streaming while /generate answers a
// bare object; "parameters" absent means do_sample TRUE (lib.rs default_parameters) but "parameters": {}
// means do_sample FALSE (serde default), i.e. greedy; frequency_penalty is length-normalised
// (count / sequence length * p, logits_process.py:171-177), not count * p as here, so a non-zero value is
// refused with that reason rather than passed through with a silently different meaning.
// This is a fresh C++ translation layer; no Rust or Python code copied.
//
// DESIGN: translate at the edge, the same shape as this fork's own tools/server/server-ollama.cpp:582-720
// (handle_chat_or_generate, read this session). The TGI request is validated and mapped onto a
// /completion request, served by the existing routes.post_completions (same task queue, sampling and
// streaming), and the answer is mapped back. Sampling is pinned to TGI's semantics: TGI applies no
// min_p/top_k/top_p unless asked, so the translation sets top_k 0, top_p 1, min_p 0, typical_p 1,
// repeat_penalty 1 explicitly instead of inheriting this server's defaults; and TGI is greedy unless
// do_sample or a sampling knob is set.
//
// decoder_input_details (rows S69-TGI-PICK-DECODER-INPUT-DETAILS-VIA-PROMPT-LOGPROBS-2026-10-19-01): the prompt
// is tokenized here (/tokenize with pieces) and sent as ids, the /completion request carries prompt_logprobs,
// and details.prefill is [{id, text, logprob}] per prompt token with the first logprob null (lib.rs:1376-1383
// PrefillToken; server.rs:310 details are returned when details || decoder_input_details).
// best_of > 1 (row S69-TGI-PICK-BEST-OF-VIA-N-CMPL-RANK-2026-10-19-01): one /completion with n_cmpl = best_of
// (children get seed + 1, + 2, ... per server-task.h add_child). SOURCE-READ infer/mod.rs:318-355: the answer is
// the sequence with the highest MEAN generated-token logprob; the others go in details.best_of_sequences
// (lib.rs:1442-1455; server.rs:322-347). validation.rs:229-238 sampling is required; :298-302 a client seed is
// refused; :451-460 best_of is capped by max_best_of, here the slot count (n_cmpl children each need a slot).
// prompt_logprobs refuses n_cmpl > 1, so best_of with decoder_input_details scores the prompt in a second
// one-token request; every sequence shares that prefill, as in TGI where all best_of requests share the input.
// adapter_id: a loaded --lora adapter (named by its file name without extension, or its path) becomes lora [{id, scale 1}]; otherwise the
// name is a --lora-pool-dir adapter (lora_adapter). An unknown name is a 422, never a silent base-model answer.
// NOT MAPPED, refused with 422 and a reason (never silently dropped): non-zero frequency_penalty.
// NUANCES (documented, measurable): Token.logprob is the logprob this server reports for n_probs
// (pre-sampling distribution); Token.special is true only for the final token of an eos-terminated
// generation (the translation layer has no vocabulary handle; the end-of-generation token is the special
// token a generation actually emits).

struct server_tgi_request {
    common_json completion; // /completion body
    std::string inputs;
    bool return_full_text = false;
    bool details          = false;
    bool sampling         = false;
    int  top_n_tokens     = 0;
    bool stream           = false;
    int  truncate         = 0;     // > 0: keep only the last `truncate` prompt tokens
    bool decoder_input_details = false; // details.prefill from prompt_logprobs
    int  best_of          = 1;     // > 1: n_cmpl fan-out, best by mean token logprob
    std::string adapter_id; // parameters.adapter_id, resolved to the per-request LoRA fields by handle() ("" = base model)
};

// Validates a TGI GenerateRequest body and fills `out`. Returns "" on success, else the TGI validation
// message (without the "Input validation error: " prefix). `compat_route` selects "/" semantics
// (stream flag honoured, return_full_text defaults to true). `n_ctx` is the slot context (TGI's
// repetition penalty covers the whole sequence). `max_best_of` is the largest best_of accepted (TGI's
// --max-best-of; here the slot count).
std::string server_tgi_translate(const common_json & body, bool compat_route, bool stream_route, int n_ctx,
                                 int max_best_of, server_tgi_request & out);

// Maps a /completion stop_type to TGI's FinishReason string.
std::string server_tgi_finish_reason(const std::string & stop_type);

struct server_routes_tgi {
    explicit server_routes_tgi(server_routes & routes);

    server_http_context::handler_t post_generate;        // POST /generate
    server_http_context::handler_t post_generate_stream; // POST /generate_stream
    server_http_context::handler_t post_compat;          // POST /  ({"stream": bool})
    server_http_context::handler_t get_info;             // GET  /info

private:
    server_routes & routes;

    server_http_res_ptr handle(const server_http_req & req, bool compat_route, bool stream_route) const;
    server_http_res_ptr handle_info(const server_http_req & req) const;
    std::string resolve_adapter(const server_http_req & req, const common_json & props, const std::string & adapter_id,
                                common_json & completion) const;
    common_json props(const server_http_req & req) const;
};
