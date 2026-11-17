#pragma once

#include "server-http.h"

#include <memory>
#include <string>

// Forward declaration; full type lives in server-context.h. Deliberately not included here so
// this header stays cheap for anything that only needs to construct/register the shim (mirrors
// server-models.h's own forward-declare-only style).
struct server_routes;
struct server_ollama_state; // parsed Modelfile + strict-name switch (server-ollama.cpp)

// Startup configuration of the shim (both default OFF; see common_params::ollama_modelfile / ollama_strict_names).
struct server_ollama_config {
    std::string modelfile_path;   // --modelfile: PARAMETER/SYSTEM/MESSAGE defaults for /api/chat + /api/generate, shown by /api/show
    bool        strict_names = false; // --ollama-strict-model-names: unknown model names answer 404 like ollama
    std::string store_dir;            // --ollama-models-dir / --ollama-model: Ollama model store (empty = none); /api/tags also lists it
    std::string store_model;          // --ollama-model NAME: its system/params/messages/license/template layers seed the defaults (a Modelfile wins)
    bool        store_write = false;  // --ollama-store-write: allow /api/copy and /api/delete to modify store_dir
    // this process is the model router: options.num_ctx travels to the router as body field "ollama_num_ctx" and the router reloads
    // the child at that context (server-models.cpp apply_num_ctx). false = single-model server: num_ctx cannot be applied, one WARN says so.
    bool        router_mode = false;
};

// Ollama-native API translation shim.
//
// Row S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01: this fork's llama-server has no
// ollama-native routes (/api/chat, /api/generate, /api/embed(+/embeddings), /api/tags, /api/show,
// /api/ps, /api/version) -- verified absent by a prior audit
// (temp/research/findings/S69_SERVING_SURFACE_AUDIT_LOCALAI_OLLAMA_2026-09-27.md), which found the
// fork's existing /v1/* OAI-compatible routes are a FEATURE match for ollama clients, not a
// PROTOCOL match: an ollama-only client sends ollama-shaped JSON (options{}, format, keep_alive,
// images[] as base64 on messages) and expects NDJSON streaming with a final
// {done:true, total_duration, eval_count, prompt_eval_count, ...} object, and gets a 404 today.
//
// SOURCE-READ (rule 1.87; MIT license, adapted with attribution -- this is a fresh C++ design
// following the same shape, no Go code copied): mudler/LocalAI (github.com/mudler/LocalAI,
// LICENSE: MIT, "Copyright (c) 2023-2025 Ettore Di Giacinto"), master branch, files read in full
// 2026-09-27: core/schema/ollama.go (request/response struct + JSON field names, incl.
// OllamaOptions/OllamaChatRequest/OllamaChatResponse/OllamaGenerateRequest/OllamaGenerateResponse/
// OllamaEmbedRequest/OllamaEmbedResponse/OllamaShowResponse/OllamaModelDetails/OllamaModelEntry/
// OllamaPsEntry), core/http/routes/ollama.go (route table), core/http/endpoints/ollama/
// {chat,generate,embed,models,helpers}.go (translation logic + NDJSON streaming pattern + the
// num_ctx-ceiling security note from issue mudler/LocalAI#11022). Also ollama/ollama
// (github.com/ollama/ollama, MIT) docs/api.md, read in full, for the wire field semantics.
//
// DESIGN DIFFERENCE FROM THE REFERENCE (why this is not a line-for-line port): LocalAI's Go
// handlers call its own inference pipeline directly (backend.ComputeChoices) and hand-build ollama
// response structs. This shim instead calls THIS FORK'S EXISTING PUBLIC server_routes handlers
// (post_chat_completions, post_embeddings_oai, get_model_info) with a translated OAI-shaped request
// body, and translates the OAI-shaped response back into ollama's wire shape -- so the actual
// inference pipeline (task queue, sampling, streaming) is reused unmodified, never forked, per the
// row's spec. Streaming works by wrapping the inner server_http_res's SSE `next()` output: each
// "data: {...}\n\n" SSE event is re-parsed as JSON and re-emitted as one ollama NDJSON line.
//
// SCOPE (documented, not silently missing): registry CRUD (/api/create|pull|push|copy|delete|blobs)
// is deliberately OUT of scope -- see the row body for why (the harness already gets full ollama
// model-lifecycle semantics from a real ollama binary via agent_router.py, and LocalAI itself does
// not implement this subset either). Also out of scope, documented per-field below in the .cpp:
// ollama's token-array `context` continuation field (legacy raw-token continuation; superseded by
// message-based chat for every modern client), tool_calls passthrough on /api/chat, and
// sub-phase timing breakdown (load_duration/prompt_eval_duration/eval_duration are not separately
// instrumented by the underlying handlers today, so only total_duration + the two token counts are
// populated -- LocalAI's own port has this same gap, see chat.go/generate.go: only TotalDuration,
// PromptEvalCount and EvalCount are ever set there either).
struct server_routes_ollama {
    explicit server_routes_ollama(server_routes & routes);

    // Throws std::runtime_error with the parser message when the Modelfile is malformed.
    void configure(const server_ollama_config & cfg);

    server_http_context::handler_t post_api_chat;
    server_http_context::handler_t post_api_generate;
    server_http_context::handler_t post_api_embed;
    // /api/embeddings is NOT an alias of /api/embed: the older endpoint takes one "prompt" and answers
    // {"embedding": [..]} (one flat vector), where /api/embed answers {"embeddings": [[..], ..]}.
    // RESEARCH-DONE: https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md "Generate Embedding"
    // (fetched 2026-09-27, MIT).
    server_http_context::handler_t post_api_embeddings_legacy;
    server_http_context::handler_t get_api_tags;
    server_http_context::handler_t post_api_show;
    server_http_context::handler_t get_api_ps;
    server_http_context::handler_t get_api_version;
    // /api/copy (POST {source,destination}) and /api/delete (DELETE {model}) edit the Ollama store's manifests; both answer 403 unless
    // --ollama-store-write was given, and 404 "model not found" when no store is configured (server/routes.go:2064-2112, 1629-1669).
    server_http_context::handler_t post_api_copy;
    server_http_context::handler_t del_api_delete;

private:
    server_routes & routes;
    std::shared_ptr<server_ollama_state> state_;

    server_http_res_ptr handle_chat_or_generate(const server_http_req & req, bool is_generate) const;
    server_http_res_ptr handle_embed(const server_http_req & req, bool legacy = false) const;
    server_http_res_ptr handle_tags(const server_http_req & req) const;
    server_http_res_ptr handle_show(const server_http_req & req) const;
    server_http_res_ptr handle_ps(const server_http_req & req) const;
    server_http_res_ptr handle_version(const server_http_req & req) const;
    server_http_res_ptr handle_copy(const server_http_req & req) const;
    server_http_res_ptr handle_delete(const server_http_req & req) const;
};
