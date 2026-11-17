// asr-align-capture.h -- attention-row capture for word timestamps, shared by anything that wants asr-align.h input
// from a running llama_context (the llama-server /v1/audio/transcriptions word path; llama-audio keeps its own
// copy of the same logic in audio.cpp, dtw_eval_cb, until it is moved onto this module).
//
// WHY A SECOND SHAPE OF CAPTURE. llama-audio reads the row of the query that produced each token while it decodes,
// one token per llama_decode, and snapshots the row after sampling. A server slot cannot do that: it runs flash
// attention, which never materialises kq_soft_max, and its decode loop belongs to the slot thread. The server instead
// replays the finished request on a second context: prompt + audio first (row 0 = the last query row of that
// prefill), then the transcript tokens in ONE batch with logits off. A batch of n tokens has n query rows at the
// positions P..P+n-1, and the row of position P+i is the one whose logits produced token i+1 -- the same row the
// token-by-token loop sees -- so rows 1..n of the teacher-forced pass equal rows 1..n of generation (row n is the
// end-of-generation row). Numerically the batch and the per-token matmuls differ by float noise only.
//
// The callback is the fork's three-valued ggml eval callback (0 = skip, 1 = wanted, synchronize first).
// Written independently; no code copied.
#pragma once

#include "asr-align.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_tensor;

struct asr_attn_capture {
    // ---- configuration -------------------------------------------------------------------------------------------
    int l0 = 0, l1 = -1;        // selected layers, inclusive
    int n_head = 0;             // heads per layer (the model's n_head)

    // ---- state ---------------------------------------------------------------------------------------------------
    bool    armed = false;
    bool    all_rows = false;   // true: append every query row of every batch (teacher-forced pass); false: keep the last row
    int64_t kv0 = 0, kv1 = 0;   // audio KV cells
    std::vector<float>   cur;   // [n_sel][n_head][n_a], last query row of the latest eval
    std::vector<uint8_t> got;   // per selected layer: seen in the latest eval
    std::vector<std::vector<float>> all;  // per selected layer, all_rows mode: [row][n_head][n_a] in arrival order
    std::string err;            // first capture problem

    int  n_sel() const { return l1 >= l0 ? l1 - l0 + 1 : 0; }
    int  n_audio() const { return (int) (kv1 - kv0); }

    // Arms the capture for audio cells [kv0, kv1). false when the range or the configuration is empty.
    bool arm(int64_t a0, int64_t a1);
    // Row 0: after the prompt prefill, every selected layer must have been seen. Copies `cur` out (n_sel*n_head*n_a)
    // and switches to all_rows mode for the teacher-forced pass. false + err when a layer was never captured.
    bool take_row0_and_begin_all_rows(std::vector<float> & row0);
    // After the teacher-forced pass: rows [n_tok][n_sel*n_head][n_a] in `tok_rows`. false + err when a layer did not
    // deliver exactly n_tok rows.
    bool take_all_rows(int n_tok, std::vector<float> & tok_rows);

    // The testable core of the callback (no ggml needed): one kq_soft_max tensor of layer `il`, shape ne[4] / strides
    // nb[4] in BYTES, element type F32 or not. `read(dst, byte_offset, n_bytes)` copies tensor bytes.
    // Returns true ("continue the graph") always, like the callback.
    using reader_fn = std::function<void(void * dst, size_t off, size_t n)>;
    bool on_softmax(int il, bool is_f32, const char * type_name, const int64_t ne[4], const size_t nb[4], const reader_fn & read);
};

// The ggml_backend_sched eval callback: user_data = asr_attn_capture *. Ignores every tensor but kq_soft_max*.
int asr_attn_capture_cb(struct ggml_tensor * t, bool ask, void * user_data);

// Layer index from a tensor name: the integer after the last '-' ("kq_soft_max-12" -> 12), -1 when there is none.
int asr_attn_layer_of(const char * name);

// Rows of the whole replay in the layout asr-align.h wants: row 0 + the n_tok teacher-forced rows, `n_rows` of them
// (n_tok + 1 when has_eog_row, else n_tok), each [n_sel*n_head][n_frames] with the padding positions cut.
// row0: n_sel*n_head*n_pos; tok_rows: n_tok*n_sel*n_head*n_pos. Pure.
std::vector<float> asr_attn_assemble(const std::vector<float> & row0, const std::vector<float> & tok_rows,
                                     int n_tok, bool has_eog_row, int n_hd, int n_pos, int n_frames);
