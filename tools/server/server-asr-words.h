// server-asr-words.h -- word timestamps for /v1/audio/transcriptions (timestamp_granularities[]=word).
// PRIOR-ART-NONE: git log --all -S and git grep over canonical + every branch found no server word-timestamp path (see temp/qwork_qworker-fork29/dup4.out); the aligner itself (examples/audio/asr-align.h) is reused, not rewritten.
//
// The times come from the fork's own aligner, the one `llama-audio --word-timestamps` uses (examples/audio/asr-align.h:
// attention rows over the audio KV cells -> head selection -> median filter -> DTW). A llama-server slot cannot hand those
// rows over (it runs flash attention, and its decode loop is the slot thread's), so the finished request is REPLAYED on a
// second llama_context of the same model, built without flash attention and with an eval callback on kq_soft_max:
//   1. prompt + audio, exactly as tokenized for the request (row 0 = the last query row of that prefill),
//   2. the transcript tokens teacher-forced in batches, logits only on the last (rows 1..n; row n is the end row).
// See examples/audio/asr-align-capture.h for why rows 1..n of the replay equal the rows of the generation.
// The second context is created lazily on the first word request, serialised by a mutex, and the request gets a clear error
// when it cannot be built (no audio projector, out of memory): a request that asked for words never gets invented times.
//
// Limits, stated: the transcript is re-tokenized from the finished text, so a model that writes a language tag it then strips
// (Qwen3-ASR "language English<asr_text>") is replayed without it; per-request LoRA overrides are not mirrored on the second
// context (the server-wide --lora list is).
#pragma once

#include "server-asr-format.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct mtmd_context;
struct common_params;

namespace asrwords {

// timestamp_granularities[] values (OpenAI: "word", "segment"). `bad` carries the first unknown value.
struct granularities {
    bool        word    = false;
    bool        segment = false;
    std::string bad;
};

// Pure. An empty list is not an error (no granularity asked).
inline granularities parse_granularities(const std::vector<std::string> & values) {
    granularities g;
    for (const auto & v : values) {
        if (v == "word") {
            g.word = true;
        } else if (v == "segment") {
            g.segment = true;
        } else if (g.bad.empty()) {
            g.bad = v;
        }
    }
    return g;
}

struct timing {
    double tokenize_ms = 0.0;
    double prefill_ms  = 0.0;   // prompt + audio decode (includes the audio encoder)
    double forced_ms   = 0.0;   // teacher-forced transcript pass
    double align_ms    = 0.0;   // median filter + DTW
    double create_ms   = 0.0;   // lazy creation of the context + projector (first request only)
    int    n_audio_pos = 0;
    int    n_tokens    = 0;
    int    n_heads     = 0;
};

class engine {
public:
    // `params`: the server's own parameters (copied); the second context is built from them with flash attention off.
    engine(llama_model * model, const common_params & params);
    ~engine();
    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    // Builds the projector side (own mtmd_context: the shared one belongs to the slot thread) and measures the seconds per
    // audio position, or says why it cannot. Cheap after the first call. Thread-safe.
    bool ready(std::string & err);

    struct word { std::string text; double t0 = 0.0, t1 = 0.0; };

    // Words of `transcript` for the request `prompt` (rendered prompt with media markers) + `files` (uploads, marker order).
    // `has_eog`: generation ended at an end-of-generation token. Thread-safe; one replay at a time.
    bool align(const std::string & prompt, const std::vector<std::vector<uint8_t>> & files, const std::string & transcript,
               bool has_eog, std::vector<word> & out, timing * tm, std::string & err);

    // Bytes the second context holds on its devices (0 before the first request), for the row's memory number.
    size_t context_bytes() const { return ctx_bytes_; }

private:
    struct impl;
    std::unique_ptr<impl> p_;
    std::mutex            mu_;
    size_t                ctx_bytes_ = 0;
};

} // namespace asrwords
