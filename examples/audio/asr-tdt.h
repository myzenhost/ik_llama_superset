// asr-tdt.h -- Token-and-Duration Transducer (TDT) decoding for an encoder-decoder ASR model (Parakeet-TDT).
//
// The encoder runs in mtmd (tools/mtmd/mtmd-audio-encoder.h, projector type parakeet, encoder-only). This file is
// the decoder half: the prediction network (embedding + stacked LSTM), the joint network, and the greedy TDT loop
// that emits a (token, duration) pair per step and skips `duration` encoder frames. Plain float math on the CPU;
// the decoder is ~18M parameters and runs once per emitted token or skipped span, not once per frame.
//
// SOURCE-READ (rewritten, no code copied; MIT): ggml-org/whisper.cpp @ 6e4ab854f67f743900934a703d5603419384c961
//   src/parakeet.cpp:2497-2636 parakeet_decode (greedy TDT loop), :2461-2495 per-token time and probability,
//   :2409-2459 word-start / punctuation tokens and the punctuation time collapse.
// The decode rules match NVIDIA NeMo's greedy TDT (blank with duration 0 advances one frame; at most
// max_symbols_per_step tokens on one frame, then one frame forward).
#pragma once

#include "asr-halluc.h" // asr_word

#include <string>
#include <vector>

struct asr_tdt_lstm_layer {
    int n_in = 0;
    std::vector<float> w_ih; // [4H x n_in] row-major, torch gate order i, f, g, o
    std::vector<float> w_hh; // [4H x H]
    std::vector<float> b;    // [4H] = bias_ih + bias_hh
};

struct asr_tdt_model {
    int    n_vocab     = 0;    // non-blank tokens; the blank id is n_vocab
    int    n_pred      = 0;    // prediction-network width (LSTM hidden size)
    int    n_joint     = 0;    // joint hidden size
    int    n_enc       = 0;    // encoder state width
    int    max_symbols = 10;   // tokens allowed on one encoder frame
    int    subsampling = 8;    // mel frames per encoder frame
    double frame_s     = 0.01; // seconds per mel frame
    std::vector<int>         durations; // frames to advance, one per duration logit
    std::vector<std::string> vocab;     // n_vocab SentencePiece pieces ("\xE2\x96\x81" starts a word)

    std::vector<float> embd;            // [(n_vocab + 1) x n_pred], the blank row is the zero start input
    std::vector<asr_tdt_lstm_layer> lstm;
    std::vector<float> pred_w, pred_b;  // [n_joint x n_pred]
    std::vector<float> enc_w,  enc_b;   // [n_joint x n_enc]
    std::vector<float> out_w,  out_b;   // [n_out x n_joint]

    int blank() const { return n_vocab; }
    int n_out() const { return n_vocab + 1 + (int) durations.size(); }
    double frame_seconds() const { return frame_s * subsampling; } // seconds per encoder frame
};

// true when the GGUF carries a TDT decoder (key tdt.vocab_size)
bool asr_tdt_is_tdt_file(const std::string & path);

// reads the tdt.* keys and tensors (F32) from the GGUF; the encoder tensors are not touched
bool asr_tdt_load(const std::string & path, asr_tdt_model & m, std::string & err);

// the shapes are consistent with each other (used by asr_tdt_load and by tests that build a model in memory)
bool asr_tdt_check(const asr_tdt_model & m, std::string & err);

struct asr_tdt_token {
    int   id       = 0;
    int   frame    = 0;   // encoder frame it was emitted on
    int   duration = 0;   // frames the duration head chose
    float p        = 0.f; // softmax over the token logits (blank included)
};

// greedy TDT over n_frames encoder states (rows of m.n_enc floats)
bool asr_tdt_greedy(const asr_tdt_model & m, const float * enc, int n_frames, std::vector<asr_tdt_token> & out);

bool asr_tdt_is_punct(const std::string & piece);   // a single . , ! ? ; : ' " - ( ) [ ] { } (word marker allowed)
bool asr_tdt_is_special(const std::string & piece); // <unk>, <|...|> control tokens: never part of the text

// text: pieces joined, the word marker turned into a space, control tokens dropped
std::string asr_tdt_text(const asr_tdt_model & m, const std::vector<asr_tdt_token> & toks);

// words with times in seconds (+ t_offset): a word starts at a piece with the word marker; a token covers
// [frame, frame + duration) encoder frames; punctuation takes the previous token's end as both ends; the word's
// probability is the mean of its tokens'
std::vector<asr_word> asr_tdt_words(const asr_tdt_model & m, const std::vector<asr_tdt_token> & toks, double t_offset = 0.0);
