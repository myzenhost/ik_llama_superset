// asr-dtw.h
//
// DTW token-level timestamps from cross-attention weights, as pure functions on float arrays.
//
// Semantics from ggml-org/whisper.cpp (MIT, src/whisper.cpp @ d1be6fde, dtw_and_backtrace
// ~L9041, median_filter ~L9131, whisper_exp_compute_token_level_timestamps_dtw ~L9166), which
// follows openai/whisper timing.py. Written independently.
//
// Pipeline (asr_dtw_token_frames): per head, normalise each frame's column over the token axis
// (zero mean, unit biased std, eps 1e-9) -> median filter over the frame axis (odd width,
// reflect padding) -> mean over heads -> negate (cost) -> DTW -> first frame of each token row.
// Tie-breaking in the DTW recurrence is that of openai/whisper: diagonal wins only if strictly
// smaller than both others, then "token advances" if strictly smaller than "frame advances",
// else frame advances.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// cost: row-major [n_tokens][n_frames]. Returns the monotone path as (token, frame) pairs,
// first to last; always starts at (0,0) and ends at (n_tokens-1, n_frames-1).
std::vector<std::pair<int, int>> asr_dtw_path(const float * cost, int n_tokens, int n_frames);

// In-place median filter along the LAST axis of a row-major [n_rows][n_cols] array. width must be
// odd and < n_cols (reflect padding needs it); otherwise the data is left untouched and false returned.
bool asr_dtw_median_filter(float * data, int n_rows, int n_cols, int width);

// For each token row, the first frame at which the path enters that row (-1 if never).
std::vector<int> asr_dtw_token_start_frames(const std::vector<std::pair<int, int>> & path, int n_tokens);

// Full pipeline. attn: [n_heads][n_tokens][n_frames] row-major attention weights.
// Returns n_tokens start frames (multiply by the frame hop, 20 ms for Whisper, and add the
// window offset to get seconds). Empty on invalid arguments.
// frame_w (optional, n_frames values in [0, 1]): the cost of frame f is multiplied by frame_w[f], so a frame
// weighted 0 (silence) costs the same on every token row and cannot pull a token's path towards it. Not in
// whisper.cpp: see asr_align_speech_weights (asr-align.h) for why an audio LLM needs it.
std::vector<int> asr_dtw_token_frames(const float * attn, int n_heads, int n_tokens, int n_frames, int medfilt_width = 7,
                                      const float * frame_w = nullptr);
