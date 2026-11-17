// asr-compression.h
//
// Whole-text compression-ratio check used to catch looping / degenerate ASR output.
//
// Semantics follow faster-whisper's get_compression_ratio()
// (SYSTRAN/faster-whisper, MIT, faster_whisper/transcribe.py @ 9fa78645):
//     ratio = len(utf8(text)) / len(zlib.compress(utf8(text)))
// and the "repeats" flag is ratio > 2.4 (the same default whisper.cpp and openai/whisper use).
//
// Backend: if the build found zlib (ASR_HAVE_ZLIB) the ratio is computed with
// compress2(Z_DEFAULT_COMPRESSION), i.e. byte-identical to Python's zlib.compress(). Otherwise
// an own, documented estimator is used (asr_compression_ratio_estimate); it is an
// approximation of deflate, NOT a bit-exact one. Both are always callable so a test can
// compare them.
//
// Pure functions; no llama/ggml dependency.
//
#pragma once

#include <string>

// Default "repeats" threshold (faster-whisper compression_ratio_threshold, whisper.cpp 2.4).
constexpr double ASR_COMPRESSION_RATIO_THOLD = 2.4;

// true when this build links zlib.
bool asr_compression_zlib_available();

// len(text) / len(zlib.compress(text)). Returns < 0 when zlib is not compiled in.
// Empty text returns 0.
double asr_compression_ratio_zlib(const std::string & text);

// Own estimator: greedy LZ77 parse (3..258 byte matches, 32 KiB window, hash chain), then a
// deflate-like size model: literal bytes cost their order-0 Shannon entropy (bits), each match
// costs a fixed ~ASR_EST_MATCH_BITS, plus a 6 byte zlib header/trailer. Empty text returns 0.
double asr_compression_ratio_estimate(const std::string & text);

// zlib when available, else the estimator.
double asr_compression_ratio(const std::string & text);

// true when asr_compression_ratio(text) > thold.
bool asr_text_repeats(const std::string & text, double thold = ASR_COMPRESSION_RATIO_THOLD);
