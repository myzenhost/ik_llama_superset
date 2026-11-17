// wavtokenizer.h
//
// Standalone ggml implementation of the WavTokenizer decoder (the "codes to
// speech" half of the OuteTTS pipeline).
//
// WHY STANDALONE, AND NOT A libllama ARCHITECTURE:
//   upstream llama.cpp carries this as LLM_ARCH_WAVTOKENIZER_DEC inside
//   src/llama-model.cpp. ik_llama.cpp is deliberately divergent from upstream
//   (its IQK / Trellis / MXFP4 quants and CPU prompt-processing speedups depend
//   on that divergence), so bolting a new architecture into src/llama.cpp would
//   put a large upstream-shaped change straight through the most divergent file
//   in the tree. Instead the decoder owns its own gguf load, its own backend and
//   its own graph -- the same shape audio.cpp uses for every model family
//   (src/models/<family>/{weights,session}.cpp). Adding audio capability
//   therefore touches ZERO core files.
//
// Graph ported from llama.cpp src/models/wavtokenizer-dec.cpp (MIT).
// Architecture: conv1d -> 6-layer posnet (resnet/resnet/attn/resnet/resnet/norm)
//               -> layer norm -> 12-layer ConvNeXt -> norm -> head.
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct wavtokenizer;

struct wavtokenizer_params {
    bool use_gpu    = false; // honoured only when built with a GPU backend
    int  n_threads  = 4;
    bool verbose    = false;
};

// Loads a WavTokenizer decoder GGUF. Returns nullptr and fills `err` on failure.
wavtokenizer * wavtokenizer_init_from_file(const std::string & path,
                                           const wavtokenizer_params & params,
                                           std::string & err);

void wavtokenizer_free(wavtokenizer * wt);

// Number of output features per frame (the ISTFT spectrogram width).
int wavtokenizer_n_embd_out(const wavtokenizer * wt);

// Vocabulary size of the acoustic codebook.
int wavtokenizer_n_vocab(const wavtokenizer * wt);

// A one-line description of the loaded decoder, for --version style output.
std::string wavtokenizer_describe(const wavtokenizer * wt);

// Runs the decoder over `n_codes` acoustic codes and returns the raw
// spectrogram embeddings, laid out [n_codes][n_embd_out] (row major), which is
// exactly what audio_embd_to_wave() consumes.
bool wavtokenizer_decode(wavtokenizer * wt,
                         const int32_t * codes,
                         int             n_codes,
                         std::vector<float> & out_embd,
                         std::string & err);
