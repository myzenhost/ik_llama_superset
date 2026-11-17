#pragma once

// Run the AUDIO ENCODER of an mmproj GGUF on its own, with no text model: PCM samples in, one encoder state per
// output frame out. This is the half of the mtmd audio path that an encoder-decoder ASR model (a transducer such as
// Parakeet-TDT) needs; the projector into an LLM is not used, and mtmd_init_from_file's text-model checks do not
// apply. Supported projector types: parakeet (encoder-only GGUFs from examples/audio/convert-parakeet-tdt-to-gguf.py).

#include "mtmd.h"

#include <cstddef>
#include <string>
#include <vector>

struct mtmd_audio_encoder;

struct mtmd_audio_encoder_params {
    int  n_threads = 4;
    bool use_gpu   = false;
    bool warmup    = false;
};

// nullptr on failure, with the reason in *err when err is not null
MTMD_API mtmd_audio_encoder * mtmd_audio_encoder_init(const char * fname, const mtmd_audio_encoder_params & params,
                                                      std::string * err = nullptr);
MTMD_API void mtmd_audio_encoder_free(mtmd_audio_encoder * enc);

MTMD_API int mtmd_audio_encoder_n_embd     (const mtmd_audio_encoder * enc); // width of one output frame
MTMD_API int mtmd_audio_encoder_sample_rate(const mtmd_audio_encoder * enc); // the rate the samples must be at
MTMD_API int mtmd_audio_encoder_hop        (const mtmd_audio_encoder * enc); // samples per mel frame
MTMD_API int mtmd_audio_encoder_subsampling(const mtmd_audio_encoder * enc); // mel frames per output frame

// samples: mono PCM at mtmd_audio_encoder_sample_rate(). out: n_frames rows of n_embd floats, row-major.
MTMD_API bool mtmd_audio_encoder_encode(mtmd_audio_encoder * enc, const float * samples, size_t n_samples,
                                        std::vector<float> & out, int & n_frames);
