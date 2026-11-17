// audio-common.h
//
// Shared audio utilities for the unified-audio surface of ik_llama.cpp
// (llama-audio / llama-tts).
//
// Design adopted from audio.cpp (https://github.com/0xShug0/audio.cpp, Apache-2.0),
// whose "Audio Utilities" layer keeps WAV I/O, resampling, VAD chunking and
// STFT/ISTFT as ONE reusable block shared by every model family instead of
// re-implementing them per model. The ISTFT/`fold` head follows the OuteTTS
// vocoder path of llama.cpp b4404 examples/tts/tts.cpp (MIT).
//
// Nothing in here depends on llama or ggml: it is plain C++17 DSP so that both
// the ASR (listen) and TTS (speak) sides of llama-audio can share it.
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

//
// ---------------------------------------------------------------- WAV I/O ---
//

struct audio_wav {
    int32_t sample_rate = 0;
    int32_t channels    = 0;
    // interleaved, [-1, 1]
    std::vector<float> samples;

    size_t n_frames() const {
        return channels > 0 ? samples.size() / (size_t) channels : 0;
    }
    double duration_s() const {
        return sample_rate > 0 ? (double) n_frames() / sample_rate : 0.0;
    }
};

// Reads a RIFF/WAVE file. Supports PCM u8 / s16 / s24 / s32 and IEEE float
// 32/64, including WAVE_FORMAT_EXTENSIBLE, and skips unknown chunks (LIST,
// fact, ...) rather than assuming the classic 44-byte header.
bool audio_wav_read(const std::string & path, audio_wav & out, std::string & err);

// Encodes mono float samples as a 16-bit PCM WAV image in memory.
std::vector<uint8_t> audio_wav_encode_s16(const std::vector<float> & mono, int32_t sample_rate);

// Writes mono float samples as a 16-bit PCM WAV file.
bool audio_wav_write_s16(const std::string & path, const std::vector<float> & mono, int32_t sample_rate, std::string & err);

//
// ------------------------------------------------------------- conversion ---
//

// Down-mixes interleaved audio to mono by averaging channels.
std::vector<float> audio_to_mono(const std::vector<float> & interleaved, int32_t channels);

// Windowed-sinc resampler (Blackman window, 32 taps per side by default).
// Falls back to a straight copy when the rates match.
std::vector<float> audio_resample(const std::vector<float> & mono, int32_t sr_in, int32_t sr_out, int taps = 32);

// Convenience: read a file and hand back mono audio at the requested rate.
bool audio_load_mono(const std::string & path, int32_t sr_out, std::vector<float> & out, std::string & err);

// In-place peak normalisation to `peak` (no-op on silence).
void  audio_peak_normalize(std::vector<float> & mono, float peak = 0.95f);
float audio_rms(const float * x, size_t n);
float audio_dbfs(float rms);

//
// -------------------------------------------------------------------- VAD ---
//

// Energy + zero-crossing voice activity detection with an adaptive noise floor.
// The chunking knobs mirror audio.cpp's `--vad-chunk-*` CLI surface so long
// inputs can be split into model-sized windows.
struct audio_vad_params {
    float frame_ms            = 30.0f;  // analysis frame
    float threshold_db        = 9.0f;   // dB above the estimated noise floor
    float min_speech_ms       = 200.0f; // drop shorter blips
    float min_silence_ms      = 400.0f; // silence needed to close a segment
    float padding_ms          = 200.0f; // context kept around each segment
    float max_segment_s       = 30.0f;  // hard split (whisper-style window)
    float merge_gap_s         = 0.4f;   // merge segments closer than this
    float zcr_max             = 0.35f;  // reject fricative-like noise bursts
};

struct audio_segment {
    double t0_s = 0.0;
    double t1_s = 0.0;
    size_t i0   = 0; // sample index, inclusive
    size_t i1   = 0; // sample index, exclusive
};

std::vector<audio_segment> audio_vad_segments(
        const std::vector<float> & mono,
        int32_t                    sample_rate,
        const audio_vad_params &   params);

// true when the buffer carries speech-like energy at all (cheap gate for
// streaming/`--listen` paths).
bool audio_has_speech(const std::vector<float> & mono, int32_t sample_rate, const audio_vad_params & params);

//
// ------------------------------------------------------------- STFT/ISTFT ---
//

void fill_hann_window(int length, bool periodic, float * output);

// Inverse real FFT with a cached twiddle table.
//
// llama.cpp b4404 recomputed cosf/sinf for every (k,m) pair, which dominates
// vocoder wall time; the table makes it a lookup. Same numerics, same layout:
// `inp_cplx` is n/2+1 interleaved (re,im) pairs, `out_real` is n floats.
struct audio_irfft_plan {
    int n = 0;
    std::vector<float> tw_real; // [n]
    std::vector<float> tw_imag; // [n]

    void init(int n_fft);
    void run(const float * inp_cplx, float * out_real) const;
};

// torch.nn.functional.fold equivalent used for overlap-add.
void audio_fold(const std::vector<float> & data,
                int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad,
                std::vector<float> & output);

// Turns a vocoder embedding matrix (n_codes x n_embd, magnitude in the first
// half and phase in the second) into a waveform via ISTFT + overlap-add.
std::vector<float> audio_embd_to_wave(
        const float * embd,
        int           n_codes,
        int           n_embd,
        int           n_fft,
        int           n_hop,
        int           n_threads);
