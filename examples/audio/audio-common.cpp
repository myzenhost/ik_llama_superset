// audio-common.cpp -- see audio-common.h for provenance.

#include "audio-common.h"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//
// ---------------------------------------------------------------- WAV I/O ---
//

namespace {

struct riff_reader {
    const uint8_t * p   = nullptr;
    size_t          n   = 0;
    size_t          off = 0;

    bool need(size_t k) const { return off + k <= n; }

    uint32_t u32() { uint32_t v; memcpy(&v, p + off, 4); off += 4; return v; }
    uint16_t u16() { uint16_t v; memcpy(&v, p + off, 2); off += 2; return v; }
};

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// WAVE_FORMAT tags
constexpr uint16_t WAVE_FMT_PCM        = 0x0001;
constexpr uint16_t WAVE_FMT_IEEE_FLOAT = 0x0003;
constexpr uint16_t WAVE_FMT_EXTENSIBLE = 0xFFFE;

} // namespace

bool audio_wav_read(const std::string & path, audio_wav & out, std::string & err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "cannot open '" + path + "'";
        return false;
    }

    const std::streamsize size = f.tellg();
    if (size < 44) {
        err = "'" + path + "' is too small to be a WAV file";
        return false;
    }
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> buf((size_t) size);
    if (!f.read(reinterpret_cast<char *>(buf.data()), size)) {
        err = "short read on '" + path + "'";
        return false;
    }

    riff_reader r{buf.data(), buf.size(), 0};

    if (memcmp(r.p, "RIFF", 4) != 0 || memcmp(r.p + 8, "WAVE", 4) != 0) {
        err = "'" + path + "' is not a RIFF/WAVE file";
        return false;
    }
    r.off = 12;

    uint16_t fmt_tag        = 0;
    uint16_t channels       = 0;
    uint32_t sample_rate    = 0;
    uint16_t bits_per_samp  = 0;

    bool     have_fmt  = false;
    const uint8_t * data_ptr = nullptr;
    size_t          data_len = 0;

    // walk the chunk list; do NOT assume the canonical 44-byte header
    while (r.need(8)) {
        char id[5] = {0};
        memcpy(id, r.p + r.off, 4);
        r.off += 4;
        const uint32_t csz = r.u32();
        const size_t   cend = r.off + csz;

        if (memcmp(id, "fmt ", 4) == 0 && csz >= 16 && r.need(16)) {
            const size_t base = r.off;
            fmt_tag       = r.u16();
            channels      = r.u16();
            sample_rate   = r.u32();
            r.u32();  // byte rate
            r.u16();  // block align
            bits_per_samp = r.u16();

            if (fmt_tag == WAVE_FMT_EXTENSIBLE && csz >= 40 && base + 40 <= r.n) {
                // cbSize(2) + validBits(2) + channelMask(4) + GUID(16);
                // the first two GUID bytes carry the real format tag
                uint16_t sub = 0;
                memcpy(&sub, r.p + base + 24, 2);
                fmt_tag = sub;
            }
            have_fmt = true;
        } else if (memcmp(id, "data", 4) == 0) {
            data_ptr = r.p + r.off;
            data_len = std::min<size_t>(csz, r.n - r.off);
        }

        r.off = cend + (cend & 1); // chunks are word-aligned
        if (r.off <= 12 || r.off > r.n) {
            break;
        }
    }

    if (!have_fmt) {
        err = "'" + path + "' has no fmt chunk";
        return false;
    }
    if (!data_ptr || data_len == 0) {
        err = "'" + path + "' has no data chunk";
        return false;
    }
    if (channels == 0) {
        err = "'" + path + "' declares 0 channels";
        return false;
    }

    out.sample_rate = (int32_t) sample_rate;
    out.channels    = (int32_t) channels;
    out.samples.clear();

    const size_t bytes = bits_per_samp / 8;
    if (bytes == 0) {
        err = "'" + path + "' declares 0 bits per sample";
        return false;
    }
    const size_t n_samp = data_len / bytes;
    out.samples.reserve(n_samp);

    if (fmt_tag == WAVE_FMT_PCM && bits_per_samp == 16) {
        for (size_t i = 0; i < n_samp; ++i) {
            int16_t v;
            memcpy(&v, data_ptr + i * 2, 2);
            out.samples.push_back((float) v / 32768.0f);
        }
    } else if (fmt_tag == WAVE_FMT_PCM && bits_per_samp == 8) {
        for (size_t i = 0; i < n_samp; ++i) {
            out.samples.push_back(((float) data_ptr[i] - 128.0f) / 128.0f);
        }
    } else if (fmt_tag == WAVE_FMT_PCM && bits_per_samp == 24) {
        for (size_t i = 0; i < n_samp; ++i) {
            const uint8_t * q = data_ptr + i * 3;
            int32_t v = (int32_t) ((uint32_t) q[0] << 8 | (uint32_t) q[1] << 16 | (uint32_t) q[2] << 24);
            out.samples.push_back((float) (v >> 8) / 8388608.0f);
        }
    } else if (fmt_tag == WAVE_FMT_PCM && bits_per_samp == 32) {
        for (size_t i = 0; i < n_samp; ++i) {
            int32_t v;
            memcpy(&v, data_ptr + i * 4, 4);
            out.samples.push_back((float) v / 2147483648.0f);
        }
    } else if (fmt_tag == WAVE_FMT_IEEE_FLOAT && bits_per_samp == 32) {
        for (size_t i = 0; i < n_samp; ++i) {
            float v;
            memcpy(&v, data_ptr + i * 4, 4);
            out.samples.push_back(v);
        }
    } else if (fmt_tag == WAVE_FMT_IEEE_FLOAT && bits_per_samp == 64) {
        for (size_t i = 0; i < n_samp; ++i) {
            double v;
            memcpy(&v, data_ptr + i * 8, 8);
            out.samples.push_back((float) v);
        }
    } else {
        err = "'" + path + "': unsupported WAV format tag " + std::to_string(fmt_tag) +
              " with " + std::to_string(bits_per_samp) + " bits per sample";
        return false;
    }

    return true;
}

std::vector<uint8_t> audio_wav_encode_s16(const std::vector<float> & mono, int32_t sample_rate) {
    struct wav_header {
        char     riff[4]        = {'R', 'I', 'F', 'F'};
        uint32_t chunk_size     = 0;
        char     wave[4]        = {'W', 'A', 'V', 'E'};
        char     fmt[4]         = {'f', 'm', 't', ' '};
        uint32_t fmt_chunk_size = 16;
        uint16_t audio_format   = 1; // PCM
        uint16_t num_channels   = 1; // mono
        uint32_t sample_rate    = 0;
        uint32_t byte_rate      = 0;
        uint16_t block_align    = 0;
        uint16_t bits_per_sample = 16;
        char     data[4]        = {'d', 'a', 't', 'a'};
        uint32_t data_size      = 0;
    } h;

    h.sample_rate = (uint32_t) sample_rate;
    h.byte_rate   = h.sample_rate * h.num_channels * (h.bits_per_sample / 8);
    h.block_align = h.num_channels * (h.bits_per_sample / 8);
    h.data_size   = (uint32_t) (mono.size() * 2);
    h.chunk_size  = 36 + h.data_size;

    std::vector<uint8_t> out;
    out.resize(sizeof(h) + mono.size() * 2);
    memcpy(out.data(), &h, sizeof(h));

    uint8_t * q = out.data() + sizeof(h);
    for (size_t i = 0; i < mono.size(); ++i) {
        const int16_t s = (int16_t) clampf(mono[i] * 32767.0f, -32768.0f, 32767.0f);
        memcpy(q + i * 2, &s, 2);
    }
    return out;
}

bool audio_wav_write_s16(const std::string & path, const std::vector<float> & mono, int32_t sample_rate, std::string & err) {
    const std::vector<uint8_t> img = audio_wav_encode_s16(mono, sample_rate);
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot write '" + path + "'";
        return false;
    }
    f.write(reinterpret_cast<const char *>(img.data()), (std::streamsize) img.size());
    if (!f) {
        err = "short write on '" + path + "'";
        return false;
    }
    return true;
}

//
// ------------------------------------------------------------- conversion ---
//

std::vector<float> audio_to_mono(const std::vector<float> & interleaved, int32_t channels) {
    if (channels <= 1) {
        return interleaved;
    }
    const size_t n = interleaved.size() / (size_t) channels;
    std::vector<float> mono(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int32_t c = 0; c < channels; ++c) {
            acc += interleaved[i * channels + c];
        }
        mono[i] = acc / (float) channels;
    }
    return mono;
}

std::vector<float> audio_resample(const std::vector<float> & mono, int32_t sr_in, int32_t sr_out, int taps) {
    if (sr_in == sr_out || sr_in <= 0 || sr_out <= 0 || mono.empty()) {
        return mono;
    }

    const double ratio  = (double) sr_out / (double) sr_in;
    const size_t n_out  = (size_t) std::llround((double) mono.size() * ratio);
    // when down-sampling the low-pass cutoff has to follow the OUTPUT rate
    const double cutoff = std::min(1.0, ratio);
    const int    half   = std::max(4, taps);

    std::vector<float> out(n_out, 0.0f);

    for (size_t i = 0; i < n_out; ++i) {
        const double center = (double) i / ratio;
        const int64_t c0    = (int64_t) std::floor(center);

        double acc = 0.0;
        double wsum = 0.0;
        for (int64_t k = c0 - half + 1; k <= c0 + half; ++k) {
            if (k < 0 || k >= (int64_t) mono.size()) {
                continue;
            }
            const double x = (double) k - center;
            const double sx = M_PI * x * cutoff;
            const double sinc = (std::fabs(sx) < 1e-9) ? 1.0 : std::sin(sx) / sx;
            // Blackman window over the [-half, half] support
            const double t = (x + half) / (2.0 * half);
            const double w = 0.42 - 0.5 * std::cos(2.0 * M_PI * t) + 0.08 * std::cos(4.0 * M_PI * t);
            const double coeff = sinc * w;
            acc  += mono[(size_t) k] * coeff;
            wsum += coeff;
        }
        out[i] = (float) (wsum != 0.0 ? acc / wsum : 0.0);
    }

    return out;
}

bool audio_load_mono(const std::string & path, int32_t sr_out, std::vector<float> & out, std::string & err) {
    audio_wav wav;
    if (!audio_wav_read(path, wav, err)) {
        return false;
    }
    std::vector<float> mono = audio_to_mono(wav.samples, wav.channels);
    out = sr_out > 0 ? audio_resample(mono, wav.sample_rate, sr_out) : mono;
    return true;
}

void audio_peak_normalize(std::vector<float> & mono, float peak) {
    float mx = 0.0f;
    for (float v : mono) {
        mx = std::max(mx, std::fabs(v));
    }
    if (mx <= 1e-6f) {
        return;
    }
    const float g = peak / mx;
    for (float & v : mono) {
        v *= g;
    }
}

float audio_rms(const float * x, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += (double) x[i] * x[i];
    }
    return (float) std::sqrt(acc / (double) n);
}

float audio_dbfs(float rms) {
    return 20.0f * std::log10(std::max(rms, 1e-10f));
}

//
// -------------------------------------------------------------------- VAD ---
//

namespace {

struct vad_frame {
    float db  = 0.0f;
    float zcr = 0.0f;
};

std::vector<vad_frame> vad_analyse(const std::vector<float> & mono, int32_t sr, int n_win) {
    std::vector<vad_frame> frames;
    if (n_win <= 0) {
        return frames;
    }
    (void) sr;
    frames.reserve(mono.size() / (size_t) n_win + 1);

    for (size_t off = 0; off < mono.size(); off += (size_t) n_win) {
        const size_t n = std::min<size_t>((size_t) n_win, mono.size() - off);
        vad_frame fr;
        fr.db = audio_dbfs(audio_rms(mono.data() + off, n));

        size_t zc = 0;
        for (size_t i = 1; i < n; ++i) {
            const float a = mono[off + i - 1];
            const float b = mono[off + i];
            if ((a >= 0.0f) != (b >= 0.0f)) {
                zc++;
            }
        }
        fr.zcr = n > 1 ? (float) zc / (float) (n - 1) : 0.0f;
        frames.push_back(fr);
    }
    return frames;
}

// Noise floor = 20th percentile frame energy. Robust when speech dominates,
// and does not collapse on an all-silence buffer the way a min() would.
float vad_noise_floor(std::vector<vad_frame> frames) {
    if (frames.empty()) {
        return -90.0f;
    }
    std::vector<float> dbs;
    dbs.reserve(frames.size());
    for (const auto & f : frames) {
        dbs.push_back(f.db);
    }
    std::sort(dbs.begin(), dbs.end());
    const size_t idx = (size_t) (dbs.size() * 0.2);
    return dbs[std::min(idx, dbs.size() - 1)];
}

} // namespace

std::vector<audio_segment> audio_vad_segments(
        const std::vector<float> & mono,
        int32_t                    sample_rate,
        const audio_vad_params &   params) {

    std::vector<audio_segment> segs;
    if (mono.empty() || sample_rate <= 0) {
        return segs;
    }

    const int n_win = std::max(1, (int) (sample_rate * params.frame_ms / 1000.0f));
    const auto frames = vad_analyse(mono, sample_rate, n_win);
    if (frames.empty()) {
        return segs;
    }

    const float floor_db = vad_noise_floor(frames);
    const float thresh   = floor_db + params.threshold_db;

    const int min_speech_fr  = std::max(1, (int) (params.min_speech_ms  / params.frame_ms));
    const int min_silence_fr = std::max(1, (int) (params.min_silence_ms / params.frame_ms));

    // 1) raw frame decisions -> runs
    bool in_speech = false;
    int  run_start = 0;
    int  silence   = 0;

    std::vector<std::pair<int,int>> runs; // [f0, f1) in frames

    for (int i = 0; i < (int) frames.size(); ++i) {
        const bool voiced = frames[i].db > thresh && frames[i].zcr <= params.zcr_max;
        if (voiced) {
            if (!in_speech) {
                in_speech = true;
                run_start = i;
            }
            silence = 0;
        } else if (in_speech) {
            if (++silence >= min_silence_fr) {
                const int f1 = i - silence + 1;
                if (f1 - run_start >= min_speech_fr) {
                    runs.emplace_back(run_start, f1);
                }
                in_speech = false;
                silence   = 0;
            }
        }
    }
    if (in_speech) {
        const int f1 = (int) frames.size();
        if (f1 - run_start >= min_speech_fr) {
            runs.emplace_back(run_start, f1);
        }
    }

    if (runs.empty()) {
        return segs;
    }

    // 2) pad, then merge anything closer than merge_gap_s
    const int pad_fr   = (int) (params.padding_ms / params.frame_ms);
    const int merge_fr = (int) (params.merge_gap_s * 1000.0f / params.frame_ms);

    std::vector<std::pair<int,int>> merged;
    for (auto & r : runs) {
        const int f0 = std::max(0, r.first - pad_fr);
        const int f1 = std::min((int) frames.size(), r.second + pad_fr);
        if (!merged.empty() && f0 - merged.back().second <= merge_fr) {
            merged.back().second = std::max(merged.back().second, f1);
        } else {
            merged.emplace_back(f0, f1);
        }
    }

    // 3) hard-split anything longer than max_segment_s
    const int max_fr = std::max(1, (int) (params.max_segment_s * 1000.0f / params.frame_ms));

    for (auto & m : merged) {
        for (int f0 = m.first; f0 < m.second; f0 += max_fr) {
            const int f1 = std::min(m.second, f0 + max_fr);

            audio_segment s;
            s.i0 = std::min(mono.size(), (size_t) f0 * (size_t) n_win);
            s.i1 = std::min(mono.size(), (size_t) f1 * (size_t) n_win);
            if (s.i1 <= s.i0) {
                continue;
            }
            s.t0_s = (double) s.i0 / sample_rate;
            s.t1_s = (double) s.i1 / sample_rate;
            segs.push_back(s);
        }
    }

    return segs;
}

bool audio_has_speech(const std::vector<float> & mono, int32_t sample_rate, const audio_vad_params & params) {
    return !audio_vad_segments(mono, sample_rate, params).empty();
}

//
// ------------------------------------------------------------- STFT/ISTFT ---
//

void fill_hann_window(int length, bool periodic, float * output) {
    const int offset = periodic ? 0 : -1;
    for (int i = 0; i < length; i++) {
        output[i] = 0.5f * (1.0f - cosf((2.0f * (float) M_PI * i) / (length + offset)));
    }
}

void audio_irfft_plan::init(int n_fft) {
    n = n_fft;
    tw_real.resize(n);
    tw_imag.resize(n);
    for (int i = 0; i < n; ++i) {
        const double angle = 2.0 * M_PI * i / (double) n;
        tw_real[i] = (float) std::cos(angle);
        tw_imag[i] = (float) std::sin(angle);
    }
}

void audio_irfft_plan::run(const float * inp_cplx, float * out_real) const {
    const int N = n / 2 + 1;

    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        // index into the twiddle table is (k*m) mod n
        int idx = 0;
        for (int m = 0; m < N; ++m) {
            const float tr = tw_real[idx];
            const float ti = tw_imag[idx];
            acc += (double) inp_cplx[2 * m] * tr - (double) inp_cplx[2 * m + 1] * ti;

            idx += k;
            if (idx >= n) {
                idx -= n * (idx / n);
            }
        }
        out_real[k] = (float) (acc / (double) N);
    }
}

void audio_fold(const std::vector<float> & data,
                int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad,
                std::vector<float> & output) {
    const int64_t output_height = n_out;
    const int64_t kernel_w      = n_win;
    const int64_t stride_w      = n_hop;
    const int64_t width         = n_out;

    output.assign((size_t) width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        const int64_t start = w_col * stride_w - n_pad;
        const int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[(size_t) w_im] += data[(size_t) col_idx];
            }
            col_idx++;
        }
    }

    output.resize((size_t) (n_out - 2 * n_pad));
}

std::vector<float> audio_embd_to_wave(
        const float * embd,
        int           n_codes,
        int           n_embd,
        int           n_fft,
        int           n_hop,
        int           n_threads) {

    if (n_codes <= 0 || n_embd <= 0) {
        return {};
    }

    const int n_win = n_fft;
    const int n_pad = (n_win - n_hop) / 2;
    const int n_out = (n_codes - 1) * n_hop + n_win;

    std::vector<float> hann(n_fft);
    fill_hann_window((int) hann.size(), true, hann.data());

    const int n_spec = n_embd * n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k * n_codes + l] = embd[l * n_embd + k];
        }
    }

    for (int k = 0; k < n_embd / 2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k)*n_codes + l];
            const float phi = E[(k + n_embd / 2) * n_codes + l];

            mag = expf(mag);
            if (mag > 1e2f) {
                mag = 1e2f;
            }

            S[2 * (k * n_codes + l) + 0] = mag * cosf(phi);
            S[2 * (k * n_codes + l) + 1] = mag * sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd / 2; ++k) {
            ST[l * n_embd + 2 * k + 0] = S[2 * (k * n_codes + l) + 0];
            ST[l * n_embd + 2 * k + 1] = S[2 * (k * n_codes + l) + 1];
        }
    }

    audio_irfft_plan plan;
    plan.init(n_fft);

    std::vector<float> res  ((size_t) n_codes * n_fft);
    std::vector<float> hann2((size_t) n_codes * n_fft);

    const int nth = std::max(1, n_threads);
    std::vector<std::thread> workers((size_t) nth);
    for (int i = 0; i < nth; ++i) {
        workers[(size_t) i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += nth) {
                plan.run(ST.data() + (size_t) l * n_embd, res.data() + (size_t) l * n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [(size_t) l * n_fft + j] *= hann[j];
                    hann2[(size_t) l * n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (auto & w : workers) {
        w.join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    audio_fold(res,   n_out, n_win, n_hop, n_pad, audio);
    audio_fold(hann2, n_out, n_win, n_hop, n_pad, env);

    for (size_t i = 0; i < audio.size(); ++i) {
        if (std::fabs(env[i]) > 1e-8f) {
            audio[i] /= env[i];
        }
    }

    return audio;
}
