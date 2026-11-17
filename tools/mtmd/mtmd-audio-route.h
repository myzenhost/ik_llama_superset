#pragma once
// Audio container routing for the mtmd helper (row S69-ENGREV-FASTERWHISPER-DECODEFMT-2026-10-27-01).
// SOURCE-READ: the named project's decode_audio pushes every container through a libav resampler to 16 kHz mono; no code copied.
// The fork decodes wav / mp3 / flac natively (miniaudio). Other containers are transcoded by an ffmpeg subprocess to raw f32.
// Pure helpers only (no I/O), so a model-free test can pin them.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mtmd_audio_route {

enum class kind {
    UNKNOWN,      // not a recognised audio container
    NATIVE,       // wav / flac / mp3: the in-process decoder
    TRANSCODE,    // ogg (opus/vorbis), amr, ADTS aac: ffmpeg
    MAYBE_VIDEO,  // mp4/m4a/mov (ftyp) or EBML (webm/mkv): try video first, then an audio-only ffmpeg transcode
};

inline kind classify(const unsigned char * p, size_t n) {
    if (!p || n < 4) {
        return kind::UNKNOWN;
    }
    if (n >= 12 && memcmp(p, "RIFF", 4) == 0 && memcmp(p + 8, "WAVE", 4) == 0) {
        return kind::NATIVE;
    }
    if (memcmp(p, "fLaC", 4) == 0 || memcmp(p, "ID3", 3) == 0) {
        return kind::NATIVE;
    }
    if (memcmp(p, "OggS", 4) == 0) {
        return kind::TRANSCODE;
    }
    if (n >= 5 && memcmp(p, "#!AMR", 5) == 0) {
        return kind::TRANSCODE;
    }
    if (n >= 8 && memcmp(p + 4, "ftyp", 4) == 0) {
        return kind::MAYBE_VIDEO;
    }
    if (p[0] == 0x1A && p[1] == 0x45 && p[2] == 0xDF && p[3] == 0xA3) {
        return kind::MAYBE_VIDEO;
    }
    // 11-bit frame sync. Layer bits ((b1 >> 1) & 3): 01 = layer III (mp3, native), 00 = ADTS aac (layer field is always 00).
    if (p[0] == 0xFF && (p[1] & 0xE0) == 0xE0) {
        const unsigned layer = (p[1] >> 1) & 0x3;
        if (layer == 0x1) {
            return kind::NATIVE;
        }
        if (layer == 0x0 && (p[1] & 0x18) != 0x08) {  // version bits 01 = reserved
            return kind::TRANSCODE;
        }
    }
    return kind::UNKNOWN;
}

// ffmpeg argument list: any container on stdin (through the seekable cache so an mp4 with the moov atom at the end works),
// mono, `rate` Hz, raw little-endian f32 on stdout. -err_detect is an input option (placed before -i).
inline std::vector<std::string> ffmpeg_argv(const std::string & bin, int rate) {
    return {
        bin, "-nostdin", "-read_ahead_limit", "-1", "-err_detect", "ignore_err", "-i", "cache:pipe:0",
        "-vn", "-ac", "1", "-ar", std::to_string(rate), "-f", "f32le", "-loglevel", "error", "pipe:1",
    };
}

// Sample count above which a clip is refused (never truncated). 0 = no cap.
inline uint64_t max_samples(double cap_seconds, int rate) {
    if (!(cap_seconds > 0.0) || rate <= 0) {
        return 0;
    }
    return (uint64_t) (cap_seconds * (double) rate + 0.5);
}

inline bool over_cap(uint64_t samples, uint64_t cap) {
    return cap != 0 && samples > cap;
}

inline std::string formats_hint() {
    return "supported audio: wav, mp3, flac natively; ogg/opus, webm, m4a/mp4, amr and aac need ffmpeg (set the ffmpeg bin dir or put ffmpeg on PATH)";
}

} // namespace mtmd_audio_route
