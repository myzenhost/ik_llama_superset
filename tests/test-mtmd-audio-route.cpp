// Audio container routing for the mtmd helper (row S69-ENGREV-FASTERWHISPER-DECODEFMT-2026-10-27-01).
//
// The native decoder (miniaudio) reads wav / mp3 / flac only. Everything else a client may upload to /v1/audio/transcriptions
// (ogg/opus, webm/opus, m4a/mp4, amr, ADTS aac) used to be refused; those go to an ffmpeg transcode to 16 kHz mono f32.
// This test pins the model-free part: which head bytes route where, the ffmpeg argument list, and the sample cap.
// The one defect this class had before: an ADTS aac head passes the old MPEG-sync test and is then undecodable.
//
// CONTROLS: a wav / flac / ID3 / MPEG-layer-III head stays on the native fast path (a router that sent everything to ffmpeg
// would pass the transcode checks and fail these); random bytes classify as unknown (a router that accepts anything would pass
// the positive checks and fail this one).
#include "mtmd-audio-route.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<unsigned char> head(std::initializer_list<unsigned char> b, size_t pad = 16) {
    std::vector<unsigned char> v(b);
    v.resize(std::max(v.size(), pad), 0);
    return v;
}

static std::vector<unsigned char> at(size_t off, const char * s, size_t total = 32) {
    std::vector<unsigned char> v(total, 0);
    memcpy(v.data() + off, s, strlen(s));
    return v;
}

using mtmd_audio_route::kind;

static kind cls(const std::vector<unsigned char> & v) {
    return mtmd_audio_route::classify(v.data(), v.size());
}

int main() {
    // native fast path (controls)
    {
        auto wav = at(0, "RIFF"); memcpy(wav.data() + 8, "WAVE", 4);
        CHECK(cls(wav) == kind::NATIVE, "wav must stay native");
        CHECK(cls(at(0, "fLaC")) == kind::NATIVE, "flac must stay native");
        CHECK(cls(at(0, "ID3")) == kind::NATIVE, "id3-tagged mp3 must stay native");
        CHECK(cls(head({0xFF, 0xFB, 0x90, 0x00})) == kind::NATIVE, "MPEG-1 layer III frame must stay native");
        CHECK(cls(head({0xFF, 0xF3, 0x90, 0x00})) == kind::NATIVE, "MPEG-2 layer III frame must stay native");
    }
    // transcode
    CHECK(cls(at(0, "OggS")) == kind::TRANSCODE, "ogg (opus/vorbis) must transcode");
    CHECK(cls(at(0, "#!AMR")) == kind::TRANSCODE, "amr must transcode");
    CHECK(cls(head({0xFF, 0xF1, 0x50, 0x80})) == kind::TRANSCODE, "ADTS aac (MPEG-4) must transcode, not pass as mp3");
    CHECK(cls(head({0xFF, 0xF9, 0x50, 0x80})) == kind::TRANSCODE, "ADTS aac (MPEG-2) must transcode, not pass as mp3");
    // containers that may or may not carry video: the caller tries video first, then audio-only transcode
    CHECK(cls(at(4, "ftyp")) == kind::MAYBE_VIDEO, "mp4/m4a is a possible video container");
    CHECK(cls(head({0x1A, 0x45, 0xDF, 0xA3})) == kind::MAYBE_VIDEO, "webm/mkv (EBML) is a possible video container");
    // unknown and short
    {
        const std::vector<unsigned char> junk = head({0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44});
        CHECK(cls(junk) == kind::UNKNOWN, "random bytes must be unknown");
        CHECK(mtmd_audio_route::classify(junk.data(), 4) == kind::UNKNOWN, "short buffer must be unknown");
        CHECK(mtmd_audio_route::classify(nullptr, 0) == kind::UNKNOWN, "null buffer must be unknown");
    }
    // ffmpeg argv: input options before -i, no video, mono, rate, raw f32 on stdout
    {
        const std::vector<std::string> a = mtmd_audio_route::ffmpeg_argv("ffmpeg", 16000);
        auto idx = [&](const char * s) { for (size_t i = 0; i < a.size(); ++i) if (a[i] == s) return (int) i; return -1; };
        CHECK(!a.empty() && a[0] == "ffmpeg", "argv[0] is the binary");
        CHECK(idx("-i") > 0 && idx("-err_detect") > 0 && idx("-err_detect") < idx("-i"), "-err_detect is an input option: before -i");
        CHECK(idx("-err_detect") > 0 && a[idx("-err_detect") + 1] == "ignore_err", "-err_detect ignore_err");
        CHECK(idx("-vn") > 0, "no video stream");
        CHECK(idx("-ac") > 0 && a[idx("-ac") + 1] == "1", "mono");
        CHECK(idx("-ar") > 0 && a[idx("-ar") + 1] == "16000", "target rate");
        CHECK(idx("-f") > 0 && a[idx("-f") + 1] == "f32le", "raw f32 little endian");
        CHECK(idx("-nostdin") > 0, "ffmpeg must not read the terminal");
        CHECK(a.back() == "pipe:1", "output on stdout");
        CHECK(a[idx("-i") + 1] == "cache:pipe:0", "stdin through the seekable cache (moov atom at the end of an mp4)");
        const std::vector<std::string> b = mtmd_audio_route::ffmpeg_argv("C:/x/ffmpeg.exe", 24000);
        CHECK(b[0] == "C:/x/ffmpeg.exe" && b[(size_t) (std::find(b.begin(), b.end(), std::string("-ar")) - b.begin()) + 1] == "24000", "binary and rate are parameters");
    }
    // cap: the sample count at which a clip is refused (never truncated)
    CHECK(mtmd_audio_route::max_samples(14400.0, 16000) == 230400000ull, "4 h at 16 kHz");
    CHECK(mtmd_audio_route::max_samples(0.0, 16000) == 0ull, "cap 0 = off");
    CHECK(mtmd_audio_route::max_samples(-1.0, 16000) == 0ull, "negative cap = off");
    CHECK(mtmd_audio_route::max_samples(10.0, 0) == 0ull, "no rate = off");
    CHECK(mtmd_audio_route::over_cap(230400001ull, 230400000ull), "one past the cap is refused");
    CHECK(!mtmd_audio_route::over_cap(230400000ull, 230400000ull), "exactly the cap is accepted");
    CHECK(!mtmd_audio_route::over_cap(5ull, 0ull), "cap 0 never refuses");
    // the user-facing refusal names the formats
    {
        const std::string h = mtmd_audio_route::formats_hint();
        CHECK(h.find("ogg") != std::string::npos && h.find("wav") != std::string::npos && h.find("ffmpeg") != std::string::npos, "hint names formats and ffmpeg");
    }

    if (g_fail) {
        printf("test-mtmd-audio-route: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-mtmd-audio-route: all passed\n");
    return 0;
}
