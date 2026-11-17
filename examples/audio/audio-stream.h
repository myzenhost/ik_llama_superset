// audio-stream.h -- live (sliding-window) transcription building blocks for `llama-audio --task stream`.
//
// Two pieces, both free of any audio API so they can be tested without a microphone:
//   audio_ring           a thread-safe ring buffer: the capture callback pushes, the decode loop takes
//   audio_stream_window  the step / length / keep window arithmetic of whisper.cpp's live-mic example
//
// SOURCE-READ (rewritten, no code copied; MIT): ggml-org/whisper.cpp @ 6e4ab854f67f743900934a703d5603419384c961
//   examples/common-sdl.cpp:139-168 callback (ring write, keeps the newest samples when one call exceeds the ring),
//   :170-211 get (the trailing N ms), :117-136 clear; examples/common-sdl.h:15-46 (class audio_async).
//   examples/stream/stream.cpp:129-139 (keep <= step, length >= step, n_new_line = max(1, length/step - 1)),
//   :260-274 (wait for one step of new audio; drop with a warning when the loop falls behind),
//   :278-291 (window = up to keep + length - new samples of the previous window, then the new ones),
//   :408-412 (every n_new_line steps the line is committed and only `keep` ms carry into the next one).
// Differences on purpose: audio_ring counts what was pushed since the last take, so "falling behind" is measured
// (stream.cpp's get(step_ms) can never return more than one step, which makes its 2*step drop check unreachable),
// and a final short step at end of input is still transcribed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

class audio_ring {
public:
    explicit audio_ring(size_t capacity);

    // capture side: append; when more than the capacity arrives, only the newest `capacity` samples are kept
    void push(const float * x, size_t n);

    // consumer side: every sample pushed since the previous take (the newest `capacity` of them) goes to `out`.
    // returns how many were pushed since the previous take -- more than out.size() means samples were lost.
    size_t take(std::vector<float> & out);

    size_t   pending() const;   // pushed since the previous take
    uint64_t total()   const;   // pushed since construction
    size_t   capacity() const { return m_buf.size(); }

private:
    mutable std::mutex m_mutex;
    std::vector<float> m_buf;
    size_t   m_pos     = 0;     // next write position
    size_t   m_pending = 0;     // pushed since the last take (may exceed capacity)
    uint64_t m_total   = 0;
};

struct audio_stream_params {
    int step_ms     = 3000;
    int length_ms   = 10000;
    int keep_ms     = 200;
    int sample_rate = 16000;
};

class audio_stream_window {
public:
    explicit audio_stream_window(const audio_stream_params & p);

    // the window to transcribe after `fresh` new samples: the tail of the previous window, then `fresh`
    const std::vector<float> & next(const std::vector<float> & fresh);

    // call after the window from next() was transcribed. true = this step ends a line: the caller commits the
    // text, and only `keep` samples carry over into the next window.
    bool done_step();

    // End the line NOW (a silence endpoint, or a silent step that is not decoded): only `keep` samples carry over and
    // the step count toward the next fixed line end starts again. Row S69-ENGREV-WHISPERCPP-STREAMVAD-2026-10-27-01.
    void end_line();

    // stream time (seconds) of the current window: [t0, t1)
    double t0() const { return (double) (m_consumed - m_window.size()) / m_p.sample_rate; }
    double t1() const { return (double) m_consumed / m_p.sample_rate; }

    int n_step() const { return m_n_step; }
    int n_len()  const { return m_n_len; }
    int n_keep() const { return m_n_keep; }
    int n_new_line() const { return m_n_new_line; }

private:
    audio_stream_params m_p;
    int m_n_step = 0, m_n_len = 0, m_n_keep = 0, m_n_new_line = 1;
    int m_n_iter = 0;
    uint64_t m_consumed = 0;      // samples handed to next() so far
    std::vector<float> m_old;     // carried from the previous window
    std::vector<float> m_window;
};
