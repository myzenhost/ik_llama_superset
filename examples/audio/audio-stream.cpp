// audio-stream.cpp -- see audio-stream.h

#include "audio-stream.h"

#include <algorithm>
#include <cstring>

audio_ring::audio_ring(size_t capacity) : m_buf(std::max<size_t>(1, capacity), 0.0f) {}

void audio_ring::push(const float * x, size_t n) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_total   += n;
    m_pending += n;
    const size_t cap = m_buf.size();
    if (n > cap) {          // only the newest `cap` samples can survive
        x += n - cap;
        n  = cap;
    }
    const size_t n0 = std::min(n, cap - m_pos);
    std::memcpy(m_buf.data() + m_pos, x, n0 * sizeof(float));
    if (n > n0) {
        std::memcpy(m_buf.data(), x + n0, (n - n0) * sizeof(float));
    }
    m_pos = (m_pos + n) % cap;
}

size_t audio_ring::take(std::vector<float> & out) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t cap  = m_buf.size();
    const size_t had  = m_pending;
    const size_t n    = std::min(had, cap);
    out.resize(n);
    const size_t s0 = (m_pos + cap - n) % cap;   // oldest of the n newest samples
    const size_t n0 = std::min(n, cap - s0);
    std::memcpy(out.data(), m_buf.data() + s0, n0 * sizeof(float));
    if (n > n0) {
        std::memcpy(out.data() + n0, m_buf.data(), (n - n0) * sizeof(float));
    }
    m_pending = 0;
    return had;
}

size_t audio_ring::pending() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_pending;
}

uint64_t audio_ring::total() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_total;
}

audio_stream_window::audio_stream_window(const audio_stream_params & p) : m_p(p) {
    m_p.step_ms   = std::max(1, m_p.step_ms);
    m_p.keep_ms   = std::max(0, std::min(m_p.keep_ms, m_p.step_ms));
    m_p.length_ms = std::max(m_p.length_ms, m_p.step_ms);
    m_n_step      = (int) ((int64_t) m_p.step_ms   * m_p.sample_rate / 1000);
    m_n_len       = (int) ((int64_t) m_p.length_ms * m_p.sample_rate / 1000);
    m_n_keep      = (int) ((int64_t) m_p.keep_ms   * m_p.sample_rate / 1000);
    m_n_new_line  = std::max(1, m_p.length_ms / m_p.step_ms - 1);
}

const std::vector<float> & audio_stream_window::next(const std::vector<float> & fresh) {
    const int n_new  = (int) fresh.size();
    const int n_take = std::min((int) m_old.size(), std::max(0, m_n_keep + m_n_len - n_new));
    m_window.resize((size_t) n_take + fresh.size());
    std::copy(m_old.end() - n_take, m_old.end(), m_window.begin());
    std::copy(fresh.begin(), fresh.end(), m_window.begin() + n_take);
    m_old = m_window;
    m_consumed += fresh.size();
    return m_window;
}

bool audio_stream_window::done_step() {
    ++m_n_iter;
    if (m_n_iter % m_n_new_line != 0) {
        return false;
    }
    const size_t k = std::min(m_window.size(), (size_t) m_n_keep);
    m_old.assign(m_window.end() - k, m_window.end());
    return true;
}

void audio_stream_window::end_line() {
    m_n_iter = 0;
    const size_t k = std::min(m_window.size(), (size_t) m_n_keep);
    m_old.assign(m_window.end() - k, m_window.end());
}
