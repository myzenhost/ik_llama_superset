#include "asr-vad-trim.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

// index of the first piece whose out_end is > x, or pieces.size()
size_t first_piece_ending_after(const std::vector<asr_vad_trim_piece> & pc, double x) {
    size_t lo = 0, hi = pc.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if ((double) pc[mid].out_end > x) { hi = mid; } else { lo = mid + 1; }
    }
    return lo;
}

// index + 1 of the last piece whose out_start is < x, or 0
size_t n_pieces_starting_before(const std::vector<asr_vad_trim_piece> & pc, double x) {
    size_t lo = 0, hi = pc.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if ((double) pc[mid].out_start < x) { lo = mid + 1; } else { hi = mid; }
    }
    return lo;
}

void put_u32(std::vector<uint8_t> & o, uint32_t v) {
    for (int i = 0; i < 4; ++i) { o.push_back((uint8_t) ((v >> (8 * i)) & 0xff)); }
}

void put_u16(std::vector<uint8_t> & o, uint16_t v) {
    for (int i = 0; i < 2; ++i) { o.push_back((uint8_t) ((v >> (8 * i)) & 0xff)); }
}

} // namespace

double asr_vad_trim_map::start_to_original(double t) const {
    if (pieces.empty() || sample_rate <= 0) {
        return t;
    }
    const double sr = (double) sample_rate;
    const double x  = t * sr;
    const size_t i  = first_piece_ending_after(pieces, x);
    if (i == pieces.size()) {
        return (double) pieces.back().in_end() / sr;        // at or past the end of the trimmed signal
    }
    const auto & pc = pieces[i];
    if (x <= (double) pc.out_start) {
        return (double) pc.in_start / sr;                   // on a join or in the pad before this piece
    }
    return ((double) pc.in_start + (x - (double) pc.out_start)) / sr;
}

double asr_vad_trim_map::end_to_original(double t) const {
    if (pieces.empty() || sample_rate <= 0) {
        return t;
    }
    const double sr = (double) sample_rate;
    const double x  = t * sr;
    const size_t k  = n_pieces_starting_before(pieces, x);
    if (k == 0) {
        return (double) pieces.front().in_start / sr;       // an end at or before the first kept sample
    }
    const auto & pc = pieces[k - 1];
    if (x >= (double) pc.out_end) {
        return (double) pc.in_end() / sr;                   // on a join or in the pad after this piece
    }
    return ((double) pc.in_start + (x - (double) pc.out_start)) / sr;
}

void asr_vad_trim_map::segment_to_original(double & t0, double & t1) const {
    if (pieces.empty() || sample_rate <= 0) {
        return;
    }
    const double o0 = start_to_original(t0);
    const double o1 = end_to_original(t1);
    t0 = o0;
    t1 = std::max(o0, o1);
}

void asr_vad_trim_map::word_to_original(double & t0, double & t1) const {
    if (pieces.empty() || sample_rate <= 0) {
        return;
    }
    const double sr  = (double) sample_rate;
    const double mid = 0.5 * (t0 + t1) * sr;
    size_t i = first_piece_ending_after(pieces, mid);
    if (i == pieces.size()) {
        i = pieces.size() - 1;
    } else if (mid < (double) pieces[i].out_start && i > 0) {
        // the midpoint is in the pad between piece i-1 and piece i: the nearer one, the earlier one on a tie
        const double d_prev = mid - (double) pieces[i - 1].out_end;
        const double d_next = (double) pieces[i].out_start - mid;
        if (d_prev <= d_next) {
            i -= 1;
        }
    }
    const auto & pc = pieces[i];
    const double x0 = std::min(std::max(t0 * sr, (double) pc.out_start), (double) pc.out_end);
    const double x1 = std::min(std::max(t1 * sr, (double) pc.out_start), (double) pc.out_end);
    t0 = ((double) pc.in_start + (x0 - (double) pc.out_start)) / sr;
    t1 = ((double) pc.in_start + (std::max(x0, x1) - (double) pc.out_start)) / sr;
}

std::vector<float> asr_vad_trim(const float * pcm, size_t n_samples, int32_t sample_rate,
                                const std::vector<asr_vad_span> & spans_16k, const asr_vad_trim_params & p,
                                asr_vad_trim_map & map) {
    map = asr_vad_trim_map();
    map.n_in        = (int64_t) n_samples;
    map.sample_rate = sample_rate;
    std::vector<float> out;
    if (pcm == nullptr || n_samples == 0 || sample_rate <= 0) {
        return out;
    }
    const int64_t n   = (int64_t) n_samples;
    const int64_t pad = std::max<int64_t>(0, (int64_t) p.pad_ms) * (int64_t) sample_rate / 1000;

    // spans at the clip rate, ordered by start
    std::vector<asr_vad_span> sp;
    sp.reserve(spans_16k.size());
    for (const auto & s : spans_16k) {
        asr_vad_span c;
        c.start = (int64_t) std::floor((double) s.start * (double) sample_rate / 16000.0);
        c.end   = (int64_t) std::ceil ((double) s.end   * (double) sample_rate / 16000.0);
        c.start = std::min(std::max<int64_t>(c.start, 0), n);
        c.end   = std::min(std::max<int64_t>(c.end,   0), n);
        if (c.end > c.start) {
            sp.push_back(c);
        }
    }
    std::sort(sp.begin(), sp.end(), [](const asr_vad_span & a, const asr_vad_span & b) {
        return a.start != b.start ? a.start < b.start : a.end < b.end;
    });

    // merge overlapping spans and spans whose gap is no longer than the pad: [in_start, in_end) runs to keep
    std::vector<asr_vad_span> keep;
    for (const auto & c : sp) {
        if (!keep.empty() && c.start - keep.back().end <= pad) {
            keep.back().end = std::max(keep.back().end, c.end);
        } else {
            keep.push_back(c);
        }
    }

    size_t total = 0;
    for (const auto & k : keep) {
        total += (size_t) (k.end - k.start);
    }
    total += keep.empty() ? 0 : (size_t) pad * (keep.size() - 1);
    out.reserve(total);
    for (size_t i = 0; i < keep.size(); ++i) {
        if (i > 0) {
            out.insert(out.end(), (size_t) pad, 0.0f);
        }
        asr_vad_trim_piece pc;
        pc.out_start = (int64_t) out.size();
        pc.in_start  = keep[i].start;
        out.insert(out.end(), pcm + keep[i].start, pcm + keep[i].end);
        pc.out_end   = (int64_t) out.size();
        map.pieces.push_back(pc);
    }
    map.n_out = (int64_t) out.size();
    return out;
}

std::vector<uint8_t> asr_wav_f32_encode(const float * pcm, size_t n_samples, int32_t sample_rate) {
    std::vector<uint8_t> o;
    const uint32_t data_bytes = (uint32_t) (n_samples * 4);
    o.reserve(58 + (size_t) data_bytes);
    // RIFF header, a non-PCM fmt chunk (18 bytes, cbSize 0) and the fact chunk the WAVE spec asks for with it
    o.insert(o.end(), {'R', 'I', 'F', 'F'});
    put_u32(o, 4 + (8 + 18) + (8 + 4) + (8 + data_bytes));
    o.insert(o.end(), {'W', 'A', 'V', 'E'});
    o.insert(o.end(), {'f', 'm', 't', ' '});
    put_u32(o, 18);
    put_u16(o, 3);                                  // WAVE_FORMAT_IEEE_FLOAT
    put_u16(o, 1);                                  // mono
    put_u32(o, (uint32_t) sample_rate);
    put_u32(o, (uint32_t) sample_rate * 4);         // bytes per second
    put_u16(o, 4);                                  // block align
    put_u16(o, 32);                                 // bits per sample
    put_u16(o, 0);                                  // cbSize
    o.insert(o.end(), {'f', 'a', 'c', 't'});
    put_u32(o, 4);
    put_u32(o, (uint32_t) n_samples);
    o.insert(o.end(), {'d', 'a', 't', 'a'});
    put_u32(o, data_bytes);
    const size_t at = o.size();
    o.resize(at + (size_t) data_bytes);
    if (n_samples > 0 && pcm != nullptr) {
        // IEEE-754 little-endian floats, the byte order of every host this builds for
        std::memcpy(o.data() + at, pcm, (size_t) data_bytes);
    }
    return o;
}
