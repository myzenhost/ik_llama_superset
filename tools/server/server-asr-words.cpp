// server-asr-words.cpp -- see server-asr-words.h. Written independently; no code copied.
// PRIOR-ART-NONE: no server word-timestamp path exists on canonical or any branch (temp/qwork_qworker-fork29/dup4.out); the aligner is reused.
#include "server-asr-words.h"

#include "server-common.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "asr-align.h"
#include "asr-align-capture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace asrwords {

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

struct chunks_guard {
    mtmd_input_chunks * c = nullptr;
    ~chunks_guard() { if (c) { mtmd_input_chunks_free(c); } }
};

struct bitmap_guard {
    std::vector<mtmd_bitmap *> v;
    ~bitmap_guard() { for (auto * b : v) { if (b) { mtmd_bitmap_free(b); } } }
};

// Tokens the projector produces for `n` zero samples at `sr` (marker only, no text), or -1.
int audio_tokens_for(mtmd_context * mctx, const char * marker, int sr, size_t n) {
    std::vector<float> pcm(n, 0.0f);
    mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(pcm.size(), pcm.data());
    if (!bmp) {
        return -1;
    }
    chunks_guard cg;
    cg.c = mtmd_input_chunks_init();
    mtmd_input_text txt;
    txt.text          = marker;
    txt.text_len      = std::char_traits<char>::length(marker);
    txt.add_special   = false;
    txt.parse_special = true;
    const mtmd_bitmap * bms[1] = { bmp };
    int total = -1;
    if (mtmd_tokenize(mctx, cg.c, &txt, bms, 1) == 0) {
        total = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(cg.c); ++i) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(cg.c, i);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                total += (int) mtmd_input_chunk_get_n_tokens(ch);
            }
        }
    }
    mtmd_bitmap_free(bmp);
    (void) sr;
    return total;
}

} // namespace

struct engine::impl {
    llama_model *  model = nullptr;
    common_params  params;
    mtmd_context * mctx  = nullptr;
    llama_context * lctx = nullptr;
    int            n_ctx = 0;
    int            sr    = 0;
    double         spp   = 0.0;     // seconds per audio position
    bool           probed = false;
    std::string    probe_err;
    int            n_layer = 0;
    int            n_head  = 0;
    asr_attn_capture cap;

    ~impl() {
        free_ctx();
        if (mctx) {
            mtmd_free(mctx);
            mctx = nullptr;
        }
    }

    void free_ctx() {
        if (lctx) {
            llama_free(lctx);
            lctx = nullptr;
        }
        n_ctx = 0;
    }

    bool probe(std::string & err) {
        if (probed) {
            err = probe_err;
            return probe_err.empty();
        }
        probed = true;
        const std::string & path = params.mmproj.path;
        if (path.empty()) {
            probe_err = "word timestamps need an audio projector (--mmproj); this server has none";
            err = probe_err;
            return false;
        }
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu          = params.mmproj_use_gpu;
        mp.device           = params.mmproj_device;
        mp.print_timings    = false;
        mp.n_threads        = params.n_threads_mtmd > 0 ? params.n_threads_mtmd
                            : params.cpuparams_batch.n_threads > 0 ? params.cpuparams_batch.n_threads
                            : params.cpuparams.n_threads;
        // the attention rows are read from kq_soft_max, which a flash-attention graph never builds: the projector side is
        // built without flash attention too, and without the embedding cache (a replay must not hit the slots' cache entries)
        mp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        mp.warmup           = false;
        mp.image_min_tokens = params.image_min_tokens;
        mp.image_max_tokens = params.image_max_tokens;
        mp.batch_max_tokens = params.mtmd_batch_max_tokens;
        mp.embd_cache_bytes = 0;
        mp.media_marker     = get_media_marker();
        mctx = mtmd_init_from_file(path.c_str(), model, mp);
        if (!mctx) {
            probe_err = "word timestamps: the second audio projector could not be created (out of memory?)";
            err = probe_err;
            return false;
        }
        sr = mtmd_get_audio_sample_rate(mctx);
        if (sr <= 0) {
            probe_err = "word timestamps need an audio projector; the loaded projector has no audio input";
            err = probe_err;
            return false;
        }
        const char * marker = mtmd_default_marker();
        (void) marker;
        const char * mk = get_media_marker();
        const int na = audio_tokens_for(mctx, mk, sr, (size_t) sr * 10);
        const int nb = audio_tokens_for(mctx, mk, sr, (size_t) sr * 20);
        if (na <= 0 || nb <= 0) {
            probe_err = "word timestamps: could not measure the audio positions per second";
            err = probe_err;
            return false;
        }
        double chunk_s = 0.0;
        if (na == nb) {
            // fixed-chunk projector (whisper style): find the chunk length where the token count first grows
            double lo = 20.0, hi = 120.0;
            if (audio_tokens_for(mctx, mk, sr, (size_t) (hi * sr)) <= na) {
                chunk_s = 0.0;
            } else {
                while (hi - lo > 0.01) {
                    const double mid = 0.5 * (lo + hi);
                    if (audio_tokens_for(mctx, mk, sr, (size_t) (mid * sr)) > na) { hi = mid; } else { lo = mid; }
                }
                chunk_s = std::round(lo * 10.0) / 10.0;
            }
        }
        spp = asr_align_seconds_per_position(10.0, na, 20.0, nb, chunk_s);
        if (spp <= 0.0) {
            probe_err = "word timestamps: the projector's audio positions do not map to time (no proportional or fixed-chunk rule fits)";
            err = probe_err;
            return false;
        }
        n_layer = llama_model_n_layer(model);
        n_head  = llama_model_n_head(model);
        if (n_layer <= 0 || n_head <= 0) {
            probe_err = "word timestamps: the model reports no layers or heads";
            err = probe_err;
            return false;
        }
        return true;
    }

    bool ensure_ctx(int need, size_t & bytes, std::string & err) {
        if (lctx && n_ctx >= need) {
            return true;
        }
        free_ctx();
        int want = std::max(4096, ((need + 1023) / 1024) * 1024);
        common_params p = params;
        p.flash_attn        = false;
        p.attn_max_batch    = 0;
        p.n_parallel        = 1;
        p.n_ctx             = want;
        p.cb_eval           = asr_attn_capture_cb;
        p.cb_eval_user_data = &cap;
        llama_context_params cp = common_context_params_to_llama(p);
        lctx = llama_init_from_model(model, cp);
        if (!lctx) {
            err = "word timestamps: the second context could not be created (out of memory?)";
            return false;
        }
        n_ctx = want;
        // the --lora ARGUMENT list (common_params), the same list p copied from params; applied to the second context too
        if (!params.lora_adapters.empty()) {
            common_set_adapter_lora(lctx, params.lora_adapters);
        }
        // memory the context holds (KV + compute buffers), for the report
        const llama_context * ctxs[1] = { lctx };
        llama_memory_row rows[16];
        const int32_t nr = llama_memory_breakdown_rows(ctxs, 1, nullptr, 0, rows, 16);
        size_t sum = 0;
        for (int32_t i = 0; i < std::min<int32_t>(nr, 16); ++i) {
            sum += rows[i].context + rows[i].compute;
        }
        bytes = sum;
        return true;
    }
};

engine::engine(llama_model * model, const common_params & params) : p_(new impl) {
    p_->model  = model;
    p_->params = params;
}

engine::~engine() = default;

bool engine::ready(std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    return p_->probe(err);
}

bool engine::align(const std::string & prompt, const std::vector<std::vector<uint8_t>> & files, const std::string & transcript,
                   bool has_eog, std::vector<word> & out, timing * tm, std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    out.clear();
    timing local;
    timing & T = tm ? *tm : local;
    T = timing();

    double t0 = now_ms();
    if (!p_->probe(err)) {
        return false;
    }
    T.create_ms = now_ms() - t0;
    if (files.size() != 1) {
        err = "word timestamps support exactly one audio upload per request";
        return false;
    }

    // ---- tokenize the request exactly as the slot did --------------------------------------------------------------------
    t0 = now_ms();
    bitmap_guard bg;
    {
        auto w = mtmd_helper_bitmap_init_from_buf(p_->mctx, files[0].data(), files[0].size(), false, mtmd_helper_init_opt_default());
        if (!w.bitmap) {
            err = "word timestamps: the audio could not be decoded";
            return false;
        }
        bg.v.push_back(w.bitmap);
    }
    if (!mtmd_bitmap_is_audio(bg.v[0])) {
        err = "word timestamps need an audio upload";
        return false;
    }
    const size_t n_samples = mtmd_bitmap_get_n_bytes(bg.v[0]) / sizeof(float);
    const double dur_s     = (double) n_samples / (double) p_->sr;
    const float * pcm      = (const float *) mtmd_bitmap_get_data(bg.v[0]);

    chunks_guard cg;
    cg.c = mtmd_input_chunks_init();
    mtmd_input_text txt;
    txt.text          = prompt.c_str();
    txt.text_len      = prompt.size();
    txt.add_special   = true;
    txt.parse_special = true;
    std::vector<const mtmd_bitmap *> bms(bg.v.begin(), bg.v.end());
    if (mtmd_tokenize(p_->mctx, cg.c, &txt, bms.data(), bms.size()) != 0) {
        err = "word timestamps: the prompt could not be tokenized with the audio";
        return false;
    }

    // audio cells: [a0, a1), chunks walked from position 0
    int64_t cell = 0, a0 = -1, a1 = -1;
    for (size_t i = 0; i < mtmd_input_chunks_size(cg.c); ++i) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(cg.c, i);
        const int64_t n = (int64_t) mtmd_input_chunk_get_n_tokens(ch);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            if (a0 < 0) {
                a0 = cell;
            } else if (cell != a1) {
                err = "word timestamps: the audio chunks are not adjacent in the prompt";
                return false;
            }
            a1 = cell + n;
        }
        cell += n;
    }
    if (a0 < 0 || a1 <= a0) {
        err = "word timestamps: the prompt carries no audio positions";
        return false;
    }
    const int n_prompt = (int) cell;

    const llama_vocab * vocab = llama_model_get_vocab(p_->model);
    std::vector<llama_token> toks = common_tokenize(vocab, transcript, false, false);
    if (toks.empty()) {
        err = "word timestamps: the transcript has no tokens";
        return false;
    }
    T.tokenize_ms = now_ms() - t0;
    T.n_audio_pos = (int) (a1 - a0);
    T.n_tokens    = (int) toks.size();

    // ---- the second context ----------------------------------------------------------------------------------------------
    size_t bytes = 0;
    if (!p_->ensure_ctx(n_prompt + (int) toks.size() + 16, bytes, err)) {
        return false;
    }
    if (bytes) {
        ctx_bytes_ = bytes;
    }
    llama_context * lctx = p_->lctx;
    llama_memory_clear(llama_get_memory(lctx), true);

    asr_attn_capture & cap = p_->cap;
    cap.l0 = 0;
    cap.l1 = p_->n_layer - 1;
    cap.n_head = p_->n_head;
    if (!cap.arm(a0, a1)) {
        err = "word timestamps: the attention capture could not be armed";
        return false;
    }
    T.n_heads = cap.n_sel() * cap.n_head;

    // ---- pass 1: prompt + audio ------------------------------------------------------------------------------------------
    t0 = now_ms();
    llama_pos n_past = 0;
    const int32_t n_batch = (int32_t) llama_n_batch(lctx);
    if (mtmd_helper_eval_chunks(p_->mctx, lctx, cg.c, 0, 0, n_batch, true, &n_past) != 0) {
        cap.armed = false;
        err = "word timestamps: the prompt could not be evaluated on the second context";
        return false;
    }
    T.prefill_ms = now_ms() - t0;
    std::vector<float> row0;
    if (!cap.take_row0_and_begin_all_rows(row0)) {
        cap.armed = false;
        err = "word timestamps: " + (cap.err.empty() ? std::string("no attention rows were captured") : cap.err);
        return false;
    }

    // ---- pass 2: the transcript, teacher-forced; logits on the last token only ---------------------------------------------
    t0 = now_ms();
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    bool ok = true;
    for (size_t i = 0; i < toks.size() && ok; ) {
        const size_t n = std::min<size_t>((size_t) n_batch, toks.size() - i);
        common_batch_clear(batch);
        for (size_t j = 0; j < n; ++j) {
            common_batch_add(batch, toks[i + j], (llama_pos) (n_past + (llama_pos) (i + j)), { 0 }, (i + j + 1 == toks.size()));
        }
        if (llama_decode(lctx, batch) != 0) {
            ok = false;
        }
        i += n;
    }
    llama_batch_free(batch);
    T.forced_ms = now_ms() - t0;
    if (!ok) {
        cap.armed = false;
        err = "word timestamps: the transcript could not be evaluated on the second context";
        return false;
    }
    std::vector<float> tok_rows;
    if (!cap.take_all_rows((int) toks.size(), tok_rows)) {
        err = "word timestamps: " + (cap.err.empty() ? std::string("attention rows are missing") : cap.err);
        return false;
    }

    // ---- align -----------------------------------------------------------------------------------------------------------
    t0 = now_ms();
    const int n_pos    = cap.n_audio();
    const int n_frames = asr_align_valid_frames(n_pos, dur_s, p_->spp);
    asr_align_input in;
    in.n_heads       = T.n_heads;
    in.n_frames      = n_frames;
    in.has_eog_row   = has_eog;
    in.sec_per_frame = p_->spp;
    in.t_offset      = 0.0;
    in.t_end         = dur_s;
    in.medfilt_width = 7;
    in.attn          = asr_attn_assemble(row0, tok_rows, (int) toks.size(), has_eog, T.n_heads, n_pos, n_frames);
    if (in.attn.empty()) {
        err = "word timestamps: the attention rows could not be assembled";
        return false;
    }
    for (llama_token id : toks) {
        asr_align_token t;
        t.id    = id;
        t.piece = common_token_to_piece(lctx, id);
        in.tokens.push_back(std::move(t));
    }
    in.speech_w = asr_align_speech_weights(pcm, n_samples, p_->sr, p_->spp, n_frames);
    asr_align_result res;
    std::string aerr;
    if (!asr_align_words(in, res, &aerr)) {
        err = "word timestamps: the aligner failed: " + aerr;
        return false;
    }
    T.align_ms = now_ms() - t0;
    for (const auto & w : res.words) {
        out.push_back({ w.word, w.start, w.end });
    }
    return true;
}

} // namespace asrwords
