// mtmd-audio-encoder.cpp -- the audio encoder of an mmproj without a text model (see mtmd-audio-encoder.h)

#include "mtmd-audio-encoder.h"

#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "mtmd-audio.h"

#include <exception>
#include <memory>

struct mtmd_audio_encoder {
    clip_ctx * ctx_a = nullptr;
    clip_ctx * ctx_v = nullptr; // freed with the encoder when a file carries both
    std::unique_ptr<mtmd_audio_preprocessor> preproc;
    int n_threads = 4;
};

mtmd_audio_encoder * mtmd_audio_encoder_init(const char * fname, const mtmd_audio_encoder_params & params, std::string * err) {
    auto fail = [&](const std::string & why) -> mtmd_audio_encoder * {
        if (err) { *err = why; }
        return nullptr;
    };
    clip_context_params cp {
        /* use_gpu           */ params.use_gpu,
        /* device            */ nullptr,
        /* flash_attn_type   */ CLIP_FLASH_ATTN_TYPE_AUTO,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* warmup            */ params.warmup,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
        /* no_alloc          */ false,
        /* progress_callback */ nullptr,
        /* progress_callback_user_data */ nullptr,
    };
    clip_init_result res {};
    try {
        res = clip_init(fname, cp);
    } catch (const std::exception & e) {
        return fail(std::string("clip_init: ") + e.what());
    }
    auto enc = std::make_unique<mtmd_audio_encoder>();
    enc->ctx_a     = res.ctx_a;
    enc->ctx_v     = res.ctx_v;
    enc->n_threads = params.n_threads > 0 ? params.n_threads : 1;
    if (res.ctx_gen_a) { clip_free(res.ctx_gen_a); }
    if (!enc->ctx_a) {
        if (enc->ctx_v) { clip_free(enc->ctx_v); }
        return fail(std::string(fname) + ": no audio encoder in this file");
    }
    const projector_type pt = clip_get_projector_type(enc->ctx_a);
    switch (pt) {
        case PROJECTOR_TYPE_PARAKEET:
            enc->preproc = std::make_unique<mtmd_audio_preprocessor_parakeet>(enc->ctx_a);
            break;
        default:
            clip_free(enc->ctx_a);
            if (enc->ctx_v) { clip_free(enc->ctx_v); }
            return fail(std::string(fname) + ": audio projector '" + PROJECTOR_TYPE_NAMES[pt] +
                        "' has no encoder-only route (supported: parakeet)");
    }
    enc->preproc->initialize();
    return enc.release();
}

void mtmd_audio_encoder_free(mtmd_audio_encoder * enc) {
    if (!enc) { return; }
    if (enc->ctx_a) { clip_free(enc->ctx_a); }
    if (enc->ctx_v) { clip_free(enc->ctx_v); }
    delete enc;
}

int mtmd_audio_encoder_n_embd(const mtmd_audio_encoder * enc)      { return clip_n_mmproj_embd(enc->ctx_a); }
int mtmd_audio_encoder_sample_rate(const mtmd_audio_encoder * enc) { return clip_get_hparams(enc->ctx_a)->audio_sample_rate; }
int mtmd_audio_encoder_hop(const mtmd_audio_encoder * enc)         { return clip_get_hparams(enc->ctx_a)->audio_hop_len; }
int mtmd_audio_encoder_subsampling(const mtmd_audio_encoder * enc) {
    const int f = (int) clip_get_hparams(enc->ctx_a)->subsampling_factor;
    return f > 0 ? f : 1;
}

bool mtmd_audio_encoder_encode(mtmd_audio_encoder * enc, const float * samples, size_t n_samples,
                               std::vector<float> & out, int & n_frames) {
    out.clear();
    n_frames = 0;
    if (!enc || !samples || n_samples == 0) {
        return false;
    }
    std::vector<mtmd_audio_mel> mels;
    if (!enc->preproc->preprocess(samples, n_samples, mels) || mels.empty()) {
        return false;
    }
    const int n_embd = clip_n_mmproj_embd(enc->ctx_a);
    for (const auto & mel : mels) {
        clip_image_f32 img;
        img.set_size({(int) mel.n_len, (int) mel.n_mel}, /* is_placeholder */ false, /* is_audio */ true);
        img.cpy_buf(mel.data);
        const int n_tok = clip_n_output_tokens(enc->ctx_a, &img);
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(img));
        // clip copies the output only into a buffer of the exact size (clip.cpp "output buffer is empty, skipping copy")
        std::vector<float> part((size_t) n_tok * (size_t) n_embd);
        if (!clip_image_batch_encode(enc->ctx_a, enc->n_threads, &batch, part)) {
            return false;
        }
        if (part.size() != (size_t) n_tok * (size_t) n_embd) {
            return false;
        }
        out.insert(out.end(), part.begin(), part.end());
        n_frames += n_tok;
    }
    return true;
}
