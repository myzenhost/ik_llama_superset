// Runtime proof for the qwen3a (Qwen3-ASR) projector port.
//
// Loads an mmproj GGUF whose clip.projector_type is "qwen3a", reports what the binary made of
// it, then pushes a mel spectrogram through the full encode path so the cgraph is actually
// built, allocated and executed.
//
// Exists because no real Qwen3-ASR GGUF+mmproj is present on this host (E:, F: and Z: swept
// 2026-08-31 -- see temp/research/findings/S69_FORK_AGENTS/asr_agent.md). This harness proves
// the port end to end against a synthetic mmproj with the real tensor names and real graph.

#include "clip.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s <mmproj.gguf> [n_frames] [n_mel] [expected_tokens]\n", argv[0]);
        return 1;
    }
    const char * fname    = argv[1];
    const int    n_frames = argc > 2 ? atoi(argv[2]) : 200;
    const int    n_mel_in = argc > 3 ? atoi(argv[3]) : 128;
    const int    exp_tok  = argc > 4 ? atoi(argv[4]) : -1; // -1 = do not check

    clip_context_params p;
    p.use_gpu           = false;
    p.verbosity         = GGML_LOG_LEVEL_INFO;
    p.flash_attn_type   = CLIP_FLASH_ATTN_TYPE_DISABLED;
    p.image_min_tokens  = -1;
    p.image_max_tokens  = -1;
    p.kq_type           = GGML_TYPE_F16;

    clip_init_result r = clip_init(fname, p);
    if (r.ctx_a == nullptr) {
        printf("\n[RESULT] FAIL: clip_init returned no audio context\n");
        return 1;
    }
    clip_ctx * ctx = r.ctx_a;

    printf("\n==== qwen3a runtime probe ====\n");
    printf("clip_has_audio_encoder   : %d\n", (int) clip_has_audio_encoder(ctx));
    printf("clip_has_vision_encoder  : %d\n", (int) clip_has_vision_encoder(ctx));
    printf("clip_has_whisper_encoder : %d\n", (int) clip_has_whisper_encoder(ctx));
    const int n_embd_proj = clip_n_mmproj_embd(ctx);
    printf("clip_n_mmproj_embd       : %d\n", n_embd_proj);

    // synthetic 128-bin mel, n_frames long (must be a multiple of the 100-frame chunk)
    const int n_mel = n_mel_in;
    std::vector<float> mel((size_t) n_mel * n_frames);
    for (size_t i = 0; i < mel.size(); i++) {
        mel[i] = 0.1f * sinf(0.01f * (float) i);
    }

    clip_image_f32_batch * batch = clip_image_f32_batch_init();
    clip_image_f32_batch_add_mel(batch, n_mel, n_frames, mel.data());

    clip_image_f32 * img  = clip_image_f32_get_img(batch, 0);
    const int        ntok = clip_n_output_tokens(ctx, img);
    printf("n_frames=%d (n_mel=%d) -> clip_n_output_tokens: %d",
           n_frames, n_mel, ntok);
    if (exp_tok >= 0) {
        printf(" (expected %d)", exp_tok);
    }
    printf("\n");

    std::vector<float> vec((size_t) ntok * n_embd_proj, 0.0f);
    const bool ok = clip_image_batch_encode(ctx, 4, batch, vec.data());
    printf("clip_image_batch_encode  : %s\n", ok ? "OK" : "FAILED");
    if (!ok) {
        clip_image_f32_batch_free(batch);
        clip_free(ctx);
        printf("\n[RESULT] FAIL: encode returned false\n");
        return 1;
    }

    int    n_bad = 0;
    double sum   = 0.0;
    for (float v : vec) {
        if (!std::isfinite(v)) {
            n_bad++;
        } else {
            sum += v;
        }
    }
    printf("output embd  : %d tokens x %d dims = %zu floats\n", ntok, n_embd_proj, vec.size());
    printf("non-finite   : %d\n", n_bad);
    printf("mean         : %.6f\n", sum / (double) vec.size());
    printf("first 6      : %.5f %.5f %.5f %.5f %.5f %.5f\n",
           vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]);

    clip_image_f32_batch_free(batch);
    clip_free(ctx);

    const bool pass = ok && n_bad == 0 && (exp_tok < 0 || ntok == exp_tok);
    printf("\n[RESULT] %s\n", pass ? "PASS: projector loaded, graph built, encode ran clean"
                                   : "FAIL");
    return pass ? 0 : 1;
}
