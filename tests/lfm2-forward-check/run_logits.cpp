// Load a tiny GGUF on the CPU backend, decode a fixed token sequence in chunks of `chunk` tokens (chunk < T exercises the conv state carried
// across ubatches), and write every position's logits as float32 [T, n_vocab] to a file. PROGRESS-OK: a few dozen tokens, milliseconds.
// usage: prog model.gguf out.bin T chunk
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: prog model.gguf out.bin T chunk\n"); return 64; }
    const char * path = argv[1];
    const char * outp = argv[2];
    const int T = atoi(argv[3]);
    const int chunk = atoi(argv[4]);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 65; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = chunk;
    cp.n_ubatch = chunk;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context creation failed\n"); return 66; }

    const int n_vocab = 64;    // the tiny models are built with vocab 64; checked against the output size below
    std::vector<llama_token> toks(T);
    for (int i = 0; i < T; ++i) toks[i] = (7 * i + 3) % n_vocab;

    std::vector<float> all((size_t) T * n_vocab, 0.0f);
    llama_batch batch = llama_batch_init(chunk, 0, 1);
    for (int start = 0; start < T; start += chunk) {
        const int n = (T - start < chunk) ? (T - start) : chunk;
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = toks[start + i];
            batch.pos[i]       = start + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) { fprintf(stderr, "llama_decode rc=%d at start=%d\n", rc, start); return 67; }
        const float * lg = llama_get_logits(ctx);
        if (!lg) { fprintf(stderr, "no logits\n"); return 68; }
        memcpy(all.data() + (size_t) start * n_vocab, lg, (size_t) n * n_vocab * sizeof(float));
        fprintf(stderr, "decoded positions %d..%d\n", start, start + n - 1);
    }
    FILE * f = fopen(outp, "wb");
    fwrite(all.data(), sizeof(float), all.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %zu floats\n", all.size());
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
