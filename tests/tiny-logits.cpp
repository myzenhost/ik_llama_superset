// tiny-logits: load a (random-weight) GGUF through the real llama library, decode a fixed token sequence in ubatch-sized
// chunks (the same llama_decode path llama-cli/server/perplexity use), optionally greedy-generate, and write the logits.
//   tiny-logits -m model.gguf [-n 48] [-ub 8] [-gen 0] [-fa 0|1] [-ctk f16] [-ctv f16] [--kivi-bits N --kivi-group G --kivi-residual R]
//               [-o logits.bin]
// stdout: "LOGITS n_pos=.. n_vocab=.. rms=.. sum=.. argmax_last=.." and, with -gen, "GEN: t t t ..."; the library's own log goes to stderr.
// Used by tests/test-wiring-tiny.py (MX types, KIVI, ...). Not a unit test on its own.
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_type parse_type(const std::string & s) {
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    std::fprintf(stderr, "bad cache type %s\n", s.c_str());
    std::exit(2);
}

int main(int argc, char ** argv) {
    std::string path, out, ctk = "f16", ctv = "f16", toks_arg;
    int n = 48, ub = 8, gen = 0, fa = 1, kb = 0, kg = 32, kr = 128, ctx = 512;
    bool dsa = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "-m") path = nx();
        else if (a == "-o") out = nx();
        else if (a == "-n") n = std::stoi(nx());
        else if (a == "-ub") ub = std::stoi(nx());
        else if (a == "-gen") gen = std::stoi(nx());
        else if (a == "-fa") fa = std::stoi(nx());
        else if (a == "-ctk") ctk = nx();
        else if (a == "-ctv") ctv = nx();
        else if (a == "-c") ctx = std::stoi(nx());
        else if (a == "-dsa") dsa = std::stoi(nx()) != 0; // cparams.dsa: the fork's glm5next DSA k-pool indexer is OPT-IN (off = dense MLA)
        else if (a == "-t") toks_arg = nx(); // comma-separated token ids; replaces the synthetic sequence (n = count)
        else if (a == "--kivi-bits") kb = std::stoi(nx());
        else if (a == "--kivi-group") kg = std::stoi(nx());
        else if (a == "--kivi-residual") kr = std::stoi(nx());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (path.empty()) { std::fprintf(stderr, "need -m\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 3; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctx; cp.n_batch = ub; cp.n_ubatch = ub; cp.flash_attn = fa != 0;
    cp.type_k = parse_type(ctk); cp.type_v = parse_type(ctv);
    cp.dsa = dsa;
    cp.kivi_bits = kb; cp.kivi_group = kg; cp.kivi_residual = kr;
    llama_context * lctx = nullptr;
    try {
        lctx = llama_init_from_model(model, cp);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "context creation threw: %s\n", e.what());
    }
    if (!lctx) { std::printf("CONTEXT_FAILED\n"); llama_model_free(model); return 4; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> toks(n);
    for (int i = 0; i < n; i++) toks[i] = 3 + (i * 37 + 5) % (n_vocab - 3);
    if (!toks_arg.empty()) {
        toks.clear();
        size_t p = 0;
        while (p < toks_arg.size()) {
            size_t q = toks_arg.find(',', p);
            if (q == std::string::npos) q = toks_arg.size();
            toks.push_back((llama_token) std::stoi(toks_arg.substr(p, q - p)));
            p = q + 1;
        }
        n = (int) toks.size();
    }

    std::vector<float> all((size_t) (n + gen) * n_vocab, 0.f);
    int done = 0;
    while (done < n) {
        const int c = std::min(ub, n - done);
        llama_batch b = llama_batch_init(c, 0, 1);
        b.n_tokens = c;
        for (int j = 0; j < c; j++) { b.token[j] = toks[done + j]; b.pos[j] = done + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1; }
        if (llama_decode(lctx, b) != 0) { std::printf("DECODE_FAILED at %d\n", done); return 5; }
        for (int j = 0; j < c; j++) std::memcpy(&all[(size_t) (done + j) * n_vocab], llama_get_logits_ith(lctx, j), n_vocab * sizeof(float));
        llama_batch_free(b);
        done += c;
    }
    std::string gen_s;
    int n_pos = n;
    int last = 0;
    { // argmax of last position
        const float * l = &all[(size_t) (n - 1) * n_vocab];
        for (int v = 1; v < n_vocab; v++) if (l[v] > l[last]) last = v;
    }
    int cur = last;
    for (int g = 0; g < gen; g++) {
        gen_s += std::to_string(cur) + " ";
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1; b.token[0] = cur; b.pos[0] = n + g; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        if (llama_decode(lctx, b) != 0) { std::printf("DECODE_FAILED gen %d\n", g); return 5; }
        const float * l = llama_get_logits_ith(lctx, 0);
        std::memcpy(&all[(size_t) (n + g) * n_vocab], l, n_vocab * sizeof(float));
        int am = 0; for (int v = 1; v < n_vocab; v++) if (l[v] > l[am]) am = v;
        cur = am;
        llama_batch_free(b);
        n_pos++;
    }
    double s2 = 0, s1 = 0;
    for (size_t i = 0; i < (size_t) n_pos * n_vocab; i++) { s2 += (double) all[i] * all[i]; s1 += all[i]; }
    std::printf("LOGITS n_pos=%d n_vocab=%d rms=%.6f sum=%.6f argmax_last=%d\n", n_pos, n_vocab, std::sqrt(s2 / ((double) n_pos * n_vocab)), s1, last);
    if (gen > 0) std::printf("GEN: %s\n", gen_s.c_str());
    if (!out.empty()) {
        FILE * f = std::fopen(out.c_str(), "wb");
        std::fwrite(all.data(), sizeof(float), (size_t) n_pos * n_vocab, f);
        std::fclose(f);
    }
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
