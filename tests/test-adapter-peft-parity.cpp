// Logits dumper for the PEFT parity harness (tests/test-adapter-peft-parity.py, row S69-ADAPTERS-PEFT-PARITY-HARNESS-2026-10-26-01).
// The Python driver builds a random-weight HF model + PEFT adapters, converts both with this repo's converters, computes the PEFT fp32 logits
// and runs this binary on the SAME token ids; it then diffs the two logit arrays. This binary does no checking of its own: it only has to
// decode exactly what it is told and write every position's logits.
//
// usage: test-adapter-peft-parity MODEL.gguf TOKENS.txt OUT.bin [--lora PATH SCALE]... [--seq-lora SEQ PATH SCALE]... [--batch]
//   TOKENS.txt   one sequence per line, space-separated token ids (no BOS is added: the ids are used verbatim)
//   --lora       context-wide adapter (llama_set_adapters_lora), any number
//   --seq-lora   per-sequence adapter (llama_set_seq_adapters_lora) for sequence index SEQ (0-based line of TOKENS.txt)
//   --draft-start SEQ POS  llama_set_draft_start: positions >= POS of sequence SEQ are draft rows (adapter.lora.gate = draft_rows)
//   --batch      decode every sequence in ONE batch (sequence i = seq id i); default decodes each sequence alone after a memory clear
//   --vocab-subset STRIDE  llama_set_output_vocab_subset on the ids t with t % STRIDE == 0 (row
//                S69-ENGREV-PEFT-LORA-FASTPATH-GUARD-LEGACY-MAP-ONLY-2026-10-27-01); exits non-zero unless the restricted projection ran
//                (used == |S|, every other entry -inf); OUT.bin then holds only the |S| subset columns (n_vocab field = |S|)
// OUT.bin: int32 n_seq, int32 n_vocab, then per sequence int32 n_tok followed by n_tok * n_vocab float32 logits (position-major).
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct lora_arg {
    std::string path;
    float       scale;
    int         seq; // -1 = context-wide
};

static bool decode_seqs(llama_context * ctx, const std::vector<std::vector<llama_token>> & seqs, const std::vector<int> & idx,
                        int n_vocab, std::vector<std::vector<float>> & out) {
    llama_memory_clear(llama_get_memory(ctx), true);
    int n = 0;
    for (int i : idx) n += (int) seqs[i].size();
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = 0;
    std::vector<int> first(idx.size());
    for (size_t s = 0; s < idx.size(); ++s) {
        first[s] = b.n_tokens;
        const auto & t = seqs[idx[s]];
        for (size_t i = 0; i < t.size(); ++i) {
            const int k = b.n_tokens++;
            b.token[k]     = t[i];
            b.pos[k]       = (llama_pos) i;
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = (llama_seq_id) idx[s];
            b.logits[k]    = 1;
        }
    }
    const int rc = llama_decode(ctx, b);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed rc=%d\n", rc);
        llama_batch_free(b);
        return false;
    }
    for (size_t s = 0; s < idx.size(); ++s) {
        auto & o = out[idx[s]];
        o.clear();
        for (size_t i = 0; i < seqs[idx[s]].size(); ++i) {
            const float * l = llama_get_logits_ith(ctx, first[s] + (int) i);
            o.insert(o.end(), l, l + n_vocab);
        }
    }
    llama_batch_free(b);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.txt OUT.bin [--lora PATH SCALE]... [--seq-lora SEQ PATH SCALE]... [--batch]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * tok_path   = argv[2];
    const char * out_path   = argv[3];
    std::vector<lora_arg> loras;
    bool batch = false;
    int  vocab_stride = 0; // >= 2: llama_set_output_vocab_subset on the ids t with t % stride == 0, and OUT.bin holds only those columns
    std::vector<std::pair<int, int>> draft_starts; // (sequence index, first draft position): llama_set_draft_start
    for (int i = 4; i < argc; ++i) {
        if (!strcmp(argv[i], "--lora") && i + 2 < argc) {
            loras.push_back({ argv[i + 1], (float) atof(argv[i + 2]), -1 });
            i += 2;
        } else if (!strcmp(argv[i], "--seq-lora") && i + 3 < argc) {
            loras.push_back({ argv[i + 2], (float) atof(argv[i + 3]), atoi(argv[i + 1]) });
            i += 3;
        } else if (!strcmp(argv[i], "--draft-start") && i + 2 < argc) {
            draft_starts.push_back({ atoi(argv[i + 1]), atoi(argv[i + 2]) });
            i += 2;
        } else if (!strcmp(argv[i], "--vocab-subset") && i + 1 < argc) {
            vocab_stride = atoi(argv[i + 1]);
            if (vocab_stride < 2) { fprintf(stderr, "--vocab-subset STRIDE needs STRIDE >= 2\n"); return 2; }
            i += 1;
        } else if (!strcmp(argv[i], "--batch")) {
            batch = true;
        } else {
            fprintf(stderr, "unknown or incomplete argument '%s'\n", argv[i]);
            return 2;
        }
    }

    std::vector<std::vector<llama_token>> seqs;
    {
        std::ifstream f(tok_path);
        if (!f) { fprintf(stderr, "cannot open %s\n", tok_path); return 2; }
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::vector<llama_token> t;
            long v;
            while (ss >> v) t.push_back((llama_token) v);
            if (!t.empty()) seqs.push_back(t);
        }
    }
    if (seqs.empty()) { fprintf(stderr, "no sequences in %s\n", tok_path); return 2; }
    int max_len = 0, tot = 0;
    for (auto & s : seqs) { max_len = std::max(max_len, (int) s.size()); tot += (int) s.size(); }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path); return 2; }
    llama_context_params cp = llama_context_default_params();
    const int n_need = batch ? tot : max_len;
    cp.n_ctx = (uint32_t) std::max(256, ((n_need + 255) / 256) * 256);
    cp.n_batch = cp.n_ubatch = (uint32_t) std::max(64, n_need);
    cp.n_seq_max = (uint32_t) std::max<size_t>(1, seqs.size());
    cp.n_threads = cp.n_threads_batch = 4;
    // F32 KV cache: the reference is PEFT fp32, and the default F16 cache alone puts ~5e-3 into the logits (measured).
    cp.type_k = cp.type_v = cp.idx_type_k = GGML_TYPE_F32;
    cp.type_k_first = cp.type_k_last = cp.type_v_first = cp.type_v_last = GGML_TYPE_F32;
    // CPU math only. n_gpu_layers = 0 keeps the weights on the host, but in a CUDA build the scheduler still
    // offloads host-weight ops to the device (op_offload defaults to true, llama-context.cpp:5446 -> sched_new
    // at :1898), and the device matmul is not fp32-exact. MEASURED 2026-10-27, b1019g CUDA build: the base model
    // with no adapter missed PEFT fp32 by 3.926e-03 (TOL 1e-4) with "28 offloaded at 64 tokens" in the log, and
    // 37 of 46 cases missed. The reference is fp32, so the fork side must be too.
    // DISABLE-REVERSIBLE: PARITY_OP_OFFLOAD=1 keeps the library default (device offload) for a CUDA arm, which
    // needs its own tolerance (row S69-ADAPTERS-W2-PEFT-PARITY-CUDA-ARM-TOLERANCE-2026-10-27-01).
    const char * oo = getenv("PARITY_OP_OFFLOAD");
    if (!(oo && atoi(oo))) {
        cp.op_offload  = false;
        cp.offload_kqv = false;
    }
    // DISABLE-REVERSIBLE: PARITY_FA env, a test-only flash-attention override; unset keeps the library default.
    const char * fa = getenv("PARITY_FA");
    if (fa) cp.flash_attn_type = atoi(fa) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create context\n"); return 2; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::map<std::string, llama_adapter_lora *> loaded;
    std::vector<llama_adapter_lora *> g_ptr;
    std::vector<float> g_scale;
    std::map<int, std::pair<std::vector<llama_adapter_lora *>, std::vector<float>>> per_seq;
    for (auto & la : loras) {
        llama_adapter_lora *& a = loaded[la.path];
        if (!a) {
            a = llama_adapter_lora_init(model, la.path.c_str());
            if (!a) { fprintf(stderr, "cannot load adapter %s\n", la.path.c_str()); return 3; }
        }
        if (la.seq < 0) { g_ptr.push_back(a); g_scale.push_back(la.scale); }
        else { per_seq[la.seq].first.push_back(a); per_seq[la.seq].second.push_back(la.scale); }
    }
    if (!g_ptr.empty()) llama_set_adapters_lora(ctx, g_ptr.data(), g_ptr.size(), g_scale.data());
    for (auto & ps : per_seq) {
        llama_set_seq_adapters_lora(ctx, ps.first, ps.second.first.data(), ps.second.first.size(), ps.second.second.data());
    }

    for (auto & ds : draft_starts) {
        // sequence i decodes as seq id i in a --batch run and as seq id 0 alone otherwise
        llama_set_draft_start(ctx, batch ? ds.first : 0, ds.second);
    }
    std::vector<llama_token> sub_ids;
    if (vocab_stride >= 2) {
        for (int t = 0; t < n_vocab; t += vocab_stride) sub_ids.push_back((llama_token) t);
        if (!llama_set_output_vocab_subset(ctx, sub_ids.data(), (int32_t) sub_ids.size())) {
            fprintf(stderr, "llama_set_output_vocab_subset refused the subset\n");
            return 5;
        }
    }
    std::vector<std::vector<float>> out(seqs.size());
    if (batch) {
        std::vector<int> idx(seqs.size());
        for (size_t i = 0; i < seqs.size(); ++i) idx[i] = (int) i;
        if (!decode_seqs(ctx, seqs, idx, n_vocab, out)) return 4;
    } else {
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (!decode_seqs(ctx, seqs, { (int) i }, n_vocab, out)) return 4;
        }
    }

    if (!sub_ids.empty()) {
        // the restricted projection must really have run (the fast path under test): used == |S|, every other entry -INFINITY.
        // Then gather the subset columns, so OUT.bin has n_vocab = |S| and the driver compares them with the full-vocab reference.
        // With an adapter attached the fast path must DECLINE (build_output_mm: llm_lora_targets) and the full LoRA projection runs: used == 0,
        // every entry finite and the subset columns carry the adapter. Without an adapter the restricted projection runs.
        const bool lora_on = !loras.empty();
        const int32_t want_used = lora_on ? 0 : (int32_t) sub_ids.size();
        if (llama_output_vocab_subset_used(ctx) != want_used) {
            fprintf(stderr, "vocab subset used=%d expected %d\n", (int) llama_output_vocab_subset_used(ctx), (int) want_used);
            return 6;
        }
        for (auto & o : out) {
            std::vector<float> g;
            for (size_t p = 0; p + (size_t) n_vocab <= o.size(); p += (size_t) n_vocab) {
                size_t k = 0;
                for (int t = 0; t < n_vocab; ++t) {
                    const bool in_sub = k < sub_ids.size() && sub_ids[k] == t;
                    const float v = o[p + (size_t) t];
                    if (in_sub) { g.push_back(v); ++k; }
                    else if (!lora_on && v != -INFINITY) { fprintf(stderr, "entry %d outside the subset is %g, expected -inf\n", t, v); return 6; }
                }
            }
            o.swap(g);
        }
    }
    FILE * fo = fopen(out_path, "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
    const int32_t ns = (int32_t) seqs.size(), nv = (int32_t) (sub_ids.empty() ? n_vocab : (int) sub_ids.size());
    fwrite(&ns, 4, 1, fo);
    fwrite(&nv, 4, 1, fo);
    for (size_t i = 0; i < seqs.size(); ++i) {
        const int32_t nt = (int32_t) seqs[i].size();
        fwrite(&nt, 4, 1, fo);
        fwrite(out[i].data(), sizeof(float), out[i].size(), fo);
    }
    fclose(fo);
    printf("test-adapter-peft-parity: wrote %d sequence(s), n_vocab %d, %zu adapter file(s) -> %s\n", ns, nv, loaded.size(), out_path);

    for (auto & it : loaded) llama_adapter_lora_free(it.second);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
