// Logits dumper for the In-Place TTT parity test (tests/test-ttt-inplace.py, row S69-FORK-INPLACE-TTT-2026-10-26-01).
// The Python driver builds a tiny random-weight Qwen3 with the OFFICIAL In-Place TTT modeling code (ByteDance-Seed/In-Place-TTT,
// Apache-2.0, run unmodified as the oracle), converts it with this repo's converter, and runs this binary on the same token ids. This
// binary does no checking of its own: it decodes exactly what it is told and writes every position's logits (and the token ids it fed,
// which for `gen` are its own greedy picks).
//
// usage: test-ttt-inplace MODEL.gguf TOKENS.txt OUT.bin MODE [args]
//   TOKENS.txt  one sequence per line, space-separated token ids (no BOS added: ids are used verbatim)
//   MODE
//     load           llama_model_load_from_file, nothing else (no context, no decode); OUT.bin holds zero sequences
//     gen NP NG     sequence 0 only: prefill its first NP ids in ONE batch, then NG greedy decode steps (argmax of the last logits)
//     tbt            every sequence alone (memory cleared between), one token per llama_decode
//     ub U           every sequence alone, ONE llama_decode of the whole sequence with n_ubatch = U
//     step S         every sequence alone, llama_decode batches of S tokens (n_ubatch large)
//     multi SA SB    sequences 0 and 1 together (seq ids 0 and 1): each llama_decode batch holds the next SA ids of sequence 0 followed by
//                    the next SB ids of sequence 1 (whichever remain), so the two sequences interleave inside one batch
//     state K        sequence 0: decode its first K ids, llama_state_seq_get_data(seq 0), FREE the context, make a new one,
//                    llama_state_seq_set_data(seq 0), decode the rest. Rows 0..K-1 come from the first context, the rest from the second.
//     rewind K B G   sequence 0 token by token to K, seq_rm tries (B: across a completed chunk, 2..4: interior, G: inside the current
//                    chunk), seq_cp 0 -> 1, sequence 1 continues from G. Prints "rewind: rm(B)=0/1 rm(2,4)=0/1 rm(G)=0/1".
//     spec NP NG K CAP DRAFT [mix]   greedy speculative decoding of sequence 0 (see the mode); DRAFT = draft GGUF or "adv"
// OUT.bin: int32 n_seq, int32 n_vocab, then per sequence: int32 n_tok, n_tok int32 token ids, n_tok * n_vocab float32 logits (row-major).
// Exit 0 = wrote OUT.bin; 2 = usage/IO; 3 = model or context creation failed; 4 = llama_decode failed; 5 = state get/set failed;
// 6 = llama_memory_seq_rm refused the discarded draft suffix (spec); 7 = the same on the draft context (spec).
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct seq_out {
    std::vector<llama_token> tok;
    std::vector<float>       logits;
};

static llama_context * make_ctx(llama_model * model, int n_ubatch, int n_seq_max) {
    // TTT_DUMP_NCTX / TTT_DUMP_THREADS: larger runs (the real-weight check); the tiny suite uses the defaults
    const char * env_ctx = getenv("TTT_DUMP_NCTX");
    const char * env_thr = getenv("TTT_DUMP_THREADS");
    const int    n_ctx   = env_ctx ? std::max(256, atoi(env_ctx)) : 256;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = (uint32_t) n_ctx;
    cp.n_batch   = (uint32_t) n_ctx;
    cp.n_ubatch  = (uint32_t) std::min(n_ubatch, n_ctx);
    cp.n_seq_max = (uint32_t) n_seq_max;
    cp.n_threads = cp.n_threads_batch = env_thr ? std::max(1, atoi(env_thr)) : 4;
    // F32 KV cache: the oracle is fp32 torch, and an F16 cache alone moves logits by ~5e-3 (measured by the PEFT parity harness).
    cp.type_k = cp.type_v = cp.idx_type_k = GGML_TYPE_F32;
    cp.type_k_first = cp.type_k_last = cp.type_v_first = cp.type_v_last = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return llama_init_from_model(model, cp);
}

// decode `n` ids of sequence `seq` starting at position `pos0` (ids taken from `ids` at offset `off`), logits for every id appended to out
struct piece {
    int seq;
    int off;
    int n;
};

static bool decode_pieces(llama_context * ctx, const std::vector<std::vector<llama_token>> & ids, const std::vector<piece> & pieces,
                          int n_vocab, std::vector<seq_out> & out) {
    int n = 0;
    for (auto & p : pieces) n += p.n;
    if (n == 0) return true;
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens    = 0;
    for (auto & p : pieces) {
        for (int i = 0; i < p.n; ++i) {
            const int k    = b.n_tokens++;
            b.token[k]     = ids[p.seq][p.off + i];
            b.pos[k]       = (llama_pos) (p.off + i);
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = (llama_seq_id) p.seq;
            b.logits[k]    = 1;
        }
    }
    const int rc = llama_decode(ctx, b);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed rc=%d\n", rc);
        llama_batch_free(b);
        return false;
    }
    int k = 0;
    for (auto & p : pieces) {
        for (int i = 0; i < p.n; ++i, ++k) {
            const float * l = llama_get_logits_ith(ctx, k);
            out[p.seq].tok.push_back(ids[p.seq][p.off + i]);
            out[p.seq].logits.insert(out[p.seq].logits.end(), l, l + n_vocab);
        }
    }
    llama_batch_free(b);
    return true;
}

static int argmax_last(const seq_out & o, int n_vocab) {
    const float * l = o.logits.data() + o.logits.size() - n_vocab;
    return (int) (std::max_element(l, l + n_vocab) - l);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.txt OUT.bin MODE [args]  (see the header of tests/test-ttt-inplace.cpp)\n", argv[0]);
        return 2;
    }
    const char *      model_path = argv[1];
    const char *      tok_path   = argv[2];
    const char *      out_path   = argv[3];
    const std::string mode       = argv[4];
    auto              iarg       = [&](int i) -> int {
        if (i >= argc) {
            fprintf(stderr, "mode %s: missing argument %d\n", mode.c_str(), i - 4);
            exit(2);
        }
        return atoi(argv[i]);
    };

    std::vector<std::vector<llama_token>> ids;
    {
        std::ifstream f(tok_path);
        if (!f) {
            fprintf(stderr, "cannot open %s\n", tok_path);
            return 2;
        }
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::vector<llama_token> t;
            long v;
            while (ss >> v) t.push_back((llama_token) v);
            if (!t.empty()) ids.push_back(t);
        }
    }
    if (ids.empty()) {
        fprintf(stderr, "no sequences in %s\n", tok_path);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = 0;
    llama_model * model   = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", model_path);
        return 3;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<seq_out> out;
    int rc = 0;
    if (mode == "load") {
        // model load only (the loader's TTT hparams line goes to the log); writes an OUT.bin with zero sequences
        printf("load: ok, %d layers\n", llama_model_n_layer(model));
    } else if (mode == "gen") {
        const int np = iarg(5), ng = iarg(6);
        out.resize(1);
        llama_context * ctx = make_ctx(model, 256, 1);
        if (!ctx) return 3;
        std::vector<std::vector<llama_token>> gen(1, std::vector<llama_token>(ids[0].begin(), ids[0].begin() + std::min<int>(np, ids[0].size())));
        if (!decode_pieces(ctx, gen, { { 0, 0, (int) gen[0].size() } }, n_vocab, out)) rc = 4;
        for (int s = 0; rc == 0 && s < ng; ++s) {
            gen[0].push_back((llama_token) argmax_last(out[0], n_vocab));
            if (!decode_pieces(ctx, gen, { { 0, (int) gen[0].size() - 1, 1 } }, n_vocab, out)) rc = 4;
        }
        llama_free(ctx);
    } else if (mode == "tbt" || mode == "ub" || mode == "step") {
        const int u    = mode == "ub" ? iarg(5) : 256;
        const int step = mode == "tbt" ? 1 : mode == "step" ? iarg(5) : 1 << 30;
        out.resize(ids.size());
        llama_context * ctx = make_ctx(model, u, 1);
        if (!ctx) return 3;
        for (size_t s = 0; rc == 0 && s < ids.size(); ++s) {
            llama_memory_clear(llama_get_memory(ctx), true);
            std::vector<std::vector<llama_token>> one(1, ids[s]);
            std::vector<seq_out>                  o1(1);
            for (int off = 0; rc == 0 && off < (int) ids[s].size(); off += step) {
                const int n = std::min<int>(step, (int) ids[s].size() - off);
                if (!decode_pieces(ctx, one, { { 0, off, n } }, n_vocab, o1)) rc = 4;
            }
            out[s] = o1[0];
        }
        llama_free(ctx);
    } else if (mode == "multi") {
        const int sa = iarg(5), sb = iarg(6);
        if (ids.size() < 2) {
            fprintf(stderr, "multi needs two sequences\n");
            return 2;
        }
        ids.resize(2);
        out.resize(2);
        llama_context * ctx = make_ctx(model, 256, 2);
        if (!ctx) return 3;
        int oa = 0, ob = 0;
        while (rc == 0 && (oa < (int) ids[0].size() || ob < (int) ids[1].size())) {
            const int na = std::min<int>(sa, (int) ids[0].size() - oa);
            const int nb = std::min<int>(sb, (int) ids[1].size() - ob);
            if (!decode_pieces(ctx, ids, { { 0, oa, na }, { 1, ob, nb } }, n_vocab, out)) rc = 4;
            oa += na;
            ob += nb;
        }
        llama_free(ctx);
    } else if (mode == "state") {
        const int k = iarg(5);
        ids.resize(1);
        out.resize(1);
        llama_context * ctx = make_ctx(model, 256, 1);
        if (!ctx) return 3;
        if (!decode_pieces(ctx, ids, { { 0, 0, k } }, n_vocab, out)) rc = 4;
        std::vector<uint8_t> buf;
        if (rc == 0) {
            buf.resize(llama_state_seq_get_size(ctx, 0, 0));
            const size_t got = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0, 0);
            if (got == 0 || got > buf.size()) rc = 5;
            buf.resize(got);
            printf("state: seq 0 after %d ids = %zu bytes\n", k, buf.size());
        }
        llama_free(ctx);
        if (rc == 0) {
            ctx = make_ctx(model, 256, 1);
            if (!ctx) return 3;
            if (llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0, 0) == 0) rc = 5;
            if (rc == 0 && !decode_pieces(ctx, ids, { { 0, k, (int) ids[0].size() - k } }, n_vocab, out)) rc = 4;
            llama_free(ctx);
        }
    } else if (mode == "rewind") {
        // sequence 0 token by token up to K; then seq_rm(0, BAD, -1) (a rewind across a completed chunk, must be refused),
        // seq_rm(0, 2, 4) (interior, must be refused), seq_rm(0, GOOD, -1) (inside the current chunk, must succeed),
        // seq_cp(0 -> 1), and sequence 1 continues token by token from GOOD. Rows 0..GOOD-1 come from sequence 0, the rest
        // from sequence 1, so the output equals the oracle only if the rewind kept the state exact AND seq_cp copied it.
        const int k = iarg(5), bad = iarg(6), good = iarg(7);
        ids.resize(1);
        ids.push_back(ids[0]);
        std::vector<seq_out> o2(2);
        llama_context * ctx = make_ctx(model, 256, 2);
        if (!ctx) return 3;
        for (int off = 0; rc == 0 && off < k; ++off) {
            if (!decode_pieces(ctx, ids, { { 0, off, 1 } }, n_vocab, o2)) rc = 4;
        }
        if (rc == 0) {
            llama_memory_t mem = llama_get_memory(ctx);
            const bool r_bad  = llama_memory_seq_rm(mem, 0, bad, -1);
            const bool r_mid  = llama_memory_seq_rm(mem, 0, 2, 4);
            const bool r_good = llama_memory_seq_rm(mem, 0, good, -1);
            printf("rewind: rm(%d)=%d rm(2,4)=%d rm(%d)=%d can_shift=%d\n", bad, (int) r_bad, (int) r_mid, good, (int) r_good,
                   (int) llama_memory_can_shift(mem));
            llama_memory_seq_cp(mem, 0, 1, -1, -1);
            for (int off = good; rc == 0 && off < (int) ids[1].size(); ++off) {
                if (!decode_pieces(ctx, ids, { { 1, off, 1 } }, n_vocab, o2)) rc = 4;
            }
        }
        llama_free(ctx);
        out.resize(1);
        out[0].tok.assign(o2[0].tok.begin(), o2[0].tok.begin() + std::min<size_t>(good, o2[0].tok.size()));
        out[0].logits.assign(o2[0].logits.begin(), o2[0].logits.begin() + std::min<size_t>((size_t) good * n_vocab, o2[0].logits.size()));
        out[0].tok.insert(out[0].tok.end(), o2[1].tok.begin(), o2[1].tok.end());
        out[0].logits.insert(out[0].logits.end(), o2[1].logits.begin(), o2[1].logits.end());
    } else if (mode == "spec") {
        // greedy speculative decoding of sequence 0: prefill NP ids, then rounds of [id_last, drafts...] verified in ONE
        // llama_decode, longest matching prefix kept, the non-matching suffix removed with llama_memory_seq_rm, until
        // NP + NG positions are decoded. Drafts per round = min(K, llama_ttt_draft_cap(model, pos of id_last)) when CAP = 1,
        // K when CAP = 0. DRAFT = a GGUF path (greedy drafts from that model) or "adv" (adversarial drafts that almost
        // never match). Optional 6th arg "mix": round s corrupts draft s % (nd + 1), so an exact drafter (the target GGUF
        // itself) keeps 0..nd drafts in turn. Rows are the target's logits at each decoded position, like `gen`.
        const int np = iarg(5), ng = iarg(6), kmax = iarg(7), use_cap = iarg(8);
        if (argc < 10) {
            fprintf(stderr, "spec: missing DRAFT\n");
            return 2;
        }
        const std::string dpath = argv[9];
        const bool        mix   = argc > 10 && std::string(argv[10]) == "mix";
        out.resize(1);
        llama_context * ctx = make_ctx(model, 256, 1);
        if (!ctx) return 3;
        llama_model *   dmodel = nullptr;
        llama_context * dctx   = nullptr;
        if (dpath != "adv") {
            dmodel = llama_model_load_from_file(dpath.c_str(), mp);
            dctx   = dmodel ? make_ctx(dmodel, 256, 1) : nullptr;
            if (!dctx) return 3;
        }
        std::vector<std::vector<llama_token>> hist(1, std::vector<llama_token>(ids[0].begin(), ids[0].begin() + std::min<int>(np, ids[0].size())));
        if (!decode_pieces(ctx, hist, { { 0, 0, (int) hist[0].size() } }, n_vocab, out)) rc = 4;
        llama_token id_last = rc == 0 ? (llama_token) argmax_last(out[0], n_vocab) : 0;
        int n_past = (int) hist[0].size(), d_past = 0, steps = 0, drafted = 0, n_acc = 0;
        llama_batch b = llama_batch_init(kmax + 1, 0, 1);
        auto one = [&](llama_context * c, llama_token t, int pos, bool logits) {
            b.n_tokens = 1;
            b.token[0] = t; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = logits;
            return llama_decode(c, b) == 0;
        };
        while (rc == 0 && (int) out[0].tok.size() < np + ng) {
            int nd = kmax;
            if (use_cap) nd = std::min<int>(nd, llama_ttt_draft_cap(model, n_past));
            std::vector<llama_token> d;
            if (dctx) {
                // bring the draft context to exactly the kept history, then feed id_last and draft greedily
                if (d_past > n_past) {
                    if (!llama_memory_seq_rm(llama_get_memory(dctx), 0, n_past, -1)) {
                        fprintf(stderr, "spec: draft seq_rm refused at %d (the draft reached %d)\n", n_past, d_past);
                        rc = 7;
                        break;
                    }
                    d_past = n_past;
                }
                for (; rc == 0 && d_past < n_past; ++d_past) {
                    if (!one(dctx, hist[0][d_past], d_past, false)) rc = 4;
                }
                llama_token cur = id_last;
                for (int i = 0; rc == 0 && i < nd; ++i) {
                    if (!one(dctx, cur, n_past + i, true)) { rc = 4; break; }
                    const float * l = llama_get_logits_ith(dctx, -1);
                    cur = (llama_token) (std::max_element(l, l + n_vocab) - l);
                    d.push_back(cur);
                    d_past = n_past + i + 1;
                }
            } else {
                for (int i = 0; i < nd; ++i) d.push_back((llama_token) ((id_last + 7 + i) % n_vocab));
            }
            if (rc != 0) break;
            if (mix && nd > 0 && steps % (nd + 1) < nd) {
                // corrupt draft j = round % (nd + 1): with an exact drafter the round keeps exactly j drafts
                const int j = steps % (nd + 1);
                d[j]        = (llama_token) ((d[j] + 7) % n_vocab);
            }
            b.n_tokens = 0;
            for (int i = 0; i <= nd; ++i) {
                const int k = b.n_tokens++;
                b.token[k] = i == 0 ? id_last : d[i - 1];
                b.pos[k] = n_past + i; b.n_seq_id[k] = 1; b.seq_id[k][0] = 0; b.logits[k] = 1;
            }
            if (llama_decode(ctx, b) != 0) { rc = 4; break; }
            steps++;
            drafted += nd;
            int k = 0;
            for (;;) {
                const float * l = llama_get_logits_ith(ctx, k);
                out[0].tok.push_back(b.token[k]);
                out[0].logits.insert(out[0].logits.end(), l, l + n_vocab);
                hist[0].push_back(b.token[k]);
                const llama_token t = (llama_token) (std::max_element(l, l + n_vocab) - l);
                if (k < nd && t == d[k]) { ++k; continue; }
                id_last = t;
                break;
            }
            n_acc += k;
            if (k < nd && !llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + k + 1, -1)) {
                fprintf(stderr, "spec: seq_rm refused at %d (the verify batch reached %d)\n", n_past + k + 1, n_past + nd);
                rc = 6;
                break;
            }
            n_past += k + 1;
        }
        llama_batch_free(b);
        printf("spec: K %d cap %d draft %s: %d rounds, %d drafted, %d kept\n", kmax, use_cap, dctx ? "model" : "adv", steps, drafted, n_acc);
        if (dctx) llama_free(dctx);
        if (dmodel) llama_model_free(dmodel);
        llama_free(ctx);
        if (rc == 0) {
            out[0].tok.resize(np + ng);
            out[0].logits.resize((size_t) (np + ng) * n_vocab);
        }
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
        return 2;
    }
    if (rc != 0) return rc;

    FILE * fo = fopen(out_path, "wb");
    if (!fo) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }
    const int32_t ns = (int32_t) out.size(), nv = (int32_t) n_vocab;
    fwrite(&ns, 4, 1, fo);
    fwrite(&nv, 4, 1, fo);
    for (auto & o : out) {
        const int32_t nt = (int32_t) o.tok.size();
        fwrite(&nt, 4, 1, fo);
        fwrite(o.tok.data(), sizeof(int32_t), o.tok.size(), fo);
        fwrite(o.logits.data(), sizeof(float), o.logits.size(), fo);
    }
    fclose(fo);
    printf("test-ttt-inplace: mode %s wrote %d sequence(s), n_vocab %d -> %s\n", mode.c_str(), ns, nv, out_path);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
