// quant-sensitivity.cpp -- per-tensor quantisation sensitivity: perturb ONE tensor by a Q4_0/Q8_0 round trip, all others clean,
// and score the output KL (nats) over the next-token distribution plus whether the greedy token flipped.
//
// Row S69-ENGREV-FERROX-QUANT-SENSITIVITY-TOOL-2026-10-27-01.
// SOURCE-READ (design only, Apache-2.0, re-written here, no code copied): ferrox crates/frink-cli/src/quant_sensitivity.rs:1-697
// (all of it; the first read stopped at 300), sha 7bae13aa10abec683a9fb3410528af4b34b5dd29. Taken from it: one-tensor-at-a-time, KL(clean||perturbed)
// in that direction (weights each term by the clean model's mass), relative MSE mean and p99 per 32-element block with an all-zero block scored 0
// (0/0 would put NaN through the tensor mean), nearest-rank p99, --layers START:END end-exclusive, the per-FAMILY share of the total KL (the
// column a static quant rule is guessing at), and the refusal to run when the weights are not plainly writable (ferrox: repack cache keyed by
// buffer address; here: any tensor whose buffer is not the plain CPU buffer).
//
//   quant-sensitivity -m model.gguf [--cand q4_0|q8_0] [--layers A:B] [--n-prompt N] [--seed S] [--tokens-file f] [--top N] [-t T]
//   quant-sensitivity --selftest
// Only F32 / F16 source tensors are perturbed (a tensor already quantised would be double-rounded); those are counted and reported as skipped.
// Perplexity's KL (tools/perplexity) scores a whole quantised model; this scores each tensor alone, which is what a per-tensor recipe needs.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------- pure helpers (selftested)
static double qs_percentile(std::vector<double> xs, double q) {   // nearest rank by index round((n-1)q), as the reference tool does
    if (xs.empty()) return 0.0;
    const size_t idx = (size_t) std::llround((double) (xs.size() - 1) * q);
    std::nth_element(xs.begin(), xs.begin() + (long) idx, xs.end());
    return xs[idx];
}

static double qs_mean(const std::vector<double> & xs) {
    if (xs.empty()) return 0.0;
    double s = 0; for (double x : xs) s += x; return s / (double) xs.size();
}

static std::vector<double> qs_softmax(const float * lg, int n) {
    double mx = -1e300; for (int i = 0; i < n; ++i) mx = std::max(mx, (double) lg[i]);
    std::vector<double> e((size_t) n); double s = 0;
    for (int i = 0; i < n; ++i) { e[(size_t) i] = std::exp((double) lg[i] - mx); s += e[(size_t) i]; }
    if (s <= 0) return std::vector<double>((size_t) n, 0.0);
    for (double & v : e) v /= s;
    return e;
}

// KL(clean || perturbed), nats. This direction: each term weighted by what the UNDAMAGED model put there.
static double qs_kl(const std::vector<double> & p, const float * perturbed_logits) {
    const std::vector<double> q = qs_softmax(perturbed_logits, (int) p.size());
    double acc = 0;
    for (size_t i = 0; i < p.size(); ++i) if (p[i] > 0) acc += p[i] * std::log(p[i] / std::max(q[i], 1e-300));
    return std::max(acc, 0.0);
}

static int qs_argmax(const float * lg, int n) { int a = 0; for (int i = 1; i < n; ++i) if (lg[i] > lg[a]) a = i; return a; }

// relative MSE per 32-element block of `orig` vs `back`; an all-zero block scores 0 (no 0/0)
static std::vector<double> qs_block_relmse(const float * orig, const float * back, size_t n, size_t block = 32) {
    std::vector<double> out;
    for (size_t s = 0; s + block <= n; s += block) {
        double e = 0, d = 0;
        for (size_t i = 0; i < block; ++i) { const double a = orig[s + i], b = back[s + i]; e += a * a; d += (a - b) * (a - b); }
        out.push_back(e > 0 ? d / e : 0.0);
    }
    return out;
}

// "blk.14.ffn_down.0.weight" -> "ffn_down", "blk.3.attn_v.weight" -> "attn_v"
static std::string qs_family(const std::string & name) {
    size_t a = name.find('.'); if (a == std::string::npos) return name;
    size_t b = name.find('.', a + 1); if (b == std::string::npos) return name;
    const std::string rest = name.substr(b + 1);
    return rest.substr(0, rest.find('.'));
}

// START:END, end exclusive, either side optional. false = invalid.
static bool qs_layer_range(const std::string & spec, int n_layers, int & from, int & to) {
    if (spec.empty()) { from = 0; to = n_layers; return true; }
    const size_t c = spec.find(':'); if (c == std::string::npos) return false;
    const std::string a = spec.substr(0, c), b = spec.substr(c + 1);
    auto num = [](const std::string & s, int & v) { if (s.empty()) return true; for (char ch : s) if (ch < '0' || ch > '9') return false; v = atoi(s.c_str()); return true; };
    from = 0; to = n_layers;
    if (!num(a, from) || !num(b, to)) return false;
    return from < to && to <= n_layers;
}

static int qs_layer_of(const std::string & name) {   // "blk.N...." -> N, else -1
    if (name.compare(0, 4, "blk.") != 0) return -1;
    return atoi(name.c_str() + 4);
}

struct qs_row { std::string name; double relmse_mean, relmse_p99, kl; bool flipped; };

struct qs_family_share { std::string family; double share; std::string worst; };
static std::vector<qs_family_share> qs_family_shares(const std::vector<qs_row> & rows) {
    double total = 0; for (const auto & r : rows) total += r.kl;
    std::map<std::string, std::pair<double, std::pair<double, std::string>>> m;
    for (const auto & r : rows) {
        auto & e = m[qs_family(r.name)];
        if (e.second.second.empty() || r.kl > e.second.first) e.second = { r.kl, r.name };
        e.first += r.kl;
    }
    std::vector<qs_family_share> out;
    for (auto & kv : m) out.push_back({ kv.first, total > 0 ? kv.second.first / total : 0.0, kv.second.second.second });
    std::sort(out.begin(), out.end(), [](const qs_family_share & a, const qs_family_share & b) { return a.share > b.share; });
    return out;
}

// ---------------------------------------------------------------- selftest
static int selftest() {
    int fails = 0;
    auto check = [&](bool ok, const char * what) { printf("selftest %-90s : %s\n", what, ok ? "PASS" : "FAIL"); fails += !ok; };
    const float lg[4] = { 1.f, -2.f, 3.5f, 0.f };
    const auto p = qs_softmax(lg, 4);
    check(qs_kl(p, lg) < 1e-12, "KL of a distribution against itself is 0");
    const float c[3] = { 2.f, 1.f, 0.f }; const auto pc = qs_softmax(c, 3);
    const float small[3] = { 2.05f, 1.f, 0.f }, large[3] = { 0.f, 1.f, 2.f };
    check(qs_kl(pc, small) > 0 && qs_kl(pc, large) > 10 * qs_kl(pc, small), "KL grows with the damage (planted 0.05 vs a reversal)");
    std::vector<double> xs; for (int i = 0; i < 100; ++i) xs.push_back(i);
    check(qs_percentile(xs, 0.99) == 98.0 && qs_percentile(xs, 0.5) == 50.0, "p99 of 0..99 is 98 (second largest, not the max); median 50");
    std::vector<float> z(32, 0.f);
    const auto zr = qs_block_relmse(z.data(), z.data(), 32);
    check(zr.size() == 1 && zr[0] == 0.0, "an all-zero block scores 0, not NaN");
    // real Q4_0 / Q8_0 round trip through ggml: the quantiser and dequantiser agree, Q8 is finer than Q4
    std::vector<float> src(32); for (int i = 0; i < 32; ++i) src[(size_t) i] = ((float) i - 16.f) * 0.25f;
    double err[2] = { 0, 0 }; const ggml_type ty[2] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 };
    for (int t = 0; t < 2; ++t) {
        ggml_quantize_init(ty[t]);
        std::vector<char> q(ggml_row_size(ty[t], 32)); std::vector<float> back(32);
        ggml_quantize_chunk(ty[t], src.data(), q.data(), 0, 1, 32, nullptr);
        ggml_get_type_traits(ty[t])->to_float(q.data(), back.data(), 32);
        err[t] = qs_mean(qs_block_relmse(src.data(), back.data(), 32));
    }
    check(err[0] > 0 && err[0] < 0.05, "Q4_0 round trip of a ramp: relative MSE in (0, 0.05)");
    check(err[1] < err[0] / 10, "Q8_0 error is at least 10x below Q4_0 (the statistic ranks the candidates)");
    std::vector<qs_row> rows = { { "blk.0.attn_v.weight", 0, 0, 1.0, false }, { "blk.1.attn_v.weight", 0, 0, 3.0, false }, { "blk.0.ffn_down.weight", 0, 0, 5.0, true } };
    const auto fs = qs_family_shares(rows);
    double sum = 0; for (const auto & f : fs) sum += f.share;
    check(qs_family("blk.14.ffn_down.0.weight") == "ffn_down" && qs_family("blk.3.attn_v.weight") == "attn_v", "family strips the layer and expert index");
    check(fs[0].family == "ffn_down" && fs[1].worst == "blk.1.attn_v.weight" && std::fabs(sum - 1.0) < 1e-12, "family shares sum to 1 and name the worst tensor");
    int a, b;
    check(qs_layer_range("", 16, a, b) && a == 0 && b == 16 && qs_layer_range("0:4", 16, a, b) && b == 4 && qs_layer_range("12:", 16, a, b) && a == 12 && qs_layer_range(":3", 16, a, b) && b == 3, "layer ranges parse, end exclusive");
    check(!qs_layer_range("4:4", 16, a, b) && !qs_layer_range("0:99", 16, a, b) && !qs_layer_range("nonsense", 16, a, b), "empty, past-the-end and malformed ranges are refused");
    check(qs_layer_of("blk.12.attn_q.weight") == 12 && qs_layer_of("output.weight") == -1, "layer index parse");
    printf("selftest: %s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------- model mode
static std::vector<float> read_f32(const ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t) n);
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_get(t, out.data(), 0, (size_t) n * sizeof(float)); }
    else {   // F16
        std::vector<ggml_fp16_t> h((size_t) n);
        ggml_backend_tensor_get(t, h.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; ++i) out[(size_t) i] = ggml_fp16_to_fp32(h[(size_t) i]);
    }
    return out;
}

static void write_f32(ggml_tensor * t, const std::vector<float> & v) {
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float)); return; }
    std::vector<ggml_fp16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = ggml_fp32_to_fp16(v[i]);
    ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
}

static bool last_logits(llama_context * ctx, std::vector<llama_token> & prompt, std::vector<float> & out, int nv) {
    llama_kv_cache_clear(ctx);
    llama_batch b = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(ctx, b) != 0) return false;
    out.assign(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + nv);
    return true;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    std::string model, layers, tokfile, cand = "q4_0"; int threads = 4, n_prompt = 16, top = 20; uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i]; auto nx = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") model = nx(); else if (a == "--cand") cand = nx(); else if (a == "--layers") layers = nx(); else if (a == "-t") threads = atoi(nx());
        else if (a == "--n-prompt") n_prompt = atoi(nx()); else if (a == "--seed") seed = (uint64_t) atoll(nx()); else if (a == "--tokens-file") tokfile = nx();
        else if (a == "--top") top = atoi(nx());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    ggml_type ct;
    if (cand == "q4_0") ct = GGML_TYPE_Q4_0; else if (cand == "q8_0") ct = GGML_TYPE_Q8_0;
    else { fprintf(stderr, "--cand must be q4_0 or q8_0 (a K-quant needs its own quantiser path)\n"); return 2; }
    if (model.empty()) { fprintf(stderr, "usage: quant-sensitivity -m model.gguf [--cand q4_0|q8_0] [--layers A:B] [--n-prompt N] [--seed S] [--tokens-file f] [--top N] | --selftest\n"); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.use_mmap = false;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    int lfrom = 0, lto = llama_model_n_layer(m);
    if (!qs_layer_range(layers, llama_model_n_layer(m), lfrom, lto)) { fprintf(stderr, "bad --layers %s (START:END, end exclusive, within %d layers)\n", layers.c_str(), llama_model_n_layer(m)); return 2; }
    std::vector<llama_token> prompt;
    if (!tokfile.empty()) { FILE * f = fopen(tokfile.c_str(), "r"); int t; while (f && fscanf(f, "%d", &t) == 1) prompt.push_back(t); if (f) fclose(f); }
    else { uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1; for (int i = 0; i < n_prompt; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; prompt.push_back((llama_token) (3 + s % (uint64_t) std::max(1, nv - 3))); } }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) prompt.size() + 8; cp.n_batch = cp.n_ubatch = (uint32_t) prompt.size(); cp.n_threads = cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    std::vector<float> clean, again;
    if (!last_logits(ctx, prompt, clean, nv) || !last_logits(ctx, prompt, again, nv)) { fprintf(stderr, "decode failed\n"); return 1; }
    const std::vector<double> pc = qs_softmax(clean.data(), nv);
    const double base_noise = qs_kl(pc, again.data());
    if (base_noise > 1e-9) { fprintf(stderr, "ERROR: the clean model is not deterministic run to run (KL %.3e): every number below would be noise\n", base_noise); return 3; }
    const int clean_top = qs_argmax(clean.data(), nv);
    ggml_quantize_init(ct);

    std::vector<qs_row> rows; int skipped_type = 0, skipped_buf = 0;
    for (const auto & kv : llama_internal_get_tensor_map(m)) {
        ggml_tensor * t = kv.second; const std::string & name = kv.first;
        const int il = qs_layer_of(name);
        if (il < lfrom || il >= lto) continue;
        if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) continue;
        if (ggml_n_dims(t) < 2 || t->ne[0] % 32 != 0) continue;
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) { ++skipped_type; continue; }
        // Any HOST buffer, whatever its name: with n_gpu_layers 0 a CUDA build keeps the weights in "CUDA_Host" (pinned host
        // memory; ggml-cuda.cu gives that buffer type the CPU type's is_host), not in "CPU". Comparing the name skipped every
        // tensor there (gate b1114d: rc 3, "14 not in the plain CPU buffer"). read_f32 / write_f32 go through
        // ggml_backend_tensor_get / _set, so the buffer's own interface does the copy.
        if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) { ++skipped_buf; continue; }
        const std::vector<float> orig = read_f32(t);
        const int64_t ncol = t->ne[0], nrow = ggml_nelements(t) / ncol;
        std::vector<char> q(ggml_row_size(ct, ncol) * (size_t) nrow); std::vector<float> back(orig.size());
        ggml_quantize_chunk(ct, orig.data(), q.data(), 0, nrow, ncol, nullptr);
        for (int64_t r = 0; r < nrow; ++r) ggml_get_type_traits(ct)->to_float(q.data() + (size_t) r * ggml_row_size(ct, ncol), back.data() + (size_t) (r * ncol), ncol);
        const auto per_block = qs_block_relmse(orig.data(), back.data(), orig.size());
        write_f32(t, back);
        std::vector<float> got;
        const bool ok = last_logits(ctx, prompt, got, nv);
        write_f32(t, orig);   // restore BEFORE anything else can fail
        if (!ok) { fprintf(stderr, "decode failed on %s\n", name.c_str()); return 1; }
        rows.push_back({ name, qs_mean(per_block), qs_percentile(per_block, 0.99), qs_kl(pc, got.data()), qs_argmax(got.data(), nv) != clean_top });
    }
    // restore check: with every tensor written back the clean logits must come back exactly (the perturbation did not leak)
    std::vector<float> fin; last_logits(ctx, prompt, fin, nv);
    const double leak = qs_kl(pc, fin.data());
    if (rows.empty()) { fprintf(stderr, "ERROR: no tensor was perturbed (skipped: %d not F32/F16, %d not in a host buffer)\n", skipped_type, skipped_buf); return 3; }
    std::sort(rows.begin(), rows.end(), [](const qs_row & a, const qs_row & b) { return a.kl > b.kl; });
    printf("quant-sensitivity: candidate %s, layers %d:%d, prompt %zu tokens, baseline run-to-run KL %.1e, restore-leak KL %.1e\n", cand.c_str(), lfrom, lto, prompt.size(), base_noise, leak);
    printf("%-28s %12s %12s %12s  top1\n", "tensor", "relmse mean", "relmse p99", "dKL nats");
    for (size_t i = 0; i < rows.size() && (int) i < top; ++i)
        printf("%-28s %12.3e %12.3e %12.3e  %s\n", rows[i].name.c_str(), rows[i].relmse_mean, rows[i].relmse_p99, rows[i].kl, rows[i].flipped ? "FLIPPED" : "kept");
    double total = 0; int flips = 0; for (const auto & r : rows) { total += r.kl; flips += r.flipped; }
    for (const auto & f : qs_family_shares(rows)) printf("family %-14s %5.1f%% of measured KL, worst single tensor %s\n", f.family.c_str(), f.share * 100.0, f.worst.c_str());
    printf("total measured KL %.4e nats across %zu tensors at %s; %d flip the greedy token on their own; skipped %d (not F32/F16), %d (not plain CPU buffer)\n", total, rows.size(), cand.c_str(), flips, skipped_type, skipped_buf);
    llama_free(ctx); llama_model_free(m); llama_backend_free();
    return leak > 1e-9 ? 4 : 0;
}
