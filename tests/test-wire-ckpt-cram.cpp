// WIRING test for --ctx-ckpt-cram-soft: common_prompt_checkpoint::cram_compress()/reload() route the checkpoint bytes
// through llama_cram_soft_store and give back EXACTLY the bytes that went in; the process counters show it fired.
//   Part A: synthetic checkpoint (3 vectors, mixed compressibility): OFF control, ON round trip, counters, store reuse,
//           invalidation by update_dft/clear, corrupted store refused (CRC).
//   Part B (needs a model path argv[1], tests/make_tiny_wire_gguf.py): a REAL llama_context. Decode a prompt, take a
//           checkpoint with update_tgt (the server's call), decode N more tokens (reference logits), then restore the
//           checkpoint -- once raw (flag OFF) and once through the CRAM store (flag ON) -- re-decode the same N tokens
//           and require bit-identical logits in all three.
#include "common.h"
#include "llama.h"
#include "../src/llama-cram-soft.h"
#include "../src/llama-cram-probe.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static void part_a() {
    std::mt19937 rng(7);
    common_prompt_checkpoint ck;
    ck.update_pos(100, 3, 99);
    ck.data_tgt.resize(300000);
    for (size_t i = 0; i < ck.data_tgt.size(); i++) { // f16-ish: low byte noisy, high byte repetitive
        ck.data_tgt[i] = (i & 1) ? (uint8_t)(0x3c + (rng() & 3)) : (uint8_t) rng();
    }
    ck.data_dft.assign(70000, 0); // very compressible
    ck.data_spec = {1, 2, 3, 4, 5};
    const auto want_t = ck.data_tgt, want_d = ck.data_dft, want_s = ck.data_spec;

    common_ckpt_cram_reset_counters();
    // OFF control: nothing compressed -> counters stay zero, data untouched
    CHECK(!ck.spilled());
    CHECK(common_ckpt_cram_get_counters().stores == 0);

    // ON
    CHECK(ck.cram_compress());
    auto c = common_ckpt_cram_get_counters();
    printf("A: stores=%llu raw=%llu stored=%llu ratio=%.3f\n", (unsigned long long) c.stores, (unsigned long long) c.raw_bytes,
           (unsigned long long) c.stored_bytes, c.stored_bytes ? (double) c.raw_bytes / c.stored_bytes : 0.0);
    CHECK(c.stores == 1);
    CHECK(c.raw_bytes == want_t.size() + want_d.size() + want_s.size());
    CHECK(c.stored_bytes < c.raw_bytes);
    CHECK(ck.spilled() && ck.empty() && ck.data_dft.empty() && ck.data_spec.empty());
    CHECK(ck.size() == 0);

    CHECK(ck.reload());
    c = common_ckpt_cram_get_counters();
    CHECK(c.reloads == 1);
    CHECK(c.read_bytes == c.raw_bytes);
    CHECK(ck.data_tgt == want_t);
    CHECK(ck.data_dft == want_d);
    CHECK(ck.data_spec == want_s);
    CHECK(!ck.spilled());

    // re-compress after reload reuses the store (no second build)
    CHECK(ck.cram_compress());
    CHECK(common_ckpt_cram_get_counters().stores == 1);
    CHECK(ck.reload() && ck.data_tgt == want_t && ck.data_dft == want_d && ck.data_spec == want_s);

    // a modified checkpoint must NOT reuse the stale store
    ck.data_dft.assign(10, 9);
    ck.cram.reset();
    CHECK(ck.cram_compress());
    CHECK(common_ckpt_cram_get_counters().stores == 2);
    CHECK(ck.reload() && ck.data_dft.size() == 10 && ck.data_dft[0] == 9 && ck.data_tgt == want_t);

    // a copy shares the immutable store and reloads independently
    CHECK(ck.cram_compress());
    common_prompt_checkpoint cp = ck;
    CHECK(cp.spilled() && cp.reload() && cp.data_tgt == want_t);
    CHECK(ck.spilled() && ck.reload() && ck.data_tgt == want_t);

    // clear() drops the store
    ck.clear();
    CHECK(ck.cram == nullptr && !ck.spilled());

    // corrupted store: reload refuses (CRC), caller falls back to reprocessing
    common_prompt_checkpoint bad;
    bad.data_tgt.assign(200000, 0x5a);
    for (size_t i = 0; i < bad.data_tgt.size(); i += 3) bad.data_tgt[i] = (uint8_t) rng();
    CHECK(bad.cram_compress());
    size_t len = 0;
    uint8_t * p = bad.cram->debug_stored_bytes_of_block(0, &len);
    CHECK(p != nullptr && len > 0);
    if (p && len) { p[len / 2] ^= 0xff; }
    CHECK(!bad.reload());
    CHECK(!bad.spilled() || bad.cram == nullptr);
}

static std::vector<float> decode_tokens(llama_context * ctx, std::vector<llama_token> toks, int n_vocab) {
    std::vector<float> out;
    for (auto t : toks) {
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b) != 0) { out.clear(); return out; }
        const float * l = llama_get_logits_ith(ctx, -1);
        out.insert(out.end(), l, l + n_vocab);
    }
    return out;
}

static void part_b(const char * path) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    CHECK(model != nullptr);
    if (!model) { return; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx != nullptr);
    if (!ctx) { llama_model_free(model); return; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt, cont;
    for (int i = 0; i < 24; i++) prompt.push_back(4 + (i * 7) % (nv - 4));
    for (int i = 0; i < 6; i++)  cont.push_back(4 + (i * 11 + 3) % (nv - 4));

    llama_batch pb = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    CHECK(llama_decode(ctx, pb) == 0);

    common_prompt_checkpoint ck;
    ck.update_pos((int64_t) prompt.size(), 0, (llama_pos) prompt.size() - 1);
    ck.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    CHECK(ck.data_tgt.size() > 0);
    const auto raw = ck.data_tgt;

    const auto ref = decode_tokens(ctx, cont, nv);
    CHECK(ref.size() == cont.size() * (size_t) nv);

    auto restore_and_decode = [&](const common_prompt_checkpoint & c, bool must_load = true) {
        llama_kv_cache_seq_rm(ctx, 0, -1, -1);
        const size_t n = llama_state_seq_set_data_ext(ctx, c.data_tgt.data(), c.data_tgt.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (must_load) { CHECK(n == c.data_tgt.size()); }
        return decode_tokens(ctx, cont, nv);
    };

    const auto off = restore_and_decode(ck); // flag OFF: raw bytes
    CHECK(off.size() == ref.size() && std::memcmp(off.data(), ref.data(), ref.size() * sizeof(float)) == 0);

    common_ckpt_cram_reset_counters();
    CHECK(ck.cram_compress());
    CHECK(ck.data_tgt.empty());
    CHECK(ck.reload());
    const auto c = common_ckpt_cram_get_counters();
    printf("B: checkpoint %zu bytes -> stored %llu, stores=%llu reloads=%llu\n", raw.size(), (unsigned long long) c.stored_bytes,
           (unsigned long long) c.stores, (unsigned long long) c.reloads);
    CHECK(c.stores == 1 && c.reloads == 1);
    CHECK(ck.data_tgt == raw);
    const auto on = restore_and_decode(ck); // flag ON: bytes came back out of the CRAM store
    CHECK(on.size() == ref.size() && std::memcmp(on.data(), ref.data(), ref.size() * sizeof(float)) == 0);

    // control: a checkpoint that lost one byte must change the logits, or the comparison above proves nothing
    common_prompt_checkpoint ctl = ck;
    ctl.data_tgt[ctl.data_tgt.size() / 2] ^= 0x40;
    ctl.data_tgt[ctl.data_tgt.size() - 3] ^= 0x40;
    const auto cl = restore_and_decode(ctl, false);
    CHECK(cl.size() != ref.size() || std::memcmp(cl.data(), ref.data(), ref.size() * sizeof(float)) != 0);

    llama_free(ctx);
    llama_model_free(model);
}

static void part_c() { // --cram-probe log helper: runs, reports, and returns the real probe's result
    const auto r = common_cram_probe_log(nullptr);
    CHECK(r.detail.size() > 0 || r.node_count_total >= 0);
}

int main(int argc, char ** argv) {
    part_a();
    part_c();
    if (argc > 1) { part_b(argv[1]); } else { printf("Part B skipped (no model path)\n"); g_fail++; }
    printf("test-wire-ckpt-cram: %s (%d failed)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
