// Ollama qwen35 blob with an embedded mtp.* head, loaded with MTP: the three norms that need +1.0 at read time
// (llama_model_loader::apply_ollama_qwen35_mtp_compat, compat_norm_add_one) must get owned storage while EVERY OTHER tensor
// stays in the file mapping. Before 2026-10-18 the compat turned mmap off for the whole load (src/llama.cpp, llama_model_load):
// on the qwen3.6:27b blob that is a 16.5 GB anonymous CPU buffer, refused under a commit-capped job, so
// test-ollama-qwen35-mtp failed in every light-lane ctest.
//
// No blob needed: the test writes a tiny qwen35 GGUF in the Ollama layout (block_count 1, trunk blk.0.*, head under
// mtp.layers.0.* / mtp.fc / mtp.pre_fc_norm_{embedding,hidden} / mtp.norm, per-layer head_count_kv array), all F32 with
// known values, and loads it through the public API.
//
// Arms (all CPU, n_gpu_layers 0):
//   A  mtp=on,  mmap on                    -> 2 layers, 1 NextN; trunk AND untouched MTP tensors in CPU_Mapped buffers;
//                                             enorm / hnorm / shared_head_norm = file + 1.0 in a non-mapped buffer;
//                                             blk.1.attn_norm (already shifted by Ollama's converter) = file, not shifted again.
//                                             FAILS before the fix: the trunk lands in a plain CPU buffer (mmap was off).
//   B  mtp=on,  use_mmap=false             -> same values; nothing mapped (explicit --no-mmap still works).
//   C  mtp=on,  LLAMA_OLLAMA_COMPAT=0      -> 1 layer, no NextN, trunk mapped: the opt-out loads as before.
//   D  mtp=off                             -> 1 layer, no NextN, trunk mapped: the translation does not fire.
//
// usage: test-ollama-qwen35-mtp-mmap   (writes a small temp GGUF in the cwd and removes it)
// Row: S69-FORK-OLLAMA-QWEN35-MTP-COMPAT-NO-MMAP-16GB-2026-10-18-01.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

static const int64_t N_EMBD = 64, N_HEAD = 2, HEAD = 32, N_FF = 128, N_VOCAB = 32;

struct tspec { std::string name; std::vector<int64_t> ne; };

// tensor values: a per-tensor offset plus a small ramp, so a shift by +1.0 (or a wrong tensor) is visible
static float value_of(int tensor_id, int64_t i) { return 0.03125f * (float) (tensor_id % 29) + 0.0009765625f * (float) (i % 64); }

static std::vector<tspec> layer_tensors(const std::string & p) {
    return {
        { p + "attn_norm.weight",           { N_EMBD } },
        { p + "post_attention_norm.weight", { N_EMBD } },
        { p + "attn_q.weight",              { N_EMBD, HEAD * N_HEAD * 2 } },
        { p + "attn_k.weight",              { N_EMBD, HEAD * N_HEAD } },
        { p + "attn_v.weight",              { N_EMBD, HEAD * N_HEAD } },
        { p + "attn_output.weight",         { HEAD * N_HEAD, N_EMBD } },
        { p + "attn_q_norm.weight",         { HEAD } },
        { p + "attn_k_norm.weight",         { HEAD } },
        { p + "ffn_gate.weight",            { N_EMBD, N_FF } },
        { p + "ffn_down.weight",            { N_FF, N_EMBD } },
        { p + "ffn_up.weight",              { N_EMBD, N_FF } },
    };
}

static std::vector<tspec> all_tensors() {
    std::vector<tspec> v = {
        { "token_embd.weight",  { N_EMBD, N_VOCAB } },
        { "output_norm.weight", { N_EMBD } },
        { "output.weight",      { N_EMBD, N_VOCAB } },
    };
    for (auto & t : layer_tensors("blk.0.")) v.push_back(t);
    for (auto & t : layer_tensors("mtp.layers.0.")) v.push_back(t);
    v.push_back({ "mtp.fc.weight",                    { 2 * N_EMBD, N_EMBD } });
    v.push_back({ "mtp.pre_fc_norm_embedding.weight", { N_EMBD } });
    v.push_back({ "mtp.pre_fc_norm_hidden.weight",    { N_EMBD } });
    v.push_back({ "mtp.norm.weight",                  { N_EMBD } });
    return v;
}

static void write_fixture(const char * path) {
    const std::vector<tspec> ts = all_tensors();
    size_t bytes = 0;
    for (const auto & t : ts) {
        size_t n = 1;
        for (int64_t d : t.ne) n *= (size_t) d;
        bytes += n * sizeof(float) + ggml_tensor_overhead();
    }
    ggml_init_params ip = { bytes + 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "qwen35");
    gguf_set_val_str(g, "tokenizer.ggml.model", "no_vocab");
    gguf_set_val_u32(g, "qwen35.vocab_size",       (uint32_t) N_VOCAB);
    gguf_set_val_u32(g, "qwen35.context_length",   128);
    gguf_set_val_u32(g, "qwen35.embedding_length", (uint32_t) N_EMBD);
    gguf_set_val_u32(g, "qwen35.block_count",      1);
    gguf_set_val_u32(g, "qwen35.feed_forward_length", (uint32_t) N_FF);
    gguf_set_val_u32(g, "qwen35.attention.head_count", (uint32_t) N_HEAD);
    const uint32_t hkv[1] = { (uint32_t) N_HEAD };   // Ollama writes a per-layer array; the compat extends it by one entry
    gguf_set_arr_data(g, "qwen35.attention.head_count_kv", GGUF_TYPE_UINT32, hkv, 1);
    gguf_set_val_u32(g, "qwen35.attention.key_length",   (uint32_t) HEAD);
    gguf_set_val_u32(g, "qwen35.attention.value_length", (uint32_t) HEAD);
    gguf_set_val_f32(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(g, "qwen35.rope.dimension_count", (uint32_t) HEAD);
    const int32_t sections[4] = { 4, 4, 8, 0 };
    gguf_set_arr_data(g, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
    gguf_set_val_f32(g, "qwen35.rope.freq_base", 10000.0f);
    gguf_set_val_u32(g, "qwen35.full_attention_interval", 1);   // every trunk layer is full attention: no delta-net tensors
    gguf_set_val_u32(g, "qwen35.ssm.conv_kernel",    4);
    gguf_set_val_u32(g, "qwen35.ssm.inner_size",     (uint32_t) N_EMBD);
    gguf_set_val_u32(g, "qwen35.ssm.state_size",     16);
    gguf_set_val_u32(g, "qwen35.ssm.time_step_rank", 2);
    gguf_set_val_u32(g, "qwen35.ssm.group_count",    2);
    int id = 0;
    for (const auto & t : ts) {
        ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) t.ne.size(), t.ne.data());
        ggml_set_name(x, t.name.c_str());
        float * d = (float *) x->data;
        for (int64_t i = 0; i < ggml_nelements(x); i++) d[i] = value_of(id, i);
        gguf_add_tensor(g, x);
        id++;
    }
    gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
}

static int file_id_of(const std::string & file_name) {
    const std::vector<tspec> ts = all_tensors();
    for (size_t i = 0; i < ts.size(); i++) if (ts[i].name == file_name) return (int) i;
    return -1;
}

static int g_fail = 0;
static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    fflush(stdout);
    g_fail += ok ? 0 : 1;
}

static std::string buft_of(const ggml_tensor * t) {
    if (!t || !t->buffer) return "(none)";
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
}

// true when every element equals the file value (+ shift)
static bool values_match(const ggml_tensor * t, const std::string & file_name, float shift) {
    const int id = file_id_of(file_name);
    if (!t || id < 0 || t->type != GGML_TYPE_F32) return false;
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] != value_of(id, (int64_t) i) + shift) return false;
    }
    return true;
}

struct arm {
    bool loaded = false;
    int n_layer = -1, n_nextn = -1;
    std::map<std::string, std::string> buft;   // tensor name -> buffer type name
    bool enorm_ok = false, hnorm_ok = false, shared_ok = false, mtp_attn_norm_raw = false, trunk_raw = false;
};

static arm run_arm(const char * path, bool mtp, bool use_mmap) {
    arm r;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = use_mmap;
    mp.mtp          = mtp;
    llama_model * m = llama_model_load_from_file(path, mp);
    if (!m) return r;
    r.loaded  = true;
    r.n_layer = llama_model_n_layer(m);
    r.n_nextn = llama_model_n_nextn_layer(m);
    for (const char * n : { "blk.0.attn_q.weight", "blk.0.attn_norm.weight", "token_embd.weight", "blk.1.attn_q.weight",
                            "blk.1.attn_norm.weight", "blk.1.nextn.eh_proj.weight", "blk.1.nextn.enorm.weight",
                            "blk.1.nextn.hnorm.weight", "blk.1.nextn.shared_head_norm.weight" }) {
        r.buft[n] = buft_of(llama_get_model_tensor(m, n));
    }
    r.enorm_ok          = values_match(llama_get_model_tensor(m, "blk.1.nextn.enorm.weight"),            "mtp.pre_fc_norm_embedding.weight", 1.0f);
    r.hnorm_ok          = values_match(llama_get_model_tensor(m, "blk.1.nextn.hnorm.weight"),            "mtp.pre_fc_norm_hidden.weight",    1.0f);
    r.shared_ok         = values_match(llama_get_model_tensor(m, "blk.1.nextn.shared_head_norm.weight"), "mtp.norm.weight",                  1.0f);
    r.mtp_attn_norm_raw = values_match(llama_get_model_tensor(m, "blk.1.attn_norm.weight"),              "mtp.layers.0.attn_norm.weight",    0.0f);
    r.trunk_raw         = values_match(llama_get_model_tensor(m, "blk.0.attn_norm.weight"),              "blk.0.attn_norm.weight",           0.0f);
    llama_model_free(m);
    return r;
}

static void print_arm(const char * label, const arm & a) {
    printf("%s: loaded %d n_layer %d nextn %d\n", label, a.loaded, a.n_layer, a.n_nextn);
    for (const auto & it : a.buft) printf("    %-38s %s\n", it.first.c_str(), it.second.c_str());
}

int main() {
    const char * path = "test-ollama-qwen35-mtp-mmap.tmp.gguf";
    write_fixture(path);
    llama_backend_init();

    set_env("LLAMA_OLLAMA_COMPAT", nullptr);
    const arm a = run_arm(path, /*mtp*/ true,  /*mmap*/ true);
    const arm b = run_arm(path, /*mtp*/ true,  /*mmap*/ false);
    const arm d = run_arm(path, /*mtp*/ false, /*mmap*/ true);
    set_env("LLAMA_OLLAMA_COMPAT", "0");
    const arm c = run_arm(path, /*mtp*/ true,  /*mmap*/ true);
    set_env("LLAMA_OLLAMA_COMPAT", nullptr);

    print_arm("A mtp=on mmap", a);
    print_arm("B mtp=on no-mmap", b);
    print_arm("C mtp=on compat=0", c);
    print_arm("D mtp=off", d);

    const std::string MAPPED = "CPU_Mapped";
    auto at = [](const arm & x, const char * n) { auto it = x.buft.find(n); return it == x.buft.end() ? std::string() : it->second; };

    check(a.loaded && a.n_layer == 2 && a.n_nextn == 1, "A: head translated, 2 layers, 1 NextN layer");
    check(at(a, "blk.0.attn_q.weight") == MAPPED && at(a, "token_embd.weight") == MAPPED,
          "A: trunk tensors stay in the file mapping (" + at(a, "blk.0.attn_q.weight") + ")");
    check(at(a, "blk.1.attn_q.weight") == MAPPED && at(a, "blk.1.nextn.eh_proj.weight") == MAPPED,
          "A: unshifted MTP tensors stay in the file mapping too");
    check(at(a, "blk.1.nextn.enorm.weight") != MAPPED && at(a, "blk.1.nextn.enorm.weight") != "(none)" &&
          at(a, "blk.1.nextn.hnorm.weight") != MAPPED && at(a, "blk.1.nextn.shared_head_norm.weight") != MAPPED,
          "A: the three shifted norms have owned (non-mapped) storage (" + at(a, "blk.1.nextn.enorm.weight") + ")");
    check(a.enorm_ok && a.hnorm_ok && a.shared_ok, "A: enorm / hnorm / shared_head_norm read file + 1.0");
    check(a.mtp_attn_norm_raw && a.trunk_raw, "A: blk.1.attn_norm and blk.0.attn_norm read the file value (no second shift)");

    check(b.loaded && b.n_layer == 2 && b.n_nextn == 1, "B: --no-mmap: head translated");
    check(at(b, "blk.0.attn_q.weight") != MAPPED && at(b, "blk.1.nextn.enorm.weight") != MAPPED, "B: --no-mmap maps nothing");
    check(b.enorm_ok && b.hnorm_ok && b.shared_ok && b.mtp_attn_norm_raw && b.trunk_raw, "B: --no-mmap: same values as A");

    check(c.loaded && c.n_layer == 1 && c.n_nextn == 0 && at(c, "blk.0.attn_q.weight") == MAPPED,
          "C (opt-out): 1 layer, no NextN, trunk mapped");
    check(d.loaded && d.n_layer == 1 && d.n_nextn == 0 && at(d, "blk.0.attn_q.weight") == MAPPED,
          "D (mtp off): 1 layer, no NextN, trunk mapped");

    llama_backend_free();
    remove(path);
    printf("RESULT: %s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
