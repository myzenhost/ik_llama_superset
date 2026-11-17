// Ollama qwen35 blob: with MTP requested, the embedded mtp.* head must be mapped onto the NextN layer
// (llama_model_loader::apply_ollama_qwen35_mtp_compat). Metadata-only loads (no_alloc), so the 17 GB blob costs
// no RAM and no tensor reads; the numeric half (+1.0 on enorm / hnorm / shared_head_norm, and draft acceptance)
// is tests/test-ollama-qwen35-mtp.py, which runs the model.
//
// Arms:
//   A  mtp=true                               -> n_layer 65, 1 NextN layer, blk.64.* and blk.64.nextn.* tensors exist
//   B  mtp=false                              -> unchanged: n_layer 64, no NextN layer (the translation must not fire)
//   C  mtp=true, LLAMA_OLLAMA_COMPAT=0 (opt-out; the loader then runs exactly the pre-change code)
//                                             -> n_layer 64, no NextN layer: the MTP head is unusable. This is the
//                                                fail-before state.
//
// Fixture: LLAMA_TEST_OLLAMA_QWEN35_BLOB, else the ollama qwen3.6:27b manifest under OLLAMA_MODELS. Exit 77 = skipped.

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

static std::string find_blob() {
    if (const char * p = getenv("LLAMA_TEST_OLLAMA_QWEN35_BLOB")) {
        return p;
    }
    std::string store = getenv("OLLAMA_MODELS") ? getenv("OLLAMA_MODELS") : "models/ollama"; // portability-ok: default store, env override
    std::ifstream f(store + "/manifests/registry.ollama.ai/library/qwen3.6/27b");
    if (!f) {
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string m = ss.str();
    const size_t at = m.find("application/vnd.ollama.image.model");
    if (at == std::string::npos) {
        return "";
    }
    const size_t d = m.find("sha256:", at);
    if (d == std::string::npos) {
        return "";
    }
    return store + "/blobs/sha256-" + m.substr(d + 7, 64);
}

struct arm_result {
    bool loaded = false;
    int  n_layer = -1;
    int  n_nextn = -1;
    bool has_eh_proj = false, has_enorm = false, has_hnorm = false, has_shared_norm = false, has_attn_q = false, has_mtp_name = false;
    long long eh0 = 0, eh1 = 0;
};

static arm_result run_arm(const std::string & blob, bool mtp) {
    arm_result r;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.no_alloc     = true;
    mp.mtp          = mtp;
    llama_model * m = llama_model_load_from_file(blob.c_str(), mp);
    if (!m) {
        return r;
    }
    r.loaded  = true;
    r.n_layer = llama_model_n_layer(m);
    r.n_nextn = llama_model_n_nextn_layer(m);
    if (ggml_tensor * t = llama_get_model_tensor(m, "blk.64.nextn.eh_proj.weight")) {
        r.has_eh_proj = true;
        r.eh0 = t->ne[0];
        r.eh1 = t->ne[1];
    }
    r.has_enorm       = llama_get_model_tensor(m, "blk.64.nextn.enorm.weight") != nullptr;
    r.has_hnorm       = llama_get_model_tensor(m, "blk.64.nextn.hnorm.weight") != nullptr;
    r.has_shared_norm = llama_get_model_tensor(m, "blk.64.nextn.shared_head_norm.weight") != nullptr;
    r.has_attn_q      = llama_get_model_tensor(m, "blk.64.attn_q.weight") != nullptr;
    r.has_mtp_name    = llama_get_model_tensor(m, "mtp.fc.weight") != nullptr;
    llama_model_free(m);
    return r;
}

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    fflush(stdout);
    g_fail += ok ? 0 : 1;
}

int main() {
    const std::string blob = find_blob();
    if (blob.empty() || !std::ifstream(blob)) {
        printf("SKIP: no ollama qwen35 blob (set LLAMA_TEST_OLLAMA_QWEN35_BLOB)\n");
        return 77;
    }
    llama_backend_init();

    set_env("LLAMA_OLLAMA_COMPAT", nullptr);
    const arm_result a = run_arm(blob, true);
    const arm_result b = run_arm(blob, false);
    set_env("LLAMA_OLLAMA_COMPAT", "0");
    const arm_result c = run_arm(blob, true);
    set_env("LLAMA_OLLAMA_COMPAT", nullptr);

    printf("A mtp=on:             loaded %d n_layer %d nextn %d eh_proj %d [%lld,%lld] enorm %d hnorm %d shared_norm %d attn_q %d mtp.* %d\n",
           a.loaded, a.n_layer, a.n_nextn, a.has_eh_proj, a.eh0, a.eh1, a.has_enorm, a.has_hnorm, a.has_shared_norm, a.has_attn_q, a.has_mtp_name);
    printf("B mtp=off:            loaded %d n_layer %d nextn %d\n", b.loaded, b.n_layer, b.n_nextn);
    printf("C mtp=on, compat=0:   loaded %d n_layer %d nextn %d eh_proj %d\n", c.loaded, c.n_layer, c.n_nextn, c.has_eh_proj);

    check(a.loaded && a.n_layer == 65 && a.n_nextn == 1, "A: mtp=on maps the head -> 65 layers, 1 NextN layer");
    check(a.has_eh_proj && a.eh0 == 10240 && a.eh1 == 5120, "A: blk.64.nextn.eh_proj.weight is [2*n_embd, n_embd] = [10240, 5120]");
    check(a.has_enorm && a.has_hnorm && a.has_shared_norm, "A: enorm, hnorm, shared_head_norm present");
    check(a.has_attn_q, "A: the MTP block's attention weights load as blk.64.attn_q");
    check(!a.has_mtp_name, "A: no tensor is left under an mtp.* name");
    check(b.loaded && b.n_layer == 64 && b.n_nextn == 0, "B (control): mtp=off load is unchanged, 64 layers, no NextN");
    check(c.loaded && c.n_layer == 64 && c.n_nextn == 0 && !c.has_eh_proj, "C (opt-out = pre-change behaviour): no NextN layer");

    llama_backend_free();
    printf("RESULT: %s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
