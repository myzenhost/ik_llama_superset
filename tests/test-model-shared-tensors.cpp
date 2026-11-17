// A draft head that leaves out token_embd / output_norm / output (convert_hf_to_gguf.py --mtp --mtp-shared-embd)
// must load against an already loaded target through llama_model_params::model_shared and then behave exactly
// like a draft that carries its own copies. Test written here for this fork.
// SOURCE-READ: tmp/_unsl-branches_g1.out (git show f36eda84 of the unslothai/llama.cpp remote, MIT)
//
// usage: test-model-shared-tensors <dir with the ggufs from make_tiny_shared_pair_gguf.py>
// exit 0 = every check passed. CPU only, tiny F32 models, milliseconds.
//
// Row: S69-UNSLOTH-ADOPTABLE-BRANCHES-NOT-YET-PORTED-01.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

static std::string g_log;

static void log_cb(ggml_log_level, const char * text, void *) {
    g_log += text;
}

static int count_of(const std::string & hay, const char * needle) {
    int n = 0;
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + 1)) {
        n++;
    }
    return n;
}

// the field is detected, not named, so the same source also builds against a tree that lacks it: that is
// how the "before" run is made
template <class P>
static auto set_shared(P & p, const llama_model * m, int) -> decltype(p.model_shared = m, bool()) {
    p.model_shared = m;
    return true;
}
template <class P>
static bool set_shared(P &, const llama_model *, long) {
    return false;
}

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

static llama_model * load(const std::string & path, const llama_model * shared, bool * had_field) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = true;
    const bool f = set_shared(mp, shared, 0);
    if (had_field) {
        *had_field = f;
    }
    return llama_model_load_from_file(path.c_str(), mp);
}

static std::vector<float> logits_of(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 16; cp.n_ubatch = 16; cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return {};
    }
    const int T = 6;
    const llama_token toks[T] = { 3, 5, 7, 9, 11, 13 };
    llama_batch b = llama_batch_init(T, 0, 1);
    b.n_tokens = T;
    for (int i = 0; i < T; i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    std::vector<float> res;
    if (llama_decode(ctx, b) == 0) {
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * lg = llama_get_logits(ctx);
        if (lg) {
            res.assign(lg, lg + (size_t) T * nv);
        }
    }
    llama_batch_free(b);
    llama_free(ctx);
    return res;
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        m = std::fmax(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf dir>\n", argv[0]);
        return 64;
    }
    const std::string dir = std::string(argv[1]) + "/";

    llama_backend_init();
    llama_log_set(log_cb, nullptr);

    bool has_field = false;
    llama_model * target = load(dir + "target.gguf", nullptr, &has_field);
    CHECK(target != nullptr, "target must load");
    if (!target) { return 2; }
    printf("info  llama_model_params::model_shared %s in this build\n", has_field ? "PRESENT" : "ABSENT");

    // control: a draft that carries its own copies loads and decodes
    llama_model * d_full = load(dir + "draft_full.gguf", nullptr, nullptr);
    CHECK(d_full != nullptr, "self-contained draft must load (control)");
    const std::vector<float> lg_full = d_full ? logits_of(d_full) : std::vector<float>();
    CHECK(!lg_full.empty(), "self-contained draft must decode (control)");
    if (d_full) { llama_model_free(d_full); }

    // the target's own logits, before any draft borrows from it
    const std::vector<float> lg_target_before = logits_of(target);
    CHECK(!lg_target_before.empty(), "target must decode");

    // a shared draft loaded on its own is refused with a message that says what to do
    {
        g_log.clear();
        llama_model * m = load(dir + "draft_shared.gguf", nullptr, nullptr);
        CHECK(m == nullptr, "a draft without token_embd must not load on its own");
        CHECK(g_log.find("load it as a draft of its target") != std::string::npos, "the refusal must say how to load it");
        if (m) { llama_model_free(m); }
    }

    // THE FEATURE: loaded against the target it works and equals the self-contained draft bit for bit
    {
        g_log.clear();
        llama_model * m = load(dir + "draft_shared.gguf", target, nullptr);
        CHECK(m != nullptr, "shared draft must load with model_shared = target");
        if (m) {
            CHECK(count_of(g_log, "taken from the target model") == 3, "all three tensors must be borrowed, saw %d", count_of(g_log, "taken from the target model"));
            const std::vector<float> lg = logits_of(m);
            CHECK(same(lg, lg_full), "shared draft logits must equal the self-contained draft (max abs diff %g)", max_abs_diff(lg, lg_full));
            llama_model_free(m);
            // the draft borrowed, so freeing it must not have freed the target's tensors
            const std::vector<float> lg_after = logits_of(target);
            CHECK(same(lg_after, lg_target_before), "target must be intact after the borrowing draft is freed");
        }
    }

    // a file that ships its own tensors is never touched by model_shared
    {
        g_log.clear();
        llama_model * m = load(dir + "draft_full.gguf", target, nullptr);
        CHECK(m != nullptr, "self-contained draft must still load with model_shared set");
        if (m) {
            CHECK(count_of(g_log, "taken from the target model") == 0, "a draft with its own tensors must borrow nothing, saw %d", count_of(g_log, "taken from the target model"));
            CHECK(same(logits_of(m), lg_full), "self-contained draft logits unchanged by model_shared");
            llama_model_free(m);
        }
    }

    // shape disagreement is refused
    {
        g_log.clear();
        llama_model * m = load(dir + "draft_badshape.gguf", target, nullptr);
        CHECK(m == nullptr, "a draft whose n_embd differs from the target must be refused");
        CHECK(g_log.find("disagree") != std::string::npos, "the refusal must name the disagreement");
        if (m) { llama_model_free(m); }
    }

    llama_model_free(target);

    // tied head: the target has no output.weight, the shared draft must follow its tied head
    {
        llama_model * tt = load(dir + "target_tied.gguf", nullptr, nullptr);
        CHECK(tt != nullptr, "tied target must load");
        llama_model * df = load(dir + "draft_full_tied.gguf", nullptr, nullptr);
        CHECK(df != nullptr, "tied self-contained draft must load (control)");
        const std::vector<float> lg_ref = df ? logits_of(df) : std::vector<float>();
        if (df) { llama_model_free(df); }
        if (tt) {
            g_log.clear();
            llama_model * m = load(dir + "draft_shared_tied.gguf", tt, nullptr);
            CHECK(m != nullptr, "shared draft must load against a tied-head target");
            if (m) {
                CHECK(same(logits_of(m), lg_ref), "shared draft logits must equal the tied self-contained draft");
                llama_model_free(m);
            }
            llama_model_free(tt);
        }
    }

    llama_backend_free();
    printf(g_fail ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
