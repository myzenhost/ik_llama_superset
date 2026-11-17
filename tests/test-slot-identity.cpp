// A slot file saved under checkpoint A must be REFUSED when restored under checkpoint B of the same architecture, naming the
// mismatched field; before this a restore returned success and the continuation was garbage (llama-context.cpp state read checked only
// the arch string; the TODO "add more model-specific info which should prevent loading the session file if not identical" stood).
//
// Row S69-ENGREV-FERROX-SLOT-FILE-MODEL-IDENTITY-2026-10-27-01 (design: ferrox slots/identity.rs, Jan slots.rs; nothing copied).
//
// usage: test-slot-identity <model.gguf>
//   any small GGUF works; tests/make_tiny_llama_gguf.py <tree-root> out.gguf builds one without a download. The test writes a
//   second copy of it into a temp file with ONE byte flipped inside the first tensor (same arch, same shapes, different weights).
// PASS: (1) save under A + restore under A succeeds and writes <file>.ident; (2) restore under B (one flipped tensor byte, inside the
// sampled window of that tensor) returns 0 and the check names field 'fingerprint'; (3) a tampered sidecar (n_layer changed) is named
// as 'n_layer' (shape is checked before the fingerprint); (4) a file with NO sidecar restores (legacy file, check rc 2); (5) a sidecar
// missing a field is a mismatch, absence is never agreement.
#include "llama.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s  (line %d)\n", msg, __LINE__); g_fail++; } else { std::printf("ok:   %s\n", msg); } } while (0)

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); }
}

static llama_context * make_ctx(llama_model * m) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 32; cp.n_ubatch = 32;
    return llama_init_from_model(m, cp);
}

static bool flip_byte_in_first_tensor(const std::string & src, const std::string & dst) {
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(src.c_str(), gp);
    if (!g) return false;
    const size_t pos = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, 0) + 1;
    gguf_free(g);
    std::ifstream in(src, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (pos >= bytes.size()) return false;
    bytes[pos] ^= 0x5a;
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), (std::streamsize) bytes.size());
    return (bool) out;
}

static void write_text(const std::string & path, const std::string & text) {
    std::ofstream f(path, std::ios::trunc);
    f << text;
}
static std::string read_text(const std::string & path) {
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    const std::string model_b_path = std::string(argv[1]) + ".slotid_b.gguf";
    const std::string slot_path    = std::string(argv[1]) + ".slotid.bin";
    const std::string ident_path   = slot_path + ".ident";
    if (!flip_byte_in_first_tensor(argv[1], model_b_path)) { std::fprintf(stderr, "cannot build model B\n"); return 2; }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * ma = llama_model_load_from_file(argv[1], mp);
    llama_model * mb = llama_model_load_from_file(model_b_path.c_str(), mp);
    if (!ma || !mb) { std::fprintf(stderr, "cannot load models\n"); return 2; }
    llama_context * ca = make_ctx(ma);
    llama_context * cb = make_ctx(mb);
    if (!ca || !cb) { std::fprintf(stderr, "cannot create contexts\n"); return 2; }

    // decode a few tokens in seq 0 of A so there is a state to save
    std::vector<llama_token> toks = { 1, 3, 4, 5, 6 };
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ca, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 2; }

    const size_t nsave = llama_state_seq_save_file(ca, slot_path.c_str(), 0, toks.data(), toks.size());
    CHECK(nsave > 0, "save under A succeeds");
    CHECK(!read_text(ident_path).empty(), "sidecar .ident written");

    char why[512];
    std::vector<llama_token> out(16); size_t n_out = 0;

    // (1) same model: identical, restore works
    CHECK(llama_state_seq_file_identity_check(ca, slot_path.c_str(), why, sizeof(why)) == 0, "identity check A->A is identical");
    CHECK(llama_state_seq_load_file(ca, slot_path.c_str(), 0, out.data(), out.size(), &n_out) > 0 && n_out == toks.size(), "restore under A succeeds");

    // (2) other checkpoint, same arch + shapes: refused, field named
    int rc = llama_state_seq_file_identity_check(cb, slot_path.c_str(), why, sizeof(why));
    std::printf("      B says: %s\n", why);
    CHECK(rc == 1, "identity check A->B is a MISMATCH");
    CHECK(std::strstr(why, "fingerprint") != nullptr, "mismatch names field 'fingerprint'");
    CHECK(llama_state_seq_load_file(cb, slot_path.c_str(), 0, out.data(), out.size(), &n_out) == 0, "restore under B is REFUSED (returns 0)");
    CHECK(llama_state_seq_load_file(cb, slot_path.c_str(), 0, nullptr, 0, &n_out) == 0, "token-count peek under B is refused too");

    // (3) tampered shape field is named before the fingerprint
    const std::string good = read_text(ident_path);
    {
        std::string t = good;
        const size_t p = t.find("n_layer=");
        if (p != std::string::npos) { t.insert(p + 8, "9"); }
        write_text(ident_path, t);
        rc = llama_state_seq_file_identity_check(ca, slot_path.c_str(), why, sizeof(why));
        std::printf("      tampered says: %s\n", why);
        CHECK(rc == 1 && std::strstr(why, "n_layer") != nullptr, "tampered n_layer is named (shape before fingerprint)");
    }
    // (5) a missing field is a mismatch
    {
        std::string t;
        size_t pos = 0;
        while (pos < good.size()) {
            size_t e = good.find('\n', pos); if (e == std::string::npos) e = good.size();
            std::string line = good.substr(pos, e - pos);
            if (line.rfind("fingerprint=", 0) != 0) { t += line + "\n"; }
            pos = e + 1;
        }
        write_text(ident_path, t);
        rc = llama_state_seq_file_identity_check(ca, slot_path.c_str(), why, sizeof(why));
        CHECK(rc == 1 && std::strstr(why, "fingerprint") != nullptr, "sidecar lacking a field is a mismatch (absence is not agreement)");
    }
    // (4) no sidecar at all: legacy file, cannot be proven wrong, restores
    std::remove(ident_path.c_str());
    CHECK(llama_state_seq_file_identity_check(ca, slot_path.c_str(), why, sizeof(why)) == 2, "no sidecar -> rc 2 (legacy)");
    CHECK(llama_state_seq_load_file(ca, slot_path.c_str(), 0, out.data(), out.size(), &n_out) > 0, "legacy file without sidecar still restores");

    llama_free(ca); llama_free(cb);
    llama_model_free(ma); llama_model_free(mb);
    std::remove(slot_path.c_str()); std::remove(model_b_path.c_str());
    llama_backend_free();
    if (g_fail) { std::printf("test-slot-identity: %d FAILED\n", g_fail); return 1; }
    std::printf("test-slot-identity: OK\n");
    return 0;
}
