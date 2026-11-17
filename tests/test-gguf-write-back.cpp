// GGUF write-back: a gguf_context obtained by READING a file is written out again with gguf_write_to_file / gguf_write_to_buf.
// The read loop used to leave each gguf_tensor_info's embedded ggml_tensor default-initialised, so t.buffer and t.data were
// indeterminate; the writer branches on t.buffer and called ggml_backend_tensor_get on garbage (SIGSEGV, backtrace in
// JusticianAI temp/arex1014/imxhess1018/diag2.out). Row S69-FORK-GGUF-READ-TENSOR-INFO-UNINITIALISED-BUFFER-2026-10-18-01.
//
// Cases (run all with no argument, or one by name so a crash in one does not hide the others):
//   full      read with no_alloc = false into a ggml context; a full write reproduces the original bytes (file and buffer)
//   noalloc   read with no_alloc = true; a full write fails cleanly (returns false, writes no byte), a meta write is exact
//   metaonly  read with no ggml context at all; same contract as noalloc
//   setdata   read with no_alloc = true, then gguf_set_tensor_data on every tensor; a full write reproduces the original
// GATE-CLAIM-OK: unit test, no resource claims.
#include "ggml.h"
#include "../ggml/src/ggml-impl.h" // gguf_write_to_buf (non-public, as in test-gguf.cpp)
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static const char * TMP = "test-gguf-write-back.tmp.gguf";

static std::vector<uint8_t> read_all(const char * fname) {
    std::vector<uint8_t> w;
    FILE * fp = fopen(fname, "rb");
    if (fp) { int c; while ((c = fgetc(fp)) != EOF) w.push_back((uint8_t) c); fclose(fp); }
    return w;
}

// the reference file: 3 kv and 3 tensors, one of them 13 bytes (I8) so the data section has padding between tensors
static std::vector<uint8_t> make_reference() {
    ggml_init_params ip = { /*.mem_size =*/ 16*ggml_tensor_overhead() + 4096, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 13);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(t0, "t0");
    ggml_set_name(t1, "t1");
    ggml_set_name(t2, "t2");
    for (int i = 0; i < 12; i++) ((float *) t0->data)[i] = 0.5f*i - 1.0f;
    for (int i = 0; i < 13; i++) ((int8_t *) t1->data)[i] = (int8_t) (7*i - 40);
    for (int i = 0; i < 8; i++)  ((float *) t2->data)[i] = -0.25f*i;

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_u32(g, "test.u32", 7);
    const float arr[3] = { 1.5f, -2.0f, 3.25f };
    gguf_set_arr_data(g, "test.arr_f32", GGUF_TYPE_FLOAT32, arr, 3);
    gguf_add_tensor(g, t0);
    gguf_add_tensor(g, t1);
    gguf_add_tensor(g, t2);

    std::vector<int8_t> buf;
    gguf_write_to_buf(g, buf, /*only_meta =*/ false);
    gguf_free(g);
    ggml_free(ctx);
    return std::vector<uint8_t>(buf.begin(), buf.end());
}

static gguf_context * load(const std::vector<uint8_t> & f, bool no_alloc, ggml_context ** ctx) {
    gguf_init_params ip = { /*.no_alloc =*/ no_alloc, /*.ctx =*/ ctx };
    return gguf_init_from_buffer(f.data(), f.size(), ip);
}

static bool same(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, const char * what) {
    if (a.size() != b.size()) {
        fprintf(stderr, "%s: size %zu, want %zu\n", what, a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "%s: first difference at byte %zu (%u vs %u)\n", what, i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// full write to a file and to a buffer, both must equal ref
static void check_full_write(const gguf_context * g, const std::vector<uint8_t> & ref, const char * what) {
    remove(TMP);
    CHECK(gguf_write_to_file(g, TMP, /*only_meta =*/ false), "%s: full gguf_write_to_file returned false", what);
    CHECK(same(read_all(TMP), ref, what), "%s: file bytes differ from the reference", what);
    remove(TMP);

    std::vector<int8_t> buf;
    gguf_write_to_buf(g, buf, /*only_meta =*/ false);
    CHECK(same(std::vector<uint8_t>(buf.begin(), buf.end()), ref, what), "%s: buffer bytes differ from the reference", what);
}

// a context that holds no tensor data: the full write must refuse without writing, the meta write must be exact
static void check_meta_contract(const gguf_context * g, const std::vector<uint8_t> & ref, const char * what) {
    remove(TMP);
    const bool ok = gguf_write_to_file(g, TMP, /*only_meta =*/ false);
    CHECK(!ok, "%s: full write of a context without tensor data returned true", what);
    const std::vector<uint8_t> partial = read_all(TMP);
    CHECK(partial.empty(), "%s: refused full write still left %zu bytes in the file", what, partial.size());
    remove(TMP);

    const size_t meta = gguf_get_meta_size(g);
    CHECK(meta > 0 && meta < ref.size(), "%s: meta size %zu (reference %zu)", what, meta, ref.size());
    CHECK(gguf_write_to_file(g, TMP, /*only_meta =*/ true), "%s: meta-only write returned false", what);
    const std::vector<uint8_t> m = read_all(TMP);
    remove(TMP);
    CHECK(m.size() == meta && same(m, std::vector<uint8_t>(ref.begin(), ref.begin() + meta), what),
        "%s: meta-only bytes differ from the reference header", what);
}

static void case_full(const std::vector<uint8_t> & ref) {
    ggml_context * ctx = nullptr;
    gguf_context * g = load(ref, /*no_alloc =*/ false, &ctx);
    CHECK(g != nullptr && ctx != nullptr, "full: load failed");
    if (!g) { return; }
    check_full_write(g, ref, "full");
    gguf_free(g);
    ggml_free(ctx);
}

static void case_noalloc(const std::vector<uint8_t> & ref) {
    ggml_context * ctx = nullptr;
    gguf_context * g = load(ref, /*no_alloc =*/ true, &ctx);
    CHECK(g != nullptr && ctx != nullptr, "noalloc: load failed");
    if (!g) { return; }
    check_meta_contract(g, ref, "noalloc");
    gguf_free(g);
    ggml_free(ctx);
}

static void case_metaonly(const std::vector<uint8_t> & ref) {
    gguf_context * g = load(ref, /*no_alloc =*/ true, nullptr);
    CHECK(g != nullptr, "metaonly: load failed");
    if (!g) { return; }
    check_meta_contract(g, ref, "metaonly");
    gguf_free(g);
}

static void case_setdata(const std::vector<uint8_t> & ref) {
    gguf_context * g = load(ref, /*no_alloc =*/ true, nullptr);
    CHECK(g != nullptr, "setdata: load failed");
    if (!g) { return; }
    // the data section of the reference: each tensor's own copy, so the context points at memory the test owns
    const size_t base = gguf_get_data_offset(g);
    std::vector<std::vector<uint8_t>> own(gguf_get_n_tensors(g));
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const size_t off = base + gguf_get_tensor_offset(g, i);
        own[i].assign(ref.begin() + off, ref.begin() + off + gguf_get_tensor_size(g, i));
        gguf_set_tensor_data(g, gguf_get_tensor_name(g, i), own[i].data());
    }
    check_full_write(g, ref, "setdata");
    gguf_free(g);
}

int main(int argc, char ** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    const std::vector<uint8_t> ref = make_reference();
    CHECK(ref.size() > 64, "reference file is %zu bytes", ref.size());

    if (which == "all" || which == "full")     { case_full(ref);     printf("case full done\n"); }
    if (which == "all" || which == "noalloc")  { case_noalloc(ref);  printf("case noalloc done\n"); }
    if (which == "all" || which == "metaonly") { case_metaonly(ref); printf("case metaonly done\n"); }
    if (which == "all" || which == "setdata")  { case_setdata(ref);  printf("case setdata done\n"); }

    printf("%s: %d failure(s) [%s]\n", fails ? "FAIL" : "OK", fails, which.c_str());
    return fails ? 1 : 0;
}
