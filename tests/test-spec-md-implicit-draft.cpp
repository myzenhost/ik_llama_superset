// `-md X` without `--spec-type` must give a draft-model stage (row S69-FORK-SPEC-MD-WITHOUT-TYPE-LOADS-DRAFT-NEVER-DRAFTS-2026-10-27-01).
// MEASURED 2026-10-27 on :8081: the draft model loaded (+1 GB VRAM) and nothing was ever drafted, because the stage chain
// defaults to NONE and common_speculative_types_from_gguf() detects only MTP / DFlash / DSpark, so a plain causal draft
// GGUF left the chain empty. common_models_handler_apply() now falls back to a 'draft' stage in that one case.
//
// Each case writes a tiny GGUF to the temp dir and runs common_models_handler_init/apply on it (no model load):
//   plain draft, no type        -> exactly one stage, DRAFT                     (fails before the fix: chain empty)
//   MTP draft, no type          -> MTP from the GGUF, no DRAFT stage             (inference must still win)
//   plain draft, --spec-type none -> no chain                                    (explicit opt-out is kept)
//   plain draft, explicit ngram -> ngram only, no DRAFT stage                    (an explicit type is never changed)
//   no draft model              -> no chain                                      (nothing named, nothing added)
// GATE-CLAIM-OK: unit test, no resource claims.
#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "speculative.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        n_fail++;
    }
}

// a GGUF with general.architecture=llama, llama.block_count=1 and one tensor; with_mtp names the MTP head tensor
static std::string write_gguf(const char * name, bool with_mtp) {
    const std::string path = (std::filesystem::temp_directory_path() / name).string();

    ggml_init_params ip = { /* .mem_size = */ 16 * 1024, /* .mem_buffer = */ nullptr, /* .no_alloc = */ false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, with_mtp ? "blk.0.nextn.eh_proj.weight" : "blk.0.attn_norm.weight");
    ggml_set_zero(t);

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_u32(g, "llama.block_count", 1);
    gguf_add_tensor(g, t);
    gguf_write_to_file(g, path.c_str(), /* only_meta = */ false);

    gguf_free(g);
    ggml_free(ctx);
    return path;
}

static common_params_speculative run(const std::string & draft_path, const std::vector<common_speculative_stage_params> & stages) {
    common_params params;
    params.speculative.mparams_dft.path = draft_path;
    params.speculative.stages           = stages;
    common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
    common_models_handler_apply(handler, params);
    return params.speculative;
}

int main() {
    const std::string plain = write_gguf("test-spec-md-implicit-plain.gguf", false);
    const std::string mtp   = write_gguf("test-spec-md-implicit-mtp.gguf",   true);

    // positive control for the GGUF writer: the MTP file must read back as MTP, the plain one as nothing
    check(common_speculative_types_from_gguf(mtp).size() == 1 &&
          common_speculative_types_from_gguf(mtp)[0] == COMMON_SPECULATIVE_TYPE_MTP, "control: MTP GGUF detected as MTP");
    check(common_speculative_types_from_gguf(plain).empty(), "control: plain GGUF detected as no type");

    {
        const auto sp = run(plain, {});
        const auto st = sp.get_resolved_stages();
        check(st.size() == 1 && st[0].type == COMMON_SPECULATIVE_TYPE_DRAFT, "plain -md, no type -> one DRAFT stage");
    }
    {
        const auto sp = run(mtp, {});
        check(sp.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) && !sp.has_stage_type(COMMON_SPECULATIVE_TYPE_DRAFT),
              "MTP -md, no type -> MTP, no DRAFT");
    }
    {
        common_speculative_stage_params none_stage;
        none_stage.type = COMMON_SPECULATIVE_TYPE_NONE;
        const auto sp = run(plain, { none_stage });
        check(!sp.has_stage_chain(), "plain -md, --spec-type none -> no chain");
    }
    {
        common_speculative_stage_params ngram;
        ngram.type = COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE;
        const auto sp = run(plain, { ngram });
        const auto st = sp.get_resolved_stages();
        check(st.size() == 1 && st[0].type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, "plain -md, explicit ngram -> ngram only");
    }
    {
        const auto sp = run("", {});
        check(!sp.has_stage_chain(), "no -md -> no chain");
    }

    std::filesystem::remove(plain);
    std::filesystem::remove(mtp);

    printf("%d failed\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
