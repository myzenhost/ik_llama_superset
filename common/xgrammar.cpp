// XGrammar (mlc-ai/xgrammar, Apache-2.0) as a sampler-shaped output constraint, the second backend in
// common_sampler::cnstr next to llguidance. Row S69-ADD-TGI-ALTERNATIVES-LLGUIDANCE-PAGEDKV-RADIX-SYNTHID-2026-09-27-01.
// SOURCE-READ: upstream/xgrammar @ 97787376f (v0.2.8) include/xgrammar/tokenizer_info.h:28-82,
// compiler.h:57-121, matcher.h:22-215. Written against that C++ API; no code copied.
//
// Request form (the grammar string of a sampling request):
//     %xgrammar <kind> [any_whitespace]\n<body>
// kind = ebnf | json_schema | json | regex | lark | structural_tag. "json" ignores the body (any JSON value).
// JSON kinds use fixed separators ", " / ": " unless "any_whitespace" is given (see llama_sampler_xgr_compile).
//
// Why a second backend: XGrammar compiles a grammar ONCE per (vocabulary, grammar) and caches it
// (GrammarCompiler cache), which pays off when the same schema is served repeatedly; llguidance computes
// masks lazily with near-zero compile time, which suits one-off grammars. Both stay available.

#include "sampling.h"
#include "log.h"
#include "grammar-failclosed.h"

#include <cstdint>
#include <string>
#include <vector>

// The ids XGrammar's TokenizerInfo gets as its stop-token set: every token of the vocabulary the model treats as end-of-generation.
// The matcher lets a stop token through exactly when the grammar can complete, so a stop token missing from this set is MASKED OUT even
// when the grammar is finished (the sampler can then never stop through it) -- or, worse, treated as a plain vocabulary token. Until
// 2026-11-07 only eos and eot went in; llama_vocab_is_eog also covers eom, and every token the loader flagged by name (<|im_end|>,
// <|end|>, <|return|>, <|endoftext|>, <|fim_pad|> ...), and a model whose generation ends on one of those never terminated.
// SOURCE-READ: sglang/python/sglang/srt/constrained/xgrammar_backend.py:216-246 (TokenizerInfo built with the model's whole EOS id list as
// the authoritative stop tokens). Apache-2.0, behaviour only, no code copied. Row S69-ENGREV-SGLANG2-XGRAMMAR-FULL-EOG-STOP-SET-2026-10-27-01.
// Not under LLAMA_USE_XGRAMMAR on purpose: tests/test-server-sglang-picks.cpp checks it without the XGrammar library.
std::vector<int32_t> common_xgrammar_stop_tokens(const llama_vocab * vocab) {
    std::vector<int32_t> stop;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n_vocab; ++t) {
        if (llama_vocab_is_eog(vocab, t)) {
            stop.push_back(t);
        }
    }
    return stop;
}

bool common_xgrammar_parse_header(const std::string & head, std::string & kind, bool & any_ws, int & max_ws, std::string & err) {
    std::vector<std::string> words;
    const std::string rest = head.size() > 9 ? head.substr(9) : std::string();  // after "%xgrammar"
    size_t i = 0;
    while (i < rest.size()) {
        while (i < rest.size() && (rest[i] == ' ' || rest[i] == '\t' || rest[i] == '\r')) ++i;
        size_t j = i;
        while (j < rest.size() && rest[j] != ' ' && rest[j] != '\t' && rest[j] != '\r') ++j;
        if (j > i) words.push_back(rest.substr(i, j - i));
        i = j;
    }
    kind   = words.empty() ? std::string() : words[0];
    any_ws = false;
    max_ws = -1;
    const std::string key = "max_whitespace=";
    for (size_t w = 1; w < words.size(); ++w) {
        if (words[w] == "any_whitespace") {
            any_ws = true;
        } else if (words[w].compare(0, key.size(), key) == 0) {
            const std::string num = words[w].substr(key.size());
            bool ok = !num.empty() && num.size() <= 6;
            for (char c : num) {
                ok = ok && c >= '0' && c <= '9';
            }
            const int n = ok ? std::stoi(num) : 0;
            if (n < 1) {
                err = "xgrammar: max_whitespace needs an integer >= 1, got '" + num + "'";
                return false;
            }
            max_ws = n;
            any_ws = true;
        } else {
            err = "xgrammar: unknown option '" + words[w] + "' (expected any_whitespace | max_whitespace=N)";
            return false;
        }
    }
    return true;
}

#ifdef LLAMA_USE_XGRAMMAR

#    include <xgrammar/xgrammar.h>

#    include <cmath>
#    include <memory>
#    include <mutex>
#    include <optional>
#    include <string>
#    include <vector>

struct llama_sampler_xgr {
    const llama_vocab *                         vocab;
    std::string                                 grammar_str;  // the full request string, for clone/logging
    std::shared_ptr<xgrammar::GrammarMatcher>   matcher;
    std::vector<int32_t>                        bitmask;
    int32_t                                     n_vocab;
    common_grammar_failclosed                   fc;           // sticky: a rejected token fails the request closed
};

// One compiler per vocabulary: it owns the (grammar -> CompiledGrammar) cache. Guarded because server slots
// can initialise samplers from different threads.
static std::mutex                                     g_xgr_mutex;
static const llama_vocab *                            g_xgr_vocab = nullptr;
static std::shared_ptr<xgrammar::GrammarCompiler>     g_xgr_compiler;

static xgrammar::TokenizerInfo llama_sampler_xgr_tokenizer_info(const llama_vocab * vocab) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<std::string> encoded;
    encoded.reserve(n_vocab);
    for (llama_token t = 0; t < n_vocab; ++t) {
        // Control tokens get an empty string, which XGrammar treats as special: a grammar must never be able
        // to "spell" <s> or <|im_end|> through the control token itself.
        encoded.emplace_back(llama_vocab_is_control(vocab, t) ? "" : llama_vocab_get_text(vocab, t));
    }
    xgrammar::VocabType vt = xgrammar::VocabType::RAW;
    switch (llama_vocab_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: vt = xgrammar::VocabType::BYTE_FALLBACK; break;  // "\u2581" + <0xNN> pieces
        case LLAMA_VOCAB_TYPE_BPE: vt = xgrammar::VocabType::BYTE_LEVEL;    break;  // GPT-2 byte-level ("G-dot")
        default:                   vt = xgrammar::VocabType::RAW;           break;
    }
    const std::vector<int32_t> stop = common_xgrammar_stop_tokens(vocab);
    const bool add_prefix_space = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;
    return xgrammar::TokenizerInfo(encoded, vt, n_vocab, stop, add_prefix_space);
}

static std::shared_ptr<xgrammar::GrammarCompiler> llama_sampler_xgr_compiler(const llama_vocab * vocab) {
    std::lock_guard<std::mutex> lock(g_xgr_mutex);
    if (g_xgr_vocab != vocab || !g_xgr_compiler) {
        g_xgr_compiler = std::make_shared<xgrammar::GrammarCompiler>(llama_sampler_xgr_tokenizer_info(vocab));
        g_xgr_vocab    = vocab;
    }
    return g_xgr_compiler;
}

// "%xgrammar <kind>\n<body>" -> compiled grammar; throws std::exception (XGrammar's own, or runtime_error) on
// an unknown kind or an invalid grammar.
static xgrammar::CompiledGrammar llama_sampler_xgr_compile(xgrammar::GrammarCompiler & c, const std::string & s) {
    const size_t nl   = s.find('\n');
    std::string  head = s.substr(0, nl);
    std::string  body = nl == std::string::npos ? std::string() : s.substr(nl + 1);
    // header: "%xgrammar <kind> [any_whitespace] [max_whitespace=N]"
    std::string kind;
    bool        any_whitespace = false;
    int         max_ws         = -1;
    std::string perr;
    if (!common_xgrammar_parse_header(head, kind, any_whitespace, max_ws, perr)) {
        throw std::runtime_error(perr);
    }
    const std::optional<int> max_whitespace_cnt = max_ws > 0 ? std::optional<int>(max_ws) : std::nullopt;
    // JSON kinds default to FIXED separators (", " and ": ", json_schema_converter.cc:1770-1781). XGrammar's own default (any_whitespace = true, compiler.h) lets a model
    // emit unbounded whitespace between tokens -- MEASURED 2026-09-28 on stories260K: a {"n": integer} schema gave
    // '{ \n"n"\n\n\n \n...' until the token budget ran out, valid JSON never finished. The fork's GBNF JSON grammar
    // bounds whitespace too. "any_whitespace" in the header opts back in.
    if (kind == "ebnf")           return c.CompileGrammar(body, "root");
    // CompileJSONSchema(schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order): compiler.h:75-83 (v0.2.8)
    if (kind == "json_schema")    return c.CompileJSONSchema(body, any_whitespace, std::nullopt, std::nullopt, true, max_whitespace_cnt);
    if (kind == "json")           return c.CompileJSONSchema("{}", any_whitespace, std::nullopt, std::nullopt, true, max_whitespace_cnt);  // empty schema = any JSON value
    if (kind == "regex")          return c.CompileRegex(body);
    if (kind == "lark")           return c.CompileLark(body);
    if (kind == "structural_tag") return c.CompileStructuralTag(body);
    throw std::runtime_error("xgrammar: unknown kind '" + kind +
                             "' (expected ebnf | json_schema | json | regex | lark | structural_tag)");
}

static const char * llama_sampler_xgr_name(const llama_sampler * /*smpl*/) {
    return "xgrammar";
}

static void llama_sampler_xgr_accept(llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_xgr *) smpl->ctx;
    if (ctx->matcher && !ctx->fc.failed && !ctx->matcher->IsTerminated()) {
        if (!ctx->matcher->AcceptToken(token)) {
            // Only reachable if a token outside the mask was sampled. The matcher keeps its previous state, so continuing would
            // drift silently from what was generated: FAIL CLOSED instead (row S69-ENGREV-APHRODITE-LLG-FAILCLOSED-2026-10-27-01).
            LOG_ERR("xgrammar: token %d rejected by the grammar -- failing closed (only EOS stays selectable)\n", token);
            ctx->fc.fail("xgrammar matcher rejected a token");
        }
    }
}

static void llama_sampler_xgr_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_xgr *) smpl->ctx;
    if (!ctx->matcher) {
        return;
    }
    llama_token eos = llama_vocab_eot(ctx->vocab);
    if (eos == LLAMA_TOKEN_NULL) {
        eos = llama_vocab_eos(ctx->vocab);
    }
    if (ctx->fc.apply_if_failed(cur_p, eos)) {
        return;
    }
    int64_t  shape[1] = { (int64_t) ctx->bitmask.size() };
    DLTensor t;
    t.data        = ctx->bitmask.data();
    t.device      = DLDevice{ kDLCPU, 0 };
    t.ndim        = 1;
    t.dtype       = xgrammar::GetBitmaskDLType();
    t.shape       = shape;
    t.strides     = nullptr;
    t.byte_offset = 0;
    ctx->matcher->FillNextTokenBitmask(&t);
    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token tok = cur_p->data[i].id;
        const bool allowed = tok >= 0 && tok < ctx->n_vocab &&
                             ((uint32_t) ctx->bitmask[(size_t) tok / 32] >> ((uint32_t) tok % 32)) & 1u;
        if (!allowed) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void llama_sampler_xgr_reset(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_xgr *) smpl->ctx;
    if (ctx->matcher) {
        ctx->matcher->Reset();
        ctx->fc.clear();
    }
}

static llama_sampler * llama_sampler_xgr_clone(const llama_sampler * smpl);

static void llama_sampler_xgr_free(llama_sampler * smpl) {
    delete (llama_sampler_xgr *) smpl->ctx;
}

static llama_sampler_i llama_sampler_xgr_i = {
    /* .name              = */ llama_sampler_xgr_name,
    /* .accept            = */ llama_sampler_xgr_accept,
    /* .apply             = */ llama_sampler_xgr_apply,
    /* .reset             = */ llama_sampler_xgr_reset,
    /* .clone             = */ llama_sampler_xgr_clone,
    /* .free              = */ llama_sampler_xgr_free,
    /* .backend_init      = */ NULL,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ NULL,
    /* .backend_set_input = */ NULL,
    /* .backend_reset     = */ NULL,
    /* .copy_state        = */ NULL,
};

static llama_sampler * llama_sampler_xgr_clone(const llama_sampler * smpl) {
    const auto * src = (const llama_sampler_xgr *) smpl->ctx;
    auto *       dst = new llama_sampler_xgr{ src->vocab, src->grammar_str, nullptr, src->bitmask, src->n_vocab };
    dst->fc = src->fc;
    if (src->matcher) {
        dst->matcher = std::make_shared<xgrammar::GrammarMatcher>(src->matcher->Fork());  // deep copy of state
    }
    return llama_sampler_init(&llama_sampler_xgr_i, dst);
}

bool llama_sampler_xgr_failed(const llama_sampler * smpl, std::string * err) {
    if (smpl == nullptr || smpl->iface != &llama_sampler_xgr_i) {
        return false;
    }
    const auto * ctx = (const llama_sampler_xgr *) smpl->ctx;
    if (ctx == nullptr || !ctx->fc.failed) {
        return false;
    }
    if (err) {
        *err = ctx->fc.error;
    }
    return true;
}

llama_sampler * llama_sampler_init_xgr(const llama_vocab * vocab, const char * grammar_str) {
    try {
        auto compiler = llama_sampler_xgr_compiler(vocab);
        auto compiled = llama_sampler_xgr_compile(*compiler, grammar_str);
        auto * ctx    = new llama_sampler_xgr;
        ctx->vocab       = vocab;
        ctx->grammar_str = grammar_str;
        ctx->n_vocab     = llama_vocab_n_tokens(vocab);
        ctx->matcher     = std::make_shared<xgrammar::GrammarMatcher>(compiled);
        ctx->bitmask.assign((size_t) xgrammar::GetBitmaskSize(ctx->n_vocab), 0);
        return llama_sampler_init(&llama_sampler_xgr_i, ctx);
    } catch (const std::exception & e) {
        // nullptr, not an empty sampler: common_sampler_init refuses the request (400) instead of running it
        // unconstrained.
        LOG_ERR("xgrammar: %s\n", e.what());
        return nullptr;
    }
}

#else

llama_sampler * llama_sampler_init_xgr(const llama_vocab *, const char *) {
    LOG_WRN("xgrammar (cmake -DLLAMA_XGRAMMAR=ON) is not enabled");
    return nullptr;
}

bool llama_sampler_xgr_failed(const llama_sampler *, std::string *) {
    return false;
}

#endif  // LLAMA_USE_XGRAMMAR
