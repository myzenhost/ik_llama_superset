// LLM_KV(arch, suffix) must APPEND ".<suffix>" to the formatted key, as upstream llama-arch.cpp does.
//
// The fork used to pass the suffix as a second format() argument. Every key whose name has no second
// %s -- "tokenizer.chat_template" in particular -- dropped it silently, so
// llama_model_chat_template(model, "rerank" / "tool_use") returned the DEFAULT template. Measured:
// llama-server /v1/rerank scored the raw Jinja text (986 tokens) for every document.

#include "../src/llama-arch.h"

#include <cstdio>
#include <string>

static int n_fail = 0;

static void expect(const std::string & got, const std::string & want, const char * what) {
    const bool ok = got == want;
    n_fail += !ok;
    printf("%-4s %-52s got '%s'%s%s%s\n", ok ? "ok" : "FAIL", what, got.c_str(),
           ok ? "" : " want '", ok ? "" : want.c_str(), ok ? "" : "'");
}

int main() {
    // the case that was broken: an arch-less name plus a suffix
    expect(LLM_KV(LLM_ARCH_QWEN3, "rerank")(LLM_KV_TOKENIZER_CHAT_TEMPLATE),
           "tokenizer.chat_template.rerank", "named chat template (rerank)");
    expect(LLM_KV(LLM_ARCH_LLAMA, "tool_use")(LLM_KV_TOKENIZER_CHAT_TEMPLATE),
           "tokenizer.chat_template.tool_use", "named chat template (tool_use)");

    // unchanged behaviour without a suffix
    expect(LLM_KV(LLM_ARCH_QWEN3)(LLM_KV_TOKENIZER_CHAT_TEMPLATE),
           "tokenizer.chat_template", "default chat template");
    expect(LLM_KV(LLM_ARCH_QWEN3)(LLM_KV_POOLING_TYPE),
           "qwen3.pooling_type", "arch-formatted key");
    expect(LLM_KV(LLM_ARCH_QWEN3)(LLM_KV_CLASSIFIER_OUTPUT_LABELS),
           "qwen3.classifier.output_labels", "classifier labels key");

    // a suffix on an arch-formatted key keeps the arch AND appends the suffix
    expect(LLM_KV(LLM_ARCH_QWEN3, "x")(LLM_KV_POOLING_TYPE),
           "qwen3.pooling_type.x", "arch-formatted key + suffix");

    // CONTROL: the named key must differ from the default one, or the lookup cannot tell them apart
    const bool distinct = LLM_KV(LLM_ARCH_QWEN3, "rerank")(LLM_KV_TOKENIZER_CHAT_TEMPLATE) !=
                          LLM_KV(LLM_ARCH_QWEN3)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    n_fail += !distinct;
    printf("%-4s named and default template keys are distinct\n", distinct ? "ok" : "FAIL");

    printf("%d failed\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
