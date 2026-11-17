// S69-CHAT-TEMPLATES-MONOLITH-LACKS-14-UPSTREAM-2026-09-27-01
// Differential probe: applies EVERY chat-template name in the fork+upstream union to one fixed
// 3-message conversation via the PUBLIC llama_chat_apply_template C API, and prints
// "name\t<len-or-error>\t<escaped output>" for each. Run once before the llama-chat.cpp
// consolidation and once after; the shared names must produce byte-identical lines (or the diff
// must be justified in the ledger), the 3 fork-only names must be unchanged, and the 14
// upstream-only names must start working (currently -1 UNKNOWN on the "before" run).
//
// Deliberately links ONLY `llama` (the public C API), not llama-common/jinja: this is a probe of
// the compiled named-template path (llama_chat_apply_template), not the jinja engine.
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Union of every template name reachable via llama_chat_apply_template on either side, measured
// 2026-09-27 (temp/_q_chat_templates.py): 40 shared with upstream's src/llama-chat.cpp, 3
// fork-only (bitnet, deepseek4, falcon_e), 14 upstream-only. Plus one known-bad control name that
// must return -1 (LLM_CHAT_TEMPLATE_UNKNOWN) on both sides.
static const std::vector<std::string> UNION_NAMES = {
    // --- 40 shared (fork monolith + upstream llama-chat.cpp both map these) ---
    "chatml", "llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip",
    "mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7",
    "phi3", "falcon3", "zephyr", "monarch", "gemma", "orion", "openchat",
    "vicuna", "vicuna-orca", "deepseek", "deepseek2", "deepseek3",
    "command-r", "llama3", "chatglm3", "chatglm4", "minicpm", "exaone3",
    "rwkv-world", "granite", "gigachat", "megrez", "llama4",
    "hunyuan-moe", "kimi-k2", "gpt-oss", "grok-2",
    "bailing", "bailing-think", "bailing2", "seed_oss",
    // --- 3 fork-only ---
    "bitnet", "deepseek4", "falcon_e",
    // --- 14 upstream-only (expected -1/UNKNOWN before this change, working after) ---
    "mistral-v7-tekken", "phi4", "deepseek-ocr", "glmedge", "exaone4",
    "exaone-moe", "granite-4.0", "granite-4.1", "yandex", "smolvlm",
    "hunyuan-dense", "hunyuan-vl", "pangu-embedded", "solar-open",
    // --- control: must stay -1/UNKNOWN on both sides ---
    "definitely-not-a-real-chat-template-xyz",
};

static std::string escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else if (c == '\t') { out += "\\t"; }
        else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof(buf), "\\x%02x", c); out += buf; }
        else { out += (char) c; }
    }
    return out;
}

int main() {
    std::vector<llama_chat_message> conversation = {
        { "system",    "You are a helpful assistant." },
        { "user",      "Hello, how are you?" },
        { "assistant", "I am doing well, thank you." },
    };

    printf("# S69-CHAT-TEMPLATES-MONOLITH-LACKS-14-UPSTREAM-2026-09-27-01 parity dump\n");
    printf("# format: name\\tlen_or_err\\toutput\n");

    for (const auto & name : UNION_NAMES) {
        char buf[8192];
        std::memset(buf, 0, sizeof(buf));
        int32_t n = llama_chat_apply_template(
            name.c_str(), conversation.data(), conversation.size(), /*add_ass=*/true, buf, sizeof(buf));
        if (n < 0) {
            printf("%s\t%d\tERROR\n", name.c_str(), (int) n);
        } else {
            std::string out(buf, buf + (n < (int32_t) sizeof(buf) ? n : (int32_t) sizeof(buf) - 1));
            printf("%s\t%d\t%s\n", name.c_str(), (int) n, escape(out).c_str());
        }
    }

    // Secondary signal: total builtin-template count on this build (43 before, 57 after: 40
    // shared + 3 fork-only stay reachable via llama_chat_apply_template, but
    // llama_chat_builtin_templates enumerates whichever LLM_CHAT_TEMPLATES map is linked in).
    int32_t total = llama_chat_builtin_templates(nullptr, 0);
    printf("# llama_chat_builtin_templates total=%d\n", (int) total);

    return 0;
}
