// Row S69-ENGREV-TABBYAPI-CHAT-HANDLER-LOADLOG-2026-10-27-01: the chat handler picked for a template
// is logged at INFO when the template loads (specialized name, or the autoparser fallback).
// Tabby-API prior art: its chat-template loader logs which template/handler it chose. SOURCE-READ: common/chat.cpp
// common_chat_template_detect_params_init (first match wins) and the common_chat_template constructor.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#undef NDEBUG
#include <cassert>

#include "chat.h"
#include "log.h"

static std::string load_and_capture(const std::string & tmpl, const char * path) {
    std::remove(path);
    common_log * log = common_log_main();
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    common_log_set_file(log, path);
    {
        common_chat_template t(tmpl, "", "");
    }
    common_log_flush(log);
    common_log_set_file(log, nullptr);
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main() {
    // Qwen3-Coder markers (chat.cpp detect: <tool_call>, <function=, <parameter=)
    const std::string coder = "<tool_call>\n<function=f>\n<parameter=p>\n{{ messages }}";
    std::string out = load_and_capture(coder, "test-chat-handler-loadlog-1.log");
    if (out.find("Using specialized template: Qwen3-Coder") == std::string::npos) {
        std::fprintf(stderr, "FAIL: no INFO line for the specialized handler. captured:\n%s\n", out.c_str());
        return 1;
    }

    const std::string plain = "{% for m in messages %}{{ m.content }}{% endfor %}";
    out = load_and_capture(plain, "test-chat-handler-loadlog-2.log");
    if (out.find("Using chat handler: autoparser") == std::string::npos) {
        std::fprintf(stderr, "FAIL: no INFO line for the autoparser fallback. captured:\n%s\n", out.c_str());
        return 1;
    }
    std::remove("test-chat-handler-loadlog-1.log");
    std::remove("test-chat-handler-loadlog-2.log");
    std::printf("OK\n");
    return 0;
}
