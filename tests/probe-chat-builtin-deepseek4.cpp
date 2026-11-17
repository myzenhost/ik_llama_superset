// Probe: does the builtin chat template named "deepseek4" (src/llama-chat.cpp) render what the OFFICIAL DeepSeek-V4-Flash-0731
// encoder renders in chat mode? Golden strings below come from the official encoder's encode_messages(..., "chat") with its
// leading BOS removed (the tokenizer adds the BOS); see tools/server/tests_cpp/gen_dsv4_golden.py for how the official encoder is called.
// Controls: "deepseek3" (the V3 format, which the name used to alias) must DIFFER, and an unknown name must fail.
// Not in CMake on purpose -- it links the static archives directly. Build (mingw, from the fork root, build-static):
//   g++ -O1 -std=c++17 -fopenmp -I include -I ggml/include tests/probe-chat-builtin-deepseek4.cpp -o probe-chat.exe
//       -Wl,--start-group build-static/src/libllama.a build-static/ggml/src/ggml.a build-static/ggml/src/ggml-cpu.a
//       build-static/ggml/src/ggml-base.a -Wl,--end-group -fopenmp -lstdc++
// Expected: "4 checks, 0 failed", rc 0.
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s\n", what); } } while (0)

static std::string render(const char * tmpl, const std::vector<llama_chat_message> & msgs, int32_t * rc = nullptr) {
    std::vector<char> buf(4096);
    int32_t n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), (int32_t) buf.size());
    if (rc) { *rc = n; }
    if (n < 0) {
        return "";
    }
    if ((size_t) n > buf.size()) {
        buf.resize(n + 1);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), (int32_t) buf.size());
    }
    return std::string(buf.data(), n);
}

int main() {
    // official encode_messages(..., "chat") minus the leading BOS
    const std::string G_PLAIN = "You are helpful.<\xEF\xBD\x9C" "User\xEF\xBD\x9C>Weather in Paris?<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C></think>";
    const std::string G_NOSYS = "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>Hi<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C></think>";
    const std::string G_MULTI = "S<\xEF\xBD\x9C" "User\xEF\xBD\x9C>u1<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C></think>a1<\xEF\xBD\x9C" "end\xE2\x96\x81" "of\xE2\x96\x81" "sentence\xEF\xBD\x9C><\xEF\xBD\x9C" "User\xEF\xBD\x9C>u2<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C></think>";

    std::vector<llama_chat_message> plain = {{"system", "You are helpful."}, {"user", "Weather in Paris?"}};
    std::vector<llama_chat_message> nosys = {{"user", "Hi"}};
    std::vector<llama_chat_message> multi = {{"system", "S"}, {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"}};

    const std::string got_plain = render("deepseek4", plain);
    const std::string got_nosys = render("deepseek4", nosys);
    const std::string got_multi = render("deepseek4", multi);
    CHECK(got_plain == G_PLAIN, "deepseek4 plain chat == official");
    CHECK(got_nosys == G_NOSYS, "deepseek4 no system == official");
    CHECK(got_multi == G_MULTI, "deepseek4 multi-turn == official");
    // control: the V3 format must NOT equal the V4 golden (a probe that can only say "match" measures nothing)
    CHECK(render("deepseek3", plain) != G_PLAIN, "control: deepseek3 differs from the V4 golden");
    int32_t rc = 0;
    render("no-such-template-xyz", plain, &rc);
    CHECK(rc < 0, "control: unknown template name fails");

    if (g_fails) {
        std::printf("deepseek4 got : [%s]\nofficial      : [%s]\n", got_plain.c_str(), G_PLAIN.c_str());
    }
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
