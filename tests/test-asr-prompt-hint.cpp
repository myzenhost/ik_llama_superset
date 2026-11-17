// A client `prompt` on /v1/audio/transcriptions is a vocabulary hint: the template's transcription
// instruction must survive it (tools/server/server-chat.cpp used to do `if (prompt.empty()) prompt = preset.user`).
#include "../tools/server/server-asr-prompt.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static bool has(const std::string & s, const std::string & t) { return s.find(t) != std::string::npos; }

int main() {
    const std::string instr = "Transcribe this audio verbatim.";

    // no client prompt: the instruction alone, as before
    CHECK(asr_compose_user_prompt(instr, "", "") == instr);
    // language suffix unchanged
    CHECK(asr_compose_user_prompt(instr, "", "de") == instr + " (language: de)");

    // client prompt: the instruction is KEPT and the hint is present
    const std::string p = asr_compose_user_prompt(instr, "Acme, Zyrtec", "");
    CHECK(has(p, instr));
    CHECK(has(p, "Acme, Zyrtec"));
    CHECK(p.find(instr) < p.find("Acme"));   // instruction first

    // both together
    const std::string q = asr_compose_user_prompt(instr, "Acme", "en");
    CHECK(has(q, instr) && has(q, "Acme") && has(q, "(language: en)"));

    // a client prompt that IS a transcription instruction is the instruction, not a hint (merge gate b1114c ASR_NATIVE:
    // the preset wrapped as a hint after itself drew a refusal on Voxtral-Mini-3B)
    const std::string gate_p = "Transcribe the speech in this audio verbatim. Output only the words spoken.";
    CHECK(asr_compose_user_prompt(instr, gate_p, "") == gate_p);
    CHECK(asr_compose_user_prompt(gate_p, gate_p, "") == gate_p);           // equal to the preset
    CHECK(asr_compose_user_prompt(instr, "  transcribe it all", "fr") == "  transcribe it all (language: fr)");
    CHECK(!has(asr_compose_user_prompt(instr, gate_p, ""), "hint"));
    // control: a hint that merely mentions the word later is still a hint
    const std::string h = asr_compose_user_prompt(instr, "Names: Acme. Please transcribe carefully", "");
    CHECK(has(h, instr) && has(h, "hint") && has(h, "Acme"));

    printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
