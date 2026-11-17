#pragma once
// Builds the user turn of an /v1/audio/transcriptions request (std-only, so a CPU test can include it).
//
// OpenAI's `prompt` field is a style / vocabulary hint, not a replacement for the task. Before this,
// a client-supplied prompt REPLACED the template's transcription instruction, so an instruction-tuned
// audio model (Voxtral) saw only "Acme, Zyrtec" and chatted instead of transcribing. The instruction
// is always kept; the client text rides after it as a hint.
// Spec: whisper.cpp initial_prompt semantics (conditions decoding, never replaces the task); SOURCE-READ, MIT, no code copied.
//
// A client prompt that IS a transcription instruction (it begins with "transcribe", any case, after leading blanks) is
// used as the instruction, as before this header existed. MEASURED 2026-11-14, merge gate b1114c step ASR_NATIVE on
// Voxtral-Mini-3B: the explicit prompt "Transcribe the speech in this audio verbatim. Output only the words spoken."
// (the preset itself) wrapped as a hint after the same instruction drew "I'm unable to assist with that request."
// (1 of 8 reference words); the same request passed in gate b1114b, before the hint wrapper.
#include <cctype>
#include <string>

static inline bool asr_prompt_is_instruction(const std::string & client_prompt, const std::string & preset_user) {
    size_t i = 0;
    while (i < client_prompt.size() && std::isspace((unsigned char) client_prompt[i])) {
        i++;
    }
    static const char kVerb[] = "transcribe";
    const size_t n = sizeof(kVerb) - 1;
    if (client_prompt.size() - i >= n) {
        bool same = true;
        for (size_t k = 0; k < n && same; k++) {
            same = std::tolower((unsigned char) client_prompt[i + k]) == kVerb[k];
        }
        if (same) {
            return true;
        }
    }
    return !preset_user.empty() && client_prompt == preset_user;
}

static inline std::string asr_compose_user_prompt(
        const std::string & preset_user,
        const std::string & client_prompt,
        const std::string & language) {
    std::string out = preset_user;
    if (!client_prompt.empty()) {
        if (asr_prompt_is_instruction(client_prompt, preset_user)) {
            out = client_prompt;
        } else {
            out += " Spelling and vocabulary hint (do not transcribe this text): " + client_prompt;
        }
    }
    if (!language.empty()) {
        out += " (language: " + language + ")";
    }
    return out;
}
