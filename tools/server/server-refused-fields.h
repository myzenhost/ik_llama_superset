// Request fields that change the returned tokens or text but have no effect in this server: they are refused BY NAME on
// every generation route, with a 501, instead of being dropped while the request answers 200 as if they had been honoured.
//
// The line this table draws: only fields whose absence changes the ANSWER. A hint the server may ignore without changing its
// answer (a scheduling priority, a tracing tag) is not refused; the once-per-key warning in server-schema.cpp covers those.
// A field that is implemented never belongs here (n / n_cmpl, echo, prompt_logprobs, allowed_token_ids, bad_words, use_beam_search,
// json_schema / grammar / response_format are all read by the request schema or the beam handler). best_of and suffix are not in
// the table either: /v1/completions already refuses them in oaicompat_completion_params_parse and the TGI route serves best_of.
//
// JSON null counts as absent (same rule as the request schema), so a client that serialises every optional field as null keeps working.
//
// To add a field: put a row in server_refused_field_table() and a case in tools/server/tests_cpp/test_server_refused_fields.cpp.
// A field the schema starts to read must be taken out of the table in the same commit; the test lists every name so that the
// table and the test cannot drift apart without one of them failing.
//
// SOURCE-READ: upstream/ferrox/crates/frink-server/src/unimplemented_fields.rs:11-55 (the measurement: undeclared fields
// answered 200, chat n=3 gave 501 while completions n=3 gave 200 with one choice), :77-110 (the field list, every member optional,
// None = the caller said nothing), :195-312 (one refusal per field, 501 for "well-formed but not served", 400 for values no
// server could serve), :315-324 (one sentence naming the field and the route). Ferrox is Apache-2.0; the spec is written here in my
// own words and structure, no code copied.
// SOURCE-NUANCE: the source refuses only OUTPUT-CHANGING fields (priority / cache_salt style hints pass), treats an absent value
// and None alike, and rejects a few values with 400 rather than 501 (truncate_prompt_tokens below 1, an empty allowed_token_ids);
// a description of "reject unknown fields" would say none of that. The fork has no typed request struct to destructure, so the guard
// against a field being added and not classified is the test that pins the table instead of a compile error.
#pragma once

#include <string>
#include <vector>

struct server_refused_field_entry {
    const char * name;
    const char * what; // what the caller asked for, in a clause that follows "is not implemented on this server: "
    const char * hint; // what to send instead; empty when there is no equivalent
};

inline const std::vector<server_refused_field_entry> & server_refused_field_table() {
    static const std::vector<server_refused_field_entry> table = {
        { "truncate_prompt_tokens", "keeping only the last k tokens of an over-long prompt",
          "shorten the prompt on the client, or raise the context with -c" },
        { "prompt_embeds",          "embeddings supplied as input in place of text", "" },
        { "guided_json",            "constrained decoding to a JSON schema through a guided_* field",
          "send the same schema as response_format / json_schema" },
        { "guided_regex",           "constrained decoding to a regular expression through a guided_* field",
          "send a GBNF grammar in the grammar field" },
        { "guided_choice",          "constrained decoding to a list of choices through a guided_* field",
          "send a GBNF grammar listing the choices in the grammar field" },
        { "guided_grammar",         "constrained decoding to a grammar through a guided_* field", "send it in the grammar field" },
        { "guided_decoding_backend", "choosing a constrained-decoding backend", "" },
        { "structured_outputs",     "a structured_outputs object",
          "send response_format / json_schema, or a GBNF grammar in the grammar field" },
        { "logits_processors",      "user-supplied logits processors", "use logit_bias, allowed_token_ids or bad_words" },
    };
    return table;
}

struct server_refused_field {
    bool        refused = false;
    std::string name;
    std::string message; // one sentence naming the field; ready for a 501 body
};

// `J` is the JSON class of the caller (nlohmann json, or the common_json alias): both give contains / at / is_null.
template <class J>
inline server_refused_field server_refuse_unimplemented_field(const J & data) {
    server_refused_field r;
    if (!data.is_object()) {
        return r;
    }
    for (const auto & e : server_refused_field_table()) {
        if (!data.contains(e.name) || data.at(e.name).is_null()) {
            continue;
        }
        r.refused = true;
        r.name    = e.name;
        r.message = std::string("`") + e.name + "` is not implemented on this server: " + e.what;
        if (e.hint[0] != '\0') {
            r.message += "; " + std::string(e.hint);
        }
        return r;
    }
    return r;
}
