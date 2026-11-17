#pragma once

#include "json-schema.h"
#include "json.h"

#include <functional>
#include <memory>
#include <string>

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_chat_schema_document & schema);

// GBNF for a raw regex that must FULLY match the whole output (anchored with ^...$ if it is not already).
// Same translator as a JSON-schema string "pattern", but no JSON quotes and no "accept any string" fallback:
// returns "" and sets `err` when the regex is invalid or uses a construct GBNF cannot express (lookaround, ...).
std::string regex_to_grammar(const std::string & regex, std::string & err, bool dotall = false);

struct common_schema_info_impl;

// Probes a raw JSON schema to extract information about its structure and type constraints (fork API; the typed
// document of json-schema.h, common_chat_schema::may_be_string, is upstream's equivalent for a built schema).
// Never throws: a schema past the depth cap is left unresolved and reads as "not a string".
class common_schema_info {
    std::unique_ptr<common_schema_info_impl> impl_;

  public:
    common_schema_info();
    ~common_schema_info();

    common_schema_info(const common_schema_info &) = delete;
    common_schema_info & operator=(const common_schema_info &) = delete;
    common_schema_info(common_schema_info &&) noexcept;
    common_schema_info & operator=(common_schema_info &&) noexcept;

    void resolve_refs(common_json & schema);
    bool resolves_to_string(const common_json & schema);
};

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)>    add_rule;
    std::function<std::string(const std::string &, const common_chat_schema &)> add_schema;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
