#pragma once

#include "json.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

// JSON schema, covering the subset that json_schema_to_grammar() can convert.

struct common_chat_schema {
    enum node_kind {
        KIND_ANY,
        KIND_REF,
        KIND_ANY_OF,
        KIND_ALL_OF,
        KIND_CONST,
        KIND_ENUM,
        KIND_NULL,
        KIND_BOOLEAN,
        KIND_NUMBER,
        KIND_INTEGER,
        KIND_STRING,
        KIND_ARRAY,
        KIND_TUPLE,
        KIND_OBJECT,
    };

    enum value_type {
        TYPE_NULL,
        TYPE_BOOLEAN,
        TYPE_NUMBER,
        TYPE_INTEGER,
        TYPE_STRING,
        TYPE_ARRAY,
        TYPE_OBJECT,
    };

    enum string_format {
        FORMAT_NONE,
        FORMAT_UUID,  // uuid, uuid1 .. uuid5
        FORMAT_DATE,
        FORMAT_TIME,
        FORMAT_DATE_TIME,
    };

    class type_set {
        uint32_t mask_ = 0;

      public:
        type_set() = default;
        type_set(std::initializer_list<value_type> types) {
            for (auto type : types) {
                add(type);
            }
        }

        static type_set all() {
            return { TYPE_NULL, TYPE_BOOLEAN, TYPE_NUMBER, TYPE_INTEGER, TYPE_STRING, TYPE_ARRAY, TYPE_OBJECT };
        }

        void add(value_type type) { mask_ |= 1u << type; }

        bool has(value_type type) const { return (mask_ & (1u << type)) != 0; }
        bool is_only(value_type type) const { return mask_ == (1u << type); }
        bool empty() const { return mask_ == 0; }

        type_set & operator|=(const type_set & other) { mask_ |= other.mask_; return *this; }
        type_set & operator&=(const type_set & other) { mask_ &= other.mask_; return *this; }

        bool operator==(const type_set & other) const { return mask_ == other.mask_; }
        bool operator!=(const type_set & other) const { return mask_ != other.mask_; }
    };

    virtual ~common_chat_schema() = default;
    virtual node_kind kind() const = 0;

    type_set value_types() const;

    // Whether a value matching the schema may be a string, through any branch of it.
    bool may_be_string() const;

    static const char * kind_name(node_kind kind);
    static const char * type_name(value_type type);
};

using common_chat_schema_ptr = std::unique_ptr<common_chat_schema>;

struct common_chat_schema_any : common_chat_schema {
    node_kind kind() const override { return KIND_ANY; }
};

// {"$ref": "#/..."}, only references into the same document are supported
struct common_chat_schema_ref : common_chat_schema {
    std::string                ref;
    const common_chat_schema * target = nullptr;  // owned by common_chat_schema_document::refs

    explicit common_chat_schema_ref(std::string ref) : ref(std::move(ref)) {}

    node_kind kind() const override { return KIND_REF; }
};

// oneOf / anyOf, or a "type" array expanded to one alternative per type
struct common_chat_schema_any_of : common_chat_schema {
    std::vector<common_chat_schema_ptr> children;

    node_kind kind() const override { return KIND_ANY_OF; }
};

struct common_chat_schema_all_of : common_chat_schema {
    std::vector<common_chat_schema_ptr> children;

    node_kind kind() const override { return KIND_ALL_OF; }
};

struct common_chat_schema_const : common_chat_schema {
    common_json value;

    explicit common_chat_schema_const(common_json value) : value(std::move(value)) {}

    node_kind kind() const override { return KIND_CONST; }
};

struct common_chat_schema_enum : common_chat_schema {
    std::vector<common_json> values;

    node_kind kind() const override { return KIND_ENUM; }
};

struct common_chat_schema_null : common_chat_schema {
    node_kind kind() const override { return KIND_NULL; }
};

struct common_chat_schema_boolean : common_chat_schema {
    node_kind kind() const override { return KIND_BOOLEAN; }
};

struct common_chat_schema_number : common_chat_schema {
    node_kind kind() const override { return KIND_NUMBER; }
};

// bounds are inclusive, exclusiveMinimum / exclusiveMaximum are folded in
struct common_chat_schema_integer : common_chat_schema {
    int64_t minimum = INT64_MIN;  // INT64_MIN for unbounded
    int64_t maximum = INT64_MAX;  // INT64_MAX for unbounded

    node_kind kind() const override { return KIND_INTEGER; }
};

struct common_chat_schema_string : common_chat_schema {
    std::string   pattern;  // empty when absent
    string_format format     = FORMAT_NONE;
    int           min_length = 0;
    int           max_length = -1;  // -1 for unbounded

    node_kind kind() const override { return KIND_STRING; }
};

struct common_chat_schema_array : common_chat_schema {
    common_chat_schema_ptr items;  // a common_chat_schema_any when "items" is absent
    int                    min_items = 0;
    int                    max_items = -1;  // -1 for unbounded

    node_kind kind() const override { return KIND_ARRAY; }
};

struct common_chat_schema_tuple : common_chat_schema {
    std::vector<common_chat_schema_ptr> items;

    node_kind kind() const override { return KIND_TUPLE; }
};

struct common_chat_schema_property {
    std::string            name;
    common_chat_schema_ptr schema;
    bool                   required = false;
};

struct common_chat_schema_object : common_chat_schema {
    std::vector<common_chat_schema_property> properties;             // in schema order
    common_chat_schema_ptr                   additional_properties;  // null when not allowed

    node_kind kind() const override { return KIND_OBJECT; }
};

struct common_chat_schema_document {
    common_chat_schema_ptr                        root;
    std::map<std::string, common_chat_schema_ptr> refs;
    common_json                                   source; // the schema JSON the document was built from (saved parsers)
};

// A document shared by the PEG parsers built from its nodes, which it keeps alive
using common_chat_schema_document_ptr = std::shared_ptr<const common_chat_schema_document>;

// Deepest schema accepted: common_chat_schema_from_json builds at most this many nested levels, the root is one of them.
// A schema reaches the builder from a client (tool "parameters", response_format, json_schema), so its nesting is
// attacker-sized; past the cap the request fails with std::invalid_argument instead of overflowing the stack.
// 256 (the fork's value; upstream's is 128). MEASURED 2026-11-06 by tests/probe-schema-depth (MSVC 14.51 Release, a 1 MiB stack = the
// llama-server PE reserve, on the main thread AND on a std::thread worker, through common_chat_schema_from_json + json_schema_to_grammar
// and through common_chat_templates_apply): the first stack overflow comes at 553 nested object layers, 827 array, 924 anyOf, so 255
// layers use under half of it. Apple gives secondary threads 512 KiB, which the worst case (about 1.9 KiB per object layer) would nearly fill
// at 255, so Apple keeps upstream's 128. Row S69-ENGREV-UPSTREAM-MERGE-SCHEMA-DEPTH-CAP-MEASURE-2026-11-04-01.
// Only tests/probe-schema-depth.cpp defines COMMON_CHAT_SCHEMA_MAX_DEPTH_VALUE itself, to measure the stack past the cap.
#ifndef COMMON_CHAT_SCHEMA_MAX_DEPTH_VALUE
#  if defined(__APPLE__)
#    define COMMON_CHAT_SCHEMA_MAX_DEPTH_VALUE 128
#  else
#    define COMMON_CHAT_SCHEMA_MAX_DEPTH_VALUE 256
#  endif
#endif
constexpr int COMMON_CHAT_SCHEMA_MAX_DEPTH = COMMON_CHAT_SCHEMA_MAX_DEPTH_VALUE;
// The same cap in raw JSON levels: a schema level costs up to two JSON levels (an object under "properties", an array under
// "anyOf"), plus slack for the document root (fork, row S69-ENGREV-LOCALAI2-SCHEMA-ALLOF-REF-CYCLE-DEPTH-GUARD-2026-10-27-01).
constexpr int COMMON_CHAT_SCHEMA_MAX_JSON_DEPTH = 2 * COMMON_CHAT_SCHEMA_MAX_DEPTH + 8;

// Is the tree deeper than `limit` JSON levels? Iterative, so it is safe on a tree no recursive walk may touch yet (fork).
bool common_json_deeper_than(const common_json & root, int limit);

// Throws std::invalid_argument naming the depth cap when the schema's JSON is deeper than COMMON_CHAT_SCHEMA_MAX_JSON_DEPTH (fork).
void common_chat_schema_check_depth(const common_json & schema);

// Throws std::runtime_error when the schema falls outside the supported subset.
common_chat_schema_document common_chat_schema_from_json(const common_json & schema);
