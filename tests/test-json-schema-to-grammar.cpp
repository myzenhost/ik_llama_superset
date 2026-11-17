#ifdef NDEBUG
#undef NDEBUG
#endif

#include "json-schema-to-grammar.h"

#include "../src/llama-grammar.h"

#include "json.h"

#include <cassert>
#include <regex>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

static std::string trim(const std::string & source) {
    std::string s(source);
    s.erase(0,s.find_first_not_of(" \n\r\t"));
    s.erase(s.find_last_not_of(" \n\r\t")+1);
    return std::regex_replace(s, std::regex("(^|\n)[ \t]+"), "$1");
}

enum TestCaseStatus {
    SUCCESS,
    FAILURE
};

struct TestCase {
    TestCaseStatus expected_status;
    std::string name;
    std::string schema;
    std::string expected_grammar;

    void _print_failure_header() const {
        fprintf(stderr, "#\n# Test '%s' failed.\n#\n%s\n", name.c_str(), schema.c_str());
    }
    void verify(const std::string & actual_grammar) const {
        if (trim(actual_grammar) != trim(expected_grammar)) {
        _print_failure_header();
        fprintf(stderr, "# EXPECTED:\n%s\n# ACTUAL:\n%s\n", expected_grammar.c_str(), actual_grammar.c_str());
        assert(false);
        }
    }
    void verify_expectation_parseable() const {
        try {
            llama_grammar_parser state;
            state.parse(expected_grammar.c_str());
            if (state.symbol_ids.find("root") == state.symbol_ids.end()) {
                throw std::runtime_error("Grammar failed to parse:\n" + expected_grammar);
            }
        } catch (const std::runtime_error & ex) {
            _print_failure_header();
            fprintf(stderr, "# GRAMMAR ERROR: %s\n", ex.what());
            assert(false);
        }
    }
    void verify_status(TestCaseStatus status) const {
        if (status != expected_status) {
            _print_failure_header();
            fprintf(stderr, "# EXPECTED STATUS: %s\n", expected_status == SUCCESS ? "SUCCESS" : "FAILURE");
            fprintf(stderr, "# ACTUAL STATUS: %s\n", status == SUCCESS ? "SUCCESS" : "FAILURE");
            assert(false);
        }
    }
};

static void test_all(const std::string & title, std::function<void(const TestCase &)> runner) {
    fprintf(stderr, "#\n# %s\n#\n", title.c_str());
    auto test = [&](const TestCase & tc) {
        fprintf(stderr, "- %s%s\n", tc.name.c_str(), tc.expected_status == FAILURE ? " (failure expected)" : "");
        runner(tc);
    };

    test({
        SUCCESS,
        "min 0",
        R"""({
            "type": "integer",
            "minimum": 0
        })""",
        R"""(
            root ::= ([0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 1",
        R"""({
            "type": "integer",
            "minimum": 1
        })""",
        R"""(
            root ::= ([1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 3",
        R"""({
            "type": "integer",
            "minimum": 3
        })""",
        R"""(
            root ::= ([1-2] [0-9]{1,15} | [3-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 9",
        R"""({
            "type": "integer",
            "minimum": 9
        })""",
        R"""(
            root ::= ([1-8] [0-9]{1,15} | [9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 10",
        R"""({
            "type": "integer",
            "minimum": 10
        })""",
        R"""(
            root ::= ([1] ([0-9]{1,15}) | [2-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 25",
        R"""({
            "type": "integer",
            "minimum": 25
        })""",
        R"""(
            root ::= ([1] [0-9]{2,15} | [2] ([0-4] [0-9]{1,14} | [5-9] [0-9]{0,14}) | [3-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 30",
        R"""({
            "type": "integer",
            "maximum": 30
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -5",
        R"""({
            "type": "integer",
            "minimum": -5
        })""",
        R"""(
            root ::= ("-" ([0-5]) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -123",
        R"""({
            "type": "integer",
            "minimum": -123
        })""",
        R"""(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max -5",
        R"""({
            "type": "integer",
            "maximum": -5
        })""",
        R"""(
            root ::= ("-" ([0-4] [0-9]{1,15} | [5-9] [0-9]{0,15}))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 1",
        R"""({
            "type": "integer",
            "maximum": 1
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-1])
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 100",
        R"""({
            "type": "integer",
            "maximum": 100
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-8] [0-9] | [9] [0-9]) | "100")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 0 max 23",
        R"""({
            "type": "integer",
            "minimum": 0,
            "maximum": 23
        })""",
        R"""(
            root ::= ([0-9] | ([1] [0-9] | [2] [0-3]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 15 max 300",
        R"""({
            "type": "integer",
            "minimum": 15,
            "maximum": 300
        })""",
        R"""(
            root ::= (([1] ([5-9]) | [2-9] [0-9]) | ([1-2] [0-9]{2} | [3] "00"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 5 max 30",
        R"""({
            "type": "integer",
            "minimum": 5,
            "maximum": 30
        })""",
        R"""(
            root ::= ([5-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -123 max 42",
        R"""({
            "type": "integer",
            "minimum": -123,
            "maximum": 42
        })""",
        R"""(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0-9] | ([1-3] [0-9] | [4] [0-2]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -10 max 10",
        R"""({
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        })""",
        R"""(
            root ::= ("-" ([0-9] | "10") | [0-9] | "10")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        FAILURE,
        "unknown type",
        R"""({
            "type": "kaboom"
        })""",
        ""
    });

    test({
        FAILURE,
        "invalid type",
        R"""({
            "type": 123
        })""",
        ""
    });

    test({
        SUCCESS,
        "empty schema (any value)",
        "{}",
        R"""(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= value
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "exotic formats",
        R"""({
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        })""",
        R"""(
            date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
            date-string ::= "\"" date "\""
            date-time ::= date "T" time
            date-time-string ::= "\"" date-time "\""
            root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
            time-string ::= "\"" time "\""
            tuple-0 ::= date-string
            tuple-2 ::= time-string
            tuple-3 ::= date-time-string
            uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""
        )"""
    });

    test({
        SUCCESS,
        "string",
        R"""({
            "type": "string"
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char* "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min length 1",
        R"""({
            "type": "string",
            "minLength": 1
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char+ "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min length 3",
        R"""({
            "type": "string",
            "minLength": 3
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{3,} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ max length",
        R"""({
            "type": "string",
            "maxLength": 3
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{0,3} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min & max length",
        R"""({
            "type": "string",
            "minLength": 1,
            "maxLength": 4
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{1,4} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "boolean",
        R"""({
            "type": "boolean"
        })""",
        R"""(
            root ::= ("true" | "false")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "integer",
        R"""({
            "type": "integer"
        })""",
        R"""(
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string const",
        R"""({
            "const": "foo"
        })""",
        R"""(
            root ::= "\"foo\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "non-string const",
        R"""({
            "const": 123
        })""",
        R"""(
            root ::= "123"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "non-string enum",
        R"""({
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
        })""",
        R"""(
            root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string array",
        R"""({
            "type": "array",
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space (string ("," space string)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "nullable string array",
        R"""({
            "type": ["array", "null"],
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            alternative-0 ::= "[" space (string ("," space string)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            null ::= "null"
            root ::= alternative-0 | null
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "tuple1",
        R"""({
            "prefixItems": [{ "type": "string" }]
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space string space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "tuple2",
        R"""({
            "prefixItems": [{ "type": "string" }, { "type": "number" }]
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space string "," space number space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    // items {} constrains nothing, the same as no items at all
    test({
        SUCCESS,
        "array with empty items",
        R"""({
            "type": "array",
            "items": {}
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space ( value ("," space value)* )? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "array with empty items and prefixItems",
        R"""({
            "type": "array",
            "items": {},
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space ( value ("," space value)* )? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "number",
        R"""({
            "type": "number"
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "minItems",
        R"""({
            "items": {
                "type": "boolean"
            },
            "minItems": 2
        })""",
        R"""(
            boolean ::= ("true" | "false")
            root ::= "[" space boolean ("," space boolean)+ space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 0",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 0
        })""",
        R"""(
            boolean ::= ("true" | "false")
            root ::= "[" space  space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 1",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 1
        })""",
        R"""(
            boolean ::= ("true" | "false")
            root ::= "[" space boolean? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 2",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 2
        })""",
        R"""(
            boolean ::= ("true" | "false")
            root ::= "[" space (boolean ("," space boolean)?)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + maxItems",
        R"""({
            "items": {
                "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= number | integer
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + max items with min + max values across zero",
        R"""({
            "items": {
                "type": "integer",
                "minimum": -12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            item ::= ("-" ([0-9] | "1" [0-2]) | [0-9] | ([1-8] [0-9] | [9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + max items with min + max values",
        R"""({
            "items": {
                "type": "integer",
                "minimum": 12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            item ::= (([1] ([2-9]) | [2-9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "simple regexp",
        R"""({
            "type": "string",
            "pattern": "^abc?d*efg+(hij)?kl$"
        })""",
        R"""(
            root ::= "\"" ("ab" "c"? "d"* "ef" "g"+ ("hij")? "kl") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp escapes",
        R"""({
            "type": "string",
            "pattern": "^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$"
        })""",
        R"""(
            root ::= "\"" ("[]{}()|+*?") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp quote",
        R"""({
            "type": "string",
            "pattern": "^\"$"
        })""",
        R"""(
            root ::= "\"" ("\"") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp with top-level alternation",
        R"""({
            "type": "string",
            "pattern": "^A|B|C|D$"
        })""",
        R"""(
            root ::= "\"" ("A" | "B" | "C" | "D") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp",
        R"""({
            "type": "string",
            "pattern": "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"
        })""",
        R"""(
            dot ::= [^\x0A\x0D]
            root ::= "\"" (("(" root-1{1,3} ")")? root-1{3,3} "-" root-1{4,4} " " "a"{3,5} "nd" dot dot dot) "\""
            root-1 ::= [0-9]
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required props in original order",
        R"""({
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "c": {"type": "string"},
                "a": {"type": "string"}
            },
            "required": [
                "a",
                "b",
                "c"
            ],
            "additionalProperties": false,
            "definitions": {}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space b-kv "," space c-kv "," space a-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "1 optional prop",
        R"""({
            "properties": {
                "a": {
                "type": "string"
                }
            },
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "N optional props",
        R"""({
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            a-rest ::= ( "," space b-kv )? b-rest
            b-kv ::= "\"b\"" space ":" space string
            b-rest ::= ( "," space c-kv )?
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "required + optional props each in original order",
        R"""({
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            d-kv ::= "\"d\"" space ":" space string
            d-rest ::= ( "," space c-kv )?
            root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "additional props",
        R"""({
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}}
        })""",
        R"""(
            additional-kv ::= string ":" space additional-value
            additional-value ::= "[" space (number ("," space number)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "additional props (true)",
        R"""({
            "type": "object",
            "additionalProperties": true
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "additional props (implicit)",
        R"""({
            "type": "object"
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "empty w/o additional props",
        R"""({
            "type": "object",
            "additionalProperties": false
        })""",
        R"""(
            root ::= "{" space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required + additional props",
        R"""({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "required": ["a"],
            "additionalProperties": {"type": "string"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "optional + additional props",
        R"""({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "additionalProperties": {"type": "number"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required + optional + additional props",
        R"""({
            "type": "object",
            "properties": {
                "and": {"type": "number"},
                "also": {"type": "number"}
            },
            "required": ["and"],
            "additionalProperties": {"type": "number"}
        })""",
        R"""(
            additional-k ::= ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            also-kv ::= "\"also\"" space ":" space number
            also-rest ::= ( "," space additional-kv )*
            and-kv ::= "\"and\"" space ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space and-kv ( "," space ( also-kv also-rest | additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with empty name",
        R"""({
            "properties": {
                "": {"type": "integer"},
                "a": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            -kv ::= "\"\"" space ":" space root
            -rest ::= ( "," space a-kv )? a-rest
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* ) ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            root0 ::= "{" space  (-kv -rest | a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with nested names",
        R"""({
            "properties": {
                "a": {"type": "integer"},
                "aa": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space aa-kv )? aa-rest
            aa-kv ::= "\"aa\"" space ":" space integer
            aa-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([a] char+ | [^"a] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (a-kv a-rest | aa-kv aa-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with common prefix",
        R"""({
            "properties": {
                "ab": {"type": "integer"},
                "ac": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            ab-kv ::= "\"ab\"" space ":" space integer
            ab-rest ::= ( "," space ac-kv )? ac-rest
            ac-kv ::= "\"ac\"" space ":" space integer
            ac-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([b] char+ | [c] char+ | [^"bc] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (ab-kv ab-rest | ac-kv ac-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "top-level $ref",
        R"""({
            "$ref": "#/definitions/foo",
            "definitions": {
                "foo": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "a"
                    ],
                    "additionalProperties": false
                }
            }
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space string
            root ::= ref-definitions-foo
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "anyOf",
        R"""({
            "anyOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                }
            },
            "type": "object"
        })""",
        R"""(
            alternative-0 ::= ref-definitions-foo
            alternative-1 ::= ref-definitions-bar
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            ref-definitions-bar ::= "{" space  (ref-definitions-bar-b-kv )? space "}"
            ref-definitions-bar-b-kv ::= "\"b\"" space ":" space number
            ref-definitions-foo ::= "{" space  (ref-definitions-foo-a-kv )? space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space number
            root ::= alternative-0 | alternative-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "anyOf $ref",
        R"""({
            "properties": {
                "a": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                },
                "b": {
                    "anyOf": [
                        {"$ref": "#/properties/a/anyOf/0"},
                        {"type": "boolean"}
                    ]
                }
            },
            "type": "object"
        })""",
        R"""(
            a ::= string | number
            a-kv ::= "\"a\"" space ":" space a
            a-rest ::= ( "," space b-kv )?
            b ::= b-0 | boolean
            b-0 ::= string
            b-kv ::= "\"b\"" space ":" space b
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | b-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "mix of allOf, anyOf and $ref (similar to https://json.schemastore.org/tsconfig.json)",
        R"""({
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"},
                {
                "anyOf": [
                    {"$ref": "#/definitions/baz"},
                    {"$ref": "#/definitions/bam"}
                ]
                }
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                },
                "bam": {
                    "properties": {"c": {"type": "number"}}
                },
                "baz": {
                    "properties": {"d": {"type": "number"}}
                }
            },
            "type": "object"
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            b-kv ::= "\"b\"" space ":" space number
            c-kv ::= "\"c\"" space ":" space number
            d-kv ::= "\"d\"" space ":" space number
            d-rest ::= ( "," space c-kv )?
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv "," space b-kv ( "," space ( d-kv d-rest | c-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "allOf with enum schema",
        R"""({
            "allOf": [
                {"$ref": "#/definitions/foo"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b"]
                }
            }
        })""",
        R"""(
            root ::= ("\"a\"" | "\"b\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "allOf with multiple enum schemas",
        R"""({
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b", "c"]
                },
                "bar": {
                    "type": "string",
                    "enum": ["b", "c", "d"]
                }
            }
        })""",
        R"""(
            root ::= ("\"b\"" | "\"c\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "conflicting names",
        R"""({
            "type": "object",
            "properties": {
                "number": {
                "type": "object",
                "properties": {
                    "number": {
                    "type": "object",
                        "properties": {
                            "root": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "root"
                        ],
                        "additionalProperties": false
                    }
                },
                "required": [
                    "number"
                ],
                "additionalProperties": false
                }
            },
            "required": [
                "number"
            ],
            "additionalProperties": false,
            "definitions": {}
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            number- ::= "{" space number-number-kv space "}"
            number-kv ::= "\"number\"" space ":" space number-
            number-number ::= "{" space number-number-root-kv space "}"
            number-number-kv ::= "\"number\"" space ":" space number-number
            number-number-root-kv ::= "\"root\"" space ":" space number
            root ::= "{" space number-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "description only (no type) treated as unconstrained",
        R"""({"description": "The 0-based index of the last line to be retrieved (inclusive). If None, read until the end of the file."})""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= value
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "literal string with escapes",
        R"""({
            "properties": {
                "code": {
                    "const": " \r \n \" \\ ",
                    "description": "Generated code",
                    "title": "Code",
                    "type": "string"
                }
            },
            "required": [
                "code"
            ],
            "title": "DecoderResponse",
            "type": "object"
        })""",
        R"""(
            code ::= "\" \\r \\n \\\" \\\\ \""
            code-kv ::= "\"code\"" space ":" space code
            root ::= "{" space code-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp with non-capturing group",
        R"""({
            "type": "string",
            "pattern": "^(?:foo|bar)baz$"
        })""",
        R"""(
            root ::= "\"" (("foo" | "bar") "baz") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp with nested non-capturing groups",
        R"""({
            "type": "string",
            "pattern": "^(?:(?:ab)+c)?d$"
        })""",
        R"""(
            root ::= "\"" ((("ab")+ "c")? "d") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "unanchored regexp",
        R"""({
            "type": "string",
            "pattern": "[0-9]+"
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= string
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    // the rules of the partial conversion (here "root-0") must not leak into the grammar
    test({
        SUCCESS,
        "regexp with unsupported shorthand",
        R"""({
            "type": "string",
            "pattern": "^[0-9]{3}\\w$"
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= string
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });

    test({
        SUCCESS,
        "regexp with escaped hyphen in a character class",
        R"""({
            "type": "string",
            "pattern": "^[a-z\\-]+$"
        })""",
        R"""(
            root ::= "\"" ([a-z\-]+) "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp with escaped hyphen outside a character class",
        R"""({
            "type": "string",
            "pattern": "^a\\-b$"
        })""",
        R"""(
            root ::= "\"" ("a\-b") "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    // a regexp that is invalid under any flavor is still an error
    test({
        FAILURE,
        "regexp with unbalanced parentheses",
        R"""({
            "type": "string",
            "pattern": "^(a$"
        })""",
        ""
    });

    // only the property with the bad pattern degrades
    test({
        SUCCESS,
        "unsupported regexp in a property",
        R"""({
            "type": "object",
            "properties": {
                "a": { "type": "string", "pattern": "^(?=a)a$" }
            },
            "required": ["a"],
            "additionalProperties": false
        })""",
        R"""(
            a ::= string
            a-kv ::= "\"a\"" space ":" space a
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space a-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )"""
    });
}

static void test_resolves_to_string() {
    fprintf(stderr, "#\n# Testing resolves_to_string\n#\n");

    auto test = [](const std::string & name, const std::string & schema_str, bool expected) {
        fprintf(stderr, "- %s\n", name.c_str());
        common_schema_info info;
        auto schema = common_json::parse(schema_str);
        info.resolve_refs(schema);
        bool result = info.resolves_to_string(schema);
        if (result != expected) {
            fprintf(stderr, "#\n# Test '%s' failed.\n#\n", name.c_str());
            fprintf(stderr, "Schema: %s\n", schema_str.c_str());
            fprintf(stderr, "Expected: %s, Got: %s\n", expected ? "true" : "false", result ? "true" : "false");
            assert(false);
        }
    };

    // Basic type checks
    test("type string", R"({"type": "string"})", true);
    test("type integer", R"({"type": "integer"})", false);
    test("type number", R"({"type": "number"})", false);
    test("type boolean", R"({"type": "boolean"})", false);
    test("type object", R"({"type": "object"})", false);
    test("type array", R"({"type": "array"})", false);

    // Type array (nullable string)
    test("type array with string", R"({"type": ["string", "null"]})", true);
    test("type array without string", R"({"type": ["integer", "null"]})", false);

    // String-specific keywords
    test("minLength implies string", R"({"minLength": 1})", true);
    test("maxLength implies string", R"({"maxLength": 10})", true);
    test("pattern implies string", R"({"pattern": "^[a-z]+$"})", true);

    // Format
    test("format date", R"({"format": "date"})", true);
    test("format uuid", R"({"format": "uuid"})", true);
    test("format email", R"({"format": "email"})", true);

    // Const
    test("const string", R"({"const": "hello"})", true);
    test("const number", R"({"const": 123})", false);

    // Enum
    test("enum with strings", R"({"enum": ["a", "b", "c"]})", true);
    test("enum with numbers", R"({"enum": [1, 2, 3]})", false);
    test("enum mixed with string", R"({"enum": [1, "a", null]})", true);

    // anyOf
    test("anyOf with string", R"({"anyOf": [{"type": "string"}, {"type": "integer"}]})", true);
    test("anyOf without string", R"({"anyOf": [{"type": "integer"}, {"type": "boolean"}]})", false);

    // oneOf
    test("oneOf with string", R"({"oneOf": [{"type": "string"}, {"type": "number"}]})", true);
    test("oneOf without string", R"({"oneOf": [{"type": "object"}, {"type": "array"}]})", false);

    // allOf - all must be strings
    test("allOf all strings", R"({"allOf": [{"type": "string"}, {"minLength": 1}]})", true);
    test("allOf mixed types", R"({"allOf": [{"type": "string"}, {"type": "integer"}]})", false);

    // $ref
    test("$ref to string",
        R"({"$ref": "#/$defs/str", "$defs": {"str": {"type": "string"}}})", true);
    test("$ref to integer",
        R"({"$ref": "#/$defs/num", "$defs": {"num": {"type": "integer"}}})", false);

    // One definition used twice without a cycle: the second use must be checked again, not read as "already visited".
    test("allOf with the same $ref to a string twice",
        R"({"$defs": {"S": {"type": "string"}}, "allOf": [{"$ref": "#/$defs/S"}, {"$ref": "#/$defs/S"}]})", true);
    test("allOf with the same $ref to an integer twice",
        R"({"$defs": {"N": {"type": "integer"}}, "allOf": [{"$ref": "#/$defs/N"}, {"$ref": "#/$defs/N"}]})", false);
    test("allOf: $ref to an integer, then anyOf reusing a string $ref used earlier",
        R"({"$defs": {"S": {"type": "string"}}, "allOf": [{"$ref": "#/$defs/S"}, {"anyOf": [{"type": "integer"}, {"$ref": "#/$defs/S"}]}]})", true);
    // Controls: a real cycle is still answered "not a string" (and still terminates).
    test("two definitions that reference each other",
        R"({"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}})", false);
    test("cycle reached through allOf",
        R"({"$ref": "#/$defs/a", "$defs": {"a": {"allOf": [{"$ref": "#/$defs/a"}]}}})", false);

    // Nested
    test("nested anyOf with string",
        R"({"anyOf": [{"anyOf": [{"type": "integer"}, {"type": "string"}]}, {"type": "boolean"}]})", true);

    fprintf(stderr, "All resolves_to_string tests passed!\n");
}

// Row S69-ENGREV-LOCALAI2-SCHEMA-ALLOF-REF-CYCLE-DEPTH-GUARD-2026-10-27-01: a client-supplied tool schema must fail
// ONE request (std::invalid_argument), never unwind the server with a nlohmann type_error or overflow the stack.
// Oracle behaviour (MIT, spec only): LocalAI pkg/functions/grammars/json_schema_test.go:609-677 (cyclic $ref
// error, depth cap, sibling reuse of one $ref still converts). Our plain recursive $ref stays a recursive rule.
static int g_guard_failures = 0;

static void guard_fail(const std::string & name, const std::string & why) {
    fprintf(stderr, "  GUARD FAIL [%s]: %s\n", name.c_str(), why.c_str());
    g_guard_failures++;
}

static std::string nested_array_schema(int layers) {
    std::string s;
    for (int i = 0; i < layers; i++) {
        s += R"({"type":"array","items":)";
    }
    s += R"({"type":"integer"})";
    for (int i = 0; i < layers; i++) {
        s += "}";
    }
    return s;
}

static std::string nested_anyof_schema(int layers) {
    std::string s;
    for (int i = 0; i < layers; i++) {
        s += R"({"anyOf":[{"type":"integer"},)";
    }
    s += R"({"type":"string"})";
    for (int i = 0; i < layers; i++) {
        s += "]}";
    }
    return s;
}

static std::string nested_object_schema(int layers) {
    std::string s;
    for (int i = 0; i < layers; i++) {
        s += R"({"type":"object","properties":{"a":)";
    }
    s += R"({"type":"integer"})";
    for (int i = 0; i < layers; i++) {
        s += R"(},"required":["a"],"additionalProperties":false})";
    }
    return s;
}

// converts `schema`; expects std::invalid_argument whose text contains `needle`
static void guard_expect_invalid(const std::string & name, const std::string & schema, const std::string & needle) {
    fprintf(stderr, "- %s (std::invalid_argument containing \"%s\" expected)\n", name.c_str(), needle.c_str());
    try {
        (void) json_schema_to_grammar(common_json::parse(schema), true);
        guard_fail(name, "converted without any error");
    } catch (const std::invalid_argument & ex) {
        if (std::string(ex.what()).find(needle) == std::string::npos) {
            guard_fail(name, std::string("invalid_argument without \"") + needle + "\": " + ex.what());
        } else {
            fprintf(stderr, "  ok: %.200s\n", ex.what());
        }
    } catch (const std::exception & ex) {
        guard_fail(name, std::string("threw ") + typeid(ex).name() + " (not invalid_argument): " + ex.what());
    }
}

// converts `schema`; expects a grammar that the grammar parser accepts and that contains every fragment
static void guard_expect_ok(const std::string & name, const std::string & schema, const std::vector<std::string> & fragments = {}) {
    fprintf(stderr, "- %s (must still convert)\n", name.c_str());
    try {
        const std::string grammar = json_schema_to_grammar(common_json::parse(schema), true);
        llama_grammar_parser state;
        state.parse(grammar.c_str());
        if (state.symbol_ids.find("root") == state.symbol_ids.end()) {
            guard_fail(name, "grammar has no root rule:\n" + grammar);
            return;
        }
        for (const auto & f : fragments) {
            if (grammar.find(f) == std::string::npos) {
                guard_fail(name, "grammar lacks \"" + f + "\":\n" + grammar);
            }
        }
    } catch (const std::exception & ex) {
        guard_fail(name, std::string("threw ") + typeid(ex).name() + ": " + ex.what());
    }
}

static void test_schema_guards() {
    fprintf(stderr, "#\n# Testing schema recursion / type guards\n#\n");

    // keep-working controls (must pass before AND after the guards)
    guard_expect_ok("32-layer array nest", nested_array_schema(32));
    guard_expect_ok("two properties reusing one non-cyclic $ref",
        R"({"$defs":{"S":{"type":"string"}},"type":"object",
            "properties":{"a":{"$ref":"#/$defs/S"},"b":{"$ref":"#/$defs/S"}},"required":["a","b"],
            "additionalProperties":false})");
    guard_expect_ok("allOf listing one non-cyclic $ref twice keeps counting both",
        R"({"$defs":{"E":{"enum":["a","b"]}},"allOf":[{"$ref":"#/$defs/E"},{"$ref":"#/$defs/E"}]})",
        {R"("\"a\"" | "\"b\"")"});
    guard_expect_ok("allOf through two different non-cyclic $refs",
        R"({"$defs":{"P":{"properties":{"x":{"type":"string"}}},"Q":{"properties":{"y":{"type":"integer"}}}},
            "allOf":[{"$ref":"#/$defs/P"},{"$ref":"#/$defs/Q"}]})",
        {"x-kv", "y-kv"});
    guard_expect_ok("plain recursive $ref stays a recursive rule",
        R"({"$defs":{"Node":{"type":"object","properties":{"value":{"type":"integer"},"next":{"$ref":"#/$defs/Node"}},
            "required":["value"],"additionalProperties":false}},"$ref":"#/$defs/Node"})",
        {"ref-defs-Node ::="});
    // nested "$defs" (an MCP tool schema keeps them inside "properties"'s sibling nodes) are hoisted to the root, so the
    // "#/$defs/S" ref resolves instead of "Error resolving ref" (row S69-ENGREV-LLAMAAGENT-HOIST-NESTED-DEFS-2026-10-27-01)
    guard_expect_ok("nested $defs hoisted to the root so its $ref resolves",
        R"({"type":"object","properties":{"a":{"$ref":"#/$defs/S"},
            "holder":{"type":"object","$defs":{"S":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}}},
            "required":["a"],"additionalProperties":false})",
        {"ref-defs-S ::=", "x-kv"});
    // (the guard's target was a plain {"type":"string"}: the builder maps that to the shared "string" primitive and never emits a
    // "ref-defs-S" rule for it (the property rule "a" is just `a ::= string`), so the expected rule name could never appear;
    // the guard keeps the hoist check but uses a target that does get its own rule)
    // the one cap is the builder cap (COMMON_CHAT_SCHEMA_MAX_DEPTH, the root is one of the levels; tests/test-chat.cpp test_chat_json_depth
    // pins the same boundary). MECHANISM-TESTED: tests/probe-schema-depth.py swept the MSVC Release build (1 MiB stack, main thread and a
    // worker thread): first stack overflow at 553 object / 827 array / 924 anyOf layers; the cap is 256 (128 on Apple) on that measurement
    // (row S69-ENGREV-UPSTREAM-MERGE-SCHEMA-DEPTH-CAP-MEASURE-2026-11-04-01).
    constexpr int deepest = COMMON_CHAT_SCHEMA_MAX_DEPTH - 1;
#if !defined(__APPLE__)
    static_assert(COMMON_CHAT_SCHEMA_MAX_DEPTH == 256, "the fork keeps 255 accepted layers outside Apple (measured 2026-11-06)");
#endif
    guard_expect_ok(std::to_string(deepest) + "-layer array nest (deepest accepted)", nested_array_schema(deepest));
    guard_expect_ok(std::to_string(deepest) + "-layer object nest (deepest accepted; the heaviest stack frames per level)", nested_object_schema(deepest));
    guard_expect_ok(std::to_string(deepest) + "-layer anyOf nest (deepest accepted)", nested_anyof_schema(deepest));
    guard_expect_ok("property literally named $ref is data, not a reference",
        R"({"type":"object","properties":{"$ref":{"type":"string"}},"required":["$ref"],"additionalProperties":false})");

    // type checks: a malformed keyword value is a request error, not a nlohmann type_error
    guard_expect_invalid("format is a number", R"({"format":5})", "format");
    guard_expect_invalid("string minLength is a string", R"({"type":"string","minLength":"3"})", "minLength");
    guard_expect_invalid("string maxLength is an object", R"({"type":"string","maxLength":{}})", "maxLength");
    guard_expect_invalid("array minItems is a string", R"({"type":"array","items":{"type":"integer"},"minItems":"1"})", "minItems");
    guard_expect_invalid("$ref is a number", R"({"$ref":5})", "$ref");
    guard_expect_invalid("allOf part with a numeric $ref", R"({"allOf":[{"$ref":5}]})", "$ref");
    guard_expect_invalid("pattern is a number", R"({"type":"string","pattern":5})", "pattern");
    guard_expect_invalid("integer minimum is a string", R"({"type":"integer","minimum":"3"})", "minimum");
    guard_expect_invalid("integer exclusiveMaximum is a bool", R"({"type":"integer","exclusiveMaximum":true})", "exclusiveMaximum");

    // depth cap (COMMON_CHAT_SCHEMA_MAX_DEPTH): 400 nested layers is refused with a message naming the cap
    constexpr int past = COMMON_CHAT_SCHEMA_MAX_DEPTH;
    guard_expect_invalid("400-layer array nest", nested_array_schema(400), "maximum depth");
    guard_expect_invalid(std::to_string(past) + "-layer array nest (one past the cap)", nested_array_schema(past), "maximum depth");
    guard_expect_invalid("400-layer anyOf nest", nested_anyof_schema(400), "maximum depth");
    guard_expect_invalid(std::to_string(past) + "-layer object nest (one past the cap)", nested_object_schema(past), "maximum depth");
    guard_expect_invalid("20000-layer array nest (would overflow any stack)", nested_array_schema(20000), "maximum depth");
    // The deep member is the LAST key on purpose. A key inserted after a deep value makes the PARSER deep-copy that value
    // when the object's key vector grows (nlohmann ordered_map: pair<const string, json> has no noexcept move), and that
    // copy is recursive: on the MSVC build (1 MiB stack) 5000 levels overflow inside common_json::parse, before any
    // converter code runs (gate b1019u, rc 0xC00000FD; MinGW's 2 MiB stack hid it). That is a parser defect with its own
    // row (S69-ENGREV-LOCALAI2-REQUEST-BODY-JSON-DEPTH-CAP-PARSER-COPY-2026-10-28-01); this case pins the converter: a
    // missing depth check would let json_schema_to_grammar copy the same 5000-level tree, the same recursion.
    guard_expect_invalid("5000-layer nest beside a \"#/\" ref (the root copy must not run)",
        R"({"$defs":{"S":{"type":"string"}},"anyOf":[{"$ref":"#/$defs/S"}],"x-deep":)" + nested_array_schema(5000) + "}",
        "maximum depth");

    // cyclic $ref reached through allOf (add_component recursed on _refs[...] with no visited set)
    guard_expect_invalid("allOf self $ref cycle",
        R"({"$defs":{"A":{"$ref":"#/$defs/A"}},"allOf":[{"$ref":"#/$defs/A"}]})", "cyclic $ref");
    guard_expect_invalid("allOf indirect $ref cycle A to B to A",
        R"({"$defs":{"A":{"$ref":"#/$defs/B"},"B":{"$ref":"#/$defs/A"}},"allOf":[{"$ref":"#/$defs/A"}]})", "cyclic $ref");
    guard_expect_invalid("allOf cycle inside an anyOf member",
        R"({"$defs":{"A":{"$ref":"#/$defs/A"}},"allOf":[{"anyOf":[{"$ref":"#/$defs/A"}]}]})", "cyclic $ref");

    // the same degenerate cycle reached by a plain $ref (no allOf): the rule would be `refdefs-A ::= refdefs-A`, a left recursion
    // the grammar parser rejects at load (row S69-ENGREV-LOCALAI2-PLAIN-REF-DEGENERATE-SELF-REF-GRAMMAR-LOAD-2026-10-28-01)
    guard_expect_invalid("plain self $ref cycle",
        R"({"$defs":{"A":{"$ref":"#/$defs/A"}},"$ref":"#/$defs/A"})", "cyclic $ref");
    guard_expect_invalid("plain indirect $ref cycle A to B to A",
        R"({"$defs":{"A":{"$ref":"#/$defs/B"},"B":{"$ref":"#/$defs/A"}},"$ref":"#/$defs/A"})", "cyclic $ref");
    guard_expect_invalid("self $ref cycle as an object property",
        R"({"type":"object","properties":{"p":{"$ref":"#/$defs/A"}},"$defs":{"A":{"$ref":"#/$defs/A"}}})", "cyclic $ref");
    // controls: an alias chain that ends in a real node, and a recursive type reached through an alias, still convert
    guard_expect_ok("alias chain A to B to a string",
        R"({"$defs":{"A":{"$ref":"#/$defs/B"},"B":{"type":"string"}},"$ref":"#/$defs/A"})");
    guard_expect_ok("recursive type reached through an alias",
        R"({"$defs":{"A":{"$ref":"#/$defs/Node"},"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/A"}},"additionalProperties":false}},"$ref":"#/$defs/A"})",
        {"ref-defs-Node ::="});

    // the info path (tool parameter inspection in chat.cpp) must also survive deep and malformed input
    fprintf(stderr, "- info path: deep anyOf nest and malformed keywords neither throw nor overflow\n");
    try {
        common_schema_info info;
        auto deep = common_json::parse(nested_anyof_schema(5000));
        info.resolve_refs(deep);
        if (info.resolves_to_string(deep)) {
            guard_fail("info deep", "resolves_to_string walked 5000 anyOf levels (no depth cap)");
        }
        // a "#/" ref makes resolve_refs copy the whole root: the depth check must stop that before the copy
        // (deep member last, so the parse itself does not copy it; see the guard case above)
        auto deep_ref = common_json::parse(R"({"$defs":{"S":{"type":"string"}},"anyOf":[{"$ref":"#/$defs/S"}],"x-deep":)" +
                                           nested_array_schema(5000) + "}");
        info.resolve_refs(deep_ref);
        (void) info.resolves_to_string(deep_ref);
        auto bad = common_json::parse(R"({"anyOf":[{"$ref":5},{"format":5},{"type":"integer"}]})");
        info.resolve_refs(bad);
        if (info.resolves_to_string(bad)) {
            guard_fail("info bad", "numeric $ref / format reported as string");
        }
    } catch (const std::exception & ex) {
        guard_fail("info path", std::string("threw ") + typeid(ex).name() + ": " + ex.what());
    }

    if (g_guard_failures != 0) {
        fprintf(stderr, "#\n# %d schema guard check(s) FAILED\n#\n", g_guard_failures);
        assert(false);
    }
    fprintf(stderr, "All schema guard tests passed!\n");
}

int main() {
    // stderr is block-buffered when redirected to a file or pipe on this host (MinGW/UCRT): a crash inside a converter
    // case (stack overflow) would lose every line printed before it, and the log would not say which case died.
    setvbuf(stderr, nullptr, _IONBF, 0);
    fprintf(stderr, "LLAMA_NODE_AVAILABLE = %s\n", getenv("LLAMA_NODE_AVAILABLE") ? "true" : "false");
    fprintf(stderr, "LLAMA_PYTHON_AVAILABLE = %s\n", getenv("LLAMA_PYTHON_AVAILABLE") ? "true" : "false");

    test_schema_guards();

    test_resolves_to_string();

    test_all("C++", [](const TestCase & tc) {
        try {
            tc.verify(json_schema_to_grammar(common_json::parse(tc.schema), true));
            tc.verify_status(SUCCESS);
        } catch (const std::invalid_argument & ex) {
            fprintf(stderr, "Error: %s\n", ex.what());
            tc.verify_status(FAILURE);
        }
    });

    // a document parsed up front gives the same grammar as the JSON, recursion included
    {
        fprintf(stderr, "- parsed document\n");
        auto schema = common_json::parse(R"""({
            "$ref": "#/$defs/node",
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/$defs/node"}, "leaf": {}},
                    "additionalProperties": false
                }
            }
        })""");
        assert(json_schema_to_grammar(common_chat_schema_from_json(schema)) == json_schema_to_grammar(schema, true));
    }

    // a property node carries its $ref target, so its grammar names the ref rule
    {
        fprintf(stderr, "- sub-schema $ref\n");
        auto parameters = common_json::parse(R"""({
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/item"}},
            "$defs": {
                "item": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                    "additionalProperties": false
                }
            }
        })""");
        TestCase tc {
            SUCCESS,
            "sub-schema $ref",
            "",
            R"""(
                char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
                ref-defs-item ::= "{" space ref-defs-item-a-kv space "}"
                ref-defs-item-a-kv ::= "\"a\"" space ":" space string
                root ::= ref-defs-item
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
                string ::= "\"" char* "\""
            )""",
        };
        auto doc = common_chat_schema_from_json(parameters);
        tc.verify(build_grammar([&](const common_grammar_builder & builder) {
            const auto & item = static_cast<const common_chat_schema_object &>(*doc.root).properties.at(0);
            builder.add_schema("root", *item.schema);
        }));
    }

    test_all("Check the expectations parse", [](const TestCase & tc) {
        if (tc.expected_status == SUCCESS) {
            tc.verify_expectation_parseable();
        }
    });
}
