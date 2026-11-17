// /v1/systemone (typed decision models): the request parsing and the answer formatting of server_decision_context,
// without a model. The served path (label logits at the last prompt token) is checked end to end by
// tools/server/tests/unit/test_systemone.py. Ported 2026-11-04 from the fork's server_systemone_context test (ik_llama.cpp
// c069e89e1 #2592) when upstream's server_decision replaced that module: same checks, upstream's names.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-decision.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

static bool throws_invalid(const server_decision_context & so, const json & body) {
    try {
        so.parse_questions(body);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

// the fork's count_images(): upstream's parse_state() appends each media file of the request to files
static size_t count_media(const server_decision_context & so, const json & body) {
    std::vector<raw_buffer> files;
    so.parse_state(body, files);
    return files.size();
}

int main() {
    // OpenJev-like: 26 labels, noul shown [true, false]
    server_decision_context so;
    so.init_for_test(COMMON_DECISION_TYPE_OPENJEV, 26, true, {});

    const json body = json::parse(R"({
        "state": {"ticket": "printer on fire"},
        "questions": {
            "route": {"type": "choice", "instructions": "who handles it", "criteria": {"it": "IT desk", "facilities": null, "security": "guards"}},
            "urgency": {"type": "score", "instructions": "how urgent", "criteria": ["none", "low", "high", "now"]},
            "fire": {"type": "noul", "instructions": "is there a fire"}
        }
    })");
    std::vector<server_decision_question> qs = so.parse_questions(body);
    CHECK(qs.size() == 3, "3 questions, got %zu", qs.size());
    if (qs.size() == 3) {
        CHECK(qs[0].id == "route" && qs[0].type == SERVER_DECISION_QUESTION_CHOICE && qs[0].options.size() == 3, "choice parsed");
        CHECK(qs[0].options[0].key == "it" && qs[0].options[1].key == "facilities", "choice keys keep request order");
        CHECK(qs[1].type == SERVER_DECISION_QUESTION_SCORE && qs[1].options.size() == 4 && qs[1].options[3].key == "3", "score levels 0..3");
        CHECK(qs[2].type == SERVER_DECISION_QUESTION_NOUL && qs[2].options.size() == 2 && qs[2].options[0].key == "true", "openjev noul is [true, false]");
        for (const auto & q : qs) CHECK(so.n_variants(q) == 1, "openjev: one variant per question");

        // choice: softmax of the label logits, temperature 1 (no metadata)
        const json a = so.format_answer(qs[0], {{2.0f, 0.0f, -1.0f}});
        const double z = std::exp(2.0) + 1.0 + std::exp(-1.0);
        CHECK(a.at("choice") == "it", "choice is the argmax");
        CHECK(near(a.at("probabilities").at("it").get<double>(), std::exp(2.0)/z), "p(it)");
        CHECK(near(a.at("probabilities").at("security").get<double>(), std::exp(-1.0)/z), "p(security)");
        const double pmax = std::exp(2.0)/z;
        CHECK(near(a.at("confidence").get<double>(), (pmax - 1.0/3)/(1 - 1.0/3)), "choice confidence");

        // score: expected level
        const json s = so.format_answer(qs[1], {{0.0f, 0.0f, 0.0f, 0.0f}});
        CHECK(near(s.at("score").get<double>(), 1.5), "uniform score expectation 1.5, got %g", s.at("score").get<double>());
        CHECK(near(s.at("confidence").get<double>(), 0.0, 1e-12), "uniform score confidence clamps to 0");
        CHECK(s.at("legend").at("2") == "high", "legend");

        // noul: probability of the "true" option (first here)
        const json n = so.format_answer(qs[2], {{1.0f, 0.0f}});
        CHECK(near(n.at("noul").get<double>(), std::exp(1.0)/(std::exp(1.0) + 1.0)), "noul = p(true)");

        // a result of the wrong width is refused, not read past
        bool threw = false;
        try { so.format_answer(qs[0], {{1.0f, 2.0f}}); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "short score vector must throw");
    }

    // bad requests
    CHECK(throws_invalid(so, json::parse(R"({"questions": {"a": {"type": "noul", "instructions": "x"}}})")), "missing state");
    CHECK(throws_invalid(so, json::parse(R"({"state": 1, "questions": {}})")), "empty questions");
    CHECK(throws_invalid(so, json::parse(R"({"state": 1, "questions": {"a": {"type": "noul"}}})")), "missing instructions");
    CHECK(throws_invalid(so, json::parse(R"({"state": 1, "questions": {"a": {"type": "guess", "instructions": "x"}}})")), "unknown type");
    CHECK(throws_invalid(so, json::parse(R"({"state": 1, "questions": {"a": {"type": "score", "instructions": "x", "criteria": ["one"]}}})")), "score with 1 level");
    {
        server_decision_context small;
        small.init_for_test(COMMON_DECISION_TYPE_OPENJEV, 2, true, {});
        CHECK(throws_invalid(small, json::parse(R"({"state": 1, "questions": {"a": {"type": "choice", "instructions": "x", "criteria": {"p": 1, "q": 2, "r": 3}}}})")),
                "more options than labels");
    }
    // images: counted from "images" and from chat-message image parts; non-data URLs refused
    CHECK(count_media(so, json::parse(R"({"state": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}], "images": ["data:image/png;base64,AA"]})")) == 2,
            "two images counted");
    {
        bool threw = false;
        try { count_media(so, json::parse(R"({"state": 1, "images": ["http://x/y.png"]})")); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "a non-data image URL must be refused");
    }

    // lev: a choice is asked in 2 orders and the second is read reversed; noul reads a 9-level rating
    {
        server_decision_context lev;
        lev.init_for_test(COMMON_DECISION_TYPE_LEV, 30, false, {{"choice.small", 2.0f}});
        const auto lq = lev.parse_questions(json::parse(R"({"state": "s", "questions": {
            "c": {"type": "choice", "instructions": "x", "criteria": {"a": 1, "b": 2}},
            "n": {"type": "noul", "instructions": "y"}}})"));
        CHECK(lq.size() == 2 && lev.n_variants(lq[0]) == 2 && lev.n_variants(lq[1]) == 1, "lev variants");
        if (lq.size() == 2) {
            CHECK(lq[1].options[0].key == "false", "lev noul is [false, true]");
            // variant 0 favours "a" by 2 logits; variant 1 shows [b, a] and favours its 2nd output (= "a") by 2
            const json a = lev.format_answer(lq[0], {{2.0f, 0.0f}, {0.0f, 2.0f}});
            const double pa = 1.0/(1.0 + std::exp(-2.0/2.0)); // temperature choice.small = 2
            CHECK(a.at("choice") == "a", "lev choice after un-reversing");
            CHECK(near(a.at("probabilities").at("a").get<double>(), pa), "lev p(a) %g want %g", a.at("probabilities").at("a").get<double>(), pa);
            // control: if the second variant were NOT reversed, p(a) would be 0.5
            CHECK(!near(a.at("probabilities").at("a").get<double>(), 0.5, 1e-3), "control: reversal matters");
            std::vector<float> r(9, 0.0f); r[8] = 50.0f;
            const json n = lev.format_answer(lq[1], {r});
            CHECK(near(n.at("noul").get<double>(), 1.0, 1e-9), "lev noul from the top rating");
        }
    }

    // token limits come from the GGUF ("<arch>.decision.<key>"), the old literals when the key is absent (row
    // S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01). Before this change the five values were static consts.
    {
        auto meta = [](std::map<std::string, std::string> kv) {
            return [kv](const std::string & key) { auto it = kv.find(key); return it == kv.end() ? std::string() : it->second; };
        };
        // old GGUF (no key): the former literals
        const server_decision_limits old_gguf = server_decision_limits::from_meta(meta({}));
        CHECK(old_gguf.h2o_max_state_tokens == 32000, "fallback h2o state %zu", old_gguf.h2o_max_state_tokens);
        CHECK(old_gguf.d1omni_max_tokens == 16384 && old_gguf.d1omni_max_tokens_image == 896 && old_gguf.d1omni_max_tokens_audio == 15360,
              "fallback d1omni %zu/%zu/%zu", old_gguf.d1omni_max_tokens, old_gguf.d1omni_max_tokens_image, old_gguf.d1omni_max_tokens_audio);
        CHECK(old_gguf.max_option_tokens == 48, "fallback option cap %zu", old_gguf.max_option_tokens);
        // a model that carries the keys: its values
        const server_decision_limits own = server_decision_limits::from_meta(meta({
            {"max_state_tokens", "64000"}, {"max_tokens", "32768"}, {"max_tokens_image", "1792"}, {"max_tokens_audio", "30720"}, {"max_option_tokens", "96"}}));
        CHECK(own.h2o_max_state_tokens == 64000, "derived h2o state %zu", own.h2o_max_state_tokens);
        CHECK(own.d1omni_max_tokens == 32768 && own.d1omni_max_tokens_image == 1792 && own.d1omni_max_tokens_audio == 30720,
              "derived d1omni %zu/%zu/%zu", own.d1omni_max_tokens, own.d1omni_max_tokens_image, own.d1omni_max_tokens_audio);
        CHECK(own.max_option_tokens == 96, "derived option cap %zu", own.max_option_tokens);
        // one key does not touch the others
        const server_decision_limits one = server_decision_limits::from_meta(meta({{"max_tokens", "8192"}}));
        CHECK(one.d1omni_max_tokens == 8192 && one.d1omni_max_tokens_image == 896 && one.h2o_max_state_tokens == 32000, "single key");
        // garbage, zero and negative keep the fallback (control: a bad key must not give a 0 limit)
        const server_decision_limits bad = server_decision_limits::from_meta(meta({
            {"max_state_tokens", "0"}, {"max_tokens", "abc"}, {"max_tokens_image", "-5"}, {"max_option_tokens", ""}}));
        CHECK(bad.h2o_max_state_tokens == 32000 && bad.d1omni_max_tokens == 16384 && bad.d1omni_max_tokens_image == 896 && bad.max_option_tokens == 48,
              "invalid keys keep the fallback");
        // the context uses the limits it was given, and a context of the test has the fallback
        server_decision_context ctx;
        ctx.init_for_test(COMMON_DECISION_TYPE_OPENJEV, 2, true, {});
        CHECK(ctx.get_limits().d1omni_max_tokens == 16384, "init_for_test fallback");
        ctx.set_limits_for_test(own);
        CHECK(ctx.get_limits().d1omni_max_tokens == 32768 && ctx.get_limits().h2o_max_state_tokens == 64000, "context holds the derived limits");
    }

    // n_options_max of the six marker-token model types (kev, laya, clef, vllm-decision2, lfm2-d1, lfm2-d1-omni) comes from the GGUF key
    // "<arch>.decision.n_options_max", 255 when the key is absent (row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01, lint rows M1-M6).
    {
        auto meta = [](std::map<std::string, std::string> kv) {
            return [kv](const std::string & key) { auto it = kv.find(key); return it == kv.end() ? std::string() : it->second; };
        };
        const server_decision_limits absent = server_decision_limits::from_meta(meta({}));
        const server_decision_limits own    = server_decision_limits::from_meta(meta({{"n_options_max", "16"}}));
        const server_decision_limits bad0   = server_decision_limits::from_meta(meta({{"n_options_max", "0"}}));
        const server_decision_limits badt   = server_decision_limits::from_meta(meta({{"n_options_max", "abc"}}));
        CHECK(absent.n_options_max == 255, "absent key -> %zu, want the 255 fallback", absent.n_options_max);
        CHECK(own.n_options_max == 16, "key 16 -> %zu", own.n_options_max);
        CHECK(bad0.n_options_max == 255 && badt.n_options_max == 255, "0 / text keep the fallback: %zu %zu", bad0.n_options_max, badt.n_options_max);
        const common_decision_type marker_types[] = {
            COMMON_DECISION_TYPE_KEV, COMMON_DECISION_TYPE_LAYA, COMMON_DECISION_TYPE_CLEF,
            COMMON_DECISION_TYPE_VLLM_DECISION2, COMMON_DECISION_TYPE_LFM2_D1, COMMON_DECISION_TYPE_LFM2_D1_OMNI,
        };
        for (const auto mt : marker_types) {
            CHECK(server_decision_n_options_max(mt, own) == 16, "type %d with key 16 -> %zu", (int) mt, server_decision_n_options_max(mt, own));
            CHECK(server_decision_n_options_max(mt, absent) == 255, "type %d without key -> %zu", (int) mt, server_decision_n_options_max(mt, absent));
        }
        // control: the types whose option count is set by their label list are not touched by the key
        CHECK(server_decision_n_options_max(COMMON_DECISION_TYPE_OPENJEV, own) == 0, "openjev");
        CHECK(server_decision_n_options_max(COMMON_DECISION_TYPE_LEV, own) == 0, "lev");
        CHECK(server_decision_n_options_max(COMMON_DECISION_TYPE_H2O_LIGHTNING, own) == 0, "h2o-lightning");
    }

    if (fails) {
        fprintf(stderr, "%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("all /v1/systemone unit checks passed\n");
    return 0;
}
