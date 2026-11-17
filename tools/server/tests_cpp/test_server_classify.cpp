// Standalone test for server-classify.h. Build: g++ -std=c++17 -I.. test_server_classify.cpp -o t && ./t
#include "server-classify.h"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    const std::vector<std::string> labels = { "contract", "statute", "case" };

    // 3-label single-label head: probabilities sum to 1 within 1e-5, labels equal the GGUF labels, winner is the max logit
    {
        const server_classify_result r = server_classify_make({ 0.5f, 2.0f, -1.0f }, labels, false);
        CHECK(r.scores.size() == 3);
        float sum = 0;
        for (const auto & s : r.scores) { sum += s.probability; }
        CHECK(std::fabs(sum - 1.0f) < 1e-5f);
        CHECK(r.scores[0].label == "contract" && r.scores[1].label == "statute" && r.scores[2].label == "case");
        CHECK(r.label == "statute");
        CHECK(std::fabs(r.score - r.scores[1].probability) < 1e-9f);
    }
    // large logits do not overflow (max-shifted softmax)
    {
        const std::vector<float> p = server_classify_probs({ 1000.0f, 1001.0f }, false);
        CHECK(std::isfinite(p[0]) && std::isfinite(p[1]));
        CHECK(std::fabs(p[0] + p[1] - 1.0f) < 1e-5f);
        CHECK(p[1] > p[0]);
    }
    // multi-label: independent sigmoids, they need NOT sum to 1
    {
        const std::vector<float> p = server_classify_probs({ 3.0f, 3.0f, -3.0f }, true);
        CHECK(std::fabs(p[0] - 1.0f / (1.0f + std::exp(-3.0f))) < 1e-6f);
        CHECK(p[0] + p[1] + p[2] > 1.5f);
    }
    // one output: a sigmoid, not a softmax that would always be 1
    {
        const std::vector<float> p = server_classify_probs({ 0.0f }, false);
        CHECK(p.size() == 1 && std::fabs(p[0] - 0.5f) < 1e-6f);
    }
    // missing labels fall back to the class index; empty input is empty
    {
        const server_classify_result r = server_classify_make({ 1.0f, 0.0f }, {}, false);
        CHECK(r.label == "0" && r.scores[1].label == "1");
        const server_classify_result e = server_classify_make({}, labels, false);
        CHECK(e.scores.empty() && e.label.empty());
    }

    if (fails) { std::printf("%d FAILED\n", fails); return 1; }
    std::printf("ok\n");
    return 0;
}
