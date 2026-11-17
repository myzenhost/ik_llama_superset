// common_grammar_mask_overlap (common/grammar-mask-overlap.h): the worker-thread overlap of the grammar mask with the next decode.
// Row S69-EXLLAMAV2-PICK-GRAMMAR-MASK-OVERLAP-MEASURE-2026-10-20-01. Before, the llguidance sampler computed the mask inside apply(), after
// the decode, so mask time and decode time ADDED. The stub job below plays a matcher step; the "decode" is a sleep on the calling thread.
// Needs no model and no llguidance.
#include "grammar-mask-overlap.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

using clk = std::chrono::steady_clock;
static long ms_since(clk::time_point t0) {
    return (long) std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
}
static void nap(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

static void setenv_portable(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main() {
    // 1. overlap: a 200 ms job and a 200 ms "decode" finish in about 200 ms, not 400 (the synchronous sum)
    {
        const auto t0 = clk::now();
        common_grammar_mask_overlap o;
        o.start([] { nap(200); return 0; });
        nap(200);  // the decode
        const int rc = o.finish();
        const long dt = ms_since(t0);
        CHECK(rc == 0, "overlap job result %d", rc);
        CHECK(dt < 340, "overlapped run took %ld ms (serial would be ~400)", dt);
        CHECK(dt >= 190, "overlapped run took %ld ms, less than the longest part", dt);
    }
    // control: the same two parts done one after the other really do add up (proves the clock sees the difference)
    {
        const auto t0 = clk::now();
        nap(200);
        nap(200);
        CHECK(ms_since(t0) >= 390, "serial control took %ld ms", ms_since(t0));
    }
    // 2. the job runs on another thread and its result comes back
    {
        const auto caller = std::this_thread::get_id();
        std::thread::id seen;
        common_grammar_mask_overlap o;
        o.start([&] { seen = std::this_thread::get_id(); return 7; });
        CHECK(o.pending(), "pending after start");
        CHECK(o.finish() == 7, "job result is returned");
        CHECK(!o.pending(), "not pending after finish");
        CHECK(seen != caller, "job ran on the calling thread");
    }
    // 3. nothing pending: finish returns the idle value at once, never waits
    {
        common_grammar_mask_overlap o;
        const auto t0 = clk::now();
        CHECK(o.finish(5) == 5, "idle finish value");
        CHECK(ms_since(t0) < 50, "idle finish waited %ld ms", ms_since(t0));
    }
    // 4. start() joins the pending job first: its work is complete before the next job begins, and only one job runs at a time
    {
        std::atomic<int> running{0};
        std::atomic<int> peak{0};
        std::atomic<int> done{0};
        auto job = [&] {
            const int r = ++running;
            int p = peak.load();
            while (r > p && !peak.compare_exchange_weak(p, r)) {}
            nap(60);
            --running;
            ++done;
            return 0;
        };
        common_grammar_mask_overlap o;
        o.start(job);
        o.start(job);
        o.start(job);
        o.finish();
        CHECK(done == 3, "jobs completed: %d", done.load());
        CHECK(peak == 1, "two jobs ran at once (peak %d)", peak.load());
    }
    // 5. drop() joins: the job's effect is visible the moment drop returns (the matcher is safe to touch)
    {
        std::atomic<bool> flag{false};
        common_grammar_mask_overlap o;
        o.start([&] { nap(80); flag = true; return 0; });
        CHECK(!flag, "job finished before drop");
        o.drop();
        CHECK(flag, "drop returned before the job finished");
        CHECK(!o.pending(), "pending after drop");
    }
    // 6. the destructor joins (the job uses a stack variable that dies right after)
    {
        std::atomic<bool> flag{false};
        {
            common_grammar_mask_overlap o;
            o.start([&] { nap(80); flag = true; return 0; });
        }
        CHECK(flag, "destructor returned before the job finished");
    }
    // 7. an exception in the job is a non-zero result, not a terminate
    {
        common_grammar_mask_overlap o;
        o.start([]() -> int { throw std::runtime_error("matcher blew up"); });
        CHECK(o.finish() != 0, "a throwing job must report failure");
        CHECK(!o.pending(), "pending after a throwing job");
    }
    // 8. a failing job result passes through unchanged (the sampler fails closed on it)
    {
        common_grammar_mask_overlap o;
        o.start([] { return -1; });
        CHECK(o.finish() == -1, "failing result passes through");
    }
    // 9. move-assign over a pending job joins it first
    {
        std::atomic<bool> flag{false};
        common_grammar_mask_overlap a;
        a.start([&] { nap(80); flag = true; return 0; });
        common_grammar_mask_overlap b;
        b.start([] { return 3; });
        a = std::move(b);
        CHECK(flag, "move-assign did not join the old job");
        CHECK(a.finish() == 3, "moved job result");
    }
    // 10. LLAMA_GRAMMAR_MASK_SYNC turns the overlap off; unset, empty and "0" leave it on
    {
        setenv_portable("LLAMA_GRAMMAR_MASK_SYNC", "");
        CHECK(common_grammar_mask_overlap::enabled(), "enabled with empty value");
        setenv_portable("LLAMA_GRAMMAR_MASK_SYNC", "0");
        CHECK(common_grammar_mask_overlap::enabled(), "enabled with 0");
        setenv_portable("LLAMA_GRAMMAR_MASK_SYNC", "1");
        CHECK(!common_grammar_mask_overlap::enabled(), "still enabled with 1");
        setenv_portable("LLAMA_GRAMMAR_MASK_SYNC", "");
    }

    if (g_fails) {
        std::printf("test-grammar-mask-overlap: %d FAILED\n", g_fails);
        return 1;
    }
    std::printf("test-grammar-mask-overlap: ok\n");
    return 0;
}
