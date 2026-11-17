// A return from main() must not lose log lines that are still queued for the worker thread.
//
// common_log_main() is leaked on purpose (upstream issue #22142), so nothing used to print the
// queue's tail at exit. Found 2026-10-27 in gate b1019h: llama-server refused a bad hot-swap
// config with rc 1, and its log ended before the refusal line.
//
// The test runs itself as a child: the child logs N numbered lines to a file through the main
// logger and returns at once, without flushing. The parent counts the lines that reached the file.
// Before the atexit drain in common/log.cpp, most runs ended with lines missing.

#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const int N_LINES = 500; // under the queue's default capacity (512), so add() never blocks
static const int N_RUNS  = 5;
static const int N_FREE_RUNS = 30;

static int count_lines(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        s.append(buf, n);
    }
    fclose(f);
    int count = 0;
    for (size_t p = s.find("drain-line "); p != std::string::npos; p = s.find("drain-line ", p + 1)) {
        count++;
    }
    return count;
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "child") == 0) {
        common_log_set_file(common_log_main(), argv[2]);
        for (int i = 0; i < N_LINES; i++) {
            LOG_INF("drain-line %d\n", i);
        }
        return 0; // deliberately no common_log_flush(): the exit path itself must print the queue
    }

    // A child that frees the singleton (as tests/test-log.cpp does) and then returns from main() must not
    // crash: the at-exit drain used to run on the freed logger (S69-FORK-TEST-LOG-AT-EXIT-DRAIN-ACCESS-
    // VIOLATION-FLAKY-2026-11-09-01, 1 crash in ~20-240 runs), so this is a probabilistic regression guard.
    if (argc == 2 && strcmp(argv[1], "free-child") == 0) {
        for (int i = 0; i < 50; i++) {
            LOG_INF("free-child line %d\n", i);
        }
        common_log_free(common_log_main());
        return 0;
    }

    const char * path = "test-log-drain-at-exit.tmp.log";
    int failed = 0;
    for (int r = 0; r < N_RUNS; r++) {
        remove(path);
        std::string cmd = std::string("\"") + argv[0] + "\" child " + path;
#ifdef _WIN32
        // cmd /c strips one pair of outer quotes, so wrap the whole line in one more
        cmd = "\"" + cmd + " >NUL 2>&1\"";
#else
        cmd += " >/dev/null 2>&1";
#endif
        const int rc    = std::system(cmd.c_str());
        const int lines = count_lines(path);
        printf("run %d: child rc %d, %d of %d lines in the log\n", r, rc, lines, N_LINES);
        if (rc != 0 || lines != N_LINES) {
            failed++;
        }
    }
    remove(path);

    int free_failed = 0;
    for (int r = 0; r < N_FREE_RUNS; r++) {
        std::string cmd = std::string("\"") + argv[0] + "\" free-child";
#ifdef _WIN32
        cmd = "\"" + cmd + " >NUL 2>&1\"";
#else
        cmd += " >/dev/null 2>&1";
#endif
        if (std::system(cmd.c_str()) != 0) {
            free_failed++;
        }
    }
    printf("free-child: %d of %d runs exited non-zero\n", free_failed, N_FREE_RUNS);
    failed += free_failed;

    printf("%s: %d of %d runs lost log lines or failed\n", failed ? "FAIL" : "PASS", failed, N_RUNS);
    return failed ? 1 : 0;
}
