#pragma once

// fork: what a failed child load says when the child never logged a line that status_error_line() (server-models.cpp)
// recognises as an error, for instance a segfault or a silent exit(3). The router then has only the exit code and the
// last few lines the child wrote, so those are kept and joined into the load-failure error.
// Row S69-ENGREV-JAN-LOAD-FAIL-CAUSE-AND-CHILD-DEATH-2026-10-27-01 (residual); no upstream code used.
// Header-only on purpose: tools/server/tests_cpp/test_server_load_fail.cpp builds it without the server.

#include <cstddef>
#include <deque>
#include <mutex>
#include <string>

// the last N lines a child wrote to stderr, newest last, each clipped; thread-safe (the monitor thread writes, request threads read)
struct server_output_ring {
    static constexpr size_t default_max_lines = 8;
    static constexpr size_t default_max_line  = 512;

    explicit server_output_ring(size_t max_lines = default_max_lines, size_t max_line = default_max_line)
        : max_lines(max_lines ? max_lines : 1), max_line(max_line ? max_line : 1) {}

    void push(const std::string & raw) {
        std::string line = raw;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) {
            return;
        }
        if (line.size() > max_line) {
            line.resize(max_line); // keep the head of an over-long line; the cause is usually at the front
        }
        std::lock_guard<std::mutex> lk(mu);
        lines.push_back(std::move(line));
        while (lines.size() > max_lines) {
            lines.pop_front();
        }
    }

    // oldest first, joined with '\n'; "" when nothing was written
    std::string text() const {
        std::lock_guard<std::mutex> lk(mu);
        std::string out;
        for (const auto & l : lines) {
            if (!out.empty()) {
                out += '\n';
            }
            out += l;
        }
        return out;
    }

private:
    size_t max_lines;
    size_t max_line;
    mutable std::mutex mu;
    std::deque<std::string> lines;
};

// The text kept for a finished child.
//   * the matched error lines when there are any (they already say why);
//   * else, when the child exited non-zero, "exit code X" plus its last output lines when it wrote any;
//   * else "" (a clean exit with nothing to report).
inline std::string server_load_failure_detail(const std::string & err_tail, const std::string & recent_output, int exit_code) {
    if (!err_tail.empty()) {
        return err_tail;
    }
    if (exit_code == 0) {
        return "";
    }
    std::string out = "exit code " + std::to_string(exit_code);
    if (!recent_output.empty()) {
        out += "; last output: ";
        for (char c : recent_output) {
            out += c == '\n' ? '|' : c; // one line in a JSON error string and in the log
        }
    }
    return out;
}
