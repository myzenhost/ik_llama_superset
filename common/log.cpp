#include "common.h"
#include "log.h"
#include "json.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

int common_log_verbosity_thold = LOG_DEFAULT_LLAMA;

int common_log_get_verbosity_thold(void) {
    return common_log_verbosity_thold;
}

void common_log_set_verbosity_thold(int verbosity) {
    common_log_verbosity_thold = verbosity;
}

static bool common_log_jsonl = false;

bool common_log_get_jsonl(void) {
    return common_log_jsonl;
}

void common_log_set_jsonl(bool jsonl) {
    common_log_jsonl = jsonl;
}

static int64_t t_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// SOURCE-READ: examples/server/server-common.cpp:26-65 @ 35f0cac75 (server_log(), the fork's own MIT
// code; reimplemented here as a small self-contained escaper rather than pulling nlohmann/json into
// this low-level file -- row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-
// 01, item 4). Escapes control characters and the two bytes JSON requires ('"' and '\\'); a message's
// own trailing "\n" (most LOG_* format strings end in one) comes through as a literal \n escape
// inside the "msg" string, same as nlohmann's dump() would render it, so the line stays one JSON
// object regardless.
//
// Coordinator review point 3 (2026-09-27): a log message can legitimately contain a truncated or
// otherwise invalid UTF-8 byte (a token piece cut mid-codepoint, a raw byte from a binary blob logged
// by mistake, etc). Passing bytes >= 0x80 straight through, as an earlier draft did, can then emit a
// malformed multi-byte sequence inside the JSON string, which a strict reader may refuse even though
// every individual byte was "escaped" correctly. This validates each multi-byte sequence before
// copying it through and replaces anything invalid (bad continuation byte, overlong encoding,
// surrogate range, or a sequence truncated by the end of the message) with U+FFFD, one byte at a time
// so it resyncs instead of eating the rest of the message.
static void json_escape_append(std::string & out, const char * s) {
    const unsigned char * p = (const unsigned char *) s;
    while (*p) {
        const unsigned char c = *p;

        if (c < 0x80) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += (char) c;
                    }
            }
            ++p;
            continue;
        }

        int      len  = 0;
        unsigned char lead_mask = 0;
        unsigned min_cp = 0;
        if      ((c & 0xE0) == 0xC0) { len = 2; lead_mask = 0x1F; min_cp = 0x80;    }
        else if ((c & 0xF0) == 0xE0) { len = 3; lead_mask = 0x0F; min_cp = 0x800;   }
        else if ((c & 0xF8) == 0xF0) { len = 4; lead_mask = 0x07; min_cp = 0x10000; }

        bool     valid = len > 0;
        unsigned cp    = valid ? (unsigned) (c & lead_mask) : 0;
        for (int i = 1; valid && i < len; i++) {
            const unsigned char cc = p[i];
            if (cc == '\0' || (cc & 0xC0) != 0x80) {
                valid = false;
                break;
            }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (valid && (cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) {
            valid = false;
        }

        if (valid) {
            out.append((const char *) p, len);
            p += len;
        } else {
            out += "\xEF\xBF\xBD"; // U+FFFD, encoded as UTF-8 (valid inside a JSON string as-is)
            ++p; // resync one byte at a time rather than assuming the intended sequence length
        }
    }
}

// colors
enum common_log_col : int {
    COMMON_LOG_COL_DEFAULT = 0,
    COMMON_LOG_COL_BOLD,
    COMMON_LOG_COL_RED,
    COMMON_LOG_COL_GREEN,
    COMMON_LOG_COL_YELLOW,
    COMMON_LOG_COL_BLUE,
    COMMON_LOG_COL_MAGENTA,
    COMMON_LOG_COL_CYAN,
    COMMON_LOG_COL_WHITE,
};

// disable colors by default
static const char* g_col[] = {
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
};

static const char * level_str(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return "debug";
        case GGML_LOG_LEVEL_INFO:  return "info";
        case GGML_LOG_LEVEL_WARN:  return "warn";
        case GGML_LOG_LEVEL_ERROR: return "error";
        case GGML_LOG_LEVEL_CONT:  return "cont";
        default:                   return "none";
    }
}

struct common_log_entry {
    enum ggml_log_level level {GGML_LOG_LEVEL_INFO};

    std::vector<char> msg;

    int64_t timestamp      { 0 }; // relative to t_start, only set when --log-timestamps is on
    int64_t timestamp_epoch { 0 }; // wall-clock epoch microseconds, captured unconditionally at add()
                                   // time (the calling thread) for --log-format json -- coordinator
                                   // review point 2 (2026-09-27): computing a timestamp with t_us() at
                                   // PRINT time instead would be the WORKER thread's time, which drifts
                                   // from the actual event time under load once the queue backs up.
    bool is_end       { false }; // signals the worker thread to stop
    bool prefix       { false };
    bool jsonl        { false };
    bool is_json      { false }; // msg already holds a serialized JSON object

    common_log_entry(size_t size = 256) : msg(size) { }

    void print(FILE * file = nullptr, bool json_format = false) const {
        FILE * fcur = file;
        if (!fcur) {
            // stderr displays DBG messages only when their verbosity level is not higher than the threshold
            // these messages will still be logged to a file
            if (level == GGML_LOG_LEVEL_DEBUG && common_log_verbosity_thold < LOG_DEFAULT_DEBUG) {
                return;
            }

            fcur = stdout;

            if (level != GGML_LOG_LEVEL_NONE && !jsonl) {
                fcur = stderr;
            }
        }

        if (jsonl) {
            if (is_json) {
                fprintf(fcur, "%s\n", msg.data());
                fflush(fcur);
                return;
            }

            common_json obj = {
                {"type",  "log"},
                {"time",  timestamp},
                {"level", level_str(level)},
                {"msg",   msg.data()},
            };
            fprintf(fcur, "%s\n", obj.dump_safe().c_str());
            fflush(fcur);
            return;
        }

        if (json_format) {
            static const char * level_names[] = {
                // indexed by ggml_log_level: NONE=0, DEBUG, INFO, WARN, ERROR, CONT
                "none", "debug", "info", "warn", "error", "cont",
            };
            const char * level_name = (level >= 0 && (size_t) level < std::size(level_names))
                ? level_names[level] : "unknown";

            std::string line;
            line.reserve(msg.size() + 64);
            line += "{\"timestamp\":";
            line += std::to_string(timestamp_epoch);
            line += ",\"level\":\"";
            line += level_name;
            line += "\",\"msg\":\"";
            json_escape_append(line, msg.data());
            line += "\"}\n";

            fwrite(line.data(), 1, line.size(), fcur);
            fflush(fcur);
            return;
        }

        if (level != GGML_LOG_LEVEL_NONE && level != GGML_LOG_LEVEL_CONT && prefix) {
            if (timestamp) {
                // [M.s.ms.us]
                fprintf(fcur, "%s%d.%02d.%03d.%03d%s ",
                        g_col[COMMON_LOG_COL_BLUE],
                        (int) (timestamp / 1000000 / 60),
                        (int) (timestamp / 1000000 % 60),
                        (int) (timestamp / 1000 % 1000),
                        (int) (timestamp % 1000),
                        g_col[COMMON_LOG_COL_DEFAULT]);
            }

            switch (level) {
                case GGML_LOG_LEVEL_INFO:  fprintf(fcur, "%sI %s", g_col[COMMON_LOG_COL_GREEN],   g_col[COMMON_LOG_COL_DEFAULT]); break;
                case GGML_LOG_LEVEL_WARN:  fprintf(fcur, "%sW %s", g_col[COMMON_LOG_COL_MAGENTA], ""                        ); break;
                case GGML_LOG_LEVEL_ERROR: fprintf(fcur, "%sE %s", g_col[COMMON_LOG_COL_RED],     ""                        ); break;
                case GGML_LOG_LEVEL_DEBUG: fprintf(fcur, "%sD %s", g_col[COMMON_LOG_COL_YELLOW],  ""                        ); break;
                default:
                    break;
            }
        }

        // the reset goes before the trailing newlines, so that every line carries its own colors
        const bool reset = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_DEBUG;

        size_t end = strlen(msg.data());
        while (end > 0 && msg[end - 1] == '\n') {
            end--;
        }

        fprintf(fcur, "%.*s%s%s", (int) end, msg.data(), reset ? g_col[COMMON_LOG_COL_DEFAULT] : "", msg.data() + end);

        fflush(fcur);
    }
};

struct common_log {
    // default capacity
    common_log(size_t capacity = 512) {
        file        = nullptr;
        stdo        = true;
        colors      = false;
        prefix      = false;
        timestamps  = false;
        json_format = false;
        running     = false;
        t_start     = t_us();

        queue.resize(capacity, common_log_entry(256));
        head = 0;
        tail = 0;

        resume();
    }

    ~common_log() {
        pause();
        if (file) {
            fclose(file);
        }
    }

private:
    std::mutex              mtx;
    std::thread             thrd;
    std::condition_variable cv_new;  // new entry
    std::condition_variable cv_full; // wait on full

    FILE * file;

    bool stdo;
    bool colors;
    bool prefix;
    bool timestamps;
    // Coordinator review point 4 (2026-09-27): print_entry() (below) reads this from the worker
    // thread without holding mtx -- resume()'s worker loop explicitly unlocks before flushing the
    // queue, unlike prefix/timestamps which print() only reads with the lock held via the entry
    // snapshot taken at add() time. An unsynchronized bool read/write race is undefined behaviour
    // even if benign in practice on this target; atomic<bool> removes the question.
    std::atomic<bool> json_format;
    bool running;

    int64_t t_start;

    // queue of entries
    std::vector<common_log_entry> queue;
    size_t head;
    size_t tail;

    bool print_entry(const common_log_entry & e) const {
        if (e.is_end) return true;

        if (stdo) {
            e.print(nullptr, json_format);
        }
        if (file) {
            e.print(file, json_format);
        }
        return false;
    }

    bool flush_queue(size_t start_head, size_t end_tail, size_t & out_head) const {
        bool stop = false;
        size_t h = start_head;
        while (h != end_tail && !stop) {
            stop = print_entry(queue[h]);
            h = (h + 1) % queue.size();
        }
        out_head = h;
        return stop;
    }

public:
    bool is_full() const {
        return ((tail + 1) % queue.size()) == head;
    }

    bool is_empty() const {
        return head == tail;
    }

    void add(enum ggml_log_level level, const char * fmt, va_list args) {
        std::unique_lock<std::mutex> lock(mtx);

        // block if the queue is full
        cv_full.wait(lock, [this]() { return !running || !is_full(); });

        if (!running) {
            // discard messages while the worker thread is paused
            return;
        }

        auto & entry = queue[tail];

        {
            // cannot use args twice, so make a copy in case we need to expand the buffer
            va_list args_copy;
            va_copy(args_copy, args);

#if 1
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args_copy);
            }
#else
            // hack for bolding arguments

            std::stringstream ss;
            for (int i = 0; fmt[i] != 0; i++) {
                if (fmt[i] == '%') {
                    ss << LOG_COL_BOLD;
                    while (fmt[i] != ' ' && fmt[i] != ')' && fmt[i] != ']' && fmt[i] != 0) ss << fmt[i++];
                    ss << LOG_COL_DEFAULT;
                    if (fmt[i] == 0) break;
                }
                ss << fmt[i];
            }
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args_copy);
            }
#endif
            va_end(args_copy);
        }

        entry.is_end        = false;
        entry.level         = level;
        entry.prefix        = prefix;
        entry.timestamp     = 0;
        entry.timestamp_epoch = t_us(); // unconditional; see the field comment for why
        entry.jsonl         = common_log_jsonl;
        entry.is_json       = false;
        if (timestamps) {
            entry.timestamp = entry.timestamp_epoch - t_start;
        }

        tail = (tail + 1) % queue.size();
        cv_new.notify_one();
    }

    void add_json(const char * type, const common_json & obj) {
        const common_json full = {
            {"type", type},
            {"data", obj},
        };

        const std::string text = full.dump_safe();

        std::unique_lock<std::mutex> lock(mtx);

        // block if the queue is full
        cv_full.wait(lock, [this]() { return !running || !is_full(); });

        if (!running) {
            // discard messages while the worker thread is paused
            return;
        }

        auto & entry = queue[tail];

        if (entry.msg.size() < text.size() + 1) {
            entry.msg.resize(text.size() + 1);
        }
        memcpy(entry.msg.data(), text.c_str(), text.size() + 1);

        entry.is_end    = false;
        entry.level     = GGML_LOG_LEVEL_NONE;
        entry.prefix    = false;
        entry.jsonl     = true;
        entry.is_json   = true;
        entry.timestamp = 0;

        tail = (tail + 1) % queue.size();
        cv_new.notify_one();
    }

    void resume() {
        std::lock_guard<std::mutex> lock(mtx);

        if (running) {
            return;
        }

        running = true;

        thrd = std::thread([this]() {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx);
                cv_new.wait(lock, [this]() { return !is_empty(); });

                size_t cached_head = head;
                size_t cached_tail = tail;

                lock.unlock(); // drop the lock during flush

                size_t next_head;
                bool stop = flush_queue(cached_head, cached_tail, next_head);

                lock.lock();
                head = next_head;
                cv_full.notify_all();

                if (stop) {
                    break;
                }
            }
        });
    }

    void pause() {
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!running) {
                return;
            }

            running = false;

            // push an entry to signal the worker thread to stop
            auto & entry = queue[tail];
            entry.is_end = true;
            tail = (tail + 1) % queue.size();

            // wakeup everyone
            cv_new.notify_one();
            cv_full.notify_all();
        }

        thrd.join();
    }

    // Wait, for at most timeout_ms, until every queued entry has been printed. head only moves
    // past an entry after the worker printed it, so an empty queue means nothing is left unwritten.
    // Unlike pause(), this never joins the worker and never waits on mtx past the deadline: it runs
    // from atexit, and in a DLL build that is DLL teardown, where ExitProcess has already killed
    // the worker -- possibly while it held mtx -- and a join is the hang of upstream issue #22142.
    bool drain(int64_t timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
        while (!lock.try_lock()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return cv_full.wait_until(lock, deadline, [this]() { return !running || is_empty(); });
    }

    void set_file(const char * path) {
        pause();

        if (file) {
            fclose(file);
        }

        if (path) {
            file = fopen(path, "w");
        } else {
            file = nullptr;
        }

        resume();
    }

    void set_stdo(bool enabled) {
        pause();

        stdo = enabled;

        resume();
    }

    bool get_colors() const {
        return colors;
    }

    void set_colors(bool colors) {
        pause();

        this->colors = colors && tty_enable_ansi();

        if (this->colors) {
            g_col[COMMON_LOG_COL_DEFAULT] = LOG_COL_DEFAULT;
            g_col[COMMON_LOG_COL_BOLD]    = LOG_COL_BOLD;
            g_col[COMMON_LOG_COL_RED]     = LOG_COL_RED;
            g_col[COMMON_LOG_COL_GREEN]   = LOG_COL_GREEN;
            g_col[COMMON_LOG_COL_YELLOW]  = LOG_COL_YELLOW;
            g_col[COMMON_LOG_COL_BLUE]    = LOG_COL_BLUE;
            g_col[COMMON_LOG_COL_MAGENTA] = LOG_COL_MAGENTA;
            g_col[COMMON_LOG_COL_CYAN]    = LOG_COL_CYAN;
            g_col[COMMON_LOG_COL_WHITE]   = LOG_COL_WHITE;
        } else {
            for (size_t i = 0; i < std::size(g_col); i++) {
                g_col[i] = "";
            }
        }

        resume();
    }

    void set_prefix(bool prefix) {
        std::lock_guard<std::mutex> lock(mtx);

        this->prefix = prefix;
    }

    void set_timestamps(bool timestamps) {
        std::lock_guard<std::mutex> lock(mtx);

        this->timestamps = timestamps;
    }

    void set_json_format(bool json_format) {
        std::lock_guard<std::mutex> lock(mtx);

        this->json_format = json_format;
    }
};

//
// public API
//

struct common_log * common_log_init() {
    return new common_log;
}

// The logger the at-exit drain may touch. common_log_main() publishes the singleton here and
// common_log_free() withdraws it BEFORE deleting, so the atexit handler never drains a freed object.
// Row S69-FORK-TEST-LOG-AT-EXIT-DRAIN-ACCESS-VIOLATION-FLAKY-2026-11-09-01: tests/test-log.cpp (and any
// embedder that calls common_log_free(common_log_main())) frees the singleton, and the drain registered
// in fd9ccd44c then ran on that freed memory when main returned -- an access violation in 1 of ~20 runs.
static std::atomic<struct common_log *> g_log_main { nullptr };

static void common_log_drain_main_at_exit() {
    struct common_log * log = g_log_main.load();
    if (log) {
        common_log_drain(log, 1000);
    }
}

struct common_log * common_log_main() {
    // We intentionally leak (i.e. do not delete) the logger singleton because
    // common_log destructor called at DLL teardown phase will cause hanging on Windows.
    // OS will release resources anyway so it should not be a significant issue,
    // though this design may cause logs to be lost if not flushed before the program exits.
    // Refer to https://github.com/ggml-org/llama.cpp/issues/22142 for details.
    static struct common_log * log;
    static std::once_flag    init_flag;
    std::call_once(init_flag, [&]() {
        log = new common_log;
        // Set default to auto-detect colors
        log->set_colors(tty_can_use_colors());
        // Because the singleton is leaked, an exit right after a LOG_ERR lost the tail of the log:
        // b1019h 2026-10-27, llama-server refused a bad hot-swap config with rc 1 and its log ended
        // before the refusal line. A bounded drain at exit prints what is queued (see drain()).
        g_log_main.store(log);
        std::atexit(common_log_drain_main_at_exit);
    });

    return log;
}

bool common_log_drain(struct common_log * log, int64_t timeout_ms) {
    return log->drain(timeout_ms);
}

void common_log_pause(struct common_log * log) {
    log->pause();
}

void common_log_resume(struct common_log * log) {
    log->resume();
}

void common_log_free(struct common_log * log) {
    // if this is the main singleton, stop the at-exit drain from reaching it before the memory goes
    struct common_log * expected = log;
    g_log_main.compare_exchange_strong(expected, nullptr);
    delete log;
}

void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log->add(level, fmt, args);
    va_end(args);
}

void common_log_add_json(struct common_log * log, const char * type, const common_json & obj) {
    if (!common_log_jsonl) {
        return;
    }

    log->add_json(type, obj);
}

void common_log_set_file(struct common_log * log, const char * file) {
    log->set_file(file);
}

void common_log_set_stdo(struct common_log * log, bool enabled) {
    log->set_stdo(enabled);
}

void common_log_set_colors(struct common_log * log, log_colors colors) {
    if (colors == LOG_COLORS_AUTO) {
        log->set_colors(tty_can_use_colors());
        return;
    }

    if (colors == LOG_COLORS_DISABLED) {
        log->set_colors(false);
        return;
    }

    GGML_ASSERT(colors == LOG_COLORS_ENABLED);
    log->set_colors(true);
}

bool common_log_get_colors(struct common_log * log) {
    return log->get_colors();
}

void common_log_set_prefix(struct common_log * log, bool prefix) {
    log->set_prefix(prefix);
}

void common_log_set_timestamps(struct common_log * log, bool timestamps) {
    log->set_timestamps(timestamps);
}

void common_log_set_json_format(struct common_log * log, bool json_format) {
    log->set_json_format(json_format);
}

void common_log_flush(struct common_log * log) {
    log->pause();
    log->resume();
}

int common_log_get_verbosity(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;
        case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;
        case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;
        case GGML_LOG_LEVEL_CONT:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_NONE:
        default:
            return LOG_LEVEL_OUTPUT;
    }
}

void common_log_default_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    auto verbosity = common_log_get_verbosity(level);
    if (verbosity <= common_log_verbosity_thold) {
        common_log_add(common_log_main(), level, "%s", text);
    }
}
