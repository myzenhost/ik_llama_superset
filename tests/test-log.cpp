#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

// row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 4,
// coordinator review point 3 (2026-09-27): read a whole file into a std::string.
static std::string read_whole_file(const char * path) {
    std::string out;
    FILE * f = fopen(path, "rb");
    if (!f) {
        return out;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return out;
}

// Coordinator review point 3: json_escape_append (common/log.cpp) must replace an invalid/truncated
// UTF-8 byte sequence with U+FFFD rather than passing raw bytes through into a JSON string. It is
// `static` (file-local), so it cannot be reached from a Python test; this exercises it end to end
// through a dedicated common_log instance redirected to a file, the same way --log-format json does.
static bool test_log_format_json_invalid_utf8() {
    struct common_log * log = common_log_init();

    const char * path = "test-log-invalid-utf8.tmp.json";
    common_log_set_file      (log, path);
    common_log_set_json_format(log, true);
    common_log_set_timestamps(log, false);
    common_log_set_prefix    (log, false);

    // 0xC3 0x28 is not valid UTF-8: 0xC3 is a 2-byte lead byte, but 0x28 ('(') is not a continuation
    // byte (top two bits must be 10). A raw passthrough would emit that pair verbatim inside the
    // JSON string, producing invalid UTF-8 that a strict JSON reader may refuse to parse.
    const char msg[] = { 'b', 'a', 'd', ':', ' ', (char) 0xC3, (char) 0x28, ' ', 'e', 'n', 'd', '\n', '\0' };
    common_log_add(log, GGML_LOG_LEVEL_INFO, "%s", msg);
    common_log_flush(log);
    common_log_free(log);

    const std::string content = read_whole_file(path);
    remove(path);

    if (content.empty()) {
        fprintf(stderr, "test_log_format_json_invalid_utf8: FAIL (empty output file)\n");
        return false;
    }

    // the invalid two-byte sequence must not survive verbatim
    const char invalid_seq[] = { (char) 0xC3, (char) 0x28 };
    if (content.find(std::string(invalid_seq, 2)) != std::string::npos) {
        fprintf(stderr, "test_log_format_json_invalid_utf8: FAIL (raw invalid UTF-8 bytes 0xC3 0x28 found in output)\n");
        return false;
    }

    // and it must have been replaced by U+FFFD (EF BF BD), not silently dropped
    if (content.find("\xEF\xBF\xBD") == std::string::npos) {
        fprintf(stderr, "test_log_format_json_invalid_utf8: FAIL (no U+FFFD replacement found in output)\n");
        return false;
    }

    // minimal structural check that the line is still one JSON object (no embedded raw '\0', starts
    // with '{', the "msg" key is present) -- a full JSON parser is not linked into this test binary.
    if (content[0] != '{' || content.find("\"msg\"") == std::string::npos) {
        fprintf(stderr, "test_log_format_json_invalid_utf8: FAIL (output does not look like a JSON object): %s\n", content.c_str());
        return false;
    }

    fprintf(stderr, "test_log_format_json_invalid_utf8: PASS\n");
    return true;
}

int main() {
    const int n_thread = 8;

    std::thread threads[n_thread];
    for (int i = 0; i < n_thread; i++) {
        threads[i] = std::thread([i]() {
            const int n_msg = 1000;

            for (int j = 0; j < n_msg; j++) {
                const int log_type = std::rand() % 4;

                switch (log_type) {
                    case 0: LOG_INF("Thread %d: %d\n", i, j); break;
                    case 1: LOG_WRN("Thread %d: %d\n", i, j); break;
                    case 2: LOG_ERR("Thread %d: %d\n", i, j); break;
                    case 3: LOG_DBG("Thread %d: %d\n", i, j); break;
                    default:
                        break;
                }

                if (rand () % 10 < 5) {
                    common_log_set_timestamps(common_log_main(), rand() % 2);
                    common_log_set_prefix    (common_log_main(), rand() % 2);
                    // coordinator review point 4 (2026-09-27): common_log::json_format is now
                    // std::atomic<bool> because the worker thread reads it in print_entry() without
                    // holding the same mutex set_json_format() takes. Toggle it concurrently from all
                    // 8 fuzz threads here, same as the pre-existing timestamps/prefix toggles, so a
                    // thread-safety regression (e.g. reverting the field to plain bool) shows up as a
                    // TSan/crash failure in this existing stress test instead of going unnoticed.
                    common_log_set_json_format(common_log_main(), rand() % 2);
                }
            }
        });
    }

    for (int i = 0; i < n_thread; i++) {
        threads[i].join();
    }

    common_log_flush(common_log_main());
    // We explicitly free the logger singleton to avoid hanging on Windows
    // related to timing issues of thread startup and DLL teardown
    common_log_free(common_log_main());

    // leave common_log_main() in its default (non-json) state for anything running after this in
    // the same process; not currently relied on (this is the last thing main() does) but cheap
    // insurance if another check is ever appended below.
    // (common_log_main() is intentionally leaked, so no free() call here.)

    if (!test_log_format_json_invalid_utf8()) {
        return 1;
    }

    return 0;
}
