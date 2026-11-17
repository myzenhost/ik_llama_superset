#include "arg.h"

#include "build-info.h"
#include "common.h"
#include "log.h"
#include "download.h"
#include "download-chunk.h"
#include "file-lock.h"
#include "gguf-header-probe.h"
#include "hf-cache.h"
#include "json.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <unordered_set>
#include <string>
#include <thread>
#include <vector>

#include "http.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

//
// downloader
//

// the suffix of the file a download writes until it completes
static const std::string PARTIAL_SUFFIX = ".downloadInProgress";

static void write_etag(const std::string & path, const std::string & etag) {
    const std::string etag_path = path + ".etag";
    fs_write_atomic(std::filesystem::u8path(etag_path), etag);
    LOG_DBG("%s: file etag saved: %s\n", __func__, etag_path.c_str());
}

static std::string read_etag(const std::string & path) {
    const std::string etag_path = path + ".etag";
    if (!std::filesystem::exists(etag_path)) {
        return {};
    }
    std::ifstream etag_in(etag_path);
    if (!etag_in) {
        LOG_ERR("%s: could not open .etag file for reading: %s\n", __func__, etag_path.c_str());
        return {};
    }
    std::string etag;
    std::getline(etag_in, etag);
    return etag;
}

static bool is_http_status_ok(int status) {
    return status >= 200 && status < 400;
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    auto parts = string_split<std::string>(hf_repo_with_tag, ':');
    std::string tag = parts.size() > 1 ? parts.back() : "";
    std::string hf_repo = parts[0];
    if (string_split<std::string>(hf_repo, '/').size() != 2) {
        throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
    }
    return {hf_repo, tag};
}

// file name from url, without the query string
static std::string url_get_filename(const std::string & url) {
    std::string filename = url;
    if (auto pos = filename.rfind('/'); pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }
    if (auto pos = filename.find('?'); pos != std::string::npos) {
        filename = filename.substr(0, pos);
    }
    return filename;
}

class ProgressBar : public common_download_callback {
    static inline std::mutex mutex;
    static inline std::map<const ProgressBar *, int> lines;
    static inline int max_line = 0;

    std::string filename;
    size_t len = 0;

    static void cleanup(const ProgressBar * line) {
        lines.erase(line);
        if (lines.empty()) {
            max_line = 0;
        }
    }

    static bool is_output_a_tty() {
        return common_is_tty(stdout);
    }

public:
    ProgressBar() = default;

    void on_start(const common_download_progress & p) override {
        filename = url_get_filename(p.url);

        for (size_t i = 0; i < filename.size(); ++i) {
            if ((filename[i] & 0xC0) != 0x80) {
                if (len++ == 39) {
                    filename.resize(i);
                    filename += "…";
                    break;
                }
            }
        }
    }

    void on_done(const common_download_progress &, bool) override {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void on_update(const common_download_progress & p) override {
        if (!p.total || !is_output_a_tty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t bar = (55 - len) * 2;
        size_t pct = (100 * p.downloaded) / p.total;
        size_t pos = (bar * p.downloaded) / p.total;

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << '\r' << "Downloading " << filename << " ";

        for (size_t i = 0; i < bar; i += 2) {
            std::cout << (i + 1 < pos ? "─" : (i < pos ? "╴" : " "));
        }
        std::cout << std::setw(4) << pct << "%\033[K";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "B";
        }
        std::cout << '\r' << std::flush;

        if (p.downloaded == p.total) {
            cleanup(this);
        }
    }

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar & operator=(const ProgressBar &) = delete;
};

bool common_is_sha256_hex(const std::string & s) {
    if (s.size() != 64) {
        return false;
    }
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

bool common_file_sha256_hex(const std::string & path, std::string & out_hex) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }
    sha256_t ctx;
    sha256_init(&ctx);
    std::vector<unsigned char> buf(1u << 20);
    while (ifs) {
        ifs.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) buf.size());
        const std::streamsize n = ifs.gcount();
        if (n > 0) {
            sha256_update(&ctx, buf.data(), (size_t) n);
        }
    }
    if (!ifs.eof()) {
        return false; // a read error, not the end of the file
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    static const char * hex = "0123456789abcdef";
    out_hex.clear();
    out_hex.reserve(2 * SHA256_DIGEST_SIZE);
    for (unsigned char b : digest) {
        out_hex.push_back(hex[b >> 4]);
        out_hex.push_back(hex[b & 15]);
    }
    return true;
}

// "bytes START-END/TOTAL" (or "bytes START-END/*") -> START; false for anything else, e.g. "bytes */TOTAL"
static bool common_content_range_start(const std::string & value, size_t & start) {
    static const std::string unit = "bytes ";
    if (value.compare(0, unit.size(), unit) != 0) {
        return false;
    }
    size_t i = unit.size();
    if (i >= value.size() || !std::isdigit((unsigned char) value[i])) {
        return false;
    }
    try {
        size_t used = 0;
        start = (size_t) std::stoull(value.substr(i), &used);
        i += used;
    } catch (const std::exception &) {
        return false;
    }
    return i < value.size() && value[i] == '-';
}

static bool common_pull_file(httplib::Client & cli,
                             const std::string & resolve_path,
                             const std::string & path_tmp,
                             bool supports_ranges,
                             common_download_progress & p,
                             common_download_callback * callback) {
    std::ofstream ofs(path_tmp, std::ios::binary | std::ios::app);
    if (!ofs.is_open()) {
        LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
        return false;
    }

    httplib::Headers headers;
    if (supports_ranges && p.downloaded > 0) {
        headers.emplace("Range", "bytes=" + std::to_string(p.downloaded) + "-");
    }

    const char * func = __func__; // avoid __func__ inside a lambda
    size_t progress_step = 0;

    auto res = cli.Get(resolve_path, headers,
        [&](const httplib::Response &response) {
            if (p.downloaded > 0 && response.status != 206) {
                LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", func, response.status);
                if (response.status == 416 || response.status == 200) {
                    // the kept partial cannot be resumed: 416 for a complete or stale one,
                    // 200 from a server that ignores the range; drop it, so the next
                    // attempt starts over instead of wedging on the same answer
                    ofs.close();
                    remove(path_tmp.c_str());
                }
                return false;
            }
            if (p.downloaded > 0) {
                // a 206 only extends the partial when it starts exactly where the partial ends; a partial from
                // another file or a server that slices elsewhere would otherwise be glued into a corrupt file.
                // Behaviour from containers/ramalama http_client.py verify_resume_point (MIT), own code.
                size_t cr_start = 0;
                if (!common_content_range_start(response.get_header_value("Content-Range"), cr_start) ||
                    cr_start != p.downloaded) {
                    LOG_WRN("%s: 206 Content-Range '%s' does not start at the partial size %zu, discarding the partial\n",
                            func, response.get_header_value("Content-Range").c_str(), p.downloaded);
                    ofs.close();
                    remove(path_tmp.c_str());
                    return false;
                }
            }
            if (p.downloaded == 0 && response.status != 200) {
                LOG_WRN("%s: download received non-successful status code: %d\n", func, response.status);
                return false;
            }
            if (p.total == 0 && response.has_header("Content-Length")) {
                try {
                    size_t content_length = std::stoull(response.get_header_value("Content-Length"));
                    p.total = p.downloaded + content_length;
                } catch (const std::exception &e) {
                    LOG_WRN("%s: invalid Content-Length header: %s\n", func, e.what());
                }
            }
            return true;
        },
        [&](const char *data, size_t len) {
            ofs.write(data, len);
            if (!ofs) {
                LOG_ERR("%s: error writing to file: %s\n", func, path_tmp.c_str());
                return false;
            }
            p.downloaded += len;
            progress_step += len;

            if (progress_step >= p.total / 1000 || p.downloaded == p.total) {
                if (callback) {
                    callback->on_update(p);
                    if (callback->is_cancelled() && (p.total == 0 || p.downloaded != p.total)) {
                        return false;
                    }
                }
                progress_step = 0;
            }
            return true;
        },
        nullptr
    );

    if (!res) {
        LOG_ERR("%s: download failed: %s (status: %d)\n",
                __func__,
                httplib::to_string(res.error()).c_str(),
                res ? res->status : -1);
        return false;
    }

    ofs.close();
    if (!ofs) {
        LOG_ERR("%s: error closing file: %s\n", __func__, path_tmp.c_str());
        return false;
    }

    return true;
}

// Parallel range download: the file is cut into fixed-size ranges, `concurrency` connections fetch them at once into a
// preallocated partial file, and the bitmap sidecar (<partial>.chunks) records every finished range, so a failed or
// interrupted attempt keeps all finished ranges and the next attempt fetches only the rest.
// Behaviour from NEXA SDK's chunked downloader (BSD-3-Clause), reimplemented here, no code copied.
// Row S69-ENGINE-MERGE-NEXA-2026-09-27-01.
static bool common_pull_chunks(const std::string & url,
                               const httplib::Headers & headers,
                               const std::string & path_tmp,
                               size_t chunk_size,
                               int concurrency,
                               common_download_progress & p,
                               common_download_callback * callback) {
    const size_t total = p.total;
    const size_t cs    = common_chunk_effective_size(total, chunk_size);
    const auto   plan  = common_chunk_plan(total, chunk_size);
    const std::string sidecar = path_tmp + COMMON_CHUNK_SIDECAR_SUFFIX;

    std::vector<char> done;
    const bool have_bitmap = common_chunk_bitmap_load(sidecar, total, cs, plan.size(), done);

    std::error_code ec;
    const bool partial_ok = have_bitmap && std::filesystem::exists(path_tmp, ec) &&
                            std::filesystem::file_size(path_tmp, ec) == total;
    if (!partial_ok) {
        // no usable record of what the partial holds: start over on a fresh preallocated file
        done.assign(plan.size(), 0);
        {
            std::ofstream create(path_tmp, std::ios::binary | std::ios::trunc);
            if (!create) {
                LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
                return false;
            }
        }
        std::filesystem::resize_file(path_tmp, total, ec);
        if (ec) {
            LOG_ERR("%s: cannot preallocate %zu bytes for %s: %s\n", __func__, total, path_tmp.c_str(), ec.message().c_str());
            return false;
        }
        common_chunk_bitmap_save(sidecar, total, cs, done);
    }

    std::mutex          mtx;
    std::atomic<size_t> next{0};
    std::atomic<bool>   failed{false};
    std::atomic<bool>   cancel{false};
    p.downloaded = common_chunk_done_bytes(plan, done);
    if (callback) {
        callback->on_update(p);
    }

    auto worker = [&]() {
        auto [cli, parts] = common_http_client(url);
        cli.set_default_headers(headers);
        std::fstream out(path_tmp, std::ios::in | std::ios::out | std::ios::binary);
        if (!out) {
            LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
            failed = true;
            return;
        }
        while (!failed && !cancel) {
            const size_t idx = next.fetch_add(1);
            if (idx >= plan.size()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (done[idx]) {
                    continue;
                }
            }
            const common_chunk & c = plan[idx];
            httplib::Headers h;
            h.emplace("Range", "bytes=" + std::to_string(c.offset) + "-" + std::to_string(c.offset + c.len - 1));

            size_t got = 0;
            bool   ok  = true;
            out.clear();
            out.seekp((std::streamoff) c.offset);
            auto res = cli.Get(parts.path, h,
                [&](const httplib::Response & response) {
                    size_t cr_start = 0;
                    if (response.status != 206 ||
                        !common_content_range_start(response.get_header_value("Content-Range"), cr_start) ||
                        cr_start != c.offset) {
                        LOG_WRN("%s: range %zu answered %d with Content-Range '%s'; expected 206 starting at %zu\n",
                                __func__, idx, response.status, response.get_header_value("Content-Range").c_str(), c.offset);
                        ok = false;
                        return false;
                    }
                    return true;
                },
                [&](const char * data, size_t len) {
                    if (got + len > c.len) {
                        ok = false; // more bytes than the range holds: would overwrite the next range
                        return false;
                    }
                    out.write(data, (std::streamsize) len);
                    if (!out) {
                        ok = false;
                        return false;
                    }
                    got += len;
                    return !failed && !cancel;
                });
            out.flush();
            if (!res || !ok || got != c.len || !out) {
                LOG_WRN("%s: range %zu (%zu bytes at %zu) failed: %s\n", __func__, idx, c.len, c.offset,
                        res ? "bad answer" : httplib::to_string(res.error()).c_str());
                failed = true;
                return;
            }
            std::lock_guard<std::mutex> lk(mtx);
            done[idx] = 1;
            p.downloaded += c.len;
            common_chunk_bitmap_save(sidecar, total, cs, done);
            if (callback) {
                callback->on_update(p);
                if (callback->is_cancelled()) {
                    cancel = true;
                }
            }
        }
    };

    const int n_threads = (int) std::min<size_t>((size_t) std::max(1, concurrency), plan.size());
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto & t : threads) {
        t.join();
    }

    const bool all = common_chunk_done_bytes(plan, done) == total && !failed && !cancel;
    if (all) {
        std::filesystem::remove(sidecar, ec);
        p.downloaded = total;
    }
    return all;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int common_download_file_single_online(const std::string & url,
                                              const std::string & path,
                                              const common_download_opts & opts,
                                              bool skip_etag,
                                              const std::string & lock_path,
                                              const std::string & alt_path) {
    static const int max_attempts        = 3;
    static const int retry_delay_seconds = 2;

    const std::string path_temporary = path + PARTIAL_SUFFIX;
    const std::string path_lock = lock_path.empty() ? path + ".lock" : lock_path;

    common_download_progress p;
    p.url = url;

    if (opts.callback) {
        opts.callback->on_start(p);
    }

    // the completion callback may itself take the same lock (the HF cache
    // finalizes blobs under it), so every exit goes through finish()
    common_file_lock lock(path_lock);
    auto finish = [&](int status) {
        lock.close();
        if (opts.callback) {
            opts.callback->on_done(p, is_http_status_ok(status));
        }
        return status;
    };

    // alt_path is the HF snapshot: the blob may have been moved there when symlinks are unavailable
    auto cache_hit = [&] {
        return std::filesystem::exists(std::filesystem::u8path(path)) ||
               (!alt_path.empty() && std::filesystem::exists(std::filesystem::u8path(alt_path)));
    };

    if (skip_etag && cache_hit()) {
        LOG_DBG("%s: using cached file: %s\n", __func__, path.c_str());
        return finish(304); // 304 Not Modified - fake cached response
    }

    // take the lock before touching the cache, so two processes cannot race on the same file
    bool waited = false;
    if (!lock.acquire([&] {
            if (!waited) {
                LOG_INF("%s: file '%s' is being downloaded by another process, waiting...\n", __func__, url_get_filename(url).c_str());
                waited = true;
            }
            return !(opts.callback && opts.callback->is_cancelled());
        })) {
        return finish(-1);
    }

    // under the lock: another process may have finished while we waited
    bool file_exists = cache_hit();

    if (file_exists && skip_etag) {
        LOG_DBG("%s: using cached file: %s\n", __func__, path.c_str());
        return finish(304); // 304 Not Modified - fake cached response
    }

    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : opts.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }
    if (!opts.bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + opts.bearer_token);
    }
    cli.set_default_headers(headers);

    std::string last_etag;
    if (file_exists) {
        last_etag = read_etag(path);
    } else {
        LOG_DBG("%s: no previous model file found %s\n", __func__, path.c_str());
    }

    auto head = cli.Head(parts.path);
    if (!head || head->status < 200 || head->status >= 300) {
        LOG_TRC("%s: HEAD failed, status: %d\n", __func__, head ? head->status : -1);
        if (file_exists) {
            LOG_TRC("%s: using cached file (HEAD failed): %s\n", __func__, path.c_str());
            return finish(304); // 304 Not Modified - fake cached response
        }
        return finish(head ? head->status : -1);
    }

    std::string etag;
    if (head->has_header("ETag")) {
        etag = head->get_header_value("ETag");
    }

    if (head->has_header("Content-Length")) {
        try {
            p.total = std::stoull(head->get_header_value("Content-Length"));
        } catch (const std::exception& e) {
            LOG_WRN("%s: invalid Content-Length in HEAD response: %s\n", __func__, e.what());
        }
    }

    bool supports_ranges = false;
    if (head->has_header("Accept-Ranges")) {
        supports_ranges = head->get_header_value("Accept-Ranges") != "none";
    }

    if (file_exists) {
        if (etag.empty()) {
            LOG_DBG("%s: using cached file (no server etag): %s\n", __func__, path.c_str());
            return finish(304); // 304 Not Modified - fake cached response
        }
        if (!last_etag.empty() && last_etag == etag) {
            LOG_DBG("%s: using cached file (same etag): %s\n", __func__, path.c_str());
            return finish(304); // 304 Not Modified - fake cached response
        }
        // pass this point, the file exists but is different from the server version, so we need to redownload it
        if (remove(path.c_str()) != 0) {
            LOG_ERR("%s: unable to delete file: %s\n", __func__, path.c_str());
            return finish(-1);
        }
    }

    { // silent
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    }

    bool success = false;
    int delay = retry_delay_seconds;
    int n_hash_mismatch = 0;

    // parallel range path: needs a known length and a range-capable server, and a file worth cutting into 2+ ranges
    const std::string path_chunks = path_temporary + COMMON_CHUNK_SIDECAR_SUFFIX;
    const bool use_chunks = supports_ranges && p.total > 0 && opts.chunk_concurrency > 1 &&
                            common_chunk_plan(p.total, opts.chunk_size).size() >= 2;
    if (!use_chunks) {
        // a range bitmap describes a preallocated partial the single stream cannot resume from
        std::error_code ec_chunks;
        if (std::filesystem::exists(path_chunks, ec_chunks)) {
            std::filesystem::remove(path_chunks, ec_chunks);
            std::filesystem::remove(path_temporary, ec_chunks);
        }
    }

    if (!skip_etag && !etag.empty()) {
        if (std::filesystem::exists(path_temporary) && read_etag(path_temporary) != etag) {
            LOG_WRN("%s: partial file does not match the server etag, discarding it: %s\n", __func__, path_temporary.c_str());
            if (remove(path_temporary.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                return finish(-1);
            }
            remove(path_chunks.c_str());
        }
        write_etag(path_temporary, etag);
    }

    if (opts.callback) {
        // report the resume offset, so a resumed file does not start at 0
        // (a range download reports its own progress: its preallocated partial is always full size)
        p.downloaded = supports_ranges && !use_chunks && std::filesystem::exists(path_temporary) ?
                       std::filesystem::file_size(path_temporary) : 0;
        opts.callback->on_update(p);
    }

    for (int i = 0; i < max_attempts; ++i) {
        if (opts.callback && opts.callback->is_cancelled()) {
            break;
        }
        if (i) {
            LOG_WRN("%s: retrying after %d seconds...\n", __func__, delay);
            std::this_thread::sleep_for(std::chrono::seconds(delay));
            delay *= retry_delay_seconds;
        }

        size_t existing_size = 0;

        if (std::filesystem::exists(path_temporary) && !use_chunks) {
            if (supports_ranges) {
                existing_size = std::filesystem::file_size(path_temporary);
            } else if (remove(path_temporary.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                break;
            }
        }

        p.downloaded = existing_size;

        LOG_DBG("%s: downloading from %s to %s (etag:%s)...\n",
                __func__, common_http_show_masked_url(parts).c_str(),
                path_temporary.c_str(), etag.c_str());

        const bool pulled = use_chunks ?
            common_pull_chunks(url, headers, path_temporary, opts.chunk_size, opts.chunk_concurrency, p, opts.callback) :
            common_pull_file(cli, parts.path, path_temporary, supports_ranges, p, opts.callback);
        if (pulled) {
            // Content integrity before the file becomes visible (containers/ramalama model_store/store.py:236-242 and
            // PR #2917, MIT; behaviour reimplemented, no code copied).
            //  1. a body shorter than the announced length is a truncated transfer: keep the partial and let the next
            //     attempt range-resume it. A longer one is left alone: a server that compresses on the fly reports the
            //     compressed length.
            //  2. when the registry announced a SHA-256, the finished file must have it. One mismatch deletes the file
            //     and downloads again, a second one fails the download with nothing at the final path.
            if (p.total > 0 && p.downloaded < p.total) {
                LOG_WRN("%s: short download, %zu of %zu bytes; keeping the partial file for a range resume\n",
                        __func__, p.downloaded, p.total);
                continue;
            }
            if (!opts.expected_sha256.empty()) {
                std::string got;
                if (!common_file_sha256_hex(path_temporary, got)) {
                    LOG_ERR("%s: cannot hash %s\n", __func__, path_temporary.c_str());
                    break;
                }
                std::string want = opts.expected_sha256;
                std::transform(want.begin(), want.end(), want.begin(), [](unsigned char c) { return (char) std::tolower(c); });
                if (got != want) {
                    LOG_ERR("%s: SHA-256 mismatch for %s: expected %s, got %s\n", __func__, url.c_str(), want.c_str(), got.c_str());
                    if (remove(path_temporary.c_str()) != 0) {
                        LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                        break;
                    }
                    if (++n_hash_mismatch >= 2) {
                        LOG_ERR("%s: second SHA-256 mismatch, giving up; no file is left at %s\n", __func__, path.c_str());
                        break;
                    }
                    continue;
                }
            }
            if (std::rename(path_temporary.c_str(), path.c_str()) != 0) {
                LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
                break;
            }
            if (!etag.empty() && !skip_etag) {
                write_etag(path, etag);
                remove((path_temporary + ".etag").c_str());
            }
            success = true;
            break;
        }
    }

    // a stopped download keeps path_temporary, so the next attempt resumes from it
    if (!success) {
        LOG_ERR("%s: download failed after %d attempts\n", __func__, max_attempts);
    }

    return finish(success ? head->status : -1);
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    httplib::Headers headers;
    for (const auto & h : params.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }

    // block_private: the host is checked before every request and redirects are followed by hand so each
    // hop is checked too (a public URL must not bounce into 127.0.0.1 or the cloud metadata address)
    const int max_hops = params.block_private ? 5 : 0;
    std::string cur_url = url;

    for (int hop = 0; ; hop++) {
        auto [cli, parts] = common_http_client(cur_url);

        if (params.block_private) {
            if (common_http_host_is_private(parts.host)) {
                throw std::runtime_error("error: refusing to fetch a private, loopback or link-local host");
            }
            cli.set_follow_location(false);
        }

        if (params.timeout > 0) {
            cli.set_read_timeout(params.timeout, 0);
            cli.set_write_timeout(params.timeout, 0);
        }

        std::vector<char> buf;
        auto res = cli.Get(parts.path, headers,
            [&](const char *data, size_t len) {
                buf.insert(buf.end(), data, data + len);
                return params.max_size == 0 ||
                       buf.size() <= static_cast<size_t>(params.max_size);
            },
            nullptr
        );

        if (!res) {
            throw std::runtime_error("error: cannot make GET request");
        }

        const int st = res->status;
        if (params.block_private && (st == 301 || st == 302 || st == 303 || st == 307 || st == 308) && res->has_header("Location")) {
            if (hop >= max_hops) {
                throw std::runtime_error("error: too many redirects");
            }
            const std::string loc = res->get_header_value("Location");
            if (loc.rfind("http://", 0) == 0 || loc.rfind("https://", 0) == 0) {
                cur_url = loc;
            } else if (!loc.empty() && loc[0] == '/') {
                cur_url = parts.scheme + "://" + common_http_format_host(parts.host) + ":" + std::to_string(parts.port) + loc;
            } else {
                throw std::runtime_error("error: unsupported redirect target");
            }
            continue;
        }

        return { st, std::move(buf) };
    }
}

int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts,
                                bool skip_etag,
                                const std::string & lock_path,
                                const std::string & alt_path) {
    if (!opts.offline) {
        ProgressBar tty_cb;
        common_download_opts online_opts = opts;
        if (!online_opts.callback) {
            online_opts.callback = &tty_cb;
        }
        return common_download_file_single_online(url, path, online_opts, skip_etag, lock_path, alt_path);
    }

    if (!std::filesystem::exists(std::filesystem::u8path(path))) {
        LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, path.c_str());
        return -1;
    }

    LOG_DBG("%s: using cached file (offline mode): %s\n", __func__, path.c_str());

    // notify the callback that the file was cached
    if (opts.callback) {
        common_download_progress p;
        p.url = url;
        p.cached = true;
        opts.callback->on_start(p);
        opts.callback->on_done(p, true);
    }

    return 304; // Not Modified - fake cached response
}

struct gguf_split_info {
    std::string prefix; // tag included
    std::string tag;
    int index;
    int count;
};

static gguf_split_info get_gguf_split_info(const std::string & path) {
    static const std::regex re_split("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
    static const std::regex re_tag("[-.]([A-Z0-9_]+)$", std::regex::icase);
    std::smatch m;

    std::string prefix = path;
    if (!string_remove_suffix(prefix, ".gguf")) {
        return {};
    }

    int index = 1;
    int count = 1;

    if (std::regex_match(prefix, m, re_split)) {
        index = std::stoi(m[2].str());
        count = std::stoi(m[3].str());
        prefix = m[1].str();
    }

    std::string tag;
    if (std::regex_search(prefix, m, re_tag)) {
        tag = m[1].str();
        for (char & c : tag) {
            c = std::toupper((unsigned char)c);
        }
    }

    return {std::move(prefix), std::move(tag), index, count};
}

// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc
static int extract_quant_bits(const std::string & filename) {
    auto split = get_gguf_split_info(filename);

    auto pos = split.tag.find_first_of("0123456789");
    if (pos == std::string::npos) {
        return 0;
    }

    return std::stoi(split.tag.substr(pos));
}

static hf_cache::hf_files get_split_files(const hf_cache::hf_files & files,
                                          const hf_cache::hf_file  & file) {
    auto split = get_gguf_split_info(file.path);

    if (split.count <= 1) {
        return {file};
    }
    hf_cache::hf_files result;

    for (const auto & f : files) {
        auto split_f = get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            result.push_back(f);
        }
    }
    return result;
}

// pick the best sibling GGUF whose filename contains `keyword` (e.g. "mmproj" / "mtp"),
// preferring deeper shared directory prefix with the model, then exact `tag` match,
// then closest quantization to the tag when given, or to the model otherwise
static hf_cache::hf_file find_best_sibling(const hf_cache::hf_files & files,
                                           const std::string        & model,
                                           const std::string        & keyword,
                                           const std::string        & tag = "") {
    hf_cache::hf_file best;
    size_t best_depth = 0;
    int best_diff = 0;
    bool best_exact = false;
    bool found = false;

    std::string tag_upper = tag;
    for (char & c : tag_upper) {
        c = (char) std::toupper((unsigned char) c);
    }

    int model_bits = 0;
    if (!tag_upper.empty()) {
        auto pos = tag_upper.find_first_of("0123456789");
        model_bits = pos == std::string::npos ? 0 : std::stoi(tag_upper.substr(pos));
    } else {
        model_bits = extract_quant_bits(model);
    }
    auto model_parts = string_split<std::string>(model, '/');
    auto model_dir = model_parts.end() - 1;

    for (const auto & f : files) {
        if (!string_ends_with(f.path, ".gguf") ||
            f.path.find(keyword) == std::string::npos) {
            continue;
        }

        auto sib_parts = string_split<std::string>(f.path, '/');
        auto sib_dir = sib_parts.end() - 1;

        auto [_, dir] = std::mismatch(model_parts.begin(), model_dir,
                                      sib_parts.begin(), sib_dir);
        if (dir != sib_dir) {
            continue;
        }

        size_t depth = dir - sib_parts.begin();
        auto bits = extract_quant_bits(f.path);
        auto diff = std::abs(bits - model_bits);

        std::string path_upper = f.path;
        for (char & c : path_upper) {
            c = (char) std::toupper((unsigned char) c);
        }
        bool exact = !tag_upper.empty() && path_upper.find("-" + tag_upper + ".") != std::string::npos;

        if (!found || depth > best_depth ||
            (depth == best_depth && exact && !best_exact) ||
            (depth == best_depth && exact == best_exact && diff < best_diff)) {
            best = f;
            best_depth = depth;
            best_diff = diff;
            best_exact = exact;
            found = true;
        }
    }
    return best;
}

static const std::vector<common_download_sidecar> sidecars = {
    { "mmproj", "mmproj",  false, &common_download_opts::download_mmproj, &common_download_hf_plan::mmproj },
    { "mtp",    "mtp-",    true,  &common_download_opts::download_mtp,    &common_download_hf_plan::mtp    },
    { "dflash", "dflash-", true,  &common_download_opts::download_dflash, &common_download_hf_plan::dflash },
    { "eagle3", "eagle3-", true,  &common_download_opts::download_eagle3, &common_download_hf_plan::eagle3 },
    { "dspark", "dspark-", true,  &common_download_opts::download_dspark, &common_download_hf_plan::dspark },
};

const std::vector<common_download_sidecar> & common_download_sidecars() {
    return sidecars;
}

bool common_download_opts_add_sidecar(common_download_opts & opts, const std::string & kind) {
    for (const auto & s : sidecars) {
        if (kind == s.name) {
            opts.*s.opt = true;
            return true;
        }
    }
    return false;
}

std::vector<std::string> common_download_plan_sidecars(const common_download_hf_plan & plan) {
    std::vector<std::string> out;
    for (const auto & s : sidecars) {
        if (!(plan.*s.file).path.empty()) {
            out.emplace_back(s.name);
        }
    }
    return out;
}

std::vector<common_download_partial> common_download_plan_partials(const common_download_hf_plan & plan) {
    std::vector<common_download_partial> out;

    // a file in the cache resolves to its snapshot entry, a file still to download to its blob
    auto add = [&](const std::string & kind, const hf_cache::hf_file & f) {
        if (!f.local_path.empty() && f.local_path != f.final_path) {
            out.push_back({ kind, f.local_path + PARTIAL_SUFFIX });
        }
    };

    for (const auto & f : plan.model_files) {
        add("", f);
    }
    for (const auto & s : sidecars) {
        add(s.name, plan.*s.file);
    }
    return out;
}

// true when `name` carries the keyword of a sidecar kind
static bool name_is_sidecar(const std::string & name) {
    for (const auto & s : sidecars) {
        if (name.find(s.keyword) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool gguf_filename_is_model(const std::string & filepath) {
    if (!string_ends_with(filepath, ".gguf")) {
        return false;
    }

    std::string filename = filepath;
    if (auto pos = filename.rfind('/'); pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }

    return filename.find("imatrix") == std::string::npos && !name_is_sidecar(filename);
}

static hf_cache::hf_file find_best_model(const hf_cache::hf_files & files,
                                         const std::string        & tag) {
    std::vector<std::string> tags;

    if (!tag.empty()) {
        tags.push_back(tag);
    } else {
        tags = {"Q4_K_M", "Q8_0"};
    }

    for (const auto & t : tags) {
        std::regex pattern(t + "[.-]", std::regex::icase);
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path) &&
                std::regex_search(f.path, pattern)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    // fallback to first available model only if tag is empty
    if (tag.empty()) {
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    return {};
}

static void list_available_gguf_files(const hf_cache::hf_files & files) {
    LOG_INF("Available GGUF files:\n");
    for (const auto & f : files) {
        if (string_ends_with(f.path, ".gguf")) {
            LOG_INF(" - %s\n", f.path.c_str());
        }
    }
}

// `-hf repo:TAG` when no file NAME carries TAG: read general.file_type from the head of each untagged model
// file with ranged GETs (common/gguf-header-probe.h) and take the first whose ftype tag is TAG. Only files whose
// name has no tag are probed (a named Q8_0 is not a Q4_K_M), only first shards, at most HF_PROBE_MAX_FILES of
// them, each at most COMMON_GGUF_PROBE_MAX bytes. LLAMA_HF_PROBE_FTYPE=0 turns it off.
static constexpr size_t HF_PROBE_MAX_FILES = 16;  // bounds the requests one plan can make

static hf_cache::hf_file find_model_by_header_tag(const hf_cache::hf_files & files, const std::string & tag,
                                                  const std::string & bearer_token) {
    const char * env = std::getenv("LLAMA_HF_PROBE_FTYPE");
    if (env && std::string(env) == "0") {
        return {};
    }
    std::string want = tag;
    for (char & c : want) {
        c = (char) std::toupper((unsigned char) c);
    }
    size_t probed = 0;
    for (const auto & f : files) {
        if (!gguf_filename_is_model(f.path) || f.url.empty()) {
            continue;
        }
        auto split = get_gguf_split_info(f.path);
        if (!split.tag.empty() && common_ftype_tag_is_known(split.tag)) {
            continue;  // the name already says what it is, and it is not TAG (find_best_model looked)
        }
        if (split.count > 1 && split.index != 1) {
            continue;
        }
        if (probed++ >= HF_PROBE_MAX_FILES) {
            LOG_WRN("%s: stopped after probing %zu untagged files for %s\n", __func__, HF_PROBE_MAX_FILES, want.c_str());
            break;
        }
        auto [cli, parts] = common_http_client(f.url);
        httplib::Headers base;
        if (!bearer_token.empty()) {
            base.emplace("Authorization", "Bearer " + bearer_token);
        }
        auto fetch = [&](uint64_t off, uint64_t len, std::vector<uint8_t> & out) {
            httplib::Headers h = base;
            h.emplace("Range", "bytes=" + std::to_string(off) + "-" + std::to_string(off + len - 1));
            uint64_t got = 0;
            bool     ok  = true;
            auto res = cli.Get(parts.path, h,
                [&](const httplib::Response & r) {
                    size_t cr = 0;
                    // 206 at the asked offset; a 200 (Range ignored) is usable only for the first read
                    ok = (r.status == 206 && common_content_range_start(r.get_header_value("Content-Range"), cr) && cr == off) ||
                         (r.status == 200 && off == 0);
                    return ok;
                },
                [&](const char * data, size_t n) {
                    const uint64_t take = std::min<uint64_t>(n, len - got);
                    out.insert(out.end(), (const uint8_t *) data, (const uint8_t *) data + take);
                    got += take;
                    return got < len;  // stop a 200 body once the range is filled
                });
            return ok && got > 0 && (res || got == len);
        };
        uint64_t    bytes = 0;
        std::string found = common_gguf_probe_tag(fetch, 0, &bytes);
        LOG_INF("%s: %s -> general.file_type %s (%llu bytes read)\n", __func__, f.path.c_str(),
                found.empty() ? "not found" : found.c_str(), (unsigned long long) bytes);
        if (!found.empty() && found == want) {
            return f;
        }
    }
    return {};
}

common_download_hf_plan common_download_get_hf_plan(const common_params_model & model, const common_download_opts & opts) {
    common_download_hf_plan plan;
    hf_cache::hf_files all;

    auto [repo, tag] = common_download_split_repo_tag(model.hf_repo);

    if (!opts.offline) {
        all = hf_cache::get_repo_files(repo, opts.bearer_token, model.hf_revision);
    }
    if (all.empty() && model.hf_revision.empty()) {
        // the cache listing follows the main ref: with a pinned revision it would be a DIFFERENT revision
        all = hf_cache::get_cached_files(repo);
    }
    if (all.empty()) {
        // without this the next message is "failed to open GGUF file ''", which names no cause
        LOG_ERR("%s: no files for repository %s: %s, and nothing for it in the local cache "
                "(set LLAMA_CACHE / HF_HUB_CACHE if the model is cached elsewhere)\n",
                __func__, repo.c_str(), opts.offline ? "offline mode" : "the Hub listing failed");
        return plan;
    }

    // if preset.ini exists in the repo root, download only that file
    for (const auto & f : all) {
        if (f.path == "preset.ini") {
            plan.preset = f;
            return plan;
        }
    }

    hf_cache::hf_file primary;

    if (!model.hf_file.empty()) {
        for (const auto & f : all) {
            if (f.path == model.hf_file) {
                primary = f;
                break;
            }
        }
        if (primary.path.empty()) {
            LOG_ERR("%s: file '%s' not found in repository\n", __func__, model.hf_file.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    } else {
        primary = find_best_model(all, tag);
        if (primary.path.empty() && !tag.empty() && !opts.offline) {
            primary = find_model_by_header_tag(all, tag, opts.bearer_token);
        }
        // a requested draft can resolve on its own, without a full model of the same tag
        const bool want_draft = std::any_of(sidecars.begin(), sidecars.end(), [&](const common_download_sidecar & s) {
            return s.draft && opts.*s.opt;
        });
        if (primary.path.empty() && !want_draft) {
            LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    }

    if (!primary.path.empty()) {
        plan.primary = primary;
        plan.model_files = get_split_files(all, primary);
    }

    // a draft resolves at the tag, the mmproj pairs with the model
    for (const auto & s : sidecars) {
        if (!(opts.*s.opt)) {
            continue;
        }
        if (s.draft) {
            plan.*s.file = find_best_sibling(all, primary.path, s.keyword, tag);
        } else if (!primary.path.empty()) {
            plan.*s.file = find_best_sibling(all, primary.path, s.keyword);
        }
    }

    const bool has_draft = std::any_of(sidecars.begin(), sidecars.end(), [&](const common_download_sidecar & s) {
        return s.draft && !(plan.*s.file).local_path.empty();
    });
    if (primary.path.empty() && !has_draft) {
        LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
        list_available_gguf_files(all);
    }

    return plan;
}

void common_download_run_tasks(const std::vector<common_download_task> & tasks) {
    std::vector<std::future<int>> futures;
    for (const auto & task : tasks) {
        futures.push_back(std::async(std::launch::async,
            [&task]() {
                return common_download_file_single(task.url, task.local_path, task.opts, task.is_hf, task.lock_path, task.alt_path);
            }
        ));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        std::string url = tasks[i].url;
        int status = futures[i].get();
        bool is_ok = is_http_status_ok(status);
        if (!is_ok) {
            throw std::runtime_error(string_format("Download '%s' failed with status code: %d", url.c_str(), status));
        }
    }
}

std::vector<std::string> common_download_get_all_parts(const std::string & url) {
    auto split = get_gguf_split_info(url);

    if (split.count <= 1) {
        return {url};
    }

    std::vector<std::string> parts;
    for (int i = 1; i <= split.count; i++) {
        auto suffix = string_format("-%05d-of-%05d.gguf", i, split.count);
        parts.push_back(split.prefix + suffix);
    }
    return parts;
}

//
// Docker registry functions
//

static std::string common_docker_get_token(const std::string & repo) {
    std::string url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + repo + ":pull";

    common_remote_params params;
    auto                 res = common_remote_get_content(url, params);

    if (res.first != 200) {
        throw std::runtime_error("Failed to get Docker registry token, HTTP code: " + std::to_string(res.first));
    }

    std::string response_str(res.second.begin(), res.second.end());
    common_json response = common_json::parse(response_str);

    if (!response.contains("token")) {
        throw std::runtime_error("Docker registry token response missing 'token' field");
    }

    return response["token"].get<std::string>();
}

std::string common_docker_resolve_model(const std::string & docker) {
    // Parse ai/smollm2:135M-Q4_0
    size_t      colon_pos = docker.find(':');
    std::string repo, tag;
    if (colon_pos != std::string::npos) {
        repo = docker.substr(0, colon_pos);
        tag  = docker.substr(colon_pos + 1);
    } else {
        repo = docker;
        tag  = "latest";
    }

    // ai/ is the default
    size_t      slash_pos = docker.find('/');
    if (slash_pos == std::string::npos) {
        repo.insert(0, "ai/");
    }

    LOG_INF("%s: Downloading Docker Model: %s:%s\n", __func__, repo.c_str(), tag.c_str());
    try {
        // --- helper: digest validation ---
        auto validate_oci_digest = [](const std::string & digest) -> std::string {
            // Expected: algo:hex ; start with sha256 (64 hex chars)
            // You can extend this map if supporting other algorithms in future.
            static const std::regex re("^sha256:([a-fA-F0-9]{64})$");
            std::smatch m;
            if (!std::regex_match(digest, m, re)) {
                throw std::runtime_error("Invalid OCI digest format received in manifest: " + digest);
            }
            // normalize hex to lowercase
            std::string normalized = digest;
            std::transform(normalized.begin()+7, normalized.end(), normalized.begin()+7, [](unsigned char c){
                return std::tolower(c);
            });
            return normalized;
        };

        std::string token = common_docker_get_token(repo);  // Get authentication token

        // Get manifest
        // TODO: cache the manifest response so that it appears in the model list
        const std::string    url_prefix = "https://registry-1.docker.io/v2/" + repo;
        std::string          manifest_url = url_prefix + "/manifests/" + tag;
        common_remote_params manifest_params;
        manifest_params.headers.push_back({"Authorization", "Bearer " + token});
        manifest_params.headers.push_back({"Accept",
            "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
        });
        auto manifest_res = common_remote_get_content(manifest_url, manifest_params);
        if (manifest_res.first != 200) {
            throw std::runtime_error("Failed to get Docker manifest, HTTP code: " + std::to_string(manifest_res.first));
        }

        std::string manifest_str(manifest_res.second.begin(), manifest_res.second.end());
        common_json manifest = common_json::parse(manifest_str);
        std::string gguf_digest;  // Find the GGUF layer
        if (manifest.contains("layers")) {
            for (const auto & layer : manifest["layers"]) {
                if (layer.contains("mediaType")) {
                    std::string media_type = layer["mediaType"].get<std::string>();
                    if (media_type == "application/vnd.docker.ai.gguf.v3" ||
                        media_type.find("gguf") != std::string::npos) {
                        gguf_digest = layer["digest"].get<std::string>();
                        break;
                    }
                }
            }
        }

        if (gguf_digest.empty()) {
            throw std::runtime_error("No GGUF layer found in Docker manifest");
        }

        // Validate & normalize digest
        gguf_digest = validate_oci_digest(gguf_digest);
        LOG_DBG("%s: Using validated digest: %s\n", __func__, gguf_digest.c_str());

        // Prepare local filename
        std::string model_filename = repo;
        std::replace(model_filename.begin(), model_filename.end(), '/', '_');
        model_filename += "_" + tag + ".gguf";
        std::string local_path = fs_get_cache_file(model_filename).string();

        const std::string blob_url = url_prefix + "/blobs/" + gguf_digest;
        common_download_opts opts;
        opts.bearer_token    = token;
        opts.expected_sha256 = gguf_digest.substr(7); // validated above as "sha256:" + 64 hex, lower-cased
        const int http_status = common_download_file_single(blob_url, local_path, opts);
        if (!is_http_status_ok(http_status)) {
            throw std::runtime_error("Failed to download Docker Model");
        }

        LOG_INF("%s: Downloaded Docker Model to: %s\n", __func__, local_path.c_str());
        return local_path;
    } catch (const std::exception & e) {
        LOG_ERR("%s: Docker Model download failed: %s\n", __func__, e.what());
        throw;
    }
}

// a cached file that lists a model under its tag: the first split of a model, not a sidecar
static bool is_listed_model(const gguf_split_info & split) {
    return split.index == 1 && !split.tag.empty() && !name_is_sidecar(split.prefix);
}

std::vector<common_cached_model_info> common_list_cached_models() {
    std::unordered_set<std::string> seen;
    std::vector<common_cached_model_info> result;

    auto files = hf_cache::get_cached_files();

    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (!is_listed_model(split)) {
            continue;
        }
        if (seen.insert(f.repo_id + ":" + split.tag).second) {
            result.push_back({f.repo_id, split.tag});
        }
    }

    return result;
}

std::string common_download_resolve_path(const std::string & hf_repo_with_tag, const std::string & hf_file) {
    auto [repo, tag] = common_download_split_repo_tag(hf_repo_with_tag);

    auto files = hf_cache::get_cached_files(repo);
    if (files.empty()) {
        return "";
    }

    if (!hf_file.empty()) {
        for (const auto & f : files) {
            if (f.path == hf_file) {
                return f.local_path;
            }
        }
        return "";
    }

    return find_best_model(files, tag).local_path;
}

// the files -hf loads from the cache, the model with every sidecar, for each cached
// tag of `repo_id` other than `tag`
static std::unordered_set<std::string> files_in_use(const std::string & repo_id, const std::string & tag) {
    common_download_opts opts;
    opts.offline = true;
    for (const auto & s : sidecars) {
        opts.*s.opt = true;
    }

    std::unordered_set<std::string> tags;
    std::unordered_set<std::string> used;

    for (const auto & f : hf_cache::get_cached_files(repo_id)) {
        auto split = get_gguf_split_info(f.path);
        if (!is_listed_model(split) || split.tag == tag || !tags.insert(split.tag).second) {
            continue;
        }

        common_params_model model;
        model.hf_repo = repo_id + ":" + split.tag;

        auto plan = common_download_get_hf_plan(model, opts);

        for (const auto & m : plan.model_files) {
            used.insert(m.local_path);
        }
        for (const auto & s : sidecars) {
            const auto & sf = plan.*s.file;
            if (!sf.local_path.empty()) {
                used.insert(sf.local_path);
            }
        }
    }
    return used;
}

static bool remove_snapshot_entries(const std::string & repo_id, const std::vector<std::filesystem::path> & to_remove);

bool common_download_remove(const std::string & hf_repo_with_tag) {
    namespace fs = std::filesystem;

    auto [repo_id, tag] = common_download_split_repo_tag(hf_repo_with_tag);

    if (tag.empty()) {
        return hf_cache::remove_cached_repo(repo_id);
    }

    std::string tag_upper = tag;
    for (char & c : tag_upper) {
        c = (char) std::toupper((unsigned char) c);
    }

    auto files = hf_cache::get_cached_files(repo_id);
    if (files.empty()) {
        return false;
    }

    // collect snapshot entries whose tag matches, but a file another tag loads stays
    const auto used = files_in_use(repo_id, tag_upper);

    std::vector<fs::path> to_remove;
    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (split.tag == tag_upper && used.find(f.local_path) == used.end()) {
            to_remove.emplace_back(f.local_path);
        }
    }

    if (to_remove.empty()) {
        return false;
    }

    return remove_snapshot_entries(repo_id, to_remove);
}

// remove snapshot entries of `repo_id`, then the blobs no remaining entry references
static bool remove_snapshot_entries(const std::string & repo_id, const std::vector<std::filesystem::path> & to_remove) {
    namespace fs = std::filesystem;
    (void) repo_id;

    // resolve blob paths from symlinks before deleting snapshot entries
    std::vector<fs::path> blobs_to_check;
    for (const auto & p : to_remove) {
        std::error_code ec;
        if (fs::is_symlink(p, ec)) {
            auto target = fs::read_symlink(p, ec);
            if (!ec) {
                blobs_to_check.push_back((p.parent_path() / target).lexically_normal());
            }
        }
    }

    // remove snapshot entries
    for (const auto & p : to_remove) {
        std::error_code ec;
        fs::remove(p, ec);
        if (ec) {
            LOG_WRN("%s: failed to remove %s: %s\n", __func__, p.string().c_str(), ec.message().c_str());
        }
    }

    if (blobs_to_check.empty()) {
        return true;
    }

    // collect blobs still referenced by remaining snapshot entries, in EVERY
    // revision's snapshot dir: get_cached_files() lists only the ref's current
    // commit, so a blob another revision still links would be deleted
    std::unordered_set<std::string> still_referenced;
    {
        fs::path snapshots;
        for (fs::path a = fs::path(to_remove.front()).parent_path(); !a.empty() && a != a.parent_path(); a = a.parent_path()) {
            if (a.filename() == "snapshots") {
                snapshots = a;
                break;
            }
        }
        std::error_code ec;
        if (!snapshots.empty() && fs::is_directory(snapshots, ec)) {
            for (fs::recursive_directory_iterator it(snapshots, ec), end; !ec && it != end; it.increment(ec)) {
                std::error_code ec2;
                if (!it->is_symlink(ec2)) {
                    continue;
                }
                auto target = fs::read_symlink(it->path(), ec2);
                if (!ec2) {
                    still_referenced.insert((it->path().parent_path() / target).lexically_normal().string());
                }
            }
        }
    }

    // remove orphaned blobs
    for (const auto & blob : blobs_to_check) {
        if (still_referenced.find(blob.string()) == still_referenced.end()) {
            std::error_code ec;
            fs::remove(blob, ec);
            if (ec) {
                LOG_WRN("%s: failed to remove blob %s: %s\n", __func__, blob.string().c_str(), ec.message().c_str());
            }
        }
    }

    return true;
}

bool common_download_remove_sidecars(const std::string & hf_repo_with_tag, const common_download_opts & opts) {
    common_params_model model;
    model.hf_repo = hf_repo_with_tag;

    common_download_opts cached = opts;
    cached.offline = true;

    auto plan = common_download_get_hf_plan(model, cached);

    std::vector<std::filesystem::path> to_remove;
    for (const auto & s : sidecars) {
        const auto & f = plan.*s.file;
        if (!f.local_path.empty()) {
            to_remove.emplace_back(std::filesystem::u8path(f.local_path));
        }
    }
    if (to_remove.empty()) {
        return false;
    }

    return remove_snapshot_entries(common_download_split_repo_tag(hf_repo_with_tag).first, to_remove);
}
