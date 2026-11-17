#include "hf-cache.h"

#include "build-info.h"
#include "common.h"
#include "file-lock.h"
#include "log.h"
#include "http.h"
#include "json.h"

#include <filesystem>
#include <fstream>
#include <atomic>
#include <set>
#include <string>
#include <string_view>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define HOME_DIR "USERPROFILE"
#include <windows.h>
#else
#define HOME_DIR "HOME"
#include <unistd.h>
#include <pwd.h>
#endif

namespace hf_cache {

namespace fs = std::filesystem;

static fs::path get_cache_directory() {
    static const fs::path cache = []() {
        struct {
            const char * var;
            fs::path path;
        } entries[] = {
            {"LLAMA_CACHE",           fs::path()},
            {"HF_HUB_CACHE",          fs::path()},
            {"HUGGINGFACE_HUB_CACHE", fs::path()},
            {"HF_HOME",               fs::path("hub")},
            {"XDG_CACHE_HOME",        fs::path("huggingface") / "hub"},
            {HOME_DIR,                fs::path(".cache") / "huggingface" / "hub"}
        };
        for (const auto & entry : entries) {
            if (fs::path base = common_get_env(entry.var); !base.empty()) {
                return entry.path.empty() ? base : base / entry.path;
            }
        }
#ifndef _WIN32
        const struct passwd * pw = getpwuid(getuid());

        if (pw && pw->pw_dir && *pw->pw_dir) {
            return fs::path(pw->pw_dir) / ".cache" / "huggingface" / "hub";
        }
#endif
        throw std::runtime_error("Failed to determine HF cache directory");
    }();

    return cache;
}

std::string get_cache_path() {
    return get_cache_directory().string();
}

static std::string folder_name_to_repo(const std::string & folder) {
    constexpr std::string_view prefix = "models--";
    if (folder.rfind(prefix, 0)) {
        return {};
    }
    std::string result = folder.substr(prefix.length());
    string_replace_all(result, "--", "/");
    return result;
}

static std::string repo_to_folder_name(const std::string & repo_id) {
    constexpr std::string_view prefix = "models--";
    std::string result = std::string(prefix) + repo_id;
    string_replace_all(result, "/", "--");
    return result;
}

static fs::path get_repo_path(const std::string & repo_id) {
    return get_cache_directory() / repo_to_folder_name(repo_id);
}

static bool is_hex_char(const char c) {
    return (c >= 'A' && c <= 'F') ||
           (c >= 'a' && c <= 'f') ||
           (c >= '0' && c <= '9');
}

static bool is_hex_string(const std::string & s, size_t expected_len) {
    if (s.length() != expected_len) {
        return false;
    }
    for (const char c : s) {
        if (!is_hex_char(c)) {
            return false;
        }
    }
    return true;
}

static bool is_alphanum(const char c) {
    return (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9');
}

static bool is_special_char(char c) {
    return c == '/' || c == '.' || c == '-';
}

// base chars [A-Za-z0-9_] are always valid
// special chars [/.-] must be surrounded by base chars
// exactly one '/' required
static bool is_valid_repo_id(const std::string & repo_id) {
    if (repo_id.empty() || repo_id.length() > 256) {
        return false;
    }
    int slash = 0;
    bool special = true;

    for (const char c : repo_id) {
        if (is_alphanum(c) || c == '_') {
            special = false;
        } else if (is_special_char(c)) {
            if (special) {
                return false;
            }
            slash += (c == '/');
            special = true;
        } else {
            return false;
        }
    }
    return !special && slash == 1;
}

static bool is_valid_hf_token(const std::string & token) {
    if (token.length() < 37 || token.length() > 256 ||
        !string_starts_with(token, "hf_")) {
        return false;
    }
    for (size_t i = 3; i < token.length(); ++i) {
        if (!is_alphanum(token[i])) {
            return false;
        }
    }
    return true;
}

static bool is_valid_commit(const std::string & hash) {
    return is_hex_string(hash, 40);
}

static bool is_valid_oid(const std::string & oid) {
    return is_hex_string(oid, 40) || is_hex_string(oid, 64);
}

std::string get_lock_path(const std::string & repo_id, const std::string & oid) {
    if (!is_valid_repo_id(repo_id) || !is_valid_oid(oid)) {
        return {};
    }
    fs::path lock_path = fs::path(get_cache_path()) / ".locks" / repo_to_folder_name(repo_id) / (oid + ".lock");
    return lock_path.string();
}

static bool is_valid_subpath(const fs::path & path, const fs::path & subpath) {
    if (subpath.is_absolute()) {
        return false; // never do a / b with b absolute
    }
    auto b = fs::absolute(path).lexically_normal();
    auto t = (b / subpath).lexically_normal();
    auto [b_end, _] = std::mismatch(b.begin(), b.end(), t.begin(), t.end());

    return b_end == b.end();
}

// URL of the rel="next" entry of an RFC 8288 Link header, "" when there is none.
static std::string link_header_next(const std::string & link) {
    size_t pos = 0;
    while (pos < link.size()) {
        const size_t lt = link.find('<', pos);
        if (lt == std::string::npos) {
            break;
        }
        const size_t gt = link.find('>', lt);
        if (gt == std::string::npos) {
            break;
        }
        // the params of this entry run to the next '<' (urls hold no raw '<')
        const size_t next_entry = link.find('<', gt);
        const std::string params = link.substr(gt + 1, next_entry == std::string::npos ? std::string::npos : next_entry - gt - 1);
        if (params.find("rel=\"next\"") != std::string::npos || params.find("rel=next") != std::string::npos) {
            return link.substr(lt + 1, gt - lt - 1);
        }
        pos = gt + 1;
    }
    return {};
}

static common_json api_get(const std::string & url,
                           const std::string & token,
                           std::string * link_out = nullptr) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers = {
        {"User-Agent", "llama-cpp/" + std::string(llama_build_info())},
        {"Accept", "application/json"}
    };

    if (is_valid_hf_token(token)) {
        headers.emplace("Authorization", "Bearer " + token);
    } else if (!token.empty()) {
        LOG_WRN("%s: invalid token, authentication disabled\n", __func__);
    }

    if (auto res = cli.Get(parts.path, headers)) {
        auto body = res->body;

        if (res->status == 200) {
            if (link_out) {
                *link_out = res->get_header_value("Link");
            }
            return common_json::parse(res->body);
        }
        try {
            body = common_json::parse(res->body)["error"].get<std::string>();
        } catch (...) { }

        throw std::runtime_error("GET failed (" + std::to_string(res->status) + "): " + body);
    } else {
        throw std::runtime_error("HTTPLIB failed: " + httplib::to_string(res.error()));
    }
}

// revision empty = the main branch (or the first branch when there is none). A non-empty revision is a
// 40-hex commit (used as is), a branch name or a tag name; one that names nothing yields {} and NEVER
// falls back to main, so a pinned download cannot silently fetch a different revision.
static std::string get_repo_commit(const std::string & repo_id,
                                   const std::string & token,
                                   const std::string & revision) {
    if (!revision.empty() && is_valid_commit(revision)) {
        LOG_INF("%s: %s pinned to commit %s\n", __func__, repo_id.c_str(), revision.c_str());
        return revision;
    }
    try {
        auto endpoint = common_get_model_endpoint();
        auto json = api_get(endpoint + "api/models/" + repo_id + "/refs", token);

        if (!json.is_object() ||
            !json.contains("branches") || !json["branches"].is_array()) {
            LOG_WRN("%s: missing 'branches' for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        fs::path refs_path = get_repo_path(repo_id) / "refs";
        std::string name;
        std::string commit;
        fs::path name_path;

        for (const auto & branch : json["branches"]) {
            if (!branch.is_object() ||
                !branch.contains("name") || !branch["name"].is_string() ||
                !branch.contains("targetCommit") || !branch["targetCommit"].is_string()) {
                continue;
            }
            std::string _name = branch["name"].get<std::string>();
            std::string _commit = branch["targetCommit"].get<std::string>();

            if (!is_valid_commit(_commit)) {
                LOG_WRN("%s: skip invalid commit: %s\n", __func__, _commit.c_str());
                continue;
            }
            const fs::path candidate = fs::u8path(_name);

            if (!is_valid_subpath(refs_path, candidate)) {
                LOG_WRN("%s: skip invalid branch: %s\n", __func__, _name.c_str());
                continue;
            }

            if (!revision.empty() && _name != revision) {
                continue;
            }

            if (_name == "main") {
                name = _name;
                commit = _commit;
                name_path = candidate;
                break;
            }

            if (name.empty() || commit.empty()) {
                name = _name;
                commit = _commit;
                name_path = candidate;
            }
        }

        if ((name.empty() || commit.empty()) && !revision.empty() && json.contains("tags") && json["tags"].is_array()) {
            // a tag: not recorded under refs/ (the hub cache layout only keeps branch refs there)
            for (const auto & tag : json["tags"]) {
                if (!tag.is_object() ||
                    !tag.contains("name") || !tag["name"].is_string() || tag["name"].get<std::string>() != revision ||
                    !tag.contains("targetCommit") || !tag["targetCommit"].is_string()) {
                    continue;
                }
                const std::string tag_commit = tag["targetCommit"].get<std::string>();
                if (is_valid_commit(tag_commit)) {
                    LOG_INF("%s: %s tag '%s' resolved to commit %s\n", __func__, repo_id.c_str(), revision.c_str(), tag_commit.c_str());
                    return tag_commit;
                }
            }
        }

        if (name.empty() || commit.empty()) {
            if (!revision.empty()) {
                LOG_ERR("%s: revision '%s' is not a branch, tag or commit of '%s'\n", __func__, revision.c_str(), repo_id.c_str());
            } else {
                LOG_WRN("%s: no valid branch for '%s'\n", __func__, repo_id.c_str());
            }
            return {};
        }

        fs_write_atomic(refs_path / name_path, commit);
        if (!revision.empty()) {
            LOG_INF("%s: %s branch '%s' resolved to commit %s\n", __func__, repo_id.c_str(), revision.c_str(), commit.c_str());
        }
        return commit;

    } catch (const common_json_error & e) {
        LOG_ERR("%s: JSON error: %s\n", __func__, e.what());
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
    }
    return {};
}

hf_files get_repo_files(const std::string & repo_id,
                        const std::string & token,
                        const std::string & revision) {
    if (!is_valid_repo_id(repo_id)) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    std::string commit = get_repo_commit(repo_id, token, revision);
    if (commit.empty()) {
        LOG_WRN("%s: failed to resolve commit for %s\n", __func__, repo_id.c_str());
        return {};
    }

    fs::path blobs_path = get_repo_path(repo_id) / "blobs";
    fs::path commit_path = get_repo_path(repo_id) / "snapshots" / commit;

    hf_files files;

    try {
        auto endpoint = common_get_model_endpoint();
        // The tree is paged: each page names the next in `Link: <url>; rel="next"`. Follow it, or the files past
        // the first page (later shards of a split GGUF) are silently lost. A next link must stay on the endpoint
        // (the bearer token goes with every request) and a repeated URL ends the walk.
        std::string page_url = endpoint + "api/models/" + repo_id + "/tree/" + commit + "?recursive=true";
        std::set<std::string> seen_pages;
        common_json json = common_json::array();
        for (int page = 0; page < 1000; ++page) {
            seen_pages.insert(page_url);
            std::string link;
            auto part = api_get(page_url, token, &link);
            if (!part.is_array()) {
                LOG_WRN("%s: response is not an array for '%s'\n", __func__, repo_id.c_str());
                return {};
            }
            for (auto & item : part) {
                json.push_back(std::move(item));
            }
            const std::string next = link_header_next(link);
            if (next.empty() || seen_pages.count(next)) {
                break;
            }
            if (next.compare(0, endpoint.size(), endpoint) != 0) {
                LOG_WRN("%s: ignoring a next page link outside the endpoint for '%s'\n", __func__, repo_id.c_str());
                break;
            }
            page_url = next;
        }

        for (const auto & item : json) {
            if (!item.is_object() ||
                !item.contains("type") || !item["type"].is_string() || item["type"] != "file" ||
                !item.contains("path") || !item["path"].is_string()) {
                continue;
            }

            hf_file file;
            file.repo_id = repo_id;
            file.path = item["path"].get<std::string>();

            const fs::path subpath = fs::u8path(file.path);

            if (!is_valid_subpath(commit_path, subpath)) {
                LOG_WRN("%s: skip invalid path: %s\n", __func__, file.path.c_str());
                continue;
            }

            if (item.contains("lfs") && item["lfs"].is_object()) {
                if (item["lfs"].contains("oid") && item["lfs"]["oid"].is_string()) {
                    file.oid = item["lfs"]["oid"].get<std::string>();
                }
            } else if (item.contains("oid") && item["oid"].is_string()) {
                file.oid = item["oid"].get<std::string>();
            }

            if (!file.oid.empty() && !is_valid_oid(file.oid)) {
                LOG_WRN("%s: skip invalid oid: %s\n", __func__, file.oid.c_str());
                continue;
            }

            file.url = endpoint + repo_id + "/resolve/" + commit + "/" + file.path;

            fs::path final_path = commit_path / subpath;
            file.final_path = final_path.string();

            if (!file.oid.empty() && !fs::exists(final_path)) {
                fs::path local_path = blobs_path / file.oid;
                file.local_path = local_path.string();
            } else {
                file.local_path = file.final_path;
            }

            files.push_back(file);
        }
    } catch (const common_json_error & e) {
        LOG_ERR("%s: JSON error: %s\n", __func__, e.what());
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
    }
    return files;
}

static std::string get_cached_ref(const fs::path & repo_path) {
    fs::path refs_path = repo_path / "refs";
    if (!fs::is_directory(refs_path)) {
        return {};
    }
    std::string fallback;

    for (const auto & entry : fs::directory_iterator(refs_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::ifstream f(entry.path());
        std::string commit;
        if (!f || !std::getline(f, commit) || commit.empty()) {
            continue;
        }
        if (!is_valid_commit(commit)) {
            LOG_WRN("%s: skip invalid commit: %s\n", __func__, commit.c_str());
            continue;
        }
        if (entry.path().filename() == "main") {
            return commit;
        }
        if (fallback.empty()) {
            fallback = commit;
        }
    }
    return fallback;
}

hf_files get_cached_files(const std::string & repo_id) {
    const fs::path cache_path = get_cache_directory();
    if (!fs::exists(cache_path)) {
        return {};
    }

    if (!repo_id.empty() && !is_valid_repo_id(repo_id)) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    hf_files files;

    for (const auto & repo : fs::directory_iterator(cache_path)) {
        if (!repo.is_directory()) {
            continue;
        }
        fs::path snapshots_path = repo.path() / "snapshots";

        if (!fs::exists(snapshots_path)) {
            continue;
        }
        std::string _repo_id = folder_name_to_repo(repo.path().filename().string());

        if (!is_valid_repo_id(_repo_id)) {
            continue;
        }
        if (!repo_id.empty() && _repo_id != repo_id) {
            continue;
        }
        std::string commit = get_cached_ref(repo.path());
        fs::path commit_path = snapshots_path / commit;

        if (commit.empty() || !fs::is_directory(commit_path)) {
            continue;
        }
        for (const auto & entry : fs::recursive_directory_iterator(commit_path)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) {
                continue;
            }
            fs::path path = entry.path().lexically_relative(commit_path);

            if (!path.empty()) {
                hf_file file;
                file.repo_id = _repo_id;
                const auto generic_path = path.generic_u8string();
                file.path = std::string(generic_path.begin(), generic_path.end());
                file.local_path = entry.path().string();
                file.final_path = file.local_path;
                files.push_back(std::move(file));
            }
        }
    }

    return files;
}

std::string finalize_file(const hf_file & file, const std::function<bool()> & keep_waiting) {
    static std::atomic<bool> symlinks_disabled{false};

    std::error_code ec;
    fs::path local_path = fs::u8path(file.local_path);
    fs::path final_path = fs::u8path(file.final_path);

    if (local_path == final_path || fs::exists(final_path, ec)) {
        return file.final_path;
    }

    if (!fs::exists(local_path, ec)) {
        return file.final_path;
    }

    // the lock file is shared with huggingface_hub: another process may be
    // finalizing the same blob
    common_file_lock lock(get_lock_path(file.repo_id, file.oid));
    if (!lock.acquire(keep_waiting)) {
        // without the lock, another process could move the blob while we check it
        throw std::runtime_error("failed to lock cache for finalization: " + file.final_path);
    }

    // acquire() only polls keep_waiting while waiting for a busy lock
    if (keep_waiting && !keep_waiting()) {
        throw std::runtime_error("download cancelled before finalization: " + file.final_path);
    }

    // re-check under the lock: the winner may have created the link already
    if (fs::exists(final_path, ec)) {
        return file.final_path;
    }

    fs::create_directories(final_path.parent_path(), ec);

    if (!symlinks_disabled) {
        fs::path target = fs::relative(local_path, final_path.parent_path(), ec);
        if (!ec) {
            fs::create_symlink(target, final_path, ec);
        }
        if (!ec) {
            return file.final_path;
        }
    }

    if (!symlinks_disabled.exchange(true)) {
        LOG_WRN("%s: failed to create symlink: %s\n", __func__, ec.message().c_str());
        LOG_WRN("%s: switching to degraded mode\n", __func__);
    }

    // a hard link keeps blobs/<oid> and shares the bytes (no privilege needed, same volume only)
    fs::create_hard_link(local_path, final_path, ec);
    if (!ec) {
        return file.final_path;
    }

    fs::rename(local_path, final_path, ec);
    if (ec) {
        LOG_WRN("%s: failed to move file to snapshots: %s\n", __func__, ec.message().c_str());
        fs::copy(local_path, final_path, ec);
        if (ec) {
            LOG_ERR("%s: failed to copy file to snapshots: %s\n", __func__, ec.message().c_str());
        }
    }
    return file.final_path;
}

bool remove_cached_repo(const std::string & repo_id) {
    if (!is_valid_repo_id(repo_id)) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return false;
    }
    fs::path repo_path = get_repo_path(repo_id);
    std::error_code ec;
    auto removed = fs::remove_all(repo_path, ec);
    if (ec) {
        LOG_ERR("%s: failed to remove repo cache %s: %s\n", __func__, repo_path.string().c_str(), ec.message().c_str());
        return false;
    }
    return removed > 0;
}

} // namespace hf_cache
