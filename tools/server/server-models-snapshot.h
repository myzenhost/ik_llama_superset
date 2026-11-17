#pragma once

// Router loaded-set snapshot (--models-snapshot FILE), merged from TorchServe's model snapshot
// (docs/snapshot.md; WorkLoadManager.java saveSnapshot, Apache-2.0; behaviour only, no code copied).
// Row S69-TORCHSERVE-PICK-ROUTER-LOADED-SET-SNAPSHOT-2026-10-18-01.
//
// The router writes the names of its loaded models to FILE after every load or unload, and on the next
// start loads them again. One deliberate difference from TorchServe: TorchServe ignores the models named
// on the command line when it restores a snapshot; here an explicit preset always wins. A model whose
// preset sets load-on-startup (to either value) is decided by that preset, and the snapshot entry for it
// is skipped and logged. Snapshot entries are added after the preset ones and only up to models_max.
//
// Everything here is pure (no router state) so tests/test-server-models-snapshot.cpp can drive it.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#define SERVER_MODELS_SNAPSHOT_HEADER "# llama-server models snapshot v1"

struct server_models_snapshot_plan {
    std::vector<std::string> load;             // the startup load list: preset entries first, then restored ones
    std::vector<std::string> restored;         // entries the snapshot added
    std::vector<std::string> skipped_unknown;  // in the snapshot, but no such model is configured now
    std::vector<std::string> skipped_explicit; // a preset sets load-on-startup for it, so the preset decides
    std::vector<std::string> skipped_cap;      // adding it would exceed models_max
};

// text -> names. Returns false (names empty, err set) when the first non-empty line is not the v1 header,
// so a file of some other format is never read as a model list. Blank lines and '#' lines are ignored,
// surrounding whitespace is trimmed, duplicates are dropped (first one kept).
static inline bool server_models_snapshot_parse(const std::string & text, std::vector<std::string> & names, std::string & err) {
    names.clear();
    err.clear();
    std::istringstream in(text);
    std::string line;
    bool header_seen = false;
    std::set<std::string> seen;
    while (std::getline(in, line)) {
        const auto b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) {
            continue;
        }
        const auto e = line.find_last_not_of(" \t\r\n");
        line = line.substr(b, e - b + 1);
        if (!header_seen) {
            if (line != SERVER_MODELS_SNAPSHOT_HEADER) {
                err = "first line is not '" SERVER_MODELS_SNAPSHOT_HEADER "'";
                return false;
            }
            header_seen = true;
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        if (seen.insert(line).second) {
            names.push_back(line);
        }
    }
    if (!header_seen) {
        err = "empty file (no '" SERVER_MODELS_SNAPSHOT_HEADER "' line)";
        return false;
    }
    return true;
}

static inline std::string server_models_snapshot_serialize(const std::vector<std::string> & names) {
    std::string out = SERVER_MODELS_SNAPSHOT_HEADER "\n";
    for (const auto & n : names) {
        out += n;
        out += '\n';
    }
    return out;
}

// preset_load: models whose preset turns load-on-startup on (already in mapping order).
// explicit_names: every model whose preset sets load-on-startup at all, to either value.
// known: every configured model name. models_max <= 0 means no cap on the restored entries.
static inline server_models_snapshot_plan server_models_snapshot_merge(
        const std::vector<std::string> & preset_load,
        const std::vector<std::string> & snapshot,
        const std::set<std::string>    & known,
        const std::set<std::string>    & explicit_names,
        int models_max) {
    server_models_snapshot_plan plan;
    plan.load = preset_load;
    std::set<std::string> in_plan(preset_load.begin(), preset_load.end());
    for (const auto & name : snapshot) {
        if (in_plan.count(name)) {
            continue; // the preset already loads it
        }
        if (!known.count(name)) {
            plan.skipped_unknown.push_back(name);
            continue;
        }
        if (explicit_names.count(name)) {
            plan.skipped_explicit.push_back(name);
            continue;
        }
        if (models_max > 0 && (int) plan.load.size() >= models_max) {
            plan.skipped_cap.push_back(name);
            continue;
        }
        plan.load.push_back(name);
        plan.restored.push_back(name);
        in_plan.insert(name);
    }
    return plan;
}

// read FILE; a missing file is not an error (first start), it yields ok=true and no names
static inline bool server_models_snapshot_read(const std::string & path, std::vector<std::string> & names, std::string & err) {
    names.clear();
    err.clear();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return true;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return server_models_snapshot_parse(ss.str(), names, err);
}

// write FILE atomically: the full text goes to FILE.tmp, which then replaces FILE in one rename, so a reader
// (or a crash mid-write) sees the old list or the new one, never half of one
static inline bool server_models_snapshot_write(const std::string & path, const std::vector<std::string> & names, std::string & err) {
    err.clear();
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            err = "cannot open " + tmp;
            return false;
        }
        const std::string text = server_models_snapshot_serialize(names);
        f.write(text.data(), (std::streamsize) text.size());
        f.flush();
        if (!f) {
            err = "write failed: " + tmp;
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec); // replaces an existing FILE (MoveFileExW REPLACE_EXISTING on Windows)
    if (ec) {
        err = "rename " + tmp + " -> " + path + ": " + ec.message();
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}
