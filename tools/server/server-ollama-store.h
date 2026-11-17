#pragma once
// Read/maintain an Ollama model store (OLLAMA_MODELS layout): resolve "name:tag" -> manifest -> GGUF blob(s).
// Header-only; depends on server-ollama-util.h (name grammar) and nlohmann json only, so tests drive it directly.
//
// SOURCE-READ (ollama/ollama @ eab97e9f92b9a25c2d52d2cc6c1b1c99bd9fae21, MIT; behaviour re-implemented in C++, no Go code copied):
//   manifest/paths.go:28-31,36-40   digest pattern sha256[:-]<64 hex>; blob file is blobs/sha256-<hex>
//   manifest/paths.go:50-67         manifest roots: manifests-v2/ (host "registry.ollama.ai"/"ollama.com" -> "ollama.com", ':' -> %3A)
//                                   and legacy manifests/<host>/<ns>/<model>/<tag>
//   manifest/paths.go:93-168        resolve order: v2 path, then legacy candidates (default public host tried under BOTH spellings)
//   manifest/paths.go:261-281       BlobsPath: "sha256:<hex>" -> blobs/sha256-<hex>
//   manifest/manifest.go:27-48      MediaTypeManifestList; Manifest{config, layers[{mediaType,digest,size}], runner, format, manifests[]}
//   manifest/manifest.go:713-735    a manifest LIST resolves to its child chosen by runner preference
//   manifest/manifest.go:792-860    child may be a bare {digest} reference to a manifest stored as a blob; preference llamacpp, ggml
//   manifest/layer.go:22-34         layer media types (model, projector, adapter, template, system, params, license, messages)
//   server/images.go:767-860        what each layer type means (model blob = GGUF, projector = mmproj GGUF, adapter = LoRA GGUF,
//                                   template = Go template text, system = text, params = JSON options, license = text, messages = JSON)
//   manifest/manifest.go:244-262,587-628  removing a name deletes its manifest file then every blob no other manifest references
#include "server-ollama-util.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ollama_util {

namespace fs = std::filesystem;

struct store_layer {
    std::string media_type;
    std::string digest; // "sha256:<64 hex>" as written in the manifest
    int64_t     size = 0;
};

struct store_model {
    std::string              manifest_path;
    std::string              model_blob;                // path of the application/vnd.ollama.image.model GGUF (first layer; shards ignored)
    std::string              projector_blob;            // first projector layer
    std::vector<std::string> adapter_blobs;
    std::vector<std::string> model_shards;              // model layers after the first (split GGUF): reported, not loaded
    std::string              template_text;             // Go template ("" if none)
    std::string              system_text;
    json                     params = json::object();   // application/vnd.ollama.image.params
    json                     messages = json::array();
    std::vector<std::string> licenses;
    std::vector<store_layer> layers;
    int64_t                  total_size = 0;            // config + every layer, as Manifest.Size() (manifest.go:142-151)
    std::string              runner;
};

// manifest/paths.go digestPattern: sha256 + ':' or '-' + exactly 64 hex. Anything else could smuggle a path into blobs/.
inline bool store_valid_digest(const std::string & d) {
    if (d.size() != 7 + 64 || d.compare(0, 6, "sha256") != 0 || (d[6] != ':' && d[6] != '-')) {
        return false;
    }
    for (size_t i = 7; i < d.size(); ++i) {
        if (!std::isxdigit((unsigned char) d[i])) {
            return false;
        }
    }
    return true;
}

// OLLAMA_MODELS, else $HOME/.ollama/models (envconfig/config.go Models()).
inline std::string store_default_dir() {
    if (const char * e = std::getenv("OLLAMA_MODELS"); e && *e) {
        return e;
    }
    const char * h = std::getenv("HOME");
    return (h && *h) ? std::string(h) + "/.ollama/models" : std::string(".ollama/models");
}

inline std::string store_blob_path(const std::string & dir, const std::string & digest) {
    std::string d = digest;
    d[6] = '-';
    for (char & c : d) c = (char) std::tolower((unsigned char) c);
    return (fs::path(dir) / "blobs" / d).string();
}

inline bool store_read_file(const std::string & path, std::string & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

inline bool is_default_public_host(const std::string & h) {
    const std::string l = lower(h);
    return l == "registry.ollama.ai" || l == "ollama.com";
}

// Candidate manifest files for a parsed name, in ollama's resolve order (paths.go:98-152).
inline std::vector<std::string> store_manifest_candidates(const std::string & dir, model_name n) {
    std::vector<std::string> out;
    if (is_default_public_host(n.host)) n.host = lower(n.host); // the public host is case-insensitive (EqualFold); tags are not
    auto enc = [](std::string h) { size_t p; while ((p = h.find(':')) != std::string::npos) h.replace(p, 1, "%3A"); return h; };
    const fs::path root(dir);
    out.push_back((root / "manifests-v2" / enc(is_default_public_host(n.host) ? "ollama.com" : n.host) / n.ns / n.model / n.tag).string());
    out.push_back((root / "manifests" / n.host / n.ns / n.model / n.tag).string());
    if (is_default_public_host(n.host)) {
        const std::string alt = lower(n.host) == "registry.ollama.ai" ? "ollama.com" : "registry.ollama.ai";
        out.push_back((root / "manifests" / alt / n.ns / n.model / n.tag).string());
    }
    return out;
}

// Pick the concrete manifest object out of parsed JSON: plain manifest as is; a manifest list resolves to the child whose
// runner is llamacpp, then ggml (manifest.go:843-849, 818-841); a bare {digest} child is read from blobs/ (792-816).
inline bool store_select_manifest(const std::string & dir, const json & m, json & out, std::string & err, int depth = 0) {
    if (!m.is_object() || depth > 2) { err = "malformed manifest"; return false; }
    if (m.value("mediaType", std::string()) != "application/vnd.ollama.manifest.list.v2+json") {
        out = m;
        return true;
    }
    const json kids = m.value("manifests", json::array());
    for (const char * want : { "llamacpp", "ggml" }) {
        for (const auto & k : kids) {
            if (!k.is_object() || lower(k.value("runner", std::string())) != want) continue;
            const std::string mt = k.value("mediaType", std::string());
            if (!mt.empty() && mt != "application/vnd.docker.distribution.manifest.v2+json") continue;
            if (k.contains("layers")) { out = k; return true; }
            const std::string dg = k.value("digest", std::string());
            if (!store_valid_digest(dg)) { err = "manifest list child has an invalid digest"; return false; }
            std::string text;
            if (!store_read_file(store_blob_path(dir, dg), text)) continue; // child blob not local: try the next runner
            json child = json::parse_no_throw(text);
            if (child.is_discarded()) { err = "manifest list child blob is not JSON"; return false; }
            return store_select_manifest(dir, child, out, err, depth + 1);
        }
    }
    err = "no compatible manifest (need runner llamacpp or ggml)";
    return false;
}

// Resolve `name` against the store at `dir`. On success every blob path in `out` exists on disk.
inline bool store_resolve(const std::string & dir, const std::string & name, store_model & out, std::string & err) {
    const model_name n = parse_name(name);
    if (!valid(n)) { err = "invalid model name '" + name + "'"; return false; }
    std::string text, found;
    for (const auto & c : store_manifest_candidates(dir, n)) {
        if (fs::is_regular_file(c) && store_read_file(c, text)) { found = c; break; }
    }
    if (found.empty()) { err = "model '" + name + "' not found in " + dir; return false; }
    json top = json::parse_no_throw(text), m;
    if (top.is_discarded()) { err = found + ": not JSON"; return false; }
    if (!store_select_manifest(dir, top, m, err)) { err = found + ": " + err; return false; }

    out = store_model();
    out.manifest_path = found;
    out.runner = m.value("runner", std::string());
    for (const auto & l : m.value("layers", json::array())) {
        store_layer sl;
        sl.media_type = l.value("mediaType", std::string());
        sl.digest     = l.value("digest", std::string());
        sl.size       = l.value("size", (int64_t) 0);
        if (!store_valid_digest(sl.digest)) { err = found + ": layer with invalid digest"; return false; }
        out.layers.push_back(sl);
    }
    out.total_size = m.contains("config") && m["config"].is_object() ? m["config"].value("size", (int64_t) 0) : 0;
    for (const auto & l : out.layers) out.total_size += l.size;
    for (const auto & l : out.layers) {
        if (l.media_type == "application/vnd.ollama.image.tensor") continue; // safetensors-era layers: not GGUF
        const std::string path = store_blob_path(dir, l.digest);
        const std::string & mt = l.media_type;
        auto need = [&](std::string & dst) {
            if (!store_read_file(path, dst)) { err = "blob missing: " + path; return false; }
            return true;
        };
        if (mt == "application/vnd.ollama.image.model") {
            if (!fs::exists(path)) { err = "blob missing: " + path; return false; }
            if (out.model_blob.empty()) out.model_blob = path; else out.model_shards.push_back(path);
        } else if (mt == "application/vnd.ollama.image.projector") {
            if (out.projector_blob.empty()) out.projector_blob = path;
        } else if (mt == "application/vnd.ollama.image.adapter") {
            out.adapter_blobs.push_back(path);
        } else if (mt == "application/vnd.ollama.image.template" || mt == "application/vnd.ollama.image.prompt") {
            if (!need(out.template_text)) return false;
        } else if (mt == "application/vnd.ollama.image.system") {
            if (!need(out.system_text)) return false;
        } else if (mt == "application/vnd.ollama.image.params") {
            std::string t; if (!need(t)) return false;
            out.params = json::parse_no_throw(t);
            if (!out.params.is_object()) { err = "params layer is not a JSON object"; return false; }
        } else if (mt == "application/vnd.ollama.image.messages") {
            std::string t; if (!need(t)) return false;
            out.messages = json::parse_no_throw(t);
            if (!out.messages.is_array()) { err = "messages layer is not a JSON array"; return false; }
        } else if (mt == "application/vnd.ollama.image.license") {
            std::string t; if (!need(t)) return false;
            out.licenses.push_back(t);
        }
    }
    if (out.model_blob.empty()) { err = "model '" + name + "' has no GGUF model layer"; return false; }
    return true;
}

// Every <host>/<ns>/<model>/<tag> manifest file under manifests-v2 and manifests, as display names (for /api/tags).
inline std::vector<std::string> store_list(const std::string & dir) {
    std::set<std::string> names;
    for (const char * sub : { "manifests-v2", "manifests" }) {
        std::error_code ec;
        const fs::path root = fs::path(dir) / sub;
        if (!fs::is_directory(root, ec)) continue;
        for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            const fs::path rel = fs::relative(it->path(), root, ec);
            std::vector<std::string> parts;
            for (const auto & p : rel) parts.push_back(p.string());
            if (parts.size() != 4) continue;
            std::string host = parts[0];
            for (size_t p; (p = host.find("%3A")) != std::string::npos; ) host.replace(p, 3, ":");
            if (is_default_public_host(host)) host = "registry.ollama.ai"; // both spellings are one logical host (paths.go:86-94)
            model_name n { "", host, parts[1], parts[2], parts[3] };
            if (valid(n)) names.insert(display_shortest(n));
        }
    }
    return std::vector<std::string>(names.begin(), names.end());
}

// All blob digests referenced by one manifest file (config + layers; list children included, one level).
inline void store_manifest_digests(const std::string & dir, const json & m, std::set<std::string> & out, int depth = 0) {
    if (!m.is_object() || depth > 2) return;
    auto add = [&](const json & l) {
        std::string d = l.value("digest", std::string());
        if (store_valid_digest(d)) { d[6] = '-'; for (char & c : d) c = (char) std::tolower((unsigned char) c); out.insert(d); }
    };
    if (m.contains("config") && m["config"].is_object()) add(m["config"]);
    for (const auto & l : m.value("layers", json::array())) add(l);
    for (const auto & k : m.value("manifests", json::array())) {
        if (k.contains("layers") || k.contains("config")) { store_manifest_digests(dir, k, out, depth + 1); continue; }
        add(k);
        std::string t, d = k.value("digest", std::string());
        if (store_valid_digest(d) && store_read_file(store_blob_path(dir, d), t)) {
            store_manifest_digests(dir, json::parse_no_throw(t), out, depth + 1);
        }
    }
}

// /api/copy: new manifest file with the same bytes under the destination name (server/routes.go CopyHandler -> CopyModel
// copies the manifest only; blobs are shared). Fails if the destination name is invalid or the source is missing.
inline bool store_copy(const std::string & dir, const std::string & src, const std::string & dst, std::string & err) {
    const model_name s = parse_name(src), d = parse_name(dst);
    if (!valid(s) || !valid(d)) { err = "invalid model name"; return false; }
    std::string text, from;
    for (const auto & c : store_manifest_candidates(dir, s)) {
        if (fs::is_regular_file(c) && store_read_file(c, text)) { from = c; break; }
    }
    if (from.empty()) { err = "model '" + src + "' not found"; return false; }
    // write in the layout the source used, so a legacy store stays legacy
    const bool v2 = from.find((fs::path(dir) / "manifests-v2").string()) == 0;
    const std::string host = v2 && is_default_public_host(d.host) ? "ollama.com" : d.host;
    std::string enc = host;
    if (v2) for (size_t p; (p = enc.find(':')) != std::string::npos; ) enc.replace(p, 1, "%3A");
    const fs::path to = fs::path(dir) / (v2 ? "manifests-v2" : "manifests") / (v2 ? enc : host) / d.ns / d.model / d.tag;
    std::error_code ec;
    fs::create_directories(to.parent_path(), ec);
    std::ofstream o(to, std::ios::binary | std::ios::trunc);
    if (!o) { err = "cannot write " + to.string(); return false; }
    o << text;
    return (bool) o;
}

// /api/delete: remove the manifest file(s) for the name, then every blob no remaining manifest references.
inline bool store_remove(const std::string & dir, const std::string & name, std::vector<std::string> & removed_blobs, std::string & err) {
    const model_name n = parse_name(name);
    if (!valid(n)) { err = "invalid model name"; return false; }
    std::set<std::string> candidates;
    bool any = false;
    for (const auto & c : store_manifest_candidates(dir, n)) {
        std::string t;
        if (!fs::is_regular_file(c) || !store_read_file(c, t)) continue;
        store_manifest_digests(dir, json::parse_no_throw(t), candidates);
        std::error_code ec; fs::remove(c, ec);
        any = any || !ec;
    }
    if (!any) { err = "model '" + name + "' not found"; return false; }
    // prune empty manifest dirs, then GC: a candidate blob survives iff some other manifest still references it
    for (const char * sub : { "manifests-v2", "manifests" }) {
        const fs::path root = fs::path(dir) / sub;
        std::error_code ec;
        std::vector<fs::path> dirs;
        for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) if (it->is_directory(ec)) dirs.push_back(it->path());
        std::sort(dirs.begin(), dirs.end(), [](const fs::path & a, const fs::path & b) { return a.string().size() > b.string().size(); });
        for (const auto & p : dirs) if (fs::is_empty(p, ec)) fs::remove(p, ec);
    }
    std::set<std::string> live;
    for (const char * sub : { "manifests-v2", "manifests" }) {
        std::error_code ec;
        const fs::path root = fs::path(dir) / sub;
        for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            std::string t;
            if (it->is_regular_file(ec) && store_read_file(it->path().string(), t)) {
                store_manifest_digests(dir, json::parse_no_throw(t), live);
            }
        }
    }
    for (const auto & d : candidates) {
        if (live.count(d)) continue;
        std::error_code ec;
        if (fs::remove(fs::path(dir) / "blobs" / d, ec)) removed_blobs.push_back(d);
    }
    return true;
}

} // namespace ollama_util
