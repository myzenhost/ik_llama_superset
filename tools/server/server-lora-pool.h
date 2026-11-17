#pragma once

// Server side of --lora-pool-dir: an on-demand LoRA adapter pool in front of llama_adapter_lora_init().
// Policy (LRU, byte/count budget, refcount pinning, refusal when every resident adapter is pinned) is common/lora-adapter-pool.h; this file adds
// what the policy cannot know: where an adapter lives on disk, how to load/free it through the llama API, and the identity (load serial) the
// prompt-cache key needs. Row S69-LORAX-PICK-ON-DEMAND-ADAPTER-POOL-2026-09-29-01 (wired by docs/agent-reports/wire-serving-features.md).
//
// Used from the server's main loop only; the pool itself is thread-safe.

#include "llama.h"
#include "lora-adapter-pool.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <unordered_map>

class server_lora_pool {
public:
    struct counters {
        uint64_t acquires   = 0; // pool lookups for a named adapter
        uint64_t hits       = 0; // already resident
        uint64_t loads      = 0; // read from disk
        uint64_t evictions  = 0; // least-recently-used adapters dropped to make room
        uint64_t refused    = 0; // all resident adapters pinned (no_room) or the adapter alone exceeds the byte budget
        uint64_t failed     = 0; // bad name, missing file, unreadable adapter
    };

    enum class result_kind { ok, bad_name, not_found, load_failed, over_budget, no_room };

    struct pin {
        result_kind          kind   = result_kind::load_failed;
        llama_adapter_lora * ptr    = nullptr;
        uint64_t             serial = 0;   // unique per load: the same name loaded again gets a new serial
        bool                 loaded_now = false;
        std::string          error;
    };

    // Called with the adapter still alive, just before it is freed, so the owner can drop every reference to it (context-wide set, per-sequence
    // lists). Must not call back into the pool.
    using before_free_fn = std::function<void(llama_adapter_lora *)>;

    server_lora_pool(llama_model * model, std::string dir, size_t max_adapters, size_t max_bytes, before_free_fn before_free)
        : model_(model), dir_(std::move(dir)), before_free_(std::move(before_free)),
          pool_(max_adapters, max_bytes,
                [this](const std::string & id, lora_adapter_pool::loaded & out) { return load(id, out); },
                [this](const std::string & id) { on_evict(id); }) {}

    // letters, digits, '.', '_' and '-'; no leading '.', at most 128 bytes. Keeps a request from naming a path outside the pool directory.
    static bool valid_name(const std::string & n) {
        if (n.empty() || n.size() > 128 || n[0] == '.') {
            return false;
        }
        for (char c : n) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
            if (!ok) {
                return false;
            }
        }
        return true;
    }

    std::string path_of(const std::string & name) const {
        return (std::filesystem::path(dir_) / (name + ".gguf")).string();
    }

    pin acquire(const std::string & name) {
        pin p;
        n_.acquires++;
        if (!valid_name(name)) {
            n_.failed++;
            p.kind  = result_kind::bad_name;
            p.error = "lora_adapter: invalid adapter name (letters, digits, '.', '_' and '-' only)";
            return p;
        }
        std::error_code ec;
        if (!pool_.contains(name) && !std::filesystem::is_regular_file(path_of(name), ec)) {
            n_.failed++;
            p.kind  = result_kind::not_found;
            p.error = "lora_adapter: no adapter named '" + name + "' in the LoRA pool";
            return p;
        }
        const auto a = pool_.acquire(name);
        switch (a.st) {
            case lora_adapter_pool::status::ok:
                p.kind       = result_kind::ok;
                p.ptr        = static_cast<llama_adapter_lora *>(a.handle.get());
                p.serial     = serial_[name];
                p.loaded_now = a.loaded_now;
                if (a.loaded_now) {
                    n_.loads++;
                } else {
                    n_.hits++;
                }
                return p;
            case lora_adapter_pool::status::load_failed:
                n_.failed++;
                p.kind  = result_kind::load_failed;
                p.error = "lora_adapter: failed to load '" + name + "' (not a LoRA for this model?)";
                break;
            case lora_adapter_pool::status::over_budget:
                n_.refused++;
                p.kind  = result_kind::over_budget;
                p.error = "lora_adapter: '" + name + "' alone exceeds the LoRA pool byte budget";
                break;
            case lora_adapter_pool::status::no_room:
                n_.refused++;
                p.kind  = result_kind::no_room;
                p.error = "lora_adapter: every resident LoRA adapter is in use by a running request; retry later";
                break;
        }
        // a failed or refused load leaves nothing resident under this name (the loader ran, the pool dropped the adapter)
        if (!pool_.contains(name)) {
            ptr_.erase(name);
            serial_.erase(name);
        }
        return p;
    }

    bool release(const std::string & name) { return pool_.release(name); }

    // 0 when the adapter is not resident
    uint64_t serial_of(const std::string & name) const {
        if (!pool_.contains(name)) {
            return 0;
        }
        const auto it = serial_.find(name);
        return it == serial_.end() ? 0 : it->second;
    }

    llama_adapter_lora * peek(const std::string & name) const {
        const auto it = ptr_.find(name);
        return it == ptr_.end() || !pool_.contains(name) ? nullptr : it->second;
    }

    const counters & stats() const { return n_; }
    size_t resident() const { return pool_.size(); }
    size_t bytes() const { return pool_.bytes(); }
    size_t refs(const std::string & name) const { return pool_.refs(name); }
    std::vector<std::string> lru_order() const { return pool_.lru_order(); }
    const std::string & dir() const { return dir_; }

private:
    bool load(const std::string & id, lora_adapter_pool::loaded & out) {
        std::error_code ec;
        const auto path = path_of(id);
        const auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            return false;
        }
        llama_adapter_lora * ad = llama_adapter_lora_init(model_, path.c_str());
        if (ad == nullptr) {
            return false;
        }
        out.handle = std::shared_ptr<void>(ad, [](void * p) { llama_adapter_lora_free(static_cast<llama_adapter_lora *>(p)); });
        out.bytes  = (size_t) size;
        ptr_[id]    = ad;
        serial_[id] = ++next_serial_;
        loads_++;
        return true;
    }

    void on_evict(const std::string & id) {
        n_.evictions++;
        const auto it = ptr_.find(id);
        if (it != ptr_.end()) {
            if (before_free_) {
                before_free_(it->second);
            }
            ptr_.erase(it);
        }
        serial_.erase(id);
    }

    llama_model *                                 model_;
    std::string                                   dir_;
    before_free_fn                                before_free_;
    counters                                      n_;
    size_t                                        loads_       = 0;
    uint64_t                                      next_serial_ = 0;
    std::unordered_map<std::string, llama_adapter_lora *> ptr_;
    std::unordered_map<std::string, uint64_t>     serial_;
    lora_adapter_pool                             pool_; // declared last: its loader/evict callbacks use the members above
};
