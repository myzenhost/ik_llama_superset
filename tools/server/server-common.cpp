#include "common.h"
#include "download.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#include "base64.hpp"

#include "server-common.h"
#include "server-loop-recovery.h"
#include "server-template-kwargs-force.h"
#include "server-token-offsets.h"

#include <cmath>
#include <filesystem>
#include <random>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <type_traits>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#ifdef _WIN32
// windows.h defines min and max as macros, which breaks std::min and std::max
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#include <limits.h>
#else
#include <unistd.h>
#endif

std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
        case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
            type_str = "exceed_context_size_error";
            code = 400;
            break;
    }
    if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        // OpenAI clients (openai-python's APIError.code, LangChain, agent frameworks) key on the STRING code
        // "context_length_exceeded" in error.code, so that is what error.code carries here. The HTTP status moves to
        // "status" (still 400); every reader of the error body takes it through error_http_status().
        // RESEARCH-DONE: NexaAI/nexa-sdk 7a0ff49c4 cli/server/handler/completion.go:330-341 (BSD-3-Clause) returns
        // code 'context_length_exceeded'; openai-python _exceptions.py reads body["code"] into APIError.code.
        return json {
            {"code",    "context_length_exceeded"},
            {"status",  code},
            {"message", message},
            {"type",    type_str},
        };
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

int error_http_status(const json & error_data, int fallback) {
    if (error_data.is_object()) {
        for (const char * key : { "code", "status" }) {
            if (error_data.contains(key) && error_data.at(key).is_number_integer()) {
                return error_data.at(key).get<int>();
            }
        }
    }
    return fallback;
}

//
// server_slot_stats
//

json server_slot_stats::to_json() const {
    json base = {
        {"cache_n",                n_prompt_cached},

        {"prompt_n",               n_prompt_processed},
        {"prompt_ms",              t_prompt_ms()},
        {"prompt_per_token_ms",    t_prompt_per_token_ms()},
        {"prompt_per_second",      n_prompt_tps()},

        {"predicted_n",            n_gen},
        {"predicted_ms",           t_gen_ms()},
        {"predicted_per_token_ms", t_gen_per_token_ms()},
        {"predicted_per_second",   n_gen_tps()},
    };

    if (n_draft_tokens > 0) {
        base["draft_n"]          = n_draft_tokens;
        base["draft_n_accepted"] = n_draft_accepted;
    }

    // reasoning_loop_recovery: only when the request turned it on, so other responses keep their exact shape
    if (rlr_on) {
        base["reasoning_recoveries"] = n_reasoning_recoveries;
        if (reasoning_coverage >= 0.0f) {
            base["reasoning_repeat_coverage"] = reasoning_coverage;
        }
        if (!rlr_note.empty()) {
            base["reasoning_recovery_note"] = rlr_note;
        }
    }

    // time to first token (from submission) and the inter-token latency of this request
    if (t_first_token > 0) {
        base["ttft_ms"] = ttft_ms();
    }
    if (n_itl > 0) {
        base["itl_mean_ms"] = itl_mean_ms();
        base["itl_min_ms"]  = itl_min_us / 1000.0;
        base["itl_max_ms"]  = itl_max_us / 1000.0;
    }

    return base;
}

//
// random string / id
//

std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

std::string gen_tool_call_id() {
    return random_string();
}

const char * get_media_marker() {
    static const std::string marker = []() {
        // allow user to pin a reproducible marker via env var
        const char * env = getenv("LLAMA_MEDIA_MARKER");
        if (env && env[0] != '\0') {
            return std::string(env);
        }
        return std::string("<__media_") + random_string() + "__>";
    }();
    return marker.c_str();
}

//
// model output modalities
//

std::vector<std::string> server_model_output_modalities(common_decision_type decision_type, bool has_cap_chat, bool has_cap_tts) {
    std::vector<std::string> out;
    switch (decision_type) {
        case COMMON_DECISION_TYPE_OPENJEV:
        case COMMON_DECISION_TYPE_LEV:
        case COMMON_DECISION_TYPE_KEV:
        case COMMON_DECISION_TYPE_NIMBLE:
        case COMMON_DECISION_TYPE_LAYA:
        case COMMON_DECISION_TYPE_CLEF:
        case COMMON_DECISION_TYPE_PPLX_DECIDER:
        case COMMON_DECISION_TYPE_VLLM_DECISION2:
        case COMMON_DECISION_TYPE_VLLM_D3:
        case COMMON_DECISION_TYPE_LFM2_D1:
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
        case COMMON_DECISION_TYPE_H2O_LIGHTNING:
            out = {"decisions"};
            break;
        case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            out = {"text", "decisions"};
            break;
        default:
            // a tts-only model cannot generate text
            if (has_cap_chat || !has_cap_tts) {
                // fallback when there is no decision type or the metadata is bad
                out = {"text"};
            }
            break;
    }
    if (has_cap_tts) {
        out.push_back("audio");
    }
    return out;
}

std::vector<std::string> server_model_output_modalities(const llama_model * model, bool has_cap_chat, bool has_cap_tts) {
    return server_model_output_modalities(common_get_decision_type(model), has_cap_chat, has_cap_tts);
}

json server_model_architecture_json(
        bool inp_image,
        bool inp_audio,
        bool inp_video,
        const std::vector<std::string> & output_modalities) {
    std::vector<std::string> input_modalities = {"text"};
    if (inp_image) {
        input_modalities.push_back("image");
    }
    if (inp_audio) {
        input_modalities.push_back("audio");
    }
    if (inp_video) {
        input_modalities.push_back("video");
    }

    return {
        {"input_modalities",  input_modalities},
        {"output_modalities", output_modalities},
    };
}

//
// lora utils
//

bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras) {
    bool found_alora = false;
    for (const auto & lora : loras) {
        if (lora.scale != 0) {
            if (llama_adapter_get_alora_n_invocation_tokens(lora.ptr) == 0) {
                return false;
            }
            found_alora = true;
        }
    }
    return found_alora;
}

bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next) {

    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    GGML_ASSERT(!are_lora_equal(current, next));

    return (
        !(lora_get_enabled_ids(current).empty() || lora_all_alora(current)) ||
        !lora_all_alora(next));
}

std::map<int, float> parse_lora_request(const json & data) {
    std::map<int, float> lora;

    // set value
    for (const auto & entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        lora[id] = scale;
    }

    return lora;
}

void validate_lora_request(const json & lora, size_t n_server_adapters) {
    if (!lora.is_array()) {
        throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
    }
    // Each entry: an object with an integer id of an adapter this server loaded and a numeric scale. These used to be
    // dropped silently (json_value falls back to a default), so a wrong id answered 200 and changed nothing.
    // SOURCE-READ: vllm/entrypoints/openai/serve/engine/serving.py:40-75 (unknown adapter -> error), lora/request.py.
    for (size_t i = 0; i < lora.size(); ++i) {
        const auto & e = lora[i];
        if (!e.is_object() || !e.contains("id") || !e["id"].is_number_integer() ||
            (e.contains("scale") && !e["scale"].is_number())) {
            throw std::runtime_error("Error: 'lora' entry " + std::to_string(i) + " must be an object with an integer 'id' and a numeric 'scale'");
        }
        const int64_t id = e["id"].get<int64_t>();
        if (id < 0 || id >= (int64_t) n_server_adapters) {
            throw std::runtime_error("Error: 'lora' entry " + std::to_string(i) + ": id " + std::to_string(id) +
                                     " out of range (server has " + std::to_string(n_server_adapters) + " adapters)");
        }
    }
}

void parse_request_lora_fields(const json & body, size_t n_server_adapters,
                               std::map<int, float> & lora, std::string & lora_adapter, float & lora_adapter_scale) {
    if (body.contains("lora")) {
        const auto & l = body.at("lora");
        validate_lora_request(l, n_server_adapters);
        lora = parse_lora_request(l);
    }
    if (body.contains("lora_adapter")) {
        const auto & v = body.at("lora_adapter");
        if (!v.is_string()) {
            throw std::runtime_error("Error: 'lora_adapter' must be a string");
        }
        lora_adapter = v.get<std::string>();
    }
    if (body.contains("lora_adapter_scale")) {
        const auto & v = body.at("lora_adapter_scale");
        if (!v.is_number()) {
            throw std::runtime_error("Error: 'lora_adapter_scale' must be a number");
        }
        lora_adapter_scale = v.get<float>();
    }
}

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2) {
    if (l1.size() != l2.size()) {
        return false;
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        // we don't check lora.path to reduce the time complexity
        if (l1[i].scale != l2[i].scale || l1[i].ptr != l2[i].ptr) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras) {
    std::vector<size_t> enabled_ids;
    for (size_t i = 0; i < loras.size(); ++i) {
        if (loras[i].scale > 0) {
            enabled_ids.push_back(i);
        }
    }
    return enabled_ids;
}

uint64_t lora_config_fingerprint(const std::vector<common_adapter_lora_info> & loras) {
    // FNV-1a over the 8 bytes of the pointer and the 4 bytes of the scale of each active adapter
    uint64_t h = 1469598103934665603ULL;
    bool any = false;
    auto mix = [&h](const void * data, size_t n) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    };
    for (const auto & lora : loras) {
        if (lora.scale == 0.0f) {
            continue;
        }
        const uint64_t ptr = (uint64_t) (uintptr_t) lora.ptr;
        const float scale = lora.scale;
        mix(&ptr, sizeof(ptr));
        mix(&scale, sizeof(scale));
        any = true;
    }
    if (!any) {
        return 0;
    }
    return h == 0 ? 1 : h;
}

size_t lora_find_alora_invocation_start(const struct llama_adapter_lora * lora, const server_tokens & tokens) {
    const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
    const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);
    if (n_invocation_tokens == 0) {
        return tokens.size();
    }

    // scan backwards through the prompt tokens to find the last occurrence of the invocation sequence
    int match_idx = static_cast<int>(n_invocation_tokens) - 1;
    for (int i = (int) tokens.size() - 1; i >= 0; --i) {
        // the token in this position matches the next token to find in the invocation sequence
        if (tokens[i] == invocation_tokens[match_idx]) {
            // if it's a full match, we've found the start
            if (match_idx == 0) {
                return (size_t) i;
            }
            // otherwise, check the next token in the sequence
            --match_idx;
        } else {
            // no match in this position, so start looking over again
            match_idx = static_cast<int>(n_invocation_tokens) - 1;
        }
    }
    return tokens.size();
}

uint64_t lora_effective_fingerprint(std::vector<common_adapter_lora_info> loras, const server_tokens & tokens) {
    if (lora_all_alora(loras)) {
        const auto enabled_ids = lora_get_enabled_ids(loras);
        if (enabled_ids.size() == 1 && lora_find_alora_invocation_start(loras[enabled_ids[0]].ptr, tokens) == tokens.size()) {
            loras[enabled_ids[0]].scale = 0.0f; // invocation sequence absent: launch runs this request without the adapter
        }
    }
    return lora_config_fingerprint(loras);
}

uint64_t control_vector_fingerprint(const float * data, size_t n, int32_t n_embd, int32_t layer_start, int32_t layer_end) {
    if (data == nullptr || n == 0) {
        return 0;
    }
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void * p_, size_t len) {
        const uint8_t * p = static_cast<const uint8_t *>(p_);
        for (size_t i = 0; i < len; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    };
    mix(&n_embd,      sizeof(n_embd));
    mix(&layer_start, sizeof(layer_start));
    mix(&layer_end,   sizeof(layer_end));
    mix(data,         n * sizeof(float));
    return h == 0 ? 1 : h;
}

uint64_t kv_config_fingerprint(uint64_t lora_fp, uint64_t cv_fp) {
    if (cv_fp == 0) {
        return lora_fp;
    }
    const uint64_t h = (lora_fp ^ 0x9E3779B97F4A7C15ULL) * 1099511628211ULL ^ cv_fp;
    return h == 0 ? 1 : h;
}

uint64_t kv_namespace_fingerprint(uint64_t fp, const std::string & cache_salt, const std::string & extra_key) {
    if (cache_salt.empty() && extra_key.empty()) {
        return fp;
    }
    uint64_t h = (fp ^ 0xC2B2AE3D27D4EB4FULL) * 1099511628211ULL;
    const auto mix_str = [&](const std::string & str, uint8_t tag) {
        // length-prefixed with a tag so ("ab","") and ("a","b") and (salt "x", key "") vs (salt "", key "x") all differ
        h ^= tag;
        h *= 1099511628211ULL;
        const uint64_t n = str.size();
        for (int i = 0; i < 8; i++) {
            h ^= (n >> (8 * i)) & 0xFF;
            h *= 1099511628211ULL;
        }
        for (unsigned char c : str) {
            h ^= c;
            h *= 1099511628211ULL;
        }
    };
    mix_str(cache_salt, 1);
    mix_str(extra_key,  2);
    return h == 0 ? 1 : h;
}

//
// base64 utils (TODO: use the base64::decode from base64.hpp)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline raw_buffer base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    raw_buffer ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// server_tokens implementation
//

namespace {

constexpr uint32_t SERVER_TOKENS_STATE_VERSION = 1;

uint32_t server_tokens_state_u32(size_t value) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Server tokens state is too large");
    }
    return value;
}

class server_tokens_state_writer {
public:
    template <typename T>
    void write(T value) {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        const auto * ptr = reinterpret_cast<const char *>(&value);
        data.insert(data.end(), ptr, ptr + sizeof(value));
    }

    template <typename T>
    void write(const std::vector<T> & values) {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        write(server_tokens_state_u32(values.size()));
        if (values.empty()) {
            return;
        }
        const auto * ptr = reinterpret_cast<const char *>(values.data());
        data.insert(data.end(), ptr, ptr + values.size() * sizeof(T));
    }

    void write_media_chunk(const mtmd_input_chunk * chunk) {
        size_t chunk_size = 0;
        if (mtmd_input_chunk_save(chunk, nullptr, 0, &chunk_size) != 0 || chunk_size == 0) {
            throw std::runtime_error("Cannot serialize media chunk in server tokens");
        }
        std::vector<char> chunk_data(server_tokens_state_u32(chunk_size));
        if (mtmd_input_chunk_save(chunk, chunk_data.data(), chunk_data.size(), nullptr) != 0) {
            throw std::runtime_error("Cannot serialize media chunk in server tokens");
        }
        write(chunk_data);
    }

    std::vector<char> take() {
        data.resize((data.size() + sizeof(llama_token) - 1) / sizeof(llama_token) * sizeof(llama_token), 0);
        return std::move(data);
    }

private:
    std::vector<char> data;
};

class server_tokens_state_reader {
public:
    server_tokens_state_reader(const char * data, size_t size) : data(data), size(size) {}

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        if (size - pos < sizeof(T)) {
            throw std::runtime_error("Unexpected end of server tokens state");
        }
        T value;
        std::memcpy(&value, data + pos, sizeof(value));
        pos += sizeof(value);
        return value;
    }

    template <typename T>
    std::vector<T> read_vector() {
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
        const uint32_t n_values = read<uint32_t>();
        // reject before resizing, so that a small corrupted payload cannot request a huge allocation
        if (n_values > remaining() / sizeof(T)) {
            throw std::runtime_error("Unexpected end of server tokens state");
        }
        std::vector<T> values(n_values);
        if (n_values > 0) {
            std::memcpy(values.data(), data + pos, values.size() * sizeof(T));
            pos += values.size() * sizeof(T);
        }
        return values;
    }

    size_t remaining() const {
        return size - pos;
    }

private:
    const char * data;
    size_t size;
    size_t pos = 0;
};

} // namespace

server_tokens::server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd) : has_mtmd(has_mtmd) {
    for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
        push_back(mtmd_chunks[i]);
    }
}

server_tokens::server_tokens(const llama_tokens & tokens, bool has_mtmd) : has_mtmd(has_mtmd), tokens(tokens) {
}

llama_pos server_tokens::pos_next(int64_t n_tokens) const {
    if (!has_mtmd) {
        if (n_tokens < 0) {
            return tokens.size();
        }

        return n_tokens;
    }

    if (n_tokens < 0) {
        llama_pos res = tokens.size();

        for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
            const auto & chunk = it->second;
            res += mtmd_input_chunk_get_n_pos(chunk.get()) - mtmd_input_chunk_get_n_tokens(chunk.get());
        }

        return res;
    }

    int64_t idx = 0;
    llama_pos pos = 0;

    GGML_ASSERT(n_tokens <= (int64_t)tokens.size());

    while (idx < n_tokens) {
        const auto media_it = map_idx_to_media.find(idx);
        if (media_it != map_idx_to_media.end()) {
            const auto & chunk = media_it->second;
            const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());

            pos += n_pos;
            idx += n_tok;
        } else {
            pos++;
            idx++;
        }
    }

    return pos;
}

size_t server_tokens::size_up_to_pos(llama_pos max_pos) const {
    if (!has_mtmd) {
        return std::min((size_t)max_pos, tokens.size());
    }

    size_t idx = 0;
    llama_pos pos = 0;

    while (idx < tokens.size()) {
        const auto media_it = map_idx_to_media.find(idx);
        if (media_it != map_idx_to_media.end()) {
            const auto & chunk = media_it->second;
            const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());

            pos += n_pos;
            idx += n_tok;
        } else {
            pos++;
            idx++;
        }

        if (pos >= max_pos) {
            break;
        }
    }

    return idx;
}

std::string server_tokens::str() const {
    std::ostringstream oss;
    oss << "tokens: ";
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        llama_token t = tokens[idx];
        oss << "idx:" << idx << " ";
        if (t == LLAMA_TOKEN_NULL) {
            oss << "<embd> ";
        } else {
            oss << t << " ";
        }
    }
    oss << "\n";
    oss << "image idx: ";
    for (const auto & it : map_idx_to_media) {
        oss << it.first << ", ";
    }
    return oss.str();
}

const mtmd::input_chunk_ptr & server_tokens::find_chunk(size_t idx) const {
    auto it = map_idx_to_media.find(idx);
    if (it != map_idx_to_media.end()) {
        return it->second;
    }
    throw std::runtime_error("Chunk not found");
}

std::pair<const mtmd::input_chunk_ptr *, size_t> server_tokens::find_next_media_chunk(size_t idx) const {
    auto it = map_idx_to_media.upper_bound(idx);
    if (it != map_idx_to_media.end()) {
        return { &it->second, it->first };
    }
    return { nullptr, 0 };
}

void server_tokens::push_back(llama_token tok) {
    if (tok == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("Invalid token");
    }
    tokens.emplace_back(tok);
}

void server_tokens::push_back(const mtmd_input_chunk * chunk) {
    auto type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        GGML_ASSERT(has_mtmd);
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        size_t start_idx = tokens.size();
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.emplace_back(LLAMA_TOKEN_NULL);
        }
        mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
        map_idx_to_media[start_idx] = std::move(new_chunk);
    } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            push_back(text_tokens[i]);
        }
    } else {
        GGML_ABORT("Invalid chunk type");
    }
}

void server_tokens::push_back_placeholder(const mtmd_input_chunk * chunk) {
    auto type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        GGML_ASSERT(has_mtmd);
        mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_get_placeholder(chunk));
        GGML_ASSERT(new_chunk != nullptr && "failed to create placeholder chunk");
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        size_t start_idx = tokens.size();
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.emplace_back(LLAMA_TOKEN_NULL);
        }
        map_idx_to_media[start_idx] = std::move(new_chunk);
    } else {
        push_back(chunk);
    }
}

void server_tokens::push_back(server_tokens & tokens) {
    size_t start_idx = size();
    for (size_t i = 0; i < tokens.size(); i++) {
        push_back(tokens[i]);
    }
    if (tokens.has_mtmd) {
        // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
        // We could also just check, but this will prevent silently dropping MTMD data.
        GGML_ASSERT(has_mtmd);
        for (auto it = tokens.map_idx_to_media.begin(); it != tokens.map_idx_to_media.end(); ) {
            auto * chunk = tokens.map_idx_to_media[it->first].get();
            mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
            map_idx_to_media[start_idx + it->first] = std::move(new_chunk);
        }
    }
}

void server_tokens::insert(const llama_tokens & inp_tokens) {
    tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
}

const llama_tokens & server_tokens::get_tokens() const {
    GGML_ASSERT(!has_mtmd);
    return tokens;
}

std::vector<char> server_tokens::serialize() const {
    static_assert(sizeof(llama_token) == sizeof(uint32_t), "unexpected llama_token size");

    server_tokens_state_writer writer;
    writer.write((llama_token) LLAMA_TOKEN_NULL);
    writer.write(SERVER_TOKENS_STATE_VERSION);
    writer.write(tokens);

    std::vector<uint32_t> media_keys;
    media_keys.reserve(map_idx_to_media.size());
    for (const auto & item : map_idx_to_media) {
        media_keys.push_back(server_tokens_state_u32(item.first));
    }
    writer.write(media_keys);

    for (const auto & item : map_idx_to_media) {
        writer.write_media_chunk(item.second.get());
    }

    return writer.take();
}

server_tokens server_tokens::deserialize(const llama_tokens & packed, bool has_mtmd) {
    static_assert(sizeof(llama_token) == sizeof(uint32_t), "unexpected llama_token size");

    if (packed.empty() || packed[0] != LLAMA_TOKEN_NULL) {
        // plain token list, as written by older versions
        return server_tokens(packed, has_mtmd);
    }

    server_tokens_state_reader reader(reinterpret_cast<const char *>(packed.data()), packed.size() * sizeof(llama_token));
    reader.read<llama_token>(); // format marker
    if (reader.read<uint32_t>() != SERVER_TOKENS_STATE_VERSION) {
        throw std::runtime_error("Unsupported server tokens state version");
    }

    const llama_tokens tokens = reader.read_vector<llama_token>();

    // the media start indices, followed by the media chunks in the same order
    const std::vector<uint32_t> media_keys = reader.read_vector<uint32_t>();
    if (!media_keys.empty() && !has_mtmd) {
        throw std::runtime_error("Cannot restore media tokens without an mmproj");
    }

    server_tokens result(tokens, has_mtmd);

    for (const uint32_t key : media_keys) {
        const size_t start_idx = key;
        const std::vector<char> chunk_data = reader.read_vector<char>();
        if (chunk_data.empty()) {
            throw std::runtime_error("Cannot load media chunk from server tokens state");
        }

        mtmd::input_chunk_ptr chunk(mtmd_input_chunk_load(chunk_data.data(), chunk_data.size()));
        if (!chunk) {
            throw std::runtime_error("Cannot load media chunk from server tokens state");
        }
        result.map_idx_to_media[start_idx] = std::move(chunk);
    }

    if (reader.remaining() >= sizeof(llama_token)) {
        throw std::runtime_error("Trailing data in server tokens state");
    }

    return result;
}

llama_tokens server_tokens::get_text_tokens() const {
    llama_tokens res;
    res.reserve(tokens.size());
    for (llama_token t : tokens) {
        if (t != LLAMA_TOKEN_NULL) {
            res.push_back(t);
        }
    }
    return res;
}

void server_tokens::set_token(llama_pos pos, llama_token id) {
    GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
    tokens[pos] = id;
}

void server_tokens::keep_first(size_t n) {
    GGML_ASSERT(n <= tokens.size());
    if (has_mtmd) {
        if (n == tokens.size()) {
            return; // nothing to do
        }
        // we throw an error if we try to remove a token in the middle of an image
        // for ex. with input of 5 text tokens and 2 images:
        //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
        // n  1   2   3   4   5   6      7      8      9      10
        // allowed to resize      ^                    ^
        // disallowed to resize          ^      ^             ^
        if (n > 0) {
            // make sure we never remove tokens in the middle of an image
            // note that the case where we keep a full image at the end is allowed:
            //   tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] != LLAMA_TOKEN_NULL
            if (tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] == LLAMA_TOKEN_NULL) {
                find_chunk(n); // will throw an error if the cut is not at a chunk boundary
            }
        }
        // remove all image chunks that are not used anymore
        for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ) {
            size_t idx = it->first;
            if (idx >= n) {
                it = map_idx_to_media.erase(it);
            } else {
                ++it;
            }
        }
    }
    tokens.resize(n);
}

std::string server_tokens::detokenize(const llama_context * ctx, bool special) const {
    llama_tokens text_tokens;
    text_tokens.reserve(tokens.size());
    for (const auto & t : tokens) {
        if (t != LLAMA_TOKEN_NULL) {
            text_tokens.push_back(t);
        }
    }
    return common_detokenize(ctx, text_tokens, special);
}

size_t server_tokens::get_common_prefix(const server_tokens & b) const {
    const size_t max_idx = std::min(tokens.size(), b.tokens.size());

    if (!has_mtmd) {
        for (size_t i = 0; i < max_idx; ++i) {
            if (tokens[i] == b.tokens[i]) {
                continue;
            }

            return i;
        }

        return max_idx;
    }

    for (size_t i = 0; i < max_idx; ++i) {
        const llama_token ai =   tokens[i];
        const llama_token bi = b.tokens[i];

        if (ai == LLAMA_TOKEN_NULL && bi == LLAMA_TOKEN_NULL) {
            const auto & a_chunk =   find_chunk(i);
            const auto & b_chunk = b.find_chunk(i);

            GGML_ASSERT(a_chunk && b_chunk);

            const std::string id_ai = mtmd_input_chunk_get_id(a_chunk.get());
            const std::string id_bi = mtmd_input_chunk_get_id(b_chunk.get());

            const size_t n_tok_a = mtmd_input_chunk_get_n_tokens(a_chunk.get());
            const size_t n_tok_b = mtmd_input_chunk_get_n_tokens(b_chunk.get());

            if (id_ai == id_bi && n_tok_a == n_tok_b) {
                GGML_ASSERT(n_tok_a > 0 && "Invalid media chunk"); // should never happen
                i += n_tok_a - 1; // will be +1 by the for loop
                continue;
            }

            return i;
        }

        if (ai == bi) {
            continue;
        }

        return i;
    }

    return max_idx; // all tokens are equal
}

common_chat_msg_spans server_tokens::find_message_spans(const common_chat_msg_delimiters & delims) const {
    std::map<size_t, size_t> skips;
    for (const auto & it : map_idx_to_media) {
        skips[it.first] = mtmd_input_chunk_get_n_tokens(it.second.get());
    }
    return delims.split(tokens, skips);
}

bool server_tokens::validate(const struct llama_context * ctx) const {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    size_t n_media = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto & t = tokens[i];
        if (t == LLAMA_TOKEN_NULL) {
            try {
                const auto & chunk = find_chunk(i);
                if (mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    return false;
                }
                const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk.get());
                const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
                if (n_tokens == 0 || n_pos <= 0 || n_tokens > tokens.size() - i) {
                    return false;
                }
                for (size_t j = i; j < i + n_tokens; ++j) {
                    if (tokens[j] != LLAMA_TOKEN_NULL) {
                        return false;
                    }
                }
                ++n_media;
                i += n_tokens - 1;
            } catch (const std::exception & e) {
                return false;
            }
        } else if (t < 0 || t >= n_vocab) {
            return false;
        }
    }
    return n_media == map_idx_to_media.size();
}

server_tokens server_tokens::clone() const {
    server_tokens res;
    res.has_mtmd = has_mtmd;
    res.tokens   = tokens;
    for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
        size_t idx = it->first;
        const mtmd::input_chunk_ptr & chunk = it->second;
        res.map_idx_to_media[idx] = mtmd::input_chunk_ptr(mtmd_input_chunk_copy(chunk.get()));
    }
    return res;
}

// SOURCE-READ: examples/server/server-common.cpp:1512-1596 @ 35f0cac75 (the fork's own MIT code,
// adapted to this class's detokenize()/clone() and the private tokens/map_idx_to_media members --
// row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 2).
// A token is treated as "inside" a reasoning span using the fork's own boundary rule: walk the
// tokens accumulating the byte length of their detokenized text; a token is skipped while that
// running length has crossed into (start, end] of an excluded range, and kept once it crosses past
// `end` (which also advances to the next range) -- token boundaries rarely align with tag text
// boundaries, so this is an approximation, not exact, matching the fork's own accepted trade-off.
// Unlike the fork's O(n^2) all-pairs search for matching <begin>...<end> spans, this pairs each
// begin with the next end that follows it and resumes searching after that end -- equivalent for
// well-formed non-overlapping tags and does not attempt to pair an unterminated trailing tag (that
// text is left in, not silently swallowed by a later unrelated end tag).
server_tokens server_tokens::get_tokens_exclude_think(const llama_context * ctx, const thinking_tokens & think_token) const {
    if (!think_token.exclude) {
        return clone();
    }
    GGML_ASSERT(!think_token.begin.empty() && !think_token.end.empty() && "think tokens cannot be empty");

    // Build full_text by concatenating the SAME per-token piece() calls used below to walk byte
    // offsets, rather than calling detokenize() separately. Coordinator review (2026-09-27) caught
    // that those two would disagree on an SPM vocab: detokenize() can special-case the leading
    // token's whitespace differently than a token-by-token common_token_to_piece() walk, silently
    // shifting every excluded range and making the exclusion a no-op (measured: raw and excluded
    // comparisons picked the same slot). One coordinate system for both removes the mismatch.
    std::vector<std::string> pieces;
    pieces.reserve(tokens.size());
    std::string full_text;
    for (const auto & tok : tokens) {
        pieces.emplace_back(tok != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, tok, true) : std::string());
        full_text += pieces.back();
    }

    // non-overlapping [start, end) byte ranges (in full_text) to exclude, in ascending order
    std::vector<std::pair<size_t, size_t>> excl_ranges;
    size_t search_from = 0;
    while (true) {
        const size_t start = full_text.find(think_token.begin, search_from);
        if (start == std::string::npos) {
            break;
        }
        const size_t end_tag = full_text.find(think_token.end, start + think_token.begin.size());
        if (end_tag == std::string::npos) {
            // unterminated reasoning span at the tail -- leave it in place rather than risk pairing
            // it with an unrelated end tag further away
            break;
        }
        const size_t end = end_tag + think_token.end.size();
        excl_ranges.emplace_back(start, end);
        search_from = end;
    }

    if (excl_ranges.empty()) {
        return clone();
    }

    server_tokens res;
    res.has_mtmd = has_mtmd;

    size_t range_idx = 0;
    size_t byte_len  = 0; // cumulative piece length after including the current token
    for (size_t n = 0; n < tokens.size(); ++n) {
        byte_len += pieces[n].size();

        bool keep = true;
        if (range_idx < excl_ranges.size()) {
            if (byte_len <= excl_ranges[range_idx].first) {
                keep = true;
            } else if (byte_len <= excl_ranges[range_idx].second) {
                keep = false;
            } else {
                keep = true;
                range_idx++;
            }
        }

        if (keep) {
            res.tokens.push_back(tokens[n]);
            auto it = map_idx_to_media.find(n);
            if (it != map_idx_to_media.end()) {
                res.map_idx_to_media[res.tokens.size() - 1] = mtmd::input_chunk_ptr(mtmd_input_chunk_copy(it->second.get()));
            }
        }
    }
    return res;
}

//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

bool json_is_array_and_contains_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (e.is_number_integer()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

server_tokens process_mtmd_prompt(
        mtmd_context * mctx,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const mtmd_helper_init_opt & init_opt,
        bool is_placeholder) {
    // these will be freed upon going out of scope
    mtmd::bitmaps bitmaps;
    std::vector<mtmd_helper::video_ptr> videos;
    // One request, one decode budget: the pixel counts the image headers DECLARE are summed against the single-image
    // limit before anything is decoded, so many medium images cannot add up to what one oversized image is refused for.
    // Row S69-ENGREV-MLXENGINE-IMAGE-PIXEL-CAP-2026-10-27-01 (mlx-engine server/chat.py:163-165, running total).
    {
        const uint64_t px_cap = mtmd_helper_image_max_decode_pixels();
        uint64_t px_total = 0;
        for (auto & file : files) {
            px_total += mtmd_helper_image_declared_pixels(file.data(), file.size());
            if (px_cap > 0 && px_total > px_cap) {
                throw std::runtime_error(string_format(
                    "Images in this request declare %llu pixels in total, above the decode limit of %llu "
                    "(LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS raises it)",
                    (unsigned long long) px_total, (unsigned long long) px_cap));
            }
        }
    }
    for (auto & file : files) {
        auto out = mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size(), is_placeholder, init_opt);
        if (!out.bitmap) {
            throw std::runtime_error("Failed to load image or audio file");
        }
        bitmaps.entries.emplace_back(out.bitmap);
        if (out.video_ctx) {
            videos.emplace_back(out.video_ctx);
        }
    }
    // process prompt
    std::vector<server_tokens> inputs;
    // multimodal
    mtmd_input_text inp_txt = {
        prompt.data(),
        prompt.size(),
        /* add_special */   true,
        /* parse_special */ true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t tokenized = mtmd_tokenize(mctx,
                                      chunks.ptr.get(),
                                      &inp_txt,
                                      bitmaps_c_ptr.data(),
                                      bitmaps_c_ptr.size());
    if (tokenized != 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    auto result = server_tokens(chunks, true);
    return result;
}

/**
 * tokenize a single input "prompt" object
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
// A prompt text that already starts with the BOS string (a client that pre-rendered a chat template, e.g. "<bos>..." or
// "<s>...") gets a second BOS from add_special, and the vocab only logs a warning. Drop the duplicate so the model sees one.
// Only prompts given as TEXT come through here: /tokenize and token-id array prompts are untouched, and the latter is the
// way to send two BOS on purpose.
// SOURCE-READ: ollama/ollama a8aaf9fcfad23dc8d6f07b8109b1ba5ed310d76c llm/llama_server.go:246-280 (MIT, fetched in full to
// tmp/_tca_ollama_llama_server_0930.go): completionPrompt strips a leading BOS string from a raw prompt when
// the tokenizer adds BOS. Here the same outcome is reached after tokenizing, so it does not depend on the BOS spelling. No code copied.
static void drop_duplicate_leading_bos(const llama_vocab * vocab, llama_tokens & toks, bool add_special) {
    if (!add_special || !llama_vocab_get_add_bos(vocab)) {
        return;
    }
    const llama_token bos = llama_vocab_bos(vocab);
    if (bos != LLAMA_TOKEN_NULL && toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
}

server_tokens tokenize_input_subprompt(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special, const mtmd_helper_init_opt & init_opt) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";
    const bool has_mtmd = mctx != nullptr;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        llama_tokens tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        if (json_prompt.is_string()) {
            drop_duplicate_leading_bos(vocab, tmp, add_special);
        }
        return server_tokens(tmp, false);
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        llama_tokens tmp = json_prompt.get<llama_tokens>();
        return server_tokens(tmp, false);
    } else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        if (json_prompt.contains(JSON_MTMD_DATA_KEY)) {
            if (!has_mtmd)
                throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");

            // JSON object with prompt and multimodal key.
            std::vector<raw_buffer> files;
            for (const auto & entry : json_prompt.at(JSON_MTMD_DATA_KEY)) {
                files.push_back(base64_decode(entry));
            }
            return process_mtmd_prompt(mctx, json_prompt.at(JSON_STRING_PROMPT_KEY), files, init_opt);
        } else {
            // Not multimodal, but contains a subobject.
            llama_tokens tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
            drop_duplicate_leading_bos(vocab, tmp, add_special);
            return server_tokens(tmp, false);
        }
   } else {
       throw std::invalid_argument("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
   }
}

std::vector<server_tokens> tokenize_input_prompts(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special, const mtmd_helper_init_opt & init_opt) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, mctx, p, add_special, parse_special, init_opt));
        }
    } else {
        result.push_back(tokenize_input_subprompt(vocab, mctx, json_prompt, add_special, parse_special, init_opt));
    }
    if (result.empty()) {
        throw std::invalid_argument("\"prompt\" must not be empty");
    }
    return result;
}

//
// OAI utils
//

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body) {
    json llama_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        // "stop_sequence" is Kobold's / TabbyAPI's name for it (server-schema.cpp alias); used only when "stop" is absent
        llama_params["stop"] = json_value(body, "stop", json_value(body, "stop_sequence", json::array()));
    }

    // "echo" is handled by the request schema (task_params::echo): the prompt is put in front of the text, and with logprobs the
    // prompt tokens are scored. It cannot be streamed.
    if (json_value(body, "echo", false) && json_value(body, "stream", false)) {
        throw std::runtime_error("echo does not support stream");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "best_of", "suffix" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

// url can be
// - http(s):// for remote files
// - file:// for local files (only allowed if media_path is set)
// - data: for base64 encoded data with uri scheme (e.g. data:image/png;base64,...)
// - raw base64 encoded data
void handle_media(
        std::vector<raw_buffer> & out_files,
        const std::string & url,
        const std::string & media_path) {
    if (!media_path.empty()) {
        // should already be enforced by arg.cpp, but checking just in case
        GGML_ASSERT(media_path.back() == DIRECTORY_SEPARATOR);
    }

    if (string_starts_with(url, "http")) {
        // download remote image
        // TODO @ngxson : maybe make these params configurable
        common_remote_params params;
        params.max_size = 1024 * 1024 * 10; // 10MB
        params.timeout  = 10; // seconds
        // SSRF: a client-supplied URL must not make the server reach its own loopback, the LAN or the cloud
        // metadata address; LLAMA_SERVER_MEDIA_ALLOW_PRIVATE=1 restores the old behaviour for trusted setups
        {
            const char * allow = std::getenv("LLAMA_SERVER_MEDIA_ALLOW_PRIVATE");
            params.block_private = !(allow != nullptr && std::string(allow) == "1");
        }
        SRV_INF("downloading image from '%s'\n", url.c_str());
        auto res = common_remote_get_content(url, params);
        if (200 <= res.first && res.first < 300) {
            SRV_INF("downloaded %zu bytes\n", res.second.size());
            raw_buffer data;
            data.insert(data.end(), res.second.begin(), res.second.end());
            out_files.push_back(data);
        } else {
            throw std::runtime_error("Failed to download image");
        }

    } else if (string_starts_with(url, "file://")) {
        if (media_path.empty()) {
            throw std::invalid_argument("file:// URLs are not allowed unless --media-path is specified");
        }
        // load local image file
        std::string file_path = url.substr(7); // remove "file://"
        raw_buffer data;
        if (!fs_validate_filename(file_path, true)) {
            throw std::invalid_argument("file path is not allowed: " + file_path);
        }
        SRV_INF("loading image from local file '%s'\n", (media_path + file_path).c_str());
        std::ifstream file(media_path + file_path, std::ios::binary);
        if (!file) {
            throw std::invalid_argument("file does not exist or cannot be opened: " + file_path);
        }
        data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        out_files.push_back(data);

    } else if (string_starts_with(url, "data:")) {
        // try to decode base64 image, video, or audio
        std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
        if (parts.size() != 2) {
            throw std::invalid_argument("Invalid uri-encoded base64 value");
        } else if (!string_starts_with(parts[0], "data:image/")
                && !string_starts_with(parts[0], "data:video/")
                && !string_starts_with(parts[0], "data:audio/")) {
            throw std::invalid_argument("Invalid uri format: " + parts[0]);
        } else if (!string_ends_with(parts[0], "base64")) {
            throw std::invalid_argument("uri must be base64 encoded");
        } else {
            auto base64_data = parts[1];
            auto decoded_data = base64_decode(base64_data);
            out_files.push_back(decoded_data);
        }

    } else {
        // try as raw base64 string
        auto decoded_data = base64_decode(url);
        if (decoded_data.empty()) {
            throw std::runtime_error("Invalid base64 value");
        }
        out_files.push_back(decoded_data);
    }
}
// load media files from an OAI content array, then replace each media part with a media marker text part
static void oaicompat_content_load_media(json & content, const server_chat_params & opt, std::vector<raw_buffer> & out_files) {
    for (auto & p : content) {
        std::string type = json_value(p, "type", std::string());
        if (type == "image_url") {
            if (!opt.allow_image) {
                throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
            }

            json image_url = json_value(p, "image_url", json::object());
            std::string url = json_value(image_url, "url", std::string());
            handle_media(out_files, url, opt.media_path);

            p["type"] = "media_marker";
            p["text"] = get_media_marker();
            p.erase("image_url");

        } else if (type == "input_audio") {
            if (!opt.allow_audio) {
                throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
            }

            // note: don't need to validate "format", it's redundant
            json input_audio = json_value(p, "input_audio", json::object());
            std::string url  = json_value(input_audio, "data",
                                    json_value(input_audio, "url", std::string()));
            handle_media(out_files, url, opt.media_path);

            p["type"] = "media_marker";
            p["text"] = get_media_marker();
            p.erase("input_audio");

        } else if (type == "input_video" || type == "video_url") {
            if (!opt.allow_video) {
                throw std::runtime_error("video input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
            }

            // accept the OpenAI-style "video_url" key as an alias of "input_video"
            json input_video = json_value(p, type, json::object());
            std::string url  = json_value(input_video, "data",
                                    json_value(input_video, "url", std::string()));
            handle_media(out_files, url, opt.media_path);

            p["type"] = "media_marker";
            p["text"] = get_media_marker();
            p.erase("input_video");
            p.erase("video_url");

        } else if (type != "text") {
            throw std::invalid_argument("unsupported content[].type");
        }
    }
}

server_tokens tokenize_oai_content_array(const llama_vocab * vocab, mtmd_context * mctx, const server_chat_params & opt, json content, bool add_special, bool parse_special, const mtmd_helper_init_opt & init_opt) {
    if (!content.is_array()) {
        throw std::invalid_argument("\"content\" must be an array");
    }

    std::vector<raw_buffer> files;
    oaicompat_content_load_media(content, opt, files);

    std::string prompt;
    for (const auto & p : content) {
        prompt += json_value(p, "text", std::string());
    }

    if (files.empty()) {
        return server_tokens(common_tokenize(vocab, prompt, add_special, parse_special), false);
    }

    return process_mtmd_prompt(mctx, prompt, files, init_opt);
}

//
// Control-token text inside message text (see server-common.h)
//

static void plain_utf8_append(std::string & out, uint32_t cpt) {
    if (cpt < 0x80) {
        out += (char) cpt;
    } else if (cpt < 0x800) {
        out += (char) (0xC0 | (cpt >> 6));
        out += (char) (0x80 | (cpt & 0x3F));
    } else if (cpt < 0x10000) {
        out += (char) (0xE0 | (cpt >> 12));
        out += (char) (0x80 | ((cpt >> 6) & 0x3F));
        out += (char) (0x80 | (cpt & 0x3F));
    } else {
        out += (char) (0xF0 | (cpt >> 18));
        out += (char) (0x80 | ((cpt >> 12) & 0x3F));
        out += (char) (0x80 | ((cpt >> 6) & 0x3F));
        out += (char) (0x80 | (cpt & 0x3F));
    }
}

// Decode the code point at s[i] and advance i past it. A malformed byte is returned as itself (>= 0x110000 marks it) and
// advances by one, so a restore pass copies stray bytes through unchanged.
static uint32_t plain_utf8_next(const std::string & s, size_t & i) {
    const unsigned char c = (unsigned char) s[i];
    size_t len = 1;
    uint32_t cpt = c;
    if      ((c & 0xE0) == 0xC0) { len = 2; cpt = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; cpt = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; cpt = c & 0x07; }
    else if (c >= 0x80)          { i += 1; return 0x110000 + c; }
    if (i + len > s.size()) { i += 1; return 0x110000 + c; }
    for (size_t k = 1; k < len; ++k) {
        const unsigned char cc = (unsigned char) s[i + k];
        if ((cc & 0xC0) != 0x80) { i += 1; return 0x110000 + c; }
        cpt = (cpt << 6) | (cc & 0x3F);
    }
    i += len;
    return cpt;
}

// Private-use blocks a run of placeholders can come from, in the order they are tried. The count is what the block really holds: a run
// longer than that would walk into ordinary characters (U+F900.. are CJK compatibility ideographs) or past the last valid code point.
struct plain_block_t { uint32_t base; uint32_t count; };
static const plain_block_t PLAIN_BLOCKS[] = {
    { 0xE000u,  0xF8FFu  - 0xE000u  + 1u }, // BMP private-use area, 6400
    { 0xF0000u, 0xFFFFDu - 0xF0000u + 1u }, // plane 15 private-use area, 65534 (U+FFFE and U+FFFF are noncharacters)
};

static size_t plain_literal_capacity() {
    size_t cap = 0;
    for (const auto & b : PLAIN_BLOCKS) {
        cap = std::max(cap, (size_t) b.count);
    }
    return cap;
}

std::vector<std::string> server_collect_plain_literals(const llama_vocab * vocab) {
    std::vector<std::string> lits;
    // BPE and SPM vocabs. SPM (Llama/Gemma family) was left out and the check returned EMPTY there, so a user message or a legal
    // document containing "<start_of_turn>model" tokenised to the real control token: a forged turn (row
    // S69-ENGREV-BEELLAMA-INPUT-PROVENANCE-SPM-GAP-2026-10-27-01). tokenize_with_plain_spans now strips the SPM dummy prefix.
    if (vocab == nullptr || (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_BPE && llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_SPM)) {
        return lits;
    }
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (int32_t t = 0; t < n; ++t) {
        const int attr = (int) llama_vocab_get_attr(vocab, t);
        if (!(attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED))) {
            continue;
        }
        const char * txt = llama_vocab_get_text(vocab, t);
        if (txt == nullptr) {
            continue;
        }
        std::string s(txt);
        // markup only: "<|im_end|>", "<think>", "<tool_call>", "[INST]". Plain-looking added tokens (whitespace runs, words)
        // are left alone so ordinary text is not carved up.
        if (s.size() < 3 || (s[0] != '<' && s[0] != '[')) {
            continue;
        }
        lits.push_back(std::move(s));
    }
    // longest first, so a literal contained in a longer one is not marked first
    std::stable_sort(lits.begin(), lits.end(), [](const std::string & a, const std::string & b) { return a.size() > b.size(); });
    lits.erase(std::unique(lits.begin(), lits.end()), lits.end());
    // Every literal gets one placeholder code point out of a private-use block. A literal left out would stay a real control token
    // inside message text (an injection hole), so too many literals is an error, never a truncation.
    if (lits.size() > plain_literal_capacity()) {
        throw std::length_error("the vocab has " + std::to_string(lits.size()) + " control-token literals, more than the " +
                                std::to_string(plain_literal_capacity()) + " placeholder code points the largest private-use block holds; "
                                "refusing to leave any of them un-neutralised");
    }
    return lits;
}

plain_spans_t plain_spans_from_json(const json & j, size_t prompt_size) {
    plain_spans_t spans;
    if (!j.is_array()) {
        return spans;
    }
    size_t prev_end = 0;
    for (const auto & e : j) {
        if (!e.is_array() || e.size() != 2 || !e[0].is_number_integer() || !e[1].is_number_integer()) {
            return {};
        }
        const int64_t b_raw  = e[0].get<int64_t>();
        const int64_t en_raw = e[1].get<int64_t>();
        if (b_raw < 0 || en_raw < 0) {
            return {};
        }
        const size_t b  = (size_t) b_raw;
        const size_t en = (size_t) en_raw;
        if (b >= en || b < prev_end || en > prompt_size) {
            return {}; // malformed: fall back to the legacy whole-string tokenisation rather than guess
        }
        spans.emplace_back(b, en);
        prev_end = en;
    }
    return spans;
}

llama_tokens tokenize_with_plain_spans(const llama_vocab * vocab, const std::string & prompt, const plain_spans_t & spans, bool add_special) {
    llama_tokens out;
    bool first = true;
    const bool spm = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;
    // SPM prepends a dummy-prefix space ("\xE2\x96\x81") to every common_tokenize call, but inside ONE tokenisation it only does so at
    // the start and right after a special token. A later part therefore carries a spurious space token run unless it is stripped:
    // tokenise "\n" + part and drop the tokens "\n" alone produces (a byte token, never merged with what follows).
    llama_tokens nl_tokens;
    if (spm) {
        nl_tokens = common_tokenize(vocab, "\n", false, false);
    }
    auto prev_is_special = [&]() {
        if (out.empty()) {
            return true;
        }
        return (llama_vocab_get_attr(vocab, out.back()) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) != 0;
    };
    auto add = [&](size_t b, size_t e, bool parse_special) {
        if (e <= b) {
            return;
        }
        llama_tokens t;
        if (spm && !first && !prev_is_special()) {
            t = common_tokenize(vocab, "\n" + prompt.substr(b, e - b), false, parse_special);
            if (t.size() >= nl_tokens.size() && std::equal(nl_tokens.begin(), nl_tokens.end(), t.begin())) {
                t.erase(t.begin(), t.begin() + (std::ptrdiff_t) nl_tokens.size());
            } else {
                t = common_tokenize(vocab, prompt.substr(b, e - b), false, parse_special); // no clean split: keep the prefix
            }
        } else {
            t = common_tokenize(vocab, prompt.substr(b, e - b), add_special && first, parse_special);
        }
        first = false;
        out.insert(out.end(), t.begin(), t.end());
    };
    size_t pos = 0;
    for (const auto & sp : spans) {
        add(pos, sp.first, true);
        add(sp.first, sp.second, false);
        pos = sp.second;
    }
    add(pos, prompt.size(), true);
    if (first) { // empty prompt: same result as the plain path
        out = common_tokenize(vocab, prompt, add_special, true);
    }
    drop_duplicate_leading_bos(vocab, out, add_special);
    return out;
}

// true when any string inside j holds a code point in [lo, hi)
static bool plain_json_has_cpt_in(const json & j, uint32_t lo, uint32_t hi) {
    if (j.is_string()) {
        const std::string s = j.get<std::string>();
        for (size_t i = 0; i < s.size();) {
            const uint32_t c = plain_utf8_next(s, i);
            if (c >= lo && c < hi) {
                return true;
            }
        }
        return false;
    }
    if (j.is_array() || j.is_object()) {
        for (const auto & e : j) {
            if (plain_json_has_cpt_in(e, lo, hi)) {
                return true;
            }
        }
    }
    return false;
}

uint32_t server_plain_literal_base(const json & body, size_t n_lits) {
    if (n_lits > plain_literal_capacity()) {
        throw std::length_error("no private-use block holds " + std::to_string(n_lits) + " placeholder code points");
    }
    for (const auto & blk : PLAIN_BLOCKS) {
        if (n_lits <= blk.count && !plain_json_has_cpt_in(body, blk.base, blk.base + (uint32_t) n_lits)) {
            return blk.base;
        }
    }
    throw std::invalid_argument("request text uses the private-use code points reserved for control-token neutralisation");
}

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    const llama_vocab * vocab,
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files,
    common_chat_session & out_session)
{
    json llama_params;

    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    // tool_choice is a string ("auto" / "none" / "required") OR OpenAI's named form
    // {"type": "function", "function": {"name": "X"}}, which forces that ONE function. Reading it with
    // json_value<std::string> turned the object into "auto" with only a log warning, so a forced tool was
    // silently not forced (upstream has the same gap; the Anthropic converter feeds this path too). The
    // chat-template enum has no "this one function" value, so express it exactly: offer only X, require a call.
    std::string tool_choice = "auto";
    if (body.contains("tool_choice") && body.at("tool_choice").is_object() &&
        json_value(body.at("tool_choice"), "type", std::string()) == "allowed_tools") {
        // OpenAI's {"type": "allowed_tools"} limits the model to a subset of `tools` without editing `tools` itself, so a
        // client keeps one tool list (and the prompt prefix it renders) while the allowed set changes per turn.
        // Chat Completions nests {"mode", "tools"} under "allowed_tools"; the Responses API, converted to this path by
        // server-chat.cpp, writes them flat. Each entry names a tool as {"type": "function", "function": {"name": X}}
        // (Chat Completions) or {"type": "function", "name": X} (Responses). mode "auto" lets the model answer in text,
        // "required" demands a call. Hosted Responses tools (web search, image generation, ...) are not served here and
        // are passed over, as they are in `tools`. Found in the LMDeploy review
        // (lmdeploy/serve/parsers/tool_parser/tool_parser.py:35-58, which filters the same way but never reads `mode`).
        const json & tc   = body.at("tool_choice");
        const json & spec = (tc.contains("allowed_tools") && tc.at("allowed_tools").is_object()) ? tc.at("allowed_tools") : tc;
        const std::string mode = json_value(spec, "mode", std::string());
        if (mode != "auto" && mode != "required") {
            throw std::invalid_argument("tool_choice allowed_tools needs \"mode\": \"auto\" or \"required\"");
        }
        if (!spec.contains("tools") || !spec.at("tools").is_array()) {
            throw std::invalid_argument("tool_choice allowed_tools needs a \"tools\" array");
        }
        std::vector<std::string> allowed;
        for (const auto & ref : spec.at("tools")) {
            if (!ref.is_object()) {
                throw std::invalid_argument("tool_choice allowed_tools entries must be objects");
            }
            const std::string ref_type = json_value(ref, "type", std::string("function"));
            if (ref_type != "function" && ref_type != "custom") {
                continue;
            }
            const std::string name = (ref.contains("function") && ref.at("function").is_object())
                                   ? json_value(ref.at("function"), "name", std::string())
                                   : json_value(ref, "name", std::string());
            if (name.empty()) {
                throw std::invalid_argument("tool_choice allowed_tools entry has no function name");
            }
            allowed.push_back(name);
        }
        if (allowed.empty()) {
            throw std::invalid_argument("tool_choice allowed_tools lists no function tool this server can call");
        }
        auto tool_name = [](const json & t) {
            return (t.contains("function") && t.at("function").is_object())
                 ? json_value(t.at("function"), "name", std::string()) : std::string();
        };
        for (const auto & name : allowed) {
            bool found = false;
            if (tools.is_array()) {
                for (const auto & t : tools) {
                    found = found || tool_name(t) == name;
                }
            }
            if (!found) {
                throw std::invalid_argument("tool_choice allowed_tools names function '" + name + "', which is not in tools");
            }
        }
        json kept = json::array();
        for (const auto & t : tools) {   // request order, not allowed-list order: the rendered prompt keeps its layout
            if (std::find(allowed.begin(), allowed.end(), tool_name(t)) != allowed.end()) {
                kept.push_back(t);
            }
        }
        tools       = kept;
        has_tools   = true;
        tool_choice = mode;
    } else if (body.contains("tool_choice") && body.at("tool_choice").is_object()) {
        const json & tc = body.at("tool_choice");
        const std::string tc_type = json_value(tc, "type", std::string());
        // Chat Completions nests the name ({"function": {"name": X}}); the Responses API, converted to this
        // path by server-chat.cpp, writes it flat ({"type": "function", "name": X}). Accept both.
        const std::string forced  = (tc.contains("function") && tc.at("function").is_object())
                                  ? json_value(tc.at("function"), "name", std::string())
                                  : json_value(tc, "name", std::string());
        if (tc_type != "function" || forced.empty()) {
            throw std::invalid_argument("tool_choice object must be {\"type\": \"function\", \"function\": {\"name\": ...}}"
                                        " (or the Responses API form {\"type\": \"function\", \"name\": ...})");
        }
        json kept = json::array();
        if (tools.is_array()) {
            for (const auto & t : tools) {
                if (t.contains("function") && t.at("function").is_object() &&
                    json_value(t.at("function"), "name", std::string()) == forced) {
                    kept.push_back(t);
                }
            }
        }
        if (kept.empty()) {
            throw std::invalid_argument("tool_choice names function '" + forced + "', which is not in tools");
        }
        tools       = kept;
        has_tools   = true;
        tool_choice = "required";
    } else {
        tool_choice = json_value(body, "tool_choice", std::string("auto"));
    }

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        // "stop_sequence": Kobold / TabbyAPI alias, used only when "stop" is absent (see server-schema.cpp)
        llama_params["stop"] = json_value(body, "stop", json_value(body, "stop_sequence", json::array()));
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        if (!response_format.is_object()) {
            throw std::invalid_argument("response_format must be an object");
        }
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            if (response_format.contains("schema") || json_schema.empty()) {
                json_schema = json_value(response_format, "schema", json::object());
            }
        } else if (response_type == "json_schema") {
            // SOURCE-READ: vllm/entrypoints/openai/chat_completion/protocol.py:761-792 (validate_response_format)
            if (!response_format.contains("json_schema") || !response_format.at("json_schema").is_object()) {
                throw std::invalid_argument("When response_format type is 'json_schema', the 'json_schema' field must be provided.");
            }
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::invalid_argument("response_format type must be one of \"text\", \"json_object\" or \"json_schema\", but got: " + response_type);
        }
    }

    // an absent or empty schema means any object
    if (json_schema.is_object() && json_schema.empty()) {
        json_schema["type"] = "object";
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::invalid_argument("'messages' is required");
    }
    json & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::invalid_argument("Expected 'messages' to be an array");
    }
    // server-wide default system prompt (--system-prompt / --system-prompt-file / fork -spf): only when the
    // request brings no system message of its own -- the request always wins.
    if (!opt.default_system_prompt.empty()) {
        bool has_system = false;
        for (const auto & msg : messages) {
            if (json_value(msg, "role", std::string()) == "system") {
                has_system = true;
                break;
            }
        }
        if (!has_system) {
            // common_json has no positional insert (MSVC C2660, measured 2026-09-27): build the new array
            json with_system = json::array();
            with_system.push_back({{"role", "system"}, {"content", opt.default_system_prompt}});
            with_system.insert(messages);
            messages = with_system;
        }
    }
    for (auto & msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::invalid_argument("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json & content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::invalid_argument("Expected 'content' to be a string or an array");
        }

        oaicompat_content_load_media(content, opt, out_files);
    }

    // TGI `tool_prompt` (row S69-ENGREV-TGI-TOOL-PROMPT-FIELD-2026-10-27-01): a per-request instruction text that comes with the
    // tools. TGI (router/src/lib.rs ChatRequest.tool_prompt, infer/chat_template.rs apply) appends "\n---\n" + the text to the
    // last message when tools are offered and tool_choice is not "none". Spec read only, no code copied. Unlike TGI there is no
    // built-in default text here: the request must name one, so requests without the field render exactly as before.
    if (has_tools && tool_choice != "none" && body.contains("tool_prompt") && !body.at("tool_prompt").is_null()) {
        if (!body.at("tool_prompt").is_string()) {
            throw std::invalid_argument("tool_prompt must be a string");
        }
        const std::string tool_prompt = body.at("tool_prompt").get<std::string>();
        if (!tool_prompt.empty() && !messages.empty() && messages.back().is_object()) {
            json & last = messages.back();
            const std::string extra = "\n---\n" + tool_prompt;
            // a last turn without content (an assistant turn that only carries tool_calls) is left alone, as TGI only extends
            // messages that have a content body
            if (!last.contains("content") || last.at("content").is_null()) {
                // nothing to extend
            } else if (last.at("content").is_string()) {
                last["content"] = last.at("content").get<std::string>() + extra;
            } else if (last.at("content").is_array()) {
                last.at("content").push_back({{"type", "text"}, {"text", extra}});
            }
        }
    }

    // Control-token text inside message text -> ordinary text. Each literal found in non-assistant message text is swapped for
    // one private-use placeholder code point here; the rendered prompt gets the literal back below and its span recorded.
    // Assistant text is the model's own earlier output (real tokens when it was generated) and is left as is.
    uint32_t plain_base = 0;
    bool     plain_on   = false;
    if (!opt.plain_literals.empty()) {
        const std::vector<std::string> & lits = opt.plain_literals;
        plain_base = server_plain_literal_base(body, lits.size());
        plain_on   = true;
        std::vector<std::vector<size_t>> by_first(256);
        for (size_t k = 0; k < lits.size(); ++k) {
            by_first[(unsigned char) lits[k][0]].push_back(k); // lits is longest first, so k order keeps that
        }
        auto mark = [&](const std::string & text) {
            std::string out;
            out.reserve(text.size());
            for (size_t i = 0; i < text.size();) {
                bool hit = false;
                for (const size_t k : by_first[(unsigned char) text[i]]) {
                    if (text.compare(i, lits[k].size(), lits[k]) == 0) {
                        plain_utf8_append(out, plain_base + (uint32_t) k);
                        i  += lits[k].size();
                        hit = true;
                        break;
                    }
                }
                if (!hit) {
                    out += text[i++];
                }
            }
            return out;
        };
        for (auto & msg : messages) {
            if (json_value(msg, "role", std::string()) == "assistant" || !msg.contains("content")) {
                continue;
            }
            json & content = msg.at("content");
            if (content.is_string()) {
                content = mark(content.get<std::string>());
            } else if (content.is_array()) {
                for (auto & p : content) {
                    if (json_value(p, "type", std::string()) == "text" && p.contains("text") && p.at("text").is_string()) {
                        p["text"] = mark(p.at("text").get<std::string>());
                    }
                }
            }
        }
    }

    auto caps = common_chat_templates_get_caps(opt.tmpls.get());

    common_chat_templates_inputs inputs;
    inputs.messages               = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                  = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice            = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema            = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar                = grammar;
    inputs.use_jinja              = opt.use_jinja;
    inputs.parallel_tool_calls    = json_value(body, "parallel_tool_calls",
                                               opt.parallel_tool_calls_default || caps["supports_parallel_tool_calls"]);
    inputs.add_generation_prompt  = json_value(body, "add_generation_prompt", true);
    inputs.continue_final_message = body.contains("continue_final_message") ?
        common_chat_continuation_parse(body.at("continue_final_message")) :
        COMMON_CHAT_CONTINUATION_NONE;
    if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && opt.prefill_assistant
        && !inputs.messages.empty() && inputs.messages.back().role == "assistant") {
        if (inputs.messages.size() >= 2 && inputs.messages[inputs.messages.size() - 2].role == "assistant") {
            throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
        }
        inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
        inputs.add_generation_prompt  = false;
    }
    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && inputs.add_generation_prompt) {
        throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
    }
    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE
        && !inputs.messages.empty()
        && inputs.messages.back().role == "assistant"
        && !inputs.messages.back().tool_calls.empty()) {
        throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
    }
    inputs.reasoning_format = opt.reasoning_format;
    if (body.contains("reasoning_format")) {
        inputs.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
    }
    inputs.enable_thinking = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
        }
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto & item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }
    // --chat-template-kwargs-force: the server's pinned keys replace what the request sent
    server_template_kwargs_apply_force(inputs.chat_template_kwargs, opt.chat_template_kwargs_force);

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    } else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    } else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    // Parse the OAI "reasoning_effort" field; "none" disables reasoning.
    int effort_budget = -1; // thinking-token budget derived from the effort level (--reasoning-effort-budget), -1 = none
    // the generation limit an effort fraction or an answer reserve applies to: the request's own limit capped by the
    // server's --n-predict, -1 when neither is finite
    auto generation_limit = [&body, &opt]() -> int {
        auto int_field = [&body](const char * key) -> int {
            return (body.contains(key) && body.at(key).is_number_integer()) ? body.at(key).get<int>() : -1;
        };
        int request_limit = int_field("max_completion_tokens");
        if (request_limit <= 0) request_limit = int_field("max_tokens");
        if (request_limit <= 0) request_limit = int_field("n_predict");
        return server_effort_table::effective_limit(request_limit, opt.n_predict_default);
    };
    if (body.contains("reasoning_effort") && !opt.chat_template_kwargs_force.count("reasoning_effort")) {
        auto reasoning_effort = json_value(body, "reasoning_effort", std::string(""));
        if (reasoning_effort == "none") {
            inputs.enable_thinking = false;
            inputs.chat_template_kwargs.erase("reasoning_effort");
        } else if (!reasoning_effort.empty()) {
            inputs.chat_template_kwargs["reasoning_effort"] = json(reasoning_effort).dump();
            if (!opt.effort_budget.empty()) {
                effort_budget = opt.effort_budget.budget(reasoning_effort, generation_limit());
            }
        }
    }
    // a forced enable_thinking also beats the request's reasoning_effort "none"
    {
        auto forced = opt.chat_template_kwargs_force.find("enable_thinking");
        if (forced != opt.chat_template_kwargs_force.end()) {
            if (forced->second == "true")  inputs.enable_thinking = true;
            if (forced->second == "false") inputs.enable_thinking = false;
        }
    }

    inputs.force_pure_content = opt.force_pure_content;
    inputs.tool_call_format   = opt.tool_call_format;

    // Tool-call argument type repair (common_chat_coerce_tool_args): hand the schemas of the offered tools to the task so the
    // parser can repair "5" -> 5 etc. on the final parse. The request field wins over the server default.
    if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        llama_params["tool_arg_schemas"]  = tools;
        llama_params["tool_arg_coercion"] = json_value(body, "tool_arg_coercion", opt.tool_arg_coercion);
        // a call the model wrote inside reasoning it never closed is delivered as a tool call (on by default, a request can
        // turn it off); the server still withholds it after a max_tokens cut
        llama_params["reasoning_call_rescue"] = json_value(body, "reasoning_call_rescue", true);
        // a call written in a slightly malformed form is rebuilt on the final parse (on by default, a request can turn it off)
        llama_params["malformed_call_repair"] = json_value(body, "malformed_call_repair", true);
        // a call the text ended inside (max_tokens / context / deadline cut it) is not returned as a call to run, and a stream
        // sends each call once it is complete (on by default; false = upstream: arguments stream as they are written and a
        // cut call is returned with whatever arguments it had)
        llama_params["withhold_incomplete_tool_calls"] = json_value(body, "withhold_incomplete_tool_calls", true);
    }

    // Apply chat template to the list of messages
    common_chat_session_params session_params;
    session_params.echo = json_value(body, "echo", false);
    out_session = common_chat_session(opt.tmpls.get(), vocab, inputs, session_params);

    // Put the neutralised literals back into the rendered prompt and record where they sit. The key is always set (empty
    // when nothing was neutralised) so the body-copy loop below cannot let a client supply its own spans.
    std::string rendered_prompt = out_session.prompt().str();
    json plain_spans_json = json::array();
    if (plain_on) {
        const std::vector<std::string> & lits = opt.plain_literals;
        const uint32_t hi = plain_base + (uint32_t) lits.size();
        std::string restored;
        restored.reserve(rendered_prompt.size());
        plain_spans_t spans;
        for (size_t i = 0; i < rendered_prompt.size();) {
            const size_t start = i;
            const uint32_t c   = plain_utf8_next(rendered_prompt, i);
            if (c >= plain_base && c < hi) {
                const size_t b = restored.size();
                restored += lits[c - plain_base];
                if (!spans.empty() && spans.back().second == b) {
                    spans.back().second = restored.size(); // adjacent literals form one span
                } else {
                    spans.emplace_back(b, restored.size());
                }
            } else {
                restored.append(rendered_prompt, start, i - start);
            }
        }
        for (const auto & sp : spans) {
            json e = json::array();
            e.push_back(sp.first);
            e.push_back(sp.second);
            plain_spans_json.push_back(e);
        }
        rendered_prompt = std::move(restored);
    }

    // the session carries format, grammar, triggers, preserved tokens, generation prompt, stops, parser and message
    // delimiters (applied by apply_chat_session); only what the fork adds on top travels in llama_params
    llama_params["prompt"]           = rendered_prompt;
    llama_params["plain_text_spans"] = plain_spans_json;

    // fastllm-1027 C8: where the thinking block ends. The end tag only counts as the end where a block ends (before a line
    // break or a new tag, outside backticks and fences) when the template itself writes a line break after the tag, so the
    // model was trained to write one; a template that puts the answer right after the tag keeps the first-tag rule. The
    // request field wins.
    {
        bool block_default = false;
        if (!out_session.thinking_end_tags().empty() && !out_session.thinking_end_tags()[0].empty()) {
            const std::string & et  = out_session.thinking_end_tags()[0];
            const std::string   src = common_chat_templates_source(opt.tmpls.get());
            for (size_t p = src.find(et); p != std::string::npos && !block_default; p = src.find(et, p + 1)) {
                const size_t a = p + et.size();
                block_default  = src.compare(a, 1, "\n") == 0 || src.compare(a, 2, "\\n") == 0;
            }
        }
        llama_params["block_level_thinking_end"] = json_value(body, "block_level_thinking_end", block_default);
        // a turn the model ends inside or right after its reasoning, with no text and no call, is a retryable error, not a
        // finish: "stop" with an empty reply reads as "done" to an agent client
        llama_params["error_on_empty_thinking_reply"] = json_value(body, "error_on_empty_thinking_reply", true);
    }

    // Reasoning budget: pass parameters through to sampling layer
    {
        // answer reserve (--reasoning-answer-reserve, server-answer-reserve.h): the request field replaces the server's;
        // an integer is a token count, a string is "N" or "P%", -1 turns it off for this request
        server_answer_reserve answer_reserve = opt.answer_reserve;
        if (body.contains("reasoning_answer_reserve") && !body.at("reasoning_answer_reserve").is_null()) {
            const json & rv = body.at("reasoning_answer_reserve");
            std::string spec;
            if (rv.is_number_integer()) {
                spec = std::to_string(rv.get<long long>());
            } else if (rv.is_string()) {
                spec = rv.get<std::string>();
            } else {
                throw std::invalid_argument("reasoning_answer_reserve must be an integer token count or a string \"N\" / \"P%\"");
            }
            std::string err;
            if (!answer_reserve.parse(spec, err)) {
                throw std::invalid_argument("invalid reasoning_answer_reserve: " + err);
            }
        }

        // no explicit budget in the request (-1): the effort level's budget (more specific than the server default), else
        // the budget the answer reserve leaves inside the generation limit, else the server default
        const int reasoning_budget = server_answer_reserve::resolve(
                json_value(body, "reasoning_budget_tokens", json_value(body, "thinking_budget_tokens", -1)),
                effort_budget,
                answer_reserve.active() ? answer_reserve.budget(generation_limit()) : -1,
                opt.reasoning_budget);

        if (!out_session.thinking_end_tags().empty()) {
            llama_params["reasoning_budget_tokens"] = reasoning_budget;
            llama_params["reasoning_budget_start_tag"] = out_session.thinking_start_tag();
            llama_params["reasoning_budget_end_tags"] = out_session.thinking_end_tags();
            llama_params["reasoning_budget_message"] = json_value(body, "reasoning_budget_message", opt.reasoning_budget_message);
            llama_params["reasoning_control"] = json_value(body, "reasoning_control", false);
        }
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::invalid_argument("logprobs is not supported with tools + stream");
        }
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
    }

    // reasoning_loop_recovery "recover" swaps the high-effort sentence inside the FIRST turn of the prompt. That turn is the
    // system turn whenever the first message is a system/developer message, or the template adds one itself (Qwen3.8 does
    // when thinking is on). Otherwise the first turn may be the user's, so a user message holding the sentence turns the
    // splice off for this request (it then falls back to stop). Set here, before the copy below, so a client cannot clear it.
    {
        std::string high = json_value(body, "reasoning_effort_high_text", opt.rlr_high_text);
        if (high.empty()) {
            high = server_loop_recovery::QWEN38_EFFORT_HIGH;
        }
        bool user_has = false;
        const json & msgs = body.at("messages");
        const std::string role0 = msgs.empty() ? std::string() : json_value(msgs[0], "role", std::string());
        if (role0 != "system" && role0 != "developer") {
            for (const auto & msg : msgs) {
                if (json_value(msg, "role", std::string()) != "user" || !msg.contains("content")) {
                    continue;
                }
                const json & c = msg.at("content");
                if (c.is_string()) {
                    user_has = user_has || c.get<std::string>().find(high) != std::string::npos;
                } else if (c.is_array()) {
                    for (const auto & part : c) {
                        if (part.is_object() && part.contains("text") && part.at("text").is_string()) {
                            user_has = user_has || part.at("text").get<std::string>().find(high) != std::string::npos;
                        }
                    }
                }
            }
        }
        llama_params["reasoning_loop_user_has_sentence"] = user_has;
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    // the session owns these, the server applies them with server_task::apply_chat_session()
    for (const char * key : { "grammar_lazy", "grammar_triggers", "preserved_tokens" }) {
        llama_params.erase(key);
    }
    if (!out_session.grammar().empty()) {
        llama_params.erase("grammar");
        llama_params.erase("json_schema");
    }

    return llama_params;
}

json format_embeddings_response_oaicompat(
        const json & request,
        const std::string & model_name,
        const json & embeddings,
        bool use_base64,
        const std::string & quant_name) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto & elem : embeddings) {
        json embedding_obj;

        if (!quant_name.empty()) {
            // the handler already replaced "embedding" with the packed / scaled integers; name the format so a client can tell
            // a 48-element binary vector from a 48-dimensional float one
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", quant_name}
            };
        } else if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json {
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n) {
    int32_t n_tokens = 0;
    bool return_text = is_tei_format && json_value(request, "return_text", false);
    // Infinity-compatible rerank options (MIT, michaelfeil/infinity: inference/batch_handler.py:177-221,
    // fastapi_schemas/pymodels.py:221-232), available on BOTH request formats:
    //  - return_documents: attach each input document. Jina/Cohere shape {"document": {"text": ...}};
    //    TEI keeps its own "return_text" -> "text".
    //  - raw_scores: false -> sigmoid(score) in (0,1); true -> the raw logit. Infinity DEFAULTS to sigmoid;
    //    this server deliberately keeps RAW when the request does not say, so callers calibrated on raw
    //    logits are unaffected. Sigmoid is monotone, so ranking and top_n selection are unchanged.
    const bool return_documents = !is_tei_format && json_value(request, "return_documents", false);
    const bool apply_sigmoid    = request.contains("raw_scores") && !json_value(request, "raw_scores", true);
    std::vector<json> elements; // Temporary vector to hold unsorted elements
    std::string score_label = is_tei_format ? "score" : "relevance_score";
    for (const auto & rank : ranks) {
        int index = json_value(rank, "index", 0);
        double score = json_value(rank, "score", 0.0);
        if (apply_sigmoid) {
            score = 1.0 / (1.0 + std::exp(-score));
        }
        json elem = json{
            {"index", index},
            {score_label, score},
        };
        n_tokens += json_value(rank, "tokens_evaluated", 0);
        if (return_text) {
            elem["text"] = std::move(texts[index]);
        } else if (return_documents) {
            elem["document"] = json{{"text", texts[index]}};
        }
        elements.push_back(elem);
    }

    std::sort(elements.begin(), elements.end(), [score_label](const json& a, const json& b) {
        return json_value(a, score_label, 0.0) > json_value(b, score_label, 0.0);
    });

    // top_n <= 0 means "all" (Infinity batch_handler.py:218). It used to reach resize() as a negative int
    // converted to size_t: top_n=0 returned nothing and top_n=-1 failed the request with HTTP 500.
    if (top_n > 0 && top_n < (int) elements.size()) {
        elements.resize(top_n);
    }
    json results = elements;

    if (is_tei_format) return results;

    json res = json{
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json{
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"results", results}
    };

    return res;
}


//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx, size_t n_top) {
    std::vector<llama_token_data> cur;

    const auto * logits = llama_get_logits_ith(ctx, idx);
    const llama_token * sampled_ids = llama_get_sampled_candidates_ith(ctx, idx);

    const int n_logits = llama_get_sampled_logits_count_ith(ctx, idx);

    cur.resize(n_logits);
    if (sampled_ids) {
        for (int i = 0; i < n_logits; i++) {
            cur[i] = llama_token_data{sampled_ids[i], logits[i], 0.0f};
        }
    } else {
        for (llama_token token_id = 0; token_id < n_logits; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    // sort tokens by logits (partial: only the leading `n_top` need ordering)
    if (n_top > cur.size()) {
        n_top = cur.size();
    }
    if (n_top > 0) {
        std::partial_sort(cur.begin(), cur.begin() + n_top, cur.end(),
            [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
    }

    // apply softmax
    float max_l = -std::numeric_limits<float>::infinity();
    if (n_top > 0) {
        max_l = cur[0].logit; // partial_sort guarantees the absolute maximum is at index 0
    } else {
        for (const auto & t : cur) {
            max_l = std::max(max_l, t.logit);
        }
    }
    float cum_sum = 0.0f;
    for (auto & t : cur) {
        float p = expf(t.logit - max_l);
        t.p = p;
        cum_sum += p;
    }
    for (auto & t : cur) {
        t.p /= cum_sum;
    }

    return cur;
}

std::string safe_json_to_str(const json & data) {
    return data.dump_safe();
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(const llama_vocab * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens) {
    auto model = llama_get_model(ctx);
    return tokens_to_str(llama_model_get_vocab(model), tokens.begin(), tokens.end());
}

std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens) {
    return tokens_to_str(vocab, tokens.begin(), tokens.end());
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & data) {
        ss << "data: " <<
            safe_json_to_str(data) <<
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_oai_resp_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & event_obj) {
        ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
        ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_anthropic_sse(const json & data) {
    std::ostringstream ss;

    auto send_event = [&ss](const json & event_obj) {
        if (event_obj.contains("event") && event_obj.contains("data")) {
            ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
            ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
        } else {
            ss << "data: " << safe_json_to_str(event_obj) << "\n\n";
        }
    };

    if (data.is_array()) {
        for (const auto & event : data) {
            send_event(event);
        }
    } else {
        send_event(data);
    }

    return ss.str();
}

bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

json tokens_to_json_detailed(const llama_vocab * vocab, const llama_tokens & tokens, bool with_pieces, bool with_offsets) {
    std::vector<std::string> pieces;
    pieces.reserve(tokens.size());
    for (const llama_token token : tokens) {
        pieces.push_back(common_token_to_piece(vocab, token));
    }
    std::vector<token_char_span> spans;
    if (with_offsets) {
        spans = token_char_offsets(pieces);
    }
    json out = json::array();
    for (size_t i = 0; i < tokens.size(); i++) {
        json piece_json;
        if (is_valid_utf8(pieces[i])) {
            piece_json = pieces[i];
        } else {
            // not valid UTF-8 on its own (byte-fallback token): store the byte values
            piece_json = json::array();
            for (unsigned char c : pieces[i]) {
                piece_json.push_back(static_cast<int>(c));
            }
        }
        json item = { { "id", tokens[i] } };
        if (with_pieces) {
            item["piece"] = piece_json;
        }
        if (with_offsets) {
            item["text"]  = piece_json;
            item["start"] = spans[i].start;
            item["stop"]  = spans[i].stop;
        }
        out.push_back(std::move(item));
    }
    return out;
}

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

std::string format_rerank_template(
        const std::string & tmpl,
        const std::string & query,
        const std::string & doc) {
    std::string prompt;
    prompt.reserve(tmpl.size());
    for (size_t pos = 0; pos < tmpl.size();) {
        if (tmpl.compare(pos, 7, "{query}") == 0) {
            prompt += query;
            pos += 7;
        } else if (tmpl.compare(pos, 10, "{document}") == 0) {
            prompt += doc;
            pos += 10;
        } else {
            prompt += tmpl[pos++];
        }
    }
    return prompt;
}

server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc,
        const std::vector<raw_buffer> & query_files,
        const std::vector<raw_buffer> & doc_files,
        const mtmd_helper_init_opt & init_opt) {
    server_tokens result = {};

    // media must be spliced through the mtmd pipeline
    if ((!query_files.empty() || !doc_files.empty()) && mctx == nullptr) {
        throw std::invalid_argument("multimodal rerank input requires --mmproj");
    }

    const char * rerank_prompt = llama_model_chat_template(model, "rerank");

    if (rerank_prompt != nullptr) {
        // multimodal: append one media marker per image (arrays already flattened by caller)
        std::string q = query;
        for (size_t i = 0; i < query_files.size(); i++) {
            q += get_media_marker();
        }
        std::string d = doc;
        for (size_t i = 0; i < doc_files.size(); i++) {
            d += get_media_marker();
        }

        std::string prompt = format_rerank_template(rerank_prompt, q, d);

        if (query_files.empty() && doc_files.empty()) {
            server_tokens tokens = tokenize_input_subprompt(vocab, mctx, prompt, false, true, init_opt);
            result.push_back(tokens);
        } else {
            // process_mtmd_prompt returns a complete mtmd server_tokens; return it directly
            // (push_back into a non-mtmd result would hit the NULL placeholder tokens)
            std::vector<raw_buffer> files;
            files.insert(files.end(), query_files.begin(), query_files.end());
            files.insert(files.end(), doc_files.begin(),   doc_files.end());
            return process_mtmd_prompt(mctx, prompt, files, init_opt);
        }
    } else {
        if (!query_files.empty() || !doc_files.empty()) {
            throw std::invalid_argument("multimodal rerank input requires a model with a rerank chat template");
        }

        // Get EOS token - use SEP token as fallback if EOS is not available
        server_tokens query_tokens = tokenize_input_subprompt(vocab, mctx, query, false, false, init_opt);
        server_tokens doc_tokens   = tokenize_input_subprompt(vocab, mctx, doc,   false, false, init_opt);
        llama_token eos_token = llama_vocab_eos(vocab);
        if (eos_token == LLAMA_TOKEN_NULL) {
            eos_token = llama_vocab_sep(vocab);
        }

        if (llama_vocab_get_add_bos(vocab)) {
            result.push_back(llama_vocab_bos(vocab));
        }
        result.push_back(query_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
        if (llama_vocab_get_add_sep(vocab)) {
            result.push_back(llama_vocab_sep(vocab));
        }
        result.push_back(doc_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
    }

    return result;
}

//
// server_subproc
//

FILE * server_reserve_stdout() {
    fflush(stdout);
    // the reserved stream is not inherited, grandchildren get stdout and stderr of their own
#ifdef _WIN32
    int fd = _dup(_fileno(stdout));
    GGML_ASSERT(fd >= 0);
    SetHandleInformation((HANDLE) _get_osfhandle(fd), HANDLE_FLAG_INHERIT, 0);
    _dup2(_fileno(stderr), _fileno(stdout));
    SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
    FILE * f = _fdopen(fd, "w");
#else
    int fd = fcntl(fileno(stdout), F_DUPFD_CLOEXEC, 0);
    GGML_ASSERT(fd >= 0);
    dup2(fileno(stderr), fileno(stdout));
    FILE * f = fdopen(fd, "w");
#endif
    GGML_ASSERT(f);
    return f;
}

bool server_subproc::has_output() {
    FILE * files[SERVER_SUBPROC_STREAMS] = { sproc.stdout_file(), sproc.stderr_file() };
    for (int i = 0; i < SERVER_SUBPROC_STREAMS; i++) {
        if (out_handles[i] >= 0) {
            continue;
        }
        if (!files[i]) {
            return false;
        }
#ifdef _WIN32
        HANDLE h = (HANDLE) _get_osfhandle(_fileno(files[i]));
        if (h == INVALID_HANDLE_VALUE) {
            return false;
        }
        out_handles[i] = (intptr_t) h;
#else
        int fd = fileno(files[i]);
        if (fd < 0) {
            return false;
        }
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        out_handles[i] = fd;
#endif
    }
    return true;
}

// SOURCE-READ: tmp/ollama_src_1107/llm_status.go (ollama 3fbc609be4b3, MIT; no code copied)
// SOURCE-NUANCE: ollama keeps only ERROR lines (not the whole stderr), joins them with '\n', and when the joined text passes
//   maxCapturedErrorBytes (8 KiB) keeps the TAIL and then drops the partial first line (status.go:44-60).
void server_subproc::append_error(const std::string & line) {
    if (line.empty()) {
        return;
    }
    static constexpr size_t cap = 8 * 1024;
    std::lock_guard<std::mutex> lk(err_mu);
    err_tail = err_tail.empty() ? line : err_tail + "\n" + line;
    if (err_tail.size() > cap) {
        err_tail.erase(0, err_tail.size() - cap);
        const size_t nl = err_tail.find('\n');
        if (nl != std::string::npos) {
            err_tail.erase(0, nl + 1);
        }
    }
}

std::string server_subproc::last_error() const {
    std::lock_guard<std::mutex> lk(err_mu);
    return err_tail;
}

// SOURCE-READ: tmp/ollama_src_1107/llm_runner_job_windows.go (ollama 3fbc609be4b3, MIT; no code copied)
// SOURCE-NUANCE: ONE job for the whole process lifetime, created lazily on the first child and RETRIED by the next call if creating
//   it failed; its handle is deliberately never closed -- the kernel closes it when the router exits, which is what kills the
//   children (runner_job_windows.go:18-31,51-53).
bool server_subproc::assign_to_kill_on_close_job(std::string & err) {
#ifdef _WIN32
    static std::mutex job_mu;
    static HANDLE     job = nullptr;
    HANDLE proc_handle = (HANDLE) sproc.native_handle();
    if (proc_handle == nullptr) {
        err = "no process handle";
        return false;
    }
    std::lock_guard<std::mutex> lk(job_mu);
    if (job == nullptr) {
        HANDLE j = CreateJobObjectW(nullptr, nullptr);
        if (j == nullptr) {
            err = "CreateJobObject failed, error " + std::to_string(GetLastError());
            return false;
        }
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = {};
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(j, JobObjectExtendedLimitInformation, &info, sizeof(info))) {
            err = "SetInformationJobObject failed, error " + std::to_string(GetLastError());
            CloseHandle(j);
            return false;
        }
        job = j; // kept open for the router's lifetime on purpose
    }
    if (!AssignProcessToJobObject(job, proc_handle)) {
        err = "AssignProcessToJobObject failed, error " + std::to_string(GetLastError());
        return false;
    }
    return true;
#else
    (void) err;
    return true;
#endif
}

bool server_subproc::output_closed() const {
    return out_closed[SERVER_SUBPROC_STDOUT] && out_closed[SERVER_SUBPROC_STDERR];
}

int server_subproc::read_output(server_subproc_stream stream, char * buf, size_t len) {
    if (out_closed[stream]) {
        return -1;
    }
    if (!has_output()) {
        out_closed[stream] = true;
        return -1;
    }
#ifdef _WIN32
    HANDLE h     = (HANDLE) out_handles[stream];
    DWORD  avail = 0;
    if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) {
        out_closed[stream] = true; // pipe broken, child gone
        return -1;
    }
    if (avail == 0) {
        return 0;
    }
    DWORD to_read = avail < (DWORD) len ? avail : (DWORD) len;
    DWORD got     = 0;
    if (!ReadFile(h, buf, to_read, &got, NULL) || got == 0) {
        out_closed[stream] = true;
        return -1;
    }
    return (int) got;
#else
    while (true) {
        ssize_t r = read((int) out_handles[stream], buf, len);
        if (r > 0) {
            return (int) r;
        }
        if (r < 0 && errno == EINTR) {
            continue;
        }
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return 0;
        }
        out_closed[stream] = true; // EOF or error
        return -1;
    }
#endif
}

server_subproc::waiter::waiter() {
#ifndef _WIN32
    int fds[2];
    GGML_ASSERT(pipe(fds) == 0);
    for (int fd : fds) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    }
    wake_fd[0] = fds[0];
    wake_fd[1] = fds[1];
#endif
}

server_subproc::waiter::~waiter() {
#ifndef _WIN32
    close((int) wake_fd[0]);
    close((int) wake_fd[1]);
#endif
}

void server_subproc::waiter::wake() {
#ifndef _WIN32
    char c = 1;
    (void) !write((int) wake_fd[1], &c, 1);
#endif
}

void server_subproc::waiter::wait(const std::vector<server_subproc *> & procs, std::vector<bool> & ready, int64_t timeout_ms) {
    ready.assign(procs.size(), false);
#ifdef _WIN32
    // no waitable wait exists for anonymous pipes, so poll them in 50 ms steps
    bool any = false;
    for (size_t i = 0; i < procs.size(); i++) {
        server_subproc * p = procs[i];
        ready[i] = !p->has_output();
        for (int s = 0; s < SERVER_SUBPROC_STREAMS && !ready[i]; s++) {
            DWORD avail = 0;
            if (!p->out_closed[s] && (!PeekNamedPipe((HANDLE) p->out_handles[s], NULL, 0, NULL, &avail, NULL) || avail > 0)) {
                ready[i] = true; // data or broken pipe, read_output() tells which
            }
        }
        any = any || ready[i];
    }
    if (!any) {
        int64_t step = timeout_ms < 0 ? 50 : std::min<int64_t>(timeout_ms, 50);
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
    }
#else
    std::vector<pollfd> pfds;
    pfds.reserve(procs.size() * SERVER_SUBPROC_STREAMS + 1);
    pfds.push_back({ (int) wake_fd[0], POLLIN, 0 });
    for (auto * p : procs) {
        const bool open = p->has_output();
        for (int s = 0; s < SERVER_SUBPROC_STREAMS; s++) {
            pfds.push_back({ open && !p->out_closed[s] ? (int) p->out_handles[s] : -1, POLLIN, 0 }); // poll() skips negative fds
        }
    }
    int timeout = timeout_ms < 0 ? -1 : (int) std::min<int64_t>(timeout_ms, std::numeric_limits<int>::max());
    int r = poll(pfds.data(), pfds.size(), timeout);
    if (r < 0 && errno != EINTR) {
        LOG_ERR("%s: poll() failed: %s\n", __func__, strerror(errno));
    }
    if (pfds[0].revents) {
        char buf[64];
        while (read((int) wake_fd[0], buf, sizeof(buf)) > 0) {}
    }
    for (size_t i = 0; i < procs.size(); i++) {
        ready[i] = !procs[i]->has_output();
        for (int s = 0; s < SERVER_SUBPROC_STREAMS; s++) {
            ready[i] = ready[i] || pfds[1 + i * SERVER_SUBPROC_STREAMS + s].revents != 0;
        }
    }
#endif
}

//
// server_role
//

server_role::server_role() {
    if (is_managed_by_router()) {
        cmd_out = server_reserve_stdout();
    }
}

server_role::~server_role() {
    if (cmd_out) {
        fclose(cmd_out);
    }
}

bool server_role::is_managed_by_router() {
    return std::getenv(SERVER_ENV_ROUTER_PORT) != nullptr;
}

server_role_type server_role::get_role() {
    const char * role = std::getenv(SERVER_ENV_CHILD_ROLE);
    return server_role_type_from_str(role ? role : "");
}

std::thread server_role::setup(const std::function<void(int)> & shutdown_handler) {
    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}

void server_role::notify_to_router(const std::string & state, const json & payload) {
    json data = {
        {"state", state},
        {"payload", payload},
    };
    std::lock_guard<std::mutex> lk(mtx_stdout);
    fprintf(cmd_out, "%s%s\n", CMD_CHILD_TO_ROUTER_STATE, safe_json_to_str(data).c_str());
    fflush(cmd_out);
}
