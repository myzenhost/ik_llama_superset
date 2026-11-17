#include "llama-adapter.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include <algorithm>
#include <map>
#include <cassert>
#include <cmath>
#include <cstring>
#include <set>
#include <cerrno>
#include <sstream>
#include <stdexcept>
#include <thread>

// vec

ggml_tensor * llama_adapter_cvec::tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const {
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = ggml_add(ctx, cur, layer_dir);
    }

    return cur;
}

bool llama_adapter_cvec::init(const llama_model & model) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(tensors.empty());
    GGML_ASSERT(ctxs.empty());
    GGML_ASSERT(bufs.empty());

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // make tensors
    tensors.reserve(hparams.n_layer);
    tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < hparams.n_layer; il++) {
        // WIRE-66 group 2: this fork's llama_model has no select_buft() method (upstream-only --
        // src/llama-model.h has neither the method nor a comment excluding it, confirmed by grep).
        // The fork's own equivalent, used the same way throughout built src/llama.cpp (e.g. :4753),
        // is the public data member buft_layer[il].buft. Same effect: the buffer type this layer's
        // matrices actually live in.
        ggml_backend_buffer_type_t buft = model.buft_layer[il].buft;
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    return true;
}

bool llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    const auto & hparams = model.hparams;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        layer_start = -1;
        layer_end   = -1;
        return true;
    }

    if (n_embd != (int) hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return false;
    }

    if (tensors.empty()) {
        if (!init(model)) {
            return false;
        }
    }

    layer_start = il_start;
    layer_end   = il_end;

    for (size_t il = 1; il < hparams.n_layer; il++) {
        assert(tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(tensors[il], data + off, 0, n_embd * ggml_element_size(tensors[il]));
        }
    }

    return true;
}

// lora

llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    const std::string name(w->name);

    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

// One row of `t` (ne[0] values) as F32. Rows are addressed through nb[1]/nb[2], so per-row metadata (row_meta_size) and
// any ggml type with a to_float are handled the way the type itself lays them out.
static void llama_dora_row_to_f32(ggml_type type, const uint8_t * row, float * out, int64_t n) {
    if (type == GGML_TYPE_F32) {
        memcpy(out, row, n * sizeof(float));
        return;
    }
    const auto * tt = ggml_get_type_traits(type);
    if (tt == nullptr || tt->to_float == nullptr) {
        throw std::runtime_error(format("DoRA: no dequantisation for tensor type %s", ggml_type_name(type)));
    }
    tt->to_float(row, out, n);
}

// All of a tensor's elements as F32 (ne0-major), from raw bytes laid out like `t` (contiguous rows).
static std::vector<float> llama_dora_to_f32(const ggml_tensor * t, const std::vector<uint8_t> & raw) {
    const int64_t n_rows = ggml_nrows(t);
    const size_t  rb     = ggml_row_size(t->type, t->ne[0]);
    if (raw.size() < rb * (size_t) n_rows) {
        throw std::runtime_error(format("DoRA: tensor '%s' data is short", t->name));
    }
    std::vector<float> out((size_t) (n_rows * t->ne[0]));
    for (int64_t r = 0; r < n_rows; ++r) {
        llama_dora_row_to_f32(t->type, raw.data() + r * rb, out.data() + r * t->ne[0], t->ne[0]);
    }
    return out;
}

// d - 1 for every output row of the base weight W (row S69-ADAPTERS-DORA-RUNTIME-KIND-2026-10-26-01).
//   W  ggml [n_in, n_out, n_exp]   (n_exp = 1 for a 2-D linear; stacked experts otherwise), read with ggml_backend_tensor_get
//   A  ggml [n_in, r, n_exp]       B ggml [r, n_out, n_exp]       M (magnitude) ggml [n_out, n_exp]
//   d[e, j] = M[e, j] / || W[e, j, :] + s * sum_k B[e, j, k] A[e, k, :] ||_2
// = PEFT DoraLinearLayer.get_weight_norm (peft 0.17.1 tuners/lora/dora.py:30-35: norm over dim=1 of W + scaling * B@A, i.e. one
// norm per output row) and mag_norm_scale = magnitude / weight_norm (:87). Rewritten from that math (Apache-2.0), no code copied.
// Accumulated in double; PEFT accumulates in fp32, the difference is far below the harness tolerance.
static std::vector<float> llama_dora_compute_dm1(const ggml_tensor * W, const std::vector<float> & A, const std::vector<float> & B,
                                                 const std::vector<float> & M, int64_t r, float s) {
    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];
    const int64_t n_exp = W->ne[2];
    const int64_t n_rows = n_out * n_exp;
    std::vector<float> dm1((size_t) n_rows);
    const size_t row_bytes = ggml_row_size(W->type, n_in);

    auto work = [&](int64_t r0, int64_t r1, std::string * err) {
        try {
            std::vector<uint8_t> raw(row_bytes);
            std::vector<float>   w((size_t) n_in);
            std::vector<double>  acc((size_t) n_in);
            for (int64_t ir = r0; ir < r1; ++ir) {
                const int64_t e = ir / n_out;
                const int64_t j = ir % n_out;
                ggml_backend_tensor_get(W, raw.data(), (size_t) e * W->nb[2] + (size_t) j * W->nb[1], row_bytes);
                llama_dora_row_to_f32(W->type, raw.data(), w.data(), n_in);
                for (int64_t i = 0; i < n_in; ++i) {
                    acc[i] = w[i];
                }
                const float * b_row = B.data() + ((size_t) e * n_out + j) * r;
                for (int64_t k = 0; k < r; ++k) {
                    const double c = (double) s * b_row[k];
                    const float * a_row = A.data() + ((size_t) e * r + k) * n_in;
                    for (int64_t i = 0; i < n_in; ++i) {
                        acc[i] += c * a_row[i];
                    }
                }
                double ss = 0.0;
                for (int64_t i = 0; i < n_in; ++i) {
                    ss += acc[i] * acc[i];
                }
                const double norm = std::sqrt(ss);
                if (!(norm > 0.0)) {
                    throw std::runtime_error(format("DoRA: ||W + s B A|| is %g for row %lld of '%s'", norm, (long long) ir, W->name));
                }
                dm1[(size_t) ir] = (float) ((double) M[(size_t) ir] / norm - 1.0);
            }
        } catch (const std::exception & ex) {
            *err = ex.what();
        }
    };

    const int64_t n_thr = std::max<int64_t>(1, std::min<int64_t>({ (int64_t) std::thread::hardware_concurrency(), 16, n_rows / 64 + 1 }));
    std::vector<std::thread> threads;
    std::vector<std::string> errs((size_t) n_thr);
    const int64_t per = (n_rows + n_thr - 1) / n_thr;
    for (int64_t t = 0; t < n_thr; ++t) {
        const int64_t r0 = t * per;
        const int64_t r1 = std::min(n_rows, r0 + per);
        if (r0 >= r1) {
            break;
        }
        threads.emplace_back(work, r0, r1, &errs[(size_t) t]);
    }
    for (auto & th : threads) {
        th.join();
    }
    for (const auto & e : errs) {
        if (!e.empty()) {
            throw std::runtime_error(e);
        }
    }
    return dm1;
}

static void llama_adapter_lora_init_impl(llama_model & model, FILE * file, llama_adapter_lora & adapter) {
    bool kind_dora = false; // adapter.kinds lists "dora"
    bool kind_lily = false; // adapter.kinds lists "lily" (row S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01)
    bool kind_ia3  = false; // adapter.kinds lists "ia3" (row S69-ADAPTERS-LINEAR-WRAPPER-IA3-RED-VEC-SCALE-2026-10-26-01)

    ggml_context * ctx_init;
    gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_init,
    };

    gguf_context_ptr ctx_gguf { gguf_init_from_file_ptr(file, meta_gguf_params) };
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load lora adapter from file");
    }

    ggml_context_ptr ctx { ctx_init };

    // must come after gguf_init_from_file_ptr, the llama_file constructor moves the file position
    llama_file gguf_file(file);

    // check metadata
    {
        const gguf_context * gguf_ctx = ctx_gguf.get();

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values.\n", __func__);

        // get metadata as string
        for (int i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            gguf_type type = gguf_get_kv_type(gguf_ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(gguf_ctx, i)), gguf_get_arr_n(gguf_ctx, i))
                : gguf_type_name(type);
            const char * name = gguf_get_key(gguf_ctx, i);
            const size_t MAX_VALUE_LEN = 40;
            const std::string value = gguf_kv_to_str(gguf_ctx, i, MAX_VALUE_LEN);

            if (type != GGUF_TYPE_ARRAY) {
                adapter.gguf_kv.emplace(name, value);
            }

            std::string print_value = value.size() > MAX_VALUE_LEN ? format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str()) : value;
            replace_all(print_value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), print_value.c_str());
        }

        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? "" : std::string(gguf_get_val_str(gguf_ctx, id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? 0.0f : gguf_get_val_f32(gguf_ctx, id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));

        // parse alora invocation sequence vector
        const auto & key = llm_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS);
        const int kid = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (kid >= 0) {
            if (gguf_get_kv_type(ctx_gguf.get(), kid) != GGUF_TYPE_ARRAY) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            const auto arr_type = gguf_get_arr_type(ctx_gguf.get(), kid);
            if (arr_type != GGUF_TYPE_UINT32) {
                throw std::runtime_error("invalid gguf element type for " + key);
            }
            const size_t seq_len = gguf_get_arr_n(ctx_gguf.get(), kid);
            const void * data = gguf_get_arr_data(ctx_gguf.get(), kid);
            adapter.alora_invocation_tokens.resize(seq_len);
            std::copy(
                (const llama_token *)data,
                (const llama_token *)data + seq_len,
                adapter.alora_invocation_tokens.begin());
        }

        // runtime kinds carried by the file (convert_lora_to_gguf.py writes ["lora", "dora"] for a PEFT use_dora adapter).
        // An unknown kind is refused here, so a file this loader cannot serve fully never runs as a partial adapter.
        const int kinds_id = gguf_find_key(ctx_gguf.get(), llm_kv(LLM_KV_ADAPTER_KINDS).c_str());
        if (kinds_id >= 0) {
            if (gguf_get_kv_type(ctx_gguf.get(), kinds_id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_gguf.get(), kinds_id) != GGUF_TYPE_STRING) {
                throw std::runtime_error("invalid gguf type for adapter.kinds (want a string array)");
            }
            for (size_t i = 0; i < gguf_get_arr_n(ctx_gguf.get(), kinds_id); ++i) {
                const std::string k = gguf_get_arr_str(ctx_gguf.get(), kinds_id, i);
                if (k == "dora") {
                    kind_dora = true;
                } else if (k == "lily") {
                    kind_lily = true;
                } else if (k == "ia3") {
                    kind_ia3 = true;
                } else if (k != "lora") {
                    throw std::runtime_error("adapter kind '" + k + "' is not supported by this build");
                }
            }
        }
    }

    // adapter.lora.gate (row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01): a per-token gate on the delta. An unknown value is
    // refused so a file this loader cannot serve never runs ungated.
    {
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);
        const int gate_id = gguf_find_key(ctx_gguf.get(), llm_kv(LLM_KV_ADAPTER_LORA_GATE).c_str());
        if (gate_id >= 0) {
            if (gguf_get_kv_type(ctx_gguf.get(), gate_id) != GGUF_TYPE_STRING) {
                throw std::runtime_error("invalid gguf type for adapter.lora.gate (want a string)");
            }
            const std::string g = gguf_get_val_str(ctx_gguf.get(), gate_id);
            if (g == "none") {
                adapter.gate = llama_adapter_lora::LLAMA_LORA_GATE_NONE;
            } else if (g == "draft_rows") {
                adapter.gate = llama_adapter_lora::LLAMA_LORA_GATE_DRAFT_ROWS;
            } else if (g == "mask_token") {
                adapter.gate = llama_adapter_lora::LLAMA_LORA_GATE_MASK_TOKEN;
                const int tid = gguf_find_key(ctx_gguf.get(), llm_kv(LLM_KV_ADAPTER_LORA_GATE_TOKEN).c_str());
                if (tid < 0 || gguf_get_kv_type(ctx_gguf.get(), tid) != GGUF_TYPE_UINT32) {
                    throw std::runtime_error("adapter.lora.gate = mask_token needs a u32 adapter.lora.gate_token");
                }
                adapter.gate_token = (llama_token) gguf_get_val_u32(ctx_gguf.get(), tid);
            } else {
                // "invocation" is NOT a gate value: invocation-activated adapters use adapter.alora.invocation_tokens
                throw std::runtime_error("adapter.lora.gate '" + g + "' is not supported by this build (none, draft_rows, mask_token)");
            }
        }
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

    // contexts for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * buft_ctx = ggml_init(params);
            if (!buft_ctx) {
                return nullptr;
            }
            ctx_map[buft] = buft_ctx;
            adapter.ctxs.emplace_back(buft_ctx);
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    std::map<std::string, ggml_tensor *> dora_m_map; // base tensor name -> {base}.dora_m (meta)
    std::map<std::string, ggml_tensor *> lily_r_map; // base tensor name -> {base}.lily_router (meta)
    std::map<std::string, ggml_tensor *> ia3_in_map;  // base tensor name -> {base}.ia3_in  (meta)
    std::map<std::string, ggml_tensor *> ia3_out_map; // base tensor name -> {base}.ia3_out (meta)
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (str_endswith(name, ".lily_a") || str_endswith(name, ".lily_b")) {
            // Lily A / B: the same pairing as lora_a / lora_b; B is [r, n_out, num_B]. Shapes are checked below.
            const bool is_a = str_endswith(name, ".lily_a");
            replace_all(name, is_a ? ".lily_a" : ".lily_b", "");
            llama_adapter_lora_weight & pw = ab_map[name];
            (is_a ? pw.a : pw.b) = cur;
            lily_r_map.emplace(name, nullptr); // marks the pair as Lily; the router fills it
        } else if (str_endswith(name, ".lily_router")) {
            replace_all(name, ".lily_router", "");
            lily_r_map[name] = cur;
        } else if (str_endswith(name, ".ia3_in")) {
            replace_all(name, ".ia3_in", "");
            ia3_in_map[name] = cur;
        } else if (str_endswith(name, ".ia3_out")) {
            replace_all(name, ".ia3_out", "");
            ia3_out_map[name] = cur;
        } else if (str_endswith(name, ".dora_m")) {
            // DoRA magnitude. RESEARCH-DONE: fail-before 2026-10-26 (temp/arex1014/adapw2_1026/before.json): at fdfaa679e the
            // converter exited on "Unexpected name '...q_proj.lora_magnitude_vector'" for all 9 DoRA cases; this loader branch
            // is the runtime half of the fix (the else-throw below keeps any other suffix loud).
            replace_all(name, ".dora_m", "");
            dora_m_map[name] = cur;
        } else if (str_endswith(name, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    }
    if (!dora_m_map.empty() && !kind_dora) {
        throw std::runtime_error("adapter has .dora_m tensors but adapter.kinds does not list \"dora\"");
    }
    if (kind_dora && dora_m_map.empty()) {
        throw std::runtime_error("adapter.kinds lists \"dora\" but the file has no .dora_m tensor");
    }
    if (!lily_r_map.empty() && !kind_lily) {
        throw std::runtime_error("adapter has .lily_* tensors but adapter.kinds does not list \"lily\"");
    }
    if (kind_lily) {
        if (kind_dora || !dora_m_map.empty()) {
            throw std::runtime_error("adapter.kinds lists both \"lily\" and \"dora\": not a combination any trainer produces");
        }
        for (const auto & it : ab_map) {
            const auto r = lily_r_map.find(it.first);
            if (r == lily_r_map.end() || r->second == nullptr) {
                throw std::runtime_error("Lily adapter: '" + it.first + "' has no .lily_router (every Lily weight needs A, B and a router)");
            }
        }
        for (const auto & it : lily_r_map) {
            if (ab_map.find(it.first) == ab_map.end()) {
                throw std::runtime_error("Lily router '" + it.first + ".lily_router' has no .lily_a/.lily_b pair");
            }
        }
        adapter.kind_lily = true;
    }
    const bool has_ia3_tensors = !ia3_in_map.empty() || !ia3_out_map.empty();
    if (has_ia3_tensors && !kind_ia3) {
        throw std::runtime_error("adapter has .ia3_in/.ia3_out tensors but adapter.kinds does not list \"ia3\"");
    }
    if (kind_ia3) {
        if (!has_ia3_tensors) {
            throw std::runtime_error("adapter.kinds lists \"ia3\" but the file has no .ia3_in/.ia3_out tensor");
        }
        if (kind_lily || kind_dora || !ab_map.empty()) {
            throw std::runtime_error("adapter.kinds lists \"ia3\" together with a LoRA-shaped kind: not a combination PEFT produces");
        }
        for (const auto & it : ia3_in_map) {
            if (ia3_out_map.count(it.first)) {
                throw std::runtime_error("IA3 weight '" + it.first + "' has both .ia3_in and .ia3_out (PEFT IA3 scales one side per module)");
            }
        }
    }
    for (const auto & it : dora_m_map) {
        if (ab_map.find(it.first) == ab_map.end()) {
            throw std::runtime_error("DoRA magnitude '" + it.first + ".dora_m' has no lora_a/lora_b pair");
        }
    }

    // get extra buffer types of the CPU
    // TODO: a more general solution for non-CPU extra buft should be implemented in the future
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12593#pullrequestreview-2718659948
    std::vector<ggml_backend_buffer_type_t> buft_extra;
    {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);

        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_extra.emplace_back(*extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & w = it.second;
        bool is_token_embd = str_endswith(name, "token_embd.weight");

        if (!w.a || !w.b) {
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }

        // device buft and device ctx
        // WIRE-66 group 2: no llama_model::get_tensor() method in this fork either. The public
        // FREE function llama_get_model_tensor(model*, name) (include/llama.h:990) is already
        // defined and built (returns nullptr for an unknown name, same contract as upstream's
        // method) -- exact functional equivalent, not a stub.
        const auto * model_tensor = llama_get_model_tensor(&model, name.c_str());
        if (!model_tensor) {
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
        }
        // build_lora_mm feeds a folded weight's LoRA delta the primal input, so folded matmul weights are fine;
        // a latent lookup table (token_embd, also the head when the output is tied) stores rotated rows, refuse it
        if (model.hdmd.inverse_blocks.count(name)) {
            throw std::runtime_error("LoRA tensor '" + name + "' targets a prism.hadamard latent lookup table, which is not supported");
        }

        auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);

        // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
        for (auto & ex : buft_extra) {
            if (ex == buft) {
                LLAMA_LOG_WARN("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error(format("%s: no CPU backend found", __func__));
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);

                break;
            }
        }

        LLAMA_LOG_DEBUG("%s: lora for '%s' -> '%s'\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

        ggml_context * dev_ctx = ctx_for_buft(buft);
        // validate tensor shape
        ggml_tensor * lily_r = nullptr;
        if (auto lr = lily_r_map.find(name); lr != lily_r_map.end()) {
            lily_r = lr->second;
            // PEFT Lily wraps nn.Linear only (peft a7e8ea4a1227 tuners/lily/model.py:_create_new_module): no embeddings, no expert stacks
            if (is_token_embd || model_tensor->ne[2] != 1 || model_tensor->ne[3] != 1) {
                throw std::runtime_error("Lily on '" + name + "': only a 2-D linear weight is supported (PEFT Lily wraps nn.Linear only)");
            }
            if (w.a->ne[2] != 1 || w.b->ne[3] != 1 || lily_r->ne[2] != 1 || lily_r->ne[0] != w.a->ne[1] || lily_r->ne[1] != w.b->ne[2] ||
                lily_r->type != GGML_TYPE_F32) {
                throw std::runtime_error(format("Lily on '%s': want A [n_in, r], B [r, n_out, num_B], F32 router [r, num_B]; got A [%lld, %lld], "
                    "B [%lld, %lld, %lld], router %s [%lld, %lld]", name.c_str(), (long long) w.a->ne[0], (long long) w.a->ne[1],
                    (long long) w.b->ne[0], (long long) w.b->ne[1], (long long) w.b->ne[2], ggml_type_name(lily_r->type),
                    (long long) lily_r->ne[0], (long long) lily_r->ne[1]));
            }
        }
        if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor->ne[0] != w.b->ne[1] || model_tensor->ne[1] != w.a->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
        } else {
            if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[1] != w.b->ne[0]) {
                throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
            }
        }

        // save tensor to adapter
        ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
        ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
        ggml_set_name(tensor_a, w.a->name);
        ggml_set_name(tensor_b, w.b->name);

        // DoRA: one (d - 1) per output row (per expert for a stacked [n_in, n_out, n_expert] base), filled after the A/B data
        ggml_tensor * tensor_dm1 = nullptr;
        if (auto dm = dora_m_map.find(name); dm != dora_m_map.end()) {
            const ggml_tensor * m = dm->second;
            if (is_token_embd) {
                // PEFT's DoRA embedding variant is a different formula (dora.py:108-127, no (d-1)*base term); the converter refuses it too
                throw std::runtime_error("DoRA on '" + name + "' (token embeddings) is not supported by the DoRA kind");
            }
            if (ggml_n_dims(model_tensor) > 3 || model_tensor->ne[3] != 1) {
                throw std::runtime_error("DoRA on '" + name + "': base tensor has more than 3 dims");
            }
            if (m->type != GGML_TYPE_F32 || m->ne[0] != model_tensor->ne[1] || m->ne[1] != model_tensor->ne[2] || m->ne[2] != 1) {
                throw std::runtime_error(format("DoRA magnitude '%s.dora_m' must be F32 [%lld, %lld], got %s [%lld, %lld, %lld]",
                    name.c_str(), (long long) model_tensor->ne[1], (long long) model_tensor->ne[2], ggml_type_name(m->type),
                    (long long) m->ne[0], (long long) m->ne[1], (long long) m->ne[2]));
            }
            if (w.a->ne[2] != model_tensor->ne[2] || w.b->ne[2] != model_tensor->ne[2]) {
                throw std::runtime_error("DoRA on '" + name + "': lora_a/lora_b expert count differs from the base tensor");
            }
            tensor_dm1 = model_tensor->ne[2] == 1
                ? ggml_new_tensor_1d(dev_ctx, GGML_TYPE_F32, model_tensor->ne[1])
                : ggml_new_tensor_2d(dev_ctx, GGML_TYPE_F32, model_tensor->ne[1], model_tensor->ne[2]);
            ggml_set_name(tensor_dm1, (name + ".dora_dm1").c_str());
        }
        ggml_tensor * tensor_r = nullptr;
        if (lily_r != nullptr) {
            tensor_r = ggml_dup_tensor(dev_ctx, lily_r);
            ggml_set_name(tensor_r, lily_r->name);
        }
        adapter.ab_map[name] = llama_adapter_lora_weight(tensor_a, tensor_b, tensor_dm1, tensor_r);
        adapter.ab_map[name].is_embd = is_token_embd;
    }

    // IA3 vectors: one F32 vector per targeted weight, [n_in] (input side) or [n_out] (output side)
    for (int side = 0; side < 2; ++side) {
        const bool is_in = side == 0;
        for (auto & it : is_in ? ia3_in_map : ia3_out_map) {
            const std::string & name = it.first;
            ggml_tensor * vec = it.second;
            const auto * model_tensor = llama_get_model_tensor(&model, name.c_str());
            if (!model_tensor) {
                throw std::runtime_error("IA3 tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
            }
            if (str_endswith(name, "token_embd.weight") || model_tensor->ne[2] != 1 || model_tensor->ne[3] != 1) {
                throw std::runtime_error("IA3 on '" + name + "': only a 2-D linear weight is supported (PEFT IA3 wraps Linear/Conv only)");
            }
            if (model.hdmd.inverse_blocks.count(name) || model.hadamard_rotation(model_tensor) != nullptr) {
                throw std::runtime_error("IA3 on '" + name + "': the base weight is Hadamard-rotated; the input scale would act on the rotated activation");
            }
            if (vec->type != GGML_TYPE_F32 || ggml_n_dims(vec) != 1 || vec->ne[0] != (is_in ? model_tensor->ne[0] : model_tensor->ne[1])) {
                throw std::runtime_error(format("IA3 vector '%s.%s' must be F32 [%lld] (n_%s of the base weight), got %s [%lld]", name.c_str(),
                    is_in ? "ia3_in" : "ia3_out", (long long) (is_in ? model_tensor->ne[0] : model_tensor->ne[1]), is_in ? "in" : "out",
                    ggml_type_name(vec->type), (long long) vec->ne[0]));
            }
            if (!is_in) {
                // PEFT applies the output scale AFTER the bias (ia3/layer.py:172-201: result = base_layer(x) * l); here the caller adds the
                // bias after the matmul wrapper, so a biased target would be scaled wrongly. Refused loudly.
                std::string bias_name = name;
                if (str_endswith(bias_name, ".weight")) {
                    bias_name.replace(bias_name.size() - 6, 6, "bias");
                    if (llama_get_model_tensor(&model, bias_name.c_str()) != nullptr) {
                        throw std::runtime_error("IA3 output scale on '" + name + "': the base layer has a bias, which PEFT scales too; "
                                                 "this build adds the bias outside the scaled product");
                    }
                }
            }
            auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);
            for (auto & ex : buft_extra) {
                if (ex == buft) {
                    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                    if (!cpu_dev) {
                        throw std::runtime_error(format("%s: no CPU backend found", __func__));
                    }
                    buft = ggml_backend_dev_buffer_type(cpu_dev);
                    break;
                }
            }
            ggml_tensor * dev_vec = ggml_dup_tensor(ctx_for_buft(buft), vec);
            ggml_set_name(dev_vec, vec->name);
            llama_adapter_lora_weight & dw = adapter.ab_map[name];
            (is_in ? dw.ia3_in : dw.ia3_out) = dev_vec;
        }
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            ggml_backend_buffer_type_t buft = it.first;
            ggml_context * ctx_dev = it.second;
            ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft) };
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            adapter.bufs.emplace_back(std::move(buf));
        }
    }

    // set tensor data
    {
        std::vector<uint8_t> read_buf;
        auto read_tensor = [&](const ggml_tensor * orig, std::vector<uint8_t> & dst) {
            const size_t offs = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), gguf_find_tensor(ctx_gguf.get(), orig->name));
            const size_t size = ggml_nbytes(orig);
            if (offs + size < offs || offs + size > gguf_file.size()) {
                throw std::runtime_error(format("LoRA tensor '%s' data is not within the file bounds, file is corrupted or incomplete", orig->name));
            }
            dst.resize(size);
            gguf_file.seek(offs, SEEK_SET);
            gguf_file.read_raw(dst.data(), size);
        };
        auto set_tensor = [&](ggml_tensor * orig, ggml_tensor * dev) {
            read_tensor(orig, read_buf);
            ggml_backend_tensor_set(dev, read_buf.data(), 0, ggml_nbytes(orig));
        };
        int n_dora = 0;
        for (auto & it : adapter.ab_map) {
            if (it.second.is_ia3()) {
                if (it.second.ia3_in  != nullptr) { set_tensor(ia3_in_map.at(it.first),  it.second.ia3_in);  }
                if (it.second.ia3_out != nullptr) { set_tensor(ia3_out_map.at(it.first), it.second.ia3_out); }
                continue;
            }
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
            if (dev.lily_router != nullptr) {
                set_tensor(lily_r_map.at(it.first), dev.lily_router);
            }
            if (dev.dora_dm1 != nullptr) {
                // d = m / ||W + s B A||_row from the dequantised base, s = alpha / rank (adapter scale 1: PEFT's scaling; the
                // converter folded rsLoRA / alpha_pattern into B). A user scale c != 1 later multiplies the whole DoRA delta
                // (LyCORIS `mult` semantics, lycoris/functional/general.py:117-135), so c = 0 is off and c = 1 is PEFT-exact.
                std::vector<uint8_t> raw;
                read_tensor(orig.a, raw);
                const std::vector<float> fa = llama_dora_to_f32(orig.a, raw);
                read_tensor(orig.b, raw);
                const std::vector<float> fb = llama_dora_to_f32(orig.b, raw);
                read_tensor(dora_m_map.at(it.first), raw);
                const std::vector<float> fm = llama_dora_to_f32(dora_m_map.at(it.first), raw);
                const ggml_tensor * base = llama_get_model_tensor(&model, it.first.c_str());
                // The norm must be of the weight the graph EFFECTIVELY multiplies by. Three bases store something else: a
                // Hadamard-rotated weight (the activation is rotated instead, llm_build_lora_mm), an NVFP4 weight whose
                // per-tensor scale is applied to the product, and BitNet's ternary weights scaled by op_params / a separate
                // scale tensor in the graph. Refused loudly rather than normalised against the wrong matrix.
                if (model.hadamard_rotation(base) != nullptr || model.weight_scale(base) != nullptr ||
                    model.arch == LLM_ARCH_BITNET || model.arch == LLM_ARCH_BITNET_25 || model.arch == LLM_ARCH_BITNET_B158) {
                    throw std::runtime_error("DoRA tensor " + it.first + ": the base weight is rotated or externally scaled "
                                             "(Hadamard / NVFP4 scale / BitNet); its row norm is not the norm of the stored tensor");
                }
                const std::vector<float> dm1 = llama_dora_compute_dm1(base, fa, fb, fm, orig.b->ne[0], dev.get_scale(adapter.alpha, 1.0f));
                ggml_backend_tensor_set(dev.dora_dm1, dm1.data(), 0, dm1.size() * sizeof(float));
                ++n_dora;
            }
        }
        if (n_dora > 0) {
            LLAMA_LOG_INFO("%s: DoRA: computed d = m / ||W + s B A||_row for %d tensor(s)\n", __func__, n_dora);
        }
    }

    // Register with the model so ~llama_model frees any adapter still attached.
    // Was commented out by WIRE-66 group 2 because llama_model had no member of this
    // type; the member (and the matching free loop) landed with this change.
    model.loras.insert(&adapter);

    LLAMA_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, kind_ia3 ? ia3_in_map.size() + ia3_out_map.size() : adapter.ab_map.size()*2);
}

llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    FILE * file = ggml_fopen(path_lora, "rb");
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path_lora, strerror(errno));
        return nullptr;
    }

    llama_adapter_lora * adapter = llama_adapter_lora_init_from_file_ptr(model, file);
    fclose(file);

    return adapter;
}

llama_adapter_lora * llama_adapter_lora_init_from_file_ptr(llama_model * model, FILE * file) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }

    llama_adapter_lora * adapter = new llama_adapter_lora(model);

    try {
        llama_adapter_lora_init_impl(*model, file, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());

        delete adapter;
    }

    return nullptr;
}

int32_t llama_adapter_meta_val_str(const llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size) {
    const auto & it = adapter->gguf_kv.find(key);
    if (it == adapter->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_adapter_meta_count(const llama_adapter_lora * adapter) {
    return (int)adapter->gguf_kv.size();
}

int32_t llama_adapter_meta_key_by_index(const llama_adapter_lora * adapter, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_adapter_meta_val_str_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

void llama_adapter_lora_free(llama_adapter_lora * adapter) {
    if (adapter == nullptr) {
        return;
    }

    if (adapter->model != nullptr) {
        adapter->model->loras.erase(adapter);
        adapter->model = nullptr;
    }

    delete adapter;
}

bool llama_adapter_lora_is_draft_gated(const struct llama_adapter_lora * adapter) {
    return adapter != nullptr && adapter->gate == llama_adapter_lora::LLAMA_LORA_GATE_DRAFT_ROWS;
}

uint64_t llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter) {
    if (!adapter) {
        return 0;
    }
    return adapter->alora_invocation_tokens.size();
}

const llama_token * llama_adapter_get_alora_invocation_tokens(const llama_adapter_lora * adapter) {
    GGML_ASSERT(adapter);
    return adapter->alora_invocation_tokens.data();
}
