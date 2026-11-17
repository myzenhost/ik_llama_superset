#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {
static ov::Output<ov::Node> reshape_flat_kv(const ov::Output<ov::Node> & kv_flat,
                                            size_t view_offset_bytes,
                                            size_t nb1_bytes,
                                            int64_t n_head,
                                            int64_t head_size,
                                            const ov::Output<ov::Node> & attention_size) {
    int64_t n_state = n_head * head_size;
    int64_t layer_start_elem = (int64_t) (view_offset_bytes / (nb1_bytes / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    auto start_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_start_elem});
    auto n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    auto kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_size, n_state_c);
    auto end_c = std::make_shared<ov::op::v1::Add>(start_c, kv_len_elems);
    auto step_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto sliced = std::make_shared<ov::op::v8::Slice>(kv_flat, start_c, end_c, step_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    auto one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto n_head_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_head});
    auto head_size_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_size});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    auto new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one_c, attention_size, n_head_c, head_size_c}, 0);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, false);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    auto ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    return ret;
}

OutputVector translate_flash_attn_ext(const NodeContext & context) {
    num_inputs_check(context, 3, 4);
    const bool has_mask = context.get_input_size() == 4;
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    const int op_case = context.get_op_case();

    if (op_case == 1 || op_case == 2) {
        int64_t n_state_head = (int64_t) context.get_view_input_ggml_shape(1, 0)[3];
        int64_t n_head = (int64_t) context.get_view_input_ggml_shape(1, 0)[1];
        size_t nb1 = context.get_view_input_stride(1, 0)[2];
        size_t offset = context.get_view_input_offset(1, 0);
        ov::Output<ov::Node> attention_size;
        if (op_case == 1) {
            attention_size = context.get_input("attention_size");
        } else {
            attention_size = context.get_input("attention_size_static");
        }
        k = reshape_flat_kv(k, offset, nb1, n_head, n_state_head, attention_size);
        v = reshape_flat_kv(v, offset, nb1, n_head, n_state_head, attention_size);
    }

    // op_case 3: this fork's cache (GgmlOvDecoder::is_fork_kv_cache_view). K and V arrive as the whole cache
    // tensor (a VIEW resolves to its base, and a CPY op_case 11 earlier in the graph has replaced that base with
    // the cache holding this ubatch's rows); the view's own ggml metadata gives the cell window [offset, n_kv)
    // and the head count. Memory is [cell][head][d], the layout reshape_flat_kv already slices.
    int64_t fork_n_head_kv = 0;
    int64_t fork_n_kv = 0;
    if (op_case == 3) {
        auto flat4 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        auto fork_view = [&](size_t idx, const ov::Output<ov::Node> & base) -> ov::Output<ov::Node> {
            auto shape = context.get_view_input_ggml_shape(idx, 0);   // [1, n_head_kv, n_kv, d]
            auto stride = context.get_view_input_stride(idx, 0);      // [nb3, nb2, nb1, nb0]
            size_t offset = context.get_view_input_offset(idx, 0);
            FRONT_END_OP_CONVERSION_CHECK(shape.size() == 4 && stride.size() == 4,
                                          "FLASH_ATTN_EXT op_case 3: K/V view metadata missing");
            auto n_kv_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) shape[2]});
            // Base written earlier in this graph by translate_cpy op_case 11:
            //   Reshape(ScatterUpdate(cache_flat, Range(start, start + n), rows_flat), ShapeOf(cache)).
            // That ScatterUpdate copies the whole cache; read only the [offset, n_kv) window instead, as
            // cache[w0, start) ++ rows ++ cache[start + n, w1). The fork writes inside the window it reads
            // (n_kv covers every used cell); a write outside it makes the concatenation the wrong length and
            // the fixed-shape Reshape below throws, so a violated assumption fails loudly.
            auto splice_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(base.get_node_shared_ptr());
            auto splice = splice_reshape ? ov::as_type_ptr<ov::op::v3::ScatterUpdate>(
                                               splice_reshape->get_input_node_shared_ptr(0))
                                         : nullptr;
            auto splice_range =
                splice ? ov::as_type_ptr<ov::op::v4::Range>(splice->get_input_node_shared_ptr(1)) : nullptr;
            if (splice_range) {
                const int64_t n_head = (int64_t) shape[1];
                const int64_t head_size = (int64_t) shape[3];
                const int64_t n_kv = (int64_t) shape[2];
                const int64_t n_state = n_head * head_size;
                const int64_t elem_size = (int64_t) stride[2] / n_state;
                const int64_t w0 = (int64_t) offset / elem_size;
                const int64_t w1 = w0 + n_kv * n_state;
                auto axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto step1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto w0_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {w0});
                auto w1_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {w1});
                auto cache_flat = splice->input_value(0);
                auto rows_flat = splice->input_value(2);
                auto start1 = std::make_shared<ov::op::v0::Unsqueeze>(splice_range->input_value(0), axis0);
                auto end1 = std::make_shared<ov::op::v1::Add>(
                    start1, std::make_shared<ov::op::v3::ShapeOf>(rows_flat, ov::element::i64));
                // On the CPU device the cache input is bound with only the cells the attention reads, [0, w1)
                // (try_make_kv_sliced_tensor), so a window that starts at 0 IS the spliced cache: reading it from the
                // ScatterUpdate output skips the head/tail Slice pair and the Concat. The CPU plugin runs those two
                // Slices with its reference kernel (profile exec_type ref_f32, ~170 us each per layer and per K/V at
                // n_kv 256), which made the one-graph decode slower than the split fragments. That holds only when the
                // flat cache is dynamic (bound sliced) or already exactly w1 long; a cache of any other static length
                // (test_fork_kv_attn d 256, n_ctx 512 > n_kv 256), GGML_OPENVINO_DISABLE_KV_SLICE=1 and a non-CPU device
                // keep the Slice/Concat form, as does a window that starts later (sliding attention).
                // GGML_OPENVINO_LEGACY_KV_WINDOW=1 restores it for A/B.
                static const bool legacy_window = ggml_openvino_getenv_int("GGML_OPENVINO_LEGACY_KV_WINDOW") != 0;
                static const bool slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") != 0;
                bool cache_is_window = false;
                {
                    const auto cache_ps = cache_flat.get_partial_shape();
                    if (cache_ps.rank().is_static() && cache_ps.size() == 1) {
                        cache_is_window = cache_ps[0].is_dynamic()
                                              ? (!slice_disabled && ggml_openvino_get_device_name().rfind("CPU", 0) == 0)
                                              : (int64_t) cache_ps[0].get_length() == w1;
                    }
                }
                ov::Output<ov::Node> window;
                if (w0 == 0 && cache_is_window && !legacy_window) {
                    window = splice->output(0);
                } else {
                    auto head = std::make_shared<ov::op::v8::Slice>(cache_flat, w0_c, start1, step1, axis0);
                    auto tail = std::make_shared<ov::op::v8::Slice>(cache_flat, end1, w1_c, step1, axis0);
                    window = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head, rows_flat, tail}, 0);
                }
                auto window4 = std::make_shared<ov::op::v1::Reshape>(
                    window,
                    ov::op::v0::Constant::create(ov::element::i64, {4},
                                                 std::vector<int64_t>{1, n_kv, n_head, head_size}),
                    false);
                auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
                return std::make_shared<ov::op::v1::Transpose>(window4, perm);
            }
            auto flat = std::make_shared<ov::op::v1::Reshape>(base, flat4, false);
            return reshape_flat_kv(flat, offset, stride[2], (int64_t) shape[1], (int64_t) shape[3], n_kv_c);
        };
        auto k_shape_view = context.get_view_input_ggml_shape(1, 0);
        fork_n_head_kv = (int64_t) k_shape_view[1];
        fork_n_kv = (int64_t) k_shape_view[2];
        k = fork_view(1, k);
        v = fork_view(2, v);
        // is_plain_kv_base admits an f32 cache (--cache-type-k/v f32) as well as f16. SDPA requires one element
        // type for Q, K, V, mask and scale, and Q, mask and scale are f16 below: an f32 cache failed validation
        // with "Mixed input types are not supported" (test_fork_kv_attn type_kv=f32, 12/12 cases).
        if (k.get_element_type() != ov::element::f16) {
            k = std::make_shared<ov::op::v0::Convert>(k, ov::element::f16);
        }
        if (v.get_element_type() != ov::element::f16) {
            v = std::make_shared<ov::op::v0::Convert>(v, ov::element::f16);
        }
    }

    float * params = reinterpret_cast<float *>(context.get_output_op_params());
    float scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<float>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> mask;
    if (has_mask) {
        mask = context.get_input(3);
        std::string mask_name = "KQ_mask_sliced";
        if (context.get_input_names()[3].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        if (op_case != 3 && context.has_input(mask_name)) {
            mask = context.get_input(mask_name);
        }
        if (op_case == 3) {
            // ggml pads the mask's query rows to GGML_KQ_MASK_PAD (ggml.h: mask [n_kv, n_batch_pad]); SDPA needs
            // exactly [.., S_q, n_kv]. Rows past S_q are padding, columns past n_kv cannot exist here.
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto s_q = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(q_f32, ov::element::i64),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {2}), zero);
            auto stop = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{s_q, ov::op::v0::Constant::create(ov::element::i64, {1}, {fork_n_kv})}, 0);
            mask = std::make_shared<ov::op::v8::Slice>(
                mask, ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 0}), stop,
                ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{1, 1}),
                ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{2, 3}));
        }
        if (mask.get_element_type() != ov::element::f16) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    auto q_shape = context.get_input_shape(0).to_shape();
    const auto k_shape = op_case == 3 ? ov::Shape{} : context.get_input_shape(1).to_shape();
    const int64_t num_heads = q_shape[1];
    // op_case 3 hands over the whole cache as input 1; the head count is the view's, not that tensor's.
    const int64_t num_heads_kv = op_case == 3 ? fork_n_head_kv : (int64_t) k_shape[1];
    const int64_t head_size = q_shape[3];
    const int64_t factor = num_heads / num_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    static const bool manual_gqa_enabled = []() {
        const char * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        if (env != nullptr) {
            return ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        return ggml_openvino_is_gpu();
    }();
    const bool use_manual_gqa_attention =
        manual_gqa_enabled && factor > 1 && num_heads_kv > 1 && !context.is_stateful();

    if (use_manual_gqa_attention) {
        // Q, K, V arrive as [B, n_heads(_kv), S, head_size], where B is the active
        // batch (n_seq_active) and may be > 1 (llama-perplexity, llama-server -np > 1)
        // or dynamic. Reshape to
        //   K_r: [B, num_heads_kv, 1, S, head_size]
        //   Q_r: [B, num_heads_kv, factor, S_q, head_size]
        // and let MatMul broadcast across the factor dim without materialising
        // an expanded K/V. The leading 0 + special_zero=true copies B at runtime,
        // so this is correct for B == 1, B > 1, and dynamic B alike. Only the head
        // dims and head_size are baked in as literals; the sequence dim stays -1.
        auto k_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, num_heads_kv, 1, -1, head_size});
        auto v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, num_heads_kv, 1, -1, head_size});
        auto q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, num_heads_kv, factor, -1, head_size});

        auto k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, true);
        auto v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, true);
        auto q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, true);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        auto qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/false, /*tB=*/true);
        auto qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_node);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        if (has_mask) {
            auto mask_unsq1 =
                std::make_shared<ov::op::v0::Unsqueeze>(mask, ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            qk_masked = std::make_shared<ov::op::v1::Add>(qk_scaled, mask_unsq1);
        } else {
            qk_masked = qk_scaled;
        }

        auto softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [B, num_heads_kv, factor, S_q, head_size]
        auto attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [B, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        // Leading 0 + special_zero=true copies B at runtime.
        auto out_4d_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, num_heads, -1, head_size});
        auto out_4d = std::make_shared<ov::op::v1::Reshape>(attn, out_4d_shape, true);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            out_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    auto tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        if (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_broadcast_shape;
            ov::Output<ov::Node> kv_unsqueezed;
            ov::Output<ov::Node> new_kv_shape;
            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_broadcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                              {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed, kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);
        }
        return kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(num_heads, num_heads_kv, head_size, k);
    v = tile_kv(num_heads, num_heads_kv, head_size, v);

    constexpr auto causal = false;
    if (has_mask) {
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, scale_node, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } else {
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, scale_node, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
