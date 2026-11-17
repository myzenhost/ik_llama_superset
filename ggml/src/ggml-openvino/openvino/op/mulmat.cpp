#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/op_types.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_input_new(context, 0);
    ov::Output<ov::Node> A = process_view_input_new(context, 1);

    if (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_input_type(1));
    }

    auto B_shape = context.get_input_shape(0).to_shape();
    auto A_shape = context.get_input_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    if (factor > 1 && batch_small > 1) {
        auto batch_large_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        if (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, true);
    }
    if (A_batch_larger) {
        B = Z;
    } else {
        A = Z;
    }

    bool transpose_b = true;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);

    const auto output_type = context.get_output_type();
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_OP_FUSED_UP_GATE (fork op, dense 2D weights): unary(gate . x) * (up . x), src = {up, gate, x}.
// unary is op_params[0] (SILU or RELU; the rest are declined in supports_op).
OutputVector translate_fused_up_gate(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    ov::Output<ov::Node> up = process_view_input_new(context, 0);
    ov::Output<ov::Node> gate = process_view_input_new(context, 1);
    ov::Output<ov::Node> x = process_view_input_new(context, 2);

    if (up.get_element_type() != x.get_element_type()) {
        up = std::make_shared<ov::op::v0::Convert>(up, x.get_element_type());
    }
    if (gate.get_element_type() != x.get_element_type()) {
        gate = std::make_shared<ov::op::v0::Convert>(gate, x.get_element_type());
    }

    ov::Output<Node> up_res = std::make_shared<ov::op::v0::MatMul>(x, up, false, true);
    ov::Output<Node> gate_res = std::make_shared<ov::op::v0::MatMul>(x, gate, false, true);

    const int32_t unary_op = context.get_output_op_params()[0];
    ov::Output<Node> act;
    if (unary_op == GGML_UNARY_OP_SILU) {
        act = std::make_shared<ov::op::v4::Swish>(gate_res);
    } else if (unary_op == GGML_UNARY_OP_RELU) {
        act = std::make_shared<ov::op::v0::Relu>(gate_res);
    } else {
        OPENVINO_THROW("FUSED_UP_GATE: unsupported unary op ", unary_op);
    }

    ov::Output<Node> res = std::make_shared<ov::op::v1::Multiply>(act, up_res);

    const auto output_type = context.get_output_type();
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
