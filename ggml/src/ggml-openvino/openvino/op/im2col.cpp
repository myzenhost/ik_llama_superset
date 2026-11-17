#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <openvino/core/shape.hpp>
#include <openvino/core/strides.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/extractimagepatches.hpp>
#include <openvino/op/pad.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/util/attr_types.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_im2col(const NodeContext & context) {
    num_inputs_check(context, 2, 2);
    const int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t p0 = params[2];
    int32_t p1 = params[3];
    int32_t d0 = params[4];
    int32_t d1 = params[5];
    bool is_2D = params[6] == 1;
    ov::Output<Node> res;

    ov::Output<Node> image = context.get_input(1);
    const ov::Shape kernel_shape = context.get_input(0).get_shape();

    const size_t IC = is_2D ? kernel_shape[1] : kernel_shape[2];
    const size_t KH = is_2D ? kernel_shape[2] : 1;
    const size_t KW = kernel_shape[3];

    int32_t stride_w = s0;
    int32_t stride_h = is_2D ? s1 : 1;
    int32_t pad_w = p0;
    int32_t pad_h = is_2D ? p1 : 0;
    int32_t dil_w = d0;
    int32_t dil_h = is_2D ? d1 : 1;

    if (!is_2D) {
        // GGML input shape: [IW, IC, N, 1]
        // OpenVINO input shape: [1, N, IC, IW]
        // Reshape image to: [N, IC, 1, IW]
        const ov::Shape image_shape = image.get_shape();
        const size_t N = image_shape[1];
        const size_t IW = image_shape[3];
        auto image_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), 1, static_cast<int64_t>(IW)});
        image = std::make_shared<ov::op::v1::Reshape>(image, image_reshape_shape, false);
    }

    const ov::Shape patch_sizes = {KH, KW};
    const ov::Strides strides = {static_cast<size_t>(stride_h), static_cast<size_t>(stride_w)};
    const ov::Shape rates = {static_cast<size_t>(dil_h), static_cast<size_t>(dil_w)};

    // S69-FORK-OPENVINO-IM2COL-DEGENERATE: ggml sizes its output with C truncating division
    // (ggml_calc_conv_output_size, ggml.c), OpenVINO's VALID ExtractImagePatches with floor division. When
    // the numerator is in (-stride, 0) -- the kernel's dilated extent overhangs the padded input -- ggml
    // gets 1 window (every tap out of range or partly in range, zero-filled) and OpenVINO gets 0: a
    // zero-sized tensor that crashed OV with an integer divide-by-zero (Cin=25) or fed uninitialised data
    // through the reshapes (Cin=1). Extend the END pad so exactly ggml's window count exists; whenever
    // the geometry is not degenerate the extension is <= the symmetric pad, so pad_end == pad and the
    // graph is unchanged.
    const ov::Shape image_dims = image.get_shape();   // [N, IC, IH, IW]
    const int64_t IH = static_cast<int64_t>(image_dims[2]);
    const int64_t IW = static_cast<int64_t>(image_dims[3]);
    auto out_len = [](int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride) -> int64_t {
        return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;  // truncating on purpose: matches ggml
    };
    auto end_pad = [](int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride, int64_t out) -> int64_t {
        const int64_t needed = (out - 1) * stride + dil * (k - 1) + 1;  // padded extent the last window needs
        return std::max<int64_t>(pad, needed - in - pad);
    };
    const int64_t ggml_OH = out_len(IH, pad_h, dil_h, static_cast<int64_t>(KH), stride_h);
    const int64_t ggml_OW = out_len(IW, pad_w, dil_w, static_cast<int64_t>(KW), stride_w);
    const int64_t pad_h_end = end_pad(IH, pad_h, dil_h, static_cast<int64_t>(KH), stride_h, ggml_OH);
    const int64_t pad_w_end = end_pad(IW, pad_w, dil_w, static_cast<int64_t>(KW), stride_w, ggml_OW);

    auto pads_begin =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h, pad_w});
    auto pads_end =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, pad_h_end, pad_w_end});

    auto pad = std::make_shared<ov::op::v1::Pad>(image, pads_begin, pads_end, ov::op::PadMode::CONSTANT);
    auto patches =
        std::make_shared<ov::op::v3::ExtractImagePatches>(pad, patch_sizes, strides, rates, ov::op::PadType::VALID);

    // [N, KH*KW*IC, OH, OW] → [N, OH, OW, KH*KW*IC]
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    auto t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);

    // [N, OH, OW, KH*KW*IC] → [N, OH, OW, KH*KW, IC]
    const ov::Shape out_shape = t1->get_output_shape(0);
    const size_t N = out_shape[0];
    const size_t OH = out_shape[1];
    const size_t OW = out_shape[2];
    auto reshape1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, reshape1_shape, false);

    // [N, OH, OW, KH*KW, IC] → [N, OH, OW, IC, KH*KW]
    auto perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
    auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);

    // flatten back to [N, OH, OW, IC*KH*KW]
    auto r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, false);

    if (!is_2D) {
        // [N, 1, OW, IC * KW] -> [1, N, OW, IC * KW]
        auto final_reshape_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{4},
            std::vector<int64_t>{1, static_cast<int64_t>(N), static_cast<int64_t>(OW), static_cast<int64_t>(IC * KW)});
        res = std::make_shared<ov::op::v1::Reshape>(res, final_reshape_shape, false);
    }

    auto output_type = context.get_output_type();
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_im2col_3d(const NodeContext & context) {
    num_inputs_check(context, 2, 2);
    const int32_t * params = context.get_output_op_params();
    int32_t s0 = params[0];
    int32_t s1 = params[1];
    int32_t s2 = params[2];
    int32_t p0 = params[3];
    int32_t p1 = params[4];
    int32_t p2 = params[5];
    int32_t d0 = params[6];
    int32_t d1 = params[7];
    int32_t d2 = params[8];
    int32_t IC = params[9];

    ov::Output<Node> image = process_view_input_new(context, 1);
    const ov::Shape kernel_shape = context.get_input(0).get_shape();
    const ov::Shape image_shape = image.get_shape();
    const ov::Shape out_shape = context.get_output_shape().to_shape();

    const size_t KD = kernel_shape[1];
    const size_t KH = kernel_shape[2];
    const size_t KW = kernel_shape[3];

    const size_t N = image_shape[0] / static_cast<size_t>(IC);
    const size_t ID = image_shape[1];
    const size_t IH = image_shape[2];
    const size_t IW = image_shape[3];

    // S69-FORK-OPENVINO-IM2COL-3D-DEGENERATE-DECLINED-NOT-TRANSLATED: same rule as translate_im2col, per axis.
    // ggml_im2col_3d sizes each axis with ggml_calc_conv_output_size (ggml.c), signed C truncating division,
    // so a numerator in (-stride, 0) -- the dilated kernel overhangs the padded input -- still yields 1
    // window whose out-of-range taps read as 0. The old unsigned arithmetic here underflowed size_t on that
    // geometry (supports_op declined it). Size every axis the ggml way and extend that axis's END pad so the
    // last window exists; on non-degenerate geometry the extension is <= the symmetric pad, so pad_end == pad
    // and the graph is unchanged.
    auto out_len = [](int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride) -> int64_t {
        return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;  // truncating on purpose: matches ggml
    };
    auto end_pad = [](int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride, int64_t out) -> int64_t {
        const int64_t needed = (out - 1) * stride + dil * (k - 1) + 1;  // padded extent the last window needs
        return std::max<int64_t>(pad, needed - in - pad);
    };
    const int64_t OD_i = out_len(static_cast<int64_t>(ID), p2, d2, static_cast<int64_t>(KD), s2);
    const int64_t OH_i = out_len(static_cast<int64_t>(IH), p1, d1, static_cast<int64_t>(KH), s1);
    const int64_t OW_i = out_len(static_cast<int64_t>(IW), p0, d0, static_cast<int64_t>(KW), s0);

    if (N == 0 || OD_i <= 0 || OH_i <= 0 || OW_i <= 0) {
        auto output_type = context.get_output_type();
        ov::Output<Node> res = ov::op::v0::Constant::create(
            output_type,
            ov::Shape{N * static_cast<size_t>(std::max<int64_t>(OD_i, 0)), static_cast<size_t>(std::max<int64_t>(OH_i, 0)),
                      static_cast<size_t>(std::max<int64_t>(OW_i, 0)), static_cast<size_t>(IC * KD * KH * KW)},
            {});
        return rename_outputs_with_suffix({res}, context.get_name());
    }
    const size_t OD = static_cast<size_t>(OD_i);
    const size_t OH = static_cast<size_t>(OH_i);
    const size_t OW = static_cast<size_t>(OW_i);
    const int64_t p2_end = end_pad(static_cast<int64_t>(ID), p2, d2, static_cast<int64_t>(KD), s2, OD_i);
    const int64_t p1_end = end_pad(static_cast<int64_t>(IH), p1, d1, static_cast<int64_t>(KH), s1, OH_i);
    const int64_t p0_end = end_pad(static_cast<int64_t>(IW), p0, d0, static_cast<int64_t>(KW), s0, OW_i);

    const size_t IH_pad = IH + static_cast<size_t>(p1 + p1_end);
    const size_t IW_pad = IW + static_cast<size_t>(p0 + p0_end);

    auto image_5d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(IC), static_cast<int64_t>(ID),
                             static_cast<int64_t>(IH), static_cast<int64_t>(IW)});
    auto image_5d = std::make_shared<ov::op::v1::Reshape>(image, image_5d_shape, false);

    auto pads_begin = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2, p1, p0});
    auto pads_end = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, p2_end, p1_end, p0_end});
    auto pad_3d = std::make_shared<ov::op::v1::Pad>(image_5d, pads_begin, pads_end, ov::op::PadMode::CONSTANT);

    const ov::Shape patch_sizes = {KH, KW};
    const ov::Strides strides = {static_cast<size_t>(s1), static_cast<size_t>(s0)};
    const ov::Shape rates = {static_cast<size_t>(d1), static_cast<size_t>(d0)};

    ov::OutputVector kd_slices;
    kd_slices.reserve(KD);

    auto perm_nod = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 2, 1, 3, 4});
    auto reshape_4d_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(IC),
                             static_cast<int64_t>(IH_pad), static_cast<int64_t>(IW_pad)});
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    auto r1_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{5},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(KH * KW), static_cast<int64_t>(IC)});
    auto perm2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
    auto r2_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{6},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC), 1, static_cast<int64_t>(KH * KW)});
    auto step_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(s2)});
    auto axes_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    for (size_t ikd = 0; ikd < KD; ++ikd) {
        auto start_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2)});
        auto stop_c = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(ikd * d2 + OD * s2)});
        auto depth_slice = std::make_shared<ov::op::v8::Slice>(pad_3d, start_c, stop_c, step_c, axes_c);
        auto depth_slice_trans = std::make_shared<ov::op::v1::Transpose>(depth_slice, perm_nod);
        auto depth_slice_4d = std::make_shared<ov::op::v1::Reshape>(depth_slice_trans, reshape_4d_shape, false);

        auto patches = std::make_shared<ov::op::v3::ExtractImagePatches>(
            depth_slice_4d, patch_sizes, strides, rates, ov::op::PadType::VALID);
        auto t1 = std::make_shared<ov::op::v1::Transpose>(patches, perm1);
        auto r1 = std::make_shared<ov::op::v1::Reshape>(t1, r1_shape, false);
        auto t2 = std::make_shared<ov::op::v1::Transpose>(r1, perm2);
        auto r2 = std::make_shared<ov::op::v1::Reshape>(t2, r2_shape, false);
        kd_slices.push_back(r2);
    }

    ov::Output<Node> res;
    if (KD == 1) {
        res = kd_slices[0];
    } else {
        res = std::make_shared<ov::op::v0::Concat>(kd_slices, 4);
    }

    auto final_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(N * OD), static_cast<int64_t>(OH), static_cast<int64_t>(OW),
                             static_cast<int64_t>(IC * KD * KH * KW)});
    res = std::make_shared<ov::op::v1::Reshape>(res, final_shape, false);

    auto output_type = context.get_output_type();
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
