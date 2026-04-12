#include "bf_postprocess.hpp"

#include <cassert>
#include <cmath>

#include <ptensor/tensor.hpp>

namespace p10::recog {

BfPostprocess::BfPostprocess(const BfAnchorsParameters& anchor_params, Nms nms) :
    anchors_(anchor_params),
    nms_(std::move(nms)) {}

void BfPostprocess::process(
    size_t input_width,
    size_t input_height,
    float preprocess_scale_ratio,
    std::span<const Tensor> model_outputs,
    std::span<FaceDetection> detections
) {
    const auto anchors = anchors_.get_anchors(input_width, input_height);
    const auto anchor_decoder = anchors_.decoder();

    const auto out_boxes = model_outputs[0].as_accessor3d<const float>().unwrap();
    const auto out_scores = model_outputs[1].as_accessor2d<const float>().unwrap();
    const auto out_landmarks = model_outputs[2].as_accessor3d<const float>().unwrap();

    assert(size_t(out_boxes.channels()) == detections.size());

    const float box_scale_x = float(input_width) / preprocess_scale_ratio;
    const float box_scale_y = float(input_height) / preprocess_scale_ratio;

    for (size_t img_idx = 0; img_idx < detections.size(); ++img_idx) {
        rect_buffer_.clear();
        conf_buffer_.clear();
        row_index_buffer_.clear();

        const auto boxes = out_boxes[img_idx];
        const auto scores = out_scores[img_idx];
        const auto landmks = out_landmarks[img_idx];

        for (size_t row_idx = 0; row_idx < size_t(boxes.rows()); ++row_idx) {
            const float conf = scores[row_idx];
            if (conf > conf_threshold_) {
                const auto& anchor = anchors[row_idx];
                const Rect2f rect = anchor_decoder.decode_rect(boxes[row_idx].as_span(), anchor)
                                        .scale(box_scale_x, box_scale_y);
                rect_buffer_.push_back(rect);
                conf_buffer_.push_back(conf);
                row_index_buffer_.push_back(row_idx);
            }
        }

        FaceDetection& result = detections[img_idx];
        result.clear();

        nms_.filter_nms(rect_buffer_, conf_buffer_, selected_);

        result.faces.reserve(selected_.size());
        result.confidences.reserve(selected_.size());
        result.landmarks.reserve(selected_.size());

        const size_t num_landmark_points = size_t(landmks.cols()) / 2;
        landmark_buffer_.resize(num_landmark_points);

        for (const size_t selected_idx : selected_) {
            const size_t anchor_row = row_index_buffer_[selected_idx];
            const auto& anchor = anchors[anchor_row];

            result.confidences.push_back(conf_buffer_[selected_idx]);
            result.faces.push_back(rect_buffer_[selected_idx].to<int>());

            anchor_decoder
                .decode_landmarks(landmks[anchor_row].as_span(), anchor, landmark_buffer_);

            result.landmarks.push_back({});
            auto& face_landmarks = result.landmarks.back();
            face_landmarks.reserve(num_landmark_points);
            for (const auto& pt : landmark_buffer_) {
                face_landmarks.push_back(Point2i(
                    int(std::round(pt.x * box_scale_x)),
                    int(std::round(pt.y * box_scale_y))
                ));
            }
        }
    }
}

}  // namespace p10::recog
