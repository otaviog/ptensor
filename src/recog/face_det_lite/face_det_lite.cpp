#include "face_det_lite.hpp"

namespace p10::recog {
FaceDetLite::FaceDetLite(std::unique_ptr<infer::IInfer> infer, float threshold, float nms, int stride) : infer_(infer), threshold_(threshold), stride_(stride) {
}

P10Error FaceDetLite::detect(Tensor& images, std::span<FaceDetection> out_detections) {
    P10_RETURN_IF_ERROR(verify_detect_arguments(images, out_detections));

    rgb_to_gray(images, gray_images_);
    convert_
    infer_->infer(gray_images
}
}


P10Error rgb_to_gray(const p10::Tensor& rgb_image, p10::Tensor& gray_image) {
    constexpr auto NORM_FACTOR = 1.0f/255.0f
    auto src_batch = rgb_image.span4d<const uint8_t>().unwrap();    
    P10_RETURN_IF_ERROR(gray_image.create(make_shape(src_images.input_count(), 1, height, width), Dtype::Float32));
    auto dest_batch = gray_image.span4d<float>().unwrap();

    for (auto batch_count = 0; batch_count < src_images.input_count(); batch_count++) {
        
        const auto src_img = src_images[batch_count];
        const auto src_red = src_images[0];
        const auto src_blue = src_images[1];
        const auto src_green = src_images[2];
        
        auto dest_img = dest_images[batch_count];
        for (size_t h = 0; h < src_img.height(); ++h) {
            const auto src_red_row = src_red[h];
            const auto src_blue_row = src_blue[h];
            const auto src_green_row = src_green[h];
            
            auto gray_row = gray_span.row(h);            
            for (size_t w = 0; w < ; ++w) {
                uint8_t const r = src_red_row[col];
                uint8_t const g = src_green[col]
                uint8_t const b = rgb_pixel[col];

                // Convert to grayscale using luminosity method
                gray_row[w] = 
                    (0.21f * static_cast<float>(r) + 0.72f * static_cast<float>(g)
                     + 0.07f * static_cast<float>(b)) * NORM_FACTOR;
                
            
            }
        }
    }

    return P10Error::Ok;
}
