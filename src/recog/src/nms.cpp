#include "nms.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace p10::recog {

void Nms::filter_nms(
    std::span<const Rect2f> rects,
    std::span<const float> confidences,
    std::vector<size_t>& selected
) {
    const auto rect_count = rects.size();
    assert(rect_count == confidences.size());

    std::vector<size_t> rect_indices_heap;
    rect_indices_heap.resize(rect_count);
    std::iota(rect_indices_heap.begin(), rect_indices_heap.end(), 0);
    std::vector<bool> suppressed(rect_count, false);

    const auto confidence_compare = [&](size_t lfs, size_t rhs) {
        return confidences[lfs] < confidences[rhs];
    };
    std::make_heap(rect_indices_heap.begin(), rect_indices_heap.end(), confidence_compare);

    selected.clear();
    while (!rect_indices_heap.empty()) {
        std::pop_heap(rect_indices_heap.begin(), rect_indices_heap.end(), confidence_compare);
        auto selected_index = rect_indices_heap.back();
        rect_indices_heap.pop_back();

        if (suppressed[selected_index]) {
            continue;
        }

        const auto selected_rect = rects[selected_index];
        for (const auto other_idx : rect_indices_heap) {
            if (!suppressed[other_idx] && selected_rect.iou(rects[other_idx]) > iou_threshold_) {
                suppressed[other_idx] = true;
            }
        }

        selected.push_back(selected_index);
    }
}

}  // namespace p10::recog
