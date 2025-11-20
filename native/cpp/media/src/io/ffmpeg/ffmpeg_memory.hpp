#pragma once
#include <memory>
extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
}

namespace p10::media {
namespace detail {
    inline constexpr auto delete_packet = [](AVPacket* ptr) { av_packet_free(&ptr); };
    inline constexpr auto delete_frame = [](AVFrame* ptr) { av_frame_free(&ptr); };
    inline constexpr auto unref_frame = [](AVFrame* ptr) { av_frame_unref(ptr); };
    inline constexpr auto unref_packet = [](AVPacket* ptr) { av_packet_unref(ptr); };
}  // namespace detail

using UniqueAvFrame = std::unique_ptr<AVFrame, decltype(detail::delete_frame)>;
using UniqueAvFrameRef = std::unique_ptr<AVFrame, decltype(detail::unref_frame)>;
using UniqueAvPacket = std::unique_ptr<AVPacket, decltype(detail::delete_packet)>;
using UniqueAvPacketRef = std::unique_ptr<AVPacket, decltype(detail::unref_packet)>;

}  // namespace p10::media
