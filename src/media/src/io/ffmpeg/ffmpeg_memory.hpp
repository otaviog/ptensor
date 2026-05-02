#pragma once
#include <memory>
extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
}

namespace p10::media {
namespace detail {
    inline constexpr auto DELETE_PACKET = [](AVPacket* ptr) { av_packet_free(&ptr); };
    inline constexpr auto DELETE_FRAME = [](AVFrame* ptr) { av_frame_free(&ptr); };
    inline constexpr auto UNREF_FRAME = [](AVFrame* ptr) { av_frame_unref(ptr); };
    inline constexpr auto UNREF_PACKET = [](AVPacket* ptr) { av_packet_unref(ptr); };
}  // namespace detail

using UniqueAvFrame = std::unique_ptr<AVFrame, decltype(detail::DELETE_FRAME)>;
using UniqueAvFrameRef = std::unique_ptr<AVFrame, decltype(detail::UNREF_FRAME)>;
using UniqueAvPacket = std::unique_ptr<AVPacket, decltype(detail::DELETE_PACKET)>;
using UniqueAvPacketRef = std::unique_ptr<AVPacket, decltype(detail::UNREF_PACKET)>;

}  // namespace p10::media
