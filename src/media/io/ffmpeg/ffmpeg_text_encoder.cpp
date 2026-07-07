#include "ffmpeg_text_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <ptensor/p10_error.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
}

#include "ffmpeg_wrap_error.hpp"
#include "text.hpp"
#include "text_parameters.hpp"

namespace p10::media {

namespace {
    AVCodecID codec_id_from_text_parameters(const TextParameters& text_params);
    int64_t to_stream_stamp(const Time& time, const Rational& time_base);
}  // namespace

P10Error
FfmpegTextEncoder::create(const TextParameters& text_params, AVFormatContext* output_format) {
    const AVCodecID codec_id = codec_id_from_text_parameters(text_params);
    if (codec_id == AV_CODEC_ID_NONE) {
        return P10Error::IoError << std::string("Unsupported text codec ")
            + text_params.codec().to_string();
    }

    // Reject the codec early if the target container cannot mux it. Otherwise
    // this only surfaces as a cryptic failure deep inside avformat_write_header
    // (e.g. MP4 has no place for raw SubRip text). Point the user at Matroska.
    if (avformat_query_codec(output_format->oformat, codec_id, FF_COMPLIANCE_NORMAL) <= 0) {
        const char* container = output_format->oformat->name != nullptr
            ? output_format->oformat->name
            : "this container";
        return P10Error::InvalidArgument
            << std::string("Container '") + container + "' does not support "
            + text_params.codec().to_string()
            + " text streams. Use a Matroska (.mkv) output for text streams.";
    }

    stream_ = avformat_new_stream(output_format, nullptr);
    if (stream_ == nullptr) {
        return P10Error::InvalidOperation << "Could not add text stream";
    }

    // Raw text subtitles need no codec context: the cue text is the packet
    // payload and the interval is carried by pts/duration.
    stream_->codecpar->codec_type = AVMEDIA_TYPE_SUBTITLE;
    stream_->codecpar->codec_id = codec_id;
    stream_->time_base = AVRational {
        .num = static_cast<int>(time_base_.num()),
        .den = static_cast<int>(time_base_.den()),
    };

    av_dict_set(&stream_->metadata, "language", text_params.language().c_str(), 0);

    return P10Error::Ok;
}

P10Result<UniqueAvPacket> FfmpegTextEncoder::encode(const Text& text) const {
    if (stream_ == nullptr) {
        return Err(P10Error::InvalidOperation << "Text stream not created");
    }

    const std::string& payload = text.text();
    UniqueAvPacket pkt(av_packet_alloc());
    if (!pkt) {
        return Err(P10Error::OutOfMemory << "Could not allocate text packet");
    }

    P10_RETURN_ERR_IF_ERROR(
        wrap_ffmpeg_error(av_new_packet(pkt.get(), static_cast<int>(payload.size())))
    );
    std::memcpy(pkt->data, payload.data(), payload.size());

    const int64_t begin_stamp = to_stream_stamp(text.begin(), time_base_);
    const int64_t duration =
        std::max<int64_t>(0, to_stream_stamp(text.end(), time_base_) - begin_stamp);

    pkt->stream_index = stream_->index;
    pkt->pts = begin_stamp;
    pkt->dts = begin_stamp;
    pkt->duration = duration;
    pkt->flags |= AV_PKT_FLAG_KEY;

    return Ok(std::move(pkt));
}

namespace {
    AVCodecID codec_id_from_text_parameters(const TextParameters& text_params) {
        switch (text_params.codec().type()) {
            case TextCodec::CodecType::SubRip:
                return AV_CODEC_ID_SUBRIP;
            default:
                return AV_CODEC_ID_NONE;
        }
    }

    int64_t to_stream_stamp(const Time& time, const Rational& time_base) {
        // Route through seconds so any source timebase (including an unset one)
        // maps cleanly onto the stream timebase (stamps per second = den/num).
        const double seconds = time.to_seconds();
        const double stamps_per_second =
            static_cast<double>(time_base.den()) / static_cast<double>(time_base.num());
        return std::llround(seconds * stamps_per_second);
    }
}  // namespace

}  // namespace p10::media
