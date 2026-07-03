#include "ffmpeg_text_decoder.hpp"

#include <string>
#include <utility>

#include "ffmpeg_memory.hpp"
#include "ffmpeg_wrap_error.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
}

#include "time/rational.hpp"
#include "time/time.hpp"

namespace p10::media {
namespace {
    P10Result<std::vector<std::vector<Text>>>
    collect_text_cues(AVFormatContext* ctx, const std::vector<int>& stream_indices);

    TextParameters describe_text_stream(const AVStream* stream);
}  // namespace

void FfmpegTextDecoder::set_source(std::string url, std::vector<int> stream_indices) {
    source_url_ = std::move(url);
    stream_indices_ = std::move(stream_indices);
}

std::vector<TextParameters> FfmpegTextDecoder::describe(const AVFormatContext* format_ctx) const {
    std::vector<TextParameters> params;
    params.reserve(stream_indices_.size());
    for (const int stream_idx : stream_indices_) {
        params.push_back(describe_text_stream(format_ctx->streams[stream_idx]));
    }
    return params;
}

P10Result<TextStreams> FfmpegTextDecoder::get_text_streams() const {
    P10_RETURN_ERR_IF_ERROR(ensure_scanned());
    return Ok(TextStreams(streams_));
}

P10Error FfmpegTextDecoder::ensure_scanned() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (scanned_) {
        return P10Error::Ok;
    }
    scanned_ = true;

    if (stream_indices_.empty() || source_url_.empty()) {
        return P10Error::Ok;
    }

    // One extra demux pass, paid only once and only when text is requested.
    AVFormatContext* scan_ctx = nullptr;
    P10_RETURN_IF_ERROR(
        wrap_ffmpeg_error(avformat_open_input(&scan_ctx, source_url_.c_str(), nullptr, nullptr))
    );

    auto cues = collect_text_cues(scan_ctx, stream_indices_);
    avformat_close_input(&scan_ctx);

    if (cues.is_error()) {
        return cues.error();
    }
    streams_ = std::move(cues).unwrap();
    return P10Error::Ok;
}

namespace {
    P10Result<std::vector<std::vector<Text>>>
    collect_text_cues(AVFormatContext* ctx, const std::vector<int>& stream_indices) {
        P10_RETURN_ERR_IF_ERROR(
            wrap_ffmpeg_error(avformat_find_stream_info(ctx, nullptr), "Failed to read stream info")
        );

        std::vector<std::vector<Text>> cues(stream_indices.size());

        while (true) {
            UniqueAvPacketRef pkt(av_packet_alloc());
            const int ret = av_read_frame(ctx, pkt.get());
            if (ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                return Err(wrap_ffmpeg_error(ret, "Failed to scan text streams"));
            }

            for (size_t group = 0; group < stream_indices.size(); ++group) {
                if (pkt->stream_index != stream_indices[group]) {
                    continue;
                }
                const AVStream* stream = ctx->streams[pkt->stream_index];
                const Rational base {stream->time_base.num, stream->time_base.den};
                const int64_t begin_stamp = pkt->pts == AV_NOPTS_VALUE ? 0 : pkt->pts;
                const int64_t end_stamp = begin_stamp + (pkt->duration > 0 ? pkt->duration : 0);
                std::string payload(reinterpret_cast<const char*>(pkt->data), pkt->size);
                cues[group].emplace_back(
                    std::move(payload),
                    Time {base, begin_stamp},
                    Time {base, end_stamp}
                );
                break;
            }
        }

        return Ok(std::move(cues));
    }

    TextParameters describe_text_stream(const AVStream* stream) {
        TextParameters params;
        if (stream->codecpar->codec_id == AV_CODEC_ID_SUBRIP) {
            params.codec(TextCodec(TextCodec::SubRip));
        } else {
            params.codec(TextCodec(std::string(avcodec_get_name(stream->codecpar->codec_id))));
        }

        const AVDictionaryEntry* language = av_dict_get(stream->metadata, "language", nullptr, 0);
        if (language != nullptr && language->value != nullptr) {
            params.language(language->value);
        }
        return params;
    }
}  // namespace

}  // namespace p10::media
