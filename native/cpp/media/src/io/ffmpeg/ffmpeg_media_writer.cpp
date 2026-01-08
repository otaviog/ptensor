#include "ffmpeg_media_writer.hpp"

#include "audio_frame.hpp"
#include "video_frame.hpp"

namespace p10::media {
static P10Result<std::shared_ptr<FfmpegMediaWriter>>
FfmpegMediaWriter::create(const std::string& path, const MediaParameters& params) {
    const auto* videoEncoder = avcodec_find_encoder(AV_CODEC_ID_H264);

    if (videoEncoder == nullptr) {
        return Err(P10Error::InvalidOperation << "Could not find H264 encoder");
    }
    m_videoStream = avformat_new_stream(m_outputFormatContext, videoEncoder);
    if (m_videoStream == nullptr) {
        return Err(P10Error::InvalidOperation << "Could not add video stream");
    }
    m_videoStream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    m_videoEncoderContext = avcodec_alloc_context3(videoEncoder);

    // Quality
    m_videoEncoderContext->bit_rate = outputOptions.videoBitRate();
    m_videoEncoderContext->rc_buffer_size = 4 * 1000 * 1000;
    m_videoEncoderContext->rc_max_rate = 2 * 1000 * 1000;
    m_videoEncoderContext->rc_min_rate = int64_t(2.5 * 1000.0 * 1000.0);
    m_videoEncoderContext->gop_size = 12;
    av_opt_set(m_videoEncoderContext->priv_data, "preset", "slower", 0);
    // Imagining
    m_videoEncoderContext->pix_fmt = AV_PIX_FMT_YUV420P;
    m_videoEncoderContext->width = outputOptions.videoResolution().first;
    m_videoEncoderContext->height = outputOptions.videoResolution().second;
    // Timing
    const auto fpsNum = static_cast<int>(outputOptions.videoFrameRate());
    const AVRational timeBase {1, fpsNum};
    const AVRational fps {fpsNum, 1};
    m_videoStream->time_base = timeBase;
    m_videoEncoderContext->time_base = timeBase;
    m_videoStream->avg_frame_rate = fps;

    m_toVideoPts = FfmpegVideoPtsGenerator(timeBase, fps);

    errorOnFail(
        avcodec_open2(m_videoEncoderContext, videoEncoder, nullptr),
        "Could not open video encoder"
    );

    errorOnFail(
        avcodec_parameters_from_context(m_videoStream->codecpar, m_videoEncoderContext),
        "Could not initialize video stream parameters"
    );

    m_videoRescaler = FfmpegVideoRescaler(
        outputOptions.videoResolution().first,
        outputOptions.videoResolution().second,
        AV_PIX_FMT_YUV420P
    );

    /******************************
     * Audio part
     ******************************/
    m_audioStream = avformat_new_stream(m_outputFormatContext, nullptr);
    if (m_audioStream == nullptr) {
        errorOnFail(std::nullopt, "Could not audio add stream");
    }

    const auto* audioCodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (audioCodec == nullptr) {
        errorOnFail(std::nullopt, "Could not find audio encoder");
    }

    m_audioEncoderContext = avcodec_alloc_context3(audioCodec);
    m_audioEncoderContext->bit_rate = outputOptions.audioBitRate();
    m_audioEncoderContext->sample_rate = outputOptions.audioSampleRate();
    m_audioEncoderContext->sample_fmt = audioCodec->sample_fmts[0];
    av_channel_layout_default(&m_audioEncoderContext->ch_layout, outputOptions.audioChannels());
    // Time
    const AVRational audioTimeBase {1, outputOptions.audioSampleRate()};
    m_audioStream->time_base = audioTimeBase;
    m_audioEncoderContext->time_base = audioTimeBase;
    m_audioEncoderContext->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
    if (m_outputFormatContext->oformat->flags & AVFMT_GLOBALHEADER)
        m_audioEncoderContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    errorOnFail(
        avcodec_open2(m_audioEncoderContext, audioCodec, nullptr),
        "Could not open audio encoder"
    );

    errorOnFail(
        avcodec_parameters_from_context(m_audioStream->codecpar, m_audioEncoderContext),
        "Could not initialize audio stream parameters"
    );

    m_audioResampler = FfmpegAudioResampler(
        m_audioEncoderContext->ch_layout,
        m_audioEncoderContext->sample_fmt,
        m_audioEncoderContext->sample_rate
    );
    m_audioFifo = FfmpegAudioFifo(
        m_audioEncoderContext->ch_layout,
        m_audioEncoderContext->sample_fmt,
        m_audioEncoderContext->sample_rate
    );  // Bug, invert fifo and resampler.

    /******************************
     * Container
     ******************************/
    errorOnFail(
        avio_open(&m_outputFormatContext->pb, uri.c_str(), AVIO_FLAG_WRITE),
        "Could not initialize stream parameters"
    );

    errorOnFail(
        avformat_write_header(m_outputFormatContext, nullptr),
        "Could not initialize stream parameters"
    );
}

FfmpegMediaWriter::~FfmpegMediaWriter() override;

void FfmpegMediaWriter::close() override;

MediaParameters FfmpegMediaWriter::get_parameters() const override;

P10Error FfmpegMediaWriter::write_video(const VideoFrame& frame) override;
P10Error FfmpegMediaWriter::write_audio(const AudioFrame& frame) override;
}  // namespace p10::media