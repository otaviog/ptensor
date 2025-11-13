#include "audio.hpp"

#include <AudioFile/AudioFile.h>
#include <ptensor/tensor.hpp>

namespace p10::io {
P10Error load_audio(const std::string& path, Tensor& tensor, int64_t& sample_rate, Dtype dtype) {
    return dtype.match([&path, &sample_rate, &tensor, dtype](auto scalar) -> P10Error {
        using scalar_t = typename decltype(scalar)::type;

        AudioFile<scalar_t> audioFile;
        if (!audioFile.load(path)) {
            return P10Error::IoError << "Failed to load audio file: " + path;
        }

        sample_rate = static_cast<int64_t>(audioFile.getSampleRate());
        size_t num_channels = audioFile.getNumChannels();
        size_t num_samples = audioFile.getNumSamplesPerChannel();

        tensor.create(make_shape(num_samples, num_channels), dtype);
        auto tensor_s = tensor.as_span2d<scalar_t>().unwrap();

        for (size_t channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
            auto channelData = audioFile.samples[channel_idx];
            auto tensor_row = tensor_s.row(channel_idx);
            for (size_t s = 0; s < num_samples; ++s) {
                tensor_row[s] = channelData[s];
            }
        }

        return P10Error::Ok;
    });
}

P10Error save_audio(const std::string& path, const Tensor& tensor, int64_t sample_rate) {
    return tensor.visit([&path, &sample_rate, &tensor](auto data) -> P10Error {
        using scalar_t = typename decltype(data)::value_type;

        auto span = tensor.as_span2d<scalar_t>();
        if (span.is_error()) {
            return P10Error::InvalidArgument << "Tensor must be 2D for audio saving.";
        }

        size_t num_samples = span.unwrap().width();
        size_t num_channels = span.unwrap().height();

        AudioFile<scalar_t> audioFile;
        audioFile.setNumChannels(static_cast<int>(num_channels));
        audioFile.setNumSamplesPerChannel(static_cast<int>(num_samples));
        audioFile.setSampleRate(static_cast<int>(sample_rate));

        for (size_t channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
            std::vector<scalar_t> channelData(num_samples);
            auto tensor_row = span.unwrap().row(channel_idx);
            for (size_t s = 0; s < num_samples; ++s) {
                channelData[s] = tensor_row[s];
            }
            audioFile.samples[channel_idx] = channelData;
        }

        if (!audioFile.save(path)) {
            return P10Error::IoError << "Failed to save audio file: " + path;
        }

        return P10Error::Ok;
    });
}
}  // namespace p10::io
