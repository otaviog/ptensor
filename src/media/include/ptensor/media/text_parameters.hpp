#pragma once

#include <string>

namespace p10::media {
/// Text (subtitle) codec type identifier.
class TextCodec {
  public:
    /// Supported text codec types.
    enum CodecType { SubRip, Other };

    TextCodec() = default;

    /// Create codec from type.
    TextCodec(const CodecType& type) : type_(type) {}

    /// Create codec from name string.
    TextCodec(const std::string& codec_name) {
        if (codec_name == "SubRip") {
            type_ = CodecType::SubRip;
        } else {
            type_ = CodecType::Other;
            codec_name_ = codec_name;
        }
    }

    /// Get codec type.
    CodecType type() const {
        return type_;
    }

    /// Get codec name as string.
    std::string to_string() const {
        switch (type_) {
            case CodecType::SubRip:
                return "SubRip";
            case CodecType::Other:
            default:
                return codec_name_;
        }
    }

  private:
    CodecType type_ = CodecType::SubRip;
    std::string codec_name_;
};

/// Parameters describing a single text (subtitle) stream.
///
/// A media file may carry several text streams (e.g. one per metadata kind).
/// Each stream must be declared before the writer opens the file, because the
/// container header is written up front and cannot grow new streams afterwards.
class TextParameters {
  public:
    /// Get the text codec.
    TextCodec codec() const {
        return codec_;
    }

    /// Set the text codec.
    TextParameters& codec(const TextCodec& codec) {
        codec_ = codec;
        return *this;
    }

    /// Get the BCP-47/ISO-639 language tag (e.g. "und", "eng").
    const std::string& language() const {
        return language_;
    }

    /// Set the language tag.
    TextParameters& language(std::string language) {
        language_ = std::move(language);
        return *this;
    }

  private:
    TextCodec codec_;
    std::string language_ = "und";
};
}  // namespace p10::media
