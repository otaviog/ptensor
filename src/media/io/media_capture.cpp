#include "io/media_capture.hpp"

#include <string>
#include <utility>

#include "camera_controls.hpp"
#include "ffmpeg/ffmpeg_device_media_capture.hpp"
#include "ffmpeg/ffmpeg_file_media_capture.hpp"

namespace p10::media {

P10Result<MediaCapture> MediaCapture::open_file(const std::string& path) {
    auto result = FfmpegFileMediaCapture::open(path);
    if (result.is_error()) {
        return Err(result.error());
    }
    return Ok(MediaCapture(result.unwrap()));
}

P10Result<std::vector<VideoDeviceInfo>> MediaCapture::list_video_devices() {
    return p10::media::list_video_devices();
}

P10Result<std::vector<AudioDeviceInfo>> MediaCapture::list_audio_devices() {
    return p10::media::list_audio_devices();
}

P10Result<MediaCapture> MediaCapture::open_stream(
    std::optional<std::pair<int, VideoParameters>> video,
    std::optional<std::pair<int, AudioParameters>> audio
) {
    auto result = FfmpegDeviceMediaCapture::open(std::move(video), std::move(audio));
    if (result.is_error()) {
        return Err(result.error());
    }
    return Ok(MediaCapture(result.unwrap()));
}

void MediaCapture::close() {
    if (impl_) {
        impl_->close();
    }
}

MediaParameters MediaCapture::get_parameters() const {
    return impl_->get_parameters();
}

bool MediaCapture::is_stream() const {
    return dynamic_cast<FfmpegDeviceMediaCapture*>(impl_.get()) != nullptr;
}

P10Result<MediaCapture::NextFrameResult> MediaCapture::next_frame(WaitMode wait) {
    return impl_->next_frame(wait);
}

P10Error MediaCapture::get_video(VideoFrame& frame) {
    return impl_->get_video(frame);
}

P10Error MediaCapture::get_audio(AudioFrame& frame) {
    return impl_->get_audio(frame);
}

P10Result<TextStreams> MediaCapture::get_text_streams() const {
    return impl_->get_text_streams();
}

std::optional<int64_t> MediaCapture::video_frame_count() const {
    return impl_->video_frame_count();
}

std::optional<double> MediaCapture::duration() const {
    return impl_->duration();
}

P10Error MediaCapture::seek(double seconds) {
    if (!impl_) {
        return P10Error::InvalidArgument << "MediaCapture not open";
    }
    return impl_->seek(seconds);
}

P10Error MediaCapture::set_auto_focus(bool enabled) {
    return impl_->set_camera_auto_control(CameraAutoControlId::Focus, enabled);
}

P10Result<bool> MediaCapture::get_auto_focus() const {
    return impl_->get_camera_auto_control(CameraAutoControlId::Focus);
}

P10Error MediaCapture::set_focus_distance(int value) {
    return impl_->set_camera_control(CameraControlId::FocusDistance, value);
}

P10Result<int> MediaCapture::get_focus_distance() const {
    return impl_->get_camera_control(CameraControlId::FocusDistance);
}

P10Result<CameraControlRange> MediaCapture::get_focus_distance_range() const {
    return impl_->get_camera_control_range(CameraControlId::FocusDistance);
}

P10Error MediaCapture::set_auto_exposure(bool enabled) {
    return impl_->set_camera_auto_control(CameraAutoControlId::Exposure, enabled);
}

P10Result<bool> MediaCapture::get_auto_exposure() const {
    return impl_->get_camera_auto_control(CameraAutoControlId::Exposure);
}

P10Error MediaCapture::set_exposure(int value) {
    return impl_->set_camera_control(CameraControlId::Exposure, value);
}

P10Result<int> MediaCapture::get_exposure() const {
    return impl_->get_camera_control(CameraControlId::Exposure);
}

P10Result<CameraControlRange> MediaCapture::get_exposure_range() const {
    return impl_->get_camera_control_range(CameraControlId::Exposure);
}

P10Error MediaCapture::set_brightness(int value) {
    return impl_->set_camera_control(CameraControlId::Brightness, value);
}

P10Result<int> MediaCapture::get_brightness() const {
    return impl_->get_camera_control(CameraControlId::Brightness);
}

P10Result<CameraControlRange> MediaCapture::get_brightness_range() const {
    return impl_->get_camera_control_range(CameraControlId::Brightness);
}

P10Error MediaCapture::set_contrast(int value) {
    return impl_->set_camera_control(CameraControlId::Contrast, value);
}

P10Result<int> MediaCapture::get_contrast() const {
    return impl_->get_camera_control(CameraControlId::Contrast);
}

P10Result<CameraControlRange> MediaCapture::get_contrast_range() const {
    return impl_->get_camera_control_range(CameraControlId::Contrast);
}

P10Error MediaCapture::set_saturation(int value) {
    return impl_->set_camera_control(CameraControlId::Saturation, value);
}

P10Result<int> MediaCapture::get_saturation() const {
    return impl_->get_camera_control(CameraControlId::Saturation);
}

P10Result<CameraControlRange> MediaCapture::get_saturation_range() const {
    return impl_->get_camera_control_range(CameraControlId::Saturation);
}

P10Error MediaCapture::set_gain(int value) {
    return impl_->set_camera_control(CameraControlId::Gain, value);
}

P10Result<int> MediaCapture::get_gain() const {
    return impl_->get_camera_control(CameraControlId::Gain);
}

P10Result<CameraControlRange> MediaCapture::get_gain_range() const {
    return impl_->get_camera_control_range(CameraControlId::Gain);
}

P10Error MediaCapture::set_auto_white_balance(bool enabled) {
    return impl_->set_camera_auto_control(CameraAutoControlId::WhiteBalance, enabled);
}

P10Result<bool> MediaCapture::get_auto_white_balance() const {
    return impl_->get_camera_auto_control(CameraAutoControlId::WhiteBalance);
}

P10Error MediaCapture::set_white_balance_temperature(int value) {
    return impl_->set_camera_control(CameraControlId::WhiteBalanceTemperature, value);
}

P10Result<int> MediaCapture::get_white_balance_temperature() const {
    return impl_->get_camera_control(CameraControlId::WhiteBalanceTemperature);
}

P10Result<CameraControlRange> MediaCapture::get_white_balance_temperature_range() const {
    return impl_->get_camera_control_range(CameraControlId::WhiteBalanceTemperature);
}

P10Error MediaCapture::set_zoom(int value) {
    return impl_->set_camera_control(CameraControlId::Zoom, value);
}

P10Result<int> MediaCapture::get_zoom() const {
    return impl_->get_camera_control(CameraControlId::Zoom);
}

P10Result<CameraControlRange> MediaCapture::get_zoom_range() const {
    return impl_->get_camera_control_range(CameraControlId::Zoom);
}

}  // namespace p10::media
