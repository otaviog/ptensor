#include "camera_controls.hpp"

#include <memory>
#include <string>

#ifdef __linux__
    #include <cerrno>
    #include <cstring>

    #include <fcntl.h>
    #include <linux/videodev2.h>
    #include <sys/ioctl.h>
    #include <unistd.h>
#endif

namespace p10::media {

namespace {

#ifdef __linux__

    uint32_t to_v4l2_cid(CameraControlId id) {
        switch (id) {
            case CameraControlId::Brightness:
                return V4L2_CID_BRIGHTNESS;
            case CameraControlId::Contrast:
                return V4L2_CID_CONTRAST;
            case CameraControlId::Saturation:
                return V4L2_CID_SATURATION;
            case CameraControlId::Gain:
                return V4L2_CID_GAIN;
            case CameraControlId::Zoom:
                return V4L2_CID_ZOOM_ABSOLUTE;
            case CameraControlId::FocusDistance:
                return V4L2_CID_FOCUS_ABSOLUTE;
            case CameraControlId::Exposure:
                return V4L2_CID_EXPOSURE_ABSOLUTE;
            case CameraControlId::WhiteBalanceTemperature:
                return V4L2_CID_WHITE_BALANCE_TEMPERATURE;
        }
        return 0;
    }

    uint32_t to_v4l2_cid(CameraAutoControlId id) {
        switch (id) {
            case CameraAutoControlId::Focus:
                return V4L2_CID_FOCUS_AUTO;
            case CameraAutoControlId::Exposure:
                return V4L2_CID_EXPOSURE_AUTO;
            case CameraAutoControlId::WhiteBalance:
                return V4L2_CID_AUTO_WHITE_BALANCE;
        }
        return 0;
    }

    P10Error errno_error(const std::string& context) {
        return P10Error::IoError << (context + ": " + std::strerror(errno));
    }

    /// V4L2 exposes camera controls through ioctl() on a device file
    /// descriptor. FFmpeg's v4l2 demuxer keeps its own fd private, so this
    /// backend opens a second, control-only fd to the same device node;
    /// V4L2 controls live at the device level and accept concurrent opens.
    class V4l2CameraControlBackend: public CameraControlBackend {
      public:
        explicit V4l2CameraControlBackend(int fd) : fd_(fd) {}

        V4l2CameraControlBackend(const V4l2CameraControlBackend&) = delete;
        V4l2CameraControlBackend& operator=(const V4l2CameraControlBackend&) = delete;
        V4l2CameraControlBackend(V4l2CameraControlBackend&&) = delete;
        V4l2CameraControlBackend& operator=(V4l2CameraControlBackend&&) = delete;

        ~V4l2CameraControlBackend() override {
            if (fd_ >= 0) {
                ::close(fd_);
            }
        }

        P10Result<int> get(CameraControlId id) const override {
            return get_raw(to_v4l2_cid(id));
        }

        P10Error set(CameraControlId id, int value) override {
            return set_raw(to_v4l2_cid(id), value);
        }

        P10Result<CameraControlRange> get_range(CameraControlId id) const override {
            return get_range_raw(to_v4l2_cid(id));
        }

        P10Result<bool> get_auto(CameraAutoControlId id) const override {
            auto result = get_raw(to_v4l2_cid(id));
            if (result.is_error()) {
                return Err(result.error());
            }
            if (id == CameraAutoControlId::Exposure) {
                return Ok(result.unwrap() == V4L2_EXPOSURE_AUTO);
            }
            return Ok(result.unwrap() != 0);
        }

        P10Error set_auto(CameraAutoControlId id, bool enabled) override {
            // V4L2_CID_EXPOSURE_AUTO is a menu control (auto/manual/shutter-
            // /aperture-priority), not a plain bool like the other auto CIDs.
            const int value = id == CameraAutoControlId::Exposure
                ? (enabled ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL)
                : (enabled ? 1 : 0);
            return set_raw(to_v4l2_cid(id), value);
        }

      private:
        P10Result<int> get_raw(uint32_t cid) const {
            v4l2_control control {};
            control.id = cid;
            if (ioctl(fd_, VIDIOC_G_CTRL, &control) < 0) {
                return Err(errno_error("Failed to read camera control"));
            }
            return Ok(control.value);
        }

        P10Error set_raw(uint32_t cid, int value) {
            v4l2_control control {};
            control.id = cid;
            control.value = value;
            if (ioctl(fd_, VIDIOC_S_CTRL, &control) < 0) {
                return errno_error("Failed to write camera control");
            }
            return P10Error::Ok;
        }

        P10Result<CameraControlRange> get_range_raw(uint32_t cid) const {
            v4l2_queryctrl query {};
            query.id = cid;
            if (ioctl(fd_, VIDIOC_QUERYCTRL, &query) < 0) {
                return Err(errno_error("Failed to query camera control range"));
            }
            return Ok(
                CameraControlRange {}
                    .min(query.minimum)
                    .max(query.maximum)
                    .step(query.step)
                    .default_value(query.default_value)
            );
        }

        int fd_ = -1;
    };

#endif  // defined(__linux__)

}  // namespace

std::unique_ptr<CameraControlBackend> open_camera_control_backend(int video_device_index) {
#ifdef __linux__
    if (video_device_index < 0) {
        return nullptr;
    }
    const std::string path = "/dev/video" + std::to_string(video_device_index);
    const int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) {
        return nullptr;
    }
    return std::make_unique<V4l2CameraControlBackend>(fd);
#else
    // TODO: implement for macOS (AVCaptureDevice) and Windows (DirectShow
    // IAMCameraControl / IAMVideoProcAmp). Until then camera property calls
    // report NotImplemented on those platforms.
    (void)video_device_index;
    return nullptr;
#endif
}

}  // namespace p10::media
