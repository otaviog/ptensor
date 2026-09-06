#include <memory>
#include <string>
#include <vector>

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#include <ptensor/media/io/media_device.hpp>

#include "camera_controls.hpp"

namespace p10::media {

namespace {

    // AVCaptureDevice on macOS does not expose the integer image controls that
    // V4L2 does (brightness, contrast, zoom, ...): those properties are marked
    // API_UNAVAILABLE(macos). Only the auto/manual capture modes for focus,
    // exposure, and white balance are reachable, so those are the sole controls
    // this backend implements; the integer accessors report NotImplemented.
    P10Error not_supported(const std::string& what) {
        return P10Error::NotImplemented
            << (what + " is not supported through AVFoundation on this device");
    }

    /// Enumerate video capture devices in the same order FFmpeg's avfoundation
    /// demuxer does, so a device index resolves to the same camera. FFmpeg uses an
    /// AVCaptureDeviceDiscoverySession over built-in and external cameras.
    NSArray<AVCaptureDevice*>* discover_video_devices() {
        if (@available(macOS 10.15, *)) {
            NSMutableArray<AVCaptureDeviceType>* types =
                [NSMutableArray arrayWithObject:AVCaptureDeviceTypeBuiltInWideAngleCamera];
            if (@available(macOS 14.0, *)) {
                [types addObject:AVCaptureDeviceTypeExternal];
            } else {
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED < 140000
                [types addObject:AVCaptureDeviceTypeExternalUnknown];
#endif
            }
            AVCaptureDeviceDiscoverySession* session = [AVCaptureDeviceDiscoverySession
                discoverySessionWithDeviceTypes:types
                                      mediaType:AVMediaTypeVideo
                                       position:AVCaptureDevicePositionUnspecified];
            return session.devices;
        }
        return nil;
    }

    /// AVCaptureDevice-backed control surface. Wraps the device with manual
    /// retain/release (the target compiles with -fno-objc-arc, matching the rest
    /// of the Apple sources).
    class AvfCameraControlBackend: public CameraControlBackend {
      public:
        explicit AvfCameraControlBackend(AVCaptureDevice* device) : device_([device retain]) {}

        AvfCameraControlBackend(const AvfCameraControlBackend&) = delete;
        AvfCameraControlBackend& operator=(const AvfCameraControlBackend&) = delete;
        AvfCameraControlBackend(AvfCameraControlBackend&&) = delete;
        AvfCameraControlBackend& operator=(AvfCameraControlBackend&&) = delete;

        ~AvfCameraControlBackend() override {
            [device_ release];
        }

        P10Result<int> get(CameraControlId /*id*/) const override {
            return Err(not_supported("Integer camera controls"));
        }

        P10Error set(CameraControlId /*id*/, int /*value*/) override {
            return not_supported("Integer camera controls");
        }

        P10Result<CameraControlRange> get_range(CameraControlId /*id*/) const override {
            return Err(not_supported("Integer camera controls"));
        }

        P10Result<bool> get_auto(CameraAutoControlId id) const override {
            switch (id) {
                case CameraAutoControlId::Focus:
                    return Ok(device_.focusMode == AVCaptureFocusModeContinuousAutoFocus);
                case CameraAutoControlId::Exposure:
                    return Ok(device_.exposureMode == AVCaptureExposureModeContinuousAutoExposure);
                case CameraAutoControlId::WhiteBalance:
                    return Ok(
                        device_.whiteBalanceMode
                        == AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance
                    );
            }
            return Err(not_supported("This auto control"));
        }

        P10Error set_auto(CameraAutoControlId id, bool enabled) override {
            switch (id) {
                case CameraAutoControlId::Focus:
                    return set_focus_auto(enabled);
                case CameraAutoControlId::Exposure:
                    return set_exposure_auto(enabled);
                case CameraAutoControlId::WhiteBalance:
                    return set_white_balance_auto(enabled);
            }
            return not_supported("This auto control");
        }

      private:
        template<typename Fn>
        P10Error with_config(Fn&& fn) {
            NSError* error = nil;
            if (![device_ lockForConfiguration:&error]) {
                const char* msg = error ? error.localizedDescription.UTF8String : "unknown error";
                return P10Error::IoError
                    << (std::string("Failed to lock camera for configuration: ") + msg);
            }
            P10Error result = std::forward<Fn>(fn)(device_);
            [device_ unlockForConfiguration];
            return result;
        }

        P10Error set_focus_auto(bool enabled) {
            const AVCaptureFocusMode mode =
                enabled ? AVCaptureFocusModeContinuousAutoFocus : AVCaptureFocusModeLocked;
            if (![device_ isFocusModeSupported:mode]) {
                return not_supported("Focus mode");
            }
            return with_config([&](AVCaptureDevice* dev) {
                dev.focusMode = mode;
                return P10Error::Ok;
            });
        }

        P10Error set_exposure_auto(bool enabled) {
            const AVCaptureExposureMode mode =
                enabled ? AVCaptureExposureModeContinuousAutoExposure : AVCaptureExposureModeLocked;
            if (![device_ isExposureModeSupported:mode]) {
                return not_supported("Exposure mode");
            }
            return with_config([&](AVCaptureDevice* dev) {
                dev.exposureMode = mode;
                return P10Error::Ok;
            });
        }

        P10Error set_white_balance_auto(bool enabled) {
            const AVCaptureWhiteBalanceMode mode = enabled
                ? AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance
                : AVCaptureWhiteBalanceModeLocked;
            if (![device_ isWhiteBalanceModeSupported:mode]) {
                return not_supported("White balance mode");
            }
            return with_config([&](AVCaptureDevice* dev) {
                dev.whiteBalanceMode = mode;
                return P10Error::Ok;
            });
        }

        AVCaptureDevice* device_ = nil;
    };

}  // namespace

/// Enumerate video devices natively: FFmpeg's avfoundation demuxer does not
/// implement avdevice_list_input_sources(), so device listing must go through
/// AVFoundation directly. Indices match what the demuxer expects in its
/// "<video>:<audio>" URL because discover_video_devices() replicates its
/// discovery-session ordering.
P10Result<std::vector<VideoDeviceInfo>> list_avf_video_devices() {
    @autoreleasepool {
        std::vector<VideoDeviceInfo> result;
        int idx = 0;
        for (AVCaptureDevice* device in discover_video_devices()) {
            const char* name = device.localizedName.UTF8String;
            VideoDeviceInfo info;
            info.index(idx).name(name != nullptr ? name : "").url(std::to_string(idx));
            result.push_back(std::move(info));
            ++idx;
        }
        return Ok(std::move(result));
    }
}

std::unique_ptr<CameraControlBackend> open_avf_camera_control_backend(int video_device_index) {
    if (video_device_index < 0) {
        return nullptr;
    }
    @autoreleasepool {
        NSArray<AVCaptureDevice*>* devices = discover_video_devices();
        if (devices == nil || video_device_index >= static_cast<int>(devices.count)) {
            return nullptr;
        }
        AVCaptureDevice* device = devices[static_cast<NSUInteger>(video_device_index)];
        if (device == nil) {
            return nullptr;
        }
        return std::make_unique<AvfCameraControlBackend>(device);
    }
}

}  // namespace p10::media
