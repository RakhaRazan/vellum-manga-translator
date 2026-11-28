#pragma once

#include "detector_types.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace vellum::detector {

/// Pure-virtual interface for Detector Engine v2 backends (YOLO11-Seg, RF-DETR-Seg, Legacy CTD).
class IDetectorEngine {
public:
    virtual ~IDetectorEngine() = default;

    virtual bool load(const DetectorConfig& config) = 0;

    virtual PageDetectionResult detectPage(
        const uint8_t* rgbData,
        int width,
        int height,
        const std::string& filename,
        const DetectorConfig& config) = 0;

    virtual void unload() = 0;
    virtual std::string activeModelName() const = 0;
    virtual std::string executionProvider() const = 0;
};

/// Factory for instantiating concrete IDetectorEngine backends.
class DetectorFactory {
public:
    static std::unique_ptr<IDetectorEngine> create(BackendType backend);
};

}  // namespace vellum::detector
