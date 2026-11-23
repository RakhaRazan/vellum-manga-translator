#pragma once

#include "detector_engine.hpp"

#include <memory>
#include <string>

namespace mangaflow::pipeline {
class ModelManager;
class TextDetector;
}  // namespace mangaflow::pipeline

namespace vellum::detector {

/// High-Context profile wrapping the existing Koharu RF-DETR-Seg ONNX detector
/// (`mangaflow::pipeline::TextDetector`) into the unified IDetectorEngine interface.
class RfDetrSegBackend final : public IDetectorEngine {
public:
    RfDetrSegBackend();
    ~RfDetrSegBackend() override;

    bool load(const DetectorConfig& config) override;

    PageDetectionResult detectPage(
        const uint8_t* rgbData,
        int width,
        int height,
        const std::string& filename,
        const DetectorConfig& config) override;

    void unload() override;
    std::string activeModelName() const override;
    std::string executionProvider() const override;

private:
    std::unique_ptr<mangaflow::pipeline::ModelManager> model_manager_;
    std::unique_ptr<mangaflow::pipeline::TextDetector> text_detector_;
    std::string active_provider_ = "CPUExecutionProvider";
    std::string model_name_ = "koharu-layout-rfdetr-seg-2xl";
    bool loaded_ = false;
};

}  // namespace vellum::detector
