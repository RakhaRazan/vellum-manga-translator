#pragma once

#include "detector_engine.hpp"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace vellum::detector {

/// Native C++17 ONNX Runtime backend for YOLO11-Seg 9-class layout & bubble segmentation.
class Yolo11SegBackend final : public IDetectorEngine {
public:
    Yolo11SegBackend() = default;
    ~Yolo11SegBackend() override = default;

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
    static Ort::Env& sharedEnv();
    static int queryPhysicalCoreCount() noexcept;

    std::unique_ptr<Ort::Session> session_;
    std::string loaded_model_path_;
    bool loaded_force_cpu_ = false;
    std::string active_provider_ = "CPUExecutionProvider";
    std::string model_name_ = "yolo11-seg";

    std::string input_name_;
    std::vector<std::string> output_names_;
    ONNXTensorElementDataType input_elem_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    int input_width_ = 1024;
    int input_height_ = 1024;
};

}  // namespace vellum::detector
