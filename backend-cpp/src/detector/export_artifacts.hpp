#pragma once

#include "detector_types.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>

namespace vellum::detector {

/// Serialize a PageDetectionResult to Schema "2.0" JSON.
nlohmann::json toSchemaV2Json(const PageDetectionResult& result);

/// Export Schema "2.0" JSON (`<export_dir>/<stem>.detections.json`),
/// per-detection cropped RGB PNGs (`<export_dir>/crops/<id>_crop.png` with 4px clamped padding),
/// and 8-bit binary masks (`<export_dir>/masks/<id>_mask.png`) when `config.export_dir` is non-empty.
bool exportPageArtifacts(
    PageDetectionResult& result,
    const uint8_t* rgbData,
    int width,
    int height,
    const DetectorConfig& config);

}  // namespace vellum::detector
