#pragma once

#include "detector_engine.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vellum::detector {

/// Adaptive high-resolution / tall Webtoon strip tiler with Weighted Boxes Fusion (WBF)
/// and IoU mask union across overlapping 1024x1024 tiles.
class AdaptiveTiler {
public:
    struct TileRect {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };

    static bool shouldTile(int width, int height, const DetectorConfig& config) noexcept;

    static std::vector<TileRect> computeTileGrid(
        int width,
        int height,
        const DetectorConfig& config);

    static PageDetectionResult detectPage(
        IDetectorEngine& engine,
        const uint8_t* rgbData,
        int width,
        int height,
        const std::string& filename,
        const DetectorConfig& config);
};

}  // namespace vellum::detector
