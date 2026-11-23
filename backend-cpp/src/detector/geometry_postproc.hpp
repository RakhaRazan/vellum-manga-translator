#pragma once

#include "detector_types.hpp"

#include <vector>

namespace vellum::detector {

/// Intersection over Union between two axis-aligned bounding boxes.
float computeIoU(const BBox& a, const BBox& b) noexcept;

/// Intersection over Area of `inner`: Area(inner ∩ outer) / Area(inner).
float computeIoA(const BBox& inner, const BBox& outer) noexcept;

/// Drop dialogue candidates (0..5) suppressed by higher-confidence SfxIgnore (8)
/// detections (IoU > config.sfx_suppress_iou) and drop micro-artifacts
/// (width < 12px, height < 12px, or mask area < 0.015% of page area).
void filterSfxAndMicroArtifacts(
    std::vector<BubbleDetection>& detections,
    int imageWidth,
    int imageHeight,
    const DetectorConfig& config);

/// Run 3x3 morphological closing on `mask` and compute the maximum inscribed
/// axis-aligned rectangle (row-histogram largest rectangle) with 5% inner padding.
/// Falls back safely to a 5% inset `bbox` if `mask` is empty or degenerates.
BBox computeSafeBBox(const BBox& bbox, const InstanceMask& mask) noexcept;

/// Associate each detection to the panel with highest IoA (>= 0.45) and mark
/// `is_cross_panel = true` when overlapping >= 2 panels with IoA >= 0.20 each.
void associateBubblesWithPanels(
    std::vector<BubbleDetection>& detections,
    const std::vector<PanelDetection>& panels) noexcept;

/// Sort panels and detections into reading order using a cycle-safe DAG
/// topological sort (vertical dominance with 15% overlap tolerance, horizontal
/// RTL/LTR, deterministic tie-breaking). Dialogue detections are sorted within
/// each `panel_id` using mask centroids, and unassociated bubbles globally.
void sortReadingOrder(
    std::vector<PanelDetection>& panels,
    std::vector<BubbleDetection>& detections,
    ReadingDirection direction);

/// Execute the full Detector Engine v2 geometry post-processing pipeline on a page.
void runGeometryPostproc(
    PageDetectionResult& result,
    const DetectorConfig& config);

/// Build a tight binary text-stroke inpainting mask (`width * height`, 0 or 255)
/// across the full `3px`-eroded `bubble_mask` (`InstanceMask`) interior of each
/// dialogue detection (`0..5`), completely separating container/bubble segmentation
/// masks (typesetting) from text strokes (inpainting) while covering top/bottom
/// tapered bubble tips and side columns outside `safe_bbox`.
std::vector<uint8_t> buildTextStrokeInpaintMask(
    const uint8_t* rgbImage,
    int width,
    int height,
    const std::vector<BubbleDetection>& detections,
    const std::vector<uint8_t>* optionalCtdStrokeMask = nullptr);

}  // namespace vellum::detector
