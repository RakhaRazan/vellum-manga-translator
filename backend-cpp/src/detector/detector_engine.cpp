#include "detector_engine.hpp"
#include "yolo_seg_backend.hpp"

namespace vellum::detector {

std::unique_ptr<IDetectorEngine> DetectorFactory::create(BackendType backend) {
    switch (backend) {
        case BackendType::Yolo11Seg:
        case BackendType::Auto:
        case BackendType::LegacyCtd:
        default:
            return std::make_unique<Yolo11SegBackend>();
    }
}

}  // namespace vellum::detector
