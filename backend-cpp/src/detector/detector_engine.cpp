#include "detector_engine.hpp"
#include "rfdetr_seg_backend.hpp"
#include "yolo_seg_backend.hpp"

namespace vellum::detector {

std::unique_ptr<IDetectorEngine> DetectorFactory::create(BackendType backend) {
    switch (backend) {
        case BackendType::Yolo11Seg:
        case BackendType::Auto:
            return std::make_unique<Yolo11SegBackend>();
        case BackendType::RfDetrSeg:
        case BackendType::LegacyCtd:
            return std::make_unique<RfDetrSegBackend>();
        default:
            return std::make_unique<Yolo11SegBackend>();
    }
}

}  // namespace vellum::detector
