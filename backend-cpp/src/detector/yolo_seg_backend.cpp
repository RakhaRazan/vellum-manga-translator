#include "yolo_seg_backend.hpp"
#include "cv_geometry.hpp"
#include "cv_refinement.hpp"
#include "cv_typography.hpp"
#include "geometry_postproc.hpp"
#include "../pipeline/ort_provider.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vellum::detector {

namespace {

constexpr int kYoloInputSize = 1024;
constexpr int kMaskChannels = 32;
constexpr float kLetterboxPadVal = 114.0f / 255.0f;
constexpr float kMaskThreshold = 0.50f;

// IEEE-754 float32 <-> float16 conversion helpers for FP16 ONNX models.
uint16_t floatToHalfBits(float value) noexcept {
    uint32_t f = 0;
    std::memcpy(&f, &value, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = f & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t m = mant | 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        return static_cast<uint16_t>(sign | (m >> shift));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float halfBitsToFloat(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t m = mant;
            int32_t e = 127 - 15 + 1;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            f = sign | (static_cast<uint32_t>(e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline float sigmoidf(float x) noexcept {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_left = 0;
    int pad_top = 0;
    int new_w = kYoloInputSize;
    int new_h = kYoloInputSize;
};

LetterboxInfo computeLetterbox(int src_w, int src_h, int dst_w, int dst_h) noexcept {
    LetterboxInfo info;
    info.scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                          static_cast<float>(dst_h) / static_cast<float>(src_h));
    info.new_w = std::max(1, static_cast<int>(std::round(static_cast<float>(src_w) * info.scale)));
    info.new_h = std::max(1, static_cast<int>(std::round(static_cast<float>(src_h) * info.scale)));
    info.pad_left = (dst_w - info.new_w) / 2;
    info.pad_top = (dst_h - info.new_h) / 2;
    return info;
}

void letterboxRgbToNchwFloat(
    const uint8_t* rgb,
    int src_w,
    int src_h,
    const LetterboxInfo& lb,
    int dst_w,
    int dst_h,
    std::vector<float>& out_nchw) {
    const size_t plane = static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h);
    out_nchw.assign(3ULL * plane, kLetterboxPadVal);

    float* ch0 = out_nchw.data();
    float* ch1 = out_nchw.data() + plane;
    float* ch2 = out_nchw.data() + 2ULL * plane;

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(lb.new_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(lb.new_h);

    for (int dy = 0; dy < lb.new_h; ++dy) {
        const float sy_f = (static_cast<float>(dy) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy_f)), 0, src_h - 1);
        const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        const float wy1 = std::clamp(sy_f - static_cast<float>(y0), 0.0f, 1.0f);
        const float wy0 = 1.0f - wy1;

        const size_t dst_row = static_cast<size_t>(dy + lb.pad_top) * static_cast<size_t>(dst_w);
        const size_t src_row0 = static_cast<size_t>(y0) * static_cast<size_t>(src_w);
        const size_t src_row1 = static_cast<size_t>(y1) * static_cast<size_t>(src_w);

        for (int dx = 0; dx < lb.new_w; ++dx) {
            const float sx_f = (static_cast<float>(dx) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sx_f)), 0, src_w - 1);
            const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            const float wx1 = std::clamp(sx_f - static_cast<float>(x0), 0.0f, 1.0f);
            const float wx0 = 1.0f - wx1;

            const size_t idx00 = (src_row0 + static_cast<size_t>(x0)) * 3ULL;
            const size_t idx01 = (src_row0 + static_cast<size_t>(x1)) * 3ULL;
            const size_t idx10 = (src_row1 + static_cast<size_t>(x0)) * 3ULL;
            const size_t idx11 = (src_row1 + static_cast<size_t>(x1)) * 3ULL;

            const float w00 = wy0 * wx0;
            const float w01 = wy0 * wx1;
            const float w10 = wy1 * wx0;
            const float w11 = wy1 * wx1;

            const size_t dst_idx = dst_row + static_cast<size_t>(dx + lb.pad_left);
            ch0[dst_idx] = (w00 * rgb[idx00 + 0] + w01 * rgb[idx01 + 0] +
                            w10 * rgb[idx10 + 0] + w11 * rgb[idx11 + 0]) / 255.0f;
            ch1[dst_idx] = (w00 * rgb[idx00 + 1] + w01 * rgb[idx01 + 1] +
                            w10 * rgb[idx10 + 1] + w11 * rgb[idx11 + 1]) / 255.0f;
            ch2[dst_idx] = (w00 * rgb[idx00 + 2] + w01 * rgb[idx01 + 2] +
                            w10 * rgb[idx10 + 2] + w11 * rgb[idx11 + 2]) / 255.0f;
        }
    }
}

struct RawYoloCandidate {
    BBox lb_bbox;  // In 1024x1024 letterbox coordinates
    int class_id = 0;
    float confidence = 0.0f;
    std::array<float, kMaskChannels> mask_coeffs{};
};

// Extract simplified boundary polygon in full image coordinates from a cropped binary mask.
std::vector<Point2f> extractSimplifiedPolygon(const InstanceMask& mask) {
    std::vector<Point2f> poly;
    if (mask.empty()) {
        return poly;
    }

    // Collect left-most and right-most foreground points per row to form a clean hull/contour
    // without OpenCV, then subsample to <= 32 vertices.
    std::vector<Point2f> left_edge;
    std::vector<Point2f> right_edge;
    const int step = std::max(1, mask.height / 16);

    for (int y = 0; y < mask.height; y += step) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(mask.width);
        int min_x = -1;
        int max_x = -1;
        for (int x = 0; x < mask.width; ++x) {
            if (mask.data[row + static_cast<size_t>(x)] != 0) {
                if (min_x < 0) min_x = x;
                max_x = x;
            }
        }
        if (min_x >= 0) {
            left_edge.push_back(Point2f{
                static_cast<float>(mask.offset_x + min_x),
                static_cast<float>(mask.offset_y + y)
            });
            if (max_x != min_x) {
                right_edge.push_back(Point2f{
                    static_cast<float>(mask.offset_x + max_x),
                    static_cast<float>(mask.offset_y + y)
                });
            }
        }
    }

    poly.reserve(left_edge.size() + right_edge.size());
    for (const auto& pt : left_edge) {
        poly.push_back(pt);
    }
    for (auto it = right_edge.rbegin(); it != right_edge.rend(); ++it) {
        poly.push_back(*it);
    }
    return poly;
}

std::vector<float> extractTensorAsFloat(const Ort::Value& val) {
    const auto info = val.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    const auto elem_type = info.GetElementType();

    std::vector<float> out(count);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* ptr = val.GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = halfBitsToFloat(ptr[i].val);
        }
    } else {
        const auto* ptr = val.GetTensorData<float>();
        std::memcpy(out.data(), ptr, count * sizeof(float));
    }
    return out;
}

}  // namespace

Ort::Env& Yolo11SegBackend::sharedEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VellumYolo11Seg");
    return env;
}

int Yolo11SegBackend::queryPhysicalCoreCount() noexcept {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<uint8_t> buf(len);
        if (GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                &len)) {
            int cores = 0;
            DWORD offset = 0;
            while (offset < len) {
                const auto* info =
                    reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + offset);
                if (info->Relationship == RelationProcessorCore) {
                    ++cores;
                }
                offset += info->Size;
            }
            if (cores > 0) {
                return cores;
            }
        }
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1u, hw));
}

bool Yolo11SegBackend::load(const DetectorConfig& config) {
    if (config.model_path.empty()) {
        return false;
    }

    if (session_ && loaded_model_path_ == config.model_path &&
        loaded_force_cpu_ == config.force_cpu) {
        return true;
    }

    unload();

    const std::filesystem::path model_fs_path = std::filesystem::u8path(config.model_path);
    if (!std::filesystem::exists(model_fs_path)) {
        return false;
    }

    const int physical_cores = queryPhysicalCoreCount();
    const int gpu_host_threads = std::max(1, std::min(2, physical_cores / 2));

    const std::vector<std::string> providers =
        mangaflow::pipeline::get_ordered_execution_providers(config.force_cpu);

    auto tryCreateSession = [&](const std::string& provider) -> std::unique_ptr<Ort::Session> {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetInterOpNumThreads(1);

        if (provider == "CPUExecutionProvider") {
            opts.SetIntraOpNumThreads(physical_cores);
            opts.AddConfigEntry("session.intra_op.allow_spinning", "1");
            return std::make_unique<Ort::Session>(sharedEnv(), model_fs_path.c_str(), opts);
        }

        opts.SetIntraOpNumThreads(gpu_host_threads);
        opts.AddConfigEntry("session.intra_op.allow_spinning", "0");

        if (!mangaflow::pipeline::append_gpu_execution_provider(opts, provider, /*device_id=*/0)) {
            return nullptr;
        }
        return std::make_unique<Ort::Session>(sharedEnv(), model_fs_path.c_str(), opts);
    };

    for (const auto& provider : providers) {
        try {
            auto sess = tryCreateSession(provider);
            if (sess) {
                session_ = std::move(sess);
                active_provider_ = provider;
                break;
            }
        } catch (...) {
            // Fall through to next execution provider
        }
    }

    if (!session_) {
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    if (session_->GetInputCount() < 1 || session_->GetOutputCount() < 2) {
        unload();
        return false;
    }

    {
        auto in_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = in_name.get();
        const auto type_info = session_->GetInputTypeInfo(0);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_elem_type_ = tensor_info.GetElementType();
        const auto shape = tensor_info.GetShape();
        if (shape.size() == 4) {
            if (shape[2] > 0) input_height_ = static_cast<int>(shape[2]);
            if (shape[3] > 0) input_width_ = static_cast<int>(shape[3]);
        }
    }

    output_names_.clear();
    const size_t out_count = session_->GetOutputCount();
    output_names_.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        auto out_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(out_name.get());
    }

    loaded_model_path_ = config.model_path;
    loaded_force_cpu_ = config.force_cpu;
    model_name_ = model_fs_path.stem().u8string();
    return true;
}

PageDetectionResult Yolo11SegBackend::detectPage(
    const uint8_t* rgbData,
    int width,
    int height,
    const std::string& filename,
    const DetectorConfig& config) {
    if (!rgbData || width <= 0 || height <= 0) {
        throw std::invalid_argument("Yolo11SegBackend::detectPage: invalid RGB buffer or dimensions.");
    }

    if (!session_ && !load(config)) {
        throw std::runtime_error("Yolo11SegBackend::detectPage: failed to load ONNX model.");
    }

    const auto t_start = std::chrono::steady_clock::now();

    const int dst_w = (input_width_ > 0) ? input_width_ : kYoloInputSize;
    const int dst_h = (input_height_ > 0) ? input_height_ : kYoloInputSize;
    const LetterboxInfo lb = computeLetterbox(width, height, dst_w, dst_h);

    std::vector<float> input_fp32;
    letterboxRgbToNchwFloat(rgbData, width, height, lb, dst_w, dst_h, input_fp32);

    const std::array<int64_t, 4> input_shape{1, 3, dst_h, dst_w};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Float16_t> input_fp16;
    Ort::Value input_tensor{nullptr};
    if (input_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        input_fp16.resize(input_fp32.size());
        for (size_t i = 0; i < input_fp32.size(); ++i) {
            input_fp16[i] = Ort::Float16_t(input_fp32[i]);
        }
        input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, input_fp16.data(), input_fp16.size(), input_shape.data(), input_shape.size());
    } else {
        input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_fp32.data(), input_fp32.size(), input_shape.data(), input_shape.size());
    }

    const char* in_names[] = {input_name_.c_str()};
    std::vector<const char*> out_names;
    out_names.reserve(output_names_.size());
    for (const auto& s : output_names_) {
        out_names.push_back(s.c_str());
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names.data(), out_names.size());
    } catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "[Vellum][YOLO11m-Seg] %s inference failed: %s\n", active_provider_.c_str(), ex.what());
        if (active_provider_ == "CPUExecutionProvider") {
            throw;
        }
        DetectorConfig cpu_cfg = config;
        cpu_cfg.force_cpu = true;
        if (!load(cpu_cfg) || !session_) {
            throw;
        }
        outputs = session_->Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names.data(), out_names.size());
    }

    const auto t_infer = std::chrono::steady_clock::now();

    // Locate output0 (detections: 3D) and output1 (protos: 4D)
    size_t det_idx = 0;
    size_t proto_idx = 1;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3) det_idx = i;
        if (shape.size() == 4) proto_idx = i;
    }

    const auto det_shape = outputs[det_idx].GetTensorTypeAndShapeInfo().GetShape();
    const auto proto_shape = outputs[proto_idx].GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<float> det_data = extractTensorAsFloat(outputs[det_idx]);
    const std::vector<float> proto_data = extractTensorAsFloat(outputs[proto_idx]);

    // Infer num_classes dynamically from output0: [1, 4 + num_classes + 32, num_anchors]
    const int64_t dim1 = (det_shape.size() >= 2) ? det_shape[1] : 0;
    const int64_t dim2 = (det_shape.size() >= 3) ? det_shape[2] : 0;
    const bool channels_first = (dim1 <= dim2);
    const int channels = static_cast<int>(channels_first ? dim1 : dim2);
    const int num_anchors = static_cast<int>(channels_first ? dim2 : dim1);
    const int num_classes = std::max(0, channels - 4 - kMaskChannels);

    const int proto_c = (proto_shape.size() == 4) ? static_cast<int>(proto_shape[1]) : kMaskChannels;
    const int proto_h = (proto_shape.size() == 4) ? static_cast<int>(proto_shape[2]) : 256;
    const int proto_w = (proto_shape.size() == 4) ? static_cast<int>(proto_shape[3]) : 256;

    std::vector<RawYoloCandidate> candidates;
    if (num_classes > 0 && num_anchors > 0) {
        auto getDetVal = [&](int ch, int anchor) -> float {
            if (channels_first) {
                return det_data[static_cast<size_t>(ch) * static_cast<size_t>(num_anchors) +
                                static_cast<size_t>(anchor)];
            }
            return det_data[static_cast<size_t>(anchor) * static_cast<size_t>(channels) +
                            static_cast<size_t>(ch)];
        };

        for (int a = 0; a < num_anchors; ++a) {
            int best_cls = -1;
            float best_score = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                const float s = getDetVal(4 + c, a);
                const float thresh = (c < kNumClasses)
                    ? config.class_thresholds[static_cast<size_t>(c)]
                    : 0.35f;
                if (s >= thresh && s > best_score) {
                    best_score = s;
                    best_cls = c;
                }
            }
            if (best_cls < 0) {
                continue;
            }

            const float cx = getDetVal(0, a);
            const float cy = getDetVal(1, a);
            const float bw = getDetVal(2, a);
            const float bh = getDetVal(3, a);

            RawYoloCandidate cand;
            cand.lb_bbox = BBox{cx - 0.5f * bw, cy - 0.5f * bh, cx + 0.5f * bw, cy + 0.5f * bh};
            cand.class_id = best_cls;
            cand.confidence = best_score;
            for (int m = 0; m < kMaskChannels; ++m) {
                cand.mask_coeffs[static_cast<size_t>(m)] = getDetVal(4 + num_classes + m, a);
            }
            candidates.push_back(cand);
        }
    }

    // Class-aware NMS
    std::sort(candidates.begin(), candidates.end(),
              [](const RawYoloCandidate& a, const RawYoloCandidate& b) {
                  return a.confidence > b.confidence;
              });

    const int max_det = std::max(150, config.max_detections);
    std::vector<RawYoloCandidate> nms_kept;
    nms_kept.reserve(std::min(static_cast<int>(candidates.size()), max_det));
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        nms_kept.push_back(candidates[i]);
        if (static_cast<int>(nms_kept.size()) >= max_det) break;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[i].class_id != candidates[j].class_id) continue;
            if (computeIoU(candidates[i].lb_bbox, candidates[j].lb_bbox) > config.nms_iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    PageDetectionResult result;
    result.schema_version = "2.0";
    result.filename = filename;
    result.image_width = width;
    result.image_height = height;
    result.reading_direction = config.reading_direction;
    result.runtime_meta.backend = BackendType::Yolo11Seg;
    result.runtime_meta.model_name = model_name_;
    result.runtime_meta.execution_provider = active_provider_;
    result.runtime_meta.tiles_processed = 1;

    const float proto_scale_x = static_cast<float>(proto_w) / static_cast<float>(dst_w);
    const float proto_scale_y = static_cast<float>(proto_h) / static_cast<float>(dst_h);
    const size_t proto_plane = static_cast<size_t>(proto_h) * static_cast<size_t>(proto_w);
    const int use_c = std::min(proto_c, kMaskChannels);

    int next_panel_id = 0;
    int next_det_id = 0;

    for (const auto& cand : nms_kept) {
        // Un-letterbox box to original image coordinates
        const float ox1 = std::clamp((cand.lb_bbox.x1 - static_cast<float>(lb.pad_left)) / lb.scale,
                                     0.0f, static_cast<float>(width));
        const float oy1 = std::clamp((cand.lb_bbox.y1 - static_cast<float>(lb.pad_top)) / lb.scale,
                                     0.0f, static_cast<float>(height));
        const float ox2 = std::clamp((cand.lb_bbox.x2 - static_cast<float>(lb.pad_left)) / lb.scale,
                                     0.0f, static_cast<float>(width));
        const float oy2 = std::clamp((cand.lb_bbox.y2 - static_cast<float>(lb.pad_top)) / lb.scale,
                                     0.0f, static_cast<float>(height));
        BBox orig_box{ox1, oy1, ox2, oy2};
        if (!orig_box.is_valid()) {
            continue;
        }

        BBox norm_box{
            ox1 / static_cast<float>(width),
            oy1 / static_cast<float>(height),
            ox2 / static_cast<float>(width),
            oy2 / static_cast<float>(height)
        };

        // Single-class or compact multi-class open-source manga bubble models (e.g. 1..3 bubble classes)
        // map cleanly to ClassId::SpeechBubble, while full 9-class Vellum models preserve their taxonomy.
        ClassId cls = ClassId::SpeechBubble;
        if (num_classes == kNumClasses && cand.class_id >= 0 && cand.class_id < kNumClasses) {
            cls = static_cast<ClassId>(cand.class_id);
        } else if (num_classes > kNumClasses && cand.class_id >= kNumClasses) {
            cls = ClassId::TextRegion;
        }

        if (cls == ClassId::Panel) {
            PanelDetection p;
            p.id = next_panel_id++;
            p.confidence = cand.confidence;
            p.bbox = orig_box;
            p.bbox_norm = norm_box;
            result.panels.push_back(std::move(p));
            continue;
        }

        BubbleDetection det;
        det.id = next_det_id++;
        det.class_id = cls;
        det.class_name = class_id_to_string(cls);
        det.confidence = cand.confidence;
        det.bbox = orig_box;
        det.bbox_norm = norm_box;

        // Decode cropped instance mask back to original image resolution
        const int bx1 = std::clamp(static_cast<int>(std::floor(ox1)), 0, width - 1);
        const int by1 = std::clamp(static_cast<int>(std::floor(oy1)), 0, height - 1);
        const int bx2 = std::clamp(static_cast<int>(std::ceil(ox2)), bx1 + 1, width);
        const int by2 = std::clamp(static_cast<int>(std::ceil(oy2)), by1 + 1, height);
        const int mw = bx2 - bx1;
        const int mh = by2 - by1;

        if (mw > 0 && mh > 0 && !proto_data.empty()) {
            det.mask.offset_x = bx1;
            det.mask.offset_y = by1;
            det.mask.width = mw;
            det.mask.height = mh;
            det.mask.data.assign(static_cast<size_t>(mw) * static_cast<size_t>(mh), 0);

            double area_px = 0.0;
            for (int y = 0; y < mh; ++y) {
                const float orig_y = static_cast<float>(by1 + y) + 0.5f;
                const float lb_y = orig_y * lb.scale + static_cast<float>(lb.pad_top);
                const float py_f = lb_y * proto_scale_y - 0.5f;
                const int py0 = std::clamp(static_cast<int>(std::floor(py_f)), 0, proto_h - 1);
                const int py1 = std::clamp(py0 + 1, 0, proto_h - 1);
                const float wy1 = std::clamp(py_f - static_cast<float>(py0), 0.0f, 1.0f);
                const float wy0 = 1.0f - wy1;

                const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(mw);
                for (int x = 0; x < mw; ++x) {
                    const float orig_x = static_cast<float>(bx1 + x) + 0.5f;
                    const float lb_x = orig_x * lb.scale + static_cast<float>(lb.pad_left);
                    const float px_f = lb_x * proto_scale_x - 0.5f;
                    const int px0 = std::clamp(static_cast<int>(std::floor(px_f)), 0, proto_w - 1);
                    const int px1 = std::clamp(px0 + 1, 0, proto_w - 1);
                    const float wx1 = std::clamp(px_f - static_cast<float>(px0), 0.0f, 1.0f);
                    const float wx0 = 1.0f - wx1;

                    float logit = 0.0f;
                    for (int c = 0; c < use_c; ++c) {
                        const size_t base = static_cast<size_t>(c) * proto_plane;
                        const float p00 = proto_data[base + static_cast<size_t>(py0) * static_cast<size_t>(proto_w) + static_cast<size_t>(px0)];
                        const float p01 = proto_data[base + static_cast<size_t>(py0) * static_cast<size_t>(proto_w) + static_cast<size_t>(px1)];
                        const float p10 = proto_data[base + static_cast<size_t>(py1) * static_cast<size_t>(proto_w) + static_cast<size_t>(px0)];
                        const float p11 = proto_data[base + static_cast<size_t>(py1) * static_cast<size_t>(proto_w) + static_cast<size_t>(px1)];
                        const float interp = wy0 * (wx0 * p00 + wx1 * p01) +
                                             wy1 * (wx0 * p10 + wx1 * p11);
                        logit += cand.mask_coeffs[static_cast<size_t>(c)] * interp;
                    }

                    if (sigmoidf(logit) >= kMaskThreshold) {
                        det.mask.data[dst_row + static_cast<size_t>(x)] = 255;
                        area_px += 1.0;
                    }
                }
            }
            det.mask.area_px = area_px;
            det.mask.polygon = extractSimplifiedPolygon(det.mask);
        }

        result.detections.push_back(std::move(det));
    }

    runGeometryPostproc(result, config);

    // Wire full 0-VRAM v2.5 CV stack (5-stage Otsu glyph refinement, waist-cut watershed
    // room polygons <= 32 vertices, and typography/color/outline metadata) into YOLO11m-Seg.
    if (!result.detections.empty()) {
        std::vector<RefinementBlock> ref_blocks;
        ref_blocks.reserve(result.detections.size());
        std::vector<RoomSeedBlock> seeds;
        seeds.reserve(result.detections.size());
        std::vector<RectI> bubble_rects;
        bubble_rects.reserve(result.detections.size());

        const bool is_rtl = (config.reading_direction == ReadingDirection::RTL);
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& d = result.detections[i];
            const BBox& core = d.safe_bbox.is_valid() ? d.safe_bbox : d.bbox;
            const RectI core_rect = RectI{
                static_cast<int>(std::floor(core.x1)),
                static_cast<int>(std::floor(core.y1)),
                static_cast<int>(std::ceil(core.x2)),
                static_cast<int>(std::ceil(core.y2))
            }.clamped(width, height);
            const RectI full_rect = RectI{
                static_cast<int>(std::floor(d.bbox.x1)),
                static_cast<int>(std::floor(d.bbox.y1)),
                static_cast<int>(std::ceil(d.bbox.x2)),
                static_cast<int>(std::ceil(d.bbox.y2))
            }.clamped(width, height);

            RefinementBlock rb;
            rb.original_index = static_cast<int>(i);
            rb.class_id = (d.class_id == ClassId::SfxIgnore) ? 1 : 0;
            rb.score = d.confidence;
            rb.bbox = core_rect;
            rb.is_vertical = is_rtl && (core_rect.height() >= core_rect.width());
            if (!d.mask.empty()) {
                rb.raw_mask_rect = RectI{
                    d.mask.offset_x,
                    d.mask.offset_y,
                    d.mask.offset_x + d.mask.width,
                    d.mask.offset_y + d.mask.height
                }.clamped(width, height);
                rb.raw_mask = d.mask.data;
            }
            ref_blocks.push_back(std::move(rb));

            RoomSeedBlock seed;
            seed.block_index = static_cast<int>(i);
            seed.bbox = core_rect;
            seeds.push_back(std::move(seed));

            if (!full_rect.empty()) {
                bubble_rects.push_back(full_rect);
            }
        }

        // 1. 5-Stage Otsu + Plain-Paper Glyph Mask Refinement (zero bubble-border bleed)
        const std::vector<uint8_t> refined_page_mask =
            refineTextMasks(rgbData, width, height, ref_blocks);

        // 2. Conjoined Balloon Waist-Cutting + Watershed Room Polygons (<= 32 vertices)
        const auto room_polys = computePageRoomPolygons(
            rgbData, width, height, seeds, bubble_rects, &refined_page_mask, config.enable_waist_cut);
        for (size_t i = 0; i < result.detections.size() && i < room_polys.size(); ++i) {
            result.detections[i].room_polygon = room_polys[i];
        }

        // 3. 0-VRAM CV Typography & Colorimetry (tilt, font size, fill color, outline)
        // Pass nullptr for optional_mask so Otsu extracts interior text strokes strictly
        // inside `safe_bbox` (avoiding YOLO11m-Seg's full-bubble wall mask skewing A_max).
        if (config.enable_cv_style_detect) {
            for (size_t i = 0; i < result.detections.size(); ++i) {
                auto& d = result.detections[i];
                RectI tb = ref_blocks[i].bbox;
                // Inset by 2px when box is large enough so stray bubble border pixels are excluded
                if (tb.width() >= 20 && tb.height() >= 20) {
                    tb = RectI{tb.x1 + 2, tb.y1 + 2, tb.x2 - 2, tb.y2 - 2}.clamped(width, height);
                }
                const bool is_vert = is_rtl && (tb.height() > static_cast<int>(tb.width() * 1.15));
                const bool is_sfx = (d.class_id == ClassId::SfxIgnore);
                const auto style = analyzeBlockTypographyFromBox(
                    rgbData, width, height, tb, is_vert, is_sfx, nullptr, nullptr);
                if (style.valid) {
                    d.angle_deg = style.angle_deg;
                    d.estimated_font_size = style.estimated_font_size;
                    d.text_color = style.text_color;
                    if (style.outline.has_value()) {
                        d.outline = BubbleOutlineInfo{style.outline->color, style.outline->width};
                    } else {
                        d.outline = std::nullopt;
                    }
                }
            }
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    result.runtime_meta.inference_ms =
        std::chrono::duration<double, std::milli>(t_infer - t_start).count();
    result.runtime_meta.postproc_ms =
        std::chrono::duration<double, std::milli>(t_end - t_infer).count();

    if (!config.keep_session_alive) {
        unload();
    }

    return result;
}

void Yolo11SegBackend::unload() {
    session_.reset();
    loaded_model_path_.clear();
    input_name_.clear();
    output_names_.clear();
}

std::string Yolo11SegBackend::activeModelName() const {
    return model_name_;
}

std::string Yolo11SegBackend::executionProvider() const {
    return active_provider_;
}

}  // namespace vellum::detector
