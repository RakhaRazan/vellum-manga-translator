#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vellum {
namespace psd {

class PsdWriteError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

enum class TextAlign : int {
    Left = 0,
    Right = 1,
    Center = 2,
};

struct ColorRGB {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

struct TextOutlineSpec {
    bool enabled = false;
    double width_px = 0.0;
    ColorRGB color{255, 255, 255};
    double opacity_pct = 100.0;  // 0..100
};

// Stage 2: Editable Photoshop Type Layer metadata (`TySh` + `EngineData` + `lfx2`)
struct TypeLayerSpec {
    std::string text;               // UTF-8; newlines normalized to '\r' and '\r'-terminated
    std::string font_name = "ArialMT";
    double font_size = 18.0;        // in points (at 72 DPI, 1 pt == 1 px)
    double leading = 0.0;           // <= 0 means 1.2 * font_size
    ColorRGB fill_color{0, 0, 0};
    TextAlign align = TextAlign::Center;
    bool faux_bold = false;
    bool faux_italic = false;
    bool vertical = false;
    bool auto_hyphenate = false;
    double box_x = 0.0;
    double box_y = 0.0;
    double box_width = 100.0;
    double box_height = 50.0;
    double rotation_deg = 0.0;
    TextOutlineSpec outline;
};

// Single raster (or raster + editable Type) layer in a PSD document.
// Layers are supplied in bottom-to-top stacking order (index 0 = bottom-most `Background`).
struct PsdLayer {
    std::string name;               // UTF-8 layer name
    int32_t left = 0;
    int32_t top = 0;
    int32_t width = 0;
    int32_t height = 0;
    uint8_t opacity = 255;          // 0..255
    bool visible = true;
    bool has_alpha = true;          // true -> RGBA (4 channels: -1, 0, 1, 2); false -> RGB (3 channels: 0, 1, 2)
    std::vector<uint8_t> pixels;    // tightly packed row-major RGBA (if has_alpha) or RGB (if !has_alpha)
    std::optional<TypeLayerSpec> type_spec;  // Stage 2: emitted only when editable_text_layers == true
};

struct PsdDocumentOptions {
    int32_t width = 0;
    int32_t height = 0;
    uint16_t channels = 3;          // 3 = RGB (opaque composite), 4 = RGBA (transparent composite)
    double dpi = 72.0;              // Written to Image Resource 1005 (ResolutionInfo)
    bool editable_text_layers = false;  // Stage 2 toggle: emit TySh + EngineData + lfx2 tagged blocks
};

// Pure C++17 PackBits RLE encoder/decoder (Apple/Adobe PSD compression = 1).
std::vector<uint8_t> packbits_encode_row(const uint8_t* src, size_t len);
std::vector<uint8_t> packbits_decode_row(const uint8_t* src, size_t src_len, size_t expected_len);

// Utility: Parse "#RRGGBB" or "#RGB" hex color string into ColorRGB.
ColorRGB parse_hex_color(const std::string& hex, ColorRGB fallback = {0, 0, 0});

// Utility: Build a non-destructive `Cleanup` patch layer from raw original RGB/RGBA and
// inpainted RGB/RGBA buffers (same dimensions). Only pixels differing by > `diff_threshold`
// (or marked in optional 1-channel `mask_gray`) receive alpha = 255, tightly cropped to the
// non-transparent bounding box. If no pixels differ, returns a 0x0 empty transparent layer.
PsdLayer build_cleanup_diff_layer(
    const uint8_t* original_rgba,
    const uint8_t* cleaned_rgba,
    int32_t width,
    int32_t height,
    const uint8_t* mask_gray = nullptr,
    uint8_t diff_threshold = 2,
    const std::string& layer_name = "Cleanup");

// Utility: Tightly crop a full-page RGBA buffer to its non-transparent bounding box
// (alpha > 0) and return a positioned PsdLayer with exact (left, top) offsets.
PsdLayer build_cropped_rgba_layer(
    const uint8_t* full_rgba,
    int32_t page_width,
    int32_t page_height,
    const std::string& layer_name,
    uint8_t opacity = 255,
    bool visible = true);

// Utility: Build a PSD Version 1 binary file in memory.
// If `composite_rgb_or_rgba` is empty, the writer automatically alpha-composites all visible
// layers bottom-to-top onto opaque white (channels == 3) or transparent (channels == 4).
std::vector<uint8_t> write_psd_bytes(
    const PsdDocumentOptions& options,
    const std::vector<PsdLayer>& layers,
    const std::vector<uint8_t>& composite_rgb_or_rgba = {});

// Convenience helper: Write PSD directly to a binary file path (UTF-8 path string).
void write_psd_file(
    const std::string& utf8_path,
    const PsdDocumentOptions& options,
    const std::vector<PsdLayer>& layers,
    const std::vector<uint8_t>& composite_rgb_or_rgba = {});

}  // namespace psd
}  // namespace vellum
