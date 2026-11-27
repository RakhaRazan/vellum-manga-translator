#include "psd_writer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace vellum {
namespace psd {

namespace {

constexpr int32_t kMaxPsdDimension = 30000;
constexpr double kPi = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Strictly Big-Endian Binary Writer
// -----------------------------------------------------------------------------

class BigEndianWriter {
public:
    BigEndianWriter() = default;
    explicit BigEndianWriter(size_t reserve_bytes) { buf_.reserve(reserve_bytes); }

    void write_u8(uint8_t v) { buf_.push_back(v); }

    void write_i8(int8_t v) { buf_.push_back(static_cast<uint8_t>(v)); }

    void write_u16(uint16_t v) {
        buf_.push_back(static_cast<uint8_t>((v >> 8) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>(v & 0xFFU));
    }

    void write_i16(int16_t v) { write_u16(static_cast<uint16_t>(v)); }

    void write_u32(uint32_t v) {
        buf_.push_back(static_cast<uint8_t>((v >> 24) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((v >> 16) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((v >> 8) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>(v & 0xFFU));
    }

    void write_i32(int32_t v) { write_u32(static_cast<uint32_t>(v)); }

    void write_f64(double v) {
        uint64_t bits = 0;
        static_assert(sizeof(double) == sizeof(uint64_t), "IEEE 754 double must be 8 bytes");
        std::memcpy(&bits, &v, sizeof(bits));
        buf_.push_back(static_cast<uint8_t>((bits >> 56) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((bits >> 48) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((bits >> 40) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((bits >> 32) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((bits >> 24) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((bits >> 16) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>((bits >> 8) & 0xFFU));
        buf_.push_back(static_cast<uint8_t>(bits & 0xFFU));
    }

    void write_fourcc(const char tag[4]) {
        buf_.push_back(static_cast<uint8_t>(tag[0]));
        buf_.push_back(static_cast<uint8_t>(tag[1]));
        buf_.push_back(static_cast<uint8_t>(tag[2]));
        buf_.push_back(static_cast<uint8_t>(tag[3]));
    }

    void write_bytes(const void* data, size_t len) {
        if (len == 0) return;
        const auto* p = static_cast<const uint8_t*>(data);
        buf_.insert(buf_.end(), p, p + len);
    }

    void write_bytes(const std::vector<uint8_t>& data) {
        if (data.empty()) return;
        buf_.insert(buf_.end(), data.begin(), data.end());
    }

    void write_zeros(size_t count) {
        if (count == 0) return;
        buf_.insert(buf_.end(), count, 0x00U);
    }

    void pad_to(size_t alignment) {
        if (alignment <= 1) return;
        const size_t rem = buf_.size() % alignment;
        if (rem != 0) {
            write_zeros(alignment - rem);
        }
    }

    size_t size() const noexcept { return buf_.size(); }

    void patch_u32(size_t offset, uint32_t v) {
        buf_[offset + 0] = static_cast<uint8_t>((v >> 24) & 0xFFU);
        buf_[offset + 1] = static_cast<uint8_t>((v >> 16) & 0xFFU);
        buf_[offset + 2] = static_cast<uint8_t>((v >> 8) & 0xFFU);
        buf_[offset + 3] = static_cast<uint8_t>(v & 0xFFU);
    }

    const std::vector<uint8_t>& data() const noexcept { return buf_; }
    std::vector<uint8_t> take() noexcept { return std::move(buf_); }

private:
    std::vector<uint8_t> buf_;
};

// -----------------------------------------------------------------------------
// UTF-8 -> UTF-16BE Conversion & Escaping Helpers
// -----------------------------------------------------------------------------

std::vector<uint16_t> utf8_to_utf16_units(const std::string& utf8) {
    std::vector<uint16_t> out;
    out.reserve(utf8.size());
    size_t i = 0;
    const size_t n = utf8.size();
    while (i < n) {
        uint32_t cp = 0;
        const uint8_t c0 = static_cast<uint8_t>(utf8[i]);
        if (c0 < 0x80U) {
            cp = c0;
            i += 1;
        } else if ((c0 & 0xE0U) == 0xC0U && i + 1 < n) {
            cp = ((c0 & 0x1FU) << 6) | (static_cast<uint8_t>(utf8[i + 1]) & 0x3FU);
            i += 2;
        } else if ((c0 & 0xF0U) == 0xE0U && i + 2 < n) {
            cp = ((c0 & 0x0FU) << 12) |
                 ((static_cast<uint8_t>(utf8[i + 1]) & 0x3FU) << 6) |
                 (static_cast<uint8_t>(utf8[i + 2]) & 0x3FU);
            i += 3;
        } else if ((c0 & 0xF8U) == 0xF0U && i + 3 < n) {
            cp = ((c0 & 0x07U) << 18) |
                 ((static_cast<uint8_t>(utf8[i + 1]) & 0x3FU) << 12) |
                 ((static_cast<uint8_t>(utf8[i + 2]) & 0x3FU) << 6) |
                 (static_cast<uint8_t>(utf8[i + 3]) & 0x3FU);
            i += 4;
        } else {
            cp = 0xFFFD;
            i += 1;
        }

        if (cp <= 0xFFFFU) {
            out.push_back(static_cast<uint16_t>(cp));
        } else if (cp <= 0x10FFFFU) {
            cp -= 0x10000U;
            out.push_back(static_cast<uint16_t>(0xD800U + (cp >> 10)));
            out.push_back(static_cast<uint16_t>(0xDC00U + (cp & 0x3FFU)));
        } else {
            out.push_back(0xFFFD);
        }
    }
    return out;
}

// Normalize newlines ("\r\n" or "\n") to Photoshop '\r' and guarantee trailing '\r'.
std::string normalize_photoshop_text(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 1);
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\r') {
            if (i + 1 < raw.size() && raw[i + 1] == '\n') {
                ++i;
            }
            out.push_back('\r');
        } else if (raw[i] == '\n') {
            out.push_back('\r');
        } else {
            out.push_back(raw[i]);
        }
    }
    if (out.empty() || out.back() != '\r') {
        out.push_back('\r');
    }
    return out;
}

// Encode a UTF-16BE string with BOM (0xFE 0xFF) escaped for PostScript/EngineData `(...)`.
std::string encode_engine_data_utf16_literal(const std::vector<uint16_t>& units) {
    std::string out;
    out.reserve(units.size() * 2 + 8);
    out.push_back('(');
    out.push_back(static_cast<char>(0xFE));
    out.push_back(static_cast<char>(0xFF));
    auto emit_byte = [&out](uint8_t b) {
        if (b == '(' || b == ')' || b == '\\') {
            out.push_back('\\');
        }
        out.push_back(static_cast<char>(b));
    };
    for (uint16_t u : units) {
        emit_byte(static_cast<uint8_t>((u >> 8) & 0xFFU));
        emit_byte(static_cast<uint8_t>(u & 0xFFU));
    }
    out.push_back(')');
    return out;
}

std::string format_engine_float(double v) {
    if (!std::isfinite(v)) v = 0.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(5) << v;
    std::string s = oss.str();
    while (s.size() > 1 && s.back() == '0' && s[s.size() - 2] != '.') {
        s.pop_back();
    }
    return s;
}

// Sanitize layer name to ASCII/Latin-1 for the legacy Pascal string (max 255 bytes).
std::string to_pascal_ascii_name(const std::string& utf8) {
    std::string out;
    out.reserve(std::min<size_t>(utf8.size(), 255));
    for (unsigned char c : utf8) {
        if (out.size() >= 255) break;
        if (c >= 0x20U && c < 0x7FU) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        out = "Layer";
    }
    return out;
}

// -----------------------------------------------------------------------------
// Photoshop Descriptor Binary Serialization Helpers
// -----------------------------------------------------------------------------

void write_descriptor_unicode_string(BigEndianWriter& w, const std::string& utf8, bool include_null = true) {
    const auto units = utf8_to_utf16_units(utf8);
    if (units.empty() && !include_null) {
        w.write_u32(0);
        return;
    }
    const uint32_t count = static_cast<uint32_t>(units.size() + (include_null ? 1U : 0U));
    w.write_u32(count);
    for (uint16_t u : units) {
        w.write_u16(u);
    }
    if (include_null) {
        w.write_u16(0x0000);
    }
}

// Key/ClassID: if 4-byte OSType, length = 0 followed by 4 bytes; otherwise length + ASCII string.
void write_descriptor_id(BigEndianWriter& w, const std::string& id) {
    if (id.size() == 4) {
        w.write_u32(0);
        w.write_bytes(id.data(), 4);
    } else {
        w.write_u32(static_cast<uint32_t>(id.size()));
        w.write_bytes(id.data(), id.size());
    }
}

void write_descriptor_enum(
    BigEndianWriter& w,
    const std::string& key,
    const std::string& type_id,
    const std::string& value_id) {
    write_descriptor_id(w, key);
    w.write_fourcc("enum");
    write_descriptor_id(w, type_id);
    write_descriptor_id(w, value_id);
}

void write_descriptor_doub(BigEndianWriter& w, const std::string& key, double val) {
    write_descriptor_id(w, key);
    w.write_fourcc("doub");
    w.write_f64(val);
}

void write_descriptor_untf(
    BigEndianWriter& w,
    const std::string& key,
    const char unit[4],
    double val) {
    write_descriptor_id(w, key);
    w.write_fourcc("UntF");
    w.write_fourcc(unit);
    w.write_f64(val);
}

void write_descriptor_bool(BigEndianWriter& w, const std::string& key, bool val) {
    write_descriptor_id(w, key);
    w.write_fourcc("bool");
    w.write_u8(val ? 1U : 0U);
}

// -----------------------------------------------------------------------------
// Stage 2: Tagged Block Builders (`luni`, `TySh`, `lfx2`)
// -----------------------------------------------------------------------------

void append_tagged_block(
    BigEndianWriter& dst,
    const char key[4],
    const std::vector<uint8_t>& payload) {
    dst.write_fourcc("8BIM");
    dst.write_fourcc(key);
    const size_t padded_len = (payload.size() + 3U) & ~size_t(3U);
    dst.write_u32(static_cast<uint32_t>(padded_len));
    dst.write_bytes(payload);
    dst.write_zeros(padded_len - payload.size());
}

void append_luni_block(BigEndianWriter& dst, const std::string& utf8_name) {
    const auto units = utf8_to_utf16_units(utf8_name);
    BigEndianWriter payload(4 + units.size() * 2 + 4);
    payload.write_u32(static_cast<uint32_t>(units.size()));
    for (uint16_t u : units) {
        payload.write_u16(u);
    }
    append_tagged_block(dst, "luni", payload.data());
}

std::string build_engine_data_payload(
    const TypeLayerSpec& spec,
    const std::vector<uint16_t>& text_utf16_units) {
    const std::string font_name = spec.font_name.empty() ? "ArialMT" : spec.font_name;
    const auto font_units = utf8_to_utf16_units(font_name);
    const double font_size = std::max(1.0, spec.font_size);
    const double leading = (spec.leading > 0.0) ? spec.leading : (font_size * 1.2);
    const double r = static_cast<double>(spec.fill_color.r) / 255.0;
    const double g = static_cast<double>(spec.fill_color.g) / 255.0;
    const double b = static_cast<double>(spec.fill_color.b) / 255.0;
    const int justification = static_cast<int>(spec.align);
    const size_t run_len = std::max<size_t>(1, text_utf16_units.size());
    const int writing_dir = spec.vertical ? 2 : 0;
    const double box_w = std::max(1.0, spec.box_width);
    const double box_h = std::max(1.0, spec.box_height);

    std::ostringstream oss;
    oss << "\n<<\n"
        << "\t/EngineDict <<\n"
        << "\t\t/Editor <<\n"
        << "\t\t\t/Text " << encode_engine_data_utf16_literal(text_utf16_units) << "\n"
        << "\t\t>>\n"
        << "\t\t/ParagraphRun <<\n"
        << "\t\t\t/DefaultRunData <<\n"
        << "\t\t\t\t/ParagraphSheet <<\n"
        << "\t\t\t\t\t/DefaultStyleSheet 0\n"
        << "\t\t\t\t\t/Properties <<\n"
        << "\t\t\t\t\t>>\n"
        << "\t\t\t\t>>\n"
        << "\t\t\t\t/Adjustments <<\n"
        << "\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n"
        << "\t\t\t\t\t/XY [ 0.0 0.0 ]\n"
        << "\t\t\t\t>>\n"
        << "\t\t\t>>\n"
        << "\t\t\t/RunArray [ <<\n"
        << "\t\t\t\t/ParagraphSheet <<\n"
        << "\t\t\t\t\t/DefaultStyleSheet 0\n"
        << "\t\t\t\t\t/Properties <<\n"
        << "\t\t\t\t\t\t/Justification " << justification << "\n"
        << "\t\t\t\t\t\t/AutoHyphenate " << (spec.auto_hyphenate ? "true" : "false") << "\n"
        << "\t\t\t\t\t>>\n"
        << "\t\t\t\t>>\n"
        << "\t\t\t\t/Adjustments <<\n"
        << "\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n"
        << "\t\t\t\t\t/XY [ 0.0 0.0 ]\n"
        << "\t\t\t\t>>\n"
        << "\t\t\t>> ]\n"
        << "\t\t\t/RunLengthArray [ " << run_len << " ]\n"
        << "\t\t\t/IsJoinable 1\n"
        << "\t\t>>\n"
        << "\t\t/StyleRun <<\n"
        << "\t\t\t/DefaultRunData <<\n"
        << "\t\t\t\t/StyleSheet <<\n"
        << "\t\t\t\t\t/StyleSheetData <<\n"
        << "\t\t\t\t\t>>\n"
        << "\t\t\t\t>>\n"
        << "\t\t\t>>\n"
        << "\t\t\t/RunArray [ <<\n"
        << "\t\t\t\t/StyleSheet <<\n"
        << "\t\t\t\t\t/StyleSheetData <<\n"
        << "\t\t\t\t\t\t/Font 0\n"
        << "\t\t\t\t\t\t/FontSize " << format_engine_float(font_size) << "\n"
        << "\t\t\t\t\t\t/FauxBold " << (spec.faux_bold ? "true" : "false") << "\n"
        << "\t\t\t\t\t\t/FauxItalic " << (spec.faux_italic ? "true" : "false") << "\n"
        << "\t\t\t\t\t\t/AutoLeading false\n"
        << "\t\t\t\t\t\t/Leading " << format_engine_float(leading) << "\n"
        << "\t\t\t\t\t\t/FillColor <<\n"
        << "\t\t\t\t\t\t\t/Type 1\n"
        << "\t\t\t\t\t\t\t/Values [ 1.0 "
        << format_engine_float(r) << " "
        << format_engine_float(g) << " "
        << format_engine_float(b) << " ]\n"
        << "\t\t\t\t\t\t>>\n"
        << "\t\t\t\t\t>>\n"
        << "\t\t\t\t>>\n"
        << "\t\t\t>> ]\n"
        << "\t\t\t/RunLengthArray [ " << run_len << " ]\n"
        << "\t\t\t/IsJoinable 2\n"
        << "\t\t>>\n"
        << "\t\t/GridInfo <<\n"
        << "\t\t\t/GridIsOn false\n"
        << "\t\t\t/ShowGrid false\n"
        << "\t\t\t/GridSize 18.0\n"
        << "\t\t\t/GridLeading 22.0\n"
        << "\t\t\t/GridColor <<\n"
        << "\t\t\t\t/Type 1\n"
        << "\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]\n"
        << "\t\t\t>>\n"
        << "\t\t\t/GridLeadingFillColor <<\n"
        << "\t\t\t\t/Type 1\n"
        << "\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]\n"
        << "\t\t\t>>\n"
        << "\t\t\t/AlignLineHeightToGridFlags false\n"
        << "\t\t>>\n"
        << "\t\t/AntiAlias 4\n"
        << "\t\t/UseFractionalGlyphWidths true\n"
        << "\t\t/Rendered <<\n"
        << "\t\t\t/Version 1\n"
        << "\t\t\t/Shapes <<\n"
        << "\t\t\t\t/WritingDirection " << writing_dir << "\n"
        << "\t\t\t\t/Children [ <<\n"
        << "\t\t\t\t\t/ShapeType 1\n"
        << "\t\t\t\t\t/Procession 0\n"
        << "\t\t\t\t\t/Lines <<\n"
        << "\t\t\t\t\t\t/WritingDirection " << writing_dir << "\n"
        << "\t\t\t\t\t\t/Children [ ]\n"
        << "\t\t\t\t\t>>\n"
        << "\t\t\t\t\t/Cookie <<\n"
        << "\t\t\t\t\t\t/Photoshop <<\n"
        << "\t\t\t\t\t\t\t/ShapeType 1\n"
        << "\t\t\t\t\t\t\t/PointBase [ 0.0 0.0 ]\n"
        << "\t\t\t\t\t\t\t/BoxBounds [ 0.0 0.0 "
        << format_engine_float(box_h) << " "
        << format_engine_float(box_w) << " ]\n"
        << "\t\t\t\t\t\t>>\n"
        << "\t\t\t\t\t>>\n"
        << "\t\t\t\t>> ]\n"
        << "\t\t\t>>\n"
        << "\t\t>>\n"
        << "\t>>\n"
        << "\t/ResourceDict <<\n"
        << "\t\t/FontSet [ <<\n"
        << "\t\t\t/Name " << encode_engine_data_utf16_literal(font_units) << "\n"
        << "\t\t\t/Script 0\n"
        << "\t\t\t/FontType 1\n"
        << "\t\t\t/Synthetic 0\n"
        << "\t\t>> ]\n"
        << "\t>>\n"
        << ">>";
    return oss.str();
}

void append_tysh_block(BigEndianWriter& dst, const TypeLayerSpec& spec) {
    const std::string normalized_text = normalize_photoshop_text(spec.text);
    const auto text_units = utf8_to_utf16_units(normalized_text);
    const std::string engine_data = build_engine_data_payload(spec, text_units);

    BigEndianWriter payload(512 + engine_data.size());

    // 1. TySh Version = 1
    payload.write_i16(1);

    // 2. 2D Affine Transform (6 x IEEE-754 big-endian double): [xx, xy, yx, yy, tx, ty]
    const double rad = spec.rotation_deg * (kPi / 180.0);
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    payload.write_f64(c);
    payload.write_f64(s);
    payload.write_f64(-s);
    payload.write_f64(c);
    payload.write_f64(spec.box_x);
    payload.write_f64(spec.box_y);

    // 3. Text version = 50, Descriptor version = 16
    payload.write_i16(50);
    payload.write_i32(16);

    // 4. Text Data Descriptor (`TxLr`)
    write_descriptor_unicode_string(payload, "", false);  // empty class name
    write_descriptor_id(payload, "TxLr");
    payload.write_u32(5);  // 5 descriptor items

    // Item 1: `Txt ` (TEXT: UTF-16BE '\r'-terminated + null terminator)
    write_descriptor_id(payload, "Txt ");
    payload.write_fourcc("TEXT");
    payload.write_u32(static_cast<uint32_t>(text_units.size() + 1U));
    for (uint16_t u : text_units) {
        payload.write_u16(u);
    }
    payload.write_u16(0x0000);

    // Item 2: `textGridding`
    write_descriptor_enum(payload, "textGridding", "textGridding", "None");

    // Item 3: `Ornt` (Orientation)
    write_descriptor_enum(payload, "Ornt", "Ornt", spec.vertical ? "Vrtc" : "Hrzn");

    // Item 4: `AntA` (Anti-aliasing)
    write_descriptor_enum(payload, "AntA", "Annt", "AnCr");

    // Item 5: `EngineData` (tdta raw byte blob)
    write_descriptor_id(payload, "EngineData");
    payload.write_fourcc("tdta");
    payload.write_u32(static_cast<uint32_t>(engine_data.size()));
    payload.write_bytes(engine_data.data(), engine_data.size());

    // 5. Warp version = 1, Warp descriptor version = 16
    payload.write_i16(1);
    payload.write_i32(16);

    // 6. Warp Descriptor (`warp`)
    write_descriptor_unicode_string(payload, "", false);
    write_descriptor_id(payload, "warp");
    payload.write_u32(4);
    write_descriptor_enum(payload, "warpStyle", "warpStyle", "warpNone");
    write_descriptor_doub(payload, "warpValue", 0.0);
    write_descriptor_doub(payload, "warpPerspective", 0.0);
    write_descriptor_doub(payload, "warpPerspectiveOther", 0.0);

    // 7. Bounding box (left, top, right, bottom as 4 x big-endian double)
    const double left = spec.box_x;
    const double top = spec.box_y;
    const double right = spec.box_x + std::max(1.0, spec.box_width);
    const double bottom = spec.box_y + std::max(1.0, spec.box_height);
    payload.write_f64(left);
    payload.write_f64(top);
    payload.write_f64(right);
    payload.write_f64(bottom);

    append_tagged_block(dst, "TySh", payload.data());
}

void append_lfx2_outline_block(BigEndianWriter& dst, const TextOutlineSpec& outline) {
    if (!outline.enabled || outline.width_px <= 0.0) {
        return;
    }

    BigEndianWriter payload(256);

    // Object Effects version = 0, Descriptor version = 16
    payload.write_i32(0);
    payload.write_i32(16);

    // Root descriptor (`Lefx`)
    write_descriptor_unicode_string(payload, "", false);
    write_descriptor_id(payload, "Lefx");
    payload.write_u32(3);  // Scl , masterFXSwitch, FrFX

    write_descriptor_untf(payload, "Scl ", "#Prc", 100.0);
    write_descriptor_bool(payload, "masterFXSwitch", true);

    // `FrFX` (FrameFX / Outside Stroke)
    write_descriptor_id(payload, "FrFX");
    payload.write_fourcc("Objc");
    write_descriptor_unicode_string(payload, "", false);
    write_descriptor_id(payload, "FrFX");
    payload.write_u32(7);

    write_descriptor_bool(payload, "enab", true);
    write_descriptor_enum(payload, "Styl", "FStl", "OutF");  // Outside stroke
    write_descriptor_enum(payload, "PntT", "FrFl", "SClr");  // Solid color
    write_descriptor_enum(payload, "Md  ", "BlnM", "Nrml");  // Normal blend
    const double opct = std::max(0.0, std::min(100.0, outline.opacity_pct));
    write_descriptor_untf(payload, "Opct", "#Prc", opct);
    write_descriptor_untf(payload, "Sz  ", "#Pxl", outline.width_px);

    // `Clr ` (`RGBC` color descriptor)
    write_descriptor_id(payload, "Clr ");
    payload.write_fourcc("Objc");
    write_descriptor_unicode_string(payload, "", false);
    write_descriptor_id(payload, "RGBC");
    payload.write_u32(3);
    write_descriptor_doub(payload, "Rd  ", static_cast<double>(outline.color.r));
    write_descriptor_doub(payload, "Grn ", static_cast<double>(outline.color.g));
    write_descriptor_doub(payload, "Bl  ", static_cast<double>(outline.color.b));

    append_tagged_block(dst, "lfx2", payload.data());
}

// -----------------------------------------------------------------------------
// Composite Blending Helper
// -----------------------------------------------------------------------------

std::vector<uint8_t> build_merged_composite(
    int32_t doc_w,
    int32_t doc_h,
    uint16_t doc_channels,
    const std::vector<PsdLayer>& layers) {
    const size_t px_count = static_cast<size_t>(doc_w) * static_cast<size_t>(doc_h);
    std::vector<uint8_t> comp(px_count * 4, 0);

    // Initialize opaque white for 3-channel RGB documents, transparent for 4-channel RGBA.
    if (doc_channels == 3) {
        for (size_t i = 0; i < px_count; ++i) {
            comp[i * 4 + 0] = 255;
            comp[i * 4 + 1] = 255;
            comp[i * 4 + 2] = 255;
            comp[i * 4 + 3] = 255;
        }
    }

    for (const auto& layer : layers) {
        if (!layer.visible || layer.opacity == 0 || layer.width <= 0 || layer.height <= 0) {
            continue;
        }
        const int32_t x0 = std::max<int32_t>(0, layer.left);
        const int32_t y0 = std::max<int32_t>(0, layer.top);
        const int32_t x1 = std::min<int32_t>(doc_w, layer.left + layer.width);
        const int32_t y1 = std::min<int32_t>(doc_h, layer.top + layer.height);
        if (x0 >= x1 || y0 >= y1) continue;

        const size_t src_stride = layer.has_alpha ? 4U : 3U;
        const uint32_t layer_op = layer.opacity;

        for (int32_t y = y0; y < y1; ++y) {
            const int32_t sy = y - layer.top;
            for (int32_t x = x0; x < x1; ++x) {
                const int32_t sx = x - layer.left;
                const size_t sidx =
                    (static_cast<size_t>(sy) * static_cast<size_t>(layer.width) + static_cast<size_t>(sx)) * src_stride;
                const uint32_t sr = layer.pixels[sidx + 0];
                const uint32_t sg = layer.pixels[sidx + 1];
                const uint32_t sb = layer.pixels[sidx + 2];
                const uint32_t sa_raw = layer.has_alpha ? layer.pixels[sidx + 3] : 255U;
                const uint32_t sa = (sa_raw * layer_op + 127U) / 255U;
                if (sa == 0) continue;

                const size_t didx =
                    (static_cast<size_t>(y) * static_cast<size_t>(doc_w) + static_cast<size_t>(x)) * 4U;

                if (sa == 255U) {
                    comp[didx + 0] = static_cast<uint8_t>(sr);
                    comp[didx + 1] = static_cast<uint8_t>(sg);
                    comp[didx + 2] = static_cast<uint8_t>(sb);
                    comp[didx + 3] = 255U;
                } else {
                    const uint32_t da = comp[didx + 3];
                    const uint32_t inv_sa = 255U - sa;
                    const uint32_t out_a = sa + (da * inv_sa + 127U) / 255U;
                    if (out_a > 0) {
                        const uint32_t dr = comp[didx + 0];
                        const uint32_t dg = comp[didx + 1];
                        const uint32_t db = comp[didx + 2];
                        const uint32_t da_inv = (da * inv_sa + 127U) / 255U;
                        comp[didx + 0] = static_cast<uint8_t>((sr * sa + dr * da_inv + out_a / 2U) / out_a);
                        comp[didx + 1] = static_cast<uint8_t>((sg * sa + dg * da_inv + out_a / 2U) / out_a);
                        comp[didx + 2] = static_cast<uint8_t>((sb * sa + db * da_inv + out_a / 2U) / out_a);
                        comp[didx + 3] = static_cast<uint8_t>(std::min<uint32_t>(255U, out_a));
                    }
                }
            }
        }
    }

    if (doc_channels == 4) {
        return comp;
    }

    std::vector<uint8_t> rgb(px_count * 3);
    for (size_t i = 0; i < px_count; ++i) {
        rgb[i * 3 + 0] = comp[i * 4 + 0];
        rgb[i * 3 + 1] = comp[i * 4 + 1];
        rgb[i * 3 + 2] = comp[i * 4 + 2];
    }
    return rgb;
}

// Encoded single-channel data ready to be written to Channel Image Data.
struct EncodedChannel {
    int16_t channel_id = 0;
    std::vector<uint8_t> bytes;  // includes 2-byte compression header + row counts + RLE data
};

EncodedChannel encode_layer_channel_rle(
    int16_t channel_id,
    const uint8_t* layer_pixels,
    int32_t width,
    int32_t height,
    size_t pixel_stride,
    size_t channel_offset) {
    EncodedChannel ch;
    ch.channel_id = channel_id;

    BigEndianWriter w;
    // Compression = 1 (RLE PackBits)
    w.write_u16(1);

    if (width <= 0 || height <= 0) {
        ch.bytes = w.take();
        return ch;
    }

    std::vector<uint8_t> row_buf(static_cast<size_t>(width));
    std::vector<std::vector<uint8_t>> encoded_rows;
    encoded_rows.reserve(static_cast<size_t>(height));

    for (int32_t y = 0; y < height; ++y) {
        const size_t row_start = static_cast<size_t>(y) * static_cast<size_t>(width) * pixel_stride;
        for (int32_t x = 0; x < width; ++x) {
            row_buf[static_cast<size_t>(x)] =
                layer_pixels[row_start + static_cast<size_t>(x) * pixel_stride + channel_offset];
        }
        auto rle = packbits_encode_row(row_buf.data(), row_buf.size());
        if (rle.size() > std::numeric_limits<uint16_t>::max()) {
            throw PsdWriteError("RLE-encoded scanline exceeds PSD v1 16-bit row count limit.");
        }
        encoded_rows.push_back(std::move(rle));
    }

    for (const auto& r : encoded_rows) {
        w.write_u16(static_cast<uint16_t>(r.size()));
    }
    for (const auto& r : encoded_rows) {
        w.write_bytes(r);
    }

    ch.bytes = w.take();
    return ch;
}

}  // namespace

// -----------------------------------------------------------------------------
// PackBits RLE Encoder & Decoder
// -----------------------------------------------------------------------------

std::vector<uint8_t> packbits_encode_row(const uint8_t* src, size_t len) {
    std::vector<uint8_t> out;
    if (!src || len == 0) {
        return out;
    }
    out.reserve(len + (len / 128) + 2);

    size_t i = 0;
    while (i < len) {
        // Check for a replicate run of length >= 2 (prefer replicate for >= 3, or >= 2 at end/standalone)
        size_t run_len = 1;
        while (i + run_len < len && run_len < 128 && src[i + run_len] == src[i]) {
            ++run_len;
        }

        if (run_len >= 3) {
            // Replicate run: header = (1 - run_len) as signed int8 -> 257 - run_len unsigned
            out.push_back(static_cast<uint8_t>(static_cast<int8_t>(1 - static_cast<int>(run_len))));
            out.push_back(src[i]);
            i += run_len;
        } else {
            // Literal run: accumulate up to 128 bytes until we hit a replicate run of >= 3
            const size_t lit_start = i;
            size_t lit_len = 0;
            while (i < len && lit_len < 128) {
                size_t next_run = 1;
                while (i + next_run < len && next_run < 3 && src[i + next_run] == src[i]) {
                    ++next_run;
                }
                if (next_run >= 3) {
                    break;
                }
                ++i;
                ++lit_len;
            }
            out.push_back(static_cast<uint8_t>(lit_len - 1));
            out.insert(out.end(), src + lit_start, src + lit_start + lit_len);
        }
    }

    return out;
}

std::vector<uint8_t> packbits_decode_row(const uint8_t* src, size_t src_len, size_t expected_len) {
    std::vector<uint8_t> out;
    out.reserve(expected_len);

    size_t i = 0;
    while (i < src_len) {
        const int8_t header = static_cast<int8_t>(src[i++]);
        if (header >= 0) {
            const size_t count = static_cast<size_t>(header) + 1U;
            if (i + count > src_len) {
                throw PsdWriteError("PackBits decode error: truncated literal run.");
            }
            out.insert(out.end(), src + i, src + i + count);
            i += count;
        } else if (header != -128) {
            const size_t count = static_cast<size_t>(1 - static_cast<int>(header));
            if (i >= src_len) {
                throw PsdWriteError("PackBits decode error: truncated replicate run.");
            }
            const uint8_t val = src[i++];
            out.insert(out.end(), count, val);
        }
        // header == -128 is a PackBits no-op
    }

    if (expected_len > 0 && out.size() != expected_len) {
        throw PsdWriteError(
            "PackBits decode length mismatch: expected " + std::to_string(expected_len) +
            ", got " + std::to_string(out.size()) + ".");
    }
    return out;
}

// -----------------------------------------------------------------------------
// Color & Layer Helpers
// -----------------------------------------------------------------------------

ColorRGB parse_hex_color(const std::string& hex, ColorRGB fallback) {
    std::string s;
    for (unsigned char c : hex) {
        if (!std::isspace(c)) {
            s.push_back(static_cast<char>(c));
        }
    }
    if (!s.empty() && s.front() == '#') {
        s.erase(s.begin());
    }
    auto hex_val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };
    if (s.size() == 3) {
        const int r = hex_val(s[0]);
        const int g = hex_val(s[1]);
        const int b = hex_val(s[2]);
        if (r >= 0 && g >= 0 && b >= 0) {
            return {
                static_cast<uint8_t>(r * 17),
                static_cast<uint8_t>(g * 17),
                static_cast<uint8_t>(b * 17),
            };
        }
    } else if (s.size() >= 6) {
        const int r1 = hex_val(s[0]), r2 = hex_val(s[1]);
        const int g1 = hex_val(s[2]), g2 = hex_val(s[3]);
        const int b1 = hex_val(s[4]), b2 = hex_val(s[5]);
        if (r1 >= 0 && r2 >= 0 && g1 >= 0 && g2 >= 0 && b1 >= 0 && b2 >= 0) {
            return {
                static_cast<uint8_t>((r1 << 4) | r2),
                static_cast<uint8_t>((g1 << 4) | g2),
                static_cast<uint8_t>((b1 << 4) | b2),
            };
        }
    }
    return fallback;
}

PsdLayer build_cleanup_diff_layer(
    const uint8_t* original_rgba,
    const uint8_t* cleaned_rgba,
    int32_t width,
    int32_t height,
    const uint8_t* mask_gray,
    uint8_t diff_threshold,
    const std::string& layer_name) {
    PsdLayer layer;
    layer.name = layer_name;
    layer.has_alpha = true;
    layer.opacity = 255;
    layer.visible = true;

    if (!original_rgba || !cleaned_rgba || width <= 0 || height <= 0) {
        return layer;
    }

    int32_t min_x = width;
    int32_t min_y = height;
    int32_t max_x = -1;
    int32_t max_y = -1;

    const size_t px_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> alpha_map(px_count, 0);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const size_t off = idx * 4U;
            const int dr = std::abs(static_cast<int>(cleaned_rgba[off + 0]) - static_cast<int>(original_rgba[off + 0]));
            const int dg = std::abs(static_cast<int>(cleaned_rgba[off + 1]) - static_cast<int>(original_rgba[off + 1]));
            const int db = std::abs(static_cast<int>(cleaned_rgba[off + 2]) - static_cast<int>(original_rgba[off + 2]));
            const bool mask_hit = (mask_gray != nullptr && mask_gray[idx] > 0);
            const bool diff_hit = (dr > diff_threshold || dg > diff_threshold || db > diff_threshold);
            if (mask_hit || diff_hit) {
                alpha_map[idx] = mask_hit ? mask_gray[idx] : 255U;
                if (x < min_x) min_x = x;
                if (y < min_y) min_y = y;
                if (x > max_x) max_x = x;
                if (y > max_y) max_y = y;
            }
        }
    }

    if (max_x < min_x || max_y < min_y) {
        return layer;  // 0x0 empty transparent layer
    }

    layer.left = min_x;
    layer.top = min_y;
    layer.width = max_x - min_x + 1;
    layer.height = max_y - min_y + 1;
    layer.pixels.resize(static_cast<size_t>(layer.width) * static_cast<size_t>(layer.height) * 4U);

    for (int32_t y = 0; y < layer.height; ++y) {
        const int32_t sy = min_y + y;
        for (int32_t x = 0; x < layer.width; ++x) {
            const int32_t sx = min_x + x;
            const size_t src_idx = static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx);
            const size_t dst_idx = static_cast<size_t>(y) * static_cast<size_t>(layer.width) + static_cast<size_t>(x);
            layer.pixels[dst_idx * 4 + 0] = cleaned_rgba[src_idx * 4 + 0];
            layer.pixels[dst_idx * 4 + 1] = cleaned_rgba[src_idx * 4 + 1];
            layer.pixels[dst_idx * 4 + 2] = cleaned_rgba[src_idx * 4 + 2];
            layer.pixels[dst_idx * 4 + 3] = alpha_map[src_idx];
        }
    }

    return layer;
}

PsdLayer build_cropped_rgba_layer(
    const uint8_t* full_rgba,
    int32_t page_width,
    int32_t page_height,
    const std::string& layer_name,
    uint8_t opacity,
    bool visible) {
    PsdLayer layer;
    layer.name = layer_name;
    layer.has_alpha = true;
    layer.opacity = opacity;
    layer.visible = visible;

    if (!full_rgba || page_width <= 0 || page_height <= 0) {
        return layer;
    }

    int32_t min_x = page_width;
    int32_t min_y = page_height;
    int32_t max_x = -1;
    int32_t max_y = -1;

    for (int32_t y = 0; y < page_height; ++y) {
        for (int32_t x = 0; x < page_width; ++x) {
            const size_t off =
                (static_cast<size_t>(y) * static_cast<size_t>(page_width) + static_cast<size_t>(x)) * 4U;
            if (full_rgba[off + 3] > 0) {
                if (x < min_x) min_x = x;
                if (y < min_y) min_y = y;
                if (x > max_x) max_x = x;
                if (y > max_y) max_y = y;
            }
        }
    }

    if (max_x < min_x || max_y < min_y) {
        return layer;
    }

    layer.left = min_x;
    layer.top = min_y;
    layer.width = max_x - min_x + 1;
    layer.height = max_y - min_y + 1;
    layer.pixels.resize(static_cast<size_t>(layer.width) * static_cast<size_t>(layer.height) * 4U);

    for (int32_t y = 0; y < layer.height; ++y) {
        const size_t src_row =
            (static_cast<size_t>(min_y + y) * static_cast<size_t>(page_width) + static_cast<size_t>(min_x)) * 4U;
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(layer.width) * 4U;
        std::memcpy(layer.pixels.data() + dst_row, full_rgba + src_row, static_cast<size_t>(layer.width) * 4U);
    }

    return layer;
}

// -----------------------------------------------------------------------------
// Main PSD Version 1 Binary Writer
// -----------------------------------------------------------------------------

std::vector<uint8_t> write_psd_bytes(
    const PsdDocumentOptions& options,
    const std::vector<PsdLayer>& layers,
    const std::vector<uint8_t>& composite_rgb_or_rgba) {
    if (options.width <= 0 || options.height <= 0) {
        throw PsdWriteError("PSD document width and height must be positive.");
    }
    if (options.width > kMaxPsdDimension || options.height > kMaxPsdDimension) {
        throw PsdWriteError("PSD v1 dimensions cannot exceed 30000x30000 pixels.");
    }
    if (options.channels != 3 && options.channels != 4) {
        throw PsdWriteError("PSD channels must be 3 (RGB) or 4 (RGBA).");
    }
    if (layers.size() > 32767) {
        throw PsdWriteError("PSD layer count exceeds 16-bit signed limit.");
    }

    // Validate layer buffers
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& l = layers[i];
        if (l.width < 0 || l.height < 0) {
            throw PsdWriteError("Layer #" + std::to_string(i) + " has negative dimensions.");
        }
        const size_t expected =
            static_cast<size_t>(l.width) * static_cast<size_t>(l.height) * (l.has_alpha ? 4U : 3U);
        if (l.pixels.size() != expected) {
            throw PsdWriteError(
                "Layer '" + l.name + "' pixel buffer size mismatch: expected " +
                std::to_string(expected) + ", got " + std::to_string(l.pixels.size()) + ".");
        }
    }

    // Prepare merged composite buffer
    const size_t expected_comp =
        static_cast<size_t>(options.width) * static_cast<size_t>(options.height) * options.channels;
    std::vector<uint8_t> auto_composite;
    const std::vector<uint8_t>* comp_ptr = &composite_rgb_or_rgba;
    if (composite_rgb_or_rgba.empty()) {
        auto_composite = build_merged_composite(options.width, options.height, options.channels, layers);
        comp_ptr = &auto_composite;
    } else if (composite_rgb_or_rgba.size() != expected_comp) {
        throw PsdWriteError(
            "Composite buffer size mismatch: expected " + std::to_string(expected_comp) +
            ", got " + std::to_string(composite_rgb_or_rgba.size()) + ".");
    }

    BigEndianWriter psd(expected_comp + 4096);

    // =========================================================================
    // Section 1: File Header (26 bytes)
    // =========================================================================
    psd.write_fourcc("8BPS");                                // Signature
    psd.write_u16(1);                                        // Version 1 (.psd)
    psd.write_zeros(6);                                      // Reserved
    psd.write_u16(options.channels);                         // Channels (3=RGB, 4=RGBA)
    psd.write_u32(static_cast<uint32_t>(options.height));    // Height
    psd.write_u32(static_cast<uint32_t>(options.width));     // Width
    psd.write_u16(8);                                        // Depth: 8 bits/channel
    psd.write_u16(3);                                        // Color Mode: 3 (RGB)

    // =========================================================================
    // Section 2: Color Mode Data (0 bytes for RGB)
    // =========================================================================
    psd.write_u32(0);

    // =========================================================================
    // Section 3: Image Resources (ResolutionInfo 0x03ED = 1005)
    // =========================================================================
    {
        BigEndianWriter res_sec(32);
        const double dpi = (options.dpi > 0.0 && std::isfinite(options.dpi)) ? options.dpi : 72.0;
        const uint32_t fixed_dpi = static_cast<uint32_t>(std::llround(dpi * 65536.0));

        res_sec.write_fourcc("8BIM");
        res_sec.write_u16(1005);   // ResolutionInfo
        res_sec.write_u16(0);      // Empty Pascal string padded to even (2 bytes)
        res_sec.write_u32(16);     // Payload size = 16 bytes
        res_sec.write_u32(fixed_dpi);  // hRes (Fixed 16.16)
        res_sec.write_u16(1);          // hResUnit = pixels per inch
        res_sec.write_u16(1);          // widthUnit = inches
        res_sec.write_u32(fixed_dpi);  // vRes (Fixed 16.16)
        res_sec.write_u16(1);          // vResUnit = pixels per inch
        res_sec.write_u16(1);          // heightUnit = inches

        psd.write_u32(static_cast<uint32_t>(res_sec.size()));
        psd.write_bytes(res_sec.data());
    }

    // =========================================================================
    // Section 4: Layer and Mask Information
    // =========================================================================
    if (layers.empty()) {
        psd.write_u32(0);
    } else {
        // Pre-encode all layer channels with RLE PackBits so we know exact channel lengths
        std::vector<std::vector<EncodedChannel>> all_layer_channels;
        all_layer_channels.reserve(layers.size());

        for (const auto& layer : layers) {
            std::vector<EncodedChannel> chs;
            const size_t stride = layer.has_alpha ? 4U : 3U;
            if (layer.has_alpha) {
                chs.reserve(4);
                chs.push_back(encode_layer_channel_rle(-1, layer.pixels.data(), layer.width, layer.height, stride, 3));
                chs.push_back(encode_layer_channel_rle(0, layer.pixels.data(), layer.width, layer.height, stride, 0));
                chs.push_back(encode_layer_channel_rle(1, layer.pixels.data(), layer.width, layer.height, stride, 1));
                chs.push_back(encode_layer_channel_rle(2, layer.pixels.data(), layer.width, layer.height, stride, 2));
            } else {
                chs.reserve(3);
                chs.push_back(encode_layer_channel_rle(0, layer.pixels.data(), layer.width, layer.height, stride, 0));
                chs.push_back(encode_layer_channel_rle(1, layer.pixels.data(), layer.width, layer.height, stride, 1));
                chs.push_back(encode_layer_channel_rle(2, layer.pixels.data(), layer.width, layer.height, stride, 2));
            }
            all_layer_channels.push_back(std::move(chs));
        }

        // Build Layer Info payload
        BigEndianWriter layer_info(4096);
        // Layer count: negative if merged composite includes alpha transparency
        const int16_t signed_layer_count = (options.channels == 4)
            ? static_cast<int16_t>(-static_cast<int>(layers.size()))
            : static_cast<int16_t>(layers.size());
        layer_info.write_i16(signed_layer_count);

        // Layer Records
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            const auto& chs = all_layer_channels[i];

            const int32_t top = layer.top;
            const int32_t left = layer.left;
            const int32_t bottom = layer.top + layer.height;
            const int32_t right = layer.left + layer.width;

            layer_info.write_i32(top);
            layer_info.write_i32(left);
            layer_info.write_i32(bottom);
            layer_info.write_i32(right);

            layer_info.write_u16(static_cast<uint16_t>(chs.size()));
            for (const auto& ch : chs) {
                layer_info.write_i16(ch.channel_id);
                layer_info.write_u32(static_cast<uint32_t>(ch.bytes.size()));
            }

            layer_info.write_fourcc("8BIM");
            layer_info.write_fourcc("norm");
            layer_info.write_u8(layer.opacity);
            layer_info.write_u8(0);  // clipping: base
            // Flags: bit 1 = 1 means hidden, 0 means visible
            const uint8_t flags = layer.visible ? 0x00U : 0x02U;
            layer_info.write_u8(flags);
            layer_info.write_u8(0);  // filler

            // Extra Data Field
            BigEndianWriter extra(128);
            extra.write_u32(0);  // Layer mask data length = 0
            extra.write_u32(0);  // Layer blending ranges length = 0

            // Layer Name: Pascal string padded to a multiple of 4 bytes
            const std::string pascal_name = to_pascal_ascii_name(layer.name);
            extra.write_u8(static_cast<uint8_t>(pascal_name.size()));
            extra.write_bytes(pascal_name.data(), pascal_name.size());
            const size_t name_total = 1U + pascal_name.size();
            const size_t name_pad = (4U - (name_total % 4U)) % 4U;
            extra.write_zeros(name_pad);

            // Additional Layer Information (4-byte aligned tagged blocks)
            append_luni_block(extra, layer.name.empty() ? pascal_name : layer.name);

            if (options.editable_text_layers && layer.type_spec.has_value()) {
                append_tysh_block(extra, *layer.type_spec);
                if (layer.type_spec->outline.enabled && layer.type_spec->outline.width_px > 0.0) {
                    append_lfx2_outline_block(extra, layer.type_spec->outline);
                }
            }

            layer_info.write_u32(static_cast<uint32_t>(extra.size()));
            layer_info.write_bytes(extra.data());
        }

        // Channel Image Data
        for (const auto& chs : all_layer_channels) {
            for (const auto& ch : chs) {
                layer_info.write_bytes(ch.bytes);
            }
        }

        // Layer Info length must be rounded up to a multiple of 2 bytes (4-byte aligned is also even)
        layer_info.pad_to(4);

        // Assemble Layer & Mask Information section:
        // [4B layer_info_len] [layer_info] [4B global_layer_mask_len = 0]
        BigEndianWriter lm_section(layer_info.size() + 16);
        lm_section.write_u32(static_cast<uint32_t>(layer_info.size()));
        lm_section.write_bytes(layer_info.data());
        lm_section.write_u32(0);  // Global layer mask info length = 0
        lm_section.pad_to(4);

        psd.write_u32(static_cast<uint32_t>(lm_section.size()));
        psd.write_bytes(lm_section.data());
    }

    // =========================================================================
    // Section 5: Image Data (Merged Composite, RLE PackBits Compression = 1)
    // =========================================================================
    {
        psd.write_u16(1);  // RLE PackBits

        const size_t ch_count = options.channels;
        const size_t h = static_cast<size_t>(options.height);
        const size_t w = static_cast<size_t>(options.width);

        std::vector<uint8_t> row_buf(w);
        std::vector<std::vector<uint8_t>> all_rows;
        all_rows.reserve(ch_count * h);

        for (size_t c = 0; c < ch_count; ++c) {
            for (size_t y = 0; y < h; ++y) {
                const size_t row_off = y * w * ch_count;
                for (size_t x = 0; x < w; ++x) {
                    row_buf[x] = (*comp_ptr)[row_off + x * ch_count + c];
                }
                auto rle = packbits_encode_row(row_buf.data(), w);
                if (rle.size() > std::numeric_limits<uint16_t>::max()) {
                    throw PsdWriteError("Composite RLE scanline exceeds PSD v1 16-bit row count limit.");
                }
                all_rows.push_back(std::move(rle));
            }
        }

        // All scanline byte counts first (ch_count * height * 2 bytes)
        for (const auto& r : all_rows) {
            psd.write_u16(static_cast<uint16_t>(r.size()));
        }
        // Followed by all compressed scanline data
        for (const auto& r : all_rows) {
            psd.write_bytes(r);
        }
    }

    return psd.take();
}

void write_psd_file(
    const std::string& utf8_path,
    const PsdDocumentOptions& options,
    const std::vector<PsdLayer>& layers,
    const std::vector<uint8_t>& composite_rgb_or_rgba) {
    const auto bytes = write_psd_bytes(options, layers, composite_rgb_or_rgba);
    std::ofstream ofs(utf8_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw PsdWriteError("Failed to open output PSD file for writing: " + utf8_path);
    }
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!ofs.good()) {
        throw PsdWriteError("Failed to write PSD file bytes to disk: " + utf8_path);
    }
}

}  // namespace psd
}  // namespace vellum
