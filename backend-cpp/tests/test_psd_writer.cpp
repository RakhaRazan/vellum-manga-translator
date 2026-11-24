#include "../src/psd_writer.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4244 4456 4996)
#endif
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace vellum::psd;

int g_tests_run = 0;
int g_tests_failed = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "[FAIL] " << __func__ << ":" << __LINE__ << " -> "   \
                      << (msg) << "\n";                                       \
            ++g_tests_failed;                                                 \
            return;                                                           \
        }                                                                     \
    } while (0)

#define RUN_TEST(fn)                                                          \
    do {                                                                      \
        const int before = g_tests_failed;                                    \
        ++g_tests_run;                                                        \
        fn();                                                                 \
        if (g_tests_failed == before) {                                       \
            std::cout << "[PASS] " << #fn << "\n";                            \
        }                                                                     \
    } while (0)

class BigEndianReader {
public:
    BigEndianReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    size_t pos() const noexcept { return pos_; }
    size_t size() const noexcept { return size_; }
    size_t remaining() const noexcept { return size_ - pos_; }

    uint8_t read_u8() {
        require(1);
        return data_[pos_++];
    }

    uint16_t read_u16() {
        require(2);
        const uint16_t v = static_cast<uint16_t>(
            (static_cast<uint16_t>(data_[pos_]) << 8) | data_[pos_ + 1]);
        pos_ += 2;
        return v;
    }

    int16_t read_i16() { return static_cast<int16_t>(read_u16()); }

    uint32_t read_u32() {
        require(4);
        const uint32_t v =
            (static_cast<uint32_t>(data_[pos_]) << 24) |
            (static_cast<uint32_t>(data_[pos_ + 1]) << 16) |
            (static_cast<uint32_t>(data_[pos_ + 2]) << 8) |
            static_cast<uint32_t>(data_[pos_ + 3]);
        pos_ += 4;
        return v;
    }

    int32_t read_i32() { return static_cast<int32_t>(read_u32()); }

    double read_f64() {
        require(8);
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) {
            bits = (bits << 8) | data_[pos_ + i];
        }
        pos_ += 8;
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

    std::string read_str(size_t n) {
        require(n);
        std::string s(reinterpret_cast<const char*>(data_ + pos_), n);
        pos_ += n;
        return s;
    }

    const uint8_t* read_ptr(size_t n) {
        require(n);
        const uint8_t* p = data_ + pos_;
        pos_ += n;
        return p;
    }

    void skip(size_t n) {
        require(n);
        pos_ += n;
    }

    void seek(size_t new_pos) {
        if (new_pos > size_) {
            throw std::runtime_error("Seek past end of PSD buffer.");
        }
        pos_ = new_pos;
    }

private:
    void require(size_t n) const {
        if (pos_ + n > size_) {
            throw std::runtime_error("Unexpected EOF while parsing PSD.");
        }
    }

    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

struct ParsedLayerSummary {
    std::string pascal_name;
    int32_t top = 0;
    int32_t left = 0;
    int32_t bottom = 0;
    int32_t right = 0;
    uint16_t channel_count = 0;
    std::vector<std::pair<int16_t, uint32_t>> channels;
    bool has_luni = false;
    bool has_tysh = false;
    bool has_lfx2 = false;
    std::vector<uint8_t> tysh_payload;
    std::vector<uint8_t> lfx2_payload;
    // Decoded channel planes (indexed in same order as `channels`)
    std::vector<std::vector<uint8_t>> decoded_channels;
};

struct ParsedPsdSummary {
    uint16_t version = 0;
    uint16_t channels = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint16_t depth = 0;
    uint16_t color_mode = 0;
    uint32_t color_mode_len = 0;
    uint32_t resources_len = 0;
    uint32_t layer_mask_len = 0;
    uint32_t layer_info_len = 0;
    int16_t signed_layer_count = 0;
    std::vector<ParsedLayerSummary> layers;
    std::vector<std::vector<uint8_t>> composite_channels;
    bool exact_eof = false;
};

ParsedPsdSummary parse_and_validate_psd(const std::vector<uint8_t>& psd) {
    ParsedPsdSummary out;
    BigEndianReader r(psd.data(), psd.size());

    // 1. File Header (26 bytes)
    if (r.read_str(4) != "8BPS") {
        throw std::runtime_error("Invalid PSD signature (expected 8BPS).");
    }
    out.version = r.read_u16();
    if (out.version != 1) {
        throw std::runtime_error("Expected PSD version 1.");
    }
    for (int i = 0; i < 6; ++i) {
        if (r.read_u8() != 0) {
            throw std::runtime_error("Reserved header bytes must be zero.");
        }
    }
    out.channels = r.read_u16();
    out.height = r.read_u32();
    out.width = r.read_u32();
    out.depth = r.read_u16();
    out.color_mode = r.read_u16();

    // 2. Color Mode Data
    out.color_mode_len = r.read_u32();
    r.skip(out.color_mode_len);

    // 3. Image Resources
    out.resources_len = r.read_u32();
    if ((out.resources_len % 2U) != 0) {
        throw std::runtime_error("Image Resources section length must be even.");
    }
    const size_t res_end = r.pos() + out.resources_len;
    while (r.pos() < res_end) {
        if (r.read_str(4) != "8BIM") {
            throw std::runtime_error("Invalid Image Resource signature (expected 8BIM).");
        }
        (void)r.read_u16();  // resource ID
        const uint8_t name_len = r.read_u8();
        r.skip(name_len);
        if (((1U + name_len) % 2U) != 0) {
            r.skip(1);
        }
        const uint32_t data_len = r.read_u32();
        r.skip(data_len + (data_len & 1U));
    }
    if (r.pos() != res_end) {
        throw std::runtime_error("Image Resources section overran declared length.");
    }

    // 4. Layer & Mask Information
    out.layer_mask_len = r.read_u32();
    const size_t lm_end = r.pos() + out.layer_mask_len;
    if (out.layer_mask_len > 0) {
        if ((out.layer_mask_len % 2U) != 0) {
            throw std::runtime_error("Layer & Mask Info section length must be even.");
        }
        out.layer_info_len = r.read_u32();
        if ((out.layer_info_len % 2U) != 0) {
            throw std::runtime_error("Layer Info length must be even.");
        }
        const size_t li_end = r.pos() + out.layer_info_len;

        out.signed_layer_count = r.read_i16();
        const int layer_count = std::abs(static_cast<int>(out.signed_layer_count));
        out.layers.resize(static_cast<size_t>(layer_count));

        for (int i = 0; i < layer_count; ++i) {
            auto& l = out.layers[static_cast<size_t>(i)];
            l.top = r.read_i32();
            l.left = r.read_i32();
            l.bottom = r.read_i32();
            l.right = r.read_i32();
            l.channel_count = r.read_u16();
            for (uint16_t c = 0; c < l.channel_count; ++c) {
                const int16_t ch_id = r.read_i16();
                const uint32_t ch_len = r.read_u32();
                l.channels.emplace_back(ch_id, ch_len);
            }
            if (r.read_str(4) != "8BIM") {
                throw std::runtime_error("Layer blend mode signature must be 8BIM.");
            }
            (void)r.read_str(4);  // blend mode key
            (void)r.read_u8();    // opacity
            (void)r.read_u8();    // clipping
            (void)r.read_u8();    // flags
            if (r.read_u8() != 0) {
                throw std::runtime_error("Layer filler byte must be 0.");
            }

            const uint32_t extra_len = r.read_u32();
            const size_t extra_end = r.pos() + extra_len;

            const uint32_t mask_len = r.read_u32();
            r.skip(mask_len);
            const uint32_t blend_ranges_len = r.read_u32();
            r.skip(blend_ranges_len);

            // Pascal string padded to 4 bytes
            const uint8_t pname_len = r.read_u8();
            l.pascal_name = r.read_str(pname_len);
            const size_t pname_total = 1U + pname_len;
            const size_t pname_pad = (4U - (pname_total % 4U)) % 4U;
            r.skip(pname_pad);

            // Additional Layer Information (4-byte aligned tagged blocks)
            while (r.pos() < extra_end) {
                const std::string sig = r.read_str(4);
                if (sig != "8BIM" && sig != "8B64") {
                    throw std::runtime_error("Invalid tagged block signature: " + sig);
                }
                const std::string key = r.read_str(4);
                const uint32_t blk_len = r.read_u32();
                if ((blk_len % 4U) != 0) {
                    throw std::runtime_error("Tagged block '" + key + "' length is not 4-byte aligned.");
                }
                const uint8_t* blk_ptr = r.read_ptr(blk_len);
                if (key == "luni") {
                    l.has_luni = true;
                } else if (key == "TySh") {
                    l.has_tysh = true;
                    l.tysh_payload.assign(blk_ptr, blk_ptr + blk_len);
                } else if (key == "lfx2") {
                    l.has_lfx2 = true;
                    l.lfx2_payload.assign(blk_ptr, blk_ptr + blk_len);
                }
            }
            if (r.pos() != extra_end) {
                throw std::runtime_error("Layer extra data length mismatch.");
            }
        }

        // Decode Layer Channel Image Data
        for (auto& l : out.layers) {
            const size_t lw = static_cast<size_t>(std::max<int32_t>(0, l.right - l.left));
            const size_t lh = static_cast<size_t>(std::max<int32_t>(0, l.bottom - l.top));
            for (const auto& [ch_id, ch_len] : l.channels) {
                (void)ch_id;
                const size_t ch_start = r.pos();
                const uint16_t comp = r.read_u16();
                if (comp != 1) {
                    throw std::runtime_error("Expected RLE PackBits compression (1) on layer channel.");
                }
                std::vector<uint16_t> row_counts(lh);
                for (size_t y = 0; y < lh; ++y) {
                    row_counts[y] = r.read_u16();
                }
                std::vector<uint8_t> plane;
                plane.reserve(lw * lh);
                for (size_t y = 0; y < lh; ++y) {
                    const uint8_t* rle_ptr = r.read_ptr(row_counts[y]);
                    const auto decoded = packbits_decode_row(rle_ptr, row_counts[y], lw);
                    plane.insert(plane.end(), decoded.begin(), decoded.end());
                }
                if (r.pos() - ch_start != ch_len) {
                    throw std::runtime_error("Layer channel byte length mismatch.");
                }
                l.decoded_channels.push_back(std::move(plane));
            }
        }

        // Skip Layer Info padding up to li_end
        if (r.pos() > li_end) {
            throw std::runtime_error("Layer Info overran declared length.");
        }
        r.seek(li_end);

        // Global Layer Mask Info
        const uint32_t global_mask_len = r.read_u32();
        r.skip(global_mask_len);

        if (r.pos() > lm_end) {
            throw std::runtime_error("Layer & Mask Info overran declared length.");
        }
        r.seek(lm_end);
    }

    // 5. Merged Composite Image Data
    const uint16_t comp = r.read_u16();
    if (comp != 1) {
        throw std::runtime_error("Expected RLE PackBits compression (1) on composite image data.");
    }
    const size_t total_rows = static_cast<size_t>(out.channels) * static_cast<size_t>(out.height);
    std::vector<uint16_t> row_counts(total_rows);
    for (size_t i = 0; i < total_rows; ++i) {
        row_counts[i] = r.read_u16();
    }
    out.composite_channels.resize(out.channels);
    for (uint16_t c = 0; c < out.channels; ++c) {
        auto& plane = out.composite_channels[c];
        plane.reserve(static_cast<size_t>(out.width) * static_cast<size_t>(out.height));
        for (uint32_t y = 0; y < out.height; ++y) {
            const uint16_t rlen = row_counts[static_cast<size_t>(c) * out.height + y];
            const uint8_t* rle_ptr = r.read_ptr(rlen);
            const auto decoded = packbits_decode_row(rle_ptr, rlen, out.width);
            plane.insert(plane.end(), decoded.begin(), decoded.end());
        }
    }

    out.exact_eof = (r.remaining() == 0);
    return out;
}

// -----------------------------------------------------------------------------
// Unit Tests
// -----------------------------------------------------------------------------

void test_packbits_roundtrip() {
    std::vector<std::vector<uint8_t>> cases;
    cases.push_back({});
    cases.push_back({42});
    cases.push_back({7, 7});
    cases.push_back({7, 7, 7});
    cases.push_back(std::vector<uint8_t>(128, 0xAA));
    cases.push_back(std::vector<uint8_t>(500, 0x55));

    // Alternating bytes (worst case literal expansion)
    {
        std::vector<uint8_t> alt(300);
        for (size_t i = 0; i < alt.size(); ++i) alt[i] = static_cast<uint8_t>(i & 1 ? 0x10 : 0xEF);
        cases.push_back(std::move(alt));
    }

    // Mixed runs and literals
    {
        std::vector<uint8_t> mix;
        for (int i = 0; i < 10; ++i) mix.push_back(static_cast<uint8_t>(i));
        mix.insert(mix.end(), 17, 0xCC);
        mix.push_back(0x01);
        mix.push_back(0x02);
        mix.insert(mix.end(), 130, 0xFF);
        mix.push_back(0x99);
        cases.push_back(std::move(mix));
    }

    // Deterministic pseudo-random 2048 bytes
    {
        std::vector<uint8_t> rnd(2048);
        uint32_t state = 0x12345678U;
        for (auto& b : rnd) {
            state = state * 1664525U + 1013904223U;
            b = static_cast<uint8_t>((state >> 24) & 0xFFU);
        }
        cases.push_back(std::move(rnd));
    }

    for (size_t idx = 0; idx < cases.size(); ++idx) {
        const auto& orig = cases[idx];
        const auto encoded = packbits_encode_row(orig.data(), orig.size());
        const auto decoded = packbits_decode_row(encoded.data(), encoded.size(), orig.size());
        CHECK(decoded == orig, "PackBits round-trip mismatch on case #" + std::to_string(idx));
    }
}

void test_stage1_layered_raster_psd() {
    const int32_t W = 64;
    const int32_t H = 48;

    // Raw original Background (opaque RGB gradient)
    std::vector<uint8_t> orig_rgba(static_cast<size_t>(W * H * 4));
    for (int32_t y = 0; y < H; ++y) {
        for (int32_t x = 0; x < W; ++x) {
            const size_t off = static_cast<size_t>((y * W + x) * 4);
            orig_rgba[off + 0] = static_cast<uint8_t>(x * 4);
            orig_rgba[off + 1] = static_cast<uint8_t>(y * 5);
            orig_rgba[off + 2] = 128;
            orig_rgba[off + 3] = 255;
        }
    }

    // Cleaned image: inpaint a 16x12 patch at (10, 8) with solid white
    std::vector<uint8_t> cleaned_rgba = orig_rgba;
    for (int32_t y = 8; y < 20; ++y) {
        for (int32_t x = 10; x < 26; ++x) {
            const size_t off = static_cast<size_t>((y * W + x) * 4);
            cleaned_rgba[off + 0] = 255;
            cleaned_rgba[off + 1] = 255;
            cleaned_rgba[off + 2] = 255;
            cleaned_rgba[off + 3] = 255;
        }
    }

    // Layer 0: Background (opaque RGB)
    PsdLayer bg;
    bg.name = "Background";
    bg.left = 0;
    bg.top = 0;
    bg.width = W;
    bg.height = H;
    bg.has_alpha = false;
    bg.pixels.resize(static_cast<size_t>(W * H * 3));
    for (int32_t i = 0; i < W * H; ++i) {
        bg.pixels[static_cast<size_t>(i * 3 + 0)] = orig_rgba[static_cast<size_t>(i * 4 + 0)];
        bg.pixels[static_cast<size_t>(i * 3 + 1)] = orig_rgba[static_cast<size_t>(i * 4 + 1)];
        bg.pixels[static_cast<size_t>(i * 3 + 2)] = orig_rgba[static_cast<size_t>(i * 4 + 2)];
    }

    // Layer 1: Cleanup (non-destructive diff patch)
    PsdLayer cleanup = build_cleanup_diff_layer(orig_rgba.data(), cleaned_rgba.data(), W, H);
    CHECK(cleanup.left == 10 && cleanup.top == 8, "Cleanup bbox offset mismatch");
    CHECK(cleanup.width == 16 && cleanup.height == 12, "Cleanup bbox size mismatch");

    // Layer 2: Text (offset raster block at (12, 10), size 10x6, black text)
    PsdLayer text;
    text.name = "Text";
    text.left = 12;
    text.top = 10;
    text.width = 10;
    text.height = 6;
    text.has_alpha = true;
    text.pixels.assign(static_cast<size_t>(10 * 6 * 4), 0);
    for (int i = 0; i < 10 * 6; ++i) {
        text.pixels[static_cast<size_t>(i * 4 + 0)] = 20;
        text.pixels[static_cast<size_t>(i * 4 + 1)] = 30;
        text.pixels[static_cast<size_t>(i * 4 + 2)] = 40;
        text.pixels[static_cast<size_t>(i * 4 + 3)] = 255;
    }

    PsdDocumentOptions opts;
    opts.width = W;
    opts.height = H;
    opts.channels = 3;
    opts.dpi = 72.0;
    opts.editable_text_layers = false;

    const auto psd_bytes = write_psd_bytes(opts, {bg, cleanup, text});
    const auto parsed = parse_and_validate_psd(psd_bytes);

    CHECK(parsed.exact_eof, "PSD parser did not reach exact EOF");
    CHECK(parsed.version == 1 && parsed.channels == 3 && parsed.depth == 8 && parsed.color_mode == 3,
          "PSD header fields mismatch");
    CHECK(parsed.width == static_cast<uint32_t>(W) && parsed.height == static_cast<uint32_t>(H),
          "PSD dimensions mismatch");
    CHECK(parsed.signed_layer_count == 3, "Expected +3 layers for 3-channel RGB PSD");
    CHECK(parsed.layers.size() == 3, "Layer count mismatch");

    CHECK(parsed.layers[0].pascal_name == "Background", "Layer 0 name mismatch");
    CHECK(parsed.layers[1].pascal_name == "Cleanup", "Layer 1 name mismatch");
    CHECK(parsed.layers[2].pascal_name == "Text", "Layer 2 name mismatch");
    CHECK(parsed.layers[2].left == 12 && parsed.layers[2].top == 10 &&
              parsed.layers[2].right == 22 && parsed.layers[2].bottom == 16,
          "Text layer exact (x, y) bounds mismatch");

    // Verify Stage 2 tags are NOT emitted when editable_text_layers == false
    CHECK(!parsed.layers[2].has_tysh && !parsed.layers[2].has_lfx2,
          "TySh/lfx2 must not be emitted when editable_text_layers == false");

    // Verify merged composite pixel at (14, 12) matches Text layer (20, 30, 40)
    const size_t px_text = static_cast<size_t>(12 * W + 14);
    CHECK(parsed.composite_channels[0][px_text] == 20 &&
              parsed.composite_channels[1][px_text] == 30 &&
              parsed.composite_channels[2][px_text] == 40,
          "Composite pixel over Text layer mismatch");

    // Verify merged composite pixel at (10, 8) matches Cleanup patch (255, 255, 255)
    const size_t px_clean = static_cast<size_t>(8 * W + 10);
    CHECK(parsed.composite_channels[0][px_clean] == 255 &&
              parsed.composite_channels[1][px_clean] == 255 &&
              parsed.composite_channels[2][px_clean] == 255,
          "Composite pixel over Cleanup layer mismatch");

    // Verify merged composite pixel at (2, 2) matches raw Background
    const size_t px_bg = static_cast<size_t>(2 * W + 2);
    CHECK(parsed.composite_channels[0][px_bg] == orig_rgba[px_bg * 4 + 0] &&
              parsed.composite_channels[1][px_bg] == orig_rgba[px_bg * 4 + 1] &&
              parsed.composite_channels[2][px_bg] == orig_rgba[px_bg * 4 + 2],
          "Composite pixel over Background layer mismatch");
}

void test_stage1_rgba_4channel_mode() {
    PsdLayer l;
    l.name = "AlphaLayer";
    l.left = 0;
    l.top = 0;
    l.width = 8;
    l.height = 8;
    l.has_alpha = true;
    l.pixels.assign(8 * 8 * 4, 128);

    PsdDocumentOptions opts;
    opts.width = 8;
    opts.height = 8;
    opts.channels = 4;

    const auto psd_bytes = write_psd_bytes(opts, {l});
    const auto parsed = parse_and_validate_psd(psd_bytes);
    CHECK(parsed.exact_eof, "4-channel PSD did not reach exact EOF");
    CHECK(parsed.channels == 4, "Expected 4 channels");
    CHECK(parsed.signed_layer_count == -1, "Expected negative layer count (-1) for 4-channel RGBA PSD");
    CHECK(parsed.composite_channels.size() == 4, "Expected 4 composite channel planes");
    CHECK(parsed.composite_channels[3][0] == 128, "Composite alpha channel mismatch");
}

void test_stage2_tysh_enginedata_and_lfx2() {
    const int32_t W = 200;
    const int32_t H = 120;

    PsdLayer bg;
    bg.name = "Background";
    bg.width = W;
    bg.height = H;
    bg.has_alpha = false;
    bg.pixels.assign(static_cast<size_t>(W * H * 3), 240);

    PsdLayer text;
    text.name = "Dialogue 01";
    text.left = 30;
    text.top = 24;
    text.width = 120;
    text.height = 48;
    text.has_alpha = true;
    text.pixels.assign(static_cast<size_t>(120 * 48 * 4), 255);

    TypeLayerSpec spec;
    spec.text = "Hello\nWorld\nテスト";
    spec.font_name = "AnimeAce";
    spec.font_size = 22.0;
    spec.fill_color = {12, 34, 56};
    spec.align = TextAlign::Center;
    spec.faux_bold = true;
    spec.auto_hyphenate = false;
    spec.box_x = 30.0;
    spec.box_y = 24.0;
    spec.box_width = 120.0;
    spec.box_height = 48.0;
    spec.outline.enabled = true;
    spec.outline.width_px = 3.0;
    spec.outline.color = {255, 255, 255};
    spec.outline.opacity_pct = 100.0;
    text.type_spec = spec;

    PsdDocumentOptions opts;
    opts.width = W;
    opts.height = H;
    opts.channels = 3;
    opts.dpi = 72.0;
    opts.editable_text_layers = true;

    const auto psd_bytes = write_psd_bytes(opts, {bg, text});
    const auto parsed = parse_and_validate_psd(psd_bytes);

    CHECK(parsed.exact_eof, "Stage 2 PSD did not reach exact EOF");
    CHECK(parsed.layers.size() == 2, "Expected 2 layers");
    CHECK(parsed.layers[1].has_luni, "Missing luni tagged block");
    CHECK(parsed.layers[1].has_tysh, "Missing TySh tagged block");
    CHECK(parsed.layers[1].has_lfx2, "Missing lfx2 tagged block");

    // Parse TySh payload structure
    const auto& tysh = parsed.layers[1].tysh_payload;
    CHECK((tysh.size() % 4U) == 0, "TySh payload must be 4-byte aligned");
    BigEndianReader tr(tysh.data(), tysh.size());
    CHECK(tr.read_i16() == 1, "TySh version must be 1");
    const double xx = tr.read_f64();
    const double xy = tr.read_f64();
    const double yx = tr.read_f64();
    const double yy = tr.read_f64();
    const double tx = tr.read_f64();
    const double ty = tr.read_f64();
    CHECK(std::fabs(xx - 1.0) < 1e-9 && std::fabs(xy) < 1e-9 &&
              std::fabs(yx) < 1e-9 && std::fabs(yy - 1.0) < 1e-9,
          "Identity rotation matrix mismatch");
    CHECK(std::fabs(tx - 30.0) < 1e-9 && std::fabs(ty - 24.0) < 1e-9,
          "Transform translation mismatch");
    CHECK(tr.read_i16() == 50, "TySh textVersion must be 50");
    CHECK(tr.read_i32() == 16, "TySh descriptorVersion must be 16");
    CHECK(tr.read_u32() == 0, "Expected empty class name in TxLr descriptor");
    CHECK(tr.read_u32() == 0 && tr.read_str(4) == "TxLr", "Expected TxLr class ID");
    const uint32_t item_count = tr.read_u32();
    CHECK(item_count == 5, "Expected 5 items in TxLr descriptor");

    // Item 1: `Txt ` (TEXT)
    CHECK(tr.read_u32() == 0 && tr.read_str(4) == "Txt ", "Expected Txt  key");
    CHECK(tr.read_str(4) == "TEXT", "Expected TEXT type");
    const uint32_t char_count = tr.read_u32();
    // "Hello\rWorld\rテスト\r" = 16 UTF-16 units + 1 null terminator = 17
    CHECK(char_count == 17, "Expected 17 UTF-16 code units (including \\r and NUL)");
    std::vector<uint16_t> utf16(char_count);
    for (uint32_t i = 0; i < char_count; ++i) {
        utf16[i] = tr.read_u16();
    }
    CHECK(utf16[5] == 0x000D, "Expected \\r after Hello");
    CHECK(utf16[11] == 0x000D, "Expected \\r after World");
    CHECK(utf16[12] == 0x30C6 && utf16[13] == 0x30B9 && utf16[14] == 0x30C8,
          "Expected UTF-16BE Katakana 'テスト'");
    CHECK(utf16[15] == 0x000D, "Expected trailing \\r");
    CHECK(utf16[16] == 0x0000, "Expected trailing NUL terminator");

    // Verify EngineData substring contents inside TySh payload
    const std::string tysh_str(reinterpret_cast<const char*>(tysh.data()), tysh.size());
    CHECK(tysh_str.find("EngineData") != std::string::npos, "Missing EngineData key");
    CHECK(tysh_str.find("/StyleRun") != std::string::npos, "Missing /StyleRun in EngineData");
    CHECK(tysh_str.find("/ParagraphRun") != std::string::npos, "Missing /ParagraphRun in EngineData");
    CHECK(tysh_str.find("/FontSize 22.0") != std::string::npos, "Missing /FontSize in EngineData");
    CHECK(tysh_str.find("/FillColor") != std::string::npos, "Missing /FillColor in EngineData");
    CHECK(tysh_str.find("/AutoHyphenate false") != std::string::npos, "Missing /AutoHyphenate in EngineData");
    CHECK(tysh_str.find("/FontSet") != std::string::npos, "Missing /FontSet in EngineData");

    // Parse lfx2 payload structure
    const auto& lfx2 = parsed.layers[1].lfx2_payload;
    CHECK((lfx2.size() % 4U) == 0, "lfx2 payload must be 4-byte aligned");
    BigEndianReader lr(lfx2.data(), lfx2.size());
    CHECK(lr.read_i32() == 0, "lfx2 objectEffectsVersion must be 0");
    CHECK(lr.read_i32() == 16, "lfx2 descriptorVersion must be 16");
    CHECK(lr.read_u32() == 0, "Expected empty class name in Lefx");
    CHECK(lr.read_u32() == 0 && lr.read_str(4) == "Lefx", "Expected Lefx class ID");
    const std::string lfx2_str(reinterpret_cast<const char*>(lfx2.data()), lfx2.size());
    CHECK(lfx2_str.find("FrFX") != std::string::npos, "Missing FrFX stroke descriptor in lfx2");
    CHECK(lfx2_str.find("OutF") != std::string::npos, "Missing OutF (Outside stroke) in lfx2");
    CHECK(lfx2_str.find("RGBC") != std::string::npos, "Missing RGBC stroke color in lfx2");
}

int export_sample_psd_and_verify(const std::string& input_img, const std::string& output_psd) {
    int w = 0, h = 0, ch = 0;
    unsigned char* raw = stbi_load(input_img.c_str(), &w, &h, &ch, 4);
    if (!raw || w <= 0 || h <= 0) {
        std::cerr << "[ERROR] Failed to decode sample image: " << input_img << "\n";
        return 1;
    }

    std::vector<uint8_t> orig_rgba(raw, raw + static_cast<size_t>(w) * static_cast<size_t>(h) * 4U);
    stbi_image_free(raw);

    // Build Background (RGB)
    PsdLayer bg;
    bg.name = "Background";
    bg.width = w;
    bg.height = h;
    bg.has_alpha = false;
    bg.pixels.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3U);
    for (size_t i = 0; i < static_cast<size_t>(w) * static_cast<size_t>(h); ++i) {
        bg.pixels[i * 3 + 0] = orig_rgba[i * 4 + 0];
        bg.pixels[i * 3 + 1] = orig_rgba[i * 4 + 1];
        bg.pixels[i * 3 + 2] = orig_rgba[i * 4 + 2];
    }

    // Build synthetic Cleanup patch (white speech-bubble fill in upper-right quadrant)
    std::vector<uint8_t> cleaned_rgba = orig_rgba;
    const int32_t cx0 = w / 2;
    const int32_t cy0 = h / 8;
    const int32_t cx1 = std::min(w, cx0 + std::max(32, w / 5));
    const int32_t cy1 = std::min(h, cy0 + std::max(32, h / 10));
    for (int32_t y = cy0; y < cy1; ++y) {
        for (int32_t x = cx0; x < cx1; ++x) {
            const size_t off = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4U;
            cleaned_rgba[off + 0] = 255;
            cleaned_rgba[off + 1] = 255;
            cleaned_rgba[off + 2] = 255;
            cleaned_rgba[off + 3] = 255;
        }
    }
    PsdLayer cleanup = build_cleanup_diff_layer(orig_rgba.data(), cleaned_rgba.data(), w, h);

    // Build Text layer with both raster pixels and Stage 2 TySh + lfx2 metadata
    PsdLayer text;
    text.name = "Text 01";
    text.left = cx0 + 8;
    text.top = cy0 + 8;
    text.width = std::max(16, (cx1 - cx0) - 16);
    text.height = std::max(16, (cy1 - cy0) - 16);
    text.has_alpha = true;
    text.pixels.assign(static_cast<size_t>(text.width) * static_cast<size_t>(text.height) * 4U, 0);
    // Draw a crisp dark horizontal bar as placeholder raster text glyph
    for (int32_t y = text.height / 3; y < (text.height * 2) / 3; ++y) {
        for (int32_t x = 4; x < text.width - 4; ++x) {
            const size_t off = (static_cast<size_t>(y) * static_cast<size_t>(text.width) + static_cast<size_t>(x)) * 4U;
            text.pixels[off + 0] = 16;
            text.pixels[off + 1] = 16;
            text.pixels[off + 2] = 16;
            text.pixels[off + 3] = 255;
        }
    }
    TypeLayerSpec tspec;
    tspec.text = "Vellum Layered PSD Sample\nTest Dialogue";
    tspec.font_name = "ArialMT";
    tspec.font_size = 20.0;
    tspec.fill_color = {0, 0, 0};
    tspec.align = TextAlign::Center;
    tspec.box_x = static_cast<double>(text.left);
    tspec.box_y = static_cast<double>(text.top);
    tspec.box_width = static_cast<double>(text.width);
    tspec.box_height = static_cast<double>(text.height);
    tspec.outline.enabled = true;
    tspec.outline.width_px = 2.0;
    tspec.outline.color = {255, 255, 255};
    text.type_spec = tspec;

    PsdDocumentOptions opts;
    opts.width = w;
    opts.height = h;
    opts.channels = 3;
    opts.dpi = 72.0;
    opts.editable_text_layers = true;

    const auto psd_bytes = write_psd_bytes(opts, {bg, cleanup, text});
    const auto parsed = parse_and_validate_psd(psd_bytes);
    if (!parsed.exact_eof || parsed.layers.size() != 3 || !parsed.layers[2].has_tysh || !parsed.layers[2].has_lfx2) {
        std::cerr << "[ERROR] Sample PSD failed structural byte validation.\n";
        return 1;
    }

    std::ofstream ofs(output_psd, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to write sample PSD: " << output_psd << "\n";
        return 1;
    }
    ofs.write(reinterpret_cast<const char*>(psd_bytes.data()), static_cast<std::streamsize>(psd_bytes.size()));
    std::cout << "[OK] Exported & byte-verified sample PSD: " << output_psd
              << " (" << w << "x" << h << ", " << psd_bytes.size() << " bytes, "
              << parsed.layers.size() << " layers, TySh+lfx2 verified)\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--export-sample") {
            return export_sample_psd_and_verify(argv[2], argv[3]);
        }

        RUN_TEST(test_packbits_roundtrip);
        RUN_TEST(test_stage1_layered_raster_psd);
        RUN_TEST(test_stage1_rgba_4channel_mode);
        RUN_TEST(test_stage2_tysh_enginedata_and_lfx2);

        std::cout << "Summary: " << (g_tests_run - g_tests_failed) << "/" << g_tests_run
                  << " tests passed.\n";
        return g_tests_failed == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] Unhandled exception: " << ex.what() << "\n";
        return 1;
    }
}
