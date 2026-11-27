#pragma once

// Vellum Math Wave M4 — Pure C++17, 0 MB VRAM NLP, Preflight & Auto-QC Ops
//
// Implements:
//   #5  - Aho-Corasick O(L+M) + Normalized Levenshtein (<= 0.25) Active Page Glossary
//         + Okapi BM25 (k1=1.5, b=0.75) Chapter Memory Retrieval
//   #7  - 64x64 Grayscale Shannon Entropy Blank-Page Guard (H < 0.22)
//         + 32x32->8x8 2D DCT 64-bit Perceptual Hash (pHash) + Bounded LRU Cache (<= 2048)
//   #13 - Decoder Logit Shannon Entropy (H_ocr > 1.35 flag)
//         + 3-Feature Mahalanobis Distance Translation Anomaly Auto-QC

#include <nlohmann/json.hpp>

#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mangaflow::pipeline::math_nlp {

// -----------------------------------------------------------------------------
// #5 — Aho-Corasick + Fuzzy Levenshtein Active Glossary + Okapi BM25 Memory
// -----------------------------------------------------------------------------

/// Scan `page_ocr_text` against `full_glossary` using an Aho-Corasick automaton O(L+M)
/// for exact matches plus a normalized UTF-8 Levenshtein fallback (<= max_norm_lev) for
/// 1-char OCR typos on terms >= 3 codepoints. Returns ONLY matching (src, tgt) pairs.
std::vector<std::pair<std::string, std::string>> matchActivePageGlossary(
    const std::string& page_ocr_text,
    const std::map<std::string, std::string>& full_glossary,
    float max_norm_lev = 0.25f);

/// Retrieve the top `top_k` story chunks most relevant to `query_text` via Okapi BM25
/// (k1 = 1.5, b = 0.75) over CJK bigrams + Latin word tokens.
std::vector<std::string> retrieveTopBm25Chunks(
    const std::string& query_text,
    const std::vector<std::string>& chunks,
    size_t top_k = 3,
    float k1 = 1.5f,
    float b = 0.75f);

// -----------------------------------------------------------------------------
// #7 — Shannon Entropy Blank-Page Guard + 64-Bit 2D DCT pHash & LRU Cache
// -----------------------------------------------------------------------------

/// Compute 64x64 downscaled grayscale Shannon entropy H = -sum(p_i * log2(p_i)) in bits.
/// Solid blank / divider pages typically yield H < 0.22.
float computeBlankPageEntropy(
    const uint8_t* rgb,
    int width,
    int height) noexcept;

/// Compute 64-bit perceptual hash via 32x32 grayscale -> 8x8 low-frequency 2D DCT
/// thresholded against the median of the 63 AC coefficients.
uint64_t computeDctPHash64(
    const uint8_t* rgb,
    int width,
    int height) noexcept;

/// Format a 64-bit pHash as a 16-character lowercase hex string.
std::string formatPHashHex(uint64_t hash);

/// Hamming distance between two 64-bit perceptual hashes.
int hammingDistance64(uint64_t a, uint64_t b) noexcept;

struct PHashCacheHit {
    uint64_t matched_hash = 0;
    int hamming_distance = 0;
    nlohmann::json payload;
};

/// Thread-safe bounded LRU cache (max 2048 entries) keyed by (target_lang, profile_slug)
/// supporting near-duplicate perceptual hash lookup (Hamming distance <= max_hamming).
class PHashLruCache {
public:
    explicit PHashLruCache(size_t capacity = 2048) : capacity_(capacity) {}

    std::optional<PHashCacheHit> lookup(
        const std::string& target_lang,
        const std::string& profile_slug,
        uint64_t phash,
        int max_hamming = 4);

    void put(
        const std::string& target_lang,
        const std::string& profile_slug,
        uint64_t phash,
        nlohmann::json payload);

    void clear();

private:
    struct Entry {
        std::string scope_key;
        uint64_t phash = 0;
        nlohmann::json payload;
    };

    size_t capacity_ = 2048;
    std::mutex mutex_;
    std::list<Entry> lru_;
};

// -----------------------------------------------------------------------------
// #13 — Decoder Logit Shannon Entropy + Translation Mahalanobis Distance QC
// -----------------------------------------------------------------------------

/// Compute Shannon entropy H = -sum_v p(v) * log2(p(v)) in bits from raw decoder logits.
float computeLogitEntropyBits(const float* logits, size_t vocab_size) noexcept;

struct TranslationQcPair {
    std::string id;
    std::string source_text;
    std::string translated_text;
};

/// Compute 3-feature Mahalanobis distance D_M(x) = sqrt((x - mu)^T Sigma^-1 (x - mu))
/// over [log((len_trans+1)/(len_src+1)), punct_mismatch, digit_mismatch] and return IDs
/// exceeding `mahalanobis_thresh` (default 3.2).
std::vector<std::string> detectTranslationAnomaliesMahalanobis(
    const std::vector<TranslationQcPair>& pairs,
    float mahalanobis_thresh = 3.2f);

}  // namespace mangaflow::pipeline::math_nlp
