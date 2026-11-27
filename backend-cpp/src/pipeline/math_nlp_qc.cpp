#include "math_nlp_qc.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mangaflow::pipeline::math_nlp {

namespace {

constexpr float kPi = 3.14159265358979323846f;

inline int rgbLumaInt(const uint8_t* px) noexcept {
    return (77 * static_cast<int>(px[0]) +
            150 * static_cast<int>(px[1]) +
            29 * static_cast<int>(px[2])) >> 8;
}

std::vector<uint32_t> decodeUtf8(std::string_view s) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char b0 = static_cast<unsigned char>(s[i]);
        if (b0 < 0x80) {
            cps.push_back(b0);
            i += 1;
        } else if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cps.push_back(((b0 & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu));
            i += 2;
        } else if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cps.push_back(((b0 & 0x0Fu) << 12) |
                          ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                          (static_cast<unsigned char>(s[i + 2]) & 0x3Fu));
            i += 3;
        } else if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cps.push_back(((b0 & 0x07u) << 18) |
                          ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                          ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                          (static_cast<unsigned char>(s[i + 3]) & 0x3Fu));
            i += 4;
        } else {
            i += 1;
        }
    }
    return cps;
}

int levenshteinBounded(
    const uint32_t* a,
    size_t na,
    const uint32_t* b,
    size_t nb,
    int max_dist) {
    if (na == 0) return static_cast<int>(nb);
    if (nb == 0) return static_cast<int>(na);
    if (static_cast<int>(na > nb ? na - nb : nb - na) > max_dist) {
        return max_dist + 1;
    }

    std::vector<int> prev(nb + 1);
    std::vector<int> cur(nb + 1);
    for (size_t j = 0; j <= nb; ++j) prev[j] = static_cast<int>(j);

    for (size_t i = 1; i <= na; ++i) {
        cur[0] = static_cast<int>(i);
        int row_min = cur[0];
        for (size_t j = 1; j <= nb; ++j) {
            const int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            const int v = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
            cur[j] = v;
            if (v < row_min) row_min = v;
        }
        if (row_min > max_dist) {
            return max_dist + 1;
        }
        prev.swap(cur);
    }
    return prev[nb];
}

// Tokenize mixed CJK + Latin text for BM25: CJK character bigrams + lowercased ASCII words.
std::vector<std::string> tokenizeForBm25(std::string_view text) {
    const auto cps = decodeUtf8(text);
    std::vector<std::string> toks;
    toks.reserve(cps.size());

    auto isCjk = [](uint32_t cp) noexcept {
        return (cp >= 0x3040 && cp <= 0x30FF) ||
               (cp >= 0x3400 && cp <= 0x9FFF) ||
               (cp >= 0xAC00 && cp <= 0xD7AF);
    };

    auto encodeCp = [](uint32_t cp, std::string& out) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    };

    std::string word;
    for (size_t i = 0; i < cps.size(); ++i) {
        const uint32_t cp = cps[i];
        if (cp < 128 && std::isalnum(static_cast<unsigned char>(cp))) {
            word.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(cp))));
        } else {
            if (word.size() >= 2) {
                toks.push_back(word);
            }
            word.clear();
            if (isCjk(cp)) {
                std::string uni;
                encodeCp(cp, uni);
                toks.push_back(uni);
                if (i + 1 < cps.size() && isCjk(cps[i + 1])) {
                    std::string bi = uni;
                    encodeCp(cps[i + 1], bi);
                    toks.push_back(std::move(bi));
                }
            }
        }
    }
    if (word.size() >= 2) {
        toks.push_back(std::move(word));
    }
    return toks;
}

int popcount64Portable(uint64_t x) noexcept {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    return static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
}

// Invert a 3x3 symmetric positive-definite matrix; returns false if singular.
bool invert3x3(const double M[3][3], double inv[3][3]) noexcept {
    const double c00 = M[1][1] * M[2][2] - M[1][2] * M[2][1];
    const double c01 = -(M[1][0] * M[2][2] - M[1][2] * M[2][0]);
    const double c02 = M[1][0] * M[2][1] - M[1][1] * M[2][0];
    const double det = M[0][0] * c00 + M[0][1] * c01 + M[0][2] * c02;
    if (std::abs(det) < 1e-12) return false;
    const double id = 1.0 / det;
    inv[0][0] = c00 * id;
    inv[0][1] = -(M[0][1] * M[2][2] - M[0][2] * M[2][1]) * id;
    inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * id;
    inv[1][0] = c01 * id;
    inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * id;
    inv[1][2] = -(M[0][0] * M[1][2] - M[0][2] * M[1][0]) * id;
    inv[2][0] = c02 * id;
    inv[2][1] = -(M[0][0] * M[2][1] - M[0][1] * M[2][0]) * id;
    inv[2][2] = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * id;
    return true;
}

}  // namespace

// =============================================================================
// #5 — Aho-Corasick + Fuzzy Levenshtein Active Page Glossary
// =============================================================================

std::vector<std::pair<std::string, std::string>> matchActivePageGlossary(
    const std::string& page_ocr_text,
    const std::map<std::string, std::string>& full_glossary,
    float max_norm_lev) {
    std::vector<std::pair<std::string, std::string>> matched;
    if (page_ocr_text.empty() || full_glossary.empty()) {
        return matched;
    }

    // Collect valid (src, tgt) entries
    std::vector<std::pair<std::string, std::string>> entries;
    entries.reserve(full_glossary.size());
    for (const auto& [k, v] : full_glossary) {
        if (!k.empty() && !v.empty()) {
            entries.emplace_back(k, v);
        }
    }
    if (entries.empty()) return matched;

    // 1. Build byte-level Aho-Corasick automaton in O(sum |pattern|).
    struct AcNode {
        std::unordered_map<uint8_t, int> next;
        int fail = 0;
        std::vector<int> outputs;
    };
    std::vector<AcNode> trie(1);
    trie.reserve(entries.size() * 8 + 1);

    for (size_t idx = 0; idx < entries.size(); ++idx) {
        int cur = 0;
        for (unsigned char ch : entries[idx].first) {
            auto it = trie[static_cast<size_t>(cur)].next.find(ch);
            if (it == trie[static_cast<size_t>(cur)].next.end()) {
                const int nxt = static_cast<int>(trie.size());
                trie[static_cast<size_t>(cur)].next.emplace(ch, nxt);
                trie.emplace_back();
                cur = nxt;
            } else {
                cur = it->second;
            }
        }
        trie[static_cast<size_t>(cur)].outputs.push_back(static_cast<int>(idx));
    }

    // Build failure links via BFS
    std::queue<int> q;
    for (const auto& [ch, nxt] : trie[0].next) {
        (void)ch;
        trie[static_cast<size_t>(nxt)].fail = 0;
        q.push(nxt);
    }
    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        for (const auto& [ch, v] : trie[static_cast<size_t>(u)].next) {
            int f = trie[static_cast<size_t>(u)].fail;
            while (f != 0 && trie[static_cast<size_t>(f)].next.find(ch) == trie[static_cast<size_t>(f)].next.end()) {
                f = trie[static_cast<size_t>(f)].fail;
            }
            auto fit = trie[static_cast<size_t>(f)].next.find(ch);
            trie[static_cast<size_t>(v)].fail = (fit != trie[static_cast<size_t>(f)].next.end() && fit->second != v)
                ? fit->second
                : 0;
            const int vf = trie[static_cast<size_t>(v)].fail;
            if (!trie[static_cast<size_t>(vf)].outputs.empty()) {
                auto& dst = trie[static_cast<size_t>(v)].outputs;
                const auto& src = trie[static_cast<size_t>(vf)].outputs;
                dst.insert(dst.end(), src.begin(), src.end());
            }
            q.push(v);
        }
    }

    // Scan page_ocr_text in O(L + M)
    std::vector<uint8_t> hit(entries.size(), 0);
    int cur = 0;
    for (unsigned char ch : page_ocr_text) {
        while (cur != 0 && trie[static_cast<size_t>(cur)].next.find(ch) == trie[static_cast<size_t>(cur)].next.end()) {
            cur = trie[static_cast<size_t>(cur)].fail;
        }
        auto it = trie[static_cast<size_t>(cur)].next.find(ch);
        if (it != trie[static_cast<size_t>(cur)].next.end()) {
            cur = it->second;
        }
        for (int out_idx : trie[static_cast<size_t>(cur)].outputs) {
            hit[static_cast<size_t>(out_idx)] = 1;
        }
    }

    // 2. Normalized Levenshtein (<= max_norm_lev) fuzzy rescue for 1-char OCR typos on unmatched keys >= 3 cps.
    const auto text_cps = decodeUtf8(page_ocr_text);
    if (!text_cps.empty() && max_norm_lev > 0.0f) {
        for (size_t idx = 0; idx < entries.size(); ++idx) {
            if (hit[idx]) continue;
            const auto key_cps = decodeUtf8(entries[idx].first);
            const size_t klen = key_cps.size();
            if (klen < 3) continue;
            const int max_d = std::max(1, static_cast<int>(std::floor(static_cast<float>(klen) * max_norm_lev)));
            if (static_cast<float>(max_d) / static_cast<float>(klen) > max_norm_lev + 1e-4f) continue;

            bool fuzzy_found = false;
            const size_t min_win = (klen > 1) ? (klen - 1) : 1;
            const size_t max_win = std::min(text_cps.size(), klen + 1);
            for (size_t pos = 0; pos + min_win <= text_cps.size() && !fuzzy_found; ++pos) {
                // Quick anchor check: at least first or second codepoint must appear in window[0..1]
                const uint32_t k0 = key_cps[0];
                const uint32_t k1 = key_cps[1];
                if (text_cps[pos] != k0 && text_cps[pos] != k1 &&
                    (pos + 1 >= text_cps.size() || (text_cps[pos + 1] != k0 && text_cps[pos + 1] != k1))) {
                    continue;
                }
                for (size_t wlen = min_win; wlen <= max_win && pos + wlen <= text_cps.size(); ++wlen) {
                    const int d = levenshteinBounded(
                        key_cps.data(), klen,
                        text_cps.data() + pos, wlen,
                        max_d);
                    if (d <= max_d) {
                        const float norm_d = static_cast<float>(d) / static_cast<float>(std::max(klen, wlen));
                        if (norm_d <= max_norm_lev) {
                            fuzzy_found = true;
                            break;
                        }
                    }
                }
            }
            if (fuzzy_found) {
                hit[idx] = 1;
            }
        }
    }

    for (size_t idx = 0; idx < entries.size(); ++idx) {
        if (hit[idx]) {
            matched.push_back(entries[idx]);
        }
    }
    return matched;
}

// =============================================================================
// #5 — Okapi BM25 Chapter Memory Retrieval
// =============================================================================

std::vector<std::string> retrieveTopBm25Chunks(
    const std::string& query_text,
    const std::vector<std::string>& chunks,
    size_t top_k,
    float k1,
    float b) {
    if (query_text.empty() || chunks.empty() || top_k == 0) {
        return {};
    }

    const auto q_tokens = tokenizeForBm25(query_text);
    if (q_tokens.empty()) return {};

    const size_t N = chunks.size();
    std::vector<std::unordered_map<std::string, int>> doc_tf(N);
    std::vector<size_t> doc_len(N, 0);
    std::unordered_map<std::string, int> df;

    double total_len = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const auto toks = tokenizeForBm25(chunks[i]);
        doc_len[i] = toks.size();
        total_len += static_cast<double>(toks.size());
        for (const auto& t : toks) {
            if (++doc_tf[i][t] == 1) {
                ++df[t];
            }
        }
    }
    const double avgdl = std::max(1.0, total_len / static_cast<double>(N));

    std::unordered_set<std::string> unique_q(q_tokens.begin(), q_tokens.end());
    std::vector<std::pair<double, size_t>> scored;
    scored.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        double score = 0.0;
        const double dl_norm = 1.0 - static_cast<double>(b) +
                               static_cast<double>(b) * (static_cast<double>(doc_len[i]) / avgdl);
        for (const auto& qt : unique_q) {
            const auto tf_it = doc_tf[i].find(qt);
            if (tf_it == doc_tf[i].end()) continue;
            const double tf = static_cast<double>(tf_it->second);
            const double df_v = static_cast<double>(df[qt]);
            const double idf = std::log((static_cast<double>(N) - df_v + 0.5) / (df_v + 0.5) + 1.0);
            score += idf * (tf * (static_cast<double>(k1) + 1.0)) /
                     (tf + static_cast<double>(k1) * dl_norm);
        }
        if (score > 0.0) {
            scored.emplace_back(score, i);
        }
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        if (std::abs(a.first - b.first) > 1e-6) return a.first > b.first;
        return a.second > b.second;  // Prefer more recent chunk on tie
    });

    const size_t keep = std::min(top_k, scored.size());
    std::vector<std::string> out;
    out.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
        out.push_back(chunks[scored[i].second]);
    }
    return out;
}

// =============================================================================
// #7 — Shannon Entropy Blank-Page Guard + 64-Bit 2D DCT pHash & LRU Cache
// =============================================================================

float computeBlankPageEntropy(
    const uint8_t* rgb,
    int width,
    int height) noexcept {
    if (!rgb || width <= 0 || height <= 0) return 0.0f;

    constexpr int kS = 64;
    std::array<uint32_t, 256> hist{};
    for (int y = 0; y < kS; ++y) {
        const int sy = std::min(height - 1, (y * height) / kS);
        const size_t row = static_cast<size_t>(sy) * static_cast<size_t>(width) * 3ULL;
        for (int x = 0; x < kS; ++x) {
            const int sx = std::min(width - 1, (x * width) / kS);
            const int lum = rgbLumaInt(rgb + row + static_cast<size_t>(sx) * 3ULL);
            ++hist[static_cast<size_t>(lum)];
        }
    }

    constexpr double kInvTotal = 1.0 / static_cast<double>(kS * kS);
    double entropy = 0.0;
    for (uint32_t c : hist) {
        if (c == 0) continue;
        const double p = static_cast<double>(c) * kInvTotal;
        entropy -= p * std::log2(p);
    }
    return static_cast<float>(entropy);
}

uint64_t computeDctPHash64(
    const uint8_t* rgb,
    int width,
    int height) noexcept {
    if (!rgb || width <= 0 || height <= 0) return 0ULL;

    constexpr int kN = 32;
    constexpr int kLow = 8;

    // Precompute 8x32 cosine table for Type-II DCT
    static const auto kCos = []() {
        std::array<std::array<float, 32>, 8> tbl{};
        for (int u = 0; u < 8; ++u) {
            for (int x = 0; x < 32; ++x) {
                tbl[static_cast<size_t>(u)][static_cast<size_t>(x)] =
                    std::cos((kPi * (2.0f * static_cast<float>(x) + 1.0f) * static_cast<float>(u)) /
                             (2.0f * 32.0f));
            }
        }
        return tbl;
    }();

    float img[kN][kN]{};
    for (int y = 0; y < kN; ++y) {
        const int sy = std::min(height - 1, (y * height) / kN);
        const size_t row = static_cast<size_t>(sy) * static_cast<size_t>(width) * 3ULL;
        for (int x = 0; x < kN; ++x) {
            const int sx = std::min(width - 1, (x * width) / kN);
            img[y][x] = static_cast<float>(rgbLumaInt(rgb + row + static_cast<size_t>(sx) * 3ULL));
        }
    }

    // Separable 2D DCT: rows (32->8), then cols (32->8)
    float tmp[kN][kLow]{};
    for (int y = 0; y < kN; ++y) {
        for (int u = 0; u < kLow; ++u) {
            float sum = 0.0f;
            for (int x = 0; x < kN; ++x) {
                sum += img[y][x] * kCos[static_cast<size_t>(u)][static_cast<size_t>(x)];
            }
            tmp[y][u] = sum;
        }
    }

    float dct[kLow * kLow]{};
    std::array<float, kLow * kLow - 1> ac_vals{};
    size_t ac_i = 0;
    for (int v = 0; v < kLow; ++v) {
        for (int u = 0; u < kLow; ++u) {
            float sum = 0.0f;
            for (int y = 0; y < kN; ++y) {
                sum += tmp[y][u] * kCos[static_cast<size_t>(v)][static_cast<size_t>(y)];
            }
            dct[v * kLow + u] = sum;
            if (u != 0 || v != 0) {
                ac_vals[ac_i++] = sum;
            }
        }
    }

    std::nth_element(ac_vals.begin(), ac_vals.begin() + ac_vals.size() / 2, ac_vals.end());
    const float med = ac_vals[ac_vals.size() / 2];

    uint64_t hash = 0ULL;
    for (int k = 0; k < kLow * kLow; ++k) {
        if (dct[k] > med) {
            hash |= (1ULL << static_cast<uint64_t>(k));
        }
    }
    return hash;
}

std::string formatPHashHex(uint64_t hash) {
    char buf[20]{};
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buf);
}

int hammingDistance64(uint64_t a, uint64_t b) noexcept {
    return popcount64Portable(a ^ b);
}

std::optional<PHashCacheHit> PHashLruCache::lookup(
    const std::string& target_lang,
    const std::string& profile_slug,
    uint64_t phash,
    int max_hamming) {
    const std::string scope = target_lang + "|" + profile_slug;
    std::lock_guard<std::mutex> lock(mutex_);

    auto best_it = lru_.end();
    int best_dist = max_hamming + 1;
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        if (it->scope_key != scope) continue;
        const int d = hammingDistance64(it->phash, phash);
        if (d < best_dist) {
            best_dist = d;
            best_it = it;
            if (d == 0) break;
        }
    }
    if (best_it == lru_.end()) {
        return std::nullopt;
    }

    // Touch LRU (move to front)
    lru_.splice(lru_.begin(), lru_, best_it);
    return PHashCacheHit{best_it->phash, best_dist, best_it->payload};
}

void PHashLruCache::put(
    const std::string& target_lang,
    const std::string& profile_slug,
    uint64_t phash,
    nlohmann::json payload) {
    const std::string scope = target_lang + "|" + profile_slug;
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        if (it->scope_key == scope && it->phash == phash) {
            it->payload = std::move(payload);
            lru_.splice(lru_.begin(), lru_, it);
            return;
        }
    }

    lru_.push_front(Entry{scope, phash, std::move(payload)});
    while (lru_.size() > capacity_) {
        lru_.pop_back();
    }
}

void PHashLruCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    lru_.clear();
}

// =============================================================================
// #13 — Decoder Logit Shannon Entropy + Translation Mahalanobis Distance QC
// =============================================================================

float computeLogitEntropyBits(const float* logits, size_t vocab_size) noexcept {
    if (!logits || vocab_size == 0) return 0.0f;

    float max_l = -std::numeric_limits<float>::infinity();
    for (size_t v = 0; v < vocab_size; ++v) {
        if (logits[v] > max_l) max_l = logits[v];
    }
    if (!std::isfinite(max_l)) return 0.0f;

    double sum_exp = 0.0;
    for (size_t v = 0; v < vocab_size; ++v) {
        const float diff = logits[v] - max_l;
        if (diff > -25.0f) {
            sum_exp += std::exp(static_cast<double>(diff));
        }
    }
    if (sum_exp <= 1e-12) return 0.0f;

    const double inv_z = 1.0 / sum_exp;
    double entropy = 0.0;
    for (size_t v = 0; v < vocab_size; ++v) {
        const float diff = logits[v] - max_l;
        if (diff > -25.0f) {
            const double p = std::exp(static_cast<double>(diff)) * inv_z;
            if (p > 1e-12) {
                entropy -= p * std::log2(p);
            }
        }
    }
    return static_cast<float>(entropy);
}

std::vector<std::string> detectTranslationAnomaliesMahalanobis(
    const std::vector<TranslationQcPair>& pairs,
    float mahalanobis_thresh) {
    std::vector<std::string> anomalies;
    if (pairs.empty()) return anomalies;

    auto extractFeatures = [](const std::string& src, const std::string& trans) -> std::array<double, 3> {
        const auto s_cps = decodeUtf8(src);
        const auto t_cps = decodeUtf8(trans);

        const double len_ratio = std::log(static_cast<double>(t_cps.size() + 1) /
                                          static_cast<double>(s_cps.size() + 1));

        auto countPunct = [](const std::vector<uint32_t>& cps) -> std::pair<int, int> {
            int q = 0, ex = 0;
            for (uint32_t cp : cps) {
                if (cp == '?' || cp == 0xFF1F) ++q;
                if (cp == '!' || cp == 0xFF01) ++ex;
            }
            return {q, ex};
        };
        const auto [sq, sex] = countPunct(s_cps);
        const auto [tq, tex] = countPunct(t_cps);
        const double punct_mismatch =
            ((sq > 0) != (tq > 0) ? 1.0 : 0.0) +
            ((sex > 0) != (tex > 0) ? 0.5 : 0.0);

        auto extractDigits = [](const std::vector<uint32_t>& cps) -> std::string {
            std::string d;
            for (uint32_t cp : cps) {
                if (cp >= '0' && cp <= '9') d.push_back(static_cast<char>(cp));
                else if (cp >= 0xFF10 && cp <= 0xFF19) d.push_back(static_cast<char>('0' + (cp - 0xFF10)));
            }
            return d;
        };
        const std::string sd = extractDigits(s_cps);
        const std::string td = extractDigits(t_cps);
        const double digit_mismatch = (!sd.empty() && sd != td) ? 1.0 : 0.0;

        return {len_ratio, punct_mismatch, digit_mismatch};
    };

    std::vector<std::array<double, 3>> feats;
    feats.reserve(pairs.size());
    for (const auto& p : pairs) {
        feats.push_back(extractFeatures(p.source_text, p.translated_text));
    }

    // Robust cross-lingual manga baseline priors (handles small-N pages without degeneracy)
    double mu[3] = {0.45, 0.0, 0.0};
    double cov[3][3] = {
        {0.55 * 0.55, 0.0, 0.0},
        {0.0, 0.45 * 0.45, 0.0},
        {0.0, 0.0, 0.30 * 0.30}
    };

    if (feats.size() >= 6) {
        double emp_mu[3] = {0.0, 0.0, 0.0};
        for (const auto& f : feats) {
            for (int k = 0; k < 3; ++k) emp_mu[k] += f[static_cast<size_t>(k)];
        }
        const double inv_n = 1.0 / static_cast<double>(feats.size());
        for (int k = 0; k < 3; ++k) emp_mu[k] *= inv_n;

        double emp_cov[3][3]{};
        for (const auto& f : feats) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    emp_cov[r][c] += (f[static_cast<size_t>(r)] - emp_mu[r]) *
                                     (f[static_cast<size_t>(c)] - emp_mu[c]);
                }
            }
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                emp_cov[r][c] *= inv_n;
            }
        }

        // Shrinkage blend (50% empirical + 50% prior) for numerical stability
        for (int k = 0; k < 3; ++k) {
            mu[k] = 0.5 * emp_mu[k] + 0.5 * mu[k];
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                cov[r][c] = 0.5 * emp_cov[r][c] + 0.5 * cov[r][c];
            }
        }
    }

    double inv_cov[3][3]{};
    if (!invert3x3(cov, inv_cov)) {
        return anomalies;
    }

    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].source_text.empty()) continue;
        double d[3] = {
            feats[i][0] - mu[0],
            feats[i][1] - mu[1],
            feats[i][2] - mu[2]
        };
        double dist2 = 0.0;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                dist2 += d[r] * inv_cov[r][c] * d[c];
            }
        }
        const float dm = static_cast<float>(std::sqrt(std::max(0.0, dist2)));
        if (dm > mahalanobis_thresh) {
            anomalies.push_back(pairs[i].id);
        }
    }

    return anomalies;
}

}  // namespace mangaflow::pipeline::math_nlp
