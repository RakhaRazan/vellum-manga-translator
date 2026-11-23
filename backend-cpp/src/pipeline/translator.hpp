#pragma once

// Vellum - Native C++17 Multi-Provider Context-Aware Comic Translator (Phase 10.3)
//
// Koharu-style multi-provider LLM translation engine using:
//   - `cpp-httplib` for local/plain HTTP endpoints (`lmstudio`, `ollama`, HTTP `custom`)
//   - Windows WinHTTP (`winhttp.lib`) for TLS/HTTPS cloud endpoints (`openai`, `gemini`,
//     `anthropic`, `deepseek`, `openrouter`, HTTPS `custom`)
//   - `nlohmann::json` for Structured JSON Schema payloads and resilient output extraction
//   - `stb_image` / `stb_image_write` for optional multimodal Vision JPEG encoding
//
// Features:
//   1. 8 Koharu-style providers: `lmstudio`, `ollama`, `openai`, `gemini`, `anthropic`,
//      `deepseek`, `openrouter`, `custom`.
//   2. Live provider connectivity testing (`POST /api/providers/test`) and LM Studio
//      status probing (`GET /api/lmstudio/status`) with fuzzy model matching & `/api/v0/models`.
//   3. Advanced controls: `temperature`, `top_p`, `top_k`, `max_tokens`, `frequency_penalty`,
//      `reasoning_effort` (`off`/`low`/`medium`/`high`) with automatic O(N) C++ `<think>...</think>`
//      stripping, `use_vision` multimodal page image attachment with text-only fallback,
//      custom Instruction Bar, Glossary injection & replacement rules, and 16 target languages.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mangaflow::pipeline {

constexpr const char* kDefaultLmStudioBaseUrl = "http://localhost:1234/v1";
constexpr const char* kDefaultLmStudioApiKey = "";
constexpr const char* kDefaultLmStudioModel = "";
constexpr const char* kDefaultOllamaBaseUrl = "http://localhost:11434/v1";
constexpr const char* kDefaultAnthropicVersion = "2023-06-01";

class ProviderConnectionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ProviderTranslationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ProviderSpec {
    std::string id;
    std::string name;
    std::string default_base_url;
    std::string default_model;
    std::string default_api_key;
    bool requires_api_key = false;
    bool requires_base_url = true;
    bool is_local = false;
    std::string protocol = "openai";  // "openai" | "anthropic"
    bool supports_json_schema = true;
    bool supports_vision = true;
    bool supports_top_k = true;
    bool supports_frequency_penalty = true;
    bool supports_reasoning_effort = true;
    std::vector<std::pair<std::string, std::string>> extra_headers;
};

struct TargetLanguageSpec {
    std::string code;
    std::string name;
    std::string native_name;
    bool rtl = false;
    std::string style_rule;
    std::vector<std::string> aliases;
};

struct DialogueBlockInput {
    std::string id;
    std::string source_text;
};

struct TranslatedDialogueItem {
    std::string id;
    std::string source_text;
    std::string translated_text;

    nlohmann::json to_json() const;
};

struct TranslatorConfig {
    std::string provider = "lmstudio";
    std::string base_url;
    std::string api_key;
    std::string preferred_model;
    double temperature = 0.2;
    std::optional<double> top_p = 0.95;
    std::optional<int> top_k = 40;
    int max_tokens = 1024;
    std::optional<double> frequency_penalty = 0.0;
    std::string reasoning_effort = "off";
    std::string translation_tone = "standard";  // "standard" | "casual" | "adult"
    bool strip_thinking = true;
    bool use_vision = false;
    int vision_max_side = 1024;
    std::string custom_instructions;
    std::map<std::string, std::string> glossary;
    double timeout_seconds = 90.0;

    static TranslatorConfig from_settings(const nlohmann::json& settings);
    static TranslatorConfig from_settings_with_overrides(
        const nlohmann::json& settings,
        const nlohmann::json& overrides);
};

struct TranslateRequestOptions {
    std::string source_lang = "ja";
    std::string target_lang = "id";
    std::optional<std::string> translation_tone;
    std::optional<std::string> custom_instructions;
    std::optional<std::map<std::string, std::string>> glossary;
    std::optional<std::string> page_image_data_uri;
    std::optional<bool> use_vision;
    std::optional<int> vision_max_side;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::optional<int> max_tokens;
    std::optional<double> frequency_penalty;
    std::optional<std::string> reasoning_effort;
};

struct PageTranslationResult {
    std::string provider;
    std::string model;
    std::string source_language;
    std::string target_language;
    bool vision_used = false;
    std::string reasoning_effort = "off";
    std::map<std::string, std::string> translations;
    std::vector<TranslatedDialogueItem> dialogues;

    nlohmann::json to_json() const;
};

struct ProviderTestResult {
    bool ok = false;
    bool connected = false;
    std::string provider;
    std::string provider_name;
    std::string endpoint;
    std::optional<std::string> active_model;
    std::optional<std::string> resolved_model_id;
    std::string preferred_model;
    std::vector<std::string> models;
    std::vector<std::string> vision_models;
    int latency_ms = 0;
    std::string message;
    std::optional<std::string> error;

    nlohmann::json to_json() const;
};

struct LmStudioStatusResult {
    bool connected = false;
    std::string endpoint;
    std::optional<std::string> active_model;
    std::optional<std::string> resolved_model_id;
    std::string preferred_model;
    std::vector<std::string> available_models;
    std::optional<std::string> error;

    nlohmann::json to_json() const;
};

class ComicTranslator {
public:
    explicit ComicTranslator(const TranslatorConfig& config = {});

    // -------------------------------------------------------------------------
    // Normalization & Catalog Helpers
    // -------------------------------------------------------------------------
    static std::string normalize_provider_id(const std::string& raw);
    static const ProviderSpec& get_provider_spec(const std::string& provider_id);
    static std::string normalize_provider_base_url(
        const std::string& provider_id,
        const std::string& raw_base_url);
    static std::string resolve_provider_api_key(
        const std::string& provider_id,
        const std::string& raw_api_key);

    static std::string normalize_source_language(const std::string& raw);
    static std::string normalize_target_language(const std::string& raw);
    static const TargetLanguageSpec& get_target_language_spec(const std::string& code);
    static std::string normalize_reasoning_effort(const std::string& raw);
    static std::string normalize_translation_tone(const std::string& raw);
    static std::map<std::string, std::string> parse_glossary(const nlohmann::json& raw);

    static bool is_embedding_model(const std::string& model_id);
    static bool is_likely_vision_model(
        const std::string& model_id,
        const std::string& provider_id);
    static double score_model_match(
        const std::string& preferred_name,
        const std::string& candidate_id);
    static std::optional<std::string> format_display_model_name(
        const std::string& provider_id,
        const std::optional<std::string>& resolved_model_id,
        const std::string& preferred_model);

    // -------------------------------------------------------------------------
    // Provider Discovery, Connection Test & LM Studio Status
    // -------------------------------------------------------------------------
    static std::vector<std::string> list_provider_models(
        const std::string& provider = "lmstudio",
        const std::string& base_url = "",
        const std::string& api_key = "",
        double timeout_seconds = 8.0,
        bool include_embeddings = true);

    static std::string resolve_provider_model(
        const std::string& provider = "lmstudio",
        const std::string& preferred_model = "",
        const std::string& base_url = "",
        const std::string& api_key = "",
        double timeout_seconds = 8.0,
        const std::optional<std::vector<std::string>>& available_models = std::nullopt);

    static ProviderTestResult test_provider_connection(
        const std::string& provider = "lmstudio",
        const std::string& base_url = "",
        const std::string& api_key = "",
        const std::string& preferred_model = "",
        double timeout_seconds = 10.0);

    static LmStudioStatusResult get_lmstudio_status(
        const nlohmann::json& settings,
        double timeout_seconds = 5.0);

    // -------------------------------------------------------------------------
    // Prompting, Vision Encoding, Reasoning Stripping & JSON Extraction
    // -------------------------------------------------------------------------
    static nlohmann::json comic_translation_json_schema();

    static std::string build_system_prompt(
        const std::string& source_lang,
        const std::string& target_lang,
        const std::string& reasoning_effort = "off",
        const std::string& custom_instructions = "",
        const std::map<std::string, std::string>& glossary = {},
        bool has_page_image = false,
        const std::string& translation_tone = "standard");

    // Strip `<think>...</think>`, `<thought>...</thought>`, `<|thought|>...<|thought|>`,
    // and unclosed reasoning blocks prior to JSON parsing.
    static std::string strip_thinking_tags(const std::string& raw_content);

    static std::map<std::string, std::string> extract_and_clean_translations(
        const std::string& raw_content,
        const std::vector<DialogueBlockInput>& input_blocks,
        bool strip_thinking = true);

    // Deterministically enforce glossary replacement rules on translated text
    // if any glossary source term still remains verbatim in the output.
    static std::string apply_glossary_replacements(
        const std::string& text,
        const std::map<std::string, std::string>& glossary);

    // Resize RGB `(width * height * 3)` image to `<= max_side` px and encode as compact JPEG data URI.
    static std::string encode_rgb_for_vision(
        const uint8_t* rgb_data,
        int width,
        int height,
        int max_side = 1024,
        int jpeg_quality = 84);

    // Decode raw image bytes (JPG/PNG/WebP/BMP) -> resize -> JPEG data URI.
    static std::string encode_bytes_for_vision(
        const uint8_t* image_bytes,
        size_t num_bytes,
        int max_side = 1024,
        int jpeg_quality = 84);

    // -------------------------------------------------------------------------
    // Instance Translation API
    // -------------------------------------------------------------------------
    const TranslatorConfig& config() const noexcept { return config_; }
    const ProviderSpec& spec() const noexcept { return *spec_; }

    std::string resolve_model(bool force_refresh = false);

    PageTranslationResult translate_blocks(
        const std::vector<DialogueBlockInput>& blocks,
        const TranslateRequestOptions& options = {});

private:
    std::pair<std::string, bool> call_openai_chat_completion(
        const std::string& model_id,
        const std::string& system_prompt,
        const std::string& user_payload,
        double temperature,
        std::optional<double> top_p,
        std::optional<int> top_k,
        int max_tokens,
        std::optional<double> frequency_penalty,
        const std::string& reasoning_effort,
        const std::optional<std::string>& page_image_data_uri) const;

    std::pair<std::string, bool> call_anthropic_messages(
        const std::string& model_id,
        const std::string& system_prompt,
        const std::string& user_payload,
        double temperature,
        std::optional<double> top_p,
        std::optional<int> top_k,
        int max_tokens,
        const std::string& reasoning_effort,
        const std::optional<std::string>& page_image_data_uri) const;

    TranslatorConfig config_;
    const ProviderSpec* spec_ = nullptr;
    std::optional<std::string> resolved_model_;
};

}  // namespace mangaflow::pipeline
