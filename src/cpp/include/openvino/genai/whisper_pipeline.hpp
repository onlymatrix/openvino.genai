// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <variant>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/whisper_generation_config.hpp"

namespace ov::genai {

using OptionalWhisperGenerationConfig = std::optional<WhisperGenerationConfig>;

using RawSpeechInput = std::vector<float>;

/**
 * @brief base class for chunk streamers. In order to use inherit from from this class and implement put, and methods
 *
 * @param m_tokenizer tokenizer
 */
class OPENVINO_GENAI_EXPORTS ChunkStreamerBase : public StreamerBase {
public:
    /// @brief put is called every time new token chunk is generated,
    /// @return bool flag to indicate whether generation should be stopped, if return true generation stops
    virtual bool put_chunk(std::vector<int64_t> tokens) = 0;
};

// Return flag corresponds whether generation should be stopped: false means continue generation, true means stop.
using ChunkStreamerVariant =
    std::variant<std::function<bool(std::string)>, std::shared_ptr<ChunkStreamerBase>, std::monostate>;

struct OPENVINO_GENAI_EXPORTS WhisperRawPerfMetrics {
    /** @brief Duration for each features extraction call */
    std::vector<MicroSeconds> features_extraction_durations;
};

struct OPENVINO_GENAI_EXPORTS WhisperPerfMetrics : public PerfMetrics {
    /** @brief Mean and standart deviation of Features Extraction Duration in milliseconds */
    MeanStdPair features_extraction_duration;

    MeanStdPair get_features_extraction_duration();

    WhisperPerfMetrics() = default;

    WhisperPerfMetrics(PerfMetrics& perf_metrics) : PerfMetrics(perf_metrics){};

    //void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;
    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt);

    WhisperPerfMetrics operator+(const WhisperPerfMetrics& metrics) const;
    WhisperPerfMetrics& operator+=(const WhisperPerfMetrics& right);

    WhisperRawPerfMetrics whisper_raw_metrics;
};

struct WhisperDecodedResultChunk {
    // start of chunk in seconds
    float start_ts;

    // end of chunk in seconds
    // -1.0f if chunk started but model did not predict an ending timestamp
    // can happen if audio is cut off in the middle of a word
    float end_ts = -1.0f;
    std::string text;
};

struct WhisperDecodedResults {
    std::vector<std::string> texts;
    std::vector<float> scores;
    std::optional<std::vector<WhisperDecodedResultChunk>> chunks = std::nullopt;
    WhisperPerfMetrics perf_metrics;

    operator std::string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    operator std::vector<std::string>() const {
        return texts;
    }

    friend std::ostream& operator<<(std::ostream& os, const WhisperDecodedResults& dr) {
        OPENVINO_ASSERT(dr.scores.size() == dr.texts.size(),
                        "The number of scores and texts doesn't match in WhisperDecodedResults.");
        if (dr.texts.empty()) {
            return os;
        }
        if (dr.texts.size() == 1) {
            os << dr.texts[0];
            return os;
        }
        for (size_t i = 0; i < dr.texts.size() - 1; ++i) {
            os << std::to_string(dr.scores[i]) << ": " << dr.texts[i] << '\n';
        }
        return os << std::to_string(dr.scores.back()) << ": " << dr.texts.back();
    }
};

/**
 * @brief Automatic speech recognition pipeline
 */
class OPENVINO_GENAI_EXPORTS WhisperPipeline {
    class WhisperPipelineImplBase;
    std::unique_ptr<WhisperPipelineImplBase> m_impl;

    class StaticWhisperPipeline;
    class WhisperPipelineStatefulImpl;

public:
    /**
     * @brief Constructs a WhisperPipeline from xml/bin files, tokenizers and configuration in the
     * same dir.
     *
     * @param models_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param properties optional properties
     */
    WhisperPipeline(const std::filesystem::path& models_path,
                    const std::string& device,
                    const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a WhisperPipeline from xml/bin files, tokenizers and configuration in the
     * same dir. Accepts arbitrary list of optional properties.
     *
     * @param models_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param properties optional properties
     */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    WhisperPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : WhisperPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~WhisperPipeline();

    /**
     * @brief High level generate that receives raw speech as a vector of floats and returns decoded output.
     *
     * @param raw_speech_input raw speech input. Required to be normalized to near [-1, 1] range and have 16k Hz
     * sampling rate.
     * @param generation_config optional GenerationConfig
     * @param streamer optional streamer. Streamer supported for short-form audio (< 30 seconds) with
     * `return_timestamps=False` only
     * @return WhisperDecodedResults decoded resulting text transcription
     */
    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                   OptionalWhisperGenerationConfig generation_config = std::nullopt,
                                   ChunkStreamerVariant streamer = std::monostate());

    /**
     * @brief High level generate that receives raw speech as a vector of floats and returns decoded output.
     * properties can be in any order pipe.generate(..., ov::genai::max_new_tokens(100),
     * ov::genai::streamer(lambda_func)).
     *
     * @param raw_speech_input raw speech input
     * @param properties properties
     * @return WhisperDecodedResults decoded resulting text transcription
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<WhisperDecodedResults, Properties...> generate(const RawSpeechInput& raw_speech_input,
                                                                              Properties&&... properties) {
        return generate(raw_speech_input, AnyMap{std::forward<Properties>(properties)...});
    }
    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input, const ov::AnyMap& config_map);

    ov::genai::Tokenizer get_tokenizer();
    WhisperGenerationConfig get_generation_config() const;
    void set_generation_config(const WhisperGenerationConfig& config);
};

OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> streamer(ChunkStreamerVariant func);
OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> generation_config(const WhisperGenerationConfig& config);
}  // namespace ov::genai
