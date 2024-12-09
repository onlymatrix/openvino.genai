// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

namespace {

ov::genai::MeanStdPair calc_mean_and_std(const std::vector<ov::genai::MicroSeconds>& durations) {
    if (durations.size() == 0) {
        return {-1, -1};
    }
    // Accepts time durations in microseconds and returns standard deviation and mean in milliseconds.
    float mean = std::accumulate(durations.begin(),
                                 durations.end(),
                                 0.0f,
                                 [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                     return acc + duration.count() / 1000.0f;
                                 });
    mean /= durations.size();

    float sum_square_durations =
        std::accumulate(durations.begin(),
                        durations.end(),
                        0.0f,
                        [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                            auto d = duration.count() / 1000.0f;
                            return acc + d * d;
                        });
    float std = std::sqrt(sum_square_durations / durations.size() - mean * mean);
    return {mean, std};
}

}  // namespace


MeanStdPair WhisperPerfMetrics::get_features_extraction_duration() {
    evaluate_statistics();
    return features_extraction_duration;
}

void WhisperPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    features_extraction_duration = ov::genai::calc_mean_and_std(whisper_raw_metrics.features_extraction_durations);
    PerfMetrics::evaluate_statistics(start_time);
};

WhisperPerfMetrics WhisperPerfMetrics::operator+(const WhisperPerfMetrics& right) const {
    PerfMetrics base_result = PerfMetrics::operator+(right);
    WhisperPerfMetrics result{base_result};

    // copy left whisper raw metrics
    result.whisper_raw_metrics = whisper_raw_metrics;

    // insert right metrics
    auto& result_features_extraction_durations = result.whisper_raw_metrics.features_extraction_durations;
    auto& right_features_extraction_durations = right.whisper_raw_metrics.features_extraction_durations;
    result_features_extraction_durations.insert(result_features_extraction_durations.end(),
                                                right_features_extraction_durations.begin(),
                                                right_features_extraction_durations.end());
    return result;
}

WhisperPerfMetrics& WhisperPerfMetrics::operator+=(const WhisperPerfMetrics& right) {
    *this = *this + right;
    return *this;
}

}  // namespace genai
}  // namespace ov
