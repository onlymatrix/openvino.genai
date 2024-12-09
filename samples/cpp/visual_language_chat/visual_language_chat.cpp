// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <windows.h>  

bool g_stop = false;

BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        g_stop = true;
        return TRUE;
    }
    return FALSE;
}

bool print_subword(std::string&& subword) {
    if (g_stop == true) {
        g_stop = false;
        return true;
    }
    return !(std::cout << subword << std::flush);
}

void printMetrics(ov::genai::PerfMetrics& metrics) {
    std::cout << "\n\n" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    // std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    //  TTFT:Time To First Token
    std::cout << "First Token: " << metrics.get_ttft().mean << " ms" << std::endl;
    // Throughput
    std::cout << "Tokens Per Second: " << metrics.get_throughput().mean << " tokens/s" << std::endl;
    // TPOT
    std::cout << "TPOT: " << metrics.get_tpot().mean << " ms/token " << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ms" << std::endl;
}

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES>");
    }

    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    
    
    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    std::string device = "GPU";  // GPU can be used as well
    ov::AnyMap enable_compile_cache;
    if ("GPU" == device) {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, enable_compile_cache);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 2048;
    generation_config.do_sample = true;
    generation_config.top_k = 100;
    generation_config.top_p = 0.8;
    generation_config.temperature = 0.7;
    generation_config.presence_penalty = 1.05;

    std::string prompt;

    pipe.start_chat();
    std::cout << "Question: ";

    std::getline(std::cin, prompt);
    std::cout << "\nAI:";
    ov::genai::DecodedResults res = pipe.generate(prompt,
                                                  ov::genai::images(rgbs),
                                                  ov::genai::generation_config(generation_config),
                                                  ov::genai::streamer(print_subword));

    ov::genai::PerfMetrics metrics = res.perf_metrics;
    printMetrics(metrics);
    std::cout << "\n----------\n"
                 "Question: ";

    while (std::getline(std::cin, prompt)) {
        if (prompt == "bye") {
            break;
        }
        std::cout << "\nAI:";
        res =
            pipe.generate(prompt, ov::genai::generation_config(generation_config), ov::genai::streamer(print_subword));
        metrics = metrics + res.perf_metrics;
        printMetrics(metrics);
        std::cout << "\n----------\n"
                     "Question: ";
    }

    pipe.finish_chat();

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
