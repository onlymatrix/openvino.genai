# C++ visual language chat

This example showcases inference of Visual language models (VLMs): [`openbmb/MiniCPM-V-2_6`](https://huggingface.co/openbmb/MiniCPM-V-2_6). The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::VLMPipeline` and runs the simplest deterministic greedy sampling algorithm. There is also a example [python](https://github.com/onlymatrix/miniCPMs/tree/main/miniCPM-V26) which provides an example of Visual-language assistant.

# Build openvino.genai
```sh
git clone https://github.com/onlymatrix/openvino.genai.git
cd openvino.genai/
git submodule update --init
```
For Windows

Compile the project using CMake:

```sh
<OpenVINO_2024.5.0 dir>\setupvars.bat
cd openvino.genai
mkdir build & cd build
cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build build -j --config Release
cmake --build . --config RelWithDebInfo --verbose -j4
```

# Download and convert the model

``` sh
conda create -n ov_minicpm python=3.10
conda activate ov_minicpm
pip install -r requirements.txt
python convert_minicpmV26.py -m /path/to/minicpmV26 -o /path/to/minicpmV26_ov
```

# Run

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

`visual_language_chat /path/to/minicpmV26_ov 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg`

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model `llava-hf/llava-v1.6-mistral-7b-hf` can benefit from being run on a dGPU. Modify the source code to change the device for inference to the `GPU`.

See [SUPPORTED_MODELS.md](../../../src/docs/SUPPORTED_MODELS.md#visual-language-models) for the list of supported models.
