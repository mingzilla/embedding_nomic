# Model Formats and Frameworks Explanation

## Summary

| stage        | framework             | output     | latest cuda support      | base runner  | wrapper runner                          |
|--------------|-----------------------|------------|--------------------------|--------------|-----------------------------------------|
| build model  | PyTorch, TensorFlow   | HF model   | CUDA Toolkit 13          | PyTorch      | sentence-transformers, vLLM             |
| convert GGUF | llama.cpp             | model.gguf | CUDA Toolkit 13          | llama.cpp    | LlamaSharp, llama-cpp-python, koboldcpp |
| convert ONNX | PyTorch^, TensorFlow^ | model.onnx | CUDA Toolkit 12, cuDNN 9 | ONNX Runtime |                                         |

- PyTorch^ - torch.onnx.export()
- TensorFlow^ - tf2onnx
- PyTorch, TensorFlow, llama.cpp - compatible with specific CUDA version used to compile the framework

## Model Format Hierarchy

```
[Original HuggingFace Model]
         |
         |-- Parameters: weights & biases
         |   |- **PyTorch** = framework for training/running ML models
         |   |- **Weights & biases** = the learned parameters
         |   +- **.bin / .safetensors** = PyTorch file formats that store parameters
         |
         |-- Model config (config.json)
         |   |- a model must have config.json, regardless if trained by PyTorch or TensorFlow
         |   +- config.json describes the model architecture - e.g. num_hidden_layers
         |
         |-- Tokenizer - uses files (tokenizer.json, vocab.txt)
         |   |- runs an algorithm to split text, this uses tokenizer.json
         |   +- converts tokens into numbers, this uses vocab.txt
         |
         +--CONVERT-->
                |
                +--[GGUF] (llama.cpp, LLamaSharp)
                |
                +--[ONNX] (ONNX Runtime, cross-platform)

Note:
(TensorFlow vs PyTorch) == (Angular vs React)
```

Typical release of a model includes:

- **PyTorch** weights (.safetensors or .bin)
- **Tokenizer** files (tokenizer.json, vocab.txt, etc.)
- **Config** (config.json)
- Sometimes **GGUF** or **ONNX** conversions
    - (community-provided, not always official)
    - ONNX is a conversion format for deployment
