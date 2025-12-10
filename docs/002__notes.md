CUDA Toolkit 12
- https://developer.nvidia.com/cuda-toolkit-archive
- CUDA 12.x family will work with onnxruntime-gpu v1.20.1
- run `nvidia-smi` to check versions
- I have v13 installed and cannot downgrade

cuDNN v 9
- CUDA Deep Neural Network library. It's a GPU-accelerated library of primitives for deep learning
- compatible cuDNN package (e.g., cuDNN 9 for CUDA 12.x) - downloaded from the NVIDIA developer website



for example, nomic-embed-text-v1.5 has:
- original huggingface model - e.g. https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- gguf model - e.g. https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
- onnx model

could you explain to me:
- what are gguf models
- what are onnx models, and what is a vocab.txt file
- how do they relate to the original huggingface model
- how are these relate to pytorch, vllm, LLamaSharp

what is Tokenizers and how does it fit into the picture?
how do these relate to cuda?