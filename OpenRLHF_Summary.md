# OpenRLHF: Architecture and Usage Summary

This document provides a summary of the architecture and usage of the OpenRLHF repository.

## Architecture

OpenRLHF is a high-performance, open-source framework for Reinforcement Learning from Human Feedback (RLHF). Its architecture is designed for scalability and efficiency, particularly for training large language models (LLMs).

Key architectural features include:

*   **Distributed by Default:** Built on the [Ray](https://www.ray.io/) framework, it enables the distribution of different RLHF components (Actor, Critic, Reward, and Reference models) across multiple GPUs and nodes. This allows for training models up to 70B parameters.
*   **Hybrid Engine:** Features a "Hybrid Engine" that allows different models and the vLLM inference engine to share GPU resources, maximizing utilization and minimizing idle time.
*   **vLLM for Inference:** Integrates with [vLLM](https://github.com/vllm-project/vllm) for accelerated sample generation, a critical bottleneck in RLHF. This provides high-throughput and memory-efficient inference.
*   **Memory-Efficient Training:** Leverages [DeepSpeed's](https://github.com/microsoft/DeepSpeed) ZeRO-3 and Auto Tensor Parallelism (AutoTP) to train large models without the overhead of heavy frameworks.
*   **Optimized PPO:** The implementation of Proximal Policy Optimization (PPO) is optimized with various techniques for improved training stability and reward quality.
*   **Hugging Face Integration:** Tightly integrated with the [Hugging Face](https://huggingface.co/) ecosystem, allowing for easy loading and fine-tuning of pre-trained models from the Hugging Face Hub.

## Usage

The primary use of OpenRLHF is to align LLMs with human preferences using RLHF. It supports a variety of algorithms and techniques, making it a versatile tool for LLM training and research.

### Core Use Cases:

*   **Supervised Fine-tuning (SFT):** The initial step in the alignment process, where a pre-trained model is fine-tuned on a specific dataset.
*   **Reward Model (RM) Training:** Training a model to learn human preferences from a dataset of comparisons.
*   **Reinforcement Learning (RL) Fine-tuning:** Using the trained reward model to further fine-tune the SFT model with algorithms like PPO, DPO, and KTO.

### Key Features and Supported Algorithms:

*   **RL Algorithms:** PPO, REINFORCE++, DPO, IPO, cDPO, KTO, and Rejection Sampling.
*   **Training Techniques:** Conditional SFT, Knowledge Distillation, and Process Reward Model (PRM) training.
*   **Efficiency:** Supports FlashAttention2, QLoRA, LoRA, and packing of training samples.
*   **Flexibility:** Compatible with Hugging Face's `apply_chat_template` for easy dataset formatting.
*   **Monitoring:** Integrated with Wandb and TensorBoard for experiment tracking.
*   **Scalability:** Provides scripts for multi-node training on SLURM clusters.

## GPU Acceleration

OpenRLHF employs a multi-faceted approach to GPU acceleration, combining several state-of-the-art technologies to achieve high performance and scalability.

### vLLM for Accelerated Inference

*   **High-Throughput Inference:** OpenRLHF integrates `vLLM` for high-throughput, memory-efficient sample generation, which is a critical bottleneck in RLHF.
*   **Synchronous and Asynchronous Modes:** It uses `vllm.LLM` for synchronous and `vllm.AsyncLLMEngine` for asynchronous inference, managed as Ray actors for distributed execution.
*   **Hybrid Engine:** The **Hybrid Engine** (`--colocate_all_models`) allows vLLM and other models to share GPU resources. The `--vllm_enable_sleep` flag enables vLLM to yield the GPU when idle, maximizing resource utilization.
*   **Configuration:** Key parameters like `--vllm_num_engines`, `--vllm_tensor_parallel_size`, and `--vllm_gpu_memory_utilization` allow for fine-grained control over the vLLM engines.
*   **Weight Synchronization:** Weights are kept consistent between the DeepSpeed training process and the vLLM inference engine through a broadcast mechanism.

### DeepSpeed for Efficient Training

*   **Foundation of Training:** The framework is built on DeepSpeed, using the `deepspeed` command to launch all training jobs.
*   **ZeRO-3 Memory Optimization:** It utilizes **ZeRO-3 (Zero Redundancy Optimizer)** (`--zero_stage 3`) to partition model states, gradients, and optimizer states, significantly reducing memory requirements for large models.
*   **Tensor Parallelism:** **Tensor Parallelism** (`--ds_tensor_parallel_size`) is supported for sharding model layers across multiple GPUs.
*   **Hybrid Engine Sleep Mode:** A **sleep mode** (`--deepspeed_enable_sleep`) is implemented for the Hybrid Engine, allowing DeepSpeed to release GPU resources when not actively training.
*   **Strategy Class:** The `DeepspeedStrategy` class encapsulates DeepSpeed's configuration and initialization, simplifying its use.

### FlashAttention and RingAttention for Optimized Attention

*   **FlashAttention2:** **FlashAttention2** (`--flash_attn`) is used as a highly optimized attention implementation to accelerate computation and reduce memory access, especially when using `--packing_samples`.
*   **RingAttention:** **RingAttention** (`--ring_attn_size`) is implemented to distribute the attention computation across multiple GPUs in a ring, enabling the training of models with very long sequences that wouldn't fit on a single GPU.
*   **Library Integration:** The `ring_flash_attn` library is used for this, and the `DeepspeedStrategy` sets up the necessary process groups for sequence parallelism.

In summary, OpenRLHF is a powerful and flexible framework for researchers and developers working on LLM alignment. It provides the necessary tools to efficiently train and fine-tune large models using state-of-the-art RLHF techniques.
