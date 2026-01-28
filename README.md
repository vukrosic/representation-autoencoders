# Representation Autoencoders (RAE) Research Sandbox

A simplified, research-first implementation of **Representation Autoencoders**. This repository is designed for experimenting with how foundation model representations (DINOv2, SigLIP, etc.) can be reconstructed into pixels.

> **Note:** This is a streamlined version of the [original RAE implementation](https://github.com/bytetriper/RAE), optimized for readability and rapid experimentation.

## 🚀 Concept
RAEs use frozen, pretrained foundation models as encoders and train a lightweight ViT decoder to reconstruct the original image. This creates a bridge between **Semantic Latent Spaces** and **Pixel Space**, enabling high-fidelity generative modeling without the need for traditional VAEs.

## 🛠️ Project Structure
- `src/stage1/`: Core RAE architecture logic.
- `src/stage1/encoders/`: Support for DINOv2, SigLIP2, and MAE.
- `src/stage1/decoders/`: Trainable ViT decoder.
- `train_simple.py`: Lightweight training script for MSE-based reconstruction.
- `RESEARCH_IDEAS.md`: A roadmap of potential experiments and research questions.

## 🏃 Quick Start

### 1. Training
To start a simple reconstruction training run:
```bash
python train_simple.py
```
*(Make sure to adjust the `data_path` in the script to point to your ImageNet-style dataset.)*

### 2. Configuration
The `RAE` class can be initialized with different foundation model backbones:
```python
model = RAE(
    encoder_cls='Dinov2withNorm',
    encoder_config_path='facebook/dinov2-base',
    decoder_config_path='facebook/vit-mae-base'
)
```

## 🔬 Research Directions
This repository is built for exploration. See [RESEARCH_IDEAS.md](./RESEARCH_IDEAS.md) for details on:
- Comparing semantic gaps between DINOv2 vs. SigLIP.
- Analyzing decoder complexity bottlenecks.
- Latent space robustness and denoising capabilities.
- Zero-shot domain transfer (e.g., medical/satellite imagery).

## 📄 License
This project inherits the MIT License from its parent project.
