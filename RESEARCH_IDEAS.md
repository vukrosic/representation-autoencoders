# RAE Research Directions

This file outlines potential research questions and experiments for the simplified Representation Autoencoder (RAE) codebase.

---

### 1. The "Semantic Gap" Research
Foundation models (like DINOv2 or SigLIP) are trained to understand *semantics*, not pixels. 
*   **Question:** Which foundation model provides the most "generative-friendly" latent space?
*   **Experiment:** Swap between `Dinov2withNorm`, `SigLIP2`, and `MAE` in the `RAE` initialization. Train for short periods (e.g., 5 epochs) and compare reconstruction fidelity.
*   **Hypothesis:** Contrastive models (SigLIP) might lose more low-level geometric detail compared to self-distillation models (DINOv2).

### 2. Decoder Bottleneck Analysis
The decoder's job is to uncompress high-level features back into pixel space.
*   **Question:** What is the minimum decoder complexity required for high-fidelity reconstruction?
*   **Experiment:** Modify `decoder_config_path` to compare `ViT-Small` vs. `ViT-Large` decoders.
*   **Metric:** Analyze if increasing depth helps more with *semantic color accuracy* or *spatial geometry*.

### 3. Latent Space Robustness
In the full RAE paper, noise (`noise_tau`) is added to latents to support Stage 2 diffusion.
*   **Question:** How does latent-level noise affect the reconstruction objective?
*   **Experiment:** Train three RAEs with `noise_tau` set to `0.0`, `0.5`, and `1.0`.
*   **Analysis:** Visualize whether the decoder learns to robustly "denoise" foundation features or if it starts creating artifacts.

### 4. Zero-Shot Domain Transfer
Foundation models are trained on internet-scale natural images.
*   **Question:** How does an RAE trained on foundation features handle out-of-distribution data?
*   **Experiment:** Train on natural images (ImageNet), then test on medical X-rays or satellite imagery.
*   **Goal:** Determine if failure is due to a lack of encoder features or the decoder's pixel-space prior.

### 5. Representation Redundancy (MRAE)
*   **Question:** Can we reconstruct the image using only a small subset of the foundation model's latent patches?
*   **Experiment:** Modify `src/stage1/rae.py` to randomly mask 50% or 75% of the latent tokens before decoding.
*   **Goal:** Test if foundation model representations are redundant enough to support "Masked Representation Autoencoding."

### 6. Semantic Editability & Linearity
*   **Question:** Are foundation model latents "linearly" editable for reconstruction?
*   **Experiment:** Perform linear interpolation between two latents ($Z = \alpha A + (1-\alpha) B$) and decode.
*   **Analysis:** Observe if reconstructive morphing is smooth (semantic) or results in "teleporting" artifacts.
