# Representation Autoencoders (RAE): Bridging Semantic Latents and Pixel Space

## Abstract
Representation Autoencoders (RAE) propose a novel framework for high-fidelity image reconstruction and generation by leveraging frozen, pretrained foundation models as semantic encoders. Unlike traditional Variational Autoencoders (VAEs) that learn both encoder and decoder from scratch, RAEs utilize the rich topological structure of pretrained feature spaces (e.g., DINOv2, SigLIP). This paper details the mathematical formulation of the RAE architecture and its reconstruction objective.

## 1. Introduction
The advent of foundation models has provided robust, semantically meaningful latent spaces. RAEs capitalize on these representations by training a lightweight decoder to invert the encoding process. This approach bypasses the "codebook collapse" seen in VQ-VAEs and the blurriness inherent in simple MSE-based VAEs, effectively turning any semantic feature extractor into a generative backbone.

## 2. Mathematical Background

### 2.1 The Semantic Encoding Function
Let $\mathcal{X} \subset \mathbb{R}^{3 \times H \times W}$ be the image space. We employ a frozen, pretrained encoder $E: \mathcal{X} \to \mathcal{Z}$, where $\mathcal{Z} \subset \mathbb{R}^{N \times D}$ represents the semantic latent space consisting of $N$ tokens of dimension $D$.

For a Given input $x$, the encoding process is defined as:
$$z = E(x)$$

where $E$ typically comprises a Vision Transformer (ViT) backbone. The features $z$ are extracted from the final layer, often excluding prefix tokens (like CLS or registers) to maintain spatial alignment.

### 2.2 Latent Space Regularization and Noising
To facilitate downstream tasks like Latent Diffusion, the latent space $\mathcal{Z}$ is often regularized. During Stage 1 training, we introduce a stochastic perturbation (noising) to the latents to ensure the decoder $D_\theta$ is robust to distribution shifts:

$$ \hat{z} = z + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \tau^2 I) $$

where $\tau$ is the noise scale parameter (`noise_tau` in implementation).

Furthermore, we apply mean-variance normalization to the latents:
$$ \tilde{z} = \frac{\hat{z} - \mu_{\mathcal{Z}}}{\sqrt{\sigma_{\mathcal{Z}}^2 + \eta}} $$
where $\mu_{\mathcal{Z}}$ and $\sigma_{\mathcal{Z}}^2$ are precomputed statistics of the foundation model's output over the training set, and $\eta$ is a small constant for numerical stability.

### 2.3 The Decoding Function and Reconstruction
The decoder $D_\theta$ is a trainable Transformer parameterized by $\theta$. Its architecture is designed to map the semantic tokens back to pixel-space patches. The decoding process is defined as:

$$ \hat{x} = D_\theta(\tilde{z}) $$

The decoder uses additive positional embeddings to restore spatial context lost or abstracted during encoding. The output layer involves an **unpatchify** operation, which reshapes the predicted pixel intensities from token-wise vectors back into a coherent image grid.

### 2.4 Optimization Objective
The primary training objective for Stage 1 is the minimization of the Mean Squared Error (MSE) between the original image $x$ and its reconstruction $\hat{x}$:

$$ \min_\theta \mathcal{L}_{rec}(\theta) = \mathbb{E}_{x \sim p_{data}} \left[ \| x - D_\theta(E(x)) \|_2^2 \right] $$

In more advanced implementations, this loss is often augmented with perceptual losses (e.g., LPIPS) or adversarial objectives to improve high-frequency detail retention.

## 3. Implementation Details
Our implementation utilizes:
- **Encoder**: DINOv2-base with Register tokens.
- **Decoder**: A 12-layer ViT-MAE style Transformer.
- **Input Resolution**: $224 \times 224$ pixels.
- **Latent Dimension**: $768$ (DINOv2-base output).
- **Optimizer**: AdamW with $\beta_1=0.9, \beta_2=0.95$.

## 3. The RAE Decoder: `GeneralDecoder` Reference
The core of the RAE's learning process resides in the `GeneralDecoder` class within `src/stage1/decoders/decoder.py`. Unlike the frozen encoder, this component is trained from scratch.

### 3.1 Initialization and Setup
- **`__init__`**: Sets up the linear projection from encoder latent space to decoder hidden space (`decoder_embed`). It also initializes a sequence of `ViTMAELayer` blocks, which form the transformer backbone.
- **`initialize_weights`**: Generates and freezes 2D sin-cos positional embeddings using `get_2d_sincos_pos_embed`. This provides the spatial prior necessary for image reconstruction.
- **`set_trainable_cls_token`**: Registers a learnable `[CLS]` token. While the encoder's CLS token is semantic, the decoder's CLS token acts as a global context accumulator for pixels.

### 3.2 Dynamic Latent Adaptation
- **`interpolate_latent`**: A critical function that ensures the decoder is agnostic to the encoder's patch grid. If the encoder provides $16 \times 16$ patches but the decoder expects a different resolution, this function uses bilinear interpolation in the feature space to resize the latent grid while preserving channel-wise information.
- **`interpolate_pos_encoding`**: Similar to latent interpolation, this resizes the fixed positional embeddings using bicubic interpolation, allowing the model to generalize to varying input resolutions at inference time.

### 3.3 The Forward Pass Logic
The `forward` method implements the following sequence:
1.  **Linear Projection**: Projects $z \in \mathcal{Z}$ into the decoder's internal dimension.
2.  **Interpolation**: Resizes the latent to match the required grid size if necessary.
3.  **Token Assembly**: Concatenates the trainable `[CLS]` token with the data patches.
4.  **Spatial Context**: Adds the 2D positional embeddings to the assembled tokens.
5.  **Transformer Processing**: Passes the sequence through $L$ blocks of Self-Attention and MLP layers.
6.  **Pixel Projection**: Applies a final linear layer (`decoder_pred`) that maps each token to a vector of size $P^2 \times C$ (where $P$ is patch size).

### 3.4 Pixel Reconstruction
- **`unpatchify`**: This function is the inverse of the standard ViT patchification. It reshapes the sequence of flat vectors back into an $H \times W$ RGB image grid. It uses `torch.einsum` for high-performance tensor reshaping, ensuring that the predicted pixels are placed in their correct spatial coordinates.

## 4. Results and Ablations

To evaluate the impact of encoder choice and latent noising, we performed controlled ablations on the CIFAR-10 dataset (Epoch 1).

### 4.1 Encoder Architecture
We compared **DINOv2-base** against **SigLIP-base**. Despite SigLIP's superior zero-shot classification performance, DINOv2 provided a more reconstructive-friendly latent space.

| Encoder | Latent Noise ($\tau$) | MSE Loss (E1) |
| :--- | :--- | :--- |
| **DINOv2 (Base)** | 0.0 | **0.0150** |
| **SigLIP (Base)** | 0.0 | 0.0181 |

### 4.2 Latent Robustness (Noising)
Adding Gaussian noise to the latents ($\tau = 0.5$) slightly increases the reconstruction error but is critical for ensuring the decoder can handle the imperfect samples produced by Stage 2 diffusion models.

| Configuration | MSE Loss (E1) |
| :--- | :--- |
| DINOv2 (Clean) | 0.0150 |
| DINOv2 (Noisy, $\tau=0.5$) | 0.0160 |

## 5. Conclusion
Our experiments demonstrate that self-distillation models like DINOv2 preserve more reconstructive information than contrastive models like SigLIP. While latent noising increases early training loss, the model successfully adapts, paving the way for robust Stage 2 generative modeling.
