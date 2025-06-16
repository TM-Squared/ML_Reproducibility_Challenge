---
# ML Reproducibility Challenge: Explainability Methods for CLIP

This repo explores and compares various **explainability methods for CLIP (Contrastive Language-Image Pre-training)** models, focusing on **Grad-ECLIP** to make CLIP's predictions more interpretable.

---
## Table of Contents

- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
  - [CLIP Overview](#clip-overview)
  - [Grad-ECLIP Explained](#grad-eclip-explained)
  - [Dense Encoding & Position Embeddings](#dense-encoding--position-embeddings)
- [Installation](#installation)

---
## Introduction

We visually compare saliency maps (heatmaps) from methods like **Grad-CLIP**, **GAME**, **MaskCLIP**, **GradCAM**, **Rollout Attention**, **Self-Attention**, **CLIP Surgery**, and **M2IB** to understand what drives CLIP's image predictions.

---
## Key Concepts

### CLIP Overview

**CLIP** learns visual representations from natural language, creating a shared embedding space. It uses a **Vision Transformer (ViT-B/16)** for images and a **Text Encoder** for text, projecting both into a common space.

**Objective Function**:
$\mathcal{L}_{CLIP} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i^v, z_i^t) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i^v, z_j^t) / \tau)}$

### Grad-ECLIP Explained

Grad-ECLIP improves CLIP's interpretability by identifying key image regions and linking them to text queries. Unlike global [CLS] token summaries, Grad-ECLIP uses **gradients of the Vision Transformer's attention mechanisms**.

**Core Idea**: Gradients show how each attention connection influences the final image-text similarity.

**Explainability Map**:
$M_{\text{Grad-ECLIP}} = \sum_{l=1}^{L} \left|\frac{\partial \mathcal{L}}{\partial A^{(l)}}\right| \odot A^{(l)}$

Our implementation of the **attention layer** is crucial as it allows access to Q, K, V matrices and original attention weights, preserving the computation graph for gradient calculation.

### Dense Encoding & Position Embeddings

**Dense encoding** is fundamental for Grad-ECLIP, ensuring spatial correspondence between features and image regions by using **all patch tokens** instead of just the [CLS] token.

**Adaptive Position Embedding**: Handles varying image resolutions using **bicubic interpolation** to adapt the original position embeddings.
$\text{PE}_{\text{interpolated}} = \text{Interpolate}(\text{PE}_{\text{original}}, H_{\text{new}}, W_{\text{new}})$

Implemented as:
```python
img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic' align_corners=False)
```

## Grad-ECLIP Algorithm

Here's how Grad-ECLIP works step-by-step:

1.  **Forward Pass**: We densely encode both the image and the text.
2.  **Similarity**: We compute the CLIP score ($s$) to see how well the image and text match.
3.  **Backward Pass**: We calculate the gradients of the loss $\mathcal{L} = -\log(s)$ with respect to the attention weights $G^{(l,h)}_{i,j} = \frac{\partial \mathcal{L}}{\partial A^{(l,h)}_{i,j}}$.
4.  **Aggregation**: We combine the absolute values of these gradients with the attention weights: $|G^{(l,h)}| \odot A^{(l,h)}$.
5.  **Visualization**: Finally, we generate the heatmap $
M = \text{Reshape}\left(\sum_{l,h} |G^{(l,h)}| \odot A^{(l,h)}\right)
$

---
## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/TM-Squared/ML_Reproducibility_Challenge.git
cd ML_Reproducibility_Challenge
pip install -r requirements.txt
```