# ML Reproducibility Challenge: Explainability Methods for CLIP



Ce dépôt explore et compare diverses **méthodes d'explicabilité pour CLIP (Contrastive Language-Image Pre-training)**, en se concentrant sur **Grad-ECLIP** pour rendre les prédictions de CLIP plus interprétables. Ma contribution porte sur l'analyse qualitative des méthodes d'explicabilité et l'implémentation du fine-tuning guidé par Grad-ECLIP.

## 📝 Table des matières
- [Introduction](#-introduction)
- [Méthodes Implémentées](#-méthodes-implémentées)
- [Résultats Qualitatifs](#-résultats-qualitatifs)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Fine-Tuning](#-fine-tuning-à-venir)
- [Contribution](#-contribution)
- [Références](#-références)

## 🌟 Introduction
Ce projet reproduit et étend **Grad-ECLIP** ([Zhao et al., 2024](https://arxiv.org/abs/2502.18816v1)), une méthode gradient-based pour expliquer les prédictions de CLIP via :
- **Cartes de salience visuelles** (régions influentes dans l'image)
- **Explications textuelles** (mots clés dans le prompt)

Mon travail couvre :
1. L'analyse comparative qualitative des méthodes d'explicabilité (Grad-ECLIP vs. Grad-CAM, Rollout, etc.)
2. L'implémentation du fine-tuning utilisant les heatmaps de Grad-ECLIP pour améliorer l'alignement région-texte.

## 🛠 Méthodes Implémentées
### Méthodes d'explicabilité évaluées
| Méthode          | Type               | Spécifique au texte? |
|------------------|--------------------|----------------------|
| Grad-ECLIP       | Gradient-based     | ✅                   |
| Grad-CAM         | Gradient-based     | ✅                   |
| Rollout          | Attention-based    | ❌                   |
| MaskCLIP         | Similarité cosine  | ✅                   |
| CLIP Surgery     | Similarité modifiée| ✅                   |

### Grad-ECLIP (Notre focus)
```python
# Pseudocode de l'explication visuelle
heatmap = sum(
    abs(∂(cosine_similarity)/∂A^(l)) ⊙ A^(l) 
    for l in layers
)
```
