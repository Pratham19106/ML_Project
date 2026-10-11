
### 7.3 Experiment 6: Frequency-Aware CBM & Spectral-Semantic Alignment

#### 1. Formulation: Fixing the AugMix Interference
Experiment 5 revealed that AugMix destroys CBMs because its color distortions conflict with semantic color targets. To explicitly resolve this **Representation-Augmentation Interference**, we combined the Concept Bottleneck Model with our previously validated **Low-Frequency Component (LFC)** filtering. LFC attenuates destructive high-frequency noise while strictly preserving color and macroscopic geometry.

#### 2. Empirical Benchmark Outcomes
- **Clean Accuracy:** **97.45%**
- **Clean Macro-F1:** **96.60%**
- **Mean Corruption Accuracy ($mCA$, 75 points):** **84.21%**
- **Relative Performance Drop ($RPD$):** **13.59%** *(Lowest relative degradation in the project)*

#### 3. Scientific Validation
Experiment 6 validates the hypothesis that semantic architectures require strictly orthogonal, color-preserving augmentations. While AugMix violently collapsed the CBM to $81.63\%$ $mCA$ ($16.37\%$ RPD), Frequency Regularization cleanly preserved the semantic bottleneck. It achieved a near-identical raw accuracy to the pure CBM ($84.21\%$ vs $84.29\%$), but yielded the absolute best relative robustness ($13.59\%$ RPD) across all 6 experiments. This solidifies the principle of **Spectral-Semantic Alignment**: interpretable architectures should be regularized in the frequency domain, not the spatial color domain.
