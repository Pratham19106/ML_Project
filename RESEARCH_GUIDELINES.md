# Research Guidelines & Scientific Master Blueprint

**Project Title:** Data-Centric, Representation-Centric, and Hybrid Robustness for Interpretable Traffic Sign Recognition  
**Domain:** German Traffic Sign Recognition Benchmark (GTSRB) under Natural Corruptions (GTSRB-C)  
**Status:** Authoritative Architectural & Experimental Guidelines  
**Target Output:** Top-Tier Conference Paper & Master's Thesis Defense  

---

## 1. The Central Scientific Narrative

### 1.1 The Core Scientific Question
> **When attempting to make convolutional neural networks robust to visual distribution shifts, is it more effective to teach the model to see more variations of the world (data-centric), or to teach it to represent the world using stable semantic characteristics (representation-centric), and do both approaches provide complementary or redundant gains?**

### 1.2 The Common Root Problem: Feature Instability
Conventional CNNs achieve high clean test accuracy by exploiting whatever visual correlations are easiest to learn—frequently shortcut features such as local surface textures, high-frequency sensor noise, background scenery, and camera distance cues. 

Under natural environmental distribution shifts (blur, fog, frost, rain, sensor noise, low contrast), these spurious shortcuts collapse.

| Dimension | Strategy A: Data-Centric Robustness | Strategy B: Representation-Centric Robustness |
| :--- | :--- | :--- |
| **Main Problem** | CNN fails under distribution shift / corruptions | CNN makes opaque decisions using fragile shortcut features |
| **Primary Mechanism**| Invariant visual representation via input manipulation | Explicit semantic structure via Concept Bottleneck |
| **Interventions** | **AugMix**, **Frequency manipulation (LFC/HFC)**, **Texture debiasing** | **Legally grounded concepts** (Shape, Color, Border, Symbol) |
| **Representation** | Unconstrained continuous latent space ($\mathbb{R}^{512}$) | Interpretable semantic bottleneck ($\hat{C} \in [0, 1]^K$) |
| **Interpretability**| Black-box latent features | Transparent, auditable concept predictions |
| **Evaluation Focus**| Clean vs. Corrupted performance ($mCA, RPD$) | Concept F1, class accuracy, concept robustness ($R_c$), Oracle-CBM |
| **Core Question** | *How should we train the network?* | *What should the network represent?* |

---

## 2. The Three Strategic Paradigms

### Strategy A — Data-Centric Robustness
Exposes the CNN backbone to controlled visual diversity during optimization to force the network to learn representations that are invariant to domain perturbations:
```text
Raw Image (X)
      │
      ▼
Data-Centric Transformations:
  ├── 1. AugMix (Stochastic branch mixing + Jensen-Shannon Consistency)
  ├── 2. Frequency Decomposition (Low-pass LFC vs High-pass HFC filtering)
  └── 3. Texture Debiasing (Stylized / edge-enhanced transformations)
      │
      ▼
CNN Backbone (ResNet-18)
      │
      ▼
Latent Feature Vector (h in R^512)
      │
      ▼
Linear Head Projection (W_fc)
      │
      ▼
Sign Class (Y in {0, ..., 42})
```
*Note: We do **NOT** use arbitrary generic augmentations (such as generic RandAugment or unprincipled CutMix). Every intervention must be tied to a specific physical degradation hypothesis (frequency, texture, or consistency).*

---

### Strategy B — Representation-Centric Robustness
Constrains the model's hypothesis space by forcing it to predict standards-grounded physical primitives before arriving at a final traffic sign classification:
```text
Raw Image (X)
      │
      ▼
Frozen Preprocessing Contract (64x64 Letterbox)
      │
      ▼
CNN Backbone (ResNet-18)
      │
      ▼
Concept Bottleneck Layer (c_hat in [0, 1]^C):
  ├── Shape:          Circle, Triangle-Up, Triangle-Down, Octagon, Diamond
  ├── Primary Color:  Red, Blue, White, Yellow
  ├── Border Color:   Red, White, Blue, None
  ├── Has Symbol:     Binary (0 or 1)
  └── Symbol Type:    Number/Text, Arrow, Vehicle, Pedestrian, Works, etc.
      │
      ▼
Predictor Function (g: C -> Y, Linear or Constrained Rule-Base)
      │
      ▼
Sign Class (Y in {0, ..., 42})
```
*Scientific Terminology Note: We do **not** claim upfront that CBM is inherently robust. We investigate: **"Concept Bottleneck Models as a representation-level robustness mechanism."***

---

### Strategy C — Hybrid Robustness
Synthesizes both input-distribution diversity and semantic structural constraints into a unified model:
```text
                                  Raw Image (X)
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                  AugMix / JSD             Frequency / Texture Filtering
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                               CNN Backbone (ResNet)
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                  Latent Features (h)          Concept Layer (c_hat)
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                              Structured Classifier
                                        │
                                        ▼
                                 Sign Class (Y)
```
*Central Question for Strategy C: Do data-centric and representation-centric methods provide **complementary, additive, or redundant** robustness gains?*

---

## 3. Formal Scientific Hypotheses

* **Hypothesis 1 ($H_1$ — Data-Centric Robustness):**  
  Augmentation-based consistency training (specifically AugMix and frequency filtering) significantly improves CNN corruption robustness over standard ERM training ($mCA_{\text{AugMix}} > mCA_{\text{standard}}$).
* **Hypothesis 2 ($H_2$ — Representation-Centric Robustness):**  
  Explicit semantic concept bottlenecks improve robustness to visual corruptions relative to an unconstrained latent representation ($mCA_{\text{CBM}} > mCA_{\text{standard}}$).
* **Hypothesis 3 ($H_3$ — Complementarity):**  
  Combining data-centric visual diversity with representation-centric semantic bottlenecks produces complementary robustness improvements exceeding either individual intervention ($mCA_{\text{Hybrid}} > \max(mCA_{\text{AugMix}}, mCA_{\text{CBM}})$).
* **Hypothesis 4 ($H_4$ — Semantic Stability):**  
  Robust classification under corruption is causally associated with the preservation of high-level geometric concepts (Shape, Border) over fine-grained interior details (Symbols).
* **Hypothesis 5 ($H_5$ — Computational Efficiency):**  
  The proposed hybrid robust CNN achieves competitive corruption robustness with a fraction of the computational and memory footprint required by Vision Transformer (ViT) architectures.

---

## 4. Factorial Experimental Design Matrix

To avoid ad-hoc experimentation, the research follows a strict **$3 \times 4 \times 15$ Factorial Design**:

$$\text{Training Strategy (Axis 1)} \times \text{Representation Strategy (Axis 2)} \times \text{Corruption Benchmark (Axis 3)}$$

### Axis 1: Training Strategy (Input Space)
1. **Standard ERM:** Clean training with standard data standardization only.
2. **AugMix:** Multi-branch stochastic augmentation with Jensen-Shannon Divergence consistency loss.
3. **Frequency-Aware:** Low-frequency component (LFC) vs. High-frequency component (HFC) spectral filtering.
4. **Combined Pipeline:** Unified AugMix + Frequency/Texture regularization.

### Axis 2: Representation Strategy (Feature Space)
1. **Black-Box CNN:** Unconstrained linear projection from 512-d latent space ($\mathbb{R}^{512} \to \mathbb{R}^{43}$).
2. **Sequential CBM:** Two-stage training: $X \to C$ (concept model), followed by frozen $C \to Y$ (task head).
3. **Joint CBM:** Multi-task end-to-end optimization ($\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{cbm}} \mathcal{L}_{\text{concept}}$).
4. **Constraint-Aware CBM:** CBM regularized with legal/semantic logic constraints (Vienna Convention ontology).

### Axis 3: Evaluation Benchmark
- **Clean Held-Out Test Set:** 12,630 GTSRB benchmark samples.
- **15 Common Corruptions across 5 Severities:**
  - **Noise:** Gaussian Noise, Shot Noise, Impulse Noise.
  - **Blur:** Defocus Blur, Glass Blur, Motion Blur, Zoom Blur.
  - **Weather:** Snow, Frost, Fog, Brightness.
  - **Digital:** Contrast, Elastic Transform, Pixelation, JPEG Compression.

---

## 5. The 7 Controlled Experiments (Step-by-Step)

We follow a strict additive ladder. **We do NOT jump directly to the full system**, because without individual baselines, ablation is impossible.

```text
[Exp 1: Standard CNN] ──────> [Exp 2: AugMix CNN] ──────> [Exp 3: Frequency CNN]
          │
          ▼
   [Exp 4: CBM] ────────────> [Exp 5: AugMix + CBM] ───> [Exp 6: Frequency + CBM]
          │
          ▼
   [Exp 7: Full System (Aug + Freq + CBM + Constraints)]
```

### Experiment 1: Standard CNN Baseline *(Current Milestone)*
$$X \to \text{ResNet-18} \to Y$$
- **Objective:** Establish the clean accuracy, macro-F1, ECE, and baseline corruption robustness matrix ($mCA_{\text{base}}$).
- **Status:** Architecture built, training loop tested, notebook prepared (`baseline_experiment_colab.ipynb`).

### Experiment 2: AugMix CNN
$$\text{AugMix}(X) \to \text{ResNet-18} \to Y$$
- **Objective:** Measure pure data-centric gain from stochastic mixing and Jensen-Shannon consistency.
- **Key Metric:** $\Delta mCA = mCA_{\text{AugMix}} - mCA_{\text{base}}$.

### Experiment 3: Frequency-Aware CNN
$$\text{LFC/HFC}(X) \to \text{ResNet-18} \to Y$$
- **Objective:** Determine whether suppressing high-frequency shortcuts and reinforcing low/mid frequencies improves robustness.

### Experiment 4: Pure Concept Bottleneck Model (CBM)
$$X \to \text{ResNet-18} \to C \to Y$$
- **Objective:** Isolate pure representation-centric robustness. Does forcing the network to predict Shape, Color, Border, and Symbol improve robustness without any data augmentation?
- **Diagnostic:** Evaluate concept F1 under corruptions and compare against Exp 1.

### Experiment 5: AugMix + CBM
$$\text{AugMix}(X) \to \text{ResNet-18} \to C \to Y$$
- **Objective:** Test if data diversity and semantic concepts provide additive or redundant benefits.
- **Scientific Test:** Does $\Delta mCA(\text{Exp 5}) \approx \Delta mCA(\text{Exp 2}) + \Delta mCA(\text{Exp 4})$?

### Experiment 6: Frequency + CBM
$$\text{LFC/HFC}(X) \to \text{ResNet-18} \to C \to Y$$
- **Objective:** Test whether low-frequency components specifically encode geometric/semantic concepts (Shape/Color) rather than local texture.

### Experiment 7: Full Hybrid System
$$X \to \text{AugMix} + \text{Freq} \to \text{CNN} \to \text{CBM} \to \text{Constraints} \to Y$$
- **Objective:** Evaluate the complete unified framework against all prior single-factor interventions.

---

## 6. Novel Evaluation Paradigms

### 6.1 Concept-Level Robustness Formulation
A major scientific contribution of this framework is the ability to explain *why* classification degrades under corruption by measuring the degradation of individual semantic concepts:

$$R_c = \frac{\text{F1}_c^{\text{clean}} - \text{F1}_c^{\text{corrupt}}}{\text{F1}_c^{\text{clean}}}$$

for each concept $c \in \{\text{Shape}, \text{Color}, \text{Border}, \text{Symbol}\}$.

**Hypothesized Degradation Hierarchy:**
$$\text{Stability}(\text{Shape}) > \text{Stability}(\text{Border}) > \text{Stability}(\text{Color}) > \text{Stability}(\text{Symbol})$$

Under severe blur or fog, fine symbols (e.g., distinguishing "30" from "80") degrade rapidly, whereas global geometry (Circle vs. Octagon) remains intact. Black-box CNNs collapse when symbols blur; CBM preserves the category and warns of symbol uncertainty.

### 6.2 The Diagnostic Oracle-CBM Protocol
To avoid the dangerous assumption that "interpretable is automatically robust," we enforce the **Oracle-CBM diagnostic test**:

$$\text{Evaluate: } \text{Accuracy}(\hat{C} \to Y) \quad \text{vs.} \quad \text{Accuracy}(C^* \to Y)$$
where $\hat{C}$ is the predicted concept vector and $C^*$ is the ground-truth concept vector.

* **Scenario 1 (Perception Failure):** $Accuracy(C^* \to Y)$ remains $\approx 100\%$ under corruption, but $Accuracy(\hat{C} \to Y)$ drops.  
  $\implies$ The classifier reasoning is sound; the visual concept perception layer failed under corruption.
* **Scenario 2 (Vocabulary Insufficiency):** $Accuracy(C^* \to Y)$ drops significantly.  
  $\implies$ The concept ontology itself is incomplete or ambiguous for distinguishing certain classes.
* **Scenario 3 (Error Amplification):** Predicted CBM accuracy is significantly worse than Black-Box CNN accuracy under corruption.  
  $\implies$ Errors in intermediate concept predictions cascade catastrophically into the final decision.

This gives our paper deep scientific diagnostic power that standard benchmark papers completely lack.

---

## 7. Strict Implementation Guidelines & Constraints

1. **NO RandAugment or Unprincipled Augmentations:**  
   Do not introduce random augmentation sweeps (RandAugment, TrivialAugment, CutMix, MixUp) unless explicitly tied to a verified hypothesis (e.g., AugMix for consistency, spectral filtering for frequency analysis).
2. **Preserve the Frozen Preprocessing Contract:**  
   All experiments must use the identical $64 \times 64$ letterboxed images with 10% context margin. Never use anisotropic stretch resizing.
3. **Preserve the Frozen Group Splits:**  
   All training runs use `train.csv` (31,378 samples); all model selection uses `val.csv` (7,830 samples, 0 track leakage); all final benchmarks use `test.csv` (12,630 samples).
4. **Primary Metric is Macro-F1:**  
   Every model is selected based on Validation Macro-F1, never clean accuracy alone.
5. **Compute Budget:**  
   ResNet-18 is the primary engine throughout the paper to fit within free Google Colab T4 GPU limits. ResNet-50 may only be introduced for a final verification table if compute permits.

