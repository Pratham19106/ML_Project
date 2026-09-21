# Research Proposal

## Title

**Data-Centric, Representation-Centric, and Hybrid Robustness for Interpretable Traffic Sign Recognition**

## 1. Abstract

Traffic Sign Recognition (TSR) is a safety-critical perception task for Advanced Driver Assistance Systems (ADAS). Conventional convolutional neural networks (CNNs) can achieve strong accuracy on clean images, but their decisions may be sensitive to blur, weather, lighting, sensor noise, texture changes, and other distribution shifts. They also provide limited visibility into the visual evidence behind a prediction.

This study compares two complementary approaches to improving TSR robustness: **data-centric robustness**, which increases useful visual variation during training, and **representation-centric robustness**, which requires the model to predict explicit, standards-grounded traffic-sign concepts before classifying the sign. A hybrid approach combines both strategies. The central question is whether robustness is better improved by changing the images a model sees during training, changing the representation it is required to learn, or combining both.

The proposed representation-centric model is a Concept Bottleneck Model (CBM) whose concepts are derived from traffic-sign standards and their German implementation: shape, colour profile, border, and symbol category. Models will be evaluated on clean and corrupted test images using classification, concept, calibration, intervention, consistency, and efficiency metrics. The study will use one frozen master dataset, shared preprocessing, splits, and evaluation protocol to ensure fair comparison across all methods.

## 2. Background and Motivation

CNNs often learn shortcuts from textures or incidental visual correlations rather than stable sign properties. Such shortcuts can fail under real-world visual corruption. Two ways to reduce this weakness are:

1. **Data-centric robustness:** expose the network to diverse but label-preserving visual conditions during training.
2. **Representation-centric robustness:** constrain the network to reason through meaningful sign characteristics that should remain relevant across visual changes.

The approaches target the same underlying problem—reliance on unstable features—but do so at different stages of learning. Their combination may provide stronger robustness and clearer failure diagnosis than either approach alone.

## 3. Problem Statement

It is unclear whether CNN robustness in traffic-sign recognition is improved more effectively by training-time visual transformations or by an explicit semantic concept bottleneck. It is also unclear whether these strategies are complementary and whether the resulting approach can remain computationally efficient relative to Vision Transformer (ViT) baselines.

## 4. Aim and Objectives

### Aim

To develop and evaluate an efficient, interpretable CNN-based TSR pipeline that is robust to visual corruptions without training directly on the final corruption benchmark.

### Objectives

1. Establish a standard CNN baseline for TSR.
2. Measure the effect of data-centric methods, including AugMix, frequency-aware transformations, and texture/shape-bias interventions.
3. Build sequential and joint CBMs that predict a standards-grounded concept vector before the final sign class.
4. Measure whether concept representations improve corruption robustness, explainability, and correctability.
5. Develop a hybrid model that combines the most effective data-centric method(s) with the CBM.
6. Compare all approaches fairly on clean accuracy, corrupted-image robustness, interpretability, calibration, and computational cost.

## 5. Research Questions and Hypotheses

### Research Questions

**RQ1.** Do data-centric transformations improve CNN robustness to unseen traffic-sign corruptions while preserving clean-image accuracy?

**RQ2.** Does a standards-grounded CBM improve robustness and failure interpretability compared with a conventional black-box CNN?

**RQ3.** Are data-centric and representation-centric robustness strategies complementary?

**RQ4.** Can a robust and interpretable CNN remain competitive with a ViT baseline at lower computational cost?

### Working Hypotheses

- **H1:** Data-centric training improves corrupted-image performance relative to the standard CNN baseline.
- **H2:** The CBM improves concept-level reliability, intervention capability, and diagnostic transparency; its robustness effect will be measured rather than assumed.
- **H3:** The hybrid model produces better robustness than either the best data-centric model or the CBM alone.
- **H4:** The hybrid CNN has lower parameter count, FLOPs, and inference latency than the ViT baseline while retaining competitive robustness.

## 6. Scope and Dataset

### Primary Dataset

The main dataset will be the **German Traffic Sign Recognition Benchmark (GTSRB)**. It supplies class labels and is suitable for constructing additional concept labels.

### Concept Grounding

The concept vocabulary will be **derived from** the Vienna Convention on Road Signs and Signals and mapped carefully to the relevant German traffic-sign implementation. The project will not claim that every ML label is itself legally mandated; instead, it will document a reproducible machine-readable ontology grounded in these standards.

### Evaluation Data

- Clean held-out GTSRB test images.
- A held-out traffic-sign corruption benchmark, such as GTSRB-C, or a reproducible equivalent corruption suite.
- Corruption types may include weather effects, fog, motion blur, noise, lighting changes, and digital distortions, evaluated across severity levels.

The corruption benchmark is used for **evaluation**, not as stored training data. AugMix and other training transformations are generated on the fly and must not reproduce the test set or leak test samples.

## 7. Standardized Master Dataset

One centrally managed master dataset will be the single source of truth for every experiment.

### 7.1 Central Data Preparation

1. Verify files, class labels, duplicates, unreadable images, and major resolution anomalies.
2. Convert images consistently to RGB.
3. Freeze train, validation, and test image IDs in version-controlled CSV files.
4. Use one documented resize/pad/crop policy that preserves aspect ratio before producing the fixed model input size.
5. Store class labels, split assignments, and concept labels in metadata rather than maintaining multiple altered datasets.
6. Version the data, ontology, annotations, and preprocessing configuration.

### 7.2 Recommended Metadata

```text
image_id, image_path, class_id, class_name, split,
shape, colour_profile, border_type, symbol_category,
annotation_source, annotator, annotation_confidence
```

### 7.3 Concept Ontology

The final categories will be frozen before annotation. An initial controlled vocabulary is:

| Concept | Initial values |
| --- | --- |
| Shape | circle, triangle, rectangle, octagon, inverted triangle, other |
| Colour profile | red, blue, yellow, white, black, green, mixed/other |
| Border | present/absent, with optional border colour |
| Symbol category | numeral, directional arrow, pedestrian, vehicle, stop, warning, restriction, text, other |

An annotation guideline must define each category, difficult cases, source mapping, and quality-control procedure. Annotation provenance and confidence must be retained.

### 7.4 Transformation Policy

| Dataset stage | Allowed operations |
| --- | --- |
| Master data | Cleaning, RGB conversion, fixed resize/pad/crop, frozen splits, labels and concept annotations |
| Training only | Method-specific augmentation, AugMix, frequency processing, and texture/shape interventions |
| Validation | Fixed preprocessing only |
| Clean test | Fixed preprocessing only |
| Corruption test | Fixed preprocessing plus the predefined evaluation corruption only |

Semantics-changing operations, especially horizontal flips or unconstrained rotations, must be excluded unless their validity for each affected sign class is explicitly established.

## 8. Methods

### 8.1 Common Backbone and Controls

All CNN variants will use a shared baseline backbone (for example, ResNet-50) and the same core training protocol unless a method explicitly requires a justified change. The protocol includes fixed splits, input resolution, optimizer, learning-rate schedule, epochs, random seeds, and reporting format.

### 8.2 Baseline: Black-Box CNN

The baseline is trained end to end:

```text
Image -> CNN backbone -> Sign class
```

It establishes clean accuracy, robustness, calibration, compute, and error-analysis reference values.

### 8.3 Data-Centric Robustness Track

This track changes the training inputs while retaining the same class-learning task.

- **Standard label-preserving augmentation:** only transformations verified as valid for traffic signs.
- **AugMix:** diverse stochastic mixtures of augmentations applied online during training.
- **Frequency-aware methods:** low- and high-frequency decomposition or perturbation to test reliance on frequency components.
- **Texture/shape-bias interventions:** training mechanisms that reduce inappropriate texture reliance and encourage shape-relevant evidence.

Each component is first evaluated independently and then through justified combinations. Augmented images are not saved as a separate master dataset.

### 8.4 Representation-Centric Track: Concept Bottleneck Model

The CBM separates visual concept prediction from class decision-making:

```text
Image -> CNN backbone -> Concepts (shape, colour, border, symbol) -> Sign class
```

Two variants will be evaluated:

- **Sequential CBM:** an image-to-concept model followed by a concept-to-class classifier.
- **Joint CBM:** one backbone trained with both concept and class losses:

$$
\mathcal{L}=\mathcal{L}_{task}(y,\hat{y}) + \lambda\sum_i \mathcal{L}_{concept}(c_i,\hat{c}_i)
$$

The class head may be a linear classifier, decision tree, or another explicitly documented constrained mapper. A standards-derived consistency layer will reject or penalize implausible concept/class combinations where appropriate.

### 8.5 Oracle CBM

The concept-to-class model will also be tested using ground-truth concepts. This separates errors caused by visual concept prediction from errors caused by an insufficient concept vocabulary or concept-to-class mapping.

### 8.6 Hybrid Track

The hybrid model combines the CBM with the best-performing data-centric intervention(s):

```text
Training-time augmentation/frequency/texture intervention
                      -> CNN backbone
                      -> standards-grounded concepts
                      -> consistency layer
                      -> sign class
```

The hybrid is evaluated only after the individual tracks establish which components are useful. Its purpose is to test complementarity, not to combine every technique indiscriminately.

### 8.7 ViT Reference Baseline

A comparable ViT baseline will serve as an efficiency and robustness reference. It is an auxiliary comparison, not the main architectural contribution.

## 9. Evaluation Framework

| Evaluation dimension | Metrics / analysis |
| --- | --- |
| Classification | Clean accuracy, macro-F1, per-class accuracy |
| Robustness | Accuracy/F1 by corruption type and severity; relative performance drop |
| Concepts | Per-concept accuracy and macro-F1 under clean and corrupted conditions |
| Calibration | Expected Calibration Error (ECE) |
| Intervention | Accuracy before/after correcting one or all incorrect concepts; intervention recovery rate |
| Consistency | Rate of invalid concept combinations or incompatible concept-to-class outputs |
| Explainability | Concept localisation/saliency analysis compared with Grad-CAM, where annotations permit |
| Efficiency | Parameters, FLOPs, training cost, and inference latency |
| Reliability | Multiple seeds, mean and standard deviation, and appropriate paired statistical tests |

The intervention recovery rate may be reported as:

$$
IR=\frac{Acc_{after\ intervention}-Acc_{before\ intervention}}
{1-Acc_{before\ intervention}}
$$

This measures the share of previously incorrect predictions recovered after concept correction.

## 10. Experimental Sequence

1. Freeze the master dataset, ontology, preprocessing, splits, and evaluation protocol.
2. Train and evaluate the black-box CNN baseline.
3. Run individual data-centric ablations and select promising combinations.
4. Train sequential, joint, and oracle CBMs.
5. Evaluate concept robustness, consistency, and interventions.
6. Build and evaluate the hybrid using selected components.
7. Compare the strongest CNN variants with the ViT reference baseline.
8. Conduct error, cost, calibration, and statistical analyses under the same protocol.

## 11. Team Work Allocation

| Role | Primary responsibility | Scientific question |
| --- | --- | --- |
| Dataset and standardisation lead | Master dataset, data versioning, splits, preprocessing, concept ontology and annotations | Are all comparisons based on equivalent data? |
| Data-centric robustness lead | AugMix, frequency methods, texture/shape interventions, ablations | Does changing training data improve robustness? |
| Representation-centric robustness lead | Sequential/joint/oracle CBM, constraints, interventions | Does semantic representation improve robustness and correctability? |
| Hybrid and comparison lead | Data-centric + CBM integration, ViT comparison, efficiency and overall statistical analysis | Are the two strategies complementary? |

Where the team has three members, the dataset-standardisation lead may also own one modelling track. All data and modules are shared project resources after their validated release.

## 12. Expected Contributions

1. A versioned traffic-sign dataset protocol with standards-grounded concept labels and annotation provenance.
2. A controlled comparison of data-centric versus representation-centric robustness strategies for TSR.
3. An intervention-capable, constraint-aware CBM for transparent sign recognition.
4. Evidence on whether semantic bottlenecks, augmentations, or their combination improve robustness under unseen corruptions.
5. A clean-accuracy, robustness, interpretability, calibration, and compute comparison against CNN and ViT baselines.

## 13. Limitations and Risk Controls

- The concept vocabulary may omit class-specific details; the oracle CBM will test whether it is sufficient.
- Convention-based characteristics must be distinguished from national implementation and the project ontology.
- Some augmentations can change sign meaning; use a class-aware transformation allowlist.
- Small signs may lose detail during resizing; document and validate the selected aspect-ratio-preserving policy.
- Robustness gains must be measured on held-out corruptions to avoid claims based only on training augmentation.
- Interpretability claims require concept accuracy and consistency evidence, not only saliency visualisations.

## 14. Proposed Contribution Statement

> This work evaluates whether traffic-sign robustness is better achieved by changing the visual variation seen during CNN training, constraining the learned representation through standards-grounded concepts, or combining both. It introduces an intervention-capable, constraint-aware concept bottleneck pipeline and evaluates robustness, transparency, and efficiency under a unified experimental protocol.
