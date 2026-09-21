# Project Progress & Memory State: Robust and Interpretable TSR

**Project Title:** Data-Centric, Representation-Centric, and Hybrid Robustness for Interpretable Traffic Sign Recognition  
**Target Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  
**Last Updated:** September 20, 2026 (Evening Session)  
**Execution Environment:** Windows (Local Code & Dataset Curation) / Google Colab T4 GPU (Deep Learning Training & Evaluation)  
**Repository Root:** `D:\ML_Project`

---

## 1. Quick-Start Memory for Future Sessions

> [!IMPORTANT]
> **READ THIS FIRST BEFORE STARTING ANY NEW SESSION OR AGENT WORK:**
> 1. **Local Machine Hardware:** Local environment has **NO CUDA GPU** (`torch.cuda.is_available() == False`). All heavy training and corruption sweeps **must run on Google Colab (T4 GPU)**. Local commands are strictly for data processing, unit tests, code authoring, and packaging.
> 2. **Track-Leakage Prevention Rule:** GTSRB contains sequential video frames (30 frames per physical sign track). **NEVER apply random splitting** across images. Splits are frozen in `data/master_v1/metadata/` using group-stratification on `group_id = ClassId + TrackId`.
> 3. **Preprocessing Contract:** **NEVER apply naive anisotropic stretch resizing** ($W, H \to 64, 64$). Always use the frozen contract: 10% context-buffered ROI crop ($\alpha = 0.10$), aspect-preserving letterbox ($\gamma = \min(S/w, S/h)$), and centered zero-padding to $64 \times 64$.
> 4. **Primary Metric:** Always evaluate and checkpoint on **Macro-Averaged F1-Score**, not clean top-1 accuracy alone, because GTSRB has severe class imbalance (from 210 to 2,250 samples per class).
> 5. **Core Scientific Architecture (Authoritative Guide: `RESEARCH_GUIDELINES.md`):**
>    - **Central Question:** Is robustness better achieved by increasing training visual diversity (data-centric) or by structuring the learned representation (representation-centric), and are their gains complementary?
>    - **Strategy A (Data-Centric):** Exclusively targeted at feature instability—**AugMix** (consistency regularization via Jensen-Shannon Divergence) and **Frequency Manipulation** (LFC/HFC decomposition & texture debiasing). *We do NOT use generic RandAugment or arbitrary CutMix.*
>    - **Strategy B (Representation-Centric):** Legally grounded **Concept Bottleneck Models (CBM)** predicting physical primitives ($\text{Shape}, \text{Color}, \text{Border}, \text{Symbol}$); evaluated via concept survival ($R_c$) and the diagnostic Oracle-CBM protocol ($C^* \to Y$).
>    - **Strategy C (Hybrid):** Synthesizing AugMix/Frequency processing with structured CBM bottlenecks to test for additive vs redundant robustness.
>    - **7-Experiment Ladder:** Exp 1 (Baseline ResNet-18) $\to$ Exp 2 (AugMix) $\to$ Exp 3 (Frequency) $\to$ Exp 4 (CBM) $\to$ Exp 5 (AugMix + CBM) $\to$ Exp 6 (Frequency + CBM) $\to$ Exp 7 (Full Hybrid).

---

## 2. Completed Milestones & Engineering Chronicle

### Phase 1: Ground-Truth Concept Ontology Grounding
- **Location:** `data/master_v1/concepts/class_concepts.csv`
- **Accomplishments:**
  - Audited all 43 GTSRB classes and mapped them to international legal standards: the **UNECE Vienna Convention on Road Signs and Signals (1968)** and the **German Straßenverkehrs-Ordnung (StVO / VzKat 2017)**.
  - Formulated 5 orthogonal concept dimensions:
    1. `shape`: `circle`, `triangle_up`, `triangle_down`, `octagon`, `diamond`.
    2. `primary_color`: `red`, `blue`, `white`, `yellow`.
    3. `border_color`: `red`, `white`, `blue`, `none`.
    4. `has_symbol`: Boolean flag (`0` or `1`).
    5. `symbol_type`: `text_number`, `arrow`, `vehicle`, `pedestrian_cyclist`, `construction`, `cross`, `pictogram_other`, `roundabout`, `none`.
  - Audited and corrected directional arrow mappings (Classes 33/34 right/left turn, 36/37 straight/left, 38/39 right/left bypass).
  - Verified 100% complete with 0 missing values across all 43 classes.

### Phase 2: Master Dataset Audit & Leakage-Free Stratified Splitting
- **Location:** `build_master_dataset.py`, `data/master_v1/metadata/`
- **Accomplishments:**
  - Complete census of raw GTSRB data: 39,209 training samples and 12,630 test samples (51,838 total).
  - Uncovered the video-track structure: 1,307 unique physical sign tracks in the training set (1,305 with 30 frames, 2 with 29 frames).
  - Proved that standard random image splitting causes severe instance/temporal leakage where near-identical consecutive frames appear in both train and validation sets, giving artificially inflated validation metrics.
  - Implemented deterministic Group-Stratified Partitioning (`random_state=42`) grouping by `ClassId + TrackId`:
    - `train.csv`: 31,378 images (1,046 tracks, 80% per class).
    - `val.csv`: 7,830 images (261 tracks, 20% per class; **0 track overlap with train**).
    - `test.csv`: 12,630 images (official held-out benchmark preserved pristine).
    - `master_metadata.csv`: 51,838 total rows joining raw bounding boxes with concept annotations.
  - Generated cryptographic SHA-256 hashes recorded in `data/master_v1/configs/dataset_version.md`.

### Phase 3: Deterministic Preprocessing Pipeline & PyTorch Dataset
- **Location:** `src/dataset.py`, `data/master_v1/configs/preprocessing.yaml`
- **Accomplishments:**
  - Designed the 4-stage preprocessing contract:
    1. Context-buffered ROI crop with $\alpha = 0.10$ expansion (prevents clipping the outer colored rims of signs).
    2. Aspect-ratio preserving letterbox resizing with bilinear interpolation (preserves circular and triangular geometry, eliminating spurious texture/aspect ratio shortcuts).
    3. Symmetric zero-padding to centered $64 \times 64$ RGB canvas.
    4. Standard transfer-learning normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
  - Implemented high-performance PyTorch `GTSRBDataset` supporting class labels and multi-label concept vectors.
  - Unit tested on real images: output tensor verified at `torch.Size([3, 64, 64])` with zero errors.

### Phase 4: Baseline Architecture with Modern "Bag of Tricks"
- **Location:** `src/models/baseline_cnn.py`
- **Accomplishments:**
  - Built `BaselineResNet18` using `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
  - Classification Head: Replaced 1,000-way linear layer with 43-class head (`Linear(512, 43)`).
  - **Kaiming Normal Initialization:** Initialized head weights with fan-out mode ($W \sim \mathcal{N}(0, \sqrt{2/d_{\text{in}}})$) and zero bias ($b = 0$) to protect pretrained representations from early gradient shocks.
  - **Zero-$\gamma$ BN Initialization:** Initialized the scale parameter $\gamma = 0$ on the final Batch Normalization layer (`bn2`) of every residual block, guaranteeing identity mapping ($y = x$) at epoch 0 for stable signal propagation.
  - **No-Bias-Decay Rule:** Implemented `get_parameter_groups()` partitioning parameters into 2D weight matrices (weight decay $\lambda = 10^{-4}$) and 1D biases / BN affine parameters ($\lambda = 0$) to avoid underfitting.
  - Unit tested forward pass: input `[4, 3, 64, 64]` $\to$ output `[4, 43]` verified.

### Phase 5: Production Baseline Training Pipeline
- **Location:** `train_baseline.py`
- **Accomplishments:**
  - Optimizer: SGD with Nesterov momentum $\mu = 0.9$.
  - Learning Rate Schedule: 3-epoch linear warmup followed by half-period cosine decay down to $\eta_{\text{min}} = 10^{-5}$.
  - Numerical Acceleration: Automatic Mixed Precision (`torch.cuda.amp.autocast`) with `GradScaler`.
  - Model Selection: Validation Macro-Averaged F1-Score computed at every epoch; automatically retains `best_baseline_resnet18.pth`.
  - Exports training history to `training_history.json` and generated training curves.

### Phase 6: Robustness & Uncertainty Calibration Evaluation Suite
- **Location:** `src/corruptions.py`, `evaluate_robustness.py`
- **Accomplishments:**
  - Implemented 15 Hendrycks common corruptions across 4 categories adapted for $64 \times 64$:
    - **Noise:** Gaussian, Shot, Impulse.
    - **Blur:** Defocus, Glass, Motion, Zoom.
    - **Weather:** Snow, Frost, Fog, Brightness.
    - **Digital:** Contrast, Elastic, Pixelate, JPEG compression.
  - Evaluates across 5 severity levels (75 corruption test sets total).
  - Computes comprehensive metrics:
    - Clean Accuracy, Clean Macro-F1, Clean Expected Calibration Error ($ECE$, 10 bins).
    - Mean Corruption Accuracy ($mCA$) and Relative Performance Degradation ($RPD$).
    - Category-level breakdown (Noise, Blur, Weather, Digital).
  - Generates publication-ready summary reports (`baseline_robustness_report.json` and markdown).

### Phase 7: Google Colab Orchestration & Packaging
- **Location:** `create_colab_package.py`, `baseline_experiment_colab.ipynb`
- **Accomplishments:**
  - Built automated packager to create `ML_Project_Data.zip` for easy Google Drive transfer.
  - Created end-to-end Google Colab notebook featuring:
    - GPU verification (T4 check).
    - Environment setup and zip extraction.
    - Manifest and file integrity verification.
    - 15-epoch baseline training run with live progress.
    - Training curves visualization (Loss, Accuracy, Macro-F1, Learning Rate).
    - Full clean test evaluation and 75-point corruption benchmark sweep.
    - Results export and zip download back to local machine.

### Phase 8: Academic Paper Reference & Citation Grounding
- **Location:** `research_content.md` (282 lines)
- **Accomplishments:**
  - Section 3: Preprocessing motivation, failure modes of anisotropic scaling, 4-stage mathematical formulation, $64 \times 64$ Pareto optimality, and track-leakage prevention.
  - Section 4: ResNet-18 architecture, head replacement, Kaiming init, Zero-$\gamma$ BN init, No-Bias-Decay rule, warmup + cosine decay, and Macro-F1 checkpointing.
  - Section 5: Robustness metrics ($mCA$, $RPD$) and Expected Calibration Error ($ECE$).
  - Section 8: 19 complete APA citations grounding every dataset, architecture, optimization trick, and legal standard.
  - Citation Defense Table: 18-row matrix explicitly demarcating externally citable foundations from primary empirical contributions for committee review.

---

## 3. Directory Layout & File Manifest

```text
D:\ML_Project\
├── AGENT_PROJECT_CONTEXT.md          # High-level initial project charter & scope
├── Structured_Research_Proposal.md   # Initial formal proposal & research questions
├── research_content.md               # 282-line publication-ready methodology & citations
├── progress.md                       # This file: Complete memory, status, and instructions
│
├── build_master_dataset.py           # Generates frozen master manifests and SHA-256 hashes
├── dataset_audit.py                  # Audits GTSRB directory structure and bounding boxes
├── create_colab_package.py           # Packages codebase and dataset into zip for Colab
├── train_baseline.py                 # Standalone ResNet-18 training script with warmup/cosine
├── evaluate_robustness.py            # Standalone 15-corruption + ECE evaluation script
├── baseline_experiment_colab.ipynb   # Complete Colab orchestrator notebook
│
├── data/
│   └── master_v1/
│       ├── concepts/
│       │   └── class_concepts.csv    # 43-class Vienna Convention concept ontology (frozen)
│       ├── metadata/
│       │   ├── train.csv             # 31,378 samples (1,046 tracks, 0 leakage)
│       │   ├── val.csv               # 7,830 samples (261 tracks, 0 leakage)
│       │   ├── test.csv              # 12,630 held-out GTSRB benchmark samples
│       │   └── master_metadata.csv   # 51,838 samples with all concepts joined
│       └── configs/
│           ├── preprocessing.yaml    # Frozen preprocessing parameters (64x64, 10% crop)
│           └── dataset_version.md    # Manifest SHA-256 checksums & audit record
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # PyTorch GTSRBDataset with letterbox & context margin
│   ├── corruptions.py                # 15 Hendrycks corruptions adapted for 64x64
│   └── models/
│       ├── __init__.py
│       └── baseline_cnn.py           # ResNet-18 + Kaiming + Zero-gamma BN + No-Bias-Decay
│
├── german/                           # Raw GTSRB dataset (Train/ and Test/)
│   ├── Train.csv
│   ├── Test.csv
│   ├── Train/                        # 43 class subfolders (0 to 42)
│   └── Test/                         # Test image files
│
└── reports/                          # Experiment outputs, metrics, and figures
```

---

## 4. Manifest Checksum Registry (SHA-256)

| Manifest File | Sample Count | Track Count | Checksum (SHA-256) |
| :--- | :---: | :---: | :--- |
| `data/master_v1/metadata/train.csv` | $31,378$ | $1,046$ | `8b3ccc81296014751cc1ebbde1e9085076472f01e6704e420c32f25a31ac3aa4` |
| `data/master_v1/metadata/val.csv` | $7,830$ | $261$ | `5658566473412da8e4b3c2d251ee563ba894366f757d7830cf5abd03dcb2af0d` |
| `data/master_v1/metadata/test.csv` | $12,630$ | — | `0a4029151060c5e748303aa6d88c7c7b53ec7e8fa0f77d5f66707837626ddc3b` |
| `data/master_v1/metadata/master_metadata.csv` | $51,838$ | $1,307$ | `b6daad60ae5c7be0cda9fcfb26ad025592451af40b7c3ab6901c940a62673b40` |

---

## 5. Immediate Next Steps & Controlled Experiment Ladder

Refer to [`RESEARCH_GUIDELINES.md`](file:///d:/ML_Project/RESEARCH_GUIDELINES.md) for the complete scientific justification and formal hypotheses ($H_1$ to $H_5$).

### Step 1: Execute Experiment 1 (Baseline ResNet-18) on Colab
1. Upload `ML_Project_Data.zip` and `baseline_experiment_colab.ipynb` to Google Drive / Colab.
2. Run all cells in `baseline_experiment_colab.ipynb` on a T4 GPU runtime.
3. Download baseline results (`best_baseline_resnet18.pt`, `training_history.json`, `robustness_matrix.csv`, `evaluation_summary.json`) into `reports/baseline_resnet18/`.
4. Establishes the anchor benchmark: Clean Acc, Macro-F1, ECE, $mCA_{\text{base}}$, and $RPD_{\text{base}}$.

### Step 2: Implement Experiment 2 (AugMix CNN)
- Build `src/augmentations/augmix.py` implementing stochastic branch mixing and Jensen-Shannon Divergence (JSD) consistency loss.
- Train ResNet-18 under AugMix using the identical optimizer and seed ($X \to \text{AugMix} \to \text{ResNet-18} \to Y$).
- Benchmark on the 15-corruption test suite; isolate pure data-centric gain: $\Delta mCA_{\text{AugMix}} = mCA_{\text{AugMix}} - mCA_{\text{base}}$.

### Step 3: Implement Experiment 3 (Frequency-Aware CNN)
- Build `src/augmentations/frequency.py` implementing Low-Frequency Component (LFC) and High-Frequency Component (HFC) Fourier decomposition.
- Test whether suppressing high-frequency shortcuts and training on filtered spectra improves out-of-distribution robustness.

### Step 4: Implement Experiment 4 (Pure Concept Bottleneck Model - CBM)
- Build `src/models/cbm.py` mapping $X \to \text{ResNet-18} \to C \to Y$ using the 43-class Vienna Convention ontology.
- Measure:
  - Clean & Corrupted classification accuracy and Macro-F1.
  - Concept-level degradation: $R_c = \frac{\text{F1}_c^{\text{clean}} - \text{F1}_c^{\text{corrupt}}}{\text{F1}_c^{\text{clean}}}$ for Shape, Color, Border, Symbol.
  - **Diagnostic Oracle-CBM Protocol:** Compare $Accuracy(\hat{C} \to Y)$ vs. $Accuracy(C^* \to Y)$ to test perception error vs. vocabulary limits.

### Step 5: Implement Experiments 5, 6, & 7 (Hybrid Synergy & Full System)
- **Exp 5 (AugMix + CBM):** Tests whether data diversity and semantic structure provide additive or redundant robustness ($\Delta mCA \stackrel{?}{=} \Delta mCA_{\text{AugMix}} + \Delta mCA_{\text{CBM}}$).
- **Exp 6 (Frequency + CBM):** Tests whether low-frequency components specifically correspond to geometric concepts (Shape/Border).
- **Exp 7 (Full System):** Unified AugMix + Frequency + CBM + Semantic Constraint Layer.

