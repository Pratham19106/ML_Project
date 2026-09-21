# Collaborator & AI Agent Onboarding Guide

**Project Title:** Data-Centric, Representation-Centric, and Hybrid Robustness for Interpretable Traffic Sign Recognition  
**Target Dataset:** German Traffic Sign Recognition Benchmark (GTSRB) under Common Corruptions (GTSRB-C)  
**Primary Compute:** Local CPU (Code, Manifests, Unit Tests) & Google Colab T4 GPU (Deep Learning Training & Evaluation)  
**Authoritative Core Guides:**
* [`RESEARCH_GUIDELINES.md`](file:///d:/ML_Project/RESEARCH_GUIDELINES.md) — Scientific master plan, formal hypotheses ($H_1$–$H_5$), 7-experiment ladder.
* [`research_content.md`](file:///d:/ML_Project/research_content.md) — Mathematical formulations, formal APA citations, and citation defense matrix.
* [`progress.md`](file:///d:/ML_Project/progress.md) — Completed milestones, file manifests, SHA-256 hashes, and execution roadmap.

---

## 1. Ready-to-Copy System Prompt for AI Agents

When a fellow team member sets up an AI assistant (Cursor, Claude, ChatGPT, Gemini, Copilot, Antigravity), **copy and paste the block below** directly into the agent's system prompt or introductory prompt:

```text
You are an expert AI research collaborator working on an academic machine learning research project:
"Data-Centric, Representation-Centric, and Hybrid Robustness for Interpretable Traffic Sign Recognition" on GTSRB.

Before generating code or suggesting experiments, you MUST adhere to these strict invariants:
1. SCIENTIFIC HYPOTHESIS: We investigate whether corruption robustness is better achieved by increasing visual diversity (Data-Centric: AugMix, Frequency/Texture filtering) or by enforcing explicit semantic structure (Representation-Centric: Concept Bottleneck Models grounded in the Vienna Convention), or if both provide complementary gains (Hybrid).
2. NO GENERIC AUGMENTATIONS: We do NOT use generic RandAugment, TrivialAugment, or unprincipled CutMix. Every data-centric method must be tied to a specific physical feature-instability mechanism (AugMix for invariant consistency; Fourier LFC/HFC decomposition for high-frequency shortcut suppression).
3. HARDWARE REALITY: The local environment has NO CUDA GPU. All heavy training and corruption sweeps run on Google Colab (T4 GPU). Local scripts are strictly for data processing, lightweight unit tests, and packaging.
4. ZERO TRACK-LEAKAGE: GTSRB contains sequential 30-frame video tracks. NEVER apply random splitting across images. Splits are frozen in data/master_v1/metadata/ grouped by `group_id = ClassId + TrackId` (train: 31,378, val: 7,830, test: 12,630).
5. FROZEN PREPROCESSING CONTRACT: NEVER apply anisotropic stretch resizing. Every image must undergo: 10% context-buffered ROI crop (alpha = 0.10), aspect-preserving letterbox, symmetric zero-padding to 64x64 canvas, and ImageNet standardization.
6. EVALUATION METRIC: Always evaluate and checkpoint on Macro-Averaged F1-Score (due to severe class imbalance from 210 to 2,250 samples), alongside clean accuracy, ECE, mCA (Mean Corruption Accuracy), and RPD (Relative Performance Degradation).
7. REPOSITORY REFERENCES: Read RESEARCH_GUIDELINES.md (scientific roadmap), research_content.md (formal math & citations), and progress.md (current code state).
```

---

## 2. Project Architecture at a Glance

```text
D:\ML_Project\
├── RESEARCH_GUIDELINES.md          # Complete scientific framework, hypotheses H1-H5, 7-experiment ladder
├── research_content.md             # Formal paper draft text, LaTeX equations, APA citations, defense table
├── progress.md                     # Current milestone state, SHA-256 hashes, and execution roadmap
├── COLLABORATOR_AND_AGENT_GUIDE.md # This guide: Agent prompt, role orientations, code conventions
│
├── data/
│   └── master_v1/
│       ├── concepts/
│       │   └── class_concepts.csv  # 43-class Vienna Convention ontology (Shape, Color, Border, Symbol)
│       ├── metadata/
│       │   ├── train.csv           # 31,378 samples (1,046 tracks, 0 leakage)
│       │   ├── val.csv             # 7,830 samples (261 tracks, 0 leakage)
│       │   ├── test.csv            # 12,630 held-out official benchmark samples
│       │   └── master_metadata.csv # 51,838 samples with joined concept attributes
│       └── configs/
│           ├── preprocessing.yaml  # Frozen 64x64 letterbox contract specifications
│           └── dataset_version.md  # Cryptographic SHA-256 manifest audit record
│
├── src/
│   ├── dataset.py                  # PyTorch GTSRBDataset implementing letterbox & context margin
│   ├── corruptions.py              # 15 Hendrycks corruptions adapted for 64x64 across 5 severities
│   ├── models/
│   │   └── baseline_cnn.py         # ResNet-18 + Kaiming Head + Zero-gamma BN + No-Bias-Decay rule
│   └── augmentations/              # (Upcoming Phase 2: AugMix, Frequency Fourier decomposition)
│
├── train_baseline.py               # Standalone training script (SGD, 3-ep Warmup, Cosine Anneal, AMP)
├── evaluate_robustness.py          # Standalone benchmark script (Clean Test, 75 Corruptions, ECE)
├── create_colab_package.py         # Automated packager into ML_Project_Data.zip for Colab
├── baseline_experiment_colab.ipynb # End-to-end Google Colab orchestrator notebook
└── reports/                        # Saved checkpoints, metrics JSONs, and CSV robustness matrices
```

---

## 3. Task-Specific Orientation for Team Members & Agents

If a researcher or agent is assigned a specific component of the research, consult the corresponding section below:

### Track 1: Data Standardization & Preprocessing Verification
* **Primary Files:** [`src/dataset.py`](file:///d:/ML_Project/src/dataset.py), [`data/master_v1/configs/preprocessing.yaml`](file:///d:/ML_Project/data/master_v1/configs/preprocessing.yaml).
* **Key Concept:** Why naive resizing fails. Stretching a $28 \times 45$ sign to $64 \times 64$ turns circles into ellipses and alters triangle angles, which invalidates the geometric concepts needed by the Concept Bottleneck Model.
* **The 4-Stage Contract:**
  1. $\alpha = 0.10$ Context expansion ($w' = w_{\text{roi}} + 2\Delta x$, clamped to raw image bounds).
  2. Proportional letterbox resizing with bilinear interpolation ($\gamma = \min(64/w_{\text{crop}}, 64/h_{\text{crop}})$).
  3. Symmetric zero-padding to place the rescaled sign at the center of an empty $64 \times 64$ black canvas.
  4. Standard transfer-learning normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
* **Verification Command:**
  ```bash
  python -c "from src.dataset import GTSRBDataset; ds = GTSRBDataset('data/master_v1/metadata/train.csv', 'german'); print(ds[0]['image'].shape)"
  # Output MUST be: torch.Size([3, 64, 64])
  ```

---

### Track 2: Data-Centric Robustness (AugMix & Frequency Processing)
* **Goal:** Implement **Experiment 2 (AugMix)** and **Experiment 3 (Frequency-Aware CNN)**.
* **Key Constraints:**
  - **Do NOT introduce generic RandAugment or unprincipled CutMix.**
  - **AugMix Implementation:** Must apply stochastic augmentation chains combined via Dirichlet weights, paired with **Jensen-Shannon Divergence (JSD)** consistency loss between clean and augmented forward passes.
  - **Frequency Processing:** Must implement 2D Fast Fourier Transform (FFT) filters to decouple Low-Frequency Components (LFC, structural geometry) from High-Frequency Components (HFC, fine edges and sensor noise).
  - Training schedule must remain **strictly identical** to the baseline (SGD, Nesterov 0.9, 3-epoch warmup, cosine decay, seed 42) to isolate data-centric causality.

---

### Track 3: Representation-Centric CBM (Concept Bottleneck Models)
* **Primary Files:** [`data/master_v1/concepts/class_concepts.csv`](file:///d:/ML_Project/data/master_v1/concepts/class_concepts.csv), upcoming `src/models/cbm.py`.
* **The Ontology:** Grounded in the **UNECE Vienna Convention (1968)** and German **StVO/VzKat (2017)**:
  - `shape`: 5 classes (`circle`, `triangle_up`, `triangle_down`, `octagon`, `diamond`).
  - `primary_color`: 4 classes (`red`, `blue`, `white`, `yellow`).
  - `border_color`: 4 classes (`red`, `white`, `blue`, `none`).
  - `has_symbol`: Binary (`0` or `1`).
  - `symbol_type`: 9 classes (`text_number`, `arrow`, `vehicle`, `pedestrian_cyclist`, `construction`, `cross`, `pictogram_other`, `roundabout`, `none`).
* **Model Design:**
  - Backbone: ResNet-18 yielding feature vector $h \in \mathbb{R}^{512}$.
  - Concept Head: $h \to \hat{C} \in [0, 1]^K$ (multi-label concept logits with BCE loss).
  - Classification Head: $\hat{C} \to \hat{Y} \in \mathbb{R}^{43}$ (predicts traffic sign class strictly from concepts).
* **Two Critical Evaluation Protocols (Must Implement):**
  1. **Concept Degradation Metric ($R_c$):**
     $$R_c = \frac{\text{F1}_c^{\text{clean}} - \text{F1}_c^{\text{corrupt}}}{\text{F1}_c^{\text{clean}}}$$
     Tests which concepts survive blur, fog, and noise.
  2. **Diagnostic Oracle-CBM Test:** Compare accuracy using predicted concepts $Accuracy(\hat{C} \to Y)$ against ground-truth concepts $Accuracy(C^* \to Y)$ to distinguish perception failure from ontology limitations.

---

### Track 4: Evaluation, Robustness Benchmarking, & Calibration
* **Primary Files:** [`evaluate_robustness.py`](file:///d:/ML_Project/evaluate_robustness.py), [`src/corruptions.py`](file:///d:/ML_Project/src/corruptions.py).
* **Benchmark Scope:** 15 Hendrycks corruptions across 4 categories (Noise, Blur, Weather, Digital) $\times$ 5 severity levels = **75 test evaluations** on the held-out test set (12,630 samples).
* **Core Metrics:**
  - **Mean Corruption Accuracy ($mCA$):** Average accuracy across all 75 corruption-severity conditions.
  - **Relative Performance Degradation ($RPD$):** $\frac{\text{Acc}_{\text{clean}} - mCA}{\text{Acc}_{\text{clean}}} \times 100\%$.
  - **Expected Calibration Error ($ECE$):** Measures probability calibration across $M = 10$ confidence bins.
* **Execution Command:**
  ```bash
  python evaluate_robustness.py \
      --checkpoint checkpoints/baseline_resnet18/best_baseline_resnet18.pt \
      --data-root german \
      --meta-path data/master_v1/metadata/test.csv \
      --output-dir reports/baseline_resnet18 \
      --batch-size 128
  ```

---

### Track 5: Academic Paper Writing & Committee Defense
* **Primary File:** [`research_content.md`](file:///d:/ML_Project/research_content.md).
* **Usage:** Contains publication-ready prose, LaTeX equations, and 19 formal APA references.
* **Defense Grounding:** When explaining methodological decisions to mentors or reviewers, refer to the **Citation Defense Breakdown Table** at the end of `research_content.md`:
  - **Externally Citable:** GTSRB benchmark (Stallkamp et al.), CBM framework (Koh et al.), Texture bias (Geirhos et al.), ResNet-18 (He et al.), Bag-of-Tricks (He et al.), Warmup & Cosine decay (Goyal et al., Loshchilov & Hutter), Corruptions benchmark (Hendrycks & Dietterich), Calibration (Guo et al.), Vienna Convention (UNECE).
  - **Primary Empirical Contributions (Not Citable):** $\alpha = 0.10$ context margin, canonical $64 \times 64$ Pareto selection, composite track-group stratification ($g_i = \text{ClassId} \oplus \text{TrackId}$).

---

## 4. Non-Negotiable Engineering Invariants

| Rule | Requirement | Rationale |
| :--- | :--- | :--- |
| **1. Hardware** | Local = CPU / Colab = T4 GPU | Local machine lacks CUDA. Running 25 epochs locally takes hours; Colab T4 takes ~4 minutes with AMP. |
| **2. Random Splitting** | **STRICTLY PROHIBITED** | Images are from 30-frame video tracks. Random splitting leaks identical physical signs, causing inflated scores. |
| **3. Anisotropic Resizing** | **STRICTLY PROHIBITED** | Deforms physical shapes (circles $\to$ ellipses), corrupting the CBM semantic bottleneck. |
| **4. Primary Metric** | **Macro-Averaged F1-Score** | GTSRB is severely imbalanced (210 to 2,250 samples). Accuracy masks failure on rare, critical signs. |
| **5. Model Engine** | **ResNet-18 Backbone** | Colab T4 compute constraint. Fits within memory, allows rapid factorial comparison across all 7 experiments. |
| **6. Reproducibility** | Master Seed = `42` | All data shuffles, Kaiming head initializations, and augmentations must be deterministically reproducible. |

---

## 5. Google Colab Execution Workflow

1. Run the local packaging script to update the archive:
   ```powershell
   python create_colab_package.py
   ```
2. Upload `ML_Project_Data.zip` and [`baseline_experiment_colab.ipynb`](file:///d:/ML_Project/baseline_experiment_colab.ipynb) to your Google Drive.
3. Open `baseline_experiment_colab.ipynb` in Google Colab, select **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
4. Run all cells. The notebook will:
   - Mount Google Drive and unpack `ML_Project_Data.zip`.
   - Verify GPU acceleration (`nvidia-smi`).
   - Run 25 epochs of baseline ResNet-18 training with Automatic Mixed Precision.
   - Plot loss, accuracy, and macro-F1 curves.
   - Execute the complete 75-point corruption benchmark and compute $mCA$, $RPD$, and $ECE$.
   - Automatically back up all checkpoints and evaluation matrices back to Google Drive (`/content/drive/MyDrive/ML_Project_Results/baseline_resnet18/`).

