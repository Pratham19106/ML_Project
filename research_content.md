# Research Paper Methodology & Content Reference
**Working Title:** Data-Centric, Representation-Centric, and Hybrid Robustness for Interpretable Traffic Sign Recognition  
**Target Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  
**Document Purpose:** Frozen academic text, mathematical formulations, and formal scholarly citations for publication and committee defense.

---

## Section 3: Data Standardization and Preprocessing Protocol

### 3.1 Motivation & The Geometric Distortion Problem

In real-world visual traffic sign recognition (TSR), input imagery captured by vehicle-mounted forward cameras exhibits substantial variations in scale, aspect ratio, and lighting. In the German Traffic Sign Recognition Benchmark (GTSRB) [Stallkamp et al., 2011, 2012], cropped image bounding boxes range from $25 \times 25$ pixels (signs captured at long distance) to $266 \times 232$ pixels (signs captured immediately adjacent to the vehicle). Empirical inspection reveals that native bounding box aspect ratios ($W/H$) span a wide distribution from $0.39$ (severely elongated) to $1.44$ (substantially widened), centered around a median of $1.00$.

Conventional deep learning pipelines routinely apply naive anisotropic resizing:
$$\mathcal{T}_{\text{naive}}: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{S \times S \times 3}$$
which squashes or stretches arbitrary image dimensions directly into a fixed square canvas (e.g., $S = 64$ or $S = 224$).

While standard black-box convolutional neural networks (CNNs) can partially memorize deformed features on clean training distributions, this transformation introduces two severe failure modes:

1. **Geometric Concept Corruption:** Our research evaluates a **Concept Bottleneck Model (CBM)** [Koh et al., 2020] whose intermediate representations predict physical, standards-grounded attributes ($\text{Shape} \in \{\text{circle}, \text{triangle\_up}, \text{triangle\_down}, \text{octagon}, \text{diamond}\}$). Anisotropic scaling distorts circular speed limit plates into ellipses and alters the interior angles of equilateral warning triangles and regular octagonal STOP signs. Forcing the concept predictor to infer physical shapes from artificially deformed silhouettes compromises the semantic fidelity of the bottleneck.
2. **Spurious Correlation & High-Frequency Shortcuts:** Standard CNNs are biased toward local texture rather than global shape [Geirhos et al., 2019]. Aggressive bilinear stretching of small patches introduces non-uniform high-frequency interpolation blur that deep networks exploit as predictive shortcuts for sign scale and camera distance [Wang et al., 2020], confounding our ability to isolate genuine data-centric robustness improvements under visual corruptions.

To eliminate these confounding factors, we establish a frozen, mathematically standardized preprocessing contract: **Context-Buffered Region-of-Interest (ROI) Cropping followed by Proportional Letterbox Transformation to a Canonical $64 \times 64$ Canvas**.

---

### 3.2 Mathematical Formulation of the Preprocessing Pipeline

The preprocessing pipeline maps each raw image $I_{\text{raw}} \in \mathbb{R}^{H_{\text{raw}} \times W_{\text{raw}} \times 3}$ and its associated ground-truth bounding box $\text{ROI} = (x_1, y_1, x_2, y_2)$ to a standardized tensor $X \in \mathbb{R}^{3 \times 64 \times 64}$ via four deterministic stages:

```text
Raw Camera Image (H_raw x W_raw)
          │
          ▼  [Stage 1: Context-Buffered ROI Crop]
Buffered Bounding Box (w_crop x h_crop)
          │
          ▼  [Stage 2: Proportional Bilinear Resizing]
Aspect-Preserved Rescaled Patch (w' x h') where max(w', h') = 64
          │
          ▼  [Stage 3: Symmetric Zero-Padding]
Centered 64 x 64 Canvas (I_canvas)
          │
          ▼  [Stage 4: Tensor Standardization]
Standardized Input Tensor X in R^(3 x 64 x 64)
```

#### Stage 1: Context-Buffered ROI Extraction
Raw GTSRB samples contain annotations specifying the inclusive, zero-indexed bounding box coordinates of the sign:
$$\text{ROI} = [x_1, y_1, x_2, y_2], \quad \text{with } w_{\text{roi}} = x_2 - x_1 + 1, \quad h_{\text{roi}} = y_2 - y_1 + 1$$

Cropping strictly along the annotated ROI boundary frequently clips the outer contrasting rim of the traffic sign (such as the circular red outer band of prohibition signs or the white border of mandatory signs). Conversely, using the uncropped camera image incorporates excessive irrelevant background clutter.

We resolve this trade-off by introducing a fixed contextual margin expansion ratio $\alpha = 0.10$ ($10\%$ context expansion):
$$\Delta x = \left\lfloor \alpha \cdot w_{\text{roi}} \right\rceil, \quad \Delta y = \left\lfloor \alpha \cdot h_{\text{roi}} \right\rceil$$

The buffered crop coordinates $(x_1', y_1', x_2', y_2')$ are clamped strictly to the spatial boundaries of the original image:
$$x_1' = \max(0, x_1 - \Delta x), \quad y_1' = \max(0, y_1 - \Delta y)$$
$$x_2' = \min(W_{\text{raw}} - 1, x_2 + \Delta x), \quad y_2' = \min(H_{\text{raw}} - 1, y_2 + \Delta y)$$

The resulting cropped image patch has dimensions $w_{\text{crop}} = x_2' - x_1' + 1$ and $h_{\text{crop}} = y_2' - y_1' + 1$.

#### Stage 2: Proportional Letterbox Resizing
Following aspect-ratio preservation principles in modern computer vision [Redmon & Farhadi, 2018], the cropped patch is scaled uniformly along both axes such that its longest dimension exactly matches the target canonical resolution $S = 64$:
$$\gamma = \min\left( \frac{S}{w_{\text{crop}}}, \frac{S}{h_{\text{crop}}} \right)$$
$$w' = \max\left(1, \left\lfloor w_{\text{crop}} \cdot \gamma \right\rceil\right), \quad h' = \max\left(1, \left\lfloor h_{\text{crop}} \cdot \gamma \right\rceil\right)$$

The patch is resampled to $(h', w')$ using bilinear interpolation:
$$I_{\text{resized}} = \text{BilinearResize}(I_{\text{crop}}, (h', w'))$$

Under this formulation, the aspect ratio is strictly preserved: $\frac{w'}{h'} \approx \frac{w_{\text{crop}}}{h_{\text{crop}}}$. Circular signs maintain a $1.00$ radial ratio, and triangular/octagonal angles remain undeformed.

#### Stage 3: Symmetric Zero-Padding
The rescaled patch $I_{\text{resized}}$ is placed squarely in the center of an empty neutral canvas $I_{\text{canvas}} \in \mathbb{R}^{S \times S \times 3}$ filled with zeros ($[0, 0, 0]$ RGB):
$$\Delta p_x = \left\lfloor \frac{S - w'}{2} \right\rfloor, \quad \Delta p_y = \left\lfloor \frac{S - h'}{2} \right\rfloor$$
$$I_{\text{canvas}}(y, x, c) = \begin{cases} 
I_{\text{resized}}(y - \Delta p_y, x - \Delta p_x, c) & \text{if } \Delta p_y \le y < \Delta p_y + h' \text{ and } \Delta p_x \le x < \Delta p_x + w' \\
0 & \text{otherwise}
\end{cases}$$

#### Stage 4: Tensor Standardization
Pixel intensities are converted to floating-point values in the range $[0.0, 1.0]$ and standardized across color channels using transfer-learning normalization parameters $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$ [Deng et al., 2009]:
$$X_{c, y, x} = \frac{\frac{I_{\text{canvas}}(y, x, c)}{255.0} - \mu_c}{\sigma_c}, \quad \forall c \in \{0, 1, 2\}$$

---

### 3.3 Selection of the Canonical $64 \times 64$ Resolution

The choice of input resolution represents an essential balance between computational efficiency and visual fidelity:
* **Standard ImageNet Resolution ($224 \times 224$):** Upscaling an original $28 \times 28$ sign to $224 \times 224$ represents an $8 \times$ magnification that synthesizes substantial interpolation blur without adding authentic physical information, while inflating GPU memory and FLOPs by $12.25 \times$ ($224^2 / 64^2$).
* **CIFAR-Scale Resolution ($32 \times 32$):** Downsampling larger signs ($>100$ px) to $32 \times 32$ degrades fine interior glyphs, making it difficult to distinguish numerical speed limits (e.g., $30$ vs. $80$ km/h) or complex pictograms (e.g., road work vs. pedestrian crossing).
* **The $64 \times 64$ Sweet Spot:** Stallkamp et al. [2012] document that native GTSRB sign sizes cluster around $30$–$50$ pixels (median $43 \times 43$). A $64 \times 64$ canvas preserves fine numeral typography and pictogram edges with minimal upsampling artifacts, while executing at $>1,000$ images/second on an NVIDIA T4 GPU.

---

### 3.4 Prevention of Video-Track Data Leakage (Stratified Group Splitting)

A pervasive methodological flaw in published TSR literature is the application of naive uniform random splitting across GTSRB training samples.

#### The Track Memorization Trap
In GTSRB [Stallkamp et al., 2011, 2012], training images were sampled from continuous video streams recorded as the vehicle approached a physical sign. Filenames follow the structure:
$$\text{Train}/\langle\text{ClassId}\rangle/\langle\text{SignId}\rangle\_\langle\text{TrackId}\rangle\_\langle\text{FrameId}\rangle.\text{png}$$

Each physical sign instance generates a discrete track containing approximately $30$ sequential frames (our empirical audit revealed $1,305$ tracks with exactly $30$ frames and $2$ tracks with $29$ frames, totaling $1,307$ unique tracks). Consecutive frames within a single track differ only by negligible sub-pixel vehicle translation and slight optical zoom.

If images are partitioned randomly into train and validation sets, consecutive frames from the *same* physical sign end up on both sides of the split. This induces severe **temporal and instance data leakage** [Kaufman et al., 2012]: the network simply memorizes the background scenery, lighting condition, and mounting pole angle of that specific physical instance, yielding severely inflated validation scores ($>99.5\%$) that mask real-world generalization failure.

#### Group-Stratified Partitioning Solution
To guarantee zero video-frame leakage, we enforce a **Group-Stratified Partition** [Pedregosa et al., 2011] using the composite grouping key:
$$g_i = \text{ClassId}_i \oplus \text{TrackId}_i$$

For every traffic sign class $k \in \{0, \dots, 42\}$, the set of unique tracks $\mathcal{G}_k$ is deterministically shuffled using master seed $42$:
1. $80\%$ of tracks are assigned strictly to the training split ($\mathcal{D}_{\text{train}}$: $31,378$ images across $1,046$ tracks).
2. $20\%$ of tracks are assigned strictly to the validation split ($\mathcal{D}_{\text{val}}$: $7,830$ images across $261$ tracks).
3. The official held-out test benchmark ($\mathcal{D}_{\text{test}}$: $12,630$ images from `Test.csv`) is preserved completely untouched, never accessed during training, validation, hyperparameter tuning, or normalization statistics.

Under this protocol, the validation set functions as a genuine held-out instance test, measuring whether the model generalizes to entirely unseen physical signs.

---

### 3.5 Provenance & Version Control Table

All partitions, bounding box specifications, and grounded concept vectors are frozen in the Master Dataset manifest registry (*reproducibility checksums generated via SHA-256 on the frozen CSV manifests*):

| Split Manifest | Samples | Unique Tracks | Role in Research | Checksum (SHA-256) |
| :--- | :---: | :---: | :--- | :--- |
| **`train.csv`** | $31,378$ | $1,046$ | Clean model optimization & baseline training | `8b3ccc81296014751cc1ebbde1e9085076472f01e6704e420c32f25a31ac3aa4` |
| **`val.csv`** | $7,830$ | $261$ | Model selection & Macro-F1 checkpointing | `5658566473412da8e4b3c2d251ee563ba894366f757d7830cf5abd03dcb2af0d` |
| **`test.csv`** | $12,630$ | — | Final clean testing & held-out corruption evaluation | `0a4029151060c5e748303aa6d88c7c7b53ec7e8fa0f77d5f66707837626ddc3b` |
| **`master_metadata.csv`** | $51,838$ | $1,307$ | Complete consolidated registry with concept vectors | `b6daad60ae5c7be0cda9fcfb26ad025592451af40b7c3ab6901c940a62673b40` |

---

## Section 4: Baseline Architecture and Optimization Protocol

### 4.1 Architecture & Head Adaptation

The standard black-box baseline adopts a **ResNet-18** backbone [He et al., 2016]. The network is parameterized by four sequential residual stages yielding a 512-dimensional spatial feature vector after global average pooling:
$$h = \text{AdaptiveAvgPool2D}(f_{\theta}(X)) \in \mathbb{R}^{512}$$

To map feature representations to the GTSRB task, the original 1,000-way ImageNet classification head is replaced by a linear projection layer parameterized for the $K = 43$ traffic sign classes:
$$z = W_{\text{fc}} h + b_{\text{fc}}, \quad W_{\text{fc}} \in \mathbb{R}^{43 \times 512}, \quad b_{\text{fc}} \in \mathbb{R}^{43}$$

#### Weight Initialization of the New Head
Because the classification head is initialized randomly while the backbone retains pretrained weights, poor initialization can send massive gradient shocks through the network, destroying learned feature extractors in early iterations. We initialize the projection matrix using **Kaiming (He) Normal Initialization** [He et al., 2015] with fan-out mode:
$$W_{\text{fc}} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right), \quad d_{\text{in}} = 512$$
and initialize the bias vector strictly to zero:
$$b_{\text{fc}} = \mathbf{0}_{43}$$

---

### 4.2 Training Dynamics Refinements ("Bag of Tricks")

To ensure stable convergence, prevent catastrophic feature distortion, and eliminate optimization pathologies, we integrate two foundational refinements from modern deep learning literature:

#### 1. Zero-$\gamma$ Batch Normalization Initialization
In a standard residual block with identity shortcut connection:
$$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}, \mathcal{W}_i)$$
the terminal operation inside the residual branch $\mathcal{F}$ is a Batch Normalization layer parameterized by learnable scale $\gamma$ and shift $\beta$:
$$\text{BN}(\mathbf{v}) = \gamma \odot \hat{\mathbf{v}} + \beta$$

Following He et al. [2019] (*Bag of Tricks for Image Classification*), we initialize $\gamma = 0$ for the last Batch Normalization layer in every residual block (`bn2` in BasicBlock). Consequently, at initialization ($t = 0$):
$$\mathcal{F}(\mathbf{x}, \mathcal{W}_i) \equiv \mathbf{0} \implies \mathbf{y} = \mathbf{x}$$

Every residual block acts as an exact identity mapping at step zero. This eliminates signal explosion and gradient instability in early iterations, allowing signal propagation unimpeded across the network depth and stabilizing transfer learning from the pretrained checkpoint.

#### 2. No-Bias-Decay (Decoupled Regularization) Rule
Standard weight decay applies an $L_2$ regularization penalty uniformly across all learnable network parameters:
$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \sum_{\theta \in \Theta} \|\theta\|_2^2$$

However, applying weight decay to 1D parameters—specifically layer biases $b$ and Batch Normalization affine parameters $(\gamma, \beta)$—leads to severe underfitting by artificially restricting scale calibration and dynamic range [He et al., 2019; Goyal et al., 2017].

We partition the network parameters $\Theta$ into two mutually exclusive optimization groups:
$$\Theta_{\text{decay}} = \left\{ \theta \in \Theta \mid \text{dim}(\theta) \ge 2 \right\} \quad (\text{2D Conv kernels, Linear weight matrices})$$
$$\Theta_{\text{no\_decay}} = \left\{ \theta \in \Theta \mid \text{dim}(\theta) < 2 \right\} \quad (\text{1D biases, BN scale } \gamma \text{ and shift } \beta)$$

Weight decay ($\lambda = 10^{-4}$) is applied strictly to $\Theta_{\text{decay}}$, while $\Theta_{\text{no\_decay}}$ receives zero weight decay ($\lambda = 0$).

---

### 4.3 Optimization Schedule & Numerical Precision

1. **Optimizer:** Stochastic Gradient Descent (SGD) with Nesterov momentum $\mu = 0.9$ [Sutskever et al., 2013].
2. **Warmup & Decay Schedule:** 
   - **Linear Warmup:** During epochs $1 \dots 3$, the learning rate increases linearly from $0.1 \times \eta_{\text{base}}$ to the base rate $\eta_{\text{base}} = 0.05$ [Goyal et al., 2017] to prevent divergence while the new head aligns.
   - **Cosine Annealing:** For epochs $4 \dots E_{\text{max}}$, the learning rate decays following a half-period cosine cycle down to $\eta_{\text{min}} = 10^{-5}$ [Loshchilov & Hutter, 2017]:
     $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{warm}}}{T_{\text{max}} - T_{\text{warm}}} \pi\right)\right)$$
3. **Objective Function:** Multi-class Cross-Entropy Loss:
   $$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(z_{i, y_i})}{\sum_{j=0}^{42} \exp(z_{i, j})}$$
4. **Mixed Precision:** Training is accelerated using Automatic Mixed Precision (AMP `torch.cuda.amp.autocast`) with dynamic gradient scaling, reducing memory footprint and maximizing Tensor Core throughput.

---

### 4.4 Model Selection & Validation Protocol

Because GTSRB exhibits severe class imbalance (ranging from 210 training samples in Class 0 to 2,250 samples in Class 2), standard classification accuracy can be deceptive. A trivial model biased toward frequent classes can achieve high accuracy while failing catastrophically on critical rare signs.

We adopt **Macro-Averaged F1-Score** as the primary criterion for validation evaluation, model checkpointing, and early stopping:
$$\text{Macro-F1} = \frac{1}{K} \sum_{k=0}^{K-1} \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

The model checkpoint achieving peak Validation Macro-F1 across all training epochs is preserved as the frozen baseline artifact (`best_baseline_resnet18.pth`) for downstream robustness and concept bottleneck evaluations.

---

## Section 5: Robustness and Uncertainty Evaluation Protocol

### 5.1 Common Visual Corruptions (Hendrycks & Dietterich Benchmark)

To benchmark out-of-distribution robustness against real-world environmental degradation, models are evaluated on the standardized common corruptions protocol [Hendrycks & Dietterich, 2019]. We adapt 15 algorithmic corruptions across 4 major corruption categories, evaluated across 5 severity levels:

```text
               ┌── Noise:    Gaussian, Shot, Impulse
               ├── Blur:     Defocus, Glass, Motion, Zoom
Corruptions ───┼── Weather:  Snow, Frost, Fog, Brightness
               └── Digital:  Contrast, Elastic, Pixelate, JPEG
```

### 5.2 Robustness Metrics

1. **Mean Corruption Accuracy ($mCA$):** The average top-1 classification accuracy across all 15 corruptions and 5 severity levels:
   $$mCA = \frac{1}{15 \times 5} \sum_{c=1}^{15} \sum_{s=1}^{5} \text{Acc}(c, s)$$
2. **Relative Performance Degradation ($RPD$):** The percentage drop in performance relative to clean test accuracy:
   $$RPD = \frac{\text{Acc}_{\text{clean}} - mCA}{\text{Acc}_{\text{clean}}} \times 100\%$$

### 5.3 Predictive Uncertainty Calibration

In safety-critical autonomous driving, a model must be calibrated: when it predicts a sign with $90\%$ confidence, it should be correct $90\%$ of the time. We compute the **Expected Calibration Error (ECE)** [Guo et al., 2017] by grouping test predictions into $M = 10$ equally spaced confidence bins $B_m \subset (0, 1]$:
$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where $\text{acc}(B_m)$ is empirical accuracy and $\text{conf}(B_m)$ is average confidence within bin $B_m$.

---

## Section 8: Methodological Grounding & Scholarly References

Every empirical decision in this research pipeline is directly grounded in established literature, international conventions, and published experimental protocols:

1. **Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C.** (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* **Neural Networks**, 32, 323–332. [DOI: 10.1016/j.neunet.2012.02.016](https://doi.org/10.1016/j.neunet.2012.02.016)
2. **Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C.** (2011). *The German Traffic Sign Recognition Benchmark: A multi-class classification competition.* In **The 2011 International Joint Conference on Neural Networks (IJCNN)** (pp. 1453–1460). IEEE.
3. **Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Been, K., & Liang, P.** (2020). *Concept Bottleneck Models.* In **International Conference on Machine Learning (ICML)** (pp. 5338–5348). PMLR.
4. **Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W.** (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* In **International Conference on Learning Representations (ICLR)**.
5. **Wang, H., Wu, X., Huang, Z., & Xing, E. P.** (2020). *High-frequency Component Helps Explain the Generalization of Convolutional Neural Networks.* In **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)** (pp. 8684–8693).
6. **Redmon, J., & Farhadi, A.** (2018). *YOLOv3: An Incremental Improvement.* arXiv preprint arXiv:1804.02767.
7. **Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L.** (2009). *ImageNet: A Large-Scale Hierarchical Image Database.* In **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)** (pp. 248–255).
8. **Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O.** (2012). *Leakage in data mining: Formulation, detection, and avoidance.* **ACM Transactions on Knowledge Discovery from Data (TKDD)**, 6(4), 1–21.
9. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python.* **Journal of Machine Learning Research (JMLR)**, 12, 2825–2830.
10. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* In **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)** (pp. 770–778).
11. **He, K., Zhang, X., Ren, S., & Sun, J.** (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* In **IEEE International Conference on Computer Vision (ICCV)** (pp. 1026–1034).
12. **He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M.** (2019). *Bag of Tricks for Image Classification with Convolutional Neural Networks.* In **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)** (pp. 558–567).
13. **Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K.** (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv preprint arXiv:1706.02677.
14. **Loshchilov, I., & Hutter, F.** (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* In **International Conference on Learning Representations (ICLR)**.
15. **Sutskever, I., Martens, J., Dahl, G., & Hinton, G.** (2013). *On the importance of initialization and momentum in deep learning.* In **International Conference on Machine Learning (ICML)** (pp. 1139–1147). PMLR.
16. **Hendrycks, D., & Dietterich, T.** (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* In **International Conference on Learning Representations (ICLR)**.
17. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017). *On Calibration of Modern Neural Networks.* In **International Conference on Machine Learning (ICML)** (pp. 1321–1330). PMLR.
18. **United Nations Economic Commission for Europe (UNECE).** (1968). *Convention on Road Signs and Signals* (Vienna Convention). **United Nations Treaty Series**, vol. 1091, p. 3.
19. **Bundesministerium für Verkehr und digitale Infrastruktur (BMVI).** (2017). *Straßenverkehrs-Ordnung (StVO) und Katalog der Verkehrszeichen (VzKat)*. Verkehrsblatt-Dokument Nr. B 3110.

---

## Citation Defense Breakdown Table

This table guides peer review defense and mentor review, clearly separating externally established foundations from our primary empirical and protocol contributions:

| Pipeline Section & Design Element | Type | Scholarly Citation / Canonical Source | Methodological Justification & Empirical Defense |
| :--- | :--- | :--- | :--- |
| **GTSRB Benchmark Dataset** | External Foundation | Stallkamp et al. [2011, 2012] | Canonical benchmark; defines 43 classes, variable dimensions ($15\times 15$ to $250\times 250$). |
| **Concept Bottleneck Framework** | External Foundation | Koh et al. [2020] | Mathematical basis for predicting human-interpretable concepts before final classification. |
| **Texture Bias in CNNs** | External Foundation | Geirhos et al. [2019] | Justifies preventing high-frequency stretching shortcuts to enforce shape reliance. |
| **High-Frequency Spectral Shortcuts** | External Foundation | Wang et al. [2020] | Explains how anisotropic bilinear interpolation creates spectral cues exploited by CNNs. |
| **Proportional Letterboxing** | External Foundation | Redmon & Farhadi [2018] (YOLO) | Standard method for preserving aspect ratios and geometrical integrity in visual detection. |
| **Transfer Learning Normalization** | External Foundation | Deng et al. [2009] (ImageNet) | Pretrained PyTorch weights require $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$. |
| **Context Expansion Ratio ($\alpha = 0.10$)** | Empirical Contribution | *Primary Novel Protocol (This Work)* | Prevents boundary clipping of contrasting sign borders; validated via edge retention audit. |
| **Canonical $64 \times 64$ Canvas** | Empirical Contribution | *Primary Protocol Selection (This Work)* | Pareto-optimal trade-off: preserves median GTSRB sign scale ($43\times 43$) with minimal interpolation. |
| **Track-Leakage Trap & Group Split** | Methodological Standard | Kaufman et al. [2012]; Pedregosa et al. [2011] | Eliminates temporal/instance leakage ($1,307$ tracks grouped by $g_i = \text{ClassId} \oplus \text{TrackId}$). |
| **ResNet-18 Backbone** | External Foundation | He et al. [2016] | Established modern residual architecture balancing representational capacity and compute. |
| **Kaiming Normal Head Init** | External Foundation | He et al. [2015] | Prevents gradient explosion through newly attached linear layer ($d_{\text{in}} = 512, d_{\text{out}} = 43$). |
| **Zero-$\gamma$ BN Initialization** | External Foundation | He et al. [2019] (*Bag of Tricks*) | Sets residual block output to exact identity at $t=0$, smoothing early convergence. |
| **No-Bias-Decay Regularization** | External Foundation | He et al. [2019]; Goyal et al. [2017] | Excludes 1D biases and BN scale/shift from $L_2$ penalty to avoid underfitting. |
| **Linear Warmup + Cosine Decay** | External Foundation | Goyal et al. [2017]; Loshchilov & Hutter [2017] | Stabilizes initial optimization and ensures asymptotic convergence to flat minima. |
| **Macro-F1 Validation Metric** | Methodological Standard | Canonical Metric | Prevents majority-class bias on imbalanced GTSRB class frequencies ($210$ to $2,250$ samples). |
| **Common Corruptions Benchmark** | External Foundation | Hendrycks & Dietterich [2019] | Standardized 15 corruptions (Noise, Blur, Weather, Digital) across 5 severities. |
| **Expected Calibration Error (ECE)** | External Foundation | Guo et al. [2017] | Assesses reliability and probability calibration of predictions in safety-critical TSR. |
| **43-Class Concept Ontology** | International Standard | UNECE Vienna Convention [1968]; BMVI StVO [2017] | Establishes ground truth for shapes, colors, and symbols grounded in international transport law. |

