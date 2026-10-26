# Stage 6 Specification: Production Sign Recognition Model Development

**SignBridge — Real-Time Sign Language Recognition & Translation System**

> [!IMPORTANT]
> **Stage 6 Scope Notice**: Stage 6 builds, benchmarks, selects, tunes, trains, and verifies the production **static sign classification model** from the 94-dimensional `FeatureSchema v1.0` representation produced by Stage 4 & 5.
> **Stage 6 does NOT implement sentence formation, translation, NLP, temporal smoothing, text-to-speech, speech recognition, UI redesign, or dynamic gesture tracking.**

---

## 1. Objective

Transform normalized 94-dimensional hand feature vectors into deterministic, low-latency static sign classifications across all 26 letters of the American Sign Language alphabet (A through Z). The resulting classifier must achieve high generalization accuracy on unseen test subjects and real-time inference latency ($< 10$ ms) suitable for 30 FPS video streaming.

---

## 2. Dataset Contract

The model consumes the validated Stage 5 production dataset artifacts from [`datasets/processed/`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/datasets/processed/):

| Artifact | Shape / Format | Data Type | Description |
|---|---|---|---|
| `features.npy` | `(8554, 94)` | `float32` | Normalized hand landmark coordinates & geometric features |
| `labels.npy` | `(8554,)` | `int64` | Contiguous class IDs $[0, 25]$ |
| `classes.json` | JSON | Bidirectional | Mapping between class labels ("A".."Z") and class IDs $(0..25)$ |
| `sample_index.json` | JSON Array | Metadata | Per-sample source path, handedness, and `train`/`test` split |
| `dataset_manifest.json` | JSON | Metadata | Provenance, checksums, and schema version |

Pre-training validation strictly enforces:
- `features.shape == (8554, 94)`
- `features.dtype == np.float32`
- `labels.dtype == np.int64`
- Exactly zero non-finite values (0 `NaN`, 0 `Inf`)
- `len(labels) == len(features) == len(sample_index) == 8554`

---

## 3. Feature Representation (FeatureSchema v1.0)

Every input vector contains exactly 94 dimensions:
1. **Coordinates** (63 dims, `0..62`): Translation-invariant, scale-invariant, handedness-canonicalized $(x, y, z)$ coordinates for all 21 hand landmarks relative to wrist origin $(0, 0, 0)$.
2. **Wrist-to-Tip Distances** (5 dims, `63..67`): Euclidean distances from wrist to the 5 fingertips.
3. **Inter-Fingertip Distances** (10 dims, `68..77`): Pairwise Euclidean distances between all 5 fingertips ($C(5, 2) = 10$).
4. **Joint Flexion Angles** (15 dims, `78..92`): Internal 3D angles in radians $[0, \pi]$ across all 3 joints for each of the 5 fingers.
5. **Palm Aspect Ratio** (1 dim, `93`): Ratio of palm width to palm length.

---

## 4. Train, Validation & Test Partitioning Methodology

To prevent data snooping and ensure unbiased generalization metrics:
- **Source Split Preservation**: The original dataset's session-level `train` and `test` splits recorded in `sample_index.json` are strictly respected.
  - Total Samples: 8,554
  - Train Partition: 6,839 (80.0%)
  - Test Partition: 1,715 (20.0%)
- **Stratified Validation Split**: The `train` partition is split into **Train (5,471 samples, 80%)** and **Validation (1,368 samples, 20%)** using a stratified split (`random_state=42`), preserving class balance.
- **Untouched Test Set**: The test set (1,715 samples) remained completely isolated during baseline evaluation, candidate benchmarking, and hyperparameter tuning. It was evaluated exactly **once** on the final refitted model.
- **Disjointness Invariant**:
  $$\text{Train} \cap \text{Validation} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Validation} \cap \text{Test} = \emptyset$$

---

## 5. Candidate Models Evaluated

Five distinct classification paradigms were benchmarked on the 94D feature space:
1. **Logistic Regression (Baseline)**: Multinomial softmax regression with L2 regularization ($C=1.0$) and `StandardScaler`.
2. **Random Forest Classifier**: Ensemble of 150/200 decision trees with Gini impurity and sub-sampling.
3. **Extra Trees (Extremely Randomized Trees)**: Ensemble of 150/200 randomized trees providing variance reduction.
4. **SVM (RBF Kernel)**: Support Vector Machine with non-linear Radial Basis Function kernel ($C=10.0$) and calibrated probabilities.
5. **Linear SVM**: Linear Support Vector Classifier with L2 penalty and calibrated probabilities.

---

## 6. Model Selection Policy

The primary selection metric is **Macro F1 Score**:
$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C F1_c$$
Because sign recognition requires consistent performance across all 26 letters, Macro F1 prevents high-frequency classes from masking poor performance on difficult signs.

Secondary criteria:
- **Inference Latency**: Must be comfortably $< 10$ ms/sample on CPU for real-time video processing.
- **Model Footprint**: Prefer compact models ($< 50$ MB) that load rapidly into memory.
- **Deterministic Behavior**: Zero stochastic variation during inference.

---

## 7. Benchmarking Results (Validation Set)

Evaluated strictly on `X_train` (5,471 samples) vs `X_val` (1,368 samples):

| Rank | Model Architecture | Preprocessing | Validation Accuracy | Macro F1 | Weighted F1 | Latency (ms) | Size (KB) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Linear SVM ($C=1.0$)** | `StandardScaler` | **97.73%** | **0.9763** | **0.9773** | 0.035 | 742.8 |
| **2** | **SVM (RBF Kernel, $C=10.0$)** | `StandardScaler` | **97.08%** | **0.9703** | **0.9709** | 0.160 | 1,318.5 |
| **3** | **Extra Trees ($n=150$)** | None | 96.64% | 0.9655 | 0.9664 | 0.029 | 9,217.7 |
| **4** | **Random Forest ($n=150$)** | None | 96.49% | 0.9639 | 0.9649 | 0.029 | 3,833.1 |
| **5** | **Logistic Regression (Baseline)** | `StandardScaler` | 95.47% | 0.9529 | 0.9547 | 0.001 | 11.8 |

---

## 8. Hyperparameter Optimization & Selection

Fine-tuning evaluated candidate configurations with `CalibratedClassifierCV(ensemble=False)`:
- `Linear SVM (C=1.0)`: Macro F1 = 0.9600
- `Linear SVM (C=2.0)`: Macro F1 = 0.9652
- `Extra Trees (n=200, depth=25)`: Macro F1 = 0.9663
- `Random Forest (n=200, depth=20)`: Macro F1 = 0.9624
- **`SVM (RBF Kernel, C=10.0, gamma='scale')`**: **Macro F1 = 0.9712**, Accuracy = **97.15%**

**Winner**: **SVM with RBF Kernel ($C=10.0$, $\gamma=\text{scale}$)** with calibrated probabilities.
- Non-linear decision boundaries capture subtle joint angle interactions (e.g. thumb-finger tucking).
- Excellent calibration for confidence thresholding in Stage 7+.
- Highly compact memory footprint (1.45 MB).

---

## 9. Final Test Performance (Untouched Test Set)

The winning model was refitted on the combined **Train + Validation** set (6,839 samples in 2.97 seconds) and evaluated **exactly once** on the **1,715 untouched test samples**:

| Metric | Measured Value | Acceptance Threshold | Status |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **98.02%** | $\ge 90.0\%$ | **EXCEEDED** |
| **Top-3 Test Accuracy** | **99.36%** | $\ge 95.0\%$ | **EXCEEDED** |
| **Test Macro Precision** | **0.9808** | $\ge 0.90$ | **EXCEEDED** |
| **Test Macro Recall** | **0.9796** | $\ge 0.90$ | **EXCEEDED** |
| **Test Macro F1** | **0.9800** | $\ge 0.90$ | **EXCEEDED** |
| **Test Weighted F1** | **0.9802** | $\ge 0.90$ | **EXCEEDED** |
| **Mean Inference Latency** | **4.160 ms** | $< 10.0\text{ ms}$ | **EXCEEDED** |
| **p95 Inference Latency** | **4.638 ms** | $< 15.0\text{ ms}$ | **EXCEEDED** |
| **Inference Throughput** | **240.4 samples/sec** | $\ge 30\text{ FPS}$ | **EXCEEDED** |
| **Model Disk Size** | **1.45 MB** | $< 50.0\text{ MB}$ | **EXCEEDED** |

---

## 10. Error Analysis & Confusion Matrix Findings

### Strongest Sign Classes ($F1 = 1.0$)
- **B, C, F, I, J, T, V** achieved **1.000 F1 score** (zero false positives and zero false negatives).
- **A, S, W, Y** achieved $> 0.992$ F1 score.

### Most Difficult Sign Classes
1. **P** ($F1 = 0.9474$, Precision: 0.926, Recall: 0.969): Downward-angled index/middle finger occasionally confused with E or K.
2. **R** ($F1 = 0.9552$, Precision: 0.955, Recall: 0.955): Crossed fingers confused with upright side-by-side U (2 errors).
3. **Z** ($F1 = 0.9559$, Precision: 0.942, Recall: 0.970): Pointing index finger occasionally confused with K.
4. **K** ($F1 = 0.9571$, Precision: 0.944, Recall: 0.971): V-shape with thumb tucked between fingers confused with P (1 error) or Z (1 error).
5. **Q** ($F1 = 0.9600$, Precision: 1.000, Recall: 0.923): Downward index/thumb pinch confused with G (3 errors) due to subtle wrist orientation angles.
6. **N** ($F1 = 0.9683$, Precision: 1.000, Recall: 0.938): Thumb under two fingers confused with M (thumb under three fingers, 2 errors).

*Notice that even the single lowest performing class ('P') still achieves an F1 score of 0.9474.*

---

## 11. Inference Latency & Efficiency Profiling

Measured on host CPU across 1,000 independent single-vector runs:
- **Mean Latency**: **4.160 ms**
- **Median Latency**: **4.078 ms**
- **95th Percentile Latency**: **4.638 ms**
- **Single-Thread Throughput**: **240.4 inferences/second**
- **Camera Frame Budget (30 FPS)**: 33.3 ms total budget. Model inference consumes only **12.5%** of the frame budget, leaving over 28 ms for OpenCV video capture and MediaPipe hand detection.

---

## 12. Model Artifacts & File Structure

Saved under [`models/sign_classifier/`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/models/sign_classifier/):
```
models/
└── sign_classifier/
    ├── model.joblib      # Serialized scikit-learn pipeline (1.45 MB)
    ├── metadata.json     # Complete provenance, hyperparameters & metrics
    ├── classes.json      # Bidirectional class mapping (26 classes)
    └── metrics.json      # Test performance summary
```

Saved visual reports under [`reports/stage6/`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/reports/stage6/):
```
reports/
└── stage6/
    ├── model_comparison.json      # 5-model benchmark ranking
    ├── classification_report.json # Per-class precision, recall, F1, support
    ├── confusion_matrix.json      # 26x26 confusion counts
    ├── error_analysis.json        # Strongest/weakest classes & confused pairs
    ├── benchmark.json             # Latency percentiles & throughput
    ├── confusion_matrix.png       # 26x26 heatmap
    ├── per_class_f1.png           # Bar chart across all letters A-Z
    ├── confidence_distribution.png# Test prediction confidence histogram
    └── model_comparison.png      # Validation score comparison across architectures
```

---

## 13. Inference API Usage

```python
from signbridge.model.persistence import load_sign_classifier
import numpy as np

# 1. Load production model
classifier = load_sign_classifier("models/sign_classifier")

# 2. Input 94D feature vector (from Stage 4 HandFeatureExtractor)
features = np.random.randn(94).astype(np.float32)

# 3. Predict static sign
result = classifier.predict(features)

print(f"Predicted Sign: {result.class_name} (Class ID: {result.class_id})")
print(f"Confidence:     {result.confidence * 100:.2f}%")
print(f"Top Margin:     {result.top_margin * 100:.2f}%")
print(f"Latency:        {result.inference_latency_ms:.2f} ms")
print(f"Top Candidates: {result.top_candidates}")
```

---

## 14. Reproducibility Record

| Parameter | Value |
|---|---|
| **Python Version** | `3.11.15` |
| **NumPy Version** | `2.4.6` |
| **Scikit-Learn Version** | `1.9.1` |
| **Joblib Version** | `1.6.0` |
| **FeatureSchema Version** | `1.0` (94 dimensions) |
| **Dataset Identity** | `ASL-Alphabet-Natural-RGB` |
| **Dataset Manifest SHA256** | `c980cd0294a4f186ea34331b41d105de8260db1241f875b1d1022f7754a0c7d8` (features) |
| **Random Seed** | `42` |
| **Total Samples** | 8,554 (Train: 5,471, Val: 1,368, Test: 1,715) |
| **Model Algorithm** | `CalibratedClassifierCV(SVC(kernel='rbf', C=10.0, gamma='scale'), ensemble=False)` with `StandardScaler` |
