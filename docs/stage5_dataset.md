# Stage 5 Specification: Autonomous Dataset Research, Selection, Acquisition, Processing & Validation

**SignBridge — Real-Time Sign Language Recognition & Translation System**

> [!IMPORTANT]
> **Stage 5 Scope Notice**: Stage 5 implements the complete dataset acquisition, inspection, quality validation, MediaPipe landmark extraction, and Stage 4 feature engineering into the 94D `FeatureSchema v1.0` representation.
> **Stage 5 does NOT train any machine-learning models, select classifiers, perform sign classification, implement translation, or build TTS/UI.**

---

## 1. Dataset Selection & Candidates Evaluated

In accordance with Part 1 and Part 2, seven candidate sign language datasets were researched and evaluated across 20 engineering criteria:

| Candidate | Modality | Classes | Samples | License | Auth Required | Status |
|---|---|---|---|---|---|---|
| **ASL-Alphabet-Natural-RGB** | Natural RGB Photos | **26** (A–Z) | **10,873** | **Unspecified / Open Access (Requires Verification)** | **None** | **SELECTED** |
| **ASL-SemCom** | RGB (Synthetic Overlays) | 24 (A–Y) | 12,290 | CC BY 4.0 | None | Eliminated (Synthetic overlays degrade MediaPipe) |
| **Kaggle ASL Alphabet** | RGB Images | 29 | 87,000 | CC BY 4.0 | Yes (Kaggle API) | Eliminated (Auth barrier) |
| **LSA64** | MP4 Video (60fps) | 64 | 3,200 | CC BY-NC-SA 4.0 | No | Eliminated (Two-hand, Gloves) |
| **WLASL** | Video (YouTube) | 2,000 | 21,083 | CC BY 4.0 | No | Eliminated (>25% link rot) |
| **ASL-MNIST** | Grayscale (28x28) | 24 | 34,627 | CC BY-SA 4.0 | No | Eliminated (Too low res) |
| **OpenHands Poses** | Keypoints | 30 | 15,000 | CC BY 4.0 | No | Eliminated (Pre-extracted poses) |

---

## 2. Dataset Selection Rationale & Rejection of ASL-SemCom

### Why ASL-SemCom Was Rejected
During empirical inspection and processing of the Zenodo ASL-SemCom dataset (DOI: `10.5281/zenodo.14635573`), we discovered that all 12,290 raw images contain pre-drawn synthetic colored circles and skeleton lines across the hand joints. When passed to MediaPipe Hands (which uses a single-shot SSD palm detector trained on natural human skin tones), these synthetic overlays severely degraded detection accuracy, causing failure rates of 70% to 90% across key classes.

### Why ASL-Alphabet-Natural-RGB Won
1. **Natural RGB Hand Imagery**: Contains authentic, un-annotated photographs of human hands in real-world lighting and varying backgrounds, yielding high MediaPipe detection confidence (>75-80%).
2. **Full 26-Letter ASL Alphabet (A–Z)**: Covers all 26 static fingerspelling letters without omissions.
3. **Unauthenticated Public Access**: Distributed on Hugging Face Datasets (`Marxulia/asl_sign_languages_alphabets_v03`) with direct HTTP downloading.
4. **Pre-partitioned Splits**: Pre-divided into 8,697 train and 2,176 test samples, supporting clean evaluation without data leakage.

---

## 3. Dataset Source & Provenance

- **Title**: ASL Alphabet (Natural RGB Images)
- **Source Repository**: `Marxulia/asl_sign_languages_alphabets_v03` (Hugging Face Datasets)
- **Source URL**: `https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03`
- **License**: `Unspecified / Community Open Access (Requires Verification)`
- **Modality**: Natural RGB Photographs
- **Total Samples**: 10,873 (8,697 Train, 2,176 Test)
- **Classes**: 26 (A through Z)

---

## 4. Processing Pipeline Architecture

```
                  Raw RGB Images (datasets/external/asl_semcom/raw/)
                                          │
                                          ▼
                               OpenCV Image Ingestion
                                          │
                                          ▼
                       MediaPipeHandDetector (Vision Stage 3)
                            [TFLite Graph Inference]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
          Hands Detected (>= 1)                           No Hands / Corrupt
                   │                                             │
                   ▼                                             ▼
       Multiple Hand Arbitration                        Quarantine System
       [Select Highest Score Hand]               [quarantine_log.json + Reason]
                   │
                   ▼
       DetectedHand Domain Contract
                   │
                   ▼
      HandFeatureExtractor (Stage 4)
  [1. LandmarkValidator: 21 keypoints, finite]
  [2. CoordinateNormalizer: Wrist origin, scale]
  [3. Handedness Canonicalization: Left x -> -x]
  [4. Geometric Features: 15 angles, 15 dists]
                   │
                   ▼
      FeatureVector (FeatureSchema v1.0)
         - 94 dimensions
         - float32
         - Zero NaNs / Infs
                   │
                   ▼
    NumPy Binary Storage & Manifest (datasets/processed/)
         ├── features.npy      [Shape: (N, 94), dtype: float32]
         ├── labels.npy        [Shape: (N,), dtype: int64]
         ├── classes.json      [Class ID mapping: 0 -> A, etc.]
         ├── sample_index.json [Per-sample metadata and train/test split]
         └── dataset_manifest.json (datasets/metadata/)
```

---

## 5. Class Mapping (26 Classes: A through Z)

| Class ID | Sign Letter | Category | Handedness | Motion |
|---|---|---|---|---|
| `0` | **A** | Vowel | Dominant (Single) | Static |
| `1` | **B** | Consonant | Dominant (Single) | Static |
| `2` | **C** | Consonant | Dominant (Single) | Static |
| `3` | **D** | Consonant | Dominant (Single) | Static |
| `4` | **E** | Vowel | Dominant (Single) | Static |
| `5` | **F** | Consonant | Dominant (Single) | Static |
| `6` | **G** | Consonant | Dominant (Single) | Static |
| `7` | **H** | Consonant | Dominant (Single) | Static |
| `8` | **I** | Vowel | Dominant (Single) | Static |
| `9` | **J** | Consonant | Dominant (Single) | Static pose captured in dataset |
| `10` | **K** | Consonant | Dominant (Single) | Static |
| `11` | **L** | Consonant | Dominant (Single) | Static |
| `12` | **M** | Consonant | Dominant (Single) | Static |
| `13` | **N** | Consonant | Dominant (Single) | Static |
| `14` | **O** | Vowel | Dominant (Single) | Static |
| `15` | **P** | Consonant | Dominant (Single) | Static |
| `16` | **Q** | Consonant | Dominant (Single) | Static |
| `17` | **R** | Consonant | Dominant (Single) | Static |
| `18` | **S** | Consonant | Dominant (Single) | Static |
| `19` | **T** | Consonant | Dominant (Single) | Static |
| `20` | **U** | Vowel | Dominant (Single) | Static |
| `21` | **V** | Consonant | Dominant (Single) | Static |
| `22` | **W** | Consonant | Dominant (Single) | Static |
| `23` | **X** | Consonant | Dominant (Single) | Static |
| `24` | **Y** | Consonant / Semi-vowel | Dominant (Single) | Static |
| `25` | **Z** | Consonant | Dominant (Single) | Static pose captured in dataset |

---

## 6. Data Leakage Prevention Strategy

- The raw dataset is pre-partitioned into discrete `train` and `test` subfolders.
- Every processed sample's split origin (`train` vs `test`) is explicitly recorded in `datasets/processed/sample_index.json`.
- In Stage 6, ML model training splits will strictly preserve this group boundary, guaranteeing that no test image or user session bleeds into the training set.
- Unit tests (`tests/test_dataset.py::test_data_leakage_protection_splits`) verify that train and test sample paths have zero intersection.

---

## 7. Storage Format Decision

Processed features are stored using native NumPy `.npy` binary serialization:
- **`features.npy`**: 2D array of shape `(N, 94)` with contiguous `np.float32` memory layout.
- **`labels.npy`**: 1D array of shape `(N,)` with `np.int64` memory layout.
- **`classes.json`**: Bidirectional mapping between integer IDs and class letter strings.
- **`sample_index.json`**: Rich per-sample metadata tracking original filename, detection score, handedness, scale factor, and split partition.
- **Rationale**:
  - Memory-mappable (`mmap_mode='r'`) for sub-millisecond loading times.
  - Avoids creating tens of thousands of tiny individual files on disk.
  - Native compatibility with PyTorch, TensorFlow, Scikit-learn, and XGBoost.

---

## 8. Quality Assurance & Standalone Validator

The standalone validation script [`scripts/dataset/validate_dataset.py`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/scripts/dataset/validate_dataset.py) executes 15 automated checks:
1. `features.npy` exists.
2. `labels.npy` exists.
3. `classes.json` exists.
4. `sample_index.json` exists.
5. `features` array is 2D.
6. Feature dimension is exactly 94 (`FeatureSchema v1.0`).
7. Label count strictly matches feature sample count.
8. Sample index record count matches sample count.
9. Feature data type is `float32`.
10. Labels data type is integer (`int64`).
11. Numerical integrity: strictly 0 `NaN`, 0 `Inf`, 0 `-Inf`.
12. Labels span the expected range $[0, 25]$.
13. All 26 classes are represented.
14. Class imbalance ratio $< 3.0$ (or reasonable subset ratio).
15. Split disjointness: train and test sets strictly disjoint (intersection = 0).
16. Cryptographic SHA256 integrity checksums generated and recorded in `datasets/metadata/dataset_manifest.json`.

---

## 9. Reproducibility & Developer Commands

```powershell
# 1. Research Candidates & Generate Comparison Report
.\.venv\Scripts\python.exe scripts/dataset/research_dataset.py

# 2. Acquire & Inspect Raw Dataset
.\.venv\Scripts\python.exe scripts/dataset/inspect_dataset.py

# 3. Test Small Subset (Dry-Run Verification)
.\.venv\Scripts\python.exe scripts/dataset/prepare_dataset.py --subset-per-class 5

# 4. Process Full Dataset Through Stage 4 Feature Pipeline (Pending User Review)
# .\.venv\Scripts\python.exe scripts/dataset/prepare_dataset.py

# 5. Validate Processed Dataset & Generate Manifest
.\.venv\Scripts\python.exe scripts/dataset/validate_dataset.py

# 6. Compute Dataset Statistics & Distribution
.\.venv\Scripts\python.exe scripts/dataset/dataset_stats.py

# 7. Run Full Automated Test Suite (Stage 0 - 5)
.\.venv\Scripts\python.exe -m pytest tests -v
```
