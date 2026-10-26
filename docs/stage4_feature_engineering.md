# Stage 4 Specification: Landmark Preprocessing & Feature Engineering

**SignBridge — Real-Time Sign Language Recognition & Translation System**

> [!IMPORTANT]
> **Stage 4 Scope Notice**: Stage 4 implements deterministic feature extraction, coordinate normalization, and geometric feature computation on 21-point hand landmarks.
> **Stage 4 does NOT train ML models, classify signs, collect datasets, or generate speech/translations.**

---

## 1. Feature Engineering Architecture

Stage 4 operates strictly downstream of the vision subsystem, decoupling raw camera capture and MediaPipe landmark extraction from future machine learning models:

```
                  DetectedHand / HandLandmarks (21 Keypoints)
                                       │
                                       ▼
                       LandmarkValidator (Module 6.1)
           [Validates: Count=21, Shape=(21,3), Finite numbers, Palm span > eps]
                                       │
                                       ▼
                      CoordinateNormalizer (Module 6.2)
           [1. Translation Invariance: Shifts wrist P0 to origin (0, 0, 0)]
           [2. Scale Invariance: Scales by palm anchor distance ||P9 - P0||]
           [3. Handedness Canonicalization: Reflects Left hands x -> -x]
                                       │
                                       ▼
                      Geometric Computations (Module 6.3)
           [1. 21 Normalized 3D Coordinates: 63 features]
           [2. 5 Wrist-to-Fingertip Distances: 5 features]
           [3. 10 Inter-Fingertip Pairwise Distances: 10 features]
           [4. 15 Joint Flexion/Extension Angles: 15 features in [0, pi]]
           [5. Palm Aspect Ratio: 1 feature]
                                       │
                                       ▼
                        FeatureVector (Module 6.4)
           [1D float32 NumPy Array: Length 94, FeatureSchema v1.0]
           [Bitwise Deterministic, Zero NaNs/Infs, Rich Metadata]
```

---

## 2. Feature Schema Specification (Version 1.0)

Total Dimensions: **94** (Contiguous `np.float32` array).

| Feature Block | Slice Range | Dimensions | Description |
|---|---|---|---|
| **Normalized Coordinates** | `[0:63]` | 63 | $21 \times 3$ normalized $(x, y, z)$ coordinates relative to wrist $P_0$. |
| **Wrist-to-Tip Distances** | `[63:68]` | 5 | Euclidean distances from wrist $P_0$ to fingertips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20). |
| **Inter-Fingertip Distances** | `[68:78]` | 10 | Pairwise Euclidean distances between all $\binom{5}{2} = 10$ fingertip combinations. |
| **Joint Angles** | `[78:93]` | 15 | 3 joint flexion/extension angles per finger (MCP, PIP, DIP) in radians $[0, \pi]$. |
| **Palm Aspect Ratio** | `[93:94]` | 1 | Ratio of palm width $(\|P_{17} - P_5\|)$ to palm length $(\|P_9 - P_0\|)$. |

### Detailed Joint Triplets (15 Angles)

For each joint triplet $(A, B, C)$, the interior angle is computed at center vertex $B$ between vectors $(A - B)$ and $(C - B)$:

1. **Thumb**:
   - `thumb_cmc_angle`: $P_0 - P_1 - P_2$
   - `thumb_mcp_angle`: $P_1 - P_2 - P_3$
   - `thumb_ip_angle`: $P_2 - P_3 - P_4$
2. **Index**:
   - `index_mcp_angle`: $P_0 - P_5 - P_6$
   - `index_pip_angle`: $P_5 - P_6 - P_7$
   - `index_dip_angle`: $P_6 - P_7 - P_8$
3. **Middle**:
   - `middle_mcp_angle`: $P_0 - P_9 - P_10$
   - `middle_pip_angle`: $P_9 - P_{10} - P_{11}$
   - `middle_dip_angle`: $P_{10} - P_{11} - P_{12}$
4. **Ring**:
   - `ring_mcp_angle`: $P_0 - P_{13} - P_{14}$
   - `ring_pip_angle`: $P_{13} - P_{14} - P_{15}$
   - `ring_dip_angle`: $P_{14} - P_{15} - P_{16}$
5. **Pinky**:
   - `pinky_mcp_angle`: $P_0 - P_{17} - P_{18}$
   - `pinky_pip_angle`: $P_{17} - P_{18} - P_{19}$
   - `pinky_dip_angle`: $P_{18} - P_{19} - P_{20}$

---

## 3. Invariance Properties & Mathematical Guarantees

### 1. Translation Invariance
Let $\mathbf{P} \in \mathbb{R}^{21 \times 3}$ be the input landmarks, and $\mathbf{t} \in \mathbb{R}^3$ an arbitrary 3D translation vector.
$$\mathbf{P}' = \mathbf{P} + \mathbf{t}$$
In `CoordinateNormalizer`, the wrist point $P_0' = P_0 + \mathbf{t}$ is subtracted from all points:
$$\Delta P_i' = P_i' - P_0' = (P_i + \mathbf{t}) - (P_0 + \mathbf{t}) = P_i - P_0 = \Delta P_i$$
Thus, $\Delta P_i'$ is mathematically identical to $\Delta P_i$ regardless of camera position or hand placement.
- **Empirical verification error**: `0.00e+00` across all 94 features.

### 2. Scale Invariance
Let $s > 0$ be a scaling factor (e.g. user moving closer to or further from the camera).
$$\mathbf{P}' = P_0 + s(\mathbf{P} - P_0)$$
The palm anchor distance is:
$$L' = \|\Delta P_9'\| = \|s(P_9 - P_0)\| = s\|P_9 - P_0\| = s L$$
The normalized coordinates are:
$$\hat{P}_i' = \frac{\Delta P_i'}{L'} = \frac{s \Delta P_i}{s L} = \frac{\Delta P_i}{L} = \hat{P}_i$$
Fingertip distances, inter-tip distances, and joint angles (scale-invariant by definition of cosine) remain completely invariant.
- **Empirical verification error**: `0.00e+00` across all 94 features.

### 3. Handedness Canonicalization
To allow models trained on right-handed users to generalize seamlessly to left-handed users (and vice versa):
$$\hat{x}_{\text{canonical}} = -\hat{x} \quad \text{if hand is Left}$$
Preserves structural joint angles and relative finger distances while reflecting coordinates into canonical right-hand space.

---

## 4. Numerical Stability Guarantees

1. **Zero-Division Protection**: Palm scaling and aspect ratios use epsilon thresholds ($\epsilon = 10^{-6}$).
2. **Arccos Domain Clamping**: Vector cosine $\cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ is strictly clamped to $[-1.0, 1.0]$ before computing $\arccos$ to eliminate float rounding errors.
3. **Collinear & Coincidental Joint Protection**: If either vector has length $< \epsilon$, angle defaults to $0.0$ without raising exceptions or producing `NaN`.
4. **NaN/Inf Pre-Screening**: `LandmarkValidator` immediately rejects any non-finite numbers before computations begin.

---

## 5. Benchmark & Performance Profile

Benchmarked on host Windows machine with Python 3.11.15 over 1,000 extractions:

| Metric | Result | Target Budget | Status |
|---|---|---|---|
| **Mean Extraction Latency** | **119.6 $\mu\text{s}$ (0.120 ms)** | $< 1.0\text{ ms}$ | **PASSED (8x faster)** |
| **p50 Latency** | **117.0 $\mu\text{s}$ (0.117 ms)** | $< 1.0\text{ ms}$ | **PASSED** |
| **p95 Latency** | **132.4 $\mu\text{s}$ (0.132 ms)** | $< 1.5\text{ ms}$ | **PASSED** |
| **p99 Latency** | **186.2 $\mu\text{s}$ (0.186 ms)** | $< 2.0\text{ ms}$ | **PASSED** |
| **Throughput** | **8,345.8 extractions/sec** | $> 1,000\text{ ext/s}$ | **PASSED** |
| **NaN / Inf Violations** | **0 / 1,000** | 0 | **PASSED** |
| **Bitwise Determinism** | **50 / 50 consecutive runs identical** | 100% | **PASSED** |

---

## 6. Quickstart & Verification Commands

### Run Full Test Suite (36 Automated Unit Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

### Run Standalone Verification & Benchmark Suite
```powershell
.\.venv\Scripts\python.exe scripts/verify_stage4.py
```

### Run Live Interactive Feature Engineering Demo
```powershell
.\.venv\Scripts\python.exe scripts/run_feature_demo.py --camera-index 0
```
- Press `[P]` to pause / resume.
- Press `[C]` to toggle left-hand canonicalization.
- Press `[S]` to output feature vector JSON snapshot to terminal.
- Press `[Q]` or `[ESC]` to exit.
