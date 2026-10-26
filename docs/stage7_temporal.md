# Stage 7 Specification: Temporal Recognition & Stable Sign Detection

**SignBridge — Real-Time Sign Language Recognition & Translation System**

> [!IMPORTANT]
> **Stage 7 Scope Boundary**: Stage 7 implements the **temporal recognition and stabilization subsystem** that converts noisy, continuous frame-level sign classifications from Stage 6 into discrete, reliable, deduplicated sign tokens.
> **Stage 7 does NOT implement sentence formation, NLP grammar correction, word builders, spell checking, text-to-speech (TTS), speech recognition, or final UI redesign.**

---

## 1. Objective

Raw frame-by-frame sign predictions from computer vision classifiers inevitably suffer from transient jitter, single-frame misclassifications, hand transition noise, and camera frame rate variations. If left unstabilized, holding the sign for "A" for one second at 30 FPS would emit up to 30 duplicate "A" characters interspersed with momentary misclassification noise (e.g., `A A A B A A C A`).

Stage 7 resolves these issues by inserting a deterministic, stateful temporal stabilization layer between the Stage 6 ML classifier and downstream NLP components (Stage 8):

$$\text{Frame Predictions} \longrightarrow \text{Confidence Filter} \longrightarrow \text{Rolling Buffer} \longrightarrow \text{Consensus Voting} \longrightarrow \text{Dwell Gating} \longrightarrow \text{Duplicate Suppression} \longrightarrow \mathbf{\text{Stable Sign Token}}$$

---

## 2. Input Contract: `FramePrediction`

The temporal engine ingests immutable [`FramePrediction`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/temporal/contracts.py#L22-L73) objects on every processed frame:

```python
@dataclass(frozen=True)
class FramePrediction:
    class_id: int          # Discrete sign identifier (0..25), or -1 if no hand
    label: str             # Sign text label ("A".."Z"), or "" if no hand
    confidence: float      # Posterior probability in [0.0, 1.0]
    timestamp: float       # Monotonic wall-clock timestamp in seconds
    has_hand: bool = True  # Whether hand detection succeeded
    top_margin: float = 0.0# Probability margin between top-1 and top-2 candidates
```

### Ingestion Validation:
- $\text{timestamp} \ge 0.0$
- $\text{confidence} \in [0.0, 1.0]$ and strictly finite (no `NaN` or `Inf`)
- $\text{class\_id} \ge 0$ when $\text{has\_hand} == \text{True}$
- Factory methods:
  - `FramePrediction.no_hand(timestamp)`: Convenient constructor for empty frames.
  - `FramePrediction.from_prediction_result(result, timestamp, has_hand)`: Seamless bridge for Stage 6 `PredictionResult`.

---

## 3. Output Contract: `StableSign` & `TemporalFrameResult`

When a sign candidate satisfies consensus, minimum dwell duration, and duplicate suppression conditions, the engine commits and emits a [`StableSign`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/temporal/contracts.py#L76-L113) token:

```python
@dataclass(frozen=True)
class StableSign:
    class_id: int             # Sign class index (0..25)
    label: str                # Sign character label (e.g. "A")
    confidence: float         # Mean confidence across stabilization window
    start_timestamp: float    # Timestamp when candidate was first observed
    end_timestamp: float      # Timestamp when candidate was committed
    duration: float           # Total dwell duration in seconds
    frame_count: int          # Total contributing frames
    top_margin: float = 0.0   # Mean margin between top-1 and top-2 candidates
```

In addition, each call to `process_frame()` returns a rich diagnostic telemetry packet, [`TemporalFrameResult`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/temporal/contracts.py#L116-L151):

```python
@dataclass(frozen=True)
class TemporalFrameResult:
    state: TemporalState                 # IDLE, CANDIDATE, STABLE, EMITTED, WAITING_FOR_CHANGE
    current_candidate: Optional[str]     # Active candidate sign label
    candidate_confidence: float          # Rolling mean confidence of candidate
    dwell_duration_ms: float             # Accumulated dwell time in ms
    stability_ratio: float               # Window consensus agreement ratio [0.0, 1.0]
    is_hand_present: bool                # Whether hand was detected
    stable_sign: Optional[StableSign]    # Emitted token on commitment frame (else None)
    processing_latency_ms: float         # Engine execution time in ms
    frame_prediction: Optional[FramePrediction]
```

---

## 4. State Machine Architecture

The temporal engine is modeled as a deterministic finite state machine (FSM) governed by 5 lifecycle states:

```
                  ┌───────────────────────────────────────────────────────────┐
                  │                           IDLE                            │◄────────────────────────┐
                  └─────────────────────────────┬─────────────────────────────┘                         │
                                                │ Hand detected with                                    │
                                                │ confidence >= threshold                               │
                                                ▼                                                       │
                  ┌───────────────────────────────────────────────────────────┐                         │
                  │                         CANDIDATE                         │                         │
                  └──────────────────────┬──────────────┬─────────────────────┘                         │
                                         │              │ Consensus lost                                │
                                         │              │ or candidate changed                          │
                                         │              └───────────────────────────────────────────────┤
                                         │ Dwell time >= minimum_dwell_ms                               │
                                         ▼                                                              │
                  ┌───────────────────────────────────────────────────────────┐                         │
                  │                          STABLE                           │                         │
                  └─────────────────────────────┬─────────────────────────────┘                         │
                                                │ Emit StableSign token                                 │
                                                ▼                                                       │
                  ┌───────────────────────────────────────────────────────────┐                         │ Hand loss > grace_ms
                  │                          EMITTED                          │                         │ or explicit reset
                  └─────────────────────────────┬─────────────────────────────┘                         │
                                                │ Subsequent frame with                                 │
                                                │ identical sign                                        │
                                                ▼                                                       │
                  ┌───────────────────────────────────────────────────────────┐                         │
                  │                    WAITING_FOR_CHANGE                     │─────────────────────────┘
                  │                  (Duplicate Suppressed)                   │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │ New distinct sign achieves consensus
                                                ▼
                                    (Transitions to CANDIDATE)
```

### State Definitions:
1. **`IDLE`**: No hand detected or hand absent beyond grace period. Buffer contains insufficient confident frames.
2. **`CANDIDATE`**: A sign has met the confidence threshold and consensus criteria. It is actively accumulating dwell duration ($t_{\text{dwell}}$).
3. **`STABLE`**: Candidate has reached or exceeded the configured minimum dwell duration ($t_{\text{dwell}} \ge T_{\text{dwell}}$).
4. **`EMITTED`**: Transient state on the exact frame when `StableSign` is committed and dispatched to consumers.
5. **`WAITING_FOR_CHANGE`**: Sign was committed. While the user continues holding the same sign, duplicate emissions are suppressed until a sign change, timeout, or neutral drop occurs.

---

## 5. Confidence Filtering

Predictions with posterior confidence below `confidence_threshold` are filtered out:
- Grounded in Stage 6 measured test distributions:
  - Correct test predictions: $\mu = 87.08\%, \sigma = 7.65\%$
  - Misclassifications: $\mu = 52.14\%$
- Default threshold: **$\tau = 0.70$** (70%)
- Confident predictions ($\ge 0.70$) contribute to candidate tracking and rolling consensus.
- Sub-threshold frames are treated as neutral/transition frames and never establish new candidates.

---

## 6. Bounded Temporal Prediction Buffer

[`TemporalPredictionBuffer`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/temporal/buffer.py) manages a sliding temporal window of recent predictions:
- **Maximum Bounded Capacity**: `max_buffer_frames = 60` (prevents memory leaks).
- **Time-Window Pruning**: Frames older than `current_timestamp - window_duration_seconds` ($T_{\text{window}} = 400.0$ ms) are popped automatically on each insertion.
- Bounded memory footprint: consumes $< 15$ KB RAM under active continuous streaming.

---

## 7. Consensus Voting & Stability Formulation

Within the active temporal window $W$, consensus is calculated across all valid confident frames $V \subseteq W$:

$$\text{Agreement Ratio}(c) = \frac{\sum_{p \in V} \mathbb{I}[p.\text{label} = c]}{|V|}$$

$$\text{Weighted Ratio}(c) = \frac{\sum_{p \in V, p.\text{label} = c} p.\text{confidence}}{\sum_{p \in V} p.\text{confidence}}$$

A candidate sign $c^*$ reaches consensus if and only if:
1. $\text{Agreement Ratio}(c^*) \ge \text{consensus\_threshold}$ ($0.60$)
2. $|V| \ge \text{minimum\_stable\_frames}$ ($3$ frames)

This guarantees that momentary single-frame anomalies (such as $A \to B \to A$) do not disrupt the candidate stream.

---

## 8. Dwell-Time Detection

A sign must be held continuously for at least `minimum_dwell_ms` ($250.0$ ms) before it is considered committed:
- **Wall-Clock Timing**: Dwell time is computed from physical timestamps ($t - t_{\text{first\_seen}}$), **not** fixed frame counts.
- **FPS Agnostic**: Evaluates consistently across 15 FPS (webcam under heavy load), 24 FPS, 30 FPS (standard), and 60 FPS (high-speed cameras).
- Emitted token records the exact physical dwell time and total contributing frames.

---

## 9. Duplicate Suppression

When a user holds a sign steady (e.g. holding 'A' for 2 seconds):
- Frame 1 to Frame $k$ (ramp-up & dwell $\approx 250$ ms): Candidate accumulates stability.
- Frame $k$: Dwell threshold reached $\to$ **Emits `A` once**.
- Subsequent frames: `is_duplicate == True` $\to$ State transitions to `WAITING_FOR_CHANGE`. Emission of 'A' is suppressed.
- The system will **never** spam repeated letters while a sign is held statically.

---

## 10. Re-Arm & Sign Change Logic

SignBridge supports 3 configurable re-arm policies:

| Policy | Behavior | Typical Use Case |
|---|---|---|
| **`on_change`** *(Default)* | Re-arms immediately when a new distinct sign reaches consensus, or hand drops for $> \text{neutral\_dwell\_ms}$. | Standard fingerspelling and conversational sign language. |
| **`timeout`** | Re-arms if the user holds the same sign longer than `duplicate_suppression_ms` ($1200$ ms). | Assisted typing with auto-repeat. |
| **`neutral_release`** | Re-arms when the hand drops, disappears, or goes neutral for $\ge \text{neutral\_dwell\_ms}$ ($150$ ms). | Intentional double-letter signing (e.g. "OO" in "BOOK" via bounce/drop). |

---

## 11. Hand-Loss Grace Period

MediaPipe hand detection occasionally drops tracking for 1–3 frames during rapid motion or partial occlusion:
- If a hand disappears, the engine enters a grace period ($T_{\text{grace}} = 250.0$ ms).
- **Within Grace Period ($< 250$ ms)**: Active candidate state is preserved. Dwell accumulation is paused, and no tokens can be emitted while the hand is missing.
- **Exceeding Grace Period ($> 250$ ms)**: If the hand remains absent beyond 250 ms, the gesture is deemed abandoned: buffer is cleared, candidate is discarded, and state resets to `IDLE`.

---

## 12. Noise & Jitter Handling

| Source of Noise | Architectural Countermeasure | Measured Result |
|---|---|---|
| **Single-frame misclassification** | Sliding-window consensus ratio ($\ge 0.60$) | $A\,A\,A\,B\,A\,A \to A$ (100% filtered) |
| **Low-confidence prediction** | Centralized confidence filter ($\ge 0.70$) | Rejected before buffer ingestion |
| **Transient hand loss** | $250$ ms hand-loss grace window | Gesture continuity maintained |
| **Camera FPS jitter** | Monotonic timestamp clamping and wall-clock dwell | Seamless operation from 15 to 60 FPS |
| **Rapid sign switching** | Dwell-time minimum ($250$ ms) | Fast random flailing emits 0 tokens |

---

## 13. Centralized Configuration Reference

[`TemporalEngineConfig`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/temporal/config.py#L12-L65) parameters:

```python
@dataclass(frozen=True)
class TemporalEngineConfig:
    confidence_threshold: float = 0.70      # Grounded in Stage 6 test distribution
    window_duration_ms: float = 400.0       # Rolling consensus window size
    max_buffer_frames: int = 60             # Hard bound on buffer size
    consensus_threshold: float = 0.60       # Minimum agreement ratio in window
    minimum_stable_frames: int = 3          # Minimum frames needed before STABLE
    minimum_dwell_ms: float = 250.0         # Minimum dwell duration before emission
    hand_loss_grace_ms: float = 250.0       # Tolerance for brief hand dropouts
    duplicate_suppression_ms: float = 1200.0# Auto-repeat window for timeout policy
    rearm_policy: str = "on_change"         # "on_change", "timeout", "neutral_release"
    neutral_dwell_ms: float = 150.0         # Neutral duration needed to re-arm
```

---

## 14. Performance & Latency Benchmarks

Measured on host CPU across 268 simulation frames and verified under live 30 FPS camera streaming:

| Metric | Measured Value | Target Threshold | Margin |
|---|:---:|:---:|:---:|
| **Mean Per-Frame Latency** | **0.0110 ms** (11.0 $\mu$s) | $< 1.0$ ms | **90.9x faster** |
| **Median Per-Frame Latency** | **0.0086 ms** (8.6 $\mu$s) | $< 1.0$ ms | **116.3x faster** |
| **p95 Per-Frame Latency** | **0.0252 ms** (25.2 $\mu$s) | $< 2.0$ ms | **79.4x faster** |
| **Theoretical Peak Throughput** | **91,153 frames/sec** | $\ge 30$ FPS | **3,038x margin** |
| **Mean Stabilization Delay** | **266.7 ms** | $250.0 \pm 50$ ms | **Accurate** |
| **Memory Consumption** | $< 15$ KB | $< 1$ MB | **98.5% buffer** |

The temporal engine adds virtually zero computational overhead to the camera pipeline, consuming less than $0.08\%$ of the 33.3 ms frame budget at 30 FPS.

---

## 15. Simulation Test Suite Results

Executed via [`scripts/verify_stage7.py`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/scripts/verify_stage7.py):

```
======================================================================
SIMULATION SUITE RESULT: 10/10 SCENARIOS PASSED
======================================================================
[SCENARIO 1]  Consistent Sign (A x 15 @ 30 FPS = 500 ms)           -> PASS (Emitted ['A'])
[SCENARIO 2]  Single-Frame Jitter Rejection (A A A B A A A A A A)  -> PASS (Emitted ['A'])
[SCENARIO 3]  Sign Transition (A x 15, then B x 15)                -> PASS (Emitted ['A', 'B'])
[SCENARIO 4]  Duplicate Suppression (A held for 45 frames = 1.5s)  -> PASS (Emitted ['A'] only)
[SCENARIO 5]  Rapid Alternating Noise (A B A B A B A B)            -> PASS (0 emissions)
[SCENARIO 6]  Hand Loss Within Grace Period (100 ms loss < 250 ms) -> PASS (Emitted ['A'])
[SCENARIO 7]  Hand Loss Exceeding Grace (500 ms loss > 250 ms)     -> PASS (Reset + ['A'])
[SCENARIO 8]  Low-Confidence Rejection (conf=0.55 < 0.70)          -> PASS (0 emissions)
[SCENARIO 9]  FPS Variation Invariance (15, 24, 30 FPS)            -> PASS (Invariant)
[SCENARIO 10] Sequential Sign Sequence (A -> B -> C)               -> PASS (Emitted ['A', 'B', 'C'])
```

---

## 16. Real-Time Diagnostic Demo

Executed via [`scripts/run_temporal_demo.py`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/scripts/run_temporal_demo.py):
- Connects: Camera (Stage 2) $\to$ MediaPipe (Stage 3) $\to$ Features (Stage 4) $\to$ Model (Stage 6) $\to$ Temporal Engine (Stage 7).
- Interactive HUD Features:
  - Active FSM State (`IDLE`, `CANDIDATE`, `STABLE`, `WAITING_FOR_CHANGE`) with color-coded badges.
  - Live Stability Consensus Gauge & Dwell Progress Bar.
  - Recent Token History Ribbon (`[A] -> [B] -> [C]`).
  - Real-time latency telemetry breakdown across detector, feature extractor, classifier, and temporal engine.
  - Keyboard Controls: `[Q/ESC]` Quit, `[P]` Pause/Resume, `[R]` Reset State, `[C]` Toggle Confidence Info, `[D]` Toggle Diagnostic HUD.

---

## 17. Known Limitations & Stage 8 Hand-Off

1. **Static Fingerspelling Focus**: Stage 7 stabilizes static hand gestures. Dynamic signs that require continuous sweeping motion (e.g. motion letters 'J' and 'Z') are classified on their characteristic keyframe pose rather than full trajectory recognition. Full dynamic trajectory tracking belongs to dynamic gesture expansions.
2. **Hand-Off to Stage 8**:
   - Downstream Stage 8 (NLP, Token Mapping, Word & Sentence Builder) will consume the stream of emitted [`StableSign`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/temporal/contracts.py#L76-L113) tokens to build words, handle spacing, backspacing, and text reconstruction.
