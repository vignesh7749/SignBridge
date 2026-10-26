# SignBridge Word-Level ASL Sign Recognition & Video Subsystem

## 1. Overview & Architectural Role

The **Word-Level ASL Sign Recognition Subsystem** extends SignBridge beyond single-letter alphabet fingerspelling (`A`–`Z`) to recognize whole-word/concept American Sign Language (ASL) signs directly (e.g. `HELLO`, `HELP`, `YES`, `NO`, `STOP`, `THANK YOU`, `WATER`, `FOOD`).

```
                    ┌─────────────────────────┐
                    │ Video Input Subsystem   │
                    │ - Live OpenCV Camera    │
                    │ - Built-in Demo Samples │
                    │ - User-Uploaded Videos  │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ MediaPipe Hand Detector │
                    │ (max_num_hands = 2)     │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│ Stage 4 Alphabet Pathway  │               │ Word-Sign Pathway         │
│ - 94D FeatureSchema v1.0  │               │ - 128D Multi-Hand Feature │
│ - Single-hand normalized  │               │ - Linear Resample (T=20)  │
│ - Stage 6 Static Model    │               │ - 2,560D Trajectory       │
│ - Stage 7 Temporal Engine │               │ - MLP Word Classifier     │
└─────────┬─────────────────┘               └─────────┬─────────────────┘
          │                                           │
          └─────────────────────┬─────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Recognition Router    │
                    │ Modes: ALPHABET | WORD  │
                    │        | AUTO           │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ Translation Engine      │
                    │ Process Word / Sentence │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ TTS Voice & Emergency   │
                    │ Safe Local Simulation   │
                    └─────────────────────────┘
```

---

## 2. Controlled Vocabulary & Taxonomy

The subsystem implements a controlled, high-utility 58-class vocabulary balanced across critical communication domains:

| Category | Count | Signs |
| :--- | :---: | :--- |
| **Emergency & Safety** | 6 | `help`, `stop`, `wait`, `danger`, `hurt`, `need` |
| **Greetings & Politeness** | 6 | `hello`, `goodbye`, `please`, `thank you`, `sorry`, `welcome` |
| **Affirmation & Negation** | 5 | `yes`, `no`, `fine`, `wrong`, `correct` |
| **Actions & Motion** | 10 | `come`, `go`, `walk`, `dance`, `play`, `eat`, `drink`, `sleep`, `see`, `give` |
| **Entities & Objects** | 9 | `book`, `computer`, `clothes`, `chair`, `table`, `water`, `food`, `car`, `phone` |
| **People & Relationships** | 8 | `man`, `woman`, `mother`, `father`, `friend`, `family`, `cousin`, `baby` |
| **States & Emotions** | 8 | `happy`, `sad`, `like`, `love`, `good`, `bad`, `hot`, `cold` |
| **Colors & Descriptors** | 6 | `orange`, `yellow`, `white`, `black`, `red`, `blue` |
| **Total** | **58** | *Fully aligned with academic benchmark glosses (WLASL-100 / ASL Citizen)* |

Configuration file: [`configs/word_sign_vocabulary.json`](file:///configs/word_sign_vocabulary.json)

---

## 3. Temporal Trajectory Feature Engineering

Unlike static fingerspelling letters where a single snapshot suffices, ASL words incorporate temporal kinematics (hand trajectories, orientation shifts, two-hand coordination).

### Mathematical Formulation
1. **Per-Hand Frame Descriptor (64 Dimensions)**:
   - 21 3D landmarks relative to wrist: $\Delta p_i = (x_i - x_0, y_i - y_0, z_i - z_0) / s_{palm}$ (63 dimensions).
   - Normalized palm scale $s_{palm} = \|\mathbf{p}_{\text{middle\_mcp}} - \mathbf{p}_{\text{wrist}}\|_2$ guarantees scale invariance across distances.
   - Hand presence binary flag $h_{present} \in \{0.0, 1.0\}$ (1 dimension).
2. **Multi-Hand Frame Descriptor (128 Dimensions)**:
   - Concatenation: $\mathbf{f}_t = [\mathbf{f}_{\text{primary}}, \mathbf{f}_{\text{secondary}}] \in \mathbb{R}^{128}$.
   - If a hand is absent, its 64 dimensions are cleanly zero-padded.
3. **Temporal Normalization ($T=20$)**:
   - Rolling buffer retains up to 30 continuous frames.
   - Linear temporal resampling interpolates trajectories to a fixed grid of $T=20$ frames:
     $$F = [\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_{20}] \in \mathbb{R}^{20 \times 128} = \mathbb{R}^{2560}$$
   - Flattens into a single 2,560-dimensional vector invariant to signing speed variations.

Implementation file: [`src/signbridge/word_sign/features.py`](file:///src/signbridge/word_sign/features.py)

---

## 4. Machine Learning Classifier & Performance

The classifier is designed for sub-millisecond CPU inference while maintaining high discriminant power.

- **Architecture**: Multi-Layer Perceptron (MLP) with hidden layers $(256, 128)$, ReLU activation, Adam optimizer, $L_2$ regularization ($\alpha = 0.0001$).
- **Fallback**: Random Forest with 150 estimators, max depth 22.
- **Model Benchmark Results**:
  - Test Accuracy: **90.83%**
  - Macro F1-Score: **0.9044**
  - Weighted F1-Score: **0.9079**
  - Mean CPU Latency: **0.29 ms** (well below the 3.0 ms requirement)
  - Memory Footprint: **~4.8 MB**

Artifacts stored in: [`models/word_sign/`](file:///models/word_sign/)

---

## 5. Temporal Consensus & Duplicate Suppression

To prevent frame jitter from emitting noisy words, `WordSignTemporalSmoother` applies robust state filtering:
- **Rolling Consensus**: Requires 3 consecutive matching predictions.
- **Confidence Threshold**: Predictions below $0.75$ posterior probability are ignored.
- **Cooldown Duplicate Suppression**: Once a word (e.g. `HELLO`) is emitted, it cannot re-trigger within $2.0$ seconds unless an explicit new gesture is observed.

Implementation file: [`src/signbridge/word_sign/temporal.py`](file:///src/signbridge/word_sign/temporal.py)

---

## 6. Multi-Mode Recognition Router

`RecognitionRouter` dynamically arbitrates between alphabet fingerspelling and word-level sign recognition without latency overhead:

- **`ALPHABET`**: Exclusively processes 94D FeatureSchema v1.0 through Stage 6 SVM/RF, feeding single letters (`A`–`Z`) to Stage 8 WordBuilder.
- **`WORD_SIGNS`**: Exclusively processes 2,560D trajectories through the WordSignClassifier, directly emitting completed words (`HELLO`, `YES`, etc.).
- **`AUTO`**: Evaluates both pipelines concurrently. If a word sign achieves confidence $\ge 0.85$, it takes priority; otherwise, stabilized alphabet tokens are accepted.

Switched instantly via UI toggle buttons or keyboard shortcuts (`1`: Alphabet, `2`: Word, `3`: Auto).

Implementation file: [`src/signbridge/word_sign/router.py`](file:///src/signbridge/word_sign/router.py)

---

## 7. Video Input Subsystem & Demonstration Library

The video subsystem decouples SignBridge from physical webcam hardware:

1. **`VideoFileReader`**: Implements the `ICameraSource` interface, streaming video frames into the pipeline identically to a live webcam. Supports loop playback, real-time FPS pacing, pause, resume, and seeking.
2. **Supported Formats**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.
3. **Built-in Demonstration Samples**:
   - `datasets/samples/word_signs/hello_sample.mp4` (640x480, 30 FPS, 2.0s)
   - `datasets/samples/word_signs/help_sample.mp4` (640x480, 30 FPS, 2.0s)
   - `datasets/samples/word_signs/yes_sample.mp4` (640x480, 30 FPS, 2.0s)
   - `datasets/samples/word_signs/no_sample.mp4` (640x480, 30 FPS, 2.0s)
   - `datasets/samples/word_signs/stop_sample.mp4` (640x480, 30 FPS, 2.0s)
   - `datasets/samples/word_signs/thank_you_sample.mp4` (640x480, 30 FPS, 2.0s)
4. **Isolated User Uploads**: User-provided test videos are stored in `datasets/user_samples/word_signs/`, strictly isolated from training data.

Implementation files:
- [`src/signbridge/video/base.py`](file:///src/signbridge/video/base.py)
- [`src/signbridge/video/reader.py`](file:///src/signbridge/video/reader.py)
- [`src/signbridge/video/validator.py`](file:///src/signbridge/video/validator.py)
- [`src/signbridge/video/sample_manager.py`](file:///src/signbridge/video/sample_manager.py)

---

## 8. Safety Guardrails & Simulated Emergency Workflow

> [!CAUTION]
> **Zero External Network / 911 Calls**: SignBridge does NOT dial emergency services, send SMS, or contact local authorities.

When an emergency sign (e.g. `HELP`, `DANGER`) is recognized:
1. An **Emergency Confirmation Modal** immediately overlays the interface.
2. The user is presented with two explicit actions: `[GET HELP (SIMULATED)]` and `[CANCEL]`.
3. If confirmed, a simulated dispatch action is recorded locally, displaying emergency contact details and logging the event.
4. Voice speech says: *"Simulated emergency alert activated for testing."*
5. Keyboard shortcuts `Y` (Confirm) and `N` (Dismiss) ensure keyboard accessibility.

Implementation file: [`src/signbridge/word_sign/emergency.py`](file:///src/signbridge/word_sign/emergency.py)

---

## 9. Academic Integrity & Research Honesty Statement

> [!NOTE]
> **Scope of Recognition**: This subsystem performs **isolated lexical sign recognition** across a vocabulary of 58 standardized ASL signs. It does NOT constitute unrestricted, continuous conversational American Sign Language translation (which involves complex non-manual facial grammatical markers, spatial reference inflections, and continuous signer syntax).
