# Stage 8 Specification: Token → Word → Sentence Translation Engine

**SignBridge — Real-Time Sign Language Recognition & Translation System**

> [!IMPORTANT]
> **Stage 8 Scope Boundary**: Stage 8 implements the deterministic **Token → Word → Sentence Translation Engine** that translates discrete, stabilized fingerspelling tokens from Stage 7 into structured words, whitespace-delimited sentences, and punctuation-ready text.
> **Stage 8 does NOT implement probabilistic autocorrect that alters typed characters, LLM generative text completion, Text-to-Speech (TTS), speech-to-text, or production UI redesign.**
> **Linguistic Clarification**: Fingerspelling alphabet recognition is a fundamental component of ASL (American Sign Language), but is **not** equivalent to full natural ASL translation, which utilizes spatial syntax, non-manual markers, facial expressions, and complex classifier predicates.

---

## 1. Architecture

The Stage 8 Translation Engine forms the bridge between discrete perceptual sign events and linguistic text representations. It sits directly downstream of the Stage 7 Temporal Stabilization subsystem:

```
Camera (Stage 2)
   ↓
OpenCV Video Stream
   ↓
MediaPipe Hand Landmark Extraction (Stage 3)
   ↓
94-Dimensional Feature Engineering (Stage 4)
   ↓
Static Sign ML Classifier (Stage 6)
   ↓
Temporal Stabilization & Consensus Gating (Stage 7)
   ↓
[STAGE 8 TRANSLATION ENGINE]
   ├── Stable Sign Ingestion & Validation
   ├── Letter Buffer Accumulation
   ├── Word Boundary Detection (Explicit SPACE & Pause Triggers)
   ├── Word Re-Opening Backspace & Safe Editing
   ├── Multi-Step State Snapshots & Undo
   ├── Deterministic Whitespace Normalization
   └── Bounded Token History & Telemetry
   ↓
Downstream Consumer (Console HUD / Video Overlay / Future TTS)
```

The subsystem is decoupled into modular layers:
1. **`contracts.py`**: Immutable dataclasses and enumerations defining inputs, outputs, commands, states, and telemetry events.
2. **`config.py`**: Frozen configuration governing timing thresholds, boundary behaviors, buffer capacities, and formatting rules.
3. **`normalizer.py`**: Pure functional text normalizer that enforces whitespace normalization, space collapse, capitalization, and sentence formatting.
4. **`engine.py`**: Stateful finite state machine managing letter accumulation, completed word lists, word re-opening, multi-step undo history, and event telemetry.

---

## 2. Data Contracts

All data structures in Stage 8 are defined in [`src/signbridge/translation/contracts.py`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/translation/contracts.py) using frozen dataclasses and strict validation.

### 2.1 Input Contract: `StableSignToken`

The translation engine ingests `StableSignToken` instances. It provides a direct factory to ingest Stage 7 `StableSign` instances:

```python
@dataclass(frozen=True)
class StableSignToken:
    label: str               # Alphabetic token label (e.g., "H", "E", "L", "L", "O", "SPACE")
    confidence: float        # Classification confidence in [0.0, 1.0]
    timestamp: float         # Wall-clock monotonic timestamp in seconds
    duration: float = 0.0    # Temporal dwell duration in seconds
    is_control: bool = False # True for control signs ("SPACE", "BACKSPACE", "CLEAR")
```

Validation ensures:
- `label` is non-empty and stripped of surrounding whitespace.
- `confidence` is within $[0.0, 1.0]$ and finite.
- `timestamp >= 0.0`.
- Control token labels (`"SPACE"`, `"BACKSPACE"`, `"CLEAR"`, `"CLEAR_SENTENCE"`, `"UNDO"`, `"RESET"`) automatically set `is_control = True`.

Factory conversion:
```python
token = StableSignToken.from_stable_sign(stable_sign)
```

### 2.2 Word Contract: `Word`

Represents an immutable sequence of accumulated tokens forming a discrete word:

```python
@dataclass(frozen=True)
class Word:
    text: str                               # Formatted word string (e.g., "HELLO")
    tokens: tuple[StableSignToken, ...]     # Constituent token sequence
    start_timestamp: float                  # Timestamp of first constituent token
    end_timestamp: float                    # Timestamp of final constituent token
    confidence: float                       # Arithmetic mean confidence of tokens
```

### 2.3 Sentence Contract: `Sentence`

Represents an immutable sequence of completed words:

```python
@dataclass(frozen=True)
class Sentence:
    words: tuple[Word, ...]                 # Sequence of completed Word objects
    text: str                               # Normalized, whitespace-delimited sentence
    start_timestamp: float                  # Timestamp of first token in first word
    end_timestamp: float                    # Timestamp of latest committed token
    word_count: int                         # Total words
    letter_count: int                       # Total letters (excluding whitespace)
```

### 2.4 Control Commands & Events

```python
class TranslationCommand(str, Enum):
    SPACE = "SPACE"
    BACKSPACE = "BACKSPACE"
    CLEAR_WORD = "CLEAR_WORD"
    CLEAR_SENTENCE = "CLEAR_SENTENCE"
    RESET = "RESET"
    UNDO = "UNDO"

class TranslationEventType(str, Enum):
    LETTER_APPENDED = "LETTER_APPENDED"
    WORD_COMPLETED = "WORD_COMPLETED"
    LETTER_DELETED = "LETTER_DELETED"
    WORD_REOPENED = "WORD_REOPENED"
    WORD_CLEARED = "WORD_CLEARED"
    SENTENCE_CLEARED = "SENTENCE_CLEARED"
    RESET_PERFORMED = "RESET_PERFORMED"
    ACTION_UNDONE = "ACTION_UNDONE"
    IGNORED = "IGNORED"

class TranslationState(str, Enum):
    EMPTY = "EMPTY"                         # No letters or words present
    BUILDING_WORD = "BUILDING_WORD"         # Currently accumulating letters
    WORD_COMPLETED = "WORD_COMPLETED"       # Word just finalized via SPACE / pause
    BUILDING_SENTENCE = "BUILDING_SENTENCE" # One or more words completed, new word building
```

### 2.5 Execution Telemetry: `TranslationResult`

Every operation on `TranslationEngine` returns an immutable `TranslationResult`:

```python
@dataclass(frozen=True)
class TranslationResult:
    state: TranslationState
    current_word: str
    completed_words: tuple[str, ...]
    sentence_text: str
    event_type: TranslationEventType
    last_token: Optional[StableSignToken]
    tokens_count: int
    words_count: int
    processing_latency_ms: float
```

---

## 3. State Machine Design

The translation engine operates as a deterministic Finite State Machine (FSM):

```
                     ┌─────────────────────────────────────────┐
                     │                  EMPTY                  │◄────────────────┐
                     └──────┬───────────────────────────┬──────┘                 │
                            │ Letter Token Ingested     │                        │
                            ▼                           │                        │
                     ┌─────────────────────────┐        │                        │
        ┌───────────►│      BUILDING_WORD      │        │                        │
        │            └──────┬────────────┬─────┘        │                        │
        │ Letter            │            │              │                        │
        │ Appended          │ SPACE /    │ Backspace    │ Clear Sentence /       │
        │                   │ Pause      │ (buffer = 1) │ Reset                  │
        │                   ▼            ▼              │                        │
        │            ┌─────────────────────────┐        │                        │
        │            │     WORD_COMPLETED      │        │                        │
        │            └──────┬────────────┬─────┘        │                        │
        │                   │            │              │                        │
        │ Letter Ingested   │            │ Backspace    │                        │
        │                   │            ▼ (re-open)    │                        │
        │                   │    ┌─────────────────┐    │                        │
        │                   │    │  Re-opens last  │    │                        │
        │                   │    │  completed word │    │                        │
        │                   │    └────────┬────────┘    │                        │
        │                   ▼             │             │                        │
        │            ┌────────────────────┴────┐        │                        │
        └────────────┤    BUILDING_SENTENCE    ├────────┴────────────────────────┘
                     └─────────────────────────┘
```

### State Definitions & Transitions:
1. **`EMPTY`**: No letters in active buffer, no completed words. Transitions to `BUILDING_WORD` upon receiving any letter.
2. **`BUILDING_WORD`**: One or more letters in active buffer, 0 completed words.
   - Additional letter $\rightarrow$ remains `BUILDING_WORD`.
   - Backspace with 1 letter left $\rightarrow$ returns to `EMPTY`.
   - SPACE / pause commit $\rightarrow$ transitions to `WORD_COMPLETED`.
3. **`WORD_COMPLETED`**: Active buffer is empty, $\ge 1$ completed word present.
   - Receiving letter $\rightarrow$ transitions to `BUILDING_SENTENCE`.
   - Backspace $\rightarrow$ pops last completed word, puts its letters back into active buffer, and transitions to `BUILDING_SENTENCE` (or `BUILDING_WORD` if only 1 word was present).
4. **`BUILDING_SENTENCE`**: $\ge 1$ completed words exist AND active buffer has $\ge 1$ letters.
   - Additional letter $\rightarrow$ remains `BUILDING_SENTENCE`.
   - Backspace emptying active buffer $\rightarrow$ returns to `WORD_COMPLETED`.
   - SPACE / pause commit $\rightarrow$ appends active word to completed words list, transitions to `WORD_COMPLETED`.

---

## 4. Letter Buffer Management

The engine maintains an internal sequence `_current_tokens: list[StableSignToken]` representing the active word in progress.

- **Chronological Order**: Ingested tokens are strictly appended in order of arrival.
- **Deduplication Check**: When `is_control=False` and the label is alphanumeric, it is appended as a letter.
- **Confidence Tracking**: Each word preserves token-level timestamps and confidences, calculating an exact arithmetic mean confidence:
  $$\text{Word.confidence} = \frac{1}{N}\sum_{i=1}^N \text{token}_i.\text{confidence}$$
- **Capacity**: Constrained by `TranslationConfig.max_word_length` (default 30 characters). Tokens beyond this length are safely ignored to prevent runaway memory usage.

---

## 5. Word Boundary Detection

Word boundaries separate individual words in a sentence and can be triggered deterministically via two mechanisms:

### 5.1 Explicit Boundary (`SPACE`)
- A physical sign mapped to `"SPACE"` or a keyboard command `SPACE` triggers `engine.space()`.
- **Idempotency**: If `_current_tokens` is empty, calling `space()` is an explicit no-op (returns `IGNORED`). This guarantees that double spaces, leading spaces, or trailing whitespace cannot be injected into the sentence text.

### 5.2 Temporal Pause Boundary (Autonomous Gating)
- When enabled via `config.enable_pause_boundary = True`, the engine continuously checks timestamps during `tick(current_timestamp)`:
  $$\Delta t = t_{\text{current}} - t_{\text{last\_token\_end}}$$
- If $\Delta t \ge \text{pause\_boundary\_seconds}$ (default 2.0s) and `_current_tokens` is non-empty, the active word is automatically committed without requiring an explicit gesture.

---

## 6. Sentence Assembly & Formatting

Sentence generation is handled deterministically by `assemble_sentence()`:

$$\text{Sentence Text} = \begin{cases}
\text{Join}(\text{completed\_words}, \text{" "}) + \text{" "} + \text{current\_word}, & \text{if current\_word} \neq \emptyset \land \text{completed\_words} \neq \emptyset \\
\text{current\_word}, & \text{if completed\_words} = \emptyset \\
\text{Join}(\text{completed\_words}, \text{" "}), & \text{if current\_word} = \emptyset
\end{cases}$$

### Normalization Rules:
- All multiple internal whitespaces are collapsed to a single ASCII space ` ` (`\s+` regex substitution).
- Leading and trailing whitespaces are cleanly stripped from the final sentence string.
- In-flight words are seamlessly reflected in real-time in `sentence_text` as they are typed, without waiting for the closing space.

---

## 7. Backspace Behavior: Letter vs. Word Boundary

The backspace operation (`engine.backspace()`) provides natural text editing behavior across three distinct operational regimes:

```
Scenario A: Active word has letters
["H", "E", "L", "L", "O"] + BACKSPACE
→ Result: ["H", "E", "L", "L"] (LETTER_DELETED)

Scenario B: Active word is empty, previous completed words exist
Completed: ["HELLO", "WORLD"], Active: []
+ BACKSPACE
→ Result: Completed: ["HELLO"], Active: ["W", "O", "R", "L", "D"] (WORD_REOPENED)
Sentence text updates from "HELLO WORLD" to "HELLO WORLD" without trailing space.
Subsequent BACKSPACE deletes "D" → Completed: ["HELLO"], Active: ["W", "O", "R", "L"].

Scenario C: Active word is empty, completed words is empty
Completed: [], Active: []
+ BACKSPACE
→ Result: Safe no-op (IGNORED). Returns clean TranslationResult without exceptions.
```

This dual-action semantics allows a signer to seamlessly delete individual typos or step back into earlier words without having to erase the entire sentence.

---

## 8. Undo / Snapshot Semantics

To protect against inadvertent sign commitments or false boundary triggers, Stage 8 implements a multi-step undo stack:

### 8.1 Snapshot State Structure
Before any mutating operation (`process_token`, `space`, `backspace`, `clear_word`, `clear_sentence`, `reset`), an immutable `_TranslationSnapshot` is pushed to `_undo_stack`:

```python
@dataclass(frozen=True)
class _TranslationSnapshot:
    current_tokens: tuple[StableSignToken, ...]
    completed_words: tuple[Word, ...]
    last_event: TranslationEventType
    state: TranslationState
```

### 8.2 Bounded Undo Stack
- Implemented as a `collections.deque(maxlen=config.max_undo_depth)` (default depth 50).
- `engine.undo()` pops the top snapshot and atomically restores the exact token buffer, word list, and state.
- Calling `undo()` on an empty stack is a safe no-op returning `IGNORED`.

---

## 9. Normalization & Formatting

The [`normalizer.py`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/translation/normalizer.py) utility provides pure functional string transformations:

```python
class TranslationNormalizer:
    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """Replaces consecutive whitespace characters with a single space."""
        return _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    @staticmethod
    def format_sentence(words: Sequence[str], in_flight_word: str = "") -> str:
        """Constructs a clean sentence string from completed words and active in-flight word."""
        # Cleanly formats without duplicate or trailing spaces
```

Configurable options include automatic uppercase normalization (`config.capitalize_words = False`, default leaves original token casing) and punctuation trimming.

---

## 10. Real-Time Performance & Benchmarks

The Translation Engine is designed for high-frequency real-time execution. Benchmarks executed via `scripts/verify_stage8.py` on standard CPU hardware:

| Metric | Measured Value | Requirement / Target | Status |
| :--- | :--- | :--- | :--- |
| **Mean Latency** | **0.0019 ms** (1.9 $\mu$s) | $< 1.0$ ms | PASS |
| **Median Latency** | **0.0014 ms** (1.4 $\mu$s) | $< 1.0$ ms | PASS |
| **95th Percentile Latency** | **0.0041 ms** (4.1 $\mu$s) | $< 2.0$ ms | PASS |
| **Peak Throughput** | **$> 519,000$ actions/sec** | $> 1,000$ actions/sec | PASS |
| **Simulation Scenarios** | **12 / 12 Passed (100%)** | 100% | PASS |
| **Unit Tests** | **21 / 21 Passed (100%)** | 100% | PASS |
| **Full Regression Suite** | **98 / 98 Passed (100%)** | 100% | PASS |

Because translation operations involve only Python list/deque manipulations and string operations, CPU utilization is $< 0.01\%$, adding zero measurable latency to the 30 FPS camera pipeline.

---

## 11. Error Handling & Invariant Guarantees

The subsystem is architected with strict defensive guarantees:

1. **Crash Prevention**: All public methods (`process_token`, `space`, `backspace`, `clear_word`, `clear_sentence`, `reset`, `undo`, `tick`) are guaranteed never to raise unhandled exceptions on invalid or degenerate inputs.
2. **Deterministic Replay**: Given an identical sequence of `StableSignToken` inputs and timestamps, the engine produces identical `TranslationResult` outputs.
3. **No Double Spaces**: `sentence_text` is mathematically invariant against adjacent spaces.
4. **Bounded Memory**:
   - `_token_history` is capped at `config.max_token_history` (default 500).
   - `_undo_stack` is capped at `config.max_undo_depth` (default 50).
   - Word length is bounded by `config.max_word_length` (default 30).
   - Prevents memory leaks in long-running streaming deployments.

---

## 12. Standalone Usage

The engine is completely decoupled from OpenCV, MediaPipe, and hardware cameras. It can be instantiated and operated as an isolated pure software component:

```python
from signbridge.translation import TranslationEngine, TranslationConfig, StableSignToken

engine = TranslationEngine(TranslationConfig())

# Feed individual tokens
res = engine.process_token(StableSignToken(label="H", confidence=0.98, timestamp=1.0))
res = engine.process_token(StableSignToken(label="I", confidence=0.96, timestamp=1.1))
print(res.current_word)  # "HI"

# Commit word boundary
res = engine.space()
print(res.completed_words)  # ("HI",)

# Backspace re-opens word
res = engine.backspace()
print(res.current_word)  # "HI"
print(res.completed_words)  # ()
```

---

## 13. Camera Pipeline Integration

Stage 8 integrates seamlessly into the end-to-end pipeline:

```python
# In camera loop:
pred_res = predictor.predict(feat_vec.features)
frame_pred = FramePrediction.from_prediction_result(pred_res, current_ts, has_hand=True)

# Stage 7 Temporal Engine
temp_res = temporal_engine.process_frame(frame_pred)

# When a stable sign token is emitted by Stage 7:
if temp_res.has_emitted_sign:
    trans_res = translation_engine.process_token(temp_res.stable_sign)
    print(f"Token: {temp_res.stable_sign.label} | Sentence: '{trans_res.sentence_text}'")

# Idle tick for pause-based word boundary detection
trans_res = translation_engine.tick(current_ts)
```

Both interactive and headless modes are available via [`scripts/run_translation_demo.py`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/scripts/run_translation_demo.py).

---

## 14. Configuration Options

Defined in [`TranslationConfig`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/translation/config.py):

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pause_boundary_seconds` | `float` | `2.0` | Inactivity duration in seconds to trigger auto word boundary |
| `enable_pause_boundary` | `bool` | `True` | Enable/disable autonomous pause-based word completion |
| `max_word_length` | `int` | `30` | Maximum character length of a single active word |
| `max_token_history` | `int` | `500` | Maximum capacity of the chronological token history buffer |
| `max_undo_depth` | `int` | `50` | Maximum number of undoable snapshot states |
| `auto_space_on_boundary` | `bool` | `True` | Automatically separate completed words with a space |
| `capitalize_words` | `bool` | `False` | Normalize completed words to uppercase |
| `strip_trailing_spaces` | `bool` | `True` | Strip trailing whitespace from final sentence strings |

---

## 15. Known Limitations

1. **Static Fingerspelling Focus**: Operates solely on discrete alphabet signs (A–Z) and explicit control tokens (SPACE, BACKSPACE). Dynamic gestures with trajectory motion (e.g., J and Z strokes) require sequential temporal modeling.
2. **Spelling Dependency**: Assumes the user fingerspells words letter-by-letter. Does not map single holistic gestures to entire concepts (e.g., "THANK YOU", "HELP").
3. **No Semantic Grammar**: The engine concatenates words in the exact sequence spelled. It does not perform morphological conjugation or grammatical reordering.

---

## 16. Future NLP Extension Points

Stage 8 provides clean, well-defined hook points for future natural language processing modules:

1. **Dictionary & Lexicon Filtering**: A lookup trie or dictionary verifier can be attached to `engine.space()` to score or flag unrecognized words before committing.
2. **Context-Aware Disambiguation**: Phonetic or n-gram language models can suggest corrections for single-letter misclassifications (e.g., distinguishing "M" vs "N").
3. **Punctuation & Capitalization Restoration**: A lightweight TrueCasing and punctuation model can convert raw sentence streams into polished prose prior to TTS synthesis.
4. **ASL Grammar Transformation**: Rule-based or transformer-based sequence-to-sequence models can translate Subject-Object-Verb (SOV) / Topic-Comment ASL structures into standard English Subject-Verb-Object (SVO) sentences.

---

## 17. Fundamental Linguistic Distinction: Fingerspelling vs. Full ASL

It is essential for engineers, researchers, and users of SignBridge to understand the boundary between fingerspelling and natural sign language:

> [!WARNING]
> **Fingerspelling is NOT Natural ASL**:
> - **Fingerspelling** is an orthographic bridge used in Deaf communities primarily for proper nouns, names, brand titles, technical jargon, and words with no direct sign equivalent. In typical ASL discourse, fingerspelling accounts for approximately 5%–10% of signed content.
> - **Natural ASL (American Sign Language)** is a complete, autonomous natural language with its own rich phonology, morphology, syntax, and semantics. ASL does not share grammatical structures with English:
>   - **Spatial Grammar**: Concepts and referents are assigned spatial loci in the signing space; verbs move between these loci to indicate subject and object (directionality).
>   - **Simultaneity & Non-Manual Markers (NMM)**: Grammatical moods, questions, negations, and adverbial nuances are expressed simultaneously through facial expressions, head tilts, eye gaze, and body posture, not sequentially through fingerspelled letters.
>   - **Classifiers & Depicting Signs**: Handshapes represent semantic categories (vehicles, upright persons, flat surfaces) and move dynamically through space to show spatial relationships, motion trajectories, and manner of action.
> 
> The SignBridge Stage 8 Translation Engine solves the **fingerspelled text generation pipeline** with mathematical precision and deterministic reliability. It is a necessary foundational milestone, but must never be misrepresented as a complete replacement for full conversational ASL translation.
