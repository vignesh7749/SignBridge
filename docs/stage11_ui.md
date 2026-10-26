# SignBridge — Stage 11: Professional UI/UX, Accessibility & Application Experience

## 1. Executive Summary & UI Architecture Overview

Stage 11 elevates SignBridge from a functional computer vision pipeline into a production-grade, accessible, and aesthetically refined desktop application. Designed specifically for university faculty evaluations, academic demonstrations, and real-world accessibility applications, the Stage 11 user interface delivers an immersive, distraction-free environment while strictly preserving the integrity of all underlying machine learning, computer vision, and speech subsystems.

### Core Architectural Principles
- **Strict Decoupling**: The presentation layer (`src/signbridge/ui/`) is cleanly decoupled from the computer vision pipeline (`src/signbridge/pipeline/`). The UI reads from the immutable `PipelineRuntimeState` snapshot without mutating internal inference or temporal buffers directly.
- **Zero Heavy GUI Dependencies**: Avoids bulky, non-deterministic frameworks (e.g., PyQt, Tkinter, Electron) in favor of high-performance, native NumPy and OpenCV rasterization. Steady-state full-dashboard rendering executes in **~4.03 ms**, maintaining $> 240$ FPS render capability and zero camera frame drops.
- **100% Offline & Private**: Zero cloud calls, zero external web requests, and zero telemetry tracking. All rendering and speech synthesis remain local.
- **Universal Dual Input**: Every interactive control supports simultaneous point-and-click mouse interaction and ergonomic single-key keyboard shortcuts (`SPACE`, `BS`, `C`, `U`, `P`, `R`, `S`, `X`, `T`, `O`, `H`, `D`, `ESC`).
- **Comprehensive Accessibility**: Built-in support for WCAG AAA High-Contrast mode, Reduced Motion mode, dynamic onboarding tutorials, and multi-tab interactive guidance.

---

## 2. Design Philosophy: Obsidian & High-Contrast Design Systems

Stage 11 implements two complete, mathematically balanced visual themes defined in `src/signbridge/ui/theme.py`:

### 1. Obsidian Professional Theme (`OBSIDIAN_THEME`)
- **Canvas Base**: Deep obsidian/charcoal `(24, 18, 14)` BGR with surface hierarchy `(34, 26, 20)`.
- **Accent Hierarchy**:
  - *Primary Accent (Cyan / Teal)*: `(235, 175, 56)` BGR — brand identity, focal badges, hotkey indicators.
  - *Secondary Accent (Emerald Green)*: `(120, 215, 60)` BGR — confirmed signs, stable tokens, word completions, ready status.
  - *Warning Accent (Amber)*: `(50, 170, 245)` BGR — active dwelling, in-progress recognition, paused state.
  - *Error Accent (Crimson)*: `(70, 70, 235)` BGR — hardware disconnection, confidence drops, isolated errors.
  - *Muted Accent (Slate Grey)*: `(95, 80, 70)` BGR — inactive state, disabled controls.
- **Typography Scale**: Clean sans-serif hierarchy using OpenCV Hershey Simplex with crisp anti-aliasing (`cv2.LINE_AA`).

### 2. High-Contrast Accessibility Theme (`HIGH_CONTRAST_THEME`)
- **WCAG AAA Compliance**: Exceeds the 7:1 contrast ratio threshold for all text and UI boundaries.
- **Color Palette**: Pure pitch black background `(0, 0, 0)`, pure white cards/text `(255, 255, 255)`, high-visibility neon yellow accent `(0, 255, 255)`, and solid 2px borders.
- **Legibility**: Optimizes visibility for users with low vision, photophobia, or harsh ambient lighting conditions.

---

## 3. Canvas Layout Architecture

The application renders on a responsive 16:10 or 16:9 canvas (default 1280x800) organized into seven distinct functional regions managed by `DashboardLayout` (`src/signbridge/ui/layout.py`):

```
+---------------------------------------------------------------------------------------------+
| [● SIGNBRIDGE] [PRO]  [● HAND ACTIVE]            [CAM: 30 FPS] [VOICE: READY] [⚙] [?] [ESC]|
+-------------------------------------------------------------+-------------------------------+
|                                                             | [ RECOGNITION (Model v1.0)  ] |
|                                                             |   +-------+  Active Sign: 'H' |
|                                                             |   |   H   |  Confidence: 96.5%|
|                                                             |   +-------+  Dwell: [███████] |
|                 CAMERA VIEWPORT (61% Width)                 +-------------------------------+
|               - Natural 640x480 RGB Video Feed              | [ TRANSLATION ENGINE        ] |
|               - Mirrored User Interaction                   |   Current Word:  HELL         |
|               - 21-Landmark Vector Hand Skeleton            |   Final Sentence:             |
|               - Bounding Box & Palm Tracking Anchor         |   "HELLO WORLD HOW ARE YOU"   |
|                                                             +-------------------------------+
|                                                             | [ VOICE OUTPUT (SAPI5)      ] |
|                                                             |   [ Speak (S) ] [ Stop (X) ]  |
|                                                             +-------------------------------+
|                                                             | [ QUICK ACTIONS             ] |
|                                                             |   [Space] [Delete] [Clear]    |
|                                                             |   [ Undo] [ Pause] [ Reset]   |
+-------------------------------------------------------------+-------------------------------+
| [STATUS STRIP] Ready. Hold sign steady to begin.                                            |
+---------------------------------------------------------------------------------------------+
```

### Layout Breakdown
1. **Header Bar (Top, 48px)**:
   - Application logo, brand identity dot, and operational mode badge (`PRO` / `DEMO`).
   - Dynamic status indicator (`HAND ACTIVE`, `DWELLING...`, `SIGN EMITTED`, `SPEAKING`).
   - Real-time telemetry: Camera FPS, Voice engine status, Settings button (`[O]`), Help button (`[H]`).
2. **Camera Viewport (Left, 61% width)**:
   - Smoothly scales camera frames maintaining aspect ratio with letterboxing.
   - Overlays the 21 MediaPipe hand landmarks and bone connections with dynamic confidence-tinted joints.
   - Hand bounding box and palm center tracking dot.
3. **Recognition Panel (Top Right)**:
   - Large focal character box displaying the current active sign candidate.
   - Two-stage progress bars: Raw Classifier Confidence meter and Temporal Dwell Progress meter.
4. **Translation Panel (Middle Right)**:
   - Active in-progress word builder highlighted in gold/amber.
   - Finalized sentence transcript with high-contrast text and multi-line wrapping.
5. **Voice Output Panel (Lower Right)**:
   - Displays offline TTS engine status, active utterance state, and last spoken sentence.
   - Instant action buttons: `Speak` (`[S]`), `Stop` (`[X]`), and `Mute/Unmute` (`[T]`).
6. **Quick Actions Toolbar (Bottom Right)**:
   - 2x3 tactile button grid: `Word Break` (`[SPACE]`), `Delete` (`[BS]`), `Clear` (`[C]`), `Undo` (`[U]`), `Pause/Resume` (`[P]`), `Reset All` (`[R]`).
7. **Footer Status Strip (Bottom, 34px)**:
   - Animated notification toasts with progress expiration indicators.
   - Real-time contextual user guidance.

---

## 4. Vector Component Library & Geometric Hit-Testing

All UI widgets are implemented in `src/signbridge/ui/components.py` using direct NumPy/OpenCV matrix operations:

- **`hit_test(mx, my, bx, by, bw, bh)`**: Pure geometric point-in-rect function with $O(1)$ computation.
- **`draw_card`**: Shaded container surfaces with configurable 1px or 2px anti-aliased borders.
- **`draw_card_header`**: Panel title banners with category accent bullets and sub-captions.
- **`draw_button`**: Interactive buttons with hover detection, click depression state, hotkey indicator pills, and disabled states.
- **`draw_progress_bar`**: Vectorized horizontal progress meters with clamped ranges $[0.0, 1.0]$.
- **`draw_badge`**: Dynamic-width pill badges for telemetry metrics, status codes, and shortcuts.
- **`draw_switch`**: iOS-style modern toggle switches with sliding circular knobs and color transitions.

---

## 5. Interactive Modal Dialogs

`src/signbridge/ui/overlays.py` provides modal dialog overlays rendered over a dimmed 85% translucent backdrop:

### 1. Settings Dialog (`ActiveModal.SETTINGS` — Shortcut `[O]`)
- **Voice Output Toggle**: Enable or mute offline TTS voice synthesis.
- **High-Contrast Theme Toggle**: Switch between Obsidian and WCAG AAA High-Contrast themes.
- **Reduced Motion Toggle**: Disable transition effects and animated skeleton overlays.
- **Hand Skeleton Overlay Toggle**: Hide or display MediaPipe landmark points and skeleton lines.
- **Speech Rate Controls**: Stepper buttons to increment/decrement speech rate (100–250 WPM).
- **Speech Volume Controls**: Stepper buttons to adjust TTS volume (0%–100%).
- **Reset to Defaults**: Restore initial accessibility and audio configurations.

### 2. Help & Reference Dialog (`ActiveModal.HELP` — Shortcut `[H]`)
- **Tab 0: Overview**: High-level introduction to SignBridge architecture and workflow.
- **Tab 1: Keyboard Shortcuts**: Complete index of all single-key shortcuts and functions.
- **Tab 2: Hand Positioning Guide**: Lighting instructions, camera distance recommendations, and finger orientation tips.
- **Tab 3: ASL Linguistic Notice**: Educational disclaimer acknowledging fingerspelling vs. full American Sign Language grammar and non-manual markers.

### 3. Onboarding Tutorial Dialog (`ActiveModal.ONBOARDING` — CLI `--onboarding`)
- 5-step visual guide welcoming new users and walking through hand positioning, holding signs for dwell, building words, and triggering speech playback.

---

## 6. State Management & Decoupling

SignBridge strictly separates presentation state from pipeline state:
- **`PipelineRuntimeState`** (`src/signbridge/pipeline/state.py`): Read-only, thread-safe, immutable snapshot produced by `SignBridgePipeline.get_state()`. Contains camera FPS, detection telemetry, prediction scores, dwell progress, and translation text.
- **`UIState`** (`src/signbridge/ui/state.py`): Tracks interactive UI state: active modal (`ActiveModal`), mouse coordinates (`mouse_x, mouse_y`), hovered and clicked button IDs, active toast notifications (`NotificationToast`), and accessibility flags (`high_contrast`, `reduced_motion`).

---

## 7. Dual Input Handling: Mouse & Keyboard Synchronization

Every user action can be triggered seamlessly via mouse or keyboard:

| Action | Mouse Target | Keyboard Shortcut | Pipeline Invocation |
| :--- | :--- | :--- | :--- |
| **Word Break** | Quick Actions `Word Break` | `[SPACE]` | `pipeline.space()` |
| **Delete Letter / Word** | Quick Actions `Delete` | `[BACKSPACE]` | `pipeline.backspace()` |
| **Clear Sentence** | Quick Actions `Clear` | `[C]` | `pipeline.clear_sentence()` |
| **Undo Translation** | Quick Actions `Undo` | `[U]` | `pipeline.undo()` |
| **Pause / Resume** | Quick Actions `Pause/Resume` | `[P]` | `pipeline.pause()` / `pipeline.resume()` |
| **Reset System** | Quick Actions `Reset All` | `[R]` | `pipeline.reset()` |
| **Speak Sentence** | Voice Card `Speak` | `[S]` | `pipeline.speak_current_sentence()` |
| **Stop Speech** | Voice Card `Stop` | `[X]` | `pipeline.stop_speech()` |
| **Mute / Unmute Voice** | Voice Card `Mute/Unmute` | `[T]` | `pipeline.toggle_voice()` |
| **Open Settings** | Header `Settings` | `[O]` | `ui_state.open_modal(SETTINGS)` |
| **Open Help** | Header `Help` | `[H]` | `ui_state.open_modal(HELP)` |
| **Toggle Demo Mode** | — | `[D]` | `demo_driver.start()` / `stop()` |
| **Close Modal / Quit** | Modal `[X]` | `[ESC]` | Closes modal or exits app |

---

## 8. High Contrast & WCAG AAA Accessibility Compliance

SignBridge was engineered to meet modern accessibility standards:
- **Contrast Ratios**: The High-Contrast theme achieves pure black `#000000` to pure white `#FFFFFF` (contrast ratio 21:1), well above the WCAG AAA 7:1 minimum.
- **Colorblind-Safe Cues**: All statuses and buttons pair color changes with text labels and hotkey badges. No state is conveyed through color alone.
- **Keyboard Traversal**: 100% of application features can be operated without touching a mouse.

---

## 9. Reduced Motion & Performance Optimizations

When `--reduced-motion` or the Settings toggle is enabled:
- Transient animation effects and dynamic skeleton pulsing are suppressed.
- Overlays render statically without visual noise, easing sensory fatigue.
- Vector rendering overhead is minimized.

---

## 10. Demonstration Driver for Evaluations & Faculty Reviews

`SignBridgeDemoDriver` (`src/signbridge/ui/demo.py`) allows faculty, evaluators, and students to review the complete end-to-end system without requiring camera access or optimal lighting:
- **Simulation Script**: Automatically types `H - E - L - L - O` $\rightarrow$ `SPACE` $\rightarrow$ `W - O - R - L - D` $\rightarrow$ `SPACE` $\rightarrow$ triggers offline voice synthesis $\rightarrow$ resets cleanly.
- **Activation**: Can be launched directly via `python scripts/run_signbridge.py --demo` or toggled live via the `[D]` hotkey.

---

## 11. Latency & Rendering Benchmark Results

Measured using `scripts/benchmark_stage11.py` over 500 consecutive render iterations on a 1280x800 canvas:

| Render Mode | Mean Latency | Median Latency | 95th Percentile | Max Render FPS |
| :--- | :--- | :--- | :--- | :--- |
| **Obsidian Theme (Full Dashboard)** | **4.033 ms** | 4.015 ms | 4.321 ms | **248.0 FPS** |
| **High-Contrast Theme (Full Dashboard)**| **4.138 ms** | 4.117 ms | 4.456 ms | **241.6 FPS** |
| **Settings Modal Overlay** | **1.722 ms** | 1.710 ms | 2.000 ms | **580.7 FPS** |
| **Help Modal Overlay** | **1.779 ms** | 1.765 ms | 1.999 ms | **562.1 FPS** |
| **Onboarding Modal Overlay** | **1.703 ms** | 1.690 ms | 1.893 ms | **587.2 FPS** |

### Frame Budget Analysis
At a target camera rate of 30 FPS, each video frame has a generous 33.33 ms budget:
- Camera Capture: ~2.1 ms
- MediaPipe Landmark Detection: ~7.0 ms
- 94D Feature Extraction: ~0.4 ms
- Static ML Inference: ~1.2 ms
- Temporal Stabilization: ~0.5 ms
- Translation FSM: ~0.2 ms
- **Stage 11 UI Rendering: ~4.0 ms**
- **Total Pipeline Latency: ~15.4 ms**
- **Remaining Headroom: ~17.9 ms (> 53% margin)**

Result: **Zero frame drops and zero camera starvation.**

---

## 12. Integration with Stages 0-10 & Boundary Verification

Stage 11 respects all previous stage boundaries with zero architectural contamination:
- **MediaPipe Detector (Stage 3)**: Untouched.
- **FeatureSchema v1.0 94D Features (Stage 4)**: Untouched.
- **Model Classifier (Stage 6)**: Untouched. No retraining.
- **Temporal Stabilization (Stage 7)**: Untouched.
- **Translation FSM (Stage 8)**: Untouched.
- **End-to-End Orchestrator (Stage 9)**: Untouched.
- **Offline TTS Controller (Stage 10)**: Untouched.

---

## 13. User Experience & Evaluation Guide

### Launching the Application
```powershell
# Standard interactive launch with camera
uv run python scripts/run_signbridge.py

# Launch in automated presentation demonstration mode (no camera needed)
uv run python scripts/run_signbridge.py --demo

# Launch with High-Contrast accessibility theme
uv run python scripts/run_signbridge.py --high-contrast

# Launch with interactive onboarding tutorial
uv run python scripts/run_signbridge.py --onboarding

# Headless verification mode (runs N frames then exits)
uv run python scripts/run_signbridge.py --headless --max-frames 30
```

---

## 14. Future Roadmap & Non-Intrusive Extensions

- Continuous dynamic sign support (future stages) without altering static UI primitives.
- Multi-language UI localization files.
- Export transcripts to text/PDF.
