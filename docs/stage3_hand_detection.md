# Stage 3 Specification: MediaPipe Hand Detection & Landmark Extraction

**SignBridge — Real-Time Sign Language Recognition & Translation System**

> [!IMPORTANT]
> **Stage 3 Scope Notice**: Stage 3 implements robust real-time hand detection, 21-point 3D landmark extraction, handedness classification, and skeletal visualization.
> **Stage 3 does NOT perform sign classification, ML training, word formation, or translation.**

---

## 1. Dependency Analysis & Version Selection

- **Selected Dependency**: `mediapipe==0.10.14`
- **Compatibility Matrix**:
  - Python: `3.11.15` (CPython 64-bit)
  - NumPy: `2.4.6`
  - OpenCV: `5.0.0.93`
  - Host OS: Windows x64
- **Reasoning**:
  1. **Built-in Solutions Architecture**: MediaPipe version `0.10.14` is the most stable and mature release featuring the bundled `mp.solutions.hands` pipeline. Later versions ($\ge 0.10.31$ and $1.0.x$) fully deprecated and removed the `mp.solutions` namespace, mandating external `.task` file downloads.
  2. **Bundled Offline Models**: Models are embedded inside the Python wheel, requiring no external network downloads or cloud dependencies.
  3. **Zero Package Downgrade**: Validated seamless operation with `numpy==2.4.6` and `opencv-python==5.0.0.93` on Windows without requiring downgrades.

---

## 2. Vision Pipeline Architecture

The data flow for Stage 3 preserves strict decoupling between camera acquisition and downstream consumers:

```
CameraManager (Dedicated Background Worker Thread)
       │
       ▼ FramePacket (Clean BGR NumPy ndarray, Timestamp, Monotonic Frame ID)
MediaPipeHandDetector
       │
       ├─► 1. Color Conversion (BGR -> RGB)
       ├─► 2. TFLite Graph Inference (MediaPipe Hands)
       └─► 3. Decoupling Adapter (Protobuf -> Domain Contracts)
       │
       ▼ HandDetectionResult (processing_time_ms, has_hands, hands: [DetectedHand])
FrameProcessor
       │
       ├─► Draw 21 Landmark Joints & Fingertip Accents
       ├─► Draw Skeletal Bone Connections (HAND_CONNECTIONS)
       ├─► Draw Bounding Box & Handedness Badge
       └─► Render Stage 3 Heads-Up Display (HUD)
       │
       ▼ Display Frame (Rendered via OpenCV GUI in scripts/run_hand_demo.py)
```

---

## 3. The 21 MediaPipe Hand Landmarks

SignBridge strictly preserves the official MediaPipe landmark numbering and topological finger chains:

| Index | Landmark Name | Topological Region |
|---|---|---|
| **0** | `WRIST` | Palm Base / Origin |
| **1** | `THUMB_CMC` | Thumb Carpal-Metacarpal Joint |
| **2** | `THUMB_MCP` | Thumb Metacarpophalangeal Joint |
| **3** | `THUMB_IP` | Thumb Interphalangeal Joint |
| **4** | `THUMB_TIP` | Thumb Fingertip |
| **5** | `INDEX_FINGER_MCP` | Index Finger Knuckle |
| **6** | `INDEX_FINGER_PIP` | Index Finger Proximal Joint |
| **7** | `INDEX_FINGER_DIP` | Index Finger Distal Joint |
| **8** | `INDEX_FINGER_TIP` | Index Fingertip |
| **9** | `MIDDLE_FINGER_MCP` | Middle Finger Knuckle |
| **10** | `MIDDLE_FINGER_PIP` | Middle Finger Proximal Joint |
| **11** | `MIDDLE_FINGER_DIP` | Middle Finger Distal Joint |
| **12** | `MIDDLE_FINGER_TIP` | Middle Fingertip |
| **13** | `RING_FINGER_MCP` | Ring Finger Knuckle |
| **14** | `RING_FINGER_PIP` | Ring Finger Proximal Joint |
| **15** | `RING_FINGER_DIP` | Ring Finger Distal Joint |
| **16** | `RING_FINGER_TIP` | Ring Fingertip |
| **17** | `PINKY_MCP` | Pinky Finger Knuckle |
| **18** | `PINKY_PIP` | Pinky Finger Proximal Joint |
| **19** | `PINKY_DIP` | Pinky Finger Distal Joint |
| **20** | `PINKY_TIP` | Pinky Fingertip |

---

## 4. Coordinate Systems

Each keypoint in `DetectedHand.landmarks` exposes both normalized and pixel coordinates:

1. **Normalized Coordinate Space (`x_norm`, `y_norm`, `z_norm`)**:
   - $x_{norm} \in [0.0, 1.0]$: Horizontal position relative to input image width.
   - $y_{norm} \in [0.0, 1.0]$: Vertical position relative to input image height.
   - $z_{norm}$: Depth coordinate representing distance relative to the wrist keypoint (`0`). Smaller values correspond to keypoints positioned closer to the optical sensor.
2. **Image Pixel Space (`x_px`, `y_px`, `z_depth`)**:
   - $x_{px} = \text{round}(x_{norm} \times W)$ (clamped to $[0, W - 1]$)
   - $y_{px} = \text{round}(y_{norm} \times H)$ (clamped to $[0, H - 1]$)
   - $z_{depth} = z_{norm} \times W$ (metric depth estimate in pixel scale)

> [!NOTE]
> Stage 3 preserves raw image coordinates and normalized frame coordinates. Feature normalization for machine learning (centering relative to wrist origin and scaling by palm distance $\|P_9 - P_0\|$) belongs strictly to **Stage 4**.

---

## 5. Handedness & Mirroring Semantics

- **Mirrored Selfie Mode (`flip_horizontal=True`, Default)**:
  - Video feed is mirrored so moving your right hand moves the hand on the right side of the screen (intuitive mirror feedback).
  - MediaPipe's internal classification assumes a front-facing mirrored camera: when you raise your right hand, MediaPipe reports `Right`.
- **Unmirrored Raw Mode (`flip_horizontal=False`)**:
  - The video feed is not flipped. Moving your right hand moves the hand on the left side of the screen (standard camera perspective).

---

## 6. Centralized Configuration

All vision parameters are managed within [`VisionConfig`](file:///c:/Users/VIGNESH/OneDrive/Desktop/SignBridge/src/signbridge/core/config.py#L19-L26):

```python
@dataclass(frozen=True)
class VisionConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    model_complexity: int = 1
    static_image_mode: bool = False
```

---

## 7. How to Run the Stage 3 Demo

Execute from the workspace root:

```powershell
# Interactive Live Hand Tracking Demo:
.\.venv\Scripts\python.exe scripts/run_hand_demo.py --camera-index 0

# Optional CLI flags:
# --max-hands 2                (Detect up to 2 hands)
# --min-detect 0.7             (Minimum detection confidence)
# --min-track 0.6              (Minimum tracking confidence)
# --no-flip                    (Disable horizontal mirror)

# Interactive Hotkeys:
# [P]         - Pause / Resume display
# [M]         - Toggle mirror mode
# [I]         - Toggle landmark index numbers (0-20)
# [S]         - Save annotated snapshot to logs/
# [Q] / [ESC] - Clean shutdown
```

---

## 8. Known Limitations

1. **Severe Occlusion**: If fingers are tucked completely behind the palm or hidden from camera line-of-sight, depth ($z$) estimation relies on model priors.
2. **Extreme Lighting**: Low-light conditions or direct backlight can reduce hand localization confidence.
3. **Out-of-Frame Hands**: If the wrist moves outside the image boundary, MediaPipe cannot reliably anchor the hand scale.
