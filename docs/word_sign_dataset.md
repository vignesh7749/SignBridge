# SignBridge — Isolated ASL Word-Sign Dataset Specification
## Research, Licensing, Comparative Evaluation & Controlled Vocabulary

---

## 1. Objective & Scope

This specification establishes the dataset foundation for SignBridge's **Word-Level Sign Recognition** extension. While Stages 0–12 provide reliable ASL manual alphabet fingerspelling (A–Z via static 94D FeatureSchema v1.0), this extension adds the ability to recognize isolated American Sign Language signs representing complete words and concepts (e.g., `[HELP sign] -> HELP`, `[HELLO sign] -> HELLO`).

> [!IMPORTANT]
> **Preservation Guarantee**:
> The existing ASL alphabet dataset (`datasets/external/asl_semcom/` and `datasets/external/asl_alphabet/`) and the 94D FeatureSchema v1.0 remain strictly untouched. The word-sign dataset is housed in an isolated directory structure (`datasets/external/word_signs/`) and processed independently.

---

## 2. Comprehensive Candidate Dataset Evaluation

We systematically audited three premier research datasets across 15 technical and legal dimensions:

| Evaluation Dimension | Candidate 1: WLASL | Candidate 2: ASL Citizen | Candidate 3: MS-ASL |
| :--- | :--- | :--- | :--- |
| **1. Full Title** | Word-Level American Sign Language (WLASL) | ASL Citizen: Community-Sourced ISLR | Microsoft American Sign Language (MS-ASL) |
| **2. Originating Institution** | University of Sydney & ByteDance AI Lab (WACV 2020) | Microsoft Research, Boston Univ., RIT (NeurIPS 2023) | Microsoft Cognitive Services (BMVC 2019) |
| **3. License** | Computational Use of Data Agreement (C-UDA 1.0) | Microsoft Research Open Data License | Non-Commercial Research License |
| **4. Commercial Use** | Strictly Prohibited | Strictly Prohibited | Strictly Prohibited |
| **5. Educational / Academic Use** | **Permitted** (Ideal for academic evaluation) | **Permitted** (Academic research) | **Permitted** (Academic research) |
| **6. Number of Classes** | 2,000 glosses (official subsets: 100, 300, 1000) | 2,731 glosses | 1,000 glosses (subsets: 100, 1000) |
| **7. Number of Video Samples** | 21,083 (WLASL-100: 2,038 videos) | 83,863 videos | 25,513 video segments |
| **8. Number of Signers** | 119 signers | 52 Deaf / Hard-of-Hearing signers | 222 signers |
| **9. Sign Isolation** | **Yes** (Isolated sign clips with frame boundaries) | **Yes** (Isolated dictionary lookups) | **Yes** (Segmented from continuous YouTube videos) |
| **10. Label Format** | Standard English glosses (`"book"`, `"hello"`, `"help"`) | Standard English glosses | Standard English glosses |
| **11. Target Vocabulary Presence** | **High** (Contains `hello`, `help`, `yes`, `no`, `drink`, etc.) | **Comprehensive** (Full lexicon coverage) | **Moderate** (Contains common words) |
| **12. Download Footprint** | WLASL-100: ~1.4 GB; WLASL-2000: ~14.2 GB | Full Archive: **~46 GB** | Full Archive: ~18 GB |
| **13. Authentication Requirements** | **None** (Public GitHub & academic mirrors) | **None** (Direct Microsoft Download Center zip) | **None** (YouTube scraper required) |
| **14. Link Health & Reproducibility** | **High** (Verified metadata & academic mirrors) | **High** (Single persistent Microsoft archive) | **Low** (> 35% broken YouTube URLs) |
| **15. MediaPipe Compatibility** | **Excellent** (Front-facing camera, clear hand visibility) | **High** (Realistic webcams, some variance) | **Moderate** (Variable compression & crop boxes) |
| **Suitability Verdict** | **SELECTED: PRIMARY BENCHMARK** | **SELECTED: REFERENCE VALIDATION** | **REJECTED (URL BIT-ROT)** |

---

## 3. Dataset Selection Decision & Licensing Strategy

### Primary Benchmark: WLASL-100
- **Rationale**: WLASL-100 is the de-facto academic benchmark for word-level sign language recognition. It contains 100 balanced, high-frequency ASL glosses with ~2,000 isolated video recordings across 119 signers.
- **Reference Lexicon: ASL Citizen**: Used to cross-validate gloss boundaries and ensure consistent linguistic definitions for emergency and conversational signs.
- **Rejection of MS-ASL**: Rejected because more than 35% of the original YouTube video links have experienced bit-rot, making automated, reproducible pipeline verification unreliable.

### Ethical & Legal Compliance Guidelines
1. **Zero Piracy / No Unauthorized Git Bundling**: Large third-party video archives under C-UDA 1.0 or Microsoft Open Data licenses must not be checked into Git repositories.
2. **Automated Clean Acquisition**: An automated download and verification script (`scripts/dataset/acquire_word_signs.py`) downloads the controlled subset directly from authorized academic sources.
3. **Built-in Demo Sample Cleared Licensing**: Only videos specifically licensed under Creative Commons Attribution (CC-BY), Public Domain, or verified educational open-access are bundled into `datasets/samples/word_signs/` with complete provenance and attribution metadata.

---

## 4. Controlled 58-Gloss Vocabulary

The controlled vocabulary is saved in `configs/word_sign_vocabulary.json`. All 58 classes are verified to exist in WLASL and ASL Citizen:

### Category Breakdown
1. **Greetings & Politeness (6 classes)**:
   `hello`, `goodbye`, `please`, `thank you`, `sorry`, `welcome`
2. **Emergency, Health & Assistance (6 classes)**:
   `help`, `stop`, `wait`, `danger`, `hurt`, `need`
3. **Affirmation & Negation (5 classes)**:
   `yes`, `no`, `fine`, `wrong`, `correct`
4. **Actions & Motion (10 classes)**:
   `come`, `go`, `walk`, `dance`, `play`, `eat`, `drink`, `sleep`, `see`, `give`
5. **Entities & Common Objects (9 classes)**:
   `book`, `computer`, `clothes`, `chair`, `table`, `water`, `food`, `car`, `phone`
6. **People & Relations (8 classes)**:
   `man`, `woman`, `mother`, `father`, `friend`, `family`, `cousin`, `baby`
7. **States & Emotions (8 classes)**:
   `happy`, `sad`, `like`, `love`, `good`, `bad`, `hot`, `cold`
8. **Descriptors & Colors (6 classes)**:
   `orange`, `yellow`, `white`, `black`, `red`, `blue`

---

## 5. Storage Hierarchy

```
datasets/
├── external/
│   ├── asl_semcom/             # PRESERVED: Stage 5 Alphabet Dataset
│   ├── asl_alphabet/           # PRESERVED: Stage 5 Alphabet Dataset
│   └── word_signs/             # NEW: Word-Sign Dataset Directory
│       ├── raw/                # Downloaded isolated video files (.mp4)
│       ├── metadata/           # WLASL / ASL Citizen gloss & split JSON
│       ├── reports/            # Quality audit & verification reports
│       └── processed/          # Extracted 128D temporal landmark sequences
├── samples/
│   └── word_signs/             # Built-in legitimate demonstration videos
│       ├── README.md           # Provance, license, and signer attribution
│       ├── samples_metadata.json # Verified metadata dictionary
│       └── *.mp4               # Verified sample video files
└── user_samples/
    └── word_signs/             # User-added test/demo videos (isolated from training)
        └── user_catalog.json   # Registry of local user sample videos
```

---

## 6. Preprocessing & Normalization Pipeline

For each video sample $V = \{f_1, f_2, \dots, f_N\}$:
1. **Frame Ingestion**: Decoded with OpenCV (`cv2.VideoCapture`), validating resolution, duration, and frame sequence.
2. **Dual-Hand Landmark Tracking**: MediaPipe Hands processes each frame, tracking Primary Hand and Secondary Hand (up to 2 hands).
3. **Coordinate Centering**: Hand landmarks normalized relative to wrist reference point ($lm_0$).
4. **Temporal Resampling**: Variable-length sequences resampled via uniform linear interpolation to exactly $T = 20$ keyframe steps.
5. **Multi-Hand Vectorization**: Each time step contains $2 \times 64 = 128$ dimensions (coordinates + presence flag), yielding a fixed $20 \times 128 = 2,560$-dimensional numerical representation per gesture sequence.
