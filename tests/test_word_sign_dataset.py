"""Tests for word-sign dataset research, vocabulary configuration, and class definitions."""

import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
VOCAB_FILE = ROOT_DIR / "configs" / "word_sign_vocabulary.json"
RESEARCH_REPORT = ROOT_DIR / "datasets" / "reports" / "word_sign_dataset_research.json"


def test_vocabulary_file_exists_and_valid_json():
    """Verify that word_sign_vocabulary.json exists and is syntactically valid."""
    assert VOCAB_FILE.exists(), f"Vocabulary file not found at {VOCAB_FILE}"
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "classes" in data
    assert "class_to_id" in data
    assert "id_to_class" in data


def test_vocabulary_minimum_classes_and_core_words():
    """Verify vocabulary contains >= 50 classes and all core evaluated signs."""
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = [c.lower() for c in data["classes"]]
    assert len(classes) >= 50, f"Expected >= 50 classes, got {len(classes)}"

    core_required = ["hello", "help", "yes", "no", "stop", "thank you", "book", "water", "food", "eat", "drink"]
    for word in core_required:
        assert word in classes, f"Core word '{word}' missing from vocabulary"


def test_vocabulary_bidirectional_mapping_consistency():
    """Verify that class_to_id and id_to_class are exact bi-directional inverses."""
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data["classes"]
    class_to_id = data["class_to_id"]
    id_to_class = data["id_to_class"]

    assert len(class_to_id) == len(classes)
    for cls_name in classes:
        cls_id = class_to_id[cls_name]
        assert id_to_class[str(cls_id)] == cls_name


def test_emergency_signs_presence():
    """Verify that emergency/safety signs are declared and present."""
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    emergency_signs = [s.lower() for s in data.get("special_emergency_signs", [])]
    assert "help" in emergency_signs
    assert "danger" in emergency_signs
    assert "stop" in emergency_signs


def test_dataset_research_report_validity():
    """Verify that dataset research report conforms to academic and licensing requirements."""
    assert RESEARCH_REPORT.exists(), f"Research report not found at {RESEARCH_REPORT}"
    with open(RESEARCH_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "candidate_evaluations" in report
    assert "comparative_analysis" in report
    assert "WLASL" in report["candidate_evaluations"]
    assert "C-UDA 1.0" in report["candidate_evaluations"]["WLASL"]["license"]
    assert report["comparative_analysis"]["recommended_primary_dataset"].startswith("WLASL")
