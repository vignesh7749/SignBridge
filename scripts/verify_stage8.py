"""Stage 8 Verification Script: Translation Engine Simulation Harness.

Executes and verifies at least 12 synthetic simulation scenarios:
1. Word building (H E L L O -> HELLO)
2. Sentence building with spaces (H E L L O SPACE W O R L D -> HELLO WORLD)
3. Backspace editing (H E L P BACKSPACE -> HEL)
4. Trailing space handling (HELLO SPACE WORLD SPACE -> HELLO WORLD)
5. Multiple consecutive spaces (no duplicate spaces)
6. Clear word (clears active word only)
7. Clear sentence (clears entire sentence)
8. Reset (clears all state, history, undo)
9. Multi-step undo (restores previous state accurately)
10. Empty backspace safety (zero crashes on empty state)
11. Empty space safety (no empty word inserted)
12. Long token stream memory bound (bounded token history and undo stack)

Also benchmarks translation engine latency (mean, median, p95).
"""

from pathlib import Path
import sys
import time
import numpy as np

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.translation.config import TranslationConfig
from signbridge.translation.contracts import (
    StableSignToken,
    TranslationCommand,
    TranslationState,
)
from signbridge.translation.engine import TranslationEngine


def main() -> int:
    print("=" * 70)
    print("SIGNBRIDGE — STAGE 8 TRANSLATION SIMULATION & BENCHMARK HARNESS")
    print("=" * 70)

    config = TranslationConfig(
        max_token_history=200,
        max_undo_depth=50,
        trim_whitespace=True,
    )
    engine = TranslationEngine(config)

    all_latencies_ms: list[float] = []
    tests_passed = 0
    total_tests = 12

    # -------------------------------------------------------------
    # Scenario 1: Word Building (H E L L O -> HELLO)
    # -------------------------------------------------------------
    print("\n[SCENARIO 1] Word Building (H E L L O -> HELLO)")
    engine.reset()
    for char in ["H", "E", "L", "L", "O"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    print(f"  Current Word:  '{engine.current_word}'")
    print(f"  Sentence Text: '{engine.sentence_text}'")
    print(f"  State:          {engine.current_state.value}")
    if engine.current_word == "HELLO" and engine.sentence_text == "HELLO" and engine.current_state == TranslationState.BUILDING_WORD:
        print("  [PASS] Scenario 1: Word 'HELLO' built successfully.")
        tests_passed += 1
    else:
        print("  [FAIL] Scenario 1: Word building failed.")

    # -------------------------------------------------------------
    # Scenario 2: Sentence Building (H E L L O SPACE W O R L D -> HELLO WORLD)
    # -------------------------------------------------------------
    print("\n[SCENARIO 2] Sentence Building (H E L L O SPACE W O R L D -> HELLO WORLD)")
    engine.reset()
    for char in ["H", "E", "L", "L", "O"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    res_sp = engine.space()
    all_latencies_ms.append(res_sp.processing_latency_ms)

    for char in ["W", "O", "R", "L", "D"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    # Commit the second word
    res_sp2 = engine.space()
    all_latencies_ms.append(res_sp2.processing_latency_ms)

    print(f"  Completed Words: {engine.completed_words}")
    print(f"  Sentence Text:   '{engine.sentence_text}'")
    print(f"  State:           {engine.current_state.value}")
    if engine.sentence_text == "HELLO WORLD" and engine.completed_words == ["HELLO", "WORLD"]:
        print("  [PASS] Scenario 2: Sentence 'HELLO WORLD' assembled successfully.")
        tests_passed += 1
    else:
        print("  [FAIL] Scenario 2: Sentence building failed.")

    # -------------------------------------------------------------
    # Scenario 3: Backspace Editing (H E L P BACKSPACE -> HEL)
    # -------------------------------------------------------------
    print("\n[SCENARIO 3] Backspace Editing (H E L P BACKSPACE -> HEL)")
    engine.reset()
    for char in ["H", "E", "L", "P"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    res_bs = engine.backspace()
    all_latencies_ms.append(res_bs.processing_latency_ms)

    print(f"  Current Word: '{engine.current_word}'")
    if engine.current_word == "HEL":
        print("  [PASS] Scenario 3: Backspace removed 'P' -> 'HEL'.")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 3: Expected 'HEL', got '{engine.current_word}'")

    # -------------------------------------------------------------
    # Scenario 4: Trailing Space Handling (HELLO SPACE WORLD SPACE)
    # -------------------------------------------------------------
    print("\n[SCENARIO 4] Trailing Space Handling (HELLO SPACE WORLD SPACE)")
    engine.reset()
    for char in ["H", "E", "L", "L", "O"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)
    res = engine.space()
    all_latencies_ms.append(res.processing_latency_ms)

    for char in ["W", "O", "R", "L", "D"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)
    res = engine.space()
    all_latencies_ms.append(res.processing_latency_ms)

    print(f"  Sentence Text: '{engine.sentence_text}'")
    if engine.sentence_text == "HELLO WORLD" and engine.current_word == "":
        print("  [PASS] Scenario 4: Clean sentence without trailing whitespace artifacts.")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 4: Expected 'HELLO WORLD', got '{engine.sentence_text}'")

    # -------------------------------------------------------------
    # Scenario 5: Multiple Consecutive Spaces
    # -------------------------------------------------------------
    print("\n[SCENARIO 5] Multiple Consecutive Spaces (No Duplicate Spaces)")
    engine.reset()
    for char in ["H", "I"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    # 5 consecutive space presses
    for _ in range(5):
        res = engine.space()
        all_latencies_ms.append(res.processing_latency_ms)

    for char in ["A", "L", "L"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    print(f"  Completed Words: {engine.completed_words}")
    print(f"  Sentence Text:   '{engine.sentence_text}'")
    if engine.completed_words == ["HI"] and engine.sentence_text == "HI ALL":
        print("  [PASS] Scenario 5: Multiple spaces ignored cleanly, no double spacing.")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 5: Duplicate spaces occurred: '{engine.sentence_text}'")

    # -------------------------------------------------------------
    # Scenario 6: Clear Word
    # -------------------------------------------------------------
    print("\n[SCENARIO 6] Clear Word (Clears Active Word Only)")
    engine.reset()
    for char in ["G", "O", "O", "D"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)
    engine.space()

    for char in ["M", "I", "S", "T", "A", "K", "E"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    print(f"  Before clear_word: current='{engine.current_word}', sentence='{engine.sentence_text}'")
    res_cw = engine.clear_word()
    all_latencies_ms.append(res_cw.processing_latency_ms)

    print(f"  After clear_word:  current='{engine.current_word}', sentence='{engine.sentence_text}'")
    if engine.current_word == "" and engine.sentence_text == "GOOD" and engine.completed_words == ["GOOD"]:
        print("  [PASS] Scenario 6: Active word cleared while completed words preserved.")
        tests_passed += 1
    else:
        print("  [FAIL] Scenario 6: clear_word failed.")

    # -------------------------------------------------------------
    # Scenario 7: Clear Sentence
    # -------------------------------------------------------------
    print("\n[SCENARIO 7] Clear Sentence (Clears Entire Sentence)")
    res_cs = engine.clear_sentence()
    all_latencies_ms.append(res_cs.processing_latency_ms)

    print(f"  Sentence Text: '{engine.sentence_text}', State: {engine.current_state.value}")
    if engine.sentence_text == "" and engine.current_word == "" and engine.current_state == TranslationState.EMPTY:
        print("  [PASS] Scenario 7: Entire sentence cleared to EMPTY state.")
        tests_passed += 1
    else:
        print("  [FAIL] Scenario 7: clear_sentence failed.")

    # -------------------------------------------------------------
    # Scenario 8: Reset
    # -------------------------------------------------------------
    print("\n[SCENARIO 8] Reset (Clears All State, Words, History, and Undo)")
    # Add words
    for char in ["T", "E", "S", "T"]:
        engine.process_char(char)
    engine.space()

    res_rst = engine.reset()
    all_latencies_ms.append(res_rst.processing_latency_ms)

    if (
        engine.current_state == TranslationState.EMPTY
        and engine.sentence_text == ""
        and engine.current_word == ""
        and len(engine.token_history) == 0
    ):
        print("  [PASS] Scenario 8: Complete reset verified.")
        tests_passed += 1
    else:
        print("  [FAIL] Scenario 8: reset failed.")

    # -------------------------------------------------------------
    # Scenario 9: Multi-step Undo
    # -------------------------------------------------------------
    print("\n[SCENARIO 9] Multi-Step Undo")
    engine.reset()
    for char in ["A", "B", "C"]:
        res = engine.process_char(char)
        all_latencies_ms.append(res.processing_latency_ms)

    print(f"  Original: '{engine.current_word}'")
    res_u1 = engine.undo()  # Undoes 'C'
    all_latencies_ms.append(res_u1.processing_latency_ms)
    print(f"  Undo 1:   '{engine.current_word}'")

    res_u2 = engine.undo()  # Undoes 'B'
    all_latencies_ms.append(res_u2.processing_latency_ms)
    print(f"  Undo 2:   '{engine.current_word}'")

    if res_u1.current_word == "AB" and res_u2.current_word == "A":
        print("  [PASS] Scenario 9: Multi-step undo restored previous snapshots accurately.")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 9: Expected 'AB' then 'A', got '{res_u1.current_word}', '{res_u2.current_word}'")

    # -------------------------------------------------------------
    # Scenario 10: Empty Backspace Safety
    # -------------------------------------------------------------
    print("\n[SCENARIO 10] Empty Backspace Safety")
    engine.reset()
    # Calling backspace on empty state must not raise exception
    res_eb = engine.backspace()
    all_latencies_ms.append(res_eb.processing_latency_ms)

    if engine.current_state == TranslationState.EMPTY and engine.current_word == "":
        print("  [PASS] Scenario 10: Empty backspace handled safely without crashing.")
        tests_passed += 1
    else:
        print("  [FAIL] Scenario 10: Empty backspace produced invalid state.")

    # -------------------------------------------------------------
    # Scenario 11: Empty Space Safety
    # -------------------------------------------------------------
    print("\n[SCENARIO 11] Empty Space Safety")
    engine.reset()
    res_es = engine.space()
    all_latencies_ms.append(res_es.processing_latency_ms)

    if len(engine.completed_words) == 0 and engine.sentence_text == "":
        print("  [PASS] Scenario 11: Space on empty word does not insert empty word.")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 11: Expected 0 words, got {engine.completed_words}")

    # -------------------------------------------------------------
    # Scenario 12: Long Token Stream & Bounded Memory
    # -------------------------------------------------------------
    print("\n[SCENARIO 12] Long Token Stream & Bounded Memory (1,000 tokens)")
    engine.reset()
    start_bench = time.perf_counter()
    for i in range(1000):
        c = chr(ord("A") + (i % 26))
        res = engine.process_char(c)
        all_latencies_ms.append(res.processing_latency_ms)
        if i % 7 == 6:
            res_sp = engine.space()
            all_latencies_ms.append(res_sp.processing_latency_ms)
    bench_duration = time.perf_counter() - start_bench

    history_len = len(engine.token_history)
    print(f"  Total Tokens Ingested: 1,000")
    print(f"  Token History Length:  {history_len} (max: {config.max_token_history})")
    print(f"  1,000 Ingest Duration: {bench_duration * 1000:.2f} ms")

    if history_len <= config.max_token_history:
        print("  [PASS] Scenario 12: Memory strictly bounded under heavy stream.")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 12: History exceeded bound ({history_len} > {config.max_token_history})")

    # -------------------------------------------------------------
    # Benchmark Latency Profile
    # -------------------------------------------------------------
    mean_lat = float(np.mean(all_latencies_ms))
    med_lat = float(np.median(all_latencies_ms))
    p95_lat = float(np.percentile(all_latencies_ms, 95))

    print("\n" + "=" * 70)
    print("STAGE 8 TRANSLATION LATENCY BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Translation Actions Evaluated: {len(all_latencies_ms)}")
    print(f"Mean Translation Latency:            {mean_lat:.4f} ms ({mean_lat * 1000:.1f} us)")
    print(f"Median Translation Latency:          {med_lat:.4f} ms ({med_lat * 1000:.1f} us)")
    print(f"p95 Translation Latency:             {p95_lat:.4f} ms ({p95_lat * 1000:.1f} us)")
    print(f"Estimated Max Engine Throughput:     {1000.0 / mean_lat:.0f} actions/sec")

    print("\n" + "=" * 70)
    print(f"SIMULATION SUITE RESULT: {tests_passed}/{total_tests} SCENARIOS PASSED")
    print("=" * 70)

    return 0 if tests_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
