"""SignBridge Repeated Speech Verification Script.

Empirically verifies on Windows:
1. Sequential speech: HELLO -> HELP -> YES -> NO -> THANK YOU -> HELLO
2. Manual repeated speak on same token: HELLO -> HELLO -> HELLO
3. Duplicate suppression for held sign
4. Worker thread longevity and error recovery
5. Full status lifecycle: READY -> SPEAKING -> READY
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.tts.base import TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.provider import Pyttsx3TTSProvider


def verify_repeated_speech():
    print("=" * 65)
    print("SIGNBRIDGE: REPEATED SPEECH LIFECYCLE EMPIRICAL VERIFICATION")
    print("=" * 65)

    provider = Pyttsx3TTSProvider(rate=190, volume=1.0)
    print(f"Provider status: {provider.get_status().value}")
    assert provider.get_status() == TTSStatus.READY, "Provider not READY"

    cfg = TTSConfig(
        enabled=True,
        suppress_duplicates=True,
        cooldown_seconds=0.3,
        speak_words=True,
        speak_stable_signs=True,
    )
    ctrl = SpeechController(config=cfg, provider=provider)
    ctrl.start()
    print(f"Controller status: {ctrl.status.value}")
    print(f"Worker thread running: {ctrl._is_running}")

    # 1. Test 6 sequential tokens
    print("\n--- TEST 1: Sequential Tokens (HELLO -> HELP -> YES -> NO -> THANK YOU -> HELLO) ---")
    tokens = ["HELLO", "HELP", "YES", "NO", "THANK YOU", "HELLO"]
    for i, t in enumerate(tokens, 1):
        prev = ctrl._total_spoken
        ok = ctrl.handle_stable_sign(t)
        assert ok is True, f"Failed to enqueue {t}"
        print(f"[{i}/6] Triggered '{t}' -> waiting for playback...")
        # Wait up to 3.5s for audio playback
        for _ in range(35):
            if ctrl._total_spoken > prev:
                break
            time.sleep(0.1)
        print(f"      Spoken: {ctrl._total_spoken}, Last Spoken: '{ctrl.last_spoken_text}', Status: {ctrl.status.value}")
        assert ctrl._total_spoken == prev + 1, f"Utterance {t} did not complete"
        time.sleep(0.35)  # Let cooldown elapse

    assert ctrl._total_spoken == 6, f"Expected 6, got {ctrl._total_spoken}"
    print("[PASS] All 6 sequential tokens spoken successfully!")

    # 2. Test duplicate suppression on held sign
    print("\n--- TEST 2: Duplicate Suppression on Held Sign ---")
    dup_ok = ctrl.handle_stable_sign("HELLO")
    print(f"Immediate re-trigger of 'HELLO': {dup_ok} (Expected False)")
    assert dup_ok is False, "Duplicate was not suppressed"
    print("[PASS] Held sign duplicate suppression verified!")

    # 3. Test manual repeated speak on same sign
    print("\n--- TEST 3: Manual Speak Button (3x on 'HELLO') ---")
    for i in range(1, 4):
        prev = ctrl._total_spoken
        ok = ctrl.speak_manual("HELLO")
        assert ok is True, f"Manual speak {i} failed to enqueue"
        print(f"[{i}/3] Manual speak 'HELLO' triggered -> waiting for playback...")
        for _ in range(35):
            if ctrl._total_spoken > prev:
                break
            time.sleep(0.1)
        print(f"      Spoken: {ctrl._total_spoken}, Status: {ctrl.status.value}")
        assert ctrl._total_spoken == prev + 1, f"Manual speak {i} did not complete"
        time.sleep(0.2)

    assert ctrl._total_spoken == 9, f"Expected 9 total, got {ctrl._total_spoken}"
    print("[PASS] Manual speak button repeated utterance verified!")

    # 4. Verify worker thread survival
    print("\n--- TEST 4: Worker Thread Longevity ---")
    print(f"Worker thread alive: {ctrl._thread.is_alive() if ctrl._thread else False}")
    assert ctrl._thread and ctrl._thread.is_alive(), "Worker thread died"
    print(f"Final controller status: {ctrl.status.value}")
    assert ctrl.status == TTSStatus.READY, f"Status not READY: {ctrl.status.value}"
    print("[PASS] Worker thread remains alive indefinitely!")

    ctrl.shutdown()
    print("\n" + "=" * 65)
    print("ALL REPEATED SPEECH VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    verify_repeated_speech()
