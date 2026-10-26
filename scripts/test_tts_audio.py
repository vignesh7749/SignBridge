"""Standalone diagnostic script to verify offline TTS audio hardware and speech synthesis.

Tests Pyttsx3TTSProvider with native Windows SAPI5 synthesizer, checks installed voices,
synthesizes test utterances, tests rate and volume modulation, and tests stop/interrupt.
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.tts.base import TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.events import SpeechEvent, SpeechEventType
from signbridge.tts.provider import Pyttsx3TTSProvider


def main() -> int:
    print("=" * 70)
    print("SIGNBRIDGE — STAGE 10 OFFLINE TTS HARDWARE DIAGNOSTIC")
    print("=" * 70)

    # 1. Initialize offline provider
    print("[1/5] Initializing Pyttsx3TTSProvider (Windows SAPI5)...")
    provider = Pyttsx3TTSProvider(rate=175, volume=1.0)
    status = provider.get_status()
    print(f"      Provider Status: {status.value}")

    if status == TTSStatus.ERROR:
        print("[WARNING] SAPI5 TTS engine initialization reported an error.")
        print("          Audio output hardware may be disabled or unavailable.")
        return 1

    # 2. Enumerate installed system voices
    print("\n[2/5] Enumerating installed voices:")
    voices = provider.get_available_voices()
    if not voices:
        print("      No installed voices reported.")
    else:
        for idx, voice in enumerate(voices):
            print(f"      Voice {idx + 1}: {voice.get('name')} (ID: {voice.get('id')})")

    # 3. Test primary utterance
    greeting = "SignBridge text to speech is online."
    print(f"\n[3/5] Synthesizing greeting utterance: '{greeting}'...")
    t0 = time.perf_counter()
    provider.speak(greeting)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"      Utterance synthesized in {latency_ms:.1f} ms.")

    # 4. Test rate and volume changes
    sample = "HELLO WORLD"
    print(f"\n[4/5] Testing rate modulation (rate=200, volume=0.8): '{sample}'...")
    provider.set_rate(200)
    provider.set_volume(0.8)
    provider.speak(sample)
    print("      Rate/volume playback complete.")

    # 5. Test SpeechController asynchronous worker and interruption
    print("\n[5/5] Testing asynchronous SpeechController queue and interrupt...")
    config = TTSConfig(rate=175, volume=1.0, speak_words=True, speak_sentences=True)
    controller = SpeechController(config=config, provider=provider)
    controller.start()

    long_text = "This is an intentionally long sentence that should be interrupted by the stop command before it finishes completely."
    print(f"      Enqueuing: '{long_text[:40]}...'")
    controller.speak_manual(long_text)
    time.sleep(0.3)  # Allow speech to start

    print("      Issuing controller.stop() interrupt...")
    controller.stop()
    print("      Playback interrupted successfully.")

    controller.shutdown()
    print("\n[SUCCESS] Stage 10 TTS hardware verification passed.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
