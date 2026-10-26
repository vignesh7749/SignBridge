# SignBridge — User Guide & Quickstart Reference

Welcome to **SignBridge**, the real-time sign language translation system. This guide explains how to use SignBridge in simple, straightforward language.

---

## 1. How to Launch SignBridge

Open PowerShell or Command Prompt in the SignBridge directory and run:

```powershell
.\.venv\Scripts\python.exe scripts\run_signbridge.py
```

### Try Demonstration Mode (No Camera Needed)
If you do not have a webcam or want to see how the system works first:
```powershell
.\.venv\Scripts\python.exe scripts\run_signbridge.py --demo
```

### High-Contrast Mode for Low Vision
If you need high-visibility colors with maximum contrast (pure black and yellow):
```powershell
.\.venv\Scripts\python.exe scripts\run_signbridge.py --high-contrast
```

---

## 2. How to Position Your Hand

For the best and most accurate recognition:
1. **Distance**: Sit about 1.5 to 2.5 feet (45 to 75 cm) away from your webcam.
2. **Lighting**: Ensure your room is well-lit from the front. Avoid having a bright window directly behind you.
3. **Background**: Simple, uncluttered backgrounds work best.
4. **Orientation**: Face your palm toward the camera with your fingers clearly visible.
5. **Mirror View**: The video feed is mirrored (like looking in a mirror) so your movements feel natural.

---

## 3. How Recognition Works

1. When your hand enters the camera view, you will see a colorful **skeleton overlay** tracking your 21 hand joints.
2. In the top-right **Recognition Panel**, you will see the detected letter.
3. Below the letter is a **Dwell Stability Meter**:
   - Hold your sign steady for about a quarter of a second ($0.25\text{ s}$).
   - The meter will fill up. When it locks in, the letter will be confirmed and added to your word!
   - If you keep holding the exact same sign, SignBridge will **not duplicate** the letter.

---

## 4. How to Create Words

- Form letters one after another. They will appear in the **Current Word** box.
- When you finish spelling a word, press the **`[SPACE]` bar** or click the **`Word Break`** button on screen.
- If you stop signing for 2 seconds, SignBridge will automatically add a space for you!

---

## 5. How to Create Sentences

- As you complete words with spaces, they will form a full sentence in the **Final Sentence** box.
- You can build sentences as long as you need.

---

## 6. How to Use Voice Output (Speech)

- **Speak Current Sentence**: Press the **`[S]` key** or click the green **`Speak`** button. SignBridge will read your sentence aloud through your computer speakers.
- **Stop Speech**: If you want to stop the voice immediately, press **`[X]`** or click **`Stop`**.
- **Mute / Unmute Voice**: Press **`[T]`** or click **`Mute`** to turn speech synthesis on or off.

---

## 7. How to Correct Text

Made a mistake while spelling? No problem:
- **Delete Last Letter**: Press **`[BACKSPACE]`** or click **`Delete`**. If the current word is empty, it will re-open the previous word so you can fix it.
- **Undo**: Made an accidental change? Press **`[U]`** or click **`Undo`** to go back one step.
- **Clear Sentence**: Want to erase everything and start over? Press **`[C]`** or click **`Clear`**.

---

## 8. How to Reset the System

- Press **`[R]`** or click **`Reset All`** to reset the letter buffer, word buffer, sentence text, and voice queue completely.

---

## 9. Keyboard Shortcuts Summary

| Key | What it does |
| :---: | :--- |
| **`[SPACE]`** | Finish word (Word Break) |
| **`[BACKSPACE]`** | Delete last letter |
| **`[S]`** | Speak completed sentence |
| **`[X]`** | Stop speech immediately |
| **`[T]`** | Mute / Unmute speech |
| **`[C]`** | Clear sentence |
| **`[U]`** | Undo last action |
| **`[P]`** | Pause / Resume camera video |
| **`[R]`** | Reset all text and speech |
| **`[O]`** | Open Settings dialog |
| **`[H]`** | Open Help dialog |
| **`[D]`** | Start / Stop automated demonstration |
| **`[ESC]`** | Close open popup dialog / Exit application |

---

## 10. Troubleshooting

### Problem: "Unable to open camera"
- **Fix**: Check if another application (like Zoom, Teams, or Windows Camera) is using your webcam. Close it and restart SignBridge.
- If you have multiple webcams, you can specify the camera number:
  ```powershell
  .\.venv\Scripts\python.exe scripts\run_signbridge.py --camera-index 1
  ```

### Problem: No sound from speakers
- **Fix**: Check that your computer volume is up. In SignBridge, press **`[T]`** to ensure Voice Output is not set to MUTED.
- Open Settings with **`[O]`** and check that the Volume is set above 50%.

### Problem: Letters changing too quickly
- **Fix**: You can increase the dwell time so SignBridge requires you to hold signs longer before committing them:
  ```powershell
  .\.venv\Scripts\python.exe scripts\run_signbridge.py --dwell-ms 350
  ```
