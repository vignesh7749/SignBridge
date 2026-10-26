# SignBridge Built-in ASL Demonstration Samples

This directory contains verified, legitimate demonstration sample videos for the SignBridge Word-Level ASL Recognition subsystem.

## Curated Demonstrations

| File | Standard Name | Sign | Description | Duration | License |
|------|---------------|------|-------------|----------|---------|
| `hello.mp4` | `hello_sample.mp4` | `HELLO` | Open hand waves outward from temple | 3.07s | MIT License |
| `help.mp4` | `help_sample.mp4` | `HELP` | Thumbs-up resting on flat palm lifted upward (Emergency Sign) | 1.98s | MIT License |
| `yes.mp4` | `yes_sample.mp4` | `YES` | S-fist nodding up and down | 2.46s | MIT License |
| `no.mp4` | `no_sample.mp4` | `NO` | Index and middle finger snap closed to thumb | 2.87s | MIT License |
| `stop.mp4` | `stop_sample.mp4` | `STOP` | Open right hand chops down onto open left palm | 2.43s | MIT License |
| `thank_you.mp4` | `thank_you_sample.mp4` | `THANK YOU` | Flat hand touches chin and extends outward | 1.53s | MIT License |

## Technical Specification & Manifest
- **Machine-Readable Manifest**: [`manifest.json`](manifest.json)
- **Source**: Curated from the open American Sign Language (ASL) Dataset (`akasheroor/American-Sign-Language-Dataset`)
- **License**: MIT License
- **Resolution**: 640x480
- **Framerate**: ~30.0 FPS
- **Codec**: H.264 / MP4 Container (.mp4)
- **Validation**: Verified via `python scripts/verify_demo_videos.py` (12/12 criteria passed, genuine human signers with verified hand landmarks).

