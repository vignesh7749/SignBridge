"""SignBridge Environment Verification Script.

Checks Python version, OpenCV import, and NumPy import.
"""

import sys

def verify_environment():
    print("=" * 60)
    print("SignBridge — Environment Verification")
    print("=" * 60)

    # 1. Verify Python Version & Interpreter
    python_version = sys.version.split()[0]
    print(f"[+] Python Executable : {sys.executable}")
    print(f"[+] Python Version    : {python_version}")
    
    major, minor, _ = sys.version_info[:3]
    if (major, minor) == (3, 11):
        print(f"    -> Python 3.11 confirmed (compatible with CV/MediaPipe ecosystem).")
    else:
        print(f"    [!] Warning: Expected Python 3.11, running on {major}.{minor}.")

    # 2. Verify OpenCV Import
    try:
        import cv2
        print(f"[+] OpenCV Import     : SUCCESS (version {cv2.__version__})")
    except ImportError as e:
        print(f"[-] OpenCV Import     : FAILED ({e})")
        return False

    # 3. Verify NumPy Import
    try:
        import numpy as np
        print(f"[+] NumPy Import      : SUCCESS (version {np.__version__})")
        # Quick sanity check on array creation
        test_arr = np.array([1, 2, 3])
        assert test_arr.sum() == 6
    except Exception as e:
        print(f"[-] NumPy Import      : FAILED ({e})")
        return False

    print("=" * 60)
    print("[SUCCESS] All required components verified successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
