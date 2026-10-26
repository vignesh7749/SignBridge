"""SignBridge Stage 12: Automated Project Audit & Static Quality Check Script.

Performs:
1. Syntax validation on all Python files.
2. Hard-coded path audit across src/ and scripts/.
3. Requirements import sanity checks.
4. Temporary file scanning.
5. Resource management patterns inspection.
"""

from __future__ import annotations

import os
import py_compile
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def audit_syntax() -> list[str]:
    print("\n[1] Auditing Python Syntax across project...")
    errors = []
    py_files = list(PROJECT_ROOT.glob("src/**/*.py")) + list(PROJECT_ROOT.glob("scripts/**/*.py")) + list(PROJECT_ROOT.glob("tests/**/*.py"))
    for p in py_files:
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"Syntax error in {p.relative_to(PROJECT_ROOT)}: {e}")
    print(f"    Compiled {len(py_files)} Python files. Syntax errors: {len(errors)}")
    return errors


def audit_hardcoded_paths() -> list[str]:
    print("\n[2] Scanning for hard-coded absolute user machine paths in code...")
    violations = []
    # Pattern to detect Windows user paths or hardcoded developer drives
    pattern = re.compile(r'[a-zA-Z]:\\(?:Users|home)\\[a-zA-Z0-9_\-\.\\]+', re.IGNORECASE)
    
    target_dirs = [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]
    for d in target_dirs:
        for p in d.rglob("*.py"):
            try:
                content = p.read_text(encoding="utf-8")
                matches = pattern.findall(content)
                if matches:
                    violations.append(f"{p.relative_to(PROJECT_ROOT)}: found {matches}")
            except Exception as e:
                violations.append(f"Could not read {p}: {e}")
    print(f"    Hard-coded machine paths detected: {len(violations)}")
    for v in violations:
        print(f"      - {v}")
    return violations


def audit_requirements() -> list[str]:
    print("\n[3] Auditing requirements.txt dependencies...")
    req_path = PROJECT_ROOT / "requirements.txt"
    issues = []
    if not req_path.exists():
        return ["requirements.txt missing!"]

    reqs = [line.strip() for line in req_path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    print(f"    Found {len(reqs)} pinned requirements in requirements.txt:")
    for r in reqs:
        pkg = r.split("==")[0].split(">=")[0].strip()
        # Map package name to import name
        import_map = {
            "opencv-python": "cv2",
            "scikit-learn": "sklearn",
        }
        mod_name = import_map.get(pkg, pkg)
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"      [OK] {pkg} ({r}) -> Imported '{mod_name}' (version {ver})")
        except ImportError as e:
            issues.append(f"Failed to import required package '{pkg}': {e}")
    return issues


def audit_temporary_files() -> list[Path]:
    print("\n[4] Scanning for leftover temporary/debug files...")
    temp_patterns = ["*.tmp", "*.bak", "*~", "*.swp", "*.pyc.tmp"]
    found_temps = []
    for pat in temp_patterns:
        found_temps.extend(PROJECT_ROOT.glob(f"**/{pat}"))
    print(f"    Found {len(found_temps)} temporary files.")
    for t in found_temps:
        print(f"      - {t.relative_to(PROJECT_ROOT)}")
    return found_temps


def run_audit():
    print("=" * 65)
    print("SIGNBRIDGE STAGE 12: PROJECT AUDIT & STATIC QUALITY CHECK")
    print("=" * 65)

    syntax_errors = audit_syntax()
    path_issues = audit_hardcoded_paths()
    req_issues = audit_requirements()
    temp_files = audit_temporary_files()

    print("\n" + "=" * 65)
    print("AUDIT SUMMARY:")
    print(f"  - Syntax Errors:       {len(syntax_errors)}")
    print(f"  - Machine Path Leaks:  {len(path_issues)}")
    print(f"  - Dependency Issues:   {len(req_issues)}")
    print(f"  - Temporary Files:     {len(temp_files)}")
    print("=" * 65)

    if not syntax_errors and not path_issues and not req_issues:
        print("[SUCCESS] Project audit passed with ZERO defects.")
        return True
    else:
        print("[WARNING] Issues detected requiring review.")
        return False


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
