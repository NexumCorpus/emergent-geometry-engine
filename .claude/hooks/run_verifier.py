#!/usr/bin/env python
"""PostToolUse hook: after any edit to engine/verifier/test code, run the verifier
suite and surface PASS/FAIL (with real output) back into the transcript.

Wired in .claude/settings.json for matcher "Edit|Write". Non-blocking: it informs,
it does not undo the edit. The point is to make a regression impossible to miss and
to enforce the "keep going until the tests pass; do not modify the tests" loop.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PY = PROJECT / ".venv" / "Scripts" / "python.exe"
WATCHED = ("geometry", "verify", "adapters", "tests")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # never break the session on a malformed payload

    fp = (payload.get("tool_input") or {}).get("file_path", "")
    if not fp.endswith(".py"):
        return 0
    try:
        rel = Path(fp).resolve().relative_to(PROJECT)
    except Exception:
        return 0
    if not rel.parts or rel.parts[0] not in WATCHED:
        return 0

    interp = str(PY) if PY.exists() else sys.executable
    proc = subprocess.run(
        [interp, "-m", "pytest", "-q", "--no-header"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout or "").strip().splitlines()[-12:]
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[verifier hook] pytest after editing {rel} -> {status}")
    print("\n".join(tail))
    if status == "FAIL":
        print("[verifier hook] Tests are RED. Do NOT modify the tests. "
              "Fix the source and keep going until green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
