#!/usr/bin/env python
"""PreToolUse hook for Bash: block obviously destructive commands (exit code 2
tells Claude Code to deny the call and feeds the reason back to the model).

Deliberately conservative: it blocks recursive force-deletes of broad paths, disk
formatting, fork bombs, and force-pushes. It is a guard rail, not a sandbox.
"""
import json
import re
import sys

DANGEROUS = [
    (r"rm\s+-rf?\s+(/|~|\$HOME|\*)\b", "recursive force-delete of a broad path"),
    (r"\brm\s+-rf?\s+\.\s*$", "recursive force-delete of the working directory"),
    (r"Remove-Item.*-Recurse.*-Force.*([A-Za-z]:\\?\s*$|\\\s*$)", "recursive force-delete of a drive root"),
    (r"\bformat\s+[A-Za-z]:", "disk format"),
    (r"\bdel\s+/[sq].*[A-Za-z]:\\", "recursive delete of a drive path"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:", "fork bomb"),
    (r"git\s+push\s+.*--force\b", "force-push (history rewrite)"),
    (r"\bmkfs\b", "filesystem creation"),
    (r">\s*/dev/sd[a-z]", "raw write to a block device"),
]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (payload.get("tool_input") or {}).get("command", "") or ""
    for pattern, why in DANGEROUS:
        if re.search(pattern, cmd, flags=re.IGNORECASE):
            print(f"BLOCKED by guard_bash: {why}.\nCommand: {cmd}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
