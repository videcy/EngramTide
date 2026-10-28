"""Command-hook fallback for Claude Code builds without ``type: "http"`` hooks.

Reads the hook's JSON input from stdin, forwards it to the running EngramTide
server and prints the response body unchanged.  Standard library only — it
starts once per prompt, so it must not import numpy or the engine.

Any failure exits 0 with no output: a memory-layer problem must never block
or decorate the user's prompt.

Usage (in ~/.claude/settings.json):
    "command": "python /path/to/EngramTide/scripts/claude_hook.py user-prompt-submit"
Events: user-prompt-submit | stop | post-compact
"""

from __future__ import annotations

import os
import sys
import urllib.request

EVENTS = {"user-prompt-submit", "stop", "post-compact"}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in EVENTS:
        return 0
    base = os.getenv("ENGRAMTIDE_HOOK_URL", "http://127.0.0.1:8765/hooks/claude-code")
    timeout = float(os.getenv("ENGRAMTIDE_HOOK_CLIENT_TIMEOUT", "8"))
    headers = {"Content-Type": "application/json"}
    token = os.getenv("ENGRAMTIDE_HOOK_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        body = sys.stdin.buffer.read()
        request = urllib.request.Request(
            f"{base.rstrip('/')}/{argv[1]}", data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            output = response.read()
    except Exception:  # noqa: BLE001
        return 0

    if output:
        sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
