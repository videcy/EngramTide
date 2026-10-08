"""command hook 兜底脚本：服务不可达时静默放行，正常时原样透传。"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "claude_hook.py"
OUTPUT = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "记忆"}}


def run(event, payload, env):
    return subprocess.run(
        [sys.executable, str(SCRIPT), event],
        input=json.dumps(payload).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )


def test_unreachable_server_exits_silently(monkeypatch):
    env = {"ENGRAMTIDE_HOOK_URL": "http://127.0.0.1:9/hooks/claude-code",
           "ENGRAMTIDE_HOOK_CLIENT_TIMEOUT": "1"}
    result = run("user-prompt-submit", {"session_id": "s"}, env)
    assert (result.returncode, result.stdout) == (0, b"")


def test_unknown_event_is_ignored():
    result = run("pre-tool-use", {"session_id": "s"}, {})
    assert (result.returncode, result.stdout) == (0, b"")


def test_forwards_body_headers_and_echoes_response():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["type"] = self.headers.get("Content-Type")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps(OUTPUT).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        env = {
            "ENGRAMTIDE_HOOK_URL": f"http://127.0.0.1:{server.server_port}/hooks/claude-code",
            "ENGRAMTIDE_HOOK_TOKEN": "secret",
        }
        result = run("user-prompt-submit", {"session_id": "s", "user_prompt": "你好"}, env)
    finally:
        thread.join(timeout=10)
        server.server_close()

    assert result.returncode == 0
    assert json.loads(result.stdout) == OUTPUT
    assert seen == {
        "path": "/hooks/claude-code/user-prompt-submit",
        "auth": "Bearer secret",
        "type": "application/json",
        "body": {"session_id": "s", "user_prompt": "你好"},
    }
