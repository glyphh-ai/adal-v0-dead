"""Claude, one clean call per prompt, through the Claude CLI: no tools, no MCP servers, no hooks, no settings, no
session kept. (Claude Code still adds a line of its own about the account; nothing from anyone's memory.) Answers
are cached on disk by prompt, so a rerun asks only what is new, and identical prompts in two arms get one answer."""

import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

MODEL = "claude-sonnet-5-5"
_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_HERE, "out", "claude_cache.json")
_lock = threading.Lock()
_cache: dict[str, str] = json.load(open(_CACHE)) if os.path.exists(_CACHE) else {}


def _one(system: str, user: str) -> str:
    key = hashlib.sha256(f"{MODEL}\n{system}\n{user}".encode()).hexdigest()
    with _lock:
        if key in _cache:
            return _cache[key]
    for attempt in range(3):
        done = subprocess.run(
            ["claude", "-p", "--model", MODEL, "--system-prompt", system, "--tools", "", "--strict-mcp-config",
             "--no-session-persistence", "--setting-sources", "", "--disable-slash-commands"],
            input=user, capture_output=True, text=True, timeout=180, cwd="/tmp",
        )  # fmt: skip
        if done.returncode == 0 and done.stdout.strip():
            break
    else:
        raise RuntimeError(f"claude failed: {done.stderr[-300:]}")
    with _lock:
        _cache[key] = done.stdout.strip()
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with open(_CACHE, "w") as fh:
            json.dump(_cache, fh)
    return _cache[key]


def ask_all(prompts: list[tuple[str, str]], workers: int = 6) -> list[str]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda p: _one(*p), prompts))
