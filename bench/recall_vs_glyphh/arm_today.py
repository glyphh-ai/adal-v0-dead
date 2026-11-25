"""Arm A: today's glyphh memory, as the server runs it, in-process.

The Agentic memory template's own spec (server: ops/templates/library/agentic-memory.json) on the kernel the server
pins, with the server's own calls (ops/templates/glyphh.py): remember = record {cue: text, kind, key: hash of the
text} with the text as the outcome; recall = facts {cue: what was asked}, top 8 (the template's prompt pull for
memory), min_score 0.05. Nothing of Chris's own glyphh is touched: the model lives in this process only.

Two rows: as told (an update is one more memory), and "forget" (before each update the old memory is forgotten, as
the product asks Claude to do when something is no longer true; an upper bound, since it assumes Claude always does).

Run with the server's environment:  ../../../server/.venv/bin/python arm_today.py"""

import hashlib
import json

import glyphh_spec

from plan import load, questions, save, told  # noqa: F401

SERVER = "/Users/timmetim/development/glyphh/server/src/glyphh_server/ops/templates/library/agentic-memory.json"
TOP, LEAST = 8, 0.05


def data(**roles):
    return {"m": {"s": roles}}


def run(forget: bool) -> dict:
    spec = json.load(open(SERVER))["spec"]
    model = glyphh_spec.Model(glyphh_spec.Spec.from_json({**spec, "name": "memory"}))
    ids: dict[tuple[int, str], str] = {}
    for t in told():
        who = (t["person"], t["attr"])
        if forget and "update" in t:
            model.forget(ids[who])
        key = hashlib.sha256(" ".join(t["text"].lower().split()).encode()).hexdigest()[:16]
        ids[who] = model.record(data(cue=t["text"], kind="memory", key=key), t["text"])
    out = {}
    for q in questions():
        found = model.facts(data(cue=q["q"]), top=TOP, min_score=LEAST)
        out[q["id"]] = [f["outcome"] for f in found]
    return out


if __name__ == "__main__":
    save("recalled_today.json", run(False))
    save("recalled_today_forget.json", run(True))
    print("today's recall written")
