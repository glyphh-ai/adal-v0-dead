"""Arm B: the new engine's memory (adal.memory.Memory, as it is in this repo), recall only: no local model, no
steering. Facts are told in order; a question recalls at most 8 facts at tau or closer, as adaL's chat does.

Two rows: as told, and "ingest", where each update first goes through adaL's ingest (the REWRITE instruction, here
answered by Claude, and the same check as Engine.rewrite: the subject is in the text, each line starts with it,
every name or value in a line is the text's own), which is how free text is meant to reach this memory.

Run with adaL's environment:  ../../.venv/bin/python arm_new.py"""

import re

from adal.backends.fake import FakeBackend
from adal.engine import REWRITE
from adal.keys import WORD, known
from adal.memory import Memory

from claude import ask_all
from plan import questions, save, told

_BULLET = re.compile(r"^\s*(\d+[.)]|[-*•])\s*")


def checked(text: str, written: str) -> list[str]:
    """Engine.rewrite's check, on a rewrite that came from Claude."""
    lines = [x for x in (_BULLET.sub("", x).strip() for x in written.splitlines()) if x]
    if not lines or not lines[0].lower().startswith("subject:"):
        return []
    subject = lines[0].split(":", 1)[1].strip().rstrip(".")
    have = set(WORD.findall(text.lower()))
    return [
        x for x in lines[1:]
        if len(subject) > 2 and subject in text and x.startswith(subject) and len(x) <= 300
        and all(w in have for w in WORD.findall(x.lower()) if not known(w))
    ]  # fmt: skip


def run(ingest: bool) -> tuple[dict, dict]:
    tok = FakeBackend()  # only its tokenizer: where words and their pieces begin
    memory, texts = Memory(tok.piece), {}
    everything = told()
    rewritten = {}
    if ingest:
        updates = [t for t in everything if "update" in t]
        answers = ask_all([("You rewrite notes into facts.", REWRITE.format(note=t["text"])) for t in updates])
        rewritten = {t["text"]: checked(t["text"], a) for t, a in zip(updates, answers, strict=True)}
    n = 0
    for t in everything:
        for line in rewritten.get(t["text"], [t["text"]]) if ingest and "update" in t else [t["text"]]:
            n += 1
            memory.learn(tok.encode(line), n)
            texts[n] = line
    out = {q["id"]: [texts[f] for f in memory.recall(q["q"] + "\n")] for q in questions()}
    return out, rewritten


if __name__ == "__main__":
    save("recalled_new.json", run(False)[0])
    out, rewritten = run(True)
    save("recalled_new_ingest.json", out)
    save("ingest_rewrites.json", rewritten)
    kept = sum(bool(v) for v in rewritten.values())
    print(f"new engine's recall written; ingest kept a fact for {kept} of {len(rewritten)} updates")
