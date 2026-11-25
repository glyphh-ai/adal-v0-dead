"""The test plan, fixed before any answer is seen: what is told, what is asked, and what counts as right.

50 invented people, four facts each (200 facts), told as a person would tell their assistant. Then every person's
account manager changes: 25 updates worded "X's account manager is now Y." and 25 worded value-first, "Update: Y
has taken over as X's account manager." Then 200 questions, each answered by Claude with whatever the memory under
test recalls for it:

  recall     100  two of each person's other facts (town, product, code), in a wording never told
  corrected   50  each person's account manager: right = the new one is said and the old one is not
  unknown     25  people never mentioned: right = no stored value is said, and Claude says it does not know
  unrelated   25  general questions: counted for how much memory is put in front of Claude for nothing

The bar, set before the run (Chris, 2026-11-11): the new engine has to clearly beat today's glyphh recall on
corrections and on reworded recall, or it is dropped."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tests.model.cases import ASKED, ATTRS, TOLD, make_cases  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PEOPLE = make_cases(50, seed=3)
STRANGERS = make_cases(25, seed=99)
UPDATE = ["{p}'s account manager is now {v}.", "Update: {v} has taken over as {p}'s account manager."]
UNRELATED = [
    "How long should I boil an egg?", "What causes the seasons?", "What is the capital of Australia?",
    "How do I remove a coffee stain?", "Why is the ocean salty?", "What is compound interest?",
    "How do I fix a squeaky door?", "Why do cats purr?", "How do I start running?", "What is a black hole?",
    "How do I sharpen a kitchen knife?", "What makes a good password?", "Why do leaves change color?",
    "How do I make cold brew coffee?", "What is inflation?", "How often should I water a cactus?",
    "Why does bread go stale?", "How can I sleep better?", "What is a neutron star?", "How do I iron a shirt?",
    "Why do we yawn?", "How do I descale a kettle?", "What is a mortgage?", "Why does ice float?",
    "How do I get better at chess?",
]  # fmt: skip


def told() -> list[dict]:
    """Everything told, in order: the 200 facts, then the 50 updates."""
    out = [{"text": TOLD[a].format(p=c["p"], v=c[a]), "person": i, "attr": a} for i, c in enumerate(PEOPLE) for a in ATTRS]
    for i, c in enumerate(PEOPLE):
        out.append({"text": UPDATE[i % 2].format(p=c["p"], v=c["new"]), "person": i, "attr": "manager", "update": i % 2})
    return out


def questions() -> list[dict]:
    qs = []
    others = [a for a in ATTRS if a != "manager"]
    for i, c in enumerate(PEOPLE):
        for k in range(2):
            a = others[(i + k) % 3]
            qs.append({"kind": "recall", "q": ASKED[a][(i + k) % 2].format(p=c["p"]), "want": c[a]})
        qs.append({
            "kind": "corrected", "q": ASKED["manager"][i % 2].format(p=c["p"]), "want": c["new"], "old": c["manager"],
            "update": i % 2,
        })  # fmt: skip
    for i, c in enumerate(STRANGERS):
        a = ATTRS[i % 4]
        qs.append({"kind": "unknown", "q": ASKED[a][i % 2].format(p=c["p"])})
    qs += [{"kind": "unrelated", "q": q} for q in UNRELATED]
    for n, q in enumerate(qs):
        q["id"] = n
    return qs


def every_value() -> set[str]:
    return {c[a] for c in PEOPLE for a in ATTRS} | {c["new"] for c in PEOPLE}


def save(name: str, data) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(data, fh, indent=1)


def load(name: str):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)
