"""Claude answers every question, once per arm, with what that arm recalled put before it the way the glyphh
plugin's hook does it today (the same wrapper and line format for every arm), then the replies are scored.

Run with adaL's environment:  ../../.venv/bin/python ask.py"""

import re

from claude import MODEL, ask_all
from plan import every_value, load, questions, save

SYSTEM = "You are a helpful assistant. Answer in one or two sentences."
WRAP = "From your glyphh, your memory across every AI you use:\n\nWhat it knows about you:\n"
ARMS = {
    "today": "Today's glyphh recall",
    "today_forget": "Today's, if Claude always forgets the old memory first",
    "new": "New engine, as told",
    "new_ingest": "New engine, updates through ingest",
}
UNSURE = re.compile(r"don't have|do not have|don't know|do not know|no information|not sure|no record|unable|"
                    r"can't find|cannot find|isn't any|is no |not aware|doesn't (include|mention|contain)", re.I)  # fmt: skip


def prompt(q: dict, lines: list[str]) -> tuple[str, str]:
    context = (WRAP + "\n".join(f"- {t}" for t in lines) + "\n\n") if lines else ""
    return SYSTEM, context + q["q"]


def score(qs: list[dict], recalled: dict, replies: list[str]) -> dict:
    values = every_value()
    by = {k: [(q, r) for q, r in zip(qs, replies, strict=True) if q["kind"] == k] for k in ("recall", "corrected", "unknown", "unrelated")}
    corrected = by["corrected"]
    row = {
        "recall": sum(q["want"] in r for q, r in by["recall"]),
        "corrected_strict": sum(q["want"] in r and q["old"] not in r for q, r in corrected),
        "corrected_new_said": sum(q["want"] in r for q, r in corrected),
        "corrected_old_said": sum(q["old"] in r for q, r in corrected),
        "unknown_clean": sum(not any(v in r for v in values) and bool(UNSURE.search(r)) for q, r in by["unknown"]),
        "unknown_value_said": sum(any(v in r for v in values) for q, r in by["unknown"]),
        "unrelated_value_said": sum(any(v in r for v in values) for q, r in by["unrelated"]),
    }
    for u, name in ((0, "is_now"), (1, "value_first")):
        part = [(q, r) for q, r in corrected if q["update"] == u]
        row[f"corrected_strict_{name}"] = sum(q["want"] in r and q["old"] not in r for q, r in part)
    for k, pairs in by.items():
        row[f"lines_{k}"] = round(sum(len(recalled[str(q["id"])]) for q, _ in pairs) / len(pairs), 1)
    return row


if __name__ == "__main__":
    qs = questions()
    results = {"model": MODEL}
    for arm, label in ARMS.items():
        recalled = load(f"recalled_{arm}.json")
        replies = ask_all([prompt(q, recalled[str(q["id"])]) for q in qs])
        save(f"replies_{arm}.json", [{**q, "recalled": recalled[str(q["id"])], "reply": r} for q, r in zip(qs, replies, strict=True)])
        results[arm] = score(qs, recalled, replies)
        r = results[arm]
        print(f"{label}\n   recall {r['recall']}/100 | corrected: new said and old not {r['corrected_strict']}/50 "
              f"(is-now {r['corrected_strict_is_now']}/25, value-first {r['corrected_strict_value_first']}/25; new said "
              f"{r['corrected_new_said']}, old said {r['corrected_old_said']}) | unknown people: clean "
              f"{r['unknown_clean']}/25, a stored value said {r['unknown_value_said']} | lines put in view: recall "
              f"{r['lines_recall']}, unknown {r['lines_unknown']}, unrelated {r['lines_unrelated']}", flush=True)  # fmt: skip
        save("results.json", results)
