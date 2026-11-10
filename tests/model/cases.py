"""Invented customers for the model tests: nobody's training data knows them, so a right answer can only come
from the memory. The same shape as thinktank's step12 to step14."""

import numpy as np

ATTRS = ["manager", "town", "product", "code"]
TOLD = {
    "manager": "{p}'s account manager is {v}.",
    "town": "{p}'s home town is {v}.",
    "product": "{p} bought the {v} last spring.",
    "code": "{p}'s discount code is {v}.",
}
ASKED = {
    "manager": ["Who is the account manager for {p}?", "Which person manages {p}'s account?"],
    "town": ["Which town does {p} live in?", "What is {p}'s home town?"],
    "product": ["Which product did {p} buy?", "What item did {p} purchase last spring?"],
    "code": ["What discount code did {p} get?", "Which discount code can {p} use?"],
}
NOTE = (
    "A message came in from {p} over the weekend. They were given the discount code {code}. Home is the town of "
    "{town}. The line dropped once and was reconnected. The account manager on file is {manager}. The product in "
    "question is the {product}. The call lasted about ten minutes."
)


def invented(rng: np.random.Generator, n: int) -> list[str]:
    cons, vow = "bdfghklmnprstvz", "aeiou"
    out: list[str] = []
    while len(out) < n:
        w = "".join(rng.choice(list(cons)) + rng.choice(list(vow)) for _ in range(rng.integers(2, 4))).capitalize()
        if w not in out:
            out.append(w)
    return out


def make_cases(n: int, seed: int = 0) -> list[dict[str, str]]:
    words = invented(np.random.default_rng(seed), 7 * n)
    return [
        {"p": f"{w[0]} {w[1]}", **dict(zip(ATTRS, w[2:6], strict=True)), "new": w[6]}
        for w in (words[7 * i : 7 * i + 7] for i in range(n))
    ]
