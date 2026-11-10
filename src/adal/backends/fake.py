"""A language model for the unit tests: a script instead of weights.

Its tokens are shaped like a real tokenizer's: a word carries its leading space, a long word is several pieces
(" Nipi", "ra"), punctuation stands alone. `script(messages)` is what this model would say; it puts 0.4 on the
script's next token while the reply still follows the script, and on stopping once it does not or the script is
done. So the memory has a real distribution to move, and a test can see exactly where it did."""

import re
from collections.abc import Callable

import numpy as np

VOCAB = 8192
STOP = 0
_TOKEN = re.compile(r" ?[A-Za-z]+|[^A-Za-z]", re.DOTALL)
PIECE = 4  # letters in one piece of a word


def _dont_know(messages: list[dict[str, str]]) -> str:
    return "I do not know."


class _Session:
    def __init__(self, script: list[int]) -> None:
        self.script, self.reply = script, []

    def logits(self) -> np.ndarray:
        n = len(self.reply)
        want = self.script[n] if self.reply == self.script[:n] and n < len(self.script) else STOP
        p = np.full(VOCAB, 0.6 / (VOCAB - 1))
        p[want] = 0.4
        return np.log(p)

    def push(self, token: int) -> None:
        self.reply.append(token)


class FakeBackend:
    name = "fake"
    stop = frozenset({STOP})

    def __init__(self, script: Callable[[list[dict[str, str]]], str] = _dont_know) -> None:
        self.script = script
        self._messages: list[dict[str, str]] = []
        self._ids: dict[str, int] = {"": STOP}
        self._pieces: list[str] = [""]

    def _id(self, piece: str) -> int:
        token = self._ids.get(piece)
        if token is None:
            if len(self._pieces) == VOCAB:
                raise RuntimeError("The fake backend's vocabulary is full.")
            token = self._ids[piece] = len(self._pieces)
            self._pieces.append(piece)
        return token

    def encode(self, text: str) -> list[int]:
        out = []
        for m in _TOKEN.finditer(text):
            word = m.group()
            head = len(word) - len(word.lstrip(" ")) + PIECE
            out.append(self._id(word[:head]))
            out += [self._id(word[i : i + PIECE]) for i in range(head, len(word), PIECE)]
        return out

    def piece(self, token: int) -> str:
        return self._pieces[token]

    def chat(self, messages: list[dict[str, str]]) -> list[int]:
        self._messages = messages
        return self.encode("\n".join(m["content"] for m in messages) + "\n")

    def session(self, prompt: list[int]) -> _Session:
        return _Session(self.encode(self.script(self._messages)))
