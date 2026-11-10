"""One tenant's memory, in RAM: contexts and the values that followed them.

A told fact leaves two kinds of row:

  value   where a value starts: a word Ada's lexicon does not know (a name, a code), or the fact's last content
          word ("X's account manager is" -> " Nip"). The key is the context with every finished stop word out
          of it, so "is", "is now" and a question about it all meet at one row
  mid     inside that word (" Nip" -> "ira"): it carries a value of several tokens to its end, and only ever
          continues the very piece it followed

Everything else in a fact (grammar, ordinary words) is the model's to write; the memory holds the values.

Recall is answer-first. At the start of a reply, the question's key is compared with every value row: if one value
is clearly the nearest (at `tau` or closer, and no different value within `delta` of it), it is mixed into the
model's own next-token probabilities, and its mid rows finish it. When two different values are about as near, the
memory stays silent rather than guess. After the value, the model writes on by itself. Nothing in the model
changes.

A fact told again in the same context is a correction: the row's other continuations are halved, so one telling
replaces one telling. A new value whose context sits within `tau_replace` of another value row that continues
differently corrects that one too.

This index is derived: it is rebuilt from the tenant's facts (store.py), in the order they were told, whenever the
tenant is loaded or a fact is forgotten. It is never the record.

Where the pieces were measured: thinktank/version_22_qwen_adapter FINDINGS.md, steps 3, 4b, 4c, 13 and 14 (keys,
mixing, corrections, as ranked answers), and this repo's model tests (free generation)."""

import hashlib
import re
from collections.abc import Callable, Iterator, Sequence

import numpy as np

from adal.keys import DIM, LOOKBACK, STOP, Keys, known

_NEXT_WORD = re.compile(r"\s*([A-Za-z]+)")
_TAIL = re.compile(r"[A-Za-z]+$")


def _halve(counts: dict[int, int], keep: int | None = None) -> None:
    for other in [o for o in counts if o != keep]:
        counts[other] //= 2
        if not counts[other]:
            del counts[other]


def _tail(text: str) -> str:
    """The word being written at the end of `text` ("" when it ends between words)."""
    found = _TAIL.search(text[-LOOKBACK:])
    return found.group().lower() if found else ""


class Memory:
    def __init__(
        self,
        piece: Callable[[int], str],
        keys: Keys | None = None,
        tau: float = 0.45,
        delta: float = 0.05,
        beta: float = 1.0,
        tau_replace: float = 0.95,
    ) -> None:
        self.piece, self.keys = piece, keys or Keys()
        self.tau, self.delta, self.beta, self.tau_replace = tau, delta, beta, tau_replace
        self._buf = np.zeros((1024, DIM), np.float32)
        self._value = np.zeros(1024, bool)  # a value row (else a mid row)
        self.counts: list[dict[int, int]] = []
        self._row: dict[bytes, int] = {}
        self._mid: dict[str, list[int]] = {}  # the piece being written -> its mid rows
        self._fact: list[int] = []  # the fact that last wrote each row

    def __len__(self) -> int:
        return len(self.counts)

    # ------------------------------------------------------------------ learn
    def _positions(self, tokens: Sequence[int]) -> Iterator[tuple[int, np.ndarray, str | None]]:
        """(position, key, the piece being written or None for a value row) for each position that leaves a row."""
        pieces = [self.piece(t) for t in tokens]
        starts: dict[int, str] = {}  # position -> the content word that starts after it
        for t in range(len(tokens) - 1):
            if not pieces[t + 1][:1].isalpha():
                word = _NEXT_WORD.match("".join(pieces[t + 1 : t + 8]))
                if word and word.group(1).lower() not in STOP:
                    starts[t] = word.group(1).lower()
        last = max(starts, default=-1)  # a fact's last content word is a value even when it is an ordinary word
        text, inside = "", False  # inside: the word being written is a value
        for t in range(len(tokens) - 1):
            text = (text + pieces[t])[-LOOKBACK:]
            if pieces[t + 1][:1].isalpha():  # the next token continues the word
                if inside:
                    yield t, self.keys.context(text), _tail(text)
                continue
            inside = t in starts and (t == last or not known(starts[t]))
            if inside:
                yield t, self.keys.context(text, settled=True), None

    def learn(self, tokens: Sequence[int], fact: int = 0) -> None:
        """Remember one told fact (`fact` is its number in the record)."""
        for t, key, tail in self._positions(tokens):
            if not key.any():
                continue
            value = tail is None
            nxt = int(tokens[t + 1])
            n = len(self.counts)
            digest = hashlib.blake2b(key.tobytes() + (tail or "").encode(), digest_size=16).digest()
            row = self._row.get(digest)
            if value and n:
                near = np.flatnonzero(((self._buf[:n] @ key) >= self.tau_replace) & self._value[:n])
                for j in near:
                    if j != row and self.counts[j] and nxt not in self.counts[j]:
                        _halve(self.counts[j])
            if row is not None:
                _halve(self.counts[row], keep=nxt)
                self.counts[row][nxt] = self.counts[row].get(nxt, 0) + 1
                self._fact[row] = fact
                continue
            if n == len(self._buf):
                self._buf = np.vstack([self._buf, np.zeros_like(self._buf)])
                self._value = np.concatenate([self._value, np.zeros_like(self._value)])
            self._buf[n], self._value[n] = key, value
            self._row[digest] = n
            self.counts.append({nxt: 1})
            self._fact.append(fact)
            if not value:
                self._mid.setdefault(tail, []).append(n)

    # ----------------------------------------------------------------- recall
    def _top(self, row: int) -> int | None:
        return max(self.counts[row], key=self.counts[row].get) if self.counts[row] else None

    def recall(self, question: str, most: int = 8) -> list[int]:
        """The facts whose values sit at `tau` or closer to `question`, nearest first (their numbers)."""
        n = len(self.counts)
        if not n:
            return []
        s = np.where(self._value[:n], self._buf[:n] @ self.keys.context(question, settled=True), -1.0)
        out: list[int] = []
        for j in np.argsort(-s):
            if s[j] < self.tau or len(out) == most:
                break
            if self.counts[j] and self._fact[j] not in out:
                out.append(self._fact[j])
        return out

    def answer(self, question: str) -> list[tuple[int, float]]:
        """The value rows that answer `question`, with their closeness: the nearest one and any that agree with it
        within `delta`. Empty when nothing is near enough, or when two different values are about as near."""
        n = len(self.counts)
        if not n:
            return []
        s = np.where(self._value[:n], self._buf[:n] @ self.keys.context(question, settled=True), -1.0)
        for j in np.flatnonzero(s >= self.tau):  # a row whose value was corrected away holds nothing
            if not self.counts[j]:
                s[j] = -1.0
        best = int(s.argmax())
        if s[best] < self.tau:
            return []
        near = [int(j) for j in np.flatnonzero(s >= max(self.tau, float(s[best]) - self.delta))]
        if any(self._top(j) != self._top(best) for j in near):
            return []
        return [(j, float(s[j])) for j in near]

    def _mix(self, p: np.ndarray, rows: list[tuple[int, float]]) -> np.ndarray:
        num, den = np.zeros_like(p), 0.0
        for j, w in rows:
            for token, count in self.counts[j].items():
                num[token] += w * count
            den += w * sum(self.counts[j].values())
        return (num + self.beta * p) / (den + self.beta) if den else p

    def mix(self, heard: str, reply: str, p: np.ndarray) -> tuple[np.ndarray, bool]:
        """The model's next-token probabilities with what is remembered mixed in, and whether anything was.
        `heard` is what the reply answers (the user's message), `reply` what has been written of it so far."""
        if not self.counts:
            return p, False
        if not reply:
            rows = self.answer(heard)
            return (self._mix(p, rows), True) if rows else (p, False)
        tail = _tail(reply)
        mids = self._mid.get(tail) if tail else None
        if not mids:
            return p, False
        s = self._buf[mids] @ self.keys.context(heard + reply)
        best = float(s.max())
        if best < self.tau:
            return p, False
        rows = [(mids[i], float(s[i])) for i in np.flatnonzero(s >= max(self.tau, best - self.delta))]
        rows = [(j, w) for j, w in rows if self.counts[j]]
        return (self._mix(p, rows), True) if rows else (p, False)
