"""The memory's key: a glyphh of the last few content words of a text.

A context is keyed by what it means, not by its wording: each of the last WINDOW content words is a glyph from Ada's
lexicon (a word the lexicon does not know, an invented name, is its own symbol at twice the voice), plus the NSM
primes of the words together. Function words stay out of the key, with one exception: the word still being written
(a name's first piece, "He" of "Heliro") is never a stop word yet. `settled` drops that exception, for a text whose
last word is known to be finished (thinktank version_22_qwen_adapter, steps 3, 4c and 14)."""

import re

import numpy as np
from glyphh_kernel import symbol
from glyphh_lexicon import LEXICON
from glyphh_nsm import decompose

DIM, SEED = 2048, 7
WINDOW = 8  # content words
LOOKBACK = 400  # characters of text a key may read
NAME_VOICE, WORD_VOICE, PRIME_VOICE = 2.0, 1.0, 0.5

# Standard English function words (determiners, pronouns, auxiliaries, prepositions, conjunctions), as a word list.
_STOP = """
a an the of to and or is are was were be been it its it's this that these those in on at for with by from as
q what who which
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself
they them their theirs themselves whom whose am being have has had having do does did doing would should could
can will shall may might must but if because until while about against between into through during before after
above below up down out off over under again further then once here there when where why how all any both each
few more most other some such no nor not only own same so than too very just s t don now
"""
STOP: frozenset[str] = frozenset(_STOP.split())
WORD = re.compile(r"[a-z]+")


def known(word: str) -> bool:
    """Whether Ada's lexicon knows the word (an invented name or a value does not)."""
    return LEXICON.entry(word) is not None or LEXICON.lemma(word) is not None


class Keys:
    def __init__(self) -> None:
        self._glyph: dict[str, np.ndarray] = {}

    def glyph(self, word: str) -> np.ndarray:
        g = self._glyph.get(word)
        if g is None:
            voice = WORD_VOICE if known(word) else NAME_VOICE
            g = LEXICON.glyph(word, SEED, DIM, reach=2).astype(np.float32) * voice
            self._glyph[word] = g
        return g

    def context(self, text: str, settled: bool = False) -> np.ndarray:
        """The key of the context that ends where `text` ends (zeros when it has no content word)."""
        low = text[-LOOKBACK:].lower()
        found = WORD.findall(low)
        tail = found.pop() if not settled and found and low[-1:].isalpha() else None
        words = ([w for w in found if w not in STOP] + ([tail] if tail else []))[-WINDOW:]
        if not words:
            return np.zeros(DIM, np.float32)
        v = sum(self.glyph(w) for w in words)
        for prime in decompose(" ".join(words)):
            v = v + symbol(f"nsm:{prime}", SEED, DIM).astype(np.float32) * PRIME_VOICE
        return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)
