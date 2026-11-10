"""The loop: a model, its tenants' memories, and what happens at each word.

Telling writes a fact to the tenant's record (store.py) and then into their index (memory.py). Chat generates one
token at a time; at each one the tenant's memory is mixed into the model's own probabilities, keyed on the user's
last message and the reply so far. Ingest has the model rewrite free text into single sentences that each start
with their subject, and code checks every line against the text before it is told (thinktank step13).

One model, one generation at a time: `_model` serializes them (batching is the GPU backend's job, later). Memories
are loaded on first use by replaying the tenant's facts, at most `max_loaded` kept in RAM, and each has its own
lock, so one tenant's load never stalls another's reply for longer than a token."""

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from adal.backends import Backend
from adal.config import Settings
from adal.errors import Invalid
from adal.keys import WORD, Keys, known
from adal.memory import Memory
from adal.store import Fact, Store, check_tenant

REWRITE = (
    "Read the note below. On the first line write 'Subject:' and the full name of the person or thing the note is "
    "about. Then write its lasting facts, one short sentence per line. Every sentence must START with the "
    'subject\'s full name, for example "<name>\'s <something> is <value>." or "<name> <verb> <value>." Use only '
    "what the note says, and leave out passing events.\n\nNote: {note}"
)
# What the model is told when the memory holds something about the question. The facts are in view so the reply
# is written knowing them; the memory also puts the value itself into the first word when one fact clearly answers.
REMEMBER = "You remember these facts, which the user told you. They are true and current; answer from them:"
UNKNOWN = "If you are asked about a person or thing you were told nothing about and do not know, say so."
_BULLET = re.compile(r"^\s*(\d+[.)]|[-*•])\s*")
ROLES = ("system", "user", "assistant")


@dataclass
class _Loaded:
    memory: Memory
    lock: threading.RLock = field(default_factory=threading.RLock)
    texts: dict[int, str] = field(default_factory=dict)  # each fact's text, by its number in the record


@dataclass(frozen=True)
class Reply:
    text: str
    fired: int  # tokens the memory took part in
    tokens: int
    finished: bool  # the model stopped by itself (not cut at the token limit)
    recalled: int = 0  # facts put in view


class Engine:
    def __init__(self, backend: Backend, settings: Settings) -> None:
        self.backend, self.settings = backend, settings
        self.store = Store(settings.data, settings.max_facts, settings.max_fact_chars)
        self.keys = Keys()  # word glyphs, shared by every tenant
        self._loaded: OrderedDict[str, _Loaded] = OrderedDict()
        self._tenants = threading.Lock()
        self._model = threading.Lock()

    # ----------------------------------------------------------------- memory
    def _memory(self, tenant: str) -> _Loaded:
        """The tenant's index, loaded from their record on first use."""
        check_tenant(tenant)
        with self._tenants:
            got = self._loaded.get(tenant)
            if got is not None:
                self._loaded.move_to_end(tenant)
                return got
            s = self.settings
            got = self._loaded[tenant] = _Loaded(
                Memory(self.backend.piece, self.keys, s.tau, s.delta, s.beta, s.tau_replace)
            )
            got.lock.acquire()  # held while it fills, so nobody reads it half-built
            while len(self._loaded) > s.max_loaded:
                self._loaded.popitem(last=False)
        try:
            for fact in self.store.facts(tenant):
                got.memory.learn(self.backend.encode(fact.text), fact.seq)
                got.texts[fact.seq] = fact.text
        except BaseException:
            with self._tenants:
                self._loaded.pop(tenant, None)
            raise
        finally:
            got.lock.release()
        return got

    def tell(self, tenant: str, texts: list[str], source: str | None = None) -> list[Fact]:
        loaded = self._memory(tenant)
        with loaded.lock:
            facts = self.store.tell(tenant, texts, source)  # the record first: what is answered is durable
            for fact in facts:
                loaded.memory.learn(self.backend.encode(fact.text), fact.seq)
                loaded.texts[fact.seq] = fact.text
        return facts

    def forget(self, tenant: str, fact_id: str) -> None:
        check_tenant(tenant)
        self.store.forget(tenant, fact_id)
        with self._tenants:
            self._loaded.pop(tenant, None)  # rebuilt from what is left, on next use

    def forget_all(self, tenant: str) -> None:
        check_tenant(tenant)
        self.store.forget_all(tenant)
        with self._tenants:
            self._loaded.pop(tenant, None)

    def facts(self, tenant: str, q: str = "", after: int = 0, limit: int = 50) -> list[Fact]:
        check_tenant(tenant)
        return self.store.facts(tenant, q, after, limit)

    # ------------------------------------------------------------------- chat
    def _generate(self, messages: list[dict[str, str]], loaded: _Loaded | None, heard: str, most: int) -> Reply:
        with self._model:
            session = self.backend.session(self.backend.chat(messages))
            reply, fired, n, finished = "", 0, 0, False
            for n in range(1, most + 1):
                logits = session.logits()
                p = np.exp(logits - logits.max())
                p /= p.sum()
                if loaded is not None:
                    with loaded.lock:
                        p, hit = loaded.memory.mix(heard, reply, p)
                    fired += hit
                token = int(p.argmax())
                if token in self.backend.stop:
                    finished = True
                    break
                reply += self.backend.piece(token)
                session.push(token)
            return Reply(reply.strip(), fired, n, finished)

    def chat(self, tenant: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> Reply:
        if not messages or messages[-1].get("role") != "user":
            raise Invalid("The last message must be the user's.", reason="messages")
        for m in messages:
            if m.get("role") not in ROLES or not isinstance(m.get("content"), str):
                raise Invalid("A message is {role: system|user|assistant, content: text}.", reason="messages")
        most = min(max_tokens or self.settings.max_tokens, self.settings.max_tokens)
        loaded = self._memory(tenant)
        heard = messages[-1]["content"] + "\n"
        with loaded.lock:
            recalled = [loaded.texts[f] for f in loaded.memory.recall(heard) if f in loaded.texts]
        note = (REMEMBER + "".join(f"\n- {t}" for t in recalled)) if recalled else UNKNOWN
        if messages[0]["role"] == "system":
            messages = [{"role": "system", "content": messages[0]["content"] + "\n\n" + note}, *messages[1:]]
        else:
            messages = [{"role": "system", "content": note}, *messages]
        reply = self._generate(messages, loaded, heard, most)
        return Reply(reply.text, reply.fired, reply.tokens, reply.finished, len(recalled))

    # ----------------------------------------------------------------- ingest
    def rewrite(self, text: str) -> tuple[list[str], int]:
        """A note as facts: the lines that pass the check, and how many the model wrote that did not."""
        # a rewrite is about as long as its note: its own limit, not a chat reply's
        most = max(self.settings.max_tokens, len(self.backend.encode(text)) * 2 + 40)
        out = self._generate([{"role": "user", "content": REWRITE.format(note=text)}], None, "", most)
        lines = [_BULLET.sub("", x).strip() for x in out.text.splitlines()]
        lines = [x for x in lines if x]
        if not out.finished:
            lines = lines[:-1]  # cut at the token limit: the last line may be half a fact
        if not lines or not lines[0].lower().startswith("subject:"):
            return [], len(lines)
        subject = lines[0].split(":", 1)[1].strip().rstrip(".")
        have = set(WORD.findall(text.lower()))
        kept = [
            x
            for x in lines[1:]
            if len(subject) > 2
            and subject in text
            and x.startswith(subject)
            and len(x) <= self.settings.max_fact_chars
            # every name or value in the line is the note's own: the model may reword, never add
            and all(w in have for w in WORD.findall(x.lower()) if not known(w))
        ]
        return kept, len(lines) - 1 - len(kept)

    def ingest(self, tenant: str, text: str, source: str | None = None) -> tuple[list[Fact], int]:
        check_tenant(tenant)
        if not text.strip():
            raise Invalid("There is nothing to read.", reason="text")
        kept, dropped = self.rewrite(text)
        return (self.tell(tenant, kept, source) if kept else []), dropped
