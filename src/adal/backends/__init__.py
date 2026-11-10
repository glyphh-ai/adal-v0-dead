"""The language model behind adaL. A backend is whatever can tokenize, lay out a chat and give next-token logits
one step at a time; the memory and the loop above it are the same for every one.

    mlx    Apple silicon, through MLX: local development and the model tests
    fake   no model: a scripted one for the unit tests

A GPU backend for production (the model, where it comes from and the infrastructure are chosen at deploy) is the
next one to add; nothing above this package changes for it."""

from typing import Protocol

import numpy as np


class Session(Protocol):
    def logits(self) -> np.ndarray:
        """The logits of the next token, after everything pushed so far."""

    def push(self, token: int) -> None:
        """Write one token."""


class Backend(Protocol):
    name: str
    stop: frozenset[int]

    def encode(self, text: str) -> list[int]: ...

    def piece(self, token: int) -> str:
        """The text of one token."""

    def chat(self, messages: list[dict[str, str]]) -> list[int]:
        """The prompt tokens for a conversation, ready for the assistant's reply."""

    def session(self, prompt: list[int]) -> Session: ...


def make(kind: str, model: str) -> Backend:
    if kind == "fake":
        from adal.backends.fake import FakeBackend

        return FakeBackend()
    if kind == "mlx":
        from adal.backends.mlx import MlxBackend

        return MlxBackend(model)
    raise RuntimeError(f"Unknown ADAL_BACKEND {kind!r} (mlx or fake).")
