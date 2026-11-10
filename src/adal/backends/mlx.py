"""The MLX backend: an open-weight model on Apple silicon, opened up to its next-token logits.

`model` is a local directory, or the name of an mlx-community model already in the Hugging Face cache
("Qwen3-14B-4bit"). Nothing is downloaded here: fetching a model is the deploy's job."""

import glob
import os

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def _path(model: str) -> str:
    if os.path.isdir(model):
        return model
    found = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--mlx-community--{model}/snapshots/*"))
    if not found:
        raise RuntimeError(f"No model at {model!r}, nor mlx-community/{model} in the Hugging Face cache.")
    return found[0]


class _Session:
    def __init__(self, model, prompt: list[int]) -> None:
        self._model = model
        self._cache = make_prompt_cache(model)
        self._last = self._step(prompt)

    def _step(self, tokens: list[int]) -> np.ndarray:
        out = self._model(mx.array(np.array(tokens, np.int32)[None]), cache=self._cache)
        return np.array(out[0, -1].astype(mx.float32), np.float64)

    def logits(self) -> np.ndarray:
        return self._last

    def push(self, token: int) -> None:
        self._last = self._step([token])


class MlxBackend:
    def __init__(self, model: str) -> None:
        self.name = os.path.basename(model.rstrip("/"))
        self._model, self._tok = load(_path(model))
        self.stop = frozenset(
            t for t in (self._tok.eos_token_id, self._tok.convert_tokens_to_ids("<|im_end|>")) if t is not None
        )
        self._piece: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def piece(self, token: int) -> str:
        p = self._piece.get(token)
        if p is None:
            p = self._piece[token] = self._tok.decode([int(token)])
        return p

    def chat(self, messages: list[dict[str, str]]) -> list[int]:
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.encode(text)

    def session(self, prompt: list[int]) -> _Session:
        return _Session(self._model, prompt)
