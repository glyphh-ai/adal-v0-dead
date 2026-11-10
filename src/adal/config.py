"""adaL's settings, all from the environment, so a deploy decides the model, where it comes from and where the
data lives without a code change."""

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    backend: str  # "mlx" (Apple silicon, local) or "fake" (tests; no model)
    model: str  # the backend's model: a local path, or a name in the Hugging Face cache
    data: str  # the directory that holds every tenant's memory; a volume that survives restarts
    token: str  # the bearer secret the Glyphh server calls with; "" only with ADAL_DEV=1
    dev: bool
    # The memory's thresholds, as measured in thinktank/version_22_qwen_adapter (FINDINGS.md):
    tau: float  # a stored context this close to the text fires (step4c, 3,000 facts: 0.449)
    delta: float  # every context within delta of the best is mixed (step4b)
    beta: float  # the model's own weight against one remembered telling (step1)
    tau_replace: float  # a new fact this close to a stored one, continuing differently, corrects it (step14)
    max_facts: int  # per tenant
    max_fact_chars: int
    max_loaded: int  # tenants kept in RAM at once; the rest are reloaded from disk when asked for
    max_tokens: int  # the most a reply may run


def load() -> Settings:
    dev = os.environ.get("ADAL_DEV", "") == "1"
    token = os.environ.get("ADAL_TOKEN", "")
    if not token and not dev:
        raise RuntimeError("ADAL_TOKEN is not set (set ADAL_DEV=1 to run without one on your own machine).")
    return Settings(
        backend=os.environ.get("ADAL_BACKEND", "mlx"),
        model=os.environ.get("ADAL_MODEL", "Qwen3-14B-4bit"),
        data=os.environ.get("ADAL_DATA", "./data"),
        token=token,
        dev=dev,
        tau=_f("ADAL_TAU", 0.45),
        delta=_f("ADAL_DELTA", 0.05),
        beta=_f("ADAL_BETA", 1.0),
        tau_replace=_f("ADAL_TAU_REPLACE", 0.95),
        max_facts=_i("ADAL_MAX_FACTS", 5000),
        max_fact_chars=_i("ADAL_MAX_FACT_CHARS", 300),
        max_loaded=_i("ADAL_MAX_LOADED", 4),
        max_tokens=_i("ADAL_MAX_TOKENS", 200),
    )
