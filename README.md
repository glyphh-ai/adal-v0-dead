# adaL

A language model with Glyphh's memory in it. Tell it a fact once: it remembers it, answers from it in wordings it
was never told, takes a correction in place and forgets nothing else. The model's weights never change; each
customer's memory sits beside it.

adaL is a private service. It is called by the Glyphh server (which signs people in, checks their plan and meters
them), never by a customer directly. The memory keys on the Ada kernel's meaning encoding (`ada-private`).

**Status: facts, private beta quality, on a local model.** What is built and tested is in
[docs/architecture.md](docs/architecture.md), with what is not. Rules (instructions that carry across sessions) are
still research (`timmetim/thinktank`, `version_22_qwen_adapter`) and are not in this service.

## Run it

```sh
uv sync --extra mlx                      # Apple silicon; the kernel needs read access to glyphh-ai/ada-private
ADAL_DEV=1 ADAL_MODEL=Qwen3-14B-4bit ADAL_DATA=./data uv run adal
```

| Setting | What | Default |
|---|---|---|
| `ADAL_BACKEND` | `mlx` (Apple silicon) or `fake` (tests) | `mlx` |
| `ADAL_MODEL` | A model directory, or an `mlx-community` model already in the Hugging Face cache | `Qwen3-14B-4bit` |
| `ADAL_DATA` | Where every tenant's memory is kept. Put it on a volume that survives restarts | `./data` |
| `ADAL_TOKEN` | The bearer secret the Glyphh server calls with. Required unless `ADAL_DEV=1` | |
| `ADAL_HOST`, `ADAL_PORT` | Where it listens | `127.0.0.1`, `8700` |
| `ADAL_MAX_FACTS`, `ADAL_MAX_FACT_CHARS`, `ADAL_MAX_LOADED`, `ADAL_MAX_TOKENS` | Per-tenant facts, a fact's length, tenants held in RAM, a reply's length | 5000, 300, 4, 200 |
| `ADAL_TAU`, `ADAL_DELTA`, `ADAL_BETA`, `ADAL_TAU_REPLACE` | The memory's thresholds (see `config.py`) | 0.45, 0.05, 1.0, 0.95 |

In production the deploy workflow decides the model, where it is fetched from and the infrastructure; all of it
reaches adaL through these settings. The GPU backend is the next thing to add (`src/adal/backends/`).

## The API

Every call but health carries `Authorization: Bearer <ADAL_TOKEN>` and `X-Glyphh-Tenant: <tenant>`.

| Call | Does |
|---|---|
| `GET /v1/health` | `{status, version, backend, model}` |
| `POST /v1/facts` `{text}` or `{texts: [...]}`, `source?` | Tells facts, all or none. A fact is one sentence that names its subject |
| `GET /v1/facts?q=&after=&limit=` | `{facts, next}`: oldest first; `q` searches the text; `next` is the `after` of the next page |
| `DELETE /v1/facts/{id}` | Forgets one. What it had corrected comes back |
| `DELETE /v1/facts` | Forgets everything: the tenant's memory is removed from disk |
| `POST /v1/ingest` `{text, source?}` | Free text: the model rewrites it into facts, code checks each against the text, the kept ones are told. `{facts, dropped}` |
| `POST /v1/chat` `{messages, max_tokens?}` | A reply with the tenant's memory in the loop. `{reply, memory: {recalled, fired}, tokens}` |

Errors are `{"error": {"code": "E_...", "message": ...}}`: `E_UNAUTHORIZED`, `E_INVALID`, `E_NOT_FOUND`, `E_LIMIT`,
`E_INTERNAL`.

## Tests

```sh
uv run pytest                 # unit tests: a scripted model, no weights; this is what CI runs
uv run pytest -m model -s     # the memory in a real model, in free generation (ADAL_MODEL, mlx)
```

The model tests are the loop this is developed in: invented customers, facts told once, asked in wordings never
told, corrected, forgotten, restarted, ingested from free text. Measured on Qwen3-14B (4-bit, M2 Max),
2026-11-10: recall 80/80, corrections 10/10 with the old value never said, another tenant sees nothing, memory
survives a restart, ingest 20/20.
