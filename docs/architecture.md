# adaL: what is built

Only what exists in this repo and is tested. What is not built is listed at the end.

## The shape

```
Glyphh server (sign-in, plans, metering)
        |  Authorization: Bearer <service token>, X-Glyphh-Tenant
        v
api.py ----> engine.py ----> backends/   the language model: tokens, chat layout, next-token logits
                 |
                 +---------> memory.py   one tenant's index, in RAM (derived)
                 +---------> store.py    every tenant's facts, on disk (the record)
                               keys.py   the key: a glyphh of the last content words (the Ada kernel)
```

One process holds one model. Generations run one at a time (`Engine._model`); everything else is concurrent.

## A fact

A fact is one sentence that names its subject: "Piheba Fupu's account manager is Nipira." Telling it
(`POST /v1/facts`) writes it to the tenant's record first, then into their index.

**The record** (`store.py`) is one SQLite file per tenant under `ADAL_DATA`, in WAL mode with `synchronous=FULL`:
a fact answered `201` survives a crash or a restart. The file is the whole of a tenant's memory: a backup is a
copy of it, removing the tenant is removing it, and a forgotten fact is deleted from it, not marked.

**The index** (`memory.py`) is rebuilt from the record, in the order the facts were told, when a tenant is first
used after a start, after being evicted from RAM (`ADAL_MAX_LOADED`), and after a fact is forgotten. It is never
the record, so a change to how the memory keys needs no migration.

A fact leaves a **value row** where each value starts (a word the lexicon does not know, such as a name or a
code, or the fact's last content word) and **mid rows** inside that word, so a value of several tokens is carried
to its end. A value row's key is the context before the value with stop words out of it, so "…manager is",
"…manager is now" and a question about the manager meet at one row.

## A correction

A fact told again in the same context replaces the old value: the row's other continuations are halved, so one
telling replaces one telling. A new value within `ADAL_TAU_REPLACE` of another value row that continues
differently corrects that one too. The old fact stays in the record (it is what was said); the index answers with
the new one. Forgetting the correction brings the old value back.

## A reply

`POST /v1/chat`:

1. **Recall.** The question's key is compared with every value row. The facts at `ADAL_TAU` or closer (at most 8)
   are put in view, in a system note, so the reply is written knowing them. A corrected value is not recalled.
2. **Steer.** When one value is clearly the nearest (no different value within `ADAL_DELTA` of it), it is mixed
   into the model's own probabilities for the reply's first token, and its mid rows finish the word. When two
   different values are about as near, the memory does not steer; the facts in view answer.
3. The model writes the rest by itself.

With nothing recalled, the model is only told to say so when it does not know.

Why both: steering alone was tried first and fails in free generation. The model says the right value and then
writes around it as if it knew nothing ("Puroni is a company that provides…"), because one token does not change
what it knows. Facts in view alone is what failed in the research at scale and under correction (every fact in the
prompt: corrections 6%). Recall picks the few current facts; steering holds the value.

## Ingest

`POST /v1/ingest` takes free text. The model rewrites it: it names the subject once, then writes one sentence per
fact, each starting with the subject. Code keeps a line only if the subject is in the text, the line starts with
it, and every name or value in the line (every word the lexicon does not know) is in the text. The model may
reword; it cannot add. A rewrite cut at the token limit loses its last line rather than keep half a fact.

## Tenants

A tenant is the `X-Glyphh-Tenant` the server sends (`^[a-z0-9][a-z0-9_-]{0,63}$`). Its facts are its own file and
its index its own object: no query crosses tenants. The service token is compared in constant time; without it
every call but `/v1/health` is refused.

## Measured

`uv run pytest -m model -s`, Qwen3-14B 4-bit on an M2 Max, 2026-11-10, 40 invented customers with four facts each:

| | |
|---|---|
| Facts told once, asked in two wordings never told, free generation | 80 / 80 |
| Corrections ("…is now X"): new value said / old value said | 10 / 10, 0 / 10 |
| The corrected customers' other facts | 10 / 10 |
| A customer never mentioned: another's value said | 0 |
| Another tenant asking | Nothing recalled |
| After a restart (nothing in RAM) | Same answers |
| Free-text notes ("they…" after the first sentence): facts kept, then answered | 20 / 20, 20 / 20 |

The research behind the key, the mixing and corrections, with larger runs (3,000 facts, 2,000 notes) scored as
ranked answers: `timmetim/thinktank`, `version_22_qwen_adapter/FINDINGS.md`.

## Not built

- **A GPU backend.** Only MLX (Apple silicon) and the scripted test model exist. Production needs one more class
  in `backends/`; nothing above it changes.
- **Rules.** Instructions that carry across sessions are research (sign-off rule 100%, topic rule 65%).
- **Streaming.** A reply is returned whole.
- **More than one generation at a time.** No batching.
- **Scale.** An index row is 2,048 floats (8 KB), about 3 rows a fact; the nearest row is found by comparing
  with all of them. Fine for thousands of facts per tenant (the default cap is 5,000); packing the keys and an
  index for the search come before more.
- **Thresholds that follow the store.** `ADAL_TAU` is the value measured at 3,000 facts; the research showed it
  should tighten as a store grows.
- **Conversation memory.** Recall reads the user's last message only.
- **Values of several words.** A value is one word (of any number of tokens) plus whatever the facts in view give.
