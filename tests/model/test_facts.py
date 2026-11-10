"""The memory in a real model, in free generation: `pytest -m model` (Apple silicon, the mlx extra, and ADAL_MODEL,
by default Qwen3-14B-4bit in the Hugging Face cache). Every customer here is invented, so a right answer can only
come from the memory. The bars are what the loop is held to; the measured numbers are printed (`-s`)."""

import os

import pytest

from adal.engine import Engine
from tests.conftest import make_settings
from tests.model.cases import ASKED, ATTRS, NOTE, TOLD, make_cases

pytestmark = pytest.mark.model
MODEL = os.environ.get("ADAL_MODEL", "Qwen3-14B-4bit")
CASES = make_cases(40)


def say(engine: Engine, tenant: str, question: str):
    return engine.chat(tenant, [{"role": "user", "content": question}])


@pytest.fixture(scope="module")
def backend():
    from adal.backends.mlx import MlxBackend

    return MlxBackend(MODEL)


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("adal"))


@pytest.fixture(scope="module")
def engine(backend, data) -> Engine:
    engine = Engine(backend, make_settings(data, max_facts=5000, max_tokens=80))
    engine.tell("acme", [TOLD[a].format(p=c["p"], v=c[a]) for c in CASES for a in ATTRS])
    return engine


def test_facts_told_once_are_answered_in_two_untold_wordings(engine):
    asked = [(c, a, form) for c in CASES[:10] for a in ATTRS for form in (0, 1)]
    right = [c[a] in say(engine, "acme", ASKED[a][form].format(p=c["p"])).text for c, a, form in asked]
    print(f"\nrecall {sum(right)}/{len(right)}")
    assert sum(right) / len(right) >= 0.95


def test_an_answer_carries_no_other_customers_values(engine):
    c = CASES[0]
    others = {x[a] for x in CASES[1:] for a in ATTRS}
    for a in ATTRS:
        reply = say(engine, "acme", ASKED[a][0].format(p=c["p"])).text
        assert not [v for v in others if v in reply], reply


def test_someone_never_mentioned_is_not_given_anothers_facts(engine):
    values = {c[a] for c in CASES for a in ATTRS}
    for q in ("Who is the account manager for Zorino Pekka?", "What discount code did Zorino Pekka get?"):
        reply = say(engine, "acme", q)
        assert reply.recalled == 0 and reply.fired == 0
        assert not [v for v in values if v in reply.text], reply.text


def test_an_unrelated_question_is_the_models_own(engine):
    reply = say(engine, "acme", "What is the capital of France?")
    assert "Paris" in reply.text and reply.recalled == 0 and reply.fired == 0


def test_corrections_replace_the_value_and_leave_the_rest(engine):
    fixed = CASES[10:20]
    engine.tell("acme", [f"{c['p']}'s account manager is now {c['new']}." for c in fixed])
    new = old = kept = 0
    for c in fixed:
        reply = say(engine, "acme", ASKED["manager"][0].format(p=c["p"])).text
        new += c["new"] in reply
        old += c["manager"] in reply
        kept += c["town"] in say(engine, "acme", ASKED["town"][1].format(p=c["p"])).text
    print(f"\ncorrected: new {new}/10, old {old}/10; the same customers' towns {kept}/10")
    assert new >= 9 and old == 0 and kept >= 9


def test_another_tenant_sees_nothing(engine):
    c = CASES[0]
    reply = say(engine, "globex", ASKED["manager"][0].format(p=c["p"]))
    assert c["manager"] not in reply.text and reply.recalled == 0


def test_memory_survives_a_restart(engine, backend, data):
    again = Engine(backend, make_settings(data, max_facts=5000, max_tokens=80))  # nothing in RAM
    for c in (CASES[0], CASES[10]):  # an untouched customer and a corrected one
        want = c["new"] if c is CASES[10] else c["manager"]
        assert want in say(again, "acme", ASKED["manager"][0].format(p=c["p"])).text


def test_a_forgotten_fact_is_no_longer_answered(engine):
    c = CASES[30]
    (fact,) = [f for f in engine.facts("acme", q=c["code"])]
    engine.forget("acme", fact.id)
    assert c["code"] not in say(engine, "acme", ASKED["code"][0].format(p=c["p"])).text
    assert c["town"] in say(engine, "acme", ASKED["town"][1].format(p=c["p"])).text


def test_free_text_is_ingested_as_facts_and_answered(engine):
    fresh = make_cases(5, seed=9)
    kept = 0
    for c in fresh:
        facts, _ = engine.ingest("notes", NOTE.format(**c))
        kept += sum(any(c[a] in f.text and c["p"] in f.text for f in facts) for a in ATTRS)
    right = sum(c[a] in say(engine, "notes", ASKED[a][0].format(p=c["p"])).text for c in fresh for a in ATTRS)
    print(f"\ningest: {kept}/20 facts kept with their subject; answered {right}/20")
    assert kept >= 18 and right >= 18
