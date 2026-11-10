"""The loop with a scripted model (backends/fake.py): what the memory does to a reply, and what survives."""

import threading

import pytest

from adal.backends.fake import FakeBackend
from adal.engine import Engine
from adal.errors import Invalid, LimitReached, NotFound
from tests.conftest import make_settings

ASK = [{"role": "user", "content": "Who is the account manager for Piheba Fupu?"}]
FACT = "Piheba Fupu's account manager is Nipira."


def ask(engine, tenant="acme", messages=ASK):
    return engine.chat(tenant, messages).text


def test_without_a_memory_the_model_answers_alone(engine):
    reply = engine.chat("acme", ASK)
    assert reply.text == "I do not know." and reply.fired == 0


def test_a_fact_told_once_is_recalled_in_another_wording(engine):
    engine.tell("acme", [FACT])
    assert "Nipira" in ask(engine)
    assert "Nipira" in ask(engine, messages=[{"role": "user", "content": "Piheba Fupu's account manager?"}])


def test_an_unrelated_question_is_left_to_the_model(engine):
    engine.tell("acme", [FACT])
    reply = engine.chat("acme", [{"role": "user", "content": "How long should I boil an egg?"}])
    assert reply.text == "I do not know." and reply.fired == 0


def test_a_correction_replaces_the_fact_and_nothing_else(engine):
    engine.tell("acme", [FACT, "Piheba Fupu's home town is Huraku.", "Sabo Doti's account manager is Mufi."])
    engine.tell("acme", ["Piheba Fupu's account manager is now Bada."])
    reply = ask(engine)
    assert "Bada" in reply and "Nipira" not in reply
    assert "Huraku" in ask(engine, messages=[{"role": "user", "content": "What is Piheba Fupu's home town?"}])
    assert "Mufi" in ask(engine, messages=[{"role": "user", "content": "Who is the account manager for Sabo Doti?"}])


def test_tenants_never_see_each_other(engine):
    engine.tell("acme", [FACT])
    assert ask(engine, "globex") == "I do not know."
    assert engine.facts("globex") == []


def test_memory_survives_a_restart(settings):
    Engine(FakeBackend(), settings).tell("acme", [FACT, "Piheba Fupu's account manager is now Bada."])
    again = Engine(FakeBackend(), settings)  # a new process: nothing in RAM
    reply = ask(again)
    assert "Bada" in reply and "Nipira" not in reply
    assert [f.text for f in again.facts("acme")] == [FACT, "Piheba Fupu's account manager is now Bada."]


def test_forgetting_a_fact_removes_it_and_brings_back_what_it_corrected(engine):
    engine.tell("acme", [FACT])
    (new,) = engine.tell("acme", ["Piheba Fupu's account manager is now Bada."])
    engine.forget("acme", new.id)
    assert "Nipira" in ask(engine)
    with pytest.raises(NotFound):
        engine.forget("acme", new.id)


def test_forgetting_everything_removes_the_tenants_file(engine, settings):
    engine.tell("acme", [FACT])
    engine.forget_all("acme")
    assert ask(engine) == "I do not know." and engine.facts("acme") == []


def test_memories_reload_after_being_evicted(tmp_path):
    engine = Engine(FakeBackend(), make_settings(str(tmp_path), max_loaded=1))
    engine.tell("acme", [FACT])
    engine.tell("globex", ["Sabo Doti's account manager is Mufi."])  # evicts acme from RAM
    assert "Nipira" in ask(engine)


def test_limits_and_bad_input(tmp_path):
    engine = Engine(FakeBackend(), make_settings(str(tmp_path), max_facts=2, max_fact_chars=40))
    with pytest.raises(Invalid):
        engine.tell("acme", ["x" * 41])
    with pytest.raises(Invalid):
        engine.tell("acme", ["   "])
    with pytest.raises(Invalid):
        engine.tell("../etc", [FACT])
    with pytest.raises(Invalid):
        engine.chat("acme", [{"role": "assistant", "content": "hi"}])
    engine.tell("acme", ["Sabo Doti lives in Huraku.", "Sabo Doti owns a hawk."])
    with pytest.raises(LimitReached):
        engine.tell("acme", ["Sabo Doti keeps bees."])
    assert len(engine.facts("acme")) == 2  # all or none


def test_ingest_keeps_only_lines_the_note_supports(settings):
    note = "A message came in from Piheba Fupu. Home is the town of Huraku. The account manager on file is Nipira."

    def rewriter(messages):
        return (
            "Subject: Piheba Fupu\n"
            "Piheba Fupu's home town is Huraku.\n"
            "Piheba Fupu's account manager is Nipira.\n"
            "Piheba Fupu's favourite colour is Zorblat.\n"  # a value the note never had
            "The call was short.\n"  # does not start with the subject
        )

    engine = Engine(FakeBackend(rewriter), settings)
    kept, dropped = engine.ingest("acme", note, source="ticket-1")
    assert [f.text for f in kept] == ["Piheba Fupu's home town is Huraku.", "Piheba Fupu's account manager is Nipira."]
    assert dropped == 2 and kept[0].source == "ticket-1"


def test_ingest_tells_nothing_when_the_model_names_no_subject(settings):
    engine = Engine(FakeBackend(lambda m: "Piheba Fupu lives in Huraku."), settings)
    assert engine.ingest("acme", "Piheba Fupu lives in Huraku.") == ([], 1)
    with pytest.raises(Invalid):
        engine.ingest("acme", "  ")


def test_tellers_and_askers_at_once(engine):
    errors = []

    def work(i):
        try:
            engine.tell(f"t{i % 3}", [f"Person{chr(65 + i)} Doti's account manager is Mufi."])
            engine.chat(f"t{i % 3}", ASK)
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sum(len(engine.facts(f"t{i}")) for i in range(3)) == 12
