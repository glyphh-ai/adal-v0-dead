import dataclasses

import pytest
from fastapi.testclient import TestClient

from adal.api import create_app
from adal.backends.fake import FakeBackend
from adal.config import Settings
from adal.engine import Engine

TOKEN = "test-token"


def make_settings(data: str, **over) -> Settings:
    base = Settings(
        backend="fake", model="fake", data=data, token=TOKEN, dev=False, tau=0.45, delta=0.05, beta=1.0,
        tau_replace=0.95, max_facts=50, max_fact_chars=300, max_loaded=4, max_tokens=120,
    )  # fmt: skip
    return dataclasses.replace(base, **over)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(str(tmp_path / "data"))


@pytest.fixture
def engine(settings) -> Engine:
    return Engine(FakeBackend(), settings)


@pytest.fixture
def client(settings):
    def call(tenant: str = "acme", token: str = TOKEN) -> TestClient:
        c = TestClient(app)
        c.headers.update({"Authorization": f"Bearer {token}", "X-Glyphh-Tenant": tenant})
        return c

    app = create_app(settings, FakeBackend())
    return call
