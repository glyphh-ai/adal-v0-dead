"""The record: every tenant's facts, on disk, in the order they were told.

One SQLite file per tenant under the data directory (WAL, synchronous FULL: a fact that was answered 201 survives a
crash or a restart). A tenant's file is the whole of their memory: the in-RAM index (memory.py) is rebuilt from it,
a backup is a copy of it, and removing the tenant is removing it. A forgotten fact is deleted from the file, not
marked."""

import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass

from adal.errors import Invalid, LimitReached, NotFound

TENANT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    source TEXT,
    told_at REAL NOT NULL
);
"""


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    source: str | None
    told_at: float
    seq: int

    def public(self) -> dict:
        return {"id": self.id, "text": self.text, "source": self.source, "told_at": self.told_at}


def check_tenant(tenant: str) -> str:
    if not TENANT.match(tenant or ""):
        raise Invalid("A tenant is 1 to 64 of a-z, 0-9, - and _, starting with a letter or digit.", reason="tenant")
    return tenant


class Store:
    def __init__(self, directory: str, max_facts: int = 5000, max_chars: int = 300) -> None:
        self.directory, self.max_facts, self.max_chars = directory, max_facts, max_chars
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()  # one writer at a time across tenants: writes are small and rare

    def _path(self, tenant: str) -> str:
        return os.path.join(self.directory, f"{check_tenant(tenant)}.sqlite3")

    def _open(self, tenant: str) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path(tenant), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.executescript(SCHEMA)
        return conn

    def exists(self, tenant: str) -> bool:
        return os.path.exists(self._path(tenant))

    def tell(self, tenant: str, texts: list[str], source: str | None = None) -> list[Fact]:
        """Keep facts, all or none, in the order given."""
        clean = [" ".join(t.split()) for t in texts]
        for t in clean:
            if not t:
                raise Invalid("A fact needs some text.", reason="text")
            if len(t) > self.max_chars:
                raise Invalid(
                    f"A fact is one sentence, at most {self.max_chars} characters.", reason="text", most=self.max_chars
                )
        with self._lock:
            conn = self._open(tenant)
            try:
                with conn:
                    have = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
                    if have + len(clean) > self.max_facts:
                        raise LimitReached(
                            f"A memory holds at most {self.max_facts} facts.", limit="facts", most=self.max_facts
                        )
                    out = []
                    for t in clean:
                        fid, now = uuid.uuid4().hex, time.time()
                        cur = conn.execute(
                            "INSERT INTO facts (id, text, source, told_at) VALUES (?, ?, ?, ?)", (fid, t, source, now)
                        )
                        out.append(Fact(fid, t, source, now, int(cur.lastrowid)))
                    return out
            finally:
                conn.close()

    def facts(self, tenant: str, q: str = "", after: int = 0, limit: int | None = None) -> list[Fact]:
        """A tenant's facts, oldest first; `q` keeps those whose text contains it; `after` is the last seq seen."""
        if not self.exists(tenant):
            return []
        conn = self._open(tenant)
        try:
            sql, args = "SELECT id, text, source, told_at, seq FROM facts WHERE seq > ?", [after]
            if q:
                sql, args = sql + " AND instr(lower(text), lower(?)) > 0", [*args, q]
            sql += " ORDER BY seq"
            if limit is not None:
                sql, args = sql + " LIMIT ?", [*args, limit]
            return [Fact(*row) for row in conn.execute(sql, args)]
        finally:
            conn.close()

    def forget(self, tenant: str, fact_id: str) -> None:
        if not self.exists(tenant):
            raise NotFound("No such fact.", id=fact_id)
        with self._lock:
            conn = self._open(tenant)
            try:
                with conn:
                    if not conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount:
                        raise NotFound("No such fact.", id=fact_id)
            finally:
                conn.close()

    def forget_all(self, tenant: str) -> None:
        """Remove a tenant's memory entirely."""
        with self._lock:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(self._path(tenant) + suffix)
                except FileNotFoundError:
                    pass
