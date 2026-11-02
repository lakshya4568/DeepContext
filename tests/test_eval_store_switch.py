"""``EVAL_STORE=stock`` runs the benchmark on the application's own store instead of the harness subclass."""

from __future__ import annotations

import pytest
from eval import pg

from deep_context.storage.postgres_store import PostgresStore


def test_the_default_store_is_the_harness_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_STORE", raising=False)
    cls = pg.make_store_class()
    assert cls is not PostgresStore and issubclass(cls, PostgresStore)


def test_the_stock_switch_returns_the_applications_own_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_STORE", "stock")
    assert pg.make_store_class() is PostgresStore
