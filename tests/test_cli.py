"""Tests for the CLI — uses monkeypatched ``classify`` so no network needed."""

from __future__ import annotations

import json
from typing import Any

import pytest

from hscode import cli as cli_module
from hscode.models import ClassificationResult


@pytest.fixture
def fake_result() -> ClassificationResult:
    return ClassificationResult(
        hs_code="85183000",
        description="Headphones and earphones",
        confidence=0.92,
        reasoning="step by step",
        supplementary_unit="PST",
        supplementary_unit_description="Pieces / items (number)",
        chapter="85",
        heading="8518",
        subheading="851830",
        status="ok",
        validated=True,
        attempts=1,
        cn_year=2026,
    )


def test_cli_text_output(monkeypatch: pytest.MonkeyPatch, fake_result, capsys) -> None:
    def fake_classify(*args: Any, **kwargs: Any) -> ClassificationResult:
        return fake_result

    monkeypatch.setattr(cli_module, "classify", fake_classify)

    exit_code = cli_module.main(["Wireless headphones", "--quiet"])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "85183000" in out
    assert "Headphones" in out
    assert "PST" in out


def test_cli_json_output(monkeypatch: pytest.MonkeyPatch, fake_result, capsys) -> None:
    monkeypatch.setattr(cli_module, "classify", lambda *a, **kw: fake_result)

    exit_code = cli_module.main(["Wireless headphones", "--json", "--quiet"])
    assert exit_code == 0

    data = json.loads(capsys.readouterr().out)
    assert data["hs_code"] == "85183000"
    assert data["validated"] is True


def test_cli_error_handling(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def boom(*a: Any, **kw: Any) -> ClassificationResult:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(cli_module, "classify", boom)

    exit_code = cli_module.main(["Some product", "--quiet"])
    assert exit_code == 2


def test_cli_requires_description() -> None:
    with pytest.raises(SystemExit):
        cli_module.main([])
