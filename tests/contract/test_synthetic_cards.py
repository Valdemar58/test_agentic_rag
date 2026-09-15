"""§9: синтетические карточки валидны по pydantic-схеме CardData реального сервиса карточек."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from contracts.card_service import CardServiceContract
from synthetic.corpus import generate_corpus
from tessa_export.storage import CARDS_DIR, export_dir

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    structlog.reset_defaults()


def test_synthetic_cards_validate_against_card_data(contract: CardServiceContract, tmp_path: Path) -> None:
    generate_corpus(tmp_path / "corpus")
    cards = sorted((export_dir(tmp_path / "corpus") / CARDS_DIR).glob("*.json"))
    assert len(cards) == 13
    for path in cards:
        card = contract.card_data.model_validate(json.loads(path.read_text(encoding="utf-8")))
        sections = card.model_dump()["sections"]
        assert sections["DocumentCommonInfo"]["fields"]["Subject"]
        assert "OutgoingRefDocs" in sections and "IncomingRefDocs" in sections
