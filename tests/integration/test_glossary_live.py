"""AC-5.1: запрос с аббревиатурой из глоссария находит те же документы, что запрос с расшифровкой.

Пары «аббревиатура ↔ расшифровка» берутся из глоссария живого стенда (в репозитории реальных терминов
корпуса нет): отбираются записи действующих документов, у которых термин — короткое сокращение, а
определение развёрнуто. Для каждой пары сравниваются документы, найденные `hybrid_search` по полной
расшифровке и по запросу с аббревиатурой, расшифрованной так же, как это делает агент до поиска
(`agent.glossary`). Для протокола считается и базовый вариант — сокращение без расшифровки.

Нужны Qdrant с построенным индексом и глоссарием (`scripts/run_ingest.py glossary`) и MCP-сервер.
Запуск: `uv run pytest -m scenario`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastmcp import Client
from qdrant_client import QdrantClient

from agent.glossary import Expansion, expand_query
from common.config import AppConfig, load_app_config
from common.settings import Settings
from ingest.glossary import GlossaryRecord
from ingest.glossary_index import GlossaryIndex
from mcp_server.server import HEALTH_PATH

pytestmark = [pytest.mark.integration, pytest.mark.scenario]

MIN_PAIRS = 5
MAX_PAIRS = 8
TOP_K = 5
# сокращение: из прописных букв и цифр, без пробелов («ПВТР», «ГСМ», «1С:УАТ» сюда не попадает)
ABBREVIATION_RE = re.compile(r"^[А-ЯЁA-Z][А-ЯЁA-Z0-9]{1,7}$")
DEFINITION_MIN_WORDS = 3


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_app_config()


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def pairs(config: AppConfig, settings: Settings) -> Iterator[list[GlossaryRecord]]:
    client = QdrantClient(url=settings.resolve_qdrant_url(), timeout=int(config.qdrant.timeout_s))
    try:
        index = GlossaryIndex(client, config.qdrant, config.embedding.dense_dim)
        if not index.exists():
            pytest.skip("глоссарий не собран: uv run python scripts/run_ingest.py glossary")
        chosen: dict[str, GlossaryRecord] = {}
        for record in index.records():
            if record.doc_status != "active" or not ABBREVIATION_RE.match(record.term):
                continue
            if len(record.definition.split()) < DEFINITION_MIN_WORDS:
                continue
            chosen.setdefault(record.term_key, record)
        if len(chosen) < MIN_PAIRS:
            pytest.skip(f"в глоссарии {len(chosen)} пар с аббревиатурами, нужно не меньше {MIN_PAIRS}")
        yield sorted(chosen.values(), key=lambda item: item.term)[:MAX_PAIRS]
    finally:
        client.close()


@pytest.fixture(scope="module")
def mcp_url(config: AppConfig, settings: Settings) -> str:
    url = settings.resolve_mcp_url(config)
    try:
        httpx.get(url.rsplit("/", 1)[0] + HEALTH_PATH, timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"MCP-сервер недоступен ({exc})")
    return url


async def _documents(client: Client[Any], query: str) -> set[str]:
    result = await client.call_tool("hybrid_search", {"query": query, "top_k": TOP_K})
    structured = result.structured_content or {}
    return {hit["doc_id"] for hit in structured.get("hits", [])}


async def test_ac51_abbreviation_finds_the_same_documents_as_full_form(
    pairs: list[GlossaryRecord], mcp_url: str, config: AppConfig
) -> None:
    lines: list[str] = []
    matched = 0
    plain_matched = 0
    async with Client(mcp_url) as client:
        for record in pairs:
            expansion = Expansion(term=record.term, definition=record.definition, doc_label=record.doc_label)
            expanded = expand_query(record.term, [expansion], chars=config.agent.glossary.definition_chars)
            full_docs = await _documents(client, record.definition)
            abbreviation_docs = await _documents(client, expanded)
            plain_docs = await _documents(client, record.term)
            common = full_docs & abbreviation_docs
            matched += bool(common)
            plain_matched += bool(full_docs & plain_docs)
            lines.append(
                f"{record.term}: общих документов {len(common)} из {len(full_docs)}; "
                f"без расшифровки {len(full_docs & plain_docs)}"
            )
    report = "\n".join(lines)
    assert matched >= MIN_PAIRS, f"пар с общими документами {matched} из {len(pairs)}:\n{report}"
    # базовый вариант печатается для протокола: он показывает, что даёт расшифровка
    print(f"\nAC-5.1: пар {len(pairs)}, с расшифровкой {matched}, без неё {plain_matched}\n{report}")
