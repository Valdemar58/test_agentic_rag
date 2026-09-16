"""AC-2.3: MCP-сервер запускается автономно отдельным процессом (`python -m mcp_server`).

Без прогрева модели и Qdrant не нужны: список инструментов и заглушка глоссария отвечают сразу.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest
from fastmcp import Client

from common.config import DEFAULT_CONFIG_PATH, ROOT, load_app_config
from mcp_server.server import HEALTH_PATH

pytestmark = pytest.mark.integration

CONFIG = load_app_config(DEFAULT_CONFIG_PATH)
STARTUP_TIMEOUT_S = 90
EXPECTED_TOOLS = {
    "hybrid_search",
    "get_document_card",
    "get_related_documents",
    "get_document_content",
    "glossary_lookup",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "mcp_server", "--host", "127.0.0.1", "--port", str(port), "--no-warm-up"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"процесс MCP-сервера завершился с кодом {process.returncode}")
            try:
                if httpx.get(base + HEALTH_PATH, timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail(f"MCP-сервер не ответил на {HEALTH_PATH} за {STARTUP_TIMEOUT_S} с")
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


async def test_standalone_server_lists_tools_and_answers(server_url: str) -> None:
    async with Client(server_url + CONFIG.mcp.path) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == EXPECTED_TOOLS
        result = await client.call_tool("glossary_lookup", {"term": "ПВТР"})
        assert result.structured_content and result.structured_content["term"] == "ПВТР"
