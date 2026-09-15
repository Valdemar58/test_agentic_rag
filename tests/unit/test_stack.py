"""Оркестрация стенда: доля памяти GPU из бюджета и взаимоисключение профилей (без Docker)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_app_config
from common.stack import (
    CommandResult,
    Stack,
    StackError,
    compose_environment,
    conflicting_service,
    detect_gpu_total_mib,
    gpu_memory_utilization,
    parse_running_services,
)

RTX_3080_MIB = 12288
RTX_5080_MIB = 16303


class FakeRunner:
    """Записывает вызовы docker/nvidia-smi и отдаёт заранее заданные ответы."""

    def __init__(self, *, running: str = "", gpu_total: str = f"{RTX_5080_MIB}\n") -> None:
        self.calls: list[list[str]] = []
        self.envs: list[Mapping[str, str] | None] = []
        self.running = running
        self.gpu_total = gpu_total

    def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str] | None = None, capture: bool = False
    ) -> CommandResult:
        self.calls.append(list(args))
        self.envs.append(env)
        if args[0] == "nvidia-smi":
            return CommandResult(0, self.gpu_total)
        if "ps" in args:
            return CommandResult(0, self.running)
        return CommandResult(0, "")


class FailingRunner:
    def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str] | None = None, capture: bool = False
    ) -> CommandResult:
        return CommandResult(1, "")


def test_gpu_memory_utilization_gives_same_gib_on_both_cards() -> None:
    # 10,5 ГиБ на карте 12 ГиБ и на карте 16 ГиБ: доли разные, объём одинаковый
    on_3080 = gpu_memory_utilization(10.5, RTX_3080_MIB, 0.95)
    on_5080 = gpu_memory_utilization(10.5, RTX_5080_MIB, 0.95)
    assert on_3080 == 0.875
    assert on_5080 == 0.66
    assert abs(on_3080 * RTX_3080_MIB - on_5080 * RTX_5080_MIB) < 0.01 * RTX_3080_MIB


def test_gpu_memory_utilization_is_capped_and_validated() -> None:
    assert gpu_memory_utilization(10.5, 8192, 0.95) == 0.95
    with pytest.raises(StackError):
        gpu_memory_utilization(10.5, 0, 0.95)


def test_detect_gpu_total_mib_parses_nvidia_smi() -> None:
    assert detect_gpu_total_mib(FakeRunner(gpu_total="16303\n")) == 16303
    with pytest.raises(StackError, match="nvidia-smi"):
        detect_gpu_total_mib(FakeRunner(gpu_total="garbage"))


def test_compose_environment_from_example_config() -> None:
    config = load_app_config(DEFAULT_CONFIG_PATH)
    env = compose_environment(config, RTX_3080_MIB)
    assert env["VLLM_GPU_MEMORY_UTILIZATION"] == "0.875"
    assert env["VLLM_QWEN_MODEL_PATH"] == "/models/Qwen3-8B-AWQ"
    assert env["VLLM_DOTS_MODEL_PATH"] == "/models/DotsMOCR"
    assert env["VLLM_QWEN_TOOL_CALL_PARSER"] == "hermes"
    assert Path(env["MODELS_DIR"]).is_absolute()
    # bind-mount и все аргументы vLLM берутся только отсюда — без дублирования в compose
    compose_text = (DEFAULT_CONFIG_PATH.parent.parent / "docker-compose.yml").read_text(encoding="utf-8")
    for key in env:
        if key != "GPU_TOTAL_MIB":
            assert f"${{{key}" in compose_text, key


def test_parse_running_services_accepts_array_and_ndjson() -> None:
    array = '[{"Service": "qdrant", "State": "running"}, {"Service": "vllm-qwen", "State": "running"}]'
    ndjson = '{"Service": "qdrant", "State": "running"}\n{"Service": "vllm-dots", "State": "running"}\n'
    assert parse_running_services(array) == {"qdrant", "vllm-qwen"}
    assert parse_running_services(ndjson) == {"qdrant", "vllm-dots"}
    assert parse_running_services("") == set()


def test_conflicting_service_only_between_gpu_profiles() -> None:
    assert conflicting_service("ingest", {"qdrant", "vllm-qwen"}) == "vllm-qwen"
    assert conflicting_service("runtime", {"vllm-dots"}) == "vllm-dots"
    assert conflicting_service("runtime", {"vllm-qwen", "qdrant"}) is None
    assert conflicting_service("base", {"vllm-qwen"}) is None


def test_up_ingest_refuses_while_runtime_is_running() -> None:
    config = load_app_config(DEFAULT_CONFIG_PATH)
    runner = FakeRunner(running='[{"Service": "vllm-qwen", "State": "running"}]')
    stack = Stack(config, runner=runner, gpu_total_mib=RTX_3080_MIB)
    with pytest.raises(StackError, match="взаимоисключены"):
        stack.up("ingest")
    assert not any("up" in call for call in runner.calls)


def test_up_with_switch_stops_rival_then_starts_profile() -> None:
    config = load_app_config(DEFAULT_CONFIG_PATH)
    runner = FakeRunner(running='[{"Service": "vllm-qwen", "State": "running"}]')
    stack = Stack(config, runner=runner, gpu_total_mib=RTX_5080_MIB)
    stack.up("ingest", switch=True, wait=True)
    commands = [call for call in runner.calls if call[0] == "docker"]
    assert commands[-2][-4:] == ["rm", "--stop", "--force", "vllm-qwen"]
    assert "--profile" in commands[-2] and "runtime" in commands[-2]
    assert commands[-1][-4:] == ["up", "-d", "--remove-orphans", "--wait"]
    assert commands[-1][commands[-1].index("--profile") + 1] == "ingest"
    env = runner.envs[-1]
    assert env is not None and env["VLLM_GPU_MEMORY_UTILIZATION"] == "0.660"


def test_up_runtime_with_observability_adds_profile_and_base_has_none() -> None:
    config = load_app_config(DEFAULT_CONFIG_PATH)
    runner = FakeRunner()
    stack = Stack(config, runner=runner, gpu_total_mib=RTX_5080_MIB)
    stack.up("runtime", observability=True)
    up_call = runner.calls[-1]
    profiles = [up_call[i + 1] for i, item in enumerate(up_call) if item == "--profile"]
    assert profiles == ["runtime", "observability"]
    stack.up("base")
    assert "--profile" not in runner.calls[-1]
    with pytest.raises(StackError, match="неизвестная цель"):
        stack.up("gpu")


def test_down_covers_all_profiles_and_volumes_flag() -> None:
    config = load_app_config(DEFAULT_CONFIG_PATH)
    runner = FakeRunner()
    stack = Stack(config, runner=runner, gpu_total_mib=RTX_5080_MIB)
    stack.down(volumes=True)
    call = runner.calls[-1]
    assert call.count("--profile") == 3 and call[-1] == "--volumes"
    with pytest.raises(StackError, match="docker compose"):
        Stack(config, runner=FailingRunner(), gpu_total_mib=RTX_5080_MIB).down()
