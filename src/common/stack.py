"""Оркестрация стенда поверх docker compose.

Две задачи, которые compose сам не решает (§2 ТЗ, NFR-3):

* профили `runtime` (vllm-qwen) и `ingest` (vllm-dots) взаимоисключены — бюджет 12 ГиБ VRAM
  не вмещает обе модели, поэтому попытка поднять один профиль при работающем другом даёт
  ошибку (или останавливает соперника при явном `switch`);
* доля `--gpu-memory-utilization` считается из бюджета в гибибайтах (`configs/app.yaml`) и
  фактического объёма памяти карты, чтобы vLLM получал одинаковый объём и на RTX 3080 12 ГиБ,
  и на карте разработки 16 ГиБ.

Все обращения к docker и nvidia-smi идут через `CommandRunner`, чтобы логику можно было
проверить без Docker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from common.config import ROOT, AppConfig

RUNTIME = "runtime"
INGEST = "ingest"
OBSERVABILITY = "observability"
BASE = "base"
GPU_SERVICE: dict[str, str] = {RUNTIME: "vllm-qwen", INGEST: "vllm-dots"}
ALL_PROFILES: tuple[str, ...] = (RUNTIME, INGEST, OBSERVABILITY)
UP_TARGETS: tuple[str, ...] = (BASE, RUNTIME, INGEST)
COMPOSE_FILE = ROOT / "docker-compose.yml"
MODELS_MOUNT = "/models"
MIB_PER_GIB = 1024


class StackError(Exception):
    """Ошибка оркестрации: конфликт профилей, нет GPU, сбой docker compose."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str


class CommandRunner(Protocol):
    def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str] | None = None, capture: bool = False
    ) -> CommandResult: ...


def run_command(
    args: Sequence[str], *, env: Mapping[str, str] | None = None, capture: bool = False
) -> CommandResult:
    """Запуск процесса; при capture=False вывод идёт в консоль как есть."""
    try:
        completed = subprocess.run(
            list(args),
            env=dict(env) if env is not None else None,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise StackError(f"команда не найдена: {args[0]} ({exc})") from exc
    return CommandResult(completed.returncode, completed.stdout or "")


def gpu_memory_utilization(vllm_memory_gib: float, total_mib: int, max_utilization: float) -> float:
    """Доля памяти карты, дающая vLLM ровно `vllm_memory_gib`, но не выше предохранителя."""
    if total_mib <= 0:
        raise StackError(f"объём памяти GPU должен быть положительным, получено {total_mib} МиБ")
    fraction = vllm_memory_gib * MIB_PER_GIB / total_mib
    return round(min(fraction, max_utilization), 3)


def parse_gpu_total_mib(output: str) -> int:
    """Разбор вывода `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise StackError("nvidia-smi не вернул объём памяти GPU")
    try:
        return int(lines[0])
    except ValueError as exc:
        raise StackError(f"не удалось разобрать объём памяти GPU из вывода nvidia-smi: {lines[0]!r}") from exc


def detect_gpu_total_mib(runner: CommandRunner) -> int:
    result = runner(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture=True)
    if result.returncode != 0:
        raise StackError(
            "nvidia-smi завершился с ошибкой: GPU не обнаружен или драйвер не установлен. "
            "Для просмотра переменных без GPU укажите --gpu-total-mib."
        )
    return parse_gpu_total_mib(result.stdout)


def docker_runs_in_wsl(platform: str = sys.platform) -> bool:
    """Docker Desktop на Windows выполняет контейнеры в WSL2, где нет UVA (отображаемой хост-памяти)."""
    return platform == "win32"


def compose_environment(config: AppConfig, total_mib: int, *, wsl: bool | None = None) -> dict[str, str]:
    """Переменные для docker-compose.yml, вычисленные из конфига и объёма памяти карты."""
    utilization = gpu_memory_utilization(config.gpu.vllm_memory_gib, total_mib, config.gpu.max_utilization)
    qwen, dots = config.vllm.qwen, config.vllm.dots
    in_wsl = docker_runs_in_wsl() if wsl is None else wsl
    # Новый GPU-раннер vLLM (V2) требует UVA; под WSL2 её нет → прежний раннер (V1). На Linux
    # переменная не задаётся вовсе: vLLM выбирает раннер сам (пустое значение он не принимает).
    wsl_only = {"VLLM_USE_V2_MODEL_RUNNER": "0"} if in_wsl else {}
    return {
        **wsl_only,
        "MODELS_DIR": str(config.models.dir_absolute),
        "GPU_TOTAL_MIB": str(total_mib),
        "VLLM_GPU_MEMORY_UTILIZATION": f"{utilization:.3f}",
        "VLLM_QWEN_MODEL_PATH": f"{MODELS_MOUNT}/{config.models.qwen.local_name}",
        "VLLM_QWEN_SERVED_NAME": qwen.served_model_name,
        "VLLM_QWEN_PORT": str(qwen.port),
        "VLLM_QWEN_MAX_MODEL_LEN": str(qwen.max_model_len),
        "VLLM_QWEN_MAX_NUM_SEQS": str(qwen.max_num_seqs),
        "VLLM_QWEN_TOOL_CALL_PARSER": qwen.tool_call_parser,
        "VLLM_QWEN_REASONING_PARSER": qwen.reasoning_parser,
        "VLLM_DOTS_MODEL_PATH": f"{MODELS_MOUNT}/{config.models.dots.local_name}",
        "VLLM_DOTS_SERVED_NAME": dots.served_model_name,
        "VLLM_DOTS_PORT": str(dots.port),
        "VLLM_DOTS_MAX_MODEL_LEN": str(dots.max_model_len),
        "VLLM_DOTS_MAX_NUM_SEQS": str(dots.max_num_seqs),
    }


def parse_running_services(ps_output: str) -> set[str]:
    """Имена сервисов из `docker compose ps --format json` (массив или по объекту на строку)."""
    text = ps_output.strip()
    if not text:
        return set()
    try:
        data = json.loads(text)
        items = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {str(item["Service"]) for item in items if item.get("State", "running") == "running"}


def conflicting_service(target: str, running: set[str]) -> str | None:
    """GPU-сервис другого профиля, который сейчас работает и мешает поднять `target`."""
    if target not in GPU_SERVICE:
        return None
    for profile, service in GPU_SERVICE.items():
        if profile != target and service in running:
            return service
    return None


class Stack:
    """Обёртка над docker compose с окружением из конфига и проверкой взаимоисключения."""

    def __init__(
        self,
        config: AppConfig,
        *,
        runner: CommandRunner = run_command,
        compose_file: Path = COMPOSE_FILE,
        gpu_total_mib: int | None = None,
    ) -> None:
        self._config = config
        self._runner = runner
        self._compose_file = compose_file
        self._gpu_total_mib = gpu_total_mib
        self._environment: dict[str, str] | None = None

    @property
    def gpu_total_mib(self) -> int:
        if self._gpu_total_mib is None:
            self._gpu_total_mib = detect_gpu_total_mib(self._runner)
        return self._gpu_total_mib

    def environment(self) -> dict[str, str]:
        if self._environment is None:
            self._environment = compose_environment(self._config, self.gpu_total_mib)
        return self._environment

    def compose(
        self, args: Sequence[str], *, profiles: Sequence[str] = (), capture: bool = False
    ) -> CommandResult:
        command = ["docker", "compose", "-f", str(self._compose_file)]
        for profile in profiles:
            command += ["--profile", profile]
        command += list(args)
        env = {**os.environ, **self.environment()}
        result = self._runner(command, env=env, capture=capture)
        if result.returncode != 0:
            raise StackError(f"docker compose завершился с кодом {result.returncode}: {' '.join(args)}")
        return result

    def running_services(self) -> set[str]:
        result = self.compose(
            ["ps", "--status", "running", "--format", "json"], profiles=ALL_PROFILES, capture=True
        )
        return parse_running_services(result.stdout)

    def up(
        self, target: str, *, observability: bool = False, switch: bool = False, wait: bool = False
    ) -> None:
        """Поднять базу и профиль `target`; соперничающий GPU-профиль — ошибка или остановка при switch."""
        if target not in UP_TARGETS:
            raise StackError(f"неизвестная цель {target!r}; допустимо: {', '.join(UP_TARGETS)}")
        conflict = conflicting_service(target, self.running_services())
        if conflict is not None:
            if not switch:
                raise StackError(
                    f"сервис {conflict} уже занимает GPU: профили runtime и ingest взаимоисключены "
                    f"(бюджет 12 ГиБ VRAM). Остановите его или запустите с --switch."
                )
            self.stop_service(conflict)
        profiles: list[str] = [] if target == BASE else [target]
        if observability:
            profiles.append(OBSERVABILITY)
        args = ["up", "-d", "--remove-orphans"]
        if wait:
            args.append("--wait")
        self.compose(args, profiles=profiles)

    def stop_service(self, service: str) -> None:
        profile = next(profile for profile, name in GPU_SERVICE.items() if name == service)
        self.compose(["rm", "--stop", "--force", service], profiles=[profile])

    def down(self, *, volumes: bool = False) -> None:
        args = ["down", "--remove-orphans"]
        if volumes:
            args.append("--volumes")
        self.compose(args, profiles=ALL_PROFILES)

    def status(self) -> None:
        self.compose(["ps", "--all"], profiles=ALL_PROFILES)

    def passthrough(self, args: Sequence[str]) -> None:
        """Произвольная команда compose с окружением из конфига (все профили включены)."""
        self.compose(args, profiles=ALL_PROFILES)
