"""Загрузка архива экспорта (FR-3, §8 ТЗ): manifest.json, links_graph.json, cards/, files/.

Правила входа инжеста:
- источник правды о составе — `manifest.json` экспорт-скрипта (модели `tessa_export.manifest`);
- файл идёт в инжест только вместе со своей карточкой: `cards/<id>.json` читается и соответствует
  форме CardData, иначе ошибка в лог и пропуск (FR-3: документ без карточки не допускается);
- sha256 считается по файлу на диске и сверяется с манифестом: расхождение — ошибка, пропуск;
- файлы на диске, которых нет в манифесте (в том числе в каталогах карточек вне сета), в инжест
  не идут — ошибка в лог;
- файлы, которые экспорт не скачал (виртуальные, неразрешённые форматы, ошибки скачивания),
  входом инжеста не являются и в знаменатель AC-3.1/M8 не входят (N2).
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ValidationError

from ingest.cards import CardReadError, CardRecord, load_card
from tessa_export.manifest import DocumentEntry, FileEntry, LinksGraph, Manifest
from tessa_export.storage import EXPORT_DIR_NAME, FILES_DIR, LINKS_GRAPH_NAME, MANIFEST_NAME

logger = logging.getLogger(__name__)

HASH_CHUNK_BYTES = 1 << 20

IssueKind = Literal[
    "card_missing",
    "card_invalid",
    "file_missing",
    "hash_mismatch",
    "file_without_card",
    "file_not_in_manifest",
]
ISSUE_TITLES: dict[IssueKind, str] = {
    "card_missing": "нет карточки",
    "card_invalid": "карточка не читается",
    "file_missing": "файла нет на диске",
    "hash_mismatch": "хэш файла не совпадает с манифестом",
    "file_without_card": "файл без карточки в сете",
    "file_not_in_manifest": "файла нет в манифесте",
}


class CorpusError(Exception):
    """Архив экспорта не найден или не читается целиком (манифест, граф связей)."""


@dataclass(frozen=True)
class CorpusIssue:
    """Файл, который не допущен в инжест, с причиной (ошибка в лог, пропуск — FR-3, AC-3.1)."""

    kind: IssueKind
    path: str = ""
    card_id: UUID | None = None
    file_row_id: UUID | None = None
    detail: str = ""

    @property
    def title(self) -> str:
        return ISSUE_TITLES[self.kind]


@dataclass(frozen=True)
class CorpusFile:
    """Скачанный экспортом файл карточки, сверенный с диском: вход инжеста."""

    card_id: UUID
    row_id: UUID
    name: str
    extension: str
    category: str | None
    relative_path: str
    path: Path
    sha256: str
    size: int
    has_text_layer: bool | None
    page_count: int | None
    duplicate_of: str | None
    smoke_note: str | None


@dataclass(frozen=True)
class CorpusDocument:
    card_id: UUID
    entry: DocumentEntry
    card: CardRecord
    files: list[CorpusFile]


@dataclass
class Corpus:
    export_dir: Path
    manifest: Manifest
    links_graph: LinksGraph
    documents: list[CorpusDocument]
    issues: list[CorpusIssue]
    not_downloaded: int

    @property
    def synthetic(self) -> bool:
        return self.manifest.synthetic

    @property
    def files(self) -> list[CorpusFile]:
        return [file for document in self.documents for file in document.files]

    def summary_lines(self) -> list[str]:
        files = self.files
        by_extension = Counter(file.extension for file in files)
        extensions = ", ".join(f"{ext} {count}" for ext, count in sorted(by_extension.items()))
        total_mib = sum(file.size for file in files) / 2**20
        lines = [
            f"Архив: {self.export_dir}",
            f"Документов с карточками: {len(self.documents)} из {len(self.manifest.documents)} в манифесте",
            f"Файлов на вход инжеста: {len(files)} ({total_mib:.1f} МиБ; {extensions})",
            f"Не скачано экспортом (вне знаменателя): {self.not_downloaded}",
            f"Не допущено в инжест: {len(self.issues)}",
        ]
        if self.synthetic:
            lines.insert(0, "ДАННЫЕ СИНТЕТИЧЕСКИЕ: метрики M1–M8 по этому корпусу не считаются")
        return lines


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_export_dir(root: Path) -> Path:
    """Принимает и каталог с manifest.json, и его родителя с подкаталогом export/."""
    for candidate in (root, root / EXPORT_DIR_NAME):
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    raise CorpusError(
        f"в {root} нет {MANIFEST_NAME}: ожидается распакованный архив экспорта "
        f"(каталог с {MANIFEST_NAME} или его родитель с подкаталогом {EXPORT_DIR_NAME}/)"
    )


def _read_model[T: BaseModel](path: Path, model: type[T]) -> T:
    try:
        return model.model_validate_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise CorpusError(f"в архиве нет файла {path.name}: {path.parent}") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "корень"
        raise CorpusError(
            f"{path.name} не соответствует формату экспорта ({exc.error_count()} ошибок; "
            f"первая: {location}: {first['msg']})"
        ) from exc
    except ValueError as exc:
        raise CorpusError(f"{path.name} не читается как JSON: {exc}") from exc


def _downloaded(entry: DocumentEntry) -> list[FileEntry]:
    return [file for file in entry.files if file.downloaded and file.path]


def _normalize(path: str) -> str:
    return Path(path).as_posix()


def _load_document(
    export_dir: Path, entry: DocumentEntry, issues: list[CorpusIssue]
) -> CorpusDocument | None:
    downloaded = _downloaded(entry)
    card_path = export_dir / entry.card_path
    try:
        card = load_card(card_path)
        if card.id != entry.card_id:
            raise CardReadError(f"id карточки {card.id} не совпадает с манифестом ({entry.card_id})")
    except CardReadError as exc:
        kind: IssueKind = "card_missing" if not card_path.exists() else "card_invalid"
        issues.extend(
            CorpusIssue(kind, _normalize(file.path or ""), entry.card_id, file.row_id, str(exc))
            for file in downloaded
        )
        if not downloaded:
            issues.append(CorpusIssue(kind, _normalize(entry.card_path), entry.card_id, None, str(exc)))
        return None

    files: list[CorpusFile] = []
    for file in downloaded:
        relative = _normalize(file.path or "")
        path = export_dir / relative
        if not path.is_file():
            issues.append(CorpusIssue("file_missing", relative, entry.card_id, file.row_id))
            continue
        digest = sha256_of(path)
        if file.sha256 and digest != file.sha256:
            detail = f"на диске {digest[:12]}…, в манифесте {file.sha256[:12]}…"
            issues.append(CorpusIssue("hash_mismatch", relative, entry.card_id, file.row_id, detail))
            continue
        files.append(
            CorpusFile(
                card_id=entry.card_id,
                row_id=file.row_id,
                name=file.name,
                extension=file.extension.lower(),
                category=file.category,
                relative_path=relative,
                path=path,
                sha256=digest,
                size=path.stat().st_size,
                has_text_layer=file.has_text_layer,
                page_count=file.page_count,
                duplicate_of=file.duplicate_of,
                smoke_note=file.smoke_note,
            )
        )
    return CorpusDocument(card_id=entry.card_id, entry=entry, card=card, files=files)


def _orphan_files(
    export_dir: Path, manifest: Manifest, known_paths: set[str], issues: list[CorpusIssue]
) -> None:
    files_root = export_dir / FILES_DIR
    if not files_root.is_dir():
        return
    known_cards = {str(entry.card_id) for entry in manifest.documents}
    for path in sorted(item for item in files_root.rglob("*") if item.is_file()):
        relative = path.relative_to(export_dir).as_posix()
        if relative in known_paths:
            continue
        card_dir = path.relative_to(files_root).parts[0]
        kind: IssueKind = "file_not_in_manifest" if card_dir in known_cards else "file_without_card"
        card_id = UUID(card_dir) if card_dir in known_cards else None
        issues.append(CorpusIssue(kind, relative, card_id, None, f"каталог {card_dir}"))


def load_corpus(root: Path) -> Corpus:
    """Читает архив экспорта; проблемы отдельных файлов не прерывают загрузку, а попадают в issues."""
    export_dir = resolve_export_dir(root)
    manifest = _read_model(export_dir / MANIFEST_NAME, Manifest)
    links_graph = _read_model(export_dir / LINKS_GRAPH_NAME, LinksGraph)

    issues: list[CorpusIssue] = []
    documents: list[CorpusDocument] = []
    known_paths: set[str] = set()
    not_downloaded = 0
    for entry in manifest.documents:
        downloaded = _downloaded(entry)
        not_downloaded += len(entry.files) - len(downloaded)
        known_paths.update(_normalize(file.path or "") for file in downloaded)
        document = _load_document(export_dir, entry, issues)
        if document is not None:
            documents.append(document)
    _orphan_files(export_dir, manifest, known_paths, issues)

    for issue in issues:
        logger.error("%s: %s — пропущен (%s)", issue.title, issue.path, issue.detail or "без подробностей")
    logger.info(
        "Архив %s: документов %d, файлов на вход %d, не допущено %d, не скачано экспортом %d",
        export_dir,
        len(documents),
        sum(len(document.files) for document in documents),
        len(issues),
        not_downloaded,
    )
    return Corpus(
        export_dir=export_dir,
        manifest=manifest,
        links_graph=links_graph,
        documents=documents,
        issues=issues,
        not_downloaded=not_downloaded,
    )
