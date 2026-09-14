# Анализ СЭД Тесса и сервиса карточек заказчика

Результат этапа 1 (§11 ТЗ). Источники: SDK Тессы по `TESSA_SDK_PATH` (`tessa_client` 0.1.6, `swagger.json`, фикстуры), сервис карточек по `CARD_SERVICE_PATH` (`robot_skills`), локальный пример реального ответа `cards/get` (приказ, тип `OrderMKC`; файл не коммитится). Ссылки вида `файл:строка` указывают на внешний код по этим путям.

## 1. SDK Тессы (`tessa_client`)

| Аспект | Факт | Где |
|---|---|---|
| Транспорт | Только sync, один `httpx.Client`; async-варианта нет | `src/tessa_client/client.py:39-93` |
| Авторизация | Логин/пароль (`DOMAIN\user`) → `POST /service/login` → токен сессии; заголовки `tessa-session`, `tessa-version: 4.2`; повторный логин при 401 | `src/tessa_client/auth.py:47-186` |
| Карточка | `client.cards.get(card_id, mode=CardGetMode.READ_ONLY)` → `CardGetResponse.card: Card` | `src/tessa_client/resources/cards.py:373-392` |
| Файл | `client.cards.get_file_content(card_id, file_id=CardFile.row_id, version_row_id=CardFile.version_row_id)` → `DownloadedFile(content: bytes, file_name, content_type)`; всё в памяти, без стриминга; `version_row_id` обязателен на практике | `resources/cards.py:578-616`, `models/card_responses.py:97-107` |
| Версии файла | `client.cards.get_file_versions(card_id, file_id)` | `resources/cards.py:618-637` |
| Списки/справочники | Только представления: `client.views.get_data(alias, parameters, page_offset, page_limit)` → позиционные строки `Columns`/`Rows` | `src/tessa_client/resources/views.py:36-86` |
| Типы карточек | `client.card_types.get(type_id)`; `CardTypeSections` на сервере пустой, имена колонок недоступны | `resources/card_types.py:11-20`, фикстура `tests/fixtures/card_type_get_response.json` |
| Typed JSON | Ключи с суффиксами типов (`ID::uid`, `Number::int`, `DocDate::dtm`), служебные поля с точкой (`.table::int`, `.state::int`); `normalize_tessa_json` снимает суффиксы, Guid остаются строками | `src/tessa_client/typed_json.py:1-44, 113` |
| Lossless | `Card.model_dump()` не восстанавливает исходный JSON (Guid без суффиксов, потеря .NET-тиков, `Info`/`KrToken` в приватном атрибуте). Для архива хранить сырой `response.json()` | `resources/cards.py:301-332` |
| Исключения | `TessaAuthenticationError` (401), `TessaPermissionError` (403), `TessaNotFoundError` (404), `TessaValidationError` (400), `TessaServerError` (5xx), `TessaOperationError` (200 + ошибка в `ValidationResult`), `TessaTimeoutError`, `TessaConnectionError`, `TessaResponseParsingError` | `src/tessa_client/exceptions.py:24-170` |
| Побочный эффект | `print(payload)` на каждый запрос | `src/tessa_client/resources/base.py:58` |

Модель `Card` (`models/card.py:93-125`): `id`, `type_id`, `type_name`, `type_caption`, `created`, `created_by_name`, `modified`, `modified_by_name`, `version`, `sections: dict[str, CardSection]`, `files: list[CardFile]`, `permissions`. `CardSection` (`:72-90`): `fields: dict` для секций-карточек, `rows: list[dict]` для табличных. `CardFile` (`models/card_file.py:69-127`): `row_id` (= FileID), `name`, `size`, `version_row_id`, `version_number`, `category_id`, `category_caption`, `is_virtual`, `type_name`, `hash_` (в примере пуст), `versions`.

## 2. Структура карточки документа (по реальному примеру `OrderMKC`)

Ответ `POST /api/v1/cards/get`: `{ Info, ValidationResult, Card, SectionRows }`. `SectionRows` — строки-шаблоны всех табличных секций с полным набором колонок (удобно для обнаружения схемы).

Скаляры `Card`: `ID`, `TypeID`, `TypeName = "OrderMKC"`, `TypeCaption = "Приказ"`, `Created/CreatedByName`, `Modified/ModifiedByName`, `Version`.

### 2.1 Секция `DocumentCommonInfo` (поля документа)

| Поле | Пример | Назначение для RAG |
|---|---|---|
| `FullNumber` | `"144"` | Номер документа |
| `SecondaryFullNumber` / `SecondaryNumber` | `"Проект приказа_194"` | Номер проекта до регистрации |
| `DocDate` | `2026-07-02T11:40:35Z` | Дата документа |
| `CreationDate` | | Дата создания карточки |
| `Subject` | «О назначении ответственных лиц …» | Название/тема |
| `DocTypeID` / `DocTypeTitle` | `"Приказ"` | Вид документа |
| `TypeDocumentID` / `TypeDocumentNameTypeDocument` | `"Приказ"` | Вид документа (дубль из справочника `TypeDocument`) |
| `DepartmentID` / `DepartmentName` | «Отдел охраны труда …» | Подразделение |
| `AuthorID` / `AuthorName` | | Автор |
| `RegistratorID` / `RegistratorName` | | Регистратор |
| `SignedByID` / `SignedByName` | | Подписант |
| `StatusID` / `StatusNameStatus` | `de9d3b6d-…` / «Отмененный» | **Статус документа (действует/отменён/проект)** — справочник заказчика |
| `StateID` / `StateName` | `6` / `$KrStates_Doc_Registered` | Состояние маршрута (Kr); не равно статусу документа |
| `Comment` | «Отменен приказом от 27.08.2026 № 173.» | Текстовое пояснение |
| `NumberSheets` / `NumberSheetsApplication` | `6` / `23` | Листов в документе и приложениях |
| `SignatureEDSVariantYesNo`, `UrgentSignUrgent` | «Да» / «Нет» | Признаки ЭЦП и срочности |
| `RefDocsLinkID/Name/ReverseName` | `null` | Служебные поля UI добавления ссылок |

Состав полей зависит от типа карточки: у `IncomingEDO` и `PrimaryDocumentMKC` наборы другие (см. фикстуры SDK). Читать поля защищённо, через маппинг с fallback.

### 2.2 Связи между документами

Эндпоинта связей в API нет; связи — табличные секции карточки:

| Секция | Колонки | Смысл |
|---|---|---|
| `OutgoingRefDocs` | `RowID, DocID, DocDescription, DocTypeName, Order, RefTypeID, RefTypeName, RefTypeReverseName` | Исходящие связи с типом. Пример: `DocID` → приказ №109, `RefTypeName = "в отмену"`, `RefTypeReverseName = "отменено"` |
| `IncomingRefDocs` | `RowID, DocID, DocDescription` | Входящие связи, **без типа**; заполняется сервером в `cards/get`. Пример: `DocID` → приказ №173 (отменяющий) |
| `OutgoingRefDocsInput` | `RowID, DocID, DocDescription, DocType, DocTypeName` | Буфер UI, для RAG не нужен |

Справочник типов связей — представление `RefType` (в коде и фикстурах перечня нет; собирается по экспорту). Тип входящей связи восстанавливается из `OutgoingRefDocs` карточки-источника.

### 2.3 Прочие секции с метаданными

`Approv` (согласующие: `UserID, UserName`), `ResponsibleErrand` (ответственные), `Performers`, `ControlTask`, `DirectionActivityDCI` (направление деятельности: «Безопасность»), `KrApprovalCommonInfoVirtual` (`StateName`, `ApprovedBy`, `DisapprovedBy`, `AuthorName` — статус согласования), `KrApprovalHistoryVirtual`, `KrStagesVirtual` (история и стадии маршрута).

### 2.4 Файлы

Пример приказа: 5 файлов — `docx` шаблон (категория «Документ», 8 версий), `docx` «Для печати … с ЭЦП» («Документ»), итоговый `pdf` «144 от 02.07.2026 …» («Документ»), `.sig` (без категории), `Лист согласования.html` (`IsVirtual: true`, `TypeName: KrVirtualFileType`, `Size: -1`, нет записи в `Permissions.FilePermissions`). `Hash` у всех пуст — хэш считаем при скачивании.

### 2.5 Права и «гриф»

Понятия грифа/уровня доступа документа в Тессе нет (проверено: SDK, swagger 1,1 МБ, фикстуры — 0 совпадений по «гриф», «ДСП», «конфиденц», `SecurityLevel`). Есть `Card.Permissions` (битовые флаги `CardPermissionFlags`, `enums.py:75-98`) на карточку, секции, строки и файлы, и `UserAccessLevel` (Regular/Administrator). Решение заказчика: документов с ограничениями нет, фильтр не нужен; `acl_groups` в индексе — заглушка.

## 3. Сервис карточек (`robot_skills`)

| Аспект | Факт | Где |
|---|---|---|
| Контракт | `CardData` (+ `CardSectionData`, `CardFileData`, `CardPermissionData`), snake_case без alias, generic: `sections: dict[str, CardSectionData]`, внутри `fields: dict` / `rows: list[dict]` | `src/robot_skills/core/cards/schemas.py:36-95` |
| Преобразование | `CardData.model_validate(card.model_dump())`; проекция **lossy**: теряются `Card.info`, `tasks`, `task_history`, у файлов `versions`, `hash_`, `flags`, `store_source` и др. | `core/cards/service.py:33-38` |
| Маршруты | `GET /health`; `POST /core/cards/get` (`CardGetIn` → `CardData`); `POST /core/cards/get-file-content` (bytes + `Content-Disposition filename*=UTF-8''…`); `POST /core/cards/get-file-versions`; `POST /core/views/{alias}/get-data`; `GET /core/card-types`, `GET /core/card-types/{id}`; `POST /core/user-info` | `core/cards/router.py:32-101`, `core/views/router.py`, `core/card_types/router.py` |
| Авторизация | HTTP Basic (учётка Тессы) на все `/core/**`; сессии по пользователю в пуле; сервисной учётки нет | `deps.py:37-50`, `tessa_session.py:76-196` |
| Ошибки | Единый `ErrorResponse {error, message, request_id, tessa_status_code, validation_items}`; 404 `resource_not_found`, если Тесса вернула `Card = null` | `errors.py:33-38, 176-216` |
| Python / инструменты | ≥3.13; `tessa-client` из приватного devpi; ruff `E,F,I,UP,B` line 110; mypy strict + pydantic plugin; async SQLAlchemy 2.0 + naming convention; Alembic `0001_slug.py`; docstrings на русском | `pyproject.toml`, `CLAUDE.md`, `db/base.py`, `alembic/versions/` |
| Поиск | Через представления с `CurrentUserId`/`Locale=ru` параметрами; примеры lookups: `Users`, `Departments`, `RefDocumentsLookup`, `TypeDocument` | `workflows/lookups.py:27-196` |

Импорт контракта в RAG: `src/contracts/` добавляет `<TESSA_SDK_PATH>/src` и `<CARD_SERVICE_PATH>/src` в `sys.path`; `robot_skills.core.cards.schemas` зависит только от pydantic и enum'ов `tessa_client`.

## 4. Следствия для проекта

1. **Экспорт**: BFS по `OutgoingRefDocs` + `IncomingRefDocs` из ответа `cards/get`; ребро с типом пишется один раз (при обходе источника). Файлы: все невиртуальные, последняя версия; хэш считаем сами. Хранить и `CardData`-JSON, и сырой ответ.
2. **Мок**: маршруты `POST /core/cards/get`, `POST /core/cards/get-file-content`, `POST /core/views/{alias}/get-data`, `GET /core/card-types`, `GET /health`; Basic auth принимается любой.
3. **MCP `get_related_documents`**: `cards/get` → секции связей; тип входящей связи через карточку-источник.
4. **Статус в индексе**: три независимых признака — `StatusID`/`StatusNameStatus` (действует/отменён/проект), `StateName` (маршрут), `KrApprovalCommonInfoVirtual.StateName` (согласование). Фильтр «только действующие» — по `StatusID`.
5. **Открыто до экспорта**: значения `StatusID` кроме «отменён»; полный перечень `RefTypeName`; соответствие `TypeName`/`DocTypeTitle` видам документов; правило выбора основного файла.
