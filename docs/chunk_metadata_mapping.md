# Маппинг «поле карточки СЭД → метаданные чанка»

Черновик на согласование с заказчиком (FR-3, §13.4 ТЗ). Метаданные чанка формируются **только из полей карточки** (не выводятся из текста документа), кроме координат в структуре документа, которые даёт Docling. Источник полей — `CardData` из сервиса карточек (`sections[...].fields` / `.rows`), названия колонок — по реальному примеру карточки типа `OrderMKC` (см. `docs/tessa_analysis.md`).

Статус строки: ✅ подтверждено примером · ❓ уточнить по экспорту · ⚙ заполняется конвейером, не карточкой.

## 1. Обязательный минимум (FR-3)

| Поле чанка | Тип | Источник в карточке | Правило | Статус |
|---|---|---|---|---|
| `doc_id` | str (uuid) | `Card.id` | Идентификатор документа в индексе = ID карточки | ✅ |
| `tessa_card_id` | str (uuid) | `Card.id` | Явный идентификатор карточки в Тессе (дублирует `doc_id` по требованию ТЗ) | ✅ |
| `card_type_name` | str | `Card.type_name` | Системное имя типа карточки, например `OrderMKC` | ✅ |
| `card_type_caption` | str | `Card.type_caption` | Отображаемое имя типа, например «Приказ» | ✅ |
| `doc_kind` | str | `DocumentCommonInfo.DocTypeTitle` → `TypeDocumentNameTypeDocument` → `Card.type_caption` | Вид документа для фильтра и хлебных крошек; первое непустое | ❓ проверить на других типах |
| `doc_number` | str | `DocumentCommonInfo.FullNumber` → `SecondaryFullNumber` | Номер; у незарегистрированных — номер проекта | ✅ |
| `doc_date` | date (ISO) | `DocumentCommonInfo.DocDate` → `CreationDate` | Дата документа; для фильтра по диапазону хранится и как `doc_date_ts` (int, unix) | ✅ |
| `doc_status` | enum `active` / `cancelled` / `draft` / `unknown` | `DocumentCommonInfo.StatusID` через таблицу в конфиге | Известно: `de9d3b6d-532b-4cb8-aa7b-e055e8986e48` → `cancelled`. Остальные значения — по экспорту. Фильтр `hybrid_search` по умолчанию: `doc_status == active` | ❓ значения для `active`, `draft` |
| `doc_status_name` | str | `DocumentCommonInfo.StatusNameStatus` | Отображаемое имя статуса как есть («Отмененный») | ✅ |
| `approval_state` | str | `KrApprovalCommonInfoVirtual.StateName` → `DocumentCommonInfo.StateName` | Статус согласования (ключ локализации `$KrStates_Doc_*`); человекочитаемое имя — по словарю в конфиге | ✅ |
| `department` | str | `DocumentCommonInfo.DepartmentName` | Подразделение; фильтр `hybrid_search` | ✅ у приказа, ❓ у других типов |
| `department_id` | str (uuid) | `DocumentCommonInfo.DepartmentID` | Для точного фильтра | ✅ |
| `author` | str | `DocumentCommonInfo.AuthorName` → `RegistratorName` → `Card.created_by_name` | Автор/регистратор | ✅ |
| `relations` | list[{`doc_id`, `relation`, `direction`}] | `OutgoingRefDocs.rows[].DocID/RefTypeName` (`direction=outgoing`), `IncomingRefDocs.rows[].DocID` (`direction=incoming`, `relation` = `RefTypeReverseName` из карточки-источника по `links_graph.json`, иначе `null`) | Связи только из карточек (§7) | ✅ структура, ❓ полный перечень типов |
| `acl_groups` | list[str] | нет в карточке | Заглушка `[]` («не заполнено»); поле обязательно в схеме с первого дня | ⚙ |
| `section_path` | list[str] | Docling | Путь в структуре: заголовки разделов сверху вниз | ⚙ |
| `clause` | str \| null | Docling | Номер пункта/статьи, если распознан (например «3.2») | ⚙ |
| `breadcrumbs` | str | `doc_kind` + `doc_number` + `doc_date` + `section_path` + `clause` | Строка вида «Приказ №144 от 02.07.2026 → Раздел 3 → п. 3.2»; она же дублируется в начало текста чанка | ⚙ |
| `file_sha256` | str | вычисляется при экспорте/инжесте | Хэш исходного файла; ключ инкрементальности | ⚙ |
| `file_name` | str | `CardData.files[].name` | Имя исходного файла | ✅ |
| `file_row_id` | str (uuid) | `CardData.files[].row_id` | ID файла в карточке (для `get-file-content`) | ✅ |

## 2. Дополнительные поля (полезны для ответов и фильтров)

| Поле чанка | Источник | Статус |
|---|---|---|
| `subject` | `DocumentCommonInfo.Subject` | ✅ |
| `comment` | `DocumentCommonInfo.Comment` (например «Отменен приказом от 27.08.2026 № 173.») | ✅ |
| `signed_by` | `DocumentCommonInfo.SignedByName` | ✅ |
| `direction_activity` | `DirectionActivityDCI.rows[].DirectionActivityName` | ✅ |
| `approvers` | `Approv.rows[].UserName` | ✅ |
| `responsible` | `ResponsibleErrand.rows[].UserName` | ✅ |
| `file_category` | `CardData.files[].category_caption` («Документ») | ✅ |
| `card_version`, `card_modified` | `Card.version`, `Card.modified` | ✅ |
| `chunk_kind` | `structural` / `fallback` / `table` / `glossary` | ⚙ |
| `parent_id`, `chunk_index` | parent-child и порядок в документе | ⚙ |

## 3. Правила и допущения

1. **Основной файл документа** (❓ О5): среди невиртуальных файлов категории «Документ» берётся `pdf`, если есть, иначе `docx`; остальные файлы той же категории считаются приложениями и тоже индексируются с тем же `doc_id` и своим `file_name`. `.sig` и виртуальные файлы (`is_virtual`, `KrVirtualFileType`) не индексируются.
2. **Пустые поля** не заполняются выдумкой: отсутствующее поле → `null`, `doc_status` → `unknown` с предупреждением в логе инжеста.
3. **Три статуса независимы**: `doc_status` (действует/отменён/проект, из `StatusID`), `approval_state` (согласование, Kr), `StateName` маршрута. Проверка актуальности в самопроверке агента — только по `doc_status`.
4. **Фильтруемые поля Qdrant** (индексы payload): `doc_id`, `doc_kind`, `doc_status`, `doc_date_ts`, `department`, `chunk_kind`, `file_sha256`.
5. Маппинг оформляется кодом в `src/ingest/metadata.py` как явная таблица «поле карточки → поле чанка» с этим документом в качестве спецификации; изменения — только через правку обоих.

## 4. Вопросы заказчику по маппингу

1. Подтвердить выбор `DocTypeTitle` как «вида документа» для категорий 8.2 (или использовать `TypeCaption` типа карточки).
2. Прислать после экспорта значения `StatusID`/`StatusNameStatus` для «действует» и «проект» (скрипт выведет все встреченные пары).
3. Подтвердить правило основного файла (п. 3.1) или указать иное.
4. Нужны ли в метаданных `approvers`/`responsible` (ФИО сотрудников) — или их исключить из индекса.
