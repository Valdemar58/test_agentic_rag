"""Шаги и элементы диалога без внешних ключей на сообщение (FR-7, FR-9)

Chainlit сохраняет шаги, сообщения и элементы фоновыми задачами: потомок (шаг внутри run, элемент
сообщения) может попасть в БД раньше родителя, и внешний ключ отклонял запись — ответы ассистента
терялись. Ссылки `message.parent_id` и `element.message_id` остаются колонками без ограничения
(как в собственной схеме Chainlit); целостность по диалогу держат ключи на `conversation`.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-16

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("fk_message_parent_id_message", "message", type_="foreignkey")
    op.drop_constraint("fk_element_message_id_message", "element", type_="foreignkey")


def downgrade() -> None:
    op.create_foreign_key(
        "fk_element_message_id_message", "element", "message", ["message_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_message_parent_id_message", "message", "message", ["parent_id"], ["id"], ondelete="SET NULL"
    )
