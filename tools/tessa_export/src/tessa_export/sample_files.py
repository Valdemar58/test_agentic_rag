"""Генераторы минимальных файлов для тестов и режима самопроверки: PDF с текстовым слоем
и без него, DOCX (с таблицей и разделом терминов по желанию), изображение-«скан»."""

from __future__ import annotations

import io

from docx import Document
from PIL import Image


def minimal_pdf_bytes(text: str | None) -> bytes:
    """Одностраничный PDF. При text=None страница пустая — имитация скана без текстового слоя."""
    if text is None:
        content = b""
    else:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    buffer = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode()
    buffer += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(buffer)


def minimal_docx_bytes(
    paragraphs: list[str], *, with_table: bool = False, terms_heading: str | None = None
) -> bytes:
    """DOCX с абзацами; опционально таблица 2×2 и заголовок раздела терминов."""
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    if terms_heading:
        document.add_heading(terms_heading, level=1)
        document.add_paragraph("СИЗ — средства индивидуальной защиты.")
    if with_table:
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Показатель"
        table.cell(0, 1).text = "Значение"
        table.cell(1, 0).text = "Норма"
        table.cell(1, 1).text = "1"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def minimal_image_bytes(image_format: str = "JPEG") -> bytes:
    """Маленькое изображение — имитация скана."""
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()
