"""Unstructured document ingestion: PDF, Word (.docx), and Excel (.xlsx).

Each loader extracts plain text (PDF/DOCX) or tabular data (Excel) so it can
feed either the RAG pipeline (text) or the feature engineering pipeline
(tables).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.base import DataConnector, DataIngestionError


class PDFConnector(DataConnector):
    """Extracts text content from PDF files, page by page."""

    source_type = "pdf"

    def load(self, path: str | Path) -> list[dict[str, str | int]]:
        from pypdf import PdfReader

        path = Path(path)
        if not path.exists():
            raise DataIngestionError(f"PDF not found: {path}")
        try:
            reader = PdfReader(str(path))
            pages = [
                {"page": i + 1, "text": page.extract_text() or ""}
                for i, page in enumerate(reader.pages)
            ]
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"Failed to parse PDF {path}: {exc}") from exc
        self._log_loaded(len(pages), path=str(path))
        return pages


class DocxConnector(DataConnector):
    """Extracts paragraph text from Word (.docx) documents."""

    source_type = "docx"

    def load(self, path: str | Path) -> list[str]:
        from docx import Document

        path = Path(path)
        if not path.exists():
            raise DataIngestionError(f"DOCX not found: {path}")
        try:
            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"Failed to parse DOCX {path}: {exc}") from exc
        self._log_loaded(len(paragraphs), path=str(path))
        return paragraphs


class ExcelConnector(DataConnector):
    """Loads tabular data from Excel workbooks."""

    source_type = "excel"

    def load(self, path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise DataIngestionError(f"Excel file not found: {path}")
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"Failed to parse Excel {path}: {exc}") from exc
        self._log_loaded(len(df), path=str(path))
        return df