"""
utils/ingestion.py
Parse uploaded transcript files into plain text.
Supports: .txt, .docx, .pdf, .csv
"""

from __future__ import annotations
import io
import chardet


def parse_transcript(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "txt":
        return _parse_txt(file_bytes)
    elif ext == "docx":
        return _parse_docx(file_bytes)
    elif ext == "pdf":
        return _parse_pdf(file_bytes)
    elif ext == "csv":
        return _parse_csv(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def _parse_txt(data: bytes) -> str:
    enc = (chardet.detect(data).get("encoding") or "utf-8")
    try:
        return data.decode(enc)
    except Exception:
        return data.decode("utf-8", errors="replace")


def _parse_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def _parse_pdf(data: bytes) -> str:
    try:
        import pdfplumber
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t.strip())
        return "\n".join(parts)
    except ImportError:
        raise ImportError("pdfplumber is required for PDF parsing. Add it to requirements.txt.")


def _parse_csv(data: bytes) -> str:
    import pandas as pd
    enc = (chardet.detect(data).get("encoding") or "utf-8")
    df = pd.read_csv(io.BytesIO(data), encoding=enc, on_bad_lines="skip")
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols:
        return "\n".join(df[text_cols[0]].dropna().astype(str).tolist())
    return "\n".join(df.astype(str).apply(" | ".join, axis=1).tolist())
