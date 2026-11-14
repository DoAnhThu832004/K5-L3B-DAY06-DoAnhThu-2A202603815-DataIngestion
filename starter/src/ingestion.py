import json
import logging
from pathlib import Path
import re
from typing import Any
import unicodedata

from .models import DocumentPage

logger = logging.getLogger(__name__)

EXTRACTION_AUDIT_LOG: list[dict[str, Any]] = []


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1\2", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned = line.strip()
        if re.match(r"^(\d+\s+)?LOVELY PROFESSIONAL UNIVERSITY(\s+\d+)?$", cleaned, re.IGNORECASE):
            continue
        if re.match(r"^Unit \d+:.*$", cleaned, re.IGNORECASE):
            continue
        if re.match(r"^(Retail Store Management|\d{1,3})$", cleaned, re.IGNORECASE):
            continue
        if re.match(r"^Notes$", cleaned, re.IGNORECASE):
            continue
        cleaned = re.sub(r"^\s*Notes\s+", "", cleaned)
        cleaned = re.sub(r"\s+Notes$", "", cleaned)
        cleaned_lines.append(cleaned)

    paragraphs: list[str] = []
    current_para: list[str] = []

    for i, line in enumerate(cleaned_lines):
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue

        is_heading = (
            len(line) < 55
            and not line.endswith((".", ",", ";", ":", "-", "–"))
            and (line[0].isupper() or re.match(r"^\d+\.", line))
        )
        ends_sentence = line.endswith((".", "!", "?", ':"', '."', "!'", "?'"))

        next_is_heading = False
        if i + 1 < len(cleaned_lines):
            next_line = cleaned_lines[i + 1]
            next_is_heading = (
                len(next_line) < 55
                and not next_line.endswith((".", ",", ";", ":", "-", "–"))
                and (next_line[:1].isupper() or re.match(r"^\d+\.", next_line))
            )

        current_para.append(line)

        if is_heading or (ends_sentence and next_is_heading):
            paragraphs.append(" ".join(current_para))
            current_para = []

    if current_para:
        paragraphs.append(" ".join(current_para))

    normalized = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    valid_pages: list[int] | None = None,
    manifest_path: Path | None = None,
) -> list[DocumentPage]:
    global EXTRACTION_AUDIT_LOG
    EXTRACTION_AUDIT_LOG.clear()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF source not found at: {path}")

    if valid_pages is None:
        candidate_paths = [
            manifest_path,
            Path("data/manifest.json"),
            path.parent.parent / "manifest.json",
            path.resolve().parents[2] / "data" / "manifest.json",
        ]
        found_manifest = None
        for candidate in candidate_paths:
            if candidate and Path(candidate).exists():
                found_manifest = Path(candidate)
                break

        if found_manifest:
            with open(found_manifest, encoding="utf-8") as f:
                manifest_data = json.load(f)
                valid_pages = manifest_data["documents"][0]["valid_content_pages"]
        else:
            valid_pages = [
                p for p in range(6, 62)
                if p not in [25, 26, 39, 40, 41, 60, 61]
            ]

    def get_metadata(page_num: int) -> dict[str, Any]:
        if 6 <= page_num <= 26:
            unit = "unit_01"
            if page_num <= 10:
                section = "operations"
            elif page_num <= 18:
                section = "store_organization"
            else:
                section = "location"
            book_page = page_num - 5
        elif 27 <= page_num <= 41:
            unit = "unit_02"
            section = "layout_types" if page_num <= 31 else "store_design"
            book_page = page_num - 5
        elif 42 <= page_num <= 61:
            unit = "unit_03"
            if page_num <= 44:
                section = "operations"
            elif page_num <= 48:
                section = "merchandise"
            else:
                section = "distribution_centre"
            book_page = page_num - 5
        else:
            unit = "other"
            section = "general"
            book_page = page_num

        return {
            "unit": unit,
            "section": section,
            "book_page": book_page,
            "physical_pdf_page": page_num,
        }

    raw_page_texts: dict[int, str] = {}

    try:
        import pypdfium2
        pdf = pypdfium2.PdfDocument(str(path))
        for p in valid_pages:
            idx = p - 1
            if 0 <= idx < len(pdf):
                page_obj = pdf[idx]
                textpage = page_obj.get_textpage()
                raw_page_texts[p] = textpage.get_text_range()
    except ImportError:
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for p in valid_pages:
                    idx = p - 1
                    if 0 <= idx < len(pdf.pages):
                        raw_page_texts[p] = pdf.pages[idx].extract_text() or ""
        except ImportError:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for p in valid_pages:
                idx = p - 1
                if 0 <= idx < len(reader.pages):
                    raw_page_texts[p] = reader.pages[idx].extract_text() or ""

    document_pages: list[DocumentPage] = []

    for page_num in valid_pages:
        raw_text = raw_page_texts.get(page_num, "")
        normalized = normalize_text(raw_text)

        meta = get_metadata(page_num)
        meta["raw_character_count"] = len(raw_text)
        meta["normalized_character_count"] = len(normalized)

        if len(normalized.strip()) < 20:
            audit_entry = {
                "document_id": document_id,
                "source": source,
                "page": page_num,
                "raw_character_count": len(raw_text),
                "status": "EMPTY_OR_UNREADABLE",
                "error": "Extracted text has less than 20 characters",
            }
            EXTRACTION_AUDIT_LOG.append(audit_entry)
            logger.warning("Empty or unreadable page: %s", audit_entry)
            continue

        doc_page = DocumentPage(
            document_id=document_id,
            source=source,
            page=page_num,
            text=normalized,
            metadata=meta,
        )
        document_pages.append(doc_page)

    logger.info("Successfully loaded %d valid pages from %s", len(document_pages), path)
    return document_pages
