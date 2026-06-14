from __future__ import annotations

import re
from pathlib import Path

from .models import Document


DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b")


def extract_metadata(document: Document) -> dict[str, str]:
    path = Path(document.source_path)
    text = document.text
    dates = sorted(set(DATE_RE.findall(text)))
    entities = _top_entities(text)
    doc_type = _guess_document_type(path.stem, text)
    return {
        "document_type": doc_type,
        "dates": ", ".join(dates[:5]),
        "entities": ", ".join(entities[:8]),
        "source_name": path.name,
    }


def metadata_match_score(query: str, metadata: dict[str, str]) -> float:
    haystack = " ".join(metadata.values()).lower()
    query_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", query) if len(term) > 2}
    if not query_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in haystack)
    return hits / len(query_terms)


def _guess_document_type(stem: str, text: str) -> str:
    combined = f"{stem} {text[:300]}".lower()
    if "policy" in combined:
        return "policy"
    if "brief" in combined or "assessment" in combined:
        return "assessment brief"
    if "ocr" in combined:
        return "technical notes"
    if "rag" in combined or "retrieval" in combined:
        return "technical notes"
    return "document"


def _top_entities(text: str) -> list[str]:
    ignored = {"The", "This", "For", "Without", "Staff", "Documents", "Query", "Notes"}
    counts: dict[str, int] = {}
    for match in ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        if value in ignored or len(value) < 4 or value.endswith(" Notes"):
            continue
        counts[value] = counts.get(value, 0) + 1
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
