from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from docintel.ocr_evaluation import normalize_ocr_text
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "evaluation" / "tat_dqa" / "development_manifest.json"
DEFAULT_IMAGE_DIR = ROOT / "data" / "evaluation" / "ocr" / "development"
DEFAULT_OUTPUT = ROOT / "evaluation" / "ocr" / "development_manifest.json"
SELECTION_SEED = "docintel-ocr-development-v1"
CATEGORIES = ("prose", "number_heavy")
SAMPLES_PER_CATEGORY = 6
RENDER_DPI = 300


@dataclass(frozen=True)
class Candidate:
    document_uid: str
    pdf_file: str
    page: int
    line: int
    text: str
    bbox: tuple[float, float, float, float]
    category: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_key(candidate: Candidate) -> str:
    identity = f"{candidate.document_uid}\0{candidate.page}\0{candidate.line}\0{candidate.text}"
    return sha256_bytes(f"{SELECTION_SEED}\0{identity}".encode())


def classify_line(text: str) -> str | None:
    letters = sum(character.isalpha() for character in text)
    digits = sum(character.isdigit() for character in text)
    alphanumeric = letters + digits
    if len(text) < 45 or len(text) > 220 or len(text.split()) < 6:
        return None
    if digits >= 4 and digits / alphanumeric >= 0.08:
        return "number_heavy"
    if digits == 0 and letters >= 35:
        return "prose"
    return None


def collect_candidates(document: dict) -> list[Candidate]:
    pdf_path = ROOT / document["pdf_file"]
    candidates = []
    with pdfplumber.open(pdf_path, unicode_norm="NFKC") as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for line_number, line in enumerate(page.extract_text_lines(return_chars=False), start=1):
                text = normalize_ocr_text(str(line["text"]))
                category = classify_line(text)
                bbox = tuple(float(line[key]) for key in ("x0", "top", "x1", "bottom"))
                if category is None or not _bbox_is_visible(bbox, float(page.width), float(page.height)):
                    continue
                candidates.append(
                    Candidate(
                        document_uid=document["uid"],
                        pdf_file=document["pdf_file"],
                        page=page_number,
                        line=line_number,
                        text=text,
                        bbox=bbox,
                        category=category,
                    )
                )
    return candidates


def _bbox_is_visible(bbox: tuple[float, float, float, float], width: float, height: float) -> bool:
    x0, top, x1, bottom = bbox
    return 0 <= x0 < x1 <= width and 0 <= top < bottom <= height


def select_candidates(candidates: list[Candidate], count_per_category: int = SAMPLES_PER_CATEGORY) -> list[Candidate]:
    selected = []
    for category in CATEGORIES:
        matching = sorted((item for item in candidates if item.category == category), key=stable_key)
        category_selected = []
        used_documents = set()
        for candidate in matching:
            if candidate.document_uid in used_documents:
                continue
            category_selected.append(candidate)
            used_documents.add(candidate.document_uid)
            if len(category_selected) == count_per_category:
                break
        if len(category_selected) < count_per_category:
            raise RuntimeError(f"Not enough distinct development documents with {category} OCR candidates")
        selected.extend(category_selected)
    return sorted(selected, key=lambda item: (item.category, stable_key(item)))


def degrade_image(image: Image.Image, identity: str) -> Image.Image:
    grayscale = image.convert("L")
    reduced_size = (max(1, round(image.width * 0.55)), max(1, round(image.height * 0.55)))
    reduced = grayscale.resize(reduced_size, Image.Resampling.LANCZOS)
    restored = reduced.resize(image.size, Image.Resampling.BICUBIC)
    angle = 0.6 if int(sha256_bytes(identity.encode())[-1], 16) % 2 else -0.6
    rotated = restored.rotate(angle, Image.Resampling.BICUBIC, fillcolor=255)
    low_contrast = ImageEnhance.Contrast(rotated).enhance(0.72)
    return low_contrast.filter(ImageFilter.GaussianBlur(0.7)).convert("RGB")


def render_candidate(candidate: Candidate) -> Image.Image:
    with pdfplumber.open(ROOT / candidate.pdf_file) as pdf:
        page = pdf.pages[candidate.page - 1]
        x0, top, x1, bottom = candidate.bbox
        padded_bbox = (
            max(0.0, x0 - 3.0),
            max(0.0, top - 1.5),
            min(float(page.width), x1 + 3.0),
            min(float(page.height), bottom + 1.5),
        )
        return page.crop(padded_bbox).to_image(resolution=RENDER_DPI, antialias=True).original.convert("RGB")


def prepare(source_manifest_path: Path, image_dir: Path, output_path: Path) -> dict:
    source_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_bytes)
    if source_manifest.get("split") != "development":
        raise RuntimeError("OCR benchmark preparation accepts only the development split")

    candidates = [candidate for document in source_manifest["documents"] for candidate in collect_candidates(document)]
    selected = select_candidates(candidates)
    image_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for candidate in selected:
        base_id = stable_key(candidate)[:16]
        clean = render_candidate(candidate)
        for variant, image in (("clean", clean), ("degraded", degrade_image(clean, base_id))):
            image_path = image_dir / f"{base_id}-{variant}.png"
            image.save(image_path, format="PNG", optimize=True)
            try:
                image_file = image_path.relative_to(ROOT).as_posix()
            except ValueError:
                image_file = str(image_path.resolve())
            samples.append(
                {
                    "id": f"{base_id}-{variant}",
                    "base_id": base_id,
                    "category": candidate.category,
                    "variant": variant,
                    "document_uid": candidate.document_uid,
                    "source_pdf": candidate.pdf_file,
                    "page": candidate.page,
                    "source_line": candidate.line,
                    "source_bbox": [round(value, 4) for value in candidate.bbox],
                    "reference": candidate.text,
                    "image_file": image_file,
                    "image_sha256": sha256_bytes(image_path.read_bytes()),
                    "width": image.width,
                    "height": image.height,
                }
            )

    manifest = {
        "schema_version": 1,
        "name": "TAT-DQA controlled OCR development benchmark",
        "split": "development",
        "source_dataset": "TAT-DQA",
        "source_dataset_url": source_manifest["dataset_url"],
        "license": source_manifest["license"],
        "source_manifest": source_manifest_path.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256_bytes(source_bytes),
        "selection": {
            "seed": SELECTION_SEED,
            "categories": list(CATEGORIES),
            "base_samples_per_category": SAMPLES_PER_CATEGORY,
            "maximum_samples_per_document_per_category": 1,
            "method": "SHA-256 ordering of visible directly extracted PDF line regions",
        },
        "rendering": {
            "dpi": RENDER_DPI,
            "clean": "300 DPI RGB crop with 3-point horizontal and 1.5-point vertical padding",
            "degraded": "55% downsample/upsample, deterministic +/-0.6 degree rotation, 0.72 contrast, 0.7 blur",
        },
        "reference": "NFKC-normalized direct PDF line text; silver standard",
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the controlled TAT-DQA development OCR benchmark.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.source_manifest, args.image_dir, args.output)
    print(f"Prepared {len(manifest['samples'])} OCR samples at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
