from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pdfplumber
from PIL import Image
from prepare_ocr_benchmark import RENDER_DPI, degrade_image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "evaluation" / "ocr-retrieval"
DEFAULT_MANIFEST_DIR = ROOT / "evaluation" / "ocr"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def render_pages(pdf_path: Path) -> list[Image.Image]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.to_image(resolution=RENDER_DPI, antialias=True).original.convert("RGB") for page in pdf.pages]


def save_image_pdf(images: list[Image.Image], path: Path) -> None:
    if not images:
        raise ValueError("At least one page image is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, format="PDF", save_all=True, append_images=images[1:], resolution=RENDER_DPI)


def prepare(split: str, output_dir: Path, manifest_path: Path, *, overwrite: bool = False) -> dict:
    source_path = ROOT / "evaluation" / "tat_dqa" / f"{split}_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite OCR retrieval manifest: {manifest_path}")
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    condition_dir = output_dir / split
    documents = []
    for index, document in enumerate(source["documents"], start=1):
        pages = render_pages(ROOT / document["pdf_file"])
        variants = {}
        for variant, images in (
            ("clean", pages),
            (
                "degraded",
                [degrade_image(image, f"{document['uid']}:{page}") for page, image in enumerate(pages, start=1)],
            ),
        ):
            path = condition_dir / variant / f"{document['uid']}.pdf"
            if path.exists() and not overwrite:
                raise RuntimeError(f"Refusing to overwrite generated OCR PDF: {path}")
            save_image_pdf(images, path)
            variants[variant] = {
                "pdf_file": path.relative_to(ROOT).as_posix(),
                "pdf_sha256": sha256_file(path),
            }
        documents.append(
            {
                "uid": document["uid"],
                "canonical_pdf_sha256": document["pdf_sha256"],
                "source_pdf": document["pdf_file"],
                "page_count": len(pages),
                "variants": variants,
            }
        )
        print(f"Rendered {index}/{len(source['documents'])}: {document['uid']}")
    manifest = {
        "schema_version": 1,
        "name": "TAT-DQA paired full-page OCR retrieval benchmark",
        "split": split,
        "source_manifest": source_path.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rendering": {
            "dpi": RENDER_DPI,
            "clean": "Full source page rendered as an image-only PDF",
            "degraded": "Frozen 55% resampling, deterministic +/-0.6 degree rotation, 0.72 contrast, 0.7 blur",
        },
        "documents": documents,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare paired image-only PDFs for OCR retrieval evaluation.")
    parser.add_argument("--split", choices=("development", "locked_test"), default="development")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest or DEFAULT_MANIFEST_DIR / f"retrieval_{args.split}_manifest.json"
    result = prepare(args.split, args.output_dir, manifest, overwrite=args.overwrite)
    print(f"Prepared {len(result['documents'])} paired OCR documents at {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
