v from __future__ import annotations

import unittest
from pathlib import Path

from src.docintel.pipeline import RetrievalPipeline


ROOT = Path(__file__).resolve().parents[1]


class RetrievalSmokeTests(unittest.TestCase):
    def test_pipeline_loads_sample_documents(self) -> None:
        pipeline = RetrievalPipeline(ROOT / "data" / "sample_docs")
        self.assertGreaterEqual(len(pipeline.documents), 4)
        self.assertGreaterEqual(len(pipeline.chunks), 4)

    def test_keyword_search_finds_privacy_document(self) -> None:
        pipeline = RetrievalPipeline(ROOT / "data" / "sample_docs")
        results = pipeline.search("privacy risks", mode="keyword", limit=3)
        self.assertTrue(any(result.chunk.document_id == "privacy_policy" for result in results))


if __name__ == "__main__":
    unittest.main()
