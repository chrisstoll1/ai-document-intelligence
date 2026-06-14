from __future__ import annotations

import unittest
from pathlib import Path

from src.docintel.benchmarks import evaluate_rankings
from src.docintel.pipeline import RetrievalPipeline
from src.docintel.web import render_results


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

    def test_combined_search_includes_metadata_components(self) -> None:
        pipeline = RetrievalPipeline(ROOT / "data" / "sample_docs")
        results = pipeline.search("generated answers show evidence", mode="combined", limit=3)
        self.assertTrue(results)
        self.assertIn("keyword", results[0].components)
        self.assertIn("vector", results[0].components)
        self.assertIn("metadata", results[0].components)

    def test_web_renderer_outputs_search_results(self) -> None:
        pipeline = RetrievalPipeline(ROOT / "data" / "sample_docs")
        results = pipeline.search("privacy risks", mode="combined", limit=1)
        html = render_results(results)
        self.assertIn("Privacy Policy Notes", html)
        self.assertIn("Source:", html)

    def test_benchmark_metrics_reward_relevant_rankings(self) -> None:
        qrels = {
            "q1": {"d1": 1},
            "q2": {"d3": 1},
        }
        rankings = {
            "q1": ["d1", "d2"],
            "q2": ["d2", "d3"],
        }
        metrics = evaluate_rankings(qrels, rankings, k=2)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 0.75)
        self.assertGreater(metrics["ndcg"], 0.0)


if __name__ == "__main__":
    unittest.main()
