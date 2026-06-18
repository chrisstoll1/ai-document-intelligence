from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .models import SearchResult
from .pipeline import RetrievalPipeline


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "data" / "sample_docs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser demo for the document retrieval prototype")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = RetrievalPipeline(args.docs)

    class SearchHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            mode = params.get("mode", ["combined"])[0]
            if mode not in {"keyword", "vector", "combined"}:
                mode = "combined"

            results = pipeline.search(query, mode=mode, limit=5) if query else []
            body = render_page(pipeline, query, mode, results)
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer((args.host, args.port), SearchHandler)
    print(f"Serving demo at http://{args.host}:{args.port}")
    print(f"Documents: {len(pipeline.documents)} | Chunks: {len(pipeline.chunks)} | Vector backend: {pipeline.vector_index.backend}")
    server.serve_forever()


def render_page(pipeline: RetrievalPipeline, query: str, mode: str, results: list[SearchResult]) -> str:
    result_html = render_results(results) if query else "<p class='text-muted'>Enter a query to search the sample collection.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evidence Retrieval Workbench</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
  <main class="container py-4">
    <header class="mb-4">
      <h1>Evidence Retrieval Workbench</h1>
      <p class="lead">Search the sample collection, compare retrieval modes, and check the exact passage used for each result.</p>
    </header>
    <table class="table table-sm table-bordered w-auto">
      <tbody>
        <tr><th>Documents</th><td>{len(pipeline.documents)}</td></tr>
        <tr><th>Chunks</th><td>{len(pipeline.chunks)}</td></tr>
        <tr><th>Vector backend</th><td>{escape(pipeline.vector_index.backend)}</td></tr>
      </tbody>
    </table>
    <form method="get" class="row gy-2 gx-2 mb-4">
      <div class="col-md-7">
        <input class="form-control" name="q" value="{escape(query)}" placeholder="Query the collection" autofocus>
      </div>
      <div class="col-md-3">
        {render_mode_select(mode)}
      </div>
      <div class="col-md-2">
        <button class="btn btn-primary w-100" type="submit">Search</button>
      </div>
    </form>
    <section class="mb-4">{result_html}</section>
  </main>
</body>
</html>"""


def render_mode_select(selected: str) -> str:
    options = []
    for value, label in [("combined", "Combined"), ("keyword", "Keyword"), ("vector", "Vector")]:
        selected_attr = " selected" if value == selected else ""
        options.append(f'<option value="{value}"{selected_attr}>{label}</option>')
    return f'<select class="form-select" name="mode">{"".join(options)}</select>'


def render_results(results: list[SearchResult]) -> str:
    if not results:
        return "<p class='text-muted'>No matching passages found.</p>"

    rows = []
    for index, result in enumerate(results, start=1):
        metadata = ", ".join(f"{key}: {value}" for key, value in result.chunk.metadata.items() if value)
        components = ", ".join(f"{key}: {value:.2f}" for key, value in result.components.items())
        rows.append(
            f"""<article class="list-group-item">
  <h2 class="h5">{index}. {escape(result.chunk.document_title)}</h2>
  <p class="mb-1"><strong>Method:</strong> {escape(result.method)} | <strong>Score:</strong> {result.score:.3f}</p>
  {f'<p class="mb-1"><strong>Components:</strong> {escape(components)}</p>' if components else ''}
  {f'<p class="mb-1"><strong>Metadata:</strong> {escape(metadata)}</p>' if metadata else ''}
  <p>{escape(result.chunk.text)}</p>
  <p class="mb-0 text-muted"><strong>Source:</strong> {escape(result.chunk.source_path)}</p>
</article>"""
        )
    return f'<div class="list-group">{"".join(rows)}</div>'


if __name__ == "__main__":
    main()
