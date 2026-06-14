from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def render_page(pipeline: RetrievalPipeline, query: str, mode: str, results: list) -> str:
    result_html = render_results(results) if query else "<p class='empty'>Enter a query to search the sample collection.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Document Intelligence Prototype</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #667085;
      --panel: #ffffff;
      --line: #d9e2ec;
      --accent: #3454d1;
      --accent-soft: #eef2ff;
      --bg: #f6f8fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(135deg, #f6f8fb 0%, #eef4ff 100%);
      color: var(--ink);
    }}
    main {{
      width: min(1100px, calc(100% - 32px));
      margin: 32px auto;
    }}
    header {{
      display: grid;
      gap: 8px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 5vw, 48px);
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      max-width: 760px;
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .stats, form, .result {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 45px rgba(30, 42, 80, 0.08);
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 14px;
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    form {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 160px 120px;
      gap: 10px;
      padding: 14px;
      margin-bottom: 18px;
    }}
    input, select, button {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
    }}
    button {{
      border: 0;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font-weight: 700;
    }}
    .result {{
      padding: 18px;
      margin-bottom: 14px;
    }}
    .result h2 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .meta, .components, .source {{
      color: var(--muted);
      font-size: 14px;
      margin: 8px 0;
    }}
    .passage {{
      line-height: 1.6;
      margin: 14px 0 0;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      margin-right: 6px;
    }}
    .empty {{
      color: var(--muted);
      padding: 18px;
    }}
    @media (max-width: 720px) {{
      form {{ grid-template-columns: 1fr; }}
      main {{ margin-top: 18px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Evidence-Based Document Retrieval</h1>
      <p class="subtitle">Search the sample document collection and inspect the passages behind each result. This demo uses the same keyword, vector and combined retrieval pipeline as the command-line prototype.</p>
    </header>
    <section class="stats">
      <span><strong>{len(pipeline.documents)}</strong> documents</span>
      <span><strong>{len(pipeline.chunks)}</strong> chunks</span>
      <span><strong>{escape(pipeline.vector_index.backend)}</strong> vector backend</span>
    </section>
    <form method="get">
      <input name="q" value="{escape(query)}" placeholder="Try: privacy risks, scanned PDFs, evidence summaries" autofocus>
      {render_mode_select(mode)}
      <button type="submit">Search</button>
    </form>
    <section>{result_html}</section>
  </main>
</body>
</html>"""


def render_mode_select(selected: str) -> str:
    options = []
    for value, label in [("combined", "Combined"), ("keyword", "Keyword"), ("vector", "Vector")]:
        selected_attr = " selected" if value == selected else ""
        options.append(f'<option value="{value}"{selected_attr}>{label}</option>')
    return f'<select name="mode">{"".join(options)}</select>'


def render_results(results: list) -> str:
    if not results:
        return "<p class='empty'>No matching passages found.</p>"

    cards = []
    for index, result in enumerate(results, start=1):
        metadata = ", ".join(f"{key}: {value}" for key, value in result.chunk.metadata.items() if value)
        components = ", ".join(f"{key}: {value:.2f}" for key, value in result.components.items())
        cards.append(
            f"""<article class="result">
  <h2>{index}. {escape(result.chunk.document_title)}</h2>
  <div><span class="badge">{escape(result.method)}</span><span class="badge">score {result.score:.3f}</span></div>
  {f'<p class="components">Components: {escape(components)}</p>' if components else ''}
  {f'<p class="meta">Metadata: {escape(metadata)}</p>' if metadata else ''}
  <p class="passage">{escape(result.chunk.text)}</p>
  <p class="source">Source: {escape(result.chunk.source_path)}</p>
</article>"""
        )
    return "".join(cards)


if __name__ == "__main__":
    main()
