# Report Figures

These SVG charts are generated directly from committed frozen evaluation artifacts.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\render_report_figures.py
```

- `retrieval_locked_test.svg` reads `evaluation/results/tat_dqa_locked_test_selected.json`.
- `generation_locked_test.svg` joins `evaluation/results/generation_locked_test_qwen.json` to `evaluation/generation/locked_test_manifest.json` by question UID.
- `system_architecture.svg` documents the as-built model, storage, API, and browser components.
- `workflow_sequence.svg` documents ingestion and grounded-query message flow.
