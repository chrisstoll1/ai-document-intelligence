# Document Intelligence

Local-first PDF ingestion, entity enrichment, hybrid evidence retrieval, and grounded generation. The backend stores original PDFs and application records locally, indexes chunks with SQLite FTS5 and Chroma, and preserves page/block/entity/citation provenance.

## Docker
Build and start the application:

```powershell
docker compose up -d --build
```

To enable grounded generation, add the GPU override:

```powershell
docker compose -f compose.yml -f compose.gpu.yml up -d --build
```

- App: http://127.0.0.1:3000
- API docs: http://127.0.0.1:3000/docs
