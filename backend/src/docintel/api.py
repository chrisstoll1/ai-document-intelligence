from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from docintel.chunking import ChunkRepository
from docintel.config import Settings
from docintel.db import initialize_database
from docintel.documents import DocumentCatalog, DocumentRecord, DocumentRepository
from docintel.extraction import ExtractionRepository, OcrUnavailableError, PdfExtractionError, PdfExtractor
from docintel.indexing import ChromaSemanticIndex
from docintel.ingestion import IngestionService
from docintel.search import HybridSearchService
from docintel.storage import InvalidPdfError, PdfStore, PdfTooLargeError


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    status: str
    error_message: str | None
    embedding_model: str | None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class SearchResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int
    page_end: int
    score: float
    keyword_rank: int | None
    semantic_rank: int | None


@dataclass(frozen=True)
class AppServices:
    ingestion: IngestionService
    documents: DocumentRepository
    pdf_store: PdfStore
    search: HybridSearchService
    semantic_index: ChromaSemanticIndex | None = None


def build_services(settings: Settings) -> AppServices:
    initialize_database(settings.database_path)
    documents = DocumentRepository(settings.database_path)
    pdf_store = PdfStore(settings.data_dir)
    chunks = ChunkRepository(settings.database_path)
    semantic_index = ChromaSemanticIndex(settings.data_dir / "chroma")
    return AppServices(
        ingestion=IngestionService(
            DocumentCatalog(pdf_store, documents),
            documents,
            PdfExtractor(),
            ExtractionRepository(settings.database_path),
            chunks,
            semantic_index,
        ),
        documents=documents,
        pdf_store=pdf_store,
        search=HybridSearchService(settings.database_path, chunks, semantic_index),
        semantic_index=semantic_index,
    )


def create_app(
    settings: Settings | None = None,
    *,
    service_builder: Callable[[Settings], AppServices] = build_services,
) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        services = service_builder(resolved_settings)
        application.state.services = services
        try:
            yield
        finally:
            if services.semantic_index is not None:
                services.semantic_index.close()

    application = FastAPI(title="Document Intelligence API", lifespan=lifespan)

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/documents", response_model=DocumentResponse)
    def upload_document(upload: UploadFile = File()) -> DocumentResponse:
        filename = Path(upload.filename or "document.pdf").name
        try:
            document = application.state.services.ingestion.ingest(upload.file, filename)
        except PdfTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except InvalidPdfError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except OcrUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except PdfExtractionError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _document_response(document, filename)

    @application.get("/api/documents", response_model=list[DocumentResponse])
    def list_documents() -> list[DocumentResponse]:
        return [
            _document_response(document, filename)
            for document, filename in application.state.services.documents.list_all()
        ]

    @application.get("/api/documents/{document_id}", response_model=DocumentResponse)
    def get_document(document_id: str) -> DocumentResponse:
        document = application.state.services.documents.get(document_id)
        filename = application.state.services.documents.latest_name(document_id)
        if document is None or filename is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return _document_response(document, filename)

    @application.get("/api/documents/{document_id}/pdf")
    def get_document_pdf(document_id: str) -> FileResponse:
        document = application.state.services.documents.get(document_id)
        filename = application.state.services.documents.latest_name(document_id)
        if document is None or filename is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return FileResponse(
            application.state.services.pdf_store.path_for(document_id),
            media_type="application/pdf",
            filename=filename,
        )

    @application.post("/api/search", response_model=list[SearchResultResponse])
    def search(request: SearchRequest) -> list[SearchResultResponse]:
        return [
            SearchResultResponse(**result.__dict__)
            for result in application.state.services.search.search(request.query, limit=request.limit)
        ]

    return application


def _document_response(document: DocumentRecord, filename: str) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=filename,
        size_bytes=document.size_bytes,
        status=document.status,
        error_message=document.error_message,
        embedding_model=document.embedding_model,
    )


app = create_app()
