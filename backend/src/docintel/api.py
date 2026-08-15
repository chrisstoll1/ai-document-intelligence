from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from docintel.chunking import ChunkRepository, ProvenanceChunker
from docintel.config import Settings
from docintel.db import initialize_database
from docintel.documents import DocumentCatalog, DocumentRecord, DocumentRepository
from docintel.extraction import ExtractionRepository, OcrUnavailableError, PdfExtractionError, PdfExtractor
from docintel.generation import GroundedGenerationService
from docintel.generators import HuggingFaceStructuredGenerator
from docintel.indexing import ChromaSemanticIndex
from docintel.ingestion import IngestionService
from docintel.lifecycle import DocumentLifecycleService
from docintel.metadata import MetadataRepository, SpacyEntityExtractor
from docintel.search import HybridSearchService
from docintel.storage import InvalidPdfError, PdfStore, PdfTooLargeError


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    status: str
    error_message: str | None
    embedding_model: str | None
    metadata_status: str
    metadata_model: str | None
    metadata_error: str | None


class EntityMentionResponse(BaseModel):
    page_number: int
    label: str
    text: str
    normalized_text: str
    char_start: int
    char_end: int
    confidence: float | None


class DocumentMetadataResponse(BaseModel):
    document_id: str
    status: str
    model: str | None
    error_message: str | None
    entities: list[EntityMentionResponse]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


class SearchRequest(QueryRequest):
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


class AnswerRequest(QueryRequest):
    limit: int = Field(default=5, ge=1, le=10)


class GroundedClaimResponse(BaseModel):
    text: str
    citation_ids: list[str]


class GroundingContextResponse(SearchResultResponse):
    context_id: str


class AnswerResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "generation_failed"]
    answer: str | None
    claims: list[GroundedClaimResponse]
    contexts: list[GroundingContextResponse]
    failure_reason: str | None


class ResetResponse(BaseModel):
    deleted_count: int


@dataclass(frozen=True)
class AppServices:
    ingestion: IngestionService
    documents: DocumentRepository
    pdf_store: PdfStore
    search: HybridSearchService
    semantic_index: ChromaSemanticIndex | None = None
    metadata: MetadataRepository | None = None
    generation: GroundedGenerationService | None = None
    lifecycle: DocumentLifecycleService | None = None


def build_services(settings: Settings) -> AppServices:
    initialize_database(settings.database_path)
    documents = DocumentRepository(settings.database_path)
    pdf_store = PdfStore(settings.data_dir)
    chunks = ChunkRepository(
        settings.database_path,
        ProvenanceChunker(max_words=settings.chunk_max_words, overlap=settings.chunk_overlap),
    )
    semantic_index = ChromaSemanticIndex(
        settings.data_dir / "chroma",
        model_name=settings.embedding_model,
        model_revision=settings.embedding_revision,
        query_prompt=settings.embedding_query_prompt,
    )
    metadata = MetadataRepository(settings.database_path)
    metadata_extractor = SpacyEntityExtractor(settings.ner_model, settings.ner_model_version)
    search = HybridSearchService(settings.database_path, chunks, semantic_index)
    generator = HuggingFaceStructuredGenerator(
        settings.generation_model,
        settings.generation_revision,
        max_new_tokens=settings.generation_max_new_tokens,
    )
    lifecycle = DocumentLifecycleService(documents, pdf_store, semantic_index)
    semantic_index.cleanup_stale_collections()
    return AppServices(
        ingestion=IngestionService(
            DocumentCatalog(pdf_store, documents),
            documents,
            PdfExtractor(),
            ExtractionRepository(settings.database_path),
            chunks,
            semantic_index,
            metadata,
            metadata_extractor,
        ),
        documents=documents,
        pdf_store=pdf_store,
        search=search,
        semantic_index=semantic_index,
        metadata=metadata,
        generation=GroundedGenerationService(search, generator),
        lifecycle=lifecycle,
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
        pdf_path = application.state.services.pdf_store.path_for(document_id)
        if not pdf_path.is_file():
            raise HTTPException(status_code=404, detail="Document file not found")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename,
        )

    @application.get("/api/documents/{document_id}/metadata", response_model=DocumentMetadataResponse)
    def get_document_metadata(document_id: str) -> DocumentMetadataResponse:
        document = application.state.services.documents.get(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        metadata = application.state.services.metadata
        entities = metadata.list_document(document_id) if metadata is not None else []
        return DocumentMetadataResponse(
            document_id=document_id,
            status=document.metadata_status,
            model=document.metadata_model,
            error_message=document.metadata_error,
            entities=[EntityMentionResponse(**entity.__dict__) for entity in entities],
        )

    @application.delete("/api/documents/{document_id}", status_code=204)
    def delete_document(document_id: str) -> Response:
        lifecycle = application.state.services.lifecycle
        if lifecycle is None:
            raise HTTPException(status_code=503, detail="Document lifecycle management is unavailable")
        if not lifecycle.delete(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return Response(status_code=204)

    @application.delete("/api/documents", response_model=ResetResponse)
    def reset_documents() -> ResetResponse:
        lifecycle = application.state.services.lifecycle
        if lifecycle is None:
            raise HTTPException(status_code=503, detail="Document lifecycle management is unavailable")
        return ResetResponse(deleted_count=lifecycle.reset())

    @application.post("/api/search", response_model=list[SearchResultResponse])
    def search(request: SearchRequest) -> list[SearchResultResponse]:
        return [
            SearchResultResponse(**result.__dict__)
            for result in application.state.services.search.search(request.query, limit=request.limit)
        ]

    @application.post("/api/answer", response_model=AnswerResponse)
    def answer(request: AnswerRequest) -> AnswerResponse:
        generation = application.state.services.generation
        if generation is None:
            raise HTTPException(status_code=503, detail="Grounded generation is unavailable")
        result = generation.answer(request.query, limit=request.limit)
        return AnswerResponse(
            status=result.status,
            answer=result.answer,
            claims=[
                GroundedClaimResponse(text=claim.text, citation_ids=list(claim.citation_ids))
                for claim in result.claims
            ],
            contexts=[
                GroundingContextResponse(context_id=context.context_id, **context.result.__dict__)
                for context in result.contexts
            ],
            failure_reason=result.failure_reason,
        )

    return application


def _document_response(document: DocumentRecord, filename: str) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=filename,
        size_bytes=document.size_bytes,
        status=document.status,
        error_message=document.error_message,
        embedding_model=document.embedding_model,
        metadata_status=document.metadata_status,
        metadata_model=document.metadata_model,
        metadata_error=document.metadata_error,
    )


app = create_app()
