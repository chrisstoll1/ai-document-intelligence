from __future__ import annotations

from docintel.documents import DocumentRepository
from docintel.indexing import ChromaSemanticIndex
from docintel.storage import PdfStore


class DocumentLifecycleService:
    def __init__(
        self,
        documents: DocumentRepository,
        pdf_store: PdfStore,
        semantic_index: ChromaSemanticIndex,
    ) -> None:
        self.documents = documents
        self.pdf_store = pdf_store
        self.semantic_index = semantic_index

    def delete(self, document_id: str) -> bool:
        document = self.documents.get(document_id)
        if document is None:
            return False
        self.documents.set_status(document_id, "deleting")
        self.semantic_index.delete_document(document_id)
        self.pdf_store.delete(document_id)
        return self.documents.delete(document_id)

    def reset(self) -> int:
        documents = [document for document, _ in self.documents.list_all()]
        for document in documents:
            self.documents.set_status(document.id, "deleting")
        self.semantic_index.reset()
        for document in documents:
            self.pdf_store.delete(document.id)
        return self.documents.delete_all()
