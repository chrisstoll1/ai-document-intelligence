from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from docintel.search import PersistentSearchResult

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence in the retrieved passages."


class GenerationContractError(RuntimeError):
    pass


class ModelClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Generation claims must not be blank")
        return value

    @field_validator("citation_ids")
    @classmethod
    def citations_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if any(not citation_id.strip() for citation_id in value):
            raise ValueError("Generation citation IDs must not be blank")
        return value


class ModelGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["answered", "insufficient_evidence"]
    claims: list[ModelClaim]

    @model_validator(mode="after")
    def validate_status_and_claims(self):
        if self.status == "answered" and not self.claims:
            raise ValueError("Answered generation requires at least one claim")
        if self.status == "insufficient_evidence" and self.claims:
            raise ValueError("Insufficient-evidence generation must not contain claims")
        return self


@dataclass(frozen=True)
class GroundingContext:
    context_id: str
    result: PersistentSearchResult


@dataclass(frozen=True)
class GroundedClaim:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroundedAnswer:
    status: Literal["answered", "insufficient_evidence", "generation_failed"]
    answer: str | None
    claims: tuple[GroundedClaim, ...]
    contexts: tuple[GroundingContext, ...]
    failure_reason: str | None = None


class SearchService(Protocol):
    def search(self, query: str, *, limit: int = 5) -> Sequence[PersistentSearchResult]: ...


class StructuredGenerator(Protocol):
    version: str

    def generate(self, question: str, contexts: Sequence[GroundingContext]) -> ModelGeneration: ...


def build_contexts(results: Sequence[PersistentSearchResult]) -> tuple[GroundingContext, ...]:
    return tuple(GroundingContext(context_id=f"C{rank}", result=result) for rank, result in enumerate(results, start=1))


def validate_generation(
    generation: ModelGeneration,
    contexts: Sequence[GroundingContext],
) -> GroundedAnswer:
    context_ids = {context.context_id for context in contexts}
    if generation.status == "insufficient_evidence":
        return GroundedAnswer(
            status="insufficient_evidence",
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            claims=(),
            contexts=tuple(contexts),
        )

    claims = []
    for model_claim in generation.claims:
        citation_ids = tuple(dict.fromkeys(model_claim.citation_ids))
        if any(citation_id not in context_ids for citation_id in citation_ids):
            return GroundedAnswer(
                status="generation_failed",
                answer=None,
                claims=(),
                contexts=tuple(contexts),
                failure_reason="invalid_citations",
            )
        claims.append(GroundedClaim(model_claim.text.strip(), citation_ids))
    return GroundedAnswer(
        status="answered",
        answer=" ".join(claim.text for claim in claims),
        claims=tuple(claims),
        contexts=tuple(contexts),
    )


class GroundedGenerationService:
    def __init__(self, search: SearchService, generator: StructuredGenerator) -> None:
        self.search = search
        self.generator = generator

    def answer(self, query: str, *, limit: int = 5) -> GroundedAnswer:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if limit <= 0:
            raise ValueError("limit must be positive")
        contexts = build_contexts(self.search.search(normalized_query, limit=limit))
        if not contexts:
            return GroundedAnswer(
                status="insufficient_evidence",
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                claims=(),
                contexts=(),
            )
        try:
            generation = self.generator.generate(normalized_query, contexts)
        except GenerationContractError as error:
            return GroundedAnswer(
                status="generation_failed",
                answer=None,
                claims=(),
                contexts=contexts,
                failure_reason=str(error),
            )
        return validate_generation(generation, contexts)
