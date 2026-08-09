import pytest
from docintel.generation import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    GenerationContractError,
    GroundedGenerationService,
    ModelClaim,
    ModelGeneration,
    build_contexts,
    validate_generation,
)
from docintel.generators import HuggingFaceStructuredGenerator, user_prompt
from docintel.search import PersistentSearchResult
from pydantic import ValidationError


def _result(chunk_id: str = "chunk-1") -> PersistentSearchResult:
    return PersistentSearchResult(
        chunk_id=chunk_id,
        document_id="document-1",
        document_name="evidence.pdf",
        text="The source states that revenue increased.",
        page_start=2,
        page_end=2,
        score=0.01,
        keyword_rank=1,
        semantic_rank=2,
    )


class FakeSearch:
    def __init__(self, results=None) -> None:
        self.results = [_result()] if results is None else results
        self.calls = []

    def search(self, query: str, *, limit: int = 5):
        self.calls.append((query, limit))
        return self.results[:limit]


class FakeGenerator:
    version = "fake-generator-v1"

    def __init__(self, output=None, *, error=None) -> None:
        self.output = output or ModelGeneration(
            status="answered",
            claims=[ModelClaim(text="Revenue increased.", citation_ids=["C1"])],
        )
        self.error = error
        self.calls = 0

    def generate(self, question, contexts):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


def test_context_aliases_preserve_retrieval_order_and_provenance() -> None:
    results = [_result("first"), _result("second")]

    contexts = build_contexts(results)

    assert [context.context_id for context in contexts] == ["C1", "C2"]
    assert [context.result.chunk_id for context in contexts] == ["first", "second"]
    assert contexts[0].result.page_start == 2


def test_valid_generation_resolves_claims_without_copying_provenance_from_model() -> None:
    contexts = build_contexts([_result()])
    generation = ModelGeneration(
        status="answered",
        claims=[ModelClaim(text="Revenue increased.", citation_ids=["C1", "C1"])],
    )

    answer = validate_generation(generation, contexts)

    assert answer.status == "answered"
    assert answer.answer == "Revenue increased."
    assert answer.claims[0].citation_ids == ("C1",)
    assert answer.contexts[0].result.document_name == "evidence.pdf"


def test_unknown_citation_rejects_entire_generation() -> None:
    generation = ModelGeneration(
        status="answered",
        claims=[ModelClaim(text="Unsupported claim.", citation_ids=["C99"])],
    )

    answer = validate_generation(generation, build_contexts([_result()]))

    assert answer.status == "generation_failed"
    assert answer.answer is None
    assert answer.claims == ()
    assert answer.failure_reason == "invalid_citations"


def test_model_schema_rejects_uncited_blank_extra_and_contradictory_output() -> None:
    with pytest.raises(ValidationError):
        ModelGeneration(status="answered", claims=[])
    with pytest.raises(ValidationError):
        ModelGeneration(status="insufficient_evidence", claims=[ModelClaim(text="Claim", citation_ids=["C1"])])
    with pytest.raises(ValidationError):
        ModelClaim(text=" ", citation_ids=["C1"])
    with pytest.raises(ValidationError):
        ModelClaim.model_validate({"text": "Claim", "citation_ids": ["C1"], "source": "invented"})


def test_explicit_refusal_uses_server_authored_message_and_keeps_contexts() -> None:
    generation = ModelGeneration(status="insufficient_evidence", claims=[])

    answer = validate_generation(generation, build_contexts([_result()]))

    assert answer.status == "insufficient_evidence"
    assert answer.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert len(answer.contexts) == 1


def test_empty_retrieval_refuses_without_invoking_generator() -> None:
    generator = FakeGenerator()
    service = GroundedGenerationService(FakeSearch([]), generator)

    answer = service.answer("What happened?")

    assert answer.status == "insufficient_evidence"
    assert generator.calls == 0


def test_generation_service_uses_requested_limit_and_handles_contract_error() -> None:
    search = FakeSearch([_result("first"), _result("second")])
    generator = FakeGenerator(error=GenerationContractError("invalid_output"))
    service = GroundedGenerationService(search, generator)

    answer = service.answer("  What happened?  ", limit=1)

    assert search.calls == [("What happened?", 1)]
    assert answer.status == "generation_failed"
    assert answer.failure_reason == "invalid_output"
    assert len(answer.contexts) == 1


def test_generation_service_rejects_blank_queries_and_invalid_limits() -> None:
    service = GroundedGenerationService(FakeSearch(), FakeGenerator())

    with pytest.raises(ValueError, match="query"):
        service.answer(" ")
    with pytest.raises(ValueError, match="limit"):
        service.answer("question", limit=0)


def test_selected_generator_is_lazy_and_prompt_preserves_provenance() -> None:
    generator = HuggingFaceStructuredGenerator("repository", "revision", max_new_tokens=64)
    contexts = build_contexts([_result()])

    prompt = user_prompt("Question?", contexts)

    assert generator.version == "repository@revision"
    assert generator._model is None
    assert "[C1]" in prompt
    assert "pages 2-2" in prompt
    assert "The source states that revenue increased." in prompt
