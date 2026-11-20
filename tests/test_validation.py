import pytest
from fastapi.testclient import TestClient

from structuralrag.data_generation.ict_queries import generate_ict_queries
from structuralrag.evaluation.evaluator import evaluate
from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.preprocessing.text_cleaner import split_sentences
from structuralrag.retrieval.bm25 import BM25Retriever
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.schema import Document

LONG = (
    "The agency reviewed eleven programs across four regions during the audit period. "
    "Officials stated that the program lacked consistent guidance for field offices nationwide. "
    "We found that data on outcomes were incomplete for most of the grants we examined. "
    "The department agreed with our recommendations and described planned corrective actions. "
    "Funding levels varied substantially between fiscal years because of supplemental appropriations."
)
OTHER = (
    "Inspectors visited nine facilities in three states between January and June. "
    "Several facilities could not produce maintenance records for critical equipment. "
    "Staff turnover at two sites exceeded forty percent during the review period. "
    "Managers said new software would improve record keeping by next spring."
)


@pytest.fixture
def report() -> Document:
    return build_document(
        "toy",
        "r1",
        "Report",
        [
            FlatHeading("Background", 1, LONG),
            FlatHeading("Findings", 1, ""),
            FlatHeading("Details", 2, OTHER),
        ],
    )


def test_sentence_split_keeps_abbreviations():
    assert split_sentences("The U.S. Army acted. It was fast.") == ["The U.S. Army acted.", "It was fast."]


def test_ict_skips_boilerplate_sentences():
    doc = build_document("toy", "r2", "R", [FlatHeading("A", 1, LONG), FlatHeading("B", 1, LONG)])
    _, queries = generate_ict_queries([doc], split="test", queries_per_doc=5, seed=1)
    assert queries == [], "sentences shared across sections give ambiguous labels"


def test_ict_removes_query_from_index(report):
    docs, queries = generate_ict_queries([report], split="test", queries_per_doc=5, seed=1)
    assert len(queries) == 2  # only sections with >= 4 sentences qualify
    by_uid = {s.uid: s for s in docs[0].sections}
    for q in queries:
        target = by_uid[next(iter(q.target_uids))]
        assert q.question not in target.text, "query sentence leaked into the index"
        assert target.text  # remaining context still there
    assert report.section("1").text == LONG, "input document must not be mutated"


def test_duplicate_section_ids_rejected(report):
    data = report.model_dump()
    data["sections"].append(data["sections"][0])
    with pytest.raises(ValueError):
        Document.model_validate(data)


def test_evaluator_end_to_end(report):
    docs, queries = generate_ict_queries([report], split="test", queries_per_doc=5, seed=1)
    res = evaluate(BM25Retriever(SectionCorpus(docs)), queries, scope="document", latency_sample=2)
    s = res.summary()
    assert s["n_queries"] == len(queries)
    assert 0 <= s["mrr@10"] <= 1 and s["latency_p50_ms"] >= 0


def test_api_schema_validation():
    from structuralrag.api.routes import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/retrieve", json={"query": "", "method": "bm25"}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "method": "nope"}).status_code == 422
