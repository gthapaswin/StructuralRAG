import pytest

from structuralrag.ingestion.govreport import row_to_document
from structuralrag.preprocessing.hierarchy_builder import FlatHeading, HierarchyError, build_document, validate_hierarchy


def test_section_ids_and_paths(ml_doc):
    ids = [s.section_id for s in ml_doc.sections]
    assert ids == ["1", "2", "2.1", "2.1.1", "2.2", "3", "3.1", "3.2"]
    cls = ml_doc.section("2.1.1")
    assert cls.parent_section_id == "2.1"
    assert cls.path == ["Machine Learning", "Supervised Learning", "Classification"]
    assert ml_doc.section("2").children == ["2.1", "2.2"]
    assert cls.is_leaf and not ml_doc.section("2").is_leaf


def test_depth_jump_is_clamped():
    doc = build_document("t", "d", "D", [FlatHeading("A", 1), FlatHeading("B", 3), FlatHeading("C", 2)])
    assert [(s.section_id, s.level) for s in doc.sections] == [("1", 1), ("1.1", 2), ("1.2", 2)]


def test_first_heading_deep_becomes_root():
    doc = build_document("t", "d", "D", [FlatHeading("A", 2), FlatHeading("B", 2)])
    assert [s.section_id for s in doc.sections] == ["1", "2"]


def test_validation_catches_broken_parent(ml_doc):
    ml_doc.sections[3].parent_section_id = "9"
    with pytest.raises(HierarchyError):
        validate_hierarchy(ml_doc)


def test_toc(ml_doc):
    assert "    2.1.1 Classification" in ml_doc.toc()


def test_govreport_row_adapter():
    row = {
        "id": "GAO_X-1",
        "document_sections": {
            "title": ["Background", "History", "Findings"],
            "paragraphs": ["", "Some history.\nMore.", "We found things."],
            "depth": [1, 2, 1],
        },
    }
    doc = row_to_document(row)
    assert [s.section_id for s in doc.sections] == ["1", "1.1", "2"]
    assert doc.section("1.1").text == "Some history. More."
    assert doc.metadata["agency"] == "GAO"
