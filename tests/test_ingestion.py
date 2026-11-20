import pymupdf

from structuralrag.ingestion.pdf_parser import parse_pdf


def _make_pdf(path, with_outline: bool):
    doc = pymupdf.open()
    pages = [
        ("Introduction", 16, "This report studies retrieval over long documents."),
        ("Methods", 16, "We compare lexical and dense retrieval."),
        ("Evaluation Metrics", 12, "Recall and reciprocal rank are reported."),
    ]
    for title, size, body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=size)
        page.insert_text((72, 110), body, fontsize=10)
    if with_outline:
        doc.set_toc([[1, "Introduction", 1], [1, "Methods", 2], [2, "Evaluation Metrics", 3]])
    doc.save(path)


def test_pdf_outline_hierarchy(tmp_path):
    path = tmp_path / "report.pdf"
    _make_pdf(path, with_outline=True)
    doc = parse_pdf(path)
    assert [(s.section_id, s.title) for s in doc.sections] == [
        ("1", "Introduction"),
        ("2", "Methods"),
        ("2.1", "Evaluation Metrics"),
    ]
    assert "lexical and dense" in doc.section("2").text
    assert doc.section("2.1").page_start == 3


def test_pdf_font_fallback(tmp_path):
    path = tmp_path / "plain.pdf"
    _make_pdf(path, with_outline=False)
    doc = parse_pdf(path)
    titles = {s.title: s.level for s in doc.sections}
    assert titles["Introduction"] == 1 and titles["Evaluation Metrics"] == 2
