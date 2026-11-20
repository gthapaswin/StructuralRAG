from structuralrag.ingestion.qasper import row_to_document, row_to_queries


def row():
    return {
        "id": "p1",
        "title": "A Paper",
        "abstract": "We study things.",
        "full_text": {
            "section_name": ["Introduction", "Datasets", "Model ::: Background", "Model ::: Decoder", "Results"],
            "paragraphs": [["Intro text."], ["Data text."], ["Background text."], ["Decoder text."], ["Res."]],
        },
        "qas": {
            "question_id": ["q1", "q2", "q3"],
            "question": ["What decoder? ", "Unanswerable?", "Table only?"],
            "answers": [
                {"answer": [{"unanswerable": False, "evidence": ["Decoder text.", "Data text."]}]},
                {"answer": [{"unanswerable": True, "evidence": []}]},
                {"answer": [{"unanswerable": False, "evidence": ["FLOAT SELECTED: Table 1"]}]},
            ],
        },
    }


def test_missing_parent_is_created():
    doc, _ = row_to_document(row())
    titles = {s.section_id: s.title for s in doc.sections}
    model = next(s for s in doc.sections if s.title == "Model")
    assert model.text == "" and model.level == 1
    background = next(s for s in doc.sections if s.title == "Background")
    assert background.parent_section_id == model.section_id  # not under "Datasets"
    assert titles["1"] == "Abstract"


def test_queries_map_evidence_to_sections():
    doc, owner = row_to_document(row())
    qs, stats = row_to_queries(row(), doc, owner, "test")
    assert len(qs) == 1 and stats["unanswerable"] == 1 and stats["no_text_evidence"] == 1
    gold = {doc.section(s).title for s in qs[0].target_section_ids}
    assert gold == {"Decoder", "Datasets"} and qs[0].question == "What decoder?"
    assert qs[0].metadata["target_level"] == 2
