# StructuralRAG

**Beyond chunking: structure-aware retrieval for long, hierarchical documents.**

Most RAG pipelines split a document into flat chunks and forget where each chunk lived.
StructuralRAG keeps the document tree (document → chapter → section → subsection) and
uses it during retrieval: given a question, find the section of a long report that answers it.

> **Status:** work in progress. The pipeline, retrievers, evaluation framework, API and dashboard
> are implemented; benchmark results will be added to this README once the full evaluation has run.

## What it does

- **Unified document schema** (Pydantic): every dataset is converted into validated section trees
  with dotted ids (`3.2.1`), parent/child links, hierarchy paths and page ranges.
- **Ingestion**
  - [GovReport](https://huggingface.co/datasets/launch/gov_report) (`structure` config): long U.S.
    government reports with real section headings and nesting depth.
  - Any PDF: hierarchy from the embedded outline, falling back to font-size heading detection (PyMuPDF).
- **Retrieval methods**
  | Method | Description |
  |---|---|
  | BM25 | Lexical baseline (`rank-bm25`) |
  | Dense | `BAAI/bge-small-en-v1.5` embeddings, cosine similarity |
  | Hybrid | Reciprocal Rank Fusion of BM25 + dense |
  | **Structure-aware** | `cos(q, content) + α·cos(q, hierarchy path) + β·cos(q, parent content)`, fused with BM25 over structurally enriched text. α, β and the BM25 weight are tuned on a separate dev split. |
- **Evaluation**: Recall@{1,3,5,10}, MRR@10, nDCG@10, bootstrap 95% CIs, paired permutation +
  Wilcoxon tests, structural diagnostics (correct document, same chapter, tree distance to the gold
  section), latency P50/P95, index build time and memory.
- **Ablations**: A1 text only → A2 + title → A3 + hierarchy path → A4 + parent context → A5 all
  metadata, plus a component ablation of the structure-aware score. This separates the effect of
  *structure* from the effect of *simply adding more text*.
- **FastAPI** service (`/retrieve`, `/documents`, `/experiments`) and a **Streamlit** dashboard
  (retrieval playground, document explorer, benchmarks, analytics, PDF upload).

## Evaluation protocol

- **Queries (Inverse Cloze Task):** a sentence is sampled from a section and *removed* from that
  section's indexed text; the sentence is the query and the section is the gold target. Labels are
  unambiguous and the query cannot be matched verbatim.
- **Splits:** tuning uses documents from the HF `validation` split, reporting uses documents from the
  `test` split, so there is no document overlap. Leakage checks run during data preparation and fail
  loudly.
- **Scopes:** `document` (route a question to a section inside a known report, STAIR-style) and
  `corpus` (search all sections of all reports).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Runs on a laptop (Apple Silicon MPS, CUDA or CPU).

```bash
uv sync                                   # install
uv run python scripts/prepare_data.py     # download GovReport, build trees, generate queries
uv run python scripts/run_evaluation.py   # tune on dev, evaluate on test -> reports/
uv run pytest -q                          # tests
uv run streamlit run app/streamlit_app.py # dashboard
uv run uvicorn structuralrag.api.routes:app --reload  # API at http://127.0.0.1:8000/docs
```

Or use `make data eval test app`. Configuration lives in [`configs/experiment.yaml`](configs/experiment.yaml).

## Repository layout

```text
src/structuralrag/
  schema.py              unified Document / Section / Query models
  ingestion/             GovReport adapter, PDF parser
  preprocessing/         text cleaning, hierarchy builder + validation
  data_generation/       ICT query generation
  retrieval/             corpus views, BM25, dense, hybrid, structure-aware
  evaluation/            metrics, structural metrics, significance tests, evaluator
  api/                   FastAPI app and schemas
scripts/                 prepare_data.py, run_evaluation.py
app/                     Streamlit dashboard
tests/                   unit + integration tests (no model download needed)
```

## Roadmap

- Generative retrieval: predict a valid section id from the question + table of contents with a
  small local instruction model and constrained decoding.
- LLM-generated questions alongside ICT queries; human-written questions from PDFTriage.
- Cross-domain evaluation (financial, academic documents).

## Acknowledgements

GovReport: Huang et al., *Efficient Attentions for Long Document Summarization*, NAACL 2021.
Inspired by structure-aware generative retrieval work such as STAIR.

## License

MIT
