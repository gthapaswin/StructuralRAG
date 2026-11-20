# StructuralRAG

**Beyond chunking: structure-aware retrieval for long, hierarchical documents.**

Most RAG pipelines split a document into flat chunks and forget where each chunk lived.
StructuralRAG keeps the document tree (document → chapter → section → subsection) and
uses it during retrieval: given a question, find the section of a long report that answers it.

## Results

GovReport test split: **200 reports, 4,533 sections, 1,179 queries**, tuned on 60 separate dev
reports. Embeddings: `BAAI/bge-small-en-v1.5`, Apple M5 (MPS). Full tables: [`reports/results.md`](reports/results.md).

**Document scope** (find the right section inside a known report):

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 (95% CI) | nDCG@10 | P50 latency |
|---|---|---|---|---|---|---|
| BM25 | 0.546 | 0.880 | 0.956 | 0.687 (0.667–0.708) | 0.753 | 4.4 ms |
| Dense | 0.511 | 0.872 | 0.950 | 0.661 (0.639–0.681) | 0.731 | 5.3 ms |
| Hybrid (RRF) | 0.545 | **0.908** | 0.967 | 0.695 (0.673–0.715) | 0.761 | 12.4 ms |
| **Structure-aware** | **0.563** | 0.892 | **0.975** | **0.703** (0.683–0.722) | **0.769** | 12.7 ms |

**Corpus scope** (search all 4,533 sections): structure-aware MRR@10 0.573 vs BM25 0.575,
hybrid 0.565, dense 0.520.

### What the numbers say

- **Structure-aware retrieval scores highest on most metrics, but its lead over strong baselines is small.**
  Paired tests on MRR@10: vs dense +0.042 (p < 0.001, significant); vs BM25 +0.015 (permutation p = 0.06,
  Wilcoxon p = 0.003, borderline); vs hybrid +0.008 (p = 0.14, **not significant**).
- **Structure helps most for deep sections.** MRR@10 for targets at depth 3/4/5 is 0.700/0.704/0.699 for
  structure-aware vs 0.633/0.673/0.560 for BM25 and 0.672/0.676/0.653 for hybrid. At depth 1 it is
  slightly worse (0.686 vs 0.708). Depth-5 has only 25 queries, so treat that bucket with care.
- **The hierarchy-path signal helps a little; parent propagation did not help.** Tuning on dev picked
  α = 0.2 (path) and β = 0 (parent). Most of the gain over dense comes from fusing with BM25 over
  structure-enriched text.
- **Naively prepending metadata hurts dense retrieval.** Dense MRR@10 falls from 0.661 (text only) to
  0.632 (all metadata) as titles, paths and parent text are added to the embedded passage. BM25 barely
  changes. Keeping structure as a *separate field* works better than concatenating it.

### Limitations

- ICT queries are sentences taken from the section itself, which favours lexical matching (BM25 is a
  strong baseline here). Human-written or LLM-generated questions that refer to sections indirectly
  may show different results; that is the next experiment.
- Single dataset (government reports) and a single small embedding model.

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

Runs on a laptop: on an Apple M5, building all test indexes took ~30 s and peak memory was ~0.9 GB.

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
