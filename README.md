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

### Generative retrieval and LLM-written questions

A local LLM (`qwen3:8b` via Ollama, zero-shot, temperature 0) reads only the report's **table of
contents** and generates a ranked list of section ids. Its output is constrained to the report's real
ids, so **100% of predictions are valid sections**. Evaluated on two question sets (document scope):
300 ICT queries (fixed random sample) and **398 LLM-written paraphrased questions** (2 per test report,
kept only if they share no 5-word span with the section and an independent LLM check confirms the
section answers them; 602 generated, 34% rejected). Full tables: [`reports/generative_results.md`](reports/generative_results.md).

| Method | ICT MRR@10 | LLM-question MRR@10 | LLM-question Recall@1 | P50 latency |
|---|---|---|---|---|
| BM25 | 0.685 | 0.815 | 0.724 | ~4 ms |
| Dense | 0.668 | 0.871 | 0.791 | ~5 ms |
| Hybrid | 0.701 | 0.862 | 0.776 | ~12 ms |
| **Structure-aware** | **0.706** | **0.877** | **0.799** | ~13 ms |
| Generative (ToC titles only) | 0.445 | 0.644 | 0.545 | — |
| Generative (ToC + 20-word snippets) | 0.510 | 0.732 | 0.641 | ~1.9 s |
| Generative + structure-aware (fusion) | 0.641 | 0.826 | 0.751 | ~1.9 s |

- **Zero-shot generative retrieval is far behind** (MRR@10 −0.15 to −0.20 vs structure-aware,
  p < 0.001) and ~150× slower. Titles alone are not enough to route a question; adding short section
  snippets to the ToC helps substantially (+0.065 to +0.088 MRR). STAIR fine-tunes its model; this
  zero-shot result is a baseline for that, not a reproduction.
- **Fusing the LLM's picks into structure-aware ranking also hurts** (−0.05 MRR, p ≤ 0.003). It only
  helped for the deepest sections (depth 5), where the buckets are too small (7–25 queries) to rely on.
- **Paraphrased questions flip the BM25-vs-dense result.** On ICT queries BM25 beats dense; on
  LLM-written questions dense is clearly better (0.871 vs 0.815), as expected when wording differs.
  Structure-aware is best on both sets: significantly better than BM25 on LLM questions (+0.062,
  p < 0.001), but **not significantly** better than dense or hybrid.

### Limitations

- ICT queries are sentences taken from the section itself, which favours lexical matching.
- The LLM question writer saw each section's title, so its questions may echo title words; this can
  favour methods that use titles and paths (structure-aware). The same model family also powers
  generative retrieval. Human-written questions (e.g. PDFTriage) are the next step.
- Single dataset (government reports), one small embedding model, one zero-shot LLM.

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
  | Generative | Local LLM (Ollama) reads the table of contents and generates section ids; JSON-schema constrained decoding guarantees valid ids. Optional fusion with structure-aware. |
- **Question sets**: ICT (sentence removed from its section) and validated LLM-written paraphrases.
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

# Optional, needs Ollama with qwen3:8b (`ollama pull qwen3:8b`):
uv run python scripts/generate_questions.py   # LLM-written questions (~30 min, cached)
uv run python scripts/run_generative_eval.py  # generative retrieval on both question sets
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
  data_generation/       ICT queries, validated LLM-written questions
  llm/                   Ollama client (schema-constrained output, response cache)
  retrieval/             corpus views, BM25, dense, hybrid, structure-aware, generative
  evaluation/            metrics, structural metrics, significance tests, evaluator
  api/                   FastAPI app and schemas
scripts/                 prepare_data, run_evaluation, generate_questions, run_generative_eval
app/                     Streamlit dashboard
tests/                   unit + integration tests (no model download needed)
```

Runs on a laptop: on an Apple M5, building all test indexes took ~30 s and peak memory was ~0.9 GB.

## Roadmap

- Fine-tune a small model (LoRA) on GovReport train-split questions for generative retrieval, the
  STAIR setting, and compare with the zero-shot baseline above.
- Human-written questions from PDFTriage.
- Cross-domain evaluation (financial, academic documents).

## Acknowledgements

GovReport: Huang et al., *Efficient Attentions for Long Document Summarization*, NAACL 2021.
Inspired by structure-aware generative retrieval work such as STAIR.

## License

MIT
