# StructuralRAG

[![CI](https://github.com/gthapaswin/StructuralRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/gthapaswin/StructuralRAG/actions/workflows/ci.yml)

**Beyond chunking: structure-aware retrieval for long, hierarchical documents.**

Most RAG pipelines split a document into flat chunks and forget where each chunk lived.
StructuralRAG keeps the document tree (document → chapter → section → subsection) and
uses it during retrieval: given a question, find the section of a long report that answers it, then write an answer in which every sentence
cites the section it came from, or say the report does not cover the question.

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

### Grounded answers with citations

On top of retrieval, the pipeline writes an answer in which **every sentence cites a section id**.
Citations are constrained by a JSON schema to the sections actually shown to the model. A second
model from a different family (`llama3.1`) checks each sentence against the text of the sections it
cites, and sentences that fail are removed and reported. If the evidence does not cover the
question, the system answers "not covered" instead of guessing. Faithfulness is scored by an NLI
model (`nli-deberta-v3-small`) that plays no part in producing the answers. The test set is 150
LLM-written questions, with the top-3 sections and the same 900-word evidence budget for every setup.

| Setup (claim check on) | Abstained | Faithfulness (NLI) | Cites gold section | Citation precision |
|---|---|---|---|---|
| A: BM25 + plain chunks | 10.7% | 0.603 | 0.821 | 0.789 |
| B: structure-aware + plain chunks | 6.7% | 0.582 | **0.893** | 0.831 |
| C: structure-aware + hierarchical context | 6.7% | **0.611** | **0.893** | **0.845** |

- **Better retrieval means better-cited answers.** Structure-aware retrieval puts the gold section in
  the evidence 91% of the time vs 81% for BM25, and answers cite it more often (+7.5 points,
  permutation p = 0.04).
- **Abstention works.** On 100 questions asked against the *wrong* report, the system said "not
  covered" **96%** of the time, while abstaining on only 6.7% of answerable questions.
- **Hierarchical context (breadcrumb path, parent intro, children of heading-only sections) and the
  claim check give small gains that are not significant** (faithfulness +0.025, p = 0.40; +0.014,
  p = 0.06). The checker removes about 1 sentence in 20. These are honest null results at n = 150.

Full tables and example answers: [`reports/answer_results.md`](reports/answer_results.md). Try it
in the dashboard: choose a report, tick *Generate a cited answer*, and ask a question.

### Fine-tuning

Three models were fine-tuned on the GovReport **train** split: 1,500 reports and 11.4k ICT questions,
with no report shared with dev or test (checked by id and by a text fingerprint). Hard negatives come
from the document tree: they are other sections of the same report that the base model ranks highest,
skipping ancestors and descendants of the gold section. The embedding model and the reranker were
trained twice: on ICT questions only, and on ICT plus **1,395 LLM-written training questions** (from
700 train reports, validated the same way as the test paraphrases, each repeated 3×). Everything ran
on one laptop (Apple M5, 16 GB): about 25 min per embedding model, 40 min per reranker and 40 min for
the LoRA adapter. All models are scored on the same test questions as the base models, within one
report, with paired tests.

**1. Embedding model** (`bge-small-en-v1.5`, contrastive loss with hard negatives, 1 epoch)

| MRR@10 | ICT (n = 1,179) | LLM-written (n = 398) |
|---|---|---|
| Dense: base → ICT-trained → ICT+LLM-trained | 0.661 → 0.712 → **0.723** | 0.871 → 0.881 → **0.916** |
| Hybrid: base → ICT → ICT+LLM | 0.695 → 0.731 → **0.736** | 0.862 → 0.866 → 0.883 |
| Structure-aware: base → ICT → ICT+LLM | 0.703 → 0.712 → **0.723** | 0.877 → 0.881 → **0.916** |

- **Training on ICT questions only helps only on ICT questions** (dense +0.051, p < 0.001; on
  paraphrases +0.009, n.s.).
- **Adding LLM-written training questions fixes that and costs nothing on ICT.** Dense on paraphrases
  rises to 0.916 (+0.045 vs base, +0.035 vs ICT-only, both p ≤ 0.006); on ICT it is also slightly
  better than the ICT-only model (+0.012, p = 0.002). Caveat: training and test paraphrases come from
  the same LLM and prompt (on different reports), so part of the gain may be learning that writer's style.
- **Fine-tuning absorbed the structure signal.** For both fine-tuned encoders, re-tuning on dev set
  the path and parent weights to 0, so structure-aware became the same as dense. The path term
  mostly compensated for a general-purpose encoder. Fine-tuned hybrid (dense + BM25) is the best
  first-stage system on ICT questions (0.736).

**2. Cross-encoder reranker** (`bge-reranker-base`, reranks the top 20 of fine-tuned structure-aware,
sees "path + text")

A cross-encoder reads at most 512 tokens, so long sections were cut off: when the question sentence came
from beyond the first 250 words (30% of ICT questions), the plain reranker's MRR fell from about 0.73 to
0.54. **MaxP** fixes this: each section is split into overlapping 200-word windows (stride 100), each
prefixed with the section path, and the section's score is its best window.

| MRR@10 | ICT | LLM-written |
|---|---|---|
| Fine-tuned structure-aware (no reranker; ICT / ICT+LLM encoder) | 0.712 / 0.723 | 0.881 / 0.916 |
| + zero-shot reranker: plain / MaxP | 0.648 / 0.680 | 0.916 / 0.917 |
| + ICT-trained reranker: plain / MaxP | 0.690 / **0.727** | 0.874 / 0.862 |
| + ICT+LLM-trained reranker: plain / MaxP | 0.659 / 0.691 | **0.943** / 0.937 |

- **MaxP fixes the truncation problem.** It adds +0.032 to +0.037 MRR on ICT questions for every
  reranker (permutation p < 0.001), and the ICT-trained reranker with MaxP (0.727) is the best ICT result, though
  not significantly above fine-tuned structure-aware (+0.015, p = 0.17). On paraphrases, where the
  answer is usually not one buried sentence, MaxP changes nothing significant. It roughly doubles the
  latency (~1–1.5 s vs ~0.55 s per query).
- **The ICT+LLM-trained reranker is the best system in the project on paraphrased questions**: 0.943
  MRR@10 and 90.7% Recall@1, vs 0.916 for the zero-shot reranker (p = 0.003) and 0.874 for the
  ICT-only reranker (p < 0.001).
- **No single reranker wins on both question types.** The ICT+LLM reranker is worse than no
  reranker on ICT questions (−0.065 vs the ICT+LLM encoder alone, p < 0.001), and the ICT-only
  reranker is worse on paraphrases. Rerankers are sensitive to the kind of question they were trained on.

**3. Generative retrieval, LoRA** (`Qwen3-1.7B`, 4k examples, 1,000 steps with MLX; the target is
the gold id followed by the 4 hardest negatives)

| MRR@10 | ICT (n = 300) | LLM-written (n = 398) |
|---|---|---|
| Qwen3-1.7B zero-shot | 0.257 | 0.388 |
| **Qwen3-1.7B fine-tuned** | **0.549** (+0.29, p < 0.001) | **0.704** (+0.32, p < 0.001) |
| qwen3:8b zero-shot (earlier run, constrained decoding) | 0.510 | 0.732 |
| Structure-aware | 0.706 | 0.877 |

- **Fine-tuning more than doubles the small model's accuracy.** The fine-tuned 1.7B model is on par
  with the zero-shot 8B model, which is 4.7× larger: better on ICT, close on paraphrases. It is also faster
  (1.3 s vs 3.1 s per query), since it answers in JSON instead of prose.
- **It is still well below embedding retrieval** (−0.16 MRR). The model sees only titles and 20-word
  snippets, while the retrievers read every section's full text.

Full tables: [`reports/finetune_results.md`](reports/finetune_results.md) and
[`reports/generative_ft_results.md`](reports/generative_ft_results.md).

### Second domain: research papers with human-written questions

To check that the results are not specific to government reports or to synthetic questions, every
retriever was run on [QASPER](https://huggingface.co/datasets/allenai/qasper) (test split): **402 NLP
papers, 6,067 sections, 1,256 questions written by NLP practitioners** who saw only the title and
abstract. The gold sections hold the evidence paragraphs another annotator marked. This is a **zero-shot
transfer**: structure weights and fine-tuned models all come from GovReport; nothing is tuned on QASPER.
Full tables: [`reports/crossdomain_results.md`](reports/crossdomain_results.md).

| Method (document scope) | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| BM25 | 0.225 | 0.751 | 0.439 |
| Dense (base) | 0.349 | 0.752 | 0.516 |
| Hybrid (base) | 0.316 | 0.781 | 0.506 |
| **Structure-aware (base)** | 0.355 | 0.806 | **0.540** |
| Dense, GovReport ICT-trained / ICT+LLM-trained | 0.317 / 0.344 | 0.737 / 0.757 | 0.497 / 0.521 |
| Structure-aware + zero-shot reranker (MaxP) | 0.324 | 0.821 | 0.522 |
| **Structure-aware + GovReport ICT-trained reranker (MaxP)** | **0.389** | **0.839** | **0.572** |
| Structure-aware + GovReport ICT+LLM-trained reranker (MaxP) | 0.381 | 0.802 | 0.558 |

- **The structural signal carries over, and matters more here.** On human-written questions, base
  structure-aware beats dense (+0.024, p = 0.002), hybrid (+0.035, p < 0.001) and BM25 (+0.102,
  p < 0.001). On GovReport its lead over hybrid was not significant. Research papers have short,
  informative headings ("Experiments ::: Results"), which the path term uses.
- **Fine-tuned embeddings do not transfer.** The GovReport ICT-trained encoder is worse than the base
  model on papers (−0.019 dense, p = 0.008); adding LLM-written questions brings it back to the base
  level (0.521 vs 0.516, n.s.). Both stay below base structure-aware.
- **The fine-tuned reranker does transfer.** With MaxP windows, the GovReport-trained reranker is the
  best QASPER system: +0.032 MRR over structure-aware (p = 0.005) and +3.4 points Recall@1. The
  zero-shot reranker gives no gain (−0.018, n.s.). Learning *how to judge relevance* from long
  documents with hard negatives transfers better than learning a new embedding space.
- Absolute scores are lower than on GovReport even though papers have fewer sections (15 vs 23 per
  document): paper sections are more alike, and the questions were written without seeing the paper. Most gold sections are at depth 1–2 (919 and 276 questions); depth 3–4
  have only 59 and 2.

### When do section headings help? (heading study)

Structure-aware retrieval helps on some collections and not others, so the project tests *why*. Only
the headings change between conditions; section text and questions stay the same. Four question
sets: GovReport ICT and LLM-written questions, QASPER, and **Natural Questions** (1,744 real Google
queries over Wikipedia pages, a third domain). Within-document retrieval, MRR@10; tests treat
documents as the independent units.

| MRR@10 | Flat (no headings) | Real headings | Shuffled headings (placebo) | Generic "Section 3.2" |
|---|---|---|---|---|
| GovReport ICT | 0.689 | 0.691 | 0.690 | 0.689 |
| GovReport LLM-written | 0.864 | 0.878 | 0.866 | 0.867 |
| QASPER | 0.509 | **0.542** | 0.507 | 0.507 |
| NQ | **0.642** | 0.616 | 0.624 | 0.631 |

- **Headings help on QASPER (+0.035 over shuffled headings, p < 0.001), do nothing on GovReport ICT,
  and hurt on NQ (−0.026).** On QASPER the gain shrinks steadily as headings are made generic or noisy.
- **What explains it is how specific the target section's own heading is.** Specificity
  w = log(N/df)/log N (df = documents sharing the heading) needs no labels. Targets with generic
  headings ("Introduction", "Conclusions") lose to neighbours whose headings sound related to the
  question; targets with unique headings gain. This holds in all four sets (Spearman 0.15–0.31,
  p < 0.001) and survives controls for section length, depth, position and document size.
- **Negative results.** A document-level informativeness score and learned per-query weights do not
  beat a fixed weight; making common headings neutral helps GovReport and NQ but hurts QASPER,
  where common headings ("Results", "Datasets") are what questions target.

Details: [`reports/heading_study/FINDINGS.md`](reports/heading_study/FINDINGS.md); full tables in
[`results.md`](reports/heading_study/results.md); examples in
[`examples.md`](reports/heading_study/examples.md). Reproduce with `make heading-study`.

### Limitations

- ICT queries are sentences taken from the section itself, which favours lexical matching.
- The LLM question writer saw each section's title, so its questions may echo title words; this can
  favour methods that use titles and paths (structure-aware). The same model family also powers
  generative retrieval. The QASPER evaluation uses human-written questions, but only for retrieval.
- The answer evaluation uses the same LLM-written questions. The small NLI judge is strict with
  paraphrase, so absolute faithfulness scores are low; compare setups rather than read the scores
  in isolation. Faithfulness was not checked against human judgments.
- The LLM-written training and test questions come from the same model and prompt (different
  reports), which may overstate the gain from training on them. The MLX backend has no constrained decoding, so the zero-shot 1.7B model's prose
  answers are parsed for section ids (both models get the same parser).
- Three domains (government reports, NLP papers, Wikipedia), all English; scanned PDFs with noisy
  headings are only simulated (OCR noise on headings), not evaluated directly.

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
- **FastAPI** service (`/retrieve`, `/answer`, `/documents`, `/experiments`) and a **Streamlit** dashboard
  (retrieval playground with cited answers, document explorer, benchmarks, analytics, PDF upload).

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
uv run python scripts/run_answer_eval.py      # cited answers (+ `ollama pull llama3.1`, ~1.5 h)

# Optional fine-tuning (train split; Apple Silicon for the LLM step):
uv run python scripts/prepare_train_data.py   # 1,500 train reports, ICT questions, overlap checks
uv run --group finetune python scripts/finetune_embeddings.py  # ~25 min (ICT questions)
uv run --group finetune python scripts/finetune_reranker.py    # ~40 min
uv run python scripts/generate_questions.py --split train --max-docs 700  # LLM questions for training
uv run --group finetune python scripts/finetune_embeddings.py --questions mixed  # ICT + LLM
uv run --group finetune python scripts/finetune_reranker.py --questions mixed
uv run python scripts/run_finetune_eval.py    # base vs fine-tuned, plain vs MaxP reranking
uv run --group finetune python scripts/finetune_generative.py   # LoRA with MLX, ~40 min
uv run --group finetune python scripts/run_generative_ft_eval.py

# Second domain (QASPER research papers, human-written questions):
uv run python scripts/prepare_qasper.py
uv run python scripts/run_crossdomain_eval.py

# Heading study (third domain: download two NQ dev shards first, see scripts/prepare_nq.py):
make heading-study
uv run streamlit run app/streamlit_app.py # dashboard
uv run uvicorn structuralrag.api.routes:app --reload  # API at http://127.0.0.1:8000/docs
```

Or use `make data eval test app`. Configuration lives in [`configs/experiment.yaml`](configs/experiment.yaml).

### Docker

The image uses CPU-only PyTorch and builds the test data on first start.

```bash
docker build -t structuralrag .
docker run -p 8501:8501 structuralrag        # dashboard at http://localhost:8501
docker run -p 8000:8000 structuralrag api    # API at http://localhost:8000/docs
```

Generated answers need an Ollama server: pass `-e OLLAMA_HOST=http://host.docker.internal:11434`.
To host the dashboard as a Hugging Face Space, see [`deploy/huggingface-space/DEPLOY.md`](deploy/huggingface-space/DEPLOY.md).
CI (GitHub Actions) runs ruff and the test suite on every push.

## Repository layout

```text
src/structuralrag/
  schema.py              unified Document / Section / Query models
  ingestion/             GovReport, QASPER and Natural Questions adapters, PDF parser
  preprocessing/         text cleaning, hierarchy builder + validation
  data_generation/       ICT queries, validated LLM-written questions
  llm/                   Ollama and MLX clients (JSON output, response cache)
  retrieval/             corpus views, BM25, dense, hybrid, structure-aware, generative, reranker
  generation/            evidence context (plain / hierarchical), cited answers, claim check
  evaluation/            retrieval + answer metrics (NLI faithfulness), significance tests, tuning
  training/              hard negatives from the document tree, training question mixes
  analysis/              heading study: heading variants, within-document scoring, specificity
  api/                   FastAPI app and schemas
scripts/                 prepare_data, run_evaluation, generate_questions, run_generative_eval,
                         run_answer_eval, prepare_train_data, finetune_*, run_*_ft_eval,
                         prepare_qasper, run_crossdomain_eval
app/                     Streamlit dashboard
docker/, Dockerfile      container image (dashboard or API)
deploy/huggingface-space Hugging Face Space files
.github/workflows/       CI
tests/                   unit + integration tests (no model download needed)
```

Runs on a laptop: on an Apple M5, building all test indexes took ~30 s and peak memory was ~0.9 GB.

## Roadmap

- One reranker that works on both question types (e.g. train on both with MaxP windows).
- Give the generative model more of each section (longer snippets) to close the gap to embeddings.
- Tune the structure weights per domain, or learn them, instead of reusing GovReport settings.
- A third domain with messier structure (financial filings, scanned PDFs).
- Evaluate cited answers on QASPER's human-written reference answers.

## Acknowledgements

GovReport: Huang et al., *Efficient Attentions for Long Document Summarization*, NAACL 2021.
QASPER: Dasigi et al., *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers*, NAACL 2021.
Inspired by structure-aware generative retrieval work such as STAIR.

## License

MIT
