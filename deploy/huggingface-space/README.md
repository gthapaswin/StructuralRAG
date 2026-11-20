---
title: StructuralRAG
emoji: 📑
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Structure-aware retrieval for long government reports
---

# StructuralRAG demo

Structure-aware retrieval over long U.S. government reports (GovReport): pick a report,
ask a question, and see which sections each method returns. Source and full results:
https://github.com/gthapaswin/StructuralRAG

On first start the Space downloads GovReport and builds the section trees (about 2 minutes).
LLM features (generative retrieval, cited answers) need Ollama and are disabled here.
