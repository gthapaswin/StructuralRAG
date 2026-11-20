"""Structure-specific diagnostics beyond exact-section hits."""

from __future__ import annotations

from structuralrag.retrieval.corpus import SectionCorpus


def _ancestors(section_id: str) -> set[str]:
    parts = section_id.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts))}


def structural_metrics(top1_uid: str, gold_uids: set[str], corpus: SectionCorpus) -> dict[str, float]:
    """Diagnostics for the top-ranked result.

    * valid_id: the returned id exists in the corpus.
    * correct_document: top-1 is in the gold document.
    * same_branch: top-1 shares the gold's top-level chapter (right region, maybe wrong leaf).
    * tree_distance: hops between top-1 and gold in the section tree (same document only).
    """
    valid = top1_uid in corpus.uid_to_idx
    doc, sid = top1_uid.split("::", 1)
    best = {"valid_id": float(valid), "correct_document": 0.0, "same_branch": 0.0, "tree_distance": float("nan")}
    for g in gold_uids:
        gdoc, gsid = g.split("::", 1)
        if gdoc != doc:
            continue
        best["correct_document"] = 1.0
        best["same_branch"] = float(sid.split(".")[0] == gsid.split(".")[0])
        a, b = _ancestors(sid) | {sid}, _ancestors(gsid) | {gsid}
        common = max((len(x.split(".")) for x in a & b), default=0)
        best["tree_distance"] = float(len(sid.split(".")) + len(gsid.split(".")) - 2 * common)
    return best
