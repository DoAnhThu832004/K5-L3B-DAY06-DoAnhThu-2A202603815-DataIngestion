import re
from typing import Any


def _normalize_text_for_matching(text: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def evaluate_chunk_match(
    chunk: dict[str, Any],
    gold_source_id: str,
    gold_page: int | list[int],
    evidence_quote: str,
) -> tuple[bool, bool, int, int]:
    chunk_source_id = chunk.get("document_id") or ""
    if chunk_source_id and chunk_source_id != gold_source_id:
        return False, False, 0, 0

    chunk_page = chunk.get("page")
    page_matches = (chunk_page in gold_page) if isinstance(gold_page, list) else (chunk_page == gold_page)
    if not page_matches:
        return False, False, 0, 0

    chunk_clean = _normalize_text_for_matching(chunk.get("text", ""))

    clauses = [
        _normalize_text_for_matching(part)
        for part in evidence_quote.split("...")
        if len(_normalize_text_for_matching(part)) > 10
    ]

    if not clauses:
        probe = _normalize_text_for_matching(evidence_quote)[:40]
        has_probe = probe in chunk_clean
        return has_probe, has_probe, int(has_probe), 1

    matched_count = 0
    for clause in clauses:
        probe = clause[:35].strip()
        if probe in chunk_clean:
            matched_count += 1

    is_strict_hit = (matched_count == len(clauses))
    is_partial_hit = (matched_count >= max(1, (len(clauses) + 1) // 2))

    return is_strict_hit, is_partial_hit, matched_count, len(clauses)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    gold_source_id = question.get("gold_source_id", "")
    gold_page = question.get("gold_page")
    evidence_quote = question.get("evidence_quote", "")

    evaluated_hits: list[dict[str, Any]] = []
    strict_hit_at_1 = False
    strict_hit_at_3 = False
    partial_hit_at_1 = False
    partial_hit_at_3 = False

    for rank, h in enumerate(hits, start=1):
        strict_hit, partial_hit, matched_cnt, total_cnt = evaluate_chunk_match(
            h, gold_source_id, gold_page, evidence_quote
        )
        if rank == 1:
            if strict_hit:
                strict_hit_at_1 = True
            if partial_hit:
                partial_hit_at_1 = True

        if rank <= 3:
            if strict_hit:
                strict_hit_at_3 = True
            if partial_hit:
                partial_hit_at_3 = True

        full_text = h.get("text", "")
        evaluated_hits.append({
            "rank": rank,
            "chunk_id": h.get("chunk_id"),
            "page": h.get("page"),
            "score": h.get("score"),
            "is_evidence_hit": strict_hit,
            "is_partial_hit": partial_hit,
            "matched_clauses": f"{matched_cnt}/{total_cnt}",
            "metadata": h.get("metadata", {}),
            "text": full_text,
            "text_preview": (full_text[:200] + "...") if len(full_text) > 200 else full_text,
        })

    trace = {
        "id": question.get("id"),
        "split": question.get("split"),
        "type": question.get("type"),
        "question": question.get("question"),
        "filter": question.get("filter"),
        "gold_source_id": gold_source_id,
        "gold_page": gold_page,
        "evidence_quote": evidence_quote,
        "hit_at_1": strict_hit_at_1,
        "hit_at_3": strict_hit_at_3,
        "partial_hit_at_1": partial_hit_at_1,
        "partial_hit_at_3": partial_hit_at_3,
        "hits": evaluated_hits,
    }
    return trace


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    dev_traces = [t for t in traces if t.get("split") == "dev"]
    test_traces = [t for t in traces if t.get("split") == "test"]

    def calc_metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        if total == 0:
            return {
                "total": 0,
                "hit1_count": 0,
                "hit1_ratio": "0/0",
                "hit1_rate": 0.0,
                "hit3_count": 0,
                "hit3_ratio": "0/0",
                "hit3_rate": 0.0,
                "partial_hit3_count": 0,
                "partial_hit3_ratio": "0/0",
            }
        hit1 = sum(1 for t in subset if t.get("hit_at_1"))
        hit3 = sum(1 for t in subset if t.get("hit_at_3"))
        partial_hit3 = sum(1 for t in subset if t.get("partial_hit_at_3"))
        return {
            "total": total,
            "hit1_count": hit1,
            "hit1_ratio": f"{hit1}/{total}",
            "hit1_rate": round(hit1 / total, 4),
            "hit3_count": hit3,
            "hit3_ratio": f"{hit3}/{total}",
            "hit3_rate": round(hit3 / total, 4),
            "partial_hit3_count": partial_hit3,
            "partial_hit3_ratio": f"{partial_hit3}/{total}",
            "partial_hit3_rate": round(partial_hit3 / total, 4),
        }

    return {
        "dev": calc_metrics(dev_traces),
        "test": calc_metrics(test_traces),
        "overall": calc_metrics(traces),
    }
