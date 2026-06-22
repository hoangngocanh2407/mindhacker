import json

from src.answer_gen import generate_answer


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_result_entry(question_id: int, top_articles: list[dict], top_k_final: int = 5) -> dict:
    selected = top_articles[:top_k_final]

    relevant_docs = _dedupe_preserve_order([a["relevant_doc_tag"] for a in selected])
    relevant_articles = _dedupe_preserve_order([a["relevant_article_tag"] for a in selected])

    return {
        "id": question_id,
        "answer": generate_answer(selected, max_articles=top_k_final),
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }


def write_results(entries: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
