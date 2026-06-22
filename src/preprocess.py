import json
from pathlib import Path


def load_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dedupe_articles(records: list[dict]) -> list[dict]:
    seen_tags = set()
    deduped = []
    for record in records:
        tag = record["relevant_article_tag"]
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        deduped.append(record)
    return deduped


def drop_unknown(records: list[dict]) -> tuple[list[dict], list[dict]]:
    clean, dropped = [], []
    for record in records:
        if record["doc_id"] == "UNKNOWN":
            dropped.append(record)
        else:
            clean.append(record)
    return clean, dropped


def preprocess_corpus(raw_path: str, output_path: str) -> dict:
    records = load_corpus(raw_path)
    deduped = dedupe_articles(records)
    clean, dropped = drop_unknown(deduped)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    dropped_path = output_path_obj.with_name(output_path_obj.name + ".dropped.json")
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dropped_path, "w", encoding="utf-8") as f:
        json.dump(dropped, f, ensure_ascii=False, indent=2)

    return {
        "total": len(records),
        "after_dedupe": len(deduped),
        "dropped_unknown": len(dropped),
        "final": len(clean),
    }
