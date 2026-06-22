import json
import re

ARTICLE_TAG_RE = re.compile(r"^[^|]+\|[^|]+\|.+$")
REQUIRED_FIELDS = ("id", "answer", "relevant_docs", "relevant_articles")


def _article_id_from_tag(tag: str) -> str | None:
    parts = tag.split("|")
    if len(parts) < 3:
        return None
    return parts[-1].strip()


def validate_entry(entry: dict) -> list[str]:
    errors = []
    entry_id = entry.get("id", "?")

    for field in REQUIRED_FIELDS:
        if field not in entry:
            errors.append(f"id={entry_id}: missing field '{field}'")

    if "answer" in entry and not entry["answer"].strip():
        errors.append(f"id={entry_id}: empty answer")

    if "relevant_articles" in entry:
        if not entry["relevant_articles"]:
            errors.append(f"id={entry_id}: relevant_articles is empty")

        article_ids = []
        for tag in entry["relevant_articles"]:
            if not ARTICLE_TAG_RE.match(tag):
                errors.append(f"id={entry_id}: malformed relevant_article tag '{tag}'")
                continue
            article_id = _article_id_from_tag(tag)
            if article_id:
                article_ids.append(article_id)

        if article_ids and "answer" in entry:
            if not any(article_id in entry["answer"] for article_id in article_ids):
                errors.append(f"id={entry_id}: answer does not cite any relevant article")

    return errors


def validate_results(path: str, expected_count: int = 2000) -> list[str]:
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)

    errors = []
    if len(entries) != expected_count:
        errors.append(f"expected {expected_count} entries, found {len(entries)}")

    seen_ids = set()
    for entry in entries:
        errors.extend(validate_entry(entry))
        entry_id = entry.get("id")
        if entry_id in seen_ids:
            errors.append(f"duplicate id {entry_id}")
        seen_ids.add(entry_id)

    missing_ids = sorted(set(range(1, expected_count + 1)) - seen_ids)
    if missing_ids:
        errors.append(f"missing ids: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")

    return errors
