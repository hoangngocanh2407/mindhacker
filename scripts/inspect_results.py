import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles often default to cp1252, which can't encode Vietnamese
    # diacritics and crashes mid-print. Force UTF-8 so this script never dies
    # partway through printing a sample.
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import get_question_text  # noqa: E402

# NOTE: validate_results() (src/validate.py) only checks JSON shape, tag
# format, and that the answer cites one of the relevant_articles. It does
# NOT confirm the retrieved law articles are actually the correct ones for
# the question. This script exists so a human can eyeball that before any
# leaderboard submission.

DEFAULT_RESULTS_PATH = os.path.join(ROOT, "submission", "bm25_kf5", "results.json")
QUESTIONS_PATH = os.path.join(ROOT, "R2AIStage1DATA.json")


def main(results_path: str, limit: int) -> None:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = {q["id"]: get_question_text(q) for q in json.load(f)}
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Showing {min(limit, len(results))} of {len(results)} entries from {results_path}")
    print(
        "Reminder: format validation does not guarantee the retrieved law articles "
        "are content-correct — read these by hand before submitting.\n"
    )

    for entry in results[:limit]:
        answer = entry.get("answer", "")
        snippet = answer[:500] + ("..." if len(answer) > 500 else "")
        entry_id = entry.get("id")
        print(f"id={entry_id}")
        print(f"  question: {questions.get(entry_id, '<unknown question id>')}")
        print(f"  answer: {snippet}")
        print(f"  relevant_articles: {entry.get('relevant_articles')}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a sample of a results.json file for manual eyeballing.")
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help="Path to a results.json file (default: submission/bm25/results.json).",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of results to print (default 20).")
    args = parser.parse_args()
    main(args.results, args.limit)
