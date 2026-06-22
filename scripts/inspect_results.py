import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles often default to cp1252, which can't encode Vietnamese
    # diacritics and crashes mid-print. Force UTF-8 so this script never dies
    # partway through printing a sample.
    sys.stdout.reconfigure(encoding="utf-8")

# NOTE: validate_results() (src/validate.py) only checks JSON shape, tag
# format, and that the answer cites one of the relevant_articles. It does
# NOT confirm the retrieved law articles are actually the correct ones for
# the question. This script exists so a human can eyeball that before any
# leaderboard submission.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT, "submission", "results.json")


def main(n: int) -> None:
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Showing {min(n, len(results))} of {len(results)} entries from {RESULTS_PATH}")
    print(
        "Reminder: format validation does not guarantee the retrieved law articles "
        "are content-correct — read these by hand before submitting.\n"
    )

    for entry in results[:n]:
        answer = entry.get("answer", "")
        snippet = answer[:500] + ("..." if len(answer) > 500 else "")
        print(f"id={entry.get('id')}")
        print(f"  answer: {snippet}")
        print(f"  relevant_articles: {entry.get('relevant_articles')}")
        print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
