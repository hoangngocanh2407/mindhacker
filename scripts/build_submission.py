import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles often default to cp1252, which can't encode Vietnamese
    # diacritics in printed summaries/errors and crashes mid-run.
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import run_pipeline  # noqa: E402
from src.preprocess import preprocess_corpus  # noqa: E402
from src.validate import validate_results  # noqa: E402

RAW_CORPUS = os.path.join(ROOT, "vbpl_dat.json")
RAW_QUESTIONS = os.path.join(ROOT, "R2AIStage1DATA.json")
CLEAN_CORPUS = os.path.join(ROOT, "data", "processed", "corpus_clean.json")
RESULTS_PATH = os.path.join(ROOT, "submission", "results.json")
ZIP_PATH = os.path.join(ROOT, "submission", "submission.zip")


def main() -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print("Preprocessing corpus...")
    summary = preprocess_corpus(RAW_CORPUS, CLEAN_CORPUS)
    print(f"  {summary}")

    print("Running retrieval + answer generation on all questions...")
    run_pipeline(RAW_QUESTIONS, CLEAN_CORPUS, RESULTS_PATH, top_k_retrieve=15, top_k_final=5)

    print("Validating results.json (format + citation correctness)...")
    errors = validate_results(RESULTS_PATH, expected_count=2000)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for error in errors[:20]:
            print(f"  - {error}")
        sys.exit(1)
    print("  Validation passed.")
    print("  NOTE: format validation does NOT confirm the retrieved law is actually correct.")
    print("  Run scripts/inspect_results.py and eyeball a sample before using a submission slot.")

    print("Building submission.zip (flat)...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(RESULTS_PATH, arcname="results.json")
    print(f"Done: {ZIP_PATH}")


if __name__ == "__main__":
    main()
