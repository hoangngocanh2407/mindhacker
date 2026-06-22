import argparse
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone

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
DENSE_EMBEDDINGS_PATH = os.path.join(ROOT, "data", "processed", "dense_embeddings.npy")
DENSE_META_PATH = os.path.join(ROOT, "data", "processed", "dense_meta.json")

TOP_K_RETRIEVE = 15
TOP_K_FINAL = 5
RRF_K = 60


def _output_paths(mode: str) -> tuple[str, str, str, str]:
    out_dir = os.path.join(ROOT, "submission", mode)
    return (
        out_dir,
        os.path.join(out_dir, "results.json"),
        os.path.join(out_dir, "submission.zip"),
        os.path.join(out_dir, "run_meta.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VLegalQA submission.")
    parser.add_argument(
        "--retriever",
        choices=["bm25", "hybrid"],
        default="bm25",
        help=(
            "bm25 (default): the current submission baseline, BM25-only. "
            "hybrid: BM25+dense fusion, EXPERIMENTAL — last evaluated hybrid run scored "
            "worse on ARTICLES_F2MACRO than bm25 (see docs/reports/phase2_report.md). "
            "Only runs when explicitly requested; if the dense model fails to load, "
            "this fails loudly instead of silently falling back to bm25."
        ),
    )
    args = parser.parse_args()

    start_time = time.time()
    out_dir, results_path, zip_path, run_meta_path = _output_paths(args.retriever)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Mode: {args.retriever}")
    print("Preprocessing corpus...")
    corpus_summary = preprocess_corpus(RAW_CORPUS, CLEAN_CORPUS)
    print(f"  {corpus_summary}")

    with open(CLEAN_CORPUS, encoding="utf-8") as f:
        clean_corpus = json.load(f)

    dense_retriever = None
    embedding_model_name = None
    if args.retriever == "hybrid":
        print("Building dense retriever (explicit hybrid mode requested)...")
        # No try/except here on purpose: if this fails, the script must fail
        # loudly, not silently produce a bm25-only submission under the
        # "hybrid" label.
        from src.dense_retrieval import EMBEDDING_MODEL_NAME, DenseRetriever

        dense_retriever = DenseRetriever(
            clean_corpus,
            cache_embeddings_path=DENSE_EMBEDDINGS_PATH,
            cache_meta_path=DENSE_META_PATH,
        )
        embedding_model_name = EMBEDDING_MODEL_NAME

    print("Running retrieval + answer generation on all questions...")
    run_pipeline(
        RAW_QUESTIONS,
        CLEAN_CORPUS,
        results_path,
        top_k_retrieve=TOP_K_RETRIEVE,
        top_k_final=TOP_K_FINAL,
        dense_retriever=dense_retriever,
        rrf_k=RRF_K,
    )

    print("Validating results.json (format + citation correctness)...")
    errors = validate_results(results_path, expected_count=2000)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for error in errors[:20]:
            print(f"  - {error}")
        sys.exit(1)
    print("  Validation passed.")
    print("  NOTE: format validation does NOT confirm the retrieved law is actually correct.")
    print(f"  Run: python scripts/inspect_results.py --results {results_path}")

    print("Building submission.zip (flat)...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")

    elapsed = time.time() - start_time
    run_meta = {
        "mode": args.retriever,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k_retrieve": TOP_K_RETRIEVE,
        "top_k_final": TOP_K_FINAL,
        "rrf_k": RRF_K if args.retriever == "hybrid" else None,
        "embedding_model": embedding_model_name,
        "corpus_summary": corpus_summary,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"Done: {zip_path}")
    print(f"Run metadata: {run_meta_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
