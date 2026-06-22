"""VLegalQA — cross-encoder reranker run, for a Kaggle GPU notebook.

Paste this whole file into a single Kaggle notebook cell (GPU + Internet ON).
It reuses the repo's tested src/ modules (uploaded as a Kaggle Dataset), builds
a wide BM25 (+ optional dense) candidate pool, reranks on GPU with
BAAI/bge-reranker-v2-m3, and writes a validated, flat submission.zip to
/kaggle/working/.

See docs/kaggle_rerank_instructions.md for the full step-by-step (dataset
setup, enabling GPU/Internet, downloading the result).
"""
import json
import os
import sys
import time
import zipfile

# Reduce CUDA fragmentation OOM (set before torch is imported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- Config -----------------------------------------------------------------
# Folder (a Kaggle Dataset) that contains this repo: src/, vbpl_dat.json,
# R2AIStage1DATA.json. Adjust the dataset slug to match what you uploaded.
REPO_DIR = "/kaggle/input/vlegalqa-repo"
OUT_DIR = "/kaggle/working"

TOP_K_RETRIEVE = 50   # wide candidate pool -> higher recall ceiling for the reranker
TOP_K_FINAL = 3       # best cutoff found on the leaderboard (ARTICLES_F2)
USE_DENSE = True      # also pull dense candidates into the pool (raises recall ceiling)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 512  # cap seq len — legal articles reach 245k chars; uncapped = GPU OOM
RERANK_BATCH_SIZE = 16   # lower this (8/4) if you still hit CUDA OOM
# ---------------------------------------------------------------------------

sys.path.insert(0, REPO_DIR)

# Kaggle images ship sentence-transformers + torch; install the small extras.
os.system(f"{sys.executable} -m pip install -q rank-bm25 pyvi")

from src.preprocess import preprocess_corpus  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.reranker import Reranker  # noqa: E402
from src.validate import validate_results  # noqa: E402

RAW_CORPUS = os.path.join(REPO_DIR, "vbpl_dat.json")
RAW_QUESTIONS = os.path.join(REPO_DIR, "R2AIStage1DATA.json")
CLEAN_CORPUS = os.path.join(OUT_DIR, "corpus_clean.json")
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")
ZIP_PATH = os.path.join(OUT_DIR, "submission.zip")

start = time.time()

print("Preprocessing corpus...")
print(" ", preprocess_corpus(RAW_CORPUS, CLEAN_CORPUS))

with open(CLEAN_CORPUS, encoding="utf-8") as f:
    clean_corpus = json.load(f)

dense_retriever = None
if USE_DENSE:
    from src.dense_retrieval import DenseRetriever

    print("Building dense candidate retriever (GPU)...")
    dense_retriever = DenseRetriever(
        clean_corpus,
        cache_embeddings_path=os.path.join(OUT_DIR, "dense_embeddings.npy"),
        cache_meta_path=os.path.join(OUT_DIR, "dense_meta.json"),
    )

print(f"Loading reranker {RERANKER_MODEL} (GPU)...")
reranker = Reranker(RERANKER_MODEL, max_length=RERANK_MAX_LENGTH, batch_size=RERANK_BATCH_SIZE)

print(f"Running pipeline: candidates=BM25 top-{TOP_K_RETRIEVE}"
      f"{' ∪ dense' if USE_DENSE else ''} -> rerank -> top-{TOP_K_FINAL}...")
run_pipeline(
    RAW_QUESTIONS,
    CLEAN_CORPUS,
    RESULTS_PATH,
    top_k_retrieve=TOP_K_RETRIEVE,
    top_k_final=TOP_K_FINAL,
    dense_retriever=dense_retriever,
    reranker=reranker,
)

print("Validating...")
errors = validate_results(RESULTS_PATH, expected_count=2000)
if errors:
    print(f"VALIDATION FAILED ({len(errors)}):")
    for e in errors[:20]:
        print(" ", e)
    raise SystemExit(1)
print("  Validation passed.")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(RESULTS_PATH, arcname="results.json")

print(f"Done in {time.time() - start:.0f}s -> {ZIP_PATH}")
print("Download submission.zip from the Kaggle notebook's Output panel.")
