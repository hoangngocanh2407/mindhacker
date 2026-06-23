"""VLegalQA — cross-encoder reranker run, for a Kaggle GPU notebook.

Workflow: clone the (private) code repo from GitHub each run so the code is
always the latest, and read the competition data from a Kaggle Dataset (the
data files are gitignored / not in the repo). Paste this whole file into one
Kaggle cell (GPU + Internet ON).

It builds a wide BM25 (+ optional dense) candidate pool, reranks on GPU with
BAAI/bge-reranker-v2-m3, and writes a validated flat submission.zip to
/kaggle/working/.

Prereqs (one-time): a GitHub Personal Access Token stored as a Kaggle Secret,
and a Kaggle Dataset holding the two data files. See
docs/kaggle_rerank_instructions.md for the full step-by-step.
"""
import json
import os
import subprocess
import sys
import time
import zipfile

# Reduce CUDA fragmentation OOM (set before torch is imported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- Config -----------------------------------------------------------------
GITHUB_REPO = "hoangngocanh2407/mindhacker"   # owner/repo (private)
GIT_BRANCH = "main"
GITHUB_PAT_SECRET = "GITHUB_PAT"              # name of the Kaggle Secret holding the token

# Kaggle Dataset folder containing vbpl_dat.json + R2AIStage1DATA.json.
# Adjust to match your dataset's mounted path (see the Input panel).
DATA_DIR = "/kaggle/input/vlegalqa-data"

CLONE_DIR = "/kaggle/working/mindhacker"
OUT_DIR = "/kaggle/working"

# Candidate pool. top-50 ∪ dense took ~50+ min (170k pairs). top-20 BM25-only
# (~40k pairs) runs in ~10-15 min and keeps almost all recoverable recall
# (BM25 recall is front-loaded). Widen later (50/100, USE_DENSE=True) if the
# reranker is recall-limited and you can afford the time.
TOP_K_RETRIEVE = 20      # candidate depth per source
TOP_K_FINAL = 3          # best cutoff found on the leaderboard (ARTICLES_F2)
USE_DENSE = False        # add dense candidates too (raises recall ceiling, ~2x slower)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 512  # cap seq len — legal articles reach 245k chars; uncapped = GPU OOM
RERANK_BATCH_SIZE = 32   # raise for speed if VRAM allows; lower (8/4) if CUDA OOM
# ---------------------------------------------------------------------------

# --- Clone the private code repo (token read from Kaggle Secrets, never printed) ---
from kaggle_secrets import UserSecretsClient  # noqa: E402

_pat = UserSecretsClient().get_secret(GITHUB_PAT_SECRET)
subprocess.run(["rm", "-rf", CLONE_DIR], check=False)
subprocess.run(
    ["git", "clone", "--quiet", "--depth", "1", "--branch", GIT_BRANCH,
     f"https://{_pat}@github.com/{GITHUB_REPO}.git", CLONE_DIR],
    check=True,
)
# Scrub the token from the cloned repo's remote so it can't leak into output.
subprocess.run(
    ["git", "-C", CLONE_DIR, "remote", "set-url", "origin",
     f"https://github.com/{GITHUB_REPO}.git"],
    check=False,
)
del _pat
print(f"Cloned {GITHUB_REPO}@{GIT_BRANCH} -> {CLONE_DIR}")
print("  HEAD:", subprocess.run(
    ["git", "-C", CLONE_DIR, "log", "-1", "--oneline"],
    capture_output=True, text=True).stdout.strip())

sys.path.insert(0, CLONE_DIR)
os.system(f"{sys.executable} -m pip install -q rank-bm25 pyvi")

from src.pipeline import run_pipeline  # noqa: E402
from src.preprocess import preprocess_corpus  # noqa: E402
from src.reranker import Reranker  # noqa: E402
from src.validate import validate_results  # noqa: E402

RAW_CORPUS = os.path.join(DATA_DIR, "vbpl_dat.json")
RAW_QUESTIONS = os.path.join(DATA_DIR, "R2AIStage1DATA.json")
CLEAN_CORPUS = os.path.join(OUT_DIR, "corpus_clean.json")
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")
ZIP_PATH = os.path.join(OUT_DIR, "submission.zip")

assert os.path.exists(RAW_CORPUS), f"Not found: {RAW_CORPUS} — fix DATA_DIR to your dataset path"
assert os.path.exists(RAW_QUESTIONS), f"Not found: {RAW_QUESTIONS} — fix DATA_DIR"

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
