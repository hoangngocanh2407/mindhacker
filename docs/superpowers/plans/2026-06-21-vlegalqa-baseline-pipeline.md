# VLegalQA Baseline Pipeline — Implementation Plan (Revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end, valid `submission.zip` for the VLegalQA SME legal-assistant competition using BM25 retrieval over the law corpus and template-based answer generation — a baseline that runs on all 2000 questions and passes format + citation validation, so the team has something submittable within the first 1–2 days instead of risking a format rejection near the deadline.

**Architecture:** A linear pipeline: clean the law corpus (dedupe + drop unusable records, auto-creating output dirs) → BM25 keyword retrieval per question over a combined `doc_name + article_id + text` index → template-based answer that cites the retrieved article(s) → export to the exact `results.json` schema with deduped tag lists → validate format **and** citation correctness → manually eyeball a sample → zip flat.

This revision keeps the same Phase-1-only scope as before, and tightens four things found in review: the validator now checks that the answer actually cites a retrieved article (not just that the field is non-empty); export dedupes both `relevant_docs` and `relevant_articles`; BM25 indexes document title + article id, not just body text; preprocessing auto-creates its own output directory. It also adds two new steps: a mandatory schema-inspection task before `run_pipeline` is implemented (no blind `question["question"]`), and a manual result-inspection script before anything goes near the leaderboard.

**Out of scope (explicitly, do not add):** dense/embedding retrieval, cross-encoder reranking, LLM-based answer generation, any HTTP API, any UI, Docker, any database. This is a pure local Python pipeline that produces a zip file. Phase 2 (embeddings, reranker, LLM generation) is a separate future plan.

**Tech Stack:** Python 3, `rank-bm25` (pure-Python BM25, no Java/Lucene dependency), `pytest` for tests, stdlib `json`/`zipfile`/`re`/`pathlib`/`argparse` for everything else.

## Global Constraints

- Output file must be named exactly `results.json`, containing one entry per question with fields `id`, `answer`, `relevant_docs`, `relevant_articles`.
- Each string in `relevant_docs`/`relevant_articles` must follow `<mã văn bản>|<tên văn bản>` / `<mã văn bản>|<tên văn bản>|<điều>` — reuse the corpus's existing `relevant_doc_tag`/`relevant_article_tag` fields verbatim, do not reconstruct them.
- Neither `relevant_docs` nor `relevant_articles` may contain the same tag twice within one entry.
- `answer` must contain a literal "Điều N" substring that matches the article number of **at least one** tag in that same entry's `relevant_articles` — this is checked by the validator, not just assumed from the template.
- `results.json` must contain entries for all 2000 question ids (1–2000), no duplicates, no missing ids.
- `submission.zip` must contain `results.json` flat at the zip root — no subdirectories.
- Vòng Riêng allows only 5 total leaderboard submissions across the whole competition — this plan produces a locally-validated, manually-eyeballed file; the actual upload is a deliberate manual step outside this plan's scope.
- Corpus rows with `doc_id == "UNKNOWN"` cannot produce a valid `<mã văn bản>|...` tag and must be excluded from the retrievable corpus (logged, not silently lost).
- Rows with identical `relevant_article_tag` (333 known exact-duplicate pairs) must be deduplicated before indexing.
- The question file's actual field name for question text must be confirmed by inspection before `run_pipeline` is written — never assume the key is `"question"` without checking.
- No dense embeddings, no reranker, no LLM calls, no API/UI/Docker/DB anywhere in this plan.

---

## File Structure

```
D:\MindHacker\
  requirements.txt              # rank-bm25, pytest
  pytest.ini                    # pythonpath = . so `from src.x import y` works
  .gitignore                    # data/processed, submission/, __pycache__
  src/
    __init__.py
    preprocess.py                # corpus cleanup: dedupe, drop UNKNOWN, mkdir output dir
    retrieval.py                 # tokenize() + BM25Retriever over doc_name+article_id+text
    answer_gen.py                # generate_answer() template
    schema.py                    # get_question_text() adapter for the question file
    export.py                    # build_result_entry() (deduped), write_results()
    validate.py                  # validate_entry() (incl. citation check), validate_results()
    pipeline.py                  # run_pipeline() glue, uses schema.get_question_text()
  scripts/
    build_submission.py          # CLI: real data end-to-end -> submission.zip
    inspect_results.py           # CLI: print N sample results for manual eyeballing
  tests/
    __init__.py
    test_preprocess.py
    test_retrieval.py
    test_answer_gen.py
    test_schema.py
    test_validate.py
    test_pipeline_smoke.py
  data/processed/                # generated: corpus_clean.json (gitignored)
  submission/                    # generated: results.json, submission.zip (gitignored)
```

Existing files `R2AIStage1DATA.json` and `vbpl_dat.json` stay at the project root untouched.

---

### Task 0: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd "D:/MindHacker" && git init
```

Expected: `Initialized empty Git repository in D:/MindHacker/.git/`

- [ ] **Step 2: Create requirements.txt**

```
rank-bm25>=0.2.2
pytest>=7.4
```

- [ ] **Step 3: Create pytest.ini**

```ini
[pytest]
pythonpath = .
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.pyc
data/processed/
submission/
```

- [ ] **Step 5: Create empty package markers**

`src/__init__.py` — empty file.
`tests/__init__.py` — empty file.

- [ ] **Step 6: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: `rank-bm25` and `pytest` install without errors.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini .gitignore src/__init__.py tests/__init__.py
git commit -m "chore: scaffold VLegalQA baseline pipeline project"
```

---

### Task 1: Corpus preprocessing (dedupe + drop UNKNOWN + auto-create output dir)

**Files:**
- Create: `src/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces: `load_corpus(path: str) -> list[dict]`, `dedupe_articles(records: list[dict]) -> list[dict]`, `drop_unknown(records: list[dict]) -> tuple[list[dict], list[dict]]`, `preprocess_corpus(raw_path: str, output_path: str) -> dict` (returns `{"total", "after_dedupe", "dropped_unknown", "final"}`, creates `Path(output_path).parent` if missing). Later tasks (retrieval, pipeline) consume the corpus records written to `output_path` — each record keeps its original keys: `doc_id`, `doc_name`, `article_id`, `text`, `relevant_doc_tag`, `relevant_article_tag`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preprocess.py
import json

from src.preprocess import dedupe_articles, drop_unknown, load_corpus, preprocess_corpus

SAMPLE_RECORDS = [
    {
        "doc_id": "80/2021/NĐ-CP",
        "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "article_id": "Điều 1",
        "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều.",
        "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
    },
    {
        "doc_id": "80/2021/NĐ-CP",
        "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "article_id": "Điều 1",
        "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều.",
        "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
        "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
    },
    {
        "doc_id": "UNKNOWN",
        "doc_name": "Nghị định 18-CP về Luật Đầu tư nước ngoài",
        "article_id": "Điều 5",
        "text": "Quy định về đầu tư nước ngoài tại Việt Nam.",
        "relevant_doc_tag": "UNKNOWN|Nghị định 18-CP về Luật Đầu tư nước ngoài",
        "relevant_article_tag": "UNKNOWN|Nghị định 18-CP về Luật Đầu tư nước ngoài|Điều 5",
    },
    {
        "doc_id": "59/2020/QH14",
        "doc_name": "Luật Doanh nghiệp 59/2020/QH14",
        "article_id": "Điều 4",
        "text": "Doanh nghiệp nhỏ và vừa được hiểu là doanh nghiệp có quy mô nhỏ.",
        "relevant_doc_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14",
        "relevant_article_tag": "59/2020/QH14|Luật Doanh nghiệp 59/2020/QH14|Điều 4",
    },
]


def test_dedupe_articles_removes_exact_duplicates():
    deduped = dedupe_articles(SAMPLE_RECORDS)
    tags = [r["relevant_article_tag"] for r in deduped]
    assert len(deduped) == 3
    assert tags.count(
        "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1"
    ) == 1


def test_drop_unknown_removes_unknown_doc_id():
    clean, dropped = drop_unknown(SAMPLE_RECORDS)
    assert all(r["doc_id"] != "UNKNOWN" for r in clean)
    assert len(dropped) == 1
    assert dropped[0]["doc_id"] == "UNKNOWN"


def test_preprocess_corpus_writes_clean_file_and_returns_summary(tmp_path):
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "clean.json"
    raw_path.write_text(json.dumps(SAMPLE_RECORDS, ensure_ascii=False), encoding="utf-8")

    summary = preprocess_corpus(str(raw_path), str(out_path))

    assert summary == {"total": 4, "after_dedupe": 3, "dropped_unknown": 1, "final": 2}
    clean = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(clean) == 2
    assert all(r["doc_id"] != "UNKNOWN" for r in clean)


def test_preprocess_corpus_creates_missing_output_directory(tmp_path):
    raw_path = tmp_path / "raw.json"
    nested_out_path = tmp_path / "does" / "not" / "exist" / "clean.json"
    raw_path.write_text(json.dumps(SAMPLE_RECORDS, ensure_ascii=False), encoding="utf-8")

    assert not nested_out_path.parent.exists()

    preprocess_corpus(str(raw_path), str(nested_out_path))

    assert nested_out_path.exists()
    assert (nested_out_path.parent / "clean.json.dropped.json").exists()


def test_load_corpus_returns_list_of_dicts(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(SAMPLE_RECORDS, ensure_ascii=False), encoding="utf-8")
    records = load_corpus(str(raw_path))
    assert len(records) == 4
    assert records[0]["doc_id"] == "80/2021/NĐ-CP"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_preprocess.py -v
```

Expected: FAIL — `ImportError: cannot import name 'dedupe_articles' from 'src.preprocess'` (module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/preprocess.py
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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    dropped_path = output_path + ".dropped.json"
    with open(dropped_path, "w", encoding="utf-8") as f:
        json.dump(dropped, f, ensure_ascii=False, indent=2)

    return {
        "total": len(records),
        "after_dedupe": len(deduped),
        "dropped_unknown": len(dropped),
        "final": len(clean),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_preprocess.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Sanity-check against the real corpus**

```bash
python -c "from src.preprocess import preprocess_corpus; print(preprocess_corpus('vbpl_dat.json', 'data/processed/corpus_clean.json'))"
```

Expected: a dict like `{'total': 4755, 'after_dedupe': 4422, 'dropped_unknown': N, 'final': 4422-N}`. `data/processed/` did not exist before this command — if it's created automatically and the command doesn't raise `FileNotFoundError`, Step 3's `mkdir` is working. If `after_dedupe` isn't close to `4755 - 333`, stop and inspect `vbpl_dat.json` before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: add corpus preprocessing (dedupe + drop UNKNOWN + auto-mkdir)"
```

---

### Task 2: BM25 retrieval over doc_name + article_id + text

**Files:**
- Create: `src/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: corpus records from Task 1 (`output_path` of `preprocess_corpus`), each with at least `text`, `doc_name`, `article_id`, `relevant_doc_tag`, `relevant_article_tag`.
- Produces: `tokenize(text: str) -> list[str]`, `class BM25Retriever: __init__(self, corpus: list[dict])`, `.search(self, query: str, top_k: int = 15) -> list[dict]` — each returned dict is a shallow copy of the matching corpus record plus a `"score": float` key, sorted by score descending, length ≤ `top_k`. Task 3 (`answer_gen`) and Task 5 (`export`) consume this return shape directly.

**Note on the tokenizer:** `tokenize()` is a deliberately simple regex word-splitter (lowercase + `\w+`). It does no Vietnamese word-segmentation, no stopword removal, no diacritic normalization beyond what Python's `str.lower()` does. This is a Phase-1 baseline tokenizer, not a tuned one — flagged here so nobody mistakes it for a finished retrieval component when planning Phase 2.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval.py
from src.retrieval import BM25Retriever, tokenize

TEXT_MATCH_CORPUS = [
    {
        "relevant_doc_tag": "A|Doc A",
        "relevant_article_tag": "A|Doc A|Điều 1",
        "doc_name": "Doc A",
        "article_id": "Điều 1",
        "text": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi tham gia đấu thầu.",
    },
    {
        "relevant_doc_tag": "B|Doc B",
        "relevant_article_tag": "B|Doc B|Điều 2",
        "doc_name": "Doc B",
        "article_id": "Điều 2",
        "text": "Quy định về hợp đồng lao động và xử lý vi phạm kỷ luật.",
    },
    {
        "relevant_doc_tag": "C|Doc C",
        "relevant_article_tag": "C|Doc C|Điều 3",
        "doc_name": "Doc C",
        "article_id": "Điều 3",
        "text": "Thủ tục chuyển đổi hộ kinh doanh thành doanh nghiệp.",
    },
]

# Body text alone shares almost no keywords with the query below — only
# doc_name carries the matching terms. This is exactly the case the combined
# index is meant to fix.
TITLE_MATCH_CORPUS = [
    {
        "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80",
        "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80|Điều 5",
        "doc_name": "Nghị định 80/2021/NĐ-CP về hỗ trợ doanh nghiệp nhỏ và vừa",
        "article_id": "Điều 5",
        "text": "Cơ quan có trách nhiệm báo cáo định kỳ hằng năm cho Bộ Kế hoạch và Đầu tư.",
    },
    {
        "relevant_doc_tag": "X|Luật Đấu thầu",
        "relevant_article_tag": "X|Luật Đấu thầu|Điều 9",
        "doc_name": "Luật Đấu thầu",
        "article_id": "Điều 9",
        "text": "Quy định về hồ sơ dự thầu và thời hạn nộp hồ sơ.",
    },
]


def test_tokenize_lowercases_and_splits_on_punctuation():
    tokens = tokenize("Ưu đãi, thuế và đất đai?")
    assert tokens == ["ưu", "đãi", "thuế", "và", "đất", "đai"]


def test_search_returns_most_relevant_article_first():
    retriever = BM25Retriever(TEXT_MATCH_CORPUS)
    results = retriever.search("ưu đãi thuế khi tham gia đấu thầu cho doanh nghiệp", top_k=2)
    assert results[0]["relevant_article_tag"] == "A|Doc A|Điều 1"
    assert len(results) <= 2


def test_search_result_includes_score_key():
    retriever = BM25Retriever(TEXT_MATCH_CORPUS)
    results = retriever.search("hợp đồng lao động vi phạm kỷ luật", top_k=1)
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


def test_search_matches_via_doc_name_when_body_text_does_not():
    retriever = BM25Retriever(TITLE_MATCH_CORPUS)
    results = retriever.search("Nghị định 80/2021 hỗ trợ doanh nghiệp nhỏ và vừa", top_k=1)
    assert results[0]["relevant_article_tag"] == "80/2021/NĐ-CP|Nghị định 80|Điều 5"


def test_search_matches_via_article_id():
    retriever = BM25Retriever(TEXT_MATCH_CORPUS)
    results = retriever.search("nội dung của Điều 3 là gì", top_k=1)
    assert results[0]["relevant_article_tag"] == "C|Doc C|Điều 3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retrieval.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval'`.

- [ ] **Step 3: Write the implementation**

```python
# src/retrieval.py
"""BM25 retrieval over the law corpus.

`tokenize()` is intentionally a simple regex word-splitter — lowercase plus
`\\w+` — with no Vietnamese segmentation, stemming, or stopword removal.
It's a Phase-1 baseline, good enough for keyword overlap on legal terms
like "Điều 4" or "doanh nghiệp nhỏ và vừa", not a tuned tokenizer.
"""
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _searchable_text(record: dict) -> str:
    return f"{record['doc_name']} {record['article_id']} {record['text']}"


class BM25Retriever:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._tokenized = [tokenize(_searchable_text(record)) for record in corpus]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            record = dict(self.corpus[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retrieval.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval.py tests/test_retrieval.py
git commit -m "feat: BM25 retrieval indexes doc_name + article_id + text, not body text alone"
```

---

### Task 3: Template-based answer generation

**Files:**
- Create: `src/answer_gen.py`
- Test: `tests/test_answer_gen.py`

**Interfaces:**
- Consumes: list of corpus-record dicts as returned by `BM25Retriever.search()` (Task 2), each with `article_id`, `doc_name`, `text`.
- Produces: `generate_answer(top_articles: list[dict], max_articles: int = 5) -> str`. Task 5 (`export.build_result_entry`) calls this directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_answer_gen.py
from src.answer_gen import generate_answer

ARTICLE = {
    "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "article_id": "Điều 1",
    "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều của Luật.",
    "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
}


def test_generate_answer_includes_article_citation():
    answer = generate_answer([ARTICLE])
    assert "Điều 1" in answer
    assert "Nghị định 80/2021/NĐ-CP" in answer


def test_generate_answer_handles_empty_list():
    answer = generate_answer([])
    assert "không tìm thấy điều luật" in answer.lower()


def test_generate_answer_truncates_long_text():
    long_article = dict(ARTICLE)
    long_article["text"] = "a" * 1000
    answer = generate_answer([long_article])
    assert "..." in answer
    assert len(answer) < 1000


def test_generate_answer_respects_max_articles():
    articles = [ARTICLE, ARTICLE, ARTICLE]
    answer = generate_answer(articles, max_articles=1)
    assert answer.count("Điều 1") == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_answer_gen.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.answer_gen'`.

- [ ] **Step 3: Write the implementation**

```python
# src/answer_gen.py
def generate_answer(top_articles: list[dict], max_articles: int = 5) -> str:
    if not top_articles:
        return "Không tìm thấy điều luật liên quan trong cơ sở dữ liệu để trả lời câu hỏi này."

    parts = []
    for article in top_articles[:max_articles]:
        snippet = article["text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "..."
        parts.append(f"Theo {article['article_id']} {article['doc_name']}: {snippet}")
    return " ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_answer_gen.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/answer_gen.py tests/test_answer_gen.py
git commit -m "feat: add template-based answer generation with article citations"
```

---

### Task 4: Inspect the real question file schema + write an adapter

**Why this task exists and comes before the pipeline:** every later task assumes the question file's text field is called `"question"`. That assumption must be confirmed against the real file, not hard-coded blind — if the organizers ever change the field name, `run_pipeline` should fail with a clear message pointing at one place to fix, not a silent `KeyError` buried in a loop.

**Files:**
- Create: `src/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `get_question_text(record: dict) -> str`. Task 6 (`pipeline.run_pipeline`) calls this instead of indexing `record["question"]` directly.

- [ ] **Step 1: Inspect the real file**

```bash
python -c "import json; d = json.load(open('R2AIStage1DATA.json', encoding='utf-8')); print(list(d[0].keys())); print(d[0]); print(len(d))"
```

Expected (confirmed against the current dataset as of 2026-06-21): keys are `['id', 'question']`, e.g. `{'id': 1, 'question': '...'}`, length `2000`. **If your copy of the file prints different keys, do not proceed to Step 2 with `"question"` — update `QUESTION_TEXT_KEYS` in Step 3 to match what you actually saw first.**

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_schema.py
import pytest

from src.schema import get_question_text


def test_get_question_text_returns_value_for_confirmed_key():
    assert get_question_text({"id": 1, "question": "Câu hỏi mẫu?"}) == "Câu hỏi mẫu?"


def test_get_question_text_raises_clear_error_for_unknown_schema():
    with pytest.raises(KeyError) as exc_info:
        get_question_text({"id": 1, "content": "không phải field đúng"})
    assert "QUESTION_TEXT_KEYS" in str(exc_info.value)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.schema'`.

- [ ] **Step 4: Write the implementation**

```python
# src/schema.py
# Confirmed against R2AIStage1DATA.json on 2026-06-21: every record has the
# shape {"id": int, "question": str}. If the organizers change the field
# name, add the new key here rather than patching call sites.
QUESTION_TEXT_KEYS = ("question",)


def get_question_text(record: dict) -> str:
    for key in QUESTION_TEXT_KEYS:
        if key in record:
            return record[key]
    raise KeyError(
        f"No known question-text key found in record (tried {QUESTION_TEXT_KEYS}); "
        f"record keys were: {sorted(record.keys())}. Update QUESTION_TEXT_KEYS in src/schema.py."
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_schema.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/schema.py tests/test_schema.py
git commit -m "feat: add question-schema adapter, confirmed against real R2AIStage1DATA.json"
```

---

### Task 5: Submission export (deduped) + format & citation validation

**Files:**
- Create: `src/export.py`
- Create: `src/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `generate_answer` from Task 3; corpus-record dicts (with `relevant_doc_tag`, `relevant_article_tag`) as returned by `BM25Retriever.search()` from Task 2.
- Produces: `build_result_entry(question_id: int, top_articles: list[dict], top_k_final: int = 5) -> dict`, `write_results(entries: list[dict], output_path: str) -> None`, `validate_entry(entry: dict) -> list[str]`, `validate_results(path: str, expected_count: int = 2000) -> list[str]`. Task 6 (`pipeline.run_pipeline`) and `scripts/build_submission.py` call all four.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate.py
import json

from src.export import build_result_entry, write_results
from src.validate import validate_entry, validate_results

ARTICLE = {
    "doc_name": "Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "article_id": "Điều 1",
    "text": "Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều của Luật.",
    "relevant_doc_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành",
    "relevant_article_tag": "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành|Điều 1",
}


def test_build_result_entry_has_required_fields_and_valid_tags():
    entry = build_result_entry(1, [ARTICLE], top_k_final=5)
    assert entry["id"] == 1
    assert "Điều 1" in entry["answer"]
    assert entry["relevant_docs"] == [ARTICLE["relevant_doc_tag"]]
    assert entry["relevant_articles"] == [ARTICLE["relevant_article_tag"]]
    assert validate_entry(entry) == []


def test_build_result_entry_dedupes_relevant_docs_and_relevant_articles():
    # Same article passed in twice (e.g. retrieval returned a near-duplicate
    # chunk) must collapse to a single entry in BOTH output lists.
    entry = build_result_entry(2, [ARTICLE, ARTICLE], top_k_final=5)
    assert entry["relevant_docs"] == [ARTICLE["relevant_doc_tag"]]
    assert entry["relevant_articles"] == [ARTICLE["relevant_article_tag"]]
    assert len(entry["relevant_articles"]) == 1


def test_validate_entry_flags_missing_field():
    errors = validate_entry({"id": 1, "answer": "x", "relevant_docs": ["a|b"]})
    assert any("relevant_articles" in e for e in errors)


def test_validate_entry_flags_malformed_article_tag():
    entry = {
        "id": 2,
        "answer": "...",
        "relevant_docs": ["X|Y"],
        "relevant_articles": ["chỉ có hai phần|không đủ"],
    }
    errors = validate_entry(entry)
    assert any("malformed" in e for e in errors)


def test_validate_entry_flags_empty_answer():
    entry = {"id": 3, "answer": "  ", "relevant_docs": ["a|b"], "relevant_articles": ["a|b|Điều 1"]}
    errors = validate_entry(entry)
    assert any("empty answer" in e for e in errors)


def test_validate_entry_flags_answer_that_does_not_cite_any_relevant_article():
    entry = {
        "id": 4,
        "answer": "Câu trả lời này không trích dẫn điều luật nào cả.",
        "relevant_docs": ["80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP"],
        "relevant_articles": ["80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP|Điều 1"],
    }
    errors = validate_entry(entry)
    assert any("does not cite" in e for e in errors)


def test_validate_entry_passes_when_answer_cites_one_of_several_articles():
    entry = {
        "id": 5,
        "answer": "Theo Điều 4 Luật Doanh nghiệp, quy định như sau...",
        "relevant_docs": ["A|Doc A", "B|Doc B"],
        "relevant_articles": ["A|Doc A|Điều 1", "B|Doc B|Điều 4"],
    }
    assert validate_entry(entry) == []


def test_validate_results_detects_missing_ids(tmp_path):
    entries = [build_result_entry(1, [ARTICLE])]
    path = tmp_path / "results.json"
    write_results(entries, str(path))

    errors = validate_results(str(path), expected_count=2)

    assert any("expected 2 entries" in e for e in errors)
    assert any("missing ids" in e for e in errors)


def test_validate_results_passes_complete_valid_set(tmp_path):
    entries = [build_result_entry(1, [ARTICLE]), build_result_entry(2, [ARTICLE])]
    path = tmp_path / "results.json"
    write_results(entries, str(path))

    errors = validate_results(str(path), expected_count=2)

    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_validate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.export'`.

- [ ] **Step 3: Write the implementation**

```python
# src/export.py
import json

from src.answer_gen import generate_answer


def _dedupe_preserve_order(tags: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


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
```

```python
# src/validate.py
import json
import re

ARTICLE_TAG_RE = re.compile(r"^[^|]+\|[^|]+\|(Điều\s+\d+[a-zA-Z]?)$")
REQUIRED_FIELDS = ("id", "answer", "relevant_docs", "relevant_articles")


def _extract_article_label(tag: str) -> str | None:
    match = ARTICLE_TAG_RE.match(tag)
    return match.group(1) if match else None


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

        labels = []
        for tag in entry["relevant_articles"]:
            label = _extract_article_label(tag)
            if label is None:
                errors.append(f"id={entry_id}: malformed relevant_article tag '{tag}'")
            else:
                labels.append(label)

        if labels and "answer" in entry and not any(label in entry["answer"] for label in labels):
            errors.append(
                f"id={entry_id}: answer does not cite any of {labels} from relevant_articles"
            )

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validate.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/export.py src/validate.py tests/test_validate.py
git commit -m "feat: dedupe export tags and validate that answers cite a retrieved article"
```

---

### Task 6: End-to-end pipeline glue + smoke test

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: `BM25Retriever` (Task 2), `get_question_text` (Task 4), `build_result_entry`/`write_results` (Task 5).
- Produces: `run_pipeline(questions_path: str, clean_corpus_path: str, output_path: str, top_k_retrieve: int = 15, top_k_final: int = 5, limit: int | None = None) -> list[dict]`. `scripts/build_submission.py` (Task 7) calls this directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_smoke.py
import json

from src.pipeline import run_pipeline
from src.validate import validate_entry

QUESTIONS = [
    {"id": 1, "question": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi đấu thầu là gì?"},
    {"id": 2, "question": "Hợp đồng lao động xử lý vi phạm kỷ luật như thế nào?"},
]

CORPUS = [
    {
        "relevant_doc_tag": "A|Doc A",
        "relevant_article_tag": "A|Doc A|Điều 1",
        "doc_name": "Doc A",
        "article_id": "Điều 1",
        "text": "Ưu đãi thuế cho doanh nghiệp nhỏ và vừa khi tham gia đấu thầu.",
    },
    {
        "relevant_doc_tag": "B|Doc B",
        "relevant_article_tag": "B|Doc B|Điều 2",
        "doc_name": "Doc B",
        "article_id": "Điều 2",
        "text": "Quy định về hợp đồng lao động và xử lý vi phạm kỷ luật.",
    },
]


def test_run_pipeline_produces_valid_entries(tmp_path):
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(QUESTIONS, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(str(q_path), str(c_path), str(out_path), top_k_retrieve=2, top_k_final=1)

    assert len(entries) == 2
    for entry in entries:
        assert validate_entry(entry) == []
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == entries


def test_run_pipeline_respects_limit(tmp_path):
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps(QUESTIONS, ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    entries = run_pipeline(str(q_path), str(c_path), str(out_path), limit=1)

    assert len(entries) == 1
    assert entries[0]["id"] == 1


def test_run_pipeline_raises_clear_error_on_unexpected_question_schema(tmp_path):
    q_path = tmp_path / "questions.json"
    c_path = tmp_path / "corpus.json"
    out_path = tmp_path / "results.json"
    q_path.write_text(json.dumps([{"id": 1, "content": "schema mismatch"}], ensure_ascii=False), encoding="utf-8")
    c_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")

    try:
        run_pipeline(str(q_path), str(c_path), str(out_path))
        assert False, "expected KeyError for unexpected question schema"
    except KeyError as exc:
        assert "QUESTION_TEXT_KEYS" in str(exc)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_smoke.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline'`.

- [ ] **Step 3: Write the implementation**

```python
# src/pipeline.py
import json

from src.export import build_result_entry, write_results
from src.retrieval import BM25Retriever
from src.schema import get_question_text


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    questions_path: str,
    clean_corpus_path: str,
    output_path: str,
    top_k_retrieve: int = 15,
    top_k_final: int = 5,
    limit: int | None = None,
) -> list[dict]:
    questions = _load_json(questions_path)
    if limit is not None:
        questions = questions[:limit]

    corpus = _load_json(clean_corpus_path)
    retriever = BM25Retriever(corpus)

    entries = []
    for question in questions:
        query = get_question_text(question)
        top_articles = retriever.search(query, top_k=top_k_retrieve)
        entries.append(build_result_entry(question["id"], top_articles, top_k_final=top_k_final))

    write_results(entries, output_path)
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_smoke.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline_smoke.py
git commit -m "feat: end-to-end pipeline glue, using schema adapter instead of hard-coded key"
```

---

### Task 7: Real-data run + zip packaging

**Files:**
- Create: `scripts/build_submission.py`

**Interfaces:**
- Consumes: `preprocess_corpus` (Task 1), `run_pipeline` (Task 6), `validate_results` (Task 5).
- Produces: `submission/results.json` and `submission/submission.zip` on disk.

- [ ] **Step 1: Write the script**

```python
# scripts/build_submission.py
import os
import sys
import zipfile

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
    summary = preprocess_corpus(RAW_CORPUS, CLEAN_CORPUS)  # creates its own output dir
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
```

- [ ] **Step 2: Run it on the real data**

```bash
python scripts/build_submission.py
```

Expected: prints the preprocessing summary, then "Validation passed.", the eyeball reminder, then "Done: D:\MindHacker\submission\submission.zip". If validation fails, read the printed errors — do not proceed until they're empty.

- [ ] **Step 3: Manually confirm the zip is flat**

```bash
python -c "import zipfile; print(zipfile.ZipFile('submission/submission.zip').namelist())"
```

Expected: `['results.json']` — exactly one entry, no folder prefix.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_submission.py
git commit -m "feat: add build_submission.py for end-to-end baseline submission"
```

---

### Task 8: Manual result inspection before using a submission slot

**Why this task exists:** `validate_results` only checks JSON shape, tag format, and that the answer's citation matches one of the retrieved articles — it cannot tell whether BM25 actually retrieved the *right* law. The only way to catch "valid JSON, wrong article" is a human reading a sample.

**Files:**
- Create: `scripts/inspect_results.py`

**Interfaces:**
- Consumes: `get_question_text` (Task 4), `submission/results.json` (Task 7's output), `R2AIStage1DATA.json` (raw questions).
- Produces: console output only — terminal step of this plan, nothing downstream consumes it.

- [ ] **Step 1: Write the script**

```python
# scripts/inspect_results.py
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.schema import get_question_text  # noqa: E402

RESULTS_PATH = os.path.join(ROOT, "submission", "results.json")
QUESTIONS_PATH = os.path.join(ROOT, "R2AIStage1DATA.json")


def main(n: int) -> None:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = {q["id"]: get_question_text(q) for q in json.load(f)}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    for entry in results[:n]:
        print(f"--- id={entry['id']} ---")
        print(f"Q: {questions.get(entry['id'], '<unknown question id>')}")
        print(f"relevant_articles: {entry['relevant_articles']}")
        print(f"answer: {entry['answer'][:300]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a sample of results.json for manual eyeballing.")
    parser.add_argument("--n", type=int, default=20, help="Number of results to print (default 20).")
    args = parser.parse_args()
    main(args.n)
```

- [ ] **Step 2: Run it and actually read the output**

```bash
python scripts/inspect_results.py --n 30
```

Expected: 30 blocks of `id` / question / `relevant_articles` / `answer`. For each one, manually judge: does the retrieved law plausibly relate to the question's SME/legal scenario? If a noticeable fraction look unrelated (e.g. retrieving labor-law articles for a tax question), that's a retrieval quality problem to address in Phase 2, not a format bug — record specific bad cases for later, don't block this baseline on fixing them.

- [ ] **Step 3: Commit**

```bash
git add scripts/inspect_results.py
git commit -m "feat: add manual result inspection script for eyeballing retrieval quality"
```

---

## Tests added or changed in this revision

| File | Change |
|---|---|
| `tests/test_preprocess.py` | Added `test_preprocess_corpus_creates_missing_output_directory`. |
| `tests/test_retrieval.py` | Added `TITLE_MATCH_CORPUS` + `test_search_matches_via_doc_name_when_body_text_does_not` and `test_search_matches_via_article_id`, proving the combined index actually changes ranking vs. body-text-only. |
| `tests/test_schema.py` | New file: `test_get_question_text_returns_value_for_confirmed_key`, `test_get_question_text_raises_clear_error_for_unknown_schema`. |
| `tests/test_validate.py` | `test_build_result_entry_dedupes_relevant_docs_and_relevant_articles` now asserts a **single** deduped `relevant_articles` entry (previously asserted the duplicate was kept — that was the bug). Added `test_validate_entry_flags_answer_that_does_not_cite_any_relevant_article` and `test_validate_entry_passes_when_answer_cites_one_of_several_articles`. |
| `tests/test_pipeline_smoke.py` | Added `test_run_pipeline_raises_clear_error_on_unexpected_question_schema`. |

## Commands to run everything

```bash
pip install -r requirements.txt
pytest -v
python scripts/build_submission.py
python scripts/inspect_results.py --n 30
python -c "import zipfile; print(zipfile.ZipFile('submission/submission.zip').namelist())"
```

## Final checklist before using a Vòng Riêng submission slot

- [ ] `pytest -v` — all tests pass, no skips.
- [ ] `python scripts/build_submission.py` prints "Validation passed." with zero errors.
- [ ] `zipfile.ZipFile('submission/submission.zip').namelist()` returns exactly `['results.json']`.
- [ ] `python scripts/inspect_results.py --n 30` (or higher) has been read end-to-end by a human; retrieved articles look topically plausible for most sampled questions.
- [ ] No dense embeddings, reranker, LLM call, API, UI, Docker, or DB exist anywhere in `src/` or `scripts/` — this is still the Phase-1 baseline.
- [ ] Upload to the leaderboard is a manual action performed by a person outside this plan — nothing here auto-submits.
