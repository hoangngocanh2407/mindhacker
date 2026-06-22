# VLegalQA Baseline Pipeline — As-Built Plan (Revision 2)

> **Archived reference, frozen.** This file is no longer updated. For current phase status and the overall roadmap, see [vlegalqa-roadmap.md](vlegalqa-roadmap.md). For a condensed summary of what this file describes, see [docs/reports/phase1_report.md](../../reports/phase1_report.md).

> **Status: implemented.** This revision documents the pipeline as it actually exists in `src/` and `scripts/`, after a round of small fixes applied directly to the code (not a rewrite). Earlier revisions of this plan described an upcoming `src/schema.py` module and a stricter regex-based citation check — neither matches what was actually built. This document replaces those descriptions with the real module layout and behavior so the plan stays a trustworthy reference.

**Goal:** A locally-runnable Phase-1 baseline for the VLegalQA SME legal-assistant competition: BM25 retrieval over the law corpus + template-based answer generation, producing a format-and-citation-validated `submission/results.json` zipped flat into `submission/submission.zip`.

**Architecture:** Linear pipeline, no ML model beyond BM25: clean the law corpus (dedupe + drop unusable records, auto-creating its own output directory) → BM25 keyword retrieval per question over a combined `doc_name + article_id + text` index → template-based answer that cites the retrieved article(s) → export with deduped tag lists → validate format **and** citation correctness → manually eyeball a sample → zip flat.

**Out of scope (unchanged from prior revisions, still true of the current code):** dense/embedding retrieval, cross-encoder reranking, LLM-based answer generation, any HTTP API, UI, Docker, or database. Nothing in `src/` or `scripts/` calls out to a model or network service.

**Tech Stack:** Python 3.13, `rank-bm25` for BM25, `pytest` for tests, stdlib `json`/`zipfile`/`re`/`pathlib`/`sys` for everything else.

## Global Constraints

- Output file is named exactly `results.json`, one entry per question, fields `id`, `answer`, `relevant_docs`, `relevant_articles`.
- Every string in `relevant_docs`/`relevant_articles` is reused verbatim from the corpus's `relevant_doc_tag`/`relevant_article_tag` fields — never reconstructed.
- Neither `relevant_docs` nor `relevant_articles` contains the same tag twice within one entry (enforced by export-time dedupe, checked by the test suite, not by the validator — see Task 5 below).
- `answer` must contain the article-id portion (e.g. `"Điều 1"`) of at least one tag in that entry's `relevant_articles` — enforced by `validate_entry()`.
- `results.json` covers all question ids with no duplicates and no gaps (checked against `expected_count` by `validate_results()`).
- `submission.zip` contains `results.json` flat at the zip root.
- Corpus rows with `doc_id == "UNKNOWN"` are excluded from the retrievable corpus and logged to a `.dropped.json` sidecar file, not silently discarded.
- Rows with identical `relevant_article_tag` are deduplicated before indexing.
- The question file's text field is resolved through an adapter, not a hard-coded key — see Task 4 below.
- No dense embeddings, reranker, LLM calls, API, UI, Docker, or DB anywhere in this codebase.

---

## File Structure (actual)

```
D:\MindHacker\
  requirements.txt              # rank-bm25>=0.2.2, pytest>=7.4
  pytest.ini                    # pythonpath = .
  .gitignore                    # broadened beyond the original plan — now also covers
                                 # build/dist artifacts, venvs, .env files, lint/type-check
                                 # caches, Jupyter checkpoints, editor/OS files, *.log, and
                                 # explicitly ignores the two raw data files
                                 # (vbpl_dat.json, R2AIStage1DATA.json) in addition to
                                 # data/processed/ and submission/
  src/
    __init__.py
    preprocess.py                # load_corpus, dedupe_articles, drop_unknown, preprocess_corpus
    retrieval.py                 # tokenize, _searchable_text, BM25Retriever
    answer_gen.py                # generate_answer
    export.py                    # _dedupe_preserve_order, build_result_entry, write_results
    validate.py                  # _article_id_from_tag, validate_entry, validate_results
    pipeline.py                  # QUESTION_TEXT_KEYS, get_question_text, run_pipeline
  scripts/
    build_submission.py          # CLI: real data end-to-end -> submission.zip
    inspect_results.py           # CLI: print N sample results for manual eyeballing
  tests/
    __init__.py
    test_preprocess.py
    test_retrieval.py
    test_answer_gen.py
    test_validate.py             # covers both src/export.py and src/validate.py
    test_pipeline_smoke.py       # also covers get_question_text() directly
  data/processed/                # generated: corpus_clean.json + corpus_clean.json.dropped.json
  submission/                    # generated: results.json, submission.zip
```

There is **no `src/schema.py`** and **no `tests/test_schema.py`** — an earlier plan revision proposed putting the question-text adapter in its own module; in the actual implementation `get_question_text()` lives directly in `src/pipeline.py` next to its only caller, `run_pipeline()`.

---

## Module-by-module description

### `src/preprocess.py`

`load_corpus(path) -> list[dict]`, `dedupe_articles(records) -> list[dict]`, `drop_unknown(records) -> tuple[list[dict], list[dict]]`, `preprocess_corpus(raw_path, output_path) -> dict` returning `{"total", "after_dedupe", "dropped_unknown", "final"}`.

**Fix applied:** `preprocess_corpus()` now creates its output directory before writing — `Path(output_path).parent.mkdir(parents=True, exist_ok=True)` — and does the same for the `.dropped.json` sidecar path. Before this fix, running the script against a fresh checkout with no `data/processed/` directory yet would raise `FileNotFoundError`. Covered by `test_preprocess_corpus_creates_missing_nested_output_directory`, which writes to a doubly-nested non-existent directory and asserts both output files land correctly.

### `src/retrieval.py`

`tokenize(text) -> list[str]` (regex `\w+` lowercase split), `BM25Retriever(corpus)` with `.search(query, top_k=15) -> list[dict]` (each result is a shallow copy of the matching corpus record plus a `"score"` float).

**Fix applied:** the BM25 index is no longer built from `record["text"]` alone. A private helper, `_searchable_text(record)`, concatenates `doc_name + " " + article_id + " " + text` (using `.get(..., "")` so a record missing any of those keys doesn't crash indexing), and that combined string is what gets tokenized into the index. This means a question that names a law or decree by title, or by article number, can match even when the article body itself doesn't repeat those words. Covered by `test_search_matches_via_doc_name_when_body_text_does_not_contain_query` and `test_search_matches_via_article_id_when_body_text_does_not_contain_it`, both of which use corpora where the body text deliberately omits the query terms.

`tokenize()` carries an explicit code comment marking it a Phase-1 baseline tokenizer — lowercase + `\w+` only, no Vietnamese word segmentation, stemming, or stopword removal — so it isn't mistaken for a tuned component later.

### `src/answer_gen.py`

`generate_answer(top_articles, max_articles=5) -> str`. Unchanged from earlier revisions: builds `"Theo {article_id} {doc_name}: {snippet}"` per article (snippet truncated to 400 chars), joined with spaces, or a fixed "không tìm thấy điều luật liên quan" sentence if `top_articles` is empty.

### `src/export.py`

`build_result_entry(question_id, top_articles, top_k_final=5) -> dict`, `write_results(entries, output_path) -> None`.

**Fix applied:** a private `_dedupe_preserve_order(items)` helper is now applied to **both** `relevant_docs` and `relevant_articles` (the earlier revision only deduped `relevant_docs`). If the same article appears twice in `top_articles` — e.g. BM25 returning a near-duplicate or the same article surviving a future rerank step twice — the output entry collapses it to one occurrence in both lists, preserving first-seen order. Covered by `test_build_result_entry_dedupes_relevant_docs_and_relevant_articles`, which passes `[ARTICLE, ARTICLE]` in and asserts both output lists have length 1.

### `src/validate.py`

`validate_entry(entry) -> list[str]`, `validate_results(path, expected_count=2000) -> list[str]`.

**Fix applied:** `validate_entry()` now checks that `answer` cites at least one of the entry's own `relevant_articles`. The article id is extracted with a private `_article_id_from_tag(tag)` helper — `tag.split("|")`, and if there are at least 3 parts, the last part (stripped) is the article id; otherwise the tag is flagged as malformed via `ARTICLE_TAG_RE` (a loose `^[^|]+\|[^|]+\|.+$` shape check, not a strict `"Điều N"` regex). If none of the extracted article ids appear as a substring of `answer`, the entry gets exactly the error string `"id=<id>: answer does not cite any relevant article"`. This is a plain substring check, not a regex match on `answer` — it does not verify the citation is grammatically a proper "Điều N" reference, only that the same text appears. Covered by `test_validate_entry_flags_answer_that_does_not_cite_any_relevant_article` and `test_validate_entry_passes_when_answer_cites_relevant_article`.

`validate_results()` is unchanged: loads the file, checks total count against `expected_count`, runs `validate_entry()` on every entry, flags duplicate ids, and reports any missing ids from `1..expected_count`.

### `src/pipeline.py`

`QUESTION_TEXT_KEYS = ("question", "query", "question_text", "content")`, `get_question_text(record) -> str`, `run_pipeline(questions_path, clean_corpus_path, output_path, top_k_retrieve=15, top_k_final=5, limit=None) -> list[dict]`.

**Fix applied:** `run_pipeline()` no longer indexes `question["question"]` directly. `get_question_text(record)` tries each key in `QUESTION_TEXT_KEYS` in order and returns the first **truthy** value found (so an empty string at a matching key is treated as not found and falls through to the next key). If no key yields a truthy value, it raises `ValueError(f"Cannot find question text field in record id={record.get('id')}")`. Against the real `R2AIStage1DATA.json` the matching key is `"question"` (confirmed by inspection: every record is `{"id": int, "question": str}`, 2000 records total) — the adapter exists so that assumption is enforced with a clear error rather than relied on silently. Covered by `test_get_question_text_supports_question_key`, `test_get_question_text_supports_query_key`, `test_get_question_text_raises_value_error_when_no_known_key`, `test_run_pipeline_produces_valid_entries_with_query_key`, and `test_run_pipeline_raises_value_error_on_unrecognized_question_schema`.

### `scripts/build_submission.py`

Runs `preprocess_corpus` → `run_pipeline` (top_k_retrieve=15, top_k_final=5, all 2000 questions) → `validate_results` (exits non-zero with the first 20 errors printed if validation fails) → zips `results.json` flat into `submission.zip`.

**Fix applied:** the very first lines after the imports now call `sys.stdout.reconfigure(encoding="utf-8")` (guarded by `hasattr(sys.stdout, "reconfigure")`). Without this, running the script in a default Windows console (codepage `cp1252`) raises `UnicodeEncodeError` as soon as it tries to print anything containing Vietnamese diacritics — which happens routinely, since corpus summaries and validation error messages both contain Vietnamese doc names.

### `scripts/inspect_results.py`

Reads `submission/results.json`, prints the first N entries (default 20, overridable via a positional CLI argument, e.g. `python scripts/inspect_results.py 50`), each showing `id`, an `answer` snippet (first 500 chars), and `relevant_articles`. Prints an explicit reminder that `validate_results()` only checks format and citation presence, not whether the retrieved law is actually correct for the question.

**Fix applied:** same `sys.stdout.reconfigure(encoding="utf-8")` guard as `build_submission.py`, for the same reason — this script's whole purpose is printing Vietnamese legal text, so it hit the `cp1252` crash on every run before the fix.

---

## Tests (current full list)

| File | Covers |
|---|---|
| `tests/test_preprocess.py` | dedupe, UNKNOWN-dropping, summary counts, auto-mkdir for nested output paths, raw load |
| `tests/test_retrieval.py` | tokenizer behavior, ranking by body text, score key presence, ranking via doc_name only, ranking via article_id only |
| `tests/test_answer_gen.py` | citation presence, empty-list fallback sentence, long-text truncation, `max_articles` cap |
| `tests/test_validate.py` | export dedupe (both lists), all `validate_entry` error paths including the citation check, `validate_results` count/missing-id checks |
| `tests/test_pipeline_smoke.py` | `get_question_text` for `"question"` and `"query"` keys and its `ValueError` on unknown schema, `run_pipeline` end-to-end with both key variants, `ValueError` propagation through `run_pipeline`, `limit` parameter |

All 30 tests passed as of the last run against this code.

## Commands to run everything

```bash
pip install -r requirements.txt
pytest -v
python scripts/build_submission.py
python scripts/inspect_results.py 30
python -c "import zipfile; print(zipfile.ZipFile('submission/submission.zip').namelist())"
```

Last real run against `vbpl_dat.json` / `R2AIStage1DATA.json` produced: `{"total": 4755, "after_dedupe": 4341, "dropped_unknown": 65, "final": 4276}`, a `submission/results.json` that passed `validate_results(expected_count=2000)` with zero errors, and `submission.zip` containing exactly `['results.json']`.

## Remaining risks / not yet done

- **Only a handful of results have been eyeballed.** `scripts/inspect_results.py` has been run on a small sample, not a systematic one. Before using one of the 5 Vòng Riêng submission slots, run `python scripts/inspect_results.py 50` (or 100) and read through it — judge whether the retrieved articles are plausibly on-topic for each question. The validator cannot catch "valid JSON, cites a real article, wrong law."
- **65 corpus records were dropped for `doc_id == "UNKNOWN"`** — noticeably more than the "at least one" estimated during initial data analysis. These are logged in `data/processed/corpus_clean.json.dropped.json` but have not been individually reviewed. Some may be law articles worth keeping under a corrected `doc_id` rather than excluding outright — worth a manual pass before treating the corpus as final, since dropping a relevant article straight away costs recall.
- **The 4755 → 4341 dedupe count (414 duplicates removed) differs from the ~333 pairs estimated earlier.** Not necessarily a bug — the original estimate was a rough read of the raw data — but it hasn't been independently re-verified against the dedupe logic, so treat the corpus size as provisional until someone double-checks a sample of the 414.
- **BM25 over `doc_name + article_id + text` with a bare regex tokenizer is still a Phase-1 baseline**, not tuned retrieval. No stopword removal, no Vietnamese-specific segmentation (e.g. compound legal terms like "doanh nghiệp nhỏ và vừa" are tokenized word-by-word, not as a phrase), no query expansion, and no rerank step. Retrieval quality should be expected to plateau well below what a Phase 2 system would reach.
- **No dense/embedding retrieval, no cross-encoder reranker, no LLM-based answer generation exist in this codebase**, by design for Phase 1. The answers are template-filled article excerpts, not synthesized prose — sufficient for the automated "cites a correct article" grading criterion, but not optimized for the four QA criteria currently scored at 0.0 (clarity, completeness, etc.). Any of these would be a separate, explicitly out-of-scope follow-up plan, not a patch to this one.
