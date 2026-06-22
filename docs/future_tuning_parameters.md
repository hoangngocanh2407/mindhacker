# Future Tuning Parameters — VLegalQA

Tracking doc for knobs we've identified but deliberately **not** changing right now, so each pipeline change stays small and isolated. Add to this list instead of tuning ad-hoc mid-implementation. When picking up one of these, measure on the internal eval set (once it exists) before burning a real submission.

Context for current numbers:
- Submission #989 (2026-06-22), BM25-only, `top_k_retrieve=15`, `top_k_final=5`: `ARTICLES_F2MACRO=0.1609` (`precision=0.086`, `recall=0.23`), `DOCS_F2MACRO=0.1693`.
- Hybrid submission (2026-06-22), BM25+dense RRF, same cutoffs, `rrf_k=60`: `ARTICLES_F2MACRO=0.1461` (worse), `DOCS_F2MACRO=0.186` (better). Root cause analysis in [docs/reports/phase2_report.md](reports/phase2_report.md): equal-weight RRF lets dense's noisy article-level ranking displace BM25's correct article-level picks, even though dense agrees with BM25 fairly often at the document level. **Hybrid is not the default submission mode** as a result — see Phase 2 report for the full evidence (side-by-side BM25/dense/fusion top-5 comparisons on real questions).

## Retrieval cutoffs

- **`top_k_retrieve`** (currently 15) — how many candidates BM25/dense each return before fusion/truncation. Recall 23% suggests this might already be excluding correct articles before they ever reach the final list. Try 20–30.
- **`top_k_final`** (currently 5) — how many articles actually go into `relevant_docs`/`relevant_articles` per question. Since IR is scored with F2 (recall weighted 2x precision), raising this to 8–10 trades precision for recall — worth a sweep once there's an eval set to measure the trade-off instead of guessing.
- Note these two interact — sweep them together, not independently, since a wider `top_k_retrieve` only helps if `top_k_final` doesn't immediately cut it back down.

## Fusion (RRF)

- **Weighted RRF — IMPLEMENTED (Phase 3).** `reciprocal_rank_fusion(..., weights=[...])` + `run_pipeline(..., dense_weight=...)` + `build_submission.py --dense-weight <float>`. BM25 fixed at 1.0, dense adjustable; default 1.0 = old equal-weight behavior. Confirmed root cause from Phase 2 (equal-weight RRF trusts dense's noisy article-level ranking as much as BM25). Tune the actual value (try 0.3–0.5) by submitting to leaderboard — no internal eval set (cut). See [reports/phase3_report.md](reports/phase3_report.md).
- **`rrf_k`** (currently 60, the standard default) — controls how quickly rank position decays. Lower `k` rewards top ranks more aggressively; higher `k` flattens. Worth trying 10–100 in combination with dense_weight, but second-order vs the weight itself.
- **Document-level expansion instead of article-level fusion** — alternative architecture: use BM25 to pick the right document(s) first, then only let dense re-rank/expand *within* those documents' articles, instead of letting dense freely inject articles from documents BM25 didn't surface. Untried, bigger change than weighted RRF.
- **Chunked dense embeddings** — see Chunking section / Backlog in roadmap; embedding at sub-article level might sharpen dense's article-level discrimination. Untried, needs article_id remapping.

## Tokenization (BM25)

- Current tokenizer is plain regex (`\w+` + lowercase), no Vietnamese word segmentation. Legal compound terms like "doanh nghiệp nhỏ và vừa" are tokenized word-by-word, not as a phrase — a real Vietnamese segmenter (e.g. `pyvi`, `underthesea`) could meaningfully change BM25 term-matching behavior. Untested whether it helps or hurts on this domain.
- Stopword removal — not done. Could reduce noise in BM25 scoring but risks dropping legally meaningful short words (e.g. "và", "hoặc" rarely matter, but legal connector words sometimes do).

## Chunking

- Long articles (some up to ~245k chars) are currently NOT chunked — each "Điều" stays one retrieval unit. For BM25 this is fine (no context limit). For the dense embedding model added in Phase 2, `sentence-transformers` silently truncates anything past the model's max sequence length, so very long articles are only partially represented in their embedding. If eval shows dense retrieval underperforming specifically on long-article questions, revisit chunking (with `article_id` remapping back to the parent article on output) — deferred for now because it adds real implementation complexity (chunk-to-article remapping in export) for an unconfirmed payoff.

## Embedding model choice

- Currently using `bkai-foundation-models/vietnamese-bi-encoder` (chosen for: open-source, well under the 14B param cap, released well before 2026-03-01, prior benchmarks on Vietnamese legal retrieval). Alternatives considered but not used yet: `AITeamVN/Vietnamese_Embedding` (longer context, 2048 tokens, but need to re-verify exact release date against the cutoff before using), `truro7/vn-law-embedding`. Worth A/B once eval set exists.

## Answer generation

- Still template-based (Phase 1/2 scope is retrieval-only improvements). Swapping in an actual LLM (e.g. Qwen3 ≤14B) for answer synthesis is a separate, larger change — not a tuning knob, tracked as its own future phase rather than here.
