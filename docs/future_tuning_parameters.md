# Future Tuning Parameters — VLegalQA

Tracking doc for knobs we've identified but deliberately **not** changing right now, so each pipeline change stays small and isolated. Add to this list instead of tuning ad-hoc mid-implementation. When picking up one of these, measure on the internal eval set (once it exists) before burning a real submission.

Context for current numbers:
- Submission #989 (2026-06-22), BM25-only, `top_k_retrieve=15`, `top_k_final=5`: `ARTICLES_F2MACRO=0.1609` (`precision=0.086`, `recall=0.23`), `DOCS_F2MACRO=0.1693`.
- Hybrid submission (2026-06-22), BM25+dense RRF, same cutoffs, `rrf_k=60`: `ARTICLES_F2MACRO=0.1461` (worse), `DOCS_F2MACRO=0.186` (better). Root cause analysis in [docs/reports/phase2_report.md](reports/phase2_report.md): equal-weight RRF lets dense's noisy article-level ranking displace BM25's correct article-level picks, even though dense agrees with BM25 fairly often at the document level. **Hybrid is not the default submission mode** as a result — see Phase 2 report for the full evidence (side-by-side BM25/dense/fusion top-5 comparisons on real questions).

## Retrieval cutoffs — MEASURED (top_k_final sweep, BM25, leaderboard)

`--top-k-final` flag added; each value writes to its own `submission/bm25_kf<N>/`.

| top_k_final | ARTICLES_F2 | precision | recall | DOCS_F2 |
|---|---|---|---|---|
| 1 | (built `bm25_kf1`, chưa nộp) | | | |
| 2 | 0.1308 | 0.130 | 0.1383 | 0.1799 |
| **3 ← CHỐT** | **0.1669** | 0.120 | 0.200 | 0.1656 |
| 5 (default cũ) | 0.1609 | 0.086 | 0.230 | 0.1693 |
| 10 | 0.1352 | 0.0515 | 0.2667 | 0.1503 |

**kf3 là tối ưu** (peak rõ): 5→3 tăng (recall đổi lấy precision có lợi), nhưng 3→2 thì recall sụp (0.20→0.138, mất điều ở hạng 3) → F2 giảm mạnh. **Default đã đổi 5→3.** (Lưu ý: DOCS_F2 lại peak ở kf2=0.1799 nhưng ARTICLES là metric ưu tiên → chốt theo ARTICLES = kf3.)

**Kết luận đo được:**
- **Giảm cutoff cải thiện F2** (5→3: 0.1609→0.1669) vì gold set mỗi câu rất ít điều (~1-2): cắt bớt slot cứu precision (0.086→0.12) nhiều hơn mất recall (0.23→0.20). Xu hướng còn lên khi đi xuống → đang thử kf2, kf1.
- Tăng 5→10 thì F2 giảm (precision sụp, recall chỉ +0.037 vì slot 6-10 hầu như không chứa gold).
- **Recall thấp & gần phẳng dù mở rộng cutoff** → điều luật đúng không nằm trong top-10 cho ~73% câu → **trần là chất lượng RANKING của BM25, không phải cutoff.**
- `top_k_retrieve=15` không phải nút thắt.

**Cảnh báo nhiễu:** chênh lệch 0.1609 vs 0.1669 nhỏ, đo trên gold 50 câu → có phần nhiễu, nhưng P/R dịch chuyển nhất quán theo cơ chế nên hướng (giảm cutoff) là thật.

**Hệ quả:** tuning cutoff cho gain nhỏ-thật, gần cạn. Để nâng trần đáng kể phải cải thiện ranking quality (Tokenization tiếng Việt cho BM25 + Backlog reranker cần GPU).

## Fusion (RRF)

- **Weighted RRF — IMPLEMENTED (Phase 3).** `reciprocal_rank_fusion(..., weights=[...])` + `run_pipeline(..., dense_weight=...)` + `build_submission.py --dense-weight <float>`. BM25 fixed at 1.0, dense adjustable; default 1.0 = old equal-weight behavior. Confirmed root cause from Phase 2 (equal-weight RRF trusts dense's noisy article-level ranking as much as BM25). Tune the actual value (try 0.3–0.5) by submitting to leaderboard — no internal eval set (cut). See [reports/phase3_report.md](reports/phase3_report.md).
- **`rrf_k`** (currently 60, the standard default) — controls how quickly rank position decays. Lower `k` rewards top ranks more aggressively; higher `k` flattens. Worth trying 10–100 in combination with dense_weight, but second-order vs the weight itself.
- **Document-level expansion instead of article-level fusion** — alternative architecture: use BM25 to pick the right document(s) first, then only let dense re-rank/expand *within* those documents' articles, instead of letting dense freely inject articles from documents BM25 didn't surface. Untried, bigger change than weighted RRF.
- **Chunked dense embeddings** — see Chunking section / Backlog in roadmap; embedding at sub-article level might sharpen dense's article-level discrimination. Untried, needs article_id remapping.

## Tokenization (BM25)

- **Vietnamese word segmentation — IMPLEMENTED.** `src/retrieval.segment_tokenize` (pyvi `ViTokenizer`) joins compounds with underscores ("doanh nghiệp" → "doanh_nghiệp") so multi-word legal terms become single BM25 tokens. Enable via `run_pipeline(use_segmentation=True)` / `build_submission.py --segment`. BM25 only (dense unaffected). Default off. De-risked: pyvi installs cleanly on Win/Py3.13 (wheel, no compiler), segments full corpus in ~0.6 min. **Effect is broad: changes top-3 on 1829/2000 questions (91%) vs the plain tokenizer** — high potential to move the score either way; A/B on leaderboard (`bm25_kf3_seg` built, awaiting submission).
- Stopword removal — not done. Could reduce noise but risks dropping legally meaningful short words. Untried.
- BM25 hyperparameters (k1, b) — untried. With extreme doc-length variance (51 to 245k chars), length-normalization `b` could matter.

## Chunking

- Long articles (some up to ~245k chars) are currently NOT chunked — each "Điều" stays one retrieval unit. For BM25 this is fine (no context limit). For the dense embedding model added in Phase 2, `sentence-transformers` silently truncates anything past the model's max sequence length, so very long articles are only partially represented in their embedding. If eval shows dense retrieval underperforming specifically on long-article questions, revisit chunking (with `article_id` remapping back to the parent article on output) — deferred for now because it adds real implementation complexity (chunk-to-article remapping in export) for an unconfirmed payoff.

## Embedding model choice

- Currently using `bkai-foundation-models/vietnamese-bi-encoder` (chosen for: open-source, well under the 14B param cap, released well before 2026-03-01, prior benchmarks on Vietnamese legal retrieval). Alternatives considered but not used yet: `AITeamVN/Vietnamese_Embedding` (longer context, 2048 tokens, but need to re-verify exact release date against the cutoff before using), `truro7/vn-law-embedding`. Worth A/B once eval set exists.

## Answer generation

- Still template-based (Phase 1/2 scope is retrieval-only improvements). Swapping in an actual LLM (e.g. Qwen3 ≤14B) for answer synthesis is a separate, larger change — not a tuning knob, tracked as its own future phase rather than here.
