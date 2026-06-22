# Phase 3 Report — Weighted RRF

**Status:** Code hoàn chỉnh, đã nộp leaderboard, **kết luận: weighted RRF KHÔNG cải thiện ARTICLES_F2 — BM25-only vẫn tốt nhất.** Giả thuyết đã được kiểm chứng và bác bỏ bằng dữ liệu thật. Thay thế Phase 3 cũ (eval set nội bộ — đã CUT vì không có thời gian/khả năng gán nhãn tay).
**Date:** 2026-06-22.

## Kết luận (TL;DR)

Hạ trọng số dense trong RRF KHÔNG kéo được `ARTICLES_F2MACRO` lên trên baseline BM25-only. Mọi mức dense weight > 0 đều làm ARTICLES_F2 tệ hơn hoặc bằng BM25-only. **Quyết định: giữ BM25-only làm cấu hình nộp cho ARTICLES; dừng tune dense weight (ngõ cụt cho mục tiêu chính).** Code weighted RRF giữ lại (đã test sạch, không hại gì, có thể hữu ích nếu sau này ưu tiên DOCS_F2).

| Cấu hình | ARTICLES_F2 | DOCS_F2 | Ghi chú |
|---|---|---|---|
| BM25-only (= hybrid w=0.0) | **0.1609** | 0.1693 | Tốt nhất cho ARTICLES |
| hybrid w=0.3 | 0.1419 | 0.1729 | Tệ hơn BM25 |
| hybrid w=1.0 (Phase 2) | 0.1461 | **0.186** | Tốt nhất cho DOCS, tệ cho ARTICLES |

**Kiểm chứng code đúng (không bug):** chạy hybrid `--dense-weight 0.0` rồi so `relevant_articles` với BM25-only → **giống hệt 2000/2000 câu**. Đường fusion chính xác như thiết kế; kết quả trên là thực nghiệm thật, không phải lỗi. Chênh lệch w=0.3 < w=1.0 nhiều khả năng là nhiễu của bộ gold 50 câu, nhưng tín hiệu tổng thể (dense không giúp ARTICLES) là nhất quán.

## Mục tiêu

Sửa đúng root cause đã xác định ở Phase 2: hybrid RRF cho dense trọng số ngang BM25, mà dense nhiễu ở cấp điều luật cụ thể (đúng văn bản nhưng sai điều), nên nhiễu của dense đẩy bật điều đúng của BM25 ra khỏi top → `ARTICLES_F2MACRO` giảm 0.1609 → 0.1461. Giải pháp: cho phép hạ trọng số dense trong RRF (BM25 cố định 1.0).

Đây là phương án **rẻ nhất đánh đúng root cause** — không model mới, không dependency mới, chạy gần như tức thì (embedding corpus đã cache từ Phase 2). Chọn thay cho cross-encoder reranker vì máy hiện chỉ có GPU MX450 2GB + PyTorch CPU-only, rerank ~80k cặp là không khả thi (đã chuyển reranker vào backlog).

## Thay đổi code

| File | Thay đổi |
|---|---|
| `src/fusion.py` | `reciprocal_rank_fusion()` thêm tham số `weights: list[float] \| None = None`. `None` = trọng số 1.0 đều (y hệt hành vi cũ). Công thức điểm đổi từ `1/(k+rank)` thành `weight * 1/(k+rank)`. Raise `ValueError` nếu `len(weights) != len(result_lists)`. |
| `src/pipeline.py` | `run_pipeline()` thêm `dense_weight: float = 1.0`, truyền `weights=[1.0, dense_weight]` vào fusion. BM25 cố định 1.0. |
| `scripts/build_submission.py` | Thêm cờ `--dense-weight` (default 1.0). Truyền vào pipeline. `run_meta.json` ghi `dense_weight` (chỉ với hybrid, bm25 = null). |

Không đụng: `answer_gen.py`, `validate.py`, `export.py`, `dense_retrieval.py`, `preprocess.py`, `embedding_cache.py`. Default submission vẫn là `bm25`.

## Test

`tests/test_fusion.py` thêm 4 test, `tests/test_pipeline_smoke.py` thêm 1 test:
- `test_fusion_weight_zero_for_dense_excludes_dense_only_article` — weights=[1.0, 0.0]: điều chỉ có ở dense bị điểm 0, xếp dưới điều của BM25.
- `test_fusion_weight_flips_winner_between_sources` — weights=[0.0, 1.0]: điều của dense thắng.
- `test_fusion_weights_none_matches_explicit_equal_weights` — `None` cho kết quả y hệt `[1.0, 1.0]` (chứng minh backward-compatible).
- `test_fusion_raises_when_weights_length_mismatches`.
- `test_run_pipeline_accepts_dense_weight_and_produces_valid_entries` — pipeline nhận `dense_weight` và xuất entry hợp lệ.

**55/55 test pass** (50 cũ + 5 mới). 5 test fusion cũ vẫn pass → `weights=None` không đổi hành vi.

## Kết quả chạy thật

| Run | Cache | Elapsed | Ghi chú |
|---|---|---|---|
| `--retriever bm25` (default) | n/a | 91.4s | Không đổi, zip flat `['results.json']`, `dense_weight: null` trong meta |
| `--retriever hybrid --dense-weight 1.0` | HIT | 267.0s | Bằng hành vi Phase 2 (weights ngang) |
| `--retriever hybrid --dense-weight 0.3` | HIT | 247.4s | Cấu hình mới |

**Bằng chứng tham số có tác dụng:** so sánh `relevant_articles` giữa w=1.0 và w=0.3 trên toàn bộ 2000 câu → **1822/2000 câu đổi kết quả top-5**. Eyeball câu 2 ("ưu đãi khi tham gia đấu thầu") — vốn là ví dụ root cause Phase 2 nơi dense kéo `Luật Cạnh tranh 23/2018/QH14` vào top-5: với w=0.3, Luật Cạnh tranh đã bị đẩy ra, top-5 giờ toàn `80/2021/NĐ-CP` (nhóm SME) + Luật Đầu tư — đúng kỳ vọng (nghiêng về BM25 hơn).

## Hệ quả cho hướng đi tiếp

- **BM25-only là cấu hình nộp tốt nhất hiện tại** (ARTICLES_F2 = 0.1609). Đây vẫn là default của `build_submission.py`, không cần đổi gì.
- **Dừng tune dense weight** — đã chứng minh là ngõ cụt cho ARTICLES_F2. Không nộp thêm biến thể dense weight nữa (lãng phí).
- Dense chỉ hơn ở DOCS_F2 (w=1.0). Nếu sau này biết công thức xếp hạng chung của BTC ưu tiên DOCS, có thể cân nhắc lại; còn hiện tại ARTICLES là metric quan trọng hơn (gắn citation QA) nên ưu tiên BM25.
- **Lever tiếp theo đáng làm: cải thiện chính BM25** — vì đó là retriever đang thắng. Cụ thể là Phase 4 (query processing: chuẩn hóa viết tắt/synonym) tác động trực tiếp lên chất lượng match của BM25, recall hiện mới 20%.
- Không có eval set nội bộ (đã CUT) → đo qua leaderboard (gold 50 câu, nhiễu). Vì nhiễu cao, chỉ nên nộp khi có thay đổi đủ lớn về bản chất, không dò vi mô.
