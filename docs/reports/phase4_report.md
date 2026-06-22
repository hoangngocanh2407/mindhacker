# Phase 4 Report — Query Abbreviation Expansion

**Status:** Code hoàn chỉnh, đã nộp A/B. **Kết quả: trung tính trên gold-50 (điểm giống hệt BM25), không hại — giữ làm tùy chọn bật cho bài nộp cuối.**
**Date:** 2026-06-22.

## Kết luận (TL;DR)

`--expand-query` cho điểm **giống hệt baseline BM25 từng con số** (ARTICLES_F2 0.1609, DOCS_F2 0.1693). Đúng dự báo: lever chỉ chạm ~4% câu (~1-2 câu trong gold 50) → vô hình với eval nhỏ. **Không làm tệ đi** (xác nhận an toàn). Trên full 2000 nó đổi 55 câu thật nhưng không đo được. Quyết định: **giữ `--expand-query` như tùy chọn, nên bật cho bài nộp cuối** (chỉ có thể giúp câu viết tắt trong test ẩn, chứng minh không hại) — nhưng không phải lever xoay chuyển điểm.

| Cấu hình | ARTICLES_F2 | DOCS_F2 |
|---|---|---|
| BM25 plain | 0.1609 | 0.1693 |
| BM25 + `--expand-query` | 0.1609 | 0.1693 (giống hệt) |

## Mục tiêu

BM25 là retriever tốt nhất (ARTICLES_F2 = 0.1609). Câu hỏi đôi khi dùng viết tắt pháp lý (TNHH, GTGT, BHXH...) còn corpus gần như luôn viết dạng đầy đủ → BM25 không khớp được từ khóa chính. Mở rộng viết tắt trong query (append dạng đầy đủ) để BM25 khớp được.

## Thiết kế chốt

Spec: [docs/superpowers/specs/2026-06-22-query-abbreviation-expansion-design.md](../superpowers/specs/2026-06-22-query-abbreviation-expansion-design.md). Chỉ abbreviation expansion (bỏ synonym expansion vì rủi ro loãng precision vốn đã thấp 0.08).

## Module mới / sửa

| File | Thay đổi |
|---|---|
| `src/query_processing.py` (mới) | `LEGAL_ABBREVIATIONS` (10 viết tắt curated: TNHH, TNDN, TNCN, GTGT, BHXH, BHYT, BHTN, DNNVV, SHTT, UBND) + `expand_query()` — **append** dạng đầy đủ (không replace), match whole-word đúng IN HOA, mỗi dạng đầy đủ append tối đa 1 lần. |
| `src/pipeline.py` | `run_pipeline()` thêm `expand_abbreviations: bool = False`; khi bật, áp `expand_query()` lên `query_texts` tại 1 điểm trước khi đưa vào cả BM25 và dense. Mặc định False = không đổi hành vi. |
| `scripts/build_submission.py` | Cờ `--expand-query` (store_true, default off), dùng được mọi `--retriever`; `run_meta.json` ghi `expand_query`. |

Không đổi: answer_gen, validate, export, fusion, dense_retrieval, preprocess. BM25-only vẫn default, expansion mặc định TẮT.

## Curate dict — bằng chứng dữ liệu

Mỗi viết tắt được chọn vì dạng đầy đủ phổ biến trong corpus (mismatch thật):

| Viết tắt | full trong corpus | abbr trong corpus |
|---|---|---|
| TNHH | "trách nhiệm hữu hạn" 374 | 13 |
| GTGT | "giá trị gia tăng" 707 | 3 |
| TNDN | "thu nhập doanh nghiệp" 239 | 16 |
| BHXH | "bảo hiểm xã hội" 2042 | 39 |
| BHYT | "bảo hiểm y tế" 2117 | — |
| BHTN | "bảo hiểm thất nghiệp" 719 | — |
| SHTT | "sở hữu trí tuệ" 352 | — |

Loại bỏ AI/KOL/IELTS (không pháp lý) và CPTPP/PPP (corpus cũng dùng dạng tắt, expansion không chắc giúp).

## Test

`tests/test_query_processing.py` (7 test): append đúng, giữ nguyên câu không viết tắt, không khớp substring ("TNHHX"), không khớp lowercase, nhiều viết tắt, không lặp dạng đầy đủ, dict keys IN HOA. `tests/test_pipeline_smoke.py` +1: `run_pipeline(expand_abbreviations=True)` khớp được qua dạng đầy đủ khi corpus chỉ có full form. **Tổng 63/63 test pass.**

## Kết quả chạy thật (bm25, có vs không expand)

| Run | expand_query | Elapsed | Validation |
|---|---|---|---|
| `--retriever bm25` | false | 356.3s | passed, zip flat |
| `--retriever bm25 --expand-query` | true | 267.9s | passed, zip flat |

**Tác động đúng & phẫu thuật:** so top-5 hai bản → **55/2000 câu đổi kết quả**, và **toàn bộ 55 câu đó đều chứa viết tắt** (55/55) — expansion không bao giờ chạm câu không có viết tắt. (63 câu chứa viết tắt; 8 câu còn lại không đổi top-5 vì BM25 đã khớp sẵn.) Đúng hành vi an toàn như thiết kế.

## Việc còn lại

- **Chưa nộp leaderboard.** Người dùng nộp `submission/bm25/submission.zip` (đang là bản `--expand-query`) để A/B với BM25 thường (0.1609). **Lưu ý:** chỉ ~3% câu của bộ gold 50 kỳ vọng bị ảnh hưởng (~1-2 câu) → rất có thể trong vùng nhiễu, đừng kỳ vọng nhảy điểm; nếu điểm không tệ đi thì có thể giữ vì nó đúng nguyên tắc và miễn phí.
- Dict 10 viết tắt có thể mở rộng thêm nếu phát hiện viết tắt khác trong câu hỏi — nhưng lợi ích biên giảm dần.
