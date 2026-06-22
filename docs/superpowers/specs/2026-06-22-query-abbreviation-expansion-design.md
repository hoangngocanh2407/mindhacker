# Phase 4 — Query Abbreviation Expansion (Design)

**Date:** 2026-06-22. **Status:** approved, ready to implement.

## Vấn đề

BM25 là retriever tốt nhất hiện tại (ARTICLES_F2 = 0.1609), nhưng câu hỏi đôi khi dùng viết tắt pháp lý (TNHH, GTGT, BHXH...) trong khi corpus gần như luôn viết dạng đầy đủ ("trách nhiệm hữu hạn", "giá trị gia tăng", "bảo hiểm xã hội"). Khi đó BM25 không khớp được các từ khóa quan trọng nhất của câu hỏi.

Bằng chứng (đo trên dữ liệu thật): `TNHH` 13 lần vs "trách nhiệm hữu hạn" 374 lần trong corpus; `GTGT` 3 vs "giá trị gia tăng" 707; `BHXH` 39 vs "bảo hiểm xã hội" 2042. Có 85/2000 câu hỏi (4.2%) chứa ít nhất một viết tắt pháp lý.

**Kỳ vọng tác động có giới hạn:** chỉ ~4% câu bị ảnh hưởng → trên bộ gold 50 câu, kỳ vọng ~2 câu thay đổi, nhiều khả năng trong vùng nhiễu. Đây là cải tiến đúng nguyên tắc, rủi ro thấp, nhưng không kỳ vọng nhảy điểm lớn. Làm vì nó an toàn (chỉ giúp câu có viết tắt, không hại câu khác) và rẻ.

## Phạm vi

CHỈ abbreviation expansion. KHÔNG synonym expansion (đã loại: rủi ro làm loãng precision vốn đã thấp 0.08). KHÔNG model mới. KHÔNG đổi answer_gen/validate/export/fusion/dense/preprocess. BM25-only vẫn là default; expansion là cờ tùy chọn, mặc định TẮT.

## Thiết kế

### `src/query_processing.py` (mới)
- `LEGAL_ABBREVIATIONS: dict[str, str]` — bảng curated thủ công, key là viết tắt IN HOA, value là dạng đầy đủ. Chỉ gồm viết tắt pháp lý rõ ràng quan sát từ dữ liệu: TNHH, TNDN, GTGT, TNCN, BHXH, BHYT, BHTN, DNNVV, SHTT, UBND, GCN, ĐKKD. Loại bỏ viết tắt không pháp lý (AI, KOL, IELTS) và viết tắt mơ hồ.
- `expand_query(query: str, abbreviations: dict[str, str] = LEGAL_ABBREVIATIONS) -> str`:
  - Với mỗi viết tắt xuất hiện trong query (match whole-word, đúng dạng IN HOA qua `re.search(r"\b"+abbr+r"\b", query)`), **append** dạng đầy đủ vào cuối query (không replace — giữ cả viết tắt gốc).
  - Mỗi dạng đầy đủ chỉ append một lần kể cả viết tắt xuất hiện nhiều lần.
  - Thứ tự append theo thứ tự xuất hiện trong dict (ổn định, dễ test).
  - Query không có viết tắt nào → trả về nguyên văn.

### `src/pipeline.py`
- `run_pipeline()` thêm tham số `expand_abbreviations: bool = False`.
- Sau khi build `query_texts`, nếu bật thì `query_texts = [expand_query(t) for t in query_texts]` — áp tại MỘT điểm, trước khi đưa vào cả BM25 và dense.
- Mặc định False → hành vi cũ không đổi, test cũ không phá.

### `scripts/build_submission.py`
- Thêm cờ `--expand-query` (store_true, default False), dùng được với mọi `--retriever`.
- Truyền `expand_abbreviations=args.expand_query` vào `run_pipeline()`.
- `run_meta.json` ghi `"expand_query": args.expand_query`.

## Tests (`tests/test_query_processing.py` + bổ sung pipeline)
- `expand_query` append dạng đầy đủ cho viết tắt biết.
- Query không viết tắt → giữ nguyên.
- Không khớp substring: "TNHHX" hoặc "tnhh" thường không bị mở rộng (word-boundary + đúng IN HOA).
- Nhiều viết tắt trong 1 câu → append nhiều dạng đầy đủ.
- Viết tắt lặp lại → dạng đầy đủ chỉ append 1 lần.
- `run_pipeline(..., expand_abbreviations=True)` chạy được, entry hợp lệ; mặc định False giữ hành vi cũ.

## Verification
- `pytest -v` toàn bộ pass.
- `build_submission.py --retriever bm25` (không cờ) — không đổi so với trước (regression check).
- `build_submission.py --retriever bm25 --expand-query` — chạy xong, `run_meta.json` ghi `expand_query: true`, zip flat. So sánh số câu đổi kết quả vs bản không expand (kỳ vọng ~vài chục–trăm câu có viết tắt bị ảnh hưởng).
- Người dùng nộp bản `--expand-query` để A/B với BM25 thường (việc nộp do người dùng tự làm).
