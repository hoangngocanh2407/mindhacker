# Phase 1 Report — BM25 Baseline Pipeline

**Status:** Done, submitted to leaderboard (vòng công khai), result analyzed below.
**Date range:** 2026-06-21 → 2026-06-22.

## Mục tiêu Phase 1

Một pipeline chạy được hết 2000 câu hỏi, sinh `submission/results.json` hợp lệ format + đúng quy chế trích dẫn, không dùng dense embedding/reranker/LLM/API/UI/Docker/DB — chỉ BM25 + answer template.

## Module đã viết

| File | Vai trò |
|---|---|
| `src/preprocess.py` | `load_corpus`, `dedupe_articles`, `drop_unknown`, `preprocess_corpus` |
| `src/retrieval.py` | `tokenize`, `BM25Retriever` |
| `src/answer_gen.py` | `generate_answer` — ghép template trích dẫn "Điều X" |
| `src/export.py` | `build_result_entry`, `write_results` |
| `src/validate.py` | `validate_entry`, `validate_results` |
| `src/pipeline.py` | `get_question_text`, `run_pipeline` |
| `scripts/build_submission.py` | CLI end-to-end: preprocess → pipeline → validate → zip |
| `scripts/inspect_results.py` | CLI in mẫu kết quả để eyeball thủ công |

## Các fix quan trọng đã áp dụng trong quá trình làm

1. **Validator kiểm citation thật** — `validate_entry()` trích article id (phần cuối sau `|` của mỗi tag trong `relevant_articles`) và bắt lỗi nếu `answer` không chứa bất kỳ id nào trong đó (`"answer does not cite any relevant article"`). Trước fix này, validator chỉ kiểm field tồn tại, không kiểm nội dung trích dẫn có khớp không.
2. **Export dedupe cả hai list** — `build_result_entry()` dedupe `relevant_docs` **và** `relevant_articles` (ban đầu chỉ dedupe `relevant_docs`), tránh cùng một điều luật xuất hiện 2 lần trong 1 kết quả.
3. **BM25 index mở rộng** — không chỉ index `text`, mà index `doc_name + article_id + text` (qua `searchable_text()`), để câu hỏi nhắc tên luật/số điều vẫn match được dù body text không lặp lại từ đó.
4. **Preprocess tự tạo thư mục output** — `preprocess_corpus()` gọi `Path(output_path).parent.mkdir(parents=True, exist_ok=True)`, tránh `FileNotFoundError` khi chạy lần đầu trên máy sạch chưa có `data/processed/`.
5. **Adapter đọc câu hỏi linh hoạt** — `get_question_text()` thử lần lượt các key `"question"/"query"/"question_text"/"content"`, raise `ValueError` rõ ràng nếu không tìm thấy, tránh hard-code mù `question["question"]`.
6. **Fix UTF-8 stdout cho Windows** — cả `build_submission.py` và `inspect_results.py` gọi `sys.stdout.reconfigure(encoding="utf-8")` ở đầu file. Phát hiện vì console Windows mặc định `cp1252` không encode được tiếng Việt, làm script crash giữa lúc in kết quả.

## Số liệu tiền xử lý corpus thật

Chạy trên `vbpl_dat.json` (4755 bản ghi gốc):

```
{"total": 4755, "after_dedupe": 4341, "dropped_unknown": 65, "final": 4276}
```

- 414 bản ghi trùng `relevant_article_tag` bị loại (nhiều hơn ước tính ban đầu ~333 — chưa re-verify, xem rủi ro bên dưới).
- 65 bản ghi `doc_id == "UNKNOWN"` bị loại, log lại ở `data/processed/corpus_clean.json.dropped.json` (chưa được review tay).

## Test

5 file test (`test_preprocess.py`, `test_retrieval.py`, `test_answer_gen.py`, `test_validate.py`, `test_pipeline_smoke.py`), **30/30 pass** tại thời điểm chốt Phase 1 (sau đó Phase 2 thêm test mới, tổng hiện tại 37/37 — xem [phase2_report.md](phase2_report.md)).

## Kết quả nộp bài thật

Nộp lúc 2026-06-22 11:41 (vòng công khai, không phải Vòng Riêng — 10 lượt/ngày), ID 989:

```json
{"ARTICLES_F2MACRO": 0.1609, "DOCS_F2MACRO": 0.1693,
 "ARTICLES_PRECISION": 0.086, "ARTICLES_RECALL": 0.23,
 "DOCS_PRECISION": 0.1307, "DOCS_RECALL": 0.2067,
 "CHINH_XAC_NOI_DUNG": 0.0, "DAY_DU": 0.0, "THUC_TIEN": 0.0, "RO_RANG": 0.0}
```

**Đọc kết quả:**
- IR (điều luật) F2 = 0.16, recall chỉ 23% — BM25 thuần bỏ sót phần lớn điều luật đúng, đúng như dự đoán cho baseline chưa tối ưu.
- `CHINH_XAC_NOI_DUNG` (nhóm QA chấm tự động ngay) = 0.0 vì trích dẫn của answer khớp với *retrieval của chính mình* (đã pass `validate_entry`), nhưng retrieval đó phần lớn sai so với gold thật → trích dẫn sai luật.
- 3 nhóm QA còn lại = 0.0 vì chưa "đẩy" (promote) bài này để LLM-judge chấm — không phản ánh chất lượng thật.
- Cảnh báo `gold=50 pred=2000` trên log web: bộ gold dùng chấm điểm công khai hiện chỉ có 50 câu có nhãn, không phải toàn bộ 2000 — điểm số trên là ước lượng nhiễu trên mẫu nhỏ, không phải điểm cuối cùng.

## Rủi ro/việc còn thiếu khi chốt Phase 1

- Chưa eyeball hệ thống 50-100 kết quả (mới xem rải rác vài chục).
- Chưa review 65 bản ghi UNKNOWN bị drop — có thể có điều luật quan trọng bị loại oan.
- Số liệu dedupe thật (414) khác ước tính ban đầu (~333) — chưa re-verify nguyên nhân lệch.
- Tokenizer BM25 là regex thuần, chưa word-segmentation tiếng Việt.
