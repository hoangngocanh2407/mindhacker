# VLegalQA — Roadmap tổng (Phase 1 → Phase 5)

> **Quy ước từ nay:** file này là **kế hoạch tổng**, chỉ cập nhật trạng thái (status) và phạm vi từng phase — **không** viết chi tiết implementation hay biến thành "as-built" report nữa. Chi tiết những gì đã làm thật trong mỗi phase nằm ở file báo cáo riêng trong `docs/reports/`. Lý do tách: trước đây file plan duy nhất bị ghi đè 2 lần thành as-built report, làm mất luôn bức tranh tổng thể các phase tiếp theo — không lặp lại lỗi đó.

**Bối cảnh cuộc thi:** VLegalQA — trợ lý pháp lý AI cho SME. Vòng công khai: nộp tối đa 10 lần/ngày. Hôm nay 2026-06-22, hạn đóng cổng 2026-06-30 23:59 (giờ VN) — còn 8 ngày. Ràng buộc model: mã nguồn mở, tải tự do, <14B tham số, phát hành trước 1/3/2026.

**Bản phân tích đề bài gốc:** [Tong_hop_va_de_xuat_huong_lam_VLegalQA.md](../Tong_hop_va_de_xuat_huong_lam_VLegalQA.md)
**Tracking các tham số sẽ tune:** [future_tuning_parameters.md](../future_tuning_parameters.md)

---

## Tổng quan trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 1 | BM25 baseline + export/validate + zip | ✅ Done, đã nộp thật (F2=0.1609) |
| 2 | Dense embedding retrieval + RRF fusion với BM25, hardening (cache, mode tách rời) | ✅ Done, đã nộp thật — hybrid tệ hơn BM25 ở ARTICLES, **BM25-only là default**, hybrid giữ làm experimental |
| 3 | Bộ eval nội bộ (hand-label) + sweep tham số tuning (rrf weight, top_k...) | ⏸ Deferred — ưu tiên hoàn thiện pipeline vận hành trước |
| 4 | Answer generation bằng LLM mở (≤14B) thay template | ⬜ Chưa làm |
| 5 | Chiến lược nộp bài cuối + working notes | ⬜ Chưa làm |

---

## Phase 1 — BM25 Baseline (DONE)

**Mục tiêu:** Pipeline chạy hết 2000 câu hỏi, xuất `results.json` đúng format + đúng quy chế trích dẫn, dùng BM25 thuần + answer template. Không dense/LLM/API/UI/Docker/DB.

**Báo cáo chi tiết:** [docs/reports/phase1_report.md](../reports/phase1_report.md)

**Kết quả tóm tắt:** Đã nộp (#989, 2026-06-22), `ARTICLES_F2MACRO=0.1609`, recall 23%, precision 8.6%. Đây là điểm baseline để so sánh các phase sau.

---

## Phase 2 — Dense Retrieval + RRF Fusion + Hardening (DONE)

**Mục tiêu:** Thêm dense embedding (`bkai-foundation-models/vietnamese-bi-encoder`) fuse với BM25 qua Reciprocal Rank Fusion, để tăng recall/precision so với BM25 thuần. Answer generation vẫn giữ template (chưa đổi).

**Báo cáo chi tiết:** [docs/reports/phase2_report.md](../reports/phase2_report.md)

**Kết quả tóm tắt:** Đã nộp thật bản hybrid lên leaderboard — **kết quả tệ hơn BM25-only ở cấp điều luật** (`ARTICLES_F2MACRO` 0.1609→0.1461), tốt hơn nhẹ ở cấp văn bản (`DOCS_F2MACRO` 0.1693→0.186). Root cause đã điều tra rõ (dense nhiễu ở cấp điều luật, RRF trọng số ngang nhau để nhiễu đó đẩy bật điều đúng của BM25) — xem báo cáo chi tiết.

**Đã quyết định và patch xong (không nộp mò thêm khi chưa có eval set):**
- BM25-only là **default submission mode** (`scripts/build_submission.py` không cờ = bm25).
- Hybrid giữ lại trong code làm **experimental mode**, chỉ chạy khi gọi `--retriever hybrid`, fail rõ ràng nếu lỗi (không fallback ngầm).
- Thêm dense embedding cache (`data/processed/dense_embeddings.npy` + `dense_meta.json`) — giảm thời gian chạy hybrid từ ~27 phút (cache miss) xuống ~6.6 phút (cache hit).
- Output tách theo mode: `submission/bm25/` và `submission/hybrid/`, mỗi thư mục có `run_meta.json`.
- `scripts/inspect_results.py` nhận `--results`/`--limit`, in thêm `question`.

44/44 test pass sau patch.

---

## Phase 3 — Bộ eval nội bộ + Tuning (DEFERRED — để sau)

> **Tạm để sau theo yêu cầu 2026-06-22:** ưu tiên hiện tại là hoàn thiện pipeline vận hành sạch (BM25 default, hybrid experimental, cache, output tách mode — đã xong ở Phase 2). Eval set quay lại làm khi pipeline đã ổn định và cần tune tham số (rrf weight, top_k...) một cách có đo lường, thay vì dò trên leaderboard.

**Mục tiêu:** Đo được P/R/F2 trên máy local mà không cần nộp bài, để sweep các tham số trong [future_tuning_parameters.md](../future_tuning_parameters.md) (top_k_retrieve, top_k_final, rrf_k, weighted fusion, tokenizer...) trước khi quyết định cấu hình nào đáng nộp thật.

**Phạm vi dự kiến (sẽ viết implementation plan riêng khi bắt đầu):**
1. Lấy mẫu ~50–80 câu từ `R2AIStage1DATA.json`, ưu tiên câu có vẻ map rõ tới 1-2 văn bản cụ thể (ví dụ nhắc rõ "Luật Hỗ trợ DNNVV", "Nghị định 80/2021/NĐ-CP"...).
2. Gán nhãn tay `relevant_articles` đúng cho từng câu (cần đọc thật văn bản luật trong `vbpl_dat.json` để xác nhận) — đây là việc cần người làm, không tự động hoá hoàn toàn được.
3. Viết script tính P/R/F2-macro đúng công thức quy chế (F2 = 5PR/(4P+R)) trên bộ nhãn này, nhận một `results.json` (hoặc list kết quả) làm input.
4. Dùng script đó để so sánh: Phase 1 (BM25-only) vs Phase 2 (BM25+dense) vs các biến thể tham số khác nhau.
5. Chốt cấu hình tốt nhất theo eval nội bộ trước khi nộp thật lên leaderboard.

**Phụ thuộc:** không phụ thuộc code Phase 1/2 thay đổi gì, chỉ cần `run_pipeline()` đã có (đã có từ Phase 1/2).

**Rủi ro đã biết:** gán nhãn tay 50-80 câu cần thời gian đọc luật, không thể làm hộ hoàn toàn nếu không có chuyên môn pháp lý xác nhận — cần bạn tham gia trực tiếp ở bước này.

---

## Phase 4 — Answer Generation bằng LLM mở (CHƯA LÀM)

**Mục tiêu:** Thay `generate_answer()` template bằng LLM mở ≤14B sinh câu trả lời tự nhiên hơn, vẫn bắt buộc bám sát điều luật retrieval được và trích đúng "Điều X" (để không phá `validate_entry()` hiện có). Nhắm tới cải thiện 4 nhóm QA hiện đang 0.0 (đặc biệt 3 nhóm LLM-judge: đầy đủ, thực tiễn, rõ ràng).

**Phạm vi dự kiến:**
1. Chọn model cụ thể (ứng viên: dòng Qwen3 4B/8B) — phải tự xác nhận lại ngày phát hành chính xác trên HuggingFace trước khi chốt dùng, đảm bảo trước 1/3/2026 và ≤14B tham số, license cho phép.
2. Viết prompt ép buộc: chỉ dùng thông tin từ điều luật được cấp, trích "Điều X" đúng văn bản, từ chối/nói rõ giới hạn nếu không có điều luật đủ liên quan.
3. Thay thế lời gọi `generate_answer()` trong `src/export.py`/`build_result_entry()` bằng lời gọi LLM, giữ nguyên interface (vẫn trả về string).
4. Re-run `validate_entry()`/`validate_results()` không đổi — phải tiếp tục pass citation check.
5. Đo lại trên bộ eval nội bộ (Phase 3) cả về citation accuracy và (nếu có thể tự đánh giá thô) độ rõ ràng/đầy đủ.

**Phụ thuộc:** cần Phase 3 (eval set) tồn tại để đo hiệu quả, không bắt buộc nhưng nên làm sau Phase 3.

**Lưu ý quy chế:** ghi rõ nguồn, link HuggingFace, ngày phát hành của model dùng — bắt buộc cho working notes (Phase 5).

---

## Phase 5 — Chiến lược nộp bài cuối + Working Notes (CHƯA LÀM)

**Mục tiêu:** Chốt bài nộp tốt nhất trước hạn 2026-06-30 23:59, và hoàn thiện working notes paper bắt buộc của cuộc thi.

**Phạm vi dự kiến:**
1. Theo dõi quy định "đẩy" (promote) bài để được LLM-judge chấm QA định kỳ — xác nhận lại quy định này áp dụng thế nào ở vòng công khai (khác Vòng Riêng).
2. Checklist cuối trước mỗi lần nộp thật: `pytest -v` pass, `validate_results()` 0 lỗi, `inspect_results.py` đã eyeball mẫu, zip flat đúng `['results.json']`.
3. Viết working notes: nguồn/phiên bản văn bản luật ngoài `vbpl_dat.json` (nếu có bổ sung), tên đầy đủ + link HuggingFace + ngày phát hành của mọi model dùng (embedding Phase 2, LLM Phase 4), mô tả pipeline đủ chi tiết để tái lập.
4. Quyết định bài nộp nào là bài "đẩy" chính thức.

**Phụ thuộc:** cần Phase 4 hoàn thành (hoặc quyết định dừng ở Phase 2/3 nếu hết thời gian) trước khi chốt bài nộp cuối.

---

## Quy ước tài liệu (để không lặp lại vấn đề lần này)

- **File này (`vlegalqa-roadmap.md`)**: kế hoạch tổng, chỉ sửa khi đổi phạm vi/trạng thái phase, không viết chi tiết code/fix vào đây.
- **`docs/reports/phaseN_report.md`**: báo cáo cố định cho phase đã xong — viết một lần khi phase đó hoàn thành, không sửa lại trừ khi phát hiện sai sót cần đính chính.
- **`docs/future_tuning_parameters.md`**: danh sách tham số sẽ tune, cập nhật liên tục khi phát sinh thêm tham số mới.
- **`docs/superpowers/plans/2026-06-21-vlegalqa-baseline-pipeline.md`**: file lịch sử — bản as-built chi tiết của Phase 1 code, giữ nguyên không sửa thêm, nội dung đã được rút gọn vào `phase1_report.md`.
