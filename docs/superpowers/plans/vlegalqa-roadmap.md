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
| 3 | Weighted RRF — hạ trọng số dense để bớt nhiễu cấp điều luật (`--dense-weight`) | ✅ Done + đã nộp — **KHÔNG cải thiện ARTICLES_F2, BM25-only vẫn tốt nhất (0.1609)**. Code giữ lại, dừng tune dense weight |
| 4 | Query processing — abbreviation expansion (`--expand-query`) | ✅ Done + nộp A/B — **trung tính trên gold-50 (điểm = BM25), không hại**. Giữ bật cho bài cuối. Synonym expansion đã loại |
| 5 | Answer generation bằng LLM mở (≤14B) + answer validation chống hallucination | ⬜ Chưa làm |
| — | ~~Bộ eval nội bộ (hand-label)~~ | ❌ CUT — không có thời gian/khả năng gán nhãn tay; đo qua leaderboard thay thế |
| — | Cross-encoder reranker, chunking điều luật dài | 🔜 Backlog — cần GPU mạnh hơn (máy hiện MX450 2GB + torch CPU-only) |

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

44/44 test pass sau patch hardening; sau khi thêm `batch_search()` tối ưu tốc độ dense (không đổi logic ranking/RRF/output) là 50/50.

---

## Phase 3 — Weighted RRF (CODE DONE, CHƯA NỘP)

**Mục tiêu:** Sửa đúng root cause Phase 2 (dense nhiễu cấp điều luật, RRF cho dense trọng số ngang BM25) bằng cách thêm trọng số riêng cho dense trong fusion, hạ ảnh hưởng của dense xuống. Rẻ nhất, không model mới, chạy gần như tức thì (embedding đã cache).

**Báo cáo chi tiết:** [docs/reports/phase3_report.md](../reports/phase3_report.md)

**Đã làm:**
- `src/fusion.py::reciprocal_rank_fusion()` thêm tham số `weights` (mặc định None = trọng số ngang, không đổi hành vi cũ).
- `run_pipeline()` thêm `dense_weight=1.0` (BM25 cố định 1.0).
- `scripts/build_submission.py --retriever hybrid --dense-weight <float>` (mặc định 1.0), `run_meta.json` ghi lại `dense_weight`.
- 55/55 test pass. Chạy thật: w=0.3 đổi ranking ở 1822/2000 câu so với w=1.0; eyeball xác nhận đẩy được điều lạc đề từ dense (vd Luật Cạnh tranh) ra khỏi top-5 cho câu hỏi SME.

**Cách đo:** nộp `submission/hybrid/submission.zip` với vài giá trị `--dense-weight` (1.0, 0.5, 0.3...) lên leaderboard, so điểm với baseline BM25 0.1609. Việc nộp do người dùng tự làm.

---

## Phase 4 — Query Abbreviation Expansion (CODE DONE, CHƯA NỘP)

**Mục tiêu:** Mở rộng viết tắt pháp lý trong query (TNHH → "trách nhiệm hữu hạn"...) để BM25 khớp được, vì corpus gần như luôn viết dạng đầy đủ.

**Báo cáo chi tiết:** [docs/reports/phase4_report.md](../reports/phase4_report.md). **Spec:** [specs/2026-06-22-query-abbreviation-expansion-design.md](../specs/2026-06-22-query-abbreviation-expansion-design.md).

**Đã làm:** `src/query_processing.py` (`expand_query` + dict 10 viết tắt curated), `run_pipeline(expand_abbreviations=...)`, cờ `--expand-query`. 63/63 test pass. Chạy thật: 55/2000 câu đổi top-5, toàn bộ đều có viết tắt (an toàn, không chạm câu khác).

**Đã loại synonym expansion** — rủi ro loãng precision (đang 0.08).

**Cách đo:** nộp bản `--expand-query` A/B với BM25 thường. Lưu ý lever nhỏ (~4% câu) → kỳ vọng trong vùng nhiễu trên gold 50 câu.

**Phụ thuộc:** không, độc lập.

---

## Phase 5 — LLM Answer Generation + Validation (CHƯA LÀM)

**Mục tiêu:** Thay `generate_answer()` template bằng LLM mở ≤14B sinh câu trả lời rõ ràng/đầy đủ hơn từ top articles, vẫn bắt buộc trích đúng "Điều X" và **chống hallucination** (không cho LLM bịa luật ngoài context). Nhắm cải thiện 4 nhóm QA hiện đang 0.0.

**Phạm vi dự kiến:**
1. Chọn model (ứng viên: Qwen3 4B/8B) — tự xác nhận ngày phát hành trên HuggingFace (trước 1/3/2026), ≤14B, license cho phép. **Lưu ý GPU:** máy hiện MX450 2GB không chạy nổi LLM cỡ này — cần cân nhắc chạy trên máy/GPU khác hoặc quantization, xác định trước khi bắt đầu.
2. Prompt ép buộc: chỉ dùng điều luật được cấp, trích "Điều X" đúng văn bản, nói rõ giới hạn nếu không đủ căn cứ.
3. Thay lời gọi `generate_answer()` trong `build_result_entry()`, giữ nguyên interface (trả về string).
4. Answer validation: `validate_entry()` citation check không đổi phải tiếp tục pass; thêm kiểm tra answer không trích điều luật ngoài `relevant_articles` (chống bịa).

**Phụ thuộc:** nên làm sau khi retrieval đã ổn (Phase 3/4), vì câu trả lời tốt chỉ có nghĩa khi retrieval đúng.

**Lưu ý quy chế:** ghi rõ nguồn, link HuggingFace, ngày phát hành model — bắt buộc cho working notes (Phase 6).

---

## Phase 6 — Chiến lược nộp bài cuối + Working Notes (CHƯA LÀM)

**Mục tiêu:** Chốt bài nộp tốt nhất trước hạn 2026-06-30 23:59, và hoàn thiện working notes paper bắt buộc của cuộc thi.

**Phạm vi dự kiến:**
1. Theo dõi quy định "đẩy" (promote) bài để được LLM-judge chấm QA định kỳ — xác nhận lại quy định này áp dụng thế nào ở vòng công khai (khác Vòng Riêng).
2. Checklist cuối trước mỗi lần nộp thật: `pytest -v` pass, `validate_results()` 0 lỗi, `inspect_results.py` đã eyeball mẫu, zip flat đúng `['results.json']`.
3. Viết working notes: nguồn/phiên bản văn bản luật ngoài `vbpl_dat.json` (nếu có bổ sung), tên đầy đủ + link HuggingFace + ngày phát hành của mọi model dùng (embedding Phase 2, LLM Phase 4), mô tả pipeline đủ chi tiết để tái lập.
4. Quyết định bài nộp nào là bài "đẩy" chính thức.

**Phụ thuộc:** cần Phase 5 hoàn thành (hoặc quyết định dừng sớm hơn nếu hết thời gian) trước khi chốt bài nộp cuối.

---

## Cross-encoder reranker — CODE DONE, chạy trên Kaggle GPU (chưa nộp)

Lý do: mọi lever retrieval rẻ trên CPU đã cạn (trần ARTICLES_F2 ~0.1669 = BM25 kf3). Reranker là lever duy nhất còn khả năng phá trần. Máy local (MX450 2GB, torch CPU-only) không chạy nổi ~80k+ cặp → chuyển sang **Kaggle GPU miễn phí**.

**Đã làm:** `src/reranker.py` (`Reranker` bọc `BAAI/bge-reranker-v2-m3`, model inject được để test), `run_pipeline(reranker=...)` — candidate pool BM25 top-K ∪ dense, dedupe, rerank, lấy top-3. 80/80 test pass (stub, không cần GPU). Notebook + hướng dẫn: [notebooks/kaggle_rerank.py](../../notebooks/kaggle_rerank.py), [docs/kaggle_rerank_instructions.md](../kaggle_rerank_instructions.md).

**Bước tiếp:** người dùng chạy notebook trên Kaggle (BM25 top-50 ∪ dense → rerank → top-3), tải `submission.zip`, nộp so với kf3 (0.1669). Trần thật khi đó bị chặn bởi candidate recall (điều đúng có trong BM25 top-50 không) — nếu reranker không vượt, tăng `TOP_K_RETRIEVE` hoặc kết luận BM25 không surface đủ.

## Backlog (để sau)
- **Chunking điều luật dài:** chia điều luật dài (tới ~245k ký tự) thành đoạn nhỏ để dense/reranker không chỉ "đọc" phần đầu. Phụ thuộc quyết định reranker. Tốn công remap chunk → article gốc khi export.

---

## Quy ước tài liệu (để không lặp lại vấn đề lần này)

- **File này (`vlegalqa-roadmap.md`)**: kế hoạch tổng, chỉ sửa khi đổi phạm vi/trạng thái phase, không viết chi tiết code/fix vào đây.
- **`docs/reports/phaseN_report.md`**: báo cáo cố định cho phase đã xong — viết một lần khi phase đó hoàn thành, không sửa lại trừ khi phát hiện sai sót cần đính chính.
- **`docs/future_tuning_parameters.md`**: danh sách tham số sẽ tune, cập nhật liên tục khi phát sinh thêm tham số mới.
- **`docs/superpowers/plans/2026-06-21-vlegalqa-baseline-pipeline.md`**: file lịch sử — bản as-built chi tiết của Phase 1 code, giữ nguyên không sửa thêm, nội dung đã được rút gọn vào `phase1_report.md`.
