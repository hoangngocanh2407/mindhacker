# Chạy reranker trên Kaggle GPU — hướng dẫn

Mục tiêu: chạy cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) trên GPU miễn phí của Kaggle để phá trần ARTICLES_F2 (~0.167) mà máy CPU local không làm nổi. Output là `submission.zip` hợp lệ tải về để nộp.

## Vì sao reranker (nhắc lại lý do kỹ thuật)

BM25 chỉ đưa điều luật đúng vào top-3 cho ~20% câu (recall 0.20). Nhưng điều đúng thường nằm đâu đó trong top-50 của BM25. Reranker chấm lại từng cặp (câu hỏi, điều luật) bằng mô hình mạnh hơn nhiều và **kéo điều đúng từ hạng sâu lên top-3** — đó là cách recall (và F2) tăng. Reranker chỉ promote được điều **có trong candidate pool**, nên pool lấy rộng (BM25 top-50, ∪ dense) để nâng trần recall.

## Các bước

### 1. Tạo Kaggle Dataset chứa repo
- Nén thư mục dự án thành 1 file zip **bao gồm**: `src/`, `vbpl_dat.json`, `R2AIStage1DATA.json`. (Không cần `tests/`, `submission/`, `docs/`.)
- Kaggle → **Datasets → New Dataset** → upload zip đó. Đặt tên sao cho slug là `vlegalqa-repo` (hoặc tên khác, nhớ sửa `REPO_DIR` trong script).
- Sau khi tạo, dữ liệu sẽ nằm ở `/kaggle/input/vlegalqa-repo/` (kiểm tra cấu trúc: phải thấy `/kaggle/input/vlegalqa-repo/src/...`, `/kaggle/input/vlegalqa-repo/vbpl_dat.json`). Nếu Kaggle giải nén tạo thêm 1 cấp thư mục con, sửa `REPO_DIR` cho khớp.

### 2. Tạo Notebook, bật GPU + Internet
- Kaggle → **Code → New Notebook**.
- Panel bên phải:
  - **Accelerator** → chọn **GPU** (T4 hoặc P100).
  - **Internet** → **On** (cần để tải model reranker từ Hugging Face).
- **Add Input** → thêm Dataset `vlegalqa-repo` vừa tạo.

### 3. Chạy
- Dán toàn bộ nội dung [notebooks/kaggle_rerank.py](../notebooks/kaggle_rerank.py) vào một cell, chạy.
- Kiểm tra `REPO_DIR` ở đầu script khớp với đường dẫn input thật.
- Tuỳ chọn ở đầu script:
  - `USE_DENSE=True` — gộp candidate từ dense để tăng recall pool (khuyến nghị bật).
  - `TOP_K_RETRIEVE=50` — độ rộng pool; tăng (75/100) nếu muốn trần recall cao hơn, đổi lại chậm hơn.
  - `TOP_K_FINAL=3` — cutoff tốt nhất đã xác nhận trên leaderboard.
- Thời gian: rerank 2000 câu × ~50-100 candidate trên T4 thường vài phút đến ~30 phút tuỳ pool.

### 4. Tải kết quả về và nộp
- Script ghi `submission.zip` (và `results.json`) vào `/kaggle/working/`.
- Tải `submission.zip` từ panel **Output** của notebook.
- Nộp lên leaderboard, so ARTICLES_F2 với kf3 hiện tại (**0.1669**).

## Kỳ vọng & cách đọc kết quả

- Reranker là lever DUY NHẤT còn lại có khả năng vượt 0.167 một cách đáng kể. Nếu nó vẫn không vượt → trần bị chặn bởi **candidate recall** (điều đúng không có trong BM25 top-50); khi đó thử tăng `TOP_K_RETRIEVE`, hoặc kết luận corpus/BM25 không đủ surface điều đúng.
- Ghi lại cho working notes: model reranker (tên, ngày phát hành 2024, ~568M params — tuân thủ quy chế ≤14B + trước 1/3/2026), `TOP_K_RETRIEVE`, `USE_DENSE`, điểm leaderboard.

## Tuân thủ quy chế

`BAAI/bge-reranker-v2-m3`: mã nguồn mở, ~568M tham số (< 14B), phát hành 2024 (trước 1/3/2026). Dense `bkai-foundation-models/vietnamese-bi-encoder` đã dùng từ Phase 2 cũng hợp lệ. Cả hai chạy trên Kaggle GPU không vi phạm ràng buộc mô hình (vẫn là mô hình mở, tự tải trọng số).
