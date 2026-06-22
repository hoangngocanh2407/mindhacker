# Chạy reranker trên Kaggle GPU — hướng dẫn

Mục tiêu: chạy cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) trên GPU miễn phí của Kaggle để phá trần ARTICLES_F2 (~0.167) mà máy CPU local không làm nổi. Output là `submission.zip` hợp lệ tải về để nộp.

## Vì sao reranker (nhắc lại lý do kỹ thuật)

BM25 chỉ đưa điều luật đúng vào top-3 cho ~20% câu (recall 0.20). Nhưng điều đúng thường nằm đâu đó trong top-50 của BM25. Reranker chấm lại từng cặp (câu hỏi, điều luật) bằng mô hình mạnh hơn nhiều và **kéo điều đúng từ hạng sâu lên top-3** — đó là cách recall (và F2) tăng. Reranker chỉ promote được điều **có trong candidate pool**, nên pool lấy rộng (BM25 top-50, ∪ dense) để nâng trần recall.

## Các bước

Workflow: **code lấy từ GitHub (git clone mỗi lần chạy → luôn mới nhất)**, **data lấy từ Kaggle Dataset** (vì file data bị gitignore, không nằm trong repo). Setup token + dataset 1 lần, sau đó mỗi lần đổi code chỉ cần chạy lại notebook (không re-upload gì).

### 1. Tạo GitHub Personal Access Token (PAT) — 1 lần
Vì repo `hoangngocanh2407/mindhacker` để **private**, Kaggle cần token để clone.
- GitHub → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
- **Repository access**: chỉ chọn repo `mindhacker`. **Permissions → Contents: Read-only** là đủ.
- Tạo xong, **copy token** (chỉ hiện 1 lần).

### 2. Lưu token vào Kaggle Secrets — 1 lần
- Trong Kaggle Notebook → menu **Add-ons → Secrets** → **Add a new secret**.
- **Label**: `GITHUB_PAT` (đúng tên này, khớp `GITHUB_PAT_SECRET` trong script). **Value**: dán token.
- Bật (attach) secret cho notebook.

### 3. Tạo Kaggle Dataset chứa DỮ LIỆU — 1 lần
- Nén **chỉ 2 file**: `vbpl_dat.json`, `R2AIStage1DATA.json`.
- Kaggle → **Datasets → New Dataset** → upload. Ghi nhớ đường dẫn mount, ví dụ `/kaggle/input/vlegalqa-data/`.
- Sửa `DATA_DIR` trong script cho khớp đường dẫn thật (kiểm tra ở panel **Input**; nếu Kaggle lồng thêm thư mục con thì trỏ vào đúng cấp chứa 2 file json).

### 4. Tạo Notebook, bật GPU + Internet
- Kaggle → **Code → New Notebook**.
- Panel phải: **Accelerator → GPU** (T4/P100); **Internet → On** (cần để clone GitHub + tải model HF).
- **Add Input** → thêm Dataset dữ liệu (mục 3). **Add-ons → Secrets** → bật `GITHUB_PAT`.

### 5. Chạy
- Dán toàn bộ [notebooks/kaggle_rerank.py](../notebooks/kaggle_rerank.py) vào một cell, chạy.
- Script tự: clone code từ GitHub (token đọc từ Secret, **không in ra**), cài `rank-bm25`/`pyvi`, đọc data từ `DATA_DIR`, build candidate, rerank trên GPU, xuất zip.
- Kiểm tra log in dòng `HEAD: <commit>` để chắc đang chạy đúng code mới nhất.
- Tuỳ chọn ở đầu script: `USE_DENSE`, `TOP_K_RETRIEVE` (50; tăng 75/100 nếu muốn trần recall cao hơn), `TOP_K_FINAL` (3), `RERANK_BATCH_SIZE`.
- Lần sau code đổi: chỉ cần **Run all** lại — clone tự lấy bản mới. Không re-upload dataset (trừ khi data đổi).
- Thời gian: rerank 2000 câu × ~50-100 candidate trên T4 thường vài phút đến ~30 phút tuỳ pool.

#### Nếu gặp CUDA out of memory
Đã xử lý sẵn: reranker cap `max_length=512` và cắt text candidate (điều luật dài tới 245k ký tự, để nguyên sẽ tràn VRAM). Nếu vẫn OOM:
- Giảm `RERANK_BATCH_SIZE` xuống 8 hoặc 4.
- Hoặc giảm `TOP_K_RETRIEVE` (pool nhỏ hơn).
- Script đã set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` để giảm phân mảnh.

### 6. Tải kết quả về và nộp
- Script ghi `submission.zip` (và `results.json`) vào `/kaggle/working/`.
- Tải `submission.zip` từ panel **Output** của notebook.
- Nộp lên leaderboard, so ARTICLES_F2 với kf3 hiện tại (**0.1669**).

## Kỳ vọng & cách đọc kết quả

- Reranker là lever DUY NHẤT còn lại có khả năng vượt 0.167 một cách đáng kể. Nếu nó vẫn không vượt → trần bị chặn bởi **candidate recall** (điều đúng không có trong BM25 top-50); khi đó thử tăng `TOP_K_RETRIEVE`, hoặc kết luận corpus/BM25 không đủ surface điều đúng.
- Ghi lại cho working notes: model reranker (tên, ngày phát hành 2024, ~568M params — tuân thủ quy chế ≤14B + trước 1/3/2026), `TOP_K_RETRIEVE`, `USE_DENSE`, điểm leaderboard.

## Tuân thủ quy chế

`BAAI/bge-reranker-v2-m3`: mã nguồn mở, ~568M tham số (< 14B), phát hành 2024 (trước 1/3/2026). Dense `bkai-foundation-models/vietnamese-bi-encoder` đã dùng từ Phase 2 cũng hợp lệ. Cả hai chạy trên Kaggle GPU không vi phạm ràng buộc mô hình (vẫn là mô hình mở, tự tải trọng số).
