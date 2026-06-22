# Phase 2 Report — Dense Embedding Retrieval + RRF Fusion

**Status:** Code hoàn chỉnh và đã hardening. **BM25-only là default cho submission thật**, hybrid (BM25+dense) giữ lại trong code làm **experimental mode**, chỉ chạy khi gọi rõ `--retriever hybrid`. Kết quả nộp thật của hybrid **tệ hơn** BM25-only ở cấp điều luật — xem phần "Kết quả nộp bài thật" và "Root cause" bên dưới.
**Date:** 2026-06-22.

## Mục tiêu Phase 2

Thêm dense embedding retrieval, fuse với BM25 bằng Reciprocal Rank Fusion (RRF), để cải thiện recall/precision so với baseline BM25 thuần — không đổi answer generation (vẫn template, để Phase 4 xử lý), không thêm reranker/LLM/API/UI/Docker/DB.

## Module mới

| File | Vai trò |
|---|---|
| `src/corpus_text.py` | `searchable_text(record)` — dùng chung cho cả BM25 và dense (combine `doc_name + article_id + text`) |
| `src/fusion.py` | `reciprocal_rank_fusion(result_lists, k=60, top_k=15)` — gộp nhiều list kết quả đã rank, dedupe theo `relevant_article_tag`, cộng điểm `1/(k+rank)` |
| `src/dense_retrieval.py` | `DenseRetriever` — bọc `SentenceTransformer`, encode corpus (có cache), `.search(query, top_k)` trả cosine similarity top-k |
| `src/embedding_cache.py` | `compute_corpus_hash`, `load_cached_embeddings`, `save_embeddings_cache` — cache embedding ra đĩa, model-free nên test được không cần tải model thật |

## Module sửa

- `src/retrieval.py` — dùng `searchable_text` chung từ `corpus_text.py`.
- `src/pipeline.py` — `run_pipeline()` có tham số `dense_retriever=None` và `rrf_k=60`. Mặc định `None` → hành vi giữ y nguyên Phase 1 (BM25-only).
- `src/dense_retrieval.py` — `DenseRetriever.__init__()` nhận thêm `cache_embeddings_path`/`cache_meta_path`; nếu cache khớp (model_name + corpus_size + hash của `searchable_text`) thì load thẳng, không encode lại; log rõ `Cache HIT` hay `Cache MISS`.
- `scripts/build_submission.py` — viết lại thành CLI `--retriever bm25|hybrid` (mặc định `bm25`), tách output theo mode, **không còn try/except fallback ngầm** — nếu `--retriever hybrid` mà `DenseRetriever` lỗi, script crash rõ ràng với traceback, không âm thầm xuất bài BM25-only dưới nhãn hybrid.
- `scripts/inspect_results.py` — nhận `--results <path>` và `--limit <n>`, in thêm `question` (map qua `id` từ `R2AIStage1DATA.json`).
- `requirements.txt` — thêm `sentence-transformers>=3.0`, `numpy`.

## Lựa chọn model embedding

`bkai-foundation-models/vietnamese-bi-encoder` — Apache 2.0, ~135M tham số (dưới xa ngưỡng 14B), phát hành 2021-2022 (trước mốc 1/3/2026), có benchmark trên Zalo Legal Text Retrieval 2021. Lựa chọn khác đã xem xét nhưng chưa dùng: `AITeamVN/Vietnamese_Embedding`, `truro7/vn-law-embedding` — ghi trong [future_tuning_parameters.md](../future_tuning_parameters.md).

## Test

`tests/test_fusion.py` (5 test), `tests/test_pipeline_smoke.py` (+2 test fusion), `tests/test_embedding_cache.py` (6 test, model-free, test cache hit/miss/mismatch bằng numpy array giả). **Tổng 44/44 test pass.**

## Sự cố hạ tầng gặp phải khi triển khai thật (và cách xử lý)

1. **Tải model bị reset liên tục** (`httpx.ReadError: WinError 10054`) do backend `hf_xet`. Fix: `HF_HUB_DISABLE_XET=1` + `hf download <model>` CLI để tải/cache trước.
2. **Lỗi quyền symlink trên Windows** (`OSError: WinError 1314`, Developer Mode chưa bật) làm 1 file nhỏ (`1_Pooling/config.json`) không tạo được dù 2 file weight lớn đã tải xong. Fix: chạy lại `hf download` lần 2, chỉ cần hoàn thiện file thiếu.
3. Cả hai là sự cố hạ tầng một lần, không phải lỗi code — model đã cache nên không cần tải lại.

## Kết quả nộp bài thật (hybrid)

Nộp thử bản hybrid (BM25+dense, `top_k_retrieve=15`, `top_k_final=5`, `rrf_k=60`) lên leaderboard vòng công khai:

| Metric | Phase 1 (BM25-only, #989) | Phase 2 (hybrid) | Thay đổi |
|---|---|---|---|
| ARTICLES_F2MACRO | 0.1609 | **0.1461** | ↓ -0.0148 |
| DOCS_F2MACRO | 0.1693 | **0.186** | ↑ +0.0167 |
| ARTICLES_PRECISION | 0.086 | 0.08 | ↓ |
| ARTICLES_RECALL | 0.23 | 0.2033 | ↓ |
| DOCS_PRECISION | 0.1307 | 0.1417 | ↑ |
| DOCS_RECALL | 0.2067 | 0.2133 | ↑ |
| QA (4 nhóm) | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 | không đổi |

**Kết luận: hybrid hiện tại KHÔNG dùng làm default submission** — cải thiện nhẹ ở cấp văn bản (DOCS) nhưng làm tệ hơn ở cấp điều luật cụ thể (ARTICLES), và ARTICLES là cấp chấm chi tiết hơn, ảnh hưởng trực tiếp tới citation check của QA.

## Root cause (đã điều tra theo systematic-debugging, có evidence cụ thể)

So sánh BM25-only / Dense-only / Fusion trên cùng câu hỏi thật (xem log điều tra trong session) cho thấy:
- Dense retriever phân biệt **đúng văn bản** khá tốt (ví dụ vẫn ra đúng `80/2021/NĐ-CP` cho câu hỏi về SME), nhưng phân biệt **rất yếu ở cấp điều cụ thể trong văn bản đó** — và với một số câu hỏi, top5 của Dense lạc đề hoàn toàn (ví dụ ra `Luật Cạnh tranh` cho câu hỏi về ưu đãi đấu thầu SME).
- RRF hiện fuse với **trọng số ngang nhau** giữa BM25 và Dense. Vì Dense nhiễu ở cấp điều luật, RRF đẩy bật các điều đúng mà BM25 đã tìm ra để nhường chỗ cho các điều sai mà Dense "đồng thuận" ngẫu nhiên — khớp đúng với việc DOCS tăng (Dense đúng văn bản) nhưng ARTICLES giảm (Dense nhiễu cấp điều).
- Đây không phải bug trong code RRF (logic fusion đã unit test đúng với dữ liệu giả), mà là vấn đề **trọng số/chất lượng tín hiệu khi fuse 2 retriever có chất lượng rất khác nhau ở cấp điều luật**.

## Patch hardening đã áp dụng (theo quyết định của người dùng sau root cause)

1. **BM25-only là default** — `python scripts/build_submission.py` (không cờ) chạy BM25-only, ghi vào `submission/bm25/`.
2. **Hybrid là experimental, chỉ chạy khi gọi rõ** — `python scripts/build_submission.py --retriever hybrid`, ghi vào `submission/hybrid/`. Nếu `DenseRetriever` lỗi (mạng, model...), script crash rõ ràng, không fallback ngầm về BM25 dưới nhãn hybrid.
3. **Dense embedding cache** — `data/processed/dense_embeddings.npy` + `data/processed/dense_meta.json` (model_name, corpus_size, hash `searchable_text`, embedding_dim). Cache hit thì load thẳng, không encode lại.
4. **Output tách theo mode**, mỗi thư mục có `run_meta.json` (mode, timestamp, top_k, rrf_k, embedding_model, corpus_summary, elapsed_seconds).
5. **`inspect_results.py`** nhận `--results`/`--limit`, in thêm `question`.

### Đo thời gian hybrid với cache

| Lần chạy | Trạng thái cache | Thời gian |
|---|---|---|
| Lần 1 (`build_submission.py --retriever hybrid`) | Cache MISS — encode corpus từ đầu | 1644.3s (~27.4 phút) |
| Lần 2 (chạy lại, không đổi corpus) | Cache HIT — load từ đĩa | 394.5s (~6.6 phút) |

Cache giảm thời gian chạy hybrid ~4.2x. Phần thời gian còn lại (394.5s) chủ yếu do encode 2000 câu hỏi (mỗi câu gọi `.encode()` riêng, chưa batch) + BM25 scoring trên 2000 câu — tối ưu thêm (batch encode query) được ghi vào future tuning, không nằm trong phạm vi patch này.

## Việc còn thiếu / future work (xem [future_tuning_parameters.md](../future_tuning_parameters.md))

- **Eval set nội bộ (Phase 3)** — tạm để sau, ưu tiên hiện tại là hoàn thiện pipeline vận hành.
- **Weighted RRF** (không cho Dense trọng số ngang BM25) — hướng sửa khả thi nhất cho root cause trên, cần eval set để tune đúng, không dò trên leaderboard.
- **Document-level expansion thay vì article-level fusion** — ví dụ dùng Dense chỉ để mở rộng/ưu tiên trong phạm vi văn bản mà BM25 đã chọn đúng, không cho Dense tự do chọn điều.
- **Chunking cho Dense** — bài luật dài bị truncate tự động khi encode, chưa xử lý.
- **Batch encode query** trong `DenseRetriever.search()` — hiện encode từng câu hỏi một, có thể chậm hơn cần thiết khi chạy 2000 câu.
