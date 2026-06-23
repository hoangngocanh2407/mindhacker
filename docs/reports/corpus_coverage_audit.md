# Corpus Coverage Audit — ROOT CAUSE của điểm IR thấp

**Date:** 2026-06-22. **Mức độ:** quan trọng nhất dự án — giải thích vì sao mọi tối ưu retrieval chạm trần ~0.18 và vì sao nhóm khác điểm cao hơn.

## Bối cảnh

`vbpl_dat.json` (corpus) **do team tự thu thập**, KHÔNG phải BTC cấp. Chỉ câu hỏi là của BTC. Đáp án gốc (gold) của BTC trỏ tới điều luật trong các văn bản pháp luật thật.

## Phát hiện

Corpus có 173 văn bản nhưng **chỉ 15 là Luật (QH)**, và thiếu hầu hết các **Luật gốc** cho chủ đề SME — chỉ thu thập nghị định/thông tư hướng dẫn:

| Mảng | Có | Luật gốc THIẾU |
|---|---|---|
| Hỗ trợ DNNVV (trung tâm cuộc thi) | NĐ 80/2021, TT 06/2022 | **Luật 04/2017/QH14** |
| Lao động | 23 NĐ/TT | **Bộ luật Lao động 45/2019/QH14** |
| Đấu thầu | luật sửa đổi 57/2024 + NĐ | **Luật Đấu thầu 22/2023/QH15** |
| Quản lý thuế | NĐ/TT | **Luật 38/2019/QH14** |
| Kế toán | NĐ 174/2016 | **Luật Kế toán 88/2015/QH13** |
| BHXH | 9 NĐ/TT | **Luật BHXH (41/2024 hoặc 58/2014)** |
| Thuế TNDN | NĐ 24/2007 + TT | **Luật Thuế TNDN** |
| Thuế GTGT | NĐ/TT | **Luật Thuế GTGT** |
| Thương mại | NĐ 09/2018 | **Luật Thương mại 36/2005/QH11** |

15 Luật CÓ trong corpus: 59/2020 (Doanh nghiệp), 143/2025 (Đầu tư), 23/2018 (Cạnh tranh), 91/2015 (Dân sự), 142/2025 (Phá sản), 64/2020 (PPP), 19/2023 (BVQLNTD), 20/2023 (GDĐT), 54/2010 (Trọng tài TM), 69/2020 (NLĐ đi làm nước ngoài), 74/2025 (Việc làm), 76/2025 + 131/2025 + 57/2024 + 51/2024 (các luật sửa đổi). → thiếu đúng các luật lõi SME/lao động/thuế/đấu thầu/BHXH/kế toán.

## Vì sao đây là root cause (không phải thuật toán)

- Recall trần ~0.21 kể cả với BM25 + dense + cross-encoder reranker pool rộng (top-50∪dense). Reranker chỉ promote được điều luật **có trong pool**; pool chỉ chứa điều luật **có trong corpus**. Điều luật trong các Luật gốc bị thiếu → không bao giờ xuất hiện → recall bị chặn cứng.
- Bằng chứng cụ thể: câu hỏi "công ty giữ bản chính bằng cấp nhân viên" → đáp án là Điều 17 Bộ luật Lao động (cấm giữ giấy tờ gốc) — corpus không có Bộ luật Lao động.
- Article-level coverage TRONG các văn bản đã có thì ổn (Luật Doanh nghiệp đủ 218 điều...), nên vấn đề là THIẾU VĂN BẢN, không phải thiếu điều trong văn bản.

## Hành động (impact cao nhất, vượt xa mọi tuning thuật toán)

1. **Thu thập các Luật gốc còn thiếu** (ưu tiên theo tần suất chủ đề câu hỏi): Luật 04/2017/QH14 (Hỗ trợ DNNVV) → Bộ luật Lao động 45/2019/QH14 → Luật Đấu thầu 22/2023/QH15 → Luật Quản lý thuế 38/2019/QH14 → Luật BHXH → Luật Thuế TNDN/GTGT → Luật Kế toán 88/2015 → Luật Thương mại 36/2005. Nguồn: vbpl.vn / thuvienphapluat.vn (như đã làm cho nghị định).
2. **Ingest vào corpus đúng schema** hiện có: mỗi điều một bản ghi `{doc_id, doc_name, article_id, text, relevant_doc_tag, relevant_article_tag}`, `doc_id` dùng mã chuẩn (vd `04/2017/QH14`), `article_id` dạng `"Điều N"`. Cần script tách văn bản luật thành từng điều.
3. Chạy lại pipeline (BM25 kf3, hoặc reranker) trên corpus đã bổ sung → recall sẽ tăng vì giờ điều luật đúng đã nằm trong corpus.

## Bài học

Trước khi tối ưu retrieval, phải kiểm tra coverage của corpus tự thu thập so với phạm vi câu hỏi. Ở đây ~6 phase tối ưu thuật toán chỉ đẩy được 0.16→0.18 vì trần thật nằm ở dữ liệu thiếu. Việc bổ sung các Luật gốc nhiều khả năng cải thiện điểm hơn tất cả các phase tuning cộng lại.
