# Trợ lý pháp lý AI cho SME — Tổng hợp đề bài và đề xuất hướng làm

## 1. Tóm tắt nhanh đề bài

Cuộc thi yêu cầu xây dựng một hệ thống AI giải quyết đồng thời hai nhiệm vụ trên cùng một câu hỏi pháp lý: truy hồi điều luật liên quan (Information Retrieval) và sinh câu trả lời có căn cứ (Question Answering). Input duy nhất là một câu hỏi tiếng Việt; output bắt buộc là một file `results.json` chứa với mỗi câu hỏi bốn trường `answer`, `relevant_docs`, `relevant_articles` theo đúng định dạng `<mã văn bản>|<tên văn bản>|<điều>`, nén phẳng vào `submission.zip` rồi nộp lên leaderboard.

Điểm IR được tính bằng F2-macro (trọng số recall gấp đôi precision), nghĩa là bỏ sót điều luật đúng bị phạt nặng hơn việc liệt kê dư vài điều không liên quan — đây là một lựa chọn thiết kế quan trọng cho retrieval threshold. Điểm QA gồm năm nhóm tiêu chí nhưng hiện chỉ nhóm "căn cứ chính xác pháp luật" (trích đúng ít nhất một điều luật trong `answer`) được chấm tự động ngay; bốn nhóm còn lại do LLM-as-judge và chuyên gia chấm sau, đang để 0.0. Phần QA không chấm toàn bộ bài nộp — mỗi đội tự chọn một bài để "đẩy" (promote) lên leaderboard, và ban tổ chức chấm QA định kỳ mỗi tuần trên bài đang được đẩy tại thời điểm đó.

Về ràng buộc mô hình: chỉ được dùng mô hình mã nguồn mở, trọng số tải được tự do, dưới 14B tham số, và phải phát hành trước 1/3/2026. Các mô hình đóng (GPT-4o, Gemini...) bị cấm hoàn toàn, kể cả dùng ngầm trong pipeline. Báo cáo working notes phải nêu rõ nguồn gốc và cách lấy mô hình để đảm bảo tái lập được.

Về thời hạn: khai mạc 3/6/2026, đóng cổng nộp bài 30/6/2026 23:59 giờ Việt Nam, công bố Top 10 vào 5/7, DemoDay chung cuộc 11/7. Hôm nay là 21/6/2026, tức **chỉ còn khoảng 9 ngày** trước khi đóng cổng — đây là ràng buộc thời gian gấp nhất của cuộc thi và cần định hình toàn bộ kế hoạch làm việc theo hướng có một baseline chạy được sớm rồi cải tiến dần, thay vì tối ưu kiến trúc hoàn hảo trước khi có gì để nộp.

Một ràng buộc nộp bài cần nhớ: mỗi đội tối đa 10 lần nộp/ngày, nhưng ở Vòng Riêng tổng cộng mỗi người dùng chỉ được 5 lần nộp toàn cuộc thi — nên các lần nộp ở vòng này phải được cân nhắc kỹ, không nộp tùy hứng để dò kết quả.

## 2. Phân tích hai file dữ liệu đã có

`R2AIStage1DATA.json` là tập câu hỏi kiểm thử: 2000 câu hỏi, mỗi câu chỉ có `id` (1–2000, không trùng) và `question` (tiếng Việt, độ dài 46–403 ký tự, trung bình ~166 ký tự). Không có nhãn đáp án, không có train/dev set nào được cung cấp — đúng như mô tả trong đề bài. Nội dung câu hỏi xoay quanh các tình huống cụ thể của SME: ưu đãi đấu thầu, hỗ trợ thuê mặt bằng, chuyển đổi hộ kinh doanh thành doanh nghiệp, xử lý vi phạm hợp đồng lao động (giữ bằng cấp gốc của nhân viên), v.v. — đúng phạm vi Luật Doanh nghiệp, hỗ trợ SME, đầu tư, lao động như đề bài mô tả.

`vbpl_dat.json` là kho điều luật (corpus) để retrieval: 4755 đoạn văn bản, mỗi đoạn gắn với một `doc_id`, `doc_name`, `article_id`, nội dung `text`, và hai trường đã dựng sẵn đúng format nộp bài là `relevant_doc_tag` và `relevant_article_tag` — điều này rất thuận lợi vì có thể tái sử dụng trực tiếp hai trường này cho `relevant_docs`/`relevant_articles` khi retrieval ra đúng điều luật, không cần tự ghép chuỗi theo công thức "Loại văn bản + Mã văn bản + Trích yếu". Corpus trải trên 173 văn bản pháp luật khác nhau (Luật, Nghị định, Thông tư), bao trùm đúng các mảng đề bài nhắc tới: Luật Doanh nghiệp (59/2020/QH14 và bản sửa đổi 76/2025/QH15), Luật Hỗ trợ DNNVV và Nghị định 80/2021/NĐ-CP hướng dẫn, Luật Đầu tư, Luật Đầu tư công, Luật Đấu thầu, Luật Phá sản/Phục hồi, thuế thu nhập doanh nghiệp, đấu giá tài sản, và một số văn bản về lao động.

Dữ liệu corpus có vài điểm cần xử lý trước khi đưa vào pipeline. Có 333 cặp `relevant_article_tag` bị lặp với nội dung text giống nhau gần như tuyệt đối — nếu không loại trùng, retrieval có thể trả về cùng một điều luật hai lần hoặc làm lệch điểm BM25/embedding do tần suất bị nhân đôi. Có ít nhất một bản ghi với `doc_id` là chuỗi "UNKNOWN" (Nghị định 18-CP về Luật Đầu tư nước ngoài) — cần kiểm tra xem có bao nhiêu bản ghi như vậy và quyết định giữ, sửa lại mã, hoặc loại bỏ. Độ dài đoạn text dao động rất lớn, từ 51 ký tự đến hơn 245.000 ký tự (một "Điều" duy nhất nhưng kèm toàn bộ phụ lục/biểu mẫu dài dạng Thông tư) — những đoạn quá dài này sẽ vượt context của hầu hết embedding model và cần được chunk nhỏ lại, nhưng vẫn phải giữ map ngược về đúng `article_id` gốc khi xuất `relevant_articles`, để không làm sai định danh điều luật khi chấm điểm.

## 3. Đề xuất kiến trúc giải pháp

Hướng phù hợp nhất với ràng buộc (không có dữ liệu train, mô hình phải nhỏ và mở, thời gian gấp) là một pipeline RAG hai tầng: retrieval rồi sinh câu trả lời có trích dẫn, không fine-tune mô hình sinh (rủi ro cao, tốn thời gian, khó tái lập đúng hạn) mà tập trung nỗ lực vào chất lượng retrieval và prompt engineering có cấu trúc.

Tầng tiền xử lý xử lý ba việc trước khi index: loại bỏ các đoạn `relevant_article_tag` bị trùng lặp y hệt nhau, chuẩn hóa hoặc loại các bản ghi `doc_id` là "UNKNOWN" sau khi tra cứu lại văn bản gốc nếu cần, và chia nhỏ các điều luật quá dài thành các chunk con (theo khoản, hoặc theo độ dài cố định khoảng 500–800 ký tự có overlap) trong khi vẫn lưu `article_id` gốc làm metadata để khi retrieval trả về một chunk con, hệ thống biết quy về đúng "Điều X" khi xuất kết quả.

Tầng retrieval nên dùng kết hợp BM25 (tốt cho thuật ngữ pháp lý chính xác như "Điều 4", "doanh nghiệp nhỏ và vừa") và dense embedding tiếng Việt chuyên biệt cho domain pháp luật, sau đó fuse điểm hai nguồn (ví dụ reciprocal rank fusion) rồi rerank bằng cross-encoder nếu thời gian cho phép. Vì điểm IR dùng F2 (recall được nhân trọng số gấp đôi precision trong công thức 5PR/(4P+R)), ngưỡng lọc số điều luật trả về nên ưu tiên không bỏ sót — lấy top-k khá rộng ở bước retrieval ban đầu (ví dụ top 15–20) rồi dùng reranker hoặc ngưỡng điểm để cắt xuống một danh sách gọn hơn (thường 3–6 điều cho một câu hỏi), tránh cả hai cực: trả quá ít (mất recall) và trả tràn lan hàng chục điều không liên quan (mất precision nặng vì macro-average tính trên từng câu hỏi).

Về mô hình embedding, có vài lựa chọn mã nguồn mở tiếng Việt phù hợp domain pháp luật đáng cân nhắc: `AITeamVN/Vietnamese_Embedding` (fine-tune từ BGE-M3, huấn luyện trên khoảng 300.000 triplet tiếng Việt, hỗ trợ độ dài tới 2048 token, phát hành 2025), `bkai-foundation-models/vietnamese-bi-encoder` (huấn luyện một phần trên chính tập Legal Text Retrieval Zalo 2021, đã có benchmark tốt trên dữ liệu pháp luật), và các embedding chuyên biệt pháp luật hơn như `truro7/vn-law-embedding` hoặc `mainguyen9/vietlegal-e5`. Với hai model sau, cần tự kiểm tra lại ngày phát hành chính thức trên trang Hugging Face của model (một số bản có thể phát hành sau 1/3/2026, vi phạm ràng buộc thời điểm) trước khi quyết định dùng, vì thông tin về các bản phát hành rất gần đây nằm ngoài độ tin cậy của tìm kiếm hiện tại.

Tầng sinh câu trả lời dùng một LLM mở dưới 14B có khả năng tiếng Việt tốt, được cấp context là các điều luật đã retrieval (kèm `doc_name` và `article_id` rõ ràng), với prompt yêu cầu nghiêm ngặt ba điều: chỉ trả lời dựa trên điều luật được cấp, trích dẫn rõ "Điều X" của đúng văn bản khi nêu căn cứ (vì hệ thống chấm tự động quét pattern "Điều X" trong `answer` để đối chiếu), và nói rõ giới hạn nếu không có điều luật nào đủ liên quan thay vì bịa. Các lựa chọn nền tảng hợp lý trong giới hạn 14B và mã nguồn mở Apache 2.0 gồm dòng Qwen3 (4B/8B/14B, phát hành tháng 4/2025, có hỗ trợ đa ngôn ngữ tốt bao gồm tiếng Việt) hoặc các bản fine-tune tiếng Việt/pháp luật từ nền Qwen như các model tiếng Việt chuyên luật đang xuất hiện trên Hugging Face. Cũng cần tự xác minh ngày phát hành cụ thể của bất kỳ bản fine-tune nào trước khi chốt dùng, vì đây là mốc thời gian cứng của quy chế và việc dùng sai sẽ bị loại khỏi bảng xếp hạng.

Tầng định dạng đầu ra nên tận dụng trực tiếp `relevant_doc_tag`/`relevant_article_tag` đã có sẵn trong `vbpl_dat.json` cho các điều luật được retrieval — gần như không cần tự ghép lại theo công thức tên văn bản, giảm rủi ro sai định dạng. Trước khi nộp, nên có một script validate độc lập kiểm tra: đủ 2000 id, không thiếu trường, format `relevant_articles` đúng `mã|tên|Điều X`, file tên đúng `results.json`, zip phẳng không có thư mục con — vì đề bài nói rõ bài thiếu câu hoặc sai định dạng sẽ không được chấm và không tính vào lượt nộp.

## 4. Đánh giá nội bộ khi không có dev set

Vì ban tổ chức không phát dữ liệu train/dev, cần tự dựng một bộ đánh giá nội bộ nhỏ để so sánh các phương án retrieval/prompt trước khi tốn lượt nộp thật. Một cách khả thi là tự gán nhãn tay 50–100 câu hỏi mẫu lấy từ `R2AIStage1DATA.json` (ưu tiên các câu có vẻ ánh xạ rõ tới một vài văn bản cụ thể trong corpus, ví dụ các câu về SME tham chiếu rõ Luật 04/2017/QH14 và Nghị định 80/2021/NĐ-CP), tính P/R/F2 trên bộ này để chọn cấu hình retrieval tốt nhất, rồi mới áp dụng cho toàn bộ 2000 câu và nộp thật. Việc này tốn thời gian tay nhưng đáng giá vì giúp tiết kiệm các lượt nộp quý giá, đặc biệt là 5 lượt nộp tổng của Vòng Riêng.

## 5. Đề xuất mốc thời gian cho 9 ngày còn lại

Vì hạn đóng cổng là 30/6, ưu tiên trong vài ngày đầu nên là dựng được một baseline đơn giản chạy hết 2000 câu hỏi và nộp được một bài hợp lệ sớm nhất có thể — kể cả khi retrieval chỉ dùng BM25 thuần và câu trả lời còn thô — để xác nhận toàn bộ pipeline từ đầu đến cuối (tiền xử lý, chunk, retrieval, sinh câu trả lời, xuất JSON, đóng zip, nộp lên dashboard) không có lỗi định dạng nào, tránh trường hợp đến gần hạn mới phát hiện bài bị từ chối. Sau khi có baseline chạy được, phần thời gian còn lại nên dồn vào cải thiện retrieval (thử embedding khác, tinh chỉnh ngưỡng top-k, thêm reranker) và cải thiện chất lượng câu trả lời/trích dẫn, đo lại trên bộ đánh giá nội bộ trước mỗi lần quyết định nộp chính thức để không lãng phí trong số 5 lượt nộp Vòng Riêng.

## 6. Lưu ý cho báo cáo working notes

Vì kết quả chỉ chính thức sau khi nộp working notes paper, nên ghi lại ngay từ đầu (không để dồn về sau): nguồn và phiên bản chính xác của mọi văn bản pháp luật dùng thêm ngoài `vbpl_dat.json` nếu có bổ sung, tên đầy đủ và link Hugging Face cùng ngày phát hành của mọi mô hình embedding/LLM sử dụng (để chứng minh tuân thủ mốc 1/3/2026 và giới hạn 14B), và mô tả pipeline đủ chi tiết để tái lập (sơ đồ retrieval, prompt dùng cho sinh câu trả lời, cách xử lý các điều luật dài/trùng lặp trong corpus).

---

Sources cho các gợi ý mô hình (cần tự xác minh lại ngày phát hành cụ thể trước khi chốt dùng):
- [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
- [bkai-foundation-models vietnamese-bi-encoder](https://www.aimodels.fyi/models/huggingFace/vietnamese-bi-encoder-bkai-foundation-models)
- [truro7/vn-law-embedding](https://huggingface.co/truro7/vn-law-embedding)
- [mainguyen9/vietlegal-e5](https://huggingface.co/mainguyen9/vietlegal-e5)
- [Qwen3 announcement](https://qwenlm.github.io/blog/qwen3/)
- [luanngo/Qwen3-4B-VietNamese-Legal-Chat](https://huggingface.co/luanngo/Qwen3-4B-VietNamese-Legal-Chat)
