# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

> Khi temperature = 0.0, câu trả lời mang tính xác định (deterministic), ngắn gọn, chuẩn xác và luôn lặp lại cùng một sự thật phổ biến (như vị thế xuất khẩu cà phê hay hang Sơn Đoòng).Khi temperature tăng dần (0.5 → 1.0), nội dung trở nên đa dạng, phong phú hơn về cách diễn đạt và chủ đề nhưng vẫn giữ được tính mạch lạc và chính xác. Khi temperature = 1.5, câu trả lời mang tính ngẫu nhiên rất cao, từ ngữ sáng tạo/phá cách hơn nhưng cấu trúc câu có xu hướng lan man, kém mạch lạc và dễ gặp hiện tượng ảo giác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Theo em nghĩ thì temperature <1 là hợp lý, vì khi đó mô hình sẽ trả lời ngắn gọn, chính xác và không bị lan man.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo em tìm hiểu thì với Output Token thì GPT 4o đắt hơn GPT 4o-mini 2 lần, theo bảng giá ở (https://developers.openai.com/api/docs/pricing) và vì workload là như nhau nên giá của output là yếu tố chính. Trường hợp GPT-4o xứng đáng với chi phí bỏ ra là khi thực hiện trên đánh giá chất lượng, nếu như yêu cầu về chất lượng và độ chính xác của công việc cao hơn đáng kể so với GPT-4o-mini thì nên sử dụng GPT-4o. Ngược lại, trường hợp yêu cầu về tốc độ, chi phí thấp hơn, hoặc là GPT-4o không có quá nhiều khác biệt về mặt chất lượng so với GPT-4o-mini thì ta sẽ dùng GPT-4o-mini.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với system prompt là giáo viên tiểu học, thì model sẽ sử dụng câu văn ngắn, từ vựng đơn giản, gần gũi và dùng các ví dụ gần gũi trong đời sống. Còn với system prompt là các chuyên gia tài chính thì sử dụng câu văn dài phức tạp, các từ mang tính chuyên môn, đi sâu vào kĩ thuật hơn. System prompt đóng vai trò thiết lập nên khung ngữ cảnh, định hình vai trò, văn phong, và đối tượng mà model sẽ hướng tới. Nhờ đó mô hình tự động đưa ra câu trả lời phù hợp với yêu cầu mà không cần phải chỉ dẫn chi tiết từng chút một.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Khi kiểm tra một đoạn văn tiếng Việt 100 từ, đếm qua count_tokens cho kết quả khoảng 120 tokens, còn theo ước lượng số từ / 0.75 ra khoảng 133.3 tokens (chênh lệch khoảng 10%). Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì hai lý do chính: (1) Thuật toán Byte-Pair Encoding (BPE) được huấn luyện phần lớn trên dữ liệu tiếng Anh nên hầu hết từ tiếng Anh thông dụng là một token nguyên vẹn; (2) Tiếng Việt có nhiều nguyên âm và dấu thanh đặc thù, trong bảng mã UTF-8 các ký tự này chiếm từ 2–3 bytes (thay vì 1 byte như ký tự ASCII tiếng Anh), dẫn đến việc tokenizer thường bẻ vụn một từ tiếng Việt thành nhiều sub-tokens.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng, giảm độ khó chịu khi phải chờ quá lâu, giúp người dùng đọc ngay nội dung mà không phải chờ đợi toàn bộ câu trả lời. Ngược lại, non-streaming phù hợp khi xử lý batch processing, cần đầu ra dạng cấu trúc JSON để code backend xử lý tiếp theo trước khi đưa ra người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff: thời gian chờ tăng dần → giảm áp lực lên API khi đang quá tải, cho API thời gian hồi phục. Delay cố định: hàng nghìn client đều retry sau đúng 1 giây → chúng có thể retry cùng lúc, tạo ra “cơn bão retry”, khiến API càng quá tải và tiếp tục thất bại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System Prompt: "Bạn là một chuyên gia về kiến trúc hệ thống, hãy giải thích cho tôi thật kỹ dưới góc độ chuyên gia và ưu tiên dùng tiếng Việt". Em chỉ định ngôn ngữ vì là người Việt, quen với tiếng Việt hơn là tiếng Anh, tiếp theo là tôi là người muốn hiểu sâu hơn về mặt kĩ thuật nên tôi muốn mô hình sẽ giải thích kỹ hơn dưới góc nhìn của một người làm kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Trợ lý hiện tại cắt cứng lịch sử lấy 3 lượt gần nhất (history = history[-6:]), dẫn đến việc mất hoàn toàn ngữ cảnh và các ràng buộc/thông tin quan trọng người dùng đã thiết lập ở đầu phiên chat. Giải pháp: Tóm tắt ngữ cảnh tự động (Conversation Summarization) kết hợp với Sliding Window. Cách triển khai: Khi số lượt hội thoại vượt quá 3 lượt, thay vì xóa bỏ các message cũ, ta gửi các message đó sang một model chi phí thấp (gpt-4o-mini) với prompt yêu cầu tóm tắt ngắn gọn các sự thật/thông tin cốt lõi. Đoạn tóm tắt này sau đó được cập nhật trực tiếp vào system prompt của phiên chat, giúp trợ lý vừa ghi nhớ được thông tin dài hạn vừa tiết kiệm chi phí token.
---
 
## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
