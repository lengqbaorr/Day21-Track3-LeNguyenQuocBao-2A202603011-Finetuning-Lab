# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

Tôi ngạc nhiên vì `attn_only` hòa `correct` ở target 0.97 dù loss 0.5369 thấp hơn 0.6257. Tôi chờ text-linear có ưu thế rõ, nhưng kết quả buộc tôi ghi nhận hòa thay vì diễn giải theo đáp án dự kiến. FT đạt format 1.0 mà regression giảm xuống 0.5222: học đúng đầu ra tác vụ không có nghĩa năng lực chung được giữ lại.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Trong phần huấn luyện, chạy lần lượt đối chứng NB4 tốn công chờ và kiểm tra nhất, phù hợp dự đoán rằng phép so sánh công bằng có chi phí. `runs.csv` ghi thời gian train `attn_only=274.0`, `wrong_lr=408.1` và `qlora=477.3` s; đây là từng run, không phải thời gian toàn notebook. Tôi chưa dự đoán tốt rắc rối môi trường Windows và khoảng trống log định tính được phát hiện khi review: preview ban đầu không đủ so sánh từng mẫu với (b) và regression. Khoảng trống này đã được khép lại bằng lần chạy chẩn đoán trên Colab tái lập các điểm chính thức, với đầu ra ghép cặp lưu trong `submission/paired_evidence.md` và `submission/qualitative_paired.json`. Tôi phải kiểm tra nguồn chứng cứ thay vì suy đoán đầu ra thiếu.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi không còn tin loss thấp nhất tự động cho adapter tốt nhất, hoặc thắng target là đủ để deploy. Loss q,v và text-linear khác nhau nhưng target hòa, còn regression gate vẫn FAILED cho adapter chính. Tôi cũng không coi lượng tử hóa là tiết kiệm bộ nhớ không trả giá: `qlora` dùng 3.86 GB, target 0.94 và latency 1793.3 ms; `correct` dùng 8.78 GB, target 0.97 và latency 1388.9 ms.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude làm người lập kế hoạch và rà soát lập luận, Codex thực thi lệnh, kiểm tra artifacts và soạn báo cáo từ số liệu lưu. Lỗi cụ thể của Codex là chọn kích hoạt `.venv` bằng `Activate.ps1` khi PowerShell chặn chạy script; lệnh bị `PSSecurityException` và lần thử phải dừng. Tôi sửa cách vận hành bằng chỉ dẫn gọi trực tiếp `.venv\Scripts\python.exe`, sau đó cài dependency và chạy smoke/NB1 được. Một lỗi quy trình AI khác là lần chạy pipeline Colab đầu tiên clone repo upstream VinUni-AI20k thay vì fork của sinh viên; lần chạy đó đã bị dừng và chạy lại từ fork, cùng commit `d27c1c0`. Khi review báo cáo, tôi phát hiện khoảng trống đầu ra từng mẫu của (b)/regression và yêu cầu bổ sung chứng cứ thay vì tạo ca thua chưa quan sát. Lần chạy chẩn đoán trên Colab đã tái lập các điểm chính thức và khép lại khoảng trống này qua `submission/paired_evidence.md` và `submission/qualitative_paired.json`. AI giảm thao tác, nhưng trách nhiệm đối chiếu số đo và quyết định không triển khai vẫn thuộc tôi.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi xác định hành vi cần và năng lực phải giữ, đóng băng tập target cùng regression, đo prompt tối ưu trước huấn luyện. Tôi kiểm tra ví dụ thực tế, nhãn, template và loss mask, đồng thời thiết kế hỗn hợp replay dữ liệu phổ thông để hạn chế quên. Tôi yêu cầu lưu đầy đủ đầu ra baseline và FT theo từng mẫu ngay từ đầu, rút kinh nghiệm từ khoảng trống phát hiện khi review và đã khép lại bằng lần chạy chẩn đoán Colab tái lập các điểm chính thức, lưu trong `submission/paired_evidence.md` và `submission/qualitative_paired.json`. Nếu gate thất bại, tôi giữ baseline cho sử dụng thực tế và thử sửa dữ liệu trước khi đề xuất triển khai adapter.
