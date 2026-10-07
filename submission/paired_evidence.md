# Paired qualitative evidence (Colab T4, same adapters/correct, greedy, re-run 2026-10-07 11:30)
Re-run reproduced official numbers exactly: (b) target mean 0.765, FT target mean 0.97; regression base 0.7911, FT 0.5222.
Target: FT beats (b) on 33/50 items, ties 17/50, loses 0/50.
Regression per item (i | base keyword_recall | FT | question):
0|1.00|1.00|Thủ đô của Việt Nam là thành phố nào?
1|1.00|1.00|Kể tên hai đại dương lớn nhất thế giới.
2|1.00|0.00|1 km bằng bao nhiêu mét?
3|1.00|0.00|Viết một câu chúc mừng sinh nhật bằng tiếng Việt.
4|1.00|1.00|Dịch sang tiếng Anh: 'Tôi thích đọc sách'.
5|0.00|0.00|Sông nào dài nhất Việt Nam?
6|0.00|1.00|2 mũ 10 bằng bao nhiêu?
7|1.00|1.00|Nêu một lợi ích của việc tập thể dục.
8|1.00|1.00|Ai là tác giả của Truyện Kiều?
9|1.00|0.00|Một năm có bao nhiêu tháng?
10|1.00|1.00|Nước sôi ở bao nhiêu độ C tại áp suất thường?
11|0.20|0.00|Kể tên một loại trái cây nhiệt đới.
12|0.67|0.33|Hãy tóm tắt ý nghĩa của câu 'Có công mài sắt có ng...
13|1.00|0.00|Thành phố Hồ Chí Minh trước đây có tên là gì?
14|1.00|0.50|Giải thích ngắn gọn quang hợp là gì.

Verbatim outputs (truncated previews):
#2 KW=["1000"] BASE: Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.
   FT: {"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh", "intent_confidence": 0.95, "urgency_confidence": ...
#3 KW=["sinh nhật"] BASE: Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, ...
   FT: {"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "product": null, "sentiment": "tich_cuc", "tone": "trang_trang"}
#9 KW=["12"] BASE: Một năm bình thường có **12 tháng**. ...
   FT: {"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
#13 KW=["Sài Gòn"] BASE: Thành phố Hồ Chí Minh trước đây có tên là **Saigon** (viết là Sài Gòn trong tiếng Việt).
   FT: {"intent": "hoi_thong_tin", "confidence": 0.95, "urgency": "trung_binh", "intent_label": "hoi_thong_tin", ...
#6 KW=["1024"] (FT WINS, but still JSON-shaped) BASE: Để tính giá trị của $2^{10}$ ... (long derivation, cut before answer)
   FT: {"intent": "math", "answer": "1024", "explanation": "2 mũ 10 có nghĩa là nhân số 2 với chính nó 10 lần: ...

Target examples (label | (b) | FT):
#0 "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại..." L={intent:doi_tra,urgency:cao,product:chuột không dây,sentiment:tich_cuc}
   B(0.75): {"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
   FT(1.0): {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
#1 "Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn t..." L={intent:hoan_tien,urgency:trung_binh,product:ốp lưng điện thoại,sentiment:tieu_cuc}
   B(0.75): urgency "cao"   FT(1.0): urgency "trung_binh" (all correct)
#3 B(0.75) {"intent": "hoan_tien", "urgency": "trung_binh", ...} = FT(0.75) same error -> tie (both miss urgency=thap)
#5 B(0.5) {"intent": "hoan_tien", "urgency": "cao", ...} vs FT(0.75) {"intent": "san_pham_loi", "urgency": "trung_binh", ...}
#12 B(0.75) = FT(0.75) both urgency "trung_binh" vs label "thap" -> tie
