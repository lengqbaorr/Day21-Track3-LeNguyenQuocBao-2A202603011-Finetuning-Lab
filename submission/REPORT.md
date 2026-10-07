# Lab 21 — Evaluation Report

**Họ tên**: Le Nguyen Quoc Bao  **MSSV**: 2A202603011  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Colab Tesla T4, `fp16`.

> Số đo chính được chép nguyên từ `results/*.json` và `results/runs.csv`, không làm tròn lại. Bằng chứng định tính bổ sung lấy từ `submission/paired_evidence.md`, một lần chạy chẩn đoán riêng; đường học và loss ở step cuối lấy từ `submission/training_log_excerpt.md`; thời gian stage lấy từ log Colab `colab_run.py` do người thực hiện cung cấp. Thông tin cá nhân, môi trường và cấu hình chạy là khai báo của người thực hiện; split được kiểm tra thêm từ NB1 và file split. Báo cáo phân biệt các nguồn này và không bổ sung đầu ra mô hình chưa lưu.

---

## 1. Setup

| | |
|---|---|
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage (`token_stats.json:n`) |
| Train / val | 225 / 25, seed 42; đối chiếu `notebooks/01_data_and_mask.py` và `data/split/{train,val}.jsonl` |
| `max_length` | 1024 theo tier; p95 đo được là 98, `suggested_max_length` là 256 (`token_stats.json`) |
| `MASK_MODE` | `assistant-only` (`mask_proof.json`, dòng `correct` trong `runs.csv`) |
| Epochs / max_steps | `EPOCHS=2` theo khai báo chạy; `max_steps=30` ở mọi dòng `runs.csv` |
| Precision | `fp16` ở mọi dòng `runs.csv`; không dùng `bf16` trên T4 trong lần chạy này |
| Eval | `EVAL_LIMIT` không đặt; `eval_limit=null`, `smoke_mode=false`; 50 target và 15 regression (`baselines_frozen.json`) |

Tôi chọn model mặc định của tier vì vừa bộ nhớ Colab free T4 theo khai báo môi trường và giữ lựa chọn model của lab để so sánh với cấu hình tham chiếu. GPU huấn luyện là Colab; máy Windows dùng cho kiểm tra CPU và viết báo cáo không có GPU. Tôi dùng corpus mặc định để kiểm tra pipeline trước khi đổi miền dữ liệu; đáp án JSON có các trường `intent`, `urgency`, `product`, `sentiment`, không chứa reasoning trace. Lần chạy được khai báo xuất phát từ fork [lengqbaorr/Day21-Track3-LeNguyenQuocBao-2A202603011-Finetuning-Lab](https://github.com/lengqbaorr/Day21-Track3-LeNguyenQuocBao-2A202603011-Finetuning-Lab), commit `d27c1c0`.

Tôi giữ `max_length=1024` để duy trì cấu hình tier giữa các run, dù p95 là 98 và độ dài lớn nhất chỉ 101. Đây là giới hạn trên có dư địa, không phải độ dài điển hình; dữ liệu hiện tại không cho thấy cần giới hạn lớn như vậy. NB1 đề xuất 256, nên giảm xuống mức đề xuất là thử nghiệm hiệu suất hợp lý tiếp theo, áp dụng đồng nhất cho mọi đối chứng. Tôi không coi 1024 là lựa chọn tối ưu đã được đo, cũng không thay cấu hình sau khi thấy kết quả để làm lệch phép so sánh.

**Template có giữ khối `<think>` không?** Có: `template_check.json` ghi `ok=true`, `open_tag_present=true`, `body_present=true`, verdict `reasoning preserved — safe to train on traces`. Đây là phép thử template với nội dung suy luận mẫu, không phải bằng chứng corpus có suy luận. Trên corpus chỉ có đáp án JSON, `masked-think` và `response-only` là no-op đối với việc bỏ reasoning vì không có trace để bỏ, như giải thích trong `src/labkit/data.py`. Tôi dùng `assistant-only` và không tuyên bố đã thực hiện thí nghiệm reasoning-trace collapse.

Thời gian từng stage dưới đây lấy từ log Colab `colab_run.py` do người thực hiện cung cấp. Đây là thời gian toàn stage, khác với `train_seconds` của từng run trong bảng autopsy; log này là nguồn riêng, không phải trường trong `results/`.

| Stage | Thời gian (s) |
|---|---|
| nb1 | 10 |
| nb2 | 391 |
| nb3 | 458 |
| nb4 | 1326 |
| nb5 | 670 |
| Tổng | 2856 |

**Kiểm tra Colab và Windows:** lỗi checksum cục bộ đã được giải quyết bằng cách checkout lại bốn file dữ liệu từ Git với `core.autocrlf=false`, khôi phục đúng byte LF đã commit; Git không ghi nhận diff ở các file này. `scripts/verify.py` cục bộ hiện báo **26 passed, 1 warning, 0 failures**, kết thúc bằng `Ready to submit`. Warning duy nhất là ghi chú verdict mô hình FAILED do regression, không phải lỗi tính toàn vẹn dữ liệu hay lỗi verifier.

---

## 2. Mask proof (NB1)

Nguồn: `results/mask_proof.json`.

| | |
|---|---|
| `mask_mode` | `assistant-only` |
| `n_supervised` | 39 |
| `n_total` | 94 |
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss (`answer_is_supervised`) | `true` |
| Câu hỏi KHÔNG nằm trong loss (`question_is_masked`) | `true` |

Dán nguyên `supervised_preview`, sau khi giải mã ký tự xuống dòng:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Preview chứa đáp án JSON và dấu kết thúc lượt; câu hỏi nằm trong `masked_preview`. Dấu `</think>` không đồng nghĩa có nội dung suy luận được giám sát: phần thinking của mẫu này rỗng. Giải mã và các assert ủng hộ mask đúng, nên tôi không quy sự suy giảm regression cho lỗi tính loss trên câu hỏi.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

NB2 đóng băng (a), (b) trong `baselines_frozen.json`; (c) chỉ được đo sau huấn luyện ở NB5. Bảng chép nguyên `results/verdict.json:comparison`, cùng tập eval đầy đủ.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.7911 | 0.0 | 3244.2 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.0 | 1008.5 |
| (c) LoRA fine-tune | 0.97 | 0.5222 | 1.0 | 1388.9 |

**(b) có thật sự mạnh hơn (a) không?** Có: target tăng từ 0.0 lên 0.765, format từ 0.0 lên 1.0; regression trong bảng giữ nguyên. `OPTIMIZED_PROMPT` không sửa, với `optimized_prompt_sha=719e74d3b6232053` đóng băng. Mốc (b) là đối thủ có chất lượng, không bị làm yếu để nâng thành tích FT. Fine-tune được đánh giá với prompt ngắn theo NB5 nhưng vẫn có latency 1388.9 ms, cao hơn 1008.5 ms của (b); prompt ngắn hơn không bảo đảm phản hồi nhanh hơn.

**Giả thuyết về latency:** adapter LoRA chưa merge thêm các phép nhân ma trận vào đường forward ở mỗi layer được gắn adapter; `correct` tác động tới 12 loại module theo `runs.csv`. Chi phí này có thể khiến FT chậm hơn (b) dù prompt ngắn hơn. `attn_only` chỉ gắn vào 2 loại module và có latency 896.5 ms, còn `qlora` cần dequantization từ 4-bit và đạt 1793.3 ms: thứ tự này phù hợp với giả thuyết overhead adapter và lượng tử hóa. Đây chưa phải kết luận nhân quả được tách riêng bằng profiling, vì độ dài đầu ra và các yếu tố thực thi cũng ảnh hưởng latency. Merge ở NB6 sẽ gộp trọng số LoRA vào base và loại bỏ các phép nhân adapter riêng; cần đo lại latency sau merge, và tôi chưa thực hiện NB6.

---

## 4. Giải phẫu cấu hình sai (NB4)

Cấu hình, loss trung bình, thời gian và VRAM chép nguyên `results/runs.csv`; target, format và latency lấy từ `results/autopsy.json`. Placement `attn-only` là q,v theo nhãn CSV. `final_loss` là `result.training_loss`, tức mean training loss của cả run, không phải loss ở step cuối; cách ghi được xác nhận trong NB3/NB4 và [training_log_excerpt.md](training_log_excerpt.md). Cột last-step loss lấy nguyên log ở step 30 trong nguồn này.

| Run | vị trí | r | trainable | LR | mean train loss (final_loss) | last-step loss | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6257 | 0.02638 | 0.97 | 396.0 | 8.78 |
| `attn_only` | attn-only | 283 | 32456704 | 0.0001 | 0.5369 | 0.02444 | 0.97 | 274.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 1e-05 | 1.5702 | 1.119 | 0.0 | 408.1 | 8.78 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.02622 | 0.94 | 477.3 | 3.86 |

| Run | format (NB5) | latency (ms, NB5) |
|---|---|---|
| `correct` | 1.0 | 1388.9 |
| `attn_only` | 1.0 | 896.5 |
| `wrong_lr` | 0.0 | 5278.7 |
| `qlora` | 1.0 | 1793.3 |

**Xếp hạng bằng TARGET:** `correct = attn_only > qlora > wrong_lr`. Mọi run có `max_steps=30`. Biến đối chứng của `attn_only` là vị trí gắn adapter, rank điều chỉnh để khớp ngân sách; của `wrong_lr` là LR; của `qlora` là bật `load_in_4bit=True`. Giữ nguyên rank khi đổi vị trí sẽ làm đổi ngân sách, không còn là đối chứng công bằng.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về rank so với vị trí gắn adapter?**

`attn_only` hòa `correct` ở target 0.97 dù mean train loss 0.5369 thấp hơn 0.6257, đồng thời last-step loss 0.02444 cũng thấp hơn 0.02638 theo [training_log_excerpt.md](training_log_excerpt.md). Vì vậy điểm về loss gây hiểu nhầm vẫn giữ nguyên: dùng cả loss trung bình lẫn loss cuối để xếp hạng đều ngụ ý ưu thế không được target xác nhận. Rank 283 cho 32456704 tham số, gần như bằng 32464896 của text-linear; độ lệch tương đối kiểm tra trực tiếp bằng `abs(32456704 - 32464896) / 32464896`. Với ngân sách và số step tương đương, vị trí không phải đòn bẩy quyết định trên model này, tác vụ hẹp tương đối dễ này và ngân sách 30 step. Kết quả cũng không chứng minh tăng rank riêng lẻ sẽ cải thiện chất lượng, vì rank dùng để khớp ngân sách chứ chưa được quét độc lập. Tham chiếu trên `Qwen3.5-0.8B` do người thực hiện cung cấp cho thấy `correct > attn_only`; tôi ghi nhận khác biệt, không áp thứ tự đó lên lần đo này. `attn_only` còn có latency 896.5 ms thấp hơn 1388.9 ms của `correct`, nhưng thiếu regression của đối chứng này nên chưa thể kết luận nó triển khai được.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` dùng LR 1e-05 thay cho 0.0001; mean train loss của cả run là 1.5702 thay vì 0.6257. Đường học thực tế trong [training_log_excerpt.md](training_log_excerpt.md) cho thấy `correct` giảm từ 2.163 ở step 5 xuống 0.1385 ở step 15, rồi về mức khoảng 0.02 cuối run, với last-step loss chính xác 0.02638. `wrong_lr` cũng giảm nhưng chậm hơn nhiều: từ 2.163 ở step 5 xuống 1.119 ở step 30. Token accuracy tại step 30 là 0.7909 so với 0.9957 của `correct`; đây là số trong log, không phải target accuracy. Đường học vì vậy cho thấy chưa học đủ trong ngân sách cố định, không phải loss hoàn toàn phẳng.

Target 0.0 và format 0.0 của `wrong_lr` phù hợp với việc LR thấp trong 30 step chưa dịch chuyển hành vi base đủ để học khuôn JSON; đây là diễn giải hành vi, không phải phép đo khoảng cách trọng số. Base dùng cùng short naive prompt ở baseline (a) cũng có target 0.0 và format 0.0. Latency 5278.7 ms của `wrong_lr` cùng xu hướng phản hồi dài, chậm như 3244.2 ms của (a), phù hợp với việc vẫn sinh diễn giải lan man thay vì JSON ngắn, cộng thêm overhead adapter; tôi không coi hai latency này là bằng nhau. Sự nhất quán giữa đường loss, token accuracy và baseline giải thích kết quả mà không cần giả định lỗi đánh giá. Chỉ nhìn loss cao có thể khiến tôi kết luận nhầm base hoặc placement không học được tác vụ; đối chứng LR cho thấy chúng học được khi dùng thang LR phù hợp. Kết quả không chứng minh cấu hình LR thấp thất bại với mọi ngân sách dài hơn.

**`grad_norm=nan` trong log:** theo [training_log_excerpt.md](training_log_excerpt.md), fp16 GradScaler phát hiện gradient inf/nan, bỏ qua update bị ảnh hưởng và hạ loss scale. Loss tiếp tục giảm sau đó, nên những dòng này không đồng nghĩa cả run diverge; skipped updates cũng không được hiểu là mọi optimizer update đã thực sự diễn ra thành công. Các run vẫn được cấu hình cùng ngân sách step.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị “không dùng QLoRA cho dòng model này” không?**

VRAM giảm từ 8.78 GB xuống 3.86 GB, tức tiết kiệm 4.92 GB, khoảng 56 phần trăm; đây là chênh lệch tính từ số đo, với tỷ lệ `(8.78 - 3.86) / 8.78`, không thay các giá trị VRAM gốc. Đổi lại, target giảm từ 0.97 xuống 0.94 dù format vẫn 1.0. Latency tăng từ 1388.9 lên 1793.3 ms, thời gian train từ 396.0 lên 477.3 s; tiết kiệm bộ nhớ không đồng nghĩa chạy nhanh hơn. Kết quả ủng hộ ưu tiên LoRA không lượng tử hóa khi bộ nhớ đủ trong thiết lập này, vì chất lượng và thời gian đều thuận lợi hơn. Tuy nhiên, phép đo trên một tác vụ không đủ khẳng định QLoRA luôn không phù hợp; khi bộ nhớ là ràng buộc bắt buộc, trade-off có thể đáng chấp nhận sau kiểm tra đầy đủ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED** (`verdict.passed=false`).
`target Δ = +0.20499999999999996` · `regression Δ = -0.26888888888888884` · `valid_trace_rate = 0.0`.

Các delta giữ nguyên biểu diễn trong JSON. Lý do gate lưu sẵn:

```text
general capability regressed by 0.269 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.
```

Target tốt lên rõ rệt và format hợp lệ, nhưng regression giảm từ 0.7911 xuống 0.5222, vượt dung sai lưu trong lý do gate. Vì thế bản fine-tune không đạt điều kiện triển khai dù target hấp dẫn. `submission/paired_evidence.md` cung cấp bằng chứng trực tiếp: ở regression #2, #9 và #13, base trả lời đúng còn FT xuất JSON triage thay cho đáp án; ở #3, FT còn tự tạo intent `chuc_mung_sinh_nhat`. Đây là biểu hiện quan sát được của catastrophic forgetting: mô hình mất khả năng thực hiện các yêu cầu phổ thông này và áp khuôn đã học từ ticket sang JSON. Hỗn hợp huấn luyện chỉ gồm tác vụ ticket là lời giải thích nhân quả phù hợp; lần chạy chẩn đoán dùng cùng adapter và greedy decoding, tái lập các điểm chính thức, nên các ví dụ củng cố phán quyết thay vì thay thế mốc đánh giá. Cách sửa ưu tiên là trộn replay dữ liệu phổ thông theo dải `1-5%` lưu trong lý do gate và deck §6.3, rồi đánh giá lại trên cùng mốc đóng băng.

`valid_trace_rate=0.0` là dự kiến với corpus không có reasoning trace, không tự nó chứng minh reasoning collapse. Template giữ được `<think>` và mask đúng giúp loại trừ một số lỗi pipeline, nhưng không bảo vệ năng lực ngoài miền. FT còn chậm hơn (b): 1388.9 so với 1008.5 ms. Tôi không nới regression gate để đổi lấy target; lợi ích tác vụ hẹp chưa bù được suy giảm năng lực và chi phí phục vụ.

**Giới hạn phép đo regression:** `keyword_recall` trong `src/labkit/evaluate.py` chỉ kiểm tra tỷ lệ keyword xuất hiện, không buộc câu trả lời tuân thủ kiểu đầu ra hay yêu cầu giao tiếp. Đây là chỉ số khá dễ dãi: regression #0 và #6 trong [qualitative_paired.json](qualitative_paired.json) vẫn đạt FT 1.0 khi đáp án nằm bên trong JSON. Vì vậy regression 0.5222 có khả năng **đánh giá cao hơn năng lực phổ thông thực sự được giữ lại**, đặc biệt về làm theo yêu cầu và cách trả lời. Tôi giữ nguyên điểm chính thức, nhưng không đọc nó như bằng chứng rằng hành vi ngoài miền đã được bảo toàn tương ứng.

### Kiểm tra tính hợp lý của số liệu

- **Baseline (a) target 0.0, format 0.0:** short naive prompt không giúp base tạo JSON triage có các trường yêu cầu; cùng mẫu thất bại xuất hiện ở tham chiếu lab 0.000 trong [SIMULATION-FINDINGS.md](../SIMULATION-FINDINGS.md).
- **Baseline (b) target 0.765:** gần mốc T4 tham chiếu 0.760 trong cùng tài liệu, phù hợp với việc prompt tối ưu là baseline mạnh. Đây là số tham chiếu từ lần chạy khác, không thay số đo hiện tại.
- **Regression của (a) và (b) cùng 0.7911:** `notebooks/02_baselines.py` dùng `system=None` cho regression probe trên cùng base, nên khác biệt prompt target không đi vào phép đo này.
- **`correct = attn_only` ở target 0.97:** ngân sách tham số được khớp và tác vụ hẹp có thể được học bằng cả hai placement; mean loss và last-step loss thấp hơn không bắt buộc tạo ưu thế target. Đường học trong [training_log_excerpt.md](training_log_excerpt.md) phù hợp với việc cả hai đã học tốt dữ liệu train.
- **QLoRA có chi phí target nhỏ:** 0.94 so với 0.97, với format 1.0 ở cả hai; đây là trade-off chất lượng vừa phải đi kèm tiết kiệm VRAM, không phải collapse định dạng. Loss cuối 0.02622 trong log vẫn thấp nhưng không loại bỏ chênh lệch target.
- **Đối chiếu độc lập:** lần chạy chẩn đoán cùng adapter, greedy decoding trong [paired_evidence.md](paired_evidence.md) tái lập chính xác các điểm target và regression chính thức. Bằng chứng này và log trainer giúp kiểm tra tính nhất quán; chúng không được dùng để thay số đo hoặc điều chỉnh gate sau khi thấy kết quả.

---

## 6. Định tính — bắt buộc có cả ca THUA

**Nguồn:** [paired_evidence.md](paired_evidence.md), lưu lần chạy chẩn đoán riêng trên Colab với cùng `adapters/correct` và greedy decoding. Lần chạy này tái lập đúng điểm chính thức: target (b) 0.765, FT 0.97; regression base 0.7911, FT 0.5222. Đây là bằng chứng bổ sung trong `submission/`, không phải một phần của `results/`, không phải lần train mới và không thay các số đo chính thức.

Trên target, FT thắng (b) **33/50**, hòa **17/50**, thua **0/50**. Vì vậy các ca FT thua trong bảng nằm ở nhóm regression. Nhãn nhóm target và câu hỏi của các ca hòa được đối chiếu thêm với `data/eval_target.jsonl`; đầu ra và điểm so sánh lấy từ paired evidence. Nguồn chỉ lưu một số preview hoặc trường được trích, nên bảng giữ nguyên dấu `...` và không dựng phần đầu ra bị thiếu. Số # được ghi kèm nhóm để phân biệt cùng chỉ số giữa target và regression.

| Nhóm / # | Input | Expected / keywords | Base hoặc (b) output | FT output | Verdict |
|---|---|---|---|---|---|
| Target #0 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | `intent=doi_tra; urgency=cao; product=chuột không dây; sentiment=tich_cuc` | `{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | **FT THẮNG**: (b) 0.75, FT 1.0; sửa intent. |
| Target #1 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn t... | `intent=hoan_tien; urgency=trung_binh; product=ốp lưng điện thoại; sentiment=tieu_cuc` | Trích trường: urgency `"cao"` | Trích trường: urgency `"trung_binh"`; nguồn ghi all correct | **FT THẮNG**: (b) 0.75, FT 1.0; sửa urgency. |
| Target #3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. | `intent=hoan_tien; urgency=thap; product=bình giữ nhiệt; sentiment=tich_cuc` | `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | Nguồn ghi cùng lỗi: `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | **HÒA**: (b) 0.75, FT 0.75; cả hai sai urgency. |
| Target #12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. | `intent=san_pham_loi; urgency=thap; product=áo khoác gió; sentiment=tich_cuc` | Trích trường: urgency `"trung_binh"` | Trích trường: urgency `"trung_binh"` | **HÒA**: (b) 0.75, FT 0.75; cùng sai nhãn `thap`. |
| Regression #2 | 1 km bằng bao nhiêu mét? | `["1000"]` | Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét. | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh", "intent_confidence": 0.95, "urgency_confidence": ...` | **FT THUA**: base 1.00, FT 0.00; phân loại thay vì đổi đơn vị. |
| Regression #9 | Một năm có bao nhiêu tháng? | `["12"]` | Một năm bình thường có **12 tháng**. ... | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}` | **FT THUA**: base 1.00, FT 0.00; không đưa đáp án. |
| Regression #13 | Thành phố Hồ Chí Minh trước đây có tên là gì? | `["Sài Gòn"]` | Thành phố Hồ Chí Minh trước đây có tên là **Saigon** (viết là Sài Gòn trong tiếng Việt). | `{"intent": "hoi_thong_tin", "confidence": 0.95, "urgency": "trung_binh", "intent_label": "hoi_thong_tin", ...` | **FT THUA**: base 1.00, FT 0.00; JSON triage thay câu trả lời kiến thức. |
| Regression #3 | Viết một câu chúc mừng sinh nhật bằng tiếng Việt. | `["sinh nhật"]` | Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, ... | `{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "product": null, "sentiment": "tich_cuc", "tone": "trang_trang"}` | **FT THUA**: base 1.00, FT 0.00; tự tạo intent thay vì viết lời chúc. |
| Regression #6 | 2 mũ 10 bằng bao nhiêu? | `["1024"]` | Để tính giá trị của $2^{10}$ ... (long derivation, cut before answer) | `{"intent": "math", "answer": "1024", "explanation": "2 mũ 10 có nghĩa là nhân số 2 với chính nó 10 lần: ...` | **FT THẮNG**: base 0.00, FT 1.00; có đáp án nhưng vẫn dạng JSON. |

**Mẫu chung:** FT áp khuôn JSON đã học lên mọi loại đầu vào trong các ví dụ này, kể cả câu hỏi kiến thức, đổi đơn vị và lời chúc sinh nhật. Các ca thua xuất triage JSON với `product=null`, các trường confidence ngoài schema, hoặc tự tạo intent `chuc_mung_sinh_nhat`, thay cho yêu cầu mà base thực hiện đúng. Đây là bằng chứng hành vi trực tiếp của catastrophic forgetting, không còn chỉ là suy luận từ điểm regression trung bình. Regression #6 cho thấy điểm keyword có thể thắng trong khi đầu ra vẫn bị kéo sang JSON; ở đây schema được biến thể với `intent=math` và `answer`, nên không phải mọi JSON đều giữ nguyên schema triage. Các ca target hòa vẫn cho thấy cả (b) và FT nhầm “Khi nào tiện” thành urgency `trung_binh`: target cải thiện không có nghĩa đã hết lỗi tác vụ.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi không triển khai bản fine-tune hiện tại. Mô hình học tốt ticket sang JSON và cải thiện target so với prompt tối ưu, nhưng đi kèm regression vượt gate và latency cao hơn. Mô hình hữu ích cho khách hàng phải giữ năng lực cần thiết ngoài phân loại ticket, nhất là khi đầu vào chứa câu hỏi phổ thông hoặc yêu cầu ngoài miền. `submission/paired_evidence.md` cho bằng chứng trực tiếp: base trả lời đúng các câu đổi đơn vị, số tháng trong năm và tên cũ của thành phố, còn FT thay bằng JSON triage. Ngay yêu cầu chúc sinh nhật cũng bị biến thành intent tự tạo. Đây là biểu hiện catastrophic forgetting quan sát được; corpus đơn nhiệm dạy khuôn đầu ra mạnh nhưng không bảo vệ khả năng làm theo yêu cầu ngoài miền. Ca tính lũy thừa cho thấy FT có thể đúng keyword mà vẫn trả JSON, nên chỉ nhìn điểm không đủ đánh giá hành vi phục vụ thực tế.

Đối chứng LR cho bằng chứng rõ về đòn bẩy cấu hình: giảm thang LR làm target và format cùng thất bại trong ngân sách cố định. Đối chứng vị trí không cho thấy ưu thế target của text-linear so với q,v khi khớp ngân sách; loss thấp hơn ở q,v không đủ tuyên bố nó tốt hơn. Tôi phải tách niềm tin từ tài liệu khỏi kết quả trên model và tác vụ đang xét. Mask có bằng chứng đúng nên không phải nơi ưu tiên sửa để chữa regression; hỗn hợp dữ liệu mới là đòn bẩy cần thử. Tôi sẽ bắt đầu bằng replay dữ liệu phổ thông, giữ mốc eval và prompt tối ưu, kiểm tra cả chất lượng lẫn thời gian phục vụ. Chỉ khi vượt lại regression gate và có ví dụ đầy đủ cho thấy hành vi ngoài miền được bảo toàn, tôi mới cân nhắc triển khai có giám sát.

**Ba điều tôi học được**:

1. Tôi phải giải mã phần được giám sát trước khi train: preview JSON, câu hỏi bị mask và các assert là bằng chứng; tên `assistant-only` tự nó chưa đủ bảo đảm.
2. Tôi không chọn adapter bằng train loss. `attn_only` có mean train loss 0.5369 thấp hơn 0.6257 nhưng cùng target 0.97 với `correct`; khớp ngân sách làm nhận xét về placement đáng tin hơn.
3. Tôi phải giữ năng lực ngoài miền trong định nghĩa thành công. Target 0.97 và format 1.0 vẫn chưa đủ khi regression là 0.5222; các đầu ra ghép cặp trong `paired_evidence.md` cho thấy cụ thể FT thay câu trả lời đúng bằng JSON triage.

**Nếu có thêm hai giờ nữa, tôi sẽ thử:** trộn replay dữ liệu phổ thông theo khuyến nghị lưu trong verdict, giữ nguyên tập eval, rồi đánh giá lại gate. Tôi sẽ lưu đầy đủ đầu ra target và regression của cả (b) lẫn FT để đo liệu replay khắc phục được hành vi ép đầu vào thành JSON đã quan sát, đồng thời xem riêng các ticket nói “Khi nào tiện”. Nếu còn thời gian, tôi thực hiện B4: cố định placement text-linear, quét rank theo rubric, giữ các cấu hình khác và số step như nhau, chọn theo target, regression và latency. Đây là kế hoạch tiếp theo, chưa phải bonus hoàn thành.

---

## Phụ lục — thưởng đã làm

Chưa thực hiện bonus nào; các mục đều chưa đánh dấu.

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link: chưa có
