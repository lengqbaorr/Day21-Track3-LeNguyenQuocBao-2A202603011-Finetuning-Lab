---
base_model: unsloth/Qwen3.5-4B
library_name: peft
language: vi
tags:
  - lora
  - peft
---

# Lab 21 — Vietnamese customer-ticket LoRA

**CẢNH BÁO / WARNING: REGRESSION GATE FAILED — KHÔNG DÙNG TRIỂN KHAI / NOT FOR DEPLOYMENT.**

Adapter cải thiện phân loại ticket nhưng làm suy giảm năng lực phổ thông. Các ví dụ ghép cặp cho thấy câu hỏi về đổi đơn vị, số tháng và kiến thức chung bị trả lời bằng JSON triage thay vì đáp án; lời chúc sinh nhật còn bị biến thành intent tự tạo. Đây là bằng chứng catastrophic forgetting. Chỉ dùng để học tập, nghiên cứu và phân tích đối chứng; không dùng cho khách hàng hoặc thay thế một trợ lý phổ thông.

The adapter improves ticket classification but fails the regression gate. Paired examples show general questions being answered with triage JSON instead of the requested answer, including an invented birthday intent. This is observed catastrophic forgetting. This artifact is for educational analysis and research, not production deployment or general assistance.

## Task / Tác vụ

Ticket chăm sóc khách hàng tiếng Việt → JSON với đúng các trường `intent`, `urgency`, `product`, `sentiment`. Corpus mặc định của lab có 250 mẫu, split 225 train / 25 validation, seed 42; đáp án không chứa reasoning traces.

Vietnamese customer-support tickets → JSON with four fields: `intent`, `urgency`, `product`, and `sentiment`. The default lab corpus has 250 examples, split into 225 training and 25 validation examples with seed 42, without reasoning traces.

## Training / Huấn luyện

Adapter `correct`: LoRA all-linear, placement `text-linear`, gắn vào 12 loại module thuộc phần text của base `unsloth/Qwen3.5-4B`. Huấn luyện 30 steps trên Colab Tesla T4 với `fp16`, không dùng `bf16` và không lượng tử hóa base. Loss mask là `assistant-only`.

The `correct` adapter uses all-linear LoRA over 12 text-linear module types, trained for 30 steps on a Colab Tesla T4 in `fp16`, with an unquantized base and assistant-only loss masking.

| Setting / Cấu hình | Value / Giá trị |
|---|---|
| Rank `r` | 16 |
| LoRA alpha | 32 |
| Learning rate | 0.0001 (`1e-4`) |
| Trainable parameters | 32464896 |
| Training steps | 30 |
| Precision | fp16 |
| Placement / module types | text-linear / 12 |
| Training time (s) | 396.0 |
| Mean training loss (`final_loss`) | 0.6257 |
| Peak VRAM (GB) | 8.78 |

Các số ở bảng cấu hình chép từ dòng `correct` của `results/runs.csv`; rank và alpha cũng khớp `adapters/correct/adapter_config.json`. `final_loss` là mean loss của toàn run, không phải last-step loss.

Configuration values come from the `correct` row in `results/runs.csv`. The `final_loss` field denotes mean training loss over the run, not the final step's loss.

## Evaluation / Đánh giá

Chép nguyên `results/verdict.json:comparison`, trên full eval: 50 target và 15 regression. Baseline (b) là base với optimized prompt; FT dùng prompt ngắn của lab. Không sửa `OPTIMIZED_PROMPT`.

Values below are copied exactly from `results/verdict.json:comparison`. Evaluation uses all 50 target and 15 regression examples. Baseline (b) uses the optimized prompt; FT uses the lab's short prompt.

| Metric | Optimized-prompt baseline (b) | LoRA fine-tune (c) |
|---|---|---|
| Target | 0.765 | 0.97 |
| Regression | 0.7911 | 0.5222 |
| Format | 1.0 | 1.0 |
| Latency (ms) | 1008.5 | 1388.9 |

`verdict.passed=false`; `target_delta=0.20499999999999996`, `regression_delta=-0.26888888888888884`, `valid_trace_rate=0.0`. Lý do gate lưu sẵn:

```text
general capability regressed by 0.269 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.
```

Regression dùng keyword recall, có thể cho điểm khi đáp án nằm trong JSON dù hành vi không làm đúng yêu cầu giao tiếp. Vì vậy điểm regression có thể đánh giá cao hơn năng lực phổ thông còn giữ lại. `valid_trace_rate=0.0` là dự kiến với corpus không có traces, không tự chứng minh reasoning collapse. Cần thử replay dữ liệu phổ thông và đánh giá lại gate trước khi cân nhắc ứng dụng.

Regression is measured by keyword recall, which can reward a correct keyword inside unwanted JSON. The score may overstate retained general ability. A zero valid-trace rate is expected for this corpus without reasoning traces. Replay data and renewed regression evaluation are needed before considering any application.

## Offline research usage / Ví dụ nghiên cứu cục bộ

Ví dụ dùng `PeftModel` dưới đây chỉ đọc model và adapter đã có trên đĩa. Cần môi trường PyTorch, Transformers, PEFT và Accelerate tương thích cùng GPU phù hợp; thay `base_dir` bằng thư mục chứa đầy đủ base model. Chạy từ thư mục gốc repo. Đoạn mã này chưa được thực thi trong tác vụ chuẩn bị model card.

This example loads an existing local base-model directory and local adapter only. It requires a compatible GPU environment and installed dependencies. It is an illustrative research snippet and was not executed while preparing this card.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_dir = "path/to/local/Qwen3.5-4B"
adapter_dir = "adapters/correct"

tokenizer = AutoTokenizer.from_pretrained(
    adapter_dir, local_files_only=True, trust_remote_code=True
)
base = AutoModelForCausalLM.from_pretrained(
    base_dir,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(
    base, adapter_dir, local_files_only=True
).eval()

messages = [
    {"role": "system", "content": "Phân loại ticket sau."},
    {"role": "user", "content": "Shop ơi, đơn hàng bàn phím cơ giao chậm. Nhờ shop kiểm tra."},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=96,
        pad_token_id=tokenizer.pad_token_id,
    )
answer = tokenizer.decode(
    output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
)
print(answer)
```

## Provenance and publication status / Nguồn và trạng thái

Tác giả / Author: Le Nguyen Quoc Bao, MSSV 2A202603011.

- [Source repository / Repo nguồn](https://github.com/lengqbaorr/Day21-Track3-LeNguyenQuocBao-2A202603011-Finetuning-Lab), graded-run commit `d27c1c0`.
- Official metrics: `results/verdict.json`, `results/runs.csv`.
- Diagnostic evidence: `submission/paired_evidence.md`, `submission/qualitative_paired.json`, `submission/training_log_excerpt.md`.
- [Intended HF destination / URL HF dự kiến](https://huggingface.co/lnqbaor/lab21-qwen3.5-4b-ticket-lora).

Model card này được chuẩn bị cục bộ; không upload adapter, không kiểm tra sự tồn tại của repo HF và không khẳng định đã xuất bản. This card is prepared locally; no adapter was uploaded and no remote publication was verified.
