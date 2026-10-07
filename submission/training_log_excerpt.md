# Training log excerpt (Colab T4, graded run, `scripts/colab_run.py nb1..nb5`)

Copied verbatim from the trainer's `logging_steps=5` lines. 30 optimizer steps per run.
`final_loss` in `results/runs.csv` = `result.training_loss` = MEAN training loss over the
whole run (notebooks/03 line 170, notebooks/04 line 110), not the last logged step.

| step (epoch) | correct (LR 1e-4) | attn_only (r=283) | wrong_lr (LR 1e-5) | qlora (4-bit) |
|---|---|---|---|---|
| 5 (0.36)  | 2.163   | 2.163   | 2.163 | 2.155   |
| 10 (0.71) | 1.38    | 0.8241  | 2.066 | 1.731   |
| 15 (1.00) | 0.1385  | 0.1481  | 1.606 | 0.2408  |
| 20 (1.36) | 0.02846 | 0.04006 | 1.326 | 0.05115 |
| 25 (1.71) | 0.0177  | 0.02135 | 1.141 | 0.0308  |
| 30 (2.00) | 0.02638 | 0.02444 | 1.119 | 0.02622 |
| **mean = `final_loss`** | 0.6257 | 0.5369 | 1.5702 | 0.7058 |

mean_token_accuracy at step 30: correct 0.9957 · attn_only 0.9942 · wrong_lr 0.7909 · qlora 0.9915

`grad_norm` is logged as `nan` at some steps (correct: steps 5 and 30; wrong_lr: 5, 15, 30;
qlora: 5): fp16 GradScaler found inf/nan gradients and skipped that update, then lowered
the loss scale. Loss keeps falling afterwards, so training was not derailed.

NB3 also printed: `mask_mode = assistant-only supervised 9014/20951 tokens (43.0%)` and
`layer_types: {linear_attention: 24, full_attention: 8}`; QLoRA run printed
`precision fix: recast 496/496 trainable tensors bf16 -> fp32 for the fp16 GradScaler`.
