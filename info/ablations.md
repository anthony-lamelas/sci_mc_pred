# ScienceQA Model Ablations

This document tracks the experiments, configurations, and results for the `SmolVLM-500M-Instruct` model on the ScienceQA visual challenge dataset.

---

## 1. Baseline Run (Zero-Shot / No Fine-tuning)
- **Model:** `HuggingFaceTB/SmolVLM-500M-Instruct`
- **Setup:** Zero-shot evaluation using the base instruction-tuned visual-language model. Batch size of 4 for inference on Google Colab T4 (Free Tier).
- **Issue Discovered:** Right-padding was implicitly used during batch generation.
- **Result:** **54.3% Accuracy**

---

## 2. Training Run 1 (Basic QLoRA)
- **Goal:** Improve baseline via supervised fine-tuning while adhering to strict < 5M trainable parameter competition limits.
- **Hyperparameters:**
  - `r` (Rank): 6
  - `lora_alpha`: 16
  - `target_modules`: `"all-linear"`
  - `Precision`: 4-bit (`nf4` with double quant)
  - `Batch Size`: 1 (with 8 Gradient Accumulation Steps)
  - `Epochs`: 2
  - `Trainable Parameters`: 4,331,136 (Compliant)
- **Evaluation Details:** 
  - Cross-Entropy Loss was calculated over the *entire* text sequence (Images + Instruction Prompt + Answers).
  - Inference was still run with right-padding.
- **Result:** **~37% Accuracy**
- **Analysis (Why the drop?):** 
  - **Decoder Padding Failure:** Because batch inference used right-padding, shorter sequences ended in `[PAD]` rather than the `"Answer:"` token. The decoder predicting off padding generated garbage.
  - **Capacity Dilution:** Because the entire string was scored, the severely bottlenecked 4.3M parameter adapter spent a large margin of its capacity trying to learn the phrasing of the questions rather than just the logical reasoning of the multiple-choice tokens.

---

## 3. Training Run 2 (Prompt Masking + Left Padding)
- **Goal:** Resolve issues from Run 1 via mathematical masking and correct inference padding.
- **Adjustments:**
  - `padding_side='left'` forced on the tokenizer before inference to push sequence ends dynamically back to the actual prompt text.
  - `-100` label masking applied to all instruction/question context inside the `DataLoader`, restricting the Cross-Entropy loss gradients *exclusively* to the final answer tokens (e.g. `[A, B, C, D]`). 
- **Results:** 
  - **Epoch 1:** Train Loss (0.3154) | Val Loss (0.7562)
  - **Epoch 2:** Train Loss (0.2163) | Val Loss (0.9002)
  - **Submission File:** `submission_20260427_112131.csv`
  - **Final Leaderboard Accuracy:** **69%** (Best yet!)
- **Analysis:** 
  - The massive increase from ~37% to 69% proves the prompt masking algorithm, left-padding adjustments, and probability logit extraction heavily salvaged the model's logic capabilities!
  - **Optimal Weights:** Evaluating the CSV metric logs revealed the lowest true validation loss (0.3115) belonged to `Epoch 2, Step 3108`. We bypassed the traditional string-generation pipeline by using `torch.argmax` on raw target logits to pull out the correct multiple-choice probability.

---

## 4. Training Run 3 (Improved Inference Decoding)
- **Goal:** Enhance inference accuracy by switching from logit-based token extraction to text-based decoding with longer generation.
- **Adjustments:**
  - Switched inference to generate `max_new_tokens=2` and decode full text output instead of extracting first token logits.
  - Used text parsing to extract answer letters from generated responses (e.g., "Answer: A").
  - Loaded the final trained checkpoint (`models/20260426_233953/final`) instead of epoch 1.
  - Maintained left-padding and prompt masking from Run 2.
- **Results:** 
  - **Final Leaderboard Accuracy:** **74%**
- **Analysis:** 
  - The shift to text-based decoding allowed the model to generate more natural responses, improving accuracy by 5% over the logit extraction method.
  - This demonstrates that for instruction-tuned models, full generation can outperform direct logit manipulation in multiple-choice tasks.

---

## 5. Training Run 4 (Batch Inference with Memory Fixes)
- **Goal:** Optimize for free-tier Colab by reducing memory usage and enabling batch processing.
- **Adjustments:**
  - Reduced `batch_size` to 2 (later 1) and `num_workers` to 1 to avoid OOM.
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for better GPU memory management.
  - Added `torch.cuda.empty_cache()` after each batch.
  - Used `num_beams=1` (greedy decoding) instead of beam search.
  - Improved decoding to extract only generated tokens rather than the full prompt + output sequence.
- **Results:** 
  - **Final Leaderboard Accuracy:** **74.446%** (same as previous best)
- **Analysis:** 
  - **Memory Fixes Worked:** The OOM and parsing issues were corrected, but the final accuracy did not improve, which means the current checkpoint and decoding pipeline were already at the same performance level.
  - **Earlier Checkpoint Match:** An earlier checkpoint in epoch 2 produced **74.25%**, effectively identical to the best result and confirming the model had already plateaued by the end of training.
  - **No Improvement from Greedy/Batch Tuning:** This suggests the remaining gains are not from low-level generation settings like batch size, beams, or padding; instead, they are more likely in prompt formatting, answer extraction, or model scoring.
  - **Next Run Plan:** Match the training and inference prompt formats exactly, add stronger parsing for generated choices, lower the learning rate, and skip data augmentation for now.
  - **Epoch Recommendation:** Run for **3 epochs** with checkpointing and select the best validation checkpoint, since epoch 2 already flattened and the final checkpoint showed no gain.

---

## 6. Training Run 5 (Lower LR + Matched Prompt Format + Extended Training)
- **Goal:** Improve on plateau by lowering learning rate, matching prompt format exactly between training/inference, and extending to 3 epochs.
- **Hyperparameters:**
  - `lr`: 1e-4 (halved from 2e-4)
  - `EPOCHS`: 3
  - Prompt format: `"Answer: "` with trailing space (exact match between training target and inference)
  - Stronger answer parsing: prioritize `ANSWER: X` pattern, fall back to bare letter, then strip non-letters
- **Training Results:**
  - **Epoch 1:** Val Loss 0.3045 (at step 3108)
  - **Epoch 2:** Val Loss 0.2877 (at step 3108) — **Best validation loss yet!**
  - **Epoch 3:** Val Loss 0.2941 → 0.2967 (clear overfitting; train loss near 0)
- **Initial Inference Result:** **44.8% Accuracy**
- **Root Cause Analysis:**
  - **Training Success, Inference Failure:** The new model trained *better* (lower val loss) than all previous runs.
  - **Checkpoint Mismatch Bug:** The 44.8% result used the *old checkpoint* (`20260426_233953`) instead of the new trained model, causing a train/inference format mismatch.
---

## 6. Training Run 6 (Checkpoint Fix + Overfitting Investigation)
- **Goal:** Use the correct new checkpoint from Run 5 to evaluate the improved training.
- **Adjustments:**
  - Loaded the best checkpoint from Run 5: `models/20260429_183205/checkpoints/epoch_2_step_3108` (val loss 0.2877)
  - Maintained all other settings: LR=1e-4, EPOCHS=3, matched prompt format, stronger parsing.
- **Results:** 
  - **Final Leaderboard Accuracy:** **50.9%**
- **Analysis (Why worse than baseline?):**
  - **Overfitting Suspected:** Despite lower validation loss (0.2877), the test accuracy dropped significantly below the zero-shot baseline (54.3%). This suggests the model overfit to the training data, losing generalization ability.
  - **Possible Causes:**
    - **Training Data Overfitting:** With only ~10k training samples and a small model (4.3M params), the adapter may have memorized training patterns instead of learning robust reasoning.
    - **Checkpoint Selection Issue:** Although epoch 2 step 3108 had the lowest val loss, it might still be overfit compared to earlier checkpoints or the base model.
    - **Prompt/Format Sensitivity:** The exact prompt matching might have made the model too rigid, reducing flexibility on unseen test questions.
  - **Next Steps:** Try earlier checkpoints (e.g., epoch 1), add regularization (dropout, weight decay), or reduce training epochs to prevent overfitting.

## 7. Inference Sweep Update
- Replaced the validation tuning cell with a single test-only inference sweep.
- Each config now writes its own CSV immediately after test inference.
- This avoids waiting for validation config tuning before generating submission files.

---

## 8. Training Run 7 (DoRA + Augmentation + Metadata + 336px)
- **Goal:** Recover from the severe overfitting seen in Runs 5/6 and implement easy performance enhancements.
- **Adjustments:**
  - Increased `IMG_SIZE` from 224 to 336 for better visual parsing.
  - Upgraded LoRA to DoRA (`use_dora=True`).
  - Added Image Data Augmentation (`RandomCrop`, `RandomRotation`, `ColorJitter`) to the training split.
  - Injected `Subject` and `Topic` metadata into the prompt and moved the `Question:` before the context.
  - Evaluated on intermediate checkpoint: `epoch_2_step_2072` (Val Loss: ~0.375).
- **Results:** 
  - **Final Leaderboard Accuracy:** **65.0%**
- **Analysis:** 
  - The combination of image size, metadata, and DoRA helped the model recover from the 50.9% overfitting drop, though it hasn't quite surpassed the 74.4% peak from Run 4. 
  - The heavy regularization from the data augmentation makes it harder for the model to memorize the training set. This suggests we should allow the model to finish its 3rd training epoch, as it likely needs more steps to converge with these augmentations active!
