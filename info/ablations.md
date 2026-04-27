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
  - **Final Leaderboard Accuracy:** **~65%** (Best yet!)
- **Analysis:** 
  - The massive increase from ~37% to ~65% proves the prompt masking algorithm and left-padding inference adjustments heavily salvaged the model's logic capabilities!
  - **Overfitting Warning:** While training loss plummeted in Epoch 2, Validation Loss spiked. This heavily indicates the adapter weights began overfitting on the training data. The optimal mathematical setup is likely sitting natively inside an `Epoch 1` mid-step checkpoint!
