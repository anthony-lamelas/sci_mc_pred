1. Expand LoRA targets to MLP layers — Currently LoRA only adapts attention projections
(q/k/v/o). Adding MLP layers (gate/up/down_proj) lets the model adapt its internal
representations, not just how it attends. Research shows MLP-only LoRA can outperform
attention-only even at higher rank. Reduce rank to fit the 5M param budget.
2. Upgrade LoRA to DoRA — DoRA decomposes weight updates into magnitude and
direction components, giving better fine-tuning quality than standard LoRA with zero extra
inference cost. A config-level change with documented gains on VLM benchmarks.
3. Add self-generated image captions to prompts — Use the model itself to describe each
image, then feed that description back as extra text context during training and inference. Small
VLMs struggle with raw visual extraction;
text bridges the gap.
4. Try targeting both attention and MLP layers (q_proj, v_proj, gate_proj, up_proj, down_proj) at
a lower rank, while keeping param count under 5M.
5. Experiment with changes like "The correct answer is:" vs "Answer:" or reordering context vs
question.
6. You have metadata columns like subject, grade, and topic. Try adding one of them to your
prompt and see if it helps the model reason better.
7. Your image is 224×224, zoom in on a few. If you can't read the axis labels or map text at that
resolution, neither can your model. Try a bigger image size.
8. Data augmentation for images — Random crop, slight rotation, brightness jitter. Helps the
model generalize beyond the exact 224×224 framing.
9. Gradient checkpointing — increase batch size or image resolution without OOM, trading
compute for memory