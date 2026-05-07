

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

CHOICE_LETTERS = "ABCDEFGHIJ"


def build_prompt(row: pd.Series, include_answer: bool = False) -> str:
    """Build the SmolVLM text prompt.

    Format must match starter_notebook.ipynb so existing checkpoints stay valid:
    `<image>\\nContext:...\\n\\nQuestion:...\\nChoices:\\n  A. ...\\n  B. ...\\nAnswer: `
    The trailing space after `Answer:` matters (it's part of training targets).
    """
    context_parts: list[str] = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)

    choices = row["choices"]
    choices_str = "\n".join(
        f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)
    )

    prompt = "<image>\n"
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {row['question']}\nChoices:\n{choices_str}\nAnswer: "

    if include_answer:
        prompt += CHOICE_LETTERS[int(row["answer"])]
    return prompt


class ScienceQADataset(Dataset):
    """Returns raw PIL images alongside the source row for flexible scoring.

    `data_dir` is where the CSVs (and conventionally the `images/` folder)
    live. If your images are attached under a different path (common on
    Kaggle when CSVs and images come from separate Datasets), pass
    `image_dir` explicitly; it defaults to `data_dir`.

    `img_size=224` matches the resize used during training of the existing
    checkpoints, so we keep it as the default for inference to avoid
    distribution shift. Pass `img_size=None` to skip the manual resize and
    let the processor handle preprocessing natively (preferred for retraining
    on higher-resolution inputs).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_size: int | None = 224,
        image_dir: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.image_dir = Path(image_dir) if image_dir is not None else data_dir
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        img = Image.open(self.image_dir / rel_path).convert("RGB")
        if self.img_size is not None:
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return img

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "id": row["id"],
            "image": self._load_image(row["image_path"]),
            "row": row,
        }


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load train/val/test CSV and parse the `choices` JSON column."""
    df = pd.read_csv(data_dir / f"{split}.csv")
    df["choices"] = df["choices"].apply(json.loads)
    return df


@torch.inference_mode()
def score_choice_loglik(
    model,
    processor,
    image: Image.Image,
    prompt_prefix: str,
    target_text: str,
) -> Tuple[float, int]:
    """Sum log P(target | preceding context) under the model.

    Robust to BPE merging across the prefix/target boundary: SmolVLM's
    tokenizer can merge a trailing space in the prefix with the first
    character of the target into a single token (e.g. "Answer: " + "A"
    becoming "...:"," A" instead of "...:", " ", "A"). We find the first
    position where the prefix-only and prefix+target token sequences
    diverge and score everything from that position onward, which gives
    the correct conditional log-likelihood regardless of merging.

    Returns (total_log_prob, num_target_tokens_scored).
    """
    prefix_inputs = processor(
        text=[prompt_prefix], images=[image], return_tensors="pt"
    )
    full_inputs = processor(
        text=[prompt_prefix + target_text], images=[image], return_tensors="pt"
    )

    prefix_ids = prefix_inputs["input_ids"][0].tolist()
    full_ids = full_inputs["input_ids"][0].tolist()
    full_len = len(full_ids)

    target_start = len(prefix_ids)
    for i, (p_tok, f_tok) in enumerate(zip(prefix_ids, full_ids)):
        if p_tok != f_tok:
            target_start = i
            break

    n_target = full_len - target_start
    if n_target <= 0 or target_start == 0:
        return 0.0, 0

    full_inputs = {
        k: (v.to(model.device) if torch.is_tensor(v) else v)
        for k, v in full_inputs.items()
    }
    outputs = model(**full_inputs)
    logits = outputs.logits[0]  # [L, V]

    # Position p in `logits` predicts the token at position p+1 of `input_ids`.
    target_logits = logits[target_start - 1 : full_len - 1]
    target_ids = full_inputs["input_ids"][0, target_start:full_len]

    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    token_lls = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_lls.sum().item()), int(n_target)


@torch.inference_mode()
def predict_answer(
    model,
    processor,
    image: Image.Image,
    row: pd.Series,
    mode: str = "full",
    length_normalize: bool = True,
) -> Tuple[int, List[float]]:
    """Predict the answer index for one example via per-choice log-likelihood.

    Args:
        mode: "full"   -> score "{letter}. {choice text}" (best signal)
              "letter" -> score "{letter}" only (fast; calibrated alternative
                          to first-token logit argmax)
        length_normalize: divide each choice's log-likelihood by its token
            count to remove length bias. Recommended when `mode="full"`.

    Returns (predicted_index, per_choice_scores).
    """
    prompt_prefix = build_prompt(row, include_answer=False)  # ends with "Answer: "
    scores: list[float] = []

    for i, choice_text in enumerate(row["choices"]):
        if mode == "letter":
            target = CHOICE_LETTERS[i]
        elif mode == "full":
            target = f"{CHOICE_LETTERS[i]}. {choice_text}"
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")

        ll, n_tok = score_choice_loglik(
            model, processor, image, prompt_prefix, target
        )
        scores.append(ll / max(n_tok, 1) if length_normalize else ll)

    return int(np.argmax(scores)), scores


@torch.inference_mode()
def predict_answer_tta(
    model,
    processor,
    image: Image.Image,
    row: pd.Series,
    n_shuffles: int = 4,
    mode: str = "full",
    length_normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> Tuple[int, List[float]]:
    """Choice-shuffle test-time augmentation.

    For each of (1 original + n_shuffles random) permutations, score the
    choices and accumulate the score back onto each choice's *original*
    index. The model's letter-position bias gets averaged out across
    permutations.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_choices = len(row["choices"])
    perms: list[np.ndarray] = [np.arange(n_choices)]
    for _ in range(max(n_shuffles, 0)):
        perm = np.arange(n_choices)
        rng.shuffle(perm)
        perms.append(perm)

    aggregated = np.zeros(n_choices, dtype=np.float64)
    for perm in perms:
        permuted_row = row.copy()
        permuted_row["choices"] = [row["choices"][p] for p in perm]
        _, scores = predict_answer(
            model,
            processor,
            image,
            permuted_row,
            mode=mode,
            length_normalize=length_normalize,
        )
        for new_idx, original_idx in enumerate(perm):
            aggregated[original_idx] += scores[new_idx]

    return int(np.argmax(aggregated)), aggregated.tolist()


def pick_device_dtype(prefer_4bit: bool = True) -> Tuple[str, torch.dtype, bool]:
    """Pick the best available device, dtype, and whether to enable 4-bit.

    On Apple Silicon / CPU, bitsandbytes 4-bit kernels are unavailable, so we
    fall back to fp16 (MPS) or fp32 (CPU). On CUDA we use 4-bit if requested.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16, prefer_4bit
    if torch.backends.mps.is_available():
        return "mps", torch.float16, False
    return "cpu", torch.float32, False


def grade_predictions(
    predictions: Sequence[int], truths: Sequence[int]
) -> Tuple[int, int, float]:
    """Return (correct, total, accuracy)."""
    total = len(predictions)
    correct = sum(int(p == t) for p, t in zip(predictions, truths))
    acc = correct / total if total else 0.0
    return correct, total, acc
