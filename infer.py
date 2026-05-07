"""Per-choice log-likelihood inference for ScienceQA Visual Challenge.

Replaces the first-token logit argmax / text-decoding approach with proper
per-choice scoring: for each candidate answer, compute P(answer | image,
prompt) under the model and pick the argmax. Supports choice-shuffle TTA.

Examples:
    # Generate a test submission with the current best adapter
    python infer.py --adapter models/20260426_233953/final --split test

    # Measure per-choice scoring accuracy on val (no submission written)
    python infer.py --adapter models/20260426_233953/final --split val --no-save

    # 4-shuffle TTA on val to estimate the upper bound
    python infer.py --split val --tta-shuffles 4 --no-save

    # Quick smoke test on the first 20 val examples (Mac-friendly)
    python infer.py --split val --limit 20 --no-save
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor

# Class was renamed AutoModelForVision2Seq -> AutoModelForImageTextToText in
# transformers 4.45 and the old name was removed in v5. Try the new name
# first so this works on whatever version the runtime ships.
try:
    from transformers import AutoModelForImageTextToText as _AutoVisionModel
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoVisionModel

from eval_utils import (
    ScienceQADataset,
    grade_predictions,
    load_split,
    pick_device_dtype,
    predict_answer,
    predict_answer_tta,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_DIR = Path("data")
SUBMISSIONS_DIR = Path("submissions")


def load_model_and_processor(
    adapter_path: str | None,
    prefer_4bit: bool = True,
):
    """Load SmolVLM with optional QLoRA adapter, auto-selecting precision.

    On CUDA: 4-bit nf4 quantization (if `prefer_4bit`) for low memory.
    On MPS / CPU: skip bitsandbytes (its 4-bit kernels are CUDA-only) and
    fall back to fp16 / fp32.
    """
    device, dtype, use_4bit = pick_device_dtype(prefer_4bit=prefer_4bit)
    print(f"Device: {device} | dtype: {dtype} | 4-bit: {use_4bit}")

    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = _AutoVisionModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = _AutoVisionModel.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--adapter",
        type=str,
        default="models/20260426_233953/final",
        help="Path to a PEFT/LoRA adapter directory. Use empty string for base model.",
    )
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument(
        "--mode",
        choices=["full", "letter"],
        default="full",
        help="'full' scores '{letter}. {choice_text}'; 'letter' scores just the letter.",
    )
    p.add_argument(
        "--no-length-norm",
        action="store_true",
        help="Disable length normalization of per-choice log-likelihoods.",
    )
    p.add_argument(
        "--tta-shuffles",
        type=int,
        default=0,
        help="Number of additional shuffled choice orderings to average over.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Resize input images to NxN before passing to the processor. "
        "224 matches existing checkpoints; pass 0 to skip the manual resize.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for quick smoke tests).",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable bitsandbytes 4-bit quantization even on CUDA.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. If omitted, auto-named under submissions/.",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the submission CSV (useful for val-only eval runs).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def auto_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = ["loglik", args.mode]
    if args.tta_shuffles > 0:
        parts.append(f"tta{args.tta_shuffles}")
    if args.no_length_norm:
        parts.append("nonorm")
    tag = "_".join(parts)
    return SUBMISSIONS_DIR / f"submission_{tag}_{timestamp}.csv"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    model, processor = load_model_and_processor(
        adapter_path=args.adapter or None,
        prefer_4bit=not args.no_4bit,
    )

    df = load_split(DATA_DIR, args.split)
    if args.limit is not None:
        df = df.head(args.limit)
    img_size = args.img_size if args.img_size > 0 else None
    ds = ScienceQADataset(df, DATA_DIR, img_size=img_size)

    has_labels = args.split in {"train", "val"}
    predictions: list[dict] = []
    truths: list[int] = []
    preds_only: list[int] = []

    desc = f"Scoring {args.split}"
    if args.tta_shuffles > 0:
        desc += f" (TTA x{args.tta_shuffles + 1})"

    for item in tqdm(ds, desc=desc):
        if args.tta_shuffles > 0:
            pred, _ = predict_answer_tta(
                model,
                processor,
                item["image"],
                item["row"],
                n_shuffles=args.tta_shuffles,
                mode=args.mode,
                length_normalize=not args.no_length_norm,
                rng=rng,
            )
        else:
            pred, _ = predict_answer(
                model,
                processor,
                item["image"],
                item["row"],
                mode=args.mode,
                length_normalize=not args.no_length_norm,
            )

        predictions.append({"id": item["id"], "answer": int(pred)})
        preds_only.append(int(pred))
        if has_labels:
            truths.append(int(item["row"]["answer"]))

    if has_labels:
        correct, total, acc = grade_predictions(preds_only, truths)
        print(f"\n{args.split} accuracy: {correct}/{total} = {acc:.4f}")

    if args.no_save:
        return

    out_path = Path(args.out) if args.out else auto_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
