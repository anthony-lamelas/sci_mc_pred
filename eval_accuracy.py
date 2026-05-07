"""Re-rank every saved checkpoint by *validation accuracy*, not val loss.

Run 5/6 in info/ablations.md showed that lower val loss does NOT imply higher
leaderboard accuracy on this task: the masked-letter cross-entropy can keep
dropping while generalization collapses. This script measures what we
actually care about - per-choice MCQ accuracy on val - and writes it back to
models/training_metrics.csv as a new `Val Accuracy` column.

The base SmolVLM is loaded once and adapters are hot-swapped (same trick as
batch_evaluate.py) so re-evaluating all checkpoints stays cheap.

Examples:
    # Score every adapter that doesn't yet have a Val Accuracy entry
    python eval_accuracy.py

    # Quick run on a 200-sample subset (good for in-loop probing decisions)
    python eval_accuracy.py --limit 200

    # Force re-evaluation even if accuracy is already populated
    python eval_accuracy.py --force
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

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
)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_DIR = Path("data")
METRICS_CSV = Path("models/training_metrics.csv")
ACCURACY_COL = "Val Accuracy"


def discover_checkpoints(search_pattern: str = "models/*/checkpoints/*") -> list[dict]:
    """Find every adapter directory and parse its (model_name, epoch, step)."""
    checkpoints = [c for c in glob.glob(search_pattern) if os.path.isdir(c)]
    parsed: list[dict] = []
    for ckpt in checkpoints:
        parts = ckpt.split(os.sep)
        if len(parts) < 4:
            continue
        model_name = parts[1]
        ckpt_name = parts[3]

        epoch_match = re.search(r"epoch_(\d+)", ckpt_name)
        if not epoch_match:
            continue
        step_match = re.search(r"step_(\d+)", ckpt_name)

        epoch = int(epoch_match.group(1))
        step = str(step_match.group(1)) if step_match else "Final"
        parsed.append(
            {"path": ckpt, "model_name": model_name, "epoch": epoch, "step": step}
        )

    parsed.sort(
        key=lambda c: (
            c["model_name"],
            c["epoch"],
            10**9 if c["step"] == "Final" else int(c["step"]),
        )
    )
    return parsed


def load_metrics_df() -> pd.DataFrame:
    if METRICS_CSV.exists():
        df = pd.read_csv(METRICS_CSV)
    else:
        df = pd.DataFrame(
            columns=["Model Name", "Epoch", "Step", "Train Loss", "Val Loss"]
        )
    if ACCURACY_COL not in df.columns:
        df[ACCURACY_COL] = pd.NA
    return df


def existing_accuracy(
    df: pd.DataFrame, model_name: str, epoch: int, step: str
) -> float | None:
    mask = (
        (df["Model Name"].astype(str) == model_name)
        & (df["Epoch"].astype(int) == epoch)
        & (df["Step"].astype(str) == step)
    )
    if not mask.any():
        return None
    val = df.loc[mask, ACCURACY_COL].iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def upsert_accuracy(
    df: pd.DataFrame, model_name: str, epoch: int, step: str, accuracy: float
) -> pd.DataFrame:
    mask = (
        (df["Model Name"].astype(str) == model_name)
        & (df["Epoch"].astype(int) == epoch)
        & (df["Step"].astype(str) == step)
    )
    if mask.any():
        df.loc[mask, ACCURACY_COL] = round(accuracy, 4)
    else:
        new_row = {
            "Model Name": model_name,
            "Epoch": epoch,
            "Step": step,
            "Train Loss": pd.NA,
            "Val Loss": pd.NA,
            ACCURACY_COL: round(accuracy, 4),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N val examples (faster, noisier).",
    )
    p.add_argument(
        "--mode",
        choices=["full", "letter"],
        default="full",
        help="Per-choice scoring mode (see eval_utils.predict_answer).",
    )
    p.add_argument(
        "--no-length-norm",
        action="store_true",
        help="Disable length normalization of per-choice log-likelihoods.",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable bitsandbytes 4-bit quantization even on CUDA.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-score checkpoints even if Val Accuracy is already populated.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image resize before processor (224 matches training).",
    )
    return p.parse_args()


def build_base_model(prefer_4bit: bool):
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
            MODEL_ID, dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(device)
    return model


def main() -> None:
    args = parse_args()
    print("[1/4] Discovering checkpoints...")
    checkpoints = discover_checkpoints()
    if not checkpoints:
        print("No checkpoints found under models/*/checkpoints/*; nothing to do.")
        return
    print(f"  Found {len(checkpoints)} adapter checkpoints.")

    print("[2/4] Filtering already-scored checkpoints...")
    metrics_df = load_metrics_df()
    if args.force:
        to_eval = checkpoints
    else:
        to_eval = [
            c
            for c in checkpoints
            if existing_accuracy(metrics_df, c["model_name"], c["epoch"], c["step"])
            is None
        ]
    if not to_eval:
        print("All checkpoints already have Val Accuracy populated. Use --force to redo.")
        return
    print(f"  {len(to_eval)} checkpoints to evaluate.")

    print("[3/4] Loading validation set + base model...")
    val_df = load_split(DATA_DIR, "val")
    if args.limit is not None:
        val_df = val_df.head(args.limit)
    img_size = args.img_size if args.img_size > 0 else None
    val_ds = ScienceQADataset(val_df, DATA_DIR, img_size=img_size)

    base_model = build_base_model(prefer_4bit=not args.no_4bit)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, to_eval[0]["path"])
    model.eval()

    print(f"[4/4] Scoring {len(to_eval)} checkpoint(s) on {len(val_ds)} val examples...")
    for ckpt in to_eval:
        print(
            f"\n--> {ckpt['model_name']} | Epoch {ckpt['epoch']} | Step {ckpt['step']}"
        )
        model.load_adapter(ckpt["path"], adapter_name="default")
        model.set_adapter("default")

        preds: list[int] = []
        truths: list[int] = []
        for item in tqdm(val_ds, desc="Scoring", leave=False):
            pred, _ = predict_answer(
                model,
                processor,
                item["image"],
                item["row"],
                mode=args.mode,
                length_normalize=not args.no_length_norm,
            )
            preds.append(int(pred))
            truths.append(int(item["row"]["answer"]))

        correct, total, acc = grade_predictions(preds, truths)
        print(f"  Val Accuracy: {correct}/{total} = {acc:.4f}")

        metrics_df = upsert_accuracy(
            metrics_df, ckpt["model_name"], ckpt["epoch"], ckpt["step"], acc
        )
        metrics_df.to_csv(METRICS_CSV, index=False)

    print("\nDone. Best checkpoints by Val Accuracy:")
    ranked = metrics_df.dropna(subset=[ACCURACY_COL]).sort_values(
        ACCURACY_COL, ascending=False
    )
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
