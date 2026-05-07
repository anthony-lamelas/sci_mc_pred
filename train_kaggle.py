
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

# Class was renamed AutoModelForVision2Seq -> AutoModelForImageTextToText in
# transformers 4.45 and the old name was removed in v5. Try the new name first
# so the script works on whatever transformers version the runtime ships.
try:
    from transformers import AutoModelForImageTextToText as _AutoVisionModel
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoVisionModel

from eval_utils import (
    CHOICE_LETTERS,
    ScienceQADataset,
    build_prompt,
    grade_predictions,
    load_split,
    predict_answer,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


# ---------------------------------------------------------------------------
# Path auto-detection
# ---------------------------------------------------------------------------

def _looks_like_data_root(p: Path) -> bool:
    """A data root has train.csv AND val.csv together (more discriminating
    than train.csv alone, which can collide with stray files in dataset uploads).
    """
    return (p / "train.csv").exists() and (p / "val.csv").exists()


def _find_data_root(root: Path, max_depth: int = 4) -> Path | None:
    """Recursively look for a directory containing both train.csv and val.csv,
    bounded depth so we don't scan huge image folders.
    """
    if _looks_like_data_root(root):
        return root
    if max_depth <= 0:
        return None
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                found = _find_data_root(child, max_depth - 1)
                if found is not None:
                    return found
    except (PermissionError, OSError):
        pass
    return None


def auto_data_dir() -> Path:
    """Find the competition data directory.

    Walks /kaggle/input/<dataset>/... up to 4 levels deep looking for a
    directory that contains train.csv + val.csv. Falls back to ./data.
    """
    kg = Path("/kaggle/input")
    if kg.exists():
        for entry in sorted(kg.iterdir()):
            if not entry.is_dir():
                continue
            found = _find_data_root(entry)
            if found is not None:
                return found
    if _looks_like_data_root(Path("data")):
        return Path("data")
    return Path("data")


def auto_output_dir(timestamp: str) -> Path:
    """Output goes to /kaggle/working/models/<ts> on Kaggle, else ./models/<ts>."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / "models" / timestamp
    return Path("models") / timestamp


# ---------------------------------------------------------------------------
# Dataset + collate (target-only label masking)
# ---------------------------------------------------------------------------

class ScienceQATrainDataset(Dataset):
    """Yields image, prefix string, and full string for proper label masking.

    `data_dir` is where the CSVs live; `image_dir` (defaults to `data_dir`)
    is the base directory for the relative `image_path` column. They differ
    when the CSVs and images are attached as separate Kaggle Datasets.

    `img_size=None` lets the SmolVLM processor handle preprocessing natively
    (recommended for new training runs - more visual signal). Pass an int
    to manually resize first (matches the legacy 224 pipeline).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_size: int | None = None,
        image_dir: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.image_dir = Path(image_dir) if image_dir is not None else data_dir
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / row["image_path"]).convert("RGB")
        if self.img_size is not None:
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)

        prefix = build_prompt(row, include_answer=False)  # ends with "Answer: "
        answer_idx = int(row["answer"])
        target = f"{CHOICE_LETTERS[answer_idx]}. {row['choices'][answer_idx]}"
        return {
            "image": img,
            "prefix": prefix,
            "full": prefix + target,
        }


def make_train_collate(processor):
    """Build a collate fn that masks every label position before the target.

    For each example we tokenize prefix-only and full separately (batched),
    then per-row find the first divergence index between the two
    `input_ids` sequences. That index is where the target starts in the
    full sequence. Everything before it gets `-100` so it's ignored by the
    cross-entropy loss. Pad tokens are also masked.
    """
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        images = [b["image"] for b in batch]
        prefixes = [b["prefix"] for b in batch]
        fulls = [b["full"] for b in batch]

        processor.tokenizer.padding_side = "right"
        pref_inputs = processor(
            text=prefixes, images=images, return_tensors="pt", padding=True
        )
        full_inputs = processor(
            text=fulls, images=images, return_tensors="pt", padding=True
        )

        labels = full_inputs["input_ids"].clone()

        for i in range(labels.shape[0]):
            pref_ids = pref_inputs["input_ids"][i]
            full_ids = full_inputs["input_ids"][i]
            pref_len = int(pref_inputs["attention_mask"][i].sum().item())

            target_start = pref_len
            for j in range(pref_len):
                if int(pref_ids[j]) != int(full_ids[j]):
                    target_start = j
                    break
            labels[i, :target_start] = -100

        labels[labels == pad_id] = -100
        full_inputs["labels"] = labels
        return full_inputs

    return collate


# ---------------------------------------------------------------------------
# In-loop accuracy probe (per-choice log-likelihood)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def probe_accuracy(
    model,
    processor,
    df_subset: pd.DataFrame,
    data_dir: Path,
    img_size: int | None,
    desc: str = "Probe",
    image_dir: Path | None = None,
) -> float:
    """Score `df_subset` with predict_answer; return MCQ accuracy."""
    ds = ScienceQADataset(df_subset, data_dir, img_size=img_size, image_dir=image_dir)
    preds: list[int] = []
    truths: list[int] = []

    was_training = model.training
    model.eval()
    for item in tqdm(ds, desc=desc, leave=False):
        pred, _ = predict_answer(
            model,
            processor,
            item["image"],
            item["row"],
            mode="full",
            length_normalize=True,
        )
        preds.append(int(pred))
        truths.append(int(item["row"]["answer"]))
    if was_training:
        model.train()

    _, _, acc = grade_predictions(preds, truths)
    return acc


# ---------------------------------------------------------------------------
# Cosine LR schedule with warmup
# ---------------------------------------------------------------------------

def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_dir: str | None = None
    image_dir: str | None = None
    out_dir: str | None = None
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 8
    num_workers: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    dropout: float = 0.1
    img_size: int = 0  # 0 = let processor handle (recommended for new runs)
    probe_size: int = 200
    probes_per_epoch: int = 3
    seed: int = 42
    no_4bit: bool = False


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=str, default=None,
                   help="Override auto-detected data directory (where train/val/test.csv live).")
    p.add_argument("--image-dir", type=str, default=None,
                   help="Override base directory for the image_path column. "
                   "Defaults to --data-dir. Use when CSVs and images are in "
                   "separate Kaggle Datasets.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override auto-detected output directory.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--img-size", type=int, default=0,
                   help="Resize images to NxN before processor; 0 = native.")
    p.add_argument("--probe-size", type=int, default=200,
                   help="Val examples for the in-loop accuracy probe; 0 = full val.")
    p.add_argument("--probes-per-epoch", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization (uses fp16 instead).")
    a = p.parse_args()
    return TrainConfig(**vars(a))


def build_model(cfg: TrainConfig):
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if not cfg.no_4bit and torch.cuda.is_available():
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
        model = prepare_model_for_kbit_training(model)
    else:
        model = _AutoVisionModel.from_pretrained(
            MODEL_ID, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
        )

    model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def train(cfg: TrainConfig) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(cfg.data_dir) if cfg.data_dir else auto_data_dir()
    image_dir = Path(cfg.image_dir) if cfg.image_dir else data_dir
    out_dir = Path(cfg.out_dir) if cfg.out_dir else auto_output_dir(timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    print(f"Timestamp: {timestamp}")
    print(f"Data dir:  {data_dir}")
    print(f"Image dir: {image_dir}")
    print(f"Out dir:   {out_dir}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_df = load_split(data_dir, "train")
    val_df = load_split(data_dir, "val")
    if cfg.probe_size and cfg.probe_size < len(val_df):
        probe_df = val_df.sample(cfg.probe_size, random_state=cfg.seed).reset_index(drop=True)
    else:
        probe_df = val_df
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Probe: {len(probe_df)}")

    model = build_model(cfg)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    img_size = cfg.img_size if cfg.img_size > 0 else None
    train_ds = ScienceQATrainDataset(train_df, data_dir, img_size=img_size, image_dir=image_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=make_train_collate(processor),
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_optim_steps = max(1, (len(train_loader) * cfg.epochs) // cfg.grad_accum)
    warmup_steps = max(1, int(total_optim_steps * cfg.warmup_ratio))
    scheduler = cosine_with_warmup(optimizer, total_optim_steps, warmup_steps)

    metrics_csv = out_dir.parent / "training_metrics.csv"
    cols = ["Model Name", "Epoch", "Step", "Train Loss", "Val Loss", "Val Accuracy"]
    if not metrics_csv.exists():
        pd.DataFrame(columns=cols).to_csv(metrics_csv, index=False)

    save_steps = max(1, len(train_loader) // max(1, cfg.probes_per_epoch))
    print(
        f"\nTraining: {cfg.epochs} epoch(s), {len(train_loader)} steps/epoch, "
        f"probe every {save_steps} steps "
        f"({cfg.probes_per_epoch} probes/epoch on {len(probe_df)} val examples)"
    )

    best_acc = 0.0
    best_path: str | None = None

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        steps_since_probe = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True)

        for idx, batch in enumerate(pbar):
            inputs = {
                k: (v.to(model.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            outputs = model(**inputs)
            loss = outputs.loss / cfg.grad_accum
            loss.backward()
            running_loss += float(outputs.loss.item())
            steps_since_probe += 1

            if (idx + 1) % cfg.grad_accum == 0 or (idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(
                    {
                        "loss": f"{running_loss / max(steps_since_probe, 1):.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            if (idx + 1) % save_steps == 0 or (idx + 1) == len(train_loader):
                avg_loss = running_loss / max(steps_since_probe, 1)

                ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch + 1}_step_{idx + 1}"
                model.save_pretrained(ckpt_path)

                acc = probe_accuracy(
                    model,
                    processor,
                    probe_df,
                    data_dir,
                    img_size=img_size,
                    desc=f"Probe e{epoch + 1}s{idx + 1}",
                    image_dir=image_dir,
                )

                row = {
                    "Model Name": timestamp,
                    "Epoch": epoch + 1,
                    "Step": idx + 1,
                    "Train Loss": round(avg_loss, 4),
                    "Val Loss": pd.NA,
                    "Val Accuracy": round(acc, 4),
                }
                pd.DataFrame([row])[cols].to_csv(
                    metrics_csv, mode="a", header=False, index=False
                )
                print(
                    f"  [Epoch {epoch + 1} step {idx + 1}/{len(train_loader)}] "
                    f"loss={avg_loss:.4f} | val_acc={acc:.4f}"
                )

                if acc > best_acc:
                    best_acc = acc
                    best_path = str(ckpt_path)
                    print(f"  NEW BEST val accuracy: {best_acc:.4f}")

                running_loss = 0.0
                steps_since_probe = 0
                model.train()

    final_path = out_dir / "final"
    model.save_pretrained(final_path)
    print(f"\nFinal adapter saved to: {final_path}")
    if best_path is not None:
        print(f"Best by val accuracy:  {best_path}  (acc={best_acc:.4f})")
    print(f"Metrics:                {metrics_csv}")
    return {
        "timestamp": timestamp,
        "final_path": str(final_path),
        "best_path": best_path,
        "best_accuracy": best_acc,
        "out_dir": str(out_dir),
    }


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
