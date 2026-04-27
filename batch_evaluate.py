import sys
import glob
import json
import os
import re
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_DIR = Path("data")
IMG_SIZE = 224
CHOICE_LETTERS = "ABCDEFGHIJ"
METRICS_CSV = "models/training_metrics.csv"
BATCH_SIZE = 2
NUM_WORKERS = 2

# Manually requested Train Losses from the Ablations log
TRAIN_LOSS_MAP = {
    "20260426_233953": {
        "epoch_1_final": 0.3154,
        "epoch_2_final": 0.2163
    }
}

print("[1/5] Identifying Checkpoints...")
search_path = "models/*/checkpoints/*"
checkpoints = [c for c in glob.glob(search_path) if os.path.isdir(c)]

existing_records = set()
if os.path.exists(METRICS_CSV):
    existing_df = pd.read_csv(METRICS_CSV)
    for _, row in existing_df.iterrows():
        step_val = str(row["Step"]) if pd.notna(row["Step"]) else "Final"
        existing_records.add((str(row["Model Name"]), int(row["Epoch"]), step_val))
else:
    pd.DataFrame(columns=["Model Name", "Epoch", "Step", "Train Loss", "Val Loss"]).to_csv(METRICS_CSV, index=False)

to_evaluate = []
for ckpt in checkpoints:
    parts = ckpt.split(os.sep)
    model_name = parts[1] 
    ckpt_name = parts[3] 
    
    epoch_match = re.search(r"epoch_(\d+)", ckpt_name)
    step_match = re.search(r"step_(\d+)", ckpt_name)
    
    if not epoch_match: continue
    
    epoch = int(epoch_match.group(1))
    step = str(step_match.group(1)) if step_match else "Final"
    
    if (model_name, epoch, step) not in existing_records:
        train_loss = None
        if step == "Final" and model_name in TRAIN_LOSS_MAP:
            key = f"epoch_{epoch}_final"
            if key in TRAIN_LOSS_MAP[model_name]:
                train_loss = TRAIN_LOSS_MAP[model_name][key]
                
        to_evaluate.append({
            "path": ckpt,
            "model_name": model_name,
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss
        })

to_evaluate.sort(key=lambda x: (x["model_name"], x["epoch"], 999999 if x["step"]=="Final" else int(x["step"])))

if not to_evaluate:
    print("All checkpoints already exist in CSV! Nothing to do.")
    sys.exit(0)

print(f"Found {len(to_evaluate)} checkpoints to evaluate.")

# Data Loading
print("[2/5] Loading Validation Dataset...")
val_df = pd.read_csv(DATA_DIR / "val.csv")
val_df["choices"] = val_df["choices"].apply(json.loads)

def build_prompt(row, include_answer=False):
    context_parts = []
    if pd.notna(row.get("lecture")) and str(row.get("lecture")).strip(): context_parts.append(str(row["lecture"]).strip())
    if pd.notna(row.get("hint")) and str(row.get("hint")).strip(): context_parts.append(str(row["hint"]).strip())
    context_str = "\n".join(context_parts)
    choices_str = "\n".join(f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    prompt = "<image>\n"
    if context_str: prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {row['question']}\nChoices:\n{choices_str}\nAnswer:"
    if include_answer: prompt += f" {CHOICE_LETTERS[int(row['answer'])]}"
    return prompt

class ScienceQADataset(Dataset):
    def __init__(self, df, data_dir, img_size=224):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size
    def __len__(self): return len(self.df)
    def _load_image(self, rel_path):
        return Image.open(self.data_dir / rel_path).convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {"image": self._load_image(row["image_path"]), "text": build_prompt(row, include_answer=True)}

val_ds = ScienceQADataset(val_df, DATA_DIR, img_size=IMG_SIZE)
processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = inputs["input_ids"].clone()
    for i in range(labels.shape[0]):
        seq_len = (inputs["attention_mask"][i] == 1).sum().item()
        labels[i, :seq_len - 2] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

print("[3/5] Loading Base Model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base_model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True)

# Instantiate with the first one so we can hot-swap the rest
print("[4/5] Preloading Peft Framework...")
model = PeftModel.from_pretrained(base_model, to_evaluate[0]["path"])
model.eval()

print(f"[5/5] Beginning Batch Evaluation Sequence...")

for item in to_evaluate:
    print(f"\n--> Evaluating: {item['model_name']} | Epoch {item['epoch']} | Step {item['step']}")
    
    # Hot-swap the weights
    model.load_adapter(item["path"], adapter_name="default")
    model.set_adapter("default")
    
    total_val_loss = 0
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(**inputs)
            total_val_loss += outputs.loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Result -> Val Loss: {avg_val_loss:.4f}")
    
    # Append instantly
    new_row = {
        "Model Name": item["model_name"],
        "Epoch": item["epoch"],
        "Step": item["step"],
        "Train Loss": item["train_loss"],
        "Val Loss": round(avg_val_loss, 4)
    }
    pd.DataFrame([new_row]).to_csv(METRICS_CSV, mode="a", header=False, index=False)

print("\nAll outstanding checkpoints successfully logged to internal CSV!")
