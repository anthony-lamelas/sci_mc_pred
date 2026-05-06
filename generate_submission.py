import sys
import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_DIR = Path("data")
IMG_SIZE = 336
BATCH_SIZE = 1
NUM_WORKERS = 1
CHOICE_LETTERS = "ABCDEFGHIJ"

def build_prompt(row):
    context_parts = []
    if pd.notna(row.get("lecture")) and str(row.get("lecture")).strip(): context_parts.append(str(row["lecture"]).strip())
    if pd.notna(row.get("hint")) and str(row.get("hint")).strip(): context_parts.append(str(row["hint"]).strip())
    context_str = "\n".join(context_parts)
    choices_str = "\n".join(f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    prompt = "<image>\n"
    
    # Matching the new metadata prompt format
    subject = row.get("subject", "")
    topic = row.get("topic", "")
    if pd.notna(subject) and str(subject).strip(): prompt += f"Subject: {subject}\n"
    if pd.notna(topic) and str(topic).strip(): prompt += f"Topic: {topic}\n"

    prompt += f"Question: {row['question']}\n"
    if context_str: prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Choices:\n{choices_str}\nAnswer: "
    return prompt

class ScienceQATestDataset(Dataset):
    def __init__(self, df, data_dir, img_size=336):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size
        
    def __len__(self): return len(self.df)
    
    def _load_image(self, rel_path):
        return Image.open(self.data_dir / rel_path).convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "image": self._load_image(row["image_path"]), 
            "text": build_prompt(row)
        }

def collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "texts": [item["text"] for item in batch]
    }

def decode_generated_sequences(decoded_sequences):
    predictions = []
    for decoded in decoded_sequences:
        generated_text = decoded.strip()
        if not generated_text:
            predictions.append(0)
            continue

        normalized = generated_text.upper().strip()
        match = re.search(r'ANSWER[:\s]*([A-J])', normalized)
        if match is None:
            match = re.search(r'\b([A-J])\b', normalized)

        if match:
            predictions.append(CHOICE_LETTERS.index(match.group(1)))
        else:
            fallback = re.sub(r'[^A-J]', '', normalized)
            if fallback:
                predictions.append(CHOICE_LETTERS.index(fallback[0]))
            else:
                predictions.append(0)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint folder")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path '{args.checkpoint}' does not exist.")
        sys.exit(1)
        
    print(f"Loading Test Dataset...")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df["choices"] = test_df["choices"].apply(json.loads)
    test_ds = ScienceQATestDataset(test_df, DATA_DIR, img_size=IMG_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    print("Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = 'left'

    print("Loading Base Model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base_model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True)

    print(f"Loading PEFT Checkpoint from {args.checkpoint}...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    # Just run greedy-2tok to be fast, but matches notebook format
    config = {"max_new_tokens": 2, "num_beams": 1, "desc": "Greedy-2tok"}
    print(f"Running Inference: {config['desc']}...")
    
    predictions = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=f"Test [{config['desc']}]", leave=False):
            processor_inputs = processor(
                text=batch['texts'],
                images=batch['images'],
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in processor_inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=config['max_new_tokens'],
                num_beams=config['num_beams'],
                do_sample=False,
                return_dict_in_generate=True,
            )

            generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
            decoded_sequences = processor.batch_decode(generated_ids, skip_special_tokens=True)
            batch_preds = decode_generated_sequences(decoded_sequences)
            predictions.extend(batch_preds)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save
    output_dir = Path('submissions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse checkpoint name for a clean suffix
    ckpt_name = args.checkpoint.replace('/', '_').replace('\\', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_{config["desc"]}_{ckpt_name}_{timestamp}.csv'
    save_path = output_dir / filename
    
    pd.DataFrame({'id': test_df['id'], 'answer': predictions}).to_csv(save_path, index=False)
    print(f'\nSaved test submission to: {save_path}')
