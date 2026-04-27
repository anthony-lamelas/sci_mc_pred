import json

notebook_path = 'starter_notebook.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

new_cell_content = [
    "# ── 4b. Run Inference ────────────────────────────────────────────────────────\n",
    "predictions = []\n",
    "\n",
    "# Fix padding for batch inference\n",
    "processor.tokenizer.padding_side = 'left'\n",
    "\n",
    "# Pre-calculate Vocabulary IDs for exact Choice mapping\n",
    "CHOICE_LETTERS = \"ABCDEFGHIJ\"\n",
    "target_tokens = [\" \" + c for c in CHOICE_LETTERS]\n",
    "target_ids = [processor.tokenizer.encode(tok, add_special_tokens=False)[-1] for tok in target_tokens]\n",
    "target_ids_tensor = torch.tensor(target_ids, device=model.device)\n",
    "\n",
    "model.eval()\n",
    "with torch.inference_mode():\n",
    "    for batch in tqdm(test_loader, desc=\"Generating Predictions\"):\n",
    "        inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}\n",
    "        \n",
    "        # Enforce PyTorch to return strict math logits for the very first generated token\n",
    "        outputs = model.generate(\n",
    "            **inputs, \n",
    "            max_new_tokens=1, \n",
    "            return_dict_in_generate=True, \n",
    "            output_scores=True,\n",
    "            do_sample=False\n",
    "        )\n",
    "        \n",
    "        # Access the raw probability scores for the first outputted token across all items in batch\n",
    "        first_token_logits = outputs.scores[0] \n",
    "        \n",
    "        # Mathematically pluck out ONLY the 10 columns matching our A-J token targets\n",
    "        target_logits = first_token_logits[:, target_ids_tensor]\n",
    "        \n",
    "        # Dynamically select the exact index with highest probability (argmax) \n",
    "        winning_indices = torch.argmax(target_logits, dim=-1)\n",
    "        predictions.extend(winning_indices.cpu().numpy().tolist())\n"
]

target_idx = -1
for idx, cell in enumerate(nb['cells']):
    if "4b. Run Inference" in str(cell.get("source", [])):
        target_idx = idx
        break

if target_idx != -1:
    nb['cells'][target_idx]["source"] = new_cell_content
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Inference cell patched successfully.")
else:
    print("Could not find 4b. Run Inference block.")

