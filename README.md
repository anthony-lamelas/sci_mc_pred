# ScienceQA Vision-Language Model Fine-Tuning

This repository contains the code, experiments, and visualizations for fine-tuning `SmolVLM-500M-Instruct` on the ScienceQA multimodal multiple-choice dataset. The goal of this project was to maximize validation accuracy under strict hardware and parameter constraints (max 5M trainable parameters).

## Main Files

### `starter_notebook.ipynb`
This is the core execution pipeline of the project. It handles data loading, parameter-efficient fine-tuning (PEFT), and the custom text-generation inference loop.
- **Current State:** The notebook is currently hardcoded to match the hyperparameter configuration of our **peak submission (74.4% Accuracy)**. 

### `results.ipynb`
This notebook is purely for data visualization and generating figures for the final report.
- **Functionality:** It reads hardcoded leaderboard accuracies derived from the ablations log and parses the training logs to plot performance curves.
- **Outputs:** Running this notebook automatically generates three high-resolution graphs:
  1. `accuracy_progression.png`: A bar chart showing the leaderboard accuracy jump across experimental runs.
  2. `healthy_loss_curve.png`: The stable training/validation convergence curve of the peak 74.4% model.
  3. `overfit_loss_curve.png`: A chart demonstrating deceptive validation loss versus test accuracy for an overfitted 3-epoch model.

## Directory Structure

### `/models`
The `models` directory is the central storage location for all trained weights and evaluation metrics.
- **Checkpoints:** Each training run creates a new subdirectory named by its starting timestamp (e.g., `20260426_233953/`). Inside these timestamped folders, you will find the intermediate checkpoint states (saved 3 times per epoch) as well as the `/final` LoRA/DoRA adapter weights.
- **`training_metrics.csv`:** This is the master log file. During every training run, the `batch_evaluate.py` script automatically appends the current Step, Epoch, Train Loss, and Held-Out Validation Loss to this file. It serves as the data source for the visualization notebook.

## Setup and Execution

To get started and run the notebooks locally:

1. **Create Virtual Environment:**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the Virtual Environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   Ensure all required libraries (like `transformers`, `peft`, `bitsandbytes`, `pandas`, `matplotlib`, etc.) are installed:
   ```bash
   pip install -r requirements.txt
   ```

4. **Running the Pipeline:**
   - **Training & Inference:** Open `starter_notebook.ipynb` in your IDE (like VSCode) or Jupyter Notebook. Because it is pre-configured to the optimal Run 4 settings, you can simply click "Run All" to download the dataset, train the 5M parameter adapter, and generate a submission file.
   - **Visualizations:** Open `results.ipynb` and click "Run All" to parse the CSV logs and generate the `.png` charts into your root directory.
