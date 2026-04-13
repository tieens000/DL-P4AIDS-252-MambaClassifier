# Mamba Classifier for Vietnamese Text

[![Notebook](https://img.shields.io/badge/Colab-Notebook-orange?logo=googlecolab)](https://colab.research.google.com/drive/1AQJ5iYRikCVSjPtinBsDEYt846oXhfrl?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)

A robust, minimal text classifier based on the [Mamba](https://arxiv.org/abs/2312.00752) architecture (State Space Models). This repository demonstrates how to adapt a pure PyTorch implementation of Mamba for Natural Language Processing (NLP) tasks—specifically sentiment analysis on the Vietnamese Students Feedback dataset.

The core Mamba implementation is inspired by [@alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py).

---

## 🚀 Key Features

* **Mamba Architecture**: Replaces traditional Transformers/LSTMs with an efficient Selective State Space Model, ensuring $O(L)$ sequence scaling and fast auto-regressive processing.
* **Vietnamese NLP Ready**: 
  * Integrates `underthesea` for Vietnamese word segmentation.
  * Leverages pretrained embeddings from `vinai/phobert-base`.
* **Robust Training Pipeline** (`train.py`):
  * **Optimizer**: AdamW with weight decay and gradient clipping.
  * **Scheduler**: `ReduceLROnPlateau` for adaptive learning rate adjustments.
  * **Early Stopping**: Prevents overfitting with customizable patience blocks. 
* **Experiment Tracking**: Deep integration with [Weights & Biases (W&B)](https://wandb.ai/) to log macro/weighted F1-scores, losses, and real-time confusion matrices.
* **Hardware Auto-Detection**: Seamlessly detects and switches between CUDA (NVIDIA), MPS (Apple Silicon), or CPU.

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── models/
│   │   ├── backbone.py    # Core Mamba architecture (MambaBlock, ResidualBlock, etc.)
│   │   ├── head.py        # Classification head components
│   │   └── model.py       # Full MambaClassifier assembly
├── train.py               # Main training loop and data preparation
├── test.py                # Inference script
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```

---

## 🛠️ Installation

### 1. Create a Virtual Environment

We recommend using Conda to manage your Python environment.

```bash
conda create -n mamba-classifier python=3.10 -y
conda activate mamba-classifier
```

### 2. Install Dependencies

Install required libraries via `pip`. Make sure to install `underthesea` if you haven't natively installed it.

```bash
pip install -r requirements.txt
pip install underthesea
```

### 3. Weights & Biases (Optional but Recommended)

To monitor and log experiments, log in to your W&B account:

```bash
wandb login
```

---

## 🏃‍♂️ Usage

### Training the Model
To train the model on the `uitnlp/vietnamese_students_feedback` dataset, simply run:

```bash
python train.py
```

**During training, the script will automatically**:
1. Download the dataset via Hugging Face `datasets`.
2. Tokenize and segment words using PhoBERT and Underthesea.
3. Download the pre-trained embedding weights from `vinai/phobert-base`.
4. Train the model while logging validation loss and accuracy.
5. Save the best checkpoint to `best_model.pt` in the working directory and sync it to W&B.

### Configurations
Training hyperparameters can be adjusted within the `get_default_config()` function in `train.py`. Key properties include:
- `d_model`: Embedding dimension (default: `128`)
- `n_layers`: Number of Mamba layers (default: `2`)
- `batch_size`: (default: `64`)
- `num_epochs`: Maximum number of epochs (default: `30`)
- `learning_rate`: (default: `5e-4`)

---

## 📊 Evaluation Metrics

The script uses `scikit-learn` metrics to evaluate the model on the exact test set, emitting:
- **`Accuracy`**
- **`F1-Score`** (`macro`, `weighted`, and `per_class` metrics)
- **`Confusion Matrix`** (Synced with Wandb plotting tools)

---

## 📜 Acknowledgements
* Base Mamba structure referenced from [mamba.py by alxndrTL](https://github.com/alxndrTL/mamba.py).
* Initial original architecture paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
