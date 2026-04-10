"""
Huấn luyện bộ phân loại văn bản Mamba trên tập IMDB.

Tải dữ liệu qua Hugging Face ``datasets``, token hóa bằng BERT uncased,
huấn luyện :class:`src.models.model.MambaClassifier` với AdamW, scheduler
giảm LR theo validation loss, early stopping và ghi log lên Weights & Biases.
"""

import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import underthesea
import re

import wandb
from src.models.model import MambaClassifier


def _standardize_one(text: str) -> str:
    # Xóa dấu ở cuối câu
    text = re.sub(r"[\.,\?]+$", "", text)

    # Xóa punctuation
    text = (
        text.replace(",", " ")
        .replace(".", " ")
        .replace(";", " ")
        .replace("“", " ")
        .replace(":", " ")
        .replace("”", " ")
        .replace('"', " ")
        .replace("'", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace("-", " ")
    )

    # Xóa multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def standardize_data(examples):
    """Chuẩn hóa ``text``: một chuỗi hoặc list chuỗi (khi ``datasets.map(..., batched=True)``)."""
    texts = examples["text"]
    if isinstance(texts, str):
        examples["text"] = _standardize_one(texts)
    else:
        examples["text"] = [_standardize_one(t) for t in texts]
    return examples


def word_segment(examples):
    texts = examples["text"]
    if isinstance(texts, str):
        examples["text"] = underthesea.word_tokenize(texts, format="text")
    else:
        examples["text"] = [
            underthesea.word_tokenize(t, format="text") for t in texts
        ]
    return examples


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize dữ liệu
    Args:
        examples: Dữ liệu cần tokenize
        tokenizer: Tokenizer
        max_length: Chiều dài tối đa của token
    Returns:
        Dict ``input_ids``, ``attention_mask`` (và các khóa tokenizer khác), ``labels``.
        Với một câu: ``return_tensors="pt"``. Với batch (list câu): ``return_tensors=None``
        để Arrow lưu list, DataLoader vẫn nhận list số nguyên.
    """
    texts = examples["text"]
    batched = not isinstance(texts, str)
    tok_out = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None if batched else "pt",
    )
    return {**tok_out, "labels": examples["label"]}


def collate_fn(batch):
    """
    Gộp các tensor input_ids và labels thành một tensor.
    Args:
        batch: List các mẫu đã token hóa.
    Returns:
        Dict ``input_ids``, ``labels``.
    """
    # Chuyển input_ids của từng mẫu thành tensor
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    # Chuyển labels của từng mẫu thành tensor
    labels = torch.tensor([item["labels"] for item in batch])
    # Padding để đảm bảo tất cả các tensor có cùng chiều dài
    input_ids = pad_sequence(input_ids, batch_first=True)
    # Thêm chiều cuối 1 để khớp đầu vào mô hình
    input_ids = input_ids.unsqueeze(-1)
    # Trả về dict có khóa input_ids và labels
    return {"input_ids": input_ids, "labels": labels}


def get_default_config():
    """Trả về dict siêu tham số mặc định cho mô hình và huấn luyện."""
    return {
        "d_model": 64,
        "n_layers": 2,
        "num_classes": 3,  # uitnlp/vietnamese_students_feedback: sentiment 0/1/2
        "dropout": 0.3,
        "learning_rate": 5e-5,
        "weight_decay": 0.1,
        "batch_size": 32,
        "num_epochs": 20,
        "max_length": 256,
        "max_grad_norm": 1.0,
        "seed": 42,
        "dataset_name": "uitnlp/vietnamese_students_feedback",
        "tokenizer_name": "vinai/phobert-base",
        "scheduler_mode": "min",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "max_patience": 5,
        "label_smoothing": 0.1,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _try_git_revision() -> str:
    # Jupyter/Colab: __file__ is undefined when code runs from a notebook cell.
    try:
        root = Path(__file__).resolve().parent
    except NameError:
        root = Path.cwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _prep_hf_split(split_ds):
    """Khớp schema code với uitnlp/vietnamese_students_feedback (sentence/sentiment/topic)."""
    ds = split_ds.rename_columns({"sentence": "text", "sentiment": "label"})
    cols = [c for c in ds.column_names if c not in ("text", "label")]
    if cols:
        ds = ds.remove_columns(cols)
    return ds


def prepare_data(config):
    """
    Tải dữ liệu, token hóa train/validation/test, tạo DataLoader.

    Args:
        config: Phải chứa batch_size, max_length.

    Returns:
        Tuple (train_loader, val_loader, test_loader, tokenizer).
    """
    dataset = load_dataset(config["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    train_src = _prep_hf_split(dataset["train"])
    val_src = _prep_hf_split(dataset["validation"])
    test_src = _prep_hf_split(dataset["test"])

    def _tokenize(x):
        return tokenize_function(
            word_segment(standardize_data(x)), tokenizer, config["max_length"]
        )

    tokenized_train = train_src.map(
        _tokenize,
        batched=True,
        remove_columns=train_src.column_names,
    )
    tokenized_val = val_src.map(
        _tokenize,
        batched=True,
        remove_columns=val_src.column_names,
    )
    tokenized_test = test_src.map(
        _tokenize,
        batched=True,
        remove_columns=test_src.column_names,
    )

    train_loader = DataLoader(
        tokenized_train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        tokenized_val,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        tokenized_test, 
        batch_size=config["batch_size"], 
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, tokenizer


def setup_model(config, tokenizer):
    """
    Khởi tạo MambaClassifier và chọn thiết bị (CUDA / MPS / CPU).
    Trích xuất pretrained embedding từ PhoBERT nếu tokenizer_name hỗ trợ.
    Args:
        config: Cần d_model, n_layers, num_classes, dropout.
        tokenizer: Để lấy vocab_size cho embedding.
    Returns:
        Tuple (model, device).
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Trích xuất pretrained embedding từ PhoBERT (tải trực tiếp file weights,
    # tránh import AutoModel gây lỗi torchvision trên Colab)
    print(f"Loading pretrained embeddings from {config['tokenizer_name']}...")
    try:
        # Thử safetensors trước (nhẹ hơn, nhanh hơn)
        model_path = hf_hub_download(config["tokenizer_name"], "model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    except Exception:
        # Fallback sang pytorch_model.bin
        model_path = hf_hub_download(config["tokenizer_name"], "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Tìm key embedding trong state_dict
    embed_key = None
    for key in state_dict:
        if "word_embeddings.weight" in key:
            embed_key = key
            break
    if embed_key is None:
        raise KeyError("Không tìm thấy word_embeddings.weight trong pretrained model.")

    pretrained_embeddings = state_dict[embed_key].clone()
    embed_dim = pretrained_embeddings.shape[1]
    print(f"Pretrained embedding: vocab={pretrained_embeddings.shape[0]}, dim={embed_dim}")
    del state_dict  # Giải phóng bộ nhớ

    model = MambaClassifier(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        vocab_size=tokenizer.vocab_size,
        label_smoothing=config["label_smoothing"],
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,} (pretrained embedding)")
    return model, device


def train_epoch(model, train_loader, optimizer, device, max_grad_norm: float):
    """
    Một epoch huấn luyện: forward, backward, clip gradient, cập nhật trọng số.
    Args:
        model: MambaClassifier
        train_loader: DataLoader cho tập huấn luyện
        optimizer: Optimizer
        device: Thiết bị (CUDA / MPS / CPU)
    Returns:
        Tuple (loss_trung_bình, accuracy_trên_train).
    """
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:

        input_ids = batch["input_ids"].clone().detach().to(device).float()
        labels = batch["labels"].clone().detach().to(device)

        outputs = model(input_ids, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        train_loss += loss.item()

        progress_bar.set_postfix(
            {"loss": loss.item(), "acc": train_correct / train_total}
        )

    return train_loss / len(train_loader), train_correct / train_total


def evaluate(model, test_loader, device, num_classes: int):
    """
    Đánh giá trên tập kiểm tra (không backward).
    Args:
        model: MambaClassifier
        test_loader: DataLoader cho tập kiểm tra
        device: Thiết bị (CUDA / MPS / CPU)
        num_classes: Số lớp (cho confusion matrix / F1)
    Returns:
        Tuple (loss_trung_bình, accuracy_trên_test, dict_metrics).
        dict_metrics gồm f1_macro, f1_weighted, f1_per_class, y_true, y_pred.
    """
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device).float()
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels)
            test_loss += outputs["loss"].item()
            predictions = torch.argmax(outputs["logits"], dim=1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
            all_preds.append(predictions.detach().cpu())
            all_labels.append(labels.detach().cpu())

    avg_loss = test_loss / len(test_loader)
    acc = test_correct / test_total
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_list = [float(x) for x in np.atleast_1d(f1_per).ravel()]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    extra = {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_list,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return avg_loss, acc, extra


def save_checkpoint(model, optimizer, epoch, test_acc):
    """Lưu checkpoint tốt nhất vào best_model.pt trong thư mục làm việc.
    Args:
        model: MambaClassifier
        optimizer: Optimizer
        epoch: Số epoch
        test_acc: Độ chính xác trên tập kiểm tra
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc,
        },
        "best_model.pt",
    )


def train():
    """
    Vòng huấn luyện đầy đủ: W&B, scheduler, lưu model khi val_loss giảm,
    early stopping sau max_patience epoch không cải thiện.
    """
    config = get_default_config()
    _set_seed(config["seed"])
    train_loader, val_loader, test_loader, tokenizer = prepare_data(config)
    model, device = setup_model(config, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())

    wb_config = {
        **config,
        "optimizer_type": "AdamW",
        "scheduler_type": "ReduceLROnPlateau",
        "model/total_parameters": n_params,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "device": str(device),
        "git_revision": _try_git_revision(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    run = wandb.init(project="mamba-classification", config=wb_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler_mode"],
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        verbose=True,
    )

    best_val_loss = float("inf")
    best_test_f1_weighted = 0.0
    patience_counter = 0
    max_patience = config["max_patience"]
    class_names = [str(i) for i in range(config["num_classes"])]

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, config["max_grad_norm"]
        )
        val_loss, val_acc, val_ex = evaluate(
            model, val_loader, device, config["num_classes"]
        )
        test_loss, test_acc, test_ex = evaluate(
            model, test_loader, device, config["num_classes"]
        )
        best_test_f1_weighted = max(best_test_f1_weighted, test_ex["f1_weighted"])

        lr = optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_ex["f1_macro"],
            "val_f1_weighted": val_ex["f1_weighted"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1_macro": test_ex["f1_macro"],
            "test_f1_weighted": test_ex["f1_weighted"],
            "learning_rate": lr,
        }
        for i, f1c in enumerate(test_ex["f1_per_class"]):
            metrics[f"test_f1_class_{i}"] = f1c

        try:
            metrics["test/confusion_matrix"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_ex["y_true"],
                preds=test_ex["y_pred"],
                class_names=class_names,
            )
        except Exception:
            pass

        cm = test_ex["confusion_matrix"]
        cm_rows = [
            [ti, pj, int(cm[ti, pj])]
            for ti in range(cm.shape[0])
            for pj in range(cm.shape[1])
        ]
        metrics["test/confusion_counts"] = wandb.Table(
            columns=["true", "pred", "count"], data=cm_rows
        )

        wandb.log(metrics)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"f1_macro={test_ex['f1_macro']:.4f} f1_weighted={test_ex['f1_weighted']:.4f} "
            f"lr={lr:.2e}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, test_acc)
            artifact = wandb.Artifact(
                f"best-model-{run.id}",
                type="model",
                metadata={
                    "epoch": epoch + 1,
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "test_f1_weighted": test_ex["f1_weighted"],
                    "test_f1_macro": test_ex["f1_macro"],
                },
            )
            artifact.add_file("best_model.pt")
            run.log_artifact(artifact)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_test_f1_weighted"] = best_test_f1_weighted


if __name__ == "__main__":
    train()
