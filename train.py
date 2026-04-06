"""
Huấn luyện bộ phân loại văn bản Mamba trên tập IMDB.

Tải dữ liệu qua Hugging Face ``datasets``, token hóa bằng BERT uncased,
huấn luyện :class:`src.models.model.MambaClassifier` với AdamW, scheduler
giảm LR theo validation loss, early stopping và ghi log lên Weights & Biases.
"""

import torch
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
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
        "d_model": 256,
        "n_layers": 4,
        "num_classes": 3,  # uitnlp/vietnamese_students_feedback: sentiment 0/1/2
        "dropout": 0.1,
        "learning_rate": 2e-4,
        "batch_size": 32,
        "num_epochs": 3,
        "max_length": 512,
    }


def _prep_hf_split(split_ds):
    """Khớp schema code với uitnlp/vietnamese_students_feedback (sentence/sentiment/topic)."""
    ds = split_ds.rename_columns({"sentence": "text", "sentiment": "label"})
    cols = [c for c in ds.column_names if c not in ("text", "label")]
    if cols:
        ds = ds.remove_columns(cols)
    return ds


def prepare_data(config):
    """
    Tải dữ liệu, token hóa train/test, tạo DataLoader.

    Args:
        config: Phải chứa batch_size, max_length.

    Returns:
        Tuple (train_loader, test_loader, tokenizer).
    """
    dataset = load_dataset("uitnlp/vietnamese_students_feedback")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    train_src = _prep_hf_split(dataset["train"])
    test_src = _prep_hf_split(dataset["test"])

    tokenized_train = train_src.map(
        lambda x: tokenize_function(
            word_segment(standardize_data(x)), tokenizer, config["max_length"]
        ),
        batched=True,
        remove_columns=train_src.column_names,
    )
    tokenized_test = test_src.map(
        lambda x: tokenize_function(
            word_segment(standardize_data(x)), tokenizer, config["max_length"]
        ),
        batched=True,
        remove_columns=test_src.column_names,
    )

    train_loader = DataLoader(
        tokenized_train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        tokenized_test, 
        batch_size=config["batch_size"], 
        collate_fn=collate_fn
    )

    return train_loader, test_loader, tokenizer


def setup_model(config, tokenizer):
    """
    Khởi tạo MambaClassifier và chọn thiết bị (CUDA / MPS / CPU).
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
    model = MambaClassifier(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    return model, device


def train_epoch(model, train_loader, optimizer, device):
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
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        train_loss += loss.item()

        progress_bar.set_postfix(
            {"loss": loss.item(), "acc": train_correct / train_total}
        )

    return train_loss / len(train_loader), train_correct / train_total


def evaluate(model, test_loader, device):
    """
    Đánh giá trên tập kiểm tra (không backward).
    Args:
        model: MambaClassifier
        test_loader: DataLoader cho tập kiểm tra
        device: Thiết bị (CUDA / MPS / CPU)
    Returns:
        Tuple (loss_trung_bình, accuracy_trên_test).
    """
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device).float()
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels)
            test_loss += outputs["loss"].item()
            predictions = torch.argmax(outputs["logits"], dim=1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    return test_loss / len(test_loader), test_correct / test_total


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
    Vòng huấn luyện đầy đủ: W&B, scheduler, lưu model khi test_acc tăng,
    early stopping sau max_patience epoch không cải thiện.
    Args:
        config: Siêu tham số mặc định
        train_loader: DataLoader cho tập huấn luyện
        test_loader: DataLoader cho tập kiểm tra
        tokenizer: Tokenizer
    Returns:
        None
    """
    wandb.init(project="mamba-classification")
    config = get_default_config() # Lấy siêu tham số mặc định
    train_loader, test_loader, tokenizer = prepare_data(config) # Tải dữ liệu và tạo DataLoader
    model, device = setup_model(config, tokenizer) # Khởi tạo mô hình và chọn thiết bị

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"]) # Khởi tạo optimizer
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    ) # Khởi tạo scheduler

    best_test_acc = 0 # Độ chính xác tốt nhất
    patience_counter = 0 # Số epoch không cải thiện
    max_patience = 5 # Số epoch tối đa không cải thiện

    for epoch in range(config["num_epochs"]): # Vòng lặp epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        wandb.log(metrics)
        print(f"Epoch {epoch+1} metrics:", metrics)

        scheduler.step(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(model, optimizer, epoch, test_acc)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break


if __name__ == "__main__":
    train()
