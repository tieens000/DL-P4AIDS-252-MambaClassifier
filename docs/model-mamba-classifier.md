# Chi tiết xử lý: `src/models/model.py`

File định nghĩa **`MambaClassifier`** — mô hình end-to-end nối **embedding token** → **backbone Mamba** (`Mamba`) → **đầu phân loại** (`ClassifierHead`), có tùy chọn tính **CrossEntropyLoss**.

---

## 1. Vai trò trong pipeline

`MambaClassifier` là lớp PyTorch duy nhất mà `train.py` gọi trực tiếp. Luồng dữ liệu:

1. **Đầu vào:** tensor token ID (thường `(B, L, 1)` sau `collate_fn`).
2. **Embedding:** chuyển ID → vector `(B, L, d_model)`.
3. **Backbone:** Mamba giữ nguyên chiều chuỗi, vẫn `(B, L, d_model)`.
4. **Head:** gộp theo trục chuỗi → logits `(B, num_classes)`.
5. **Loss (tuỳ chọn):** nếu có `labels`, cộng thêm `loss` vào dict trả về.

---

## 2. Bước khởi tạo (`__init__`)

### 2.1 Kiểm tra `vocab_size`

- Nếu `vocab_size is None` → **`ValueError`**: bắt buộc phải có kích thước từ vựng vì dùng `nn.Embedding`.
- Trong project, `train.setup_model()` truyền `tokenizer.vocab_size` (PhoBERT).

### 2.2 `self.embedding`

- **`nn.Embedding(vocab_size, d_model)`**
- Mỗi token ID (0 … `vocab_size - 1`) ánh xạ tới một vector `d_model` chiều, **học được** trong quá trình huấn luyện (không nạp trọng số BERT từ checkpoint; chỉ “khớp kích thước vocab” với tokenizer).

### 2.3 `self.backbone`

- Tạo **`MambaConfig(d_model=d_model, n_layers=n_layers)`** — các hyperparameter khác lấy **mặc định** của dataclass (xem `docs/backbone-mamba.md`).
- **`Mamba(config)`**: xếp chồng `n_layers` khối residual Mamba.

### 2.4 `self.classifier`

- **`ClassifierHead(d_model, num_classes, dropout)`** — mean pooling + LayerNorm + Dropout + Linear (xem `docs/head-classifier.md`).

### 2.5 `self.loss_fct`

- **`nn.CrossEntropyLoss()`** — logits `(B, C)` so với nhãn `(B,)` dạng class index.

---

## 3. Bước forward — từng dòng logic

Chữ ký: `forward(self, x, labels=None) -> dict`.

### 3.1 Bước 1: Embedding

```python
x Embedding = self.embedding(x.long().squeeze(-1))
```

- **`x.long()`:** embedding yêu cầu chỉ số kiểu nguyên.
- **`squeeze(-1)`:** loại chiều cuối nếu tensor có dạng `(B, L, 1)` (đúng với `train.collate_fn`).
- **Đầu ra:** `x` có shape **`(B, L, d_model)`**.

### 3.2 Bước 2: Backbone Mamba

```python
sequence_output = self.backbone(x)
```

- **Đầu vào / đầu ra:** đều **`(B, L, d_model)`** — mỗi vị trí thời gian một vector biểu diễn sau khi qua toàn bộ `ResidualBlock`.

### 3.3 Bước 3: Đầu phân loại

```python
logits = self.classifier(sequence_output)
```

- **Đầu ra:** **`(B, num_classes)`** — logits chưa softmax.

### 3.4 Bước 4: Ghép `output`

- Luôn có **`output = {"logits": logits}`**.

### 3.5 Bước 5: Loss (nếu có nhãn)

- Nếu `labels is not None`:
  - `loss = self.loss_fct(logits, labels)`
  - `output["loss"] = loss`
- Nếu không có `labels` (ví dụ inference): chỉ trả về `logits`, không có `loss`.

---

## 4. Sơ đồ tensor (tóm tắt)

| Giai đoạn        | Shape tensor chính      |
|-----------------|--------------------------|
| Input `x`       | `(B, L)` sau squeeze, hoặc tương đương qua embedding |
| Sau embedding   | `(B, L, d_model)`        |
| Sau backbone    | `(B, L, d_model)`        |
| Logits          | `(B, num_classes)`       |
| Labels (nếu có) | `(B,)`                   |

---

## 5. Phụ thuộc module

| Import           | Vai trò                          |
|------------------|----------------------------------|
| `Mamba`, `MambaConfig` | Lõi SSM + residual (`backbone.py`) |
| `ClassifierHead`     | Pooling + FC (`head.py`)           |

Chi tiết từng lớp con được mô tả trong **`docs/backbone-mamba.md`** và **`docs/head-classifier.md`**.
