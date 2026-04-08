# Chi tiết xử lý: `src/models/head.py`

File định nghĩa **`ClassifierHead`** — lớp nhỏ gắn **sau backbone Mamba** để biến biểu diễn chuỗi `(B, L, d_model)` thành **logits phân loại** `(B, num_classes)`.

---

## 1. Ý tưởng thiết kế

- Backbone Mamba trả về **một vector `d_model` cho mỗi bước thời gian** (mỗi token), nhưng phân loại cần **một vector cố định cho cả câu**.
- `ClassifierHead` dùng **mean pooling** trên trục chuỗi: coi mọi vị trí đóng góp đều nhau, rồi chuẩn hóa và chiếu tuyến tính ra số lớp.

Không có cơ chế attention riêng; không dùng token `[CLS]`.

---

## 2. Khởi tạo (`__init__`)

Tham số:

| Tham số       | Kiểu   | Ý nghĩa |
|---------------|--------|---------|
| `d_model`     | `int`  | Chiều đặc trưng tại mỗi bước thời gian (khớp output backbone). |
| `num_classes` | `int` | Số lớp logits đầu ra. |
| `dropout`     | `float` | Xác suất dropout **trước** lớp `Linear` cuối (mặc định 0.1). |

Các submodule:

1. **`self.dropout = nn.Dropout(dropout)`**  
   - Tắt ngẫu nhiên một phần phần tử của vector đã pool (chỉ khi `training=True`).

2. **`self.norm = nn.LayerNorm(d_model)`**  
   - Chuẩn hóa theo chiều feature cuối cùng của vector `(B, d_model)` sau pooling — ổn định phân phối đầu vào cho `fc`.

3. **`self.fc = nn.Linear(d_model, num_classes)`**  
   - Ánh xạ tuyến tính → logits thô (chưa softmax).

---

## 3. Forward — từng bước

Chữ ký: `forward(self, x: Tensor) -> Tensor`.

### Bước 1: Mean pooling theo chiều chuỗi

```python
x = x.mean(dim=1)  # (batch_size, d_model)
```

- **Đầu vào `x`:** `(batch_size, seq_len, d_model)`.
- **`dim=1`:** lấy trung bình cộng **theo toàn bộ `seq_len`** cho từng batch và từng chiều `d_model`.
- **Đầu ra:** `(batch_size, d_model)` — một vector câu duy nhất.

### Bước 2: LayerNorm

```python
x = self.norm(x)
```

- Chuẩn hóa mỗi vector `(d_model,)` trong batch (mean/var theo feature).

### Bước 3: Dropout

```python
x = self.dropout(x)
```

- Regularization trước lớp phân loại cuối.

### Bước 4: Linear → logits

```python
return self.fc(x)  # (batch_size, num_classes)
```

- **Đầu ra:** logits dùng trực tiếp với `CrossEntropyLoss` trong `MambaClassifier` (loss tự gộp softmax + NLL cho logits).

---

## 4. Liên kết với `MambaClassifier`

Trong `model.py`:

```text
sequence_output = backbone(x)   # (B, L, d_model)
logits = classifier(sequence_output)  # (B, num_classes)
```

`ClassifierHead` **không** nhận `labels`; chỉ biến đổi đặc trưng. Loss được tính **ở ngoài** (trong `MambaClassifier.forward`).

---

## 5. Lưu ý khi thử nghiệm

- **Pooling:** đổi `mean` → `max` hoặc lấy vị trí cuối `x[:, -1, :]` sẽ thay đổi inductive bias nhưng cần sửa code.
- **`seq_len = 0`:** mean trên tensor rỗng sẽ gây NaN; trong pipeline tokenizer/padding hiện tại thường luôn có `L > 0`.
