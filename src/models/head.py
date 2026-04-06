"""
Đầu phân loại phía sau backbone: pooling theo trục chuỗi → LayerNorm → Linear.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        """
        Mean pooling trên chiều ``seq_len``, chuẩn hóa và chiếu xuống ``num_classes``.

        Args:
            d_model: Chiều đặc trưng mỗi bước thời gian (khớp đầu ra backbone).
            num_classes: Số lớp phân loại.
            dropout: Xác suất dropout trước lớp tuyến tính cuối.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch_size, seq_len, d_model)`` — đầu ra chuỗi từ Mamba.

        Returns:
            Logits ``(batch_size, num_classes)``.
        """
        # Average pooling over sequence length
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Apply layer norm and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Final classification layer
        return self.fc(x)
