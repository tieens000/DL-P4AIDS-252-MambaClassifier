"""
MambaClassifier là mô hình tổng hợp: embedding → backbone Mamba → đầu phân loại.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.backbone import Mamba, MambaConfig
from src.models.head import ClassifierHead


class MambaClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        n_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        vocab_size: Optional[int] = None,
    ):
        """
        Mamba end-to-end cho phân loại (embedding + SSM + đầu FC).

        Args:
            d_model:
                Ví dụ: d_model = 16 nghĩa là mỗi từ được biểu diễn bởi 16 số thực
            n_layers: Số tầng ResidualBlock.
            num_classes: Số lớp logits. Ví dụ: num_classes = 10 nghĩa là có 10 lớp phân loại.
            dropout: Tỷ lệ dropout trên đầu phân loại. Ví dụ: dropout = 0.1 nghĩa là 10% các neuron được dropout.
            vocab_size: Kích thước bảng từ vựng cho nn.Embedding.
        """
        super().__init__()
        if vocab_size is None:
            raise ValueError("vocab_size is required for nn.Embedding.")

        # Thêm lớp embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Khởi tạo backbone Mamba
        config = MambaConfig(d_model=d_model, n_layers=n_layers)
        self.backbone = Mamba(config)

        # Khởi tạo đầu phân loại
        self.classifier = ClassifierHead(
            d_model=d_model, num_classes=num_classes, dropout=dropout
        )

        # Khởi tạo hàm loss
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """
        Luồng xuôi: embedding → backbone → logits; tùy chọn tính loss.

        Args:
            x: Tensor (B, L, 1) (token id dạng float) hoặc tương thích.
            labels: Tensor nhãn (B,) long; nếu None không tính loss.

        Returns:
            Dict có khóa logits; thêm loss nếu có labels.
        """
        # Áp dụng embedding
        x = self.embedding(x.long().squeeze(-1))  # Xử lý token IDs

        # Lấy đầu ra từ backbone
        sequence_output = self.backbone(x)

        # Lấy logits phân loại
        logits = self.classifier(sequence_output)

        output = {"logits": logits}

        # Compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output["loss"] = loss

        return output
