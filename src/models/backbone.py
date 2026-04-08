"""
Lõi kiến trúc Mamba (PyTorch): cấu hình, xếp chồng tầng residual, khối SSM.

Tham khảo triển khai ``mamba_simple`` / **mamba-minimal** (johnma2006). Khác
biệt chính: conv 1D bằng ``nn.Conv1d``, selective scan thuần PyTorch (hoặc
tuần tự / CUDA qua ``mamba_ssm`` nếu bật).

Cấu trúc module:

- ``Mamba``: nhiều ``ResidualBlock``.
- ``ResidualBlock``: ``ResidualBlock(x) = MambaBlock(RMSNorm(x)) + x``.
- ``MambaBlock``: nhánh ``in_proj`` tách ``x, z``; conv1d depthwise + SiLU
  trên ``x``; SSM (selective scan); nhân có điều kiện với ``silu(z)`` rồi
  ``out_proj`` (Hình 3 bài Mamba).
"""

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.pscan import pscan


@dataclass
class MambaConfig:
    """
    Hyperparameters for Mamba architecture

    Args:
        d_model: Dimension của input
        n_layers: Số lượng layers
        dt_rank: Rank của tensor delta
        d_state: Dimension của state
        expand_factor: Hệ số mở rộng cho inner dimension
        d_conv: Kích thước kernel của convolution
        dt_min: Giá trị tối thiểu cho tensor delta
        dt_max: Giá trị tối đa cho tensor delta
        dt_init: Phương pháp khởi tạo cho tensor delta
        dt_scale: Hệ số scale cho tensor delta
        dt_init_floor: Giá trị floor cho tensor delta
        rms_norm_eps: Epsilon cho RMSNorm
        base_std: Standard deviation cơ bản cho RMSNorm
        bias: Bias cho các layer tuyến tính
        conv_bias: Bias cho layer convolution
        inner_layernorms: Whether to apply layernorms to the internal activations
        mup: Whether to use muP
        mup_base_width: Base width for muP
        pscan: Whether to use parallel scan mode
        use_cuda: Whether to use CUDA implementation
    Returns:
        MambaConfig object
    """

    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    dropout: float = 0.1  # dropout rate for regularization inside MambaBlock

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(nn.Module):
    """
    Xếp chồng nhiều tầng ResidualBlock; forward xử lý cả chuỗi.
    Args:
        config: MambaConfig object
    Returns:
        Mamba object
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x):
        """
        Args:
            x: ``(B, L, D)`` embedding chuỗi.

        Returns:
            ``(B, L, D)`` sau khi qua tất cả các tầng.
        """
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):
        """
        Suy luận từng bước thời gian (một token): cập nhật ``caches`` mỗi tầng.

        Args:
            x: ``(B, D)`` — một bước embedding sau khi đã embed token hiện tại.
            caches: List cache, mỗi phần tử ứng với một ``ResidualBlock``.

        Returns:
            Tuple ``(output, caches)`` với ``output`` ``(B, D)``.
        """
        # x : (B, D); caches : list of (h, inputs) per layer

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    """
    ResidualBlock là một khối cơ bản của Mamba, bao gồm MambaBlock và RMSNorm.
    Args:
        config: MambaConfig object
    Returns:
        ResidualBlock object
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        """
        Đi qua RMSNorm và MambaBlock, sau đó thêm vào skip connection.
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        """
        Một bước thời gian với cache conv/SSM; x là (B, D).
        Args:
            x: (B, D)
            cache: (h, inputs)
            h: (B, ED, N)
            inputs: (B, ED, d_conv-1)
        Returns:
            (B, D)
            (h, inputs)
        """

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    """
    MambaBlock là một khối cơ bản của Mamba, bao gồm in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj.
    Args:
        config: MambaConfig object
    Returns:
        MambaBlock object
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.dropout = nn.Dropout(config.dropout)

        # Tách đầu vào thành hai nhánh
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # Conv1D theo thời gian
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # Từ x -> delta, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # Từ delta -> d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # Khởi tạo dt_proj
        # dt_proj weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Khởi tạo bias cho dt_proj
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )  # Nghịch đảo của softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # Khởi tạo sẽ set tất cả bias của Linear về 0, cần đánh dấu bias này là _no_reinit

        # Khởi tạo A_log
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(
            torch.log(A)
        )  # Tại sao lưu A trong log ? để giữ A < 0 (cf -torch.exp(...)) ? cho gradient stability ?
        self.A_log._no_weight_decay = True

        # Khởi tạo D
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # Từ d_inner -> d_model
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # Sử dụng trong Jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(
                self.config.dt_rank, config.rms_norm_eps, config.mup
            )
            self.B_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup
            )
            self.C_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup
            )
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        """
        Áp RMSNorm nội bộ cho Δ, B, C.
        Args:
            dt: (B, L, dt_rank)
            B: (B, L, N)
            C: (B, L, N)
        Returns:
            (B, L, dt_rank), (B, L, N), (B, L, N)
        """
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        """
        Đi qua in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj.
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # x : (B, L, D)
        _, L, _ = x.shape
        # Tách đầu vào thành hai nhánh
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner), (B, L, d_inner)

        # Nhánh x
        # Chuyển đổi shape từ (B, L, d_inner) -> (B, d_inner, L)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[
            :, :, :L
        ]  # Convolution theo thời gian, với bộ lọc ngắn
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # Áp activation function SiLU
        x = F.silu(x)

        # Đi qua SSM
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output  # Các phép toán còn lại được thực hiện trong hàm ssm (fused với CUDA pscan)

        # Nhánh z
        # Áp activation function SiLU
        z = F.silu(z)

        # Nhân hai nhánh
        output = y * z

        # Áp dụng dropout trước out_proj để regularize
        output = self.dropout(output)

        # Đi qua out_proj
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        """
        State Space Model với tham số phụ thuộc đầu vào: tính Δ, B, C, quét selective.
        Args:
            x: (B, L, d_inner)
            z: (B, L, d_inner)
        Returns:
            (B, L, d_inner)
        """

        A = -torch.exp(self.A_log.float())  # (d_inner, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*d_state)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)
        # Áp RMSNorm nội bộ cho Δ, B, C (inner_layernorms=True)
        delta, B, C = self._apply_layernorms(delta, B, C)
        # Áp ma trận nhân cho delta
        delta = self.dt_proj.weight @ delta.transpose(
            1, 2
        )  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # Chọn hàm selective_scan tương ứng với config
        if self.config.use_cuda:
            # Đây là các thứ cần thiết cho hàm selective_scan_cuda
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" các phép toán được gộp
            y = self.selective_scan_cuda(
                x,
                delta,
                A,
                B,
                C,
                D,
                z=z,
                delta_softplus=True,
                delta_bias=self.dt_proj.bias.float(),
            )
            y = y.transpose(1, 2)  # (B, L, d_inner)

        else:
            # Áp ma trận nhân cho delta = softplus(dt_proj(delta))
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            # Chọn hàm selective_scan tương ứng với config
            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        """
        Selective scan song song qua PScan.
        Args:
            x: (B, L, d_inner)
            delta: (B, L, d_inner)
            A: (d_inner, N)
            B: (B, L, N)
            C: (B, L, N)
            D: (d_inner)
        Returns:
            (B, L, d_inner)
        """

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, d_inner, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, d_inner, N) @ (B, L, N, 1) -> (B, L, d_inner, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        """
        Selective scan tuần tự.
        Args:
            x: (B, L, d_inner)
            delta: (B, L, d_inner)
            A: (d_inner, N)
            B: (B, L, N)
            C: (B, L, N)
            D: (d_inner)
        Returns:
            (B, L, d_inner)
        """

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, d_inner, N)

        h = torch.zeros(
            x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device
        )  # (B, d_inner, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, d_inner, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, d_inner, N) @ (B, L, N, 1) -> (B, L, d_inner, 1)

        y = y + D * x

        return y

    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        """
        Một bước thời gian với cache conv/SSM; x là (B, D).
        Args:
            x: (B, D)
            cache: (h, inputs)
            h: (B, d_inner, N)
            inputs: (B, d_inner, d_conv-1)
        Returns:
            (B, D)
            (h, inputs)
        """
        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=1)  # (B, d_inner), (B, d_inner)

        # Nhánh x
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[
            :, :, self.config.d_conv - 1
        ]  # (B, d_inner)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # Nhánh z
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        # Chuẩn bị cache cho lời gọi tiếp theo
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  # (B, d_inner, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        """
        Một bước SSM với trạng thái ẩn h; dùng cho step.
        Args:
            x: (B, d_inner)
            h: (B, d_inner, N)
        Returns:
            (B, d_inner)
            (h, inputs)
        """
        A = -torch.exp(
            self.A_log.float()
        )  # (d_inner, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*d_state)

        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, dt_rank), (B, d_state), (B, d_state)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  # (B, d_inner)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, d_inner, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, d_inner, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, d_inner, N)

        if h is None:
            h = torch.zeros(
                x.size(0),
                self.config.d_inner,
                self.config.d_state,
                device=deltaA.device,
            )  # (B, d_inner, N)

        h = deltaA * h + BX  # (B, d_inner, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, d_inner, N) @ (B, d_state, 1) -> (B, d_inner, 1)

        y = y + D * x

        return y, h


class RMSNorm(nn.Module):
    """
    Chuẩn hóa RMS; khi ``use_mup=True`` không dùng tham số scale (theo muP).
    """

    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Scale theo độ lớn RMS của chiều cuối; nhân ``weight`` nếu không muP.
        Args:
            x: (B, L, d_inner)
        Returns:
            (B, L, d_inner)
        """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
