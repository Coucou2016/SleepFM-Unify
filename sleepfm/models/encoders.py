"""1D EfficientNet-style encoders (based on rthapa84/sleepfm-codebase)."""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        expansion: int,
        activation,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(in_channel, in_channel * expansion, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channel * expansion,
            in_channel * expansion,
            kernel_size=3,
            groups=in_channel * expansion,
            padding=padding,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(in_channel * expansion, out_channel, kernel_size=1, stride=1)
        self.b0 = nn.BatchNorm1d(in_channel * expansion)
        self.b1 = nn.BatchNorm1d(in_channel * expansion)
        self.d = nn.Dropout()
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.b0(self.conv1(x)))
        y = self.act(self.b1(self.conv2(y)))
        y = self.conv3(y)
        y = self.d(y)
        if self.stride == 1:
            return x + y
        return y


class MBConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channels: int,
        expansion: int,
        layers: int,
        activation=nn.ReLU6,
        stride: int = 2,
    ):
        super().__init__()
        stack = OrderedDict()
        for i in range(layers - 1):
            stack[f"s{i}"] = Bottleneck(in_channel, in_channel, expansion, activation)
        stack[f"s{layers + 1}"] = Bottleneck(
            in_channel, out_channels, expansion, activation, stride=stride
        )
        self.stack = nn.Sequential(stack)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.stack(x))


class EffNet(nn.Module):
    """Modality-specific 1D CNN encoder for contrastive pretraining."""

    def __init__(
        self,
        in_channel: int,
        num_additional_features: int = 0,
        depth=None,
        channels=None,
        dilation: int = 1,
        stride: int = 2,
        expansion: int = 6,
        embedding_dim: int = 512,
    ):
        super().__init__()
        if depth is None:
            depth = [1, 2, 2, 3, 3, 3, 3]
        if channels is None:
            channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]

        self.stage1 = nn.Conv1d(
            in_channel, channels[0], kernel_size=3, stride=stride, padding=1, dilation=dilation
        )
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_additional_features > 0:
            x, additional = x
        x = self.b0(self.stage1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.pool(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.act(self.aap(x)[:, :, 0])
        x = self.drop(x)
        if self.num_additional_features > 0:
            x = torch.cat((x, additional), dim=1)
        return self.fc(x)


class EffNetSupervised(nn.Module):
    """Supervised baseline: all modalities concatenated on channel axis.

    Paper comparison CNN: train on the ``train`` split (not pretrain), evaluate
    on ``test`` with the same staging / apnea metrics as the linear probe.
    """

    def __init__(
        self,
        in_channel: int,
        num_classes: int = 5,
        num_additional_features: int = 0,
        depth=None,
        channels=None,
        dilation: int = 1,
        stride: int = 2,
        expansion: int = 6,
    ):
        super().__init__()
        if depth is None:
            depth = [1, 2, 2, 3, 3, 3, 3]
        if channels is None:
            channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]

        self.stage1 = nn.Conv1d(
            in_channel, channels[0], kernel_size=3, stride=stride, padding=1, dilation=dilation
        )
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_additional_features > 0:
            x, additional = x
        x = self.b0(self.stage1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.pool(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.act(self.aap(x)[:, :, 0])
        x = self.drop(x)
        if self.num_additional_features > 0:
            x = torch.cat((x, additional), dim=1)
        return self.fc(x)


class SeqStagingBaseline(nn.Module):
    """Lightweight epoch CNN + temporal head for night-level staging.

    **Not** a U-Sleep port — labeled SeqStagingBaseline for fairer sequence
    comparison against SleepFM night eval without vendoring u-sleep.

    Input: ``(B, L, C, T)`` concatenated modalities per epoch, or ``(B, C, T)``
    for single-epoch mode (temporal head becomes a no-op identity path).
    Output logits: ``(B, L, num_classes)`` or ``(B, num_classes)``.
    """

    def __init__(
        self,
        in_channel: int,
        num_classes: int = 5,
        hidden: int = 64,
        temporal: str = "gru",
        temporal_layers: int = 1,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.hidden = hidden
        self.epoch_encoder = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.temporal_kind = temporal
        if temporal == "gru":
            self.temporal = nn.GRU(
                hidden, hidden, num_layers=temporal_layers, batch_first=True, bidirectional=True
            )
            self.head = nn.Linear(hidden * 2, num_classes)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
            self.head = nn.Linear(hidden, num_classes)

    def encode_epochs(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T) → (N, H)."""
        h = self.epoch_encoder(x).squeeze(-1)
        return h

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) or (B, L, C, T)
            padding_mask: optional (B, L) with True = pad (same as NightSequenceDataset)
        """
        if x.ndim == 3:
            # Route through temporal with L=1 so GRU bidirectional head dims match
            x = x.unsqueeze(1)
            if padding_mask is None:
                padding_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
            logits = self.forward(x, padding_mask=padding_mask)
            return logits.squeeze(1)
        if x.ndim != 4:
            raise ValueError(f"SeqStagingBaseline expects (B,C,T) or (B,L,C,T), got {tuple(x.shape)}")
        b, length, c, t = x.shape
        flat = x.reshape(b * length, c, t)
        h = self.encode_epochs(flat).view(b, length, -1)
        if self.temporal_kind == "gru":
            if padding_mask is not None:
                h = h.clone()
                pad = padding_mask.bool()
                h[pad] = 0
                lengths = (~pad).sum(dim=1).clamp(min=1).to(dtype=torch.long)
                packed = pack_padded_sequence(
                    h, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.temporal(packed)
                out, _ = pad_packed_sequence(
                    packed_out, batch_first=True, total_length=length
                )
            else:
                out, _ = self.temporal(h)
            logits = self.head(out)
        else:
            key_padding = None
            if padding_mask is not None:
                key_padding = padding_mask.bool()
            out = self.temporal(h, src_key_padding_mask=key_padding)
            logits = self.head(out)
        return logits
