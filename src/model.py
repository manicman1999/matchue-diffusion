"""A U-Net that looks at a noisy image and predicts the noise mixed into it.

The shape of the network with the default settings, for 28x28 MNIST digits:

    28x28,  32 ch  ─────────── skip ───────────►  28x28,  32 ch  ──► predicted noise
          ▼ downsample                                  ▲ upsample
    14x14,  64 ch  ─────────── skip ───────────►  14x14,  64 ch
          ▼ downsample                                  ▲ upsample
     7x7,  128 ch  + attention ─── skip ───────►   7x7,  128 ch  + attention
          └──────────► middle, 7x7, 128 ch ─────────────┘

The way down shrinks the image so deeper layers see the big picture: which digit
and roughly how it sits. The way up expands back to full size. Skip connections
hand fine detail from the way down straight across to the way up, so it never
has to squeeze through the 7x7 bottleneck.

Every residual block also receives the timestep, so the same weights can act
differently on heavy noise (guess the composition) and light noise (fix texture).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Encode integer timesteps [B] as sines and cosines at many frequencies [B, dim].

    A raw number like 637 makes a poor network input. Spreading it across waves
    of different speeds gives every timestep a distinct fingerprint that changes
    smoothly with t, the same trick transformers use for token positions.
    """
    half = dim // 2
    frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a shortcut around them, nudged by the timestep."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # The shortcut needs a 1x1 convolution when the channel count changes.
        self.shortcut: nn.Module = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # The timestep shifts every channel up or down by a learned amount.
        h = h + self.time(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.shortcut(x) + h


class Attention(nn.Module):
    """Self-attention across pixels, so distant parts of the image can coordinate.

    Convolutions only see small neighborhoods. Attention lets every pixel look at
    every other pixel, which helps keep a whole stroke consistent. By default it
    runs only at 7x7, where there are few enough pixels for it to be cheap.
    """

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        # Each of q, k, v becomes [batch, heads, pixels, channels per head].
        q, k, v = qkv.transpose(-1, -2).unbind(dim=1)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.out(attended)


class Level(nn.Module):
    """A run of residual blocks at one resolution, each optionally followed by attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        count: int,
        time_dim: int,
        dropout: float,
        attention: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_dim, dropout)
            for i in range(count)
        )
        self.attention = nn.ModuleList(
            Attention(out_channels) if attention else nn.Identity()
            for _ in range(count)
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        for block, attention in zip(self.blocks, self.attention):
            x = attention(block(x, t_emb))
        return x


class Upsample(nn.Module):
    """Double the resolution, then smooth the blocky result with a convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Predict the noise in `x` (images in [-1, 1]) at integer timesteps `t`.

    `channels` sets the width of each resolution level, from full size down.
    `attention` says which of those levels get self-attention.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128),
        attention: tuple[bool, ...] = (False, False, True),
        blocks_per_level: int = 2,
        dropout: float = 0.1,
        image_channels: int = 1,
    ) -> None:
        super().__init__()
        base = channels[0]
        time_dim = base * 4
        self.embedding_dim = base
        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(image_channels, base, 3, padding=1)

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        width = base
        for i, level_width in enumerate(channels):
            self.down.append(
                Level(width, level_width, blocks_per_level, time_dim, dropout, attention[i])
            )
            width = level_width
            if i < len(channels) - 1:
                self.downsample.append(nn.Conv2d(width, width, 3, stride=2, padding=1))

        self.middle = Level(width, width, 2, time_dim, dropout, attention=True)

        # Each level on the way up takes the current features plus the skip saved
        # at the same resolution on the way down, concatenated along channels.
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in reversed(range(len(channels))):
            self.up.append(
                Level(width + channels[i], channels[i], blocks_per_level, time_dim,
                      dropout, attention[i])
            )
            width = channels[i]
            if i > 0:
                self.upsample.append(Upsample(width))

        # Start out predicting zero noise rather than random garbage, which makes
        # the first steps of training calmer.
        to_image = nn.Conv2d(width, image_channels, 3, padding=1)
        nn.init.zeros_(to_image.weight)
        nn.init.zeros_(to_image.bias)
        self.out = nn.Sequential(nn.GroupNorm(32, width), nn.SiLU(), to_image)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Map noisy images [B, C, H, W] and timesteps [B] to predicted noise [B, C, H, W]."""
        t_emb = self.time_mlp(timestep_embedding(t, self.embedding_dim))
        x = self.stem(x)

        skips: list[Tensor] = []
        for i, level in enumerate(self.down):
            x = level(x, t_emb)
            skips.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)

        x = self.middle(x, t_emb)

        for i, level in enumerate(self.up):
            x = level(torch.cat([x, skips.pop()], dim=1), t_emb)
            if i < len(self.upsample):
                x = self.upsample[i](x)

        return self.out(x)


if __name__ == "__main__":
    import time

    device = "cuda"
    model = UNet().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    x = torch.randn(128, 1, 28, 28, device=device)
    t = torch.randint(1, 1001, (128,), device=device)
    noise_prediction = model(x, t)
    print(f"input {tuple(x.shape)} -> output {tuple(noise_prediction.shape)}")

    # Rough speed of one training step: forward and backward on a batch of 128.
    for _ in range(3):
        model(x, t).square().mean().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        model(x, t).square().mean().backward()
    torch.cuda.synchronize()
    print(f"forward + backward, batch 128: {(time.perf_counter() - start) / 10 * 1000:.0f} ms")
