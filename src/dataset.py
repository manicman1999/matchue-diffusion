"""MNIST, prepared for diffusion training.

The whole training set is 60,000 grayscale digits of 28x28 pixels, about 47 MB
as raw bytes, so it lives on the GPU. That skips DataLoader workers entirely and
turns every batch into a single indexing operation.

Diffusion mixes images with Gaussian noise, which is centered on zero with a
standard deviation of one. Pixels are therefore scaled from [0, 255] to [-1, 1]
so the images and the noise live on a comparable scale.
"""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.utils import save_image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(
    device: str | torch.device = "cuda", train: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return uint8 images [N, 1, 28, 28] and int64 labels [N] on `device`."""
    dataset = MNIST(root=DATA_DIR, train=train, download=True)
    # torchvision stores the digits as [N, height, width]; PyTorch layers expect
    # a channel dimension, which is 1 for grayscale.
    images = dataset.data[:, None]
    return images.to(device), dataset.targets.to(device)


def to_model_range(images: Tensor) -> Tensor:
    """uint8 pixels in [0, 255] -> floats in [-1, 1]."""
    return images.float() / 127.5 - 1.0


def to_image_range(x: Tensor) -> Tensor:
    """Floats in [-1, 1] -> floats in [0, 1], ready for save_image."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def batches(
    images: Tensor, labels: Tensor, batch_size: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield shuffled (x, labels) training batches forever.

    Diffusion training is usually measured in steps rather than epochs, so this
    reshuffles and keeps going instead of stopping after one pass. There is no
    flipping or other augmentation: a mirrored 2, 3, or 7 is no longer a digit.
    """
    count = images.shape[0]
    while True:
        order = torch.randperm(count, device=images.device)
        for start in range(0, count - batch_size + 1, batch_size):
            index = order[start:start + batch_size]
            yield to_model_range(images[index]), labels[index]


if __name__ == "__main__":
    images, labels = load_mnist()
    print(f"images: {tuple(images.shape)} {images.dtype} on {images.device}")
    print(f"labels: {tuple(labels.shape)}, counts per digit: {labels.bincount().tolist()}")

    x, y = next(batches(images, labels, batch_size=64))
    print(f"batch: {tuple(x.shape)}, range [{x.min():.2f}, {x.max():.2f}], "
          f"mean {x.mean():.3f}, std {x.std():.3f}")
    print("first row:", " ".join(str(d) for d in y[:8].tolist()))

    out = Path(__file__).resolve().parent.parent / "outputs" / "mnist-batch.png"
    out.parent.mkdir(exist_ok=True)
    save_image(to_image_range(x), out, nrow=8)
    print(f"saved {out}")
