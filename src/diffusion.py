"""The diffusion process: how images are noised, and how the noise is removed.

Timesteps run from 1 (barely noisy) to T (pure noise). Each schedule table has an
extra entry at index 0 meaning "no noise at all", so a timestep indexes the
tables directly and t - 1 always means "one step cleaner".

Notation, per step t:
    beta[t]       fresh noise variance added when going from step t - 1 to t
    alpha[t]      1 - beta[t], how much of the signal each step keeps
    alpha_bar[t]  alpha[1] * ... * alpha[t], how much of the original image
                  survives all the way to step t
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


def linear_betas(steps: int = 1000) -> Tensor:
    """The original DDPM schedule: per-step noise rises evenly from 1e-4 to 0.02."""
    return torch.linspace(1e-4, 0.02, steps, dtype=torch.float64)


def cosine_betas(steps: int = 1000, offset: float = 0.008) -> Tensor:
    """The cosine schedule from Improved DDPM (Nichol & Dhariwal, 2021).

    The linear schedule destroys small images too quickly: for a 28x28 digit,
    the last few hundred steps are almost pure noise and teach the model little.
    Here the surviving fraction of the image follows a cosine curve instead, so
    information fades more evenly across all the steps.
    """
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    surviving = torch.cos((t + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_bars = surviving / surviving[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    # The final steps would otherwise approach beta = 1, which is numerically unstable.
    return betas.clamp(max=0.999)


class Diffusion:
    """The noise schedule, plus the training loss and the sampler built on it."""

    def __init__(self, betas: Tensor, device: str | torch.device = "cuda") -> None:
        self.steps = len(betas)
        betas = torch.cat([betas.new_zeros(1), betas])
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        self.alpha_bars = alpha_bars.float().to(device)

    def add_noise(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Jump straight from clean images x0 to step t, with no loop.

        x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
        """
        alpha_bar = self.alpha_bars[t][:, None, None, None]
        return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    def loss(self, model: nn.Module, x0: Tensor) -> tuple[Tensor, Tensor]:
        """Noise each image to a random step and score the model's noise guess.

        Returns the squared error per image [B] and the timesteps used [B], so
        the training loop can see which noise levels are hard.
        """
        t = torch.randint(1, self.steps + 1, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        predicted = model(self.add_noise(x0, t, noise), t).float()
        return (predicted - noise).square().mean(dim=(1, 2, 3)), t

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        noise: Tensor,
        generator: torch.Generator | None = None,
        on_step: Callable[[int, Tensor, Tensor], None] | None = None,
    ) -> Tensor:
        """Turn pure noise [B, C, H, W] into images in [-1, 1], one step at a time.

        At every step the model's noise estimate gives a guess at the finished
        image. That guess is still blurry when the image is very noisy, so rather
        than jumping to it, we only move to step t - 1: a blend of the guess and
        the current image, plus a little fresh randomness that lets the sample
        commit to one specific image instead of an average of many.

        `on_step(t, x_t, x0_guess)`, if given, is called at every step with the
        current noisy images and the model's guess at the clean ones.
        """
        x = noise
        for step in range(self.steps, 0, -1):
            t = torch.full((x.shape[0],), step, device=x.device)
            predicted_noise = model(x, t).float()
            beta = self.betas[step]
            alpha_bar = self.alpha_bars[step]
            alpha_bar_prev = self.alpha_bars[step - 1]

            # Remove all of the predicted noise at once to get the model's current
            # guess at the clean image, kept within the valid pixel range.
            x0_guess = (x - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            x0_guess = x0_guess.clamp(-1, 1)
            if on_step is not None:
                on_step(step, x, x0_guess)

            # The most likely image one step cleaner, given that guess and where
            # we are now. At the final step this weights the guess fully.
            guess_weight = alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)
            current_weight = self.alphas[step].sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)
            x = guess_weight * x0_guess + current_weight * x

            if step > 1:
                variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                x = x + variance.sqrt() * torch.randn(
                    x.shape, device=x.device, generator=generator,
                )
        return x
