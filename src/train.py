"""Train the U-Net to predict noise on MNIST.

From the repository root:

    .venv/Scripts/python src/train.py --run first

Everything for a run goes in outputs/runs/<run>/: a checkpoint, a log.csv, and a
sample grid every few thousand steps. Ctrl+C saves a checkpoint before exiting,
and running the same command again resumes from it.
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from dataset import batches, load_mnist, to_image_range
from diffusion import Diffusion, cosine_betas
from model import UNet

ROOT = Path(__file__).resolve().parent.parent

# The loss is also reported separately for this many equal ranges of timesteps,
# from least noisy to most, since the overall average hides where progress is.
LOSS_BANDS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="mnist", help="name of the run folder")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1_000,
                        help="steps to ramp the learning rate up from zero")
    # 0.999 averages over roughly the last 1,000 steps: short enough that each
    # timelapse frame reflects the model at that point, not one far behind it.
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=500)
    return parser.parse_args()


def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """Move every averaged weight a small step toward the current training weight."""
    with torch.no_grad():
        for averaged, current in zip(ema.parameters(), model.parameters()):
            averaged.lerp_(current, 1 - decay)


def save_checkpoint(
    path: Path, step: int, model: nn.Module, ema: nn.Module, optimizer: torch.optim.Optimizer,
) -> None:
    # Write to a temporary file first, so an interrupted save never corrupts
    # the previous checkpoint.
    temporary = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, temporary)
    temporary.replace(path)


def save_samples(diffusion: Diffusion, ema: nn.Module, noise: torch.Tensor, path: Path) -> None:
    # Fixing the sampler's fresh noise too means every grid differs only
    # because the model changed.
    generator = torch.Generator(noise.device).manual_seed(1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        images = diffusion.sample(ema, noise, generator)
    save_image(to_image_range(images), path, nrow=8)


def main() -> None:
    args = parse_args()
    device = "cuda"
    run_dir = ROOT / "outputs" / "runs" / args.run
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    log_path = run_dir / "log.csv"

    images, labels = load_mnist(device)
    data = batches(images, labels, args.batch_size)
    diffusion = Diffusion(cosine_betas(), device)

    model = UNet(image_channels=images.shape[1]).to(device)
    # A slow running average of the weights. Training weights jitter from batch
    # to batch; the average is steadier and produces noticeably cleaner samples.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        print(f"resumed {args.run} from step {step:,}")

    # The same 64 starting images every time, so grids from different steps
    # show the same samples improving.
    fixed_noise = torch.randn(
        64, *images.shape[1:], device=device, generator=torch.Generator(device).manual_seed(0),
    )

    band_loss = torch.zeros(LOSS_BANDS, device=device)
    band_count = torch.zeros(LOSS_BANDS, device=device)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as log_file:
        log = csv.writer(log_file)
        if write_header:
            log.writerow(["step", "loss", *(f"loss_band{i + 1}" for i in range(LOSS_BANDS)),
                          "lr", "steps_per_sec"])

        model.train()
        window_start, window_step = time.perf_counter(), step
        try:
            while step < args.steps:
                x0, _ = next(data)
                lr = args.lr * min(1.0, (step + 1) / args.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, t = diffusion.loss(model, x0)
                optimizer.zero_grad(set_to_none=True)
                loss.mean().backward()
                # Rare huge gradients would otherwise throw the weights far off course.
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                # Early on, average over a shorter window so the EMA is not
                # stuck near the random starting weights.
                update_ema(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))

                band = (t - 1) * LOSS_BANDS // diffusion.steps
                band_loss.index_add_(0, band, loss.detach())
                band_count.index_add_(0, band, torch.ones_like(loss))

                if step % args.log_every == 0:
                    bands = (band_loss / band_count.clamp(min=1)).tolist()
                    mean_loss = (band_loss.sum() / band_count.sum()).item()
                    speed = (step - window_step) / (time.perf_counter() - window_start)
                    remaining = (args.steps - step) / speed / 60
                    print(f"step {step:>7,}  loss {mean_loss:.4f}  by noise "
                          + " ".join(f"{b:.4f}" for b in bands)
                          + f"  {speed:.1f} steps/s  {remaining:.0f} min left")
                    log.writerow([step, mean_loss, *bands, lr, speed])
                    log_file.flush()
                    band_loss.zero_()
                    band_count.zero_()
                    window_start, window_step = time.perf_counter(), step

                if step % args.sample_every == 0 or step == args.steps:
                    save_checkpoint(checkpoint_path, step, model, ema, optimizer)
                    path = run_dir / "samples" / f"step-{step:07d}.png"
                    save_samples(diffusion, ema, fixed_noise, path)
                    print(f"saved checkpoint and {path.relative_to(ROOT)}")
                    window_start, window_step = time.perf_counter(), step
        except KeyboardInterrupt:
            save_checkpoint(checkpoint_path, step, model, ema, optimizer)
            print(f"\nstopped at step {step:,}; checkpoint saved, rerun to resume")


if __name__ == "__main__":
    main()
