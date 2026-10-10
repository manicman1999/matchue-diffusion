"""Generate digits from a trained run's checkpoint.

From the repository root:

    .venv/Scripts/python src/sample.py --run mnist --count 64 --seed 0

Different seeds give different digits; the same seed always gives the same ones.
Add --gif to also save the denoising itself: the noisy images on the left and
the model's guess at the finished digits on the right, step by step.
"""

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.utils import make_grid, save_image

from dataset import to_image_range
from diffusion import Diffusion, cosine_betas
from model import UNet

ROOT = Path(__file__).resolve().parent.parent


def to_pil(images: Tensor, nrow: int, scale: int) -> Image.Image:
    """Images in [-1, 1] [B, C, H, W] -> one upscaled grid image."""
    grid = make_grid(to_image_range(images), nrow=nrow, padding=2)
    pixels = (grid * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(pixels)
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def save_denoising_gif(
    frames: list[tuple[int, Tensor, Tensor]], nrow: int, scale: int, path: Path,
) -> None:
    """Save each (t, noisy images, guessed clean images) frame side by side."""
    font = ImageFont.load_default(size=8 * scale)
    header, gap = 14 * scale, 4 * scale
    rendered = []
    for t, noisy, guess in frames:
        left, right = to_pil(noisy, nrow, scale), to_pil(guess, nrow, scale)
        frame = Image.new("RGB", (left.width * 2 + gap, left.height + header), "black")
        frame.paste(left, (0, header))
        frame.paste(right, (left.width + gap, header))
        draw = ImageDraw.Draw(frame)
        draw.text((left.width // 2, header // 2), f"noisy image, t = {t}",
                  fill="white", font=font, anchor="mm")
        draw.text((left.width + gap + right.width // 2, header // 2), "model's guess",
                  fill="white", font=font, anchor="mm")
        rendered.append(frame)
    durations = [600] + [40] * (len(rendered) - 2) + [3000]
    rendered[0].save(path, save_all=True, append_images=rendered[1:],
                     duration=durations, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="mnist", help="name of the run folder")
    parser.add_argument("--count", type=int, default=64, help="number of digits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="output image path")
    parser.add_argument("--gif", action="store_true", help="also save a denoising GIF")
    parser.add_argument("--gif-every", type=int, default=10,
                        help="sampling steps between GIF frames")
    parser.add_argument("--gif-scale", type=int, default=4, help="GIF pixel upscaling")
    args = parser.parse_args()

    device = "cuda"
    checkpoint = torch.load(ROOT / "outputs" / "runs" / args.run / "checkpoint.pt",
                            map_location=device)
    # The averaged (EMA) weights give cleaner samples than the raw training weights.
    model = UNet().to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()

    frames: list[tuple[int, Tensor, Tensor]] = []

    def record(t: int, noisy: Tensor, guess: Tensor) -> None:
        if t % args.gif_every == 0 or t == 1:
            frames.append((t, noisy.float().cpu(), guess.float().cpu()))

    generator = torch.Generator(device).manual_seed(args.seed)
    noise = torch.randn(args.count, 1, 28, 28, device=device, generator=generator)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        images = Diffusion(cosine_betas(), device).sample(
            model, noise, generator, on_step=record if args.gif else None,
        )

    nrow = math.ceil(math.sqrt(args.count))
    out = args.out or ROOT / "outputs" / f"{args.run}-seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_image_range(images), out, nrow=nrow)
    print(f"{args.count} samples from step {checkpoint['step']:,} saved to {out}")

    if args.gif:
        final = images.float().cpu()
        frames.append((0, final, final))
        gif = out.with_name(f"{out.stem}-denoising.gif")
        save_denoising_gif(frames, nrow, args.gif_scale, gif)
        print(f"{len(frames)}-frame denoising GIF saved to {gif}")


if __name__ == "__main__":
    main()
