"""Turn a run's sample grids into a labeled timelapse GIF.

From the repository root, after training:

    .venv/Scripts/python src/timelapse.py --run mnist

Every grid starts from the same fixed noise, so the GIF shows the same digits
forming as the model improves.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="mnist", help="name of the run folder")
    parser.add_argument("--scale", type=int, default=2, help="pixel upscaling factor")
    parser.add_argument("--frame-ms", type=int, default=250)
    parser.add_argument("--hold-ms", type=int, default=2500, help="pause on the last frame")
    args = parser.parse_args()

    run_dir = ROOT / "outputs" / "runs" / args.run
    paths = sorted((run_dir / "samples").glob("step-*.png"))
    if not paths:
        raise SystemExit(f"no sample grids in {run_dir / 'samples'}; train first")

    header = 18 * args.scale
    font = ImageFont.load_default(size=11 * args.scale)
    frames = []
    for path in paths:
        grid = Image.open(path).convert("RGB")
        grid = grid.resize((grid.width * args.scale, grid.height * args.scale), Image.NEAREST)
        frame = Image.new("RGB", (grid.width, grid.height + header), "black")
        frame.paste(grid, (0, header))
        step = int(path.stem.removeprefix("step-"))
        ImageDraw.Draw(frame).text((frame.width // 2, header // 2), f"step {step:,}",
                                   fill="white", font=font, anchor="mm")
        frames.append(frame)

    out = run_dir / "timelapse.gif"
    durations = [args.frame_ms] * (len(frames) - 1) + [args.hold_ms]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"{len(frames)} frames saved to {out}")


if __name__ == "__main__":
    main()
