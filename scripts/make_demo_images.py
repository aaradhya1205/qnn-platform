"""Generate demo images to upload into the dashboard.

Each image is a handwritten digit from scikit-learn's dataset, redrawn as dark
ink on off-white 'paper' and deliberately perturbed (small rotation, shift,
blur, sensor noise) so it is NOT a pixel-exact copy of anything the model
trained on. Run:  python scripts/make_demo_images.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.datasets import load_digits

OUT = Path(__file__).resolve().parent.parent / "demo_images"
SIDE = 224


def render(digit_8x8: np.ndarray, rng: np.random.RandomState, angle: float, shift: tuple[int, int],
           noise: float, blur: float) -> Image.Image:
    ink = digit_8x8 / digit_8x8.max()
    small = Image.fromarray((ink * 255).astype(np.uint8), mode="L")
    canvas_side = SIDE
    big = small.resize((int(canvas_side * 0.78),) * 2, Image.BICUBIC)
    canvas = Image.new("L", (canvas_side, canvas_side), 0)
    offset = ((canvas_side - big.width) // 2 + shift[0], (canvas_side - big.height) // 2 + shift[1])
    canvas.paste(big, offset)
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
    canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(canvas, dtype=float) / 255.0
    arr = np.clip(arr + rng.normal(0, noise, arr.shape), 0, 1)
    paper = 245 - arr * 235  # dark ink on off-white paper
    return Image.fromarray(paper.astype(np.uint8), mode="L").convert("RGB")


def main():
    OUT.mkdir(exist_ok=True)
    digits = load_digits()
    rng = np.random.RandomState(7)
    specs = []
    for d in range(10):
        idx = np.where(digits.target == d)[0]
        specs.append((d, int(rng.choice(idx)), rng.uniform(-9, 9), (int(rng.randint(-8, 9)), int(rng.randint(-8, 9))), 0.04, 1.4))
    extras = [(3, 0.0, (0, 0), 0.02, 0.8, "clean"), (8, 12.0, (6, -6), 0.10, 2.0, "rotated_noisy")]
    for d, i, angle, shift, noise, blur in specs:
        render(digits.images[i], rng, angle, shift, noise, blur).save(OUT / f"digit_{d}.png")
    for d, angle, shift, noise, blur, tag in extras:
        idx = np.where(digits.target == d)[0]
        i = int(rng.choice(idx))
        render(digits.images[i], rng, angle, shift, noise, blur).save(OUT / f"digit_{d}_{tag}.png")
    print(f"Wrote {len(list(OUT.glob('*.png')))} images to {OUT}")


if __name__ == "__main__":
    main()
