Upload-ready demo images for the Quantum Image Lab.

Each is a handwritten digit from scikit-learn's dataset redrawn as dark ink on paper and deliberately
perturbed (rotation, shift, blur, noise), so none is a pixel-exact copy of a training image.
`digit_5.png` etc. are useful as failure cases when the model was trained on 3 vs 8 only.
Regenerate with `python scripts/make_demo_images.py`.
