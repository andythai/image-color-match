import numpy as np
from PIL import Image

inp = Image.open("input.png").convert("RGB")
tgt = Image.open("target.jpg").convert("RGB").resize(inp.size, Image.LANCZOS)

def feats(x):  # degree-3 polynomial in RGB incl. cross terms
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b,
                     r**3, g**3, b**3, r*r*g, r*r*b, g*g*r, g*g*b, b*b*r, b*b*g, r*g*b], 1)

# fit on a downsampled aligned pair (blurs out JPEG noise / minor misalignment)
small = (inp.width // 4, inp.height // 4)
X = np.asarray(inp.resize(small, Image.LANCZOS), np.float64).reshape(-1, 3) / 255
Y = np.asarray(tgt.resize(small, Image.LANCZOS), np.float64).reshape(-1, 3) / 255
W, *_ = np.linalg.lstsq(feats(X), Y, rcond=None)

full = np.asarray(inp, np.float64).reshape(-1, 3) / 255
out = np.clip(feats(full) @ W, 0, 1).reshape(inp.height, inp.width, 3)
Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save("output.png")

err = np.abs(feats(X) @ W - Y).mean() * 255
print(f"mean abs error vs target: {err:.2f}/255")
assert err < 15
