"""
GAFIS - Pix2Pix Dataset Generation Pipeline
--------------------------------------------
Phase A: SOCOFing -> resize -> CLAHE -> normalize -> save as target/xxxxx.png
Phase B: target -> blur/contrast/occlusion -> + PolyU noise -> save as input/xxxxx.png

Run this in your local Jupyter environment (Windows paths as in your notebook).
Requires: opencv-python, numpy
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil

# ---------------------------
# Paths - EDIT THESE IF NEEDED
# ---------------------------
SOCOFING_ROOT = Path(
    r"C:\Users\sarva\OneDrive\Desktop\GAFIS\dataset_img\SocoFing_Real\Real\Real"
)
NOISE_LIB_DIR = Path("NoiseLibrary")          # already built in your notebook (100 .npy files)

OUTPUT_ROOT = Path("dataset")
TARGET_DIR = OUTPUT_ROOT / "target"
INPUT_DIR = OUTPUT_ROOT / "input"

TARGET_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256

# ---------------------------
# Load noise library once
# ---------------------------
noise_files = list(NOISE_LIB_DIR.glob("*.npy"))
if len(noise_files) == 0:
    raise RuntimeError(
        f"No .npy noise files found in {NOISE_LIB_DIR}. "
        "Run the PolyU noise-extraction cell first."
    )
print(f"Loaded {len(noise_files)} PolyU noise maps")


# ---------------------------
# Phase A: Preprocessing -> clean target
# ---------------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img


# ---------------------------
# Phase B, step 1: synthetic latent degradation
# ---------------------------
def realistic_smudge_mask(h, w, n_blobs, blob_scale):
    """Irregular soft-edged smudge mask instead of a rectangle. blob_scale controls size."""
    mask = np.zeros((h, w), dtype=np.float32)

    radius = int(min(h, w) * blob_scale)
    for _ in range(n_blobs):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        r = np.random.randint(max(radius // 2, 4), radius)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)

    # feather the edges so it's a gradient, not a hard boundary
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(radius / 3, 2))
    mask = np.clip(mask / mask.max(), 0, 1)
    return mask


def directional_smear(img, angle_deg, length):
    """Motion-blur kernel at a given angle -- mimics a finger sliding across a sensor."""
    length = max(3, length | 1)  # force odd
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def degrade(img, level="moderate"):
    latent = img.copy().astype(np.float32)
    h, w = latent.shape

    if level == "mild":
        global_sigma, n_blobs, blob_scale, strength = 1.0, 2, 0.18, 0.45
    elif level == "moderate":
        global_sigma, n_blobs, blob_scale, strength = 1.5, 3, 0.28, 0.70
    else:  # severe
        global_sigma, n_blobs, blob_scale, strength = 2.0, 4, 0.38, 0.90

    # mild overall softening across the whole print (real scans are never perfectly sharp)
    latent = cv2.GaussianBlur(latent, (0, 0), global_sigma)

    # directional smear, mimicking a finger sliding on the sensor
    smear_angle = np.random.uniform(0, 180)
    smear_length = np.random.randint(9, 21)
    smeared = directional_smear(latent, smear_angle, smear_length)

    smudge_mask = realistic_smudge_mask(h, w, n_blobs, blob_scale) * strength

    # randomize wet (dark ink pooling) vs dry (light/faded) smudge character
    if np.random.rand() < 0.5:
        fade_target = np.random.uniform(15, 70)     # wet/dark smear
    else:
        fade_target = np.random.uniform(200, 245)   # dry/faded smear

    latent = latent * (1 - smudge_mask) + (smeared * 0.5 + fade_target * 0.5) * smudge_mask

    return np.clip(latent, 0, 255).astype(np.uint8)


# ---------------------------
# Phase B, step 2: PolyU sensor noise injection
# ---------------------------
def add_polyu_noise(img, noise_files):
    noise_path = random.choice(noise_files)
    noise = np.load(noise_path)

    noise = cv2.resize(noise, (IMG_SIZE, IMG_SIZE))
    if noise.ndim == 3:
        noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)

    alpha = np.random.uniform(0.10, 0.25)

    out = img.astype(np.float32) + alpha * noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ---------------------------
# Main generation loop
# ---------------------------
def generate_dataset(level_choices=("mild", "moderate", "severe"),
                      level_weights=(0.25, 0.5, 0.25)):
    """
    level_choices/level_weights let you mix degradation strengths
    across the dataset instead of using a single fixed level.
    Set level_weights=(0,1,0) to force everything to 'moderate', etc.
    """
    count = 0
    skipped = 0

    subject_folders = sorted(p for p in SOCOFING_ROOT.iterdir() if p.is_dir())

    for subject_folder in subject_folders:
        for img_path in sorted(subject_folder.glob("*.BMP")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue

            # Phase A
            clean = preprocess(img)

            # Phase B
            level = random.choices(level_choices, weights=level_weights, k=1)[0]
            degraded = degrade(clean, level=level)
            degraded = add_polyu_noise(degraded, noise_files)

            fname = f"{count:05d}.png"
            cv2.imwrite(str(TARGET_DIR / fname), clean)
            cv2.imwrite(str(INPUT_DIR / fname), degraded)

            count += 1
            if count % 500 == 0:
                print(f"Processed {count} images...")

    print(f"\nDone. Saved {count} pairs. Skipped {skipped} unreadable files.")
    return count


# ---------------------------
# Zip target/ and input/ for download
# ---------------------------
def zip_outputs():
    target_zip = shutil.make_archive("target_images", "zip", root_dir=TARGET_DIR)
    input_zip = shutil.make_archive("input_images", "zip", root_dir=INPUT_DIR)
    print(f"Zipped: {target_zip}")
    print(f"Zipped: {input_zip}")
    return target_zip, input_zip


if __name__ == "__main__":
    n = generate_dataset()
    zip_outputs()