"""
run_demo.py
-----------
Restores a photo WITHOUT the window (GUI) and prints the results.
Good for testing and for showing the steps in your report.

Usage:
    python run_demo.py                  (uses sample_old_photo.jpg)
    python run_demo.py my_old_photo.jpg
"""

import os
import sys
import cv2
import numpy as np

import restoration

path = sys.argv[1] if len(sys.argv) > 1 else "sample_old_photo.jpg"

old = restoration.load_image(path)
if old is None:
    print("Could not open the image:", path)
    print("Run  python make_damaged_sample.py  first to create a test photo.")
    sys.exit(1)

# The steps we want to apply
options = {
    "scratches": True, "scratch_level": 5,
    "noise": True,     "noise_level": 6,
    "tint": True,
    "contrast": True,  "contrast_level": 2,
    "sharpen": True,   "sharpen_level": 5,
    "look": "color",
}

# Run the steps one by one and save each result (useful for the report)
steps = [
    ("1_scratches_removed", lambda im: restoration.remove_scratches(im, 5)),
    ("2_noise_removed", lambda im: restoration.remove_noise(im, 6)),
    ("3_tint_removed", lambda im: restoration.remove_tint(im, 0.6)),
    ("4_contrast_improved", lambda im: restoration.improve_contrast(im, 2)),
    ("5_sharpened", lambda im: restoration.sharpen(im, 0.75)),
]

os.makedirs("steps", exist_ok=True)
current = old
print("Running the restoration steps...")
for name, function in steps:
    current = function(current)
    restoration.save_image("steps/" + name + ".jpg", current)
    print("  done:", name)

restored = restoration.restore_photo(old, options)
restoration.save_image("restored_photo.jpg", restored)

# Before and after, side by side
side_by_side = np.hstack([old, restored])
restoration.save_image("before_after.jpg", side_by_side)

print()
print("RESULTS")
print("-" * 40)
print("Noise      before: %8.2f   after: %8.2f   (lower is cleaner)" % (restoration.noise_level(old), restoration.noise_level(restored)))
print("Contrast   before: %8.1f   after: %8.1f" % (restoration.contrast(old), restoration.contrast(restored)))

# If the clean original is available we can also calculate PSNR
if os.path.exists("sample_original.jpg") and path == "sample_old_photo.jpg":
    original = restoration.load_image("sample_original.jpg")
    if original is not None and original.shape == old.shape:
        print("PSNR vs original  before: %.2f dB   after: %.2f dB" % (
            restoration.psnr(original, old), restoration.psnr(original, restored)))

print("-" * 40)
print("Saved: restored_photo.jpg, before_after.jpg and the steps/ folder")
