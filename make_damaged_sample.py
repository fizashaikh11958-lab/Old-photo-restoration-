"""
make_damaged_sample.py
----------------------
Makes a fake "old damaged photo" so the project can be tested
even if you do not have a real old photo.

Usage:
    python make_damaged_sample.py               (draws a simple picture first)
    python make_damaged_sample.py my_photo.jpg  (damages your own clean photo)

Two files are created:
    sample_original.jpg   - the clean picture
    sample_old_photo.jpg  - the same picture with fading, yellow tint,
                            noise, scratches and dust
"""

import sys
import cv2
import numpy as np

import restoration

np.random.seed(7)   # same result every time


def draw_simple_picture():
    """Draws a small house, a tree and a sun using OpenCV shapes."""
    height, width = 450, 600
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky: blue at the top, light at the bottom
    for y in range(height):
        shade = y / height
        img[y, :] = (255 - 40 * shade, 200 + 30 * shade, 120 + 60 * shade)

    cv2.circle(img, (500, 80), 40, (80, 230, 255), -1)             # sun
    cv2.ellipse(img, (150, 330), (260, 90), 0, 180, 360, (70, 150, 60), -1)   # hill
    cv2.rectangle(img, (0, 330), (width, height), (60, 140, 50), -1)          # grass

    cv2.rectangle(img, (250, 200), (430, 340), (90, 120, 200), -1)            # house
    pts = np.array([[235, 200], [340, 120], [445, 200]], np.int32)
    cv2.fillPoly(img, [pts], (40, 50, 140))                                   # roof
    cv2.rectangle(img, (320, 260), (360, 340), (30, 60, 90), -1)              # door
    cv2.rectangle(img, (270, 230), (305, 265), (250, 230, 180), -1)           # window
    cv2.rectangle(img, (385, 230), (420, 265), (250, 230, 180), -1)           # window

    cv2.rectangle(img, (95, 250), (115, 345), (30, 70, 110), -1)              # tree trunk
    cv2.circle(img, (105, 225), 50, (40, 120, 40), -1)                        # leaves
    cv2.circle(img, (75, 255), 35, (50, 140, 50), -1)
    cv2.circle(img, (135, 255), 35, (50, 140, 50), -1)

    cv2.line(img, (300, 345), (260, 450), (150, 190, 210), 25)                # path
    return img


def make_old(clean):
    """Adds the usual problems of an old photo."""
    img = clean.astype(np.float32)

    # 1. Fading: lower contrast and lift the dark parts
    img = img * 0.55 + 80

    # 2. Yellow tint: reduce blue, keep red (B, G, R order)
    img = img * np.array([0.70, 0.88, 1.0], dtype=np.float32)

    # 3. Noise (film grain)
    img = img + np.random.normal(0, 9, img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)
    height, width = img.shape[:2]

    # 4. Scratches: long thin lines, mostly bright, some dark
    for _ in range(14):
        x = np.random.randint(0, width)
        slant = np.random.randint(-25, 25)
        colour = 235 if np.random.rand() < 0.75 else 25
        cv2.line(img, (x, 0), (x + slant, height), (colour, colour, colour), 1)

    # 5. Dust and small spots
    for _ in range(70):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        colour = 240 if np.random.rand() < 0.7 else 20
        cv2.circle(img, (x, y), np.random.randint(1, 3), (colour, colour, colour), -1)

    return img


if __name__ == "__main__":
    if len(sys.argv) > 1:
        clean = restoration.load_image(sys.argv[1])
        if clean is None:
            print("Could not open the image:", sys.argv[1])
            sys.exit(1)
    else:
        clean = draw_simple_picture()

    old = make_old(clean)
    restoration.save_image("sample_original.jpg", clean)
    restoration.save_image("sample_old_photo.jpg", old)
    print("Created sample_original.jpg and sample_old_photo.jpg")
