"""
restoration.py
--------------
All the image processing steps used to restore an old photo.
Each step is a small function that takes an image and returns a new image.

Images are handled as NumPy arrays in BGR colour order (the OpenCV way).
"""

import cv2
import numpy as np

# Big photos are made smaller so that the processing stays fast
MAX_SIZE = 1000


# ------------------------------------------------------------------
# Loading and saving
# ------------------------------------------------------------------
def load_image(path):
    """Read a photo from disk. Returns None if the file is not an image."""
    # np.fromfile + imdecode also works when the path has special characters
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Shrink the photo if it is very large
    height, width = img.shape[:2]
    biggest = max(height, width)
    if biggest > MAX_SIZE:
        scale = MAX_SIZE / biggest
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def save_image(path, img):
    """Write the photo to disk. The file type is taken from the extension."""
    extension = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ".jpg"
    ok, encoded = cv2.imencode(extension, img)
    if ok:
        encoded.tofile(path)
    return ok


# ------------------------------------------------------------------
# Step 1: remove scratches and dust
# ------------------------------------------------------------------
def remove_scratches(img, sensitivity=5):
    """
    Scratches and dust spots are thin bright or dark marks.
    1. Top-hat   finds thin BRIGHT marks (white scratches, dust)
    2. Black-hat finds thin DARK marks
    3. The marks become a mask, and inpainting fills them using the
       pixels around them.
    sensitivity: 1 (find only strong marks) to 10 (find even faint marks)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Marks thinner than this kernel will be detected
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Higher sensitivity means a lower threshold
    threshold = 70 - sensitivity * 6
    mask = ((bright > threshold) | (dark > threshold)).astype(np.uint8) * 255

    # Make the mask a little thicker so the whole scratch is covered
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


# ------------------------------------------------------------------
# Step 2: remove noise (grain)
# ------------------------------------------------------------------
def remove_noise(img, strength=7):
    """Non-local means denoising. strength: 1 (light) to 15 (strong)."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


# ------------------------------------------------------------------
# Step 3: remove the yellow / brown tint of old paper
# ------------------------------------------------------------------
def remove_tint(img, strength=0.6):
    """
    Gray world method: in a normal photo the average colour should be gray.
    So each colour channel is scaled until all three averages are equal.
    strength: 0 = no change, 1 = full correction (0.6 keeps the colours natural)
    """
    b, g, r = cv2.split(img.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    average = (mean_b + mean_g + mean_r) / 3

    b = b * (average / max(mean_b, 1))
    g = g * (average / max(mean_g, 1))
    r = r * (average / max(mean_r, 1))

    corrected = cv2.merge([b, g, r])
    # Mix the corrected photo with the original one
    result = corrected * strength + img.astype(np.float32) * (1 - strength)
    return np.clip(result, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Step 4: bring back faded contrast
# ------------------------------------------------------------------
def improve_contrast(img, strength=3):
    """
    Old photos look faded (flat and gray).
    1. Stretch the brightness so it uses the full range 0 to 255
    2. CLAHE adds local contrast to bring out the details
    The work is done only on the brightness (L) channel so colours stay natural.
    strength: 1 (light) to 6 (strong)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Stretch: ignore the darkest 1% and the brightest 1%
    low, high = np.percentile(l, (1, 99))
    if high > low:
        l = (l.astype(np.float32) - low) * 255.0 / (high - low)
        l = np.clip(l, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=strength * 0.6, tileGridSize=(8, 8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------------
# Step 5: sharpen
# ------------------------------------------------------------------
def sharpen(img, amount=0.8):
    """Unsharp mask: subtract a blurred copy to make the edges stronger."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


# ------------------------------------------------------------------
# Output looks
# ------------------------------------------------------------------
def to_sepia(img):
    """Classic brown 'old photo' look."""
    # Rows are the new B, G, R values; columns are the old B, G, R values
    kernel = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]], dtype=np.float32)
    result = cv2.transform(img.astype(np.float32), kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def to_black_white(img):
    """Black and white photo (still stored with 3 channels)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------
# Run all the selected steps one after another
# ------------------------------------------------------------------
def restore_photo(img, options):
    """
    options is a dictionary, for example:
    {
        "scratches": True, "scratch_level": 5,
        "noise": True,     "noise_level": 7,
        "tint": True,      "tint_level": 6,
        "contrast": True,  "contrast_level": 3,
        "sharpen": True,   "sharpen_level": 5,
        "look": "color"    # or "sepia" or "bw"
    }
    """
    result = img.copy()

    if options.get("scratches"):
        result = remove_scratches(result, options.get("scratch_level", 5))

    if options.get("noise"):
        result = remove_noise(result, options.get("noise_level", 7))

    if options.get("tint"):
        # slider value 0..10 becomes a strength from 0.0 to 1.0
        result = remove_tint(result, options.get("tint_level", 6) / 10.0)

    if options.get("contrast"):
        result = improve_contrast(result, options.get("contrast_level", 3))

    if options.get("sharpen"):
        # slider value 1..10 becomes an amount from 0.15 to 1.5
        result = sharpen(result, options.get("sharpen_level", 5) * 0.15)

    look = options.get("look", "color")
    if look == "sepia":
        result = to_sepia(result)
    elif look == "bw":
        result = to_black_white(result)

    return result


# ---------------------------------------------------------------------
# FULL RESTORATION
# Repairs cracks, creases and dust, fixes brightness, tint, noise and
# contrast, and can add colour with an AI model.
# ---------------------------------------------------------------------
FULL_MAX_SIZE = 800   # the full restoration works on photos up to this size


def _line_kernels(theta, length, width, gap, band):
    """Three small kernels for one direction: the line, its left side and its right side."""
    c = length // 2 + band + gap + 4
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    along = xs * np.cos(theta) + ys * np.sin(theta)
    across = -xs * np.sin(theta) + ys * np.cos(theta)

    inside = np.abs(along) <= length / 2
    centre = inside & (np.abs(across) <= width / 2)
    d = width / 2 + gap
    left = inside & (across < -d) & (across >= -d - band)
    right = inside & (across > d) & (across <= d + band)

    kernels = []
    for part in (centre, left, right):
        k = np.zeros(part.shape, np.float32)
        k[part] = 1.0 / part.sum()
        kernels.append(k)
    return kernels


def crack_strength(gray, directions=12):
    """
    A crack is a thin bright line that is brighter than BOTH of its sides.
    (The edge of a dress is brighter on one side only, so it is ignored.)
    The photo is checked in many directions and the strongest answer is kept.
    """
    g = gray.astype(np.float32)
    best = np.zeros(g.shape, np.float32)
    for i in range(directions):
        k_centre, k_left, k_right = _line_kernels(np.pi * i / directions, 15, 3, 1, 3)
        centre = cv2.filter2D(g, cv2.CV_32F, k_centre)
        left = cv2.filter2D(g, cv2.CV_32F, k_left)
        right = cv2.filter2D(g, cv2.CV_32F, k_right)
        best = np.maximum(best, np.minimum(centre - left, centre - right))
    return best


def find_faces(gray):
    """Finds faces so they can be protected. Returns a list of (x, y, w, h)."""
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(path)
        small = min(gray.shape[:2])
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(max(30, small // 12),) * 2)
        return [tuple(int(v) for v in f) for f in faces]
    except Exception:
        return []


def find_damage(img, level=5):
    """
    Makes a mask of the damage (white = damaged).
    1. long thin bright lines  = cracks and creases
    2. small bright dots       = dust
    level: 1 (only strong damage) to 10 (also faint damage)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.7)

    strength = crack_strength(smooth)

    # Faces need extra care, so cracks must be stronger there
    for (x, y, w, h) in find_faces(gray):
        pad_x, pad_y = int(w * 0.35), int(h * 0.45)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)
        strength[y0:y1, x0:x1] *= 0.65

    # Ignore the very edge of the picture (the paper border)
    strength[:5, :] = 0
    strength[-5:, :] = 0
    strength[:, :5] = 0
    strength[:, -5:] = 0

    # Higher level = lower thresholds
    high = 48 - level * 2.5
    low = 28 - level * 1.6
    min_length = 25

    strong = (strength > high).astype(np.uint8)
    weak = (strength > low).astype(np.uint8)

    # Keep weak pieces only when they touch a strong piece (this joins broken cracks)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    wanted = np.unique(labels[strong > 0])
    wanted = wanted[wanted > 0]
    joined = np.isin(labels, wanted).astype(np.uint8)

    # Keep only long pieces: cracks are long, texture is short
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    mask = np.zeros(gray.shape, np.uint8)
    for i in range(1, count):
        width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(width, height) >= min_length:
            mask[labels == i] = 255

    # Dust: tiny, very bright dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dots = (cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel) > 60).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dots, connectivity=8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] <= 30:
            mask[labels == i] = 255

    # Make the mask a little thicker so the whole crack is covered
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def repair_damage(img, level=5):
    """Fills cracks and dust by inpainting. Returns (repaired photo, mask)."""
    mask = find_damage(img, level)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA), mask


def auto_brightness(img, target=110):
    """Brightens a photo that is too dark (or darkens one that is too bright) using gamma."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = max(1.0, min(254.0, float(gray.mean())))
    gamma = np.log(target / 255.0) / np.log(mean / 255.0)
    gamma = float(np.clip(gamma, 0.6, 1.3))
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


MODEL_FOLDER = "models"
MODEL_FILES = ("colorization_deploy_v2.prototxt",
               "colorization_release_v2.caffemodel",
               "pts_in_hull.npy")


def colour_model_ready(folder=None):
    """True when the three AI model files are inside the models folder."""
    import os
    folder = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FOLDER)
    return all(os.path.exists(os.path.join(folder, name)) for name in MODEL_FILES)


def colorize_ai(img, folder=None):
    """
    Adds colour to a black and white / sepia photo with a trained neural network
    (Zhang et al., "Colorful Image Colorization"), run through OpenCV.
    The model only guesses the colours, so they look natural but not exact.
    """
    import os
    folder = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FOLDER)
    net = cv2.dnn.readNetFromCaffe(os.path.join(folder, MODEL_FILES[0]),
                                   os.path.join(folder, MODEL_FILES[1]))
    points = np.load(os.path.join(folder, MODEL_FILES[2]))

    # Put the colour cluster centres into the network
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

    # The network looks only at the brightness (L channel) and predicts the colours (a, b)
    scaled = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    l_channel = cv2.split(cv2.resize(lab, (224, 224)))[0] - 50
    net.setInput(cv2.dnn.blobFromImage(l_channel))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    # Join the original brightness with the predicted colours
    l_full = cv2.split(lab)[0]
    result = np.concatenate((l_full[:, :, np.newaxis], ab), axis=2)
    result = np.clip(cv2.cvtColor(result, cv2.COLOR_LAB2BGR), 0, 1)
    return (result * 255).astype(np.uint8)


def full_restore(img, use_colour=False, look="color", level=5):
    """
    Full restoration in one go:
      1. repair cracks, creases and dust
      2. light noise removal
      3. fix the brightness
      4. remove the colour cast of old paper
      5. bring back contrast
      6. gentle sharpening
      7. optional AI colour (or sepia / black and white)
    Returns (restored photo, notes) where notes is a list of text lines.
    """
    notes = []

    # Work on a smaller copy if the photo is big (the crack search is slow)
    height, width = img.shape[:2]
    biggest = max(height, width)
    if biggest > FULL_MAX_SIZE:
        scale = FULL_MAX_SIZE / biggest
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    # Two passes: the first removes the big cracks, the second finds the thinner
    # cracks that were hidden before
    result, mask = repair_damage(img, level)
    first = 100.0 * np.count_nonzero(mask) / mask.size
    result, mask = repair_damage(result, min(10, level + 2))
    second = 100.0 * np.count_nonzero(mask) / mask.size
    notes.append("Cracks and dust repaired (pass 1: %.1f%%, pass 2: %.1f%% of the photo)" % (first, second))

    result = remove_noise(result, 6)
    notes.append("Grain reduced")

    result = auto_brightness(result)
    notes.append("Brightness corrected")

    result = remove_tint(result, 0.85)
    notes.append("Old paper colour cast removed")

    result = improve_contrast(result, 1.5)
    notes.append("Faded contrast restored")

    result = sharpen(result, 0.3)
    notes.append("Details sharpened")

    if use_colour:
        if colour_model_ready():
            try:
                result = colorize_ai(to_black_white(result))
                notes.append("Colours added with the AI model")
            except Exception as error:
                notes.append("Colour step failed: %s" % error)
        else:
            notes.append("Colours not added: the AI model files are missing (see README)")
    elif look == "sepia":
        result = to_sepia(to_black_white(result))
        notes.append("Sepia look applied")
    elif look == "bw":
        result = to_black_white(result)
        notes.append("Black and white look applied")

    return result, notes


# ------------------------------------------------------------------
# Simple numbers to measure the result
# ------------------------------------------------------------------
def noise_level(img):
    """
    How much the pixels differ from their blurred neighbours.
    A lower number means a cleaner photo (less grain, scratches and dust).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.medianBlur(gray, 3)
    return float(np.mean(cv2.absdiff(gray, smooth)))


def contrast(img):
    """Standard deviation of brightness. A higher number means more contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def psnr(original, other):
    """
    Peak Signal to Noise Ratio in dB (higher is better).
    Only possible when the clean original photo is known.
    """
    return float(cv2.PSNR(original, other))
