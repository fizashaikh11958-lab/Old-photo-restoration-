"""
app.py  -  Old Photo Restoration (all in one file)
==================================================
A small website that restores old photos using image processing.
Everything is in this one file:
    1. the image processing functions (OpenCV + NumPy)
    2. the web server (Flask)
    3. the page design (HTML + CSS in the classic Windows 95/98 style)

Run on your own computer:
    pip install -r requirements.txt
    python app.py
    then open  http://127.0.0.1:5000  in the browser

Run on a hosting service:
    gunicorn app:app --workers 1 --timeout 300

Optional (adds colours to black and white photos):
    python download_models.py
"""

import base64
import io

import cv2
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024   # uploads up to 8 MB

MAX_SIZE = 900            # big photos are made smaller to keep it fast
MAX_PIXELS = 40_000_000   # refuse absurdly large pictures


# ======================================================================
# PART 1 - IMAGE PROCESSING
# Images are NumPy arrays in BGR colour order (the OpenCV way).
# ======================================================================
def decode_image(data):
    """Turns uploaded bytes into an image. Returns (image, error_message)."""
    # Look at the size first, without loading the whole picture
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
    except Exception:
        return None, "This file is not a picture. Please choose a JPG, PNG, BMP or TIFF file."

    if width * height > MAX_PIXELS:
        return None, "This picture is too large. Please choose a smaller one."

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "This picture could not be read."

    # Shrink the photo if it is very large
    biggest = max(height, width)
    if biggest > MAX_SIZE:
        scale = MAX_SIZE / biggest
        img = cv2.resize(img, (int(width * scale), int(height * scale)),
                         interpolation=cv2.INTER_AREA)
    return img, None


def image_to_base64(img):
    """JPEG bytes of the image as base64 text (used inside the web page)."""
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def remove_scratches(img, sensitivity=5):
    """
    Scratches and dust are thin bright or dark marks.
    Top-hat finds thin bright marks, black-hat finds thin dark marks.
    The marks become a mask and inpainting fills them from the pixels around.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    threshold = 70 - sensitivity * 6          # higher sensitivity = lower threshold
    mask = ((bright > threshold) | (dark > threshold)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def remove_noise(img, strength=6):
    """Non-local means denoising removes the film grain."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def remove_tint(img, strength=0.6):
    """
    Gray world method: the average colour of a normal photo is gray,
    so the three colour channels are scaled until their averages are equal.
    The result is mixed with the original to keep the colours natural.
    """
    b, g, r = cv2.split(img.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    average = (mean_b + mean_g + mean_r) / 3

    b = b * (average / max(mean_b, 1))
    g = g * (average / max(mean_g, 1))
    r = r * (average / max(mean_r, 1))

    corrected = cv2.merge([b, g, r])
    result = corrected * strength + img.astype(np.float32) * (1 - strength)
    return np.clip(result, 0, 255).astype(np.uint8)


def improve_contrast(img, strength=2):
    """
    Faded photos are flat. Stretch the brightness to the full range and then
    use CLAHE to add local contrast. Done on the L channel of LAB colours.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    low, high = np.percentile(l, (1, 99))
    if high > low:
        l = (l.astype(np.float32) - low) * 255.0 / (high - low)
        l = np.clip(l, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=strength * 0.6, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def sharpen(img, amount=0.6):
    """Unsharp mask: subtract a blurred copy to make edges stronger."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def to_sepia(img):
    """Classic brown 'old photo' look."""
    kernel = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]], dtype=np.float32)
    result = cv2.transform(img.astype(np.float32), kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def to_black_white(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def restore_photo(img, opts):
    """Runs the selected steps one after another."""
    result = img.copy()

    if opts["scratches"]:
        result = remove_scratches(result, opts["scratches_level"])
    if opts["noise"]:
        result = remove_noise(result, opts["noise_level"])
    if opts["tint"]:
        result = remove_tint(result, opts["tint_level"] / 10.0)
    if opts["contrast"]:
        result = improve_contrast(result, opts["contrast_level"])
    if opts["sharpen"]:
        result = sharpen(result, opts["sharpen_level"] * 0.15)

    if opts["look"] == "sepia":
        result = to_sepia(result)
    elif opts["look"] == "bw":
        result = to_black_white(result)
    return result


def contrast_of(img):
    """Standard deviation of brightness. Higher means more contrast."""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())


def noise_of(img):
    """How much pixels differ from their neighbours. Lower means cleaner."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(gray, cv2.medianBlur(gray, 3))))



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


# ======================================================================
# PART 2 - WEB SERVER
# ======================================================================
# key, label, slider minimum, slider maximum, starting value
STEPS = [
    ("scratches", "Remove scratches and dust", 1, 10, 5),
    ("noise", "Remove noise (grain)", 1, 15, 6),
    ("tint", "Remove yellow tint", 1, 10, 6),
    ("contrast", "Improve faded contrast", 1, 6, 2),
    ("sharpen", "Sharpen the photo", 1, 10, 4),
]


def default_options():
    opts = {"look": "color", "damage_level": 5, "ai_colour": False}
    for key, label, low, high, start in STEPS:
        opts[key] = True
        opts[key + "_level"] = start
    return opts


def read_options():
    """Reads the check boxes and sliders from the submitted form."""
    opts = {"look": request.form.get("look", "color")}
    if opts["look"] not in ("color", "sepia", "bw"):
        opts["look"] = "color"

    opts["ai_colour"] = "ai_colour" in request.form
    try:
        level = int(request.form.get("damage_level", 5))
    except ValueError:
        level = 5
    opts["damage_level"] = max(1, min(10, level))

    for key, label, low, high, start in STEPS:
        opts[key] = key in request.form
        try:
            value = int(request.form.get(key + "_level", start))
        except ValueError:
            value = start
        opts[key + "_level"] = max(low, min(high, value))
    return opts


def page(opts=None, **values):
    """Draws the page. Anything that is not given here is left empty."""
    info = dict(error=None, status="Ready. Choose a photo and press Restore.",
                size_text="No photo", before=None, after=None, previous="",
                quality=None, filename="")
    info.update(values)
    return render_template_string(PAGE, steps=STEPS, opts=opts or default_options(),
                                  colour_ready=colour_model_ready(), **info)


@app.errorhandler(413)
def too_big(error):
    return page(error="The file is too large. The limit is 8 MB.", status="Error"), 413


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return page()

    opts = read_options()
    mode = request.form.get("mode", "steps")      # "steps" or "full"

    # Get the photo: a new upload, or the one kept from the last time
    upload = request.files.get("photo")
    data = upload.read() if upload and upload.filename else b""
    filename = upload.filename if upload and upload.filename else ""

    if not data:
        kept = request.form.get("previous", "")
        try:
            data = base64.b64decode(kept, validate=True) if kept else b""
        except Exception:
            data = b""

    if not data:
        return page(opts, error="Please choose a photo first.", status="Error")

    original, error = decode_image(data)
    if error:
        return page(opts, error=error, status="Error")

    quality = {"contrast_before": contrast_of(original), "notes": None}

    if mode == "full":
        restored, notes = full_restore(original, use_colour=opts["ai_colour"],
                                       look=opts["look"], level=opts["damage_level"])
        quality["notes"] = notes
        status = "Full restoration complete. You can save the photo now."
    else:
        restored = restore_photo(original, opts)
        quality["noise_before"] = noise_of(original)
        quality["noise_after"] = noise_of(restored)
        status = "Restoration complete. You can save the photo now."

    quality["contrast_after"] = contrast_of(restored)

    before_b64 = image_to_base64(original)
    height, width = original.shape[:2]
    return page(opts, status=status, size_text="%d x %d pixels" % (width, height),
                before=before_b64, after=image_to_base64(restored), previous=before_b64,
                quality=quality, filename=filename)


# ======================================================================
# PART 3 - THE PAGE (HTML + CSS in the classic Windows style)
# ======================================================================
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Old Photo Restoration</title>
<style>
  * { box-sizing: border-box; }

  body {
    margin: 0;
    padding: 18px 12px 30px;
    background-color: #008080;               /* the classic teal desktop */
    font-family: "MS Sans Serif", "Microsoft Sans Serif", Tahoma, Arial, sans-serif;
    font-size: 13px;
    color: #000;
  }

  /* ----- the window ----- */
  .window {
    max-width: 960px;
    margin: 0 auto;
    background-color: #c0c0c0;
    border: 2px solid;
    border-color: #ffffff #404040 #404040 #ffffff;
    box-shadow: inset -1px -1px #808080, inset 1px 1px #dfdfdf, 4px 4px 0 rgba(0, 0, 0, 0.35);
    padding: 3px;
  }

  .titlebar {
    background: linear-gradient(90deg, #000080, #1084d0);
    color: #ffffff;
    font-weight: bold;
    font-size: 14px;
    padding: 3px 4px 3px 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .winbtns b {
    display: inline-block;
    width: 18px;
    height: 16px;
    margin-left: 2px;
    background-color: #c0c0c0;
    color: #000;
    font-size: 11px;
    line-height: 14px;
    text-align: center;
    border: 1px solid;
    border-color: #ffffff #404040 #404040 #ffffff;
  }

  .content { padding: 10px 8px 6px; }

  /* ----- layout ----- */
  .layout { display: flex; gap: 14px; flex-wrap: wrap; }
  .left { flex: 0 0 290px; }
  .right { flex: 1 1 520px; min-width: 0; }
  .pictures { display: flex; gap: 10px; flex-wrap: wrap; }
  .pictures fieldset { flex: 1 1 240px; }

  /* ----- group boxes ----- */
  fieldset {
    margin: 0 0 10px;
    padding: 6px 10px 10px;
    border: 2px groove #ffffff;
    min-width: 0;
  }

  legend { padding: 0 4px; font-weight: bold; }
  .note { margin: 6px 0 0; color: #404040; font-size: 12px; }

  .step { margin-top: 6px; }
  .step label { font-weight: bold; }
  .slider { display: flex; align-items: center; gap: 8px; padding-left: 22px; }
  .slider input[type=range] { flex: 1; accent-color: #000080; }
  .slider output {
    width: 26px;
    text-align: center;
    background: #ffffff;
    border: 2px solid;
    border-color: #808080 #ffffff #ffffff #808080;
    font-size: 12px;
  }

  .looks label { display: block; margin-top: 3px; }

  input[type=file] { width: 100%; font-family: inherit; font-size: 12px; }

  /* ----- buttons ----- */
  .buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }

  .btn {
    display: inline-block;
    min-width: 96px;
    padding: 5px 14px;
    background-color: #c0c0c0;
    color: #000;
    font-family: inherit;
    font-size: 13px;
    text-align: center;
    text-decoration: none;
    cursor: pointer;
    border: 2px solid;
    border-color: #ffffff #404040 #404040 #ffffff;
    box-shadow: inset -1px -1px #808080;
  }

  .btn:active {
    border-color: #404040 #ffffff #ffffff #404040;
    box-shadow: inset 1px 1px #808080;
    padding: 6px 13px 4px 15px;
  }

  .btn:focus { outline: 1px dotted #000; outline-offset: -5px; }
  .save { margin-top: 2px; }

  /* ----- sunken picture boxes ----- */
  .sunken {
    background-color: #ffffff;
    border: 2px solid;
    border-color: #808080 #ffffff #ffffff #808080;
  }

  .picbox {
    height: 320px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #808080;
    padding: 4px;
  }

  .picbox img { max-width: 100%; max-height: 100%; display: block; }

  .quality { margin: 0; line-height: 1.7; }
  .quality b { display: inline-block; width: 70px; }
  .notes { margin: 6px 0 0; padding-left: 18px; line-height: 1.6; }

  /* ----- error box ----- */
  .dialog {
    margin: 0 0 12px;
    background-color: #c0c0c0;
    border: 2px solid;
    border-color: #ffffff #404040 #404040 #ffffff;
    box-shadow: 3px 3px 0 rgba(0, 0, 0, 0.3);
  }

  .dialog .bar { background-color: #000080; color: #ffffff; font-weight: bold; padding: 2px 8px; }
  .dialog .msg { padding: 12px 14px; display: flex; align-items: center; gap: 12px; }

  .dialog .icon {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background-color: #c00000;
    color: #ffffff;
    font-weight: bold;
    font-size: 20px;
    line-height: 30px;
    text-align: center;
    flex-shrink: 0;
  }

  /* ----- status bar ----- */
  .statusbar { display: flex; gap: 3px; padding: 3px 4px 2px; }
  .statusbar .sunken { background-color: #c0c0c0; padding: 2px 8px; }
  .statusbar span:first-child { flex: 1; }
  .statusbar span:last-child { width: 170px; text-align: center; }

  @media (max-width: 700px) {
    .left { flex-basis: 100%; }
    .picbox { height: 260px; }
  }
</style>
</head>
<body>

<div class="window">
  <div class="titlebar">
    <span>Old Photo Restoration{% if filename %} - {{ filename }}{% endif %}</span>
    <span class="winbtns"><b>_</b><b>&#9633;</b><b>X</b></span>
  </div>

  <form method="post" enctype="multipart/form-data" class="content" id="mainForm">
    <input type="hidden" name="previous" value="{{ previous }}">

    {% if error %}
    <div class="dialog">
      <div class="bar">Old Photo Restoration</div>
      <div class="msg"><span class="icon">X</span><span>{{ error }}</span></div>
    </div>
    {% endif %}

    <div class="layout">

      <!-- ===== left side: the options ===== -->
      <div class="left">
        <fieldset>
          <legend>Photo</legend>
          <input type="file" name="photo" accept="image/*">
          <p class="note">JPG, PNG, BMP or TIFF, up to 8 MB.</p>
        </fieldset>

        <fieldset>
          <legend>Restoration steps</legend>
          {% for key, label, low, high, start in steps %}
          <div class="step">
            <label><input type="checkbox" name="{{ key }}" {% if opts[key] %}checked{% endif %}>
              {{ loop.index }}. {{ label }}</label>
            <div class="slider">
              <input type="range" name="{{ key }}_level" min="{{ low }}" max="{{ high }}"
                     value="{{ opts[key + '_level'] }}">
              <output>{{ opts[key + '_level'] }}</output>
            </div>
          </div>
          {% endfor %}
        </fieldset>

        <fieldset class="looks">
          <legend>Final look</legend>
          <label><input type="radio" name="look" value="color" {% if opts.look == 'color' %}checked{% endif %}> Keep colours</label>
          <label><input type="radio" name="look" value="sepia" {% if opts.look == 'sepia' %}checked{% endif %}> Sepia (brown)</label>
          <label><input type="radio" name="look" value="bw" {% if opts.look == 'bw' %}checked{% endif %}> Black and white</label>
        </fieldset>

        <fieldset>
          <legend>Full restoration</legend>
          <p class="note" style="margin-top: 0;">Repairs cracks, creases and dust, and fixes brightness,
            tint and contrast. Use the button below.</p>
          <div class="step">
            <label>Damage search level</label>
            <div class="slider">
              <input type="range" name="damage_level" min="1" max="10" value="{{ opts.damage_level }}">
              <output>{{ opts.damage_level }}</output>
            </div>
          </div>
          <div class="step">
            <label><input type="checkbox" name="ai_colour" {% if opts.ai_colour %}checked{% endif %}>
              Add colours (AI model)</label>
            {% if not colour_ready %}
            <p class="note">The AI model files are not installed on this server, so colours cannot be added.</p>
            {% endif %}
          </div>
        </fieldset>

        <div class="buttons">
          <button type="submit" class="btn" name="mode" value="steps">Restore</button>
          <button type="submit" class="btn" name="mode" value="full">Full Restoration</button>
          <a class="btn" href="/">Reset</a>
        </div>
      </div>

      <!-- ===== right side: pictures and numbers ===== -->
      <div class="right">
        <div class="pictures">
          <fieldset>
            <legend>Before</legend>
            <div class="sunken picbox">
              {% if before %}<img alt="Before" src="data:image/jpeg;base64,{{ before }}">{% else %}No photo{% endif %}
            </div>
          </fieldset>
          <fieldset>
            <legend>After</legend>
            <div class="sunken picbox">
              {% if after %}<img alt="After" src="data:image/jpeg;base64,{{ after }}">{% else %}No photo{% endif %}
            </div>
          </fieldset>
        </div>

        <fieldset>
          <legend>Photo quality</legend>
          {% if quality %}
          <p class="quality">
            <b>Contrast</b> {{ "%.1f"|format(quality.contrast_before) }} &rarr; {{ "%.1f"|format(quality.contrast_after) }}
            {% if not quality.notes %}<br>
            <b>Noise</b> {{ "%.2f"|format(quality.noise_before) }} &rarr; {{ "%.2f"|format(quality.noise_after) }} (lower is cleaner)
            {% endif %}
          </p>
          {% if quality.notes %}
          <ul class="notes">
            {% for note in quality.notes %}<li>{{ note }}</li>{% endfor %}
          </ul>
          {% endif %}
          {% else %}
          <p class="quality">Press Restore to see the numbers.</p>
          {% endif %}
        </fieldset>

        {% if after %}
        <a class="btn save" href="data:image/jpeg;base64,{{ after }}" download="restored_photo.jpg">Save Restored Photo</a>
        {% endif %}
      </div>

    </div>
  </form>

  <div class="statusbar">
    <span class="sunken">{{ status }}</span>
    <span class="sunken">{{ size_text }}</span>
  </div>
</div>

<script>
  // show the slider number next to each slider
  var sliders = document.querySelectorAll('input[type=range]');
  for (var i = 0; i < sliders.length; i++) {
    sliders[i].addEventListener('input', function () {
      this.nextElementSibling.textContent = this.value;
    });
  }

  // hourglass while the photo is being restored
  document.getElementById('mainForm').addEventListener('submit', function (event) {
    document.body.style.cursor = 'wait';
    if (event.submitter) { event.submitter.textContent = 'Working...'; }
  });
</script>

</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
