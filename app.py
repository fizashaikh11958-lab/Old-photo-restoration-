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
    gunicorn app:app --timeout 120
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
    opts = {"look": "color"}
    for key, label, low, high, start in STEPS:
        opts[key] = True
        opts[key + "_level"] = start
    return opts


def read_options():
    """Reads the check boxes and sliders from the submitted form."""
    opts = {"look": request.form.get("look", "color")}
    if opts["look"] not in ("color", "sepia", "bw"):
        opts["look"] = "color"

    for key, label, low, high, start in STEPS:
        opts[key] = key in request.form
        try:
            value = int(request.form.get(key + "_level", start))
        except ValueError:
            value = start
        opts[key + "_level"] = max(low, min(high, value))
    return opts


def empty_page(error=None, status="Ready. Choose a photo and press Restore.", code=200):
    page = render_template_string(PAGE, steps=STEPS, opts=default_options(), error=error,
                                  status=status, size_text="No photo", before=None,
                                  after=None, previous="", quality=None, filename="")
    return page, code


@app.errorhandler(413)
def too_big(error):
    return empty_page("The file is too large. The limit is 8 MB.", "Error", 413)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return empty_page()

    opts = read_options()

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
        page = render_template_string(PAGE, steps=STEPS, opts=opts,
                                      error="Please choose a photo first.",
                                      status="Error", size_text="No photo", before=None,
                                      after=None, previous="", quality=None, filename="")
        return page

    original, error = decode_image(data)
    if error:
        page = render_template_string(PAGE, steps=STEPS, opts=opts, error=error,
                                      status="Error", size_text="No photo", before=None,
                                      after=None, previous="", quality=None, filename="")
        return page

    restored = restore_photo(original, opts)

    before_b64 = image_to_base64(original)
    after_b64 = image_to_base64(restored)
    height, width = original.shape[:2]
    quality = {
        "contrast_before": contrast_of(original), "contrast_after": contrast_of(restored),
        "noise_before": noise_of(original), "noise_after": noise_of(restored),
    }

    return render_template_string(
        PAGE, steps=STEPS, opts=opts, error=None,
        status="Restoration complete. You can save the photo now.",
        size_text="%d x %d pixels" % (width, height),
        before=before_b64, after=after_b64, previous=before_b64,
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

        <div class="buttons">
          <button type="submit" class="btn" id="restoreBtn">Restore</button>
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
            <b>Contrast</b> {{ "%.1f"|format(quality.contrast_before) }} &rarr; {{ "%.1f"|format(quality.contrast_after) }}<br>
            <b>Noise</b> {{ "%.2f"|format(quality.noise_before) }} &rarr; {{ "%.2f"|format(quality.noise_after) }} (lower is cleaner)
          </p>
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
  document.getElementById('mainForm').addEventListener('submit', function () {
    document.body.style.cursor = 'wait';
    document.getElementById('restoreBtn').textContent = 'Restoring...';
  });
</script>

</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
