OLD PHOTO RESTORATION USING IMAGE PROCESSING IN PYTHON
======================================================


ABOUT THE PROJECT
-----------------
Old photos are often faded, yellow, grainy and full of scratches and dust.
This project repairs these problems automatically using image processing.
It has a simple window in the style of old Windows software where you open
a photo, choose the restoration steps, press Restore and compare the
Before and After pictures.

Technologies used:
  Python    - programming language
  OpenCV    - image processing
  NumPy     - working with image data as arrays
  Pillow    - showing pictures inside the window
  Tkinter   - the window (comes with Python)


FILES
-----
  main.py                - the program with the window (run this one)
  restoration.py         - all the image processing functions
  make_damaged_sample.py - creates a fake damaged photo for testing
  run_demo.py            - restores a photo without the window and prints numbers
  download_models.py     - downloads the AI colour model (optional)
  requirements.txt       - the libraries to install
  sample_old_photo.jpg   - ready made damaged test photo
  sample_original.jpg    - the clean version of the test photo
  example_result.jpg     - example of Before (left) and After (right)
  example_full_restoration.jpg - full restoration of a cracked photo


HOW TO RUN
----------
1. Install Python 3.8 or newer from python.org
   (tick "Add Python to PATH" while installing).

2. Open a command prompt in this folder and install the libraries:
       pip install -r requirements.txt

3. Start the program:
       python main.py

4. In the window:
       File > Open Photo      choose an old photo
       (or use sample_old_photo.jpg that is included)
       Tick the steps you want and move the sliders
       Press Restore          the After picture appears
       Press Save Photo       save the result

On Linux, Tkinter may need:  sudo apt install python3-tk

To run without the window (prints the results and saves the steps):
       python run_demo.py
       python run_demo.py my_old_photo.jpg

To make a new damaged test photo from any clean picture:
       python make_damaged_sample.py clean_picture.jpg


THE RESTORATION STEPS
---------------------
  1. Scratches and dust
     Thin bright and dark marks are found with morphological top-hat and
     black-hat. They become a mask, and inpainting (cv2.inpaint) fills
     the marks using the pixels around them.

  2. Noise (grain)
     Removed with non-local means denoising (cv2.fastNlMeansDenoisingColored).

  3. Yellow tint
     Gray world method. The average of each colour channel is made equal,
     then mixed with the original so that colours stay natural.

  4. Faded contrast
     The brightness is stretched to the full range, then CLAHE
     (adaptive histogram equalization) adds local contrast.
     This is done on the L channel of the LAB colour space.

  5. Sharpen
     Unsharp mask: a blurred copy is subtracted from the photo.

  Final look: keep colours, sepia (brown) or black and white.


FULL RESTORATION
----------------
The normal Restore button applies only the steps you tick. The
"Full Restoration" button does everything in one go, for badly damaged
photos with cracks, creases, torn spots and fading.
(example_full_restoration.jpg shows a cracked photo: before on the left,
after on the right)

  1. Cracks, creases and dust (done in two passes)
     - The photo is searched in 12 directions with a thin line detector.
       A crack is a long thin line that is brighter than BOTH of its sides.
       The edge of a dress is brighter on one side only, so it is ignored.
     - Weak pieces are kept only when they touch a strong piece (this joins
       broken cracks), and only long pieces are kept, because cracks are
       long and texture is short.
     - Faces are found with a Haar cascade and protected, so eyes and hair
       are not removed by mistake.
     - Tiny bright dots (dust) are found with top-hat.
     - Everything found is filled by inpainting (cv2.inpaint).
     - The second pass finds thinner cracks that were hidden by the first.
  2. Grain is reduced (non-local means)
  3. Brightness is corrected with gamma (dark photos become lighter)
  4. The colour cast of old paper is removed (gray world)
  5. Faded contrast is restored (histogram stretch + CLAHE)
  6. Gentle sharpening
  7. Colours (optional): "Add colours (AI model)"

  "Damage search level" (1-10): higher finds more damage, but may also
  remove real details. 5 is a good start.

ADDING COLOURS WITH THE AI MODEL (optional)
-------------------------------------------
A black and white or brown photo has no colour information, so the colours
must be guessed. This is done by a trained neural network (Zhang et al.,
"Colorful Image Colorization") that runs inside OpenCV. The colours look
natural but they are only the model's guess, not the real colours.

Setup (one time, needs internet, about 125 MB):
       python download_models.py
This creates a folder called  models  with three files:
       colorization_deploy_v2.prototxt
       colorization_release_v2.caffemodel
       pts_in_hull.npy
If a download fails the script tells you what to download by hand. The files
come from https://github.com/richzhang/colorization (branch "caffe").

Then tick "Add colours (AI model)" and press Full Restoration.
Without the model files the photo is still restored, only without colours.

RESULTS ON THE INCLUDED TEST PHOTO
----------------------------------
The test photo is a clean picture that was damaged on purpose (fading,
yellow tint, noise, 14 scratches, 70 dust spots). The restored photo
was compared with the clean original:

  Noise level   (lower is cleaner)        6.83  ->  0.88
  Contrast      (standard deviation)      27.4  ->  65.3
  PSNR vs original (higher is better)    13.96  ->  16.34 dB

Numbers are printed by run_demo.py. For a real old photo there is no clean
original, so only noise level and contrast can be compared.


LIMITATIONS
-----------
  - Full restoration cannot rebuild parts that are completely missing or
    torn away; it only fills thin damage from the pixels around it.
  - Faint cracks may stay, and strong settings may soften fine details.
  - AI colours are guesses and can be wrong (for example clothes or flowers).
  - Full restoration works on photos up to 800 pixels and can take up to
    a minute.
  - Photos larger than 1000 pixels are made smaller to keep it fast.
  - Very wide scratches or large torn areas are not repaired.
  - Fine details such as hair can sometimes be removed along with dust.
  - Strong settings can make flat areas look blotchy.


FUTURE IMPROVEMENTS
-------------------
  - Let the user paint a mask over damaged areas by hand
  - Deep learning inpainting for large missing areas
  - Face enhancement
  - Batch mode to restore a whole folder of photos
