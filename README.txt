OLD PHOTO RESTORATION - WEBSITE (ALL IN ONE FILE)
=================================================

app.py contains everything: the image processing, the web server (Flask)
and the page design (HTML + CSS in the classic Windows 95/98 style).

FILES
-----
  app.py                 - the whole project
  requirements.txt       - libraries to install
  sample_old_photo.jpg   - damaged test photo to try the website
  download_models.py     - downloads the AI colour model (optional)
  example_full_restoration.jpg - full restoration of a cracked photo
  README.txt             - this file


1. RUN IT ON YOUR OWN COMPUTER
------------------------------
  pip install -r requirements.txt
  python app.py

Open  http://127.0.0.1:5000  in the browser, choose sample_old_photo.jpg,
press Restore. After the first Restore you can change the sliders and press
Restore again without choosing the photo again.


2. HOSTING IT ON THE INTERNET (FREE OPTIONS)
--------------------------------------------
A. RENDER  (recommended, easiest)
   1. Create a free account on github.com.
      Create a new repository and upload app.py and requirements.txt
      (Add file > Upload files).
   2. Create a free account on render.com and log in with GitHub.
   3. Click New > Web Service and select your repository.
   4. Fill in:
        Language / Runtime : Python 3
        Build Command      : pip install -r requirements.txt
        Start Command      : gunicorn app:app --workers 1 --timeout 300
        Instance Type      : Free
   5. Click Create Web Service. After a few minutes you get a link like
      https://your-name.onrender.com
   Good to know:
      - A free service goes to sleep after 15 minutes without visitors.
        The first visit after that takes about a minute to wake it up.
      - Nothing is stored on the server, so restarts do not matter.

B. PYTHONANYWHERE  (also free, no GitHub needed)
   1. Create a free "Beginner" account on pythonanywhere.com.
   2. Files tab: create a folder  photoapp  and upload app.py and
      requirements.txt into it.
   3. Consoles tab > Bash console:
        mkvirtualenv --python=python3.10 photoenv
        pip install -r /home/YOURNAME/photoapp/requirements.txt
   4. Web tab > Add a new web app > Manual configuration > Python 3.10.
   5. On the Web tab set the "Virtualenv" path to
        /home/YOURNAME/.virtualenvs/photoenv
   6. Open the WSGI configuration file and replace its content with:
        import sys
        sys.path.insert(0, "/home/YOURNAME/photoapp")
        from app import app as application
   7. Click the green Reload button.
      Your site is  https://YOURNAME.pythonanywhere.com
   Good to know:
      - The free account has little disk space (512 MB) and the free web
        app has to be extended every month using the button on the Web tab.

C. SHOWING IT WITHOUT HOSTING
   For a viva or demo you can simply run  python app.py  on your laptop.


D. COLOURS ON A HOSTED SITE (optional)
   The AI colour model file is about 125 MB, which is too big for a normal
   GitHub upload. Instead let the server download it while it is built.
   On Render use this Build Command:
        pip install -r requirements.txt && python download_models.py
   Without the model the website works normally; the "Add colours" box just
   says that the model is not installed.
   The model needs a few hundred MB of memory, so it may not fit on the
   smallest free plans. On your own computer it works fine.

3. THE RESTORATION STEPS
------------------------
  1. Scratches and dust : morphological top-hat / black-hat find thin marks,
                          cv2.inpaint fills them from the surrounding pixels
  2. Noise (grain)      : non-local means denoising
  3. Yellow tint        : gray world colour balance
  4. Faded contrast     : histogram stretch + CLAHE on the L channel (LAB)
  5. Sharpen            : unsharp mask
  Final look            : keep colours, sepia or black and white

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

Results on the included test photo (damaged on purpose):
  Noise level (lower is cleaner)   6.83 -> 0.80
  Contrast                         27.4 -> 65.2


4. NOTES
--------
  - Photos larger than 900 pixels are made smaller to keep it fast.
  - Uploads are limited to 8 MB and are not saved on the server.
  - Very wide scratches or torn areas are not repaired.
  - Full Restoration works on photos up to 800 pixels and takes about 10
    seconds on a normal computer, longer on a free hosting plan.
  - AI colours are guesses and can be wrong.
