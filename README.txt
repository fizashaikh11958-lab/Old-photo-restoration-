OLD PHOTO RESTORATION - WEBSITE (ALL IN ONE FILE)
=================================================

app.py contains everything: the image processing, the web server (Flask)
and the page design (HTML + CSS in the classic Windows 95/98 style).

FILES
-----
  app.py                 - the whole project
  requirements.txt       - libraries to install
  sample_old_photo.jpg   - damaged test photo to try the website
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
        Start Command      : gunicorn app:app --timeout 120
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


3. THE RESTORATION STEPS
------------------------
  1. Scratches and dust : morphological top-hat / black-hat find thin marks,
                          cv2.inpaint fills them from the surrounding pixels
  2. Noise (grain)      : non-local means denoising
  3. Yellow tint        : gray world colour balance
  4. Faded contrast     : histogram stretch + CLAHE on the L channel (LAB)
  5. Sharpen            : unsharp mask
  Final look            : keep colours, sepia or black and white

Results on the included test photo (damaged on purpose):
  Noise level (lower is cleaner)   6.83 -> 0.80
  Contrast                         27.4 -> 65.2


4. NOTES
--------
  - Photos larger than 900 pixels are made smaller to keep it fast.
  - Uploads are limited to 8 MB and are not saved on the server.
  - Very wide scratches or torn areas are not repaired.
