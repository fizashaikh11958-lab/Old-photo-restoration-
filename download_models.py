"""
download_models.py
------------------
Downloads the three files of the AI colour model (about 125 MB) into a
folder called  models  next to this script.

    python download_models.py

The files are the "Colorful Image Colorization" model of Zhang et al.
If a download fails, the script prints where to get the file by hand.
"""

import os
import urllib.request

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Each file has a few possible addresses; the first one that works is used
FILES = {
    "colorization_deploy_v2.prototxt": [
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt",
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    ],
    "pts_in_hull.npy": [
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/pts_in_hull.npy",
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    ],
    "colorization_release_v2.caffemodel": [
        "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
        "https://people.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
    ],
}


def download(name, urls):
    target = os.path.join(FOLDER, name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print("already there:", name)
        return True

    for url in urls:
        try:
            print("downloading", name, "...")
            urllib.request.urlretrieve(url, target)
            if os.path.getsize(target) > 0:
                print("  done")
                return True
        except Exception as error:
            print("  failed:", error)
            if os.path.exists(target):
                os.remove(target)
    return False


if __name__ == "__main__":
    os.makedirs(FOLDER, exist_ok=True)
    failed = [name for name, urls in FILES.items() if not download(name, urls)]

    print()
    if not failed:
        print("All model files are ready in:", FOLDER)
    else:
        print("These files could not be downloaded:", ", ".join(failed))
        print("Download them by hand and put them in:", FOLDER)
        print("Project page: https://github.com/richzhang/colorization (branch: caffe)")
