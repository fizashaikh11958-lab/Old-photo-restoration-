"""
main.py
-------
Old Photo Restoration - the main program with a window (GUI).
The window is made with Tkinter and styled like old Windows software.

Run it with:   python main.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import restoration

# ---------- Classic colours and fonts ----------
GRAY = "#C0C0C0"       # the typical grey of old windows
NAVY = "#000080"       # title bar blue
WHITE = "#FFFFFF"
FONT = ("Microsoft Sans Serif", 9)
FONT_BOLD = ("Microsoft Sans Serif", 9, "bold")
FONT_TITLE = ("Microsoft Sans Serif", 11, "bold")

CANVAS_SIZE = 340      # size of the Before and After picture boxes


class PhotoRestorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Old Photo Restoration")
        self.root.configure(bg=GRAY)
        self.root.geometry("1060x740")
        self.root.resizable(False, False)

        self.original = None     # the photo that was opened
        self.restored = None     # the photo after restoration
        self.photos = {}         # keeps the pictures alive, otherwise Tkinter hides them

        # Variables connected to the check boxes and sliders
        self.use_scratches = tk.BooleanVar(value=True)
        self.scratch_level = tk.IntVar(value=5)
        self.use_noise = tk.BooleanVar(value=True)
        self.noise_level = tk.IntVar(value=6)
        self.use_tint = tk.BooleanVar(value=True)
        self.tint_level = tk.IntVar(value=6)
        self.use_contrast = tk.BooleanVar(value=True)
        self.contrast_level = tk.IntVar(value=2)
        self.use_sharpen = tk.BooleanVar(value=True)
        self.sharpen_level = tk.IntVar(value=4)
        self.look = tk.StringVar(value="color")

        # Full restoration options
        self.damage_level = tk.IntVar(value=5)
        self.use_ai_colour = tk.BooleanVar(value=False)

        self.status = tk.StringVar(value="Ready. Open a photo to begin.")
        self.size_text = tk.StringVar(value="No photo")
        self.quality_text = tk.StringVar(value="Open a photo to see the numbers.")

        self.build_title_strip()
        self.build_menu()
        self.build_toolbar()
        self.build_body()
        self.build_statusbar()

        self.root.bind("<Control-o>", lambda event: self.open_photo())
        self.root.bind("<Control-s>", lambda event: self.save_photo())

    # --------------------------------------------------------------
    # Building the window
    # --------------------------------------------------------------
    def make_button(self, parent, text, command, width=13):
        """A raised grey button like in old programs."""
        return tk.Button(parent, text=text, command=command, width=width,
                         font=FONT, bg=GRAY, activebackground=GRAY,
                         relief=tk.RAISED, bd=3)

    def build_title_strip(self):
        strip = tk.Frame(self.root, bg=NAVY)
        strip.pack(fill="x", padx=3, pady=(3, 0))
        self.title_label = tk.Label(strip, text="Old Photo Restoration", bg=NAVY,
                                    fg=WHITE, font=FONT_TITLE, anchor="w")
        self.title_label.pack(side="left", padx=8, pady=3)

    def build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Photo...   Ctrl+O", command=self.open_photo)
        file_menu.add_command(label="Save Restored Photo...   Ctrl+S", command=self.save_photo)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        process_menu = tk.Menu(menubar, tearoff=0)
        process_menu.add_command(label="Restore Photo (chosen steps)", command=self.restore)
        process_menu.add_command(label="Full Restoration", command=self.full_restore)
        process_menu.add_command(label="Reset Options", command=self.reset)
        menubar.add_cascade(label="Process", menu=process_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="How It Works", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def build_toolbar(self):
        bar = tk.Frame(self.root, bg=GRAY, relief=tk.RAISED, bd=2)
        bar.pack(fill="x", padx=3, pady=3)
        self.make_button(bar, "Open Photo", self.open_photo).pack(side="left", padx=4, pady=4)
        self.make_button(bar, "Restore", self.restore).pack(side="left", padx=4, pady=4)
        self.make_button(bar, "Full Restoration", self.full_restore, 16).pack(side="left", padx=4, pady=4)
        self.make_button(bar, "Save Photo", self.save_photo).pack(side="left", padx=4, pady=4)
        self.make_button(bar, "Reset", self.reset).pack(side="left", padx=4, pady=4)

    def build_body(self):
        body = tk.Frame(self.root, bg=GRAY)
        body.pack(fill="both", expand=True, padx=6, pady=2)

        # ----- Left side: the options -----
        left = tk.LabelFrame(body, text=" Restoration Steps ", bg=GRAY, font=FONT_BOLD,
                             bd=2, relief=tk.GROOVE)
        left.pack(side="left", fill="y", padx=(0, 8))

        self.add_step(left, "1. Remove scratches and dust", self.use_scratches, self.scratch_level, 1, 10)
        self.add_step(left, "2. Remove noise (grain)", self.use_noise, self.noise_level, 1, 15)
        self.add_step(left, "3. Remove yellow tint", self.use_tint, self.tint_level, 1, 10)
        self.add_step(left, "4. Improve faded contrast", self.use_contrast, self.contrast_level, 1, 6)
        self.add_step(left, "5. Sharpen the photo", self.use_sharpen, self.sharpen_level, 1, 10)

        look_box = tk.LabelFrame(left, text=" Final look ", bg=GRAY, font=FONT, bd=2, relief=tk.GROOVE)
        look_box.pack(fill="x", padx=8, pady=(6, 8))
        for text, value in [("Keep colours", "color"), ("Sepia (brown)", "sepia"), ("Black and white", "bw")]:
            tk.Radiobutton(look_box, text=text, value=value, variable=self.look, bg=GRAY,
                           activebackground=GRAY, selectcolor=WHITE, font=FONT,
                           anchor="w").pack(fill="x", padx=4)

        # ----- Right side: pictures and numbers -----
        right = tk.Frame(body, bg=GRAY)
        right.pack(side="left", fill="both", expand=True)

        pictures = tk.Frame(right, bg=GRAY)
        pictures.pack(fill="x")

        self.before_canvas = self.make_picture_box(pictures, " Before ")
        self.after_canvas = self.make_picture_box(pictures, " After ")
        self.show_placeholder(self.before_canvas)
        self.show_placeholder(self.after_canvas)

        bottom = tk.Frame(right, bg=GRAY)
        bottom.pack(fill="x", pady=(8, 0))

        quality = tk.LabelFrame(bottom, text=" Photo Quality ", bg=GRAY, font=FONT_BOLD,
                                bd=2, relief=tk.GROOVE)
        quality.pack(side="left", fill="both", expand=True)
        tk.Label(quality, textvariable=self.quality_text, bg=GRAY, font=FONT, height=10,
                 justify="left", anchor="nw", wraplength=380).pack(fill="both", padx=8, pady=6)

        self.build_full_box(bottom).pack(side="left", fill="y", padx=(8, 0))

    def build_full_box(self, parent):
        """The Full Restoration group box."""
        box = tk.LabelFrame(parent, text=" Full Restoration ", bg=GRAY, font=FONT_BOLD,
                            bd=2, relief=tk.GROOVE)
        tk.Label(box, text="Repairs cracks, creases and dust,\nfixes brightness, tint and contrast.",
                 bg=GRAY, font=FONT, justify="left", anchor="w").pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(box, text="Damage search level:", bg=GRAY, font=FONT,
                 anchor="w").pack(fill="x", padx=8)
        tk.Scale(box, variable=self.damage_level, from_=1, to=10, orient="horizontal",
                 bg=GRAY, troughcolor=WHITE, highlightthickness=0, length=230,
                 font=FONT).pack(padx=8)
        tk.Checkbutton(box, text="Add colours (AI model)", variable=self.use_ai_colour,
                       bg=GRAY, activebackground=GRAY, selectcolor=WHITE, font=FONT_BOLD,
                       anchor="w").pack(fill="x", padx=8)
        self.make_button(box, "Full Restoration", self.full_restore, 22).pack(pady=8)
        return box

    def add_step(self, parent, text, check_var, level_var, low, high):
        """One restoration step: a check box and a slider under it."""
        tk.Checkbutton(parent, text=text, variable=check_var, bg=GRAY, activebackground=GRAY,
                       selectcolor=WHITE, font=FONT_BOLD, anchor="w").pack(fill="x", padx=8, pady=(6, 0))
        tk.Scale(parent, variable=level_var, from_=low, to=high, orient="horizontal",
                 bg=GRAY, troughcolor=WHITE, highlightthickness=0, length=210,
                 font=FONT).pack(padx=(26, 8), pady=(0, 2))

    def make_picture_box(self, parent, title):
        box = tk.LabelFrame(parent, text=title, bg=GRAY, font=FONT_BOLD, bd=2, relief=tk.GROOVE)
        box.pack(side="left", padx=(0, 6))
        canvas = tk.Canvas(box, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=WHITE,
                           bd=2, relief=tk.SUNKEN, highlightthickness=0)
        canvas.pack(padx=6, pady=6)
        return canvas

    def build_statusbar(self):
        bar = tk.Frame(self.root, bg=GRAY)
        bar.pack(side="bottom", fill="x", padx=3, pady=3)
        tk.Label(bar, textvariable=self.status, bg=GRAY, font=FONT, anchor="w",
                 relief=tk.SUNKEN, bd=2).pack(side="left", fill="x", expand=True)
        tk.Label(bar, textvariable=self.size_text, bg=GRAY, font=FONT, width=22,
                 relief=tk.SUNKEN, bd=2).pack(side="left", padx=(3, 0))

    # --------------------------------------------------------------
    # Showing pictures
    # --------------------------------------------------------------
    def show_placeholder(self, canvas):
        canvas.delete("all")
        canvas.create_text(CANVAS_SIZE // 2, CANVAS_SIZE // 2, text="No photo",
                           font=FONT, fill="#808080")

    def show_image(self, canvas, img, name):
        """Fits the photo inside the picture box and draws it."""
        canvas.delete("all")
        height, width = img.shape[:2]
        scale = min(CANVAS_SIZE / width, CANVAS_SIZE / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   # OpenCV uses BGR, Tkinter needs RGB
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.photos[name] = photo
        canvas.create_image(CANVAS_SIZE // 2, CANVAS_SIZE // 2, image=photo)

    # --------------------------------------------------------------
    # Button actions
    # --------------------------------------------------------------
    def open_photo(self):
        path = filedialog.askopenfilename(
            title="Open Photo",
            filetypes=[("Pictures", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return

        img = restoration.load_image(path)
        if img is None:
            messagebox.showerror("Open Photo", "This file could not be opened as a picture.")
            return

        self.original = img
        self.restored = None
        self.show_image(self.before_canvas, img, "before")
        self.show_placeholder(self.after_canvas)

        height, width = img.shape[:2]
        file_name = path.replace("\\", "/").split("/")[-1]
        self.title_label.config(text="Old Photo Restoration  -  " + file_name)
        self.size_text.set("%d x %d pixels" % (width, height))
        self.status.set("Photo opened. Choose the steps and press Restore.")
        self.quality_text.set("Contrast: %.1f      Noise: %.2f      (press Restore to compare)" % (
            restoration.contrast(img), restoration.noise_level(img)))

    def get_options(self):
        """Collects the check boxes and sliders into one dictionary."""
        return {
            "scratches": self.use_scratches.get(), "scratch_level": self.scratch_level.get(),
            "noise": self.use_noise.get(), "noise_level": self.noise_level.get(),
            "tint": self.use_tint.get(), "tint_level": self.tint_level.get(),
            "contrast": self.use_contrast.get(), "contrast_level": self.contrast_level.get(),
            "sharpen": self.use_sharpen.get(), "sharpen_level": self.sharpen_level.get(),
            "look": self.look.get(),
        }

    def restore(self):
        if self.original is None:
            messagebox.showinfo("Restore", "Please open a photo first.")
            return

        self.status.set("Restoring the photo, please wait...")
        self.root.config(cursor="watch")      # the hourglass of old times
        self.root.update()

        try:
            self.restored = restoration.restore_photo(self.original, self.get_options())
        except Exception as error:
            self.root.config(cursor="")
            self.status.set("Something went wrong.")
            messagebox.showerror("Restore", "The photo could not be restored.\n\n" + str(error))
            return

        self.root.config(cursor="")
        self.show_image(self.after_canvas, self.restored, "after")
        self.status.set("Restoration complete. You can save the photo now.")
        self.quality_text.set(
            "Contrast:  %.1f  ->  %.1f\nNoise:       %.2f  ->  %.2f   (lower is cleaner)" % (
                restoration.contrast(self.original), restoration.contrast(self.restored),
                restoration.noise_level(self.original), restoration.noise_level(self.restored)))

    def full_restore(self):
        """Repairs cracks and dust, fixes the tones and can add colours."""
        if self.original is None:
            messagebox.showinfo("Full Restoration", "Please open a photo first.")
            return

        if self.use_ai_colour.get() and not restoration.colour_model_ready():
            messagebox.showwarning(
                "Add colours",
                "The AI colour model files were not found in the 'models' folder.\n\n"
                "Run   python download_models.py   to get them (about 125 MB).\n\n"
                "The photo will now be restored without added colours.")

        self.status.set("Full restoration is running. This can take up to a minute...")
        self.root.config(cursor="watch")
        self.root.update()

        try:
            self.restored, notes = restoration.full_restore(
                self.original, use_colour=self.use_ai_colour.get(),
                look=self.look.get(), level=self.damage_level.get())
        except Exception as error:
            self.root.config(cursor="")
            self.status.set("Something went wrong.")
            messagebox.showerror("Full Restoration", "The photo could not be restored.\n\n" + str(error))
            return

        self.root.config(cursor="")
        self.show_image(self.after_canvas, self.restored, "after")
        self.status.set("Full restoration complete. You can save the photo now.")
        self.quality_text.set(
            "Contrast: %.1f -> %.1f\n" % (
                restoration.contrast(self.original), restoration.contrast(self.restored))
            + "\n".join("- " + note for note in notes))

    def save_photo(self):
        if self.restored is None:
            messagebox.showinfo("Save Photo", "There is no restored photo yet.\nPress Restore first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Restored Photo", defaultextension=".jpg",
            filetypes=[("JPEG picture", "*.jpg"), ("PNG picture", "*.png"), ("Bitmap picture", "*.bmp")])
        if not path:
            return

        if restoration.save_image(path, self.restored):
            self.status.set("Saved: " + path)
        else:
            messagebox.showerror("Save Photo", "The photo could not be saved.")

    def reset(self):
        """Puts all the options back to the starting values."""
        self.use_scratches.set(True)
        self.scratch_level.set(5)
        self.use_noise.set(True)
        self.noise_level.set(6)
        self.use_tint.set(True)
        self.tint_level.set(6)
        self.use_contrast.set(True)
        self.contrast_level.set(2)
        self.use_sharpen.set(True)
        self.sharpen_level.set(4)
        self.look.set("color")
        self.damage_level.set(5)
        self.use_ai_colour.set(False)

        self.restored = None
        self.show_placeholder(self.after_canvas)
        self.status.set("Options reset.")

    def show_help(self):
        messagebox.showinfo(
            "How It Works",
            "1. Scratches and dust\n"
            "   Thin bright or dark marks are found with morphology\n"
            "   (top-hat and black-hat) and filled in by inpainting.\n\n"
            "2. Noise\n"
            "   Grain is removed with non-local means denoising.\n\n"
            "3. Yellow tint\n"
            "   The gray world method makes the average colour neutral.\n\n"
            "4. Faded contrast\n"
            "   Histogram stretching and CLAHE bring back the details.\n\n"
            "5. Sharpen\n"
            "   An unsharp mask makes the edges clearer.\n\n"
            "FULL RESTORATION\n"
            "   Cracks are found as long thin lines that are brighter than\n"
            "   both sides, then filled by inpainting (two passes). Faces are\n"
            "   protected. Then noise, brightness, tint and contrast are fixed.\n"
            "   With the AI model, colours are added to the photo.")

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Old Photo Restoration  Version 1.0\n\n"
            "Python  -  OpenCV  -  NumPy  -  Tkinter")


# ---------- Start the program ----------
if __name__ == "__main__":
    window = tk.Tk()
    app = PhotoRestorerApp(window)
    window.mainloop()
