"""A small desktop window around the converter.

Deliberately plain Tkinter: no extra packages, starts instantly, and keeps the
depth model loaded so the second photo onwards converts in well under a second.
"""

import queue
import sys
import threading
import time
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, ttk
except ImportError:  # pragma: no cover - depends on the Python build
    tk = None

from . import logbook
from .pipeline import SUFFIXES, Converter, Settings, VideoSettings
from .video import VIDEO_SUFFIXES, clock, convert_video

FILETYPES = [
    ("Photos and videos", " ".join(f"*{s}" for s in sorted(SUFFIXES | VIDEO_SUFFIXES))),
    ("Photos", " ".join(f"*{s}" for s in sorted(SUFFIXES))),
    ("Videos", " ".join(f"*{s}" for s in sorted(VIDEO_SUFFIXES))),
    ("All files", "*.*"),
]
PREVIEW_WIDTH = 620
# How wide the explanatory line under a setting is allowed to run before it
# wraps, which is what stops a long hint deciding how wide the window is.
HINT_WIDTH = 560
# Characters of room for a slider's reading.  Fixed so the sliders line up, and
# generous because Windows lays this out in a wider font than Linux does -- the
# surround slider's "off (black)" was cut to "off (black" there before this.
VALUE_WIDTH = 10
# The result sits on a dark mat, the way a print is mounted -- it is what a
# side-by-side pair is easiest to judge against, and it stops a bright photo
# glaring against the window behind it.
MAT = "#1b1b1b"
MAT_TEXT = "#9a9a9a"
EMPTY = ("Nothing converted yet\n\n"
         "The side-by-side pair appears here as each file finishes,\n"
         "and frame by frame while a clip is being converted.")
NO_CAPTION = " "  # keeps the caption's line of height, so nothing jumps later
# The cap on the finished pair's width, for a viewer that will not open
# something enormous.  Native is no cap at all, which is the usual answer.
WIDTHS = [("Native size", 0), ("Up to 4096 px", 4096), ("Up to 6144 px", 6144),
          ("Up to 8192 px", 8192), ("Up to 12288 px", 12288)]
# The window always uses the best depth model at its best working resolution, and
# neither is on show: what is on show is what the picture looks like and what it
# comes out as, which is what there is any judgement in.
#
# What is recommended depends on whether the picture moves: a clip wants a
# gentler depth than a still, because an error the eye forgives in something it
# glances at becomes a shimmer it cannot ignore over several minutes.
# Where the manual sliders sit when they are not being driven by the scene.
# `Settings` itself defaults to matching the scene, so these are the starting
# points for someone who has turned that off: a real pair of eyes at a
# comfortable distance, and something gentler once the picture moves.
RECOMMENDED = {False: (65.0, 3.0), True: (45.0, 3.0)}
# How often the window redraws the frame a conversion is currently on.  Often
# enough to look live, seldom enough that Tk is not the slowest part of it.
PREVIEW_EVERY = 1.0


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


# How each file's own row reports on it, so a batch can be followed without
# reading the one status line at the bottom.
MARKS = {"pending": " ", "working": "\u25b6", "done": "\u2713", "skipped": "\u2013",
         "failed": "\u2715", "stopped": "\u2013"}
COLOURS = {"done": "#2e7d32", "skipped": "#8a6d1f", "failed": "#b00020", "working": "#1565c0",
           "stopped": "#8a6d1f"}
# The queue is a list of files on white, the way a list of files usually is.
ROW_BG = "#ffffff"
ROW_SELECTED = "#cde4ff"
ROW_FG = "#202020"
QUEUE_WIDTH = 340
# Long names lose their middle rather than their end: the extension is the half
# that says what the file is, and a batch off a camera differs only in the digits
# just before it.
NAME_CHARS = 36
QUEUE_EMPTY = "Nothing queued.\n\nAdd photos or videos\nand they appear here."


class Row:
    """One file in the queue: what it is, how far through it is, how it went.

    A Listbox line could hold none of this.  A clip is the one thing here that
    takes minutes rather than a moment, and what it wants is a bar creeping
    along, so every file gets a small frame of widgets of its own instead.
    """

    def __init__(self, parent, small, on_click):
        self.frame = tk.Frame(parent, background=ROW_BG, padx=8, pady=5)
        self.frame.columnconfigure(0, weight=1)
        self.name = tk.Label(self.frame, background=ROW_BG, anchor="w")
        self.name.grid(row=0, column=0, sticky="ew")
        self.percent = tk.Label(self.frame, background=ROW_BG, anchor="e", width=5,
                                foreground="#555", font=small)
        self.percent.grid(row=0, column=1, sticky="e", padx=(6, 0))
        self.bar = ttk.Progressbar(self.frame, style="Queue.Horizontal.TProgressbar",
                                   maximum=1000)
        self.bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        self.detail = tk.Label(self.frame, background=ROW_BG, anchor="w", foreground="#777",
                               font=small)
        self.detail.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._spinning = False
        for widget in (self.frame, self.name, self.percent, self.detail):
            widget.bind("<Button-1>", lambda event: on_click(self, event))

    @staticmethod
    def _fit(name):
        if len(name) <= NAME_CHARS:
            return name
        keep = NAME_CHARS - 1
        return f"{name[:keep // 2]}\u2026{name[-(keep - keep // 2):]}"

    def show(self, name, state, detail, fraction):
        """`fraction` of None means nobody knows -- which for a file being worked
        on is a photo, too quick to measure, and gets a bar that simply moves."""
        self.name.config(text=f"{MARKS[state]}  {self._fit(name)}",
                         foreground=COLOURS.get(state, ROW_FG))
        self.detail.config(text=detail)
        self.detail.grid() if detail else self.detail.grid_remove()
        if state == "working" and fraction is None:
            self._spin(True)
        else:
            self._spin(False)
            if fraction is not None:
                self.bar.config(value=round(fraction * 1000))
        if state in ("skipped", "failed", "stopped"):
            self.percent.config(text="\u2013")
        elif state == "pending":
            self.percent.config(text="")
        else:  # ... for a file being worked on that cannot say how far along
            self.percent.config(text=f"{fraction:.0%}" if fraction is not None else "\u2026")

    def _spin(self, on):
        """A bar with nothing to report moves rather than stands still, which is
        the difference between working and hung."""
        if on == self._spinning:
            return
        self._spinning = on
        if on:
            self.bar.config(mode="indeterminate", value=0)
            self.bar.start(40)
        else:
            self.bar.stop()
            self.bar.config(mode="determinate")

    def select(self, on):
        colour = ROW_SELECTED if on else ROW_BG
        for widget in (self.frame, self.name, self.percent, self.detail):
            widget.config(background=colour)

    def destroy(self):
        self.frame.destroy()


class App:
    def __init__(self, root):
        self.root = root
        self.files = []
        self.states = []  # one ("state", "detail") per file, in step with it
        self.running = False
        self.cancel = threading.Event()
        # Set means carry on, so a pause is this being cleared and the worker
        # waiting on it.  An event rather than a flag because the waiting has to
        # cost nothing: a paused run holds no lock and burns no CPU.
        self.carry_on = threading.Event()
        self.carry_on.set()
        self.paused_status = ""  # what the status line said before it was paused
        # What the sliders would say if nobody had touched them, which depends on
        # what is in the queue; see `_sync_recommendation`.
        self.recommended = RECOMMENDED[False]
        self.events = queue.Queue()
        self.converter = Converter()
        self.output_dir = None
        self.preview = None
        self.finished = 0
        self.errors = []
        self.notices = []
        self.cross_used = False  # the viewing order the current run is writing

        root.title("StereoCraft - side-by-side 3D")
        self._set_icon(root)
        root.minsize(1060, 720)
        # A size below the default for the second line of a queue row and the
        # explanations under the settings, asked of the theme rather than named,
        # so it follows whatever the machine's own size is.
        self.small_font = tkfont.nametofont("TkDefaultFont").copy()
        self.small_font.configure(size=max(7, abs(self.small_font.cget("size")) - 1))
        # Thin, because there is one of these on every row of the queue and a
        # full-height bar would make each file twice as tall as it needs to be.
        ttk.Style().configure("Queue.Horizontal.TProgressbar", thickness=8)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        # The queue keeps a fixed width and everything else takes the slack:
        # file names are all much of a length, whereas the result is what has
        # something to do with every extra pixel.
        frame.columnconfigure(0, minsize=QUEUE_WIDTH + 24)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        self._build_queue(frame)
        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.columnconfigure(0, weight=1)
        self._build_settings(right)
        self._build_result(right)

        footer = ttk.Frame(frame)
        footer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        footer.columnconfigure(3, weight=1)
        self.convert_button = ttk.Button(footer, text="Convert", command=self.start)
        self.convert_button.grid(row=0, column=0)
        # Stopping a clip throws away every frame of it, and an hour in that is
        # a high price for wanting the machine back for ten minutes.  Pausing is
        # the cheap version of the same wish, so it stands beside Stop rather
        # than making Stop the only way out.  Its width is fixed at the longer
        # of the two words it shows, so that pressing it does not shove Stop and
        # the bar sideways.
        self.pause_button = ttk.Button(footer, text="Pause", width=8,
                                       command=self.toggle_pause)
        self.pause_button.grid(row=0, column=1, padx=(6, 0))
        # A photo is over before anyone could ask for it back; a clip runs for
        # minutes, so there has to be a way out of one.
        self.stop_button = ttk.Button(footer, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=2, padx=(6, 0))
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.grid(row=0, column=3, sticky="ew", padx=8)
        self.progress_percent = ttk.Label(footer, text="0%", width=5, anchor="e")
        self.progress_percent.grid(row=0, column=4, sticky="e")
        self.status = ttk.Label(footer, text="Add a photo or a video to begin")
        self.status.grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 0))

        # Every setting a Reset would put back.  Watching all of them is what
        # tells each tab's button whether it has anything left to undo, and the
        # eyes and the focus which of them are the user's to set.
        for variable in (self.automatic, self.eyes, self.focus, self.cross,
                         self.photo_depth, self.fmt, self.quality, self.max_size,
                         self.save_depth, self.target, self.temporal, self.crf,
                         self.codec, self.audio, self.full_width):
            variable.trace_add("write", lambda *_: self._refresh_controls())
        self._refresh_controls()

        root.after(100, self._drain)

    @staticmethod
    def _set_icon(root):
        """Put the app's icon on the window.

        Tk draws its own title bar icon and defaults to the Tcl feather, so the
        one built into the exe never reaches the window and has to be set here.
        """
        path = Path(__file__).with_name("stereocraft.ico")
        if not path.exists():  # running from a checkout without the icon
            return
        try:
            root.iconbitmap(default=str(path))  # Windows takes the .ico itself
        except tk.TclError:
            try:  # X11 wants an image rather than an .ico, so decode it first
                from PIL import Image, ImageTk

                with Image.open(path) as image:
                    root._icon = ImageTk.PhotoImage(image)  # Tk drops it unreferenced
                root.iconphoto(True, root._icon)
            except Exception:  # a window with the wrong icon still converts photos
                pass

    def _build_queue(self, parent):
        """The list of files, down the left-hand side of the window.

        A column rather than a band across the top: a queue grows downwards, so
        given the height it shows a whole batch at once, and the width it gives
        up is width the picture beside it would not have used anyway.
        """
        box = ttk.LabelFrame(parent, text="Queue", padding=8)
        box.grid(row=0, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(1, weight=1)

        buttons = ttk.Frame(box)
        buttons.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.add_button = ttk.Button(buttons, text="Add files...", command=self.add_files)
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(buttons, text="Remove", command=self.remove_selected)
        self.remove_button.pack(side="left", padx=4)
        self.clear_button = ttk.Button(buttons, text="Clear", command=self.clear)
        self.clear_button.pack(side="left")

        # A frame of widgets inside a canvas is Tk's way of having a list that
        # scrolls and holds more than text.
        area = ttk.Frame(box)
        area.grid(row=1, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)
        self.queue_canvas = tk.Canvas(area, background=ROW_BG, width=QUEUE_WIDTH,
                                      highlightthickness=1, highlightbackground="#b5b5b5",
                                      borderwidth=0)
        self.queue_canvas.grid(row=0, column=0, sticky="nsew")
        self.rows_frame = tk.Frame(self.queue_canvas, background=ROW_BG)
        held = self.queue_canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.rows_frame.bind("<Configure>", lambda *_: self.queue_canvas.config(
            scrollregion=self.queue_canvas.bbox("all")))
        # The rows are as wide as the canvas rather than as wide as their
        # contents, so a long name is clipped instead of widening the window.
        self.queue_canvas.bind("<Configure>", lambda event: self.queue_canvas.itemconfigure(
            held, width=event.width))
        # A tall list is a list worth scrolling, and the bar only appears once
        # there is more in the queue than fits.
        scroll = ttk.Scrollbar(area, orient="vertical", command=self.queue_canvas.yview)
        self.queue_canvas.config(
            yscrollcommand=lambda first, last: self._scrollbar(scroll, first, last))
        # The wheel is caught for the whole window and acted on only over the
        # queue.  Binding it to the canvas instead would miss: every row is a
        # widget in its own right, and the pointer is almost always on one of
        # them rather than on the canvas behind.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(sequence, self._wheeled)

        self.rows = []
        self.selected = set()
        self.anchor = None  # where a shift-click measures its run from
        self.empty_label = tk.Label(self.rows_frame, text=QUEUE_EMPTY, background=ROW_BG,
                                    foreground="#8a8a8a", justify="center", pady=28)
        self._show_empty()

    @staticmethod
    def _scrollbar(scroll, first, last):
        """Show the queue's scrollbar only while it has somewhere to scroll."""
        scroll.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            scroll.grid_remove()
        else:
            scroll.grid(row=0, column=1, sticky="ns", padx=(2, 0))

    def _wheeled(self, event):
        """Turn the queue, if the queue is what the pointer is over."""
        under = self.root.winfo_containing(event.x_root, event.y_root)
        if under is None or not str(under).startswith(str(self.queue_canvas)):
            return
        # Windows and macOS send a delta, X11 sends button 4 or 5 instead.
        up = event.num == 4 if getattr(event, "num", 0) in (4, 5) else event.delta > 0
        self.queue_canvas.yview_scroll(-2 if up else 2, "units")

    def _clicked(self, row, event):
        """Click for one, ctrl-click to add to the selection, shift-click for a
        run of them: what the list this replaced did, since Remove works from it."""
        index = self.rows.index(row)
        if event.state & 0x0004:  # ctrl
            self.selected ^= {index}
        elif event.state & 0x0001 and self.anchor is not None:  # shift
            first, last = sorted((self.anchor, index))
            self.selected = set(range(first, last + 1))
        else:
            self.selected = {index}
        if not event.state & 0x0001:
            self.anchor = index
        self._paint_selection()
        self._refresh_controls()

    def _paint_selection(self):
        for index, row in enumerate(self.rows):
            row.select(index in self.selected)

    def _show_empty(self):
        """A queue with nothing in it says so, rather than being a white void."""
        if self.files:
            self.empty_label.pack_forget()
        else:
            self.empty_label.pack(fill="x")

    def _see(self, index):
        """Scroll a row into view, for following a long batch down the queue."""
        self.queue_canvas.update_idletasks()
        total = self.rows_frame.winfo_height()
        row = self.rows[index].frame
        if total <= 0 or not row.winfo_ismapped():
            return
        first, last = self.queue_canvas.yview()
        top, bottom = row.winfo_y() / total, (row.winfo_y() + row.winfo_height()) / total
        if top < first or bottom > last:
            self.queue_canvas.yview_moveto(max(0.0, top - 0.02))

    def _build_settings(self, parent):
        """The settings, as three tabs of what they apply to.

        Split that way rather than by what they do: a run is usually all photos
        or all clips, and a tab puts the half that has nothing to say about the
        queue out of the way instead of greying it out in front of you.
        """
        self.tabs = ttk.Notebook(parent)
        self.tabs.grid(row=0, column=0, sticky="ew")
        # One Reset per tab, each with the question of whether it has anything
        # left to put back; see `_reset_button`.
        self._resets = []
        self._manual = []  # sliders that only apply when NOT matching the scene
        # ...and the ones that only apply when it is.  Both of them are the Depth
        # slider on a tab, which is a percentage of frame width -- so they also
        # need a flat projection to mean anything, and hand over to the
        # "Depth (VR180)" slider on General when the picture goes on a sphere.
        self._auto_only = []
        self._vr180_only = []  # ...and the ones with nothing to say about a flat pair
        self._values = []  # every slider's reading, for the test that they all fit
        self._build_general(self.tabs)
        self.photo_tab = self._build_photo(self.tabs)
        self.video_tab = self._build_video(self.tabs)

    def _build_general(self, tabs):
        """What the scene looks like, whether it moves or not."""
        box = ttk.Frame(tabs, padding=10)
        tabs.add(box, text="General")
        box.columnconfigure(1, weight=1)

        # On by default, because sizing the eyes to the scene beats any single
        # number: the same 65mm that suits a portrait renders a close-up
        # unfusable and a telephoto shot flat.
        self.automatic = tk.BooleanVar(value=True)
        self.eyes = tk.DoubleVar(value=self.recommended[0])
        # Held as 1/metres rather than metres.  That is the quantity the geometry
        # is linear in, so an inch of travel changes the picture by as much at one
        # end of the slider as at the other; in metres the far half would do
        # almost nothing and the near half everything.
        self.focus = tk.DoubleVar(value=1.0 / self.recommended[1])
        self.cross = tk.BooleanVar(value=False)

        # First, because it decides which of the two below it are yours to set:
        # matching the scene drives the eyes and the focus itself, and takes its
        # instructions from the Depth slider on the tab for what is in the queue.
        ttk.Checkbutton(box, text="Match the scene automatically (recommended)",
                        variable=self.automatic).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text="Sizes the eyes and the focus to what is actually in the picture,"
                            " and aims for the Depth set on the Photo or Video tab.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self._manual.append(self._slider(
            box, 2, "Eye separation", self.eyes, 20.0, 80.0, "{:.0f} mm",
            "How far apart your two eyes are. 65mm is the human average; smaller is a"
            " gentler effect, and a video is recommended gentler than a photo."))
        self._manual.append(self._slider(
            box, 4, "Focus distance", self.focus, 1.0 / 20, 1.0 / 0.5,
            lambda v: f"{1.0 / v:.1f} m",
            "How far away the window sits. Whatever is at this distance looks like it is"
            " in the screen; nearer comes out of it, further recedes behind it."))

        ttk.Checkbutton(box, text="Cross-eyed order (only for free-viewing without a headset)",
                        variable=self.cross).grid(row=6, column=0, columnspan=3, sticky="w",
                                                  pady=(4, 0))

        # The projection, and under it the one setting that only means anything
        # once it is a sphere.  Greyed rather than hidden when it is not, so the
        # window does not change shape as you look at it.
        self.projection = tk.StringVar(value=Settings.projection)
        self.surround = tk.DoubleVar(value=Settings.vr180_surround)
        self.vr180_depth = tk.DoubleVar(value=Settings.target_deg)
        self.projection.trace_add("write", lambda *_: self._refresh_controls())

        row = ttk.Frame(box)
        row.grid(row=7, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(row, text="Projection").pack(side="left")
        for value, text in (("flat", "Flat (a screen in front of you)"),
                            ("vr180", "VR180 (wrapped around you)")):
            ttk.Radiobutton(row, text=text, value=value,
                            variable=self.projection).pack(side="left", padx=(8, 0))
        ttk.Label(box, text="VR180 puts the picture at its true angular size instead of on a"
                            " screen the player scales to taste. Most of the sphere is dark,"
                            " because a camera only ever pointed at part of it.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(
            row=8, column=0, columnspan=3, sticky="w")

        self._vr180_only.append(self._slider(
            box, 9, "Surround", self.surround, 0.0, 2.0,
            lambda v: "off" if v < 0.02 else f"{v:.2f}",
            "Fills the dark with a blurred spread of the picture, the way a social video site"
            " does behind a clip that does not fill the frame. Above 1 it is brighter than the"
            " picture was, for a scene too dark to light itself."))
        # The Depth sliders on the Photo and Video tabs are percentages of frame
        # width, which a 180-degree frame makes meaningless -- so this is the one
        # they hand over to here, and they are greyed while it applies.  Both
        # kinds of input share it: the case for a clip asking less than a still
        # is about depth-map error shimmering, and that argument has not been
        # measured on a sphere the way it has on a plane.
        self._vr180_only.append(self._slider(
            box, 11, "Depth (VR180)", self.vr180_depth, 0.2, 3.0, "{:.2f} deg",
            "How much near-to-far separation the sphere aims for, in degrees of arc."
            " 0.6 is the shipped default and a cautious one; real eyes on a close scene"
            " would give several times it. Raise it until the depth reads, then back off."))

        self._reset_button(box, 13, self.reset_general, self._general_at_default)

    def _build_photo(self, tabs):
        """How much depth a still asks for, and what it is written as."""
        box = ttk.Frame(tabs, padding=10)
        tabs.add(box, text="Photo")
        box.columnconfigure(1, weight=1)

        self.photo_depth = tk.DoubleVar(value=Settings.target_pct)
        self.fmt = tk.StringVar(value=Settings.fmt)
        self.quality = tk.DoubleVar(value=Settings.quality)
        self.max_size = tk.StringVar(value=WIDTHS[0][0])
        self.save_depth = tk.BooleanVar(value=False)

        self._auto_only.append(self._slider(
            box, 0, "Depth", self.photo_depth, 0.5, 3.5, "{:.1f}%",
            "How much separation matching the scene aims for, as a share of the width."
            " 2% is a comfortable pair of eyes; more is stronger and harder to fuse."))

        row = ttk.Frame(box)
        row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(row, text="File format").pack(side="left")
        for value, text in (("auto", "Same as the photo"), ("jpg", "JPEG"), ("png", "PNG")):
            ttk.Radiobutton(row, text=text, value=value,
                            variable=self.fmt).pack(side="left", padx=(8, 0))
        ttk.Label(box, text="A side-by-side pair is twice the pixels of the photo, so JPEG at a"
                            " high quality is usually the sensible one. PNG is lossless and large.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 4))

        self.quality_scale = self._slider(
            box, 4, "JPEG quality", self.quality, 70.0, 100.0, "{:.0f}",
            "95 keeps the compression out of the depth cues; below about 85 the edges the"
            " warp opened up start to show as blocks.")

        row = ttk.Frame(box)
        row.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(row, text="Output width").pack(side="left")
        self.width_box = ttk.Combobox(row, textvariable=self.max_size, state="readonly", width=18,
                                      values=[name for name, _ in WIDTHS])
        self.width_box.pack(side="left", padx=(8, 0))
        ttk.Label(box, text="A cap for a viewer that will not open something enormous. The pair"
                            " comes out about twice as wide as the photo went in.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=7, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Checkbutton(box, text="Also save the depth map", variable=self.save_depth).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._reset_button(box, 9, self.reset_photo, self._photo_at_default)
        return box

    def _build_video(self, tabs):
        """What a clip costs to look at for several minutes.

        Its own tab rather than mixed in with the two sliders on General: those
        are about what the scene looks like, these are about what holds up over
        a few thousand frames of it.
        """
        box = ttk.Frame(tabs, padding=10)
        tabs.add(box, text="Video")
        box.columnconfigure(1, weight=1)

        self.target = tk.DoubleVar(value=VideoSettings.target_pct)
        self.temporal = tk.DoubleVar(value=VideoSettings.temporal)
        self.codec = tk.StringVar(value=VideoSettings.codec)
        self.crf = tk.DoubleVar(value=VideoSettings.crf)
        self.audio = tk.BooleanVar(value=VideoSettings.audio)
        self.full_width = tk.BooleanVar(value=VideoSettings.full_width)

        self._auto_only.append(self._slider(
            box, 0, "Depth", self.target, 0.4, 3.0, "{:.1f}%",
            "How much separation a clip aims for. Lower than a photo on purpose:"
            " an error you would not notice in a still shimmers once it moves."))
        self._slider(box, 2, "Steadiness", self.temporal, 0.0, 0.95, "{:.2f}",
                     "How much of each frame's depth carries into the next. Higher is"
                     " calmer and very slightly softer; 0 turns it off.")
        self._slider(box, 4, "Encoder quality", self.crf, 14.0, 28.0, "CRF {:.0f}",
                     "Lower is better and bigger. 18 is visually lossless; 23 is about half"
                     " the file and still good.")

        row = ttk.Frame(box)
        row.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(row, text="Codec").pack(side="left")
        for value, text in (("auto", "Auto"), ("h264", "H.264 (plays anywhere)"),
                            ("hevc", "HEVC (needed above 4K)")):
            ttk.Radiobutton(row, text=text, value=value, variable=self.codec).pack(side="left", padx=(8, 0))

        ttk.Checkbutton(box, text="Keep the soundtrack", variable=self.audio).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(box, text="Full width (every native pixel, twice as wide)",
                        variable=self.full_width).grid(row=8, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text="Off squeezes each eye to half width, which is what players and"
                            " headsets expect and what their decoders can keep up with.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(row=9, column=0, columnspan=3, sticky="w")

        # These only act when the clip is short of something, so leaving them on
        # costs nothing on footage that has nothing to gain -- which is why they
        # are plain checkboxes rather than numbers to get right.
        # `outpaint` has no switch here on purpose.  It ships without weights --
        # see the note at the top of `outpaint` for what was measured and why --
        # so a box offering it would promise something the app cannot do.  The
        # setting and the command-line flags are still there for anyone picking
        # the experiment back up.
        self.upscale = tk.BooleanVar(value=VideoSettings.upscale)
        self.interpolate = tk.BooleanVar(value=VideoSettings.interpolate)
        self._vr180_only.append(ttk.Checkbutton(
            box, text="Add detail to a clip too small to fill the frame", variable=self.upscale))
        self._vr180_only[-1].grid(row=10, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self._vr180_only.append(ttk.Checkbutton(
            box, text="Smooth a clip slower than 60fps", variable=self.interpolate))
        self._vr180_only[-1].grid(row=11, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text="Both run before the conversion and only when the clip needs them."
                            " Each roughly doubles how long it takes, and together they can"
                            " take several times as long as the conversion itself.",
                  foreground="#777", wraplength=HINT_WIDTH, justify="left").grid(
            row=12, column=0, columnspan=3, sticky="w")

        self._reset_button(box, 13, self.reset_video, self._video_at_default)
        return box

    def _build_result(self, parent):
        """Where the finished pair is shown.

        A picture wants a dark mat around it, but a dark rectangle with nothing
        in it reads as something broken -- so until there is a picture the mat
        says what will be arriving in it, and afterwards a line underneath says
        what it is looking at.
        """
        box = ttk.LabelFrame(parent, text="Result", padding=8)
        box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(1, weight=1)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1, minsize=PREVIEW_WIDTH // 2 + 16)

        self.canvas = tk.Label(box, background=MAT, foreground=MAT_TEXT, text=EMPTY,
                               justify="center", anchor="center", compound="center",
                               borderwidth=1, relief="solid", padx=10, pady=10)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.caption = ttk.Label(box, text=NO_CAPTION, foreground="#777", anchor="center")
        self.caption.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    def _slider(self, parent, row, label, variable, low, high, fmt, hint):
        """`fmt` is a format string, or a callable for a value the slider does not
        hold directly -- focus distance being stored the other way up.

        The value column is fixed so the sliders line up, which means a reading
        wider than `VALUE_WIDTH` is not squeezed, it is cut off -- and worse on
        Windows, whose default font is wider than the one this is laid out
        against.  `_values` collects the labels so a test can check every
        reading each slider can produce still fits.
        """
        show = fmt if callable(fmt) else fmt.format
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        value = ttk.Label(parent, text=show(variable.get()), width=VALUE_WIDTH, anchor="e")
        self._values.append((value, show, low, high))
        # Watching the variable rather than the slider, so a value put back by a
        # Reset shows up the same as one dragged there.
        variable.trace_add("write", lambda *_: value.config(text=show(variable.get())))
        scale = ttk.Scale(parent, from_=low, to=high, variable=variable, orient="horizontal")
        scale.grid(row=row, column=1, sticky="ew", padx=8)
        value.grid(row=row, column=2, sticky="w")
        ttk.Label(parent, text=hint, foreground="#777", wraplength=HINT_WIDTH,
                  justify="left").grid(row=row + 1, column=0, columnspan=3, sticky="w", pady=(0, 4))
        return scale  # so the caller can say when it applies; see `_refresh_controls`

    def _reset_button(self, parent, row, reset, at_default):
        """Each tab puts its own settings back, and only its own.

        One button for the lot would be a button that undoes work on a tab you
        cannot see; and it goes grey when its tab is already at its defaults,
        which is also how you can tell at a glance that it is.
        """
        button = ttk.Button(parent, text="Reset to default", command=reset)
        button.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self._resets.append((button, at_default))
        return button

    def reset_general(self):
        eyes, focus = self.recommended
        self.automatic.set(True)
        self.eyes.set(eyes)
        self.focus.set(1.0 / focus)
        self.cross.set(False)
        self.projection.set(Settings.projection)
        self.surround.set(Settings.vr180_surround)
        self.vr180_depth.set(Settings.target_deg)

    def _general_at_default(self):
        """What the eyes and the focus go back to follows the queue: a clip is
        recommended gentler than a still.  See `_sync_recommendation`."""
        return (self.automatic.get() and self._sliders_at(self.recommended)
                and not self.cross.get()
                and self.projection.get() == Settings.projection
                and round(self.surround.get(), 2) == Settings.vr180_surround
                and round(self.vr180_depth.get(), 2) == Settings.target_deg)

    def reset_photo(self):
        self.photo_depth.set(Settings.target_pct)
        self.fmt.set(Settings.fmt)
        self.quality.set(Settings.quality)
        self.max_size.set(WIDTHS[0][0])
        self.save_depth.set(False)

    def _photo_at_default(self):
        return (round(self.photo_depth.get(), 2) == Settings.target_pct
                and self.fmt.get() == Settings.fmt
                and round(self.quality.get()) == Settings.quality
                and self.max_size.get() == WIDTHS[0][0]
                and not self.save_depth.get())

    def reset_video(self):
        self.target.set(VideoSettings.target_pct)
        self.temporal.set(VideoSettings.temporal)
        self.crf.set(VideoSettings.crf)
        self.codec.set(VideoSettings.codec)
        self.audio.set(VideoSettings.audio)
        self.full_width.set(VideoSettings.full_width)
        self.upscale.set(VideoSettings.upscale)
        self.interpolate.set(VideoSettings.interpolate)

    def _video_at_default(self):
        return (round(self.target.get(), 2) == VideoSettings.target_pct
                and round(self.temporal.get(), 2) == VideoSettings.temporal
                and round(self.crf.get()) == VideoSettings.crf
                and self.codec.get() == VideoSettings.codec
                and self.audio.get() == VideoSettings.audio
                and self.full_width.get() == VideoSettings.full_width
                and self.upscale.get() == VideoSettings.upscale
                and self.interpolate.get() == VideoSettings.interpolate)

    def _sliders_at(self, values):
        eyes, focus = values
        return (round(self.eyes.get(), 1) == round(eyes, 1)
                and round(1.0 / self.focus.get(), 2) == round(focus, 2))

    def _sync_recommendation(self):
        """Follow the queue: a video recommends gentler settings than a photo.

        Sliders still sitting where the app left them are the app's to move, and
        move.  Sliders the user has set are theirs, and are left alone -- they
        only find out what changed from the note at the bottom.  A queue holding
        both takes the video's advice, being the more conservative of the two.
        """
        wanted = RECOMMENDED[any(is_video(path) for path in self.files)]
        if wanted == self.recommended:
            return None
        untouched = self._sliders_at(self.recommended)
        self.recommended = wanted
        if not untouched:
            return None
        self.eyes.set(wanted[0])
        self.focus.set(1.0 / wanted[1])
        return wanted

    # --- file list ---------------------------------------------------------
    def add_files(self):
        chosen = filedialog.askopenfilenames(title="Choose photos or videos", filetypes=FILETYPES)
        first = not self.files
        for name in chosen:
            path = Path(name)
            if path not in self.files:
                self.files.append(path)
                self.states.append(("pending", ""))
                row = Row(self.rows_frame, self.small_font, self._clicked)
                row.frame.pack(fill="x")
                self.rows.append(row)
                self._set_row(len(self.files) - 1, "pending")
        # Opening a queue puts up the tab that has something to say about it.
        # Only the first time, so a tab chosen since is not taken away again.
        if first and self.files:
            self.tabs.select(self.video_tab if is_video(self.files[0]) else self.photo_tab)
        self._show_empty()
        moved = self._sync_recommendation()
        self._refresh_status()
        if moved:
            self.status.config(text=f"{self.status.cget('text')}  -  eyes eased to "
                                    f"{moved[0]:.0f}mm for video")

    def remove_selected(self):
        for index in sorted(self.selected, reverse=True):
            self.rows.pop(index).destroy()
            del self.files[index]
            del self.states[index]
        self.selected.clear()
        self.anchor = None
        self._paint_selection()
        self._show_empty()
        self._sync_recommendation()
        self._refresh_status()

    def clear(self):
        for row in self.rows:
            row.destroy()
        self.rows.clear()
        self.selected.clear()
        self.anchor = None
        self.files.clear()
        self.states.clear()
        self._show_empty()
        self._sync_recommendation()
        self._refresh_status()
        self._clear_result()

    def _set_row(self, index, state, detail="", fraction=None):
        """Bring one file's row up to date.

        `fraction` left out means the state decides: nothing done, everything
        done, or -- for a file being worked on with no frames to count -- nobody
        knows, which is a bar that moves rather than one that fills.
        """
        self.states[index] = (state, detail)
        if fraction is None:
            fraction = {"pending": 0.0, "done": 1.0}.get(state)
        self.rows[index].show(self.files[index].name, state, detail, fraction)
        if state == "working":
            self._see(index)  # follow a long batch down the queue

    def _refresh_controls(self):
        """Offer only what there is currently something to do with."""
        idle = not self.running
        for button, usable in (
            (self.add_button, idle),
            # A batch works from the list as it stood when Convert was pressed,
            # so the list stays put until it has finished with it.
            (self.remove_button, idle and bool(self.selected)),
            (self.clear_button, idle and bool(self.files)),
            *((button, not at_default()) for button, at_default in self._resets),
            (self.stop_button, self.running and not self.cancel.is_set()),
            # Still yours while paused -- it is the only way back out of one.
            (self.pause_button, self.running and not self.cancel.is_set()),
            # The eyes and the focus are yours only when the scene is not
            # setting them; the Depth sliders are the other way round, being
            # what matching the scene aims for.  And they are percentages of
            # frame width, so a spherical frame takes them out of play in favour
            # of the one that asks in degrees -- they used to stay lit and do
            # nothing, which is the fault this pair of conditions exists to fix.
            *((slider, not self.automatic.get()) for slider in self._manual),
            *((slider, self.automatic.get() and self.projection.get() != "vr180")
              for slider in self._auto_only),
            *((widget, self.projection.get() == "vr180") for widget in self._vr180_only),
            (self.quality_scale, self.fmt.get() != "png"),
        ):
            button.state(["!disabled"] if usable else ["disabled"])
        # Asked of the event rather than remembered separately, so the button
        # cannot come to disagree with what the worker is actually doing.
        self.pause_button.config(text="Pause" if self.carry_on.is_set() else "Resume")

    def _set_progress(self, value, maximum=None):
        """The bar and the number beside it, which only ever move together."""
        if maximum is not None:
            self.progress.config(maximum=max(1, maximum))
        self.progress.config(value=value)
        self.progress_percent.config(text=f"{value / float(self.progress.cget('maximum')):.0%}")

    def pick_output(self):
        folder = filedialog.askdirectory(title="Choose an output folder")
        self.output_dir = Path(folder) if folder else None
        self.dest_label.config(text=f"Saving to {self.output_dir.name}" if self.output_dir
                               else "Saving beside each file")

    def _refresh_status(self):
        count = len(self.files)
        clips = sum(is_video(path) for path in self.files)
        if not count:
            what = "Add a photo or a video to begin"
        elif clips == count:
            what = f"{count} video{'s' if count > 1 else ''} ready"
        elif clips:
            what = f"{count - clips} photo{'s' if count - clips > 1 else ''} and {clips} " \
                   f"video{'s' if clips > 1 else ''} ready"
        else:
            what = f"{count} photo{'s' if count > 1 else ''} ready"
        self.status.config(text=what)
        self._refresh_controls()

    # --- conversion --------------------------------------------------------
    def start(self):
        if not self.files:
            self.add_files()
            return
        self.convert_button.state(["disabled"])
        self.running = True
        self.cross_used = self.cross.get()
        self.cancel.clear()
        self.carry_on.set()
        self.finished = 0
        self.errors = []
        self.notices = []
        for index in range(len(self.files)):
            self._set_row(index, "pending")  # a re-run starts everything over
        self._refresh_controls()
        self._clear_result("Converting...")
        self._set_progress(0, maximum=len(self.files))
        automatic = self.automatic.get()
        common = dict(
            eyes_mm="auto" if automatic else round(self.eyes.get(), 1),
            focus_m="auto" if automatic else round(1.0 / self.focus.get(), 3),
            cross_eyed=self.cross.get(),
            projection=self.projection.get(),
            vr180_surround=round(self.surround.get(), 2),
            target_deg=round(self.vr180_depth.get(), 2),
            on_notice=self._notice,
            on_oversize=self._ask_oversize,
        )
        settings = (Settings(target_pct=round(self.photo_depth.get(), 2),
                             quality=int(round(self.quality.get())), fmt=self.fmt.get(),
                             max_size=dict(WIDTHS)[self.max_size.get()],
                             save_depth=self.save_depth.get(), **common),
                    VideoSettings(target_pct=round(self.target.get(), 2),
                                  temporal=round(self.temporal.get(), 2),
                                  crf=int(round(self.crf.get())), codec=self.codec.get(),
                                  audio=self.audio.get(), full_width=self.full_width.get(),
                                  upscale=self.upscale.get(),
                                  interpolate=self.interpolate.get(), **common))
        threading.Thread(target=self._work, args=(list(self.files), self.output_dir, settings),
                         daemon=True).start()

    def stop(self):
        """Ask the run to stop.  A clip gives up on the frame it is on and leaves
        no half-written file; a batch of photos stops after the one in hand."""
        self.cancel.set()
        # A paused run is asleep in `_hold` and would never see this; letting it
        # go is what turns Stop-while-paused into a stop rather than a hang.
        self.carry_on.set()
        self.status.config(text="Stopping...")
        self._refresh_controls()

    def toggle_pause(self):
        """Hold the run where it stands, or let it carry on.

        A pause takes effect at the next frame rather than at once -- the frame
        in hand is finished and written first, which is what makes resuming free
        and leaves the encoder a clip it can still finish.  So the button says
        what has been asked for and the worker says when it has happened.
        """
        if self.carry_on.is_set():
            self.paused_status = self.status.cget("text")
            self.carry_on.clear()
            self.status.config(text="Pausing...")
        else:
            self.carry_on.set()
            # Put back here as well as on the worker's word, for the resume that
            # comes before the worker has even noticed the pause: there is no
            # "carrying on" to report, and "Pausing..." would sit there for the
            # rest of the clip.
            self.status.config(text=self.paused_status or "Converting...")
        self._refresh_controls()

    def _hold(self):
        """Wait here while the run is paused.  Runs on the worker thread.

        Called from the progress callbacks, which is the one moment a conversion
        is between frames: the encoder has everything it has been given and the
        decoder is blocked on a full pipe, so both sit there for as long as this
        does and neither minds.  The card is handed back before the wait rather
        than held through it -- see `Converter.let_go`, which is where the
        measurements are and why taking it again costs a couple of seconds and
        not a frame of the picture.

        Returns how long it waited, which the caller takes off the clock it is
        estimating from: an hour paused is not an hour of slow conversion, and
        the time left would otherwise read as though it were.
        """
        if self.carry_on.is_set():
            return 0.0
        began = time.monotonic()
        freed = self.converter.let_go()
        logbook.note("paused", freed=logbook._gb(freed))
        self.events.put(("paused", freed))
        self.carry_on.wait()
        waited = time.monotonic() - began
        logbook.note("resumed", after=f"{waited:.0f}s")
        self.events.put(("resumed", None))
        return waited

    def _notice(self, message):
        """Something the run needs to say that is not the result.

        Runs on the worker thread, so it goes to the main loop as an event like
        everything else.  It exists at all because these used to go to stderr,
        and the window is frozen with no console -- so a conversion that quietly
        did less than it was asked to said so to nobody.
        """
        self.events.put(("notice", message))

    def _ask_oversize(self, oversize):
        """Put a too-large photo to the user.  Runs on the worker thread, so the
        question goes to the main loop as an event and waits for the answer."""
        reply = queue.Queue(maxsize=1)
        self.events.put(("ask", (oversize, reply)))
        return reply.get()

    def _work(self, files, output_dir, settings):
        for_photos, for_videos = settings
        try:
            self.events.put(("status", "Loading depth model..."))
            try:
                self.converter.settings = for_photos
                self.converter.depth_model  # pay the load cost before the first file
            except Exception as error:
                self.events.put(("error", (None, f"Could not load the depth model: {error}")))
                return
            for index, path in enumerate(files):
                # Between files, which is where a batch of photos waits: one
                # photo is over long before anybody could pause it.
                self._hold()
                if self.cancel.is_set():
                    self.events.put(("stopped", index))
                    continue
                self.events.put(("status", f"Converting {path.name} ({index + 1}/{len(files)})"))
                self.events.put(("working", index))
                moving = is_video(path)
                self.converter.settings = for_videos if moving else for_photos
                try:
                    if moving:
                        info = convert_video(path, output_dir, self.converter,
                                             self._progress(index), self._previewer(path),
                                             on_stage=self._stage(index))
                    else:
                        info = self.converter.convert(path, output_dir)
                except Exception as error:
                    self.events.put(("error", (index, f"{path.name}: {error}")))
                    continue
                if info is None:  # too large and skipped, or stopped part-way
                    kind = "stopped" if self.cancel.is_set() else "skipped"
                    self.events.put((kind, index if kind == "stopped" else (index, path)))
                    continue
                self.events.put(("done", (index, info)))
        finally:
            self.events.put(("finished", None))

    def _stage(self, index):
        """Report the passes that run before a conversion.

        Adding detail and smoothing take minutes each, and a window that says
        "converting" through both of them reads as a hang.  The queue bar is
        deliberately left where it is -- these are not frames of the finished
        clip, and advancing it here would only make it fall back when the
        conversion proper started counting from nothing.
        """
        last = [0.0]

        def report(label, done, total):
            if self.cancel.is_set():
                return False
            now = time.monotonic()
            if not (done and total and now - last[0] < 0.3):
                last[0] = now
                share = f" {done}/{total}" if total else ""
                self.events.put(("stage", (index, f"{label}{share}",
                                           done / total if (done and total) else None)))
            # Held after the row has been brought up to date, so what it shows
            # through a pause is where the pass actually stopped; and asked
            # again afterwards, because Stop is what a pause often turns into.
            self._hold()
            return not self.cancel.is_set()

        return report

    def _progress(self, index):
        """Report a clip's frames back to the window, and carry the Stop button's
        answer back to the conversion."""
        warm, held = 0.0, 0.0

        def report(done, total, seconds):
            nonlocal warm, held
            if self.cancel.is_set():
                return False
            # The first frame carries the graphics driver's warm-up with it, and
            # charging the whole clip for it makes the opening estimate nonsense.
            if done == 1:
                warm = seconds
            # Time paused comes off the same way the warm-up does: neither was
            # spent converting, and either one left in makes the estimate wrong.
            spent = seconds - warm - held
            rate = (done - 1) / spent if done > 1 and spent > 0 else 0
            left = clock((total - done) / rate) if rate and total > done else ""
            self.events.put(("frame", (index, done, total, left)))
            held += self._hold()
            return not self.cancel.is_set()

        return report

    def _previewer(self, path):
        """Show the frame being worked on, now and then.

        The array has already been made for the encoder, so this costs the
        shrinking and nothing else -- and it is done here on the worker thread,
        leaving the window with an image it only has to draw.
        """
        last = 0.0

        def show(pixels):
            nonlocal last
            now = time.monotonic()
            if now - last < PREVIEW_EVERY:
                return
            last = now
            try:
                from PIL import Image

                image = Image.fromarray(pixels)
                image.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH // 2), Image.BILINEAR)
                self.events.put(("preview", (image, f"{path.name} - the frame being converted")))
            except Exception:  # a preview is a nicety; the encode carries on
                pass

        return show

    def _drain(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.config(text=payload)
            elif kind == "working":
                self._set_row(payload, "working", "converting...")
            elif kind == "error":
                # Collected rather than popped one dialog at a time, so a long
                # batch cannot bury the user in modal windows.
                index, message = payload
                self.errors.append(message)
                self.status.config(text=message)
                if index is not None:
                    # The row keeps the gist; the dialog at the end has it all.
                    reason = message.split(": ", 1)[-1]
                    self._set_row(index, "failed",
                                  reason if len(reason) <= 60 else reason[:57] + "...")
            elif kind == "notice":
                # On the status line as it happens, and kept for the summary --
                # a long batch would otherwise scroll its warnings past unread.
                self.notices.append(payload)
                self.status.config(text=payload)
            elif kind == "ask":
                oversize, reply = payload
                reply.put(self._oversize_dialog(oversize))
            elif kind == "stage":
                index, text, fraction = payload
                self._set_row(index, "working", text, fraction=fraction)
            elif kind == "frame":
                index, done, total, left = payload
                share = f"{done}/{total}" if total else f"{done}"
                self._set_row(index, "working",
                              f"frame {share}{f' - {left} left' if left else ''}",
                              fraction=done / total if total else None)
                # The bar runs across the whole queue, and a clip advances it a
                # fraction of a file at a time rather than jumping at the end.
                if total:
                    self._set_progress(self.finished + done / total)
            elif kind == "paused":
                # The worker has reached a frame boundary and stopped there,
                # which is a different thing from the button having been pressed.
                # What it gave back is worth saying: someone who paused to get
                # the card for something else wants to know they have it.
                gave = f" ({payload / 2 ** 30:.1f} GB of the card given back)" if payload else ""
                self.status.config(text=f"Paused - press Resume to carry on{gave}")
            elif kind == "resumed":
                self.status.config(text=self.paused_status or "Converting...")
            elif kind == "preview":
                from PIL import ImageTk

                image, caption = payload
                self._show_result(ImageTk.PhotoImage(image), caption)
            elif kind == "stopped":
                self.finished += 1
                self._set_progress(self.finished)
                self._set_row(payload, "stopped", "stopped")
            elif kind == "skipped":
                index, photo = payload
                self.finished += 1
                self._set_progress(self.finished)
                self._set_row(index, "skipped", "too large - skipped")
                self.status.config(text=f"Skipped {photo.name} - too large for memory")
            elif kind == "done":
                index, info = payload
                self.finished += 1
                self._set_progress(self.finished)
                width, height = info["output_size"]
                was = info["resized_from"]
                note = f"  (resized from {was[0]}x{was[1]})" if was else ""
                if "frames" in info:
                    detail = f"{width}x{height}, {info['frames']} frames in {clock(info['seconds'])}"
                    # How much of the sphere the surround reached, when it was
                    # asked for.  Worth a word in the row: a clip that panned has
                    # far more of its own scene to give than one that sat still,
                    # and this is the only place that difference shows.
                    if info.get("surround"):
                        detail += f", {info['surround']:.0%} of the sphere filled"
                else:
                    detail = f"{width}x{height} in {info['seconds']:.1f}s" + (" (resized)" if was else "")
                self._set_row(index, "done", detail)
                self.status.config(text=f"{info['output'].name}  -  {detail}{note}")
                # A clip has its own last frame on the mat already, so it only
                # needs the caption saying what that frame turned out to be; a
                # photo is not seen at all until it is written.
                # The line under the picture says what it is and how to look
                # at it, the two things the file itself cannot tell you; the
                # status line below already has the timings.
                order = "cross-eyed order" if self.cross_used else "left eye on the left"
                caption = f"{info['output'].name}  -  {width}x{height}, {order}"
                if "frames" in info:
                    self.caption.config(text=caption)
                else:
                    self._show(info["output"], caption)
            elif kind == "finished":
                self.convert_button.state(["!disabled"])
                self.running = False
                if self.cancel.is_set():
                    self.status.config(text="Stopped")
                self.cancel.clear()
                # A pause asked for after the last frame of the last file never
                # reached a `_hold`, and would otherwise still be sitting on the
                # button when the next run started.
                self.carry_on.set()
                self._refresh_controls()
                self._report(self.errors, self.notices)
        self.root.after(100, self._drain)

    @staticmethod
    def _report(errors, notices):
        """What a finished run has to say for itself.

        Warnings share the dialog with failures rather than getting one of their
        own: a run that quietly did less than it was asked to is the thing this
        is here to stop, and two dialogs in a row is how people learn to dismiss
        the first without reading it.
        """
        if not errors and not notices:
            return
        lines = list(errors)
        if notices:
            if lines:
                lines.append("")
            lines.append("Also worth knowing:")
            lines += [f"  - {note}" for note in notices]
        show = messagebox.showerror if errors else messagebox.showinfo
        show("StereoCraft", "\n".join(lines))

    def _oversize_dialog(self, oversize):
        """The modal question itself, on the main thread where Tk wants it."""
        if oversize.target is None:
            messagebox.showerror("StereoCraft", oversize.describe())
            return "skip"
        resize = messagebox.askyesno(
            "StereoCraft - photo too large",
            f"{oversize.describe()}\n\nResize it and convert, or skip this photo?",
            default="yes", icon="question")
        return "resize" if resize else "skip"

    def _show_result(self, image, caption):
        """Put a picture on the mat, in place of whatever it was saying."""
        self.preview = image  # Tk drops an image nothing is holding on to
        self.canvas.config(image=image, text="")
        self.caption.config(text=caption)

    def _clear_result(self, text=EMPTY):
        """Take the picture off it again, leaving something to read instead."""
        self.preview = None
        self.canvas.config(image="", text=text)
        self.caption.config(text=NO_CAPTION)

    def _show(self, path, caption):
        try:
            from PIL import Image, ImageTk

            with Image.open(path) as image:
                image.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH // 2), Image.LANCZOS)
                self._show_result(ImageTk.PhotoImage(image.copy()), caption)
        except Exception:  # a preview is a nicety; the file is already written
            pass


def main():
    logbook.start()
    if tk is None:
        print("The desktop window needs Tkinter, which this Python was built without.\n"
              "On Debian/Ubuntu: sudo apt install python3-tk", file=sys.stderr)
        return 1
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
