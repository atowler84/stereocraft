# StereoCraft

Turn a single photo — or a video — into a **side-by-side 3D** one.

One job, done well: estimate a depth map, re-project the picture into a left and
a right eye view, and write the pair as one frame you can drop straight into a
headset or view free-eyed. A photo keeps every pixel it arrived with; a clip is
squeezed to half a frame per eye, because that is what players will decode.

| Photo | Output | Time |
| --- | --- | --- |
| 1.9 MP | 3152 × 1197 | 0.5 s |
| 7.1 MP | 6040 × 2304 | 1.3 s |
| 12.5 MP | 6044 × 4080 | 1.5 s |

Measured on an RTX 4080 Super with the model already loaded; add about fifteen
seconds for the first photo of a session. The GPU path falls back to the CPU
automatically if a photo will not fit in video memory, and offers to resize a
photo that will not fit there either -- see [when a photo is too
big](#when-a-photo-is-too-big).

There is a CPU path too and it is a great deal slower, because most of the cost
is the depth network rather than the pixels: a snapshot costs nearly as much as
a raw. A 1.9 MP photo takes about 28 s on an eight-core Ryzen 7 7800X3D, against
half a second on the card. The `da2` models are quicker there -- 18 s for
`da2-large`, 6 s for `da2-base`, 2 s for `da2-small` -- but they measure nothing,
so the geometry goes back to being approximate. See [which
model](#which-model).

```
photo.jpg  ->  photo_sbs.jpg        (left | right, full width, no downscaling)
clip.mp4   ->  clip_sbs.mp4         (left | right, half width per eye, sound kept)
```

The name says what the file is, because nothing inside it does — players pick
the projection out of the file name and nowhere else:

```
_sbs             flat, left | right
_sbs_cross       flat, right | left, for cross-eyed free-viewing
_180_sbs         VR180, left | right          (--projection vr180)
_180_sbs_cross   VR180, right | left
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps depth-anything-3
pip install -e . --no-deps
```

The `--no-deps` on the third line matters. Depth Anything 3's own dependency list
replaces Torch with a different CUDA build and pulls in a reconstruction and
visualisation stack — open3d, pycolmap, moviepy, flask, jupyter — that depth
inference never touches. `requirements.txt` carries what it actually reaches for
at runtime instead, each entry checked by blocking its import and converting a
photo.

The desktop window additionally needs Tkinter, which some Python builds omit:

```bash
sudo apt install python3-tk
```

JPEG, PNG, HEIC, WebP, TIFF and BMP all work, so photos come straight off a
phone without converting anything first.

Video needs ffmpeg, which does the decoding and encoding either side of the
conversion. A copy sitting next to the app is used if there is one, so the
portable build can carry its own:

```bash
sudo apt install ffmpeg
```

Depth weights download themselves on first run (~1.3 GB for the large model) and
are cached. If a `hf-cache/` folder exists next to the app it is used instead of
`~/.cache/huggingface`, so an existing download is picked up automatically.

## A Windows app

`packaging/windows/build.ps1` freezes the lot -- Python, Torch, the weights --
into one folder that runs on a Windows machine with nothing installed on it.
Build it on Windows, with any Python from 3.10 to 3.14 (PyInstaller cannot
cross-compile, so this one step has to happen there). Depth Anything 3's wheel
declares a ceiling of 3.13, which was just the newest version when it was
published; the build installs it past that, and it runs on 3.14 unchanged.

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

It leaves a zip in `%USERPROFILE%\StereoCraft-build`. Unzip it anywhere -- a USB
stick is fine -- and double-click `StereoCraft.exe` for the window, or run
`StereoCraft-cli.exe --help` for the command line. Nothing is installed and
nothing is downloaded on first run: the weights ship in `models\da3` beside the
exe and ffmpeg sits next to it, so both photos and video work on a machine that
has never seen the internet.

The window has no console of its own, which on Windows means every ffmpeg it
starts would be given one -- a black box flashing up for the probe and two more
sitting there for the length of an encode. They are launched with
`CREATE_NO_WINDOW` instead, so a conversion happens in the window and nowhere
else.

| switch | |
| --- | --- |
| `-Cuda` | build against CUDA Torch: about three times the size, and the difference between 20 seconds a photo and a tenth of one |
| `-Models da3,da2-large` | which checkpoints to ship; `da3` alone by default |
| `-SkipFfmpeg` | leave ffmpeg out, and with it video on a machine that has none |
| `-Work <dir>` | where to build, `%USERPROFILE%\StereoCraft-build` by default |
| `-SkipZip` | leave the folder without packing it |
| `-Python <path>` | which `python.exe` to build with; found on its own otherwise |
| `-TorchIndex cu130` | a different CUDA build of Torch. The default `cu126` runs on any driver from 525 up, where `cu130` wants 580 or newer. All three of `cu126`, `cu128` and `cu130` publish wheels for Python 3.14 |

With `-SkipZip` the app is left in `dist-<flavour>\StereoCraft` under the build
folder, ready to copy wherever it is going to live. Move it somewhere of its own
before building again, because the next build overwrites that folder.

A build wants room to work in -- roughly 5 GB for the CPU one and 13 GB for
CUDA, most of it the environment being frozen. All of it is disposable
afterwards except `StereoCraft-build\models` and `StereoCraft-build\ffmpeg`,
worth keeping so that a rebuild does not fetch them all over again.

Depth Anything 3 made the folder bigger: OpenCV, torchvision, scipy and pandas
all have to come along now, on top of the 1.3 GB of weights. A CUDA build
measures 5.8 GB, against 5.4 GB before. The CPU one has not been rebuilt since
and its old figure of 1.9 GB will be low.

The first photo of a session pays for loading the model, and on CUDA the first
run on a new machine pays again while the driver builds its kernel cache.

One thing worth knowing about the frozen app: it runs with TorchScript turned
off. Depth Anything 3 puts `@torch.jit.script` on one small matrix helper, and
TorchScript compiles from source, which a frozen app does not carry -- so the
launcher disables the JIT before Torch is imported. The function then runs as
ordinary Python, which for a 4x4 inverse costs nothing measurable.

Windows has no way to know an unsigned exe, so the first run brings up a
SmartScreen box -- More info, then Run anyway -- and only a signing certificate
makes that go away.

## Use it

```bash
stereocraft photo.jpg
```

```bash
stereocraft ~/Pictures/holiday --output ~/Pictures/3d
```

```bash
stereocraft clip.mp4
```

```bash
stereocraft-gui
```

Photos and clips go in the same run and the same window queue, and the depth
model is loaded once for the lot. The first photo pays the two second load and
the rest convert in well under a second each.

The window is the queue down the left, the settings and the finished pair on the
right. Each file in the queue is a row of its own with a progress bar and a
percentage, which is what a clip wants -- it is the only thing here that takes
minutes -- and underneath it the frame it is on and how long is left. The bar
across the bottom counts the whole queue, in files and in percent; a photo has
no frames to count, so its own bar moves rather than inventing a figure.

The settings are three tabs -- General for what the scene looks like, Photo and
Video for what each is written as -- and the tab that governs the first file
added comes up with it. Each tab has its own Reset to default, which puts that
tab back and leaves the others alone, and goes grey once there is nothing left
on it to undo.

What is not on the window is what there is no judgement in: the model, the
resolution it runs at, the comfort limit and which device does the work are all
left where they belong, and a photo too big for memory is asked about rather
than settled in advance.

## When a photo is too big

Every conversion is sized up before the photo is even decoded, against what the
machine actually has free at that moment. A photo that will not fit in video
memory moves to the CPU, which usually has far more room. Only when it will not
fit there either is there a decision to make, and the app puts it to you rather
than guessing:

```
holiday.jpg is 11648x8736 (101.8 MP) and needs about 13.0 GB, but only 6.2 GB of memory is free.
Resizing to 7409x5556 (41.2 MP) would fit -- 64% of the width, 40% of the pixels.
The side-by-side image would come out about 14370x5556 instead of 22596x8736.
Resize and convert it, or skip it? [r/s]
```

The size offered is the largest one predicted to fit, so the detail given up is
the least that gets the photo converted. The window asks the same question in a
dialog. Nothing is ever downscaled without an answer: `--oversize skip` and a
non-interactive run both leave the photo alone and say so, and `--oversize
resize` takes the offer every time, for scripts that would rather have a
slightly smaller 3D photo than none.

The estimate is not calibrated to any particular machine. What a conversion
costs is worked out from the model you loaded, the precision it runs at and the
resolution it will run at, so a `small` model on a 4 GB card is judged on what
it actually needs rather than on what `large` would have needed. Free memory is
read from the machine itself -- the driver on CUDA and Apple silicon, the kernel
on Linux, Windows and macOS -- and both devices are priced, since a generous card
in a busy machine can have more to spare than the system does. When the depth
model alone is what does not fit, no resize can help, and it says which lighter
model would:

```
No smaller size would fit either: the depth model needs that much whatever size the photo is.
The base depth model would fit, and costs little in quality (--model base).
```

## Video

```bash
stereocraft clip.mp4
```

Every frame gets exactly what a photo gets. Two things are added around it, and
both come from the picture moving rather than from anything about video files.

### The defaults are gentler

A clip aims for 1.3% of frame width where a photo aims for 2.0%, and pins the
depth network rather than letting it follow the frame.

Any error in the depth map becomes a horizontal position error in proportion to
the separation. In a still that is a silhouette a pixel out of place and nobody
sees it; in a clip it is an edge that shimmers and everybody does. The gaps the
warp opens up scale with it too, and a filled gap that reads as plausible while
it holds still crawls once the edge it belongs to moves. And a photo gets a
glance where a clip gets several minutes, so what is merely noticeable becomes
tiring.

`--target` overrides it in either direction, and is worth a try on your own
footage: 1.3 is a reasoned starting point, not a measured one.

The focus distance is left to `auto` exactly as it is for a photo. It puts most
of the scene behind the window with only a near subject in front of it, which is
the arrangement that stays comfortable — things poking out of the screen are what
break at the frame edge, and motion makes that worse rather than better.

### Depth is held still between frames

Depth Anything is a per-frame model, and a per-frame estimate wobbles. In a depth
map that reads as noise; turned into a stereo pair it is the *geometry* that
wobbles, which is a great deal harder to look at. `--temporal` carries some of
each frame's depth into the next, and a frame that differs wholesale from the one
before it is treated as a cut, where the memory starts again.

This section used to describe something more elaborate, and the honest version is
shorter. Depth-Anything V2 had to have its percentile range smoothed over time as
well, because re-measuring it every frame made the whole map slide about. Metric
depth needs no range at all — metres are metres whatever else is in the frame.

But that did not make things quieter, which is worth saying plainly because the
opposite was expected. Renormalising every frame had also been cancelling the
model's own scale wobble, and measured on a static shot with sensor noise, metric
depth is about a third *noisier* frame to frame. Smoothing the metric scale back
out was tried and made it worse still: the noise is spread through the map rather
than sitting in one global factor.

It does not matter, which is why only the plain average is left. In the units
that count — how far the disparity field actually moves between frames — both
models sit near a tenth of a pixel before any smoothing at all, against roughly a
third of a pixel for the smallest movement an eye can pick out:

| | no smoothing | `--temporal 0.5` |
| --- | --- | --- |
| Depth Anything 3 | 0.14 px | 0.07 px |
| Depth-Anything V2 | 0.11 px | 0.05 px |

Smoothing still costs a little edge sharpness, so `--temporal 0` declines it.

### Half a frame per eye

Unlike a photo, a clip does not come out at native width. Each eye is squeezed to
half the frame, so 1080p in is 1920×1080 out rather than 3792×1080.

That is what players and headsets expect, and more to the point what their
hardware decoders will take — 4K doubled is 7616 px wide, past the level h.264
defines and past what most headsets will decode at all. `--full` keeps every
native pixel for a player known to handle it, and `--codec hevc` is worth pairing
with it.

The soundtrack comes across untouched wherever the container will take it as it
stands, and is re-encoded to AAC only where it would otherwise be refused.
`--no-audio` leaves it behind.

### Which encoder does the writing

Not always the same one. `--codec` picks the format; what actually encodes it
depends on how the ffmpeg to hand was built, because x264 and x265 are GPL and
an LGPL ffmpeg carries neither. The order tried is:

1. **libx264 / libx265** — the best of them at a given file size
2. **NVENC**, then QSV, then AMF — the graphics card, if there is one
3. **libopenh264**, then MediaFoundation — software, and not GPL

Whichever it lands on, `--crf` still means quality, though the encoders spell it
differently underneath (`-crf`, `-cq`, `-global_quality`). Falling past x264 is
said out loud, since a clip that comes out softer than the last one should not
look like the app's doing:

```
clip_sbs.mp4: encoding with h264_nvenc; this ffmpeg has no libx264
```

To ask a finished file what wrote it:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream_tags=encoder -of csv=p=0 clip_sbs.mp4
```

The Windows build ships a GPL ffmpeg, so x264 is there and this rarely comes up.
It matters on a machine whose ffmpeg was built without it.

### How long it takes

| Clip | Per frame | Per minute of footage |
| --- | --- | --- |
| 1280 × 720 | 222 ms | 6.7 min |
| 1920 × 1080 | 238 ms | 7.1 min |
| 3840 × 2160 | 369 ms | 11.1 min |

On an RTX 4080 Super with the model already loaded. Depth Anything 3 costs a good
deal more than the model this used to use, and is asked for more resolution as
well, so a minute of footage is the better part of ten minutes of work.

The CPU is not really in the running: one 1080p frame takes about 11 s, which is
five and a half hours for a minute of footage. `--model da2-small` is the only
combination that finishes in an evening, and gives up the metric geometry to do
it. A clip that is going to take more than ten minutes says so before it starts
rather than letting you find out:

```
clip.mp4: this is a CPU conversion -- about 6s a frame, so 11m51s for 120 frames.
```

Anything that long is worth being able to watch and to stop. The command line
rewrites a line with the frame count and an estimate; the window shows the frame
it is working on and counts the queue down. Either can be stopped part-way, and
neither leaves a half-written file behind.

## Which settings?

None of them, most of the time: press Convert.

It used to be worth explaining why there was no correct setting to hunt for. A
photo carries no scale, the argument went, so the depth map says only what is
nearer than what and no combination of settings could reconstruct the real
geometry. That is no longer true, and the whole of this section is different
because of it.

The depth model measures in **metres**. So the renderer does not approximate
the separation between your eyes -- it works it out. Two eyes a real distance
apart, both looking at a screen a real distance away, see a point at distance Z
separated by

```
d = f · B · (1/Z − 1/Zc)
```

and that is what gets rendered. Things at the focus distance sit in the screen,
nearer things come out of it, and everything beyond recedes towards a **finite**
separation rather than being stretched out by however the depth map happened to
be scaled. That finite limit is the difference between geometry and a good guess,
and it is what makes a distant background sit properly behind the frame instead
of pulling apart.

### What auto does, and why it is not just the human number

The two settings are the eye separation and the focus distance, and by default
both are chosen per photo. It is worth knowing why, because "just use 65mm, the
human number" is the obvious answer and it is measurably wrong:

| Scene | Separation a real 65mm pair would need |
| --- | --- |
| a close-up 0.2–1.0 m away | 15.8% of frame width |
| a car 1.9–24.8 m away | 2.4% |
| a telephoto shot 19.4–22.5 m away | 0.3% |

Around 2% is comfortable. So literal eyes render a close-up no one can fuse and
a telephoto shot that is nearly flat. Both are *correct* -- that really is what
someone standing at the camera would see -- but you are not standing at the
camera, you are looking at a screen.

So the separation is chosen to suit the scene, which is what a stereographer
does rather than a fudge: a wider baseline than human for a distant landscape, a
narrower one for a close-up. On the nine photos above it picks 9mm for a macro,
56mm for a car at conversational distance, and 618mm for the telephoto shot, and
lands every one of them between 1.3% and 2.0%.

What matters is that only the amplitude moves. The *shape* stays exactly what
the metric geometry says -- parallax falling off as 1/Z, distant things
converging -- and that shape is the whole gain. Scaling it is a choice about how
big you want to feel, not an approximation.

- **Eye separation** (`--eyes`) — millimetres. `auto`, or a number: 65 for real
  human eyes, more for a landscape you want depth out of, less for a close-up.
- **Focus distance** (`--focus`) — metres. `auto`, or the distance you want
  sitting in the plane of the screen.
- **Target** (`--target`) — what `auto` aims for, as a percentage of frame width.
  The one to reach for if the whole thing is too strong or too flat: it keeps the
  geometry and changes only how much of it there is.

Whatever it settles on is printed alongside the output, because there is no
guessing it otherwise:

```
photo_sbs.jpg  2008x768  56mm@2.1m  0.2s
```

That is 56mm eyes with the screen plane at 2.1m — a car at conversational
distance. A macro comes out nearer 9mm, a landscape several hundred. It is the
number to start from when you want to pin one yourself.

### Which model

`--model da3` is Depth Anything 3, and it is the only one that measures in
metres. `da2-large`, `da2-base` and `da2-small` are Depth-Anything V2, kept as a
fallback: they rank depth without measuring it, so their output is stretched onto
an assumed 1–50 m range and the geometry that comes out is approximate. Worth
reaching for if DA3 makes a mess of a particular photo, which does happen -- DA3's
own 1.4B variant was tested for this app and rejected because it falls apart on
portraits.

They are close on depth quality. On two test photos the edge-alignment scores
were 0.284 against 0.284 and 0.129 against 0.136 -- a wash. The reason to prefer
DA3 is the metres, not a sharper map.

### Where the metres come from

The conversion needs the lens. It is taken from the photo's EXIF where that
survives, and otherwise assumed to be a 28mm-equivalent, which is what most
phones point at the world.

Getting it wrong matters less than it sounds. A wrong focal length scales the
whole scene by the same factor, and `auto` then picks a baseline that cancels it
-- so the picture is unchanged and only the metre readings are off. It is worth
knowing before trusting a `--save-depth` map as a measurement.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `-e`, `--eyes` | `auto` | Distance between the two eyes, in millimetres, or `auto` to size it to the scene. 65 is the human average; a landscape wants far more and a close-up far less. |
| `-f`, `--focus` | `auto` | How far away the screen plane sits, in metres, or `auto`. Whatever is at this distance sits in the screen; nearer comes out, further recedes. |
| `-t`, `--target` | `2.0` photo, `1.3` video | What `auto` aims for: near-to-far separation as a percentage of frame width. The knob to reach for when the effect is too strong or too flat. |
| `--limit` | `3.0` | Ceiling on separation, as a percentage of frame width, so something very close cannot demand more parallax than an eye can fuse. |
| `-m`, `--model` | `da3` | `da3` measures depth in metres. `da2-large`, `da2-base`, `da2-small` only rank it and are fitted onto an assumed range — a fallback, see [which model](#which-model). |
| `--depth-size` | `auto` photo, `1400` video | Longest side fed to the depth network (shortest, for the `da2` models). `auto` follows the photo up to 2048 px; bigger gives cleaner subject silhouettes, which is what the warp cares about. A clip is pinned, since the finer structure is what the temporal smoothing then averages away. |
| `--cross` | off | Write right\|left for cross-eyed viewing instead of left\|right. Not a quality setting: it only matters when free-viewing on a monitor. Named `_sbs_cross`, because shown to a headset as an ordinary pair it puts each eye on the other one's view. |
| `--max-size` | `0` | Cap the output width. Native by default; useful if a viewer chokes on very wide images. |
| `--format`, `-q` | `auto`, `95` | Output container and JPEG quality. |
| `--save-depth` | off | Also write a 16-bit `_depth.png`, near white and far black, scaled across its own range so it can be looked at. The two distances it scaled by go in the PNG's metadata as `stereocraft:near_m` and `stereocraft:far_m`, so `metres = far - (value / 65535) * (far - near)` gets them back. Read [where the metres come from](#where-the-metres-come-from) before trusting them. |
| `--projection` | `flat` | `flat` writes a rectilinear pair, shown on a virtual screen. `vr180` wraps the same geometry onto a hemisphere at its true angular scale — see [VR180](#vr180). |
| `--vr180-size` | `auto` | Stored width per eye for `--projection vr180`. `auto` keeps as much of the source's own detail as fits, capped at 4096 (2048 for a clip). |
| `--vr180-surround` | off | Fill the part of the sphere the picture never reached with a dim, blurred spread of it, rather than leaving it black. A fixed function of each frame, so a clip cannot crawl with it. |
| `--device` | `auto` | `cuda`, `mps` or `cpu`. |
| `--oversize` | `ask` | A photo too big for memory: `ask` what to do, `skip` it, or `resize` it to the largest size that fits. |

Video only:

| Flag | Default | What it does |
| --- | --- | --- |
| `--temporal` | `0.5` | How much of the previous frame's depth to carry over, 0 to 0.95. Steadies a clip that shimmers, at a little edge sharpness; `0` turns it off. |
| `--full` | off | Keep every native pixel, doubling the frame width, instead of squeezing each eye to half width. Needs a player that will decode it. |
| `--codec` | `h264` | `hevc` is worth it above 4K, where h264 runs out of level. Which encoder produces it depends on the ffmpeg to hand — see [which encoder does the writing](#which-encoder-does-the-writing). |
| `--crf` | `18` | Encoder quality; lower is better and larger. Means the same thing whichever encoder runs, though they spell it differently underneath. |
| `--no-audio` | off | Leave the soundtrack behind rather than carrying it across. |

As a library:

```python
import stereocraft
stereocraft.convert("photo.jpg", eyes_mm=65, focus_m=3)
stereocraft.convert_video("clip.mp4", target_pct=1.0)
stereocraft.convert("photo.jpg", projection="vr180")
```

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

That is 89 of them in under two seconds — the geometry against the formula it is
supposed to implement, the automatically chosen baseline landing on target from a
0.2 m close-up to a 20 m telephoto, what the memory budget will and will not
offer, argument handling, and the video plumbing. None of it needs the depth
model.

```bash
pytest
```

runs the other nine as well, which convert a real photo and a real clip end to
end and take about half a minute, most of it loading the model.

They exist because two bugs got past careful manual checking in a single week.
`-shortest` quietly trimmed three frames off a ninety-frame clip and survived a
verification pass that counted frame *sizes* rather than frames. `--save-depth`
started writing centimetres, which is correct and looks like a black rectangle,
and survived because nothing asserted the map was legible. Both now have a test
named after what went wrong, and the second was checked by putting the bug back
and watching three of them fail.

## How it works

1. **Depth** — Depth Anything 3 predicts depth in metres, converted to inverse
   depth (1/Z) because that is the quantity that behaves: it varies linearly
   across a slanted surface where depth itself does not, so the upsample below
   interpolates it correctly, and it is what the disparity formula wants anyway.
2. **Edge alignment** — the network runs at its own resolution and the result is
   lifted to full resolution with a guided filter that uses the photo as the
   guide. Depth edges land on picture edges instead of the soft ramps plain
   interpolation leaves, and that is what keeps silhouettes clean in the warp.
3. **Geometry** — two eyes are placed a real distance apart, aimed at a real
   focus distance, and each pixel's separation comes out as `f·B·(1/Z − 1/Zc)`.
   Both distances are chosen to suit the scene unless you pin them.
4. **Rendering** — each pixel is splatted into the column its disparity puts it
   in, with a z-buffer so nearer surfaces occlude, then resampled backwards with
   bilinear weights to recover sub-pixel detail.
5. **Disocclusions** — a small baseline only uncovers a few pixels of hidden
   background per edge. Those gaps are filled from whichever side is further
   away, with the background stretched gently across so there is no flat smear
   and no ghost of the foreground edge.
6. **Framing** — both eyes shift content sideways, leaving a sliver at the frame
   edge no real pixel reaches. It is trimmed rather than invented, costing about
   1% of the width. A clip pins that trim to its settings rather than measuring
   it per frame, since frames that changed size part-way through are not
   something any encoder will take.
7. **Time**, for a clip only — the percentile range and then the depth map are
   carried forward from frame to frame, so the geometry stops wobbling. See
   [Video](#video).

There is no mesh, no inpainting network and no OpenGL context: the whole render
is a handful of tensor ops, which is why it runs at native resolution rather than
the 768px ceiling a mesh pipeline imposes.

## VR180

`--projection vr180` writes the same stereo geometry wrapped onto a 180-degree
hemisphere instead of laid on a plane. The difference in a headset is real: a
flat pair is a screen hanging in front of you at whatever size the player feels
like, where VR180 puts the picture at its **true angular scale**, so something
that subtended thirty degrees to the camera subtends thirty degrees to you.

It is also mostly black, and that is not a bug to be fixed later.

```
stereocraft --projection vr180 photo.jpg
photo_180_sbs.jpg  3536x1768  4mm@0.3m  25% of a sphere (28mm assumed)
```

The `25% of a sphere` is how much of the hemisphere the photograph reaches, by
solid angle. A 28mm phone lens covers 65 by 51 degrees, which is 15% of what you
can turn your head and look at; a 16mm ultrawide manages 30%; a 49mm lens 5%. No
amount of projection will invent the rest, so **only the part that exists is
stored**, and the file says where on the sphere it belongs — GPano for a photo,
projection bounds for a clip. The edge is faded rather than cut, an honest
absence reading better than a hard-edged rectangle floating in a void.

**Why the dark stays.** Storing only the piece the lens reached was built,
measured, and taken out again. It saved four fifths of the pixels and recorded
where the piece belonged, in the fields both formats provide for exactly that —
and no player reads them. Skybox assumes every eye is a full 180 by 180, so a
65-by-91-degree patch handed to it came out **2.7× too close and 40% stretched
sideways**: sharp, convincing and wrong. The square needs nothing read to be
right, so the square is what gets written. It is in the history if a player ever
catches up.

| Lens | Field of view | Real | Invented |
| --- | --- | --- | --- |
| 28mm (most phones) | 65.5° × 51.5° | 15% | 85% |
| 24mm | 73.7° × 53.1° | 17% | 83% |
| 16mm ultrawide | 96.7° × 73.7° | 30% | 70% |
| 13mm ultrawide | 108.3° × 92.2° | 40% | 60% |

**The dark can be lit instead.** `--vr180-surround` fills it with a dim,
blurred spread of the picture — what a social video site puts behind a clip that
does not fill the frame. It reads as the light coming off the picture rather
than as a second, blurrier photograph, and it makes an enormous difference to
how a 15%-covered frame feels to sit inside.

It earns its place for the reason an outpainting model does not: it is a **fixed
function of the frame**, so it moves exactly as the picture moves and cannot
crawl or boil between frames. Nothing is invented that was not already on
screen. Two things differ from the rectangular case, both because this is a
sphere — the wash is spread outward from the edge rather than scaled up from the
middle, so what sits beside you is the colour the camera saw *in that
direction*; and both eyes get the same wash, because a periphery with a parallax
of its own would fight the real picture over where your eyes should converge.

It costs nothing measurable in time, and about 0.03 MB in size — a smooth blur
is almost free to compress. The reported coverage does not move, because it
counts what the camera saw and not what was painted in.

**Resolution.** The square is sized to keep the photograph's own detail where
it can, which for a 65-degree lens means a side nearly three times the source
width, capped at 4096 for a photo and 2048 for a clip. Past the cap the picture
is used softer than it arrived: a 49mm photo at 4096 comes back 918 pixels wide
with fifteen megapixels of black around it. `--vr180-size` sets the stored width
by hand.

**It is a different question from the flat path, so it is asked differently.**
`--target` and `--limit` are percentages of frame width, and a percentage of a
180-degree frame is not a quantity anyone's comfort is described in — 2% of it is
3.6 degrees of parallax, several times what an eye can fuse. So the spherical
path aims at 0.6 degrees of near-to-far separation and caps at 1.2, in `vr180.py`.
`--eyes` and `--focus` still mean what they always did.

**The lens stops being a detail.** [Where the metres come
from](#where-the-metres-come-from) notes that a focal length wrong by some factor
only rescales the scene, and the focus distance absorbs it. That is true while
the picture stays flat and false the moment it goes on a sphere: the focal length
is precisely what decides where each pixel lands *in angle*, so a wrong one puts
the whole picture at the wrong apparent size — convincing, and not life-sized.
EXIF supplies it where a photo still has one. Where it does not, the output says
`(28mm assumed)`, because a guessed angular scale is worth admitting to. The
metric DA3 checkpoint reports no intrinsics of its own, so a photo stripped of
its EXIF by a download or a screenshot is guessed at.

**What the file says about itself.** A photo carries GPano XMP, whose
`CroppedArea*` and `FullPano*` fields exist precisely to say "a piece of a
panorama, and here is where". A clip carries the Spherical Video V2 boxes:
`st3d` for the side-by-side pair and `sv3d/proj/equi` for the projection and its
bounds — which is not a VR180 special case at all, a VR180 file being a
360-degree equirectangular one whose bounds crop it to the middle 180. ffmpeg
writes neither, so the boxes are spliced in afterwards and every chunk offset in
the file moved to match.

None of which any player has yet been observed to act on beyond the projection
itself, which is why nothing tighter than the format's own hemisphere is written.
`--cross` writes stereo mode 4, right-left, which ffmpeg itself discards —
a clip that goes unlabelled asks a question, where one labelled left-right would
confidently tell a headset to swap the viewer's eyes.

One consequence worth knowing: a VR180 clip carries `st3d`, so a **desktop**
player that reads it will show a single eye rather than the side-by-side pair.
That is the metadata working. Use `--projection flat` for 2D spot checks.

**The one thing it still does not do** is invent the periphery, which is the
honest state of the art: a 512-tall diffusion model filling 85% of the frame,
hallucinated depth behind hallucinated colour, on an app whose whole argument is
that its geometry is measured rather than guessed — and on a clip, boiling
differently in every frame. `--vr180-surround` is the part of that idea worth
having: it lights the dark without pretending to know what was in it.

Video takes the flag too. The frame has to be settled before the first one is
decoded — every frame must come out the size of the first — and no clip carries
a focal length, so it assumes the same 28mm `depth` falls back on. A moving
picture is still the weaker case: the square costs the pixels and the missing
periphery is missing either way. `--projection flat` remains the default
everywhere.

## Viewing

A photo comes out full-width SBS: each eye keeps its own full width, so the file
is twice as wide as the source. A clip comes out half-width per eye, at the size
it went in, which is the arrangement players and their hardware decoders expect —
`--full` overrides that.

Quest, Pico and Vision Pro read both directly through any local media viewer; on
a desktop, free-viewing works with the parallel method, or use `--cross` and
cross your eyes.

Nothing in an mp4 or a JPEG announces how it is meant to be looked at, so
players go by the file name — which is why each kind of output carries its own.
`_sbs` and `_180_sbs` are both widely recognised, the second carrying the two
tokens players key on separately: `180` sets the projection, `sbs` the layout. A
player that still guesses wrong has a setting for it.

Those names are also how a second run over the same folder knows to leave its own
output alone, so renaming a file back to something plain will get it converted
again.

## License

MIT, see [LICENSE](LICENSE).

StereoCraft is MIT. Everything it depends on is Apache-2.0, BSD or MIT, with two
exceptions worth knowing about if this folder is ever handed to anyone: the
bundled **ffmpeg is the GPL build**, chosen for x264 and x265, and **pillow-heif
bundles libx265** in its wheel whether or not anything writes HEIC. The default
depth weights are Apache-2.0; the `da2-base` and `da2-large` fallbacks are
CC BY-NC 4.0 and not licensed for commercial use.

For using it yourself none of that matters, and none of it is legal advice.

This repository began as a fork of
[3D-Photo-Inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting) by way of
[Spatial-Photo](https://github.com/fake-oskars/Spatial-Photo). None of that code
remains; their notices stay with the commits that carried it.
