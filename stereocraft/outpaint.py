"""Widening a frame before it goes on the sphere, rather than after.

`plate` gathers the scene a shot was filmed in and hands the rest of the sphere
to a model.  This is that model, and the geometry it is asked to work in.

**Why the widening happens flat.**  The obvious place to fill the empty part of
the sphere is on the sphere, and it is the wrong place.  Every model worth using
here learnt from photographs, and an equirectangular frame is not one: straight
lines bend in it, and they bend more the further from the centre you look --
which is exactly where the model is being asked to invent.  So the plate is
sampled back onto an ordinary perspective canvas first, widened there, and
projected onto the sphere afterwards by the same `vr180.project` the live picture
goes through.  The model only ever sees a photograph with a hole around it.

**Per axis, because portrait video is the case that matters.**  A 28mm lens on a
720x1280 frame already covers 97.6 degrees vertically and only 65.5 across.  A
square canvas would crop the source rather than extend it, and what is actually
missing from the hemisphere is mostly sideways.  So the reach is added to each
axis separately and the canvas comes out the shape that implies.

**Why one pass and not several.**  Expanding in stages sounds safer and measured
worse: against a scene with the answer held out, one feathered pass scored a
facet ratio of 8.8 where three staged passes scored 10.3, and it landed closer to
what was really there -- 36.1 against 47.1 in Lab.  Each stage also leaves its own
boundary behind, so the staged version had three of them.

**What the mask has to look like.**  The first attempt handed the model a hard
rectangular hole and got back a picture frame: wooden slats, mitred corners, an
ornament rather than a room.  That is what a hard rectangle asks for.  Feathering
the mask and denoising from a stretched, blurred copy of the picture rather than
from a void is what turns it into a continuation, and it is the difference
between the two results rather than a refinement of one.
"""

import math

import torch
import torch.nn.functional as F

from . import vr180

# Past this the perspective stretch at the edge outruns anything a model renders
# usefully -- at 140 degrees the corner is sampled at three times the scale of
# the centre, and beyond it the canvas is mostly rubber.
MAX_FOV = 140.0
# Degrees of sphere to reach for beyond the source's own field of view.  A knob
# rather than a constant because the right amount is a matter of taste and of
# the footage: more reach fills more of the sphere and gives the model less to
# go on, and where that trade lands is the viewer's call.  See `Settings`.
REACH_DEG = 40.0
# Long side of the canvas handed to the model.  1024 is what the diffusion
# backends were trained at; asking for much more costs quality rather than
# buying it, and the periphery is upsampled on the way to the frame anyway.
CANVAS_PIXELS = 1024
# Both diffusion backends want a multiple of 16, and LaMa a multiple of 8.
ALIGN = 16
# How soft the mask edge is, as a share of the canvas.  The number that stops a
# model decorating a rectangle instead of continuing a room.
FEATHER_SHARE = 1.0 / 22.0
# How far past the picture's edge the subject may be carried, in degrees.  The
# same width `vr180.FALLOFF_DEG` fades the picture out over, and for the same
# reason: what is invented there is already half transparent by the time anyone
# sees it.
FADE_DEG = 4.0
# How much of the picture has to be room before the second pass is worth making.
#
# Measured rather than chosen.  On the footage this pass was built for the split
# came out 52% room to 48% subject, and at that ratio the second pass is not a
# small improvement on the first -- it is worse: what is left to condition it on,
# once the subject is grown over, is mostly smear, and a model given a smear
# returns a muddy brown texture with no room in it.  The first pass on the same
# frame returns bedding, a headboard and a wall.
#
# So the number is set above where it failed rather than below.  What that means
# in practice is that the second pass runs on a wider shot, where there is a room
# to show it, and stands aside on a close-up, where there is not.  It is the
# close-up this whole feature is for, so it stands aside more often than not --
# which is the honest arrangement rather than a disappointing one.
ROOM_SHARE = 0.7
# And how heavily the seed is blurred, likewise.
SEED_BLUR_SHARE = 1.0 / 24.0

# What the model is asked for, and it is asked twice for two different things.
#
# The first pass continues the picture as it stands, so its prompt is about the
# place carrying on outward.  The second is the one that ends up further out than
# the fade, and it must contain nothing that moves -- so it asks for the room
# *empty*.  That distinction is not decoration.  Given the general prompt and a
# bedroom to extend, the model helpfully supplies more people: a second head
# above the pillow, a spare pair of legs to the right.  In a still that is merely
# odd; in a plate built once for a whole shot it is a person who will never move
# again, sitting beside one who does.
#
# FLUX Fill takes no negative prompt, which is why this has to be said in the
# positive one rather than forbidden in a negative.
PROMPT = ("photograph of the rest of the room continuing outward, walls, floor, "
          "ceiling, bedding, furniture, soft indoor lighting, one continuous "
          "scene, photorealistic")
ROOM_PROMPT = ("photograph of an empty unoccupied room, nobody present, no people, "
               "bare walls, floor, ceiling, empty bed, plain bedding, furniture, "
               "soft indoor lighting, one continuous scene, photorealistic")
# SDXL takes a negative prompt; FLUX Fill has no use for one.  "picture frame"
# leads it because that is the failure a rectangular hole invites.
NEGATIVE = ("picture frame, border, ornate frame, wooden slats, collage, tiled, "
            "text, watermark, extra people, duplicated person, extra limbs, "
            "person, body, cluttered, busy, banding, flat colour bands")
# Steps and guidance per backend, measured rather than copied: these are what
# produced the results in the module docstring.
STEPS = {"flux": 30, "sdxl": 40}
GUIDANCE = {"flux": 30.0, "sdxl": 6.0}

# What each backend is called, in the order `auto` prefers them.  Quality first:
# the diffusion models invent a room where LaMa continues a texture, and on an
# indoor close-up -- which is what this whole pass is for -- that is the whole
# difference.  LaMa stays because it is 200 MB against 7 GB and needs no prompt.
PREFERRED = ("flux", "sdxl", "lama")


def source_fov(width, height, focal_px):
    """The angle the source covers, per axis, in degrees."""
    return vr180.fov(focal_px, width), vr180.fov(focal_px, height)


def canvas_fov(width, height, focal_px, reach_deg=REACH_DEG):
    """The source's own field of view plus `reach_deg`, per axis and capped."""
    across, down = source_fov(width, height, focal_px)
    return min(across + reach_deg, MAX_FOV), min(down + reach_deg, MAX_FOV)


def canvas_size(fov_x, fov_y, pixels=CANVAS_PIXELS, align=ALIGN):
    """A canvas about `pixels` on its long side, at the aspect those two fields
    of view imply and rounded to something every backend will accept."""
    across = math.tan(math.radians(fov_x) / 2.0)
    down = math.tan(math.radians(fov_y) / 2.0)
    if across >= down:
        width, height = pixels, pixels * down / across
    else:
        width, height = pixels * across / down, pixels
    return (max(align, int(width) - int(width) % align),
            max(align, int(height) - int(height) % align))


def to_perspective(colour, covered, fov_x, fov_y, size, rotation=None):
    """Sample a full-sphere plate onto a perspective canvas.

    Returns `(canvas, known, focal_px)` -- the picture, which of it came from the
    plate rather than from nowhere, and the focal length that puts it back.

    `rotation` points the canvas somewhere other than straight down the plate's
    own axis, which is what lets the sphere be filled a face at a time: one
    canvas cannot cover a hemisphere, perspective going to infinity at 180
    degrees, so the rest of it is reached by turning and asking again.
    """
    width, height = size
    focal = (width / 2.0) / math.tan(math.radians(fov_x) / 2.0)
    xs = (torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0)[None, :]
    ys = -(torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0)[:, None]
    xs, ys = xs.expand(height, width), ys.expand(height, width)
    zs = torch.full_like(xs, focal)
    norm = torch.sqrt(xs * xs + ys * ys + zs * zs)
    x, y, z = xs / norm, ys / norm, zs / norm

    # The plate is a whole sphere, so azimuth wraps and every direction lands
    # somewhere; `border` only matters at the seam directly behind the viewer.
    if rotation is not None:
        x, y, z = vr180.turn(x, y, z, torch.as_tensor(rotation, dtype=torch.float32))
    grid = torch.stack((torch.atan2(x, z) / math.pi,
                        -torch.asin(y.clamp(-1, 1)) / (math.pi / 2)), dim=-1)[None]
    out = F.grid_sample(colour[None], grid, mode="bilinear",
                        padding_mode="border", align_corners=False)[0]
    seen = F.grid_sample(covered.float()[None, None], grid, mode="bilinear",
                         padding_mode="border", align_corners=False)[0, 0]
    return out, seen > 0.999, focal


def seed(canvas, known):
    """What the model denoises from: the picture stretched over the whole canvas
    and blurred, with the real pixels put back on top.

    *Stretched*, not blurred in place, and the difference is the whole result.
    A blur of the canvas is the picture with a halo round it -- still a rectangle
    of scene in a field of something else, which is the arrangement that gets a
    picture frame painted round it.  Scaling the picture up to fill the canvas
    first means every pixel the model starts from is already the colour and
    brightness of somewhere in the room, and what it has to do is turn a vague
    room into a sharp one rather than invent a border.
    """
    from . import stereo

    height, width = canvas.shape[-2:]
    rows = torch.nonzero(known.any(1)).flatten()
    cols = torch.nonzero(known.any(0)).flatten()
    if not len(rows) or not len(cols):
        return canvas
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    spread = F.interpolate(canvas[None, :, top:bottom, left:right], size=(height, width),
                           mode="bilinear", align_corners=False)
    radius = max(1, int(max(width, height) * SEED_BLUR_SHARE))
    spread = stereo._box(spread, radius)[0]
    return torch.where(known[None], canvas, spread)


def feather(known, size):
    """The mask, softened.  A hard rectangle is a thing to decorate; a soft edge
    is a picture to carry on.

    Boxed twice rather than once: one box blur is a linear ramp with a corner at
    each end, and the corner is a straight line for a model to find.  Two is
    close enough to a Gaussian that there is nothing left to find.
    """
    from . import stereo

    width, height = size
    radius = max(1, int(max(width, height) * FEATHER_SHARE / 2))
    soft = (~known).float()[None, None]
    soft = stereo._box(stereo._box(soft, radius), radius)[0, 0]
    return soft.clamp(0, 1)


def without(canvas, known, keep):
    """The picture with whatever is not `keep` grown over by what is.

    For the pass that must not contain anyone: marking the subject as a hole to
    be filled does not work, and it fails for a reason worth writing down.  An
    inpainting model weighs the picture far above the prompt, and a person-shaped
    hole in a bed is a request for a person however firmly the words say "empty,
    nobody" -- measured twice, once with the mask wrong and once with it right.

    So the subject is not left as a hole.  It is grown over by the room around
    it, using the same push-pull `vr180.surround` spreads a wash with: average
    down until every level has something in it, then come back up putting the
    real detail back wherever there was any.  A pyramid rather than repeated
    blurring, because the hole here is the size of a person and a blur of any
    radius small enough to keep the room sharp is far too small to cross her --
    tried, and it left a black core exactly where the model most needed not to
    find one.

    What that leaves is a smear where she was, which is harmless: it sits inside
    the picture's own rectangle and the live frame is composited back over it.
    What it buys is a canvas with nobody in it, and a model asked to extend that
    extends a room.
    """
    room = keep & known
    if not bool(room.any()) or bool(room.all()):
        return canvas

    colour = (canvas * room)[None]
    weight = room.float()[None, None]
    stack = []
    while min(colour.shape[-2:]) > 1:
        stack.append((colour, weight))
        colour = F.avg_pool2d(colour, 2, ceil_mode=True)
        weight = F.avg_pool2d(weight, 2, ceil_mode=True)

    out = colour / weight.clamp_min(1e-6)
    for colour, weight in reversed(stack):
        out = F.interpolate(out, size=colour.shape[-2:], mode="bilinear", align_corners=False)
        seen = weight.clamp(0, 1)
        out = seen * (colour / weight.clamp_min(1e-6)) + (1 - seen) * out
    return torch.where(room[None], canvas, out[0].clamp(0, 1))


def detail(image, where=None):
    """How busy a picture is: the mean absolute Laplacian of its luminance.

    The number a periphery has to be judged against is not an absolute one but
    the *picture's own* -- a plain wall and a patterned quilt are both correct,
    and which is right here is whatever the shot itself looks like.
    """
    grey = image.mean(0, keepdim=True)[None]
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                          dtype=grey.dtype, device=grey.device)[None, None]
    lap = F.conv2d(F.pad(grey, (1, 1, 1, 1), mode="replicate"), kernel)[0, 0].abs()
    return float(lap.mean() if where is None else lap[where].mean())


def match_detail(painted, known_image, known, tolerance=1.15, most=6):
    """Calm the invented part down until it is no busier than the picture it
    continues.

    The complaint this answers is specific and was measured: asked to widen a
    real frame, the model came back two and a half times busier than the
    photograph it was extending -- framed pictures, a patterned quilt, ornaments
    -- which in the corner of your eye is a thing that pulls at you rather than a
    room you are sitting in.  How *much* is invented does not matter out there.
    How much of it is competing for attention does.

    So the picture's own busyness is measured and the invented part is low-passed
    until it comes down to it.  Deterministic, and a fixed function of the plate,
    so it changes nothing about why none of this can boil.
    """
    from . import stereo

    want = detail(known_image, known) * tolerance
    if want <= 0 or detail(painted, ~known) <= want:
        return painted
    out = painted
    for radius in range(1, most + 1):
        out = torch.where(known[None], painted, stereo._box(painted[None], radius)[0])
        if detail(out, ~known) <= want:
            break
    return out


def available(app_dir=None):
    """Which backends have weights to hand, in the order `auto` prefers."""
    return tuple(name for name in PREFERRED if _weights_for(name, app_dir))


def _weights_for(name, app_dir=None):
    """Where `name`'s weights are, or None.  Beside the app if they were shipped
    with it, else the cache if something has already fetched them -- the same
    order `depth.checkpoint` uses, minus the download: a 7 GB fetch in the middle
    of a conversion is not a fallback, it is a surprise."""
    import os

    from .depth import _app_dir

    root = app_dir or _app_dir()
    if name == "lama":
        from . import plate

        path = os.path.join(root, "models", "lama", plate.FILENAME)
        return path if os.path.isfile(path) else None
    folder = os.path.join(root, "models", name)
    if os.path.isdir(folder) and os.listdir(folder):
        return folder
    # Else whatever is already in the cache, which is where a source checkout
    # keeps them.  Asked `local_files_only`, deliberately: a thirteen-gigabyte
    # download part-way through a conversion is not a fallback, it is a surprise,
    # and the pass has three quieter ones behind it.
    try:
        from huggingface_hub import snapshot_download

        repo, revision = REPOS[name]
        return snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception:
        return None


# Roughly what each backend needs live, in bytes.  FLUX is quantised to 4 bits
# and still 13.4 GB of weights; offloaded it wants that much *host* memory, and
# resident it wants that much of the card.  Measured from the files rather than
# guessed, and rounded up for the runtime around them.
NEEDS = {"flux": 15e9, "sdxl": 8e9, "lama": 1e9}


def room_for(name, device=None):
    """Whether this machine can actually hold that backend.

    Worth asking rather than finding out.  A backend too large does not fail
    politely: FLUX offloaded keeps its weights in host memory, and on a WSL
    instance given half a 32 GB machine that is 13.4 GB inside a 15.9 GB virtual
    machine -- which takes the VM down rather than the process, and so looks like
    the distro crashing and says nothing at all about why.  There is a smaller
    backend behind each of these and a plain blurred wash behind those, so
    declining is always better than trying.
    """
    from . import budget

    wanted = NEEDS.get(name, 0)
    host = budget._available_ram()
    if host is not None and host < wanted:
        return False
    if device is not None and getattr(device, "type", None) == "cuda":
        free = budget.free_bytes(device)
        # Offloaded backends stream through the card rather than living on it,
        # so what they need there is a working set, not the whole model.
        if free is not None and free < min(wanted, 6e9):
            return False
    return True


def choose(asked="auto", app_dir=None, device=None):
    """The backend to use, or None if there is nothing to use.

    `auto` walks `PREFERRED` and takes the first whose weights are present *and*
    which this machine can hold -- so a large one that was bundled but will not
    fit steps aside for a smaller one rather than taking the run with it.
    """
    have = available(app_dir)
    if asked and asked != "auto":
        return asked if asked in have else None
    for name in have:
        if room_for(name, device):
            return name
    return None


def load(name, device=None):
    """The backend, as something that takes `(image, mask)` and gives a picture.

    `image` is `[3, H, W]` in [0, 1] and `mask` `[H, W]` in [0, 1] with 1 where
    the model should paint.  What comes back is the same shape, and what it did
    outside the mask is the caller's to discard.

    Loaded and returned rather than kept, because these are gigabytes and the
    render behind this pass needs the card back -- the same discipline `prepass`
    keeps for the upscaler.
    """
    import torch

    if name == "lama":
        from . import plate

        model = plate.load(device=device or "cpu")

        def run(image, mask, prompt=PROMPT):
            pad_h, pad_w = (-image.shape[1]) % 8, (-image.shape[2]) % 8
            img = image[None].to(model_device(model))
            msk = mask[None, None].to(img.device)
            if pad_h or pad_w:
                img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
                msk = F.pad(msk, (0, pad_w, 0, pad_h), value=1.0)
            with torch.no_grad():
                out = model(img, (msk > 0.5).float())
            return out[0, :, :image.shape[1], :image.shape[2]].float().clamp(0, 1).cpu()

        return run, model

    from PIL import Image

    def as_pil(t, grey=False):
        a = (t.clamp(0, 1).numpy() * 255).astype("uint8")
        return Image.fromarray(a if grey else a.transpose(1, 2, 0))

    if name == "sdxl":
        from diffusers import AutoPipelineForInpainting

        pipe = AutoPipelineForInpainting.from_pretrained(
            _repo("sdxl"), torch_dtype=torch.float16, variant="fp16").to(device or "cuda")
        pipe.set_progress_bar_config(disable=True)

        def run(image, mask, prompt=PROMPT):
            height, width = image.shape[-2:]
            out = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                       image=as_pil(image), mask_image=as_pil(mask, grey=True),
                       width=width, height=height, num_inference_steps=STEPS["sdxl"],
                       strength=1.0, guidance_scale=GUIDANCE["sdxl"],
                       generator=torch.Generator(device or "cuda").manual_seed(0)).images[0]
            return _from_pil(out)

        return run, pipe

    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained(_repo("flux"), torch_dtype=torch.bfloat16)
    # Offloaded rather than resident: 13 GB of model against a 16 GB card that
    # still has to render an 8192x4096 frame afterwards.  It costs about half a
    # minute a shot, which is nothing spread over the hundreds of frames a shot
    # actually has.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    def run(image, mask, prompt=PROMPT):
        height, width = image.shape[-2:]
        out = pipe(prompt=prompt, image=as_pil(image), mask_image=as_pil(mask, grey=True),
                   width=width, height=height, num_inference_steps=STEPS["flux"],
                   guidance_scale=GUIDANCE["flux"], max_sequence_length=512,
                   generator=torch.Generator("cpu").manual_seed(0)).images[0]
        return _from_pil(out)

    return run, pipe


def model_device(model):
    import torch

    return next((p.device for p in model.parameters()), torch.device("cpu"))


def _from_pil(image):
    import numpy as np
    import torch

    return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float().div_(255.0)


# Where each backend's weights come from.  Pinned by revision and checked by
# hash where there is one to check, the same as everything else here.
REPOS = {"sdxl": ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                  "115134f363124c53c7d878647567d04daf26e41e"),
         "flux": ("Meatfucker/Flux.1-Fill-dev-bnb-nf4",
                  "ff1bbd4713c8a26521f48b1ca65bd4f93ade105c")}


def _repo(name):
    """The folder beside the app if the weights were shipped with it, else the
    repository id, which `from_pretrained` will find in the cache."""
    import os

    local = _weights_for(name)
    return local if local else REPOS[name][0]


def band(known, size, fov_x, degrees):
    """A soft weight that is 1 just outside the known picture and falls to 0
    `degrees` further out -- the strip the subject is allowed to be continued
    into, and no further."""
    from . import stereo

    width = size[0]
    radius = max(1, int(round(degrees * width / fov_x)))
    grown = F.max_pool2d(known.float()[None, None], (1, 2 * radius + 1), stride=1,
                         padding=(0, radius))
    grown = F.max_pool2d(grown, (2 * radius + 1, 1), stride=1, padding=(radius, 0))[0, 0]
    soft = stereo._box(grown[None, None], radius)[0, 0].clamp(0, 1)
    return (soft * soft * (3.0 - 2.0 * soft)) * (~known).float()


def widen(colour, covered, room, source_size, focal_px, run,
          reach_deg=REACH_DEG, fade_deg=None, on_note=None):
    """Widen a plate, and say how far it now reaches.

    Two passes, and the difference between them is the whole of what makes this
    hold still.  The first is conditioned on everything the plate has, so the
    picture's edge continues *naturally* -- a leg running off the frame becomes a
    leg, not a stump.  The second is conditioned on the far field alone, so what
    it produces is the room and nothing that moves.  The first is kept in a strip
    a few degrees wide where the picture is already fading out, and the second
    everywhere beyond it.

    That split is the answer to the question this design keeps running into: a
    plate is built once a shot and the subject is not going to stay where it was.
    Anything of theirs that reaches further out than the fade would sit frozen
    beside a picture that moves.  Inside the fade it is half transparent already,
    so when it stops agreeing with them -- and it will -- it stops agreeing
    invisibly.
    """
    from . import vr180

    width, height = source_size
    fade_deg = FADE_DEG if fade_deg is None else fade_deg
    fov_x, fov_y = canvas_fov(width, height, focal_px, reach_deg)
    size = canvas_size(fov_x, fov_y)
    canvas, known_all, focal = to_perspective(colour, covered, fov_x, fov_y, size)
    _, known_room, _ = to_perspective(colour, room & covered, fov_x, fov_y, size)
    if not bool(known_all.any()):
        return colour, covered

    if on_note:
        on_note(f"widening to {fov_x:.0f} by {fov_y:.0f} degrees, "
                f"{100 * float(known_all.float().mean()):.0f}% of it real")

    natural = run(seed(canvas, known_all), feather(known_all, size), PROMPT)

    # The room pass sees the same rectangle of picture with the subject grown
    # over rather than cut out -- see `without` for why that difference decides
    # the result -- and is asked for the margin alone.  It is only worth making
    # when there is enough room left to show it.
    share = (float((known_room & known_all).float().sum())
             / max(float(known_all.float().sum()), 1.0))
    emptied = only_room = None
    if share >= ROOM_SHARE:
        emptied = without(canvas, known_all, known_room)
        only_room = run(seed(emptied, known_all), feather(known_all, size), ROOM_PROMPT)
    elif on_note:
        on_note(f"the subject fills {100 * (1 - share):.0f}% of the frame, so the "
                f"surround is carried on from the picture as it stands")

    if only_room is None:
        painted = natural
    else:
        keep = band(known_all, size, fov_x, fade_deg)[None]
        painted = torch.lerp(only_room, natural, keep)
    _debug(natural=natural, canvas=canvas,
           **({} if only_room is None else {"only_room": only_room, "emptied": emptied}),
           known_all=known_all.float()[None].expand_as(natural),
           known_room=known_room.float()[None].expand_as(natural))
    painted = match_detail(painted, canvas, known_all)
    painted = torch.where(known_all[None], canvas, painted)

    on_plate, reached = vr180.project(painted, focal, _spot_of(colour))
    return torch.where(covered[None], colour, on_plate), covered | reached


def _spot_of(colour):
    from . import vr180

    return vr180.Patch(360.0, 180.0, colour.shape[-1], colour.shape[-2])


def _debug(**pictures):
    """Write the intermediate pictures out when STEREOCRAFT_OUTPAINT_DEBUG names a
    folder.  Off by default and free when it is: the point of a widening that
    goes wrong is that the finished frame does not say which of the two passes
    did it."""
    import os

    where = os.environ.get("STEREOCRAFT_OUTPAINT_DEBUG")
    if not where:
        return
    from PIL import Image

    os.makedirs(where, exist_ok=True)
    for name, picture in pictures.items():
        a = (picture.detach().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(a.transpose(1, 2, 0)).save(os.path.join(where, f"{name}.png"))
