"""Real-world video super-resolution, for a source too small to fill the sphere.

The VR180 frame is the ceiling now -- 4096 an eye, against a Quest 3's 25 pixels
a degree -- and a source is judged by whether it has the detail to fill one.
`vr180.natural_size` answers that, and most phone video answers no: 720 across a
phone's 65 degrees stretches to 1759 of the 4096 that will be written.
The projection can resample into the rest and does, but resampling moves no
information; this puts some there.

**Why a temporal model and not the obvious one.**  Real-ESRGAN is smaller,
faster and better documented, and it has no concept of time: each frame is
restored on its own, so the detail it invents at one frame differs from the
detail it invents at the next and the picture crawls.  That is the same boil
that made a blurred surround the right answer for the periphery rather than a
diffusion fill.  BasicVSR propagates features along the sequence in both
directions, so what it invents is anchored to what it invented either side of
it, and a still frame stays still.

**What it is exactly.**  RealBasicVSR: a cleaning module run over each frame to
take the compression and noise off, then BasicVSR -- SPyNet for optical flow,
one residual stack propagating backwards through the clip and another forwards,
and a pixel-shuffle head that ends at four times the size.  6.3M parameters, and
notably *not* BasicVSR++, which is where the deformable convolution and the
second-order propagation live.  There is none of that here: the architecture was
read off the checkpoint's own keys rather than assumed, and it is plain
convolutions and one `grid_sample` throughout.

**Why it is written out here rather than installed.**  The reference
implementation wants `mmcv-full` and `mmedit`, which is a large stack to take on
for six megabytes of weights, and this project has form for declining that --
see the `evo` note at the top of `requirements.txt`, and the two DA3 modules
`depth.DepthEstimator._load_da3` stubs out.  Every layer below is checked
against the checkpoint by loading it `strict=True`, which is the thing that
would catch a mistake: a wrong architecture that still loads produces a
plausible picture rather than an error.
"""

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

# Apache-2.0.  The authors publish to Google Drive; this is a mirror, so it is
# pinned by revision and checked by hash before anything is loaded from it.
REPO = "akhaliq/RealBasicVSR_x4"
FILENAME = "RealBasicVSR_x4.pth"
REVISION = "2e3382d49222d0835e19001b868f84aaf541a0e0"
SHA256 = "f2d73c0fe3e33869c812f608be77cbdbfc356443b718b78c9a6cb97027ce5de7"
# The published checkpoint carries the discriminator, the perceptual loss and an
# optimizer state as well -- 210 MB, of which the generator is 25.
WEIGHTS_PREFIX = "generator_ema."
SCALE = 4
# The most source pixels to send through the upsample head at once.
#
# The head is where nearly all of this model's memory goes, and not by a little:
# measured at 8.33 GB for every megapixel of source, for a *single* frame, dead
# linear from 480x854 through to 1080x1920.  `upsample2` and `conv_hr` hold 64
# channels at four times the frame -- 4320x7680, 4.25 GB a tensor, several of
# them alive at once -- and then `out_size` takes it all straight back down
# again, which is a great deal of memory spent on pixels that are discarded a
# line later.
#
# It is a per-frame cost and not a per-chunk one, which the chunking below was
# once built on the opposite belief about: at 1080x1920 the head peaks at 18.46
# GB for a chunk of four and 19.39 GB for a chunk of seven.  Shortening the
# chunk was never going to pay for it, which is why a clip that plainly had room
# was told there was none.
#
# Banded, it costs this many pixels' worth whatever size the frame is, which is
# what makes a big frame affordable and a long chunk nearly free.  240k is a
# little under 2 GB at the figure above.
HEAD_BAND_PIXELS = 240_000
# Rows of source each band is grown by before it is computed, and trimmed of
# afterwards.
#
# The head is four 3x3 convolutions and two pixel shuffles and nothing else --
# every one of them local -- so a halo does not approximate the whole-frame
# answer, it reproduces it.  Measured against the unbanded head: at a halo of one
# the bands differ by 1.4e-01 and only at the joins, and at two they are
# bit-identical in fp32.  In fp16 the joins differ by 4.9e-04 against 9.8e-04 for
# the picture away from them, so the joins are quieter than the frame's own
# rounding -- which is the point.  This is not a seam being hidden well; there is
# no seam.
#
# Two is what the receptive field works out to and what the measurement found,
# and the two agreeing is the reason to trust it.  Note that the bilinear skip
# below survives banding for a reason that is easy to lose: `scale_factor` with
# `align_corners=False` is a pure affine map that does not know the tensor's
# extent, so a slice enlarged is the enlargement sliced.  The bicubic resize to
# `out_size` is *not* extent-independent, which is why it stays on the assembled
# frame rather than moving into the band loop.
HEAD_HALO = 2
# Frames thrown away at each end of a chunk.  Propagation runs both ways along
# the sequence, so a frame near either end has seen less of the clip than one in
# the middle and is reconstructed differently -- measured at 0.23 of full scale
# at the very edge against 0.04 four frames in.  Left in, those are a pulse at
# every chunk boundary; discarded, they are the price of not having one.
OVERLAP = 6
# The cleaning module's stopping rule, taken from the reference rather than from
# a remembered number -- see `RealBasicVSR` for what the remembered number cost.
# At most three passes, and stop as soon as the mean absolute residual falls
# below the threshold; the reference's 255 is in 0-255 units, which is 1.0 on a
# picture in [0, 1], so in practice one pass and no more.
CLEANING_LIMIT = 3
CLEANING_THRESHOLD = 1.0
# How much of the model's picture to keep, against a plain enlargement of the
# frame it came from.
#
# The network adds a residual to a bilinear upsample -- literally, see
# `BasicVSR.forward` -- so a blend back towards that upsample is a blend back
# towards the source: at 1.0 the model's picture entire, at 0 an honest resample
# and nothing invented.  It is the knob for the half of the cartoon look that the
# cleaning fix does not reach, which is the invented half: RealBasicVSR was
# trained on synthetic footage degraded far past anything a phone produces, and
# fed something cleaner it still reconstructs with the confidence it learnt on
# the wreckage -- pores, fabric weave and leaves come back as smooth, decided
# shapes that are plausible and not what was there.
#
# Not measured, unlike most numbers in this file, because there is nothing here
# to measure against: the thing being traded is how much invented detail looks
# like too much, and that is a matter of taste and of how close the headset puts
# it.  0.75 is a starting point that keeps most of the sharpening and takes the
# plastic edge off it; `--upscale-detail` is there because the right answer is
# per-clip.
DETAIL = 0.75


class ResidualBlockNoBN(nn.Module):
    """Two convolutions and a skip.  No normalisation, which is the point: batch
    norm on a restoration network takes the contrast with it."""

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


class ResidualBlocksWithInputConv(nn.Module):
    """A conv to get to 64 channels, then a stack of residual blocks.

    Laid out as `main` with the stack at index 2 because that is where the
    checkpoint's keys put it -- `main.0` the conv, `main.1` the activation,
    `main.2.<n>` the blocks.
    """

    def __init__(self, in_channels, out_channels=64, blocks=30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*(ResidualBlockNoBN(out_channels) for _ in range(blocks))),
        )

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):
    """Upsample by making the channels into pixels.  Cheaper than a transposed
    convolution and without its chequerboard."""

    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale * scale, 3, 1, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


def flow_warp(x, flow, padding_mode="zeros"):
    """Move `x` by `flow`, which is in pixels and ordered (dx, dy)."""
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2)[None] + flow
    grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((grid_x, grid_y), dim=3), mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


class SPyNetBasicModule(nn.Module):
    """One level of the flow pyramid: five 7x7 convolutions ending at two
    channels, which are the flow."""

    def __init__(self):
        super().__init__()
        widths = ((8, 32), (32, 64), (64, 32), (32, 16), (16, 2))
        layers = []
        for index, (a, b) in enumerate(widths):
            block = nn.Sequential()
            block.add_module("conv", nn.Conv2d(a, b, 7, 1, 3))
            if index < len(widths) - 1:
                block.add_module("relu", nn.ReLU(inplace=False))
            layers.append(block)
        self.basic_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    """Optical flow, coarse to fine over six levels.

    Each level warps the second frame by the flow so far and asks only for the
    correction, which is what keeps a five-convolution module able to find
    motion far larger than its own receptive field.
    """

    def __init__(self, levels=6):
        super().__init__()
        self.basic_module = nn.ModuleList(SPyNetBasicModule() for _ in range(levels))
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.shape
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(len(self.basic_module) - 1):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]

        flow = ref[0].new_zeros(n, 2, h // (2 ** (len(ref) - 1)), w // (2 ** (len(ref) - 1)))
        for level in range(len(ref)):
            if level == 0:
                up = flow
            else:
                up = F.interpolate(flow, scale_factor=2, mode="bilinear",
                                   align_corners=True) * 2.0
            flow = up + self.basic_module[level](torch.cat(
                [ref[level], flow_warp(supp[level], up.permute(0, 2, 3, 1),
                                       padding_mode="border"), up], 1))
        return flow

    def forward(self, ref, supp):
        """Padded to a multiple of 32, because six halvings is what the pyramid
        wants and a frame is rarely built that way."""
        h, w = ref.shape[2:4]
        gh, gw = int(math.ceil(h / 32)) * 32, int(math.ceil(w / 32)) * 32
        ref = F.interpolate(ref, size=(gh, gw), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(gh, gw), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w),
                             mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= w / gw
        flow[:, 1, :, :] *= h / gh
        return flow


class BasicVSR(nn.Module):
    """Propagate features backwards along the clip, then forwards, then upsample.

    The two directions are the whole idea: a detail that is clear in a later
    frame reaches an earlier one, and the other way round, so what the network
    reconstructs is anchored to the rest of the shot instead of being invented
    afresh sixty times a second.
    """

    def __init__(self, channels=64, blocks=30):
        super().__init__()
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(channels + 3, channels, blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(channels + 3, channels, blocks)
        self.fusion = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(channels, channels, 2)
        self.upsample2 = PixelShufflePack(channels, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Flow between every adjacent pair, both ways, a pair at a time.

        The obvious way to write this is one `spynet` call on a batch of `t - 1`
        pairs, and that is what it was.  Every pair is independent of every
        other, so the flows are the same either way -- but the batched call
        builds a six-level pyramid for the whole chunk at once and runs a
        five-convolution module at 32 and 64 channels over all of it at the
        widest level.  Measured at 1080x1920 and a chunk of seven: 5.65 GB
        batched against 1.18 GB one pair after another, for the same numbers.
        """
        n, t, c, h, w = lrs.size()
        backward = lrs.new_empty(n, t - 1, 2, h, w)
        forward = lrs.new_empty(n, t - 1, 2, h, w)
        for i in range(t - 1):
            a, b = lrs[:, i], lrs[:, i + 1]
            backward[:, i] = self.spynet(a, b)
            forward[:, i] = self.spynet(b, a)
        return backward, forward

    def head(self, feat, lr, out_size=None):
        """One frame, from fused features to finished pixels, in bands.

        Banded because this is the one expensive thing the model does -- see
        `HEAD_BAND_PIXELS` -- and banded exactly rather than approximately,
        because everything in here is a local operation and `HEAD_HALO` covers
        the reach of all of it.

        The resize to `out_size` waits until the bands are assembled: it is a
        bicubic with antialiasing onto a given size, which depends on the extent
        of what it is given and so is the one step that would not survive being
        done a band at a time.
        """
        n, _, h, w = feat.shape
        bands = max(1, math.ceil(h * w / HEAD_BAND_PIXELS))
        rows = math.ceil(h / bands)
        out = feat.new_empty(n, 3, h * SCALE, w * SCALE)
        for start in range(0, h, rows):
            stop = min(h, start + rows)
            lo, hi = max(0, start - HEAD_HALO), min(h, stop + HEAD_HALO)
            band = self.lrelu(self.upsample1(feat[:, :, lo:hi]))
            band = self.lrelu(self.upsample2(band))
            band = self.lrelu(self.conv_hr(band))
            band = self.conv_last(band)
            # The network learns the difference from a plain enlargement, which
            # is why it can be this small and still hold the picture together.
            band += F.interpolate(lr[:, :, lo:hi], scale_factor=SCALE,
                                  mode="bilinear", align_corners=False)
            keep = (start - lo) * SCALE
            out[:, :, start * SCALE:stop * SCALE] = band[:, :, keep:keep + (stop - start) * SCALE]
            del band
        if out_size is not None and out.shape[-2:] != tuple(out_size):
            out = F.interpolate(out, size=tuple(out_size), mode="bicubic",
                                align_corners=False, antialias=True)
        return out

    def forward(self, lrs, out_size=None):
        """`out_size` resizes each frame as it is finished rather than after the
        whole sequence is built.  The finished frames are four times the input,
        so shrinking them to the width the caller actually wants is what makes a
        long chunk affordable -- and a long chunk is what keeps the propagation
        honest at the ends.

        Written into one tensor as they are finished rather than collected and
        stacked at the end.  `torch.stack` on a list of finished frames wants a
        second copy of every one of them at the moment the chunk is complete,
        which is the most expensive moment to ask for it.
        """
        n, t, c, h, w = lrs.size()
        flows_backward, flows_forward = self.compute_flow(lrs)

        # The backward pass produces a 64-channel map per frame -- 265 MB a frame
        # at 1080x1920 -- and nothing looks at any of them again until the
        # forward pass arrives at that index.  Held in host memory instead they
        # cost a round trip over PCIe, and what that buys is a memory model that
        # barely has a chunk length in it: measured at 1080x1920, a chunk of
        # fourteen peaks 0.2 GB above a chunk of seven.  Which is the right way
        # round, the long chunk being the one with the better joins.
        stack = []
        feat_prop = lrs.new_zeros(n, self.fusion.in_channels // 2, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat_prop = flow_warp(feat_prop, flows_backward[:, i].permute(0, 2, 3, 1))
            feat_prop = self.backward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1))
            stack.append(feat_prop.to("cpu"))
        stack.reverse()

        out_h, out_w = out_size if out_size is not None else (h * SCALE, w * SCALE)
        finished = lrs.new_empty(n, t, 3, out_h, out_w)
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(t):
            if i > 0:
                feat_prop = flow_warp(feat_prop, flows_forward[:, i - 1].permute(0, 2, 3, 1))
            feat_prop = self.forward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1))

            back = stack[i].to(lrs.device)
            stack[i] = None  # let go of the host copy as it is spent
            fused = self.lrelu(self.fusion(torch.cat([back, feat_prop], dim=1)))
            del back
            finished[:, i] = self.head(fused, lrs[:, i], out_size)
            del fused
        return finished


class RealBasicVSR(nn.Module):
    """BasicVSR with a cleaning pass in front of it.

    Real footage arrives with compression on it, and a restoration network fed
    its artefacts will faithfully sharpen them.  The cleaning module predicts a
    residual that takes them off first.

    **How many times it runs is not a fixed number, and getting that wrong is
    what made converted clips look like cartoons.**  This ran it twice, on the
    belief that twice is what the reference does at test time.  It is not: the
    reference runs it at most three times and stops as soon as the residual it
    predicted is small, with a threshold of 255/255 -- which, on a picture in
    [0, 1], effectively every real frame meets on the first pass.  So the
    reference cleans once and this cleaned twice, and the second pass is a
    denoiser run over the output of a denoiser.  On footage that arrived clean
    there is no compression left for it to find by then, so what it takes off is
    the picture's own texture: skin goes to wax, foliage to a green mass, and
    what BasicVSR then sharpens is a painting of the scene rather than the scene.
    """

    def __init__(self, cleaning_blocks=20, blocks=30, cleaning_limit=CLEANING_LIMIT,
                 cleaning_thres=CLEANING_THRESHOLD):
        super().__init__()
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, 64, cleaning_blocks),
            nn.Conv2d(64, 3, 3, 1, 1, bias=True),
        )
        self.basicvsr = BasicVSR(blocks=blocks)
        self.cleaning_limit = cleaning_limit
        self.cleaning_thres = cleaning_thres

    def forward(self, lqs, out_size=None):
        """Cleaned a frame at a time rather than as one batch of the whole chunk.

        The module has no notion of sequence -- it restores each frame on its own
        -- so the picture is the same either way, but the batched call runs a
        twenty-block residual stack over every frame at once and measured 11.24
        GB at 1080x1920 on a chunk of seven.

        The stopping rule is the one place the batch genuinely mattered: it
        compares the mean residual *over the chunk* against the threshold, not
        the mean over one frame.  So the mean is accumulated across a pass and
        the decision taken after it, which is the same number as before -- every
        frame holds the same count of elements, so the mean of the per-frame
        means is the batch's mean.  Accumulated in fp32 because a chain of fp16
        additions is not owed this much.
        """
        n, t, c, h, w = lqs.size()
        lqs = lqs.clone()  # cleaned in place from here, and the caller's to keep
        for _ in range(self.cleaning_limit):
            residual = torch.zeros((), device=lqs.device, dtype=torch.float32)
            for i in range(t):
                residue = self.image_cleaning(lqs[:, i])
                lqs[:, i] += residue
                residual += residue.abs().mean().float()
            # Stop as soon as there is nothing much left to take off, which is
            # the reference's own rule and on clean footage is after one pass.
            if float(residual) / t < self.cleaning_thres:
                break
        return self.basicvsr(lqs, out_size=out_size)


def _blocks(state, prefix):
    """How many residual blocks a stack in this checkpoint actually has, so the
    module is built to the weights rather than to a remembered number."""
    seen = {int(k[len(prefix):].split(".")[0]) for k in state if k.startswith(prefix)}
    return max(seen) + 1 if seen else 0


def checkpoint():
    """Where to load the weights from: beside the app if they were shipped with
    it, else the cache, else Hugging Face -- the same order `depth.checkpoint`
    uses, and for the same reason."""
    from .depth import _app_dir, _use_local_cache

    bundled = os.path.join(_app_dir(), "models", "realbasicvsr", FILENAME)
    if os.path.isfile(bundled):
        return bundled
    _use_local_cache()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, FILENAME, revision=REVISION)


def verify(path):
    """The weights come from a mirror rather than from the authors, so they are
    checked before anything is built out of them."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise RuntimeError(f"{path} is not the checkpoint this was written against "
                           f"(sha256 {digest.hexdigest()})")
    return path


def load(device="cpu", dtype=torch.float32, path=None):
    """The model, with the generator weights in it and nothing else.

    Loaded `strict=True` deliberately.  Every layer above was written from the
    checkpoint's own keys, and the one failure worth guarding against is an
    architecture that is subtly wrong and still loads -- which does not raise,
    it just quietly restores the picture into something else.
    """
    path = verify(path or checkpoint())
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    weights = {k[len(WEIGHTS_PREFIX):]: v for k, v in state.items()
               if k.startswith(WEIGHTS_PREFIX)}
    if not weights:  # a bare generator dump rather than a training checkpoint
        weights = {k[len("generator."):]: v for k, v in state.items()
                   if k.startswith("generator.")}

    model = RealBasicVSR(
        cleaning_blocks=_blocks(weights, "image_cleaning.0.main.2."),
        blocks=_blocks(weights, "basicvsr.backward_resblocks.main.2."),
    )
    model.load_state_dict(weights, strict=True)
    return model.to(device=device, dtype=dtype).eval()


# What a chunk costs, measured on a 4080 in fp16 rather than reasoned about.
#
# **This was wrong in shape, and it is worth saying how.**  It read
#
#     6.5 GB fixed + 400 bytes per source pixel per frame of the chunk
#
# on the belief that memory grew with chunk length and that a fixed 6.5 GB sat
# under it.  Measured stage by stage, neither half held: the upsample head alone
# peaked at 18.46 GB for a chunk of four and 19.39 GB for a chunk of seven at
# 1080x1920 -- a per-*frame* cost, scaling with the frame's area and hardly at
# all with the chunk.  So the term that actually blew the budget was hiding
# inside a constant that claimed not to grow, and the plan responded to trouble
# by shortening a chunk that was never the problem.  A portrait 1080p clip on a
# 16 GB card was told there was no room for a pass it had ample room for.
#
# With the head banded, the cleaning done a frame at a time and the backward
# stack parked in host memory, the same measurement gives three honest terms:
#
#     source       chunk 6   chunk 10   chunk 16
#     720x1280      2.62 GB   2.88 GB    3.27 GB
#     1080x1920     3.18 GB   3.53 GB    4.06 GB
#     1280x2276     3.62 GB   4.04 GB    4.67 GB
#
# which separate to within 0.2 GB across all nine, and each term is something
# the code can be pointed at.
#
# The numbers below are fitted to `run` rather than to the model, which is the
# difference between what the model costs and what a conversion costs: `run`
# blends every finished frame back towards a plain enlargement of its source,
# and holds fp32 copies of both while it does.  Measured at the entry point the
# plan is actually sizing:
#
#     source       chunk 6   chunk 16
#     720x1280      3.25 GB   4.17 GB
#     1080x1920     3.76 GB   4.92 GB
#     1280x2276     4.19 GB   5.49 GB
#
# Every constant here is rounded up from its fit, so all six come out predicted
# a little high.  Guessing high costs a few frames of chunk; guessing low costs
# the conversion.
FIXED_BYTES = 2.4e9
# The single-frame working set, per source pixel: the propagation features, the
# residual stacks' own temporaries, and the assembled four-times frame the head
# writes its bands into.  Paid once however long the chunk is, because only one
# frame is ever in the head at a time.
#
# The old number here was also 400 -- it was never the magnitude that was wrong,
# it was that this was multiplied by the chunk length.  Fitted at 351.
WORKING_BYTES_PER_PIXEL = 400
# And per source pixel per frame of the chunk, which is all that genuinely
# scales with chunk length now: the caller's copy of the frames and the model's
# cleaned clone of them, 6 bytes each, and both flow fields at 8.  Measured at
# 20.1, 20.1 and 20.3 bytes at the three sizes above, which is the arithmetic
# exactly.
CHUNK_BYTES_PER_PIXEL = 20
# And per pixel of the *finished* frame per frame of the chunk.  Six of it is the
# frames themselves, three channels of fp16 held until the chunk is handed back.
# The rest is what `run` does with each one on the way out -- `torch.lerp`
# between the model's picture and a bicubic enlargement of the source, both in
# fp32 -- which is a per-frame transient rather than a per-chunk cost, but sits
# here because it is what the allocator has to have room for while the finished
# frames are still resident.  Fitted at 9.5 across the three sizes.
FINISHED_BYTES_PER_PIXEL = 10
# However much room there seems to be.
#
# This used to stand for memory -- a longer chunk was held to cost a great deal
# of it -- and that is no longer why it is here.  A frame of chunk costs about
# 88 MB at 1080x1920 now, so the card would carry a hundred of them.  What the
# ceiling stands for is time: the window advances by `chunk - 2 * overlap`, a
# chunk is computed before any of it comes back, and the pass counts frames
# going in so that a Stop button has something to act on.  Sixteen is a few
# seconds of work at a time, which is a reasonable thing to ask someone to wait
# for and a reasonable thing to throw away when they stop.
#
# It could rise, and there is a case for it: eighteen would put `BEST_OVERLAP`
# at the full six frames rather than five.  That is a change to how every clip
# is converted rather than to whether it converts at all, so it is left alone
# here and noted as available.
MAX_CHUNK = 16
# The overlap gives way before the card does.  Shortening it costs temporal
# context at the joins, which is a visible seam; not shortening it costs the
# conversion, a chunk having to be longer than what it discards.
#
# It reaches this floor much less often than it did.  The case that used to be
# quoted here -- six frames each end wanting a chunk of thirteen, which at
# 1080x1920 was 17.9 GB and past a 16 GB card before a single frame was
# upscaled -- costs 3.8 GB now, and is not the constraint on anything.
MIN_OVERLAP = 1
# And it gives way again to arithmetic.  The window advances by
# `chunk - 2 * overlap`, so six frames of overlap on a chunk of fourteen advances
# two and computes every frame seven times over -- which measured as four and a
# half minutes for a two-second clip.  Holding the overlap to a third of the
# chunk keeps the advance at a third of it, so nothing is computed more than
# three times.  A quarter here would be cheaper -- 1.7 passes a frame rather
# than 3 -- but it caps the overlap at four frames whatever the clip, and the
# joins are what the whole chunked arrangement is trying to hide.
OVERLAP_SHARE = 3
# Below this many frames at a time there is no point running this model at all.
# Propagating along the sequence is the whole reason for choosing it over a
# per-frame upscaler, and over three or four frames there is no sequence to
# propagate along -- it would be slower than Real-ESRGAN and no steadier.  The
# caller is expected to skip the pass and say why rather than do it badly.
MIN_USEFUL_CHUNK = 6


# The longest run-up this can ever produce, however roomy the card: the overlap
# is held to a share of a chunk that is itself capped, so `OVERLAP` is an ideal
# rather than a reachable number.  Worth having as its own name because the
# caller warns when memory shortened the run-up, and measuring that against the
# ideal makes it warn on every machine ever built.
BEST_OVERLAP = min(OVERLAP, MAX_CHUNK // OVERLAP_SHARE)
# And the shortest run-up still worth staying quiet about.  The edge measurement
# above is 0.23 of full scale at the very edge against 0.04 four frames in, so
# by four the propagation has effectively converged and the frame between here
# and `BEST_OVERLAP` is below anything a headset shows.  `BEST_OVERLAP` is the
# right yardstick for whether memory shortened the run-up and the wrong one for
# whether the viewer should be told: a 16 GB card lands on four routinely, and a
# warning every run teaches the reader to ignore the one that matters.
GOOD_OVERLAP = min(4, BEST_OVERLAP)


def chunk_plan(free_bytes, width, height, out_width=None, overlap=OVERLAP):
    """How many frames to hand the model at once, and how many to throw away at
    each end of them, for the room actually free.

    A chunk has to be longer than the overlap it discards or it emits nothing
    and the loop never ends, so when the room will not stretch to that the
    overlap is what shortens.  Returns `(chunk, overlap)`, and the caller is
    expected to say so when the overlap came back shorter than it asked for --
    it is a visible seam, and a quiet one is worse than a mentioned one.
    """
    scale = (out_width / (width * SCALE)) if out_width else 1.0
    finished = ((width * SCALE * scale) * (height * SCALE * scale)
                * FINISHED_BYTES_PER_PIXEL)
    per_frame = finished + CHUNK_BYTES_PER_PIXEL * width * height
    # The frame's own working set comes off the top rather than being charged
    # per frame of the chunk, which is what it actually is -- see `FIXED_BYTES`.
    room = max(0.0, free_bytes * 0.8 - FIXED_BYTES
               - WORKING_BYTES_PER_PIXEL * width * height)
    fits = int(room // max(per_frame, 1))
    if fits < MIN_USEFUL_CHUNK:
        return 0, 0  # not enough room to be worth doing; the caller says so
    while overlap > MIN_OVERLAP and fits < 2 * overlap + 1:
        overlap -= 1
    chunk = max(2 * overlap + 1, min(MAX_CHUNK, fits))
    return chunk, max(MIN_OVERLAP, min(overlap, chunk // OVERLAP_SHARE))


def chunk_length(free_bytes, width, height, out_width=None, overlap=OVERLAP):
    """Just the chunk, for a caller that does not care what the overlap became."""
    return chunk_plan(free_bytes, width, height, out_width, overlap)[0]


def plain(frame, out_size=None):
    """The frame enlarged and nothing more: no detail added, none taken away.

    What `detail` blends against, and the picture the whole pass is trying to
    beat.  Bicubic without antialiasing, this being an enlargement -- the filter
    only has averaging to do on the way down.
    """
    if out_size is None:
        out_size = (frame.shape[-2] * SCALE, frame.shape[-1] * SCALE)
    if tuple(frame.shape[-2:]) == tuple(out_size):
        return frame.clamp(0, 1)
    return F.interpolate(frame[None].float(), size=tuple(out_size), mode="bicubic",
                         align_corners=False)[0].clamp_(0, 1)


@torch.inference_mode()
def run(model, frames, out_size=None, chunk=None, overlap=OVERLAP, device=None, dtype=None,
        detail=DETAIL):
    """Upscale an iterable of `[3, H, W]` frames in [0, 1], yielding them in order.

    Chunked with an overlap that is computed and thrown away, so every frame
    emitted has a full run-up of context in both directions.  The window slides
    by `chunk - 2 * overlap`, which means the overlap is paid for twice -- that
    is the cost of not having a visible pulse everywhere two chunks meet.

    `detail` is how much of the model's picture to keep against a plain
    enlargement of the source, 1.0 for all of it and 0 for none -- see `DETAIL`.
    The blend is done here rather than inside the model because it is per frame
    and needs the frame that went in, which the model no longer has.
    """
    device = device or next(model.parameters()).device
    dtype = dtype or next(model.parameters()).dtype
    window, done, first = [], 0, True
    detail = float(detail)

    def process(batch, take_from, take_to):
        seq = torch.stack(batch).to(device=device, dtype=dtype)[None]
        out = model(seq, out_size=out_size)[0]
        for index in range(take_from, take_to):
            frame = out[index].float().clamp_(0, 1)
            if detail < 1.0:
                # Against whatever size actually came out, rather than against
                # the size that was asked for: the two agree, and only one of
                # them is a fact.
                honest = plain(batch[index].to(device=device), frame.shape[-2:])
                frame = torch.lerp(honest, frame, detail)
            yield frame

    for frame in frames:
        window.append(frame)
        if chunk is None:
            chunk = 2 * overlap + 1
        if len(window) < chunk:
            continue
        yield from process(window, 0 if first else overlap, len(window) - overlap)
        first = False
        window = window[len(window) - 2 * overlap:]
        done += 1

    if window:
        # The tail: everything not yet emitted, with no overlap to discard at the
        # end because there is no next chunk to take it from.
        start = 0 if first else min(overlap, len(window))
        if start < len(window):
            yield from process(window, start, len(window))
