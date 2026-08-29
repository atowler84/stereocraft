"""Frame interpolation, for a clip too slow to be comfortable in a headset.

30fps is markedly worse to sit inside than to watch on a wall, and 60 is the
target for a reason beyond taste: the decode ceiling falls as frame rate rises,
so 60 is the fastest a clip can run and still be written at the full 4096 an
eye.  Above that, smoothness is bought with angular resolution.

**RIFE rather than ffmpeg's `minterpolate`.**  `minterpolate` estimates one
motion vector per block, which is a 2015-era answer: it tears where objects
uncover each other and cannot represent motion that varies inside a block.  RIFE
estimates dense flow with a learned network and fuses the two warps with a mask
that is itself predicted, which is what lets it survive an occlusion edge.
Measured on a reconstruction test -- drop every other frame, put them back,
compare against what was really there -- `minterpolate` scores 29.66 dB on a
smooth pan where holding the previous frame scores 16.31.  RIFE is kept honest
against the same number in the tests, which is the point of using it.

**Why the architecture is written out here.**  The authors publish the model
source inside the weights archive on Google Drive rather than in the repository,
so there is nothing to install and nothing to fetch.  Everything below was read
off the checkpoint's own tensor shapes, which pin it down further than they
might: block 1 takes 28 channels, and 28 is the warped pair (6) plus their
warped features (8) plus the timestep (1) plus the mask (1) plus the propagated
feature (8) plus the flow (4).  There is only one arrangement that adds up.

**What that leaves uncertain, and how it is settled.**  The pyramid's scale per
block is not written in the weights, and a wrong schedule would load cleanly and
interpolate to slightly the wrong instant -- fine in a still, juddering in
motion.  So it was measured rather than guessed, against real frames with the
answer held out, and the tests assert the result: 35.19 dB against
`minterpolate`'s 29.66 and a floor of 16.31 for doing nothing at all.  35 is
also where this network's published figures sit, which is a second opinion from
somebody who had the source.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

# MIT.  The authors publish to Google Drive; this is a mirror, pinned and hashed.
REPO = "raymondt/rife"
FILENAME = "flownet.pkl"
REVISION = "9f18bb034dedaacc69a267cb59647af050135e68"
# Coarse to fine, and settled by measurement rather than by memory.  Against
# real frames, reconstructing the ones held out: this schedule scores 35.19 dB
# where 8,4,2,1,1 scores 34.05 and running every block at full resolution scores
# 17.69 -- which is the floor, holding the previous frame scoring 16.31.  That
# spread is what makes the test worth having: a wrong schedule does not look
# slightly worse, it collapses.
SCALES = (16.0, 8.0, 4.0, 2.0, 1.0)
# The pyramid halves twice per block, so the frame has to divide by enough for
# the coarsest of them.
ALIGN = 64


def warp(source, flow):
    """Move `source` by `flow`, which is in pixels and ordered (dx, dy)."""
    _, _, h, w = source.shape
    horizontal = torch.linspace(-1.0, 1.0, w, device=source.device, dtype=source.dtype)
    vertical = torch.linspace(-1.0, 1.0, h, device=source.device, dtype=source.dtype)
    grid = torch.cat([horizontal.view(1, 1, 1, -1).expand(-1, -1, h, -1),
                      vertical.view(1, 1, -1, 1).expand(-1, -1, -1, w)], 1)
    scaled = torch.cat([flow[:, 0:1] / ((w - 1.0) / 2.0),
                        flow[:, 1:2] / ((h - 1.0) / 2.0)], 1)
    return F.grid_sample(source, (grid + scaled).permute(0, 2, 3, 1), mode="bilinear",
                         padding_mode="border", align_corners=True)


def _conv(inp, out, stride=2):
    """Convolution and activation.  The activation carries no parameters -- the
    checkpoint has none for it -- which is what says it is a leaky ReLU rather
    than the PReLU some versions of this network use."""
    return nn.Sequential(nn.Conv2d(inp, out, 3, stride, 1), nn.LeakyReLU(0.2, True))


class ResConv(nn.Module):
    """A convolution scaled by a learned per-channel gain and added back.

    The gain is what `beta` is in the checkpoint, and it is why these stack
    eight deep without the signal running away.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """One level of the flow pyramid.

    Runs at `1 / scale` of the frame, halves twice more inside `conv0`, and
    comes back out through a transposed convolution and a pixel shuffle -- so
    what it returns is at the resolution it was handed.  The thirteen channels
    it ends on are four of flow, one of mask and eight of feature for the next
    block to carry.
    """

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_channels, channels // 2), _conv(channels // 2, channels))
        self.convblock = nn.Sequential(*(ResConv(channels) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(channels, 4 * 13, 4, 2, 1),
                                      nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1.0):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear",
                                 align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        out = self.lastconv(self.convblock(self.conv0(x)))
        out = F.interpolate(out, scale_factor=scale, mode="bilinear", align_corners=False)
        return out[:, :4] * scale, out[:, 4:5], out[:, 5:]


class Head(nn.Module):
    """A small feature encoder, run once per input frame.

    Four channels out at the frame's own resolution -- down by two and back --
    which is what the blocks warp alongside the pictures themselves.
    """

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class IFNet(nn.Module):
    """Five blocks, coarse to fine, each correcting the flow the last one left.

    Every block sees both frames warped by the flow so far, their features
    warped the same way, the mask, the feature the last block passed on, and the
    instant being asked for -- so a timestep is an input rather than a fixed
    midpoint, and any frame rate divides into any other.
    """

    def __init__(self, widths=((15, 192), (28, 128), (28, 96), (28, 64), (28, 32))):
        super().__init__()
        self.block = nn.ModuleList(IFBlock(inp, channels) for inp, channels in widths)
        self.encode = Head()

    def forward(self, img0, img1, timestep=0.5, scales=SCALES):
        if not torch.is_tensor(timestep):
            timestep = torch.full_like(img0[:, :1], float(timestep))
        f0, f1 = self.encode(img0), self.encode(img1)
        flow = mask = feat = None
        warped0, warped1 = img0, img1
        for block, scale in zip(self.block, scales):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1),
                                         None, scale=scale)
            else:
                delta, mask, feat = block(
                    torch.cat((warped0, warped1, warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4]),
                               timestep, mask, feat), 1), flow, scale=scale)
                flow = flow + delta
            warped0 = warp(img0, flow[:, :2])
            warped1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return (warped0 * mask + warped1 * (1 - mask)).clamp(0, 1)


def _widths(state):
    """The blocks this checkpoint actually has, read off its own shapes so the
    module is built to the weights rather than to a remembered number."""
    widths, index = [], 0
    while f"block.{index}.conv0.0.0.weight" in state:
        first = state[f"block.{index}.conv0.0.0.weight"]
        second = state[f"block.{index}.conv0.1.0.weight"]
        widths.append((first.shape[1], second.shape[0]))
        index += 1
    return tuple(widths)


def checkpoint():
    from .depth import _app_dir, _use_local_cache

    bundled = os.path.join(_app_dir(), "models", "rife", FILENAME)
    if os.path.isfile(bundled):
        return bundled
    _use_local_cache()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, FILENAME, revision=REVISION)


def load(device="cpu", dtype=torch.float32, path=None):
    """The network, with the interpolation weights in it and nothing else.

    `teacher` and `caltime` are in the checkpoint and are not loaded: the first
    is the distillation branch used while training, the second estimates a
    timestep for footage of unknown rate, and this asks for the instants it
    wants.  So the load cannot be `strict` in the direction of unused keys --
    but it is checked in the direction that matters, that every parameter the
    model has was found.
    """
    import re

    state = torch.load(path or checkpoint(), map_location="cpu", weights_only=False)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    # The checkpoint numbers the blocks into the name -- block0, block1 -- where
    # a ModuleList wants block.0.  Renamed rather than the list unrolled into
    # five attributes, so the forward can still iterate over them.
    state = {re.sub(r"^block(\d+)\.", r"block.\1.", k): v for k, v in state.items()}
    widths = _widths(state)
    if not widths:
        raise RuntimeError("no interpolation blocks in this checkpoint")
    model = IFNet(widths=widths)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"the checkpoint is missing {len(missing)} tensors this model "
                           f"needs, starting {missing[:3]} -- it is not the one this was "
                           f"written against")
    return model.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def between(model, first, second, at=0.5):
    """The frame `at` of the way from `first` to `second`, both `[3, H, W]`."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    a, b = first[None].to(device, dtype), second[None].to(device, dtype)
    _, _, h, w = a.shape
    ph, pw = -h % ALIGN, -w % ALIGN
    if ph or pw:
        a = F.pad(a, (0, pw, 0, ph), mode="replicate")
        b = F.pad(b, (0, pw, 0, ph), mode="replicate")
    out = model(a, b, timestep=at)
    return out[0, :, :h, :w].float()


@torch.inference_mode()
def run(model, frames, source_fps, target_fps):
    """Re-time an iterable of `[3, H, W]` frames, yielding them at `target_fps`.

    Holds two source frames at a time and nothing else, so a long clip costs no
    more than a short one.

    Nothing here assumes the rates divide: the network takes the instant it is
    asked for, so 50 into 60 is as ordinary as 30 into 60 and a clip is never
    resampled twice to make the arithmetic tidy.  An instant that lands on a
    source frame is that frame, untouched -- running a network to reproduce
    something already in hand would only soften it.
    """
    step = float(source_fps) / float(target_fps)
    previous, previous_index, wanted = None, -1, 0

    for index, frame in enumerate(frames):
        if previous is None:
            previous, previous_index = frame, index
            continue
        while wanted * step <= index + 1e-6:
            fraction = wanted * step - previous_index
            if fraction <= 1e-6:
                yield previous
            elif fraction >= 1.0 - 1e-6:
                yield frame
            else:
                yield between(model, previous, frame, fraction)
            wanted += 1
        previous, previous_index = frame, index

    if previous is not None and wanted == 0:
        yield previous  # a single-frame clip has no motion to re-time
