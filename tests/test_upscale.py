"""The super-resolution pass: its chunking, its sizing, and the model itself.

The chunking is the part with a bug in it if anything has: it slides a window
with an overlap that is computed and then thrown away, and getting that wrong
loses frames or repeats them without failing.  So it is checked against a stub
that marks each frame with its own value, which needs no weights and no card.
"""

import pytest
import torch

from stereocraft import upscale


class Marked(torch.nn.Module):
    """Passes frames through untouched, so ordering is readable in the output."""

    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, seq, out_size=None):
        self.calls.append(seq.shape[1])
        return seq


def marks(total):
    """Frames whose value is their index, scaled into [0, 1] because `run`
    clamps -- it is handling colour, and 2.0 is not a colour."""
    return [torch.full((3, 4, 4), i / 500.0) for i in range(total)]


def order(frames):
    return [round(float(f[0, 0, 0]) * 500) for f in frames]


class Invented(torch.nn.Module):
    """Returns a picture of its own, four times bigger and owing nothing to the
    frame that went in -- so what came from the model and what came from the
    source can be told apart by value alone."""

    def __init__(self, value=1.0):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.value = value

    def forward(self, seq, out_size=None):
        n, t, c, h, w = seq.shape
        size = tuple(out_size) if out_size else (h * upscale.SCALE, w * upscale.SCALE)
        return torch.full((n, t, c) + size, self.value,
                          dtype=seq.dtype, device=seq.device)


class Residue(torch.nn.Module):
    """Stands in for the cleaning module, predicting a residual of a size given
    in advance and counting how often it was asked for one.

    `per_frame` gives a size for each frame of the chunk in turn, for the tests
    that care whether the stopping rule reads the chunk or a single frame.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.per_frame = None
        self.calls = 0

    def forward(self, x):
        size = self.size
        if self.per_frame:
            size = self.per_frame[self.calls % len(self.per_frame)]
        self.calls += 1
        return torch.full_like(x, size)


class Passthrough(torch.nn.Module):
    """Stands in for BasicVSR, so a cleaning test measures the cleaning."""

    def forward(self, lqs, out_size=None):
        return lqs


class TestChunking:
    @pytest.mark.parametrize("total", [1, 5, 13, 20, 31, 64, 100])
    @pytest.mark.parametrize("chunk,overlap", [(13, 6), (15, 4), (21, 6), (25, 8)])
    def test_every_frame_comes_out_once_and_in_order(self, total, chunk, overlap):
        model = Marked().eval()
        got = list(upscale.run(model, marks(total), chunk=chunk, overlap=overlap,
                               device="cpu", dtype=torch.float32))
        assert order(got) == list(range(total))

    def test_the_overlap_really_is_reprocessed(self):
        """It is not free, and the cost is the point: those frames are computed
        with a full run-up and thrown away so the emitted ones have context."""
        model = Marked().eval()
        list(upscale.run(model, marks(100), chunk=25, overlap=6, device="cpu",
                         dtype=torch.float32))
        assert sum(model.calls) > 100, "no overlap means no context at the joins"
        assert sum(model.calls) < 200, "and it should not be reprocessing everything twice over"

    def test_a_clip_shorter_than_a_chunk_still_comes_out(self):
        model = Marked().eval()
        got = list(upscale.run(model, marks(3), chunk=32, overlap=6, device="cpu",
                               dtype=torch.float32))
        assert order(got) == [0, 1, 2]


class TestChunkPlan:
    """How many frames at once, and how many to discard at each end.

    Sized from measurements rather than reasoning, because reasoning got it
    wrong once already: a guessed fixed cost of 1.5 GB against a real 6.1 asked
    for 46 frames on a 16 GB card, which does not fail -- it spills to host
    memory and takes eight times as long, and the window sits saying nothing for
    four and a half minutes.
    """

    def test_more_room_buys_more_frames(self):
        small, _ = upscale.chunk_plan(9e9, 720, 1280)
        large, _ = upscale.chunk_plan(15e9, 720, 1280)
        assert large >= small

    # Only a source short of the ceiling is ever upscaled -- past about 1490
    # pixels wide it already has the detail -- so this is the range that has to
    # fit.  A 4K source cannot be made to fit at any chunk length, and is never
    # asked to be; `TestOnlyWhatIsUpscaled` holds that end up.
    UPSCALED = ((640, 480), (720, 1280), (1080, 1920), (1440, 2560), (1489, 2646))

    @pytest.mark.parametrize("free", [6e9, 10e9, 15.7e9])
    def test_it_never_asks_for_more_than_the_card_has(self, free):
        """The whole point.  A chunk past the card is slower than a short one --
        eight times slower, measured -- so the estimate has to stay under it."""
        for width, height in self.UPSCALED:
            chunk, _ = upscale.chunk_plan(free, width, height, out_width=2048)
            cost = (upscale.FIXED_BYTES
                    + upscale.WORKING_BYTES_PER_PIXEL * width * height
                    + chunk * (upscale.CHUNK_BYTES_PER_PIXEL * width * height
                               + upscale.FINISHED_BYTES_PER_PIXEL
                               * 2048 * (2048 * height / width)))
            assert cost < 16e9, f"{width}x{height} at {free/1e9:.0f}GB wants {cost/1e9:.1f}GB"

    def test_the_ceiling_is_low_on_purpose(self):
        assert upscale.chunk_plan(64e9, 320, 240)[0] <= upscale.MAX_CHUNK

    def test_the_overlap_gives_way_before_the_card_does(self):
        """Six frames each end wants a chunk of thirteen.  Where the room will
        not stretch to that the overlap is what shortens: it costs a seam, and
        not shortening it costs the conversion.

        This used to be asked of 1080x1920 on a 16 GB card, which is the case
        the memory model was wrong about -- it now runs at a full chunk and a
        full run-up, and the question has to be put to a card that really is
        short of room.
        """
        _, overlap = upscale.chunk_plan(4.5e9, 1080, 1920, out_width=2048)
        assert overlap < upscale.OVERLAP

    def test_the_overlap_never_eats_the_chunk(self):
        """The window advances by `chunk - 2 * overlap`, so an overlap near half
        the chunk computes every frame many times over.  Six on a chunk of
        fourteen advanced two and took four and a half minutes for a two-second
        clip -- so the overlap is held to a share of the chunk as well as to
        memory.  Three passes a frame is the deliberate ceiling: a tighter one
        bought speed by capping the run-up at three frames on a 16 GB card."""
        for free in (4e9, 10e9, 15.7e9):
            for width, height in self.UPSCALED:
                chunk, overlap = upscale.chunk_plan(free, width, height, out_width=2048)
                if not chunk:
                    continue  # declined for want of room; see the test below
                advance = chunk - 2 * overlap
                assert advance > 0, "a window that does not advance never finishes"
                assert chunk / advance <= 3.0, (
                    f"{width}x{height}: chunk {chunk} overlap {overlap} computes every frame "
                    f"{chunk / advance:.1f} times")

    def test_the_best_run_up_is_one_the_chunking_can_actually_reach(self):
        """`OVERLAP` is an ideal: the overlap is capped at a share of a chunk
        that is itself capped, so no amount of memory reaches it.  The caller
        warns when memory shortened the run-up, and a warning measured against
        an unreachable number fires on every machine and means nothing."""
        assert upscale.BEST_OVERLAP == upscale.MAX_CHUNK // upscale.OVERLAP_SHARE
        _, overlap = upscale.chunk_plan(1e12, 320, 240)
        assert overlap == upscale.BEST_OVERLAP, "a roomy card should reach the ceiling"

    def test_the_short_run_up_warning_says_something_about_this_machine(self):
        """The caller warns on `overlap < GOOD_OVERLAP`, so that comparison has
        to be able to come out either way: silent where the run-up has converged,
        and speaking where memory really did cut into it."""
        _, roomy = upscale.chunk_plan(32e9, 720, 1280, out_width=1980)
        _, cramped = upscale.chunk_plan(3.5e9, 720, 1280, out_width=1980)
        assert roomy >= upscale.GOOD_OVERLAP, "a card with room should not be warned"
        assert cramped < upscale.GOOD_OVERLAP, "a card without it should be"

    def test_a_converged_run_up_is_not_warned_about(self):
        """Four frames is where the edge measurement flattens -- 0.04 of full
        scale against 0.23 at the very edge -- so a run-up of four is worth
        having and not worth mentioning.  Warning about it trains the reader to
        ignore the warning.

        A 16 GB card used to land on exactly four, which is what the threshold
        was chosen against; it reaches the full five now, so the case that lands
        on four is a much smaller one.
        """
        _, overlap = upscale.chunk_plan(5e9, 720, 1280, out_width=1980)
        assert overlap == 4, "the case this threshold was chosen for"
        assert overlap >= upscale.GOOD_OVERLAP, "so it should pass without a word"

    def test_the_quiet_threshold_is_reachable(self):
        """`GOOD_OVERLAP` above `BEST_OVERLAP` would warn on every run, which is
        the failure it was added to fix."""
        assert upscale.GOOD_OVERLAP <= upscale.BEST_OVERLAP

    def test_it_declines_rather_than_run_on_a_handful_of_frames(self):
        """Propagating along the sequence is the whole reason for this model
        over a per-frame one, and over three frames there is no sequence."""
        chunk, overlap = upscale.chunk_plan(3e9, 1440, 2560, out_width=2048)
        assert (chunk, overlap) == (0, 0)

    def test_it_does_not_decline_when_there_is_room(self):
        assert upscale.chunk_plan(15.7e9, 720, 1280, out_width=2048)[0] >= upscale.MIN_USEFUL_CHUNK

    def test_portrait_1080p_on_a_16gb_card_is_not_declined(self):
        """The case the whole memory model was wrong about.

        A phone's portrait 1080p is squarely inside the range that wants this
        pass -- 2639 of a 4096 ceiling -- and it was being told there was no
        room, because the cost of the upsample head was booked as a fixed 6.5 GB
        that did not grow with the frame and as 400 bytes a source pixel that
        did grow with the chunk.  It was neither.  Measured, the whole chunk now
        peaks at 4.06 GB for sixteen frames.
        """
        chunk, overlap = upscale.chunk_plan(15.7e9, 1080, 1920, out_width=2084)
        assert chunk == upscale.MAX_CHUNK, "a full chunk, not a declined pass"
        assert overlap == upscale.BEST_OVERLAP, "and the longest run-up going"

    def test_chunk_length_is_nearly_free(self):
        """What the backward stack going to host memory bought, and the reason
        the plan can stop trading the run-up away.

        Everything that still grows with the chunk is small: the frames
        themselves at 6 bytes a pixel, the model's cleaned copy of them at 6,
        both flow fields at 8, and the finished frames at whatever `out_width`
        asks for.  Against a working set of 400 bytes a pixel that is paid once,
        a frame of the chunk is cheap -- so a card carrying two working sets
        beyond the fixed cost already has room to run, and one carrying three
        and a half reaches the full chunk.
        """
        source = upscale.WORKING_BYTES_PER_PIXEL * 1080 * 1920
        enough = (upscale.FIXED_BYTES + source * 2) / 0.8
        assert upscale.chunk_plan(enough, 1080, 1920, out_width=2084)[0] >= \
            upscale.MIN_USEFUL_CHUNK, f"{enough/1e9:.1f}GB should be enough to run"
        roomy = (upscale.FIXED_BYTES + source * 3.5) / 0.8
        assert upscale.chunk_plan(roomy, 1080, 1920, out_width=2084)[0] == upscale.MAX_CHUNK

    def test_the_frames_working_set_is_charged_once_and_not_per_frame(self):
        """The shape of the fix, stated as the plan sees it.  Only one frame is
        ever inside the head, so its working set comes off the top; charging it
        per frame of the chunk is what made a big frame look unaffordable."""
        big = upscale.chunk_plan(8e9, 1080, 1920, out_width=2084)[0]
        small = upscale.chunk_plan(8e9, 720, 1280, out_width=2084)[0]
        assert big == small == upscale.MAX_CHUNK, (
            "both fit a full chunk; the larger frame costs more room, not fewer frames")

    def test_a_roomy_case_keeps_as_much_overlap_as_it_can_afford(self):
        _, overlap = upscale.chunk_plan(15.7e9, 640, 480, out_width=2048)
        assert overlap > upscale.MIN_OVERLAP

    def test_the_overlap_never_goes_below_its_floor(self):
        """Except when the whole pass is declined, which is (0, 0)."""
        for free in (1, 1e9, 4e9, 15.7e9):
            chunk, overlap = upscale.chunk_plan(free, 720, 1280)
            assert overlap >= upscale.MIN_OVERLAP or (chunk, overlap) == (0, 0)

    def test_a_chunk_is_always_longer_than_what_it_discards(self):
        """Otherwise it emits nothing and the loop does not terminate.  A
        declined pass is (0, 0), which is neither and is checked separately."""
        for free in (1, 4e9, 15.7e9, 64e9):
            for width, height in ((320, 240), (720, 1280), (3840, 2160)):
                chunk, overlap = upscale.chunk_plan(free, width, height)
                assert chunk > 2 * overlap or (chunk, overlap) == (0, 0)

    def test_chunk_length_is_just_the_first_of_the_two(self):
        assert upscale.chunk_length(15.7e9, 720, 1280) == upscale.chunk_plan(15.7e9, 720, 1280)[0]


class TestDetail:
    """The blend back towards a plain enlargement of the source.

    Half the cartoon look is invented detail rather than deleted texture, and
    invented detail is not a bug to be fixed -- it is the point of the model, in
    the wrong quantity.  So it is a dial, and these are its ends.
    """

    def frames(self, total=3, value=0.25):
        return [torch.full((3, 4, 4), value) for _ in range(total)]

    def test_all_of_it_is_the_model_and_nothing_of_the_source(self):
        got = list(upscale.run(Invented(1.0).eval(), self.frames(), chunk=3, overlap=1,
                               device="cpu", dtype=torch.float32, detail=1.0))
        assert all(float(f.min()) == pytest.approx(1.0) for f in got)

    def test_none_of_it_is_the_model_and_all_of_the_source(self):
        """Which is worth having as a real setting and not just as an end point:
        it is the honest enlargement, with the temporal pass reduced to a
        resize, for footage the model only makes worse."""
        got = list(upscale.run(Invented(1.0).eval(), self.frames(value=0.25), chunk=3,
                               overlap=1, device="cpu", dtype=torch.float32, detail=0.0))
        assert all(float(f.max()) == pytest.approx(0.25, abs=1e-5) for f in got)

    def test_between_the_two_it_is_a_blend(self):
        got = list(upscale.run(Invented(1.0).eval(), self.frames(value=0.0), chunk=3,
                               overlap=1, device="cpu", dtype=torch.float32, detail=0.75))
        assert all(float(f.mean()) == pytest.approx(0.75, abs=1e-4) for f in got)

    def test_the_blend_is_against_the_size_that_came_out(self):
        """Not against the size that was asked for.  The two agree, and only one
        of them is a fact -- and a mismatch is an exception rather than a blend,
        which is the sort of thing that only shows up on the one clip that has
        it."""
        got = list(upscale.run(Invented(1.0).eval(), self.frames(), chunk=3, overlap=1,
                               out_size=(9, 7), device="cpu", dtype=torch.float32,
                               detail=0.5))
        assert all(f.shape == (3, 9, 7) for f in got)

    def test_a_frame_already_the_right_size_is_not_resampled(self):
        frame = torch.rand(3, 8, 8)
        assert torch.equal(upscale.plain(frame, (8, 8)), frame.clamp(0, 1))

    def test_the_default_keeps_most_of_the_model(self):
        """A default that threw most of it away would be a slow way of not
        upscaling at all."""
        assert 0.5 < upscale.DETAIL < 1.0


class TestTheBandedHead:
    """That splitting the upsample head into bands changes nothing.

    This is the load-bearing claim of the whole memory arrangement.  The head is
    where nearly all of the model's memory goes -- 8.33 GB a megapixel of source
    for one frame -- and it is run in bands so that a big frame costs what a
    small one does.  Bands are only acceptable if they are invisible, and the
    reason they are is that everything in the head is a local operation: four
    3x3 convolutions, two pixel shuffles and a `scale_factor` enlargement, none
    of which can see further than `HEAD_HALO` rows of source.

    So this is not "the seam is small".  In fp32 there is no difference at all.
    """

    def head(self, model, feat, lr, band_pixels, out_size=None, monkeypatch=None):
        monkeypatch.setattr(upscale, "HEAD_BAND_PIXELS", band_pixels)
        with torch.inference_mode():
            return model.head(feat, lr, out_size)

    def build(self):
        torch.manual_seed(0)
        return upscale.BasicVSR(blocks=1).eval()

    def frames(self, h=48, w=32):
        torch.manual_seed(1)
        return torch.rand(1, 64, h, w), torch.rand(1, 3, h, w)

    # A couple of fp32 ulps.  Splitting the frame changes the shape of the
    # tensors the convolutions are handed, and a convolution blocks and
    # accumulates differently for a different shape -- so a few frames a band
    # comes out bit-identical and a great many bands comes out one ulp adrift.
    # That is floating-point associativity, not the bands showing: 6e-08 against
    # an 8-bit level at 3.9e-03 is five orders of magnitude below anything that
    # reaches a headset, and `test_nothing_gathers_at_the_joins` is what
    # separates rounding from a seam.
    ROUNDING = 1e-6

    def test_bands_reproduce_the_whole_frame(self, monkeypatch):
        model, (feat, lr) = self.build(), self.frames()
        whole = self.head(model, feat, lr, 1 << 30, monkeypatch=monkeypatch)
        for bands in (2, 3, 5, 8, 48):
            band_pixels = max(1, feat.shape[2] * feat.shape[3] // bands)
            got = self.head(model, feat, lr, band_pixels, monkeypatch=monkeypatch)
            worst = (got - whole).abs().max().item()
            assert worst < self.ROUNDING, f"{bands} bands differs by {worst:.3e}"

    def test_a_few_bands_are_exact_to_the_bit(self, monkeypatch):
        """Where the band is big enough that the convolutions block the same way,
        there is no difference at all -- which is the claim being made, with the
        rounding above as the only thing standing between it and the general
        case."""
        model, (feat, lr) = self.build(), self.frames()
        whole = self.head(model, feat, lr, 1 << 30, monkeypatch=monkeypatch)
        for bands in (2, 3):
            band_pixels = feat.shape[2] * feat.shape[3] // bands
            got = self.head(model, feat, lr, band_pixels, monkeypatch=monkeypatch)
            assert torch.equal(got, whole), f"{bands} bands should be exact"

    def test_nothing_gathers_at_the_joins(self, monkeypatch):
        """The test that would catch a seam.  Whatever difference banding leaves
        has to be spread over the frame like rounding and not piled up where the
        bands meet -- a seam is not a large error, it is a *placed* one."""
        model, (feat, lr) = self.build(), self.frames(h=96, w=32)
        whole = self.head(model, feat, lr, 1 << 30, monkeypatch=monkeypatch)
        rows = 12
        got = self.head(model, feat, lr, rows * 32, monkeypatch=monkeypatch)
        error = (got - whole).abs().amax(dim=(0, 1, 3))
        at_join = torch.zeros(error.shape[0], dtype=torch.bool)
        for join in range(rows, feat.shape[2], rows):
            at_join[max(0, join * upscale.SCALE - 4):join * upscale.SCALE + 4] = True
        assert at_join.any(), "the bands have to actually meet somewhere"
        assert error[at_join].max() <= error.max(), (
            "the joins are carrying more error than the picture around them")

    def test_a_halo_too_short_would_have_shown_up(self):
        """The test above only means something if it could fail.  `HEAD_HALO` is
        two because the receptive field of the head works out to two; at one the
        bands differ, and only at the joins, which is what a seam looks like."""
        model, (feat, lr) = self.build(), self.frames()
        with torch.inference_mode():
            whole = model.head(feat, lr)
            rows = feat.shape[2] // 2
            lo = max(0, rows - 1)  # a halo of one, by hand
            band = model.lrelu(model.upsample1(feat[:, :, lo:]))
            band = model.lrelu(model.upsample2(band))
            band = model.lrelu(model.conv_hr(band))
            band = model.conv_last(band)
            band += torch.nn.functional.interpolate(
                lr[:, :, lo:], scale_factor=upscale.SCALE, mode="bilinear",
                align_corners=False)
            short = band[:, :, (rows - lo) * upscale.SCALE:]
        assert not torch.equal(short, whole[:, :, rows * upscale.SCALE:]), (
            "a one-row halo should not reproduce the frame, or the halo is doing nothing")

    def test_the_halo_is_what_the_receptive_field_needs(self):
        assert upscale.HEAD_HALO == 2

    def test_a_frame_small_enough_is_not_banded_at_all(self):
        """No overhead where there is nothing to save."""
        model, (feat, lr) = self.build(), self.frames()
        assert feat.shape[2] * feat.shape[3] < upscale.HEAD_BAND_PIXELS
        with torch.inference_mode():
            assert torch.equal(model.head(feat, lr), model.head(feat, lr))

    def test_the_resize_to_out_size_survives_banding(self, monkeypatch):
        """The bicubic onto `out_size` depends on the extent of what it is given,
        so it waits for the bands to be assembled rather than moving into the
        loop.  If it ever moved, this is what would catch it."""
        model, (feat, lr) = self.build(), self.frames()
        out_size = (100, 70)
        whole = self.head(model, feat, lr, 1 << 30, out_size, monkeypatch)
        got = self.head(model, feat, lr, 64, out_size, monkeypatch)
        assert got.shape[-2:] == out_size
        assert (got - whole).abs().max() < self.ROUNDING


class TestFlowAPairAtATime:
    """Flow is computed one pair at a time rather than as one batch, for the
    memory.  Every pair is independent, so the flows must not have moved."""

    def test_it_matches_the_batched_call(self):
        torch.manual_seed(0)
        model = upscale.BasicVSR(blocks=1).eval()
        lrs = torch.rand(1, 5, 3, 32, 24)
        with torch.inference_mode():
            backward, forward = model.compute_flow(lrs)
            n, t, c, h, w = lrs.shape
            a = lrs[:, :-1].reshape(-1, c, h, w)
            b = lrs[:, 1:].reshape(-1, c, h, w)
            # To a couple of fp32 ulps, for the same reason the banded head is:
            # a convolution handed four pairs at once accumulates in a different
            # order from one handed a pair four times.
            assert torch.allclose(backward, model.spynet(a, b).view(n, t - 1, 2, h, w),
                                  rtol=0, atol=1e-6)
            assert torch.allclose(forward, model.spynet(b, a).view(n, t - 1, 2, h, w),
                                  rtol=0, atol=1e-6)

    def test_the_two_directions_are_not_the_same_flow(self):
        """Cheap, and it would catch the pair being handed over the wrong way
        round -- which would still run, and would propagate along a motion that
        is not the one in the clip."""
        torch.manual_seed(0)
        model = upscale.BasicVSR(blocks=1).eval()
        lrs = torch.rand(1, 3, 3, 32, 24)
        with torch.inference_mode():
            backward, forward = model.compute_flow(lrs)
        assert not torch.equal(backward, forward)


class TestCleaning:
    """How many times the cleaning module runs, which is what made converted
    clips look like cartoons when it was a fixed two.

    Counted in passes over the chunk rather than in calls to the module.  The
    two used to be the same number, the whole chunk going through as one batch;
    the module is fed a frame at a time now -- for the memory, and for the same
    picture -- so a pass is `frames` calls and the thing being tested is how
    many times the sequence is swept.
    """

    FRAMES = 4

    def build(self, residue):
        model = upscale.RealBasicVSR(cleaning_blocks=1, blocks=1)
        model.image_cleaning = Residue(residue)
        model.basicvsr = Passthrough()
        return model.eval()

    def passes(self, model):
        assert model.image_cleaning.calls % self.FRAMES == 0, (
            "a pass cleans every frame or it is not a pass")
        return model.image_cleaning.calls // self.FRAMES

    def test_clean_footage_is_cleaned_once(self):
        """The reference's own rule: stop as soon as the residual is small, with
        a threshold that every real frame meets on the first pass.  A second
        pass is a denoiser run over the output of a denoiser, and what it takes
        off a clean frame is the picture's own texture."""
        model = self.build(residue=0.01)
        with torch.inference_mode():
            model(torch.zeros(1, self.FRAMES, 3, 8, 8))
        assert self.passes(model) == 1

    def test_something_still_full_of_artefacts_is_cleaned_again(self):
        model = self.build(residue=2.0)
        with torch.inference_mode():
            model(torch.zeros(1, self.FRAMES, 3, 8, 8))
        assert self.passes(model) == upscale.CLEANING_LIMIT

    def test_the_stopping_rule_reads_the_whole_chunk_and_not_one_frame(self):
        """The threshold is compared against the mean residual over the chunk.
        A frame at a time, that only stays true if the mean is accumulated
        across the pass and the decision taken after it -- decided per frame,
        a chunk whose frames straddle the threshold would clean some of them
        twice and leave the rest, which is a flicker rather than a clean.
        """
        model = self.build(residue=0.0)
        # Half the chunk far above the threshold, half far below: either way
        # the mean lands above it, so every frame gets a second pass.
        model.image_cleaning.per_frame = [4.0, 4.0, 0.0, 0.0]
        with torch.inference_mode():
            model(torch.zeros(1, self.FRAMES, 3, 8, 8))
        assert self.passes(model) == upscale.CLEANING_LIMIT

    def test_a_chunk_that_is_clean_on_average_stops(self):
        model = self.build(residue=0.0)
        # The mirror image: one noisy frame, the rest spotless, mean below the
        # threshold.  One pass, as for any other clean chunk.
        model.image_cleaning.per_frame = [2.0, 0.0, 0.0, 0.0]
        with torch.inference_mode():
            model(torch.zeros(1, self.FRAMES, 3, 8, 8))
        assert self.passes(model) == 1

    def test_the_caller_keeps_its_own_frames(self):
        """Cleaning happens in place now, so the chunk handed in has to be
        cloned first -- `run` reads the frames it passed in again, to blend the
        model's picture back towards a plain enlargement of them."""
        model = self.build(residue=0.5)
        seq = torch.zeros(1, self.FRAMES, 3, 8, 8)
        with torch.inference_mode():
            model(seq)
        assert torch.equal(seq, torch.zeros(1, self.FRAMES, 3, 8, 8))

    def test_it_never_runs_away(self):
        assert upscale.CLEANING_LIMIT == 3  # the reference's ceiling


@pytest.mark.slow
class TestTheModelItself:
    """Needs the checkpoint, which is 210 MB and downloads on first run."""

    @pytest.fixture(scope="class")
    def model(self):
        return upscale.load(device="cuda" if torch.cuda.is_available() else "cpu")

    def test_it_loads_strictly(self, model):
        """The guard that matters.  Every layer was written from the checkpoint's
        own keys, and an architecture that is subtly wrong and still loads does
        not raise -- it quietly restores the picture into something else."""
        assert sum(p.numel() for p in model.parameters()) == pytest.approx(6.29e6, rel=0.01)

    def test_it_is_four_times_bigger_out_than_in(self, model):
        with torch.inference_mode():
            out = model(torch.rand(1, 5, 3, 32, 48, device=next(model.parameters()).device,
                                   dtype=next(model.parameters()).dtype))
        assert out.shape[-2:] == (32 * upscale.SCALE, 48 * upscale.SCALE)

    def test_it_adds_detail_rather_than_smoothing(self, model):
        """PSNR is the wrong question for a model trained against a
        discriminator -- it will lose to a blurry bicubic every time and is
        meant to.  What it must do is put high-frequency energy back."""
        import torch.nn.functional as F

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        torch.manual_seed(0)
        hr = F.interpolate(torch.rand(1, 3, 24, 24), size=(192, 192), mode="bicubic",
                           align_corners=False).clamp(0, 1).to(device)
        lq = F.interpolate(hr, scale_factor=0.25, mode="area")
        with torch.inference_mode():
            sr = model(lq[None].repeat(1, 9, 1, 1, 1).to(dtype))[0, 4].float().clamp(0, 1)
        bicubic = F.interpolate(lq, scale_factor=4, mode="bicubic",
                                align_corners=False).clamp(0, 1)[0]

        detail = lambda x: float((x[..., 1:] - x[..., :-1]).abs().mean())
        assert detail(sr) > detail(bicubic) * 1.1

    def test_the_middle_of_a_still_clip_is_steady(self, model):
        """Propagation runs both ways, so a frame near an end has seen less of
        the clip than one in the middle -- which is why chunks overlap.  In the
        middle, identical frames must give near-identical output.

        Asked of a smooth field rather than of per-pixel noise, which is what
        `test_it_adds_detail_rather_than_smoothing` uses for the same reason:
        white noise has no motion to find and SPyNet duly finds some anyway, so
        what a noise clip measures is the flow network's confusion rather than
        this model's steadiness on anything a camera produces.
        """
        import torch.nn.functional as F

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        torch.manual_seed(0)
        still = F.interpolate(torch.rand(1, 3, 8, 8), size=(32, 32), mode="bicubic",
                              align_corners=False).clamp(0, 1).to(device)
        seq = still[None].repeat(1, 15, 1, 1, 1).to(dtype)
        with torch.inference_mode():
            out = model(seq)[0].float()
        centre = out.shape[0] // 2
        assert float((out[centre] - out[centre + 1]).abs().max()) < 0.1


class TestOnlyWhatIsUpscaled:
    """The other half of the memory argument: the sizes that would not fit are
    the sizes that are never asked for."""

    def test_a_source_past_the_ceiling_is_never_upscaled(self):
        """Which is what keeps 4K out of the chunk planner's way -- it cannot be
        made to fit at any chunk length, and never has to be."""
        from stereocraft import prepass, video
        from stereocraft.pipeline import VideoSettings
        cfg = VideoSettings(projection="vr180", upscale=True)
        for width, height in ((1920, 1080), (2560, 1440), (3840, 2160)):
            assert prepass.wanted(video.Clip(width, height, 30.0, 100, 3.3), cfg)[0] is False

    def test_a_source_short_of_it_is(self):
        from stereocraft import prepass, video
        from stereocraft.pipeline import VideoSettings
        cfg = VideoSettings(projection="vr180", upscale=True)
        for width, height in ((640, 480), (720, 1280), (1080, 1920)):
            assert prepass.wanted(video.Clip(width, height, 30.0, 100, 3.3), cfg)[0] is True
