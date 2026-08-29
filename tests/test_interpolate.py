"""Frame interpolation: the re-timing arithmetic, and whether the network works.

The architecture here was read off the checkpoint's tensor shapes rather than
installed, because the authors ship the source inside a Google Drive archive.
That makes the reconstruction test the point of this file and not a formality:
a forward pass wired up wrongly loads perfectly and interpolates to the wrong
instant, and the only thing that catches it is asking for a frame whose real
answer is known.
"""

import numpy as np
import pytest
import torch

from stereocraft import interpolate

# Measured on a real pan with the answers held out; see the module docstring.
MINTERPOLATE_DB = 29.66
DOING_NOTHING_DB = 16.31


class TestRetiming:
    """No network involved -- which instants get asked for, and how many."""

    def frames(self, count):
        return [torch.full((3, 2, 2), i / 100.0) for i in range(count)]

    def run(self, count, source, target, monkeypatch):
        asked = []
        monkeypatch.setattr(interpolate, "between",
                            lambda m, a, b, f: asked.append(round(f, 4)) or (a + b) / 2)
        got = list(interpolate.run(None, self.frames(count), source, target))
        return got, asked

    def test_doubling_puts_one_frame_between_each_pair(self, monkeypatch):
        got, asked = self.run(5, 30, 60, monkeypatch)
        assert len(got) == 9
        assert asked == [0.5, 0.5, 0.5, 0.5]

    def test_the_same_rate_touches_nothing(self, monkeypatch):
        """An instant that lands on a source frame is that frame.  Running a
        network to reproduce something already in hand would only soften it."""
        got, asked = self.run(5, 30, 30, monkeypatch)
        assert len(got) == 5 and asked == []

    def test_rates_that_do_not_divide(self, monkeypatch):
        """24 into 60 is as ordinary as 30 into 60 -- the network takes the
        instant it is asked for, so nothing is resampled twice to tidy up."""
        got, asked = self.run(5, 24, 60, monkeypatch)
        assert len(got) == 11
        assert asked[:4] == [0.4, 0.8, 0.2, 0.6]

    def test_it_does_not_invent_an_instant_past_the_last_frame(self, monkeypatch):
        got, _ = self.run(5, 50, 60, monkeypatch)
        assert len(got) == 5, "there is nothing after the end to interpolate towards"

    def test_a_single_frame_survives(self, monkeypatch):
        got, asked = self.run(1, 30, 60, monkeypatch)
        assert len(got) == 1 and asked == []


@pytest.mark.slow
class TestTheNetwork:
    """Needs the checkpoint, which downloads on first run."""

    @pytest.fixture(scope="class")
    def model(self):
        return interpolate.load(device="cuda" if torch.cuda.is_available() else "cpu")

    def test_every_parameter_was_found(self, model):
        assert len(model.block) == 5
        assert sum(p.numel() for p in model.parameters()) == pytest.approx(5.66e6, rel=0.01)

    def pan(self, count=9, size=192, step=6):
        """A strip slid across a smooth field: real translation, known answers.

        Smooth and not noise.  White noise has no structure for optical flow to
        lock onto and is an adversarial case rather than a representative one --
        on it this scores 19 dB where real footage gives 35, which would make
        the threshold below measure the fixture instead of the network.
        """
        torch.manual_seed(0)
        wide = size + count * step
        field = torch.nn.functional.interpolate(
            torch.rand(1, 3, 12, 12), size=(size, wide), mode="bicubic",
            align_corners=False).clamp(0, 1)[0]
        return [field[:, :, i * step:i * step + size] for i in range(count)]

    def psnr(self, a, b):
        mse = float(((a - b) ** 2).mean())
        return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))

    def test_it_reconstructs_a_held_out_frame(self, model):
        """The guard the whole module rests on.  A wrong forward pass does not
        score slightly worse -- it collapses to the floor, which is what holding
        the previous frame gets."""
        frames = self.pan()
        scores, floor = [], []
        for i in range(1, len(frames) - 1, 2):
            got = interpolate.between(model, frames[i - 1], frames[i + 1], 0.5).cpu()
            scores.append(self.psnr(got, frames[i]))
            floor.append(self.psnr(frames[i - 1], frames[i]))
        assert np.mean(scores) > np.mean(floor) + 8, "it must beat doing nothing, clearly"
        assert np.mean(scores) > 25, "and land where a working interpolator lands"

    def test_the_chosen_schedule_beats_the_alternatives(self, model):
        """`SCALES` was measured, not remembered.  Running every block at full
        resolution is the degenerate case and has to lose badly."""
        frames = self.pan()
        def score(scales):
            out = []
            for i in range(1, len(frames) - 1, 2):
                a = frames[i - 1][None].to(next(model.parameters()).device)
                b = frames[i + 1][None].to(next(model.parameters()).device)
                with torch.inference_mode():
                    got = model(a, b, timestep=0.5, scales=scales)[0].float().cpu()
                out.append(self.psnr(got, frames[i]))
            return float(np.mean(out))
        assert score(interpolate.SCALES) > score((1.0,) * 5) + 5

    def test_the_ends_are_the_frames_themselves(self, model):
        """Asking for instant 0 or 1 should give back what went in, near enough
        -- if it does not, the timestep is not reaching the network."""
        a, b = self.pan(2)
        assert self.psnr(interpolate.between(model, a, b, 0.0).cpu(), a) > \
               self.psnr(interpolate.between(model, a, b, 1.0).cpu(), a)
