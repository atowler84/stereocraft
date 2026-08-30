"""What the depth estimator does around the model, rather than the model itself.

The model needs weights and a card; these are the bits of plumbing either side
of it that can be checked without both.
"""

from stereocraft.depth import _one_image_at_a_time


class _Processor:
    """Stands in for DA3's input processor, remembering how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "processed"


class _Model:
    def __init__(self, processor):
        self.input_processor = processor


class TestDA3IsNotGivenAThreadPoolPerFrame:
    """DA3's input processor farms per-image work out to a pool of eight
    threads, built and torn down on every call.  This app hands it one image at
    a time, so there is nothing to parallelise -- and on Windows the building
    and tearing down leaks 3.69 MB of commit each time, measured.  Over the
    thirty thousand frames of a long clip that is eighty-six gigabytes promised
    and never used, which is a machine at its commit limit and an encoder dying
    on a frame nine hours in."""

    def test_the_processor_is_asked_to_work_sequentially(self):
        processor = _Processor()
        model = _Model(processor)
        _one_image_at_a_time(model)
        model.input_processor(["an image"], None, None, 504, "upper_bound_resize")
        assert processor.calls == [{"sequential": True}]

    def test_and_still_passes_everything_else_through(self):
        processor = _Processor()
        model = _Model(processor)
        _one_image_at_a_time(model)
        assert model.input_processor("a", b="c") == "processed"
        assert processor.calls[0]["b"] == "c"

    def test_a_caller_that_asks_for_parallel_still_gets_it(self):
        """Defaulted rather than forced: this is our preference for the one-image
        case, not a claim that the pool is broken."""
        processor = _Processor()
        model = _Model(processor)
        _one_image_at_a_time(model)
        model.input_processor(["an image"], sequential=False)
        assert processor.calls[0]["sequential"] is False

    def test_a_model_without_one_is_left_alone(self):
        """A future DA3 that no longer works this way should not crash the app
        on the way past."""
        class Bare:
            pass

        model = Bare()
        _one_image_at_a_time(model)  # must not raise
        assert not hasattr(model, "input_processor")
