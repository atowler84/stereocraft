"""The frozen build's entry point.

Worth testing from source even though it only runs frozen, because the one bug
it has had was invisible from source and silent when frozen: the JIT was turned
off wholesale to stop `@torch.jit.script` compiling, which also replaced
`RecursiveScriptModule` with a stub, which took `torch.jit.load` with it -- and
the only symptom was that the surround's painted edge quietly stopped appearing
in the packaged app while working perfectly in the checkout.

So both halves are asserted here: the decorator must not compile, and loading a
serialised graph must still work.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

LAUNCHER = Path(__file__).resolve().parent.parent / "packaging" / "windows" / "launcher.py"


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("stereocraft_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def unscripted(launcher):
    """Applies the patch and puts the real one back, so no other test inherits it."""
    original = torch.jit.script
    launcher._unscript()
    yield
    torch.jit.script = original


class TestUnscript:
    def test_the_decorator_stops_compiling(self, unscripted):
        @torch.jit.script
        def affine_inverse(A: torch.Tensor):
            R, T, P = A[..., :3, :3], A[..., :3, 3:], A[..., 3:, :]
            return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)

        assert not isinstance(affine_inverse, torch.jit.ScriptFunction)

    def test_and_what_it_decorated_still_runs(self, unscripted):
        """DA3's helper is a 4x4 affine inverse, so plain Python costs nothing --
        but it has to give the same answer."""

        @torch.jit.script
        def affine_inverse(A: torch.Tensor):
            R, T, P = A[..., :3, :3], A[..., :3, 3:], A[..., 3:, :]
            return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)

        matrix = torch.eye(4)[None]
        matrix[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(affine_inverse(matrix) @ matrix, torch.eye(4)[None], atol=1e-6)

    def test_called_with_arguments_it_hands_back_a_decorator(self, unscripted):
        @torch.jit.script(optimize=True)
        def double(x):
            return x * 2

        assert double(3) == 6

    def test_torchscript_itself_is_left_alone(self, unscripted):
        """The half that broke.  `torch.jit.load` compiles nothing -- it reads a
        graph that was serialised long ago -- and turning the JIT off took it out
        along with the compiler."""
        assert hasattr(torch.jit.RecursiveScriptModule, "_construct")

    def test_the_blunt_switch_is_not_set_any_more(self):
        """`PYTORCH_JIT=0` is what broke this, and if it comes back so does the
        bug -- silently, and only once frozen.

        Read out of the syntax tree rather than grepped for, so the paragraph
        above `_unscript` explaining the history does not count as using it.
        """
        import ast

        tree = ast.parse(LAUNCHER.read_text())
        names = {node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)
                 and "PYTORCH_JIT" == node.value}
        assert not names

    @pytest.mark.slow
    def test_a_serialised_module_still_loads(self, unscripted):
        from stereocraft import plate

        model = plate.load(device="cpu")
        image, mask = torch.rand(1, 3, 64, 64), torch.zeros(1, 1, 64, 64)
        mask[..., 20:40, 20:40] = 1
        with torch.no_grad():
            assert model(image, mask).shape == (1, 3, 64, 64)
