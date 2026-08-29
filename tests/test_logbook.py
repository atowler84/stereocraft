"""The log exists because the packaged app cannot say anything any other way.

It is a windowed exe with no console, and `launcher` rebinds stdout and stderr
to the null device on purpose.  So when a long conversion dies there is nothing
to look at afterwards -- which is exactly the case the file is for, and why the
tests below care most about it never being the thing that breaks.
"""

import logging

import pytest

from stereocraft import logbook


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """A logbook that has not been started, writing somewhere disposable."""
    monkeypatch.setattr(logbook, "_started", False)
    monkeypatch.setattr(logbook, "directory", lambda: str(tmp_path))
    for handler in list(logbook.log.handlers):
        logbook.log.removeHandler(handler)
    yield tmp_path
    for handler in list(logbook.log.handlers):
        handler.close()
        logbook.log.removeHandler(handler)


class TestItWritesSomethingDown:

    def test_starting_makes_a_file(self, fresh):
        path = logbook.start()
        assert path and (fresh / "stereocraft.log").exists()

    def test_a_note_lands_in_it_as_one_greppable_line(self, fresh):
        logbook.start()
        logbook.note("surround done", shots=12, peak="1.20G")
        text = (fresh / "stereocraft.log").read_text()
        assert "surround done shots=12 peak=1.20G" in text

    def test_starting_twice_does_not_double_every_line(self, fresh):
        logbook.start()
        logbook.start()
        logbook.note("once")
        assert (fresh / "stereocraft.log").read_text().count("once") == 1

    def test_a_note_before_the_start_is_dropped_rather_than_raising(self, fresh):
        logbook.note("nobody is listening")  # must not raise


class TestItNeverTakesTheAppDown:
    """A log that cannot be opened is a shame, not a reason not to convert."""

    def test_nowhere_to_write_is_survivable(self, monkeypatch):
        monkeypatch.setattr(logbook, "_started", False)
        monkeypatch.setattr(logbook, "directory", lambda: "")
        for handler in list(logbook.log.handlers):
            logbook.log.removeHandler(handler)
        assert logbook.start() == ""
        logbook.note("still works")

    def test_a_directory_that_refuses_writing_is_not_chosen(self, tmp_path):
        """Asked by writing, rather than by inspecting permissions: the case
        this exists for is the app installed somewhere the user does not own,
        and the only portable way to know is to try."""
        assert logbook._writable(str(tmp_path))
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("in the way")
        assert not logbook._writable(str(blocked / "logs"))


class TestTheMemoryTrail:
    """The point of the whole file: a conversion killed by the OS unwinds
    nothing and raises nothing, so the only evidence is the trail written before
    it went.  Peak rather than current, because current is a sample and misses
    the spike that did it."""

    def test_it_reports_a_peak_at_least_as_big_as_the_current(self):
        current, peak = logbook.memory()
        assert peak >= current

    def test_and_the_numbers_are_plausible_for_a_python_holding_torch(self):
        current, _ = logbook.memory()
        assert current > 1_000_000  # more than a megabyte, less than absurd
        assert current < 1_000_000_000_000

    def test_the_machine_has_some_ram(self):
        assert logbook.total_memory() > 0

    def test_a_stage_records_what_it_cost(self, fresh):
        logbook.start()
        with logbook.stage("surround", frames=90):
            pass
        text = (fresh / "stereocraft.log").read_text()
        assert "surround begin frames=90" in text
        assert "surround done" in text and "peak=" in text

    def test_a_stage_that_dies_says_so_and_still_raises(self, fresh):
        logbook.start()
        with pytest.raises(MemoryError):
            with logbook.stage("surround"):
                raise MemoryError("no")
        text = (fresh / "stereocraft.log").read_text()
        assert "surround failed" in text and "error=MemoryError" in text

    def test_the_beginning_is_on_disk_before_the_pass_runs(self, fresh):
        """The whole design rests on this.  A pass that never returns must
        already have left its name in the file."""
        logbook.start()
        with logbook.stage("surround"):
            assert "surround begin" in (fresh / "stereocraft.log").read_text()
