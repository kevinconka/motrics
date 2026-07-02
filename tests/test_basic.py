"""Smoke tests that exercise the built extension end to end."""

import motrics


def test_version_is_exposed() -> None:
    assert isinstance(motrics.__version__, str)
    assert motrics.__version__


def test_version_function_matches_dunder() -> None:
    assert motrics.version() == motrics.__version__
