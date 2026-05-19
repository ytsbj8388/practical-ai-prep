"""Tests for `src.transcript` — speaker name canonicalization.

The canonicalization rule has to balance two real cases that look superficially
similar:

  1. ``"Daniel"`` and ``"Daniel Whitenack"`` appearing in the same transcript →
     same person, where the bare first name is just shorthand. Merge.
  2. ``"Chris Benson"`` and ``"Chris Shallue"`` (host + guest) in the same
     transcript → two different people. Keep separate.

The earlier prefix-match rule conflated these and refused to parse case (2),
which caused ~50 of 350 podcast episodes to fail with a `ValueError` because
``"Chris Benson"`` (the recurring host) frequently appears alongside guests whose
first name is also Chris.
"""

from __future__ import annotations

import pytest

from src.transcript import _canonicalize_speakers


# ---------------------------------------------------------------------------
# Single-group tests
# ---------------------------------------------------------------------------


def test_two_chris_full_names_kept_separate():
    """Host + guest share a first name but are different people — keep both."""
    result = _canonicalize_speakers({"Chris Benson", "Chris Shallue"})
    assert result == {
        "Chris Benson": "Chris Benson",
        "Chris Shallue": "Chris Shallue",
    }


def test_short_form_resolves_to_full_name():
    """A bare 'Daniel' next to a full 'Daniel Whitenack' is just shorthand — merge."""
    result = _canonicalize_speakers({"Daniel", "Daniel Whitenack"})
    assert result == {
        "Daniel": "Daniel Whitenack",
        "Daniel Whitenack": "Daniel Whitenack",
    }


def test_truly_ambiguous_short_form():
    """A bare 'Chris' with multiple full forms in the same transcript is ambiguous."""
    raw = {"Chris", "Chris Benson", "Chris Shallue"}
    with pytest.raises(ValueError) as excinfo:
        _canonicalize_speakers(raw)
    msg = str(excinfo.value)
    assert "ambiguous short-form" in msg.lower(), (
        f"error message should explain why parsing failed, got: {msg!r}"
    )
    assert "chris" in msg.lower(), "error should name the offending first name"


# ---------------------------------------------------------------------------
# Multi-group + edge cases
# ---------------------------------------------------------------------------


def test_distinct_first_names_unchanged():
    """The common case from episode 1 — four distinct first names, all map to themselves."""
    raw = {"Adam Stacoviak", "Jerod Santo", "Daniel Whitenack", "Chris Benson"}
    result = _canonicalize_speakers(raw)
    assert result == {name: name for name in raw}


def test_three_full_forms_same_first_name_all_kept():
    """Three guests sharing a first name — all distinct, each maps to itself."""
    raw = {"Chris Benson", "Chris Shallue", "Chris DeBellis"}
    result = _canonicalize_speakers(raw)
    assert result == {
        "Chris Benson": "Chris Benson",
        "Chris Shallue": "Chris Shallue",
        "Chris DeBellis": "Chris DeBellis",
    }


def test_only_short_form_maps_to_itself():
    """If 'Daniel' is the only form in the transcript, it stays as 'Daniel'."""
    result = _canonicalize_speakers({"Daniel"})
    assert result == {"Daniel": "Daniel"}


def test_short_form_with_other_groups_unaffected():
    """A short-form merge in one group must not bleed into other groups."""
    raw = {"Daniel", "Daniel Whitenack", "Adam Stacoviak"}
    result = _canonicalize_speakers(raw)
    assert result == {
        "Daniel": "Daniel Whitenack",
        "Daniel Whitenack": "Daniel Whitenack",
        "Adam Stacoviak": "Adam Stacoviak",
    }


def test_first_name_match_is_case_insensitive():
    """Grouping uses lowercased first tokens so 'DANIEL' and 'Daniel Whitenack' merge."""
    result = _canonicalize_speakers({"DANIEL", "Daniel Whitenack"})
    assert result == {
        "DANIEL": "Daniel Whitenack",
        "Daniel Whitenack": "Daniel Whitenack",
    }
