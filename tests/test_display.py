"""Tests for the display module."""

from kernels_bench.display import _format_params, _make_bar, _pad_right, _truncate, _visible_len


def test_visible_len_plain():
    assert _visible_len("hello") == 5


def test_visible_len_with_ansi():
    assert _visible_len("\x1b[1mhello\x1b[0m") == 5
    assert _visible_len("\x1b[38;2;244;183;63mtext\x1b[0m") == 4


def test_pad_right_plain():
    assert _pad_right("hi", 5) == "hi   "


def test_pad_right_with_ansi():
    s = "\x1b[1mhi\x1b[0m"
    padded = _pad_right(s, 5)
    assert _visible_len(padded) == 5
    assert padded.startswith("\x1b[1mhi\x1b[0m")


def test_make_bar_full():
    bar = _make_bar(1.0, 1.0, 10)
    assert bar == "\u2588" * 10


def test_make_bar_empty():
    bar = _make_bar(0.0, 1.0, 10)
    assert bar == "\u2591" * 10


def test_make_bar_half():
    bar = _make_bar(0.5, 1.0, 10)
    assert len(bar) == 10
    assert "\u2588" in bar
    assert "\u2591" in bar


def test_make_bar_zero_max():
    bar = _make_bar(5.0, 0.0, 10)
    assert bar == "\u2591" * 10


def test_format_params_empty():
    assert _format_params({}) == ""


def test_format_params():
    assert _format_params({"M": 1024, "N": 512}) == "M=1024, N=512"


def test_truncate_short():
    assert _truncate("hello", 10) == "hello"


def test_truncate_long():
    result = _truncate("a very long string", 10)
    assert len(result) == 10
    assert result.endswith("...")
