"""Capture-property strategies for getting *raw* YUY2 out of OpenCV.

Reading this camera means convincing OpenCV not to convert the stream. That
turns out to be the hardest part of the whole library, because
``CAP_PROP_CONVERT_RGB`` is honoured inconsistently across backends and OpenCV
versions:

* On the DSHOW backend, setting a *different* property after
  ``CAP_PROP_CONVERT_RGB`` silently resets it to the default (see
  https://github.com/opencv/opencv/issues/19367). So the flag has to be set
  **last**, after fourcc and frame size -- the exact opposite of what looks
  natural.
* Some builds want it set *before* the format instead.
* Some need it applied twice, once before and once after the frame size.
* Some negotiate a usable raw mode only when fourcc is left alone entirely.

Rather than pick one and hope, we define each of these as a named strategy and
try them in turn, re-opening the device between attempts because backend state
is sticky. The first strategy that yields a decodable frame wins, and
:meth:`ThermalCamera.info` reports which one it was, so a machine that needs an
unusual path is diagnosable rather than mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2

Applier = Callable[["cv2.VideoCapture", int, int, int], None]


def _fourcc() -> int:
    """YUY2 as an OpenCV fourcc int, across OpenCV 4.x and 5.x.

    ``VideoWriter_fourcc`` moved to ``VideoWriter.fourcc`` in OpenCV 5.
    """
    fn = getattr(cv2, "VideoWriter_fourcc", None)
    if fn is None:
        fn = cv2.VideoWriter.fourcc
    return int(fn(*"YUY2"))


def _set_size(cap, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def _convert_off(cap) -> None:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)


def _small_buffer(cap) -> None:
    """Keep read() near real time.

    Deliberately part of each strategy rather than applied around them: on
    DSHOW this counts as "a later property" and would revert CONVERT_RGB, so
    each strategy has to place it where its own ordering stays intact.
    """
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass


# --- The strategies ----------------------------------------------------------

def _convert_last(cap, width, height, fps) -> None:
    """fourcc -> size -> CONVERT_RGB. Correct for DSHOW (issue #19367)."""
    _small_buffer(cap)
    cap.set(cv2.CAP_PROP_FOURCC, _fourcc())
    _set_size(cap, width, height, fps)
    _convert_off(cap)


def _convert_first(cap, width, height, fps) -> None:
    """CONVERT_RGB -> fourcc -> size. Correct for some MSMF builds."""
    _convert_off(cap)
    _small_buffer(cap)
    cap.set(cv2.CAP_PROP_FOURCC, _fourcc())
    _set_size(cap, width, height, fps)


def _convert_twice(cap, width, height, fps) -> None:
    """Belt and braces: set the flag on both sides of every other property."""
    _convert_off(cap)
    _small_buffer(cap)
    cap.set(cv2.CAP_PROP_FOURCC, _fourcc())
    _convert_off(cap)
    _set_size(cap, width, height, fps)
    _convert_off(cap)


def _no_fourcc(cap, width, height, fps) -> None:
    """Leave the format alone; some backends only cooperate if unprompted."""
    _small_buffer(cap)
    _set_size(cap, width, height, fps)
    _convert_off(cap)


def _raw_format(cap, width, height, fps) -> None:
    """Ask for undecoded output via CAP_PROP_FORMAT = -1, then size."""
    _small_buffer(cap)
    cap.set(cv2.CAP_PROP_FOURCC, _fourcc())
    _set_size(cap, width, height, fps)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    _convert_off(cap)


@dataclass(frozen=True)
class Strategy:
    name: str
    apply: Applier
    note: str


#: Tried in order. ``convert_last`` is first because it is the one that fixes
#: the common Windows/DSHOW case.
STRATEGIES: list[Strategy] = [
    Strategy("convert_last", _convert_last,
             "CONVERT_RGB set after fourcc and frame size (DSHOW-safe order)"),
    Strategy("convert_twice", _convert_twice,
             "CONVERT_RGB re-applied around every other property"),
    Strategy("convert_first", _convert_first,
             "CONVERT_RGB set before fourcc and frame size"),
    Strategy("raw_format", _raw_format,
             "CAP_PROP_FORMAT=-1 to request undecoded frames"),
    Strategy("no_fourcc", _no_fourcc,
             "frame size and CONVERT_RGB only, fourcc left untouched"),
]


def convert_rgb_readback(cap) -> float | None:
    """What the backend *says* CONVERT_RGB is now, or None if unsupported.

    A readback of 1.0 after we set 0.0 is proof the backend rejected the
    request -- worth reporting, though some backends lie in both directions,
    so it is a hint rather than a verdict. The frame shape is the real test.
    """
    try:
        value = cap.get(cv2.CAP_PROP_CONVERT_RGB)
    except cv2.error:
        return None
    return None if value is None else float(value)
