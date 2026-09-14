"""Video, webcam, and RTSP reader with paper-style two-second sampling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_seconds: float
    frame_index: int
    source_id: str


def parse_source(source: Union[str, int]):
    if isinstance(source, int):
        return source
    stripped = str(source).strip()
    return int(stripped) if stripped.isdigit() else stripped


class VideoSource:
    def __init__(self, source: Union[str, int], source_id: Optional[str] = None):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video inputs") from exc
        self.cv2 = cv2
        self.source = parse_source(source)
        self.source_id = source_id or str(source)
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError("could not open video source %r" % source)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0  # [ASSUMPTION] only used when the source omits FPS.

    def frames(self) -> Generator[FramePacket, None, None]:
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            timestamp_ms = float(self.capture.get(self.cv2.CAP_PROP_POS_MSEC))
            timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else index / self.fps
            yield FramePacket(frame, timestamp, index, self.source_id)
            index += 1

    def sampled_frames(self, interval_seconds: float = 2.0) -> Generator[FramePacket, None, None]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        next_time = 0.0
        epsilon = 0.5 / self.fps
        for packet in self.frames():
            if packet.timestamp_seconds + epsilon >= next_time:
                yield packet
                next_time += interval_seconds

    def close(self) -> None:
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def read_image(path: Union[str, Path]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for image inputs") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError("could not read image %s" % path)
    return image

