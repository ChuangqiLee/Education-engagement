"""Generic multi-source adapter.

Exact hardware synchronization is [UNRESOLVED] in the paper. This adapter
performs timestamp-ordered, best-effort multiplexing.
"""

import heapq
from typing import Dict, Iterable, Iterator, Mapping, Union

from .video_source import FramePacket, VideoSource


class MultiCameraSource:
    def __init__(self, sources: Mapping[str, Union[str, int]]):
        if not sources:
            raise ValueError("at least one source is required")
        self.sources = {name: VideoSource(value, name) for name, value in sources.items()}

    def sampled_frames(self, interval_seconds: float = 2.0) -> Iterator[FramePacket]:
        iterators = {name: iter(source.sampled_frames(interval_seconds)) for name, source in self.sources.items()}
        heap = []
        serial = 0
        for name, iterator in iterators.items():
            try:
                packet = next(iterator)
                heapq.heappush(heap, (packet.timestamp_seconds, serial, name, packet))
                serial += 1
            except StopIteration:
                pass
        while heap:
            _, _, name, packet = heapq.heappop(heap)
            yield packet
            try:
                following = next(iterators[name])
                heapq.heappush(heap, (following.timestamp_seconds, serial, name, following))
                serial += 1
            except StopIteration:
                pass

    def close(self) -> None:
        for source in self.sources.values():
            source.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

