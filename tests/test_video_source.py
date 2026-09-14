import pytest

from engagement.data.synthetic import make_demo_video
from engagement.io.multi_camera import MultiCameraSource
from engagement.io.video_source import VideoSource


pytest.importorskip("cv2")


def test_two_second_sampling(tmp_path):
    path = make_demo_video(tmp_path / "sample.mp4", frames=19, fps=3.0)
    with VideoSource(path) as source:
        packets = list(source.sampled_frames(2.0))
    assert [round(item.timestamp_seconds) for item in packets] == [0, 2, 4, 6]


def test_multiple_sources_are_multiplexed(tmp_path):
    first = make_demo_video(tmp_path / "one.mp4", frames=7, fps=3.0)
    second = make_demo_video(tmp_path / "two.mp4", frames=7, fps=3.0)
    with MultiCameraSource({"front": first, "overhead": second}) as source:
        packets = list(source.sampled_frames(2.0))
    assert {item.source_id for item in packets} == {"front", "overhead"}

