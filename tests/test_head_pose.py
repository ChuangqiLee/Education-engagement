import numpy as np
import pytest

from engagement.head_pose.geometry import DEFAULT_MODEL_POINTS, approximate_camera_matrix, select_pose_points
from engagement.head_pose.pnp import HeadPoseEstimator


cv2 = pytest.importorskip("cv2")


def _project(rvec):
    camera = approximate_camera_matrix((640, 480))
    image_points, _ = cv2.projectPoints(
        DEFAULT_MODEL_POINTS,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray((0, 0, 1200), dtype=np.float64).reshape(3, 1),
        camera,
        np.zeros((4, 1)),
    )
    return image_points.reshape(6, 2), camera


def test_front_pose_is_close_to_zero():
    points, camera = _project((0, 0, 0))
    pose = HeadPoseEstimator(camera_matrix=camera).estimate_from_points(points, (640, 480))
    assert abs(pose.pitch) < 0.5
    assert abs(pose.yaw) < 0.5
    assert abs(pose.roll) < 0.5


def test_yaw_direction_and_magnitude_are_reasonable():
    points, camera = _project((0, np.deg2rad(18), 0))
    pose = HeadPoseEstimator(camera_matrix=camera).estimate_from_points(points, (640, 480))
    assert pose.yaw > 10
    assert pose.yaw < 25


def test_insufficient_landmarks_is_clear():
    with pytest.raises(ValueError, match="at least"):
        select_pose_points(np.zeros((10, 3)))

