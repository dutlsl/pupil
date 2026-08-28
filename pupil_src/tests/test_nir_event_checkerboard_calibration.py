import cv2
import numpy as np

from nir_event_checkerboard_calibration import (
    CheckerboardSpec,
    CheckerboardStereoCollector,
    Intrinsics,
    StereoObservation,
    calibrate_stereo,
    event_frame_from_events,
    find_checkerboard_corners,
)


def _checkerboard_image(columns=10, rows=7, square=30, inverted=False):
    margin = square
    image = np.full(
        (rows * square + 2 * margin, columns * square + 2 * margin), 127, dtype=np.uint8
    )
    for row in range(rows):
        for column in range(columns):
            bright = (row + column + int(inverted)) % 2 == 0
            value = 255 if bright else 0
            y0 = margin + row * square
            x0 = margin + column * square
            image[y0 : y0 + square, x0 : x0 + square] = value
    return image


def test_event_frame_keeps_on_and_off_polarity_separate():
    image = event_frame_from_events(
        np.array([3, 16]),
        np.array([3, 16]),
        np.array([1, 0]),
        image_size=(20, 20),
    )
    assert image[3, 3] > 127
    assert image[16, 16] < 127


def test_finds_normal_and_inverted_checkerboard():
    checkerboard = CheckerboardSpec((9, 6), 10.0)
    for inverted in (False, True):
        corners = find_checkerboard_corners(
            _checkerboard_image(inverted=inverted), checkerboard
        )
        assert corners is not None
        assert corners.shape == (54, 1, 2)


def test_live_collector_accepts_an_already_reconstructed_event_frame():
    checkerboard = CheckerboardSpec((9, 6), 10.0)
    collector = CheckerboardStereoCollector(checkerboard)
    image = _checkerboard_image()
    assert collector.add_pair("pose-01", image, event_image=image)
    assert len(collector.observations) == 1
    assert collector.eye0_image_size == (360, 270)
    assert collector.event_image_size == (360, 270)


def test_fixed_intrinsic_stereo_calibration_recovers_transform():
    checkerboard = CheckerboardSpec((5, 4), 8.0)
    object_points = checkerboard.object_points()
    eye0_matrix = np.array([[260.0, 0.0, 160.0], [0.0, 255.0, 120.0], [0.0, 0.0, 1.0]])
    event_matrix = np.array([[230.0, 0.0, 150.0], [0.0, 235.0, 115.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros((5, 1), dtype=np.float64)
    rotation, _ = cv2.Rodrigues(np.array([0.03, -0.04, 0.01]))
    translation = np.array([[24.0], [1.5], [3.0]])

    observations = []
    for index in range(8):
        board_rvec = np.array([0.04 * index, -0.02 * index, 0.01 * index])
        board_rotation, _ = cv2.Rodrigues(board_rvec)
        board_translation = np.array(
            [[-12.0 + 4.0 * index], [-8.0 + 2.0 * index], [420.0 + 15.0 * index]]
        )
        event_rotation = rotation @ board_rotation
        event_translation = rotation @ board_translation + translation
        event_rvec, _ = cv2.Rodrigues(event_rotation)
        eye0_corners, _ = cv2.projectPoints(
            object_points, board_rvec, board_translation, eye0_matrix, distortion
        )
        event_corners, _ = cv2.projectPoints(
            object_points, event_rvec, event_translation, event_matrix, distortion
        )
        observations.append(StereoObservation(str(index), eye0_corners, event_corners))

    result = calibrate_stereo(
        observations,
        checkerboard=checkerboard,
        eye0_image_size=(320, 240),
        event_image_size=(300, 230),
        eye0_intrinsics=Intrinsics(eye0_matrix, distortion),
        event_intrinsics=Intrinsics(event_matrix, distortion),
    )
    assert np.allclose(result.rotation_event_from_eye0, rotation, atol=1e-5)
    assert np.allclose(result.translation_event_from_eye0, translation, atol=1e-4)
