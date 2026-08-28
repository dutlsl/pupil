from gaze_mapping.gazer_3d.gazer_headset import Gazer3D
from gaze_mapping.matching import HybridBinocularMatcher, RealtimeMatcher


def _pupil(eye_id, timestamp, confidence=0.9):
    return {
        "id": eye_id,
        "timestamp": timestamp,
        "confidence": confidence,
    }


def _matches(matcher, pupil_datum):
    return list(matcher.on_pupil_datum(pupil_datum))


def _pupil_3d(eye_id, timestamp, confidence=0.9):
    pupil_datum = _pupil(eye_id, timestamp, confidence)
    pupil_datum.update(
        {
            "sphere": {"center": [1.0, 2.0, 3.0]},
            "circle_3d": {"normal": [0.0, 0.0, 1.0]},
        }
    )
    return pupil_datum


class _FittedBinocularModel:
    is_fitted = True

    def predict(self, features):
        return iter(({"norm_pos": [0.5, 0.5]},))


def test_hybrid_matcher_emits_only_when_eye0_arrives_and_reuses_eye1_snapshot():
    matcher = HybridBinocularMatcher(eye1_max_age_ms=50)
    eye1 = _pupil(1, 10.000)
    eye0_first = _pupil(0, 10.001)
    eye0_second = _pupil(0, 10.002)

    assert _matches(matcher, eye1) == []
    assert _matches(matcher, eye0_first) == [[eye0_first, eye1]]
    assert _matches(matcher, eye0_second) == [[eye0_second, eye1]]


def test_hybrid_matcher_updates_eye1_snapshot_without_emitting_gaze():
    matcher = HybridBinocularMatcher(eye1_max_age_ms=50)
    first_eye1 = _pupil(1, 20.000)
    updated_eye1 = _pupil(1, 20.010)
    eye0 = _pupil(0, 20.011)

    assert _matches(matcher, first_eye1) == []
    assert _matches(matcher, updated_eye1) == []
    assert _matches(matcher, eye0) == [[eye0, updated_eye1]]


def test_hybrid_matcher_falls_back_to_eye0_when_eye1_is_stale_or_future():
    matcher = HybridBinocularMatcher(eye1_max_age_ms=20)
    stale_eye1 = _pupil(1, 30.000)
    stale_eye0 = _pupil(0, 30.021)

    assert _matches(matcher, stale_eye1) == []
    assert _matches(matcher, stale_eye0) == [[stale_eye0]]

    future_eye1 = _pupil(1, 40.010)
    earlier_eye0 = _pupil(0, 40.009)
    assert _matches(matcher, future_eye1) == []
    assert _matches(matcher, earlier_eye0) == [[earlier_eye0]]


def test_hybrid_matcher_invalidates_with_a_newer_low_confidence_eye1_sample():
    matcher = HybridBinocularMatcher(eye1_max_age_ms=50)
    valid_eye1 = _pupil(1, 50.000)
    invalid_eye1 = _pupil(1, 50.010, confidence=0.2)
    eye0 = _pupil(0, 50.011)

    assert _matches(matcher, valid_eye1) == []
    assert _matches(matcher, invalid_eye1) == []
    assert _matches(matcher, eye0) == [[eye0]]


def test_hybrid_matcher_does_not_revive_eye1_after_a_newer_invalid_observation():
    matcher = HybridBinocularMatcher(eye1_max_age_ms=50)
    valid_eye1 = _pupil(1, 55.000)
    invalid_eye1 = _pupil(1, 55.010, confidence=0.2)
    delayed_valid_eye1 = _pupil(1, 55.005)
    eye0 = _pupil(0, 55.011)

    assert _matches(matcher, valid_eye1) == []
    assert _matches(matcher, invalid_eye1) == []
    assert _matches(matcher, delayed_valid_eye1) == []
    assert _matches(matcher, eye0) == [[eye0]]


def test_gazer3d_selects_hybrid_matcher_only_for_integrated_mode(monkeypatch):
    monkeypatch.delenv("PUPIL_HYBRID_INTEGRATED", raising=False)
    normal_gazer = Gazer3D.__new__(Gazer3D)
    normal_gazer.init_matcher()
    assert isinstance(normal_gazer.matcher, RealtimeMatcher)

    monkeypatch.setenv("PUPIL_HYBRID_INTEGRATED", "true")
    monkeypatch.setenv("PUPIL_HYBRID_EYE1_MAX_AGE_MS", "25")
    hybrid_gazer = Gazer3D.__new__(Gazer3D)
    hybrid_gazer.init_matcher()
    assert isinstance(hybrid_gazer.matcher, HybridBinocularMatcher)
    assert hybrid_gazer.matcher.eye1_max_age_seconds == 0.025


def test_gazer3d_preserves_eye0_timestamp_for_hybrid_binocular_output():
    eye0 = _pupil_3d(0, 60.010)
    eye1 = _pupil_3d(1, 60.000)

    hybrid_gazer = Gazer3D.__new__(Gazer3D)
    hybrid_gazer.matcher = HybridBinocularMatcher()
    hybrid_gazer.binocular_model = _FittedBinocularModel()
    hybrid_gaze = next(Gazer3D.predict(hybrid_gazer, iter([[eye0, eye1]])))
    assert hybrid_gaze["timestamp"] == eye0["timestamp"]

    regular_gazer = Gazer3D.__new__(Gazer3D)
    regular_gazer.matcher = RealtimeMatcher()
    regular_gazer.binocular_model = _FittedBinocularModel()
    regular_gaze = next(Gazer3D.predict(regular_gazer, iter([[eye0, eye1]])))
    assert abs(regular_gaze["timestamp"] - 60.005) < 1e-9
