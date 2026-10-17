from types import SimpleNamespace
import math

import pytest

from posture_feedback.core import KEYS, Metric, Monitor, direction, extract, line_tilt


def landmark(x, y, visibility=1):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def monitor(**kwargs):
    return Monitor({key: 2 for key in KEYS}, dwell=1, **kwargs)


def test_metric_math_and_signed_direction():
    assert line_tilt((20, 30), (10, 20)) == pytest.approx(45)
    assert line_tilt((20, 20), (10, 30)) == pytest.approx(-45)
    face = [landmark(0, 0) for _ in range(478)]
    face[468], face[473] = landmark(.4, .4), landmark(.6, .45)
    face[1], face[234], face[454] = landmark(.55, .2), landmark(.3, .2), landmark(.7, .2)
    pose = [landmark(0, 0) for _ in range(33)]
    pose[0], pose[11], pose[12] = landmark(.55, .2), landmark(.3, .5), landmark(.7, .55)
    mirrored, _ = extract(face, pose, 100, 100, mirrored=True)
    true_view, _ = extract(face, pose, 100, 100, mirrored=False)
    assert all(mirrored[key] == pytest.approx(-true_view[key]) for key in mirrored)
    assert mirrored['head turn'] == pytest.approx(25)
    assert mirrored['lateral lean'] == pytest.approx(12.5)
    assert direction('lateral lean', mirrored['lateral lean']).endswith('RIGHT')


def test_visibility_gating_and_missing_presence():
    pose = [landmark(0, 0) for _ in range(33)]
    pose[0], pose[11], pose[12] = landmark(.5, .2), landmark(.3, .5, .1), landmark(.7, .5)
    assert extract(None, pose, 100, 100, min_visibility=.6)[0] == {}
    m = monitor()
    m.update({'head roll': 10}, 0)
    m.update({}, 2)
    assert m.active_seconds == 0
    assert m.metrics['head roll'].over_since is None
    assert len(m.metrics['head roll'].buffer) == 0
    assert m.status_text() == 'WAITING FOR PERSON / LANDMARKS'
    m.update({'head roll': 10}, 10)
    assert not m.metrics['head roll'].in_alert


def test_dwell_hysteresis_pause_and_resume():
    m = Metric('head roll', 4, dwell=2, smooth_n=1)
    assert not m.observe(6, 0, 0, True)[1]
    assert not m.observe(6, 0, 1, True)[1]
    assert m.observe(6, 0, 2, True)[1]
    assert m.observe(3, 0, 3, True)[1]  # hysteresis after onset
    assert not m.observe(1, 0, 4, True)[1]
    m.observe(6, 0, 5, True)
    m.pause(clear_buffer=True)
    assert not m.observe(6, 0, 20, True)[1]
    assert m.observe(6, 0, 22, True)[1]
    assert m.episodes == 2
    assert m.alert_seconds == pytest.approx(2)


@pytest.mark.parametrize('dip', [3, 4, 0, -5])
def test_prealert_dwell_resets_on_trigger_interruption(dip):
    m = Metric('head roll', 4, dwell=2, smooth_n=1)
    m.observe(5, 0, 0, True)
    assert not m.observe(dip, 0, 1, True)[1]
    assert not m.observe(5, 0, 2, True)[1]
    assert not m.observe(5, 0, 3, True)[1]
    assert m.observe(5, 0, 4, True)[1]
    assert m.episodes == 1


def test_below_threshold_samples_clear_pending_dwell():
    m = Metric('head roll', 4, dwell=2, smooth_n=1)
    for time, value in enumerate((5, 3, 3)):
        assert not m.observe(value, 0, time, True)[1]
    assert m.over_since is None and m.episodes == 0


def test_partial_calibration_late_metric_gets_baseline_and_is_judged():
    m = monitor()
    m.calibrate({'head roll': 10}, 0)
    assert 'shoulder tilt' in m.waiting_baselines
    assert m.metrics['shoulder tilt'].dev_n == 0
    m.update({'head roll': 10, 'shoulder tilt': 12}, 1)
    assert 'baselines pending' in m.status_text()
    assert m.metric_waiting_text('shoulder tilt') == 'baseline 0.0/2s'
    m.update({'head roll': 10, 'shoulder tilt': 12}, 3)
    assert m.metrics['shoulder tilt'].neutral == 12
    assert m.metrics['shoulder tilt'].dev_n == 1
    assert m.metrics['head roll'].neutral == 10
    assert m.metrics['head roll'].dev_n == 2
    assert 'head turn' in m.waiting_baselines  # explicit incomplete state
    result = m.update({'head roll': 10, 'shoulder tilt': 20}, 4)
    assert result['shoulder tilt'][0] > 2


def test_calibration_needs_continuous_detection_per_metric():
    m = monitor(calibrated_default=True, auto_calibrate=2)
    m.update({'head roll': 10}, 0)
    m.update({}, 1)
    m.update({'head roll': 10}, 2)
    assert m.update({'head roll': 10}, 3)["head roll"][0] is None
    m.update({'head roll': 10}, 4)
    assert m.metrics['head roll'].neutral == 10
    assert m.metrics['head roll'].dev_n == 1
    assert m.active_seconds == 0  # waiting time is not judged time
    m.update({'head roll': 11}, 5)
    assert m.active_seconds == 1


def test_pause_resets_unfinished_calibration_and_stops_judging():
    m = monitor(calibrated_default=True, auto_calibrate=2)
    m.update({'head roll': 10}, 0)
    m.toggle_pause()
    m.update({'head roll': 10}, 10)
    assert m.metrics['head roll'].neutral is None
    assert m.status_text() == 'PAUSED'
    m.toggle_pause()
    m.update({'head roll': 10}, 11)
    assert m.metrics['head roll'].neutral is None
    m.update({'head roll': 10}, 13)
    assert m.metrics['head roll'].neutral == 10
    m.toggle_pause()
    m.update({'head roll': 30}, 14)
    assert m.metrics['head roll'].dev_n == 1
    assert m.active_seconds == 0


def test_manual_calibration_can_be_queued_without_person():
    m = monitor()
    m.calibrate({}, 1)
    assert m.calibration_armed
    assert len(m.waiting_baselines) == 4
    m.update({'head roll': 5}, 2)
    m.update({'head roll': 5}, 4)
    assert m.metrics['head roll'].neutral == 5


def test_calibration_preserves_pause_until_explicit_resume():
    m = monitor()
    m.toggle_pause()
    m.calibrate({'head roll': 3}, 1)
    assert not m.active
    m.update({'head roll': 20}, 2)
    assert m.metrics['head roll'].dev_n == 0
    assert m.status_text() == 'PAUSED'
    m.toggle_pause()
    m.update({'head roll': 20}, 3)
    assert m.metrics['head roll'].dev_n == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
@pytest.mark.parametrize('kind,index,missing', [
    ('face', 1, 'head turn'), ('face', 468, 'head roll'),
    ('pose', 0, 'lateral lean'), ('pose', 11, 'shoulder tilt'),
])
def test_nonfinite_landmarks_are_excluded_before_drawing(bad, kind, index, missing):
    import cv2
    import numpy as np
    from posture_feedback.cli import draw_overlay
    face = [landmark(.5, .5) for _ in range(478)]
    face[468], face[473] = landmark(.4, .4), landmark(.6, .45)
    face[1], face[234], face[454] = landmark(.55, .2), landmark(.3, .2), landmark(.7, .2)
    pose = [landmark(.5, .5) for _ in range(33)]
    pose[0], pose[11], pose[12] = landmark(.55, .2), landmark(.3, .5), landmark(.7, .55)
    (face if kind == 'face' else pose)[index].x = bad
    values, lines = extract(face, pose, 640, 480)
    assert missing not in values and missing not in lines
    assert all(math.isfinite(v) for v in values.values())
    m = monitor()
    statuses = m.update(values, 0)
    rendered = draw_overlay(cv2, np.zeros((480, 640, 3), dtype=np.uint8), m, statuses, lines)
    assert rendered.shape == (480, 640, 3)


def test_nonfinite_visibility_does_not_claim_a_valid_pose():
    pose = [landmark(.5, .5) for _ in range(33)]
    pose[11], pose[12] = landmark(.3, .5), landmark(.7, .5, float('nan'))
    assert extract(None, pose, 640, 480) == ({}, {})


def test_collapsed_eye_and_shoulder_spans_are_not_judged():
    face = [landmark(.5, .5) for _ in range(478)]
    face[234], face[454] = landmark(.3, .2), landmark(.7, .2)
    face[468], face[473] = landmark(.5, .4), landmark(.5, .5)
    face[1] = landmark(.55, .3)
    values, lines = extract(face, None, 640, 480)
    assert 'head roll' not in values and 'head roll' not in lines
    assert 'head turn' in values  # independent valid geometry remains usable
    pose = [landmark(.5, .5) for _ in range(33)]
    pose[11], pose[12], pose[0] = landmark(.498, .5), landmark(.502, .5), landmark(.51, .2)
    assert extract(None, pose, 640, 480) == ({}, {})


def test_well_separated_landmarks_keep_roll_and_lean():
    face = [landmark(.5, .5) for _ in range(478)]
    face[234], face[454] = landmark(.3, .2), landmark(.7, .2)
    face[468], face[473], face[1] = landmark(.4, .4), landmark(.6, .45), landmark(.55, .3)
    pose = [landmark(.5, .5) for _ in range(33)]
    pose[11], pose[12], pose[0] = landmark(.3, .5), landmark(.7, .55), landmark(.55, .2)
    values, _ = extract(face, pose, 640, 480)
    assert set(values) == set(KEYS)
    assert values['head roll'] > 3
    assert values['lateral lean'] == pytest.approx(12.5)


def test_implausible_ratio_is_skipped_instead_of_becoming_an_extreme_alert():
    pose = [landmark(.5, .5) for _ in range(33)]
    pose[11], pose[12], pose[0] = landmark(.3, .5), landmark(.5, .5), landmark(.99, .2)
    values, lines = extract(None, pose, 640, 480)
    assert 'shoulder tilt' in values
    assert 'lateral lean' not in values and 'lateral lean' not in lines


def assert_consistent_segments(m):
    for segment in m.segments():
        assert 0 <= segment['active_seconds'] <= segment['duration_seconds']
        for stats in segment['metrics'].values():
            assert 0 <= stats['alert_seconds'] <= stats['active_seconds'] <= segment['active_seconds']


def test_recalibration_closes_all_old_counters_and_starts_clean_segment():
    m = monitor()
    m.update({'head roll': 5, 'shoulder tilt': 6}, 0)
    m.update({'head roll': 5, 'shoulder tilt': 6}, 1)
    m.update({'head roll': 5, 'shoulder tilt': 6}, 2)
    old = m.segments()[0]
    m.calibrate({'head roll': 5}, 2)
    assert m.completed_segments[0] == old
    assert m.active_seconds == 0 and m.total_active_seconds == 2
    assert all(metric.dev_n == 0 for metric in m.metrics.values())
    assert m.metrics['shoulder tilt'].neutral is None
    m.update({'head roll': 5}, 3)
    m.update({'head roll': 5}, 4)
    m.calibrate({'head roll': 8}, 4)
    assert [s['start_seconds'] for s in m.segments()] == [0, 2, 4]
    assert [s['end_seconds'] for s in m.segments()] == [2, 4, 4]
    assert m.completed_segments[0] == old  # preserved after repeated recalibration
    assert m.total_active_seconds == 3
    assert_consistent_segments(m)


def test_mode_switches_preserve_stats_clear_baselines_and_do_not_join_timers():
    m = monitor()
    m.update({'head roll': 5}, 0)
    m.update({'head roll': 5}, 1)
    m.toggle_mode(1)
    m.update({'head roll': 5}, 2)
    assert m.status_text() == 'WAITING FOR CALIBRATION (press c)'
    assert m.has_samples  # the previous absolute segment has samples
    assert m.metrics['head roll'].dev_n == 0
    m.calibrate({'head roll': 5}, 2)
    m.update({'head roll': 5}, 3)
    m.update({'head roll': 6}, 4)
    m.toggle_mode(4)
    m.update({'head roll': 6}, 5)
    assert m.metrics['head roll'].over_since == 5  # no old dwell carried forward
    m.toggle_mode(5)
    assert m.waiting_baselines == list(KEYS)
    assert m.total_active_seconds == 2
    assert_consistent_segments(m)


def test_magnitude_and_range_do_not_cancel_signed_drift():
    m = Metric('head roll', 2, 0, smooth_n=1)
    m.observe(5, 0, 0, True)
    m.observe(-5, 0, 1, True)
    stats = m.summary()
    assert stats['mean_drift'] == 0
    assert stats['mean_absolute_drift'] == 5
    assert (stats['min_drift'], stats['max_drift']) == (-5, 5)
