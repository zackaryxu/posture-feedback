import json

import pytest

from posture_feedback import cli
from posture_feedback.core import KEYS, Monitor


def make_record(value=5, source='video'):
    m = Monitor({key: 2 for key in KEYS}, dwell=0)
    m.update({'head roll': value}, 0)
    m.update({'head roll': value}, 1)
    return cli.session_record(m, cli.datetime.now(cli.timezone.utc), 1, source)


def test_session_persistence_and_report(tmp_path):
    path = cli.session_path(tmp_path)
    cli.append_session(path, make_record())
    stored = json.loads(path.read_text().strip())
    assert stored['source_type'] == 'video'
    assert stored['segments'][0]['metrics']['head roll']['episodes'] == 1
    assert stored['active_seconds'] == 1
    out = cli.report(path)
    assert out.splitlines()[0] == '1 session'
    assert 'head roll: 1.0s in alert, 1 episode,' in out
    assert 'image' not in path.read_text() and 'frame' not in path.read_text()


def test_saved_session_keeps_segment_times_baselines_and_counters_together(tmp_path):
    m = Monitor({key: 2 for key in KEYS}, dwell=0)
    started = cli.datetime.now(cli.timezone.utc)
    m.update({'head roll': 5, 'shoulder tilt': 6}, 0)
    m.update({'head roll': 5, 'shoulder tilt': 6}, 1)
    m.calibrate({'head roll': 5}, 1)
    m.update({'head roll': 5}, 2)
    m.update({'head roll': 6}, 3)
    m.toggle_mode(3)
    m.update({'head roll': 5}, 4)
    record = cli.session_record(m, started, 4, 'video')
    path = cli.session_path(tmp_path)
    cli.append_session(path, record)
    stored = json.loads(path.read_text())
    assert stored['started_utc'] == started.isoformat()
    assert stored['duration_seconds'] == 4
    assert stored['active_seconds'] == 2
    assert [s['start_seconds'] for s in stored['segments']] == [0, 1, 3]
    assert stored['segments'][0]['metrics']['shoulder tilt']['samples'] == 2
    assert stored['segments'][1]['metrics']['shoulder tilt']['samples'] == 0
    assert stored['segments'][1]['baseline']['shoulder tilt'] is None
    for segment in stored['segments']:
        for stats in segment['metrics'].values():
            assert stats['alert_seconds'] <= stats['active_seconds'] <= segment['active_seconds']
    out = cli.report(path)
    assert 'absolute: 2 segments' in out and 'calibrated: 1 segment\n' in out


def test_report_signed_cancellation_has_absolute_magnitude_and_range(tmp_path):
    path = cli.session_path(tmp_path)
    cli.append_session(path, make_record(5))
    cli.append_session(path, make_record(-5))
    out = cli.report(path)
    assert out.startswith('2 sessions\n')
    assert '2.0s in alert, 2 episodes' in out
    assert 'mean absolute 5.0deg, range [-5.0, +5.0]' in out
    assert 'signed mean +0.0deg (mixed LEFT/RIGHT)' in out
    assert 'centered' not in out


def test_report_separates_demo_camera_video_and_legacy(tmp_path):
    path = cli.session_path(tmp_path)
    for source in ('demo', 'camera', 'video'):
        cli.append_session(path, make_record(source=source))
    legacy = make_record()['segments'][0]
    legacy['schema'] = 'posture-feedback-session.v1'
    for stats in legacy['metrics'].values():
        for key in ('mean_absolute_drift', 'min_drift', 'max_drift'):
            stats.pop(key)
    cli.append_session(path, legacy)
    out = cli.report(path)
    for source in ('demo', 'camera', 'video', 'legacy/unknown'):
        assert f'{source}: 1 session; 1.0s active' in out
    assert 'magnitude/range unavailable for legacy data' in out


def test_legacy_opposing_signed_means_do_not_imply_centered_posture(tmp_path):
    path = cli.session_path(tmp_path)
    for value in (5, -5):
        legacy = make_record(value)['segments'][0]
        legacy['schema'] = 'posture-feedback-session.v1'
        for stats in legacy['metrics'].values():
            for key in ('mean_absolute_drift', 'min_drift', 'max_drift'):
                stats.pop(key)
        cli.append_session(path, legacy)
    out = cli.report(path)
    assert 'absolute magnitude/range unavailable for legacy data' in out
    assert 'direction uncertain' in out
    assert 'centered' not in out


@pytest.mark.parametrize('args', [
    [], ['--camera', '0', '--demo'], ['--camera', '0', '--report'],
    ['--camera', '0', '--input-video', 'anything.mp4'], ['--demo', '--report'],
])
def test_missing_or_conflicting_input_never_reaches_capture(args, monkeypatch):
    def forbidden(_):
        pytest.fail('input validation must run before any capture/inference')
    monkeypatch.setattr(cli, 'run', forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_explicit_camera_zero_is_valid_but_parsing_does_not_open_it():
    parser = cli.build_parser()
    args = parser.parse_args(['--camera', '0'])
    cli.validate(args, parser)
    assert args.camera == 0 and not args.demo and not args.input_video


@pytest.mark.parametrize('args', [
    ['--camera', '-1'], ['--demo', '--roll-thresh', '0'],
    ['--demo', '--dwell', 'nan'], ['--demo', '--min-visibility', '1.5'],
    ['--demo', '--idle-exit', '-2'],
])
def test_invalid_cli_values_rejected(args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_headless_calibration_requires_automatic_interval(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['--demo', '--no-window', '--calibrated-default'])
    assert error.value.code == 2
    assert 'headless calibrated mode requires --auto-calibrate' in capsys.readouterr().err


def test_missing_video_rejected(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(['--input-video', str(tmp_path / 'missing.mp4'), '--no-window'])
    assert error.value.code == 2


def test_demo_loop_and_cleanup(tmp_path, monkeypatch, capsys):
    released = []
    monkeypatch.setattr(cli.DemoSource, 'release', lambda self: released.append(True))
    assert cli.main(['--demo', '--no-window', '--data-dir', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'Processed 180 frames' in out
    assert 'ALERT head roll' in out
    assert '1 episode,' in out
    assert released == [True]
    assert not (tmp_path / 'sessions.jsonl').exists()


def test_partial_automatic_calibration_in_full_loop_finishes_late_metrics(tmp_path, monkeypatch, capsys):
    original_read = cli.DemoSource.read

    def partial(self):
        ok, frame, values = original_read(self)
        if ok:
            values = {'head roll': values['head roll'], **(
                {'shoulder tilt': values['shoulder tilt']} if self.index > 45 else {})}
        return ok, frame, values

    monkeypatch.setattr(cli.DemoSource, 'read', partial)
    assert cli.main(['--demo', '--no-window', '--calibrated-default', '--auto-calibrate', '1',
                     '--save-demo', '--data-dir', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'Status: WAITING FOR BASELINES' in out
    assert 'Baseline ready: head roll' in out and 'Baseline ready: shoulder tilt' in out
    assert 'Baselines still waiting: head turn, lateral lean' in out
    row = json.loads((tmp_path / 'sessions.jsonl').read_text())
    segment = row['segments'][0]
    assert segment['baseline']['shoulder tilt'] is not None
    assert segment['metrics']['shoulder tilt']['samples'] > 0
    assert segment['metrics']['head turn']['samples'] == 0
    assert row['source_type'] == 'demo'


def test_one_frame_alert_is_saved_with_zero_active_duration(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.DemoSource, 'frames', 1)
    assert cli.main(['--demo', '--no-window', '--save-demo', '--dwell', '0', '--roll-thresh', '.1',
                     '--data-dir', str(tmp_path)]) == 0
    row = json.loads((tmp_path / 'sessions.jsonl').read_text())
    stats = row['segments'][0]['metrics']['head roll']
    assert row['active_seconds'] == 0 and row['duration_seconds'] == 0
    assert stats['samples'] == 1 and stats['episodes'] == 1
    assert 'Numeric summary saved' in capsys.readouterr().out


def test_no_person_status_and_idle_exit_never_claim_measuring(tmp_path, monkeypatch, capsys):
    original_read = cli.DemoSource.read

    def no_person(self):
        ok, frame, values = original_read(self)
        return ok, frame, {} if ok else values

    monkeypatch.setattr(cli.DemoSource, 'read', no_person)
    assert cli.main(['--demo', '--no-window', '--save-demo', '--idle-exit', '.2',
                     '--data-dir', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'Status: WAITING FOR PERSON / LANDMARKS' in out
    assert 'Status: MEASURING' not in out
    assert 'idle exit' in out and 'No judged samples; no session saved.' in out
    assert not (tmp_path / 'sessions.jsonl').exists()


def test_overlay_exposes_no_person_state(monkeypatch):
    import cv2
    import numpy as np
    labels = []
    monkeypatch.setattr(cv2, 'putText', lambda canvas, text, *a, **k: labels.append(text))
    m = Monitor({key: 2 for key in KEYS}, 1)
    m.update({}, 0)
    cli.draw_overlay(cv2, np.zeros((480, 640, 3), dtype=np.uint8), m, {}, {})
    assert any('WAITING FOR PERSON / LANDMARKS' in label for label in labels)
    assert any('no landmarks' in label for label in labels)


def test_small_video_overlay_has_space_for_status_and_controls():
    import cv2
    import numpy as np
    m = Monitor({key: 2 for key in KEYS}, 1)
    m.update({}, 0)
    rendered = cli.draw_overlay(cv2, np.zeros((240, 320, 3), dtype=np.uint8), m, {}, {})
    assert rendered.shape[0] >= 480 and rendered.shape[1] >= 640


def test_empty_input_cleanup(tmp_path, monkeypatch, capsys):
    released = []
    monkeypatch.setattr(cli.DemoSource, 'read', lambda self: (False, None, None))
    monkeypatch.setattr(cli.DemoSource, 'release', lambda self: released.append(True))
    assert cli.main(['--demo', '--no-window', '--data-dir', str(tmp_path)]) == 1
    assert 'yielded no frames' in capsys.readouterr().err
    assert released == [True]
    assert not (tmp_path / 'sessions.jsonl').exists()


def test_inference_initialization_failure_releases_video(tmp_path, monkeypatch, capsys):
    import cv2
    import mediapipe as mp
    released = []
    video = tmp_path / 'input.avi'
    video.touch()

    class Capture:
        def isOpened(self):
            return True
        def release(self):
            released.append(True)

    def fail(**kwargs):
        raise RuntimeError('model initialization failed')

    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: Capture())
    monkeypatch.setattr(mp.solutions.pose, 'Pose', fail)
    assert cli.main(['--input-video', str(video), '--no-window', '--data-dir', str(tmp_path)]) == 1
    assert 'model initialization failed' in capsys.readouterr().err
    assert released == [True]


def test_pause_then_calibrate_key_path_remains_paused(tmp_path, monkeypatch):
    import cv2
    keys = iter([ord(' '), ord('c'), ord('q')])
    monkeypatch.setattr(cli.DemoSource, 'frames', 3)
    for name in ('namedWindow', 'resizeWindow', 'setWindowProperty', 'imshow', 'destroyAllWindows'):
        monkeypatch.setattr(cv2, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda delay: next(keys))
    assert cli.main(['--demo', '--save-demo', '--data-dir', str(tmp_path)]) == 0
    row = json.loads((tmp_path / 'sessions.jsonl').read_text())
    calibrated = row['segments'][-1]
    assert calibrated['mode'] == 'calibrated'
    assert calibrated['active_seconds'] == 0
    assert all(stats['samples'] == 0 for stats in calibrated['metrics'].values())
