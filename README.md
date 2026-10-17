# Posture feedback

Local posture drift feedback from a camera, video file, or synthetic demo. Frames stay
in memory; saved sessions contain numeric summaries only.

## Install

Python 3.11 is the tested interpreter. On macOS/Linux:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install .
```

The pinned OpenCV contrib wheel includes GUI support. A desktop display is needed
for the window; use `--no-window` on headless systems. MediaPipe may fetch its
Pose model on first use.

## Run

```sh
./posture-monitor.command --demo --no-window
./posture-monitor.command --input-video /path/to/video.mp4 --no-window
./posture-monitor.command --camera 0
./posture-monitor.command --report
```

Choose exactly one mode: `--camera INDEX`, `--demo`, `--input-video FILE`,
or `--report`. Bare invocation prints usage without accessing a camera.
`--demo` uses generated graphics and values; it does not save a session unless
`--save-demo` is passed. Default thresholds are 3° head roll, 8% head turn,
3° shoulder tilt, and 6% lateral lean, with 120 seconds of dwell. The demo
uses 2 seconds of dwell. Change these with `--roll-thresh`, `--turn-thresh`,
`--shoulder-thresh`, `--lean-thresh`, and `--dwell`. `--data-dir` selects the
numeric-session directory. `--idle-exit` and `--seconds` stop unattended runs.

Window keys: Space pauses/resumes, `c` starts a new calibration, `a` toggles
absolute/calibrated baseline, `m` switches to a compact overlay, and `q` quits.
For display adjustments, `d` changes prominence, `+`/`-` resize, and `0` resets size. Use
`--auto-calibrate 5` to capture each metric's baseline after
five continuous seconds of valid landmarks. `--auto-calibrate` also selects
calibrated mode. Headless calibrated runs require it. `--no-mirror` displays the
original orientation.

Manual calibration captures currently visible metrics immediately. Missing metrics
wait for two continuous seconds of valid landmarks (or the `--auto-calibrate`
interval). Ready metrics work while others wait; each missing baseline is shown
explicitly. Loss of landmarks or pausing resets unfinished calibration intervals.
Recalibration and mode switches preserve previous statistics in separate segments.
Each calibrated segment starts with new baselines; `c` arms calibration when no
automatic interval was requested. The report separates demo, video, and camera
sessions, and shows absolute drift magnitude, signed mean/range, and alert seconds.
Any judged sample or alert event can be saved, including a single-frame run.

## Test

```sh
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
.venv/bin/python -c 'import cv2, mediapipe as mp, numpy; print(cv2.__version__, mp.__version__, numpy.__version__, hasattr(mp, "solutions"))'
./posture-monitor.command --demo --no-window --data-dir /tmp/posture-demo-data
```

## Limitations

This is visual drift feedback, not a medical assessment. A frontal, well-lit,
reasonably level view is needed. Missing, non-finite, low-visibility, and undersized
landmark spans are skipped; extreme turn/lean ratios are also excluded. These checks
reduce tracking errors but do not establish anatomical accuracy. Absolute mode is
sensitive to camera angle. Calibration captures the current pose and does not
prove it is ideal. Head turn and lean are approximate 2D ratios. Headless runs
cannot accept keyboard controls. Video timing assumes the file's frame rate.
FaceMesh provides no per-landmark visibility score; face availability and geometry
gate face metrics. A missing face does not prevent visible pose metrics from working.
