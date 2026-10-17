"""Video/camera loop and command-line interface."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

from .core import KEYS, UNITS, Monitor, direction, extract


def data_directory():
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "posture-feedback"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "posture-feedback"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "posture-feedback"


def session_path(data_dir):
    return Path(data_dir) / "sessions.jsonl"


def session_record(monitor, started, duration, source_type):
    return {"schema": "posture-feedback-session.v2", "source_type": source_type,
            "started_utc": started.isoformat(), "duration_seconds": duration,
            "active_seconds": monitor.total_active_seconds,
            "segments": monitor.segments(duration)}


def append_session(path, record):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def format_statistics(segments):
    lines = []
    for key in KEYS:
        metrics = [s["metrics"][key] for s in segments]
        samples = sum(m["samples"] for m in metrics)
        if not samples:
            lines.append(f"{key}: no judged samples")
            continue
        total = sum(m["alert_seconds"] for m in metrics)
        episodes = sum(m["episodes"] for m in metrics)
        signed = sum(m["mean_drift"] * m["samples"] for m in metrics) / samples
        sampled = [m for m in metrics if m["samples"]]
        if all("mean_absolute_drift" in m for m in sampled):
            absolute = sum(m["mean_absolute_drift"] * m["samples"] for m in sampled) / samples
            low = min(m["min_drift"] for m in sampled)
            high = max(m["max_drift"] for m in sampled)
            magnitude = f"mean absolute {absolute:.1f}{UNITS[key]}, range [{low:+.1f}, {high:+.1f}]"
            side = "mixed LEFT/RIGHT" if low < 0 < high else direction(key, signed)
        else:
            magnitude = "absolute magnitude/range unavailable for legacy data"
            side = "legacy aggregate; direction uncertain" if signed == 0 else "net " + direction(key, signed)
        lines.append(f"{key}: {total:.1f}s in alert, {episodes} episode{'s' if episodes != 1 else ''}, "
                     f"{magnitude}; signed mean {signed:+.1f}{UNITS[key]} ({side})")
    return lines


def report(path):
    if not path.exists():
        return "No sessions recorded."
    sessions = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid session record on line {number}") from exc
            if row.get("schema") in ("posture-feedback-session.v1", "posture-feedback-session.v2"):
                sessions.append(row)
    if not sessions:
        return "No sessions recorded."
    lines = [f"{len(sessions)} session{'s' if len(sessions) != 1 else ''}"]
    # Synthetic data is always reported separately from actual video/camera runs.
    sources = sorted({row.get("source_type", "legacy/unknown") for row in sessions})
    for source in sources:
        rows = [r for r in sessions if r.get("source_type", "legacy/unknown") == source]
        segments = [s for row in rows for s in row.get("segments", [row])]
        lines.append(f"{source}: {len(rows)} session{'s' if len(rows) != 1 else ''}; "
                     f"{sum(r['active_seconds'] for r in rows):.1f}s active")
        for mode in sorted({s["mode"] for s in segments}):
            selected = [s for s in segments if s["mode"] == mode]
            lines.append(f"  {mode}: {len(selected)} segment{'s' if len(selected) != 1 else ''}")
            lines.extend("  " + line for line in format_statistics(selected))
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description="Local posture drift feedback (numeric summaries only).")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-video", type=Path, help="process a local video file")
    inputs.add_argument("--demo", action="store_true", help="deterministic synthetic frame and landmark demo")
    inputs.add_argument("--report", action="store_true", help="show aggregate numeric session report")
    p.add_argument("--data-dir", type=Path, default=data_directory())
    p.add_argument("--save-demo", action="store_true", help="persist synthetic demo summary")
    inputs.add_argument("--camera", type=int, help="explicit camera index (no default camera access)")
    p.add_argument("--no-window", action="store_true", help="run without GUI")
    p.add_argument("--no-mirror", action="store_true", help="use camera/video's original orientation")
    p.add_argument("--calibrated-default", action="store_true")
    p.add_argument("--auto-calibrate", type=float, default=0.0, metavar="SECONDS")
    p.add_argument("--roll-thresh", type=float, default=3.0)
    p.add_argument("--turn-thresh", type=float, default=8.0)
    p.add_argument("--shoulder-thresh", type=float, default=3.0)
    p.add_argument("--lean-thresh", type=float, default=6.0)
    p.add_argument("--dwell", type=float, default=None, metavar="SECONDS")
    p.add_argument("--min-visibility", type=float, default=0.6)
    p.add_argument("--model-complexity", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--seconds", type=float, default=0.0, help="stop after this much input time")
    p.add_argument("--idle-exit", type=float, default=600.0, help="stop after this long without detections; 0 disables")
    p.add_argument("--start-mini", action="store_true")
    p.add_argument("--start-diminished", action="store_true")
    p.add_argument("--window-width", type=int, default=640)
    p.add_argument("--window-height", type=int, default=480)
    p.add_argument("--mini-width", type=int, default=320)
    p.add_argument("--mini-height", type=int, default=200)
    return p


def validate(args, parser):
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    for name in ("roll_thresh", "turn_thresh", "shoulder_thresh", "lean_thresh"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("auto_calibrate", "seconds", "idle_exit"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.dwell is not None and (not math.isfinite(args.dwell) or args.dwell < 0):
        parser.error("--dwell must be finite and nonnegative")
    if not math.isfinite(args.min_visibility) or not 0 <= args.min_visibility <= 1:
        parser.error("--min-visibility must be within 0..1")
    for name in ("window_width", "window_height", "mini_width", "mini_height"):
        if getattr(args, name) < 120:
            parser.error(f"--{name.replace('_', '-')} must be at least 120")
    if not args.report and args.no_window and args.calibrated_default and not args.auto_calibrate:
        parser.error("headless calibrated mode requires --auto-calibrate SECONDS (>0)")
    if args.save_demo and not args.demo:
        parser.error("--save-demo requires --demo")
    if args.input_video and not args.input_video.is_file():
        parser.error(f"video file does not exist: {args.input_video}")


class DemoSource:
    """Synthetic graphics and values; no personal media or model files."""
    fps = 15.0
    frames = 180

    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np
        self.index = 0

    def read(self):
        if self.index >= self.frames:
            return False, None, None
        index = self.index
        self.index += 1
        frame = self.np.full((480, 640, 3), (28, 32, 38), dtype=self.np.uint8)
        phase = "centered" if index < 45 else "held drift" if index < 135 else "reset"
        self.cv2.putText(frame, f"SYNTHETIC DEMO - {phase}", (12, frame.shape[0] - 48),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
        drift = 1 if 45 <= index < 135 else 0
        self.cv2.line(frame, (220, 270), (420, 270 + 32 * drift), (120, 180, 230), 5)
        self.cv2.circle(frame, (320 + 35 * drift, 170), 55, (120, 180, 230), 3)
        self.cv2.circle(frame, (300 + 35 * drift, 155), 4, (240, 240, 240), -1)
        self.cv2.circle(frame, (340 + 35 * drift, 155 + 15 * drift), 4, (240, 240, 240), -1)
        values = {"head roll": 0.4 + drift * 14, "head turn": drift * 24,
                  "shoulder tilt": drift * 9, "lateral lean": drift * 18}
        return True, frame, values

    def release(self):
        pass


def draw_overlay(cv2, frame, monitor, statuses, lines, mini=False):
    scale = max(1.0, 640 / frame.shape[1], 480 / frame.shape[0])
    canvas = cv2.resize(frame, None, fx=scale, fy=scale) if scale > 1 else frame.copy()
    if mini:
        canvas[:] = (30, 34, 40)
    else:
        # Keep text readable against a bright or busy camera background.
        shade = canvas.copy()
        cv2.rectangle(shade, (0, 0), (canvas.shape[1], 154), (30, 34, 40), -1)
        cv2.rectangle(shade, (0, canvas.shape[0] - 30),
                      (canvas.shape[1], canvas.shape[0]), (30, 34, 40), -1)
        cv2.addWeighted(shade, 0.85, canvas, 0.15, 0, canvas)
    for key, endpoints in lines.items():
        if not all(math.isfinite(value) and abs(value * scale) < 2**30
                   for point in endpoints for value in point):
            continue
        status = statuses.get(key)
        color = (0, 0, 255) if status and status[1] else (0, 180, 255) if status and monitor.metrics[key].over_since is not None else (0, 200, 0)
        if not mini:
            points = [tuple(int(value * scale) for value in point) for point in endpoints]
            cv2.line(canvas, points[0], points[1], color, 2)
    state = monitor.status_text().split(";")[0]
    cv2.putText(canvas, f"{monitor.mode.upper()}  {state}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    for row, key in enumerate(KEYS):
        status = statuses.get(key)
        label = monitor.metric_waiting_text(key) if status is None or status[0] is None else f"{status[0]:+.1f}{UNITS[key]}"
        suffix = " ALERT" if status and status[1] else ""
        color = (0, 0, 255) if suffix else (220, 220, 220)
        cv2.putText(canvas, f"{key}: {label}{suffix}", (12, 60 + row * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if not mini:
        cv2.putText(canvas, "space pause  c calibrate  a mode  m compact  q quit",
                    (12, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
    return canvas


def run(args):
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import cv2
    import numpy as np

    thresholds = {"head roll": args.roll_thresh, "head turn": args.turn_thresh,
                  "shoulder tilt": args.shoulder_thresh, "lateral lean": args.lean_thresh}
    dwell = args.dwell if args.dwell is not None else (2.0 if args.demo else 120.0)
    monitor = Monitor(thresholds, dwell, args.calibrated_default, args.auto_calibrate)
    source_type = "demo" if args.demo else "video" if args.input_video else "camera"
    started = datetime.now(timezone.utc)
    source = None
    pose = face = None
    cv_windows = False
    processed = 0
    last_time = 0.0
    absent_since = None
    last_status = None
    mini = args.start_mini
    diminished = args.start_diminished
    size_factor = 1.0
    latest_values = {}
    try:
        if args.demo:
            source = DemoSource(cv2, np)
            print("Synthetic demo: 12 seconds, centered / held drift / reset; dwell=", dwell)
        else:
            import mediapipe as mp
            source = cv2.VideoCapture(str(args.input_video) if args.input_video else args.camera)
            if not source.isOpened():
                raise RuntimeError("could not open video" if args.input_video else f"could not open camera {args.camera}")
            if not args.input_video:
                source.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                source.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            pose = mp.solutions.pose.Pose(model_complexity=args.model_complexity,
                                          min_detection_confidence=0.6, min_tracking_confidence=0.6)
            face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                   min_detection_confidence=0.6, min_tracking_confidence=0.6)
        fps = source.fps if args.demo else source.get(cv2.CAP_PROP_FPS) if args.input_video else 0.0
        if not fps or not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        clock_start = time.monotonic()
        while True:
            if args.demo:
                ok, frame, raw_values = source.read()
            else:
                ok, frame = source.read()
                raw_values = None
            if not ok or frame is None:
                if processed == 0:
                    raise RuntimeError("input opened but yielded no frames")
                if not (args.demo or args.input_video):
                    raise RuntimeError("camera stopped returning frames")
                break
            now = processed / fps if (args.demo or args.input_video) else time.monotonic() - clock_start
            if args.seconds and now >= args.seconds:
                break
            processed += 1
            last_time = now
            if not args.demo:
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pres = pose.process(rgb)
                fres = face.process(rgb)
                face_lms = fres.multi_face_landmarks[0].landmark if fres.multi_face_landmarks else None
                pose_lms = pres.pose_landmarks.landmark if pres.pose_landmarks else None
                raw_values, lines = extract(face_lms, pose_lms, frame.shape[1], frame.shape[0],
                                            mirrored=not args.no_mirror, min_visibility=args.min_visibility)
            else:
                lines = {}
            latest_values = raw_values
            previous_baselines = set(monitor.baseline_captured_at)
            statuses = monitor.update(raw_values, now)
            for metric_key in set(monitor.baseline_captured_at) - previous_baselines:
                print(f"Baseline ready: {metric_key} at {now:.1f}s")
            current_status = monitor.status_text()
            if current_status != last_status:
                print("Status: " + current_status)
                last_status = current_status
            if raw_values:
                absent_since = None
            else:
                if absent_since is None:
                    absent_since = now
                if args.idle_exit and now - absent_since >= args.idle_exit:
                    print("idle exit: no measurements available")
                    break
            for key, status in statuses.items():
                if status and status[3]:
                    print(f"ALERT {key}: {direction(key, status[0])} at {now:.1f}s")
            if not args.no_window:
                name = "Posture feedback"
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv_windows = True
                image = draw_overlay(cv2, frame, monitor, statuses, lines, mini)
                width = args.mini_width if mini else args.window_width
                height = args.mini_height if mini else args.window_height
                scale = (0.5 if diminished else 1.0) * size_factor
                cv2.resizeWindow(name, max(160, int(width * scale)), max(120, int(height * scale)))
                try:
                    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 0 if diminished else 1)
                except cv2.error:
                    pass
                cv2.imshow(name, image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):
                    monitor.toggle_mode(now)
                elif key == ord("c"):
                    monitor.calibrate(latest_values, now)
                    print("Calibration requested; missing baselines wait for continuous landmarks.")
                elif key == ord(" "):
                    monitor.toggle_pause()
                elif key == ord("m"):
                    mini = not mini
                elif key == ord("d"):
                    diminished = not diminished
                elif key in (ord("+"), ord("="), ord("]")):
                    size_factor = min(3.0, size_factor * 1.25)
                elif key in (ord("-"), ord("_"), ord("[")):
                    size_factor = max(0.3, size_factor / 1.25)
                elif key == ord("0"):
                    size_factor = 1.0
    finally:
        if source is not None:
            source.release()
        if pose is not None:
            pose.close()
        if face is not None:
            face.close()
        if cv_windows:
            cv2.destroyAllWindows()
    record = session_record(monitor, started, last_time, source_type)
    print(f"Processed {processed} frames; {monitor.total_active_seconds:.1f}s active; source={source_type}")
    for mode in sorted({s["mode"] for s in record["segments"]}):
        print(f"{mode}:")
        print("\n".join(format_statistics([s for s in record["segments"] if s["mode"] == mode])))
    if monitor.waiting_baselines:
        print("Baselines still waiting: " + ", ".join(monitor.waiting_baselines))
    if monitor.has_samples and (not args.demo or args.save_demo):
        path = session_path(args.data_dir)
        append_session(path, record)
        print(f"Numeric summary saved: {path}")
    elif not monitor.has_samples:
        print("No judged samples; no session saved.")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)
    try:
        if args.report:
            print(report(session_path(args.data_dir)))
            return 0
        return run(args)
    except Exception as exc:
        print(f"posture-feedback: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
