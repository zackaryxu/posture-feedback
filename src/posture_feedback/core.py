"""Metric extraction and state machine; independent of camera and GUI."""
from collections import deque
from dataclasses import dataclass, field
import math

KEYS = ("head roll", "head turn", "shoulder tilt", "lateral lean")
UNITS = {"head roll": "deg", "head turn": "%", "shoulder tilt": "deg", "lateral lean": "%"}


def line_tilt(a, b):
    """Positive when the image-right endpoint is lower in the image."""
    left, right = sorted((a, b), key=lambda p: p[0])
    return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))


def direction(key, signed):
    if signed == 0:
        return "centered"
    side = "RIGHT" if signed > 0 else "LEFT"
    return {
        "head roll": f"head tilting to your {side}",
        "head turn": f"turned to your {side}",
        "shoulder tilt": f"{side.lower()} shoulder dropping",
        "lateral lean": f"leaning to your {side}",
    }[key]


def extract(face, pose, width, height, mirrored=True, min_visibility=0.6):
    """Return signed metrics and draw points from MediaPipe-like landmarks.

    Invalid/low-confidence metrics are absent. Nose/shoulders use Pose indices
    0, 11, 12; FaceMesh uses iris 468/473, nose 1, cheeks 234/454.
    """
    hand = 1 if mirrored else -1
    values, lines = {}, {}
    def finite_points(*points):
        return all(math.isfinite(coordinate) for point in points for coordinate in point)
    if face is not None and len(face) > 473:
        def fp(i):
            return face[i].x * width, face[i].y * height
        iris_a, iris_b = fp(468), fp(473)
        nose, cheek_a, cheek_b = fp(1), fp(234), fp(454)
        if finite_points(cheek_a, cheek_b):
            center = (cheek_a[0] + cheek_b[0]) / 2
            half_width = abs(cheek_b[0] - cheek_a[0]) / 2
            if half_width * 2 >= max(12.0, width * 0.02):
                if (finite_points(iris_a, iris_b)
                        and abs(iris_b[0] - iris_a[0]) >= max(3.0, half_width * 0.1)):
                    values["head roll"] = line_tilt(iris_a, iris_b) * hand
                    lines["head roll"] = (iris_a, iris_b)
                if finite_points(nose):
                    turn = (nose[0] - center) / half_width * 100 * hand
                    if math.isfinite(turn) and abs(turn) <= 150:
                        values["head turn"] = turn
                        lines["head turn"] = ((center, nose[1]), nose)
    if pose is not None and len(pose) > 12:
        def pp(i):
            return pose[i].x * width, pose[i].y * height
        left, right, nose = pp(11), pp(12), pp(0)
        shoulder_width = abs(left[0] - right[0])
        if (finite_points(left, right) and shoulder_width >= max(12.0, width * 0.02)
                and all(math.isfinite(pose[i].visibility) and pose[i].visibility >= min_visibility
                        for i in (11, 12))):
            values["shoulder tilt"] = line_tilt(left, right) * hand
            lines["shoulder tilt"] = (left, right)
            if finite_points(nose) and math.isfinite(pose[0].visibility) and pose[0].visibility >= min_visibility:
                center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
                lean = (nose[0] - center[0]) / shoulder_width * 100 * hand
                if math.isfinite(lean) and abs(lean) <= 150:
                    values["lateral lean"] = lean
                    lines["lateral lean"] = (nose, center)
    return values, lines


@dataclass
class Metric:
    key: str
    threshold: float
    dwell: float
    hysteresis: float = 0.5
    smooth_n: int = 8
    neutral: float | None = None
    buffer: deque = field(default_factory=deque)
    over_since: float | None = None
    over_sign: int = 0
    in_alert: bool = False
    episodes: int = 0
    alert_seconds: float = 0.0
    active_seconds: float = 0.0
    dev_sum: float = 0.0
    dev_abs_sum: float = 0.0
    min_dev: float | None = None
    max_dev: float | None = None
    dev_n: int = 0
    last_t: float | None = None

    def __post_init__(self):
        if (not math.isfinite(self.threshold) or self.threshold <= 0
                or not math.isfinite(self.dwell) or self.dwell < 0):
            raise ValueError("threshold must be positive and dwell nonnegative")
        if not 0 < self.hysteresis <= 1 or self.smooth_n < 1:
            raise ValueError("invalid hysteresis or smoothing length")
        self.buffer = deque(maxlen=self.smooth_n)

    def pause(self, clear_buffer=False):
        self.over_since = None
        self.over_sign = 0
        self.in_alert = False
        self.last_t = None
        if clear_buffer:
            self.buffer.clear()

    def observe(self, raw, baseline, now, active):
        if not math.isfinite(raw):
            self.pause(clear_buffer=True)
            return None
        self.buffer.append(raw)
        value = sum(self.buffer) / len(self.buffer)
        dev = value - baseline if baseline is not None else None
        if not active or dev is None:
            self.pause()
            return (dev, False, 0.0, False)
        dt = 0.0 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        self.active_seconds += dt
        # Attribute each elapsed interval to the state observed at its start.
        if self.in_alert:
            self.alert_seconds += dt
        self.dev_sum += dev
        self.dev_abs_sum += abs(dev)
        self.min_dev = dev if self.min_dev is None else min(self.min_dev, dev)
        self.max_dev = dev if self.max_dev is None else max(self.max_dev, dev)
        self.dev_n += 1
        sign = 1 if dev > 0 else -1
        if self.in_alert and (abs(dev) < self.threshold * self.hysteresis or sign != self.over_sign):
            self.in_alert = False
            self.over_since = None
        # Hysteresis retains an existing alert only. Dwell requires consecutive
        # smoothed samples above the trigger, on the same side of baseline.
        if not self.in_alert:
            if abs(dev) > self.threshold:
                if self.over_since is None or sign != self.over_sign:
                    self.over_since = now
                    self.over_sign = sign
            else:
                self.over_since = None
                self.over_sign = 0
        held = max(0.0, now - self.over_since) if self.over_since is not None else 0.0
        alert = self.over_since is not None and held >= self.dwell
        onset = alert and not self.in_alert
        if onset:
            self.episodes += 1
        self.in_alert = alert
        return (dev, alert, held, onset)

    @property
    def mean_dev(self):
        return self.dev_sum / self.dev_n if self.dev_n else 0.0

    def summary(self):
        return {"threshold": self.threshold, "mean_drift": self.mean_dev,
                "mean_absolute_drift": self.dev_abs_sum / self.dev_n if self.dev_n else 0.0,
                "min_drift": self.min_dev, "max_drift": self.max_dev,
                "alert_seconds": self.alert_seconds, "episodes": self.episodes,
                "samples": self.dev_n, "active_seconds": self.active_seconds}


class Monitor:
    """A run has separate segments whenever mode or calibration is changed.

    Manual calibration captures visible metrics immediately. Missing metrics
    capture their first baseline after a continuous valid interval, independently
    of other metrics. Their counters stay at zero while waiting.
    """

    def __init__(self, thresholds, dwell, calibrated_default=False, auto_calibrate=0.0):
        self.thresholds = dict(thresholds)
        self.dwell = dwell
        self.auto_calibrate = auto_calibrate
        self.calibration_delay = auto_calibrate or 2.0
        self.mode = "calibrated" if calibrated_default or auto_calibrate else "absolute"
        self.active = True
        self.completed_segments = []
        self.now = 0.0
        self._start_segment(0.0)

    def _start_segment(self, now):
        self.metrics = {key: Metric(key, self.thresholds[key], self.dwell) for key in KEYS}
        self.segment_start = now
        self.active_seconds = 0.0
        self.last_active_t = None
        self.last_values = {}
        self.calibration_since = {}
        self.baseline_captured_at = {}
        self.calibration_armed = bool(self.auto_calibrate)

    @property
    def waiting_baselines(self):
        return [k for k, m in self.metrics.items() if self.mode == "calibrated" and m.neutral is None]

    def _segment_record(self, end):
        return {"mode": self.mode, "start_seconds": self.segment_start,
                "end_seconds": end, "duration_seconds": end - self.segment_start,
                "active_seconds": self.active_seconds,
                "baseline": {k: (0.0 if self.mode == "absolute" else m.neutral)
                             for k, m in self.metrics.items()},
                "baseline_captured_seconds": dict(self.baseline_captured_at),
                "missing_baselines": self.waiting_baselines,
                "metrics": {k: m.summary() for k, m in self.metrics.items()}}

    def segments(self, end=None):
        return [*self.completed_segments, self._segment_record(self.now if end is None else end)]

    @property
    def total_active_seconds(self):
        return sum(s["active_seconds"] for s in self.completed_segments) + self.active_seconds

    @property
    def has_samples(self):
        return any(m["samples"] or m["episodes"] for s in self.segments() for m in s["metrics"].values())

    def calibrate(self, values, now=None):
        now = self.now if now is None else now
        self.completed_segments.append(self._segment_record(now))
        self.mode = "calibrated"
        self._start_segment(now)
        self.now = now
        self.calibration_armed = True
        for key, raw in values.items():
            if key in self.metrics and math.isfinite(raw):
                self.metrics[key].neutral = raw
                self.baseline_captured_at[key] = now
        self.last_values = {k: v for k, v in values.items() if k in self.metrics and math.isfinite(v)}

    def toggle_mode(self, now=None):
        now = self.now if now is None else now
        self.completed_segments.append(self._segment_record(now))
        self.mode = "calibrated" if self.mode == "absolute" else "absolute"
        self._start_segment(now)
        self.now = now

    def toggle_pause(self):
        self.active = not self.active
        self.last_active_t = None
        self.calibration_since.clear()
        for metric in self.metrics.values():
            metric.pause(clear_buffer=True)

    def status_text(self):
        if not self.active:
            return "PAUSED"
        if not self.last_values:
            return "WAITING FOR PERSON / LANDMARKS"
        if self.mode == "calibrated":
            ready = set(self.last_values) - set(self.waiting_baselines)
            if not ready:
                return "WAITING FOR BASELINES" if self.calibration_armed else "WAITING FOR CALIBRATION (press c)"
            if self.waiting_baselines:
                return "MEASURING; baselines pending: " + ", ".join(self.waiting_baselines)
        missing = [k for k in KEYS if k not in self.last_values]
        return "MEASURING; no landmarks: " + ", ".join(missing) if missing else "MEASURING"

    def metric_waiting_text(self, key):
        if key in self.waiting_baselines:
            if not self.calibration_armed:
                return "baseline: press c"
            if key not in self.last_values:
                return "baseline waiting; no landmarks"
            held = self.now - self.calibration_since.get(key, self.now)
            return f"baseline {held:.1f}/{self.calibration_delay:g}s"
        return "no landmarks" if key not in self.last_values else ""

    def update(self, values, now):
        self.now = now
        values = {k: v for k, v in values.items() if k in self.metrics and math.isfinite(v)}
        self.last_values = values
        result = {}
        eligible = False
        for key, metric in self.metrics.items():
            if key not in values:
                metric.pause(clear_buffer=True)
                self.calibration_since.pop(key, None)
                continue
            raw = values[key]
            if self.mode == "calibrated" and metric.neutral is None:
                if self.active and self.calibration_armed:
                    since = self.calibration_since.setdefault(key, now)
                    metric.buffer.append(raw)
                    if now - since >= self.calibration_delay:
                        metric.neutral = sum(metric.buffer) / len(metric.buffer)
                        self.baseline_captured_at[key] = now
                        metric.buffer.clear()
                    else:
                        result[key] = (None, False, 0.0, False)
                        continue
                else:
                    result[key] = (None, False, 0.0, False)
                    continue
            baseline = 0.0 if self.mode == "absolute" else metric.neutral
            result[key] = metric.observe(raw, baseline, now, self.active)
            eligible = eligible or self.active
        if eligible:
            if self.last_active_t is not None:
                self.active_seconds += max(0.0, now - self.last_active_t)
            self.last_active_t = now
        else:
            self.last_active_t = None
        return result
