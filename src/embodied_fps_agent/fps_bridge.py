"""VALORANT / ValTrainer sensitivity bridge from physical mouse motion to view angles."""

import math

import numpy as np


def cm_per_360(dpi: int, sensitivity: float, deg_per_count: float = 0.07) -> float:
    """Exact ValTrainer PlayerView.Cm360 formula."""
    if dpi <= 0 or sensitivity <= 0 or deg_per_count <= 0:
        raise ValueError("dpi, sensitivity, and deg_per_count must be positive")
    return 360.0 / (deg_per_count * sensitivity * dpi) * 2.54


def degrees_per_meter(dpi: int, sensitivity: float, deg_per_count: float = 0.07) -> float:
    """Degrees of camera rotation per meter of physical mouse travel on the desk."""
    counts_per_meter = dpi / 0.0254
    return counts_per_meter * deg_per_count * sensitivity


def focal_pixels(width: int, hfov_deg: float) -> float:
    """Horizontal focal length in pixels for a pinhole camera with given HFOV."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))


def pixel_to_angle_delta(
    pixel_xy: np.ndarray,
    width: int,
    height: int,
    hfov_deg: float,
    current_pitch_deg: float = 0.0,
) -> tuple[float, float]:
    """Convert a 2D image pixel (u, v) into camera-relative (d_yaw_deg, d_pitch_deg)."""
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    fx = focal_pixels(width, hfov_deg)
    x_norm = (u - width / 2.0) / fx
    y_norm = (height / 2.0 - v) / fx
    p0 = math.radians(current_pitch_deg)
    cos_p0, sin_p0 = math.cos(p0), math.sin(p0)
    d_yaw_deg = math.degrees(math.atan2(x_norm, cos_p0 - y_norm * sin_p0))
    ray_norm = math.sqrt(1.0 + x_norm * x_norm + y_norm * y_norm)
    sin_target_pitch = max(-1.0, min(1.0, (sin_p0 + y_norm * cos_p0) / ray_norm))
    d_pitch_deg = math.degrees(math.asin(sin_target_pitch)) - current_pitch_deg
    return d_yaw_deg, d_pitch_deg


class FpsMouseBridge:
    """Convert simulated physical mouse observations into VALORANT view deltas and shots."""

    def __init__(self, mouse_cfg: dict, control_cfg: dict, initial_xy: np.ndarray):
        self.dpi = int(mouse_cfg["dpi"])
        self.sensitivity = float(mouse_cfg["sensitivity"])
        self.deg_per_count = float(mouse_cfg["deg_per_count"])
        self.deg_per_meter = degrees_per_meter(self.dpi, self.sensitivity, self.deg_per_count)
        self.cm_360 = cm_per_360(self.dpi, self.sensitivity, self.deg_per_count)
        self._previous_xy = np.asarray(initial_xy, dtype=float).copy()
        self._control = control_cfg
        self.pressed = False

    def meters_for_angles(self, d_yaw_deg: float, d_pitch_deg: float) -> np.ndarray:
        """Desk displacement [dx_m, dy_m] required for a desired (d_yaw, d_pitch) rotation."""
        return np.array([d_yaw_deg, d_pitch_deg], dtype=float) / self.deg_per_meter

    def update(
        self, mouse_xy: np.ndarray, on_table: bool, depth: float, finger_contact: bool
    ) -> tuple[float, float, str | None]:
        """Return (d_yaw_deg, d_pitch_deg, trigger_event) for one simulation step."""
        current = np.asarray(mouse_xy, dtype=float)
        delta_m = current - self._previous_xy
        self._previous_xy = current.copy()
        d_yaw, d_pitch = 0.0, 0.0
        if on_table:
            d_yaw = float(delta_m[0] * self.deg_per_meter)
            d_pitch = float(delta_m[1] * self.deg_per_meter)
        event = None
        if not self.pressed and on_table and finger_contact:
            if depth >= self._control["press_threshold"]:
                self.pressed = True
                event = "down"
        elif self.pressed and depth <= self._control["release_threshold"]:
            self.pressed = False
            event = "up"
        return d_yaw, d_pitch, event
