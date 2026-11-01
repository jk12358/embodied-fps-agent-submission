"""ValTrainer-compatible 3D VALORANT shooting range, raycast hit logic, and telemetry."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embodied_fps_agent.fps_bridge import focal_pixels

FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/DejaVuSans.ttf"),
)


@dataclass
class BotTarget:
    target_id: int
    head_pos: np.ndarray
    head_radius: float
    torso_radius: float
    torso_height: float
    vx: float
    spawn_time: float
    spawn_yaw: float
    spawn_pitch: float
    flick_dir: np.ndarray
    initial_angle_deg: float
    turn_in: float = 1.0
    peak_along_deg: float = 0.0
    first_move_ms: float | None = None


def _load_font(size: int):
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def direction_vector(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Exact ValTrainer PlayerView.Dir(yawDeg, pitchDeg) with -Z forward and +X right."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    return np.array(
        [math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)], dtype=float
    )


def angles_to_point(eye: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    """World (yaw_deg, pitch_deg) from eye to a 3D point."""
    rel = np.asarray(point, dtype=float) - np.asarray(eye, dtype=float)
    dist = float(np.linalg.norm(rel))
    if dist < 1e-8:
        return 0.0, 0.0
    unit = rel / dist
    yaw = math.degrees(math.atan2(unit[0], -unit[2]))
    pitch = math.degrees(math.asin(np.clip(unit[1], -1.0, 1.0)))
    return yaw, pitch


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class ValTrainerRange:
    """First-person 3D shooting range implementing Head Flicks, Microshot, and Strafe Bots."""

    def __init__(self, config: dict):
        self.screen = config["screen"]
        self.mouse = config["mouse"]
        self.drill = config["drill"]
        self.width = int(self.screen["width"])
        self.height = int(self.screen["height"])
        self.hfov_deg = float(self.screen["hfov_deg"])
        self.fx = focal_pixels(self.width, self.hfov_deg)
        self.eye = np.array([0.0, float(self.drill["eye_height_m"]), 0.0], dtype=float)
        self.yaw = 0.0
        self.pitch = 0.0
        self.sim_time = 0.0
        self.rng = np.random.default_rng(int(self.drill["seed"]))
        self.total_targets = int(self.drill["targets"])
        self.spawned = 0
        self.hits = 0
        self.headshots = 0
        self.shots_fired = 0
        self.history: list[dict] = []
        self.last_feedback: dict | None = None
        self.bot: BotTarget | None = None
        self._font_sm = _load_font(15)
        self._font_md = _load_font(18)
        self._spawn_next()

    @property
    def finished(self) -> bool:
        return self.bot is None and self.spawned >= self.total_targets

    def _spawn_next(self) -> None:
        if self.spawned >= self.total_targets:
            self.bot = None
            return
        mode = self.drill["mode"]
        if mode == "head_flicks":
            dist = float(self.rng.uniform(12.5, 16.5))
            side = -1.0 if self.spawned % 2 == 0 else 1.0
            yaw_off = side * float(self.rng.uniform(7.0, 20.0))
            pitch_off = float(self.rng.uniform(-2.5, 3.5))
            target_yaw = _wrap_deg(yaw_off)
            target_pitch = np.clip(pitch_off, -15.0, 15.0)
            vx = 0.0
        elif mode == "microshot":
            dist = float(self.rng.uniform(12.0, 15.0))
            ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            mag = float(self.rng.uniform(2.0, 5.2))
            target_yaw = _wrap_deg(self.yaw + mag * math.cos(ang))
            target_pitch = float(np.clip(self.pitch + mag * 0.6 * math.sin(ang), -10.0, 10.0))
            vx = 0.0
        else:  # strafe_bots
            dist = float(self.rng.uniform(13.0, 15.5))
            side = -1.0 if self.spawned % 2 == 0 else 1.0
            target_yaw = _wrap_deg(side * float(self.rng.uniform(4.0, 11.0)))
            target_pitch = float(self.rng.uniform(-1.5, 2.0))
            speed = float(self.drill.get("strafe_speed_mps", 2.6))
            vx = -side * speed
        head_pos = self.eye + direction_vector(target_yaw, target_pitch) * dist
        delta = np.array([_wrap_deg(target_yaw - self.yaw), target_pitch - self.pitch], dtype=float)
        norm = float(np.linalg.norm(delta))
        flick_dir = delta / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=float)
        turn_in = float(self.rng.uniform(1.15, 1.75)) if abs(vx) > 0 else 999.0
        self.bot = BotTarget(
            target_id=self.spawned,
            head_pos=head_pos,
            head_radius=float(self.drill["head_radius_m"]),
            torso_radius=float(self.drill["torso_radius_m"]),
            torso_height=float(self.drill["torso_height_m"]),
            vx=vx,
            spawn_time=self.sim_time,
            spawn_yaw=self.yaw,
            spawn_pitch=self.pitch,
            flick_dir=flick_dir,
            initial_angle_deg=norm,
            turn_in=turn_in,
        )
        self.spawned += 1

    def step_world(self, dt: float, d_yaw_deg: float = 0.0, d_pitch_deg: float = 0.0) -> None:
        """Advance target motion and apply camera rotation delta."""
        self.sim_time += float(dt)
        self.yaw = _wrap_deg(self.yaw + float(d_yaw_deg))
        self.pitch = float(np.clip(self.pitch + float(d_pitch_deg), -89.0, 89.0))
        if self.bot is None:
            return
        if abs(self.bot.vx) > 0:
            self.bot.head_pos[0] += self.bot.vx * float(dt)
            self.bot.turn_in -= float(dt)
            if abs(self.bot.head_pos[0]) > 3.8 or self.bot.turn_in <= 0.0:
                self.bot.head_pos[0] = float(np.clip(self.bot.head_pos[0], -3.8, 3.8))
                self.bot.vx = -self.bot.vx
                self.bot.turn_in = float(self.rng.uniform(0.75, 1.35))
        moved = np.array(
            [_wrap_deg(self.yaw - self.bot.spawn_yaw), self.pitch - self.bot.spawn_pitch],
            dtype=float,
        )
        along = float(np.dot(moved, self.bot.flick_dir))
        if self.bot.first_move_ms is None and along >= 0.25:
            self.bot.first_move_ms = (self.sim_time - self.bot.spawn_time) * 1000.0
        self.bot.peak_along_deg = max(self.bot.peak_along_deg, along)

    def _raycast_bot(self, bot: BotTarget, ray_dir: np.ndarray) -> tuple[str, float]:
        oc = self.eye - bot.head_pos
        b = float(np.dot(oc, ray_dir))
        c = float(np.dot(oc, oc) - bot.head_radius * bot.head_radius)
        disc = b * b - c
        if disc >= 0.0:
            t = -b - math.sqrt(disc)
            if t > 0.0:
                return "head", t
        # Torso vertical cylinder check below the head sphere
        y_top = bot.head_pos[1] - bot.head_radius * 0.85
        y_bot = y_top - bot.torso_height
        dx, dz = float(ray_dir[0]), float(ray_dir[2])
        ox, oz = float(self.eye[0] - bot.head_pos[0]), float(self.eye[2] - bot.head_pos[2])
        a_xz = dx * dx + dz * dz
        if a_xz > 1e-8:
            b_xz = ox * dx + oz * dz
            c_xz = ox * ox + oz * oz - bot.torso_radius * bot.torso_radius
            d_xz = b_xz * b_xz - a_xz * c_xz
            if d_xz >= 0.0:
                t_cyl = (-b_xz - math.sqrt(d_xz)) / a_xz
                if t_cyl > 0.0:
                    y_hit = float(self.eye[1] + t_cyl * ray_dir[1])
                    if y_bot <= y_hit <= y_top:
                        return "body", t_cyl
        return "miss", float("inf")

    def shoot(self) -> dict:
        """Fire one Vandal round along the current view ray and advance to the next target."""
        if self.bot is None:
            return {"hit_zone": "none", "hit": False, "headshot": False}
        bot = self.bot
        self.shots_fired += 1
        ray_dir = direction_vector(self.yaw, self.pitch)
        zone, hit_dist = self._raycast_bot(bot, ray_dir)
        head_yaw, head_pitch = angles_to_point(self.eye, bot.head_pos)
        yaw_err = _wrap_deg(self.yaw - head_yaw)
        pitch_err = self.pitch - head_pitch
        ang_err = math.hypot(yaw_err, pitch_err)
        dist_to_head = float(np.linalg.norm(bot.head_pos - self.eye))
        head_ang_rad = math.degrees(math.asin(min(0.99, bot.head_radius / dist_to_head)))
        err_vec = np.array([yaw_err, pitch_err], dtype=float)
        signed_along_err = float(np.dot(err_vec, bot.flick_dir))
        overshot = bool(bot.peak_along_deg > bot.initial_angle_deg + head_ang_rad)
        hit = zone in {"head", "body"}
        headshot = zone == "head"
        if hit:
            self.hits += 1
        if headshot:
            self.headshots += 1
        result = {
            "target_id": bot.target_id,
            "hit_zone": zone,
            "hit": hit,
            "headshot": headshot,
            "damage": 160 if headshot else (40 if hit else 0),
            "angular_error_deg": round(ang_err, 4),
            "head_angular_radius_deg": round(head_ang_rad, 4),
            "yaw_error_deg": round(yaw_err, 4),
            "pitch_error_deg": round(pitch_err, 4),
            "signed_along_flick_deg": round(signed_along_err, 4),
            "overshoot": overshot,
            "reaction_to_move_ms": (
                round(bot.first_move_ms, 2) if bot.first_move_ms is not None else None
            ),
            "time_to_shot_ms": round((self.sim_time - bot.spawn_time) * 1000.0, 2),
            "hit_distance_m": round(hit_dist, 3) if math.isfinite(hit_dist) else None,
        }
        self.history.append(result)
        self.last_feedback = {**result, "time": self.sim_time}
        self._spawn_next()
        return result

    def _project(self, point_3d: np.ndarray) -> tuple[float, float, float] | None:
        fwd = direction_vector(self.yaw, self.pitch)
        yaw_rad = math.radians(self.yaw)
        right = np.array([math.cos(yaw_rad), 0.0, math.sin(yaw_rad)], dtype=float)
        up = np.cross(right, fwd)
        rel = np.asarray(point_3d, dtype=float) - self.eye
        z_cam = float(np.dot(rel, fwd))
        if z_cam <= 0.15:
            return None
        u = self.width / 2.0 + self.fx * float(np.dot(rel, right)) / z_cam
        v = self.height / 2.0 - self.fx * float(np.dot(rel, up)) / z_cam
        return u, v, z_cam

    def screenshot(self) -> np.ndarray:
        """Render the first-person 3D VALORANT range view without leaking target coords."""
        img = Image.new("RGB", (self.width, self.height), (22, 26, 34))
        draw = ImageDraw.Draw(img)
        horizon_proj = self._project(self.eye + direction_vector(self.yaw, 0.0) * 30.0)
        horizon_y = int(np.clip(horizon_proj[1] if horizon_proj else 270, 0, self.height))
        draw.rectangle([0, 0, self.width, horizon_y], fill=(26, 31, 42))
        draw.rectangle([0, horizon_y, self.width, self.height], fill=(38, 42, 50))
        # 3D range floor grid and back wall markers
        for z_m in (-8.0, -12.0, -16.0, -20.0):
            p_left = self._project(np.array([-6.0, 0.0, z_m]))
            p_right = self._project(np.array([6.0, 0.0, z_m]))
            if p_left and p_right:
                draw.line(
                    [(p_left[0], p_left[1]), (p_right[0], p_right[1])],
                    fill=(54, 60, 72),
                    width=1,
                )
        for x_m in range(-6, 7, 2):
            p_near = self._project(np.array([float(x_m), 0.0, -4.0]))
            p_far = self._project(np.array([float(x_m), 0.0, -22.0]))
            if p_near and p_far:
                draw.line(
                    [(p_near[0], p_near[1]), (p_far[0], p_far[1])],
                    fill=(50, 56, 68),
                    width=1,
                )
        # Back wall panel at Z = -20m
        wall_pts = [
            self._project(np.array([-5.5, 0.0, -20.0])),
            self._project(np.array([5.5, 0.0, -20.0])),
            self._project(np.array([5.5, 3.6, -20.0])),
            self._project(np.array([-5.5, 3.6, -20.0])),
        ]
        if all(wall_pts):
            poly = [(p[0], p[1]) for p in wall_pts if p is not None]
            draw.polygon(poly, fill=(32, 37, 47), outline=(66, 74, 90))
        # Render practice bot
        if self.bot is not None:
            proj = self._project(self.bot.head_pos)
            if proj is not None:
                u, v, z_cam = proj
                r_px = max(4.0, self.fx * self.bot.head_radius / z_cam)
                tw_px = max(6.0, self.fx * self.bot.torso_radius / z_cam)
                th_px = max(10.0, self.fx * self.bot.torso_height / z_cam)
                torso_top = v + r_px * 0.85
                torso_bot = torso_top + th_px
                # Legs & torso with VALORANT purple-magenta enemy outline
                leg_bot = torso_bot + th_px * 0.75
                draw.rectangle(
                    [u - tw_px * 0.75, torso_bot, u - tw_px * 0.1, leg_bot],
                    fill=(40, 44, 54),
                    outline=(180, 50, 120),
                    width=2,
                )
                draw.rectangle(
                    [u + tw_px * 0.1, torso_bot, u + tw_px * 0.75, leg_bot],
                    fill=(40, 44, 54),
                    outline=(180, 50, 120),
                    width=2,
                )
                draw.rounded_rectangle(
                    [u - tw_px, torso_top, u + tw_px, torso_bot],
                    radius=4,
                    fill=(46, 50, 62),
                    outline=(225, 60, 135),
                    width=2,
                )
                # Head sphere: distinct high-contrast VALORANT crimson-gold target head
                draw.ellipse(
                    [u - r_px - 2, v - r_px - 2, u + r_px + 2, v + r_px + 2],
                    fill=(255, 52, 72),
                    outline=(255, 190, 210),
                    width=2,
                )
                inner = max(2.0, r_px * 0.48)
                draw.ellipse(
                    [u - inner, v - inner, u + inner, v + inner],
                    fill=(255, 238, 75),
                )
        # First-person weapon silhouette (bottom right)
        draw.polygon(
            [(760, 540), (835, 425), (885, 425), (960, 540)],
            fill=(28, 30, 36),
            outline=(70, 76, 88),
        )
        # Center VALORANT cyan crosshair at (W/2, H/2)
        cx, cy = self.width // 2, self.height // 2
        for x0, y0, x1, y1 in (
            (cx - 9, cy, cx - 3, cy),
            (cx + 3, cy, cx + 9, cy),
            (cx, cy - 9, cx, cy - 3),
            (cx, cy + 3, cx, cy + 9),
        ):
            draw.line([(x0, y0), (x1, y1)], fill=(0, 255, 190), width=2)
        # Top HUD bar
        draw.rectangle([0, 0, self.width, 34], fill=(15, 18, 24))
        draw.text(
            (14, 8),
            f"VALTRAINER RANGE | DRILL: {self.drill['mode'].upper()} | "
            f"SHOTS: {self.shots_fired}/{self.total_targets} | "
            f"HITS: {self.hits} | HS: {self.headshots}",
            fill=(235, 240, 248),
            font=self._font_sm,
        )
        if self.last_feedback and self.sim_time - self.last_feedback["time"] <= 0.35:
            zone = self.last_feedback["hit_zone"].upper()
            palette = {"HEAD": (60, 250, 140), "BODY": (250, 200, 70)}
            color = palette.get(zone, (250, 80, 80))
            draw.text((self.width - 170, 7), f"LAST: {zone}", fill=color, font=self._font_md)
        return np.asarray(img, dtype=np.uint8)
