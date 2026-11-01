"""Embodied MuJoCo mouse flick-and-shoot controller and digital baseline controller."""

import math

import numpy as np

from embodied_fps_agent.config import ROOT
from embodied_fps_agent.fps_bridge import FpsMouseBridge
from embodied_fps_agent.robot import MouseRobot


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class EmbodiedMouseController:
    """Execute view rotations and shots by moving and clicking the MuJoCo-mounted mouse."""

    def __init__(self, config: dict):
        self.screen = config["screen"]
        self.mouse_cfg = config["mouse"]
        self.control = config["control"]
        self.robot = MouseRobot(
            ROOT / "assets/mujoco/mounted_mouse.xml", self.control["initial_mouse_xy"]
        )
        self.bridge = FpsMouseBridge(self.mouse_cfg, self.control, self.robot.mouse_xy)
        self.trace: list[dict] = []

    def flick_and_shoot(
        self,
        d_yaw_deg: float,
        d_pitch_deg: float,
        env,
        record_frame=None,
        enable_click: bool = True,
    ) -> dict:
        if not (math.isfinite(d_yaw_deg) and math.isfinite(d_pitch_deg)):
            raise ValueError("Expected finite (d_yaw_deg, d_pitch_deg)")
        if abs(d_yaw_deg) > 90.0 or abs(d_pitch_deg) > 60.0:
            raise ValueError("Requested rotation exceeds safe single-flick range")
        robot, bridge, control = self.robot, self.bridge, self.control
        delta_m = bridge.meters_for_angles(d_yaw_deg, d_pitch_deg)
        target_angles = robot.inverse_kinematics(robot.mouse_xy + delta_m)
        start_angles = np.array(
            [robot.data.qpos[robot.model.joint(name).qposadr[0]] for name in ("shoulder", "elbow")]
        )
        move_end = float(control["move_seconds"])
        press_start, aligned_since = None, None
        hold_at = float(control["press_seconds"])
        release_at = hold_at + float(control["hold_seconds"])
        finish_at = release_at + float(control["release_seconds"])
        dt = float(robot.model.opt.timestep)
        start_sim_time = float(robot.data.time)
        events: list[dict] = []
        shot_result = None
        accum_yaw, accum_pitch = 0.0, 0.0
        int_q = np.zeros(2, dtype=float)
        deepest, next_frame = 0.0, 0.0
        max_steps = round(float(control["max_action_seconds"]) / dt)
        for step in range(max_steps):
            t = step * dt
            rem_yaw = d_yaw_deg - accum_yaw
            rem_pitch = d_pitch_deg - accum_pitch
            rem_err = math.hypot(rem_yaw, rem_pitch)
            q_ref = start_angles + (target_angles - start_angles) * smoothstep(t / move_end)
            cur_q = np.array(
                [robot.data.qpos[robot.model.joint(n).qposadr[0]] for n in ("shoulder", "elbow")]
            )
            err_q = q_ref - cur_q
            if t >= move_end:
                int_q = np.clip(int_q + err_q * dt, -0.02, 0.02)
            command = q_ref + 4.0 * err_q + 20.0 * int_q
            aligned = rem_err <= float(control["angle_tolerance_deg"])
            if press_start is None and t >= move_end:
                aligned_since = (t if aligned_since is None else aligned_since) if aligned else None
                if (
                    aligned_since is not None
                    and t - aligned_since >= float(control["settle_seconds"])
                    and t + finish_at + float(control["finish_seconds"])
                    < float(control["max_action_seconds"])
                ):
                    press_start = t
            down = 0.0
            elapsed = t - press_start if press_start is not None else None
            if enable_click and elapsed is not None:
                if elapsed < hold_at:
                    down = smoothstep(elapsed / float(control["press_seconds"]))
                elif elapsed < release_at:
                    down = 1.0
                elif elapsed < finish_at:
                    down = 1.0 - smoothstep(
                        (elapsed - release_at) / float(control["release_seconds"])
                    )
            robot.step(command, down * float(control["finger_travel"]))
            step_dyaw, step_dpitch, event = bridge.update(
                robot.mouse_xy, robot.on_table, robot.button_depth, robot.finger_contact
            )
            accum_yaw += step_dyaw
            accum_pitch += step_dpitch
            env.step_world(dt, step_dyaw, step_dpitch)
            deepest = max(deepest, robot.button_depth)
            if event:
                if event == "down" and shot_result is None:
                    shot_result = env.shoot()
                events.append({"time": float(robot.data.time), "event": event})
            if t + dt / 2.0 >= next_frame:
                self.trace.append(
                    {
                        "time": float(robot.data.time),
                        "mouse_xy": robot.mouse_xy.tolist(),
                        "yaw": round(env.yaw, 4),
                        "pitch": round(env.pitch, 4),
                        "button_depth": robot.button_depth,
                        "on_table": robot.on_table,
                        "finger_contact": robot.finger_contact,
                    }
                )
                if record_frame is not None:
                    record_frame(robot, env)
                next_frame += 1.0 / float(control["fps"])
            if elapsed is not None and elapsed >= finish_at + float(control["finish_seconds"]):
                break
        released = (
            not bridge.pressed
            and robot.button_depth <= float(control["release_threshold"])
            and not robot.finger_contact
        )
        clicked = [e["event"] for e in events] == ["down", "up"] and released
        act_time = (
            events[0]["time"] - start_sim_time
            if events
            else (float(robot.data.time) - start_sim_time)
        )
        return {
            "mode": "embodied_mujoco",
            "physical_click": bool(clicked),
            "target_reached": bool(
                math.hypot(d_yaw_deg - accum_yaw, d_pitch_deg - accum_pitch)
                <= float(control["angle_tolerance_deg"])
            ),
            "released": bool(released),
            "max_press_mm": round(deepest * 1000.0, 3),
            "commanded_dyaw_deg": round(d_yaw_deg, 4),
            "commanded_dpitch_deg": round(d_pitch_deg, 4),
            "achieved_dyaw_deg": round(accum_yaw, 4),
            "achieved_dpitch_deg": round(accum_pitch, 4),
            "actuation_duration_s": round(act_time, 4),
            "events": events,
            "shot": shot_result,
        }


class DigitalMouseController:
    """Instant OS/digital mouse baseline without physical arm inertia or button stroke lag."""

    def __init__(self, config: dict):
        self.config = config
        self.trace: list[dict] = []

    def flick_and_shoot(
        self,
        d_yaw_deg: float,
        d_pitch_deg: float,
        env,
        record_frame=None,
        enable_click: bool = True,
    ) -> dict:
        if not (math.isfinite(d_yaw_deg) and math.isfinite(d_pitch_deg)):
            raise ValueError("Expected finite (d_yaw_deg, d_pitch_deg)")
        env.step_world(0.0, d_yaw_deg, d_pitch_deg)
        shot_result = env.shoot() if enable_click else None
        self.trace.append(
            {
                "time": float(env.sim_time),
                "yaw": round(env.yaw, 4),
                "pitch": round(env.pitch, 4),
            }
        )
        return {
            "mode": "digital_direct",
            "physical_click": bool(enable_click),
            "target_reached": True,
            "released": True,
            "max_press_mm": 0.0,
            "commanded_dyaw_deg": round(d_yaw_deg, 4),
            "commanded_dpitch_deg": round(d_pitch_deg, 4),
            "achieved_dyaw_deg": round(d_yaw_deg, 4),
            "achieved_dpitch_deg": round(d_pitch_deg, 4),
            "actuation_duration_s": 0.0,
            "events": [{"time": float(env.sim_time), "event": "digital_click"}]
            if enable_click
            else [],
            "shot": shot_result,
        }
