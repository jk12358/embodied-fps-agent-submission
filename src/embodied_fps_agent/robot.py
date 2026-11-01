"""Planar arm, rigid mouse attachment, and a physical button-pushing actuator."""

import os
from pathlib import Path

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


class MouseRobot:
    def __init__(self, scene: Path, initial_xy):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.mouse_id = self.model.body("mouse").id
        self.button_qpos = self.model.joint("button_down").qposadr[0]
        self._table = self.model.geom("table").id
        self._housing = self.model.geom("mouse_housing").id
        self._finger = self.model.geom("fingertip").id
        self._button = self.model.geom("left_button_top").id
        angles = self.inverse_kinematics(initial_xy)
        for name, value in zip(("shoulder", "elbow"), angles, strict=True):
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        self.data.ctrl[:] = [*angles, 0]
        mujoco.mj_forward(self.model, self.data)

    @property
    def mouse_xy(self) -> np.ndarray:
        return self.data.xpos[self.mouse_id, :2].copy()

    @property
    def button_depth(self) -> float:
        return max(0.0, float(-self.data.qpos[self.button_qpos]))

    def inverse_kinematics(self, target_xy) -> np.ndarray:
        base = self.model.body("robot_base").pos[:2]
        l1 = self.model.body("forearm").pos[0]
        l2 = self.model.body("mouse_mount").pos[0]
        x, y = np.asarray(target_xy) - base
        cosine = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if not np.isfinite(cosine) or abs(cosine) > 1:
            raise ValueError("Mouse target is outside arm reach")
        elbow = np.arccos(np.clip(cosine, -1, 1))
        shoulder = np.arctan2(y, x) - np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow))
        angles = np.array([shoulder, elbow])
        ranges = np.array([self.model.joint(name).range for name in ("shoulder", "elbow")])
        if np.any(angles < ranges[:, 0]) or np.any(angles > ranges[:, 1]):
            raise ValueError("Mouse target exceeds joint limits")
        return angles

    def _contact(self, first: int, second: int) -> bool:
        return any(
            {int(contact.geom1), int(contact.geom2)} == {first, second} and contact.dist <= 0.0001
            for contact in self.data.contact
        )

    @property
    def on_table(self) -> bool:
        return (
            self._contact(self._housing, self._table)
            or float(self.data.xpos[self.mouse_id, 2]) <= 0.061
        )

    @property
    def finger_contact(self) -> bool:
        return self._contact(self._finger, self._button)

    def step(self, angles, finger_travel: float) -> None:
        self.data.ctrl[:] = [*angles, finger_travel]
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Nonfinite simulator state")
