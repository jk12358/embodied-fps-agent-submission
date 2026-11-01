"""Stream synchronized MuJoCo robotic mouse and first-person VALORANT range frames."""

from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_fps_agent.range_env import _load_font


class Recording:
    def __init__(self, output: Path, model, fps: int, mode: str):
        self.output = output
        self.model = model
        self.fps = fps
        self.mode = mode
        self._stack = ExitStack()
        self._font = _load_font(16)

    def __enter__(self):
        try:
            self.renderer = self._stack.enter_context(
                mujoco.Renderer(self.model, width=640, height=480)
            )
            self.writer = self._stack.enter_context(
                imageio.get_writer(str(self.output), fps=self.fps)
            )
        except BaseException:
            self._stack.close()
            raise
        return self

    def append(self, robot, env) -> None:
        self.renderer.update_scene(robot.data, camera="overview")
        frame = Image.new("RGB", (1600, 608), (16, 18, 24))
        frame.paste(Image.fromarray(self.renderer.render()), (0, 96))
        frame.paste(Image.fromarray(env.screenshot()), (640, 64))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, 1600, 48], fill=(10, 12, 18))
        draw.text(
            (16, 14),
            f"{self.mode} | sim {robot.data.time:.2f}s | "
            f"mouse ({robot.mouse_xy[0] * 100:.1f}, {robot.mouse_xy[1] * 100:.1f}) cm | "
            f"button press {robot.button_depth * 1000:.2f} mm | "
            f"view (yaw {env.yaw:+.1f} deg, pitch {env.pitch:+.1f} deg)",
            fill=(235, 240, 248),
            font=self._font,
        )
        self.writer.append_data(np.asarray(frame))

    def __exit__(self, *args):
        return self._stack.__exit__(*args)
