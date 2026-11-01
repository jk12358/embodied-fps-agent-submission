"""Render a side-by-side preview of the MuJoCo mouse arm and VALORANT range without API calls."""

import os
from pathlib import Path

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image, ImageDraw

from embodied_fps_agent.config import ROOT, load_config
from embodied_fps_agent.episode import EmbodiedMouseController
from embodied_fps_agent.range_env import ValTrainerRange, _load_font


def render_preview(output: Path | None = None) -> Path:
    config = load_config()
    env = ValTrainerRange(config)
    controller = EmbodiedMouseController(config)
    target_path = output or ROOT / "outputs/preview/valtrainer-range.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(controller.robot.model, width=640, height=480) as renderer:
        renderer.update_scene(controller.robot.data, camera="overview")
        frame = Image.new("RGB", (1600, 608), (16, 18, 24))
        frame.paste(Image.fromarray(renderer.render()), (0, 96))
        frame.paste(Image.fromarray(env.screenshot()), (640, 64))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, 1600, 48], fill=(10, 12, 18))
        draw.text(
            (16, 14),
            f"Embodied FPS Agent Preview | cm/360 = {controller.bridge.cm_360:.2f} cm | "
            f"drill = {config['drill']['mode']}",
            fill=(235, 240, 248),
            font=_load_font(16),
        )
        frame.save(target_path)
    return target_path


if __name__ == "__main__":
    saved = render_preview()
    print(f"Saved preview to {saved}")
