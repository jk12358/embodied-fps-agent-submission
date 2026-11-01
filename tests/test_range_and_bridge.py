"""Tests for ValTrainer sensitivity math, 3D range raycast, and MuJoCo mouse actuation."""

import math

import pytest

from embodied_fps_agent.config import load_config
from embodied_fps_agent.episode import EmbodiedMouseController
from embodied_fps_agent.fps_bridge import cm_per_360, pixel_to_angle_delta
from embodied_fps_agent.policy import detect_target_head_pixels
from embodied_fps_agent.range_env import ValTrainerRange, _wrap_deg, angles_to_point


def test_valtrainer_cm360_and_pixel_angle_consistency():
    config = load_config()
    cm = cm_per_360(
        config["mouse"]["dpi"],
        config["mouse"]["sensitivity"],
        config["mouse"]["deg_per_count"],
    )
    assert math.isclose(cm, 360.0 / (0.07 * 0.35 * 800) * 2.54, rel_tol=1e-6)
    env = ValTrainerRange(config)
    screenshot = env.screenshot()
    detected = detect_target_head_pixels(screenshot)
    assert detected is not None
    dyaw, dpitch = pixel_to_angle_delta(
        detected, env.width, env.height, env.hfov_deg, current_pitch_deg=env.pitch
    )
    true_yaw, true_pitch = angles_to_point(env.eye, env.bot.head_pos)
    assert abs(_wrap_deg(env.yaw + dyaw - true_yaw)) < 0.25
    assert abs((env.pitch + dpitch) - true_pitch) < 0.25
    # Verify exact spherical unprojection when camera is pitched and yawed
    env.yaw = 14.0
    env.pitch = 3.2
    proj = env._project(env.bot.head_pos)
    assert proj is not None
    dyaw2, dpitch2 = pixel_to_angle_delta(
        [proj[0], proj[1]], env.width, env.height, env.hfov_deg, current_pitch_deg=env.pitch
    )
    assert abs(_wrap_deg(env.yaw + dyaw2 - true_yaw)) < 1e-4
    assert abs((env.pitch + dpitch2) - true_pitch) < 1e-4


def test_embodied_mouse_flick_scores_headshot_from_visual_detection():
    config = load_config()
    config["drill"]["targets"] = 2
    env = ValTrainerRange(config)
    controller = EmbodiedMouseController(config)
    px = detect_target_head_pixels(env.screenshot())
    assert px is not None
    dyaw, dpitch = pixel_to_angle_delta(px, env.width, env.height, env.hfov_deg)
    result = controller.flick_and_shoot(dyaw, dpitch, env)
    assert result["physical_click"] is True
    assert result["target_reached"] is True
    assert result["released"] is True
    assert result["max_press_mm"] >= 2.0
    assert result["shot"] is not None
    assert result["shot"]["headshot"] is True
    assert result["shot"]["hit_zone"] == "head"


def test_embodied_mouse_without_click_does_not_fire():
    config = load_config()
    config["drill"]["targets"] = 1
    env = ValTrainerRange(config)
    controller = EmbodiedMouseController(config)
    result = controller.flick_and_shoot(3.0, 1.0, env, enable_click=False)
    assert result["physical_click"] is False
    assert result["shot"] is None
    assert env.shots_fired == 0


def test_excessive_rotation_is_rejected():
    config = load_config()
    env = ValTrainerRange(config)
    controller = EmbodiedMouseController(config)
    with pytest.raises(ValueError, match="exceeds"):
        controller.flick_and_shoot(120.0, 0.0, env)
