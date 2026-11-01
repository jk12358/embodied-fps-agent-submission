"""Load and validate VALORANT range and robotic mouse experiment conditions from TOML."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID_DRILLS = {"head_flicks", "microshot", "strafe_bots"}


def load_config(path: Path | None = None) -> dict:
    with (path or ROOT / "configs/range.toml").open("rb") as stream:
        config = tomllib.load(stream)
    screen = config["screen"]
    mouse = config["mouse"]
    control = config["control"]
    drill = config["drill"]
    if screen["width"] != 960 or screen["height"] != 540:
        raise ValueError("Range layout requires a 960 x 540 (16:9) screen")
    if not 60.0 <= screen["hfov_deg"] <= 130.0:
        raise ValueError("hfov_deg must be within [60, 130]")
    if mouse["dpi"] <= 0 or mouse["sensitivity"] <= 0 or mouse["deg_per_count"] <= 0:
        raise ValueError("Mouse DPI, sensitivity, and deg_per_count must be positive")
    if not 0 <= control["release_threshold"] < control["press_threshold"]:
        raise ValueError("Click hysteresis thresholds are invalid")
    if not 0 < control["finger_travel"] <= 0.025:
        raise ValueError("finger_travel is outside the finger joint range")
    for key in ("move", "settle", "press", "hold", "release", "finish"):
        if control[f"{key}_seconds"] <= 0:
            raise ValueError(f"{key}_seconds must be positive")
    if control["fps"] <= 0 or control["angle_tolerance_deg"] <= 0:
        raise ValueError("fps and angle_tolerance_deg must be positive")
    if control["max_action_seconds"] <= sum(
        control[f"{key}_seconds"]
        for key in ("move", "settle", "press", "hold", "release", "finish")
    ):
        raise ValueError("max_action_seconds must allow a full flick-and-click sequence")
    if drill["mode"] not in VALID_DRILLS:
        raise ValueError(f"drill.mode must be one of {sorted(VALID_DRILLS)}")
    if drill["targets"] < 1 or drill["head_radius_m"] <= 0 or drill["torso_radius_m"] <= 0:
        raise ValueError("Invalid drill target configuration")
    if config["api"]["max_calls"] < 1 or config["api"]["timeout_seconds"] <= 0:
        raise ValueError("Invalid API limits")
    return config
