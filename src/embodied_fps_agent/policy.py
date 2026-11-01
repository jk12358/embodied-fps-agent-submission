"""Visual FPS policies: Claude VLM direct control and a VLM-tuned rule-based fast controller."""

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from embodied_fps_agent.costs import BudgetExceeded, CostLedger
from embodied_fps_agent.fps_bridge import pixel_to_angle_delta


@dataclass(frozen=True)
class AimAction:
    x: int
    y: int
    d_yaw_deg: float
    d_pitch_deg: float
    source: str


@dataclass(frozen=True)
class GSMParams:
    kp_yaw: float = 1.0
    kp_pitch: float = 1.0
    lead_seconds: float = 0.0


def parse_aim_coords(payload: dict) -> tuple[int, int]:
    if not isinstance(payload, dict) or set(payload) != {"x", "y"}:
        raise ValueError("Expected exactly x and y")
    if any(type(payload[k]) is not int or not 0 <= payload[k] <= 1000 for k in ("x", "y")):
        raise ValueError("Coordinates must be integers in normalized range 0..1000")
    return payload["x"], payload["y"]


def parse_gsm_params(payload: dict) -> GSMParams:
    if not isinstance(payload, dict) or set(payload) != {"kp_yaw", "kp_pitch", "lead_seconds"}:
        raise ValueError("Expected exactly kp_yaw, kp_pitch, and lead_seconds")
    vals = {}
    for key, lo, hi in (
        ("kp_yaw", 0.3, 1.6),
        ("kp_pitch", 0.3, 1.6),
        ("lead_seconds", 0.0, 0.6),
    ):
        v = payload[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
            raise ValueError(f"{key} must be a number in [{lo}, {hi}]")
        vals[key] = round(float(v), 4)
    return GSMParams(**vals)


def detect_target_head_pixels(screenshot: np.ndarray) -> np.ndarray | None:
    """Detect the target head center (u, v) purely from RGB screenshot pixels."""
    roi = screenshot[36:, :, :]
    r = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    b = roi[:, :, 2].astype(np.int16)
    head_mask = (r > 220) & (b < 110) & ((g > 180) | (g < 85))
    ys, xs = np.nonzero(head_mask)
    if len(xs) == 0:
        return None
    return np.array([float(np.mean(xs)), float(np.mean(ys) + 36.0)], dtype=float)


class ShootingGSM:
    """Rule-based fast visual controller, a simplified take on GameSense (Lu et al., ACL 2025).

    Unlike the original (learned ResNet/DQN with VLM-rewritten code), this tracks the head with
    a color mask and only exposes numeric gains that the VLM can set.
    """

    def __init__(self, screen_cfg: dict, params: GSMParams):
        self.width = int(screen_cfg["width"])
        self.height = int(screen_cfg["height"])
        self.hfov_deg = float(screen_cfg["hfov_deg"])
        self.params = params

    def act_from_frames(
        self,
        frame_t0: np.ndarray,
        frame_t1: np.ndarray | None = None,
        dt_obs: float = 0.04,
        current_pitch_deg: float = 0.0,
    ) -> AimAction:
        p0 = detect_target_head_pixels(frame_t0)
        if p0 is None:
            raise RuntimeError("Target head not visible in screenshot")
        dyaw0, dpitch0 = pixel_to_angle_delta(
            p0, self.width, self.height, self.hfov_deg, current_pitch_deg=current_pitch_deg
        )
        vyaw, vpitch = 0.0, 0.0
        target_px = p0
        if frame_t1 is not None and dt_obs > 0:
            p1 = detect_target_head_pixels(frame_t1)
            if p1 is not None:
                dyaw1, dpitch1 = pixel_to_angle_delta(
                    p1, self.width, self.height, self.hfov_deg, current_pitch_deg=current_pitch_deg
                )
                vyaw = (dyaw1 - dyaw0) / dt_obs
                vpitch = (dpitch1 - dpitch0) / dt_obs
                dyaw0, dpitch0 = dyaw1, dpitch1
                target_px = p1
        cmd_yaw = self.params.kp_yaw * dyaw0 + self.params.lead_seconds * vyaw
        cmd_pitch = self.params.kp_pitch * dpitch0 + self.params.lead_seconds * vpitch
        norm_x = int(np.clip(round(target_px[0] * 1000.0 / (self.width - 1)), 0, 1000))
        norm_y = int(np.clip(round(target_px[1] * 1000.0 / (self.height - 1)), 0, 1000))
        return AimAction(
            x=norm_x,
            y=norm_y,
            d_yaw_deg=round(float(cmd_yaw), 4),
            d_pitch_deg=round(float(cmd_pitch), 4),
            source="gsm_visual",
        )


class ClaudePolicy:
    """Budget-guarded Claude VLM policy for direct aiming and fast-controller gain setting."""

    def __init__(
        self,
        api_cfg: dict,
        screen_cfg: dict | None = None,
        client=None,
        audit_path: Path | None = None,
    ):
        self.live_api = client is None
        self.config = api_cfg
        self.screen = screen_cfg or {"width": 960, "height": 540, "hfov_deg": 103.0}
        self.calls = 0
        self.records: list[dict] = []
        self.token_count_calls = 0
        self.ledger = CostLedger(api_cfg, audit_path)
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
                raise RuntimeError("Set ANTHROPIC_API_KEY in .env before calling Claude API")
            from anthropic import Anthropic

            client = Anthropic(max_retries=0, timeout=api_cfg["timeout_seconds"])
        self.client = client

    def _encode_png(self, screenshot: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(screenshot).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _aim_request(self, screenshot: np.ndarray) -> dict:
        return dict(
            model=self.config["model"],
            max_tokens=self.config["max_tokens"],
            system=(
                "You are operating a first-person VALORANT shooting range. "
                "The image is your only observation. Locate the center of the enemy bot's "
                "circular head target (bright red circle with a yellow center core above the "
                "dark armor torso) and call aim_and_shoot once with normalized coordinates "
                "x and y in 0..1000, where (0,0) is top-left and (1000,1000) is bottom-right."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self._encode_png(screenshot),
                            },
                        },
                        {"type": "text", "text": "Locate the enemy bot head and shoot."},
                    ],
                }
            ],
            tools=[
                {
                    "name": "aim_and_shoot",
                    "strict": True,
                    "description": "Aim at the normalized image coordinates of the bot head.",
                    "input_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "x": {"type": "integer", "description": "Normalized x, 0 to 1000"},
                            "y": {"type": "integer", "description": "Normalized y, 0 to 1000"},
                        },
                        "required": ["x", "y"],
                    },
                }
            ],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )

    def _tune_request(self, screenshot: np.ndarray, telemetry_summary: dict) -> dict:
        return dict(
            model=self.config["model"],
            max_tokens=self.config["max_tokens"],
            system=(
                "You are the high-level VLM optimizer in a GameSense (ACL 2025) architecture "
                "controlling a physical MuJoCo robotic mouse arm in a VALORANT shooting range. "
                "The low-level GSM computes:\n"
                "  cmd_yaw = kp_yaw * static_dyaw + lead_seconds * vyaw\n"
                "  cmd_pitch = kp_pitch * static_dpitch + lead_seconds * vpitch\n"
                "Here kp_yaw and kp_pitch scale the static angle to the target, while "
                "lead_seconds predicts target motion during physical arm actuation. "
                "When mean_command_tracking_error_deg is small (< 0.1 deg) and digital_round1 "
                "achieves high hit rate with kp_yaw=1.0 and kp_pitch=1.0, keep kp_yaw=1.0 and "
                "kp_pitch=1.0 (changing them away from 1.0 introduces static angle error!) and "
                "set lead_seconds equal to embodied mean_actuation_duration_s."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self._encode_png(screenshot),
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Previous round telemetry summary:\n"
                                f"{json.dumps(telemetry_summary, indent=2)}\n"
                                "Call update_gsm_params once with optimized parameters."
                            ),
                        },
                    ],
                }
            ],
            tools=[
                {
                    "name": "update_gsm_params",
                    "strict": True,
                    "description": "Update proportional gains and velocity lead time (seconds).",
                    "input_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kp_yaw": {"type": "number", "description": "Yaw gain in [0.3, 1.6]"},
                            "kp_pitch": {
                                "type": "number",
                                "description": "Pitch gain in [0.3, 1.6]",
                            },
                            "lead_seconds": {
                                "type": "number",
                                "description": "Motion prediction lead time in [0.0, 0.6] seconds",
                            },
                        },
                        "required": ["kp_yaw", "kp_pitch", "lead_seconds"],
                    },
                }
            ],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )

    def estimate(self, screenshot: np.ndarray, request: dict | None = None) -> tuple[dict, dict]:
        req = request or self._aim_request(screenshot)
        count_req = {k: v for k, v in req.items() if k != "max_tokens"}
        self.token_count_calls += 1
        counted = self.client.messages.count_tokens(**count_req)
        quote = self.ledger.quote(counted.input_tokens, self.config["max_tokens"])
        print(json.dumps({"api_preflight": quote, "model": self.config["model"]}), flush=True)
        return req, quote

    def _invoke(self, request: dict, tool_name: str) -> dict:
        if self.calls >= self.config["max_calls"]:
            raise RuntimeError("Claude API call limit reached")
        if self.live_api and self.ledger.path is None:
            raise RuntimeError("Live generation requires a persistent cost audit path")
        _, quote = self.estimate(np.zeros((1, 1, 3), dtype=np.uint8), request=request)
        index = self.ledger.reserve(quote)
        self.calls += 1
        try:
            response = self.client.messages.create(**request)
        except BaseException as error:
            self.ledger.fail(index, type(error).__name__)
            raise
        try:
            receipt = self.ledger.settle(index, response.usage.model_dump())
        except (ValueError, TypeError, KeyError) as error:
            self.ledger.fail(index, type(error).__name__)
            raise
        print(
            json.dumps({"api_receipt": receipt, "accounted_usd": str(self.ledger.accounted())}),
            flush=True,
        )
        if receipt["estimate_exceeded"] or self.ledger.accounted() > self.ledger.budget:
            raise BudgetExceeded("Actual usage exceeded the estimate; stop further generation")
        record = {
            "request_id": getattr(response, "_request_id", None),
            "model": response.model,
            "usage": response.usage.model_dump(),
            "tool": tool_name,
            "status": "invalid_response",
            "output": None,
        }
        self.records.append(record)
        blocks = [b for b in response.content if b.type == "tool_use" and b.name == tool_name]
        if len(blocks) != 1:
            raise ValueError(f"Claude did not return exactly one {tool_name} tool call")
        record["output"] = blocks[0].input
        record["status"] = "valid_action"
        return blocks[0].input

    def act(self, screenshot: np.ndarray, current_pitch_deg: float = 0.0) -> AimAction:
        raw = self._invoke(self._aim_request(screenshot), "aim_and_shoot")
        x, y = parse_aim_coords(raw)
        width, height = int(self.screen["width"]), int(self.screen["height"])
        px = np.array([x * (width - 1) / 1000.0, y * (height - 1) / 1000.0], dtype=float)
        d_yaw, d_pitch = pixel_to_angle_delta(
            px, width, height, float(self.screen["hfov_deg"]), current_pitch_deg=current_pitch_deg
        )
        return AimAction(
            x=x,
            y=y,
            d_yaw_deg=round(d_yaw, 4),
            d_pitch_deg=round(d_pitch, 4),
            source="claude_vlm_direct",
        )

    def tune_gsm(self, screenshot: np.ndarray, telemetry_summary: dict) -> GSMParams:
        raw = self._invoke(self._tune_request(screenshot, telemetry_summary), "update_gsm_params")
        return parse_gsm_params(raw)
