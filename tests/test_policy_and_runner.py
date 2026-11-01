"""Tests for Claude VLM direct policy, GameSense GSM, and Embodied vs Digital tuning."""

from types import SimpleNamespace

import pytest

from embodied_fps_agent.config import load_config
from embodied_fps_agent.policy import detect_target_head_pixels
from embodied_fps_agent.runner import run_gamesense_tuning, run_vlm_direct_session


class MockMessages:
    def __init__(self, handler):
        self.handler = handler

    def count_tokens(self, **kwargs):
        # Ensure no ground-truth coordinates or labels leak into the prompt
        dump = str(kwargs)
        assert "head_pos" not in dump
        assert "spawn_yaw" not in dump
        return SimpleNamespace(input_tokens=1420)

    def create(self, **kwargs):
        return self.handler(kwargs)


class MockAnthropic:
    def __init__(self, handler):
        self.messages = MockMessages(handler)


def _tool_response(tool_name: str, payload: dict):
    return SimpleNamespace(
        _request_id="req_mock_test",
        model="claude-haiku-5-5",
        usage=SimpleNamespace(
            model_dump=lambda: {
                "input_tokens": 1420,
                "output_tokens": 48,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        ),
        content=[SimpleNamespace(type="tool_use", name=tool_name, input=payload)],
    )


def test_vlm_direct_session_distinguishes_headshot_and_miss(tmp_path):
    config = load_config()
    config["drill"]["mode"] = "head_flicks"
    config["drill"]["targets"] = 2
    config["api"]["budget_usd"] = 0.05

    # 1. Mock policy that visually locates the target head in the PNG
    import base64
    import io

    import numpy as np
    from PIL import Image

    def visual_handler(req):
        b64 = req["messages"][0]["content"][0]["source"]["data"]
        arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
        px = detect_target_head_pixels(arr)
        x = int(round(px[0] * 1000.0 / (arr.shape[1] - 1)))
        y = int(round(px[1] * 1000.0 / (arr.shape[0] - 1)))
        return _tool_response("aim_and_shoot", {"x": x, "y": y})

    metrics = run_vlm_direct_session(
        config, tmp_path / "hit_run", client=MockAnthropic(visual_handler)
    )
    assert metrics["Complete"] is True
    assert metrics["headshots"] == 2
    assert metrics["headshot_rate"] == 1.0

    # 2. Mock policy that aims at the corner (miss)
    def miss_handler(_req):
        return _tool_response("aim_and_shoot", {"x": 50, "y": 50})

    miss_metrics = run_vlm_direct_session(
        config, tmp_path / "miss_run", client=MockAnthropic(miss_handler)
    )
    assert miss_metrics["Complete"] is True
    assert miss_metrics["hits"] == 0
    assert miss_metrics["headshot_rate"] == 0.0


def test_gamesense_tuning_compensates_embodied_lag_on_strafe_bots(tmp_path):
    config = load_config()
    config["drill"]["mode"] = "strafe_bots"
    config["drill"]["targets"] = 4
    config["api"]["budget_usd"] = 0.05

    def tune_handler(_req):
        return _tool_response(
            "update_gsm_params",
            {"kp_yaw": 1.0, "kp_pitch": 1.0, "lead_seconds": 0.46},
        )

    result = run_gamesense_tuning(
        config, tmp_path / "gamesense_tune", client=MockAnthropic(tune_handler)
    )
    dig = result["digital_round1"]
    emb_uncal = result["embodied_round1_uncalibrated"]
    emb_tuned = result["embodied_round2_vlm_tuned"]

    # Digital baseline has 0s actuation delay so uncalibrated GSM hits moving bots
    assert dig["headshot_rate"] >= 0.75
    # Embodied physical mouse has ~0.25s actuation delay so uncalibrated GSM misses behind
    assert emb_uncal["headshot_rate"] < dig["headshot_rate"]
    assert emb_uncal["mean_actuation_duration_s"] > 0.20
    # After VLM tuning of lead_seconds, Embodied physical mouse recovers headshot accuracy
    assert emb_tuned["headshot_rate"] > emb_uncal["headshot_rate"]
    assert emb_tuned["headshot_rate"] >= 0.75


def test_vlm_direct_respects_call_limit(tmp_path):
    config = load_config()
    config["drill"]["targets"] = 2
    config["api"]["max_calls"] = 1
    config["api"]["budget_usd"] = 0.05

    def handler(_req):
        return _tool_response("aim_and_shoot", {"x": 500, "y": 500})

    with pytest.raises(RuntimeError, match="call limit"):
        run_vlm_direct_session(config, tmp_path / "limit_run", client=MockAnthropic(handler))
