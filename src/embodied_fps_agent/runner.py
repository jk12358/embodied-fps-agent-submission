"""Run and evaluate VALORANT range drills with Claude VLM and the rule-based fast controller."""

import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from embodied_fps_agent.episode import DigitalMouseController, EmbodiedMouseController
from embodied_fps_agent.fps_bridge import cm_per_360
from embodied_fps_agent.policy import ClaudePolicy, GSMParams, ShootingGSM
from embodied_fps_agent.range_env import ValTrainerRange
from embodied_fps_agent.recording import Recording


def _prepare_output_dir(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def summarize_trials(trials: list[dict], total_targets: int) -> dict:
    shots = [t["execution"]["shot"] for t in trials if t["execution"].get("shot")]
    hits = sum(1 for s in shots if s.get("hit"))
    headshots = sum(1 for s in shots if s.get("headshot"))
    physical_clicks = sum(1 for t in trials if t["execution"].get("physical_click"))
    reached = sum(1 for t in trials if t["execution"].get("target_reached"))
    act_times = [float(t["execution"]["actuation_duration_s"]) for t in trials]
    cmd_errors = [
        float(
            (
                (t["execution"]["achieved_dyaw_deg"] - t["execution"]["commanded_dyaw_deg"]) ** 2
                + (t["execution"]["achieved_dpitch_deg"] - t["execution"]["commanded_dpitch_deg"])
                ** 2
            )
            ** 0.5
        )
        for t in trials
    ]
    ang_errors = [float(s["angular_error_deg"]) for s in shots]
    along_errors = [float(s["signed_along_flick_deg"]) for s in shots]
    overshoots = sum(1 for s in shots if s.get("overshoot"))
    count = max(1, len(trials))
    return {
        "targets": total_targets,
        "completed_trials": len(trials),
        "physical_clicks": physical_clicks,
        "target_reached_count": reached,
        "hits": hits,
        "headshots": headshots,
        "hit_rate": round(hits / total_targets, 4),
        "headshot_rate": round(headshots / total_targets, 4),
        "overshoot_rate": round(overshoots / max(1, len(shots)), 4),
        "mean_angular_error_deg": round(sum(ang_errors) / max(1, len(ang_errors)), 4),
        "mean_signed_along_flick_deg": round(sum(along_errors) / max(1, len(along_errors)), 4),
        "mean_command_tracking_error_deg": round(sum(cmd_errors) / count, 4),
        "mean_actuation_duration_s": round(sum(act_times) / count, 4),
    }


def run_vlm_direct_session(
    config: dict,
    output: Path,
    render: bool = False,
    actuation: str = "embodied",
    client=None,
    enable_click: bool = True,
) -> dict:
    """Evaluate Mode 1 (Claude VLM direct screenshot-to-flick control)."""
    _prepare_output_dir(output)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    env = ValTrainerRange(config)
    controller = (
        EmbodiedMouseController(config)
        if actuation == "embodied"
        else DigitalMouseController(config)
    )
    policy = ClaudePolicy(
        config["api"],
        config["screen"],
        client=client,
        audit_path=output / "api-costs.json",
    )
    started = time.monotonic()
    api_seconds = 0.0
    trials: list[dict] = []
    rec_ctx = (
        Recording(
            output / "session.mp4",
            controller.robot.model,
            int(config["control"]["fps"]),
            f"vlm_direct:{config['drill']['mode']}",
        )
        if render and actuation == "embodied"
        else nullcontext()
    )
    with rec_ctx as recorder:
        record_fn = recorder.append if recorder is not None else None
        for idx in range(int(config["drill"]["targets"])):
            if env.finished:
                break
            obs = env.screenshot()
            Image.fromarray(obs).save(frames_dir / f"target-{idx:03d}-obs.png")
            t0 = time.monotonic()
            action = policy.act(obs, current_pitch_deg=env.pitch)
            api_seconds += time.monotonic() - t0
            execution = controller.flick_and_shoot(
                action.d_yaw_deg,
                action.d_pitch_deg,
                env,
                record_frame=record_fn,
                enable_click=enable_click,
            )
            Image.fromarray(env.screenshot()).save(frames_dir / f"target-{idx:03d}-after.png")
            trials.append({"target_index": idx, "action": asdict(action), "execution": execution})
            if not execution["physical_click"] or not execution["target_reached"]:
                break
    wall_seconds = time.monotonic() - started
    summary = summarize_trials(trials, int(config["drill"]["targets"]))
    metrics = {
        "mode": "vlm_direct",
        "actuation": actuation,
        "drill": config["drill"]["mode"],
        "cm_per_360": round(
            cm_per_360(
                config["mouse"]["dpi"],
                config["mouse"]["sensitivity"],
                config["mouse"]["deg_per_count"],
            ),
            3,
        ),
        **summary,
        "Complete": len(trials) == int(config["drill"]["targets"])
        and summary["physical_clicks"] == int(config["drill"]["targets"]),
        "api_calls": policy.calls,
        "api_seconds": round(api_seconds, 4),
        "wall_seconds": round(wall_seconds, 4),
        "api_cost": policy.ledger.summary(),
        "trials": trials,
    }
    (output / "trace.json").write_text(json.dumps(controller.trace, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_gsm_round(
    config: dict,
    gsm_params: GSMParams,
    actuation: str = "embodied",
    output: Path | None = None,
    render: bool = False,
    dt_obs: float = 0.10,
) -> dict:
    """Run one full drill round using the fast visual ShootingGSM."""
    if output is not None:
        _prepare_output_dir(output)
        (output / "frames").mkdir(parents=True, exist_ok=True)
    env = ValTrainerRange(config)
    controller = (
        EmbodiedMouseController(config)
        if actuation == "embodied"
        else DigitalMouseController(config)
    )
    gsm = ShootingGSM(config["screen"], gsm_params)
    trials: list[dict] = []
    rec_ctx = (
        Recording(
            output / "session.mp4",
            controller.robot.model,
            int(config["control"]["fps"]),
            f"gamesense:{actuation}:lead={gsm_params.lead_seconds:.2f}s",
        )
        if render and output is not None and actuation == "embodied"
        else nullcontext()
    )
    with rec_ctx as recorder:
        record_fn = recorder.append if recorder is not None else None
        for idx in range(int(config["drill"]["targets"])):
            if env.finished:
                break
            f0 = env.screenshot()
            moving = (
                abs(float(config["drill"].get("strafe_speed_mps", 0.0))) > 0
                and config["drill"]["mode"] == "strafe_bots"
            )
            if moving:
                env.step_world(dt_obs, 0.0, 0.0)
                f1 = env.screenshot()
            else:
                f1 = None
            if output is not None:
                Image.fromarray(f1 if f1 is not None else f0).save(
                    output / "frames" / f"target-{idx:03d}-obs.png"
                )
            action = gsm.act_from_frames(f0, f1, dt_obs=dt_obs, current_pitch_deg=env.pitch)
            execution = controller.flick_and_shoot(
                action.d_yaw_deg,
                action.d_pitch_deg,
                env,
                record_frame=record_fn,
            )
            trials.append({"target_index": idx, "action": asdict(action), "execution": execution})
    summary = summarize_trials(trials, int(config["drill"]["targets"]))
    result = {
        "mode": "gamesense_gsm",
        "actuation": actuation,
        "drill": config["drill"]["mode"],
        "gsm_params": asdict(gsm_params),
        **summary,
        "trials": trials,
    }
    if output is not None:
        (output / "trace.json").write_text(json.dumps(controller.trace, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_gamesense_tuning(
    config: dict,
    output: Path,
    render: bool = False,
    client=None,
) -> dict:
    """Compare Digital vs Embodied uncalibrated GSM, then tune GSM with Claude VLM."""
    _prepare_output_dir(output)
    initial_params = GSMParams(
        kp_yaw=float(config["gsm"]["kp_yaw"]),
        kp_pitch=float(config["gsm"]["kp_pitch"]),
        lead_seconds=float(config["gsm"]["lead_seconds"]),
    )
    digital_baseline = run_gsm_round(
        config, initial_params, actuation="digital", output=output / "round1_digital"
    )
    embodied_round1 = run_gsm_round(
        config,
        initial_params,
        actuation="embodied",
        output=output / "round1_embodied_uncalibrated",
        render=render,
    )
    diag_env = ValTrainerRange(config)
    diag_frame = diag_env.screenshot()
    policy = ClaudePolicy(
        config["api"],
        config["screen"],
        client=client,
        audit_path=output / "api-costs.json",
    )
    telemetry_for_vlm = {
        "drill": embodied_round1["drill"],
        "digital_round1_hit_rate": digital_baseline["hit_rate"],
        "digital_round1_params": digital_baseline["gsm_params"],
        **{
            k: embodied_round1[k]
            for k in (
                "gsm_params",
                "targets",
                "completed_trials",
                "target_reached_count",
                "hits",
                "headshots",
                "hit_rate",
                "headshot_rate",
                "overshoot_rate",
                "mean_angular_error_deg",
                "mean_signed_along_flick_deg",
                "mean_command_tracking_error_deg",
                "mean_actuation_duration_s",
            )
        },
    }
    t0 = time.monotonic()
    tuned_params = policy.tune_gsm(diag_frame, telemetry_for_vlm)
    api_seconds = time.monotonic() - t0
    embodied_round2 = run_gsm_round(
        config,
        tuned_params,
        actuation="embodied",
        output=output / "round2_embodied_vlm_tuned",
        render=render,
    )
    comparison = {
        "mode": "gamesense_vlm_tuning",
        "drill": config["drill"]["mode"],
        "digital_round1": {k: v for k, v in digital_baseline.items() if k != "trials"},
        "embodied_round1_uncalibrated": {k: v for k, v in embodied_round1.items() if k != "trials"},
        "vlm_tuned_params": asdict(tuned_params),
        "embodied_round2_vlm_tuned": {k: v for k, v in embodied_round2.items() if k != "trials"},
        "api_calls": policy.calls,
        "api_seconds": round(api_seconds, 4),
        "api_cost": policy.ledger.summary(),
    }
    (output / "metrics.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison
