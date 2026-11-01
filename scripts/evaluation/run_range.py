"""Run VALORANT shooting range evaluations with Claude VLM or the rule-based fast controller."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from embodied_fps_agent.config import ROOT, VALID_DRILLS, load_config
from embodied_fps_agent.policy import ClaudePolicy, GSMParams
from embodied_fps_agent.range_env import ValTrainerRange
from embodied_fps_agent.runner import (
    run_gamesense_tuning,
    run_gsm_round,
    run_vlm_direct_session,
)


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run VALORANT shooting range evaluation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/range.toml")
    parser.add_argument(
        "--mode",
        choices=("vlm_direct", "gamesense_gsm", "gamesense_tune"),
        default="vlm_direct",
    )
    parser.add_argument("--drill", choices=sorted(VALID_DRILLS), default=None)
    parser.add_argument("--actuation", choices=("embodied", "digital"), default="embodied")
    parser.add_argument("--targets", type=int, default=None)
    parser.add_argument("--lead-seconds", type=float, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.drill is not None:
        config["drill"]["mode"] = args.drill
    if args.targets is not None:
        config["drill"]["targets"] = args.targets
    if args.lead_seconds is not None:
        config["gsm"]["lead_seconds"] = args.lead_seconds
    if args.max_calls is not None:
        config["api"]["max_calls"] = args.max_calls
    if args.budget_usd is not None:
        config["api"]["budget_usd"] = args.budget_usd
    config = load_config_dict(config)

    if args.estimate_only:
        env = ValTrainerRange(config)
        policy = ClaudePolicy(config["api"], config["screen"])
        _, quote = policy.estimate(env.screenshot())
        print(json.dumps({"estimate_only": quote}, indent=2))
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or (
        ROOT / "outputs/evaluation" / f"{args.mode}-{config['drill']['mode']}" / stamp
    )

    if args.mode == "vlm_direct":
        metrics = run_vlm_direct_session(
            config, output, render=args.render, actuation=args.actuation
        )
    elif args.mode == "gamesense_gsm":
        params = GSMParams(
            kp_yaw=float(config["gsm"]["kp_yaw"]),
            kp_pitch=float(config["gsm"]["kp_pitch"]),
            lead_seconds=float(config["gsm"]["lead_seconds"]),
        )
        metrics = run_gsm_round(
            config,
            params,
            actuation=args.actuation,
            output=output,
            render=args.render,
        )
    else:
        metrics = run_gamesense_tuning(config, output, render=args.render)

    print(json.dumps({k: v for k, v in metrics.items() if k != "trials"}, indent=2))


def load_config_dict(config: dict) -> dict:
    if config["drill"]["targets"] < 1:
        raise ValueError("targets must be >= 1")
    if config["api"]["max_calls"] < 1:
        raise ValueError("max_calls must be >= 1")
    return config


if __name__ == "__main__":
    main()
