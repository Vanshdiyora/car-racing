"""Core race engine — runs a head-to-head Human vs AI race.

Both players share the same track (same seed) but run in separate
environments.  The engine synchronises steps, collects metrics, and
drives the renderer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.env.car_env import make_env
from src.race.ai_control import AIController
from src.race.human_control import HumanController
from src.race.metrics import RaceMetrics, compute_score
from src.race.renderer import RaceRenderer

logger = logging.getLogger(__name__)


def run_race(
    race_cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Execute a full Human vs AI race and return results.

    Parameters
    ----------
    race_cfg : dict
        Loaded from ``configs/race_config.yaml``.
    env_cfg : dict
        Loaded from ``configs/env_config.yaml``.
    train_cfg : dict
        Loaded from ``configs/train_config.yaml`` (needed to reconstruct agent).
    """
    r = race_cfg.get("race", {})
    d = race_cfg.get("display", {})
    s = race_cfg.get("scoring", {})
    mode = r.get("mode", "human_vs_ai")
    seed = 42

    # Set up environments — same seed guarantees same track
    human_env = make_env(env_cfg, seed=seed, render=False)
    ai_env = make_env(env_cfg, seed=seed, render=False)

    # Also create a raw env for RGB frames
    import gymnasium as gym

    raw_human_env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    raw_ai_env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    raw_human_env.reset(seed=seed)
    raw_ai_env.reset(seed=seed)

    # Controllers
    human_ctrl = HumanController(race_cfg.get("controls")) if mode != "ai_only" else None
    ai_ctrl = AIController(r["ai_model_path"], train_cfg, env_cfg) if mode != "human_only" else None

    # Renderer
    renderer = RaceRenderer(
        width=d.get("window_width", 1200),
        height=d.get("window_height", 600),
        show_hud=d.get("show_hud", True),
        fps=r.get("fps", 60),
    )

    # Metrics
    human_metrics = RaceMetrics(label="Human")
    ai_metrics = RaceMetrics(label="AI")
    human_metrics.start()
    ai_metrics.start()

    h_obs, _ = human_env.reset(seed=seed)
    a_obs, _ = ai_env.reset(seed=seed)

    max_steps = r.get("max_time_seconds", 300) * r.get("fps", 60)
    step = 0

    logger.info("Race started! Mode=%s", mode)

    try:
        while step < max_steps:
            # --- Human action ---
            if human_ctrl is not None:
                h_action = human_ctrl.get_action()
                if human_ctrl.quit_requested:
                    break
            else:
                h_action = 0

            # --- AI action ---
            if ai_ctrl is not None:
                a_action = ai_ctrl.get_action(a_obs)
            else:
                a_action = 0

            # Step both environments
            h_obs, h_reward, h_term, h_trunc, h_info = human_env.step(h_action)
            a_obs, a_reward, a_term, a_trunc, a_info = ai_env.step(a_action)

            raw_human_env.step(h_action)
            raw_ai_env.step(a_action)

            # Record metrics
            human_metrics.record_step(float(h_reward))
            ai_metrics.record_step(float(a_reward))

            # Get RGB frames for display
            h_frame = raw_human_env.render()
            a_frame = raw_ai_env.render()

            renderer.render_frame(
                human_frame=h_frame,
                ai_frame=a_frame,
                human_metrics=human_metrics.summary(),
                ai_metrics=ai_metrics.summary(),
            )

            step += 1

            # End if either is done
            if (h_term or h_trunc) and (a_term or a_trunc):
                break

    except KeyboardInterrupt:
        logger.info("Race interrupted by user")

    # Compute final scores
    human_score = compute_score(
        human_metrics,
        track_weight=s.get("track_completion_weight", 1.0),
        time_weight=s.get("time_weight", 0.5),
        penalty_weight=s.get("penalty_weight", 0.3),
    )
    ai_score = compute_score(
        ai_metrics,
        track_weight=s.get("track_completion_weight", 1.0),
        time_weight=s.get("time_weight", 0.5),
        penalty_weight=s.get("penalty_weight", 0.3),
    )

    renderer.render_results(human_score, ai_score, human_metrics.summary(), ai_metrics.summary())
    renderer.close()
    human_env.close()
    ai_env.close()
    raw_human_env.close()
    raw_ai_env.close()

    results = {
        "human": human_metrics.summary(),
        "ai": ai_metrics.summary(),
        "human_score": human_score,
        "ai_score": ai_score,
        "winner": "human" if human_score > ai_score else "ai" if ai_score > human_score else "tie",
    }
    logger.info("Race finished — winner: %s (H:%.0f vs AI:%.0f)", results["winner"], human_score, ai_score)
    return results
