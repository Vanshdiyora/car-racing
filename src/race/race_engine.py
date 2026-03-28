"""Core race engine — runs a head-to-head Human vs AI race.

Human plays on a raw continuous-action env (no wrappers, no frame skip)
for a real driving feel.  AI plays on its wrapped discrete env for
inference.  RGB frames for display come from each env's .render() —
no duplicate environments, no desync.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np

from src.env.car_env import make_env, make_race_env
from src.race.ai_control import AIController
from src.race.human_control import HumanController
from src.race.metrics import RaceMetrics, compute_score
from src.race.renderer import RaceRenderer

logger = logging.getLogger(__name__)


def _get_car_speed(env: gym.Env) -> float:
    """Extract car speed from the unwrapped CarRacing env."""
    car = getattr(env.unwrapped, "car", None)
    if car is None:
        return 0.0
    vx = car.hull.linearVelocity[0]
    vy = car.hull.linearVelocity[1]
    return float(np.sqrt(vx ** 2 + vy ** 2))


def _is_on_grass(env: gym.Env) -> bool:
    """Check if any wheel is off the road (empty tiles set = on grass)."""
    car = getattr(env.unwrapped, "car", None)
    if car is None:
        return False
    return any(len(getattr(w, "tiles", {1})) == 0 for w in car.wheels)


def _get_tile_count(env: gym.Env) -> int:
    """Get the number of visited track tiles."""
    return getattr(env.unwrapped, "tile_visited_count", 0)


def _get_track_progress(env: gym.Env) -> float:
    """Fraction of track tiles visited (0.0 – 1.0)."""
    total = len(getattr(env.unwrapped, "track", []))
    visited = _get_tile_count(env)
    return visited / max(total, 1)


def run_race(
    race_cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Execute a full Human vs AI race and return results."""
    r = race_cfg.get("race", {})
    d = race_cfg.get("display", {})
    s = race_cfg.get("scoring", {})
    mode = r.get("mode", "human_vs_ai")
    seed_cfg = r.get("seed", "random")
    if seed_cfg == "random" or seed_cfg is None:
        seed = int(np.random.default_rng().integers(0, 100_000))
    else:
        seed = int(seed_cfg)
    fps = r.get("fps", 60)
    logger.info("Track seed: %d", seed)
    max_time = r.get("max_time_seconds", 300)

    # ---- Environments ----
    # Human: raw continuous env (no wrappers, no frame skip) for real gameplay
    human_env = make_race_env(seed=seed, continuous=True, max_episode_steps=max_time * fps)

    # AI: wrapped discrete env for model inference + separate raw env for display
    # Override max_episode_steps for race mode — training config has 1000 which is too low
    race_env_cfg = {**env_cfg}
    race_env_cfg["environment"] = {
        **env_cfg.get("environment", {}),
        "max_episode_steps": max_time * fps,
    }
    ai_env = make_env(race_env_cfg, seed=seed, render=False)
    # Raw continuous env for AI display — receives converted continuous actions
    ai_display_env = make_race_env(seed=seed, continuous=True, max_episode_steps=max_time * fps)

    # Controllers
    human_ctrl = HumanController(race_cfg.get("controls")) if mode != "ai_only" else None
    ai_ctrl = AIController(r["ai_model_path"], train_cfg, env_cfg) if mode != "human_only" else None

    # Renderer
    renderer = RaceRenderer(
        width=d.get("window_width", 1200),
        height=d.get("window_height", 600),
        show_hud=d.get("show_hud", True),
        fps=fps,
    )

    # Metrics
    human_metrics = RaceMetrics(label="Human")
    ai_metrics = RaceMetrics(label="AI")
    human_metrics.start()
    ai_metrics.start()

    a_obs, _ = ai_env.reset(seed=seed)

    max_steps = max_time * fps
    n_laps = r.get("n_laps", 3)
    step = 0
    race_start = time.time()

    # Lap tracking — a "lap" = visiting all track tiles once (progress resets)
    h_prev_progress = 0.0
    a_prev_progress = 0.0
    h_laps = 0
    a_laps = 0
    h_finished = False
    a_finished = False

    logger.info("Race started! Mode=%s  laps=%d  max_steps=%d", mode, n_laps, max_steps)

    # Discrete action lookup for AI display env (maps discrete int → continuous)
    _DISCRETE_TO_CONT = {
        0: np.array([0.0, 0.0, 0.0], dtype=np.float32),  # nothing
        1: np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # left
        2: np.array([1.0, 0.0, 0.0], dtype=np.float32),   # right
        3: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # gas
        4: np.array([0.0, 0.0, 0.8], dtype=np.float32),   # brake
    }

    try:
        while step < max_steps:
            # --- Human action (continuous) ---
            if human_ctrl is not None:
                h_action = human_ctrl.get_action()
                if human_ctrl.quit_requested:
                    break
            else:
                h_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # --- AI action (discrete for wrapped env) ---
            if ai_ctrl is not None:
                a_action = ai_ctrl.get_action(a_obs)
            else:
                a_action = 0

            # Step human env (raw, continuous) — single physics step
            h_obs, h_reward, h_term, h_trunc, h_info = human_env.step(h_action)

            # Step AI inference env (wrapped, discrete) — gets preprocessed obs back
            a_obs, a_reward, a_term, a_trunc, a_info = ai_env.step(a_action)

            # Step AI display env (raw, continuous) with same action converted
            a_cont = _DISCRETE_TO_CONT.get(int(a_action), _DISCRETE_TO_CONT[0])
            ai_display_env.step(a_cont)

            # Read real metrics from the raw envs
            h_speed = _get_car_speed(human_env)
            a_speed = _get_car_speed(ai_display_env)
            h_on_grass = _is_on_grass(human_env)
            a_on_grass = _is_on_grass(ai_display_env)
            h_progress = _get_track_progress(human_env)
            a_progress = _get_track_progress(ai_display_env)

            # --- Lap detection ---
            # CarRacing terminates when all tiles are visited (progress ~1.0).
            # A "lap" = env terminated/fully visited. Reset the env for next lap.
            if h_term and not h_finished:
                h_laps += 1
                human_metrics.finish_lap()
                if h_laps >= n_laps:
                    h_finished = True
                    logger.info("Human finished %d laps!", n_laps)
                else:
                    h_obs, _ = human_env.reset(seed=seed)

            if a_term and not a_finished:
                a_laps += 1
                ai_metrics.finish_lap()
                if a_laps >= n_laps:
                    a_finished = True
                    logger.info("AI finished %d laps!", n_laps)
                else:
                    a_obs, _ = ai_env.reset(seed=seed)
                    ai_display_env.reset(seed=seed)

            # Use base CarRacing reward only (no shaping) for fair comparison
            human_metrics.record_step(float(h_reward), on_grass=h_on_grass)
            ai_metrics.record_step(float(a_info.get("base_reward", a_reward)), on_grass=a_on_grass)
            human_metrics.tiles_visited = _get_tile_count(human_env)
            ai_metrics.tiles_visited = _get_tile_count(ai_display_env)

            # Get RGB frames for display
            h_frame = human_env.render()
            a_frame = ai_display_env.render()

            elapsed = time.time() - race_start

            # Overall progress including completed laps
            h_overall_pct = (h_laps * 100.0 + h_progress * 100.0) / n_laps
            a_overall_pct = (a_laps * 100.0 + a_progress * 100.0) / n_laps

            renderer.render_frame(
                human_frame=h_frame,
                ai_frame=a_frame,
                human_metrics={
                    **human_metrics.summary(),
                    "speed": round(h_speed, 1),
                    "on_grass": h_on_grass,
                    "track_pct": round(h_overall_pct, 1),
                    "elapsed": round(elapsed, 1),
                    "lap": f"{h_laps + 1}/{n_laps}" if not h_finished else f"{n_laps}/{n_laps} DONE",
                },
                ai_metrics={
                    **ai_metrics.summary(),
                    "speed": round(a_speed, 1),
                    "on_grass": a_on_grass,
                    "track_pct": round(a_overall_pct, 1),
                    "elapsed": round(elapsed, 1),
                    "lap": f"{a_laps + 1}/{n_laps}" if not a_finished else f"{n_laps}/{n_laps} DONE",
                },
            )

            step += 1

            # Race ends when BOTH have finished all laps, or time runs out
            if h_finished and a_finished:
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
    ai_display_env.close()

    results = {
        "human": human_metrics.summary(),
        "ai": ai_metrics.summary(),
        "human_score": human_score,
        "ai_score": ai_score,
        "winner": "human" if human_score > ai_score else "ai" if ai_score > human_score else "tie",
    }
    logger.info("Race finished — winner: %s (H:%.0f vs AI:%.0f)", results["winner"], human_score, ai_score)
    return results
