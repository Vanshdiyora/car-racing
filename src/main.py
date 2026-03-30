"""Main entry point — CLI hub for training, evaluation, racing, and benchmarking."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="car-racing",
        description="PPO Autonomous Driving — Human vs AI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Train the PPO agent")
    p_train.add_argument("--train-config", default="configs/train_config.yaml")
    p_train.add_argument("--env-config", default="configs/env_config.yaml")
    p_train.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    p_train.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")

    # ---- eval ----
    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("checkpoint", help="Path to model checkpoint")
    p_eval.add_argument("--env-config", default="configs/env_config.yaml")
    p_eval.add_argument("--train-config", default="configs/train_config.yaml")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--render", action="store_true")

    # ---- race ----
    p_race = sub.add_parser("race", help="Human vs AI race!")
    p_race.add_argument("--race-config", default="configs/race_config.yaml")
    p_race.add_argument("--env-config", default="configs/env_config.yaml")
    p_race.add_argument("--train-config", default="configs/train_config.yaml")
    p_race.add_argument("--seed", type=int, default=None, help="Track seed (overrides config). Omit for random.")
    p_race.add_argument("--track", type=str, default=None, help="Custom track name from configs/tracks.yaml (e.g. oval, monaco, monza)")
    p_race.add_argument("--tracks-file", default="configs/tracks.yaml", help="Path to tracks definition YAML")
    p_race.add_argument("--theme", type=str, default=None, help="UI theme: default, night, retro, neon")
    p_race.add_argument("--themes-file", default="configs/themes.yaml", help="Path to themes YAML")

    # ---- benchmark ----
    p_bench = sub.add_parser("benchmark", help="Run benchmark comparison")
    p_bench.add_argument("checkpoint", help="Path to model checkpoint")
    p_bench.add_argument("--env-config", default="configs/env_config.yaml")
    p_bench.add_argument("--train-config", default="configs/train_config.yaml")
    p_bench.add_argument("--episodes", type=int, default=50)
    p_bench.add_argument("--output", default="outputs/reports/benchmark.json")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "race":
        _cmd_race(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)


# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> None:
    from src.utils import load_env_config, load_train_config
    from src.utils.config_loader import deep_merge
    from src.train.train import run_training

    train_cfg = load_train_config(args.train_config)
    env_cfg = load_env_config(args.env_config)

    if args.timesteps:
        train_cfg = deep_merge(train_cfg, {"training": {"total_timesteps": args.timesteps}})

    run_training(train_cfg=train_cfg, env_cfg=env_cfg, n_envs=args.n_envs)


def _cmd_eval(args: argparse.Namespace) -> None:
    from src.agent.ppo_agent import PPOAgent
    from src.env.car_env import make_env
    from src.evaluate.evaluate import evaluate_with_details
    from src.utils import load_env_config, load_train_config, setup_logger

    setup_logger("car_racing")
    env_cfg = load_env_config(args.env_config)
    train_cfg = load_train_config(args.train_config)

    tmp = make_env(env_cfg, seed=0)
    n_actions = tmp.action_space.n  # type: ignore[union-attr]
    tmp.close()

    agent = PPOAgent(train_cfg, env_cfg, n_actions, compile_policy=False)
    agent.load(args.checkpoint)

    results = evaluate_with_details(
        agent, env_cfg, n_episodes=args.episodes, deterministic=True
    )
    print(f"\nEvaluation over {args.episodes} episodes:")
    print(f"  Mean reward : {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Min / Max   : {results['min_reward']:.1f} / {results['max_reward']:.1f}")
    print(f"  Mean steps  : {results['mean_steps']:.0f}")

    if args.render:
        from src.evaluate.evaluate import evaluate_agent
        evaluate_agent(agent, env_cfg, n_episodes=1, render=True)


def _cmd_race(args: argparse.Namespace) -> None:
    import yaml
    from src.race.race_engine import run_race
    from src.utils import load_env_config, load_race_config, load_train_config, setup_logger

    setup_logger("car_racing")
    race_cfg = load_race_config(args.race_config)

    if args.seed is not None:
        race_cfg.setdefault("race", {})["seed"] = args.seed

    # Load custom track if specified
    if args.track:
        import pathlib
        tracks_path = pathlib.Path(args.tracks_file)
        if not tracks_path.exists():
            print(f"Error: tracks file not found: {tracks_path}")
            return
        with open(tracks_path) as f:
            tracks_data = yaml.safe_load(f)
        all_tracks = tracks_data.get("tracks", {})
        if args.track not in all_tracks:
            print(f"Error: track '{args.track}' not found. Available: {', '.join(all_tracks.keys())}")
            return
        race_cfg["track"] = all_tracks[args.track]
        print(f"Track: {all_tracks[args.track].get('name', args.track)}")

    # Load theme if specified
    if args.theme:
        import pathlib
        themes_path = pathlib.Path(args.themes_file)
        if themes_path.exists():
            with open(themes_path) as f:
                themes_data = yaml.safe_load(f)
            all_themes = themes_data.get("themes", {})
            if args.theme in all_themes:
                race_cfg["theme"] = all_themes[args.theme]
                print(f"Theme: {all_themes[args.theme].get('name', args.theme)}")
            else:
                print(f"Warning: theme '{args.theme}' not found. Available: {', '.join(all_themes.keys())}")

    results = run_race(
        race_cfg=race_cfg,
        env_cfg=load_env_config(args.env_config),
        train_cfg=load_train_config(args.train_config),
    )
    print(f"\nRace complete! Winner: {results['winner'].upper()}")
    print(f"   Human score: {results['human_score']:.0f}")
    print(f"   AI score:    {results['ai_score']:.0f}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from src.evaluate.benchmark import benchmark_agent
    from src.utils import load_env_config, load_train_config, setup_logger

    setup_logger("car_racing")
    benchmark_agent(
        checkpoint=args.checkpoint,
        env_cfg=load_env_config(args.env_config),
        train_cfg=load_train_config(args.train_config),
        n_episodes=args.episodes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
