"""Tests for the race metrics module."""

from src.race.metrics import RaceMetrics, compute_score


def test_race_metrics_basic():
    m = RaceMetrics(label="test")
    m.start()
    m.record_step(1.0)
    m.record_step(2.0, on_grass=True)
    assert m.total_reward == 3.0
    assert m.steps == 2
    assert m.grass_steps == 1


def test_compute_score():
    m = RaceMetrics(label="test")
    m.total_reward = 500.0
    m.penalties = 10.0
    m.lap_times = [60.0, 55.0]  # total 115s
    score = compute_score(m, track_weight=1.0, time_weight=0.5, penalty_weight=0.3)
    # track=500, time_bonus=(300-115)*0.5=92.5, penalty=10*0.3=3
    assert abs(score - (500 + 92.5 - 3.0)) < 0.01


def test_summary_keys():
    m = RaceMetrics(label="AI")
    s = m.summary()
    expected = {"label", "total_reward", "steps", "tiles_visited",
                "grass_steps", "penalties", "laps_completed",
                "avg_lap_time", "total_time"}
    assert expected == set(s.keys())
