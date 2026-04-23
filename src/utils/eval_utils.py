"""
eval_utils.py
=============

Konsistente Evaluations-Pipeline fuer alle Experimente der RLE Mini-Challenge.

Der Auftrag verlangt explizit:
* mindestens 100 Episoden pro Experiment
* aggregierte Metriken (Mean, Std, Min, Max, ...)
* "Aepfel mit Aepfeln vergleichen" -> alle Experimente werden mit derselben
  Funktion ausgewertet

Alle Ergebnisse landen als JSON in results/<experiment_name>/eval.json,
sodass du sie spaeter mit pandas einlesen und im Bericht vergleichen kannst.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from env_utils import make_eval_env


def evaluate_and_save(
    model: Optional[BaseAlgorithm],
    experiment_name: str,
    n_episodes: int = 100,
    deterministic: bool = False,
    seed: int = 1000,
    results_dir: Union[str, Path] = "results",
    eval_env: Optional[VecEnv] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """
    Evaluiert ein SB3-Modell ueber n_episodes auf einer sauberen Eval-Env
    (kein Reward-Clipping, kein EpisodicLife) und speichert eine
    JSON-Statistik nach results/<experiment_name>/eval.json.

    Args:
        model:           Trainiertes SB3-Modell. Falls None -> Zufallsagent
                         (fuer die Baseline). Dann wird env.action_space.sample()
                         verwendet.
        experiment_name: Name des Experiments, z.B. "ppo_initial",
                         "dqn_double", "baseline_random". Wird auch als
                         Ordnername verwendet.
        n_episodes:      Anzahl Eval-Episoden. Default 100 wie im Auftrag.
        deterministic:   Action-Auswahl deterministisch (argmax) oder
                         stochastisch (sampling). Stochastisch ist robuster
                         gegen Atari-Determinismus-Probleme.
        seed:            Eval-Env-Seed. Sollte ueber Experimente gleich
                         bleiben fuer Vergleichbarkeit.
        results_dir:     Wurzel-Ordner. Default "results/".
        eval_env:        Optional eigene Env. Wenn None, wird make_eval_env()
                         genutzt.
        extra_metadata:  Zusaetzliche Felder (z.B. {"algo": "PPO",
                         "total_timesteps": 5_000_000}). Werden ins JSON
                         geschrieben.

    Returns:
        Dict mit allen Statistiken (auch direkt nutzbar im Code).
    """
    out_dir = Path(results_dir) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    own_env = eval_env is None
    if own_env:
        eval_env = make_eval_env(seed=seed)

    print(f"[eval] {experiment_name}: starte {n_episodes} Episoden ...")
    t0 = time.time()

    if model is None:
        # Zufalls-Baseline -> wir brauchen unsere eigene Loop, weil
        # evaluate_policy ein Modell verlangt.
        episode_returns, episode_lengths = _rollout_random(eval_env, n_episodes)
    else:
        episode_returns, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
        )

    duration = time.time() - t0

    stats = _aggregate(episode_returns, episode_lengths)
    stats["experiment_name"] = experiment_name
    stats["n_episodes"] = n_episodes
    stats["deterministic"] = deterministic
    stats["seed"] = seed
    stats["eval_duration_seconds"] = round(duration, 1)
    if extra_metadata:
        stats["metadata"] = extra_metadata

    out_path = out_dir / "eval.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    _print_summary(stats)
    print(f"[eval] gespeichert: {out_path}")

    if own_env:
        eval_env.close()

    return stats


def _rollout_random(eval_env: VecEnv, n_episodes: int) -> tuple[list, list]:
    """Spielt n_episodes mit uniform-zufaelligen Aktionen."""
    returns: list[float] = []
    lengths: list[int] = []

    obs = eval_env.reset()
    ep_return = 0.0
    ep_length = 0

    while len(returns) < n_episodes:
        action = np.array([eval_env.action_space.sample()])
        obs, reward, done, info = eval_env.step(action)
        ep_return += float(reward[0])
        ep_length += 1
        if done[0]:
            returns.append(ep_return)
            lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0

    return returns, lengths


def _aggregate(returns: list, lengths: list) -> dict:
    """Berechnet die im Bericht geforderten Aggregations-Metriken."""
    returns_f = [float(r) for r in returns]
    lengths_i = [int(l) for l in lengths]

    return {
        "return_mean": statistics.mean(returns_f),
        "return_std": statistics.stdev(returns_f) if len(returns_f) > 1 else 0.0,
        "return_min": min(returns_f),
        "return_max": max(returns_f),
        "return_median": statistics.median(returns_f),
        "return_q25": float(np.percentile(returns_f, 25)),
        "return_q75": float(np.percentile(returns_f, 75)),
        "length_mean": statistics.mean(lengths_i),
        "length_std": statistics.stdev(lengths_i) if len(lengths_i) > 1 else 0.0,
        # Rohdaten mitspeichern -> spaeter Histogramme moeglich
        "episode_returns": returns_f,
        "episode_lengths": lengths_i,
    }


def _print_summary(stats: dict) -> None:
    print(
        f"[eval] {stats['experiment_name']:25s} | "
        f"Mean: {stats['return_mean']:7.1f} +/- {stats['return_std']:6.1f} | "
        f"Min: {stats['return_min']:5.0f} | Max: {stats['return_max']:5.0f} | "
        f"Median: {stats['return_median']:6.1f}"
    )


def compare_experiments(results_dir: Union[str, Path] = "results") -> None:
    """
    Druckt eine Vergleichstabelle aller Experimente in results_dir.
    Nuetzlich fuer den finalen Vergleich im Bericht.
    """
    results_dir = Path(results_dir)
    eval_files = sorted(results_dir.glob("*/eval.json"))

    if not eval_files:
        print(f"Keine eval.json-Dateien in {results_dir}/ gefunden.")
        return

    print(f"\n{'Experiment':30s} {'Mean':>10s} {'Std':>10s} "
          f"{'Min':>8s} {'Max':>8s} {'Median':>10s}")
    print("-" * 80)
    for f in eval_files:
        with open(f) as fp:
            s = json.load(fp)
        print(f"{s['experiment_name']:30s} "
              f"{s['return_mean']:10.1f} {s['return_std']:10.1f} "
              f"{s['return_min']:8.0f} {s['return_max']:8.0f} "
              f"{s['return_median']:10.1f}")


if __name__ == "__main__":
    # Schnelltest: Random-Baseline ueber 100 Episoden.
    print("Evaluation der Random-Baseline (kann ein paar Minuten dauern) ...")
    evaluate_and_save(
        model=None,
        experiment_name="baseline_random",
        n_episodes=100,
        extra_metadata={"description": "Uniform random action policy"},
    )
    compare_experiments()