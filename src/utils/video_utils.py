"""
video_utils.py
==============

Aufzeichnung trainierter Agenten als MP4-Videos.

Wird typischerweise nach einem abgeschlossenen Training aufgerufen, um
fuer den Bericht und die Sprechstunde visuelle Belege zu haben. Bewusst
getrennt von eval_utils.py, weil:

* Die 100-Episoden-Eval soll moeglichst schnell laufen -> kein Rendering.
* Videos brauchen render_mode='rgb_array' und sind ~50x langsamer.
* Du willst nur ein paar wenige Episoden auf Video, nicht alle 100.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import ale_py
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ALE-Namespace registrieren (siehe env_utils.py).
gym.register_envs(ale_py)


def record_episodes(
    model: BaseAlgorithm,
    experiment_name: str,
    n_episodes: int = 3,
    deterministic: bool = False,
    seed: int = 2000,
    results_dir: Union[str, Path] = "results",
    env_id: str = "ALE/SpaceInvaders-v5",
) -> list[float]:
    """
    Spielt n_episodes mit dem trainierten Modell und speichert jede als MP4.

    Wichtig: Wir bauen die Env hier MANUELL (nicht ueber make_eval_env),
    weil RecordVideo den Wrapper VOR dem Atari-Preprocessing braucht
    (sonst zeichnet es 84x84 Graustufen statt 210x160 Farbbilder auf,
    was im Bericht haesslich aussieht).

    Args:
        model:           Trainiertes SB3-Modell.
        experiment_name: z.B. "ppo_initial". Videos landen in
                         results/<experiment_name>/videos/.
        n_episodes:      Wie viele Episoden aufzeichnen. Default 3.
        deterministic:   Action-Auswahl. Sollte zur Eval konsistent sein.
        seed:            Eigener Seed, damit Videos nicht dieselben
                         Anfangsbedingungen wie die 100-Episoden-Eval haben.
        results_dir:     Wurzel-Ordner.
        env_id:          Default Space Invaders.

    Returns:
        Liste der Episoden-Returns (echte Game-Scores).
    """
    out_dir = Path(results_dir) / experiment_name / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Rohe Env mit rgb_array, damit RecordVideo schoene Farbbilder
    #    in voller Aufloesung bekommt.
    raw_env = gym.make(env_id, render_mode="rgb_array", frameskip=1)

    # 2) RecordVideo INNEN drumherum, vor dem Preprocessing.
    raw_env = RecordVideo(
        raw_env,
        video_folder=str(out_dir),
        name_prefix=experiment_name,
        episode_trigger=lambda episode_id: True,
        disable_logger=True,
    )

    # 3) Jetzt das ueblicheAtari-Preprocessing fuer das Modell.
    #    AtariWrapper macht NoopReset + MaxAndSkip + EpisodicLife + FireReset
    #    + WarpFrame + ClipReward in einem Aufwasch. Fuer Eval-Videos
    #    deaktivieren wir Reward-Clipping und EpisodicLife.
    env = AtariWrapper(
        raw_env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        clip_reward=False,
    )
    env = RecordEpisodeStatistics(env)

    # 4) In VecEnv wrappen, damit das Modell direkt nutzbar ist.
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env.seed(seed)

    print(f"[video] {experiment_name}: zeichne {n_episodes} Episoden auf ...")

    returns: list[float] = []
    obs = vec_env.reset()

    while len(returns) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_env.step(action)
        if done[0]:
            # info enthaelt das "episode"-Dict von RecordEpisodeStatistics
            ep_info = info[0].get("episode")
            if ep_info is not None:
                ep_return = float(ep_info["r"])
                returns.append(ep_return)
                print(f"[video] Episode {len(returns)}/{n_episodes}: "
                      f"Return = {ep_return:.0f}")

    vec_env.close()
    print(f"[video] gespeichert in: {out_dir}/")

    return returns


if __name__ == "__main__":
    # Mini-Demo: Zufallsmodell -> nicht sinnvoll, aber zeigt dass es laeuft.
    # Im echten Workflow: model = PPO.load("checkpoints/ppo_initial.zip")
    print("Dieses Modul ist zum Importieren gedacht. Beispiel:")
    print()
    print("  from stable_baselines3 import PPO")
    print("  from video_utils import record_episodes")
    print()
    print("  model = PPO.load('checkpoints/ppo_initial.zip')")
    print("  record_episodes(model, 'ppo_initial', n_episodes=3)")