"""
env_utils.py
============

Hilfsfunktionen rund um die ALE/SpaceInvaders-v5 Umgebung fuer die RLE
Mini-Challenge. Stellt drei Modi zur Verfuegung:

* "train"  -> vektorisierte Atari-Env (DeepMind-Preprocessing) fuer SB3
* "eval"   -> einzelne Atari-Env ohne Reward-Clipping / EpisodicLife,
              damit echte Game-Scores ueber 100 Episoden gemessen werden
              koennen (Auftrag verlangt Mittelwert, Std, Min, Max usw.)
* "human"  -> unverarbeitete Env mit Tastatursteuerung zum Selber-Spielen
              (fuer die menschliche Baseline). Optional mit Video-
              Aufzeichnung und Score-Export.

Voraussetzungen:
    pip install "gymnasium[atari]" stable-baselines3[extra] ale-py
    pip install moviepy  # fuer RecordVideo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    ResizeObservation,
)
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
)

# Wichtig fuer Gymnasium 1.x: ALE-Envs muessen explizit registriert werden,
# sonst kennt gym.make("ALE/SpaceInvaders-v5") den Namespace nicht.
gym.register_envs(ale_py)


# Standard-Defaults gemaess DeepMind-DQN-Paper. Wenn eine deiner vier
# Erweiterungen das Preprocessing aendern soll (z.B. Frame Skip 2 oder
# Stack 2), kannst du diese Werte in den make_*-Funktionen ueberschreiben.
DEFAULT_ENV_ID = "ALE/SpaceInvaders-v5"
DEFAULT_FRAME_SKIP = 4
DEFAULT_FRAME_STACK = 4
DEFAULT_NOOP_MAX = 30
DEFAULT_SCREEN_SIZE = 84


# ---------------------------------------------------------------------------
# TRAIN: vektorisierte Env fuer SB3-Algorithmen (PPO, DQN, A2C, ...)
# ---------------------------------------------------------------------------
def make_train_env(
    env_id: str = DEFAULT_ENV_ID,
    n_envs: int = 8,
    seed: int = 0,
    use_subproc: bool = True,
    frame_stack: int = DEFAULT_FRAME_STACK,
    clip_reward: bool = True,
    episodic_life: bool = True,
    monitor_dir: Optional[str] = None,
) -> VecEnv:
    """
    Liefert eine vektorisierte Trainingsumgebung mit Standard-DeepMind-
    Preprocessing (Frame Skip 4, 84x84 grayscale, NoopReset 30,
    Frame Stack 4, optional Reward Clipping & EpisodicLife).
    """
    wrapper_kwargs = {
        "noop_max": DEFAULT_NOOP_MAX,
        "frame_skip": DEFAULT_FRAME_SKIP,
        "screen_size": DEFAULT_SCREEN_SIZE,
        "terminal_on_life_loss": episodic_life,
        "clip_reward": clip_reward,
    }

    vec_env_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv

    vec_env = make_atari_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=vec_env_cls,
    )
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env


# ---------------------------------------------------------------------------
# EVAL: einzelne Env mit echten Scores fuer die 100-Episoden-Auswertung
# ---------------------------------------------------------------------------
def make_eval_env(
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 1000,
    frame_stack: int = DEFAULT_FRAME_STACK,
    render_mode: Optional[str] = None,
    monitor_dir: Optional[str] = None,
) -> VecEnv:
    """
    Liefert eine Evaluations-Env als VecEnv (n_envs=1), so dass dasselbe
    SB3-Modell wie im Training direkt benutzt werden kann.

    Wichtige Unterschiede zum Training:
    * KEIN Reward Clipping (echter Game-Score)
    * KEIN EpisodicLife (eine Episode = Game Over, nicht Lebensverlust)
    """
    wrapper_kwargs = {
        "noop_max": DEFAULT_NOOP_MAX,
        "frame_skip": DEFAULT_FRAME_SKIP,
        "screen_size": DEFAULT_SCREEN_SIZE,
        "terminal_on_life_loss": False,
        "clip_reward": False,
    }

    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    vec_env = make_atari_env(
        env_id=env_id,
        n_envs=1,
        seed=seed,
        monitor_dir=monitor_dir,
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
    )
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env


# ---------------------------------------------------------------------------
# HUMAN: rohe Env zum Selber-Spielen (menschliche Baseline)
# ---------------------------------------------------------------------------
def make_human_env(
    env_id: str = DEFAULT_ENV_ID,
    video_folder: Optional[str] = None,
    name_prefix: str = "human",
) -> gym.Env:
    """
    Liefert die rohe Atari-Env mit voller Aufloesung fuer manuelles Spielen.

    Wichtig: render_mode muss 'rgb_array' sein, weil gymnasium.utils.play.play()
    die Frames selbst in einem pygame-Fenster anzeigt UND RecordVideo die
    Frames als Numpy-Array braucht.

    Args:
        env_id:        Gymnasium-ID, default ALE/SpaceInvaders-v5.
        video_folder:  Wenn gesetzt, wird jede gespielte Episode als MP4
                       in diesen Ordner gespeichert. Ordner wird angelegt
                       falls noetig.
        name_prefix:   Praefix fuer die MP4-Dateinamen.

    Returns:
        Gymnasium-Env, in der Reihenfolge: Atari -> RecordVideo (optional)
        -> RecordEpisodeStatistics. RecordEpisodeStatistics ist aussen,
        damit env.return_queue / env.length_queue die echten ungeklippten
        Game-Scores enthalten.
    """
    env = gym.make(env_id, render_mode="rgb_array", frameskip=1)

    if video_folder is not None:
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        # episode_trigger=lambda x: True nimmt jede einzelne Episode auf.
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda episode_id: True,
            disable_logger=True,
        )

    env = RecordEpisodeStatistics(env, buffer_length=1000)
    return env


def save_human_session_stats(env: gym.Env, output_path: str) -> dict:
    """
    Liest die in RecordEpisodeStatistics gesammelten Episode-Returns aus
    und speichert sie zusammen mit aggregierten Kennzahlen als JSON.

    Aufrufen, NACHDEM die play()-Session beendet wurde (Fenster zu).

    Returns:
        Das gespeicherte Statistik-Dict (auch direkt nutzbar).
    """
    import statistics

    # RecordEpisodeStatistics legt deque() an, wir wandeln in Listen um.
    returns = [float(r) for r in list(env.return_queue)]
    lengths = [int(l) for l in list(env.length_queue)]

    stats = {
        "name": "human_baseline",
        "n_episodes": len(returns),
        "episode_returns": returns,
        "episode_lengths": lengths,
    }
    if returns:
        stats["return_mean"] = statistics.mean(returns)
        stats["return_std"] = statistics.stdev(returns) if len(returns) > 1 else 0.0
        stats["return_min"] = min(returns)
        stats["return_max"] = max(returns)
        stats["return_median"] = statistics.median(returns)
        stats["length_mean"] = statistics.mean(lengths)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[stats] {len(returns)} Episoden gespielt.")
    if returns:
        print(f"[stats] Mean Return: {stats['return_mean']:.1f} "
              f"(Std: {stats['return_std']:.1f}, Min: {stats['return_min']:.0f}, "
              f"Max: {stats['return_max']:.0f})")
    print(f"[stats] Gespeichert nach: {output_path}")

    return stats


# ---------------------------------------------------------------------------
# Convenience: Single-Env-Factory fuer eigene VecEnvs (z.B. CleanRL-Style)
# ---------------------------------------------------------------------------
def make_single_env_fn(
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    idx: int = 0,
    capture_video: bool = False,
    video_dir: str = "videos",
    clip_reward: bool = True,
    episodic_life: bool = True,
) -> Callable[[], gym.Env]:
    """
    Gibt eine thunk-Funktion zurueck, die eine einzelne, voll gewrappte
    Atari-Env baut. Praktisch wenn du selbst gym.vector.SyncVectorEnv
    bauen willst (z.B. fuer CleanRL-aehnliches PPO ausserhalb von SB3).
    """

    def thunk() -> gym.Env:
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = gym.make(env_id, render_mode=render_mode, frameskip=1)
        env = RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = RecordVideo(env, video_dir)

        env = NoopResetEnv(env, noop_max=DEFAULT_NOOP_MAX)
        env = MaxAndSkipEnv(env, skip=DEFAULT_FRAME_SKIP)
        if episodic_life:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)

        env = ResizeObservation(env, (DEFAULT_SCREEN_SIZE, DEFAULT_SCREEN_SIZE))
        env = GrayscaleObservation(env)
        env = FrameStackObservation(env, stack_size=DEFAULT_FRAME_STACK)

        env.action_space.seed(seed + idx)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Manuelles Spielen via `python env_utils.py [--record OUTPUT_DIR]`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from gymnasium.utils.play import play

    parser = argparse.ArgumentParser(description="Space Invaders manuell spielen.")
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help=("Wenn gesetzt, werden Videos und Score-Statistiken in diesen "
              "Ordner geschrieben. Beispiel: results/human_baseline/"),
    )
    args = parser.parse_args()

    # Tastenbelegung Space Invaders. ALE-Action-Set:
    #   0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
    keys_to_action = {
        (ord(" "),): 1,                   # nur Feuer
        (ord("d"),): 2,                   # rechts
        (ord("a"),): 3,                   # links
        (ord("d"), ord(" ")): 4,          # rechts + Feuer
        (ord("a"), ord(" ")): 5,          # links + Feuer
    }

    if args.record:
        video_folder = str(Path(args.record) / "videos")
        env = make_human_env(video_folder=video_folder, name_prefix="human")
        print(f"[record] Videos -> {video_folder}/")
        print(f"[record] Stats  -> {args.record}/stats.json")
    else:
        env = make_human_env()

    print("Steuerung: A = links, D = rechts, Leertaste = Feuer.")
    print("Fenster schliessen oder Strg+C beendet das Spiel.\n")

    try:
        play(
            env,
            keys_to_action=keys_to_action,
            noop=0,
            fps=30,
            zoom=3,
        )
    finally:
        # Wird auch bei Strg+C ausgefuehrt, damit nichts verloren geht.
        if args.record:
            save_human_session_stats(env, str(Path(args.record) / "stats.json"))