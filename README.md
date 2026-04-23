# Reinforcement Learning für SpaceInvaders
Module: RLE
Semester: FS26
Student: Linus Ackermann

## Installation

Voraussetzungen: Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip-sync requirements.txt
```

## Projektstruktur

| Datei | Zweck |
|---|---|
| `src/utils/env_utils.py` | Erstellt die Space-Invaders-Env in drei Modi: `train` (vektorisiert mit DeepMind-Preprocessing für SB3), `eval` (echte Game-Scores ohne Reward-Clipping) und `human` (rohe Env zum Selber-Spielen, optional mit Video-Aufnahme). |
| `src/utils/eval_utils.py` | Konsistente 100-Episoden-Evaluation aller Experimente mit JSON-Export der Metriken (Mean, Std, Min, Max, Median, Quartile) nach `results/<name>/eval.json`. |
| `src/utils/video_utils.py` | Zeichnet trainierte Agenten als MP4-Videos auf, getrennt von der Eval, damit die 100-Episoden-Auswertung schnell bleibt. |

## Verwendung

### Menschliche Baseline aufnehmen

Spielt Space Invaders manuell, zeichnet jede Episode als MP4 auf und speichert die Scores als JSON. Steuerung: `A` = links, `D` = rechts, `Leertaste` = Feuer.

```bash
python src/utils/env_utils.py --record results/human_baseline/
```

Ergebnisse landen in `results/human_baseline/videos/` (MP4 pro Episode) und `results/human_baseline/stats.json` (aggregierte Metriken).

### Random-Baseline auswerten

Spielt 100 Episoden mit einer uniformen Zufallsstrategie und speichert die Statistik:

```bash
python src/utils/eval_utils.py
```

Ergebnisse landen in `results/baseline_random/eval.json`.