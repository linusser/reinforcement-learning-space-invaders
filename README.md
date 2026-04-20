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