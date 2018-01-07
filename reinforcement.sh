#!/bin/bash
cd /home/junyic/Work/game_AI/board_game_AI
python -u analyze_selfplay.py >> ./log/analyze_selfplay.log
python -u dnn.py >> ./log/train_dnn.log
python -u ai_battleground.py >> ./log/battle.log
