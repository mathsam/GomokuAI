#!/bin/bash
export DISPLAY=""
cd /home/junyic/Work/game_AI/board_game_AI
python -u analyze_selfplay.py >> ./log/analyze_selfplay.log 2>&1
if [ $? -eq 0 ]
then
  python -u dnn.py >> ./log/train_dnn.log  2>&1
  if [ $? -eq 0 ]
  then
    python -u ai_battleground.py >> ./log/battle.log  2>&1
  fi
fi
