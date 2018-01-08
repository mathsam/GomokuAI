#!/bin/bash
cd /home/junyic/Work/game_AI/board_game_AI
python -u generate_MCUCT_samples.py >> ./log/selfplay_proc1.log &
python -u generate_MCUCT_samples.py >> ./log/selfplay_proc2.log &
python -u generate_MCUCT_samples.py >> ./log/selfplay_proc3.log &
python -u generate_MCUCT_samples.py >> ./log/selfplay_proc4.log &
tail -f ./log/selfplay_proc1.log -f ./log/selfplay_proc2.log -f ./log/selfplay_proc3.log -f ./log/selfplay_proc4.log
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
