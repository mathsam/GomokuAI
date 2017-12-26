import matplotlib.pyplot as plt

from ai import MCUCT
from game_recorder import Recorder
from gomoku_board import GomokuBoard

plt.ion()

ax = None

while True:
    ai = MCUCT(GomokuBoard, C=0.3, min_num_sim=1e4, run_type='single')
    recorder = Recorder(ai, r'./selfplay_history')
    ax = ai.game_board.draw(ax, pause_time=0.01)

    while ai.game_board.judge() is None:
        best_next_move = ai.best_move()
        recorder.record()
        ai.update_state(best_next_move)
        ai.game_board.draw(ax, pause_time=0.001)

    recorder.record()
    plt.clf()
