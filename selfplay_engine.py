import matplotlib.pyplot as plt

from ai import MCUCT
from gomoku_board import GomokuBoard

plt.ion()

ai = MCUCT(GomokuBoard, C=0.25, min_num_sim=3e4)
ai.update_state((5, 5))
ax = ai.game_board.draw(None, 0.01)

while ai.game_board.judge() is None:
    best_next_move = ai.best_move()
    ai.update_state(best_next_move)
    ai.game_board.draw(ax, 0.01)
