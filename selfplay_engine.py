import matplotlib.pyplot as plt

#from ai import MCUCT
from ai_dnn import MCUCT_DNN
from gomoku_board import GomokuBoard

plt.ion()

#ai = MCUCT(GomokuBoard, C=0.3, min_num_sim=3e4)
ai = MCUCT_DNN(GomokuBoard, min_num_sim=2**12)
game_board = GomokuBoard()

ax = None

while game_board.judge() is None:
    best_next_move = ai.best_move()
    ai.update_state(best_next_move)
    game_board.update_state(best_next_move)
    ax = game_board.draw(ax, 0.01)
