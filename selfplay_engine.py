import matplotlib.pyplot as plt

import board
from ai import MCUCT
from gomoku_board import GomokuBoard

plt.ion()
grid_size = 9
ticks = range(grid_size)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.grid('on')


gboard = GomokuBoard(grid_size, grid_size, board.PLAYER_A,
                     [(board.PLAYER_A, grid_size//2, grid_size//2)])
winner = gboard.judge()

while winner is None:
    pic = ax.imshow(gboard.convert_into_2d_array())
    pic.set_clim([0, 2])
    plt.pause(0.1)
    mc_engine = MCUCT(gboard, wall_time=400)
    best_next_state = mc_engine.best_move()
    gboard.update_state(best_next_state)
    winner = gboard.judge()

ax.imshow(gboard.convert_into_2d_array())
print winner


import pandas as pd
if len(MCUCT.history[board.PLAYER_A]) > len(MCUCT.history[board.PLAYER_B]):
    MCUCT.history[board.PLAYER_B].append(0)
elif len(MCUCT.history[board.PLAYER_A]) < len(MCUCT.history[board.PLAYER_B]):
    MCUCT.history[board.PLAYER_A].append(0)
pd.DataFrame(MCUCT.history).plot()
