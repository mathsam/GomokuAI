from gomoku_board import GomokuBoard
from ai import MCUCT
import board
import matplotlib.pyplot as plt

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
    ax.imshow(gboard.convert_into_2d_array())
    plt.pause(0.1)
    mc_engine = MCUCT(gboard, wall_time=60)
    best_next_state = mc_engine.best_move()
    gboard.update_state(best_next_state)
    winner = gboard.judge()

ax.imshow(gboard.convert_into_2d_array())
print winner
