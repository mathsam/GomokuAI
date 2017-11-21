import matplotlib.pyplot as plt
import pandas as pd

from ai import MCUCT
from gomoku_board import GomokuBoard

plt.ion()

gboard = GomokuBoard()
gboard.update_state((4, 4))
ax = gboard.draw()

ai = MCUCT(C=0.3, cached_depth=5, wall_time=300)
history_stats = []


while gboard.judge() is None:
    best_next_state = ai.best_move(gboard)
    history_stats.append(ai.game_tree.stats)
    gboard.update_state(best_next_state)
    gboard.draw(ax, 0.15)


def summarize_stats(history_stats):
    player_A = history_stats[1::2]
    player_B = history_stats[::2]
    df = pd.DataFrame()
    for i in range(len(player_A)):
        win_rate = (player_A[i][:, 1] / player_A[i][:, 0])
        choice_idx = win_rate.argmax()
        upper_bound = player_A[i][:, 2]
        df.loc[i*2+1, 'A_win_rate'] = win_rate[choice_idx]
        df.loc[i*2+1, 'A_upper_bound'] = upper_bound[choice_idx]
    for i in range(len(player_B)):
        win_rate = (player_B[i][:, 1] / player_B[i][:, 0])
        choice_idx = win_rate.argmax()
        upper_bound = player_B[i][:, 2]
        df.loc[i*2, 'B_win_rate'] = win_rate[choice_idx]
        df.loc[i*2, 'B_upper_bound'] = upper_bound[choice_idx]
    return df


stats = summarize_stats(history_stats)
stats.plot(marker='o')
