import os
import random

import matplotlib.pyplot as plt
import pandas as pd

from ai import MCUCT
from ai_dnn import MCUCT_DNN
from board import PLAYER_A, TIE
from gomoku_board import GomokuBoard

plt.ion()

save_file = 'DNN_vs_PureMCTS.csv'
if os.path.isfile(save_file):
    game_record = pd.read_csv(save_file)
else:
    game_record = pd.DataFrame()

fig = plt.figure()


while True:

    ai_pureMC = MCUCT(GomokuBoard, C=0.3, min_num_sim=1e4, run_type='single')
    ai_dnn = MCUCT_DNN(GomokuBoard, min_num_sim=1e4)
    two_players = [[ai_pureMC, 'pure MCTS'], [ai_dnn, 'DNN']]
    first_player = random.randint(0, 1)

    game_board = GomokuBoard()
    ax = game_board.draw(fig, pause_time=0.01)

    curr_player_idx = first_player
    print('First player %s' %two_players[curr_player_idx][1])
    while game_board.judge() is None:
        best_move = two_players[curr_player_idx][0].best_move()
        curr_player_idx = (curr_player_idx + 1) % 2
        for ai, _ in two_players:
            ai.update_state(best_move)
        game_board.update_state(best_move)
        ax = game_board.draw(ax, pause_time=0.01)

    new_idx = len(game_record)
    game_record.loc[new_idx, 'first_player'] = two_players[first_player][1] 

    if game_board.judge() == TIE:
        print('Game Tie')
        game_record.loc[new_idx, 'winner'] = 'TIE'
    elif game_board.judge() == PLAYER_A:
        print('%s Win' %two_players[first_player][1])
        game_record.loc[new_idx, 'winner'] = two_players[first_player][1]
    else:
        second_player = (first_player + 1) % 2
        print('%s Win' %two_players[second_player][1])
        game_record.loc[new_idx, 'winner'] = two_players[second_player][1]

    game_record.to_csv(save_file, index=False)
    fig.clear()
