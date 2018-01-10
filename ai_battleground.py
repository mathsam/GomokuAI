import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use CPU only
import sys

import pandas as pd
from ai_dnn import MCUCT_DNN
from board import PLAYER_A, TIE
from gomoku_board import GomokuBoard
import time


CUT_OFF = 0.57 # minimum winning rate to accept the challenger model
if os.environ.get('DISPLAY', '') == '':
    SHOW_LIVE_GAME = False
else:
    SHOW_LIVE_GAME = True
    import matplotlib.pyplot as plt

print '\n--------- Battle starting -----------'
print 'Battle start at ', time.ctime()
start_time = time.time()

base_dir = r'./dnn_data'
training_status = eval(open(os.path.join(base_dir, 'training_status')).read())

champion_model = os.path.join(base_dir, 'v%d' %training_status['current_champion'])
challenger_model = os.path.join(base_dir, 'v%d' %training_status['current_challenger'])
print 'champion model ', champion_model
print 'challenger model ', challenger_model

if SHOW_LIVE_GAME:
    plt.ion()
    fig = plt.figure()

save_file = os.path.join(base_dir,
    'DNN%s_vs_DNN%s.csv' %(training_status['current_champion'], training_status['current_challenger']))
game_record = pd.DataFrame()
 

TOTAL_GAMES = 100
for game_i in range(TOTAL_GAMES):
    print('\n\nPlaying game %d\n' %game_i)

    #ai_pureMC = MCUCT(GomokuBoard, C=0.3, min_num_sim=1e4, run_type='single')
    ai_dnn0 = MCUCT_DNN(GomokuBoard, min_num_sim=2**10, load_path=champion_model, training_mode=True)
    ai_dnn1 = MCUCT_DNN(GomokuBoard, min_num_sim=2**10, load_path=challenger_model, training_mode=True)
    two_players = [[ai_dnn0, 'champion'], [ai_dnn1, 'challenger']]
    #first_player = random.randint(0, 1)
    first_player = game_i % 2

    game_board = GomokuBoard()
    if SHOW_LIVE_GAME:
        ax = game_board.draw(fig, pause_time=0.01)

    curr_player_idx = first_player
    print('First player %s' %two_players[curr_player_idx][1])
    while game_board.judge() is None:
        best_move = two_players[curr_player_idx][0].best_move()
        curr_player_idx = (curr_player_idx + 1) % 2
        for ai, _ in two_players:
            ai.update_state(best_move)
        game_board.update_state(best_move)
        if SHOW_LIVE_GAME:
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
    print game_record.groupby('winner').count()
    if SHOW_LIVE_GAME:
        fig.clear()

game_record['dummy'] = 1
game_record.to_csv(save_file, index=False)
print(game_record.groupby(['first_player', 'winner']).sum())

challenger_winning_rate = ((game_record['winner'] == 'challenger').sum() +
                           0.5*(game_record['winner'] == 'TIE').sum()) / float(TOTAL_GAMES)
print('Challenger winning rate: %f' %challenger_winning_rate)

if challenger_winning_rate > CUT_OFF:
    training_status['current_champion'] = training_status['current_challenger']
    training_status['current_challenger'] += 1

    with open(os.path.join(base_dir, 'training_status'), 'w') as f:
        f.write(str(training_status))
else:
    print('Challenger fail')

print 'Battle end at ', time.ctime()
end_time = time.time()
print 'Time consumed for battle ', end_time - start_time

if not challenger_winning_rate > CUT_OFF:
    sys.exit(1)
