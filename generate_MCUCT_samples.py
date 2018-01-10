import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use CPU only
from ai_dnn import MCUCT_DNN
from game_recorder import Recorder
from gomoku_board import GomokuBoard

base_dir = r'./dnn_data'

plt.ion()
fig = plt.figure()

while True:

    training_status = eval(open(os.path.join(base_dir, 'training_status')).read())
    champion_model = os.path.join(base_dir, 'v%d' %training_status['current_champion'])
    save_dir = os.path.join(base_dir, 'selfplay', 'v%d' %training_status['current_champion'])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print('\n\nLoad champion model from %s' %champion_model)
    ai = MCUCT_DNN(GomokuBoard, training_mode=True, min_num_sim=2**10, load_path=champion_model)
    recorder = Recorder(ai, save_dir)
    ax = ai.game_board.draw(fig, pause_time=0.001)

    while ai.game_board.judge() is None:
        best_next_move = ai.best_move()
        recorder.record()
        ai.update_state(best_next_move)
        ai.game_board.draw(ax, pause_time=0.001)

    recorder.record()
    fig.clear()
