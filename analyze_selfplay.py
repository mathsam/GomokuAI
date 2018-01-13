import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

print '\n------------------ Generating Training Samples ----------------'
print 'Start at ', time.ctime()

base_dir = r'./dnn_data'
training_status = eval(open(os.path.join(base_dir, 'training_status')).read())
data_dir = os.path.join(base_dir, 'selfplay', 'v%d' %training_status['current_champion'])
print('Load selfplay from %s' %data_dir)


X_files = sorted(glob.glob(os.path.join(data_dir, 'X*.csv')))

if len(X_files) < 1000:
    print('Only has %d samples yet' %len(X_files))
    print('----------------EXIT-----------------\n')
    sys.exit(1)
else:
    print('Get %d number of games' %len(X_files))

prev_version = training_status['current_champion'] - 1
while len(X_files) < 12000:
    if prev_version < -1:
        break
    data_dir = os.path.join(base_dir, 'selfplay', 'v%d' %prev_version)
    print "Load sample from %s" %data_dir
    X_files.extend(sorted(glob.glob(os.path.join(data_dir, 'X*.csv'))))
    print('Get %d number of games in total' %len(X_files))
    prev_version -= 1

num_re = re.compile(r'\D+_(\d+)\.csv')

X_df_list = []
Y_df_list = []

for fx in X_files:
    xname = fx.split(r'/')[-1]
    xnum = num_re.search(xname).group(1)
    fy = os.path.join(os.path.dirname(fx), 'Y_%s.csv' %xnum)

    X_df_list.append(pd.read_csv(fx))
    Y_df_list.append(pd.read_csv(fy))

game_summary = pd.DataFrame(columns=['game_length', 'winner', 'first_move'])

for df_X, df_Y in zip(X_df_list, Y_df_list):
    row_i = len(game_summary)
    game_summary.loc[row_i, 'game_length'] = len(df_Y) + 1
    game_summary.loc[row_i, 'winner'] = df_Y.loc[0, 'value']
    game_summary.loc[row_i, 'first_move'] = int(np.asscalar(df_X.columns[df_X.loc[1,:] == 1]))

game_summary.to_csv(os.path.join('analysis', 'game_summary.csv'), index=False)

#train_test_split = int(0.8*len(X_df_list))


def concat_and_shuffle(X, Y, sampling_window=5):
    """
    concat Dataframes in lists X and Y and shuffle indices in them
    :param X: list of DataFrame
    :param Y: list of DataFrame
    :param skip_stride: pick one state out of every `sampling_window` states to avoid serial correlation between
        states in order to reduce overfitting
    :return: (DataFrame, DataFrame)
    """
    X = pd.concat(X, axis=0).reset_index(drop=True)
    Y = pd.concat(Y, axis=0).reset_index(drop=True)
    # change stone color so that current player is always black stone
    X = X.multiply(Y['curr_player'], axis='index')
    assert(all(X.index == Y.index))
    shuffled_idx = np.array(X.index)
    if sampling_window > 1:
        shuffled_idx = shuffled_idx[::sampling_window]
        print('Sampling window is %d' %sampling_window)
    np.random.shuffle(shuffled_idx)
    X = X.loc[shuffled_idx,:].reset_index(drop=True)
    Y = Y.loc[shuffled_idx,:].reset_index(drop=True)
    return X, Y


#X_train, Y_train = concat_and_shuffle(X_df_list[:train_test_split], Y_df_list[:train_test_split], sampling_window=1)
#X_test, Y_test = concat_and_shuffle(X_df_list[train_test_split:], Y_df_list[train_test_split:], sampling_window=1)
X_train, Y_train = concat_and_shuffle(X_df_list, Y_df_list, sampling_window=1)

X_train.to_csv(os.path.join(base_dir, 'X_train.csv'), index=False)
Y_train.to_csv(os.path.join(base_dir, 'Y_train.csv'), index=False)
#X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
#Y_test.to_csv(os.path.join(save_dir, 'Y_test.csv'), index=False)


print 'End at ', time.ctime()
print('---------------- Training sample generated -----------------\n')
