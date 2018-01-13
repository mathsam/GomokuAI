import os
import time
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm

data_dir = r'./dnn_data'
TOTAL_NUM_VERSIONS = 9

XY = pd.DataFrame()

for v in range(0, TOTAL_NUM_VERSIONS):
    champion_version = v
    challenger_version = v+1

    file_name = os.path.join(data_dir, 'DNN%d_vs_DNN%d.csv' %(champion_version, challenger_version))
    game_result = pd.read_csv(file_name)

    for i in game_result.index:
        game_num = len(XY)
        if game_result.loc[i, 'winner'] == 'challenger':
            XY.loc[game_num, 'is_challenger_win'] = True
        else:
            XY.loc[game_num, 'is_challenger_win'] = False

        if game_result.loc[i, 'first_player'] == 'challenger':
            XY.loc[game_num, 'is_challenger_first'] = 1
        else:
            XY.loc[game_num, 'is_challenger_first'] = -1
        XY.loc[game_num, '%dv' %champion_version] = -1
        XY.loc[game_num, '%dv' %challenger_version] = 1

XY.fillna(0, inplace=True)
# adding intercept messed up the Elo score
# X = sm.add_constant(XY.iloc[:, 1:])
X = XY.iloc[:, 1:]
Y = XY.iloc[:, 0]

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit = sm.Logit(Y, X)
result = logit.fit()
print result.summary()

elo_score = result.params * 400 / np.log(10)
print elo_score

elo_score_vs_time = pd.DataFrame()

for v in range(0, TOTAL_NUM_VERSIONS+1):
    sample_dir = os.path.join(data_dir, 'v%d' %v)
    # last modification time
    t = pd.to_datetime(datetime.datetime.fromtimestamp(os.path.getmtime(sample_dir)))
    elo_score_vs_time.loc[t, 'Elo score'] = elo_score['%dv' %v]


elo_score_vs_time.plot(marker='o')
