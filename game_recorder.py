import os

import numpy as np
import pandas as pd

import board as gboard


class Recorder(object):

    def __init__(self, ai, save_path):
        self.ai = ai
        cols = range(gboard.Board.num_cols * gboard.Board.num_rows)
        self.game_board = pd.DataFrame(columns=cols)
        cols.append('curr_player')
        cols.append('value')
        self.game_stats = pd.DataFrame(columns=cols)

        self.save_path = save_path
        self.counter_file = os.path.join(save_path, 'counter')
        if not os.path.isfile(self.counter_file):
            with open(self.counter_file, 'w') as counter_file:
                counter_file.write('0')

    def record(self):
        if self.ai.game_board.judge() is not None:
            winner = self.ai.game_board.judge()
            if winner != gboard.TIE:
                self.game_stats.loc[self.game_stats['curr_player'] == winner, 'value'] = 1
                self.game_stats.loc[self.game_stats['curr_player'] != winner, 'value'] = -1
            self.game_stats.fillna(0, inplace=True)

            counter = int(open(self.counter_file, 'r').readline())
            open(self.counter_file, 'w').write(str(counter+1))

            self.game_board.to_csv(os.path.join(self.save_path, 'X_%d.csv' %counter), index=False)
            self.game_stats.to_csv(os.path.join(self.save_path, 'Y_%d.csv' %counter), index=False)
            return

        curr_row = len(self.game_stats)
        self.game_board.loc[curr_row, :] = self.ai.game_board.convert_into_2d_array().flatten()
        self.game_stats.loc[curr_row, 'curr_player'] = self.ai.game_board.current_player()

        stats = self.ai.game_stats
        move_prob = stats / np.sum(stats)
        avail_moves = self.ai.game_board.avial_moves()
        for p, (i, j) in zip(move_prob, avail_moves):
            idx = i * self.ai.game_board.num_cols + j
            self.game_stats.loc[curr_row, idx] = p
